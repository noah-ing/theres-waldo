"""JAX-based neural network model for Waldo detection."""

from typing import Dict, Tuple, NamedTuple, Optional, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.core import FrozenDict
import optax

class TrainState(train_state.TrainState):
    """Custom train state with dropout RNG."""
    dropout_rng: Any

class BoundingBox(NamedTuple):
    """Bounding box coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class SelfAttention(nn.Module):
    """Basic self-attention module."""
    num_heads: int
    hidden_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        batch_size, seq_len, dim = x.shape
        head_dim = dim // self.num_heads
        scale = head_dim ** -0.5
        
        # Simple QKV projection
        qkv = nn.Dense(3 * dim)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention
        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        if training and self.dropout_rate > 0:
            attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not training)
        
        x = jnp.transpose((attn @ v), (0, 2, 1, 3))
        x = x.reshape(batch_size, seq_len, dim)
        return nn.Dense(dim)(x)

class ScaleAwareAttention(nn.Module):
    """Scale-aware self-attention module."""
    num_heads: int
    hidden_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, scale_embedding: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        batch_size, seq_len, dim = x.shape
        head_dim = dim // self.num_heads
        scale = head_dim ** -0.5
        
        # Project scale embedding to match sequence length
        scale_proj = nn.Dense(seq_len)(scale_embedding.transpose((0, 2, 1))).transpose((0, 2, 1))
        
        # Incorporate scale information through gating
        scale_gate = nn.sigmoid(nn.Dense(dim)(scale_proj))
        x = x * (1 + scale_gate)
        
        # QKV with scale awareness
        qkv = nn.Dense(3 * dim)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scale-aware attention
        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        if training and self.dropout_rate > 0:
            attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not training)
        
        x = jnp.transpose((attn @ v), (0, 2, 1, 3))
        x = x.reshape(batch_size, seq_len, dim)
        return nn.Dense(dim)(x)

class WaldoDetector(nn.Module):
    """Scale-aware Vision Transformer for Waldo detection."""
    
    num_heads: int = 8  # CPU optimized architecture
    num_layers: int = 8  # CPU optimized depth
    hidden_dim: int = 512  # Balanced capacity
    mlp_dim: int = 2048  # Balanced representation
    dropout_rate: float = 0.2  # From systemPatterns.md
    attention_dropout_rate: float = 0.2  # Matched dropout
    scale_bins: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25)  # Size-based bins

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Dict[str, jnp.ndarray]:
        # Simple patch embedding (16x16 from patterns)
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(16, 16),
            strides=(16, 16),
            use_bias=False,
            name='patch_embedding'
        )(x)
        
        batch_size, h, w, c = x.shape
        x = x.reshape(batch_size, h * w, c)
        x = nn.LayerNorm(epsilon=1e-6, name='patch_norm')(x)
        
        # Add class token
        class_token = self.param(
            'class_token',
            nn.initializers.truncated_normal(stddev=0.02),
            (1, 1, self.hidden_dim)
        )
        class_tokens = jnp.tile(class_token, [batch_size, 1, 1])
        x = jnp.concatenate([class_tokens, x], axis=1)
        
        # Add positional embeddings
        positions = jnp.arange(0, x.shape[1])[None]
        angles = jnp.arange(0, self.hidden_dim, 2)[None] / self.hidden_dim
        angles = positions[:, :, None] * jnp.exp(-angles * jnp.log(10000))
        pos_embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        x = x + pos_embedding
        
        # Scale embedding for size-aware attention
        scale_embedding = self.param(
            'scale_embedding',
            nn.initializers.truncated_normal(stddev=0.02),
            (1, len(self.scale_bins), self.hidden_dim // 2)  # Reduced dimension for efficiency
        )
        scale_embedding = jnp.tile(scale_embedding, [batch_size, 1, 1])
        
        # Transformer encoder with scale awareness
        for i in range(self.num_layers):
            # Scale-aware attention block
            y = nn.LayerNorm(epsilon=1e-6)(x)
            y = ScaleAwareAttention(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.attention_dropout_rate if training else 0.0,
            )(y, scale_embedding, training)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            x = x + y
            
            # MLP block
            y = nn.LayerNorm(epsilon=1e-6)(x)
            y = nn.Dense(self.mlp_dim)(y)
            y = nn.gelu(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            y = nn.Dense(self.hidden_dim)(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            x = x + y
        
        # Detection heads
        x = jnp.concatenate([
            x[:, 0],  # class token
            jnp.mean(x[:, 1:], axis=1),  # global average
        ], axis=-1)
        x = nn.LayerNorm(epsilon=1e-6, name='final_norm')(x)
        
        # Face-specific feature extraction with hierarchical attention
        face_features = nn.Sequential([
            nn.Dense(1024),
            nn.gelu,
            nn.LayerNorm(epsilon=1e-6),  # Normalize features
            nn.Dense(512),
            nn.gelu,
            nn.LayerNorm(epsilon=1e-6),  # Normalize features
            nn.Dense(256),
            nn.gelu,
            nn.LayerNorm(epsilon=1e-6),  # Final normalization
        ])(x)
        
        # Face-focused confidence scoring
        scores = nn.Dense(1)(face_features)
        scores = jax.nn.sigmoid(scores)
        
        # Only predict box if confident about face detection
        box_features = face_features * scores  # Scale features by confidence
        
        # Scale-aware box prediction
        boxes_raw = nn.Dense(4)(box_features)
        
        # Adaptive size constraints based on scale bins
        xy_center = jax.nn.sigmoid(boxes_raw[..., :2])
        
        # Scale-aware size prediction
        wh_raw = boxes_raw[..., 2:]
        
        # Compute scale-based weights
        scale_weights = jnp.zeros_like(wh_raw[..., 0:1])
        for i, (min_size, max_size) in enumerate(zip(self.scale_bins[:-1], self.scale_bins[1:])):
            size_range = max_size - min_size
            mask = (wh_raw >= min_size) & (wh_raw < max_size)
            scale_weights = jnp.where(mask, 
                                    2.0 - jnp.abs(2 * (wh_raw - min_size) / size_range - 1),
                                    scale_weights)
        
        # Apply scale-aware constraints
        wh = jnp.clip(jax.nn.sigmoid(wh_raw), self.scale_bins[0], self.scale_bins[-1])
        wh = wh * (1.0 + 0.2 * scale_weights)  # Allow 20% flexibility based on scale
        
        # Convert to corner format
        half_wh = wh / 2
        boxes = jnp.concatenate([
            xy_center - half_wh,  # x1,y1
            xy_center + half_wh   # x2,y2
        ], axis=-1)
        
        return {
            'boxes': boxes,  # (batch_size, 4) for x1,y1,x2,y2
            'scores': scores,  # (batch_size, 1) confidence scores
        }

def create_train_state(rng: jnp.ndarray, 
                      learning_rate: float, 
                      model_kwargs: Dict,
                      num_train_steps: Optional[int] = None,
                      warmup_epochs: Optional[int] = None,
                      steps_per_epoch: Optional[int] = None) -> TrainState:
    """Creates initial training state."""
    # Extract model parameters with proven defaults
    detector_kwargs = {
        'num_heads': model_kwargs.get('num_heads', 8),  # CPU optimized
        'num_layers': model_kwargs.get('num_layers', 8),  # CPU optimized
        'hidden_dim': model_kwargs.get('hidden_dim', 512),  # Balanced
        'mlp_dim': model_kwargs.get('mlp_dim', 2048),  # Balanced
        'dropout_rate': model_kwargs.get('dropout_rate', 0.2),  # From patterns
        'scale_bins': model_kwargs.get('scale_bins', (0.01, 0.05, 0.1, 0.15, 0.2, 0.25)),  # Scale-aware bins
    }
    model = WaldoDetector(**detector_kwargs)
    
    # Get image size from data config
    data_config = model_kwargs.get('data', {})
    image_size = data_config.get('image_size', (800, 800))
    params = model.init(rng, jnp.ones((1, *image_size, 3)))['params']
    
    # Optimizer configuration from systemPatterns.md
    tx = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=0.01,  # Standard weight decay
        b1=0.9,  # Standard beta1
        b2=0.999,  # Standard beta2
        eps=1e-8,  # Standard epsilon
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=rng)

def compute_loss(params: Dict,
                batch: Dict[str, jnp.ndarray],
                state: TrainState,
                rng: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
    """Computes scale-aware loss for Waldo detection."""
    outputs = state.apply_fn(
        {'params': params},
        batch['image'],
        training=True,
        rngs={'dropout': rng}
    )
    
    # First focus on Waldo detection confidence
    pred_conf = jax.nn.sigmoid(outputs['scores'])
    
    # Aggressive focal loss for better Waldo detection
    alpha = 0.75  # Higher alpha to focus more on positive samples (Waldo)
    gamma = 2.0   # Higher gamma for harder examples
    
    # Compute focal loss
    p_t = batch['scores'] * pred_conf + (1 - batch['scores']) * (1 - pred_conf)
    alpha_t = batch['scores'] * alpha + (1 - batch['scores']) * (1 - alpha)
    focal_term = jnp.power(1 - p_t, gamma)
    score_loss = optax.sigmoid_binary_cross_entropy(
        outputs['scores'],
        batch['scores']
    )
    score_loss = (alpha_t * focal_term * score_loss).mean()
    
    # Only compute box loss when confident about Waldo
    pred_boxes = outputs['boxes']
    true_boxes = batch['boxes']
    
    # Compute IoU
    intersect_mins = jnp.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = jnp.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = jnp.maximum(intersect_maxs - intersect_mins, 0.0)
    intersect_area = jnp.prod(intersect_wh, axis=-1)
    
    pred_wh = jnp.maximum(pred_boxes[..., 2:] - pred_boxes[..., :2], 0.0)
    true_wh = jnp.maximum(true_boxes[..., 2:] - true_boxes[..., :2], 0.0)
    pred_area = jnp.prod(pred_wh, axis=-1)
    true_area = jnp.prod(true_wh, axis=-1)
    
    union_area = pred_area + true_area - intersect_area
    iou = intersect_area / (union_area + 1e-6)
    
    # GIoU computation
    enclose_mins = jnp.minimum(pred_boxes[..., :2], true_boxes[..., :2])
    enclose_maxs = jnp.maximum(pred_boxes[..., 2:], true_boxes[..., 2:])
    enclose_wh = jnp.maximum(enclose_maxs - enclose_mins, 0.0)
    enclose_area = jnp.prod(enclose_wh, axis=-1)
    
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    giou_loss = (1 - giou).mean()
    
    # L1 loss weighted by confidence
    l1_loss = jnp.mean(jnp.abs(pred_boxes - true_boxes) * pred_conf)
    
    # Compute scale-based weights
    true_wh = batch['boxes'][..., 2:] - batch['boxes'][..., :2]
    true_size = jnp.sqrt(jnp.prod(true_wh, axis=-1))
    
    # Get scale bins from model config
    scale_bins = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25)  # Match with model's default bins
    
    scale_weights = jnp.ones_like(true_size)
    for i, (min_size, max_size) in enumerate(zip(scale_bins[:-1], scale_bins[1:])):
        size_range = max_size - min_size
        mask = (true_size >= min_size) & (true_size < max_size)
        scale_weights = jnp.where(mask,
                                2.0 - jnp.abs(2 * (true_size - min_size) / size_range - 1),
                                scale_weights)
    
    # Scale-aware box loss
    box_loss = (giou_loss + 0.05 * l1_loss) * scale_weights
    
    # Balanced loss with scale awareness
    total_loss = score_loss + jnp.mean(box_loss * pred_conf)
    
    metrics = {
        'loss': total_loss,
        'giou_loss': giou_loss,
        'l1_loss': l1_loss,
        'box_loss': jnp.mean(box_loss),  # Reduce to scalar
        'score_loss': score_loss,
        'waldo_conf': jnp.mean(pred_conf),  # Track Waldo detection confidence
    }
    
    return total_loss, metrics

@jax.jit
def train_step(state: TrainState,
               batch: Dict[str, jnp.ndarray],
               rng: jnp.ndarray) -> Tuple[TrainState, Dict]:
    """Performs a single training step."""
    rng, dropout_rng = jax.random.split(rng)
    
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, batch, state, dropout_rng)
    
    state = state.apply_gradients(grads=grads)
    return state, metrics

@jax.jit
def train_step_mixed_precision(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    dynamic_scale: dynamic_scale_lib.DynamicScale
) -> Tuple[TrainState, Dict, dynamic_scale_lib.DynamicScale]:
    """Performs a single training step with mixed precision and dynamic scaling."""
    rng, dropout_rng = jax.random.split(rng)
    
    def _loss_fn(params):
        """Loss function with mixed precision."""
        outputs = state.apply_fn(
            {'params': params},
            batch['image'],
            training=True,
            rngs={'dropout': dropout_rng}
        )
        return compute_loss(params, batch, state, dropout_rng)
    
    # Run forward and backward pass with dynamic scaling
    dynamic_scale, is_finite, (loss, metrics), grads = dynamic_scale.value_and_grad(
        _loss_fn, has_aux=True)(state.params)
    
    # Handle non-finite gradients
    metrics = jax.tree_map(lambda x: jnp.where(is_finite, x, 0.0), metrics)
    state = jax.tree_map(
        lambda x, y: jnp.where(is_finite, x, y),
        state.apply_gradients(grads=grads),
        state
    )
    
    return state, metrics, dynamic_scale

@jax.jit
def eval_step(state: TrainState,
              batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Performs evaluation step."""
    outputs = state.apply_fn(
        {'params': state.params},
        batch['image'],
        training=False,
        rngs={'dropout': state.dropout_rng}
    )
    return outputs
