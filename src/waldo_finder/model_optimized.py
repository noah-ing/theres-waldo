"""Enhanced JAX-based neural network model for Waldo detection with advanced training techniques."""

from typing import Dict, Tuple, NamedTuple, Optional, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib
import optax

class TrainState(train_state.TrainState):
    """Custom train state with dropout RNG and EMA parameters."""
    dropout_rng: Any
    ema_params: Optional[Dict] = None

class GridMaskAugmentation:
    """GridMask augmentation for structured regularization."""
    def __init__(self, ratio=0.4, rotate=45, invert=True):
        self.ratio = ratio
        self.rotate = rotate
        self.invert = invert
    
    def __call__(self, rng: jnp.ndarray, image: jnp.ndarray) -> jnp.ndarray:
        """Apply GridMask augmentation."""
        h, w = image.shape[1:3]
        mask = self._create_grid_mask(rng, h, w)
        if self.invert:
            mask = 1 - mask
        return image * mask[None, :, :, None]
    
    def _create_grid_mask(self, rng: jnp.ndarray, h: int, w: int) -> jnp.ndarray:
        """Create a grid mask pattern."""
        grid_size = int(min(h, w) * self.ratio)
        mask = jnp.ones((h, w))
        
        # Create grid pattern
        xx, yy = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
        
        # Apply rotation
        if self.rotate:
            angle = jnp.deg2rad(self.rotate)
            rot_xx = xx * jnp.cos(angle) - yy * jnp.sin(angle)
            rot_yy = xx * jnp.sin(angle) + yy * jnp.cos(angle)
            xx, yy = rot_xx, rot_yy
        
        # Create grid pattern
        mask = jnp.where(
            (xx % (grid_size * 2) < grid_size) & 
            (yy % (grid_size * 2) < grid_size),
            0.0, 1.0
        )
        return mask

class ConsistencyRegularization:
    """Implements consistency regularization with multiple views."""
    def __init__(self, temperature=0.07, num_crops=2):
        self.temperature = temperature
        self.num_crops = num_crops
    
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Compute consistency loss between different views."""
        # Normalize features
        features = features / jnp.linalg.norm(features, axis=-1, keepdims=True)
        
        # Split features into crops
        crops = jnp.split(features, self.num_crops)
        
        # Compute similarity matrix
        sim_matrix = jnp.exp(
            jnp.dot(features, features.T) / self.temperature
        )
        
        # Mask out self-similarity
        mask = jnp.eye(len(features))
        sim_matrix = sim_matrix * (1 - mask)
        
        # Compute loss
        loss = -jnp.log(
            sim_matrix / (sim_matrix.sum(axis=-1, keepdims=True) + 1e-6)
        )
        return loss.mean()

class EnhancedAttention(nn.Module):
    """Enhanced multi-head attention with relative position encoding."""
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Project input to queries, keys, and values
        qkv = nn.Dense(self.num_heads * self.head_dim * 3)(x)
        qkv = qkv.reshape(x.shape[0], -1, 3, self.num_heads, self.head_dim)
        queries, keys, values = jnp.split(qkv, 3, axis=2)
        
        # Compute relative position encoding
        seq_len = x.shape[1]
        pos_enc = self._relative_position_encoding(seq_len, self.head_dim)
        
        # Add relative position encoding to keys
        keys = keys + pos_enc[None, :, None, :]
        
        # Compute attention scores
        scale = jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bqhd,bkhd->bhqk', queries, keys) / scale
        
        # Apply attention dropout
        scores = nn.Dropout(rate=self.dropout_rate)(
            scores, deterministic=not training
        )
        
        # Compute weighted sum
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn, values)
        
        # Combine heads and project
        out = out.reshape(x.shape[0], -1, self.num_heads * self.head_dim)
        return nn.Dense(x.shape[-1])(out)
    
    def _relative_position_encoding(
        self, seq_len: int, dim: int
    ) -> jnp.ndarray:
        """Compute relative position encoding."""
        positions = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, dim, 2) * -(jnp.log(10000.0) / dim)
        )
        pos_enc = jnp.zeros((seq_len, dim))
        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(positions * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(positions * div_term))
        return pos_enc

class EnhancedWaldoDetector(nn.Module):
    """Enhanced Vision Transformer with advanced training techniques."""
    
    num_heads: int = 8
    num_layers: int = 8
    hidden_dim: int = 512
    mlp_dim: int = 2048
    dropout_rate: float = 0.3
    attention_dropout: float = 0.2
    path_dropout: float = 0.2
    stochastic_depth_rate: float = 0.2
    
    def setup(self):
        """Initialize augmentation and regularization modules."""
        self.gridmask = GridMaskAugmentation(
            ratio=0.4, rotate=45, invert=True
        )
        self.consistency = ConsistencyRegularization(
            temperature=0.07, num_crops=2
        )
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
        augment: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass with advanced training techniques."""
        # Apply GridMask augmentation during training
        if training and augment:
            rng = self.make_rng('dropout')
            x = self.gridmask(rng, x)
        
        # Enhanced patch embedding
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(16, 16),
            strides=(16, 16),
            use_bias=False,
            name='patch_embedding'
        )(x)
        
        batch_size, h, w, c = x.shape
        x = x.reshape(batch_size, h * w, c)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Class token and position embedding
        cls_token = self.param(
            'cls_token',
            nn.initializers.truncated_normal(0.02),
            (1, 1, self.hidden_dim)
        )
        cls_tokens = jnp.tile(cls_token, [batch_size, 1, 1])
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # Add learned position embeddings
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(0.02),
            (1, x.shape[1], self.hidden_dim)
        )
        x = x + pos_embedding
        
        # Transformer layers with enhanced attention
        for i in range(self.num_layers):
            # Pre-norm architecture
            attn_input = nn.LayerNorm(epsilon=1e-6)(x)
            
            # Enhanced attention mechanism
            attn_output = EnhancedAttention(
                num_heads=self.num_heads,
                head_dim=self.hidden_dim // self.num_heads,
                dropout_rate=self.attention_dropout if training else 0.0
            )(attn_input, training=training)
            
            # Residual connection with stochastic depth
            if training:
                keep_prob = 1.0 - (
                    float(i) / max(self.num_layers - 1, 1)
                ) * self.stochastic_depth_rate
                attn_output = nn.Dropout(rate=1 - keep_prob)(
                    attn_output, deterministic=False
                )
            x = x + attn_output
            
            # MLP block
            mlp_input = nn.LayerNorm(epsilon=1e-6)(x)
            mlp_output = nn.Sequential([
                nn.Dense(self.mlp_dim),
                nn.gelu,
                nn.Dropout(rate=self.dropout_rate),
                nn.Dense(self.hidden_dim)
            ])(mlp_input, deterministic=not training)
            
            if training:
                mlp_output = nn.Dropout(rate=1 - keep_prob)(
                    mlp_output, deterministic=False
                )
            x = x + mlp_output
        
        # Final layer normalization
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Extract features for consistency loss
        features = x[:, 0]  # Use CLS token features
        
        # Detection heads
        boxes = nn.Sequential([
            nn.Dense(256),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            nn.Dense(4)
        ])(features, deterministic=not training)
        
        # Enforce valid box coordinates
        x1y1, x2y2 = jnp.split(boxes, 2, axis=-1)
        x1y1 = jax.nn.sigmoid(x1y1)
        relative_offset = jax.nn.sigmoid(x2y2) * (1 - x1y1)
        x2y2 = x1y1 + relative_offset
        boxes = jnp.concatenate([x1y1, x2y2], axis=-1)
        
        # Confidence scores
        scores = nn.Sequential([
            nn.Dense(256),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            nn.Dense(1),
            nn.sigmoid
        ])(features, deterministic=not training)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'features': features  # For consistency loss
        }

def create_optimized_train_state(
    rng: jnp.ndarray,
    learning_rate: float,
    model_kwargs: Dict,
    num_train_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None
) -> TrainState:
    """Creates initial training state with advanced optimization."""
    model = EnhancedWaldoDetector(**model_kwargs)
    
    # Initialize model
    init_rng, dropout_rng = jax.random.split(rng)
    params = model.init(
        {'params': init_rng, 'dropout': dropout_rng},
        jnp.ones((1, 640, 640, 3))
    )['params']
    
    # Advanced learning rate schedule
    if warmup_steps is None:
        warmup_steps = num_train_steps // 20 if num_train_steps else 1000
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_train_steps,
        end_value=learning_rate * 0.01
    )
    
    # Advanced optimizer with gradient clipping and weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.2
        )
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_rng,
        ema_params=None  # Will be initialized during training
    )

def compute_optimized_loss(
    params: Dict,
    batch: Dict[str, jnp.ndarray],
    state: TrainState,
    rng: jnp.ndarray,
    consistency_weight: float = 0.5
) -> Tuple[jnp.ndarray, Dict]:
    """Compute loss with consistency regularization."""
    outputs = state.apply_fn(
        {'params': params},
        batch['image'],
        training=True,
        augment=True,
        rngs={'dropout': rng}
    )
    
    # Main detection losses
    giou_loss = generalized_box_iou_loss(
        outputs['boxes'],
        batch['boxes']
    ).mean()
    
    score_loss = sigmoid_focal_loss(
        outputs['scores'],
        batch['scores'],
        alpha=0.25,
        gamma=2.0
    ).mean()
    
    # Consistency regularization loss
    consistency_loss = ConsistencyRegularization()(outputs['features'])
    
    # Combined loss
    total_loss = giou_loss + score_loss + consistency_weight * consistency_loss
    
    metrics = {
        'loss': total_loss,
        'giou_loss': giou_loss,
        'score_loss': score_loss,
        'consistency_loss': consistency_loss
    }
    
    return total_loss, metrics

# Re-use existing loss helper functions
generalized_box_iou_loss = generalized_box_iou_loss  # From original model.py
sigmoid_focal_loss = sigmoid_focal_loss  # From original model.py
