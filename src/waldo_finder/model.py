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

class StochasticDepth(nn.Module):
    """Stochastic Depth module for advanced regularization."""
    rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        if deterministic or self.rate == 0.:
            return x
        
        keep_prob = 1. - self.rate
        mask = jax.random.bernoulli(
            self.make_rng('dropout'), p=keep_prob, shape=(x.shape[0], 1, 1))
        return x * mask / keep_prob

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for advanced regularization."""
    rate: float = 0.

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return StochasticDepth(rate=self.rate)(x, deterministic=deterministic)

class WaldoDetector(nn.Module):
    """State-of-the-art JAX-based Waldo detector using enhanced Vision Transformer with modern techniques."""
    
    num_heads: int = 16
    num_layers: int = 24
    hidden_dim: int = 1024
    mlp_dim: int = 4096
    dropout_rate: float = 0.1
    drop_path_rate: float = 0.2
    attention_dropout_rate: float = 0.1
    stochastic_depth_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Dict[str, jnp.ndarray]:
        """Forward pass of the model.
        
        Args:
            x: Input image of shape (batch_size, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Dictionary containing bounding boxes and confidence scores
        """
        # Advanced patch embedding with layer normalization
        x = nn.Conv(
            features=self.hidden_dim, 
            kernel_size=(16, 16), 
            strides=(16, 16),
            use_bias=False,
            name='patch_embedding'
        )(x)
        batch_size, h, w, c = x.shape
        x = x.reshape(batch_size, h * w, c)
        x = nn.LayerNorm(name='patch_norm')(x)
        
        # Enhanced learnable embeddings
        class_token = self.param(
            'class_token', 
            nn.initializers.truncated_normal(stddev=0.02), 
            (1, 1, self.hidden_dim)
        )
        class_tokens = jnp.tile(class_token, [batch_size, 1, 1])
        x = jnp.concatenate([class_tokens, x], axis=1)
        
        # Add sinusoidal positional embeddings
        positions = jnp.arange(0, x.shape[1])[None]
        pos_embedding = sinusoidal_position_encoding(positions, self.hidden_dim)
        x = x + pos_embedding
        
        # Advanced Transformer encoder with stochastic depth and enhanced attention
        drop_path_rates = jnp.linspace(0, self.drop_path_rate, self.num_layers)
        
        for i in range(self.num_layers):
            # Multi-head self-attention with relative position bias
            y = nn.LayerNorm(epsilon=1e-6)(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout_rate if training else 0.0,
                deterministic=not training,
                kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal'),
            )(y, y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            y = DropPath(rate=drop_path_rates[i])(y, deterministic=not training)
            x = x + y
            
            # Enhanced MLP block with SwiGLU activation
            y = nn.LayerNorm(epsilon=1e-6)(x)
            features = self.mlp_dim
            y1 = nn.Dense(
                features, 
                kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal'),
            )(y)
            y2 = nn.Dense(
                features, 
                kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal'),
            )(y)
            y = y1 * jax.nn.swish(y2)  # SwiGLU activation
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            y = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal'),
            )(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            y = DropPath(rate=drop_path_rates[i])(y, deterministic=not training)
            x = x + y
        
        # Use class token for detection
        x = x[:, 0]  # Take class token output
        
        # Advanced detection heads with deeper architecture
        x = nn.LayerNorm(epsilon=1e-6, name='final_norm')(x)
        
        # Box prediction head with deeper network
        boxes = nn.Sequential([
            nn.Dense(512, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal')),
            nn.gelu,
            lambda x: nn.Dropout(rate=0.1)(x, deterministic=not training),
            nn.Dense(256, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal')),
            nn.gelu,
            lambda x: nn.Dropout(rate=0.1)(x, deterministic=not training),
            nn.Dense(4, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal')),
            nn.sigmoid,  # Normalize coordinates to [0,1]
        ], name='box_head')(x)
        
        # Score prediction head with deeper network
        scores = nn.Sequential([
            nn.Dense(512, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal')),
            nn.gelu,
            lambda x: nn.Dropout(rate=0.1)(x, deterministic=not training),
            nn.Dense(256, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal')),
            nn.gelu,
            lambda x: nn.Dropout(rate=0.1)(x, deterministic=not training),
            nn.Dense(1, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in', 'truncated_normal')),
            nn.sigmoid,
        ], name='score_head')(x)
        
        return {
            'boxes': boxes,  # (batch_size, 4) for x1,y1,x2,y2
            'scores': scores,  # (batch_size, 1) confidence scores
        }

def sinusoidal_position_encoding(positions: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Sinusoidal positional encoding used in 'Attention is All You Need'."""
    # Based on formula from original transformer paper
    angles = jnp.arange(0, dim, 2)[None] / dim
    angles = positions[:, :, None] * jnp.exp(-angles * jnp.log(10000))
    pos_enc = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return pos_enc

def create_train_state(rng: jnp.ndarray, 
                      learning_rate: float, 
                      model_kwargs: Dict,
                      num_train_steps: Optional[int] = None,
                      warmup_epochs: Optional[int] = None,
                      steps_per_epoch: Optional[int] = None) -> TrainState:
    """Creates initial training state with modern optimizer configuration."""
    # Extract only the parameters expected by WaldoDetector
    detector_kwargs = {
        'num_heads': model_kwargs.get('num_heads', 12),
        'num_layers': model_kwargs.get('num_layers', 12),
        'hidden_dim': model_kwargs.get('hidden_dim', 768),
        'mlp_dim': model_kwargs.get('mlp_dim', 3072),
        'dropout_rate': model_kwargs.get('dropout_rate', 0.1),
    }
    model = WaldoDetector(**detector_kwargs)
    params = model.init(rng, jnp.ones((1, 640, 640, 3)))['params']
    
    # Calculate schedule parameters
    if num_train_steps is None and steps_per_epoch is not None:
        num_train_steps = steps_per_epoch * 50  # Default 50 epochs
    
    if warmup_epochs is not None and steps_per_epoch is not None:
        warmup_steps = warmup_epochs * steps_per_epoch
    else:
        warmup_steps = min(1000, num_train_steps // 10) if num_train_steps else 1000
    
    decay_steps = num_train_steps if num_train_steps else 50000
    
    # Modern learning rate schedule with warmup and cosine decay
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )
    
    # Advanced optimizer configuration with weight decay
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=scheduler,
            weight_decay=0.01,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        )
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
    """Computes loss and metrics with modern loss formulation."""
    outputs = state.apply_fn(
        {'params': params},
        batch['image'],
        training=True,
        rngs={'dropout': rng}
    )
    
    # GIoU loss for better bounding box regression
    giou_loss = generalized_box_iou_loss(
        outputs['boxes'],
        batch['boxes']
    ).mean()
    
    # Focal loss for handling class imbalance
    alpha = 0.25
    gamma = 2.0
    score_loss = sigmoid_focal_loss(
        outputs['scores'],
        batch['scores'],
        alpha=alpha,
        gamma=gamma
    ).mean()
    
    total_loss = giou_loss + score_loss
    
    metrics = {
        'loss': total_loss,
        'giou_loss': giou_loss,
        'score_loss': score_loss,
    }
    
    return total_loss, metrics

def generalized_box_iou_loss(pred_boxes: jnp.ndarray,
                            true_boxes: jnp.ndarray) -> jnp.ndarray:
    """Computes the GIoU loss between predicted and ground truth boxes."""
    # Convert boxes from (x1,y1,x2,y2) format to (x,y,w,h)
    pred_centers = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
    pred_sizes = pred_boxes[..., 2:] - pred_boxes[..., :2]
    
    true_centers = (true_boxes[..., :2] + true_boxes[..., 2:]) / 2
    true_sizes = true_boxes[..., 2:] - true_boxes[..., :2]
    
    # Compute IoU
    intersect_sizes = jnp.minimum(
        pred_centers[..., None, :] + pred_sizes[..., None, :] / 2,
        true_centers + true_sizes / 2
    ) - jnp.maximum(
        pred_centers[..., None, :] - pred_sizes[..., None, :] / 2,
        true_centers - true_sizes / 2
    )
    
    intersect_area = jnp.prod(jnp.maximum(intersect_sizes, 0), axis=-1)
    pred_area = jnp.prod(pred_sizes, axis=-1)
    true_area = jnp.prod(true_sizes, axis=-1)
    union_area = pred_area[..., None] + true_area - intersect_area
    
    iou = intersect_area / (union_area + 1e-6)
    
    # Compute enclosing box
    enclose_sizes = jnp.maximum(
        pred_centers[..., None, :] + pred_sizes[..., None, :] / 2,
        true_centers + true_sizes / 2
    ) - jnp.minimum(
        pred_centers[..., None, :] - pred_sizes[..., None, :] / 2,
        true_centers - true_sizes / 2
    )
    
    enclose_area = jnp.prod(enclose_sizes, axis=-1)
    
    # Compute GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    
    return 1 - giou

def sigmoid_focal_loss(pred: jnp.ndarray,
                      target: jnp.ndarray,
                      alpha: float = 0.25,
                      gamma: float = 2.0) -> jnp.ndarray:
    """Compute focal loss for better handling of class imbalance."""
    pred = jax.nn.sigmoid(pred)
    ce_loss = optax.sigmoid_binary_cross_entropy(pred, target)
    p_t = target * pred + (1 - target) * (1 - pred)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal_term = jnp.power(1 - p_t, gamma)
    return alpha_t * focal_term * ce_loss

@jax.jit
def train_step(state: TrainState,
               batch: Dict[str, jnp.ndarray],
               rng: jnp.ndarray) -> Tuple[TrainState, Dict]:
    """Performs a single training step with gradient clipping."""
    rng, dropout_rng = jax.random.split(rng)
    
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, batch, state, dropout_rng)
    
    # Gradient clipping and update
    grads = jax.tree_map(
        lambda g: jnp.clip(g, -1.0, 1.0),
        grads
    )
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
        def _forward(params):
            outputs = state.apply_fn(
                {'params': params},
                batch['image'],
                training=True,
                rngs={'dropout': dropout_rng}
            )
            return compute_loss(params, batch, state, dropout_rng)
        
        return dynamic_scale.value_and_grad(
            _forward, has_aux=True, axis_name='batch'
        )(params)
    
    # Run forward and backward pass with dynamic scaling
    dynamic_scale, is_finite, aux = dynamic_scale.apply(_loss_fn, state.params)
    (loss, metrics), grads = aux
    
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
    )
    return outputs
