"""JAX-based neural network model for Waldo detection."""

from typing import Dict, Tuple, NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

class BoundingBox(NamedTuple):
    """Bounding box coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class WaldoDetector(nn.Module):
    """Modern JAX-based Waldo detector using Vision Transformer backbone with DETR-style detection."""
    
    num_heads: int = 12
    num_layers: int = 12
    hidden_dim: int = 768
    mlp_dim: int = 3072
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Dict[str, jnp.ndarray]:
        """Forward pass of the model.
        
        Args:
            x: Input image of shape (batch_size, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Dictionary containing bounding boxes and confidence scores
        """
        # Image embedding with modern ViT architecture
        x = nn.Conv(features=self.hidden_dim, kernel_size=(16, 16), strides=(16, 16))(x)
        batch_size, h, w, c = x.shape
        x = x.reshape(batch_size, h * w, c)
        
        # Add learnable class token for global features
        class_token = self.param('class_token', 
                               nn.initializers.normal(stddev=0.02), 
                               (1, 1, self.hidden_dim))
        class_tokens = jnp.tile(class_token, [batch_size, 1, 1])
        x = jnp.concatenate([class_tokens, x], axis=1)
        
        # Add sinusoidal positional embeddings
        positions = jnp.arange(0, x.shape[1])[None]
        pos_embedding = sinusoidal_position_encoding(positions, self.hidden_dim)
        x = x + pos_embedding
        
        # Transformer encoder with modern improvements
        for _ in range(self.num_layers):
            # Multi-head self-attention with pre-norm
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate if training else 0.0,
                deterministic=not training,
            )(y, y)
            x = x + y
            
            # MLP block with GELU activation
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.mlp_dim)(y)
            y = nn.gelu(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            y = nn.Dense(self.hidden_dim)(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            x = x + y
        
        # Use class token for detection
        x = x[:, 0]  # Take class token output
        
        # Detection heads
        boxes = nn.Sequential([
            nn.Dense(256),
            nn.gelu,
            nn.Dense(4),
            nn.sigmoid,  # Normalize coordinates to [0,1]
        ])(x)
        
        scores = nn.Sequential([
            nn.Dense(256),
            nn.gelu,
            nn.Dense(1),
            nn.sigmoid,
        ])(x)
        
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
                      model_kwargs: Dict) -> train_state.TrainState:
    """Creates initial training state with modern optimizer configuration."""
    model = WaldoDetector(**model_kwargs)
    params = model.init(rng, jnp.ones((1, 640, 640, 3)))['params']
    
    # Modern learning rate schedule with warmup and cosine decay
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=50000,
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
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

def compute_loss(params: Dict,
                batch: Dict[str, jnp.ndarray],
                state: train_state.TrainState,
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
def train_step(state: train_state.TrainState,
               batch: Dict[str, jnp.ndarray],
               rng: jnp.ndarray) -> Tuple[train_state.TrainState, Dict]:
    """Performs a single training step with gradient clipping."""
    rng, dropout_rng = jax.random.split(rng)
    
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, batch, state, dropout_rng)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, metrics

@jax.jit
def eval_step(state: train_state.TrainState,
              batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Performs evaluation step."""
    outputs = state.apply_fn(
        {'params': state.params},
        batch['image'],
        training=False,
    )
    return outputs
