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
        """Compute consistency loss between different views with numerical stability."""
        # Normalize features with epsilon for numerical stability
        epsilon = 1e-6
        features = features / (jnp.linalg.norm(features, axis=-1, keepdims=True) + epsilon)
        
        # Ensure features can be evenly split
        batch_size = features.shape[0]
        if batch_size % self.num_crops != 0:
            # Pad features to make it divisible
            pad_size = self.num_crops - (batch_size % self.num_crops)
            features = jnp.pad(features, ((0, pad_size), (0, 0)))
        
        # Split features into crops
        crops = jnp.split(features, self.num_crops)
        
        # Compute similarity matrix with numerical stability
        # Clip dot product to avoid exponential overflow
        dot_product = jnp.clip(
            jnp.dot(features[:batch_size], features[:batch_size].T),
            a_min=-1.0/self.temperature,
            a_max=1.0/self.temperature
        )
        sim_matrix = jnp.exp(dot_product * self.temperature)
        
        # Mask out self-similarity
        mask = jnp.eye(batch_size)
        sim_matrix = sim_matrix * (1 - mask)
        
        # Add small constant for numerical stability in denominator
        denominator = sim_matrix.sum(axis=-1, keepdims=True) + epsilon
        
        # Compute loss with clipping to avoid log(0)
        loss = -jnp.log(jnp.clip(sim_matrix / denominator, epsilon, 1.0))
        
        # Return mean of finite values only
        return jnp.nan_to_num(loss.mean(), nan=0.0, posinf=1.0, neginf=0.0)

class EnhancedAttention(nn.Module):
    """Enhanced multi-head attention with relative position encoding."""
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        B, L, _ = x.shape  # batch size, sequence length
        
        # Single projection for Q, K, V
        qkv = nn.Dense(3 * self.num_heads * self.head_dim)(x)
        
        # Reshape to [batch, length, num_heads, 3 * head_dim]
        qkv = qkv.reshape(B, L, self.num_heads, 3 * self.head_dim)
        
        # Split into Q, K, V
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Add relative position encoding to keys
        pos_enc = self._relative_position_encoding(L, self.head_dim)
        k = k + pos_enc[None, :, None, :]
        
        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim)
        attn_weights = (q @ jnp.swapaxes(k, -2, -1)) / scale
        
        # Apply dropout to attention weights
        if training and self.dropout_rate > 0:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=False
            )
        
        # Softmax and apply attention to values
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        out = attn_weights @ v
        
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
        augment: bool = False,
        deterministic: bool = None
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass with advanced training techniques."""
        # Set deterministic mode based on training flag if not explicitly provided
        if deterministic is None:
            deterministic = not training
            
        # Apply GridMask augmentation during training
        if training and augment and not deterministic:
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
            # MLP block with separate dropout control
            mlp_output = nn.Dense(self.mlp_dim)(mlp_input)
            mlp_output = nn.gelu(mlp_output)
            if not deterministic:
                mlp_output = nn.Dropout(rate=self.dropout_rate)(
                    mlp_output, deterministic=False
                )
            mlp_output = nn.Dense(self.hidden_dim)(mlp_output)
            
            if training and not deterministic:
                mlp_output = nn.Dropout(rate=1 - keep_prob)(
                    mlp_output, deterministic=False
                )
            x = x + mlp_output
        
        # Final layer normalization
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Extract features for consistency loss
        features = x[:, 0]  # Use CLS token features
        
        # Detection heads with enhanced multi-box prediction
        # First branch: predict multiple centers and sizes
        center_size = nn.Dense(512)(features)
        center_size = nn.gelu(center_size)
        if not deterministic:
            center_size = nn.Dropout(rate=self.dropout_rate)(
                center_size, deterministic=False
            )
        # Output shape: [batch_size, 6, 4] for max 6 boxes per image
        center_size = nn.Dense(6 * 4)(center_size)
        center_size = center_size.reshape(-1, 6, 4)  # [cx, cy, w, h] for each box
        
        # Convert center and size to coordinates with constraints for each box
        cx, cy, w, h = jnp.split(center_size, 4, axis=-1)  # Split along last dimension
        
        # Constrain center to [0,1] and size to reasonable bounds for all boxes
        cx = jax.nn.sigmoid(cx)  # Center x in [0,1]
        cy = jax.nn.sigmoid(cy)  # Center y in [0,1]
        w = 0.1 + 0.3 * jax.nn.sigmoid(w)  # Width in [0.1, 0.4]
        h = 0.1 + 0.3 * jax.nn.sigmoid(h)  # Height in [0.1, 0.4]
        
        # Convert to box coordinates ensuring they stay in [0,1] for all boxes
        x1 = jnp.clip(cx - w/2, 0, 1)
        y1 = jnp.clip(cy - h/2, 0, 1)
        x2 = jnp.clip(cx + w/2, 0, 1)
        y2 = jnp.clip(cy + h/2, 0, 1)
        
        # Stack coordinates for all boxes
        boxes = jnp.concatenate([x1, y1, x2, y2], axis=-1)  # Shape: [batch_size, 6, 4]
        
        # Enhanced confidence scores with size penalty for multiple boxes
        # Score head with separate dropout control
        scores = nn.Dense(512)(features)
        scores = nn.gelu(scores)
        if not deterministic:
            scores = nn.Dropout(rate=self.dropout_rate)(
                scores, deterministic=False
            )
        scores = nn.Dense(6)(scores)  # One score per box
        scores = scores.reshape(-1, 6)  # Shape: [batch_size, 6]
        
        # Apply size penalty to confidence scores for all boxes
        box_sizes = (x2 - x1) * (y2 - y1)
        size_penalties = jnp.exp(-5.0 * jnp.abs(box_sizes - 0.15))  # Penalize boxes too far from expected size
        scores = jax.nn.sigmoid(scores)[..., None] * size_penalties  # Add dimension for size penalty multiplication
        scores = scores[..., 0]  # Remove extra dimension after multiplication
        
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
    # Extract only the parameters expected by EnhancedWaldoDetector
    detector_kwargs = {
        'num_heads': model_kwargs.get('model', {}).get('num_heads', 8),
        'num_layers': model_kwargs.get('model', {}).get('num_layers', 8),
        'hidden_dim': model_kwargs.get('model', {}).get('hidden_dim', 512),
        'mlp_dim': model_kwargs.get('model', {}).get('mlp_dim', 2048),
        'dropout_rate': model_kwargs.get('model', {}).get('dropout_rate', 0.3),
        'attention_dropout': model_kwargs.get('model', {}).get('attention_dropout', 0.2),
        'path_dropout': model_kwargs.get('model', {}).get('path_dropout', 0.2),
        'stochastic_depth_rate': model_kwargs.get('model', {}).get('stochastic_depth_rate', 0.2)
    }
    model = EnhancedWaldoDetector(**detector_kwargs)
    
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
    consistency_weight: float = 0.5,
    size_target: float = 0.15,
    size_penalty: float = 5.0
) -> Tuple[jnp.ndarray, Dict]:
    """Compute enhanced loss with center-size prediction and constraints."""
    outputs = state.apply_fn(
        {'params': params},
        batch['image'],
        training=True,
        augment=True,
        rngs={'dropout': rng}
    )
    
    # Handle scores shape consistency
    pred_scores = outputs['scores']  # [batch_size, num_boxes]
    gt_scores = batch['scores']  # [batch_size, num_boxes]
    
    # Ensure scores have same number of boxes
    if pred_scores.shape[1] > gt_scores.shape[1]:
        pred_scores = pred_scores[:, :gt_scores.shape[1]]
    elif pred_scores.shape[1] < gt_scores.shape[1]:
        pad_width = ((0, 0), (0, gt_scores.shape[1] - pred_scores.shape[1]))
        pred_scores = jnp.pad(pred_scores, pad_width, mode='constant')
    
    # Create valid mask from ground truth scores
    valid_mask = jnp.expand_dims(gt_scores > 0, axis=-1)  # Shape: [batch_size, num_boxes, 1]
    
    # Convert ground truth boxes to center-size format with shape handling
    gt_boxes = batch['boxes']  # [batch_size, num_boxes, 4]
    gt_x1, gt_y1, gt_x2, gt_y2 = jnp.split(gt_boxes, 4, axis=-1)  # Each has shape [batch_size, num_boxes, 1]
    gt_w = gt_x2 - gt_x1  # [batch_size, num_boxes, 1]
    gt_h = gt_y2 - gt_y1  # [batch_size, num_boxes, 1]
    gt_cx = (gt_x1 + gt_x2) / 2  # [batch_size, num_boxes, 1]
    gt_cy = (gt_y1 + gt_y2) / 2  # [batch_size, num_boxes, 1]
    
    # Extract predicted boxes with shape handling
    pred_boxes = outputs['boxes']  # [batch_size, num_boxes, 4]
    # Ensure pred_boxes has same number of boxes as gt_boxes
    if pred_boxes.shape[1] > gt_boxes.shape[1]:
        pred_boxes = pred_boxes[:, :gt_boxes.shape[1], :]
    elif pred_boxes.shape[1] < gt_boxes.shape[1]:
        pad_width = ((0, 0), (0, gt_boxes.shape[1] - pred_boxes.shape[1]), (0, 0))
        pred_boxes = jnp.pad(pred_boxes, pad_width, mode='constant')
    
    pred_x1, pred_y1, pred_x2, pred_y2 = jnp.split(pred_boxes, 4, axis=-1)  # Each has shape [batch_size, num_boxes, 1]
    pred_w = pred_x2 - pred_x1  # [batch_size, num_boxes, 1]
    pred_h = pred_y2 - pred_y1  # [batch_size, num_boxes, 1]
    pred_cx = (pred_x1 + pred_x2) / 2  # [batch_size, num_boxes, 1]
    pred_cy = (pred_y1 + pred_y2) / 2  # [batch_size, num_boxes, 1]
    
    # All tensors should already have shape [batch_size, num_boxes, 1]
    
    # Keep valid_mask as is - it's already [batch_size, num_boxes, 1]
    
    # Center point loss with explicit shape handling
    center_loss = jnp.square(pred_cx - gt_cx) + jnp.square(pred_cy - gt_cy)  # [batch_size, num_boxes]
    center_loss = jnp.mean(center_loss * valid_mask)  # valid_mask is already [batch_size, num_boxes]
    
    # Size loss with explicit shape handling
    size_loss = jnp.abs(pred_w - gt_w) + jnp.abs(pred_h - gt_h)  # [batch_size, num_boxes]
    size_loss = jnp.mean(size_loss * valid_mask)  # valid_mask is already [batch_size, num_boxes]
    
    # Box size penalty with explicit shape handling
    pred_size = pred_w * pred_h  # [batch_size, num_boxes]
    size_deviation = jnp.abs(pred_size - size_target)  # [batch_size, num_boxes]
    size_penalty_loss = jnp.exp(size_penalty * size_deviation) * valid_mask  # [batch_size, num_boxes]
    size_penalty_loss = jnp.mean(size_penalty_loss)
    
    # Custom IoU loss implementation for multi-box predictions
    def compute_iou(box1, box2):
        # Compute intersection coordinates
        x1 = jnp.maximum(box1[..., 0], box2[..., 0])
        y1 = jnp.maximum(box1[..., 1], box2[..., 1])
        x2 = jnp.minimum(box1[..., 2], box2[..., 2])
        y2 = jnp.minimum(box1[..., 3], box2[..., 3])
        
        # Compute areas
        intersection = jnp.maximum(0, x2 - x1) * jnp.maximum(0, y2 - y1)
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = box1_area + box2_area - intersection
        
        # Compute IoU with numerical stability
        iou = intersection / (union + 1e-6)
        return iou
    
    # Compute IoU loss
    ious = compute_iou(pred_boxes, gt_boxes)
    giou_loss = (1 - ious) * valid_mask[..., 0]  # Remove last dimension for IoU
    giou_loss = jnp.mean(giou_loss)
    
    # Confidence score loss with shape handling
    score_loss = sigmoid_focal_loss(
        pred_scores,
        gt_scores,
        alpha=0.25,
        gamma=2.0
    ) * valid_mask[..., 0]  # Remove last dim for proper broadcasting
    score_loss = jnp.mean(score_loss)
    
    # Consistency regularization loss
    consistency_loss = ConsistencyRegularization()(outputs['features'])
    
    # Combined loss with weights
    total_loss = (
        2.0 * giou_loss +  # Higher weight for IoU
        0.5 * score_loss +  # Lower weight for confidence
        1.5 * center_loss +  # Higher weight for center accuracy
        1.0 * size_loss +    # Base weight for size
        0.5 * size_penalty_loss +  # Moderate weight for size penalty
        consistency_weight * consistency_loss
    )
    
    metrics = {
        'loss': total_loss,
        'giou_loss': giou_loss,
        'score_loss': score_loss,
        'center_loss': center_loss,
        'size_loss': size_loss,
        'size_penalty_loss': size_penalty_loss,
        'consistency_loss': consistency_loss
    }
    
    return total_loss, metrics

# Import loss functions from original model
from waldo_finder.model import generalized_box_iou_loss, sigmoid_focal_loss
