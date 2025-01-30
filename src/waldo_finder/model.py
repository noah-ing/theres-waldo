"""Model definitions for Waldo detection."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple
from dataclasses import field

class SceneDetector(nn.Module):
    """Scene-level Waldo detector using pre-trained features."""
    feature_extractor: nn.Module
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    mlp_dim: int = 2048
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.3
    scale_bins: list = field(default_factory=lambda: [0.05, 0.075, 0.1, 0.125, 0.15])
    scale_aware: dict = field(default_factory=lambda: {
        'enabled': True,
        'size_weights': [3.0, 2.0, 1.0, 2.0, 3.0],
        'adaptive_loss': True,
        'size_normalization': True,
    })

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for scene detection.
        
        Args:
            x: Input image of shape (batch_size, height, width, channels)
            train: Whether in training mode
            
        Returns:
            boxes: Predicted boxes of shape (batch_size, 4) [x1, y1, x2, y2]
            confidence: Confidence scores of shape (batch_size, 1)
        """
        # Extract features using pre-trained stage 1 model
        features = self.feature_extractor(x, train=False)  # Don't train feature extractor
        
        # Project and reshape features for transformer
        x = nn.Dense(self.hidden_dim * 4)(features)  # Project to larger dimension
        x = x.reshape(-1, 4, self.hidden_dim)  # Reshape to (batch_size, seq_len, hidden_dim)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        
        # Multi-head self-attention layers for scene understanding
        for _ in range(self.num_layers):
            # Attention block
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout_rate,
            )(x, x, deterministic=not train)
            x = x + attn_output
            x = nn.LayerNorm()(x)
            
            # MLP block with residual connection
            y = nn.Dense(self.mlp_dim)(x)
            y = nn.gelu(y)
            y = nn.Dense(self.hidden_dim)(y)
            x = x + y
            x = nn.LayerNorm()(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Box coordinate prediction with parameterization
        box_features = nn.Dense(4)(x)  # Base coordinates
        offset_features = nn.Dense(4)(x)  # Offset values
        
        # Get base coordinates (x1,y1) and compute x2,y2 as offsets
        x1y1 = jax.nn.sigmoid(box_features[..., :2])  # Force [0,1]
        offsets = jax.nn.sigmoid(offset_features[..., 2:])  # Force positive
        
        # Compute x2,y2 as offset from x1,y1
        x2y2 = x1y1 + offsets * (1 - x1y1)  # Ensures x2>x1, y2>y1 while staying in [0,1]
        
        # Combine into final box coordinates
        boxes = jnp.concatenate([x1y1, x2y2], axis=-1)  # Shape: (batch_size, 4)
        
        # Confidence prediction head
        confidence = nn.Dense(1)(x)
        
        return boxes, confidence

class WaldoClassifier(nn.Module):
    """Binary classifier for Waldo detection (Stage 1)."""
    num_heads: int = 8
    num_layers: int = 6
    hidden_dim: int = 256
    mlp_dim: int = 1024
    patch_size: int = 16
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass for binary classification.
        
        Args:
            x: Input image of shape (batch_size, height, width, channels)
            train: Whether in training mode
            
        Returns:
            logits: Classification logits of shape (batch_size, 1)
        """
        # Patch embedding
        batch_size, height, width, channels = x.shape
        patch_height, patch_width = self.patch_size, self.patch_size
        num_patches = (height // patch_height) * (width // patch_width)
        
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(patch_height, patch_width),
            strides=(patch_height, patch_width),
            name='embedding',
        )(x)
        
        x = x.reshape(batch_size, -1, self.hidden_dim)
        
        # Add position embeddings
        positions = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.hidden_dim),
        )
        x = x + positions
        
        # Transformer encoder
        for _ in range(self.num_layers):
            # Attention block
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
            )(x, x, deterministic=not train)
            x = x + attn_output
            x = nn.LayerNorm()(x)
            
            # MLP block with residual connection
            y = nn.Dense(self.mlp_dim)(x)
            y = nn.gelu(y)
            y = nn.Dense(self.hidden_dim)(y)
            x = x + y
            x = nn.LayerNorm()(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Classification head
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(1)(x)
        
        return x
