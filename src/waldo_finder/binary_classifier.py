"""Binary classifier for learning Waldo's appearance."""

from typing import Any, Dict
import flax.linen as nn
import jax.numpy as jnp

class WaldoClassifier(nn.Module):
    """Simple Vision Transformer for binary classification."""
    
    num_heads: int = 8
    num_layers: int = 6
    hidden_dim: int = 256
    mlp_dim: int = 1024
    dropout_rate: float = 0.2
    attention_dropout_rate: float = 0.2
    image_size: tuple = (128, 128)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass for binary classification."""
        # Simple patch embedding (16x16)
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(16, 16),
            strides=(16, 16),
            use_bias=False,
            name='patch_embedding'
        )(x)
        
        # Reshape to sequence
        batch_size, h, w, c = x.shape
        x = x.reshape(batch_size, h * w, c)
        
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
        
        # Transformer encoder
        for _ in range(self.num_layers):
            # Attention block
            y = nn.LayerNorm(epsilon=1e-6)(x)
            attention = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout_rate if training else 0.0,
            )
            y = attention(y, y, deterministic=not training)
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
        
        # Global average pooling
        x = jnp.mean(x[:, 1:], axis=1)  # Skip class token
        
        # Classification head
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.2)(x, deterministic=not training)
        x = nn.Dense(1)(x)  # Single output for binary classification
        
        return x  # Return logits
