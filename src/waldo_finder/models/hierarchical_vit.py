"""
Hierarchical Vision Transformer for scene-level Waldo detection.
Implements multi-scale feature learning with progressive feature pooling.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import List, Tuple, Optional

class HierarchicalViT(nn.Module):
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 32,
        in_channels: int = 3,
        num_layers: int = 12,
        num_heads: int = 16,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        num_levels: int = 3,
        pool_ratios: List[int] = [2, 2, 2]
    ):
        super().__init__()
        
        # Basic parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.pool_ratios = pool_ratios
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings for each level
        self.pos_embeds = nn.ModuleList([
            nn.Embedding(
                self.num_patches // (prod(pool_ratios[:i]) if i > 0 else 1),
                hidden_dim
            )
            for i in range(num_levels)
        ])
        
        # Initialize position embeddings
        for embed in self.pos_embeds:
            nn.init.normal_(embed.weight, std=0.02)
        
        # Transformer layers for each level
        self.transformers = nn.ModuleList([
            TransformerLevel(
                depth=num_layers // num_levels,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_levels)
        ])
        
        # Pooling layers between levels
        self.pools = nn.ModuleList([
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=pool_ratios[i], stride=pool_ratios[i]
            )
            for i in range(num_levels - 1)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # Initial patch embedding
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Process through levels
        features = []
        for level in range(self.num_levels):
            # Add position embeddings
            positions = torch.arange(x.size(1), device=x.device)
            x = x + self.pos_embeds[level](positions).unsqueeze(0)
            
            # Apply transformer blocks
            x = self.transformers[level](x)
            features.append(x)
            
            # Pool if not last level
            if level < self.num_levels - 1:
                x = rearrange(x, 'b (h w) c -> b c h w', h=H//prod(self.pool_ratios[:level]))
                x = self.pools[level](x)
                H, W = H//self.pool_ratios[level], W//self.pool_ratios[level]
                x = rearrange(x, 'b c h w -> b (h w) c')
        
        if return_features:
            return x, features
        return x, None

def prod(lst):
    """Helper function to compute product of a list"""
    result = 1
    for x in lst:
        result *= x
    return result

class TransformerLevel(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float
    ):
        super().__init__()
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
