"""
Hierarchical Vision Transformer for scene-level Waldo detection.
Implements multi-scale feature learning with progressive feature pooling.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from typing import List, Tuple, Optional, Dict

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
        pool_ratios: List[int] = [2, 2, 2],
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        # Basic parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.pool_ratios = pool_ratios
        self.use_checkpoint = use_checkpoint
        
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
        """Forward pass with optional gradient checkpointing"""
        # Initial patch embedding
        x = self.patch_embed(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, H*W, C]
        
        # Process through levels
        features = []
        curr_h, curr_w = H, W
        
        for level in range(self.num_levels):
            # Add position embeddings
            positions = torch.arange(x.size(1), device=x.device)
            x = x + self.pos_embeds[level](positions).unsqueeze(0)
            
            # Apply transformer blocks with explicit checkpoint settings
            if self.use_checkpoint and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0])
                    return custom_forward
                
                x = checkpoint.checkpoint(
                    create_custom_forward(self.transformers[level]),
                    x,
                    use_reentrant=False,  # Recommended setting for better memory efficiency
                    preserve_rng_state=True
                )
            else:
                x = self.transformers[level](x)
            
            # Store features in spatial format
            level_features = rearrange(x, 'b (h w) c -> b c h w', h=curr_h)
            features.append(level_features)
            
            # Pool if not last level
            if level < self.num_levels - 1:
                x = rearrange(x, 'b (h w) c -> b c h w', h=curr_h)
                x = self.pools[level](x)
                curr_h, curr_w = curr_h//self.pool_ratios[level], curr_w//self.pool_ratios[level]
                x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Reshape final output to 4D format (B, C, H, W)
        if not return_features:
            x = rearrange(x, 'b (h w) c -> b c h w', h=curr_h)
        return x, features if return_features else None

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
