"""
Context-aware attention mechanisms for scene-level Waldo detection.
Implements both global scene attention and local region focus capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple

class ContextAwareAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        window_size: Optional[int] = None
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        
        # Projections for Q, K, V
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Context projection
        self.to_context = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Relative position bias
        if window_size is not None:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            
            # Initialize relative position bias
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.to_q(x)
        k = self.to_k(x if context is None else context)
        v = self.to_v(x if context is None else context)
        
        # Split heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Add relative position bias if using windowed attention
        if self.window_size is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size * self.window_size,
                self.window_size * self.window_size,
                -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots = dots + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn = F.softmax(dots, dim=-1)
        
        # Compute weighted values
        out = torch.matmul(attn, v)
        
        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Project to output
        out = self.to_out(out)
        
        # Add context information
        if context is not None:
            context_features = self.to_context(context)
            out = out + context_features
        
        return out

class GlobalLocalAttention(nn.Module):
    """Combines global scene attention with local region focus"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        window_size: int = 7,
        global_ratio: float = 0.5
    ):
        super().__init__()
        self.global_ratio = global_ratio
        
        # Global attention
        self.global_attn = ContextAwareAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Local windowed attention
        self.local_attn = ContextAwareAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            window_size=window_size
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Global attention
        global_out = self.global_attn(x, context)
        
        # Local attention (windowed)
        local_out = self.local_attn(x)
        
        # Fuse global and local features
        fused = torch.cat([global_out, local_out], dim=-1)
        out = self.fusion(fused)
        
        return out

class CrossScaleAttention(nn.Module):
    """Attention mechanism for cross-scale feature interaction"""
    def __init__(
        self,
        dim: int,
        num_scales: int = 3,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_scales = num_scales
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_scales)
        ])
        
        # Cross-scale attention
        self.cross_attn = ContextAwareAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_scales, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        features: list[torch.Tensor]
    ) -> torch.Tensor:
        assert len(features) == self.num_scales, \
            f"Expected {self.num_scales} scale features, got {len(features)}"
        
        # Project each scale
        projected = [
            proj(feat)
            for proj, feat in zip(self.scale_projections, features)
        ]
        
        # Cross-scale attention
        attended = []
        for i in range(self.num_scales):
            # Use current scale as query, others as context
            context = torch.cat([
                feat for j, feat in enumerate(projected) if j != i
            ], dim=1)
            out = self.cross_attn(projected[i], context)
            attended.append(out)
        
        # Fuse all scales
        fused = torch.cat(attended, dim=-1)
        out = self.fusion(fused)
        
        return out
