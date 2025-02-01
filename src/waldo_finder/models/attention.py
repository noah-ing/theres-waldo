"""Attention mechanisms for Waldo detection, combining global and local context."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class GlobalLocalAttention(nn.Module):
    """Attention module combining global and local context with efficient memory usage."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        window_size: int = 3,  # Reduced to match feature map size
        use_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        
        # Linear projections
        self.qkv = nn.Linear(dim, 3 * num_heads * head_dim, bias=use_bias)
        self.proj = nn.Linear(num_heads * head_dim, dim, bias=use_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Initialize relative position indices
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Initialize weights
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def _reshape_for_attention(
        self,
        x: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape input tensor for attention computation."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        return attn, v, B
        
    def _compute_output_shape(self, x: torch.Tensor) -> Tuple[int, int]:
        """Compute spatial dimensions using PyTorch operations."""
        if len(x.shape) == 4:
            _, _, H, W = x.shape
        else:
            _, N, _ = x.shape
            # Use floor_divide and sqrt for shape computation
            H = W = int(N ** 0.5)  # N is always a perfect square in our case
        return H, W
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with shape-preserving operations."""
        # Handle 4D input (B, C, H, W)
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        else:
            B, N, C = x.shape
            H, W = self._compute_output_shape(x)
        
        # Compute attention
        attn, v, B = self._reshape_for_attention(x, H, W)
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Reshape back to 4D
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        return x
        
    def extra_repr(self) -> str:
        """String representation of module."""
        return (f'num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, '
                f'window_size={self.window_size}, '
                f'scale={self.scale:.2f}')
