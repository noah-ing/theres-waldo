"""
Simple CNN backbone for Waldo detection with squeeze-and-excitation blocks
and improved regularization for better feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    """Enhanced convolution block with SE attention and regularization"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE attention
        self.se = SEBlock(out_channels)
        
        # Residual connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
            
        # Regularization
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First conv
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE attention
        out = self.se(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out, inplace=True)
        
        return out

class SimpleWaldoNet(nn.Module):
    """Enhanced CNN backbone with SE attention and regularization"""
    def __init__(
        self,
        img_size: int = 384,
        embedding_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Initial conv block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Enhanced conv blocks
        self.layer1 = self._make_layer(64, 128, stride=2, dropout=dropout)
        self.layer2 = self._make_layer(128, 256, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 512, stride=2, dropout=dropout)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding projection with dropout
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1
    ) -> nn.Sequential:
        """Create a layer of conv blocks"""
        return ConvBlock(in_channels, out_channels, stride, dropout)
        
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with intermediate feature maps"""
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # Conv blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        # Project to embedding space
        embeddings = self.embedding(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
