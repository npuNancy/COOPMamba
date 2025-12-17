"""
Mamba Block Implementation for Vision Tasks
"""

import torch
import torch.nn as nn
import warnings
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:
    warnings.warn("mamba_ssm not installed. Using placeholder implementation.", ImportWarning)
    Mamba = None


class MambaBlock(nn.Module):
    """
    Mamba block for processing visual features with state space models.
    
    Args:
        dim (int): Input feature dimension
        d_state (int): State space dimension (default: 16)
        d_conv (int): Convolution kernel size (default: 4)
        expand (int): Expansion factor for inner dimension (default: 2)
    """
    
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        if Mamba is not None:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Placeholder: simple linear layer if mamba_ssm not available
            self.mamba = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Linear(dim * expand, dim)
            )
    
    def forward(self, x):
        """
        Forward pass through Mamba block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D) or (B, H, W, D)
            
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # Store original shape
        original_shape = x.shape
        
        # Reshape if needed (from (B, H, W, D) to (B, L, D))
        if len(x.shape) == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> b (h w) d')
        
        # Apply normalization and mamba
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = x + residual
        
        # Reshape back if needed
        if len(original_shape) == 4:
            x = rearrange(x, 'b (h w) d -> b h w d', h=H, w=W)
        
        return x


class VisionMamba(nn.Module):
    """
    Vision Mamba model for image classification.
    
    Args:
        img_size (int): Input image size (default: 224)
        patch_size (int): Patch size (default: 16)
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1000)
        embed_dim (int): Embedding dimension (default: 384)
        depth (int): Number of Mamba blocks (default: 12)
        d_state (int): State space dimension (default: 16)
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        d_state=16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state=d_state)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H', W')
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, L, D)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling and classification
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, D)
        x = self.head(x)  # (B, num_classes)
        
        return x
