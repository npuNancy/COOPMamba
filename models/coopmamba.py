"""
COOPMamba: Combining Mamba State Space Models with Context Optimization
"""

import torch
import torch.nn as nn
from einops import rearrange

from .mamba_block import MambaBlock
from .coop_module import CoOpModule


class COOPMamba(nn.Module):
    """
    COOPMamba: A vision model combining Mamba state space models with CoOp-style prompt learning.
    
    This model uses Mamba blocks for efficient visual feature extraction and CoOp modules
    for learnable prompt-based classification.
    
    Args:
        img_size (int): Input image size (default: 224)
        patch_size (int): Patch size for tokenization (default: 16)
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1000)
        embed_dim (int): Embedding dimension (default: 384)
        depth (int): Number of Mamba blocks (default: 12)
        d_state (int): State space dimension for Mamba (default: 16)
        n_ctx (int): Number of context tokens for CoOp (default: 16)
        class_token_position (str): Position of class token in prompt (default: 'end')
        use_coop (bool): Whether to use CoOp classification head (default: True)
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
        n_ctx=16,
        class_token_position='end',
        use_coop=True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_coop = use_coop
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Mamba encoder blocks
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state=d_state)
            for _ in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        if use_coop:
            # CoOp-based classification
            self.coop = CoOpModule(
                n_ctx=n_ctx,
                ctx_dim=embed_dim,
                n_classes=num_classes,
                class_token_position=class_token_position
            )
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        else:
            # Standard linear classification
            self.head = nn.Linear(embed_dim, num_classes)
        
    def forward_features(self, x):
        """
        Extract features using Mamba blocks.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Features of shape (B, D)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, D, H', W')
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, L, D)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize and pool
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling: (B, D)
        
        return x
    
    def forward(self, x):
        """
        Forward pass for classification.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        # Extract features
        features = self.forward_features(x)
        
        if self.use_coop:
            # CoOp-based classification
            prompts = self.coop()  # (n_classes, n_ctx + 1, D)
            class_features = prompts.mean(dim=1)  # (n_classes, D)
            
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * features @ class_features.t()
        else:
            # Standard linear classification
            logits = self.head(features)
        
        return logits
    
    def get_learnable_params(self):
        """
        Get learnable parameters for optimization.
        
        Returns:
            list: List of parameter groups
        """
        if self.use_coop:
            # Separate backbone and CoOp parameters
            backbone_params = []
            coop_params = []
            
            for name, param in self.named_parameters():
                if 'coop' in name or 'logit_scale' in name:
                    coop_params.append(param)
                else:
                    backbone_params.append(param)
            
            return [
                {'params': backbone_params, 'lr_scale': 1.0},
                {'params': coop_params, 'lr_scale': 1.0}
            ]
        else:
            return [{'params': self.parameters(), 'lr_scale': 1.0}]


def create_coopmamba(
    model_size='base',
    num_classes=1000,
    img_size=224,
    use_coop=True,
    **kwargs
):
    """
    Create a COOPMamba model with predefined configurations.
    
    Args:
        model_size (str): Model size - 'tiny', 'small', 'base', or 'large'
        num_classes (int): Number of output classes
        img_size (int): Input image size
        use_coop (bool): Whether to use CoOp classification
        **kwargs: Additional arguments to pass to COOPMamba
    
    Returns:
        COOPMamba: Configured model instance
    """
    configs = {
        'tiny': {'embed_dim': 192, 'depth': 6, 'd_state': 8},
        'small': {'embed_dim': 384, 'depth': 12, 'd_state': 16},
        'base': {'embed_dim': 512, 'depth': 18, 'd_state': 16},
        'large': {'embed_dim': 768, 'depth': 24, 'd_state': 32},
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(configs.keys())}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    return COOPMamba(
        img_size=img_size,
        num_classes=num_classes,
        use_coop=use_coop,
        **config
    )
