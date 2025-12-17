"""
Context Optimization (CoOp) Module for Prompt Learning
"""

import torch
import torch.nn as nn


class CoOpModule(nn.Module):
    """
    Context Optimization module for learning soft prompts.
    
    Implements learnable context vectors that can be prepended to
    class embeddings for improved zero-shot and few-shot learning.
    
    Args:
        n_ctx (int): Number of context tokens (default: 16)
        ctx_dim (int): Dimension of context embeddings (default: 512)
        n_classes (int): Number of classes
        ctx_init (str, optional): Initialization string for context tokens
        class_token_position (str): Position of class token - 'end' or 'middle' (default: 'end')
    """
    
    def __init__(
        self,
        n_ctx=16,
        ctx_dim=512,
        n_classes=1000,
        ctx_init=None,
        class_token_position='end'
    ):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.n_classes = n_classes
        self.class_token_position = class_token_position
        
        # Initialize context vectors
        if ctx_init:
            # Initialize from a string (not implemented in this basic version)
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
        
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Class name embeddings (learnable)
        self.class_embeddings = nn.Parameter(torch.empty(n_classes, ctx_dim))
        nn.init.normal_(self.class_embeddings, std=0.02)
        
    def forward(self, class_indices=None):
        """
        Construct prompts with learnable context.
        
        Args:
            class_indices (torch.Tensor, optional): Indices of classes to generate prompts for.
                If None, generates prompts for all classes.
        
        Returns:
            torch.Tensor: Prompt embeddings of shape (B, n_ctx + 1, ctx_dim) or (n_classes, n_ctx + 1, ctx_dim)
        """
        ctx = self.ctx  # (n_ctx, ctx_dim)
        
        if class_indices is not None:
            # Select specific class embeddings
            class_emb = self.class_embeddings[class_indices]  # (B, ctx_dim)
        else:
            # Use all class embeddings
            class_emb = self.class_embeddings  # (n_classes, ctx_dim)
        
        # Expand context for each class
        if class_indices is not None:
            ctx = ctx.unsqueeze(0).expand(class_emb.shape[0], -1, -1)  # (B, n_ctx, ctx_dim)
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_classes, -1, -1)  # (n_classes, n_ctx, ctx_dim)
        
        # Construct prompts based on class token position
        if self.class_token_position == 'end':
            # [ctx_1, ctx_2, ..., ctx_n, class]
            prompts = torch.cat([ctx, class_emb.unsqueeze(1)], dim=1)
        elif self.class_token_position == 'middle':
            # [ctx_1, ..., ctx_n//2, class, ctx_n//2+1, ..., ctx_n]
            half_n_ctx = self.n_ctx // 2
            prompts = torch.cat([
                ctx[:, :half_n_ctx, :],
                class_emb.unsqueeze(1),
                ctx[:, half_n_ctx:, :]
            ], dim=1)
        else:
            raise ValueError(f"Unknown class_token_position: {self.class_token_position}")
        
        return prompts
    
    def get_ctx_vectors(self):
        """Return the learnable context vectors."""
        return self.ctx


class CoOpClassifier(nn.Module):
    """
    Classifier using CoOp-style learnable prompts.
    
    Args:
        feature_dim (int): Dimension of input features
        n_ctx (int): Number of context tokens
        n_classes (int): Number of classes
        class_token_position (str): Position of class token
    """
    
    def __init__(
        self,
        feature_dim=384,
        n_ctx=16,
        n_classes=1000,
        class_token_position='end'
    ):
        super().__init__()
        
        self.coop = CoOpModule(
            n_ctx=n_ctx,
            ctx_dim=feature_dim,
            n_classes=n_classes,
            class_token_position=class_token_position
        )
        
        # Temperature parameter for scaling logits
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)
        
    def forward(self, features):
        """
        Compute classification logits using CoOp prompts.
        
        Args:
            features (torch.Tensor): Input features of shape (B, D)
        
        Returns:
            torch.Tensor: Classification logits of shape (B, n_classes)
        """
        # Generate prompts for all classes
        prompts = self.coop()  # (n_classes, n_ctx + 1, D)
        
        # Pool prompts (mean pooling over context dimension)
        class_features = prompts.mean(dim=1)  # (n_classes, D)
        
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * features @ class_features.t()  # (B, n_classes)
        
        return logits
