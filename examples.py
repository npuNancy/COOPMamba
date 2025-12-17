"""
Example usage of COOPMamba model
"""

import torch
from models import create_coopmamba


def example_basic_usage():
    """
    Basic example of creating and using COOPMamba.
    """
    print("=" * 60)
    print("Example 1: Basic Model Creation and Forward Pass")
    print("=" * 60)
    
    # Create a small model
    model = create_coopmamba(
        model_size='small',
        num_classes=10,
        img_size=224,
        use_coop=True
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output (logits): {logits}")
    
    # Get predictions
    probs = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(probs, dim=1)
    
    print(f"Predicted classes: {predicted_classes}")
    print()


def example_feature_extraction():
    """
    Example of using COOPMamba for feature extraction.
    """
    print("=" * 60)
    print("Example 2: Feature Extraction")
    print("=" * 60)
    
    # Create model
    model = create_coopmamba(
        model_size='tiny',
        num_classes=100,
        img_size=224,
        use_coop=False  # Use standard linear head
    )
    
    # Extract features
    x = torch.randn(4, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        features = model.forward_features(x)
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")
    print()


def example_model_sizes():
    """
    Example showing different model sizes.
    """
    print("=" * 60)
    print("Example 3: Different Model Sizes")
    print("=" * 60)
    
    sizes = ['tiny', 'small', 'base']
    
    for size in sizes:
        model = create_coopmamba(
            model_size=size,
            num_classes=1000,
            img_size=224,
            use_coop=True
        )
        
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{size.capitalize()}: {num_params:.2f}M parameters")
    
    print()


def example_coop_prompts():
    """
    Example showing CoOp prompt learning.
    """
    print("=" * 60)
    print("Example 4: CoOp Learnable Prompts")
    print("=" * 60)
    
    # Create model with CoOp
    model = create_coopmamba(
        model_size='small',
        num_classes=5,
        img_size=224,
        use_coop=True,
        n_ctx=8  # 8 context tokens
    )
    
    print("Model created with CoOp module")
    print(f"Number of context tokens: {model.coop.n_ctx}")
    print(f"Context dimension: {model.coop.ctx_dim}")
    
    # Get prompt embeddings
    with torch.no_grad():
        prompts = model.coop()  # Get prompts for all classes
    
    print(f"Prompt shape (all classes): {prompts.shape}")
    print(f"  - {prompts.shape[0]} classes")
    print(f"  - {prompts.shape[1]} tokens (context + class token)")
    print(f"  - {prompts.shape[2]} embedding dimension")
    print()


if __name__ == '__main__':
    print("\n")
    print("*" * 60)
    print("COOPMamba Examples")
    print("*" * 60)
    print()
    
    example_basic_usage()
    example_feature_extraction()
    example_model_sizes()
    example_coop_prompts()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
