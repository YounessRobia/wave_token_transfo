"""
Quick Example: Wavelet Transformer on a Single Image
Demonstrates basic usage and feature extraction.
"""

import torch
from wno.model import create_wavelet_tiny
import matplotlib.pyplot as plt
import numpy as np


def demo_basic_usage():
    """Demonstrate basic model usage"""
    
    print("="*60)
    print("Wavelet Transformer - Quick Demo")
    print("="*60)
    
    # 1. Create model
    print("\n1. Creating Wavelet Transformer (Tiny)...")
    model = create_wavelet_tiny(
        image_size=(32, 32),
        in_channels=3,
        num_classes=10,
        n_levels=3
    )
    
    print(f"   Model created successfully!")
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   FLOPs: {model.get_flops() / 1e9:.2f}G")
    
    # 2. Create sample image
    print("\n2. Creating sample image...")
    batch_size = 2
    image = torch.randn(batch_size, 3, 32, 32)
    print(f"   Image shape: {image.shape}")
    
    # 3. Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(image)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Predicted classes: {logits.argmax(dim=1)}")
    
    # 4. Extract features
    print("\n4. Extracting multi-scale features...")
    with torch.no_grad():
        features, metadata = model.forward_features(image)
    
    print(f"   Features shape: {features.shape}")
    print(f"   Number of tokens: {features.shape[1]}")
    print(f"   Token distribution by scale:")
    
    scales = metadata['scales'].cpu().numpy()
    for level in range(model.n_levels):
        n_tokens = (scales == level).sum()
        print(f"      Level {level}: {n_tokens} tokens")
    
    # 5. Visualize token structure
    print("\n5. Visualizing token structure...")
    visualize_token_structure(features, metadata, save_path='results/quick_demo_tokens.png')
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Train on CIFAR-10: cd experiments && python cifar10_classification.py")
    print("  - Run full tests: python test_installation.py")
    print("  - Read docs: cat README.md")


def visualize_token_structure(features, metadata, save_path='tokens.png'):
    """Visualize the multi-scale token structure"""
    
    # Extract metadata
    scales = metadata['scales'].cpu().numpy()
    orientations = metadata['orientations'].cpu().numpy()
    positions = metadata['positions'].cpu().numpy()
    
    # Compute token norms (as a measure of importance)
    token_norms = torch.norm(features[0], dim=-1).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    n_levels = metadata['scales'].unique().numel()
    
    # Plot tokens by scale
    for level in range(n_levels):
        mask = scales == level
        if mask.sum() == 0:
            continue
        
        level_norms = token_norms[mask]
        level_pos = positions[mask]
        
        # Get grid size
        if len(level_pos) == 0:
            continue
        
        h_coords = level_pos[:, 0]
        w_coords = level_pos[:, 1]
        
        # Scatter plot
        axes[level].scatter(w_coords, h_coords, c=level_norms, 
                           s=100, cmap='hot', alpha=0.6)
        axes[level].set_title(f'Scale {level}\n({mask.sum()} tokens)', fontsize=12)
        axes[level].set_xlabel('Width')
        axes[level].set_ylabel('Height')
        axes[level].invert_yaxis()
        axes[level].grid(True, alpha=0.3)
        axes[level].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"   Token visualization saved to {save_path}")


def demo_comparison():
    """Quick comparison of different configurations"""
    
    print("\n" + "="*60)
    print("Model Size Comparison")
    print("="*60)
    
    configs = [
        ('Tiny', lambda: create_wavelet_tiny(image_size=(32,32), num_classes=10)),
    ]
    
    for name, create_fn in configs:
        model = create_fn()
        print(f"\n{name}:")
        print(f"  Parameters: {model.get_num_params():,}")
        print(f"  Embed dim: {model.embed_dim}")
        print(f"  Depth: {model.depth}")
        
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        print(f"  Output shape: {out.shape}")


if __name__ == '__main__':
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run demos
    demo_basic_usage()
    demo_comparison()
    
    print("\n✅ Quick example complete!")





