"""
Compare Wavelet Transformer with Standard Vision Transformer
Analysis and visualization tools.
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from wno.model import create_wavelet_tiny, create_wavelet_small


class SimpleViT(nn.Module):
    """
    Simple Vision Transformer baseline for comparison
    """
    def __init__(self, 
                 image_size=32,
                 patch_size=4,
                 num_classes=10,
                 embed_dim=192,
                 depth=6,
                 num_heads=3,
                 mlp_ratio=4.0):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Patch embedding: [B, 3, 32, 32] -> [B, D, 8, 8] -> [B, 64, D]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling and classification
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        
        return x


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_size=(1, 3, 32, 32), device='cuda', num_runs=100):
    """Measure average inference time"""
    model.eval()
    model = model.to(device)
    
    x = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # Convert to ms
    
    return avg_time


def analyze_attention_patterns(model, test_loader, device, num_samples=5):
    """
    Analyze attention patterns in Wavelet Transformer
    """
    model.eval()
    model = model.to(device)
    
    # Get sample images
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    
    # Forward pass through model to get features
    with torch.no_grad():
        features, metadata = model.forward_features(images)
    
    # Analyze token statistics by scale
    scales = metadata['scales'].cpu().numpy()
    orientations = metadata['orientations'].cpu().numpy()
    
    # Compute feature magnitudes per scale
    feature_norms = torch.norm(features, dim=-1).mean(dim=0).cpu().numpy()  # [N]
    
    # Group by scale
    n_levels = model.n_levels
    scale_stats = []
    
    for level in range(n_levels):
        mask = scales == level
        if mask.sum() > 0:
            scale_norms = feature_norms[mask]
            scale_stats.append({
                'level': level,
                'mean_norm': scale_norms.mean(),
                'std_norm': scale_norms.std(),
                'num_tokens': mask.sum()
            })
    
    return scale_stats


def plot_model_comparison():
    """Create comprehensive comparison plots"""
    
    # Model configurations
    configs = [
        ('Wavelet-Tiny', lambda: create_wavelet_tiny(image_size=(32, 32), num_classes=10)),
        ('Wavelet-Small', lambda: create_wavelet_small(image_size=(32, 32), num_classes=10)),
        ('ViT-Tiny', lambda: SimpleViT(embed_dim=96, depth=4)),
        ('ViT-Small', lambda: SimpleViT(embed_dim=192, depth=6)),
    ]
    
    # Metrics to compare
    param_counts = []
    inference_times = []
    model_names = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for name, create_fn in configs:
        print(f"\nAnalyzing {name}...")
        model = create_fn()
        
        # Count parameters
        params = count_parameters(model)
        param_counts.append(params / 1e6)  # Convert to millions
        
        # Measure inference time
        inf_time = measure_inference_time(model, device=device)
        inference_times.append(inf_time)
        
        model_names.append(name)
        
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        print(f"  Inference time: {inf_time:.2f} ms")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameter count
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    axes[0].bar(model_names, param_counts, color=colors)
    axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[0].set_title('Model Size Comparison', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Inference time
    axes[1].bar(model_names, inference_times, color=colors)
    axes[1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[1].set_title('Inference Speed Comparison', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png', dpi=150)
    plt.close()
    
    print("\n✅ Comparison plot saved to ../results/model_comparison.png")


def visualize_wavelet_features(model, test_loader, device):
    """
    Visualize learned wavelet features
    """
    model.eval()
    model = model.to(device)
    
    # Get sample image
    images, labels = next(iter(test_loader))
    img = images[0:1].to(device)
    
    # Forward through tokenizer only
    with torch.no_grad():
        tokens, metadata = model.tokenizer(img)
    
    # Visualize token magnitudes by scale
    token_norms = torch.norm(tokens[0], dim=-1).cpu().numpy()  # [N]
    
    scales = metadata['scales'].cpu().numpy()
    positions = metadata['positions'].cpu().numpy()
    
    n_levels = model.n_levels
    
    fig, axes = plt.subplots(1, n_levels, figsize=(15, 5))
    
    for level in range(n_levels):
        mask = scales == level
        if mask.sum() == 0:
            continue
        
        level_norms = token_norms[mask]
        level_pos = positions[mask]
        
        # Get grid size
        h_max = level_pos[:, 0].max() + 1
        w_max = level_pos[:, 1].max() + 1
        
        # Create 2D grid
        grid = np.zeros((h_max, w_max))
        for i, (h, w) in enumerate(level_pos):
            if h < h_max and w < w_max:
                grid[h, w] = level_norms[i]
        
        # Plot
        im = axes[level].imshow(grid, cmap='hot', interpolation='nearest')
        axes[level].set_title(f'Scale {level}', fontsize=12)
        axes[level].axis('off')
        plt.colorbar(im, ax=axes[level], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('../results/wavelet_features.png', dpi=150)
    plt.close()
    
    print("✅ Wavelet features visualization saved to ../results/wavelet_features.png")


def main():
    """Run all comparison analyses"""
    
    print("="*60)
    print("Wavelet Transformer Analysis and Comparison")
    print("="*60)
    
    # Model comparison
    print("\n1. Comparing model architectures...")
    plot_model_comparison()
    
    # Load test data for visualizations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(testset, batch_size=16, shuffle=True)
    
    # Create model for analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_wavelet_tiny(image_size=(32, 32), num_classes=10, n_levels=3)
    
    # Visualize wavelet features
    print("\n2. Visualizing wavelet features...")
    visualize_wavelet_features(model, test_loader, device)
    
    # Analyze attention patterns
    print("\n3. Analyzing attention patterns...")
    scale_stats = analyze_attention_patterns(model, test_loader, device)
    
    print("\nToken statistics by scale:")
    for stats in scale_stats:
        print(f"  Level {stats['level']}: {stats['num_tokens']} tokens, "
              f"mean norm = {stats['mean_norm']:.3f} ± {stats['std_norm']:.3f}")
    
    print("\n✅ Analysis complete! Check ../results/ for visualizations.")


if __name__ == '__main__':
    main()





