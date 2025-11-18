"""
Wavelet Transformer Feature Visualization Script
==================================================
Loads the best.pth model and visualizes key features of the wavelet transformer
on 10 random CIFAR10 test images.

Visualizations include:
1. Original images with predictions
2. Multi-scale wavelet decomposition (approximation and detail coefficients)
3. Token embeddings at different scales
4. Attention maps from W-MSA blocks
5. Feature maps from different transformer layers
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# Add parent directory to path to import wno modules
sys.path.append(str(Path(__file__).parent))

from wno.model import WaveletTransformer, create_wavelet_tiny, create_wavelet_small
from wno.wavelets_2d import WaveletTransform2D


# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 normalization parameters
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def load_model(checkpoint_path, device='cuda'):
    """Load the best model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer model configuration from state dict
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Detect embed_dim from state dict
    embed_dim = None
    for key in state_dict.keys():
        if 'tokenizer.proj_approx.0.weight' in key:
            embed_dim = state_dict[key].shape[0]
            break

    # Detect number of blocks
    blocks = [k for k in state_dict.keys() if k.startswith('blocks.')]
    depth = max([int(k.split('.')[1]) for k in blocks]) + 1 if blocks else 4

    # Detect n_levels from pos_encoder.scale_embed.weight shape
    n_levels = None
    for key in state_dict.keys():
        if 'pos_encoder.scale_embed.weight' in key:
            n_levels = state_dict[key].shape[0]
            break
    if n_levels is None:
        n_levels = 3  # default

    print(f"Detected configuration from checkpoint:")
    print(f"  - Embedding dimension: {embed_dim}")
    print(f"  - Depth (num blocks): {depth}")
    print(f"  - Wavelet levels: {n_levels}")

    # Create model using factory function based on detected config
    if embed_dim == 96 and depth == 4:
        print("  - Model type: Tiny")
        model = create_wavelet_tiny(
            image_size=(32, 32),
            in_channels=3,
            num_classes=10,
            n_levels=n_levels,
            wavelet='db4',
            use_kernel=True
        )
    elif embed_dim == 192 and depth == 6:
        print("  - Model type: Small")
        model = create_wavelet_small(
            image_size=(32, 32),
            in_channels=3,
            num_classes=10,
            n_levels=n_levels,
            wavelet='db4',
            use_kernel=True
        )
    else:
        # Default fallback
        print("  - Model type: Custom")
        model = WaveletTransformer(
            image_size=(32, 32),
            in_channels=3,
            num_classes=10,
            embed_dim=embed_dim or 96,
            depth=depth,
            num_heads=3,
            n_levels=n_levels,
            wavelet='db4',
            use_kernel=True
        )

    # Load state dict
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Print model info
    if 'epoch' in checkpoint:
        print(f"\nLoaded model from epoch {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    config = {
        'embed_dim': embed_dim or 96,
        'depth': depth,
        'image_size': (32, 32),
        'num_classes': 10,
        'n_levels': n_levels
    }

    return model, config


def get_test_loader(batch_size=10, num_samples=10):
    """Get CIFAR-10 test data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./experiments/data',
        train=False,
        download=True,
        transform=transform
    )

    # Sample random indices
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(test_dataset, indices)

    test_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Denormalize image tensor for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def extract_wavelet_features(model, images, device):
    """Extract wavelet coefficients and intermediate features."""
    with torch.no_grad():
        images = images.to(device)

        # Get wavelet transform using the tokenizer's wavelet transform object
        wavelet_coeffs = model.tokenizer.wt.forward(images)

        # Get tokens and metadata
        tokens, metadata = model.tokenizer(images)

        # Add positional encoding
        tokens = model.pos_encoder(tokens, metadata)

        # Extract features from each transformer block
        block_features = []
        x = tokens
        for block in model.blocks:
            x = block(x, metadata)
            block_features.append(x.clone())

        # Final normalization
        x = model.norm(x)

        # Global pooling (mimicking model's forward method)
        if model.pool_type == 'mean':
            pooled = x.mean(dim=1)  # [B, D]
        elif model.pool_type == 'max':
            pooled = x.max(dim=1)[0]  # [B, D]
        else:
            pooled = x[:, 0]  # [B, D]

        # Get final predictions
        logits = model.head(pooled)
        predictions = torch.softmax(logits, dim=-1)

    return {
        'wavelet_coeffs': wavelet_coeffs,
        'tokens': tokens,
        'metadata': metadata,
        'block_features': block_features,
        'predictions': predictions,
        'logits': logits
    }


def plot_wavelet_decomposition(images, wavelet_coeffs, predictions, save_path):
    """Visualize multi-scale wavelet decomposition for each image."""
    num_images = images.shape[0]
    n_levels = len(wavelet_coeffs['approximations'])

    # Create figure with subplots
    fig = plt.figure(figsize=(20, num_images * 3))
    gs = GridSpec(num_images, n_levels + 2, figure=fig, hspace=0.3, wspace=0.3)

    for idx in range(num_images):
        # Original image
        ax = fig.add_subplot(gs[idx, 0])
        img = denormalize(images[idx].cpu()).permute(1, 2, 0).numpy()
        ax.imshow(img)

        # Get prediction
        pred_class = predictions[idx].argmax().item()
        pred_conf = predictions[idx].max().item()
        ax.set_title(f'Original\n{CIFAR10_CLASSES[pred_class]}\n({pred_conf:.2%})',
                     fontsize=10)
        ax.axis('off')

        # Wavelet coefficients at each level
        for level in range(n_levels):
            ax = fig.add_subplot(gs[idx, level + 1])

            # Get coefficients for this level and image
            approx = wavelet_coeffs['approximations'][level][idx].cpu()
            horiz = wavelet_coeffs['horizontal'][level][idx].cpu()
            vert = wavelet_coeffs['vertical'][level][idx].cpu()
            diag = wavelet_coeffs['diagonal'][level][idx].cpu()

            # Combine detail coefficients (RGB channels)
            if approx.dim() == 3:  # [C, H, W]
                approx_vis = approx.mean(dim=0)  # Average across channels
                detail_vis = (horiz.abs() + vert.abs() + diag.abs()).mean(dim=0)
            else:
                approx_vis = approx
                detail_vis = horiz.abs() + vert.abs() + diag.abs()

            # Create composite visualization
            h, w = approx_vis.shape
            composite = torch.zeros(h * 2, w * 2)
            composite[:h, :w] = approx_vis
            composite[:h, w:] = horiz.mean(dim=0) if horiz.dim() == 3 else horiz
            composite[h:, :w] = vert.mean(dim=0) if vert.dim() == 3 else vert
            composite[h:, w:] = diag.mean(dim=0) if diag.dim() == 3 else diag

            im = ax.imshow(composite.numpy(), cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Level {level}\n({h}×{w})', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Show detail energy distribution
        ax = fig.add_subplot(gs[idx, n_levels + 1])
        energies = []
        labels = []
        for level in range(n_levels):
            h_energy = wavelet_coeffs['horizontal'][level][idx].abs().mean().item()
            v_energy = wavelet_coeffs['vertical'][level][idx].abs().mean().item()
            d_energy = wavelet_coeffs['diagonal'][level][idx].abs().mean().item()
            energies.extend([h_energy, v_energy, d_energy])
            labels.extend([f'L{level}-H', f'L{level}-V', f'L{level}-D'])

        ax.barh(range(len(energies)), energies)
        ax.set_yticks(range(len(energies)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Mean Magnitude', fontsize=9)
        ax.set_title('Detail Energy', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Multi-Scale Wavelet Decomposition', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved wavelet decomposition to {save_path}")
    plt.close()


def plot_token_embeddings(metadata, tokens, predictions, save_path):
    """Visualize token embeddings colored by scale and orientation."""
    num_images = min(4, tokens.shape[0])  # Show first 4 images
    num_tokens_per_image = tokens.shape[1]

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 5, 10))
    if num_images == 1:
        axes = axes.reshape(2, 1)

    for idx in range(num_images):
        # Get token features for this image
        token_features = tokens[idx].cpu().numpy()  # [num_tokens, embed_dim]

        # Use t-SNE or PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        token_2d = pca.fit_transform(token_features)

        # Get metadata for coloring (metadata is not batched, it's shared across batch)
        # Since metadata is shared, we use the same for all images
        scales = metadata['scales'].cpu().numpy()
        orientations = metadata['orientations'].cpu().numpy()

        # Plot 1: Colored by scale
        ax = axes[0, idx]
        scatter = ax.scatter(token_2d[:, 0], token_2d[:, 1],
                            c=scales, cmap='viridis',
                            s=20, alpha=0.6)
        ax.set_title(f'Image {idx+1}: Tokens by Scale\n{CIFAR10_CLASSES[predictions[idx].argmax()]}',
                     fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax, label='Scale Level')
        ax.grid(alpha=0.3)

        # Plot 2: Colored by orientation
        ax = axes[1, idx]
        scatter = ax.scatter(token_2d[:, 0], token_2d[:, 1],
                            c=orientations, cmap='Set1',
                            s=20, alpha=0.6)
        ax.set_title(f'Tokens by Orientation', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        cbar = plt.colorbar(scatter, ax=ax, label='Orientation', ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['Approx', 'Horiz', 'Vert', 'Diag'])
        ax.grid(alpha=0.3)

    plt.suptitle('Token Embedding Visualization (PCA)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved token embeddings to {save_path}")
    plt.close()


def plot_attention_patterns(model, images, metadata, save_path, device):
    """Visualize attention patterns from W-MSA blocks."""
    num_images = min(3, images.shape[0])
    num_blocks = len(model.blocks)

    fig = plt.figure(figsize=(num_blocks * 4, num_images * 4))
    gs = GridSpec(num_images, num_blocks, figure=fig, hspace=0.3, wspace=0.3)

    with torch.no_grad():
        images = images.to(device)
        tokens, meta = model.tokenizer(images)
        tokens = model.pos_encoder(tokens, meta)

        for img_idx in range(num_images):
            x = tokens[img_idx:img_idx+1]  # Single image

            for block_idx, block in enumerate(model.blocks):
                # Extract attention weights from W-MSA
                # We need to modify forward to return attention
                # For now, we'll visualize token-to-token similarity

                ax = fig.add_subplot(gs[img_idx, block_idx])

                # Compute attention-like patterns using token similarity
                token_norms = F.normalize(x[0], dim=-1)  # [num_tokens, embed_dim]
                attn_sim = torch.mm(token_norms, token_norms.t())  # [num_tokens, num_tokens]

                # Show attention matrix
                im = ax.imshow(attn_sim.cpu().numpy(), cmap='hot', aspect='auto')
                ax.set_title(f'Block {block_idx+1}', fontsize=10)
                ax.set_xlabel('Token Index')
                ax.set_ylabel('Token Index')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Pass through block for next iteration
                x = block(x, meta)

    plt.suptitle('Token-to-Token Similarity Patterns Across Transformer Blocks',
                 fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved attention patterns to {save_path}")
    plt.close()


def plot_feature_evolution(block_features, metadata, predictions, save_path):
    """Visualize how features evolve through transformer blocks."""
    num_images = min(3, block_features[0].shape[0])
    num_blocks = len(block_features)

    fig, axes = plt.subplots(num_images, num_blocks + 1,
                             figsize=((num_blocks + 1) * 3, num_images * 3))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for img_idx in range(num_images):
        # Plot feature norms at each block
        for block_idx in range(num_blocks):
            ax = axes[img_idx, block_idx]

            # Get features for this image and block
            features = block_features[block_idx][img_idx].cpu().numpy()  # [num_tokens, embed_dim]

            # Compute feature magnitude for each token
            feature_norms = np.linalg.norm(features, axis=-1)  # [num_tokens]

            # Reshape based on scale structure (approximate)
            # This is simplified - actual structure is hierarchical
            scales = metadata['scales'].cpu().numpy()

            # Plot as bar chart
            ax.bar(range(len(feature_norms)), feature_norms, alpha=0.7)
            ax.set_title(f'Block {block_idx+1}', fontsize=10)
            ax.set_xlabel('Token Index', fontsize=9)
            ax.set_ylabel('Feature Norm', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        # Plot final statistics
        ax = axes[img_idx, num_blocks]

        # Compute statistics across blocks
        mean_norms = []
        max_norms = []
        for block_idx in range(num_blocks):
            features = block_features[block_idx][img_idx].cpu().numpy()
            feature_norms = np.linalg.norm(features, axis=-1)
            mean_norms.append(feature_norms.mean())
            max_norms.append(feature_norms.max())

        ax.plot(range(1, num_blocks + 1), mean_norms, marker='o', label='Mean Norm', linewidth=2)
        ax.plot(range(1, num_blocks + 1), max_norms, marker='s', label='Max Norm', linewidth=2)
        ax.set_title(f'Prediction: {CIFAR10_CLASSES[predictions[img_idx].argmax()]}', fontsize=10)
        ax.set_xlabel('Block', fontsize=9)
        ax.set_ylabel('Feature Norm', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Feature Evolution Through Transformer Blocks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature evolution to {save_path}")
    plt.close()


def plot_prediction_confidence(predictions, images, save_path):
    """Visualize prediction confidence and top-5 classes."""
    num_images = predictions.shape[0]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx in range(num_images):
        ax = axes[idx]

        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(predictions[idx], 5)
        top5_probs = top5_probs.cpu().numpy()
        top5_classes = [CIFAR10_CLASSES[i] for i in top5_indices.cpu().numpy()]

        # Create horizontal bar chart
        colors = ['green' if i == 0 else 'skyblue' for i in range(5)]
        bars = ax.barh(range(5), top5_probs, color=colors, alpha=0.7)

        ax.set_yticks(range(5))
        ax.set_yticklabels(top5_classes, fontsize=10)
        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_title(f'Image {idx+1}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, top5_probs)):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{prob:.1%}', va='center', fontsize=9)

    plt.suptitle('Top-5 Prediction Confidence', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved prediction confidence to {save_path}")
    plt.close()


def main():
    """Main visualization pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Wavelet Transformer features')
    parser.add_argument('--checkpoint', type=str,
                       default='./experiments/checkpoints/model_best.pth',
                       help='Path to model checkpoint (default: ./experiments/checkpoints/model_best.pth)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of random test samples to visualize (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./results/visualizations',
                       help='Directory to save visualizations (default: ./results/visualizations)')
    args = parser.parse_args()

    print("=" * 80)
    print("Wavelet Transformer Feature Visualization")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Paths
    checkpoint_path = args.checkpoint
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Load test data
    print(f"\nLoading CIFAR-10 test data ({args.num_samples} random samples)...")
    test_loader = get_test_loader(batch_size=args.num_samples, num_samples=args.num_samples)

    # Get batch of images
    images, labels = next(iter(test_loader))
    print(f"Loaded {images.shape[0]} test images: {images.shape}")

    # Extract features
    print("\nExtracting wavelet features and predictions...")
    features = extract_wavelet_features(model, images, device)

    print(f"Number of tokens per image: {features['tokens'].shape[1]}")
    print(f"Embedding dimension: {features['tokens'].shape[2]}")
    print(f"Number of transformer blocks: {len(features['block_features'])}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    print("\n[1/5] Multi-scale wavelet decomposition...")
    plot_wavelet_decomposition(
        images,
        features['wavelet_coeffs'],
        features['predictions'],
        results_dir / 'wavelet_decomposition.png'
    )

    print("\n[2/5] Token embeddings (PCA projection)...")
    plot_token_embeddings(
        features['metadata'],
        features['tokens'],
        features['predictions'],
        results_dir / 'token_embeddings.png'
    )

    print("\n[3/5] Attention patterns...")
    plot_attention_patterns(
        model,
        images,
        features['metadata'],
        results_dir / 'attention_patterns.png',
        device
    )

    print("\n[4/5] Feature evolution through blocks...")
    plot_feature_evolution(
        features['block_features'],
        features['metadata'],
        features['predictions'],
        results_dir / 'feature_evolution.png'
    )

    print("\n[5/5] Prediction confidence...")
    plot_prediction_confidence(
        features['predictions'],
        images,
        results_dir / 'prediction_confidence.png'
    )

    # Print prediction summary
    print("\n" + "=" * 80)
    print("Prediction Summary")
    print("=" * 80)

    predictions = features['predictions']
    for i in range(len(images)):
        pred_class = predictions[i].argmax().item()
        pred_conf = predictions[i].max().item()
        true_class = labels[i].item()

        status = "✓" if pred_class == true_class else "✗"
        print(f"{status} Image {i+1}: Predicted: {CIFAR10_CLASSES[pred_class]} ({pred_conf:.2%}) | "
              f"True: {CIFAR10_CLASSES[true_class]}")

    accuracy = (predictions.argmax(dim=1).cpu() == labels).float().mean().item()
    print(f"\nAccuracy on these {args.num_samples} samples: {accuracy:.1%}")

    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {results_dir.absolute()}")
    print("=" * 80)
    print("\nVisualization files created:")
    print("  1. wavelet_decomposition.png - Multi-scale wavelet coefficients")
    print("  2. token_embeddings.png      - Token embeddings (PCA)")
    print("  3. attention_patterns.png    - Token similarity patterns")
    print("  4. feature_evolution.png     - Features through transformer blocks")
    print("  5. prediction_confidence.png - Top-5 predictions")
    print("\nDone!")


if __name__ == '__main__':
    main()
