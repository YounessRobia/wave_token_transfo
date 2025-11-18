"""
CIFAR-10 Image Classification with Wavelet Transformer
Complete training pipeline with data augmentation and evaluation.
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

from wno.model import create_wavelet_small, create_wavelet_tiny
from wno.train_utils import (
    train_epoch, validate, save_checkpoint, 
    CosineAnnealingWarmupRestarts, count_parameters
)


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Create CIFAR-10 data loaders with augmentation
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_wavelet_transformer(args):
    """
    Main training function
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # Create data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        data_dir=args.data_dir
    )
    
    # Create model
    print(f"\nCreating Wavelet Transformer ({args.model_size})...")
    if args.model_size == 'tiny':
        model = create_wavelet_tiny(
            image_size=(32, 32),
            in_channels=3,
            num_classes=10,
            n_levels=args.n_levels,
            use_kernel=args.use_kernel,
            wavelet=args.wavelet
        )
    else:
        model = create_wavelet_small(
            image_size=(32, 32),
            in_channels=3,
            num_classes=10,
            n_levels=args.n_levels,
            use_kernel=args.use_kernel,
            wavelet=args.wavelet
        )
    
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Estimated FLOPs: {model.get_flops() / 1e9:.2f}G")
    
    # Memory optimization info
    print(f"\n{'='*60}")
    print(f"Memory Optimization Settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Number of tokens per image: {model.tokenizer.get_token_count((32, 32))}")
    est_memory = (args.batch_size * model.tokenizer.get_token_count((32, 32)) ** 2 * 4 * 3) / 1e9
    print(f"  Estimated attention memory: ~{est_memory:.2f} GB")
    print(f"{'='*60}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs,
        max_lr=args.lr,
        min_lr=args.lr * 0.01,
        warmup_steps=args.warmup_epochs
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            accumulation_steps=args.accumulation_steps
        )
        
        # Validate
        test_stats = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_stats['loss'])
        train_accs.append(train_stats['acc1'])
        test_losses.append(test_stats['loss'])
        test_accs.append(test_stats['acc1'])
        
        # Save checkpoint
        is_best = test_stats['acc1'] > best_acc
        best_acc = max(test_stats['acc1'], best_acc)
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                is_best,
                filename=os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth'),
                best_filename=os.path.join(args.save_dir, 'model_best.pth')
            )
        
        print(f"Best accuracy so far: {best_acc:.2f}%")
    
    # Final results
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, train_accs, test_accs, args)
    
    return model, train_losses, test_losses, train_accs, test_accs


def plot_training_curves(train_losses, test_losses, train_accs, test_accs, args):
    """Plot and save training curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Test Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, test_accs, 'r-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Test Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../results/cifar10_training_curves_{args.model_size}.png', dpi=150)
    plt.close()
    
    print(f"\nTraining curves saved to ../results/cifar10_training_curves_{args.model_size}.png")


def visualize_predictions(model, test_loader, device, num_images=16):
    """Visualize model predictions"""
    
    model.eval()
    
    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # Denormalize images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images = images.cpu() * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(num_images):
        img = images[idx].permute(1, 2, 0).numpy()
        axes[idx].imshow(img)
        
        true_label = classes[labels[idx]]
        pred_label = classes[predicted[idx].cpu()]
        
        color = 'green' if labels[idx] == predicted[idx].cpu() else 'red'
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                           color=color, fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/cifar10_predictions.png', dpi=150)
    plt.close()
    
    print("Predictions visualization saved to ../results/cifar10_predictions.png")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification with Wavelet Transformer')
    
    # Model parameters
    parser.add_argument('--model-size', type=str, default='tiny', choices=['tiny', 'small'],
                       help='Model size')
    parser.add_argument('--n-levels', type=int, default=2,
                       help='Number of wavelet decomposition levels (2=memory-friendly, 3=better accuracy)')
    parser.add_argument('--wavelet', type=str, default='db4',
                       help='Wavelet family')
    parser.add_argument('--use-kernel', action='store_true', default=True,
                       help='Use PSD kernel in attention')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (16=safe for 6GB GPU, 32=for 8GB+, 64=for 12GB+)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs')
    parser.add_argument('--accumulation-steps', type=int, default=8,
                       help='Gradient accumulation steps (effective batch = batch_size * accumulation_steps, default: 16*8=128)')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Save parameters
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Train model
    model, train_losses, test_losses, train_accs, test_accs = train_wavelet_transformer(args)
    
    # Visualize predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_cifar10_loaders(args.batch_size, args.workers, args.data_dir)
    visualize_predictions(model, test_loader, device)
    
    print("\n✅ Experiment complete!")


if __name__ == '__main__':
    main()

