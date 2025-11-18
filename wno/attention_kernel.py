"""
PSD Kernel for Scale-Aware Attention
Positive semi-definite kernel that respects wavelet structure.
"""

import torch
import torch.nn as nn
import math


class ScaleAwarePSDKernel(nn.Module):
    """
    Positive semi-definite kernel for scale-aware attention.
    K(τ, τ') = k_scale(j, j') * k_orient(ξ, ξ') * k_spatial(p, p', j)
    """
    
    def __init__(self, 
                 n_levels: int = 3,
                 n_orientations: int = 4,
                 scale_bandwidth: int = 1,
                 spatial_radius_base: int = 7,
                 learnable_scale: bool = True,
                 learnable_orient: bool = True):
        """
        Args:
            n_levels: Number of wavelet scales
            n_orientations: Number of orientations (4 for 2D)
            scale_bandwidth: Max scale difference to attend to
            spatial_radius_base: Base spatial radius (at finest scale)
            learnable_scale: Whether scale kernel parameters are learnable
            learnable_orient: Whether orientation kernel is learnable
        """
        super().__init__()
        self.n_levels = n_levels
        self.n_orientations = n_orientations
        self.scale_bandwidth = scale_bandwidth
        self.spatial_radius_base = spatial_radius_base
        
        # Scale kernel parameters
        if learnable_scale:
            self.log_sigma_scale = nn.Parameter(torch.tensor(0.5).log())
        else:
            self.register_buffer('log_sigma_scale', torch.tensor(0.5).log())
        
        # Orientation kernel: Gram matrix G = UU^T
        if learnable_orient:
            self.orient_U = nn.Parameter(torch.randn(n_orientations, n_orientations) * 0.1)
        else:
            self.register_buffer('orient_U', torch.eye(n_orientations))
        
        # Spatial kernel parameters
        self.log_sigma_spatial = nn.Parameter(torch.tensor(2.0).log())
        
        # Precompute scale normalization factors
        self.register_buffer('scale_norms', 
                            torch.tensor([2 ** j for j in range(n_levels)], dtype=torch.float32))
    
    def compute_scale_kernel(self, scales1: torch.Tensor, scales2: torch.Tensor) -> torch.Tensor:
        """
        Compute k_scale(j, j').
        
        Args:
            scales1: [N1] scale indices
            scales2: [N2] scale indices
        Returns:
            K_scale: [N1, N2] kernel matrix
        """
        sigma_scale = torch.exp(self.log_sigma_scale)
        
        # Pairwise scale differences [N1, N2]
        scale_diff = scales1.unsqueeze(1) - scales2.unsqueeze(0)  # [N1, N2]
        
        # RBF kernel
        K_scale = torch.exp(-0.5 * (scale_diff.float() ** 2) / (sigma_scale ** 2))
        
        # Apply scale bandwidth mask: zero out if |j - j'| > bandwidth
        mask = torch.abs(scale_diff) <= self.scale_bandwidth
        K_scale = K_scale * mask.float()
        
        return K_scale
    
    def compute_orient_kernel(self, orients1: torch.Tensor, orients2: torch.Tensor) -> torch.Tensor:
        """
        Compute k_orient(ξ, ξ') = e_ξ^T G e_ξ' where G = UU^T.
        
        Args:
            orients1: [N1] orientation indices
            orients2: [N2] orientation indices
        Returns:
            K_orient: [N1, N2] kernel matrix
        """
        # Compute Gram matrix G = UU^T + epsilon*I (ensure PSD with numerical stability)
        G = self.orient_U @ self.orient_U.T  # [n_orient, n_orient]
        G = G + 1e-6 * torch.eye(G.shape[0], device=G.device)  # Add small diagonal for stability
        
        # Look up entries: K[i, j] = G[orients1[i], orients2[j]]
        K_orient = G[orients1][:, orients2]  # [N1, N2]
        
        return K_orient
    
    def compute_spatial_kernel(self, 
                               positions1: torch.Tensor, 
                               positions2: torch.Tensor,
                               scales1: torch.Tensor,
                               scales2: torch.Tensor) -> torch.Tensor:
        """
        Compute k_spatial(p, p', j) with scale-normalized distances.
        
        Args:
            positions1: [N1, 2] (h, w) coordinates
            positions2: [N2, 2] (h, w) coordinates
            scales1: [N1] scale indices
            scales2: [N2] scale indices
        Returns:
            K_spatial: [N1, N2] kernel matrix
        """
        sigma_spatial = torch.exp(self.log_sigma_spatial)
        
        # Pairwise distances [N1, N2, 2]
        pos_diff = positions1.unsqueeze(1) - positions2.unsqueeze(0)  # [N1, N2, 2]
        distances_sq = (pos_diff ** 2).sum(dim=-1)  # [N1, N2]
        
        # Scale normalization: use minimum scale between pairs
        # scale_norms[j] = 2^j
        scale_min = torch.minimum(
            self.scale_norms[scales1].unsqueeze(1),
            self.scale_norms[scales2].unsqueeze(0)
        )  # [N1, N2]
        
        # Normalized distances
        distances_norm_sq = distances_sq / (scale_min ** 2 + 1e-6)
        
        # RBF kernel
        K_spatial = torch.exp(-0.5 * distances_norm_sq / (sigma_spatial ** 2))
        
        # Apply spatial radius mask
        radius_threshold = self.spatial_radius_base ** 2
        mask = distances_sq <= radius_threshold * (scale_min ** 2)
        K_spatial = K_spatial * mask.float()
        
        return K_spatial
    
    def forward(self, metadata1: dict, metadata2: dict = None) -> torch.Tensor:
        """
        Compute full kernel matrix K(τ, τ').
        
        Args:
            metadata1: Dictionary with 'scales', 'orientations', 'positions' for query tokens
            metadata2: Optional, for key tokens (if None, use metadata1 for self-attention)
        
        Returns:
            K: [N1, N2] kernel matrix (or [N, N] if metadata2 is None)
        """
        if metadata2 is None:
            metadata2 = metadata1
        
        scales1 = metadata1['scales']
        orients1 = metadata1['orientations']
        positions1 = metadata1['positions']
        
        scales2 = metadata2['scales']
        orients2 = metadata2['orientations']
        positions2 = metadata2['positions']
        
        # Compute component kernels
        K_scale = self.compute_scale_kernel(scales1, scales2)
        K_orient = self.compute_orient_kernel(orients1, orients2)
        K_spatial = self.compute_spatial_kernel(positions1, positions2, scales1, scales2)
        
        # Product (element-wise multiplication)
        K = K_scale * K_orient * K_spatial
        
        # Add small diagonal regularization for numerical stability
        if metadata1 is None or metadata2 is None or scales1.shape[0] == scales2.shape[0]:
            # Self-attention case: add diagonal
            K = K + 1e-6 * torch.eye(K.shape[0], device=K.device)
        
        return K


def test_psd_kernel():
    """Test PSD kernel properties"""
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    kernel = ScaleAwarePSDKernel(
        n_levels=3,
        n_orientations=4,
        scale_bandwidth=1,
        spatial_radius_base=5
    )
    
    # Create sample metadata for a small grid
    n_tokens = 64  # 8x8 grid at scale 0
    grid_size = 8
    
    positions = torch.stack(torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing='ij'
    ), dim=-1).reshape(-1, 2)
    
    metadata = {
        'scales': torch.zeros(n_tokens, dtype=torch.long),
        'orientations': torch.zeros(n_tokens, dtype=torch.long),
        'positions': positions
    }
    
    # Compute kernel matrix
    K = kernel(metadata)
    
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel is symmetric: {torch.allclose(K, K.T, atol=1e-6)}")
    
    # Check PSD: all eigenvalues should be non-negative
    eigvals = torch.linalg.eigvalsh(K)
    print(f"Min eigenvalue: {eigvals.min().item():.6f}")
    print(f"Max eigenvalue: {eigvals.max().item():.6f}")
    print(f"Is PSD: {(eigvals >= -1e-6).all().item()}")
    
    # Visualize kernel matrix
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Full kernel matrix
    im0 = axes[0].imshow(K.detach().numpy(), cmap='viridis')
    axes[0].set_title("Full Kernel Matrix K")
    axes[0].set_xlabel("Token index")
    axes[0].set_ylabel("Token index")
    plt.colorbar(im0, ax=axes[0])
    
    # Kernel from center token
    center_idx = n_tokens // 2
    K_center = K[center_idx].reshape(grid_size, grid_size)
    im1 = axes[1].imshow(K_center.detach().numpy(), cmap='hot')
    axes[1].set_title(f"Attention from Center Token")
    axes[1].set_xlabel("Grid column")
    axes[1].set_ylabel("Grid row")
    plt.colorbar(im1, ax=axes[1])
    
    # Eigenvalue spectrum
    axes[2].plot(eigvals.detach().numpy(), 'o-')
    axes[2].set_xlabel("Eigenvalue index")
    axes[2].set_ylabel("Eigenvalue")
    axes[2].set_title("Eigenvalue Spectrum")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../results/phase2_psd_kernel.png', dpi=150)
    plt.close()
    
    print("\n✅ PSD kernel test complete")


if __name__ == "__main__":
    test_psd_kernel()

