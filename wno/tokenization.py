"""
Wavelet Tokenization Module
Convert images to wavelet tokens with scale/orientation metadata.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .wavelets_2d import WaveletTransform2D


class WaveletTokenizer(nn.Module):
    """
    Convert image to wavelet tokens with scale/orientation metadata.
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 embed_dim: int = 192,
                 wavelet: str = 'db4',
                 n_levels: int = 3,
                 pool_coarse: bool = True):
        """
        Args:
            in_channels: Number of input image channels
            embed_dim: Token embedding dimension
            wavelet: Wavelet family
            n_levels: Number of decomposition levels
            pool_coarse: Whether to pool coarsest approximation to single token
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.pool_coarse = pool_coarse
        
        # Wavelet transform
        self.wt = WaveletTransform2D(wavelet=wavelet, level=n_levels)
        
        # Projection layers for each coefficient type at each level
        # Approximation: project C channels -> embed_dim
        self.proj_approx = nn.ModuleList([
            nn.Linear(in_channels, embed_dim)
            for _ in range(n_levels)
        ])
        
        # Details (H, V, D): project C channels -> embed_dim
        self.proj_details = nn.ModuleList([
            nn.ModuleDict({
                'horizontal': nn.Linear(in_channels, embed_dim),
                'vertical': nn.Linear(in_channels, embed_dim),
                'diagonal': nn.Linear(in_channels, embed_dim)
            })
            for _ in range(n_levels)
        ])
        
        # Optional: coarse pooling
        if pool_coarse:
            self.coarse_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input image [B, C, H, W]
        
        Returns:
            tokens: [B, N, D] where N is total number of tokens
            metadata: Dictionary containing:
                - 'scales': [N] tensor of scale indices
                - 'orientations': [N] tensor of orientation indices (0=A, 1=H, 2=V, 3=D)
                - 'positions': [N, 2] tensor of (h, w) coordinates
                - 'level_counts': List of token counts per level
        """
        B, C, H, W = x.shape
        
        # Wavelet decomposition
        coeffs = self.wt.forward(x)
        
        # Storage for tokens and metadata
        all_tokens = []
        scales = []
        orientations = []
        positions = []
        level_counts = []
        
        # Process each level
        for level in range(self.n_levels):
            # Approximation coefficients
            cA = coeffs['approximations'][level]  # [B, C, h, w]
            B_a, C_a, h_a, w_a = cA.shape
            
            if level == self.n_levels - 1 and self.pool_coarse:
                # Pool coarsest approximation to single token
                cA_pooled = self.coarse_pool(cA).squeeze(-1).squeeze(-1)  # [B, C]
                cA_tokens = self.proj_approx[level](cA_pooled).unsqueeze(1)  # [B, 1, D]
                all_tokens.append(cA_tokens)
                
                scales.append(torch.full((1,), level, dtype=torch.long))
                orientations.append(torch.full((1,), 0, dtype=torch.long))  # 0 = approx
                positions.append(torch.tensor([[h_a//2, w_a//2]], dtype=torch.long))
                level_counts.append(1)
            else:
                # Flatten spatial dimensions: [B, C, h, w] -> [B, h*w, C]
                cA_flat = cA.permute(0, 2, 3, 1).reshape(B, h_a * w_a, C_a)
                cA_tokens = self.proj_approx[level](cA_flat)  # [B, h*w, D]
                all_tokens.append(cA_tokens)
                
                # Metadata for approximation
                n_tokens_a = h_a * w_a
                scales.append(torch.full((n_tokens_a,), level, dtype=torch.long))
                orientations.append(torch.zeros(n_tokens_a, dtype=torch.long))
                
                # Position grid
                pos_grid = torch.stack(torch.meshgrid(
                    torch.arange(h_a), torch.arange(w_a), indexing='ij'
                ), dim=-1).reshape(-1, 2)
                positions.append(pos_grid)
                level_counts.append(n_tokens_a)
            
            # Detail coefficients (H, V, D)
            for orient_idx, orient_name in enumerate(['horizontal', 'vertical', 'diagonal'], 1):
                cDetail = coeffs[orient_name][level]  # [B, C, h, w]
                B_d, C_d, h_d, w_d = cDetail.shape
                
                # Flatten and project
                cDetail_flat = cDetail.permute(0, 2, 3, 1).reshape(B, h_d * w_d, C_d)
                cDetail_tokens = self.proj_details[level][orient_name](cDetail_flat)
                all_tokens.append(cDetail_tokens)
                
                # Metadata
                n_tokens_d = h_d * w_d
                scales.append(torch.full((n_tokens_d,), level, dtype=torch.long))
                orientations.append(torch.full((n_tokens_d,), orient_idx, dtype=torch.long))
                
                pos_grid = torch.stack(torch.meshgrid(
                    torch.arange(h_d), torch.arange(w_d), indexing='ij'
                ), dim=-1).reshape(-1, 2)
                positions.append(pos_grid)
                level_counts.append(n_tokens_d)
        
        # Concatenate all tokens
        tokens = torch.cat(all_tokens, dim=1)  # [B, N_total, D]
        
        # Concatenate metadata
        metadata = {
            'scales': torch.cat(scales).to(x.device),
            'orientations': torch.cat(orientations).to(x.device),
            'positions': torch.cat(positions).to(x.device),
            'level_counts': level_counts,
            'B': B
        }
        
        return tokens, metadata
    
    def get_token_count(self, image_size: Tuple[int, int]) -> int:
        """Calculate total number of tokens for given image size"""
        H, W = image_size
        total = 0
        
        for level in range(self.n_levels):
            h, w = H // (2 ** (level + 1)), W // (2 ** (level + 1))
            
            if level == self.n_levels - 1 and self.pool_coarse:
                total += 1  # Pooled approximation
            else:
                total += h * w  # Approximation
            
            total += 3 * h * w  # H, V, D details
        
        return total


def test_tokenizer():
    """Test wavelet tokenizer"""
    
    tokenizer = WaveletTokenizer(
        in_channels=3,
        embed_dim=192,
        n_levels=3,
        pool_coarse=True
    )
    
    # Test image
    x = torch.randn(2, 3, 128, 128)
    
    # Tokenize
    tokens, metadata = tokenizer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output tokens: {tokens.shape}")
    print(f"Token count per level: {metadata['level_counts']}")
    print(f"Total tokens: {sum(metadata['level_counts'])}")
    
    # Verify metadata
    print(f"\nMetadata shapes:")
    print(f"  Scales: {metadata['scales'].shape}")
    print(f"  Orientations: {metadata['orientations'].shape}")
    print(f"  Positions: {metadata['positions'].shape}")
    
    # Expected token count
    expected = tokenizer.get_token_count((128, 128))
    print(f"\nExpected tokens: {expected}")
    print(f"Actual tokens: {tokens.shape[1]}")
    
    assert tokens.shape[1] == expected, "Token count mismatch!"
    
    print("\n✅ Tokenizer test passed")


if __name__ == "__main__":
    test_tokenizer()





