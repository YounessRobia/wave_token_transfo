"""
Scale-Aware Positional Encoding for Wavelet Tokens
"""

import torch
import torch.nn as nn
import math


class WaveletPositionalEncoding(nn.Module):
    """
    Scale-aware positional encoding for wavelet tokens.
    """
    
    def __init__(self, 
                 embed_dim: int = 192,
                 n_levels: int = 3,
                 n_orientations: int = 4,
                 max_positions: int = 1024,
                 dropout: float = 0.1):
        """
        Args:
            embed_dim: Token embedding dimension
            n_levels: Number of wavelet decomposition levels
            n_orientations: Number of orientations (4 for 2D: A, H, V, D)
            max_positions: Maximum spatial positions to encode
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        
        # Learnable scale embeddings (one per level)
        self.scale_embed = nn.Embedding(n_levels, embed_dim)
        
        # Learnable orientation embeddings
        self.orient_embed = nn.Embedding(n_orientations, embed_dim)
        
        # Spatial positional encoding: sinusoidal (similar to Transformer)
        # We'll use 2D sinusoidal encoding
        self.spatial_encoding_type = 'sinusoidal'  # or 'learned'
        
        if self.spatial_encoding_type == 'sinusoidal':
            # Pre-compute sinusoidal encodings
            self.register_buffer('spatial_table', 
                                self._make_spatial_encoding_table(max_positions, embed_dim))
        else:
            # Learnable spatial embeddings
            self.pos_h_embed = nn.Embedding(max_positions, embed_dim // 2)
            self.pos_w_embed = nn.Embedding(max_positions, embed_dim // 2)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scale normalization factors (to normalize spatial frequencies by scale)
        self.register_buffer('scale_factors', 
                            torch.tensor([2 ** j for j in range(n_levels)], dtype=torch.float32))
    
    def _make_spatial_encoding_table(self, max_pos: int, d_model: int) -> torch.Tensor:
        """
        Create 2D sinusoidal positional encoding table.
        Encoding for position (h, w):
            PE(h, w, 2i) = sin(h / 10000^(2i/d))
            PE(h, w, 2i+1) = cos(h / 10000^(2i/d))
        and similarly for w in the second half of dimensions.
        """
        # Create frequency matrix
        position = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2, dtype=torch.float32) * 
                            -(math.log(10000.0) / (d_model // 2)))
        
        # Table for 1D positions [max_pos, d_model // 2]
        pe_1d = torch.zeros(max_pos, d_model // 2)
        pe_1d[:, 0::2] = torch.sin(position * div_term)
        pe_1d[:, 1::2] = torch.cos(position * div_term)
        
        return pe_1d
    
    def forward(self, tokens: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Add positional encoding to tokens.
        
        Args:
            tokens: [B, N, D]
            metadata: Dictionary with 'scales', 'orientations', 'positions'
        
        Returns:
            tokens_with_pe: [B, N, D]
        """
        B, N, D = tokens.shape
        device = tokens.device
        
        # Extract metadata
        scales = metadata['scales']  # [N]
        orientations = metadata['orientations']  # [N]
        positions = metadata['positions']  # [N, 2]
        
        # 1. Scale embedding
        scale_pe = self.scale_embed(scales)  # [N, D]
        
        # 2. Orientation embedding
        orient_pe = self.orient_embed(orientations)  # [N, D]
        
        # 3. Spatial encoding (scale-normalized)
        if self.spatial_encoding_type == 'sinusoidal':
            # Get spatial encodings for h and w separately
            pos_h = positions[:, 0]  # [N]
            pos_w = positions[:, 1]  # [N]
            
            # Normalize positions by scale (coarser scales have smaller spatial extent)
            # scale_factors[scales] gives 2^j for each token's scale
            scale_norm = self.scale_factors[scales]  # [N]
            pos_h_norm = (pos_h.float() * scale_norm).long().clamp(0, self.spatial_table.shape[0] - 1)
            pos_w_norm = (pos_w.float() * scale_norm).long().clamp(0, self.spatial_table.shape[0] - 1)
            
            # Look up encodings
            pe_h = self.spatial_table[pos_h_norm]  # [N, D//2]
            pe_w = self.spatial_table[pos_w_norm]  # [N, D//2]
            
            spatial_pe = torch.cat([pe_h, pe_w], dim=-1)  # [N, D]
        else:
            # Learnable spatial embeddings
            pos_h = positions[:, 0].clamp(0, self.pos_h_embed.num_embeddings - 1)
            pos_w = positions[:, 1].clamp(0, self.pos_w_embed.num_embeddings - 1)
            
            pe_h = self.pos_h_embed(pos_h)  # [N, D//2]
            pe_w = self.pos_w_embed(pos_w)  # [N, D//2]
            spatial_pe = torch.cat([pe_h, pe_w], dim=-1)  # [N, D]
        
        # Combine all positional encodings
        total_pe = scale_pe + orient_pe + spatial_pe  # [N, D]
        
        # Add to tokens (broadcast across batch)
        tokens_with_pe = tokens + total_pe.unsqueeze(0)  # [B, N, D]
        
        return self.dropout(tokens_with_pe)


def visualize_positional_encoding():
    """Visualize the positional encoding patterns"""
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    pe_module = WaveletPositionalEncoding(
        embed_dim=192,
        n_levels=3,
        n_orientations=4
    )
    
    # Create dummy metadata for visualization
    # Grid of positions at different scales
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for level in range(3):
        # Create grid of positions at this scale
        grid_size = 16 // (2 ** level)
        positions = []
        for h in range(grid_size):
            for w in range(grid_size):
                positions.append([h, w])
        
        positions = torch.tensor(positions, dtype=torch.long)
        n_pos = len(positions)
        
        # Create metadata
        metadata = {
            'scales': torch.full((n_pos,), level, dtype=torch.long),
            'orientations': torch.zeros(n_pos, dtype=torch.long),  # All approximation
            'positions': positions
        }
        
        # Get tokens (dummy)
        tokens = torch.zeros(1, n_pos, 192)
        
        # Apply positional encoding
        tokens_pe = pe_module(tokens, metadata)
        
        # Extract encoding [1, n_pos, 192] -> [n_pos, 192]
        pe_vectors = tokens_pe[0].detach().numpy()
        
        # Reshape to grid
        pe_grid = pe_vectors.reshape(grid_size, grid_size, 192)
        
        # Plot first 3 PCA components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pe_pca = pca.fit_transform(pe_vectors)
        pe_pca_grid = pe_pca.reshape(grid_size, grid_size, 3)
        
        # Normalize to [0, 1] for visualization
        pe_pca_grid = (pe_pca_grid - pe_pca_grid.min()) / (pe_pca_grid.max() - pe_pca_grid.min())
        
        axes[0, level].imshow(pe_pca_grid)
        axes[0, level].set_title(f"Level {level} Positional Encoding\n(PCA RGB)")
        axes[0, level].axis('off')
        
        # Plot similarity matrix
        similarity = pe_vectors @ pe_vectors.T
        axes[1, level].imshow(similarity, cmap='viridis')
        axes[1, level].set_title(f"Level {level} Token Similarity")
        axes[1, level].axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/phase2_positional_encoding.png', dpi=150)
    plt.close()
    
    print("✅ Positional encoding visualization complete")


if __name__ == "__main__":
    visualize_positional_encoding()





