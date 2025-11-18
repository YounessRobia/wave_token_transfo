"""
Complete Wavelet Transformer Architecture
Full model with tokenization, attention, and classification head.
"""

import torch
import torch.nn as nn
from typing import Tuple
from .tokenization import WaveletTokenizer
from .positional_encoding import WaveletPositionalEncoding
from .wmsa import WaveletTransformerBlock


class WaveletTransformer(nn.Module):
    """
    Complete Wavelet Transformer for Image Classification.
    
    Architecture:
        Input Image -> Wavelet Tokenization -> Positional Encoding ->
        Stack of W-MSA Blocks -> Global Pooling -> Classification Head
    """
    
    def __init__(self,
                 image_size: Tuple[int, int] = (32, 32),
                 in_channels: int = 3,
                 num_classes: int = 10,
                 embed_dim: int = 192,
                 depth: int = 6,
                 num_heads: int = 3,
                 n_levels: int = 3,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 use_kernel: bool = True,
                 pool_coarse: bool = True,
                 wavelet: str = 'db4'):
        """
        Args:
            image_size: Input image size (H, W)
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Token embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            n_levels: Number of wavelet decomposition levels
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_kernel: Use PSD kernel in attention
            pool_coarse: Pool coarsest wavelet approximation
            wavelet: Wavelet family
        """
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_levels = n_levels
        
        # Wavelet tokenizer
        self.tokenizer = WaveletTokenizer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            wavelet=wavelet,
            n_levels=n_levels,
            pool_coarse=pool_coarse
        )
        
        # Positional encoding
        self.pos_encoder = WaveletPositionalEncoding(
            embed_dim=embed_dim,
            n_levels=n_levels,
            n_orientations=4,
            dropout=drop_rate
        )
        
        # Stack of Wavelet Transformer blocks
        self.blocks = nn.ModuleList([
            WaveletTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                n_levels=n_levels,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                use_kernel=use_kernel
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Global pooling
        self.pool_type = 'mean'  # or 'cls' or 'max'
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Extract features from input image.
        
        Args:
            x: Input image [B, C, H, W]
        Returns:
            features: [B, N, D] token features
            metadata: Token metadata dictionary
        """
        # Tokenize
        tokens, metadata = self.tokenizer(x)
        
        # Add positional encoding
        tokens = self.pos_encoder(tokens, metadata)
        
        # Forward through transformer blocks
        for block in self.blocks:
            tokens = block(tokens, metadata)
        
        # Final normalization
        tokens = self.norm(tokens)
        
        return tokens, metadata
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, C, H, W]
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Extract features
        features, metadata = self.forward_features(x)
        
        # Global pooling
        if self.pool_type == 'mean':
            # Mean pooling across all tokens
            pooled = features.mean(dim=1)  # [B, D]
        elif self.pool_type == 'max':
            # Max pooling
            pooled = features.max(dim=1)[0]  # [B, D]
        else:
            # Use first token (approximation token if pool_coarse=True)
            pooled = features[:, 0]  # [B, D]
        
        # Classification head
        logits = self.head(pooled)
        
        return logits
    
    def get_num_params(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self, image_size=None):
        """
        Estimate FLOPs for a single forward pass.
        This is a rough approximation.
        """
        if image_size is None:
            image_size = self.image_size
        
        # Get number of tokens
        n_tokens = self.tokenizer.get_token_count(image_size)
        
        # Tokenization FLOPs (wavelet transform + projection)
        # Wavelet transform is O(HW) for separable filters
        H, W = image_size
        flops_tokenize = H * W * self.in_channels * 100  # Rough estimate
        
        # Attention FLOPs per block: 4*N*D^2 + 2*N^2*D
        flops_attn_per_block = 4 * n_tokens * self.embed_dim**2 + 2 * n_tokens**2 * self.embed_dim
        
        # MLP FLOPs per block: 8*N*D^2
        flops_mlp_per_block = 8 * n_tokens * self.embed_dim**2
        
        # Total
        flops_blocks = self.depth * (flops_attn_per_block + flops_mlp_per_block)
        
        # Classification head
        flops_head = self.embed_dim * self.num_classes
        
        total_flops = flops_tokenize + flops_blocks + flops_head
        
        return total_flops


def create_wavelet_tiny(**kwargs):
    """Create a tiny Wavelet Transformer (for testing)"""
    model = WaveletTransformer(
        embed_dim=96,
        depth=4,
        num_heads=3,
        **kwargs
    )
    return model


def create_wavelet_small(**kwargs):
    """Create a small Wavelet Transformer"""
    model = WaveletTransformer(
        embed_dim=192,
        depth=6,
        num_heads=3,
        **kwargs
    )
    return model


def create_wavelet_base(**kwargs):
    """Create a base Wavelet Transformer"""
    model = WaveletTransformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    return model


def test_model():
    """Test complete model"""
    
    # Create model
    model = create_wavelet_small(
        image_size=(32, 32),
        in_channels=3,
        num_classes=10,
        n_levels=3
    )
    
    # Test input
    x = torch.randn(4, 3, 32, 32)
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {model.get_num_params():,}")
    print(f"Estimated FLOPs: {model.get_flops() / 1e9:.2f}G")
    
    # Test feature extraction
    features, metadata = model.forward_features(x)
    print(f"Feature shape: {features.shape}")
    print(f"Number of tokens: {features.shape[1]}")
    
    print("\n✅ Model test passed")


if __name__ == "__main__":
    test_model()





