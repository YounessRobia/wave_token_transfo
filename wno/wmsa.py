"""
Wavelet Multi-Scale Attention (W-MSA) Implementation
"""

import torch
import torch.nn as nn
import math
from .attention_kernel import ScaleAwarePSDKernel


class WaveletMultiScaleAttention(nn.Module):
    """
    Wavelet Multi-Scale Attention (W-MSA) using PSD kernel.
    """
    
    def __init__(self,
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 n_levels: int = 3,
                 n_orientations: int = 4,
                 scale_bandwidth: int = 1,
                 spatial_radius_base: int = 7,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 use_kernel: bool = True):
        """
        Args:
            embed_dim: Token embedding dimension
            num_heads: Number of attention heads
            n_levels: Number of wavelet scales
            n_orientations: Number of orientations
            scale_bandwidth: Scale locality bandwidth
            spatial_radius_base: Spatial locality radius
            qkv_bias: Whether to use bias in QKV projections
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            use_kernel: Whether to use PSD kernel (if False, standard attention)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_kernel = use_kernel
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # PSD Kernel
        if use_kernel:
            self.kernel = ScaleAwarePSDKernel(
                n_levels=n_levels,
                n_orientations=n_orientations,
                scale_bandwidth=scale_bandwidth,
                spatial_radius_base=spatial_radius_base
            )
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            metadata: Dictionary with token metadata
        
        Returns:
            out: [B, N, D] output tokens
        """
        B, N, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, N, d]
        
        if self.use_kernel:
            # Compute PSD kernel matrix
            K_psd = self.kernel(metadata)  # [N, N]
            
            # Standard scaled dot-product attention scores
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
            
            # Modulate by PSD kernel (element-wise product)
            # K_psd is shared across batch and heads
            K_psd_expanded = K_psd.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            attn_logits = attn_logits * K_psd_expanded
            
            # Softmax
            attn = torch.softmax(attn_logits, dim=-1)
            attn = self.attn_drop(attn)
        else:
            # Standard attention (baseline)
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn_logits, dim=-1)
            attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class WaveletTransformerBlock(nn.Module):
    """
    Single Wavelet Transformer block:
    x -> LayerNorm -> W-MSA -> Residual -> LayerNorm -> FFN -> Residual
    """
    
    def __init__(self,
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 n_levels: int = 3,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 use_kernel: bool = True,
                 use_wno_ffn: bool = False):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            n_levels: Number of wavelet scales
            mlp_ratio: FFN hidden dimension = mlp_ratio * embed_dim
            qkv_bias: QKV bias
            drop: Dropout rate
            attn_drop: Attention dropout
            use_kernel: Use PSD kernel in attention
            use_wno_ffn: Use WNO layer instead of standard MLP
        """
        super().__init__()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # W-MSA
        self.attn = WaveletMultiScaleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            n_levels=n_levels,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_kernel=use_kernel
        )
        
        # FFN
        if use_wno_ffn:
            # Use WNO layer from Phase 1 (requires special handling)
            # For now, we'll use standard MLP
            # TODO: Integrate WNOLayer properly
            use_wno_ffn = False
        
        if not use_wno_ffn:
            # Standard MLP
            hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(drop)
            )
    
    def forward(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            metadata: Token metadata
        Returns:
            x: [B, N, D]
        """
        # Attention block with residual
        x = x + self.attn(self.norm1(x), metadata)
        
        # FFN block with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


def test_wmsa():
    """Test W-MSA module"""
    
    from .tokenization import WaveletTokenizer
    from .positional_encoding import WaveletPositionalEncoding
    
    # Create tokenizer
    tokenizer = WaveletTokenizer(
        in_channels=3,
        embed_dim=192,
        n_levels=3
    )
    
    # Create positional encoder
    pe = WaveletPositionalEncoding(embed_dim=192, n_levels=3)
    
    # Create W-MSA block
    block = WaveletTransformerBlock(
        embed_dim=192,
        num_heads=3,
        n_levels=3,
        use_kernel=True
    )
    
    # Test input
    x = torch.randn(2, 3, 64, 64)
    
    # Tokenize
    tokens, metadata = tokenizer(x)
    print(f"Tokens shape: {tokens.shape}")
    
    # Add positional encoding
    tokens = pe(tokens, metadata)
    
    # Forward through W-MSA block
    out = block(tokens, metadata)
    print(f"Output shape: {out.shape}")
    
    # Verify shapes match
    assert out.shape == tokens.shape
    
    print("\n✅ W-MSA test passed")


if __name__ == "__main__":
    test_wmsa()





