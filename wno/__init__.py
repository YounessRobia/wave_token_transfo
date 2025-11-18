"""
Wavelet Neural Operator - Phase 2
Wavelet Multi-Scale Attention for Vision Tasks
"""

from .wavelets_2d import WaveletTransform2D
from .tokenization import WaveletTokenizer
from .positional_encoding import WaveletPositionalEncoding
from .attention_kernel import ScaleAwarePSDKernel
from .wmsa import WaveletMultiScaleAttention, WaveletTransformerBlock
from .model import (
    WaveletTransformer,
    create_wavelet_tiny,
    create_wavelet_small,
    create_wavelet_base
)

__all__ = [
    'WaveletTransform2D',
    'WaveletTokenizer',
    'WaveletPositionalEncoding',
    'ScaleAwarePSDKernel',
    'WaveletMultiScaleAttention',
    'WaveletTransformerBlock',
    'WaveletTransformer',
    'create_wavelet_tiny',
    'create_wavelet_small',
    'create_wavelet_base',
]

__version__ = '2.0.0'





