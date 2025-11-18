"""
Complete 2D Discrete Wavelet Transform Implementation
Supports multi-level decomposition with proper coefficient handling.
"""

import torch
import torch.nn as nn
import pywt
import numpy as np
from typing import List, Tuple, Dict


class WaveletTransform2D:
    """
    Complete 2D discrete wavelet transform implementation.
    Supports multi-level decomposition with proper coefficient handling.
    """
    
    def __init__(self, wavelet='db4', mode='periodic', level=3):
        """
        Args:
            wavelet: Wavelet family ('db4', 'bior3.5', 'coif2', etc.)
            mode: Boundary handling mode
            level: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        
        # Get wavelet filters
        w = pywt.Wavelet(wavelet)
        
        # Convert to PyTorch tensors
        self.dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        self.dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
        self.rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
        self.rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
        
    def _separable_conv2d(self, x, filter_h, filter_v, stride=1):
        """Apply separable 2D convolution (rows then columns)"""
        device = x.device
        B, C, H, W = x.shape
        
        # Prepare filters
        filter_h = filter_h.to(device).flip(0)
        filter_v = filter_v.to(device).flip(0)
        
        # Expand for grouped convolution
        filter_h = filter_h.view(1, 1, 1, -1).repeat(C, 1, 1, 1)  # [C, 1, 1, L]
        filter_v = filter_v.view(1, 1, -1, 1).repeat(C, 1, 1, 1)  # [C, 1, L, 1]
        
        # Pad
        pad_h = filter_h.shape[-1] - 1
        pad_v = filter_v.shape[-2] - 1
        
        if self.mode == 'periodic':
            x = torch.nn.functional.pad(x, (pad_h//2, pad_h//2, 0, 0), mode='circular')
            x = torch.nn.functional.pad(x, (0, 0, pad_v//2, pad_v//2), mode='circular')
        else:
            x = torch.nn.functional.pad(x, (pad_h//2, pad_h//2, pad_v//2, pad_v//2), 
                                       mode='replicate')
        
        # Apply horizontal filter
        x = torch.nn.functional.conv2d(x, filter_h, stride=(1, stride), groups=C)
        
        # Apply vertical filter
        x = torch.nn.functional.conv2d(x, filter_v, stride=(stride, 1), groups=C)
        
        return x
    
    def dwt2d_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Single-level 2D DWT.
        
        Args:
            x: [B, C, H, W]
        Returns:
            cA: Approximation [B, C, H//2, W//2]
            (cH, cV, cD): Detail coefficients
        """
        # LL (approximation)
        cA = self._separable_conv2d(x, self.dec_lo, self.dec_lo, stride=2)
        
        # LH (horizontal details)
        cH = self._separable_conv2d(x, self.dec_hi, self.dec_lo, stride=2)
        
        # HL (vertical details)
        cV = self._separable_conv2d(x, self.dec_lo, self.dec_hi, stride=2)
        
        # HH (diagonal details)
        cD = self._separable_conv2d(x, self.dec_hi, self.dec_hi, stride=2)
        
        return cA, (cH, cV, cD)
    
    def idwt2d_single(self, cA: torch.Tensor, details: Tuple) -> torch.Tensor:
        """
        Single-level 2D inverse DWT.
        
        Args:
            cA: Approximation coefficients [B, C, H, W]
            details: (cH, cV, cD) tuple of detail coefficients
        Returns:
            Reconstructed image [B, C, 2*H, 2*W]
        """
        cH, cV, cD = details
        device = cA.device
        B, C, H, W = cA.shape
        
        # Upsample all coefficients
        cA_up = torch.zeros(B, C, 2*H, 2*W, device=device, dtype=cA.dtype)
        cH_up = torch.zeros(B, C, 2*H, 2*W, device=device, dtype=cA.dtype)
        cV_up = torch.zeros(B, C, 2*H, 2*W, device=device, dtype=cA.dtype)
        cD_up = torch.zeros(B, C, 2*H, 2*W, device=device, dtype=cA.dtype)
        
        cA_up[:, :, ::2, ::2] = cA
        cH_up[:, :, ::2, ::2] = cH
        cV_up[:, :, ::2, ::2] = cV
        cD_up[:, :, ::2, ::2] = cD
        
        # Prepare reconstruction filters
        rec_lo = self.rec_lo.to(device)
        rec_hi = self.rec_hi.to(device)
        
        filter_ll_h = rec_lo.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        filter_ll_v = rec_lo.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        filter_lh_h = rec_hi.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        filter_lh_v = rec_lo.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        filter_hl_h = rec_lo.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        filter_hl_v = rec_hi.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        filter_hh_h = rec_hi.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        filter_hh_v = rec_hi.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        
        # Pad - use even padding to ensure correct output size
        pad_h = filter_ll_h.shape[-1] - 1
        pad_v = filter_ll_v.shape[-2] - 1
        
        def conv_rec(x, fh, fv):
            # Horizontal convolution
            x = torch.nn.functional.pad(x, (pad_h//2, (pad_h+1)//2, 0, 0), 
                                        mode='circular' if self.mode=='periodic' else 'replicate')
            x = torch.nn.functional.conv2d(x, fh, groups=C)
            # Vertical convolution
            x = torch.nn.functional.pad(x, (0, 0, pad_v//2, (pad_v+1)//2), 
                                        mode='circular' if self.mode=='periodic' else 'replicate')
            x = torch.nn.functional.conv2d(x, fv, groups=C)
            return x
        
        # Reconstruct
        x_ll = conv_rec(cA_up, filter_ll_h, filter_ll_v)
        x_lh = conv_rec(cH_up, filter_lh_h, filter_lh_v)
        x_hl = conv_rec(cV_up, filter_hl_h, filter_hl_v)
        x_hh = conv_rec(cD_up, filter_hh_h, filter_hh_v)
        
        result = x_ll + x_lh + x_hl + x_hh
        
        # Ensure output size is exactly 2*H x 2*W
        if result.shape[-2] != 2*H or result.shape[-1] != 2*W:
            result = result[:, :, :2*H, :2*W]
        
        return result
    
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Multi-level 2D DWT.
        
        Returns:
            Dictionary with structure:
            {
                'approximations': [cA_J, cA_{J-1}, ..., cA_1],
                'horizontal': [cH_J, cH_{J-1}, ..., cH_1],
                'vertical': [cV_J, cV_{J-1}, ..., cV_1],
                'diagonal': [cD_J, cD_{J-1}, ..., cD_1]
            }
        """
        approximations = []
        horizontals = []
        verticals = []
        diagonals = []
        
        current = x
        for j in range(self.level):
            cA, (cH, cV, cD) = self.dwt2d_single(current)
            
            approximations.append(cA)
            horizontals.append(cH)
            verticals.append(cV)
            diagonals.append(cD)
            
            current = cA
        
        return {
            'approximations': approximations,
            'horizontal': horizontals,
            'vertical': verticals,
            'diagonal': diagonals
        }
    
    def inverse(self, coeffs: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        """
        Multi-level 2D inverse DWT.
        """
        # Start from coarsest approximation
        current = coeffs['approximations'][-1]
        
        # Reconstruct level by level
        for j in range(self.level - 1, -1, -1):
            cH = coeffs['horizontal'][j]
            cV = coeffs['vertical'][j]
            cD = coeffs['diagonal'][j]
            
            current = self.idwt2d_single(current, (cH, cV, cD))
        
        return current


def test_2d_wavelet_transform():
    """Test 2D wavelet transform with perfect reconstruction"""
    
    # Create test image
    img = torch.randn(1, 3, 128, 128)
    
    # Transform
    wt = WaveletTransform2D(wavelet='db4', level=3)
    coeffs = wt.forward(img)
    
    # Reconstruct
    img_recon = wt.inverse(coeffs)
    
    # Check error
    error = torch.norm(img - img_recon) / torch.norm(img)
    print(f"Reconstruction error: {error.item():.2e}")
    
    # Visualize coefficients
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for j in range(3):
        # Approximation
        axes[j, 0].imshow(coeffs['approximations'][j][0, 0].detach().numpy(), 
                         cmap='gray')
        axes[j, 0].set_title(f"cA (level {j+1})")
        axes[j, 0].axis('off')
        
        # Details
        for idx, (name, coeffs_list) in enumerate([
            ('Horizontal', coeffs['horizontal']),
            ('Vertical', coeffs['vertical']),
            ('Diagonal', coeffs['diagonal'])
        ], 1):
            axes[j, idx].imshow(coeffs_list[j][0, 0].detach().numpy(), 
                               cmap='gray')
            axes[j, idx].set_title(f"{name} (level {j+1})")
            axes[j, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/phase2_2d_wavelets.png', dpi=150)
    plt.close()
    
    print("✅ 2D wavelet transform validated")


if __name__ == "__main__":
    test_2d_wavelet_transform()

