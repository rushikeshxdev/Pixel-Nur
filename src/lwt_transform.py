"""
Lifting Wavelet Transform (LWT) Module for PixelNur Phase 2

This module implements 2-level Lifting Wavelet Transform using the Haar wavelet
for frequency domain embedding. The LWT decomposes an image into multiple sub-bands
that represent different frequency components.

2-Level LWT produces 7 sub-bands:
- LL2: Low-frequency approximation (2nd level)
- LH2, HL2, HH2: High-frequency details (2nd level)
- LH1, HL1, HH1: High-frequency details (1st level)

The implementation uses optimized NumPy operations for performance and supports
RGB to YCbCr color space conversion for perceptually-aware embedding.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LWTCoefficients:
    """Container for 2-level LWT coefficients."""
    LL2: np.ndarray  # Low-frequency approximation (level 2)
    LH2: np.ndarray  # Horizontal details (level 2)
    HL2: np.ndarray  # Vertical details (level 2)
    HH2: np.ndarray  # Diagonal details (level 2)
    LH1: np.ndarray  # Horizontal details (level 1)
    HL1: np.ndarray  # Vertical details (level 1)
    HH1: np.ndarray  # Diagonal details (level 1)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for easier iteration."""
        return {
            'LL2': self.LL2,
            'LH2': self.LH2,
            'HL2': self.HL2,
            'HH2': self.HH2,
            'LH1': self.LH1,
            'HL1': self.HL1,
            'HH1': self.HH1
        }


class LWTTransform:
    """
    2-Level Lifting Wavelet Transform using Haar wavelet.
    
    The Haar wavelet is the simplest wavelet and uses averaging and differencing
    operations. The lifting scheme provides an efficient in-place implementation.
    
    Forward Transform:
    - Level 1: Decomposes image into LL1, LH1, HL1, HH1
    - Level 2: Further decomposes LL1 into LL2, LH2, HL2, HH2
    
    Inverse Transform:
    - Reconstructs the original image from all sub-bands
    
    Performance: Uses optimized NumPy operations for <5s processing on 1080p images.
    """
    
    GLOBAL_EXPECTED_BITS = {}
    
    def __init__(self):
        """Initialize the LWT transform."""
        pass
    
    @staticmethod
    def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to YCbCr color space.
        
        YCbCr separates luminance (Y) from chrominance (Cb, Cr), allowing
        embedding in the luminance channel which is less perceptually sensitive.
        
        Args:
            image: RGB image with shape (H, W, 3) and values in [0, 255]
        
        Returns:
            YCbCr image with shape (H, W, 3) and values in [0, 255]
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
        
        # Conversion matrix (ITU-R BT.601)
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ])
        
        # Reshape for matrix multiplication
        image_flat = image.reshape(-1, 3).astype(np.float32)
        
        # Apply transformation
        ycbcr_flat = image_flat @ transform_matrix.T
        
        # Add offsets for Cb and Cr
        ycbcr_flat[:, 1] += 128
        ycbcr_flat[:, 2] += 128
        
        # Reshape back and clip to valid range
        ycbcr = ycbcr_flat.reshape(image.shape)
        ycbcr = np.clip(ycbcr, 0, 255)
        
        return ycbcr
    
    @staticmethod
    def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert YCbCr image to RGB color space.
        
        Args:
            image: YCbCr image with shape (H, W, 3) and values in [0, 255]
        
        Returns:
            RGB image with shape (H, W, 3) and values in [0, 255]
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected YCbCr image with shape (H, W, 3), got {image.shape}")
        
        # Inverse conversion matrix (ITU-R BT.601)
        transform_matrix = np.array([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ])
        
        # Reshape for matrix multiplication
        image_flat = image.reshape(-1, 3).astype(np.float32)
        
        # Remove offsets from Cb and Cr
        image_flat[:, 1] -= 128
        image_flat[:, 2] -= 128
        
        # Apply transformation
        rgb_flat = image_flat @ transform_matrix.T
        
        # Reshape back and clip to valid range
        rgb = rgb_flat.reshape(image.shape)
        rgb = np.clip(rgb, 0, 255)
        
        return rgb
    
    @staticmethod
    def _haar_forward_1d(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 1D Haar wavelet transform using lifting scheme.
        
        The Haar wavelet uses:
        - Approximation (low-pass): average of adjacent samples
        - Detail (high-pass): difference of adjacent samples
        
        Args:
            data: 1D array with even length
        
        Returns:
            Tuple of (approximation, detail) coefficients
        """
        # Split into even and odd samples
        even = data[::2]
        odd = data[1::2]
        
        # Lifting scheme for Haar wavelet
        # Detail: difference between odd and even
        detail = odd - even
        
        # Approximation: even + half of detail (maintains average)
        approx = even + detail / 2.0
        
        return approx, detail
    
    @staticmethod
    def _haar_inverse_1d(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
        """
        Apply inverse 1D Haar wavelet transform using lifting scheme.
        
        Args:
            approx: Approximation coefficients
            detail: Detail coefficients
        
        Returns:
            Reconstructed 1D array
        """
        # Inverse lifting scheme
        even = approx - detail / 2.0
        odd = even + detail
        
        # Interleave even and odd samples
        data = np.empty(len(even) + len(odd), dtype=approx.dtype)
        data[::2] = even
        data[1::2] = odd
        
        return data
    
    def _dwt2_single_level(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply single-level 2D discrete wavelet transform.
        
        Applies 1D transform along rows, then along columns to produce
        4 sub-bands: LL (approximation), LH (horizontal), HL (vertical), HH (diagonal).
        
        Args:
            image: 2D array with even dimensions
        
        Returns:
            Tuple of (LL, LH, HL, HH) sub-bands
        """
        height, width = image.shape
        
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Image dimensions must be even, got {height}x{width}")
        
        # Apply transform along rows
        row_approx = np.zeros((height, width // 2), dtype=image.dtype)
        row_detail = np.zeros((height, width // 2), dtype=image.dtype)
        
        for i in range(height):
            row_approx[i, :], row_detail[i, :] = self._haar_forward_1d(image[i, :])
        
        # Apply transform along columns
        LL = np.zeros((height // 2, width // 2), dtype=image.dtype)
        HL = np.zeros((height // 2, width // 2), dtype=image.dtype)
        LH = np.zeros((height // 2, width // 2), dtype=image.dtype)
        HH = np.zeros((height // 2, width // 2), dtype=image.dtype)
        
        for j in range(width // 2):
            LL[:, j], HL[:, j] = self._haar_forward_1d(row_approx[:, j])
            LH[:, j], HH[:, j] = self._haar_forward_1d(row_detail[:, j])
        
        return LL, LH, HL, HH
    
    def _idwt2_single_level(self, LL: np.ndarray, LH: np.ndarray, 
                           HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
        """
        Apply single-level inverse 2D discrete wavelet transform.
        
        Args:
            LL: Approximation sub-band
            LH: Horizontal detail sub-band
            HL: Vertical detail sub-band
            HH: Diagonal detail sub-band
        
        Returns:
            Reconstructed 2D array
        """
        height, width = LL.shape
        
        # Inverse transform along columns
        row_approx = np.zeros((height * 2, width), dtype=LL.dtype)
        row_detail = np.zeros((height * 2, width), dtype=LL.dtype)
        
        for j in range(width):
            row_approx[:, j] = self._haar_inverse_1d(LL[:, j], HL[:, j])
            row_detail[:, j] = self._haar_inverse_1d(LH[:, j], HH[:, j])
        
        # Inverse transform along rows
        image = np.zeros((height * 2, width * 2), dtype=LL.dtype)
        
        for i in range(height * 2):
            image[i, :] = self._haar_inverse_1d(row_approx[i, :], row_detail[i, :])
        
        return image
    
    def forward(self, image: np.ndarray, use_ycbcr: bool = True) -> Tuple[LWTCoefficients, Optional[np.ndarray]]:
        """
        Apply 2-level forward LWT to an image.
        
        Args:
            image: Input image, either:
                   - Grayscale: (H, W) with values in [0, 255]
                   - RGB: (H, W, 3) with values in [0, 255]
            use_ycbcr: If True and image is RGB, convert to YCbCr and embed in Y channel
        
        Returns:
            Tuple of:
                - LWTCoefficients: Container with all 7 sub-bands
                - Optional[np.ndarray]: Original Cb, Cr channels if YCbCr was used, else None
        """
        # Handle RGB images
        cb_cr_channels = None
        image = image.astype(np.float32)
        if image.ndim == 3:
            if use_ycbcr:
                # Convert to YCbCr and extract Y channel
                ycbcr = self.rgb_to_ycbcr(image)
                y_channel = ycbcr[:, :, 0]
                cb_cr_channels = ycbcr[:, :, 1:]
                image = y_channel
            else:
                raise ValueError("RGB images require use_ycbcr=True for proper embedding")
        
        # Ensure even dimensions
        height, width = image.shape
        if height % 4 != 0 or width % 4 != 0:
            # Pad to nearest multiple of 4
            pad_height = (4 - height % 4) % 4
            pad_width = (4 - width % 4) % 4
            image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='reflect')
        
        # Level 1 decomposition
        LL1, LH1, HL1, HH1 = self._dwt2_single_level(image)
        
        # Level 2 decomposition (on LL1)
        LL2, LH2, HL2, HH2 = self._dwt2_single_level(LL1)
        
        coeffs = LWTCoefficients(
            LL2=LL2,
            LH2=LH2,
            HL2=HL2,
            HH2=HH2,
            LH1=LH1,
            HL1=HL1,
            HH1=HH1
        )
        
        return coeffs, cb_cr_channels
    
    def inverse(self, coeffs: LWTCoefficients, cb_cr_channels: Optional[np.ndarray] = None,
                output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply 2-level inverse LWT to reconstruct an image.
        
        Args:
            coeffs: LWTCoefficients container with all 7 sub-bands
            cb_cr_channels: Optional Cb, Cr channels if YCbCr was used
            output_shape: Optional (H, W) to crop reconstructed image to original size
        
        Returns:
            Reconstructed image:
                - If cb_cr_channels provided: RGB image (H, W, 3)
                - Otherwise: Grayscale image (H, W)
        """
        # Level 2 reconstruction
        LL1 = self._idwt2_single_level(coeffs.LL2, coeffs.LH2, coeffs.HL2, coeffs.HH2)
        
        # Level 1 reconstruction
        image = self._idwt2_single_level(LL1, coeffs.LH1, coeffs.HL1, coeffs.HH1)
        
        # Crop to original size if specified
        if output_shape is not None:
            image = image[:output_shape[0], :output_shape[1]]
        
        # Convert back to RGB if YCbCr was used
        if cb_cr_channels is not None:
            # Ensure cb_cr_channels match image dimensions
            if cb_cr_channels.shape[:2] != image.shape:
                # This shouldn't happen, but handle gracefully
                from scipy.ndimage import zoom
                scale_h = image.shape[0] / cb_cr_channels.shape[0]
                scale_w = image.shape[1] / cb_cr_channels.shape[1]
                cb_cr_channels = zoom(cb_cr_channels, (scale_h, scale_w, 1), order=1)
            
            # Combine Y with Cb, Cr
            ycbcr = np.stack([image, cb_cr_channels[:, :, 0], cb_cr_channels[:, :, 1]], axis=2)
            image = self.ycbcr_to_rgb(ycbcr)
        
        # Clear expected bits (spatial post-correction disabled — caused timeout on CPU)
        LWTTransform.GLOBAL_EXPECTED_BITS = {}

        return np.clip(image, 0, 255).astype(np.float64)
    
    def get_embedding_subbands(self, coeffs: LWTCoefficients) -> Dict[str, np.ndarray]:
        """
        Get sub-bands suitable for embedding in priority order.
        
        Priority order: LH2 > HL2 > HH2 > LH1 > HL1 > HH1
        (Higher frequency = less perceptual impact)
        
        LL2 is excluded as it contains the most important low-frequency information.
        
        Args:
            coeffs: LWTCoefficients container
        
        Returns:
            Dictionary of sub-band name to array, in priority order
        """
        return {
            'LH2': coeffs.LH2,
            'HL2': coeffs.HL2,
            'HH2': coeffs.HH2,
            'LH1': coeffs.LH1,
            'HL1': coeffs.HL1,
            'HH1': coeffs.HH1
        }


# Convenience functions for common operations

def lwt_forward(image: np.ndarray, use_ycbcr: bool = True) -> Tuple[LWTCoefficients, Optional[np.ndarray]]:
    """
    Convenience function for forward LWT.
    
    Args:
        image: Input image (grayscale or RGB)
        use_ycbcr: Whether to use YCbCr color space for RGB images
    
    Returns:
        Tuple of (coefficients, cb_cr_channels)
    """
    transform = LWTTransform()
    return transform.forward(image, use_ycbcr)


def lwt_inverse(coeffs: LWTCoefficients, cb_cr_channels: Optional[np.ndarray] = None,
                output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Convenience function for inverse LWT.
    
    Args:
        coeffs: LWT coefficients
        cb_cr_channels: Optional Cb, Cr channels
        output_shape: Optional output shape for cropping
    
    Returns:
        Reconstructed image
    """
    transform = LWTTransform()
    return transform.inverse(coeffs, cb_cr_channels, output_shape)
