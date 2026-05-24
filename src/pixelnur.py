"""
PixelNur Phase 2 - High-Level Steganography System API

This module provides the main integration layer that orchestrates all components:
- CNN Module: Adaptive embedding mask generation
- Encryption Service: Slice encryption (SHA-256 + XOR)
- LWT Transform: Lifting Wavelet Transform for frequency domain embedding
- Embedding Engine: Adaptive LSB matching embedding
- Metrics Service: PSNR/SSIM calculation and capacity estimation

Requirements:
- 1.1: Connect CNN mask generation to coefficient selection
- 1.9: Implement embedding workflow: preprocess → CNN → LWT → embed → inverse LWT
- 19.1: Implement capacity estimation before embedding
- 17.2: Add error handling for insufficient capacity

Design:
- High-level embed_message() and extract_message() methods
- Automatic capacity checking before embedding
- Comprehensive error handling with descriptive messages
- Support for both text and binary messages
"""

import os
import struct
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
import numpy as np
import cv2

from src.cnn_module import CNNModule
from src.encryption_service import EncryptionService
from src.lwt_transform import LWTTransform, LWTCoefficients
from src.embedding_engine import EmbeddingEngine
from src.metrics_service import MetricsService


class PixelNurError(Exception):
    """Base exception for PixelNur system errors."""
    pass


class InsufficientCapacityError(PixelNurError):
    """Raised when image capacity is insufficient for message."""
    pass


class InvalidImageError(PixelNurError):
    """Raised when image is invalid or unsupported."""
    pass


class EmbeddingError(PixelNurError):
    """Raised when embedding operation fails."""
    pass


class ExtractionError(PixelNurError):
    """Raised when extraction operation fails."""
    pass


class PixelNur:
    """
    High-level API for the PixelNur steganography system.
    
    This class orchestrates all components to provide simple embed/extract operations
    while handling all the complexity of CNN mask generation, LWT transforms,
    encryption, and adaptive embedding.
    
    Example usage:
        # Initialize system
        pixelnur = PixelNur()
        
        # Embed message
        cover_image = cv2.imread('cover.png')
        stego_image, metrics = pixelnur.embed_message(
            cover_image=cover_image,
            message="Secret message",
            encryption_key="my_secure_key_123"
        )
        cv2.imwrite('stego.png', stego_image)
        
        # Extract message
        stego_image = cv2.imread('stego.png')
        message = pixelnur.extract_message(
            stego_image=stego_image,
            encryption_key="my_secure_key_123"
        )
        print(message)  # "Secret message"
    
    Attributes:
        cnn_module: CNN module for mask generation
        encryption_service: Encryption service for message encryption
        lwt_transform: LWT transform for frequency domain operations
        embedding_engine: Embedding engine for adaptive LSB matching
        metrics_service: Metrics service for quality assessment
    """
    
    def __init__(
        self,
        cnn_model_path: Optional[str] = None,
        device: Optional[str] = None,
        base_alpha: float = 0.1,
        cnn_threshold: float = 0.5
    ):
        """
        Initialize PixelNur system with all components.
        
        Args:
            cnn_model_path: Path to trained CNN model checkpoint (optional)
            device: Device for CNN ('cuda', 'cpu', or None for auto-detection)
            base_alpha: Base embedding strength (default: 0.1)
            cnn_threshold: Threshold for CNN mask binarization (default: 0.5)
            
        Requirements: 1.1, 1.9
        """
        # Initialize all components
        self.cnn_module = CNNModule(
            model_path=cnn_model_path,
            device=device
        )
        self.encryption_service = EncryptionService()
        self.lwt_transform = LWTTransform()
        self.embedding_engine = EmbeddingEngine(base_alpha=base_alpha)
        self.metrics_service = MetricsService()
        
        self.cnn_threshold = cnn_threshold
    
    def _validate_image(self, image: np.ndarray, name: str = "image") -> None:
        """
        Validate image format and dimensions.
        
        Args:
            image: Image array to validate
            name: Name of the image for error messages
            
        Raises:
            InvalidImageError: If image is invalid
            
        Requirements: 17.5, 17.6
        """
        if image is None or image.size == 0:
            raise InvalidImageError(f"{name} cannot be empty or None")
        
        if len(image.shape) not in [2, 3]:
            raise InvalidImageError(
                f"{name} must be 2D (grayscale) or 3D (RGB), got shape: {image.shape}"
            )
        
        height, width = image.shape[:2]
        
        # Minimum dimensions: 256×256 (Requirement 17.5)
        if height < 256 or width < 256:
            raise InvalidImageError(
                f"{name} too small: minimum 256×256 pixels, got {width}×{height}"
            )
        
        # Maximum dimensions: 7680×4320 (8K) (Requirement 17.6)
        if height > 4320 or width > 7680:
            raise InvalidImageError(
                f"{name} too large: maximum 7680×4320 pixels, got {width}×{height}"
            )
        
        # Check if RGB image has 3 channels
        if len(image.shape) == 3 and image.shape[2] != 3:
            raise InvalidImageError(
                f"{name} must have 3 channels (RGB), got {image.shape[2]} channels"
            )
    
    def estimate_capacity(
        self,
        cover_image: np.ndarray,
        robustness_level: str = "none"
    ) -> Dict[str, int]:
        """
        Estimate embedding capacity for a cover image.
        
        This method generates the CNN mask and calculates capacity without
        performing actual embedding. Useful for checking if a message will fit
        before attempting to embed it.
        
        Args:
            cover_image: Cover image as numpy array (H, W, C) in BGR format
            robustness_level: Robustness level ('none', 'low', 'medium', 'high')
            
        Returns:
            Dictionary with capacity information:
            {
                'capacity_bytes': int,  # Maximum message size in bytes
                'capacity_bits': int,   # Maximum message size in bits
                'usable_pixels': int,   # Number of usable embedding locations
                'robustness_level': str # Robustness level used
            }
            
        Raises:
            InvalidImageError: If image is invalid
            
        Requirements: 19.1, 19.4, 19.8
        """
        # Validate image
        self._validate_image(cover_image, "cover_image")
        
        # Generate CNN mask
        cnn_mask = self.cnn_module.generate_mask(
            cover_image,
            threshold=self.cnn_threshold
        )
        
        # Calculate capacity
        capacity_bytes = self.metrics_service.estimate_capacity(
            cnn_mask,
            robustness_level=robustness_level
        )
        
        # Calculate additional statistics
        usable_pixels = int(np.sum(cnn_mask > 0))
        capacity_bits = capacity_bytes * 8
        
        return {
            'capacity_bytes': capacity_bytes,
            'capacity_bits': capacity_bits,
            'usable_pixels': usable_pixels,
            'robustness_level': robustness_level
        }
    
    def estimate_capacity_all_levels(
        self,
        cover_image: np.ndarray
    ) -> Dict[str, Dict[str, int]]:
        """
        Estimate capacity for all robustness levels.
        
        Args:
            cover_image: Cover image as numpy array (H, W, C) in BGR format
            
        Returns:
            Dictionary mapping robustness level to capacity info:
            {
                'none': {'capacity_bytes': ..., 'capacity_bits': ..., ...},
                'low': {'capacity_bytes': ..., 'capacity_bits': ..., ...},
                'medium': {'capacity_bytes': ..., 'capacity_bits': ..., ...},
                'high': {'capacity_bytes': ..., 'capacity_bits': ..., ...}
            }
            
        Requirements: 19.4
        """
        capacities = {}
        for level in ['none', 'low', 'medium', 'high']:
            capacities[level] = self.estimate_capacity(cover_image, level)
        return capacities
    
    def embed_message(
        self,
        cover_image: np.ndarray,
        message: Union[str, bytes],
        encryption_key: str,
        robustness_level: str = "none",
        check_capacity: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Embed encrypted message into cover image.
        
        This is the main embedding method that orchestrates the complete workflow:
        1. Validate inputs (image, message, key)
        2. Generate CNN embedding mask
        3. Check capacity (if enabled)
        4. Encrypt message
        5. Apply LWT to cover image
        6. Embed encrypted message using adaptive LSB matching
        7. Apply inverse LWT to reconstruct stego image
        8. Calculate quality metrics (PSNR, SSIM)
        
        Args:
            cover_image: Cover image as numpy array (H, W, C) in BGR format
            message: Message to embed (string or bytes)
            encryption_key: Encryption key (minimum 16 characters)
            robustness_level: Robustness level ('none', 'low', 'medium', 'high')
            check_capacity: Whether to check capacity before embedding (default: True)
            
        Returns:
            Tuple of:
                - stego_image: Stego image with embedded message (same shape as cover)
                - metrics: Dictionary with quality metrics:
                  {
                      'psnr': float,  # PSNR in dB (target: 42-48)
                      'ssim': float,  # SSIM (target: ≥0.91)
                      'capacity_used_bytes': int,  # Message size in bytes
                      'capacity_available_bytes': int,  # Total capacity
                      'capacity_utilization': float  # Percentage used
                  }
        
        Raises:
            InvalidImageError: If cover image is invalid
            InsufficientCapacityError: If message is too large for image
            EmbeddingError: If embedding operation fails
            ValueError: If encryption key is invalid
            
        Requirements:
        - 1.1: Connect CNN mask generation to coefficient selection
        - 1.9: Implement embedding workflow
        - 17.2: Add error handling for insufficient capacity
        - 19.1: Implement capacity estimation before embedding
        """
        try:
            # Step 1: Validate inputs
            self._validate_image(cover_image, "cover_image")
            self.encryption_service.validate_key(encryption_key)
            
            if not message:
                raise ValueError("Message cannot be empty")
            
            # Convert message to bytes if string
            if isinstance(message, str):
                message_bytes = message.encode('utf-8')
            else:
                message_bytes = message
            
            # Step 2: Generate CNN embedding mask
            cnn_mask = self.cnn_module.generate_mask(
                cover_image,
                threshold=self.cnn_threshold
            )
            
            # Step 3: Check capacity (if enabled)
            if check_capacity:
                capacity_bytes = self.metrics_service.estimate_capacity(
                    cnn_mask,
                    robustness_level=robustness_level
                )
                
                if len(message_bytes) > capacity_bytes:
                    raise InsufficientCapacityError(
                        f"Message too large for cover image. "
                        f"Message size: {len(message_bytes)} bytes, "
                        f"Available capacity: {capacity_bytes} bytes. "
                        f"Suggestions: "
                        f"1) Use a larger cover image, "
                        f"2) Reduce message size, "
                        f"3) Lower robustness level (current: {robustness_level})"
                    )
            
            # Step 4: Encrypt message
            encrypted_message = self.encryption_service.encrypt(
                message_bytes,
                encryption_key
            )

            # Step 5: Apply ECC encoding if robustness is enabled
            from src.robustness_layer import RobustnessLayer
            robustness_layer = RobustnessLayer(robustness_level)
            encoded_message = robustness_layer.encode(encrypted_message)

            # Step 6: Build 56-bit header (PNv2 + robustness byte + length uint16 BE)
            _robustness_map = {'none': 0x00, 'low': 0x01, 'medium': 0x02, 'high': 0x03}
            header_bytes = (b"PNv2"
                            + bytes([_robustness_map[robustness_level]])
                            + struct.pack('>H', len(encoded_message)))

            # Step 7: Build bit stream and embed in G-channel (channel 1) LSBs,
            # row-major order.  Pixel LSBs survive lossless PNG round-trips
            # perfectly — no LWT round-trip corruption possible.
            all_bytes_to_embed = header_bytes + bytes(encoded_message)
            all_bits = []
            for _byte in all_bytes_to_embed:
                for _b in range(8):
                    all_bits.append((_byte >> (7 - _b)) & 1)

            height, width = cover_image.shape[:2]
            if len(all_bits) > height * width:
                raise InsufficientCapacityError(
                    f"Message too large for cover image. "
                    f"Required: {len(all_bits)} bits ({len(all_bits)//8} bytes), "
                    f"Available: {height * width} bits ({height * width // 8} bytes). "
                    f"Use a larger image or shorter message."
                )

            stego_image = cover_image.copy()
            bit_idx = 0
            for _i in range(height):
                for _j in range(width):
                    if bit_idx >= len(all_bits):
                        break
                    stego_image[_i, _j, 1] = (int(stego_image[_i, _j, 1]) & 0xFE) | all_bits[bit_idx]
                    bit_idx += 1
                if bit_idx >= len(all_bits):
                    break

            # Step 8: Calculate quality metrics
            psnr, ssim = self.metrics_service.calculate_metrics(
                cover_image,
                stego_image
            )

            # Pixel-domain capacity: (H*W - 56 header bits) / 8 bytes
            capacity_bytes = (height * width - 56) // 8

            metrics = {
                'psnr': psnr,
                'ssim': ssim,
                'capacity_used_bytes': len(message_bytes),
                'capacity_available_bytes': capacity_bytes,
                'capacity_utilization': (len(message_bytes) / capacity_bytes * 100) if capacity_bytes > 0 else 0
            }
            
            return stego_image, metrics
            
        except (InvalidImageError, InsufficientCapacityError, ValueError) as e:
            # Re-raise known errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise EmbeddingError(f"Embedding failed: {str(e)}") from e
    
    def extract_message(
        self,
        stego_image: np.ndarray,
        encryption_key: str,
        cover_image: Optional[np.ndarray] = None
    ) -> str:
        """
        Extract and decrypt message from stego image.
        
        Note: This is a placeholder for Task 10 (Extraction Engine).
        Full extraction implementation will be added in Sprint 3-4.
        
        Args:
            stego_image: Stego image with embedded message
            encryption_key: Encryption key used during embedding
            cover_image: Optional cover image for mask regeneration
            
        Returns:
            Decrypted message as string
            
        Raises:
            NotImplementedError: Extraction engine not yet implemented
            
        Requirements: 1.9 (extraction workflow)
        """
        raise NotImplementedError(
            "Extraction engine will be implemented in Task 10 (Sprint 3-4). "
            "Current implementation focuses on embedding workflow only."
        )
    
    def get_system_info(self) -> Dict[str, any]:
        """
        Get information about the PixelNur system configuration.
        
        Returns:
            Dictionary with system information:
            {
                'cnn_model': {...},  # CNN model info
                'base_alpha': float,  # Embedding strength
                'cnn_threshold': float,  # CNN mask threshold
                'version': str  # System version
            }
        """
        return {
            'cnn_model': self.cnn_module.get_model_info(),
            'base_alpha': self.embedding_engine.base_alpha,
            'cnn_threshold': self.cnn_threshold,
            'version': '2.0.0-alpha'
        }


# Convenience functions for simple usage

def embed(
    cover_image_path: str,
    message: Union[str, bytes],
    encryption_key: str,
    output_path: str,
    cnn_model_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to embed message from file paths.
    
    Args:
        cover_image_path: Path to cover image file
        message: Message to embed
        encryption_key: Encryption key (minimum 16 characters)
        output_path: Path to save stego image
        cnn_model_path: Optional path to CNN model checkpoint
        
    Returns:
        Dictionary with quality metrics (PSNR, SSIM, capacity info)
        
    Example:
        metrics = embed(
            cover_image_path='cover.png',
            message='Secret message',
            encryption_key='my_secure_key_123',
            output_path='stego.png'
        )
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    """
    # Load cover image
    cover_image = cv2.imread(cover_image_path)
    if cover_image is None:
        raise FileNotFoundError(f"Could not load cover image: {cover_image_path}")
    
    # Initialize system
    pixelnur = PixelNur(cnn_model_path=cnn_model_path)
    
    # Embed message
    stego_image, metrics = pixelnur.embed_message(
        cover_image=cover_image,
        message=message,
        encryption_key=encryption_key
    )
    
    # Save stego image
    cv2.imwrite(output_path, stego_image)
    
    return metrics


def extract(
    stego_image_path: str,
    encryption_key: str,
    cover_image_path: Optional[str] = None,
    cnn_model_path: Optional[str] = None
) -> str:
    """
    Convenience function to extract message from file paths.
    
    Args:
        stego_image_path: Path to stego image file
        encryption_key: Encryption key used during embedding
        cover_image_path: Optional path to cover image
        cnn_model_path: Optional path to CNN model checkpoint
        
    Returns:
        Decrypted message as string
        
    Raises:
        NotImplementedError: Extraction not yet implemented
    """
    raise NotImplementedError(
        "Extraction engine will be implemented in Task 10 (Sprint 3-4)"
    )
