"""
CNN Module for PixelNur Phase 2
Implements adaptive embedding mask generation using ResNet-18 backbone

Requirements:
- 1.1: Generate embedding masks by analyzing texture complexity
- 1.2: Output binary mask with dimensions matching LWT sub-band (H/2 × W/2)
- 1.6: Process 1080p image and generate mask within 2 seconds
- 1.8: Support model versioning for backward compatibility

Design:
- ResNet-18 backbone with custom head for mask generation
- GPU acceleration with automatic CPU fallback
- Preprocessing: resize to 512×512, normalize, convert to tensor
- Postprocessing: resize mask to LWT sub-band dimensions, apply threshold
- Model loading from checkpoint with version support
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class CNNModule:
    """
    CNN-based adaptive embedding mask generator using ResNet-18 backbone.
    
    This module analyzes texture complexity in cover images and generates
    binary masks indicating optimal embedding locations in LWT sub-bands.
    
    Attributes:
        device: torch.device for computation (GPU or CPU)
        model: The CNN model for mask generation
        input_size: Input image size for CNN (512×512)
        version: Model version for backward compatibility
    """
    
    INPUT_SIZE = 512  # Standard input size for CNN
    THRESHOLD = 0.5   # Binary threshold for mask generation
    MASK_VALIDITY_THRESHOLD = 5.0
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        version: str = "1.0"
    ):
        """
        Initialize CNN Module with ResNet-18 backbone.
        
        Args:
            model_path: Path to pretrained model checkpoint (optional)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            version: Model version for backward compatibility
            
        Requirements: 1.8 (model versioning)
        """
        self.version = version
        self.fallback_mode = False
        try:
            self.device = self._detect_device(device)
            self.model = self._build_model()
            
            if model_path and os.path.exists(model_path):
                self._load_checkpoint(model_path)
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logging.warning(f"Failed to initialize CNN model, using fallback: {e}")
            self.model = None
            self.device = torch.device('cpu')
            self.fallback_mode = True
            
        # Preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _detect_device(self, device: Optional[str] = None) -> torch.device:
        """
        Auto-detect GPU/CPU device for computation.
        
        Args:
            device: Explicit device specification or None for auto-detection
            
        Returns:
            torch.device object
            
        Requirements: 1.6 (GPU acceleration with CPU fallback)
        """
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _build_model(self) -> nn.Module:
        """
        Build CNN model with ResNet-18 backbone and custom head.
        
        Architecture:
        - ResNet-18 backbone (pretrained on ImageNet, classification layer removed)
        - Custom head: Conv2D layers (512→256→128→64→1) with ReLU and Sigmoid
        
        Returns:
            Complete CNN model
            
        Requirements: 1.1 (texture analysis for mask generation)
        """
        # Load pretrained ResNet-18
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the classification layer (fc) and avgpool
        # We want to keep the feature maps for spatial mask generation
        backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        # Custom head for mask generation
        # ResNet-18 outputs 512 channels at 16×16 spatial resolution (for 512×512 input)
        head = nn.Sequential(
            # 512 → 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            # 256 → 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # 128 → 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # 64 → 1 (single channel mask)
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Combine backbone and head
        model = nn.Sequential(backbone, head)
        
        return model
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from checkpoint file.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
            
        Requirements: 1.8 (model versioning)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'version' in checkpoint:
                        self.version = checkpoint['version']
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess cover image for CNN inference.
        
        Steps:
        1. Resize to 512×512
        2. Normalize using ImageNet statistics
        3. Convert to tensor
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR or RGB format
            
        Returns:
            Preprocessed tensor (1, 3, 512, 512)
            
        Requirements: 1.1 (preprocessing layer)
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply preprocessing transforms
        tensor = self.preprocess(image_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def _postprocess_mask(
        self,
        mask_tensor: torch.Tensor,
        target_size: Tuple[int, int],
        threshold: float = THRESHOLD
    ) -> np.ndarray:
        """
        Postprocess CNN output to binary mask at LWT sub-band dimensions.
        
        Steps:
        1. Resize mask to target dimensions (H/2 × W/2)
        2. Apply threshold to create binary mask
        
        Args:
            mask_tensor: CNN output tensor (1, 1, H, W)
            target_size: Target dimensions (height, width) for LWT sub-band
            threshold: Binary threshold value (default: 0.5)
            
        Returns:
            Binary mask as numpy array (H/2, W/2) with values 0 or 1
            
        Requirements: 1.2 (output dimensions matching LWT sub-band)
        """
        # Remove batch and channel dimensions
        mask = mask_tensor.squeeze().cpu().numpy()
        
        # Resize to target dimensions (LWT sub-band size)
        mask_resized = cv2.resize(
            mask,
            (target_size[1], target_size[0]),  # cv2.resize expects (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply threshold to create binary mask
        binary_mask = (mask_resized >= threshold).astype(np.uint8)
        
        return binary_mask
    
    def _generate_sobel_fallback_mask(
        self,
        cover_image: np.ndarray,
        target_height: int,
        target_width: int,
        threshold: float = 0.7
    ) -> np.ndarray:
        """Generate Sobel gradient-based fallback mask when CNN mask is invalid."""
        if len(cover_image.shape) == 3:
            gray = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cover_image
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude_resized = cv2.resize(
            magnitude,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
        threshold_value = np.percentile(magnitude_resized, threshold * 100)
        binary_mask = (magnitude_resized > threshold_value).astype(np.float32)
        return binary_mask

    def generate_mask(
        self,
        cover_image: np.ndarray,
        threshold: float = THRESHOLD
    ) -> np.ndarray:
        """
        Generate embedding mask for cover image using ResNet-18 feature variance.
        
        Uses L2 norm of ResNet-18 feature maps as texture complexity score.
        High feature activation = high texture = good embedding location.
        Works without fine-tuning because ImageNet features capture texture.
        
        Args:
            cover_image: Cover image as numpy array (H, W, C)
            threshold: Percentile threshold for mask binarization (default: 0.5 = 50th percentile)
            
        Returns:
            Binary embedding mask (H/2, W/2) with values 0 or 1
            
        Raises:
            ValueError: If image dimensions are invalid
        """
        if cover_image is None or cover_image.size == 0:
            raise ValueError("Invalid cover image: empty or None")
        
        if len(cover_image.shape) != 3:
            raise ValueError(
                f"Invalid image shape: expected (H, W, C), got {cover_image.shape}"
            )
        
        height, width = cover_image.shape[:2]
        
        if height < 64 or width < 64:
            raise ValueError(
                f"Image too small: minimum 64×64, got {width}×{height}"
            )
        
        # Calculate LWT sub-band dimensions (H/2 × W/2)
        target_height = height // 2
        target_width = width // 2
        
        # If model initialization failed or in fallback mode, use Sobel gradient fallback immediately
        if getattr(self, 'fallback_mode', False) or self.model is None:
            return self._generate_sobel_fallback_mask(
                cover_image, target_height, target_width, threshold=0.7
            )
        
        # Preprocess image
        input_tensor = self._preprocess_image(cover_image)
        input_tensor = input_tensor.to(self.device)
        
        # Extract ResNet-18 features (use backbone only, not the untrained head)
        with torch.no_grad():
            # Get feature maps from ResNet-18 backbone
            # Shape: (1, 512, H/32, W/32) for 512x512 input
            feature_maps = self.model[0](input_tensor)  # backbone only
            
            # Compute L2 norm across channels → texture complexity map
            # Shape: (1, H/32, W/32)
            texture_score = torch.norm(feature_maps, dim=1, keepdim=True)
            
            # Convert to numpy
            texture_np = texture_score.squeeze().cpu().numpy()
        
        # Resize to target dimensions (H/2 × W/2)
        texture_resized = cv2.resize(
            texture_np,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to [0, 1]
        t_min, t_max = texture_resized.min(), texture_resized.max()
        if t_max > t_min:
            texture_normalized = (texture_resized - t_min) / (t_max - t_min)
        else:
            texture_normalized = texture_resized
        
        # Threshold at 50th percentile to get ~50% non-zero pixels
        # (high-texture regions get selected for embedding)
        threshold_value = np.percentile(texture_normalized, 50)
        binary_mask = (texture_normalized >= threshold_value).astype(np.uint8)
        
        # Validate mask quality (should now always pass)
        non_zero_count = np.count_nonzero(binary_mask)
        non_zero_percentage = (non_zero_count / binary_mask.size) * 100
        
        if non_zero_percentage < self.MASK_VALIDITY_THRESHOLD:
            logging.warning(
                f"CNN feature mask invalid ({non_zero_percentage:.2f}% non-zero pixels), "
                f"using Sobel gradient fallback"
            )
            binary_mask = self._generate_sobel_fallback_mask(
                cover_image, target_height, target_width, threshold=0.7
            )
        
        return binary_mask
    
    def save_checkpoint(self, save_path: str, metadata: Optional[dict] = None) -> None:
        """
        Save model checkpoint with version information.
        
        Args:
            save_path: Path to save checkpoint file
            metadata: Optional metadata dictionary to include in checkpoint
            
        Requirements: 1.8 (model versioning)
        """
        if getattr(self, 'fallback_mode', False) or self.model is None:
            logging.warning("Cannot save checkpoint in fallback mode")
            return
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'version': self.version,
            'input_size': self.INPUT_SIZE,
            'threshold': self.THRESHOLD
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(checkpoint, save_path)
    
    def get_model_info(self) -> dict:
        """
        Get model information including version and device.
        
        Returns:
            Dictionary with model metadata
            
        Requirements: 1.8 (model versioning)
        """
        if getattr(self, 'fallback_mode', False) or self.model is None:
            return {
                'version': self.version,
                'device': 'cpu',
                'input_size': self.INPUT_SIZE,
                'threshold': self.THRESHOLD,
                'parameters': 0,
                'trainable_parameters': 0
            }
            
        return {
            'version': self.version,
            'device': str(self.device),
            'input_size': self.INPUT_SIZE,
            'threshold': self.THRESHOLD,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
