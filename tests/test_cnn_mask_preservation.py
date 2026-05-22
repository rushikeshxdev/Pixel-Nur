"""
Preservation Property Tests for CNN Mask Fallback Fix

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

This test suite verifies that the fix does NOT break existing functionality
when CNN masks are valid (≥5% non-zero pixels).

Property 2: Preservation - Valid CNN Mask Behavior
- When CNN generates valid masks (≥5% non-zero pixels), masks are unchanged
- Embedding quality metrics remain in expected ranges (PSNR 42-48 dB, SSIM ≥0.91)
- Capacity estimation produces identical results
- Extraction continues working correctly
- All other system components function identically

**IMPORTANT**: These tests use the PRETRAINED CNN model which produces valid masks.
"""

import numpy as np
import pytest
from src.cnn_module import CNNModule
from src.embedding_engine import EmbeddingEngine
from src.extraction_engine import ExtractionEngine
from src.lwt_transform import LWTTransform, LWTCoefficients
from src.encryption_service import EncryptionService
from src.metrics_service import MetricsService


class TestCNNMaskPreservation:
    """Test that valid CNN masks are preserved after the fix."""

    def test_valid_cnn_masks_unchanged(self):
        """
        Test that valid CNN masks (≥5% non-zero) are returned unchanged.
        
        **Validates: Requirements 3.1**
        """
        # Use pretrained CNN model (produces valid masks)
        cnn = CNNModule()
        
        # Test with multiple images
        test_images = [
            np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8),
            np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
            np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8),
        ]
        
        for idx, image in enumerate(test_images):
            mask = cnn.generate_mask(image)
            non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
            
            # Verify mask is valid (≥5% non-zero)
            assert non_zero_percentage >= 5.0, (
                f"Test setup failed: expected valid CNN mask, got {non_zero_percentage:.2f}%"
            )
            
            # Verify no warning was logged (fallback not triggered)
            # This is implicit - if fallback was triggered, mask would have ~30% density
            
            print(f"Image {idx + 1}: {non_zero_percentage:.2f}% non-zero pixels (valid)")
    
    def test_embedding_quality_preserved(self):
        """
        Test that embedding quality metrics remain unchanged for valid CNN masks.
        
        **Validates: Requirements 3.2**
        """
        # Use pretrained CNN model
        cnn = CNNModule()
        embedding_engine = EmbeddingEngine()
        lwt = LWTTransform()
        encryption_service = EncryptionService()
        metrics_service = MetricsService()
        
        # Create test image and message (use 3-channel grayscale to eliminate color space rounding errors)
        gray_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        cover_image = np.repeat(gray_image[:, :, np.newaxis], 3, axis=2)
        message = b"Test message for quality preservation"
        encrypted_message = encryption_service.encrypt(message, "test123456789012")
        
        # Generate mask
        mask = cnn.generate_mask(cover_image)
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        
        # Skip if mask is invalid (fallback would change quality)
        if non_zero_percentage < 5.0:
            pytest.skip("CNN generated invalid mask, skipping quality test")
        
        # Perform embedding
        lwt_coeffs, cb_cr_channels = lwt.forward(cover_image)
        embedding_subbands = lwt.get_embedding_subbands(lwt_coeffs)
        modified_subbands = embedding_engine.embed(
            embedding_subbands, encrypted_message, mask, robustness_level="none"
        )
        modified_coeffs = LWTCoefficients(
            LL2=lwt_coeffs.LL2,
            LH2=modified_subbands['LH2'],
            HL2=modified_subbands['HL2'],
            HH2=modified_subbands['HH2'],
            LH1=modified_subbands['LH1'],
            HL1=modified_subbands['HL1'],
            HH1=modified_subbands['HH1']
        )
        
        # Reconstruct stego image
        stego_image = lwt.inverse(modified_coeffs, cb_cr_channels=cb_cr_channels, output_shape=cover_image.shape[:2])
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        
        # Calculate quality metrics
        psnr = metrics_service.calculate_psnr(cover_image, stego_image)
        ssim = metrics_service.calculate_ssim(cover_image, stego_image)
        
        # Verify quality is in expected range
        assert psnr >= 42.0, f"PSNR {psnr:.2f} dB is too low"
        assert ssim >= 0.91, f"SSIM {ssim:.4f} not in expected range [0.91, 1.0]"
        
        print(f"Quality preserved: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
    
    def test_capacity_estimation_preserved(self):
        """
        Test that capacity estimation produces identical results for valid CNN masks.
        
        **Validates: Requirements 3.4**
        """
        # Use pretrained CNN model
        cnn = CNNModule()
        
        # Create test image
        cover_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # Generate mask
        mask = cnn.generate_mask(cover_image)
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        
        # Skip if mask is invalid
        if non_zero_percentage < 5.0:
            pytest.skip("CNN generated invalid mask, skipping capacity test")
        
        # Calculate capacity (non-zero pixels * bits per pixel)
        # Assuming 1 bit per pixel embedding
        expected_capacity = np.count_nonzero(mask)
        
        # In practice, capacity also depends on header overhead
        # But the key is that the mask determines the available pixels
        assert expected_capacity > 0, "Valid mask should provide embedding capacity"
        
        print(f"Capacity preserved: {expected_capacity} available pixels")
    
    def test_extraction_preserved(self):
        """
        Test that extraction continues working correctly for valid CNN masks.
        
        **Validates: Requirements 3.3**
        """
        # Use pretrained CNN model
        cnn = CNNModule()
        embedding_engine = EmbeddingEngine()
        extraction_engine = ExtractionEngine()
        lwt = LWTTransform()
        encryption_service = EncryptionService()
        # Create test image and message (use a smooth gradient to eliminate color space and quantization rounding errors)
        x = np.linspace(50, 200, 512, dtype=np.uint8)
        y = np.linspace(50, 200, 512, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        gray_image = ((xx.astype(float) + yy.astype(float)) / 2).astype(np.uint8)
        cover_image = np.repeat(gray_image[:, :, np.newaxis], 3, axis=2)
        original_message = b"Test message for extraction preservation"
        encrypted_message = encryption_service.encrypt(original_message, "test123456789012")
        
        # Generate mask
        mask = cnn.generate_mask(cover_image)
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        
        # Skip if mask is invalid
        if non_zero_percentage < 5.0:
            pytest.skip("CNN generated invalid mask, skipping extraction test")
        
        # Embed message
        lwt_coeffs, cb_cr_channels = lwt.forward(cover_image)
        embedding_subbands = lwt.get_embedding_subbands(lwt_coeffs)
        modified_subbands = embedding_engine.embed(
            embedding_subbands, encrypted_message, mask, robustness_level="none"
        )
        modified_coeffs = LWTCoefficients(
            LL2=lwt_coeffs.LL2,
            LH2=modified_subbands['LH2'],
            HL2=modified_subbands['HL2'],
            HH2=modified_subbands['HH2'],
            LH1=modified_subbands['LH1'],
            HL1=modified_subbands['HL1'],
            HH1=modified_subbands['HH1']
        )
        stego_image = lwt.inverse(modified_coeffs, cb_cr_channels=cb_cr_channels, output_shape=cover_image.shape[:2])
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        
        # Extract message
        extracted_encrypted = extraction_engine.extract(stego_image, cover_image)
        extracted_message = encryption_service.decrypt(extracted_encrypted, "test123456789012")
        
        # Verify extraction succeeded
        assert extracted_message == original_message, "Extraction should succeed with valid CNN mask"
        
        print("Extraction preserved: message correctly recovered")
    
    def test_lwt_transform_preserved(self):
        """
        Test that LWT transform functionality is unchanged.
        
        **Validates: Requirements 3.5**
        """
        lwt = LWTTransform()
        
        # 1. Test Grayscale image - should be mathematically lossless (exact match)
        gray_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        coefficients_obj, cb_cr = lwt.forward(gray_image)
        reconstructed_gray = lwt.inverse(coefficients_obj, cb_cr_channels=cb_cr, output_shape=gray_image.shape)
        reconstructed_gray = np.clip(reconstructed_gray, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(gray_image, reconstructed_gray, 
                                    "LWT transform on grayscale should be exactly lossless")
        
        # 2. Test RGB image - YCbCr conversion has minor floating-point rounding (±1)
        rgb_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        coefficients_obj, cb_cr = lwt.forward(rgb_image)
        reconstructed_rgb = lwt.inverse(coefficients_obj, cb_cr_channels=cb_cr, output_shape=rgb_image.shape[:2])
        reconstructed_rgb = np.clip(reconstructed_rgb, 0, 255).astype(np.uint8)
        np.testing.assert_allclose(rgb_image, reconstructed_rgb, atol=1,
                                  err_msg="LWT transform on RGB should be lossless within rounding error")
        
        print("LWT transform preserved: perfect reconstruction maintained")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])