"""
Bug Condition Exploration Test for CNN Mask Fallback Fix

This test demonstrates the bug: CNN generates empty masks causing embedding to crash.

**CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
**DO NOT attempt to fix the test or the code when it fails.**

The test encodes the expected behavior (fallback mask generation) and will validate
the fix when it passes after implementation.

**Validates: Requirements 2.1, 2.2**

Bug Condition:
- CNN module generates masks with <5% non-zero pixels (often 0%)
- Embedding engine crashes with "Image too small for embedding. Usable pixels: 0"
- Priority queue is empty when mask is all zeros

Expected Behavior (after fix):
- System detects invalid masks (<5% non-zero pixels)
- System generates Sobel gradient-based fallback mask
- Fallback mask has ≥20% non-zero pixels
- Embedding succeeds without crashes
"""

import numpy as np
import pytest
from src.cnn_module import CNNModule
from src.embedding_engine import EmbeddingEngine
from src.lwt_transform import LWTTransform
from src.encryption_service import EncryptionService


class TestCNNMaskBugCondition:
    """Test that demonstrates the bug: CNN generates empty masks causing crashes."""
    
    def test_cnn_generates_empty_masks(self):
        """
        Test that untrained CNN generates masks with <5% non-zero pixels.
        
        **EXPECTED ON UNFIXED CODE**: This assertion will FAIL because CNN generates
        empty masks (0% non-zero pixels), not the expected ≥20% from fallback.
        
        **EXPECTED AFTER FIX**: This assertion will PASS because fallback mechanism
        generates Sobel masks with ≥20% non-zero pixels.
        
        **Validates: Requirements 2.1, 2.2**
        """
        # Use pretrained CNN but temporarily modify threshold to force fallback
        cnn = CNNModule()
        # Temporarily change threshold to a high value to force empty masks
        original_threshold = cnn.THRESHOLD
        cnn.THRESHOLD = 1.1  # Very high threshold will make most masks empty
        
        try:
            # Test with multiple images to find counterexamples
            test_images = [
                np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8),
                np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
                np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8),
            ]
            
            for idx, image in enumerate(test_images):
                mask = cnn.generate_mask(image)
                
                # Calculate non-zero percentage
                non_zero_count = np.count_nonzero(mask)
                total_pixels = mask.size
                non_zero_percentage = (non_zero_count / total_pixels) * 100
                
                # Document the counterexample
                print(f"\nCounterexample {idx + 1}:")
                print(f"  Image shape: {image.shape}")
                print(f"  Mask shape: {mask.shape}")
                print(f"  Non-zero pixels: {non_zero_count}/{total_pixels}")
                print(f"  Non-zero percentage: {non_zero_percentage:.2f}%")
                
                # EXPECTED BEHAVIOR: After fix, fallback should provide ≥20% non-zero pixels
                # UNFIXED CODE: This will FAIL because CNN generates empty masks (0%)
                assert non_zero_percentage >= 20.0, (
                    f"Expected fallback mask with ≥20% non-zero pixels, "
                    f"got {non_zero_percentage:.2f}%. "
                    f"This indicates CNN generated an invalid mask and fallback was not triggered."
                )
        finally:
            # Restore original threshold
            cnn.THRESHOLD = original_threshold
    
    def test_embedding_with_empty_mask_crashes(self):
        """
        Test that embedding with empty CNN mask raises ValueError.
        
        **EXPECTED ON UNFIXED CODE**: This test will FAIL because embedding crashes
        with "Image too small for embedding. Usable pixels: 0".
        
        **EXPECTED AFTER FIX**: This test will PASS because fallback mask enables
        successful embedding.
        
        **Validates: Requirements 2.1, 2.4**
        """
        # Create test image
        cover_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # Generate CNN mask with high threshold to force empty masks
        cnn = CNNModule()
        original_threshold = cnn.THRESHOLD
        cnn.THRESHOLD = 1.1  # Force empty masks
        
        try:
            mask = cnn.generate_mask(cover_image)
            
            # Apply LWT transform
            lwt = LWTTransform()
            coefficients_obj, _ = lwt.forward(cover_image)
            coefficients_dict = coefficients_obj.to_dict()
            
            # Create test message
            encryption_service = EncryptionService()
            message = b"Test message for bug exploration"
            encrypted_message = encryption_service.encrypt(message, "test123456789012")
            
            # Attempt embedding
            embedding_engine = EmbeddingEngine()
            
            # EXPECTED BEHAVIOR: After fix, embedding should succeed
            # UNFIXED CODE: This will FAIL with ValueError about insufficient capacity
            try:
                modified_coeffs = embedding_engine.embed(
                    coefficients_dict,
                    encrypted_message,
                    mask,
                    robustness_level="none"
                )
                
                # If we reach here, embedding succeeded (expected after fix)
                assert modified_coeffs is not None
                print("\nEmbedding succeeded with fallback mask")
                
            except ValueError as e:
                error_msg = str(e)
                print(f"\nCounterexample - Embedding crashed:")
                print(f"  Error: {error_msg}")
                print(f"  Mask non-zero pixels: {np.count_nonzero(mask)}")
                print(f"  Mask shape: {mask.shape}")
                
                # Document the exact error for bug confirmation
                pytest.fail(
                    f"Embedding failed with empty mask. "
                    f"Error: {error_msg}. "
                    f"This confirms the bug exists. "
                    f"After fix, fallback mask should enable successful embedding."
                )
        finally:
            cnn.THRESHOLD = original_threshold
    
    def test_priority_queue_empty_with_zero_mask(self):
        """
        Test that priority queue is empty when mask is all zeros.
        
        **EXPECTED ON UNFIXED CODE**: This test will FAIL because CNN generates
        empty masks, resulting in empty priority queue.
        
        **EXPECTED AFTER FIX**: This test will PASS because fallback mask provides
        valid embedding locations.
        
        **Validates: Requirements 2.1, 2.2**
        """
        # Create test image
        cover_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # Generate CNN mask with high threshold to force empty masks
        cnn = CNNModule()
        original_threshold = cnn.THRESHOLD
        cnn.THRESHOLD = 1.1  # Force empty masks
        
        try:
            mask = cnn.generate_mask(cover_image)
            
            # Apply LWT transform
            lwt = LWTTransform()
            coefficients_obj, _ = lwt.forward(cover_image)
            coefficients_dict = coefficients_obj.to_dict()
            
            # Create priority queue
            embedding_engine = EmbeddingEngine()
            priority_queue = embedding_engine._create_priority_queue(coefficients_dict, mask)
            
            # Document the counterexample
            print(f"\nPriority queue analysis:")
            print(f"  Mask non-zero pixels: {np.count_nonzero(mask)}")
            print(f"  Priority queue size: {len(priority_queue)}")
            
            # EXPECTED BEHAVIOR: After fix, priority queue should have many locations
            # UNFIXED CODE: This will FAIL because queue is empty (mask is all zeros)
            assert len(priority_queue) > 0, (
                f"Priority queue is empty because mask has no valid locations. "
                f"This confirms the bug: CNN generated an empty mask. "
                f"After fix, fallback mask should provide valid embedding locations."
            )
            
            # After fix, we expect substantial capacity (at least 20% of mask pixels)
            min_expected_locations = int(mask.size * 0.20)
            assert len(priority_queue) >= min_expected_locations, (
                f"Expected at least {min_expected_locations} embedding locations "
                f"(20% of {mask.size} mask pixels), got {len(priority_queue)}. "
                f"Fallback mask should provide sufficient capacity."
            )
        finally:
            cnn.THRESHOLD = original_threshold
    
    def test_multiple_images_produce_invalid_masks(self):
        """
        Test that multiple different images produce invalid CNN masks.
        
        This test documents the scope of the bug: it's not just one specific image,
        but a systematic issue with the untrained CNN model.
        
        **EXPECTED ON UNFIXED CODE**: Most/all test images will produce invalid masks.
        **EXPECTED AFTER FIX**: All images will have valid masks via fallback.
        
        **Validates: Requirements 2.1, 2.2**
        """
        cnn = CNNModule()
        # Use high threshold to force invalid masks
        original_threshold = cnn.THRESHOLD
        cnn.THRESHOLD = 1.1  # Force invalid masks
        
        try:
            # Test various image sizes and content
            test_cases = [
                ("512x512 random", np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)),
                ("1080p random", np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)),
                ("800x600 random", np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)),
                ("4K random", np.random.randint(0, 256, (2160, 3840, 3), dtype=np.uint8)),
                ("64x64 minimum", np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)),
            ]
            
            invalid_mask_count = 0
            results = []
            
            for name, image in test_cases:
                mask = cnn.generate_mask(image)
                non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
                
                is_invalid = non_zero_percentage < 5.0
                if is_invalid:
                    invalid_mask_count += 1
                
                result = {
                    'name': name,
                    'image_shape': image.shape,
                    'mask_shape': mask.shape,
                    'non_zero_percentage': non_zero_percentage,
                    'is_invalid': is_invalid
                }
                results.append(result)
                
                print(f"\n{name}:")
                print(f"  Image: {image.shape} -> Mask: {mask.shape}")
                print(f"  Non-zero: {non_zero_percentage:.2f}%")
                print(f"  Status: {'INVALID (<5%)' if is_invalid else 'VALID (>=5%)'}")
            
            # Document findings
            print(f"\n\nBug Scope Summary:")
            print(f"  Total test cases: {len(test_cases)}")
            print(f"  Invalid masks: {invalid_mask_count}")
            print(f"  Valid masks: {len(test_cases) - invalid_mask_count}")
            
            # EXPECTED BEHAVIOR: After fix, all masks should be valid (≥20% via fallback)
            # UNFIXED CODE: Most/all masks will be invalid
            for result in results:
                assert result['non_zero_percentage'] >= 20.0, (
                    f"{result['name']}: Expected fallback mask with >=20% non-zero pixels, "
                    f"got {result['non_zero_percentage']:.2f}%. "
                    f"Fallback mechanism should ensure all masks are valid."
                )
        finally:
            cnn.THRESHOLD = original_threshold


class TestBugConditionFunction:
    """Test the bug condition detection logic."""
    
    def test_is_bug_condition_empty_mask(self):
        """Test that all-zero mask is detected as bug condition."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        is_bug_condition = non_zero_percentage < 5.0
        
        assert is_bug_condition, "All-zero mask should trigger bug condition"
        assert non_zero_percentage == 0.0
    
    def test_is_bug_condition_sparse_mask(self):
        """Test that sparse mask (<5% non-zero) is detected as bug condition."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Set 2% of pixels to 1
        num_ones = int(mask.size * 0.02)
        mask.flat[:num_ones] = 1
        np.random.shuffle(mask.flat)
        
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        is_bug_condition = non_zero_percentage < 5.0
        
        assert is_bug_condition, "Sparse mask (<5%) should trigger bug condition"
        assert 1.5 < non_zero_percentage < 2.5  # Approximately 2%
    
    def test_is_not_bug_condition_valid_mask(self):
        """Test that valid mask (≥5% non-zero) is NOT a bug condition."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Set 30% of pixels to 1 (typical for Sobel fallback)
        num_ones = int(mask.size * 0.30)
        mask.flat[:num_ones] = 1
        np.random.shuffle(mask.flat)
        
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        is_bug_condition = non_zero_percentage < 5.0
        
        assert not is_bug_condition, "Valid mask (≥5%) should NOT trigger bug condition"
        assert non_zero_percentage >= 20.0  # Should be around 30%
    
    def test_threshold_boundary_5_percent(self):
        """Test the 5% threshold boundary."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Test just below threshold (4.9%)
        num_ones = int(mask.size * 0.049)
        mask.flat[:num_ones] = 1
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        assert non_zero_percentage < 5.0, "4.9% should be below threshold"
        
        # Test at threshold (5.0%)
        mask = np.zeros((256, 256), dtype=np.uint8)
        num_ones = int(np.ceil(mask.size * 0.05))
        mask.flat[:num_ones] = 1
        non_zero_percentage = (np.count_nonzero(mask) / mask.size) * 100
        # At exactly 5%, should NOT trigger bug condition (≥5% is valid)
        assert non_zero_percentage >= 5.0, "5.0% should be at/above threshold"


if __name__ == "__main__":
    # Run tests with verbose output to see counterexamples
    pytest.main([__file__, "-v", "-s"])
