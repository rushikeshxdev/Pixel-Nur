"""
Bug Condition Exploration Test for UTF-8 Decode Extraction Fix

**Validates: Requirements 2.1, 2.2, 2.3**

This test suite verifies the bug condition: when decryption produces invalid UTF-8
bytes in extract_interface(), the system shows a raw technical error message instead
of providing a user-friendly error message with actionable guidance.

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

The test encodes the expected behavior - it will validate the fix when it passes
after implementation.

GOAL: Surface counterexamples that demonstrate the bug exists (raw codec errors shown to users).

Property 1: Fault Condition - UTF-8 Decode Error Handling
- When decryption produces invalid UTF-8 bytes, the system should catch the
  UnicodeDecodeError and return a user-friendly error message explaining possible
  causes (wrong password, modified image, no embedded data) with actionable guidance,
  NOT the raw codec error message.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple


# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Save original modules to avoid poisoning other tests
modules_to_mock = ['torch', 'cv2', 'gradio', 'src.pixelnur', 'src.extraction_engine', 'src.lwt_transform', 'src.encryption_service']
original_modules = {name: sys.modules[name] for name in modules_to_mock if name in sys.modules}

# Mock PyTorch and dependent modules before importing
sys.modules['torch'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['gradio'] = MagicMock()

# Mock the Phase 2 modules
mock_pixelnur = MagicMock()
mock_extraction_engine = MagicMock()
mock_lwt_transform = MagicMock()
mock_encryption_service = MagicMock()

# Create exception classes that can be imported
class MockExtractionError(Exception):
    pass

mock_extraction_engine.ExtractionError = MockExtractionError

sys.modules['src.pixelnur'] = mock_pixelnur
sys.modules['src.extraction_engine'] = mock_extraction_engine
sys.modules['src.lwt_transform'] = mock_lwt_transform
sys.modules['src.encryption_service'] = mock_encryption_service

# Now we need to import from pixelnur/app.py
pixelnur_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pixelnur'))
sys.path.insert(0, pixelnur_app_path)

try:
    import app as pixelnur_app
finally:
    # Restore original modules to avoid poisoning the global namespace
    for name in modules_to_mock:
        if name in original_modules:
            sys.modules[name] = original_modules[name]
        elif name in sys.modules:
            del sys.modules[name]


class TestUTF8DecodeBugCondition:
    """
    Bug Condition Exploration Tests
    
    These tests verify that invalid UTF-8 bytes from decryption currently show
    raw technical error messages instead of user-friendly guidance.
    
    EXPECTED OUTCOME: Tests FAIL on unfixed code (this confirms the bug exists)
    """
    
    @patch('app.encryption_service')
    @patch('app.extraction_engine')
    @patch('app.cv2')
    def test_extract_interface_invalid_utf8_byte_sequence_1(
        self, mock_cv2, mock_extraction_engine_instance, mock_encryption_service_instance
    ):
        """
        Test that when decryption produces invalid UTF-8 bytes (0x9b 0xf2 0xa3 0xc4),
        the system provides a user-friendly error message with actionable guidance,
        NOT the raw codec error.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        EXPECTED ON UNFIXED CODE: FAIL - shows raw codec error message
        EXPECTED ON FIXED CODE: PASS - shows user-friendly error with guidance
        """
        # Arrange: Create valid inputs
        stego_file = "test_stego.png"
        password = "ValidPassword1234"
        
        # Mock cv2.imread to return a valid image
        mock_stego_image = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_stego_image
        
        # Mock extraction_engine.extract to return encrypted message
        mock_extraction_engine_instance.extract.return_value = b"encrypted_data"
        
        # Mock encryption_service.decrypt to return INVALID UTF-8 bytes
        # This simulates wrong password producing garbage bytes
        invalid_utf8_bytes = b'\x9b\xf2\xa3\xc4'
        mock_encryption_service_instance.decrypt.return_value = invalid_utf8_bytes
        
        # Act: Call extract_interface
        result = pixelnur_app.extract_interface(stego_file, password)
        
        # Assert: The system should provide user-friendly error with actionable guidance
        assert result.startswith("❌"), \
            f"Expected error to start with '❌', got: {result}"
        
        # Should provide actionable guidance like the ExtractionError handler does
        assert "Please verify" in result or "verify" in result.lower(), \
            f"Expected error message to provide 'Please verify' guidance, got: {result}"
        assert "password" in result.lower() or "Password" in result, \
            f"Expected error message to mention password verification, got: {result}"
        
        # Should NOT show raw codec error message to user
        assert "codec can't decode" not in result, \
            f"Error message should not expose raw codec error to user. Got: {result}"
        assert "0x9b" not in result, \
            f"Error message should not show raw byte values to user. Got: {result}"
    
    @patch('app.encryption_service')
    @patch('app.extraction_engine')
    @patch('app.cv2')
    def test_extract_interface_invalid_utf8_byte_sequence_2(
        self, mock_cv2, mock_extraction_engine_instance, mock_encryption_service_instance
    ):
        """
        Test that when decryption produces invalid UTF-8 bytes (0xff 0xfe 0xfd 0xfc),
        the system provides a user-friendly error message with actionable guidance.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        This tests a different invalid byte sequence to ensure robustness.
        
        EXPECTED ON UNFIXED CODE: FAIL - shows raw codec error message
        EXPECTED ON FIXED CODE: PASS - shows user-friendly error with guidance
        """
        # Arrange
        stego_file = "test_stego.png"
        password = "WrongPassword123"
        
        mock_stego_image = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_stego_image
        
        mock_extraction_engine_instance.extract.return_value = b"encrypted_data"
        
        # Different invalid UTF-8 byte sequence
        invalid_utf8_bytes = b'\xff\xfe\xfd\xfc'
        mock_encryption_service_instance.decrypt.return_value = invalid_utf8_bytes
        
        # Act
        result = pixelnur_app.extract_interface(stego_file, password)
        
        # Assert
        assert result.startswith("❌"), \
            f"Expected error to start with '❌', got: {result}"
        
        assert "Please verify" in result or "verify" in result.lower(), \
            f"Expected error message to provide 'Please verify' guidance, got: {result}"
        assert "password" in result.lower() or "Password" in result, \
            f"Expected error message to mention password verification, got: {result}"
        
        assert "codec can't decode" not in result, \
            f"Error message should not expose raw codec error to user. Got: {result}"
        assert "0xff" not in result, \
            f"Error message should not show raw byte values to user. Got: {result}"
    
    @patch('app.encryption_service')
    @patch('app.extraction_engine')
    @patch('app.cv2')
    def test_extract_interface_partial_invalid_utf8(
        self, mock_cv2, mock_extraction_engine_instance, mock_encryption_service_instance
    ):
        """
        Test that when decryption produces bytes that start valid but end invalid
        (Hello + 0x9b 0xf2), the system provides a user-friendly error message.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        This tests partial corruption scenario.
        
        EXPECTED ON UNFIXED CODE: FAIL - shows raw codec error message
        EXPECTED ON FIXED CODE: PASS - shows user-friendly error with guidance
        """
        # Arrange
        stego_file = "test_stego.png"
        password = "PartiallyWrong123"  # 17 characters, meets minimum requirement
        
        mock_stego_image = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_stego_image
        
        mock_extraction_engine_instance.extract.return_value = b"encrypted_data"
        
        # Partially valid UTF-8: starts with "Hello" then invalid bytes
        invalid_utf8_bytes = b'Hello\x9b\xf2'
        mock_encryption_service_instance.decrypt.return_value = invalid_utf8_bytes
        
        # Act
        result = pixelnur_app.extract_interface(stego_file, password)
        
        # Assert
        assert result.startswith("❌"), \
            f"Expected error to start with '❌', got: {result}"
        
        assert "Please verify" in result or "verify" in result.lower(), \
            f"Expected error message to provide 'Please verify' guidance, got: {result}"
        assert "password" in result.lower() or "Password" in result, \
            f"Expected error message to mention password verification, got: {result}"
        
        assert "codec can't decode" not in result, \
            f"Error message should not expose raw codec error to user. Got: {result}"
    
    @patch('app.encryption_service')
    @patch('app.extraction_engine')
    @patch('app.cv2')
    def test_extract_interface_invalid_utf8_provides_actionable_guidance(
        self, mock_cv2, mock_extraction_engine_instance, mock_encryption_service_instance
    ):
        """
        Test that the error message provides actionable guidance to users when
        UTF-8 decode fails, similar to the ExtractionError handler.
        
        **Validates: Requirements 2.3**
        
        The error message should guide users to:
        - Verify their password
        - Check image integrity (hasn't been modified)
        - Confirm the image contains embedded data
        
        EXPECTED ON UNFIXED CODE: FAIL - shows raw codec error without guidance
        EXPECTED ON FIXED CODE: PASS - provides actionable guidance with bullet points
        """
        # Arrange
        stego_file = "test_stego.png"
        password = "TestPassword12345"  # 17 characters, meets minimum requirement
        
        mock_stego_image = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_stego_image
        
        mock_extraction_engine_instance.extract.return_value = b"encrypted_data"
        
        invalid_utf8_bytes = b'\x9b\xf2\xa3\xc4'
        mock_encryption_service_instance.decrypt.return_value = invalid_utf8_bytes
        
        # Act
        result = pixelnur_app.extract_interface(stego_file, password)
        
        # Assert: Check for actionable guidance similar to ExtractionError handler
        assert result.startswith("❌"), \
            f"Expected error to start with '❌', got: {result}"
        
        # Should have "Please verify:" section like ExtractionError handler
        assert "Please verify" in result, \
            f"Expected 'Please verify' section in error message, got: {result}"
        
        # Should mention all three verification points
        assert "password" in result.lower() or "Password" in result, \
            f"Expected error message to mention password verification, got: {result}"
        assert "modified" in result.lower() or "integrity" in result.lower(), \
            f"Expected error message to mention image modification check, got: {result}"
        assert "embedded data" in result.lower() or "contains" in result.lower(), \
            f"Expected error message to mention embedded data verification, got: {result}"
        
        # Should use bullet points for verification steps (like ExtractionError handler)
        assert "•" in result or "*" in result or "-" in result, \
            f"Expected bullet points for verification steps, got: {result}"
        
        # Should NOT show raw codec error or technical details
        assert "UnicodeDecodeError" not in result, \
            f"Error message should not expose exception type to user. Got: {result}"
        assert "codec can't decode" not in result, \
            f"Error message should not expose raw codec error to user. Got: {result}"


if __name__ == "__main__":
    # Run tests manually
    import traceback
    
    test_class = TestUTF8DecodeBugCondition()
    test_methods = [
        'test_extract_interface_invalid_utf8_byte_sequence_1',
        'test_extract_interface_invalid_utf8_byte_sequence_2',
        'test_extract_interface_partial_invalid_utf8',
        'test_extract_interface_invalid_utf8_provides_actionable_guidance'
    ]
    
    total_tests = len(test_methods)
    passed_tests = 0
    failed_tests = 0
    counterexamples = []
    
    print("\n" + "="*70)
    print("Bug Condition Exploration Test - UTF-8 Decode Extraction Fix")
    print("="*70)
    print("\nCRITICAL: These tests MUST FAIL on unfixed code")
    print("Failure confirms the bug exists (raw codec errors shown to users)")
    print("="*70 + "\n")
    
    for test_method_name in test_methods:
        test_method = getattr(test_class, test_method_name)
        
        try:
            test_method()
            print(f"  ✓ {test_method_name}")
            passed_tests += 1
        except AssertionError as e:
            print(f"  ✗ {test_method_name}")
            print(f"    AssertionError: {str(e)}")
            counterexamples.append(f"{test_method_name}: {str(e)}")
            failed_tests += 1
        except UnicodeDecodeError as e:
            print(f"  ✗ {test_method_name}")
            print(f"    UnicodeDecodeError: {str(e)}")
            counterexamples.append(f"{test_method_name}: UnicodeDecodeError - {str(e)}")
            failed_tests += 1
        except Exception as e:
            print(f"  ✗ {test_method_name}")
            print(f"    {type(e).__name__}: {str(e)}")
            counterexamples.append(f"{test_method_name}: {type(e).__name__} - {str(e)}")
            failed_tests += 1
    
    print("\n" + "="*70)
    print(f"Test Results: {passed_tests}/{total_tests} passed, {failed_tests} failed")
    print("="*70 + "\n")
    
    if failed_tests > 0:
        print("✅ EXPECTED OUTCOME: Tests failed on unfixed code")
        print("   This confirms the bug exists - raw codec errors shown instead of user-friendly guidance")
        print("\nCounterexamples found:")
        for i, example in enumerate(counterexamples, 1):
            print(f"  {i}. {example}")
        print("\nRoot cause: UnicodeDecodeError is caught by generic Exception handler")
        print("            which displays raw error message instead of user-friendly guidance")
        print("            Need specific handler for UnicodeDecodeError before generic handler")
        sys.exit(0)  # Exit with success - failure is expected
    else:
        print("⚠️  UNEXPECTED: All tests passed on unfixed code")
        print("   This suggests the bug may not exist or tests need adjustment")
        sys.exit(1)
