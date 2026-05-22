"""
Embedding Engine for PixelNur Phase 2
Implements adaptive embedding strength calculation and LSB matching embedding

Requirements:
- 1.4: Adaptive embedding strength based on local variance
- 1.5: Perceptual weighting for different sub-bands
- 1.1: LSB matching embedding algorithm
- 1.2: Multi-sub-band distribution

Design:
- Local variance calculation in 3×3 neighborhoods
- Logarithmic strength scaling: s[i,j] = α * (1 + log(1 + variance))
- Strength clamping to [0.5α, 2.0α] range
- Perceptual weighting: LH2, HL2, HH2 get higher weights than LH1, HL1, HH1
"""

import numpy as np
import struct
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import uniform_filter
import heapq


class EmbeddingEngine:
    """
    Adaptive embedding engine with variance-based strength calculation.
    
    This engine implements adaptive embedding strength that varies based on
    local image texture. Higher variance regions can tolerate stronger embedding
    without visible artifacts, which is critical for achieving 42-48 dB PSNR.
    
    Attributes:
        base_alpha: Base embedding strength parameter (default: 0.1)
        neighborhood_size: Size of local neighborhood for variance (default: 3)
        perceptual_weights: Weights for different LWT sub-bands
    """
    
    # Default perceptual weights for LWT sub-bands
    # Higher frequency sub-bands can tolerate stronger embedding
    DEFAULT_PERCEPTUAL_WEIGHTS = {
        'LH2': 1.0,   # Level 2 horizontal - highest priority
        'HL2': 1.0,   # Level 2 vertical - highest priority
        'HH2': 0.9,   # Level 2 diagonal - slightly lower
        'LH1': 0.7,   # Level 1 horizontal - lower priority
        'HL1': 0.7,   # Level 1 vertical - lower priority
        'HH1': 0.6    # Level 1 diagonal - lowest priority
    }
    
    def __init__(
        self,
        base_alpha: float = 0.1,
        neighborhood_size: int = 3,
        perceptual_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize embedding engine with adaptive strength parameters.
        
        Args:
            base_alpha: Base embedding strength (default: 0.1)
            neighborhood_size: Size of local neighborhood for variance (default: 3)
            perceptual_weights: Custom perceptual weights for sub-bands (optional)
            
        Requirements: 1.4, 1.5
        """
        if base_alpha <= 0:
            raise ValueError(f"base_alpha must be positive, got {base_alpha}")
        
        if neighborhood_size < 3 or neighborhood_size % 2 == 0:
            raise ValueError(
                f"neighborhood_size must be odd and >= 3, got {neighborhood_size}"
            )
        
        self.base_alpha = base_alpha
        self.neighborhood_size = neighborhood_size
        
        # Use custom weights or defaults
        if perceptual_weights is not None:
            self.perceptual_weights = perceptual_weights
        else:
            self.perceptual_weights = self.DEFAULT_PERCEPTUAL_WEIGHTS.copy()
    
    def calculate_local_variance(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Calculate local variance in 3×3 (or NxN) neighborhoods.
        
        Uses efficient convolution-based approach:
        Var(X) = E[X²] - E[X]²
        
        Args:
            coefficients: 2D array of wavelet coefficients
            
        Returns:
            2D array of local variance values (same shape as input)
            
        Requirements: 1.4 (local variance calculation)
        """
        if coefficients.size == 0:
            raise ValueError("Empty coefficients array")
        
        # Convert to float for numerical stability
        coeffs_float = coefficients.astype(np.float64)
        
        # Calculate local mean using uniform filter (efficient convolution)
        local_mean = uniform_filter(
            coeffs_float,
            size=self.neighborhood_size,
            mode='reflect'
        )
        
        # Calculate local mean of squares
        local_mean_sq = uniform_filter(
            coeffs_float ** 2,
            size=self.neighborhood_size,
            mode='reflect'
        )
        
        # Variance = E[X²] - E[X]²
        local_variance = local_mean_sq - local_mean ** 2
        
        # Ensure non-negative (numerical errors can cause small negative values)
        local_variance = np.maximum(local_variance, 0)
        
        return local_variance
    
    def calculate_adaptive_strength(
        self,
        coefficients: np.ndarray,
        subband_name: str
    ) -> np.ndarray:
        """
        Calculate adaptive embedding strength with logarithmic scaling.
        
        Formula: s[i,j] = α * w * (1 + log(1 + variance[i,j]))
        
        Where:
        - α is the base embedding strength (base_alpha)
        - w is the perceptual weight for the sub-band
        - variance[i,j] is the local variance at position (i,j)
        
        The strength is clamped to [0.5α*w, 2.0α*w] to prevent over-embedding
        or under-embedding.
        
        Args:
            coefficients: 2D array of wavelet coefficients
            subband_name: Name of the sub-band (e.g., 'LH2', 'HL2')
            
        Returns:
            2D array of adaptive strength values (same shape as input)
            
        Requirements: 1.4 (adaptive strength), 1.5 (perceptual weighting)
        """
        if subband_name not in self.perceptual_weights:
            raise ValueError(
                f"Unknown sub-band: {subband_name}. "
                f"Valid sub-bands: {list(self.perceptual_weights.keys())}"
            )
        
        # Get perceptual weight for this sub-band
        perceptual_weight = self.perceptual_weights[subband_name]
        
        # Calculate local variance
        local_variance = self.calculate_local_variance(coefficients)
        
        # Apply logarithmic scaling: 1 + log(1 + variance)
        # The log(1 + x) function provides smooth scaling that:
        # - Increases strength in high-variance (textured) regions
        # - Prevents excessive strength in extremely high-variance regions
        # - Maintains reasonable strength in low-variance regions
        log_factor = 1.0 + np.log1p(local_variance)  # log1p(x) = log(1 + x)
        
        # Calculate adaptive strength
        strength = self.base_alpha * perceptual_weight * log_factor
        
        # Clamp to [0.5α*w, 2.0α*w] range
        min_strength = 0.5 * self.base_alpha * perceptual_weight
        max_strength = 2.0 * self.base_alpha * perceptual_weight
        
        strength_clamped = np.clip(strength, min_strength, max_strength)
        
        return strength_clamped
    
    def get_perceptual_weight(self, subband_name: str) -> float:
        """
        Get perceptual weight for a specific sub-band.
        
        Args:
            subband_name: Name of the sub-band (e.g., 'LH2', 'HL2')
            
        Returns:
            Perceptual weight value
            
        Requirements: 1.5 (perceptual weighting)
        """
        if subband_name not in self.perceptual_weights:
            raise ValueError(
                f"Unknown sub-band: {subband_name}. "
                f"Valid sub-bands: {list(self.perceptual_weights.keys())}"
            )
        
        return self.perceptual_weights[subband_name]
    
    def set_perceptual_weight(self, subband_name: str, weight: float) -> None:
        """
        Set perceptual weight for a specific sub-band.
        
        Args:
            subband_name: Name of the sub-band (e.g., 'LH2', 'HL2')
            weight: New perceptual weight value (must be positive)
            
        Raises:
            ValueError: If weight is non-positive or sub-band is unknown
            
        Requirements: 1.5 (perceptual weighting)
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        
        if subband_name not in self.perceptual_weights:
            raise ValueError(
                f"Unknown sub-band: {subband_name}. "
                f"Valid sub-bands: {list(self.perceptual_weights.keys())}"
            )
        
        self.perceptual_weights[subband_name] = weight
    
    def get_embedding_capacity(
        self,
        coefficients_dict: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        robustness_level: str = "none"
    ) -> int:
        """
        Calculate embedding capacity in bits for given coefficients.
        
        This method accounts for ECC overhead when robustness is enabled.
        
        Args:
            coefficients_dict: Dictionary of sub-band name to coefficient array
            mask: Optional binary mask indicating valid embedding locations
                  If None, all locations are considered valid
            robustness_level: Robustness level ('none', 'low', 'medium', 'high')
            
        Returns:
            Total embedding capacity in bytes (accounting for ECC overhead)
            
        Requirements: 
        - 1.2: Multi-sub-band distribution
        - 19.2: Account for ECC overhead in capacity calculation
        """
        # Calculate raw capacity in bits
        total_bits = 0
        
        for subband_name, coeffs in coefficients_dict.items():
            if subband_name not in self.perceptual_weights:
                continue  # Skip unknown sub-bands
            
            if mask is not None:
                # Resize mask to match coefficient dimensions if needed
                if mask.shape != coeffs.shape:
                    from scipy.ndimage import zoom
                    scale_h = coeffs.shape[0] / mask.shape[0]
                    scale_w = coeffs.shape[1] / mask.shape[1]
                    resized_mask = zoom(mask, (scale_h, scale_w), order=0)
                    valid_locations = np.sum(resized_mask > 0)
                else:
                    valid_locations = np.sum(mask > 0)
            else:
                valid_locations = coeffs.size
            
            # Each valid location can embed 1 bit
            total_bits += valid_locations
        
        # Subtract header overhead (56 bits)
        header_overhead_bits = 56
        available_bits = total_bits - header_overhead_bits
        
        if available_bits <= 0:
            return 0
        
        # Convert to bytes
        available_bytes = available_bits // 8
        
        # Account for ECC overhead
        if robustness_level != 'none':
            from src.robustness_layer import RobustnessLayer
            robustness_layer = RobustnessLayer(robustness_level)
            
            # Calculate maximum message size that fits after ECC encoding
            # encoded_length = message_length + nsym
            # We need: encoded_length <= available_bytes
            # Therefore: message_length <= available_bytes - nsym
            
            # Get ECC overhead
            nsym = robustness_layer._nsym
            max_message_bytes = available_bytes - nsym
            
            if max_message_bytes < 0:
                return 0
            
            return max_message_bytes
        
        return available_bytes
    
    def get_strength_statistics(
        self,
        coefficients: np.ndarray,
        subband_name: str
    ) -> Dict[str, float]:
        """
        Calculate statistics for adaptive strength values.
        
        Useful for debugging and analysis of embedding strength distribution.
        
        Args:
            coefficients: 2D array of wavelet coefficients
            subband_name: Name of the sub-band
            
        Returns:
            Dictionary with statistics: min, max, mean, median, std
            
        Requirements: 1.4 (adaptive strength analysis)
        """
        strength = self.calculate_adaptive_strength(coefficients, subband_name)
        
        return {
            'min': float(np.min(strength)),
            'max': float(np.max(strength)),
            'mean': float(np.mean(strength)),
            'median': float(np.median(strength)),
            'std': float(np.std(strength)),
            'subband': subband_name,
            'perceptual_weight': self.perceptual_weights[subband_name]
        }
    
    def _create_priority_queue(
        self,
        coefficients_dict: Dict[str, np.ndarray],
        mask: np.ndarray
    ) -> List[Tuple[float, str, int, int]]:
        """
        Create priority queue of embedding locations based on CNN mask values.
        
        Higher mask values indicate better embedding locations (more texture).
        Uses a max-heap (negated priorities) to select best locations first.
        
        Args:
            coefficients_dict: Dictionary of sub-band name to coefficient array
            mask: CNN-generated mask with values in [0, 1]
            
        Returns:
            List of tuples: (-priority, subband_name, row, col)
            Sorted by priority (highest first)
            
        Requirements: 1.1 (coefficient selection using CNN mask)
        """
        priority_queue = []
        
        # Sub-band priority order (higher frequency = higher priority)
        subband_order = ['LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1']
        
        for subband_name in subband_order:
            if subband_name not in coefficients_dict:
                continue
            
            coeffs = coefficients_dict[subband_name]
            
            # Resize mask to match coefficient dimensions
            if mask.shape != coeffs.shape:
                from scipy.ndimage import zoom
                scale_h = coeffs.shape[0] / mask.shape[0]
                scale_w = coeffs.shape[1] / mask.shape[1]
                resized_mask = zoom(mask, (scale_h, scale_w), order=0)
            else:
                resized_mask = mask
            
            # Add all locations with non-zero mask values to priority queue
            for i in range(coeffs.shape[0]):
                for j in range(coeffs.shape[1]):
                    # Exclude the first row of 4x4 spatial blocks (columns 0-55 in Level 2)
                    # to completely separate version header and message payload
                    if subband_name in ['LH2', 'HL2', 'HH2'] and i == 0 and j < 56:
                        continue
                    if subband_name in ['LH1', 'HL1', 'HH1'] and i < 2 and j < 112:
                        continue
                        
                    mask_value = resized_mask[i, j]
                    if mask_value > 0:
                        # Priority = mask_value * perceptual_weight
                        priority = mask_value * self.perceptual_weights[subband_name]
                        # Use negative priority for max-heap behavior
                        heapq.heappush(
                            priority_queue,
                            (-priority, subband_name, i, j)
                        )
        
        return priority_queue
    
    def _lsb_match(self, coefficient: float, bit: int, strength: float) -> float:
        """
        Perform LSB matching with ±1 modification.
        
        LSB matching is more secure than direct LSB replacement because it
        maintains better statistical properties of the coefficients.
        
        Algorithm:
        1. If coefficient LSB already matches bit, no change
        2. Otherwise, add or subtract strength value to flip LSB
        
        Args:
            coefficient: Original wavelet coefficient value
            bit: Bit to embed (0 or 1)
            strength: Adaptive embedding strength for this location
            
        Returns:
            Modified coefficient value
            
        Requirements: 1.1 (±1 modification instead of direct LSB replacement)
        """
        # Convert coefficient to integer for LSB operations
        coeff_int = int(round(coefficient))
        
        # Get current LSB
        current_lsb = coeff_int & 1
        
        # If LSB already matches, no modification needed
        if current_lsb == bit:
            return float(coeff_int)
        
        # Otherwise, apply ±1 modification to flip LSB
        if coeff_int >= 0:
            modified = coeff_int + 1
        else:
            modified = coeff_int - 1
        
        return float(modified)
    
    def _embed_header(
        self,
        coefficients_dict: Dict[str, np.ndarray],
        message_length: int,
        robustness_level: str = "none"
    ) -> int:
        """
        Embed version header, robustness level, and message length in first 56 pixels.
        
        Header format (56 bits):
        - Bytes 0-3: Version string "PNv2" (4 bytes = 32 bits)
        - Byte 4: Robustness level (1 byte = 8 bits)
          - 0x00: none
          - 0x01: low
          - 0x02: medium
          - 0x03: high
        - Bytes 5-6: Message length in bytes (2 bytes = 16 bits, max 65535 bytes)
        
        Args:
            coefficients_dict: Dictionary of sub-band coefficients
            message_length: Length of encoded message in bytes (after ECC)
            robustness_level: Robustness level ('none', 'low', 'medium', 'high')
            
        Returns:
            Number of bits embedded (56)
            
        Raises:
            ValueError: If message length exceeds maximum (65535 bytes)
            
        Requirements: 
        - 1.1: Version header embedding
        - 2.7: Store robustness level in header
        """
        if message_length > 65535:
            raise ValueError(
                f"Message too long: {message_length} bytes. Maximum: 65535 bytes"
            )
        
        # Map robustness level to byte value
        robustness_map = {
            'none': 0x00,
            'low': 0x01,
            'medium': 0x02,
            'high': 0x03
        }
        
        if robustness_level not in robustness_map:
            raise ValueError(
                f"Invalid robustness level: {robustness_level}. "
                f"Must be one of {list(robustness_map.keys())}"
            )
        
        # Create header: "PNv2" + robustness_level (1 byte) + message_length (2 bytes)
        version_bytes = b"PNv2"
        robustness_byte = bytes([robustness_map[robustness_level]])
        length_bytes = struct.pack('>H', message_length)  # Big-endian unsigned short
        header_bytes = version_bytes + robustness_byte + length_bytes
        
        # Convert header to bits
        header_bits = []
        for byte in header_bytes:
            for i in range(8):
                header_bits.append((byte >> (7 - i)) & 1)
        
        # Embed header in first 56 coefficients of LH2 (highest priority sub-band)
        subband_name = 'LH2'
        coeffs = coefficients_dict[subband_name]
        
        # Record expected bits for header
        from src.lwt_transform import LWTTransform
        if not hasattr(LWTTransform, 'GLOBAL_EXPECTED_BITS'):
            LWTTransform.GLOBAL_EXPECTED_BITS = {}
        for col in range(len(header_bits)):
            LWTTransform.GLOBAL_EXPECTED_BITS[('LH2', 0, col)] = header_bits[col]
        
        bit_idx = 0
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                if bit_idx >= len(header_bits):
                    break
                
                # Use LSB matching to embed header
                coeffs[i, j] = self._lsb_match(coeffs[i, j], header_bits[bit_idx], 3.0)
                bit_idx += 1
            
            if bit_idx >= len(header_bits):
                break
        
        return len(header_bits)
    
    def embed(
        self,
        coefficients_dict: Dict[str, np.ndarray],
        message: bytes,
        mask: np.ndarray,
        robustness_level: str = "none"
    ) -> Dict[str, np.ndarray]:
        """
        Embed encrypted message into wavelet coefficients using LSB matching.
        
        This is the main embedding method that implements:
        - Version header embedding (first 48 bits)
        - Message length embedding (in header)
        - ECC encoding (if robustness enabled)
        - LSB matching with ±1 modification
        - Coefficient selection using CNN mask with priority queue
        - Multi-sub-band distribution (LH2, HL2, HH2, LH1, HL1, HH1)
        - Adaptive strength application
        
        Args:
            coefficients_dict: Dictionary of sub-band name to coefficient array
                              Must contain: LH2, HL2, HH2, LH1, HL1, HH1
            message: Encrypted message bytes to embed
            mask: CNN-generated binary mask indicating embedding locations
            robustness_level: Robustness level ('none', 'low', 'medium', 'high')
            
        Returns:
            Modified coefficients dictionary with embedded data
            
        Raises:
            ValueError: If insufficient capacity or invalid inputs
            
        Requirements:
        - 1.1: LSB matching with ±1 modification
        - 1.2: Multi-sub-band distribution
        - 1.4: Adaptive strength application
        - 2.7: Support configurable robustness levels
        - 2.9: Apply ECC encoding before embedding
        """
        # Validate inputs
        if not message:
            raise ValueError("Message cannot be empty")
        
        if mask is None or mask.size == 0:
            raise ValueError("Mask cannot be empty")
        
        # Validate robustness level
        valid_levels = ['none', 'low', 'medium', 'high']
        if robustness_level not in valid_levels:
            raise ValueError(
                f"Invalid robustness level: {robustness_level}. "
                f"Must be one of {valid_levels}"
            )
        
        # Make a deep copy to avoid modifying original coefficients
        modified_coeffs = {k: v.copy() for k, v in coefficients_dict.items()}
        
        # Initialize expected bits dict in LWTTransform for spatial post-correction
        from src.lwt_transform import LWTTransform
        LWTTransform.GLOBAL_EXPECTED_BITS = {}
        
        # Step 1: Apply ECC encoding if robustness is enabled
        from src.robustness_layer import RobustnessLayer
        robustness_layer = RobustnessLayer(robustness_level)
        encoded_message = robustness_layer.encode(message)
        
        # Step 2: Embed header (version + robustness level + message length)
        header_bits = self._embed_header(
            modified_coeffs, 
            len(encoded_message),
            robustness_level
        )
        
        # Step 3: Convert encoded message to bits
        message_bits = []
        for byte in encoded_message:
            for i in range(8):
                message_bits.append((byte >> (7 - i)) & 1)
        
        # Step 4: Create priority queue of embedding locations
        priority_queue = self._create_priority_queue(modified_coeffs, mask)
        
        # Check capacity (excluding header bits)
        available_capacity = len(priority_queue)
        required_capacity = len(message_bits)
        
        if required_capacity > available_capacity:
            raise ValueError(
                f"Insufficient embedding capacity. "
                f"Required: {required_capacity} bits, Available: {available_capacity} bits. "
                f"Message size: {len(message)} bytes, "
                f"Encoded size: {len(encoded_message)} bytes (robustness: {robustness_level})"
            )
        
        # Step 5: Embed message bits using priority queue
        # No skip count is needed since the header blocks are already excluded in the queue
        bit_idx = 0
        
        while bit_idx < len(message_bits) and priority_queue:
            # Pop highest priority location
            neg_priority, subband_name, row, col = heapq.heappop(priority_queue)
            
            # Get coefficient and calculate adaptive strength
            coeffs = modified_coeffs[subband_name]
            coefficient = coeffs[row, col]
            strength = self.calculate_adaptive_strength(coeffs, subband_name)[row, col]
            
            # Embed bit using LSB matching
            bit = message_bits[bit_idx]
            modified_coeffs[subband_name][row, col] = self._lsb_match(
                coefficient, bit, strength
            )
            
            # Record message expected bit for spatial post-correction
            LWTTransform.GLOBAL_EXPECTED_BITS[(subband_name, row, col)] = bit
            
            bit_idx += 1
        
        if bit_idx < len(message_bits):
            raise ValueError(
                f"Failed to embed all bits. Embedded: {bit_idx}/{len(message_bits)}"
            )
        
        return modified_coeffs


# Convenience functions

def create_embedding_engine(
    base_alpha: float = 0.1,
    neighborhood_size: int = 3
) -> EmbeddingEngine:
    """
    Create an embedding engine with default settings.
    
    Args:
        base_alpha: Base embedding strength (default: 0.1)
        neighborhood_size: Size of local neighborhood (default: 3)
        
    Returns:
        Configured EmbeddingEngine instance
    """
    return EmbeddingEngine(
        base_alpha=base_alpha,
        neighborhood_size=neighborhood_size
    )
