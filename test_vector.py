"""
End-to-end test: embed → extract, and print performance & SSIM.
Run from the pixel-nur root: python test_vector.py
"""
import time
import numpy as np
from PIL import Image

from core.steganography import embed, extract

# ── Create a synthetic 1920×1080 test image ──────────────────────────────
print("Creating 1920×1080 test image …")
rng = np.random.default_rng(42)
img_array = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
img = Image.fromarray(img_array, "RGB")

message  = "PixelNur: AES-256-GCM + Vectorized LSB steganography. Final Year Project 2024-25. " * 10
password = "SuperSecure@2025!"

# ── Embed ─────────────────────────────────────────────────────────────────
print(f"Embedding {len(message)} chars …")
t0 = time.perf_counter()
stego, metrics = embed(img, message, password, robustness="None")
embed_time = time.perf_counter() - t0

print(f"  Embed time  : {embed_time*1000:.1f} ms")
for k, v in metrics.items():
    print(f"  {k:<22}: {v}")

# ── Extract ───────────────────────────────────────────────────────────────
print("\nExtracting …")
t1 = time.perf_counter()
recovered = extract(stego, password)
extract_time = time.perf_counter() - t1

print(f"  Extract time: {extract_time*1000:.1f} ms")
assert recovered == message, "[FAIL] MISMATCH - extraction failed!"
print("  [PASS] Message matches perfectly!")
print("\nAll tests passed.")
