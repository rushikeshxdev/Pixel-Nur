"""
Simple, reliable LSB steganography with AES-256-GCM encryption.
Packet format: MAGIC(4) | BPC(1) | MSG_LEN(4) | SALT(16) | NONCE(16) | CIPHERTEXT | TAG(16)
"""

import os
import struct
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA256

MAGIC = b'SNUR'
SALT_SIZE = 16
NONCE_SIZE = 16
TAG_SIZE = 16
HEADER_SIZE = 9  # MAGIC(4) + BPC(1) + MSG_LEN(4)
ITERATIONS = 100_000

# bits-per-channel for each robustness level
ROBUSTNESS_BPC = {
    "None":   1,
    "Low":    1,
    "Medium": 2,
    "High":   2,
}


def _derive_key(password: str, salt: bytes) -> bytes:
    return PBKDF2(
        password.encode("utf-8"),
        salt,
        dkLen=32,
        count=ITERATIONS,
        hmac_hash_module=SHA256,
    )


def _encrypt(message: str, password: str) -> bytes:
    salt = get_random_bytes(SALT_SIZE)
    nonce = get_random_bytes(NONCE_SIZE)
    key = _derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_SIZE)
    ciphertext, tag = cipher.encrypt_and_digest(message.encode("utf-8"))
    return salt + nonce + ciphertext + tag


def _decrypt(data: bytes, password: str) -> str:
    if len(data) < SALT_SIZE + NONCE_SIZE + TAG_SIZE:
        raise ValueError("Encrypted data too short — likely wrong image or password.")
    salt = data[:SALT_SIZE]
    nonce = data[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    tag = data[-TAG_SIZE:]
    ciphertext = data[SALT_SIZE + NONCE_SIZE:-TAG_SIZE]
    key = _derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_SIZE)
    try:
        return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")
    except (ValueError, KeyError):
        raise ValueError("Wrong password or corrupted stego image.")


def _bytes_to_bits(data: bytes) -> list:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: list) -> bytes:
    result = bytearray()
    for i in range(0, len(bits) - 7, 8):
        val = 0
        for b in bits[i:i + 8]:
            val = (val << 1) | b
        result.append(val)
    return bytes(result)


def embed(image: Image.Image, message: str, password: str, robustness: str = "None"):
    """
    Embed an encrypted message into a cover image using LSB substitution.
    Returns (stego_image: PIL.Image, metrics: dict).
    Always save the returned image as PNG to preserve LSBs.
    """
    bpc = ROBUSTNESS_BPC[robustness]

    encrypted = _encrypt(message, password)

    # Packet: MAGIC | BPC byte | 4-byte big-endian length | encrypted payload
    packet = MAGIC + bytes([bpc]) + struct.pack(">I", len(encrypted)) + encrypted
    bits = _bytes_to_bits(packet)

    img = image.convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    h, w, _ = img_array.shape
    flat = img_array.flatten().copy()

    max_bits = len(flat) * bpc
    if len(bits) > max_bits:
        max_msg_bytes = (max_bits // 8) - HEADER_SIZE - SALT_SIZE - NONCE_SIZE - TAG_SIZE
        raise ValueError(
            f"Message too large for this image.\n"
            f"Max usable capacity: ~{max(0, max_msg_bytes)} bytes  |  "
            f"Needed: {len(encrypted)} bytes encrypted.\n"
            f"Try a larger image or a shorter message."
        )

    lsb_mask = 0xFF ^ ((1 << bpc) - 1)  # e.g. bpc=1 → 0xFE, bpc=2 → 0xFC

    bit_idx = 0
    for i in range(len(flat)):
        if bit_idx >= len(bits):
            break
        chunk = 0
        for _ in range(bpc):
            chunk = (chunk << 1) | (bits[bit_idx] if bit_idx < len(bits) else 0)
            bit_idx += 1
        flat[i] = (flat[i] & lsb_mask) | chunk

    result_array = flat.reshape(img_array.shape)
    stego = Image.fromarray(result_array, "RGB")

    # Metrics
    orig_f = img_array.astype(np.float64)
    modi_f = result_array.astype(np.float64)
    mse = np.mean((orig_f - modi_f) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

    capacity_pct = len(bits) / max_bits * 100
    max_kb = max_bits / 8 / 1024

    metrics = {
        "PSNR": f"{psnr:.2f} dB",
        "MSE": f"{mse:.6f}",
        "Image Size": f"{w}x{h} px",
        "Bits per Channel": str(bpc),
        "Robustness Level": robustness,
        "Message Length": f"{len(message)} chars",
        "Encrypted Payload": f"{len(encrypted)} bytes",
        "Capacity Used": f"{capacity_pct:.2f}%",
        "Total Capacity": f"{max_kb:.1f} KB",
    }

    return stego, metrics


def extract(image: Image.Image, password: str) -> str:
    """
    Extract and decrypt a hidden message from a stego image.
    Automatically detects bits-per-channel from the embedded header.
    """
    img_array = np.array(image.convert("RGB"), dtype=np.uint8)
    flat = img_array.flatten()

    for bpc in [1, 2]:
        try:
            result = _try_extract(flat, password, bpc)
            if result is not None:
                return result
        except ValueError as e:
            # Wrong password or bad magic — re-raise only on the last attempt
            if bpc == 2:
                raise e
        except Exception:
            continue

    raise ValueError("Could not extract a message. Check the image and password.")


def _try_extract(flat: np.ndarray, password: str, bpc: int) -> str:
    lsb_mask = (1 << bpc) - 1

    def read_bits(n_bytes):
        n_bits = n_bytes * 8
        n_channels = (n_bits + bpc - 1) // bpc
        bits = []
        for i in range(min(n_channels, len(flat))):
            val = flat[i] & lsb_mask
            for b in range(bpc - 1, -1, -1):
                bits.append((val >> b) & 1)
        return bits[:n_bits]

    # Read and validate header
    header_bits = read_bits(HEADER_SIZE)
    if len(header_bits) < HEADER_SIZE * 8:
        return None

    header_bytes = _bits_to_bytes(header_bits)
    if header_bytes[:4] != MAGIC:
        return None

    stored_bpc = header_bytes[4]
    if stored_bpc != bpc:
        return None

    msg_len = struct.unpack(">I", header_bytes[5:9])[0]
    if msg_len == 0 or msg_len > 10 * 1024 * 1024:
        return None

    # Read full payload
    total_bytes = HEADER_SIZE + msg_len
    total_channels = (total_bytes * 8 + bpc - 1) // bpc
    if total_channels > len(flat):
        raise ValueError("Image too small to contain the claimed payload size.")

    all_bits = []
    for i in range(total_channels):
        val = flat[i] & lsb_mask
        for b in range(bpc - 1, -1, -1):
            all_bits.append((val >> b) & 1)

    payload_bits = all_bits[HEADER_SIZE * 8: HEADER_SIZE * 8 + msg_len * 8]
    encrypted = _bits_to_bytes(payload_bits)

    return _decrypt(encrypted, password)
