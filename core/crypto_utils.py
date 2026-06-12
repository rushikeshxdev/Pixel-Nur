import os
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA256

SALT_SIZE = 16
NONCE_SIZE = 16
TAG_SIZE = 16
ITERATIONS = 100_000

def derive_key(password: str, salt: bytes) -> bytes:
    return PBKDF2(
        password.encode("utf-8"),
        salt,
        dkLen=32,
        count=ITERATIONS,
        hmac_hash_module=SHA256,
    )

def encrypt_message(message: str, password: str) -> bytes:
    salt = get_random_bytes(SALT_SIZE)
    nonce = get_random_bytes(NONCE_SIZE)
    key = derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_SIZE)
    ciphertext, tag = cipher.encrypt_and_digest(message.encode("utf-8"))
    return salt + nonce + ciphertext + tag

def decrypt_message(data: bytes, password: str) -> str:
    if len(data) < SALT_SIZE + NONCE_SIZE + TAG_SIZE:
        raise ValueError("Encrypted data too short — likely wrong image or password.")
    salt = data[:SALT_SIZE]
    nonce = data[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    tag = data[-TAG_SIZE:]
    ciphertext = data[SALT_SIZE + NONCE_SIZE:-TAG_SIZE]
    key = derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=TAG_SIZE)
    try:
        return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")
    except (ValueError, KeyError):
        raise ValueError("Wrong password or corrupted stego image.")
