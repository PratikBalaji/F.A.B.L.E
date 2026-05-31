"""
Application-level encryption for provider credentials (AES-256-GCM).

Why app-level (not pgsodium/Vault): ciphertext stays opaque to Postgres, so a
service-role leak or RLS misconfiguration exposes only ciphertext, never the
plaintext API key. The 32-byte key lives solely in APP_ENCRYPTION_KEY (env),
never in the database.

Ciphertext layout (before base64): nonce(12) || ciphertext || GCM tag(16).
Stored as base64 text in provider_connections.secret_enc.
"""
from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import settings

_NONCE_BYTES = 12
KEY_VERSION = 1  # bump + add a key map here when rotating keys


def _load_key() -> bytes:
    raw = settings.app_encryption_key
    if not raw:
        raise RuntimeError(
            "APP_ENCRYPTION_KEY is not set — cannot encrypt/decrypt provider credentials. "
            'Generate one: python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"'
        )
    try:
        key = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("APP_ENCRYPTION_KEY is not valid base64") from exc
    if len(key) != 32:
        raise RuntimeError(
            f"APP_ENCRYPTION_KEY must decode to 32 bytes for AES-256-GCM (got {len(key)})"
        )
    return key


def encrypt(plaintext: str) -> str:
    """Encrypt a secret and return base64(nonce || ciphertext || tag)."""
    key = _load_key()
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt(token: str) -> str:
    """Reverse of encrypt(). Takes base64 text, returns the plaintext secret."""
    key = _load_key()
    blob = base64.b64decode(token)
    nonce, ciphertext = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    return AESGCM(key).decrypt(nonce, ciphertext, None).decode("utf-8")
