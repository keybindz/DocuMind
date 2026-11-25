"""
Symmetric encryption for secrets that must be recoverable (webhook signing
secrets). Fernet = AES-128-CBC + HMAC-SHA256, authenticated, with the key
derived from config. This protects secrets from a database-only leak (a
backup, a read-only SQL injection); it does not protect against an attacker
who also has the application's environment -- that needs a KMS (Phase 5+).
"""
import base64
import hashlib

from cryptography.fernet import Fernet

from app.config import get_settings


def _fernet() -> Fernet:
    settings = get_settings()
    material = settings.WEBHOOK_ENCRYPTION_KEY or settings.SECRET_KEY
    key = base64.urlsafe_b64encode(hashlib.sha256(material.encode()).digest())
    return Fernet(key)


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    return _fernet().decrypt(token.encode()).decode()
