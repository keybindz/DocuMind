"""
Idempotency-Key support for POST /v1/documents.

Semantics (modeled on Stripe's):
  * same key + same request     -> return the original result, do nothing new
  * same key + DIFFERENT request -> 422 (the client is reusing a key by mistake)
  * keys are scoped per tenant, and expire (IDEMPOTENCY_TTL_HOURS)

Why it matters: a client uploads a 15 MB file, the response is lost to a
timeout, the client retries. Without a key that creates a second document (or
burns quota). With one, the retry is safe by construction.

The key row is inserted in the same transaction as the document, and a unique
constraint on (tenant_id, key) serializes concurrent retries: exactly one
commits, the other gets an IntegrityError and replays the winner's result.
"""
import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import IdempotencyKey

settings = get_settings()
_VALID_KEY = re.compile(r"^[\x21-\x7e]{1,255}$")  # printable ASCII, no spaces


def validate_key(key: str) -> str:
    if not _VALID_KEY.match(key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key must be 1-255 printable ASCII characters with no spaces",
        )
    return key


def fingerprint(file_hash: str, filename: str, content_type: str) -> str:
    return hashlib.sha256(f"{file_hash}|{filename}|{content_type}".encode()).hexdigest()


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)  # SQLite drops tzinfo


def find_active(db: Session, tenant_id: uuid.UUID, key: str) -> IdempotencyKey | None:
    record = db.query(IdempotencyKey).filter(IdempotencyKey.tenant_id == tenant_id, IdempotencyKey.key == key).first()
    if record is not None and _aware(record.expires_at) <= datetime.now(timezone.utc):
        db.delete(record)  # expired: forget it and treat the request as new
        db.flush()
        return None
    return record


def new_record(tenant_id: uuid.UUID, key: str, fp: str, document_id: uuid.UUID) -> IdempotencyKey:
    return IdempotencyKey(
        tenant_id=tenant_id,
        key=key,
        fingerprint=fp,
        document_id=document_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=settings.IDEMPOTENCY_TTL_HOURS),
    )
