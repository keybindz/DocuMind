"""
Centralized application settings.

All configuration is read from environment variables (or a .env file in
local dev). Nothing here should be hardcoded per-environment — the same
image is deployed to dev/staging/prod with different env vars.
"""
from functools import lru_cache

import email_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Email format is still validated (RFC syntax, domain shape) — only the
# live DNS/MX "does this domain accept mail" check is disabled. Deliverability
# checking makes signup depend on network/DNS at request time, which is
# fragile in CI, in offline dev, and behind restrictive egress rules, and it
# isn't actually necessary: an undeliverable address just bounces later.
email_validator.CHECK_DELIVERABILITY = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "DocuMind"
    ENV: str = "development"
    DEBUG: bool = True

    # --- Database ---
    # Example (docker-compose): postgresql+psycopg2://documind:documind@postgres:5432/documind
    DATABASE_URL: str = "sqlite:///./documind_dev.db"
    # Phase 3: the API connects as a restricted, non-superuser role so
    # Postgres Row-Level Security actually applies to it (superusers and
    # BYPASSRLS roles skip RLS entirely). Schema migrations need DDL rights,
    # so Alembic uses this separate owner-role URL when it's set.
    MIGRATION_DATABASE_URL: str | None = None
    # Dev convenience only. In staging/prod, schema changes go through Alembic
    # migrations (`alembic upgrade head`), never through metadata.create_all().
    AUTO_CREATE_TABLES: bool = True

    # --- Auth ---
    SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h — fine for a demo/portfolio project

    # --- Storage ---
    # Local filesystem for Phase 1. Swapped for an S3/MinIO-backed
    # implementation behind the same StorageBackend interface in a later phase.
    STORAGE_DIR: str = "./storage_data"

    # --- Chunking ---
    CHUNK_SIZE_WORDS: int = 220
    CHUNK_OVERLAP_WORDS: int = 30

    # --- Embeddings ---
    # "sentence-transformers" for real use (docker-compose default); "fake"
    # for tests — see app/services/embeddings.py for why.
    EMBEDDING_BACKEND: str = "sentence-transformers"
    EMBEDDING_DIMENSION: int = 384  # must match the selected backend's output dimension

    # --- Vector store ---
    # "qdrant" for real use (docker-compose default); "memory" for tests.
    VECTOR_STORE_BACKEND: str = "qdrant"
    QDRANT_URL: str = "http://localhost:6333"

    # --- RAG / LLM synthesis ---
    # If unset, /v1/query still performs retrieval and returns the matched
    # chunks, but skips LLM answer synthesis and says so explicitly in the
    # response rather than failing the request.
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"
    RAG_TOP_K: int = 5

    # --- Rate limiting (Phase 3) ---
    # "redis" for real use; "memory" for tests / single-process dev.
    RATE_LIMIT_BACKEND: str = "redis"
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Async ingestion (Phase 4) ---
    # "celery" for real use; "memory" for tests (see services/task_queue.py).
    TASK_QUEUE_BACKEND: str = "celery"
    # Separate Redis DB index from the rate limiter's, so flushing one never
    # touches the other.
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    INGESTION_MAX_RETRIES: int = 3       # retries after the first attempt (4 attempts total)
    IDEMPOTENCY_TTL_HOURS: int = 24

    # --- Webhooks (Phase 4) ---
    WEBHOOK_MAX_ATTEMPTS: int = 5
    WEBHOOK_TIMEOUT_SECONDS: float = 5.0
    # Off by default: webhook URLs resolving to private/loopback/link-local
    # addresses are rejected (SSRF protection). Turn on ONLY for local dev
    # (e.g. a receiver on your laptop) -- docker-compose does.
    WEBHOOK_ALLOW_PRIVATE_URLS: bool = False
    WEBHOOK_SIGNATURE_TOLERANCE_SECONDS: int = 300
    # Webhook signing secrets are stored encrypted (they must be recoverable
    # to sign deliveries, so hashing isn't an option). If unset, a key is
    # derived from SECRET_KEY.
    WEBHOOK_ENCRYPTION_KEY: str | None = None

    # --- Uploads ---
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    ALLOWED_MIME_TYPES: tuple[str, ...] = (
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
