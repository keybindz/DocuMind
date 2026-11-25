"""Pydantic schemas for request/response validation and serialization."""
import uuid
from datetime import datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import DocumentStatus, UserRole


# --- Auth ---

class SignupRequest(BaseModel):
    tenant_name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    email: EmailStr
    role: UserRole
    created_at: datetime


class SignupResponse(BaseModel):
    user: UserOut
    token: Token


# --- Documents ---

class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None = None
    char_count: int | None = None
    attempts: int = 0
    created_at: datetime
    processed_at: datetime | None = None


class DocumentContentOut(BaseModel):
    id: uuid.UUID
    filename: str
    extracted_text: str | None = None


class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


# --- Chunks ---

class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    embedding_status: str


class ChunkListOut(BaseModel):
    items: list[ChunkOut]
    total: int
    limit: int
    offset: int


# --- RAG Query ---

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] | None = Field(
        default=None, description="Restrict retrieval to these documents. Omit to search the whole tenant corpus."
    )
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    content: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str | None
    answer_synthesized: bool = Field(
        description="False when no LLM was called (e.g. ANTHROPIC_API_KEY not configured) — "
        "`retrieved_chunks` still reflects real retrieval in that case."
    )
    retrieved_chunks: list[RetrievedChunk]


# --- Webhooks ---

WebhookEvent = Literal["document.ready", "document.failed"]


class WebhookCreate(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    events: list[WebhookEvent] = Field(default_factory=lambda: ["document.ready", "document.failed"], min_length=1)


class WebhookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    url: str
    events: list[str]
    is_active: bool
    created_at: datetime


class WebhookCreatedOut(WebhookOut):
    secret: str = Field(description="Signing secret. Shown ONCE, at creation; store it now.")


class DeliveryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    webhook_id: uuid.UUID
    event_type: str
    status: str
    attempts: int
    last_status_code: int | None = None
    last_error: str | None = None
    created_at: datetime
    delivered_at: datetime | None = None


class DeliveryListOut(BaseModel):
    items: list[DeliveryOut]
    total: int
    limit: int
    offset: int


# --- Tenant ---

class TenantUsageOut(BaseModel):
    tenant_id: uuid.UUID
    name: str
    plan: str
    document_count: int
    max_documents: int
    query_rate_per_minute: int
    upload_rate_per_minute: int


class ErrorResponse(BaseModel):
    detail: str
