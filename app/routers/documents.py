import hashlib
import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.plans import limits_for
from app.database import get_db
from app.dependencies import get_current_user, rate_limit
from app.models import Chunk, Document, DocumentStatus, IdempotencyKey, Tenant, User
from app.schemas import ChunkListOut, DocumentContentOut, DocumentListOut, DocumentOut
from app.services import idempotency
from app.services.pipeline import mark_enqueue_failed
from app.services.storage import StorageBackend, get_storage_backend
from app.services.task_queue import QueueUnavailable, TaskQueue, get_task_queue
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/documents", tags=["documents"])
settings = get_settings()


def _get_owned_document(db: Session, document_id: uuid.UUID, tenant_id: uuid.UUID) -> Document:
    """
    Fetches a document scoped to the caller's tenant. Filtering by tenant_id
    in the WHERE clause (rather than fetching by id then checking) means a
    request for another tenant's document ID returns an indistinguishable
    404 -- not a 403 that would confirm the ID exists. (Row-Level Security is
    the safety net underneath this; see Phase 3.)
    """
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.tenant_id == tenant_id)
        .first()
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


def _replay_for_key(db: Session, tenant_id: uuid.UUID, key: str, fp: str) -> Document | None:
    """
    The single place that decides what an already-seen Idempotency-Key means:
      * no (live) record, or its document is gone -> None: treat as a new request
      * record exists, request differs             -> 422
      * record exists, request identical           -> the original document (a replay)
    Used before creating anything AND when a concurrent request wins the race,
    so every path agrees on the answer.
    """
    record = idempotency.find_active(db, tenant_id, key)
    if record is None:
        return None
    if record.fingerprint != fp:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Idempotency-Key was already used with a different request",
        )
    replayed = db.get(Document, record.document_id) if record.document_id else None
    if replayed is None:
        db.delete(record)  # its document was deleted since; treat as a fresh request
        db.flush()
    return replayed


def _reset_for_retry(document: Document) -> None:
    document.status = DocumentStatus.pending
    document.error_message = None
    document.attempts = 0
    document.extracted_text = None  # re-extract: a failed extraction must not be skipped on retry
    document.char_count = None


def _enqueue_or_fail(db: Session, queue: TaskQueue, document: Document) -> None:
    """
    Hands the document to the worker. If the broker is down the document
    exists but nothing will ever process it, so we say so loudly (503) and mark
    it failed -- rather than leaving it 'pending' forever -- and release any
    idempotency key so the client can retry with the SAME key.

    Production-grade alternative: a transactional outbox for ingestion jobs
    (as webhooks already use), drained by a sweeper. Skipped here because a
    clear 503 + the /retry endpoint covers the failure honestly.
    """
    try:
        queue.enqueue_ingestion(document.id, document.tenant_id)
    except QueueUnavailable:
        logger.exception("Could not enqueue ingestion for document %s", document.id)
        db.query(IdempotencyKey).filter(IdempotencyKey.document_id == document.id).delete()
        mark_enqueue_failed(db, document, "Processing queue was unavailable; use POST /v1/documents/{id}/retry")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Processing queue unavailable. Document {document.id} was saved with status 'failed'; "
                "retry with POST /v1/documents/{id}/retry."
            ),
            headers={"Retry-After": "30"},
        )


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit("upload"))],
    responses={
        200: {"description": "Identical file already uploaded by this tenant; existing document returned."},
        422: {"description": "Idempotency-Key was already used with a different request."},
        503: {"description": "Processing queue unavailable."},
    },
)
def upload_document(
    file: UploadFile,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    storage: StorageBackend = Depends(get_storage_backend),
    queue: TaskQueue = Depends(get_task_queue),
):
    """
    Accepts a document and returns **202 immediately** with status `pending`.
    A background worker extracts, chunks, and embeds it; poll
    `GET /v1/documents/{id}` or register a webhook to learn when it's `ready`.

    Send an `Idempotency-Key` header to make retries safe: repeating the same
    request with the same key returns the original document instead of
    creating a duplicate.
    """
    key = idempotency.validate_key(idempotency_key) if idempotency_key is not None else None

    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )
    # Read at most limit+1 bytes: enough to detect "too big" without buffering
    # an arbitrarily large body into memory first.
    content = file.file.read(settings.MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_BYTES} bytes",
        )
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    file_hash = hashlib.sha256(content).hexdigest()
    filename = file.filename or "untitled"
    tenant_id = current_user.tenant_id
    fp = idempotency.fingerprint(file_hash, filename, file.content_type)

    # ---- Idempotency replay ----
    if key:
        replayed = _replay_for_key(db, tenant_id, key, fp)
        if replayed is not None:
            response.headers["Idempotent-Replay"] = "true"
            return replayed

    # ---- Content-hash dedupe (a different feature from idempotency) ----
    existing = db.query(Document).filter(Document.tenant_id == tenant_id, Document.file_hash == file_hash).first()
    if existing:
        revive = existing.status == DocumentStatus.failed
        if revive:
            # Re-uploading bytes whose earlier processing failed (e.g. during a
            # queue outage) means "try again", not "here's your failed document".
            _reset_for_retry(existing)
        if key:
            db.add(idempotency.new_record(tenant_id, key, fp, existing.id))
        try:
            db.commit()
        except IntegrityError:
            # A concurrent request with this same key won the race. This request
            # is a retry of that one, so answer exactly as a replay would.
            db.rollback()
            replayed = _replay_for_key(db, tenant_id, key, fp) if key else None
            if replayed is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A request with this Idempotency-Key is already in progress",
                )
            response.headers["Idempotent-Replay"] = "true"
            return replayed
        if revive and existing is not None:
            db.refresh(existing)
            _enqueue_or_fail(db, queue, existing)
            db.refresh(existing)
            return existing  # 202: work was accepted
        response.status_code = status.HTTP_200_OK  # nothing new was accepted
        return existing

    # ---- Plan quota (after dedupe so re-uploads are free) ----
    tenant = db.get(Tenant, tenant_id)
    plan = tenant.plan if tenant else "free"
    max_docs = limits_for(plan).max_documents
    if db.query(Document).filter(Document.tenant_id == tenant_id).count() >= max_docs:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Document quota exceeded for plan '{plan}' ({max_docs} documents). Delete documents or upgrade.",
        )

    # ---- Create the document and (atomically) its idempotency record ----
    document = Document(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        uploaded_by=current_user.id,
        filename=filename,
        storage_path="",
        mime_type=file.content_type,
        file_hash=file_hash,
        size_bytes=len(content),
        status=DocumentStatus.pending,
    )
    db.add(document)
    db.flush()  # the idempotency row references the document
    if key:
        db.add(idempotency.new_record(tenant_id, key, fp, document.id))

    document.storage_path = storage.save(
        tenant_id=tenant_id, document_id=document.id, filename=filename, content=content
    )
    try:
        db.commit()
    except IntegrityError:
        # A concurrent request with the same Idempotency-Key committed first.
        db.rollback()
        storage.delete(f"{tenant_id}/{document.id}_{filename}")
        winner = _replay_for_key(db, tenant_id, key, fp) if key else None
        if winner is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A request with this Idempotency-Key is already in progress",
            )
        response.headers["Idempotent-Replay"] = "true"
        return winner

    db.refresh(document)
    _enqueue_or_fail(db, queue, document)
    db.refresh(document)  # reflects any state the worker has already reached
    return document


@router.post(
    "/{document_id}/retry",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit("upload"))],
)
def retry_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    queue: TaskQueue = Depends(get_task_queue),
):
    """Re-queue a document whose processing `failed` (e.g. after a queue outage)."""
    document = _get_owned_document(db, document_id, current_user.tenant_id)
    if document.status != DocumentStatus.failed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Only failed documents can be retried (status={document.status.value})",
        )
    _reset_for_retry(document)
    db.commit()
    db.refresh(document)
    _enqueue_or_fail(db, queue, document)
    db.refresh(document)
    return document


@router.get("", response_model=DocumentListOut)
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    base_query = db.query(Document).filter(Document.tenant_id == current_user.tenant_id)
    total = base_query.count()
    items = base_query.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    return DocumentListOut(items=items, total=total, limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Poll this to follow progress: pending -> extracting -> chunking -> embedding -> ready | failed."""
    return _get_owned_document(db, document_id, current_user.tenant_id)


@router.get("/{document_id}/content", response_model=DocumentContentOut)
def get_document_content(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = _get_owned_document(db, document_id, current_user.tenant_id)
    if document.status != DocumentStatus.ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is not ready (status={document.status.value})",
        )
    return DocumentContentOut(id=document.id, filename=document.filename, extracted_text=document.extracted_text)


@router.get("/{document_id}/chunks", response_model=ChunkListOut)
def list_chunks(
    document_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Debug/inspection endpoint: see exactly what chunks a document was split into."""
    _get_owned_document(db, document_id, current_user.tenant_id)  # 404s if not owned
    base_query = db.query(Chunk).filter(Chunk.document_id == document_id, Chunk.tenant_id == current_user.tenant_id)
    total = base_query.count()
    items = base_query.order_by(Chunk.chunk_index).offset(offset).limit(limit).all()
    return ChunkListOut(items=items, total=total, limit=limit, offset=offset)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    storage: StorageBackend = Depends(get_storage_backend),
    vector_store: VectorStore = Depends(get_vector_store),
):
    document = _get_owned_document(db, document_id, current_user.tenant_id)
    storage.delete(document.storage_path)
    vector_store.delete_by_document(tenant_id=current_user.tenant_id, document_id=document.id)
    db.delete(document)  # cascades to chunks; a still-queued job will find nothing and no-op
    db.commit()
    return None
