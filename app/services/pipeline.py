"""
The document ingestion pipeline: extract -> chunk -> embed -> ready.

This replaces the synchronous `index_document()` call from Phases 1-2. The
stage functions are the same ideas; what changed is *who runs them* (a
background worker, not the upload request) and *how they fail*:

- Each stage COMMITS, so `GET /v1/documents/{id}` shows real progress
  (extracting -> chunking -> embedding -> ready) while the job runs.
- The job is IDEMPOTENT. Celery with acks_late gives at-least-once delivery:
  a worker that dies mid-job means the job runs again. Re-running must never
  duplicate chunks or vectors, so the chunk/embed stage first deletes whatever
  an earlier attempt left behind, and a document that is already `ready` is a
  no-op.
- Failures are classified. ExtractionError (bad file) is PERMANENT: retrying a
  corrupt PDF just wastes worker time. TransientError (embedding model or
  vector store hiccup) is RETRYABLE with exponential backoff, up to a limit.
- Terminal transitions (ready/failed) create webhook delivery rows in the SAME
  transaction (outbox pattern, see services/webhooks.py).
"""
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import ObjectDeletedError, StaleDataError

from app.config import get_settings
from app.models import Chunk, Document, DocumentStatus
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingBackend
from app.services.extraction import ExtractionError, extract_text
from app.services.jobs import JobResult, TransientError, exponential_backoff
from app.services.storage import StorageBackend
from app.services.vector_store import VectorRecord, VectorStore
from app.services.webhooks import emit_document_event

logger = logging.getLogger(__name__)
settings = get_settings()


def _fail(db: Session, document: Document, message: str) -> JobResult:
    document.status = DocumentStatus.failed
    document.error_message = message[:2000]
    document.processed_at = datetime.now(timezone.utc)
    delivery_ids = emit_document_event(db, document, "document.failed")
    db.commit()
    return JobResult(delivery_ids=delivery_ids)


def fail_document(db: Session, document_id: uuid.UUID, tenant_id: uuid.UUID, message: str) -> JobResult:
    """Last resort when a job dies with an unexpected error and retries are exhausted."""
    document = db.query(Document).filter(Document.id == document_id, Document.tenant_id == tenant_id).first()
    if document is None or document.status == DocumentStatus.ready:
        return JobResult()
    return _fail(db, document, message)


def mark_enqueue_failed(db: Session, document: Document, message: str) -> None:
    """Used by the API when the broker is down: the document exists but nothing will process it."""
    document.status = DocumentStatus.failed
    document.error_message = message
    db.commit()


def run_ingestion_job(
    db: Session,
    *,
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    attempt: int,
    max_attempts: int,
    storage: StorageBackend,
    embedding_backend: EmbeddingBackend,
    vector_store: VectorStore,
) -> JobResult:
    # `db` is already scoped to tenant_id (RLS); the explicit filter is
    # belt-and-suspenders, same as in the API routes.
    document = db.query(Document).filter(Document.id == document_id, Document.tenant_id == tenant_id).first()
    if document is None:
        logger.info("Ingestion skipped: document %s no longer exists", document_id)
        return JobResult()  # deleted while queued -- nothing to do
    if document.status == DocumentStatus.ready:
        return JobResult()  # duplicate delivery of an already-finished job

    document.attempts = attempt
    try:
        # ---- Stage 1: extraction (skipped on retry if already done) ----
        if not document.extracted_text:
            document.status = DocumentStatus.extracting
            document.error_message = None
            db.commit()
            try:
                content = storage.read(document.storage_path)
            except FileNotFoundError:
                return _fail(db, document, "Uploaded file is missing from storage")
            try:
                text = extract_text(content, document.mime_type)
            except ExtractionError as exc:
                return _fail(db, document, str(exc))  # permanent: never retried
            document.extracted_text = text
            document.char_count = len(text)
            db.commit()

        # ---- Stage 2: chunking (idempotent: clear leftovers from earlier attempts) ----
        document.status = DocumentStatus.chunking
        db.commit()
        vector_store.delete_by_document(tenant_id=tenant_id, document_id=document_id)
        db.query(Chunk).filter(Chunk.document_id == document_id).delete(synchronize_session=False)

        text_chunks = chunk_text(
            document.extracted_text,
            chunk_size_words=settings.CHUNK_SIZE_WORDS,
            overlap_words=settings.CHUNK_OVERLAP_WORDS,
        )
        if not text_chunks:
            return _fail(db, document, "Chunking produced no chunks")
        chunk_rows = [
            Chunk(
                id=uuid.uuid4(),  # assigned now so we can keep plain values: ORM objects expire on commit
                document_id=document_id,
                tenant_id=tenant_id,
                chunk_index=tc.index,
                content=tc.content,
                token_count=tc.word_count,
                embedding_status="pending",
            )
            for tc in text_chunks
        ]
        chunk_ids = [c.id for c in chunk_rows]
        chunk_texts = [c.content for c in chunk_rows]
        db.add_all(chunk_rows)
        document.status = DocumentStatus.embedding
        db.commit()

        # ---- Stage 3: embedding + vector upsert (the retryable part) ----
        # Only plain values (ids, strings) are touched inside this block. If it
        # touched expired ORM objects, "the document was deleted mid-job" would
        # surface here as a DB error and be misclassified as a transient failure.
        try:
            vectors = embedding_backend.embed(chunk_texts)
            vector_store.upsert(
                [
                    VectorRecord(chunk_id=cid, document_id=document_id, tenant_id=tenant_id, vector=v)
                    for cid, v in zip(chunk_ids, vectors)
                ]
            )
        except Exception as exc:  # noqa: BLE001 - embedder/vector store failures are all retryable
            raise TransientError(f"Embedding/indexing failed: {exc}") from exc

        db.query(Chunk).filter(Chunk.document_id == document_id).update(
            {"embedding_status": "embedded"}, synchronize_session=False
        )
        document.status = DocumentStatus.ready
        document.error_message = None
        document.processed_at = datetime.now(timezone.utc)
        delivery_ids = emit_document_event(db, document, "document.ready")
        db.commit()  # raises StaleDataError/IntegrityError if the document vanished -> handled below
        return JobResult(delivery_ids=delivery_ids)

    except (IntegrityError, StaleDataError, ObjectDeletedError):
        # The document was deleted while we were working on it. Drop whatever
        # we wrote to the vector store so no orphan vectors outlive the document.
        db.rollback()
        vector_store.delete_by_document(tenant_id=tenant_id, document_id=document_id)
        logger.info("Document %s deleted mid-ingestion; cleaned up its vectors", document_id)
        return JobResult()

    except TransientError as exc:
        db.rollback()
        if attempt >= max_attempts:
            document = db.get(Document, document_id)
            if document is None:
                return JobResult()
            return _fail(db, document, f"Gave up after {attempt} attempts: {exc}")
        document = db.get(Document, document_id)
        if document is not None:
            document.attempts = attempt
            document.error_message = f"Attempt {attempt} failed, will retry: {exc}"[:2000]
            db.commit()
        delay = exponential_backoff(attempt, base=5, cap=60)
        logger.warning("Ingestion of %s failed (attempt %d/%d), retrying in %ss: %s", document_id, attempt, max_attempts, delay, exc)
        return JobResult(retry_in=delay)
