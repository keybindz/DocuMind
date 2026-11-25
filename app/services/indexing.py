"""
Indexing pipeline: turns a document's extracted text into searchable chunks.

Runs synchronously within the upload request for Phase 2, same as
extraction did in Phase 1 — this whole function is the seam Phase 4 moves
behind a queue/worker; callers won't need to change, only when they call it.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Chunk, Document, DocumentStatus
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingBackend
from app.services.vector_store import VectorRecord, VectorStore

settings = get_settings()


class IndexingError(Exception):
    pass


def index_document(
    db: Session,
    document: Document,
    *,
    embedding_backend: EmbeddingBackend,
    vector_store: VectorStore,
) -> None:
    """
    Mutates `document` in place (status, timestamps) and creates its Chunk
    rows + vector store entries. Caller is responsible for commit().
    Raises IndexingError on failure and leaves document.status == failed
    with error_message set, rather than raising past the caller — mirrors
    how extraction failures are handled.
    """
    if not document.extracted_text:
        document.status = DocumentStatus.failed
        document.error_message = "No extracted text to index"
        return

    document.status = DocumentStatus.chunking
    text_chunks = chunk_text(
        document.extracted_text,
        chunk_size_words=settings.CHUNK_SIZE_WORDS,
        overlap_words=settings.CHUNK_OVERLAP_WORDS,
    )
    if not text_chunks:
        document.status = DocumentStatus.failed
        document.error_message = "Chunking produced no chunks"
        return

    chunk_rows = [
        Chunk(
            document_id=document.id,
            tenant_id=document.tenant_id,
            chunk_index=tc.index,
            content=tc.content,
            token_count=tc.word_count,
            embedding_status="pending",
        )
        for tc in text_chunks
    ]
    db.add_all(chunk_rows)
    db.flush()  # populate chunk_rows[*].id

    document.status = DocumentStatus.embedding
    try:
        vectors = embedding_backend.embed([c.content for c in chunk_rows])
    except Exception as exc:  # noqa: BLE001
        document.status = DocumentStatus.failed
        document.error_message = f"Embedding failed: {exc}"
        for c in chunk_rows:
            c.embedding_status = "failed"
        return

    records = [
        VectorRecord(chunk_id=c.id, document_id=document.id, tenant_id=document.tenant_id, vector=v)
        for c, v in zip(chunk_rows, vectors)
    ]
    vector_store.upsert(records)
    for c in chunk_rows:
        c.embedding_status = "embedded"

    document.status = DocumentStatus.ready
    document.processed_at = datetime.now(timezone.utc)
