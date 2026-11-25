from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user, rate_limit
from app.models import Chunk, Document, User
from app.schemas import QueryRequest, QueryResponse, RetrievedChunk
from app.services.embeddings import EmbeddingBackend, get_embedding_backend
from app.services.rag import ContextChunk, synthesize_answer
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(prefix="/v1", tags=["query"])
settings = get_settings()


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit("query"))])
def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    embedding_backend: EmbeddingBackend = Depends(get_embedding_backend),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Retrieval-augmented query over the caller's tenant corpus.

    1. Embed the question
    2. Vector search, hard-filtered to the caller's tenant_id (and optionally
       a document_ids subset)
    3. Fetch the matched chunks' text from Postgres (vector store payload
       never carries content, see vector_store.py)
    4. If an LLM key is configured, synthesize a cited answer; otherwise
       return the raw retrieved chunks with answer_synthesized=False
    """
    [query_vector] = embedding_backend.embed([payload.question])

    hits = vector_store.search(
        tenant_id=current_user.tenant_id,
        query_vector=query_vector,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )

    if not hits:
        return QueryResponse(
            question=payload.question,
            answer="No relevant content was found in your documents to answer this question."
            if settings.ANTHROPIC_API_KEY
            else None,
            answer_synthesized=bool(settings.ANTHROPIC_API_KEY),
            retrieved_chunks=[],
        )

    chunk_ids = [h.chunk_id for h in hits]
    chunk_rows = {
        c.id: c
        for c in db.query(Chunk).filter(Chunk.id.in_(chunk_ids), Chunk.tenant_id == current_user.tenant_id).all()
    }
    doc_ids = {h.document_id for h in hits}
    doc_filenames = {
        d.id: d.filename
        for d in db.query(Document).filter(Document.id.in_(doc_ids), Document.tenant_id == current_user.tenant_id)
    }

    retrieved: list[RetrievedChunk] = []
    context_chunks: list[ContextChunk] = []
    for hit in hits:
        chunk_row = chunk_rows.get(hit.chunk_id)
        if chunk_row is None:
            continue  # vector store and Postgres briefly out of sync; skip rather than 500
        filename = doc_filenames.get(hit.document_id, "unknown")
        retrieved.append(
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_filename=filename,
                content=chunk_row.content,
                score=hit.score,
            )
        )
        context_chunks.append(
            ContextChunk(chunk_id=str(hit.chunk_id), document_filename=filename, content=chunk_row.content)
        )

    answer = synthesize_answer(payload.question, context_chunks)

    return QueryResponse(
        question=payload.question,
        answer=answer,
        answer_synthesized=answer is not None,
        retrieved_chunks=retrieved,
    )
