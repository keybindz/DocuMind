"""
LLM answer synthesis for the RAG query endpoint.

Deliberately optional: retrieval (the vector search) is the part of this
pipeline that's actually being showcased architecturally, and it works
completely independently of whether an LLM key is configured. If
ANTHROPIC_API_KEY is unset, /v1/query still returns real retrieved chunks
and simply skips synthesis rather than failing the request — see
`synthesize_answer`'s None return and how the router surfaces it via
`answer_synthesized`.
"""
from dataclasses import dataclass

from app.config import get_settings

settings = get_settings()


@dataclass
class ContextChunk:
    chunk_id: str
    document_filename: str
    content: str


def is_synthesis_available() -> bool:
    return bool(settings.ANTHROPIC_API_KEY)


def synthesize_answer(question: str, chunks: list[ContextChunk]) -> str | None:
    if not is_synthesis_available():
        return None
    if not chunks:
        return "No relevant content was found in your documents to answer this question."

    from anthropic import Anthropic  # local import: optional dep, only needed when a key is configured

    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    context_block = "\n\n".join(
        f"[Source {i+1}: {c.document_filename}]\n{c.content}" for i, c in enumerate(chunks)
    )
    system_prompt = (
        "You answer questions using ONLY the provided source excerpts. "
        "Cite sources inline like [Source 1]. If the excerpts don't contain "
        "the answer, say so plainly rather than guessing."
    )
    user_message = f"Sources:\n\n{context_block}\n\nQuestion: {question}"

    response = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
