"""
Text extraction service.

Runs synchronously in-process for Phase 1 since files are small and this is
a single-node demo. This is intentionally the seam where Phase 4 introduces
a message queue + worker: `extract_text()` itself won't need to change, only
*who calls it and when* (an API request handler now, a Celery task later).
"""
import io

import pypdf
from docx import Document as DocxDocument


class ExtractionError(Exception):
    pass


def extract_text(content: bytes, mime_type: str) -> str:
    try:
        if mime_type == "application/pdf":
            return _extract_pdf(content)
        if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return _extract_docx(content)
        if mime_type in ("text/plain", "text/markdown"):
            return content.decode("utf-8", errors="replace")
        raise ExtractionError(f"Unsupported mime type: {mime_type}")
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert any parser failure to a domain error
        raise ExtractionError(f"Failed to extract text: {exc}") from exc


def _extract_pdf(content: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ExtractionError("No extractable text found (file may be a scanned/image-only PDF)")
    return text


def _extract_docx(content: bytes) -> str:
    doc = DocxDocument(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs).strip()
    if not text:
        raise ExtractionError("No extractable text found in document")
    return text
