"""Shared test doubles and builders for the async pipeline."""
from collections import deque
from types import SimpleNamespace

from app.config import get_settings
from app.services.embeddings import DeterministicFakeBackend
from app.services.task_queue import InMemoryTaskQueue
from app.services.webhooks import SendResult, WebhookSender
from app.worker import handlers

settings = get_settings()


class FakeSender(WebhookSender):
    """Records every webhook delivery attempt; responses are scriptable."""

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.scripted: deque[SendResult] = deque()
        self.default = SendResult(status_code=200)

    def script(self, *results: SendResult) -> None:
        self.scripted.extend(results)

    def send(self, url, body, headers, timeout):
        self.calls.append(SimpleNamespace(url=url, body=body, headers=headers))
        return self.scripted.popleft() if self.scripted else self.default


class FlakyEmbedder(DeterministicFakeBackend):
    """Fails the first `failures` embed() calls, then behaves normally."""

    def __init__(self, failures: int):
        self.failures_left = failures
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.failures_left > 0:
            self.failures_left -= 1
            raise ConnectionError("embedding service unavailable")
        return super().embed(texts)


def build_queue(*, session_factory, storage, embedder, vector_store, sender_holder, eager: bool) -> InMemoryTaskQueue:
    """An in-memory queue running the REAL worker handlers against the given backends."""

    def ingest(document_id, tenant_id, attempt):
        return handlers.handle_ingestion(
            session_factory=session_factory,
            document_id=document_id,
            tenant_id=tenant_id,
            attempt=attempt,
            max_attempts=settings.INGESTION_MAX_RETRIES + 1,
            storage=storage,
            embedding_backend=embedder_holder_get(embedder),
            vector_store=vector_store,
        )

    def deliver(delivery_id, tenant_id, attempt):
        return handlers.handle_webhook_delivery(
            session_factory=session_factory,
            delivery_id=delivery_id,
            tenant_id=tenant_id,
            attempt=attempt,
            max_attempts=settings.WEBHOOK_MAX_ATTEMPTS,
            sender=sender_holder.sender,  # read at call time so tests can swap senders
        )

    return InMemoryTaskQueue(ingest_handler=ingest, webhook_handler=deliver, eager=eager)


def embedder_holder_get(embedder):
    # `embedder` may be a holder (so tests can swap in a FlakyEmbedder) or a backend.
    return getattr(embedder, "backend", embedder)


class HookEmbedder(DeterministicFakeBackend):
    """Calls `hook()` at the start of every embed(), i.e. while the job is mid-flight."""

    def __init__(self, hook):
        self.hook = hook

    def embed(self, texts):
        self.hook()
        return super().embed(texts)


import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class ReceiverServer:
    """A real HTTP server on 127.0.0.1 that records every POST it receives."""

    def __init__(self, status: int = 200, redirect_to: str | None = None):
        self.requests: list[SimpleNamespace] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                # Record EVERY method. A followed 302 turns a POST into a GET, and
                # a receiver that only logged POSTs would never notice it happened.
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append(
                    SimpleNamespace(
                        method=self.command,
                        path=self.path,
                        body=body,
                        headers={k.lower(): v for k, v in self.headers.items()},
                    )
                )
                self.send_response(302 if redirect_to else status)
                if redirect_to:
                    self.send_header("Location", redirect_to)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = do_GET = do_PUT = do_HEAD = _handle

            def log_message(self, *args):  # keep test output quiet
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}/hook"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()
