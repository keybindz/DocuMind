"""
Webhooks: signed event callbacks to tenant-registered URLs.

Three things here are easy to get wrong, so each is deliberate:

1. SIGNING. Every delivery carries `X-DocuMind-Signature: t=<unix>,v1=<hex>`
   where v1 = HMAC-SHA256(secret, f"{t}.{raw_body}"). Signing the timestamp
   *together with* the body lets receivers reject replays of an old, validly
   signed request (`verify_signature` enforces a tolerance window). Comparison
   is constant-time.

2. SSRF. A webhook URL is attacker-controlled input that makes OUR servers
   issue requests. Without checks, a tenant could register
   http://169.254.169.254/ (cloud metadata) or http://postgres:5432 and use us
   as a proxy into the private network. We require every resolved address to be
   globally routable, forbid credentials in URLs, never follow redirects, and
   re-check at delivery time (DNS can change after registration). Residual
   risk: a DNS-rebinding race between our check and the connection; the full
   fix is pinning the resolved IP or an egress proxy -- noted for Phase 5.

3. OUTBOX. `emit_document_event` inserts delivery rows inside the SAME DB
   transaction as the document's status change. Either both happen or neither
   does, so a crash can't leave a "ready" document whose webhook was never
   recorded. The queue message that later triggers delivery is just a nudge;
   the row is the source of truth.
"""
import abc
import hashlib
import hmac
import ipaddress
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import crypto
from app.models import Document, Webhook, WebhookDelivery
from app.services.jobs import JobResult, exponential_backoff

logger = logging.getLogger(__name__)
settings = get_settings()

SIGNATURE_HEADER = "X-DocuMind-Signature"
DOCUMENT_EVENTS = ("document.ready", "document.failed")
ALL_SUBSCRIBABLE_EVENTS = DOCUMENT_EVENTS
PING_EVENT = "webhook.ping"


# ---------------------------------------------------------------- signing

def sign_payload(secret: str, body: bytes, timestamp: int | None = None) -> str:
    ts = int(time.time()) if timestamp is None else timestamp
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify_signature(
    secret: str,
    body: bytes,
    header: str,
    *,
    tolerance_seconds: int | None = None,
    now: float | None = None,
) -> bool:
    """What a webhook *receiver* runs. Returns False on any malformation or mismatch."""
    tolerance = settings.WEBHOOK_SIGNATURE_TOLERANCE_SECONDS if tolerance_seconds is None else tolerance_seconds
    try:
        parts = dict(item.split("=", 1) for item in header.split(","))
        ts = int(parts["t"])
        provided = parts["v1"]
    except (ValueError, KeyError):
        return False
    if abs((time.time() if now is None else now) - ts) > tolerance:
        return False  # too old (replay) or from the future
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided)


# ------------------------------------------------------------------- SSRF

class WebhookURLError(ValueError):
    pass


def _resolve(host: str) -> list[str]:
    """Indirection so tests can supply DNS answers without a network."""
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def validate_webhook_url(url: str) -> None:
    allow_private = settings.WEBHOOK_ALLOW_PRIVATE_URLS
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError as exc:
        raise WebhookURLError("Malformed URL") from exc

    allowed_schemes = ("https", "http") if allow_private else ("https",)
    if parsed.scheme not in allowed_schemes:
        raise WebhookURLError("Webhook URL must use https")
    if not host:
        raise WebhookURLError("Webhook URL must include a host")
    if parsed.username or parsed.password:
        raise WebhookURLError("Credentials in webhook URLs are not allowed")
    if allow_private:
        return  # local-dev escape hatch (see config.WEBHOOK_ALLOW_PRIVATE_URLS)

    try:
        addresses = _resolve(host)
    except OSError as exc:
        raise WebhookURLError(f"Could not resolve host '{host}'") from exc
    if not addresses:
        raise WebhookURLError(f"Could not resolve host '{host}'")
    for address in addresses:
        # is_global is False for loopback, private, link-local (169.254.x.x
        # cloud metadata), CGNAT, multicast, reserved, unspecified...
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            raise WebhookURLError("Webhook URL must not resolve to a private or internal address")


# ---------------------------------------------------------------- sending

@dataclass
class SendResult:
    status_code: int | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300


class WebhookSender(abc.ABC):
    @abc.abstractmethod
    def send(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> SendResult:
        ...


class HttpxSender(WebhookSender):
    def send(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> SendResult:
        import httpx

        try:
            # follow_redirects=False: a 302 to http://169.254.169.254/ would
            # sidestep the URL validation above.
            response = httpx.post(url, content=body, headers=headers, timeout=timeout, follow_redirects=False)
            return SendResult(status_code=response.status_code)
        except httpx.HTTPError as exc:
            return SendResult(status_code=None, error=f"{type(exc).__name__}: {exc}"[:500])


_sender: WebhookSender | None = None


def get_webhook_sender() -> WebhookSender:
    global _sender
    if _sender is None:
        _sender = HttpxSender()
    return _sender


# ----------------------------------------------------------------- outbox

def _document_payload(document: Document, event_type: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "document": {
                "id": str(document.id),
                "filename": document.filename,
                "status": document.status.value,
                "error_message": document.error_message,
                "char_count": document.char_count,
                "attempts": document.attempts,
            }
        },
    }


def emit_document_event(db: Session, document: Document, event_type: str) -> list[uuid.UUID]:
    """
    Records a pending delivery for every active webhook subscribed to
    `event_type`. Runs inside the caller's transaction (outbox pattern); the
    caller commits, then enqueues the returned delivery ids.
    """
    hooks = db.query(Webhook).filter(Webhook.tenant_id == document.tenant_id, Webhook.is_active.is_(True)).all()
    payload = _document_payload(document, event_type)
    ids: list[uuid.UUID] = []
    for hook in hooks:
        if event_type not in (hook.events or []):
            continue
        delivery = WebhookDelivery(
            id=uuid.uuid4(),
            tenant_id=document.tenant_id,
            webhook_id=hook.id,
            event_type=event_type,
            payload=payload,
            status="pending",
        )
        db.add(delivery)
        ids.append(delivery.id)
    return ids


def create_ping_delivery(db: Session, webhook: Webhook) -> WebhookDelivery:
    delivery = WebhookDelivery(
        id=uuid.uuid4(),
        tenant_id=webhook.tenant_id,
        webhook_id=webhook.id,
        event_type=PING_EVENT,
        payload={
            "id": str(uuid.uuid4()),
            "type": PING_EVENT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "Test event from DocuMind"},
        },
        status="pending",
    )
    db.add(delivery)
    return delivery


# --------------------------------------------------------------- delivery

def attempt_delivery(
    db: Session,
    *,
    delivery_id: uuid.UUID,
    tenant_id: uuid.UUID,
    attempt: int,
    max_attempts: int,
    sender: WebhookSender,
) -> JobResult:
    delivery = (
        db.query(WebhookDelivery)
        .filter(WebhookDelivery.id == delivery_id, WebhookDelivery.tenant_id == tenant_id)
        .first()
    )
    if delivery is None or delivery.status != "pending":
        return JobResult()  # deleted, or already finished (duplicate queue message)

    hook = db.get(Webhook, delivery.webhook_id)
    if hook is None or not hook.is_active:
        delivery.status = "failed"
        delivery.last_error = "Webhook was deleted or disabled"
        db.commit()
        return JobResult()

    def finish_failed(error: str, status_code: int | None = None) -> JobResult:
        delivery.attempts = attempt
        delivery.status = "failed"
        delivery.last_status_code = status_code
        delivery.last_error = error[:1000]
        db.commit()
        return JobResult()

    try:
        validate_webhook_url(hook.url)  # re-check: DNS may have changed since registration
    except WebhookURLError as exc:
        return finish_failed(f"URL rejected at delivery time: {exc}")

    body = json.dumps(delivery.payload, separators=(",", ":"), sort_keys=True).encode()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "DocuMind-Webhooks/1.0",
        "X-DocuMind-Event": delivery.event_type,
        "X-DocuMind-Delivery": str(delivery.id),  # stable across retries: receivers dedupe on this
        SIGNATURE_HEADER: sign_payload(crypto.decrypt(hook.secret_encrypted), body),
    }
    result = sender.send(hook.url, body, headers, settings.WEBHOOK_TIMEOUT_SECONDS)

    delivery.attempts = attempt
    delivery.last_status_code = result.status_code
    if result.ok:
        delivery.status = "succeeded"
        delivery.last_error = None
        delivery.delivered_at = datetime.now(timezone.utc)
        db.commit()
        return JobResult()

    error = result.error or f"Receiver responded with HTTP {result.status_code}"
    if attempt >= max_attempts:
        return finish_failed(f"Gave up after {attempt} attempts: {error}", result.status_code)
    delivery.last_error = error[:1000]
    db.commit()
    return JobResult(retry_in=exponential_backoff(attempt, base=10, cap=3600))
