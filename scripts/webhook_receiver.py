#!/usr/bin/env python3
"""
Minimal webhook receiver for trying DocuMind webhooks locally. Standard library only.

    WEBHOOK_SECRET=whsec_... python scripts/webhook_receiver.py          # listens on :9000

Then register  http://host.docker.internal:9000/hook  as a webhook (this works
from inside the docker-compose containers) and upload a document.

The `verify()` function below is the reference implementation of signature
checking -- this is what a real receiver must do before trusting a request:

  1. Parse `X-DocuMind-Signature: t=<unix>,v1=<hex>`.
  2. Recompute HMAC-SHA256 over  f"{t}.{raw_body}"  with your signing secret.
  3. Compare in constant time, and reject timestamps outside a tolerance
     window (this is what defeats replay of a captured request).
  4. Use `X-DocuMind-Delivery` to deduplicate: deliveries are retried, so the
     same event can arrive more than once with the same delivery id.
"""
import hashlib
import hmac
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

TOLERANCE_SECONDS = 300
_seen_deliveries: set[str] = set()  # use a durable store (Redis/DB) in a real receiver


def verify(secret: str, body: bytes, header: str, now: float | None = None) -> bool:
    try:
        parts = dict(item.split("=", 1) for item in header.split(","))
        timestamp, provided = int(parts["t"]), parts["v1"]
    except (ValueError, KeyError):
        return False
    if abs((time.time() if now is None else now) - timestamp) > TOLERANCE_SECONDS:
        return False
    expected = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided)


class Handler(BaseHTTPRequestHandler):
    secret = os.environ.get("WEBHOOK_SECRET", "")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self.secret:
            print("!! WEBHOOK_SECRET is not set; refusing to accept unverified events", file=sys.stderr)
            return self._reply(500)
        if not verify(self.secret, body, self.headers.get("X-DocuMind-Signature", "")):
            print("!! rejected request with an invalid or expired signature")
            return self._reply(401)

        delivery_id = self.headers.get("X-DocuMind-Delivery", "")
        duplicate = delivery_id in _seen_deliveries
        _seen_deliveries.add(delivery_id)
        event = json.loads(body)
        tag = " (duplicate delivery, ignoring)" if duplicate else ""
        print(f"OK  {event['type']}  delivery={delivery_id}{tag}")
        if not duplicate:
            print(json.dumps(event["data"], indent=2))
        self._reply(200)  # any 2xx tells DocuMind the delivery succeeded; anything else is retried

    def _reply(self, status: int):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9000"))
    print(f"Listening on :{port}  (secret {'set' if Handler.secret else 'NOT SET'})")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
