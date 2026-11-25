import json
import socket

import pytest

from app.core import crypto, plans
from app.config import get_settings
from app.models import Webhook, WebhookDelivery
from app.services import webhooks as webhooks_service
from app.services.webhooks import SIGNATURE_HEADER, HttpxSender, SendResult, verify_signature
from tests.helpers import ReceiverServer

DEAD_URL = "http://127.0.0.1:9/hook"  # never actually contacted: tests use a fake sender


def _signup(client, name="Alpha"):
    r = client.post(
        "/v1/auth/signup",
        json={"tenant_name": name, "email": f"o@{name.lower()}.io", "password": "supersecret123"},
    )
    return {"Authorization": f"Bearer {r.json()['token']['access_token']}"}


def _register(client, headers, url=DEAD_URL, events=None):
    body = {"url": url}
    if events is not None:
        body["events"] = events
    r = client.post("/v1/webhooks", headers=headers, json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _upload(client, headers, name="a.txt", text="webhook test content", mime="text/plain"):
    r = client.post("/v1/documents", headers=headers, files={"file": (name, text.encode(), mime)})
    assert r.status_code in (200, 202), r.text
    return r.json()


def _deliveries(client, headers, hook_id):
    return client.get(f"/v1/webhooks/{hook_id}/deliveries", headers=headers).json()


# ------------------------------------------------------------ registration

def test_create_returns_the_secret_once_and_stores_it_encrypted(client, db_session):
    headers = _signup(client)
    created = _register(client, headers)

    assert created["secret"].startswith("whsec_") and len(created["secret"]) > 30
    listed = client.get("/v1/webhooks", headers=headers).json()
    assert len(listed) == 1 and "secret" not in listed[0]  # never shown again

    row = db_session.query(Webhook).one()
    assert row.secret_encrypted != created["secret"]  # not plaintext at rest
    assert crypto.decrypt(row.secret_encrypted) == created["secret"]


def test_each_webhook_gets_its_own_secret(client):
    headers = _signup(client)
    assert _register(client, headers)["secret"] != _register(client, headers)["secret"]


@pytest.mark.parametrize("events", [["bogus.event"], [], ["document.ready", "nope"]])
def test_invalid_event_lists_are_rejected(client, events):
    headers = _signup(client)
    r = client.post("/v1/webhooks", headers=headers, json={"url": DEAD_URL, "events": events})
    assert r.status_code == 422


def test_duplicate_events_are_collapsed(client):
    headers = _signup(client)
    created = _register(client, headers, events=["document.ready", "document.ready"])
    assert created["events"] == ["document.ready"]


def test_webhook_count_is_limited_by_plan(client, db_session):
    from app.models import Tenant

    headers = _signup(client)
    for _ in range(plans.PLANS["free"].max_webhooks):
        _register(client, headers)
    blocked = client.post("/v1/webhooks", headers=headers, json={"url": DEAD_URL})
    assert blocked.status_code == 403

    db_session.query(Tenant).update({"plan": "pro"})
    db_session.commit()
    assert client.post("/v1/webhooks", headers=headers, json={"url": DEAD_URL}).status_code == 201


# ---------------------------------------------------------------- SSRF

@pytest.fixture()
def strict_urls(monkeypatch):
    """Production URL policy, with DNS answered from a table so no network is needed."""
    monkeypatch.setattr(get_settings(), "WEBHOOK_ALLOW_PRIVATE_URLS", False)
    dns = {
        "hooks.example.com": ["93.184.216.34"],
        "internal.example.com": ["10.1.2.3"],
        "split.example.com": ["93.184.216.34", "10.1.2.3"],  # one public + one private answer
    }

    def fake_resolve(host):
        if host in dns:
            return dns[host]
        if host == "nxdomain.example.com":
            raise socket.gaierror("no such host")
        return [host.strip("[]")] if host[0].isdigit() or ":" in host else ["127.0.0.1"]  # literals, localhost

    monkeypatch.setattr(webhooks_service, "_resolve", fake_resolve)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/hook",  # loopback
        "https://localhost/hook",
        "https://169.254.169.254/latest/meta-data/",  # cloud metadata service
        "https://10.0.0.5/hook",  # RFC1918
        "https://172.16.0.9/hook",
        "https://192.168.1.10/hook",
        "https://100.64.0.1/hook",  # carrier-grade NAT
        "https://0.0.0.0/hook",
        "https://[::1]/hook",  # IPv6 loopback
        "https://[fd00::1]/hook",  # IPv6 unique-local
        "https://internal.example.com/hook",  # public-looking name resolving to a private IP
        "https://split.example.com/hook",  # ANY private answer is enough to reject
        "https://nxdomain.example.com/hook",  # unresolvable
        "http://hooks.example.com/hook",  # plain http
        "ftp://hooks.example.com/hook",
        "file:///etc/passwd",
        "https://user:pass@hooks.example.com/hook",  # credentials in the URL
        "https:///nohost",
        "not a url",
    ],
)
def test_ssrf_unsafe_urls_are_rejected(client, strict_urls, url):
    headers = _signup(client)
    r = client.post("/v1/webhooks", headers=headers, json={"url": url})
    assert r.status_code == 422, f"{url} was accepted"


def test_a_public_https_url_is_accepted(client, strict_urls):
    headers = _signup(client)
    assert client.post("/v1/webhooks", headers=headers, json={"url": "https://hooks.example.com/x"}).status_code == 201


def test_url_is_revalidated_at_delivery_time(client, webhook_sender, monkeypatch, task_queue):
    """DNS can change after registration: a hook that was fine then must not be called if it's unsafe now."""
    headers = _signup(client)
    hook = _register(client, headers)  # accepted while private URLs were allowed
    monkeypatch.setattr(get_settings(), "WEBHOOK_ALLOW_PRIVATE_URLS", False)

    _upload(client, headers)

    assert webhook_sender.calls == []  # nothing was sent
    delivery = _deliveries(client, headers, hook["id"])["items"][0]
    assert delivery["status"] == "failed"
    assert "rejected at delivery time" in delivery["last_error"]
    assert task_queue.retry_delays == []  # an unsafe URL is not retried


# ------------------------------------------------------------- delivery

def test_document_ready_event_is_delivered_signed(client, webhook_sender):
    headers = _signup(client)
    hook = _register(client, headers)
    doc = _upload(client, headers, text="hello webhook")

    assert len(webhook_sender.calls) == 1
    call = webhook_sender.calls[0]
    assert call.url == DEAD_URL
    assert call.headers["X-DocuMind-Event"] == "document.ready"
    assert verify_signature(hook["secret"], call.body, call.headers[SIGNATURE_HEADER])  # a receiver can verify it
    assert not verify_signature("whsec_wrong", call.body, call.headers[SIGNATURE_HEADER])

    payload = json.loads(call.body)
    assert payload["type"] == "document.ready"
    assert payload["data"]["document"]["id"] == doc["id"]
    assert payload["data"]["document"]["status"] == "ready"


def test_document_failed_event_is_delivered_with_the_error(client, webhook_sender):
    headers = _signup(client)
    _register(client, headers)
    _upload(client, headers, name="bad.pdf", text="not a pdf", mime="application/pdf")

    (call,) = webhook_sender.calls
    payload = json.loads(call.body)
    assert call.headers["X-DocuMind-Event"] == "document.failed"
    assert payload["data"]["document"]["status"] == "failed"
    assert payload["data"]["document"]["error_message"]


def test_webhooks_only_receive_events_they_subscribed_to(client, webhook_sender):
    headers = _signup(client)
    _register(client, headers, events=["document.failed"])

    _upload(client, headers, name="good.txt", text="fine")
    assert webhook_sender.calls == []  # a success event must not reach a failure-only hook

    _upload(client, headers, name="bad.pdf", text="not a pdf", mime="application/pdf")
    assert [c.headers["X-DocuMind-Event"] for c in webhook_sender.calls] == ["document.failed"]


def test_every_subscribed_webhook_gets_its_own_delivery(client, webhook_sender):
    headers = _signup(client)
    _register(client, headers)
    _register(client, headers)
    _upload(client, headers)
    assert len(webhook_sender.calls) == 2


def test_failed_deliveries_are_retried_with_backoff_until_one_succeeds(client, webhook_sender, task_queue):
    headers = _signup(client)
    hook = _register(client, headers)
    webhook_sender.script(SendResult(500), SendResult(None, "ReadTimeout: slow receiver"), SendResult(200))
    _upload(client, headers)

    delivery = _deliveries(client, headers, hook["id"])["items"][0]
    assert delivery["status"] == "succeeded"
    assert delivery["attempts"] == 3
    assert delivery["last_status_code"] == 200
    assert delivery["delivered_at"] is not None
    assert task_queue.retry_delays == [10, 20]  # exponential: 10s, 20s
    # Receivers dedupe on this id, so it must be stable across every retry.
    assert len({c.headers["X-DocuMind-Delivery"] for c in webhook_sender.calls}) == 1


def test_delivery_gives_up_after_max_attempts(client, webhook_sender, task_queue):
    headers = _signup(client)
    hook = _register(client, headers)
    webhook_sender.default = SendResult(503)
    _upload(client, headers)

    delivery = _deliveries(client, headers, hook["id"])["items"][0]
    assert delivery["status"] == "failed"
    assert delivery["attempts"] == 5
    assert delivery["last_status_code"] == 503
    assert "Gave up after 5 attempts" in delivery["last_error"]
    assert len(webhook_sender.calls) == 5
    assert task_queue.retry_delays == [10, 20, 40, 80]


def test_a_dead_receiver_never_breaks_document_processing(client, webhook_sender):
    headers = _signup(client)
    _register(client, headers)
    webhook_sender.default = SendResult(None, "ConnectError: refused")
    doc = _upload(client, headers)
    assert client.get(f"/v1/documents/{doc['id']}", headers=headers).json()["status"] == "ready"


def test_ping_sends_a_signed_test_event(client, webhook_sender):
    headers = _signup(client)
    hook = _register(client, headers)
    r = client.post(f"/v1/webhooks/{hook['id']}/ping", headers=headers)

    assert r.status_code == 202
    (call,) = webhook_sender.calls
    assert call.headers["X-DocuMind-Event"] == "webhook.ping"
    assert verify_signature(hook["secret"], call.body, call.headers[SIGNATURE_HEADER])
    assert _deliveries(client, headers, hook["id"])["items"][0]["status"] == "succeeded"


@pytest.mark.async_queue
def test_deleting_a_webhook_cancels_its_pending_deliveries(client, webhook_sender, task_queue):
    headers = _signup(client)
    hook = _register(client, headers)
    _upload(client, headers)
    task_queue.drain(max_jobs=1)  # run ingestion only: the delivery is now recorded + queued, not sent
    assert _deliveries(client, headers, hook["id"])["total"] == 1

    assert client.delete(f"/v1/webhooks/{hook['id']}", headers=headers).status_code == 204
    task_queue.drain()

    assert webhook_sender.calls == []  # the queued delivery found nothing and no-oped


def test_delivery_history_is_paginated(client, webhook_sender):
    headers = _signup(client)
    hook = _register(client, headers)
    for i in range(3):
        _upload(client, headers, name=f"{i}.txt", text=f"document number {i}")
    page = client.get(f"/v1/webhooks/{hook['id']}/deliveries?limit=2", headers=headers).json()
    assert page["total"] == 3 and len(page["items"]) == 2


# ---------------------------------------------------------- tenant isolation

def test_webhooks_are_invisible_and_untouchable_across_tenants(client, webhook_sender):
    a, b = _signup(client, "Alpha"), _signup(client, "Beta")
    hook = _register(client, a)

    assert client.get("/v1/webhooks", headers=b).json() == []
    assert client.delete(f"/v1/webhooks/{hook['id']}", headers=b).status_code == 404
    assert client.post(f"/v1/webhooks/{hook['id']}/ping", headers=b).status_code == 404
    assert client.get(f"/v1/webhooks/{hook['id']}/deliveries", headers=b).status_code == 404
    assert len(client.get("/v1/webhooks", headers=a).json()) == 1  # B's attempted delete did nothing


def test_one_tenants_events_never_reach_another_tenants_webhook(client, webhook_sender):
    a, b = _signup(client, "Alpha"), _signup(client, "Beta")
    _register(client, a)
    _upload(client, b, text="beta's private document")  # an event in tenant B...
    assert webhook_sender.calls == []  # ...must not be delivered to tenant A's endpoint


# ------------------------------------------------------ real HTTP, real client

@pytest.fixture()
def receiver():
    servers = []

    def make(**kwargs):
        s = ReceiverServer(**kwargs)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def test_real_http_delivery_is_signed_and_verifiable(client, backends, receiver):
    backends.sender_holder.sender = HttpxSender()  # the production sender, not the fake
    server = receiver()
    headers = _signup(client)
    hook = _register(client, headers, url=server.url)
    doc = _upload(client, headers)

    (req,) = server.requests
    assert req.headers["x-documind-event"] == "document.ready"
    assert req.headers["content-type"] == "application/json"
    assert verify_signature(hook["secret"], req.body, req.headers["x-documind-signature"])
    assert json.loads(req.body)["data"]["document"]["id"] == doc["id"]
    assert _deliveries(client, headers, hook["id"])["items"][0]["status"] == "succeeded"


def test_redirects_are_never_followed(client, backends, receiver):
    """A 302 to an internal address must not be followed: it would sidestep the URL validation."""
    backends.sender_holder.sender = HttpxSender()
    internal_target = receiver()
    redirector = receiver(redirect_to=internal_target.url)
    headers = _signup(client)
    hook = _register(client, headers, url=redirector.url)
    _upload(client, headers)

    assert internal_target.requests == []  # the redirect target was never contacted
    assert len(redirector.requests) == 5  # a 302 counts as a failed attempt and is retried
    delivery = _deliveries(client, headers, hook["id"])["items"][0]
    assert delivery["status"] == "failed" and delivery["last_status_code"] == 302


def test_unreachable_receiver_is_reported_not_raised(client, backends):
    backends.sender_holder.sender = HttpxSender()
    headers = _signup(client)
    hook = _register(client, headers, url="http://127.0.0.1:1/hook")  # nothing listens on port 1
    _upload(client, headers)
    delivery = _deliveries(client, headers, hook["id"])["items"][0]
    assert delivery["status"] == "failed"
    assert delivery["last_status_code"] is None and delivery["last_error"]
