"""The shipped receiver script must stay compatible with the server's real signing scheme."""
import importlib.util
from pathlib import Path

from app.services.webhooks import sign_payload

_spec = importlib.util.spec_from_file_location(
    "webhook_receiver", Path(__file__).resolve().parent.parent / "scripts" / "webhook_receiver.py"
)
receiver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(receiver)

SECRET, BODY = "whsec_demo", b'{"type":"document.ready"}'


def test_script_accepts_what_the_server_signs():
    assert receiver.verify(SECRET, BODY, sign_payload(SECRET, BODY, timestamp=1_800_000_000), now=1_800_000_005)


def test_script_rejects_tampering_wrong_secrets_and_replays():
    header = sign_payload(SECRET, BODY, timestamp=1_800_000_000)
    assert not receiver.verify(SECRET, BODY + b"x", header, now=1_800_000_005)
    assert not receiver.verify("whsec_other", BODY, header, now=1_800_000_005)
    assert not receiver.verify(SECRET, BODY, header, now=1_800_000_000 + 10_000)
