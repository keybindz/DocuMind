"""
Webhook signature scheme: `X-DocuMind-Signature: t=<unix>,v1=<hex>` where
v1 = HMAC-SHA256(secret, f"{t}.{body}"). These tests pin the wire format --
it's a public contract that customers' receivers implement.
"""
import hashlib
import hmac

import pytest

from app.services.webhooks import sign_payload, verify_signature

SECRET = "whsec_test_secret"
BODY = b'{"type":"document.ready"}'
NOW = 1_800_000_000


def test_known_answer_vector_matches_an_independent_computation():
    header = sign_payload(SECRET, BODY, timestamp=NOW)
    expected = hmac.new(SECRET.encode(), f"{NOW}.".encode() + BODY, hashlib.sha256).hexdigest()
    assert header == f"t={NOW},v1={expected}"


def test_valid_signature_verifies():
    header = sign_payload(SECRET, BODY, timestamp=NOW)
    assert verify_signature(SECRET, BODY, header, now=NOW + 10) is True


def test_tampered_body_is_rejected():
    header = sign_payload(SECRET, BODY, timestamp=NOW)
    assert verify_signature(SECRET, BODY + b" ", header, now=NOW) is False
    assert verify_signature(SECRET, b'{"type":"document.failed"}', header, now=NOW) is False


def test_wrong_secret_is_rejected():
    header = sign_payload(SECRET, BODY, timestamp=NOW)
    assert verify_signature("whsec_other", BODY, header, now=NOW) is False


def test_replay_of_an_old_signed_request_is_rejected():
    header = sign_payload(SECRET, BODY, timestamp=NOW)
    assert verify_signature(SECRET, BODY, header, now=NOW + 299) is True
    assert verify_signature(SECRET, BODY, header, now=NOW + 301) is False  # outside the 300s window


def test_timestamp_from_the_future_is_rejected():
    header = sign_payload(SECRET, BODY, timestamp=NOW + 3600)
    assert verify_signature(SECRET, BODY, header, now=NOW) is False


def test_timestamp_is_covered_by_the_signature():
    """An attacker can't make an old payload look fresh by editing t=."""
    old = sign_payload(SECRET, BODY, timestamp=NOW - 10_000)
    mac = old.split("v1=")[1]
    forged = f"t={NOW},v1={mac}"
    assert verify_signature(SECRET, BODY, forged, now=NOW) is False


@pytest.mark.parametrize(
    "header",
    ["", "garbage", "t=abc,v1=deadbeef", "t=1800000000", "v1=deadbeef", "t=,v1=", "t=1800000000,v1=", ",,,"],
)
def test_malformed_headers_never_verify_and_never_raise(header):
    assert verify_signature(SECRET, BODY, header, now=NOW) is False


def test_signing_defaults_to_the_current_time():
    import time

    header = sign_payload(SECRET, BODY)
    assert abs(int(header.split(",")[0][2:]) - time.time()) < 5
