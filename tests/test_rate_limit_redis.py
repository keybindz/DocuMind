"""
RedisTokenBucket against a REAL Redis. Skipped unless REDIS_TEST_URL is set,
e.g. REDIS_TEST_URL=redis://localhost:6379/15 (use a scratch DB index).
"""
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

REDIS_URL = os.environ.get("REDIS_TEST_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="set REDIS_TEST_URL to run Redis limiter tests")


@pytest.fixture()
def limiter():
    from app.services.rate_limit import RedisTokenBucket

    return RedisTokenBucket(url=REDIS_URL)


@pytest.fixture()
def key():
    return f"test:{uuid.uuid4()}"


def test_burst_then_block_then_refill(limiter, key):
    results = [limiter.check(key, capacity=3, refill_per_second=10) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert 0 < results[3].retry_after_seconds <= 0.1 + 0.05

    time.sleep(0.25)  # ~2.5 tokens at 10/s
    assert limiter.check(key, capacity=3, refill_per_second=10).allowed is True


def test_keys_are_independent(limiter, key):
    other = f"{key}:other"
    assert limiter.check(key, capacity=1, refill_per_second=0.001).allowed
    assert not limiter.check(key, capacity=1, refill_per_second=0.001).allowed
    assert limiter.check(other, capacity=1, refill_per_second=0.001).allowed


def test_check_and_decrement_is_atomic_under_concurrency(limiter, key):
    """
    80 concurrent requests against a bucket of 25 (negligible refill) must
    admit EXACTLY 25. A read-then-write implementation would over-admit.
    """
    capacity, attempts = 25, 80
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: limiter.check(key, capacity=capacity, refill_per_second=0.0001), range(attempts)))
    assert sum(r.allowed for r in results) == capacity


def test_fails_open_when_redis_is_unreachable():
    from app.services.rate_limit import RedisTokenBucket

    dead = RedisTokenBucket(url="redis://127.0.0.1:1/0")  # nothing listens on port 1
    result = dead.check("anything", capacity=1, refill_per_second=1)
    assert result.allowed is True
