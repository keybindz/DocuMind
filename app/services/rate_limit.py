"""
Token-bucket rate limiting, keyed per tenant.

Token bucket (vs fixed window) allows short bursts up to `capacity` while
enforcing a sustained average of `refill_per_second`, and has no
boundary-doubling problem at window edges.

RedisTokenBucket runs the whole check-and-decrement inside one Lua script,
so it is atomic across any number of API replicas -- no read-modify-write
race -- and it uses Redis's own clock (TIME) so replicas with skewed clocks
agree. InMemoryTokenBucket has the same semantics with an injectable clock,
for tests and single-process dev.

Failure mode: if Redis is unreachable, the limiter FAILS OPEN (requests are
allowed, a warning is logged). Rate limiting is a protection layer, not a
correctness layer; taking the whole API down because the limiter's backing
store blipped is the worse trade. Quotas (document counts) live in Postgres
and are unaffected.
"""
import abc
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_seconds: float
    remaining: float


class RateLimiter(abc.ABC):
    @abc.abstractmethod
    def check(self, key: str, *, capacity: int, refill_per_second: float, cost: int = 1) -> RateLimitResult:
        ...


_LUA_TOKEN_BUCKET = """
local capacity = tonumber(ARGV[1])
local rate     = tonumber(ARGV[2])
local cost     = tonumber(ARGV[3])
local t   = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local data   = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts     = tonumber(data[2])
if tokens == nil then tokens = capacity; ts = now end
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry = (cost - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / rate) * 2)
return {allowed, tostring(tokens), tostring(retry)}
"""


class RedisTokenBucket(RateLimiter):
    def __init__(self, url: str | None = None, client=None):
        import redis  # local import: only needed when this backend is used

        self._redis_errors = redis.RedisError
        self._client = client or redis.Redis.from_url(url or settings.REDIS_URL, socket_timeout=0.25)
        self._script = self._client.register_script(_LUA_TOKEN_BUCKET)

    def check(self, key: str, *, capacity: int, refill_per_second: float, cost: int = 1) -> RateLimitResult:
        try:
            allowed, tokens, retry = self._script(keys=[key], args=[capacity, refill_per_second, cost])
        except self._redis_errors:
            logger.warning("Rate limiter backend unavailable; failing open", exc_info=True)
            return RateLimitResult(allowed=True, retry_after_seconds=0.0, remaining=float(capacity))
        return RateLimitResult(
            allowed=bool(allowed),
            retry_after_seconds=float(retry),
            remaining=float(tokens),
        )


class InMemoryTokenBucket(RateLimiter):
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    def check(self, key: str, *, capacity: int, refill_per_second: float, cost: int = 1) -> RateLimitResult:
        now = self._clock()
        tokens, ts = self._buckets.get(key, (float(capacity), now))
        tokens = min(float(capacity), tokens + max(0.0, now - ts) * refill_per_second)
        if tokens >= cost:
            self._buckets[key] = (tokens - cost, now)
            return RateLimitResult(True, 0.0, tokens - cost)
        self._buckets[key] = (tokens, now)
        return RateLimitResult(False, (cost - tokens) / refill_per_second, tokens)


_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        if settings.RATE_LIMIT_BACKEND == "redis":
            _limiter = RedisTokenBucket()
        elif settings.RATE_LIMIT_BACKEND == "memory":
            _limiter = InMemoryTokenBucket()
        else:
            raise ValueError(f"Unknown RATE_LIMIT_BACKEND: {settings.RATE_LIMIT_BACKEND}")
    return _limiter
