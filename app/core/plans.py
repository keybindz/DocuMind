"""Per-plan limits. Single source of truth for quotas and rate limits."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketSpec:
    capacity: int          # max burst
    refill_per_minute: int  # sustained rate


@dataclass(frozen=True)
class PlanLimits:
    max_documents: int
    query_bucket: BucketSpec
    upload_bucket: BucketSpec
    max_webhooks: int = 2


PLANS: dict[str, PlanLimits] = {
    "free": PlanLimits(25, BucketSpec(10, 10), BucketSpec(5, 5), max_webhooks=2),
    "pro": PlanLimits(1_000, BucketSpec(60, 60), BucketSpec(30, 30), max_webhooks=10),
    "enterprise": PlanLimits(50_000, BucketSpec(300, 300), BucketSpec(120, 120), max_webhooks=50),
}


def limits_for(plan: str) -> PlanLimits:
    # Unknown plan strings fall back to the most restrictive tier rather than
    # failing open.
    return PLANS.get(plan, PLANS["free"])
