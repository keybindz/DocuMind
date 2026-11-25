"""Types shared by background jobs (ingestion, webhook delivery)."""
import uuid
from dataclasses import dataclass, field


class TransientError(Exception):
    """A failure worth retrying (network blip, backend briefly down)."""


@dataclass
class JobResult:
    """
    What a job handler tells its runner. The handler decides *whether* to
    retry and *how long* to wait; the runner (Celery, or the in-memory test
    queue) only carries that out. Keeping the policy in plain Python is what
    lets it be unit-tested without a broker.
    """

    retry_in: float | None = None  # seconds; None means the job is finished
    delivery_ids: list[uuid.UUID] = field(default_factory=list)  # webhook deliveries to enqueue

    @property
    def done(self) -> bool:
        return self.retry_in is None


def exponential_backoff(attempt: int, *, base: float, cap: float) -> float:
    """attempt is 1-based: the delay to wait AFTER that attempt failed."""
    return min(base * (2 ** (attempt - 1)), cap)
