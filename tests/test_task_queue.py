import time
import uuid

import pytest

from app.services.jobs import JobResult
from app.services.task_queue import CeleryTaskQueue, InMemoryTaskQueue, QueueUnavailable


def _queue(eager=False, ingest=None):
    calls = []

    def default_ingest(doc, tenant, attempt):
        calls.append(("ingest", attempt))
        return JobResult()

    return (
        InMemoryTaskQueue(
            ingest_handler=ingest or default_ingest,
            webhook_handler=lambda *a: JobResult(),
            eager=eager,
        ),
        calls,
    )


def test_lazy_queue_holds_jobs_until_drained():
    q, calls = _queue(eager=False)
    q.enqueue_ingestion(uuid.uuid4(), uuid.uuid4())
    assert calls == [] and len(q.jobs) == 1
    assert q.drain() == 1
    assert calls == [("ingest", 1)]


def test_eager_queue_runs_jobs_immediately():
    q, calls = _queue(eager=True)
    q.enqueue_ingestion(uuid.uuid4(), uuid.uuid4())
    assert calls == [("ingest", 1)] and not q.jobs


def test_retry_requests_are_requeued_with_an_incremented_attempt():
    seen = []

    def flaky(doc, tenant, attempt):
        seen.append(attempt)
        return JobResult(retry_in=3.0) if attempt < 3 else JobResult()

    q, _ = _queue(ingest=flaky)
    q.enqueue_ingestion(uuid.uuid4(), uuid.uuid4())
    q.drain()
    assert seen == [1, 2, 3]
    assert q.retry_delays == [3.0, 3.0]


def test_simulated_outage_raises_queue_unavailable_and_enqueues_nothing():
    q, _ = _queue()
    q.fail_enqueue = True
    with pytest.raises(QueueUnavailable):
        q.enqueue_ingestion(uuid.uuid4(), uuid.uuid4())
    assert not q.jobs


def test_celery_publisher_fails_fast_when_the_broker_is_down():
    """No Redis needed: nothing listens on port 1. The API must get a quick error (-> 503), not hang."""
    queue = CeleryTaskQueue(broker_url="redis://127.0.0.1:1/0")
    started = time.monotonic()
    with pytest.raises(QueueUnavailable):
        queue.enqueue_ingestion(uuid.uuid4(), uuid.uuid4())
    assert time.monotonic() - started < 15
