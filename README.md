# DocuMind — Phases 1–4

Multi-tenant document intelligence platform.

- **Phase 1:** tenant/user auth (JWT), document upload, synchronous text
  extraction (PDF, DOCX, TXT, MD).
- **Phase 2:** chunking, embeddings, a vector store, and a
  RAG query endpoint (`POST /v1/query`) with optional LLM answer synthesis.

- **Phase 3:** database-enforced tenant isolation (Postgres
  Row-Level Security), per-plan document quotas, and per-tenant Redis
  token-bucket rate limiting.

- **Phase 4 (this update):** asynchronous ingestion. Upload returns `202`
  immediately; a **Celery worker** extracts, chunks and embeds in the
  background, with retries, **idempotency keys**, and **signed webhooks**.

Still to come: observability, CI/CD and Kubernetes (Phase 5); hybrid search
and billing (Phase 6) — see "What's deliberately not here yet" below.

> **Upgrading from Phase 3?** Upload now returns **`202 Accepted`** (not
> `201`) with `status: "pending"`, and results appear once the worker
> finishes. Run `docker compose up --build` so the new `worker` service
> starts. The migration adds three tables and RLS policies for them.

> **Upgrading from Phase 2?** The Postgres container now creates a
> restricted `documind_app` role on *first* start only, so run
> `docker compose down -v` once to recreate the volume (this wipes dev data).

## Stack

- **FastAPI** — API layer
- **PostgreSQL** (via SQLAlchemy 2.0) — relational data: tenants, users, documents, chunks
- **Celery** — background worker (broker: Redis) for ingestion and webhook delivery
- **Redis** — Celery broker + per-tenant rate limiting (atomic Lua token bucket)
- **Qdrant** — vector store, for semantic search over chunk embeddings
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embedding model, no API key required
- **Anthropic API** (optional) — LLM answer synthesis over retrieved chunks
- **Alembic** — schema migrations
- **JWT** (python-jose) + **bcrypt** (passlib) — auth
- **pypdf / python-docx** — text extraction
- **Docker Compose** — local dev environment
- **pytest** — 193 tests; 161 need nothing running, 32 use real Postgres/Redis/Celery

## Architecture notes worth knowing before you read the code

- **Multi-tenancy is in the schema from day one.** Every tenant-owned table
  (`users`, `documents`) has a `tenant_id`, and every query is filtered by
  the caller's tenant. Row-Level Security as a second enforcement layer,
  plus explicit cross-tenant-leak tests, are added in Phase 3 — the schema
  won't need to change then.
- **Cross-dialect UUID type** (`app/database.py::GUID`): native `UUID` on
  Postgres, `CHAR(36)` on SQLite. This is what lets the test suite run fast
  against in-memory SQLite while production runs Postgres, from the exact
  same models.
- **Storage is behind an interface** (`app/services/storage.py`). Phase 1
  uses local disk; a later phase swaps in an S3/MinIO-backed implementation
  without touching the API or the schema — `storage_path` is already shaped
  like an S3 key (`{tenant_id}/{document_id}_{filename}`).
- **404, not 403, for cross-tenant access.** Looking up another tenant's
  document by ID returns 404 rather than 403 — it doesn't confirm the ID
  exists. See `test_tenant_isolation_cannot_see_other_tenants_documents`.
- **Embeddings and the vector store are behind swappable interfaces**, same
  pattern as storage (`app/services/embeddings.py`,
  `app/services/vector_store.py`). Production uses `sentence-transformers` +
  Qdrant; tests inject a deterministic hash-based fake embedder and an
  in-memory vector store, so the full chunk → embed → search pipeline is
  unit-testable without a GPU, a model download, or a running Qdrant
  instance. The fake backend has **no semantic understanding** — it only
  guarantees identical text → identical vector — so tests that need to
  prove *ranking* correctness query with exact chunk text rather than
  paraphrases (see the docstring on `test_query_retrieves_relevant_chunk`).
- **Vector store payloads never carry chunk text**, only `tenant_id` /
  `document_id` (`chunk_id` is the point id itself). Chunk content is the
  source of truth in Postgres and is fetched by id after a vector search
  returns matches. This keeps the vector store from becoming a second
  source of truth for content.
- **Tenant isolation extends to vector search**, not just SQL. Every
  `vector_store.search()` call requires a `tenant_id` and applies it as a
  hard filter — there's no code path that can run an unfiltered or
  cross-tenant vector search. See `test_vector_store_tenant_isolation` and
  `test_query_tenant_isolation`.
- **LLM synthesis degrades gracefully.** `POST /v1/query` always performs
  real retrieval; if `ANTHROPIC_API_KEY` isn't set it returns the retrieved
  chunks with `answer_synthesized: false` instead of failing the request.
  This is also what keeps the test suite free of any dependency on a live
  API key.

## Phase 3: how tenant isolation works now

Isolation is enforced in three independent layers, so a bug in one doesn't
become a breach:

1. **Application filters** — every query still has `WHERE tenant_id = ...`
   (unchanged from Phase 1), and cross-tenant lookups return 404, not 403.
2. **Postgres Row-Level Security** (`alembic/versions/a3c1f9e27b10_*`) —
   `ENABLE` + `FORCE` RLS with a policy on `tenants`, `users`, `documents`,
   `chunks`. `USING` hides other tenants' rows; `WITH CHECK` stops a tenant
   writing rows into someone else's tenant (or re-assigning its own rows).
   If application code forgets layer 1, the database refuses anyway —
   `test_rls_catches_a_route_that_forgot_its_tenant_filter` demonstrates it.
3. **Vector search filters** — `tenant_id` is mandatory on every
   `vector_store.search()` (Phase 2), now backed by a Qdrant payload index.

How the tenant reaches Postgres (`app/core/tenancy.py`): the verified JWT's
`tenant_id` is stored on the SQLAlchemy session, and an `after_begin` hook
runs `set_config('app.current_tenant', ..., true)` at the start of **every**
transaction. It must be per-transaction because the setting is
transaction-scoped and handlers commit mid-request; a pooled connection
therefore can never carry one request's tenant into another's
(`test_context_is_reapplied_after_commit_and_never_leaks_on_pooled_connection`).
No tenant set → empty setting → every policy matches nothing (fails closed).

Things worth knowing (and being ready to explain):

- **The API must not connect as a superuser.** Postgres exempts superusers
  and `BYPASSRLS` roles from RLS unconditionally, which would silently turn
  every policy into a no-op. The API connects as `documind_app`
  (`scripts/init-db.sql`: no superuser, no BYPASSRLS, no DDL); only Alembic
  uses the owner role (`MIGRATION_DATABASE_URL`). A test asserts the app
  role has neither flag.
- **The one escape hatch** is the *system context* (`app.bypass_rls`), used
  only by signup/login, which must find users before a tenant is known.
  Trade-off: it's a session setting, so anything able to run arbitrary SQL
  could set it (the alternative is `SECURITY DEFINER` functions for
  signup/login — stricter, more ceremony). Mitigation here:
  `test_system_context_only_on_auth_routes` fails the build if any route
  outside `/v1/auth/` can obtain it.
- **Plans** (`app/core/plans.py`):

  | Plan | Max documents | Query rate (burst / per min) | Upload rate (burst / per min) |
  |---|---|---|---|
  | free | 25 | 10 / 10 | 5 / 5 |
  | pro | 1,000 | 60 / 60 | 30 / 30 |
  | enterprise | 50,000 | 300 / 300 | 120 / 120 |

  Quota breach → `403`. Rate limit breach → `429` + `Retry-After`. Both are
  per **tenant**, not per user, so adding users doesn't multiply a budget and
  one noisy tenant can't consume another's. Re-uploading an existing file
  (dedupe hit) costs no quota. Billing/plan changes arrive in Phase 6; for
  now change a plan with
  `UPDATE tenants SET plan = 'pro' WHERE name = '...'` (as the owner role).
- **Rate limiter fails open** if Redis is unreachable (logged). It's a
  protection layer, not a correctness layer; quotas live in Postgres and are
  unaffected. The Redis check-and-decrement is one Lua script (atomic across
  API replicas, uses Redis's clock); a 80-way concurrent test proves it
  admits exactly `capacity` requests.

## Phase 4: asynchronous ingestion

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant R as Redis (broker)
    participant W as Celery worker
    participant P as Postgres (RLS)
    participant H as Customer webhook

    C->>A: POST /v1/documents (Idempotency-Key)
    A->>P: INSERT document (pending) + idempotency key, one transaction
    A->>R: publish ingest_document(document_id, tenant_id)
    A-->>C: 202 Accepted, status=pending
    R->>W: deliver job
    W->>P: extracting -> chunking -> embedding -> ready (commit per stage)
    W->>P: INSERT webhook_deliveries rows (same transaction as 'ready')
    W->>R: publish deliver_webhook(delivery_id)
    R->>W: deliver job
    W->>H: POST signed event (X-DocuMind-Signature)
    C->>A: GET /v1/documents/{id}  (poll)  ->  status=ready
```

**What happens when things go wrong** (each row has a test):

| Failure | Behavior |
|---|---|
| Corrupt/unreadable file | **Permanent**: marked `failed` after 1 attempt, never retried; `document.failed` webhook fires |
| Embedding model / vector store hiccup | **Transient**: retried with exponential backoff (5s, 10s, 20s), then `failed` ("Gave up after 4 attempts") with no half-indexed leftovers |
| Worker dies mid-job | Redelivered (`acks_late`); the job is **idempotent**, so chunks/vectors are never duplicated |
| Duplicate delivery of a finished job | No-op (guard on `status == ready`) |
| Document deleted while queued / mid-embedding | Job no-ops; any vectors it already wrote are removed |
| Broker down at upload time | `503` + `Retry-After`; document saved as `failed`; re-upload the same file or `POST /v1/documents/{id}/retry`; the idempotency key is released so the **same key** can retry |
| Client retries after a lost response | `Idempotency-Key` makes it safe: same key + same request returns the original document (`Idempotent-Replay: true`) |
| Same key, different request | `422` |
| Webhook receiver is down / slow / 5xx | Retried with backoff (10s, 20s, 40s, 80s), up to 5 attempts, then `failed`; document processing is never affected |

Design decisions worth being ready to explain:

- **Job policy lives in plain Python, not in Celery.** Handlers
  (`app/worker/handlers.py`, `app/services/pipeline.py`) return a `JobResult`
  saying "retry in N seconds" or "done"; the Celery task just translates that
  into `self.retry`. That is why the retry/backoff/idempotency logic is
  unit-tested without a broker, and the *same handlers* run in production and
  in the in-memory test queue.
- **Workers run under Row-Level Security too.** A worker has no JWT, so the
  tenant id travels in the job arguments and `tenant_session()` applies it.
  A worker is not a privileged backdoor: a test removes that call and the
  real worker process (running as the restricted role) can no longer see any
  document.
- **Transactional outbox for webhooks.** Delivery rows are inserted in the
  *same transaction* as the `ready`/`failed` status change, so an event can't
  be lost between "document finished" and "customer told". The queue message
  is only a nudge; the row is the source of truth.
- **The API publishes by task name** (`send_task`), so it never imports the
  worker's code or its heavy dependencies; the two are coupled only by a task
  name and JSON arguments.
- **Webhook signing is Stripe-style**: `X-DocuMind-Signature: t=<unix>,v1=<hex>`
  with `v1 = HMAC-SHA256(secret, "<t>.<raw body>")`. The timestamp is inside
  the signed payload, so a captured request can't be replayed later (300s
  tolerance). Secrets are shown once and stored **encrypted** (they must be
  recoverable to sign, so hashing isn't an option). `X-DocuMind-Delivery` is
  stable across retries so receivers can deduplicate.
- **Webhook URLs are an SSRF vector** (they make *our* servers send requests),
  so: HTTPS only, no credentials in the URL, every resolved address must be
  globally routable (blocks loopback, RFC1918, link-local/cloud-metadata,
  CGNAT, IPv6 ULA), redirects are **never followed**, and the URL is
  re-validated at delivery time because DNS can change after registration.
- **At-least-once, not exactly-once.** The system guarantees idempotent
  processing and stable delivery ids; it does not pretend a webhook can't
  arrive twice.

Try the webhooks locally:

```bash
# 1. Register an endpoint (the secret is returned ONCE)
curl -s -X POST $BASE/v1/webhooks -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"url":"http://host.docker.internal:9000/hook"}'      # note the "secret"

# 2. Run the reference receiver (verifies signatures, dedupes deliveries)
WEBHOOK_SECRET=whsec_... python scripts/webhook_receiver.py

# 3. Send a test event, then upload a document and watch document.ready arrive
curl -s -X POST $BASE/v1/webhooks/$HOOK_ID/ping -H "Authorization: Bearer $TOKEN"
curl -s $BASE/v1/webhooks/$HOOK_ID/deliveries -H "Authorization: Bearer $TOKEN"   # status, attempts, last error
```

**Known limitations** (deliberate scope cuts, all noted in the code):

- **Content-hash dedupe is best-effort under concurrency.** Two *simultaneous*
  uploads of identical bytes **without** an `Idempotency-Key` can both pass
  the check and create two documents. With a key the race is closed by a
  unique index (tested with 8 concurrent requests on real Postgres). A
  `UNIQUE(tenant_id, file_hash)` constraint would close it for keyless
  requests too.
- **No sweeper yet.** If publishing a webhook nudge fails after the delivery
  row is committed, the row stays `pending` until something re-enqueues it
  (a periodic sweeper is a Phase 5 item). Ingestion jobs aren't outboxed; a
  broker outage is surfaced as `503` instead.
- **SSRF residual risk:** a DNS-rebinding race between our address check and
  the connection. The full fix is pinning the resolved IP or an egress proxy.
- **Uploads live on a shared volume** between API and worker. Multi-host
  deployments need object storage (S3/MinIO) behind the existing
  `StorageBackend` interface.
- **Dedupe returns `200`, new work returns `202`.** Re-uploading a *failed*
  document's bytes re-queues it (`202`) rather than returning the failure.

## Quickstart (Docker — recommended)

```bash
docker compose up --build
```

This starts Postgres, Redis, Qdrant, the API (which runs
`alembic upgrade head` as the owner role, then serves as the restricted
role), and the Celery **worker**. The API is at
`http://localhost:8000`, interactive docs at `http://localhost:8000/docs`,
Qdrant's own dashboard at `http://localhost:6333/dashboard`.

The embedding model (~80MB) downloads on the API container's first startup
and is cached in a named volume afterwards, so subsequent restarts are
fast. To enable LLM answer synthesis on `/v1/query`, export a key before
starting:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

Without it, `/v1/query` still performs real retrieval — it just returns
`answer_synthesized: false` and `answer: null` instead of a generated
answer.

Schema now comes exclusively from Alembic (`AUTO_CREATE_TABLES=false`) —
`create_all()` can't create RLS policies. Phase 5 moves the migrate step
into CI/CD as an explicit deploy job.

## Quickstart (without Docker)

Requires Python 3.12+ and a running Postgres instance (or just use the
SQLite default for a quick look).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # edit DATABASE_URL if not using the SQLite default
uvicorn app.main:app --reload
```

## Demo script (run against either setup)

```bash
BASE=http://localhost:8000

# 1. Sign up — creates a tenant + owner user, returns a JWT
curl -s -X POST $BASE/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"tenant_name":"Acme Corp","email":"demo@acme.io","password":"supersecret123"}' | tee /tmp/signup.json
TOKEN=$(python -c "import json;print(json.load(open('/tmp/signup.json'))['token']['access_token'])")

# 2. Log in (alternative to using the signup token directly)
curl -s -X POST $BASE/v1/auth/login \
  -d "username=demo@acme.io&password=supersecret123"

# 3. Upload a document. Returns 202 + status "pending" immediately; a worker processes it.
#    The Idempotency-Key makes a client retry safe (re-run this exact command: same document back).
echo "Vector databases enable semantic search over unstructured text." > sample.txt
curl -s -X POST $BASE/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: demo-upload-1" \
  -F "file=@sample.txt;type=text/plain" | tee /tmp/upload.json
DOC_ID=$(python -c "import json;print(json.load(open('/tmp/upload.json'))['id'])")

# 3b. Poll until the worker finishes: pending -> extracting -> chunking -> embedding -> ready
curl -s $BASE/v1/documents/$DOC_ID -H "Authorization: Bearer $TOKEN"

# 4. List documents
curl -s $BASE/v1/documents -H "Authorization: Bearer $TOKEN"

# 5. Get extracted content
curl -s $BASE/v1/documents/$DOC_ID/content -H "Authorization: Bearer $TOKEN"

# 6. Confirm auth is enforced
curl -s -o /dev/null -w "%{http_code}\n" $BASE/v1/documents   # -> 401

# 7. Inspect the chunks it was split into
curl -s $BASE/v1/documents/$DOC_ID/chunks -H "Authorization: Bearer $TOKEN"

# 8. Ask a question over your documents (RAG)
curl -s -X POST $BASE/v1/query \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "What do my documents say about semantic search?", "top_k": 5}'
# -> real retrieved chunks always; "answer" is non-null only if
#    ANTHROPIC_API_KEY was set when the container started

# 9. See your plan, usage and limits
curl -s $BASE/v1/tenant -H "Authorization: Bearer $TOKEN"

# 10. Trip the rate limiter (free plan: burst of 10 queries) -> 429 + Retry-After
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code} " -X POST $BASE/v1/query \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"question":"hello"}'
done; echo

# 11. Prove isolation: sign up a SECOND tenant and try to read the first one's doc
TOKEN_B=$(curl -s -X POST $BASE/v1/auth/signup -H "Content-Type: application/json" \
  -d '{"tenant_name":"Other Co","email":"b@other.io","password":"supersecret123"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['token']['access_token'])")
curl -s -o /dev/null -w "%{http_code}\n" $BASE/v1/documents/$DOC_ID -H "Authorization: Bearer $TOKEN_B"   # -> 404

# 12. Delete it
curl -s -o /dev/null -w "%{http_code}\n" -X DELETE $BASE/v1/documents/$DOC_ID \
  -H "Authorization: Bearer $TOKEN"   # -> 204
```

Or skip curl entirely and use the Swagger UI at `/docs` — click "Authorize",
paste `Bearer <token>` from the signup response, and try every endpoint
interactively.

## Running the tests

```bash
pytest -v          # 161 tests (~45s), needs NOTHING running; 32 integration tests skip cleanly
```

The 161 use in-memory SQLite, a deterministic fake embedder, an in-memory
vector store and token bucket, a fake webhook sender, and an **in-memory task
queue that runs the real worker handlers** (see `tests/conftest.py`). The
queue has two modes: *eager* by default (jobs run on enqueue, like Celery's
`task_always_eager`) so feature tests can upload-then-query, and *lazy* for
tests marked `@pytest.mark.async_queue`, which assert what the API returns
**before** the worker has run.

- **Phases 1-3:** auth, upload/extraction, chunking, vector search + RAG,
  cross-tenant attack matrix, quotas, rate limiting (see earlier sections)
- **Phase 4 async contract** (`test_async_ingestion.py`, 16): 202-before-work,
  per-stage progress visible to polling, idempotent redelivery, permanent vs
  transient failure, backoff, retry exhaustion, broker outage -> 503 ->
  recovery, deletes racing the worker
- **Idempotency keys** (`test_idempotency.py`, 14): replay, mismatch -> 422,
  per-tenant scoping, expiry, quota, key release on outage
- **Webhooks** (`test_webhooks.py` 43, `test_webhook_signing.py` 16): a known-
  answer signature vector, replay/tamper rejection, 19 SSRF URL cases,
  delivery-time revalidation, event filtering, backoff, tenant isolation, and
  **real HTTP** delivery including "redirects are never followed"

**Integration tests (real Postgres + real Redis + a real Celery worker):**

```bash
docker compose up -d postgres redis      # init-db.sql creates the roles + a scratch DB
export RLS_TEST_ADMIN_URL=postgresql+psycopg2://documind:documind@localhost:5432/documind_test
export RLS_TEST_APP_URL=postgresql+psycopg2://documind_app:documind_app@localhost:5432/documind_test
export REDIS_TEST_URL=redis://localhost:6379/15
pytest -v                                # 193 tests (~75s)
```

- `test_rls_postgres.py` (23): wipes and rebuilds `documind_test`, runs every
  migration on real Postgres, then as the restricted role: cross-tenant
  reads/writes/updates/deletes, RLS on the Phase 4 tables, tenant-scoped
  worker sessions, a migration down/up round-trip, and **8 concurrent
  requests sharing one Idempotency-Key creating exactly one document**.
- `test_celery_integration.py` (5): starts an actual `celery worker`
  subprocess. Upload -> Redis -> worker (as the restricted DB role) ->
  Postgres -> **signed HTTP webhook** to a real receiver; permanent failure;
  jobs surviving in the broker until a worker appears; two tenants through
  one worker; a dead broker producing a fast, clean `503`.
- `test_rate_limit_redis.py` (4): the Lua token bucket, incl. 80-way atomicity.

**Sanity checks of the tests themselves.** For each safety property I
temporarily broke the code and confirmed a test fails: idempotent re-run,
permanent-vs-transient classification, redelivery no-op, orphan-vector
cleanup, idempotency fingerprint / key release / tenant scoping, SSRF
validation, redirect following, delivery-time revalidation, event filtering,
webhook tenant scoping, stable delivery ids, RLS policies, and the worker's
tenant context. Several of these initially *survived* (a test receiver that
only logged POSTs couldn't see a followed redirect; a test passed for the
wrong reason because of an unrelated crash) and were fixed.

## Migrations

An initial migration already exists (`alembic/versions/`). To use Alembic
instead of `AUTO_CREATE_TABLES`:

```bash
export AUTO_CREATE_TABLES=false
alembic upgrade head        # apply
alembic downgrade base      # roll back (tested — both directions work)
alembic revision --autogenerate -m "description"   # after changing models.py
```

## API surface

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/v1/auth/signup` | — | Create tenant + owner user, returns JWT |
| POST | `/v1/auth/login` | — | Exchange email/password for JWT |
| POST | `/v1/documents` | JWT | **202**: accept + queue for processing (`Idempotency-Key` supported; `200` if identical file already exists; `403` over quota; `429` rate limited; `503` queue down) |
| POST | `/v1/documents/{id}/retry` | JWT | Re-queue a `failed` document |
| GET | `/v1/documents` | JWT | List tenant's documents (paginated) |
| GET | `/v1/documents/{id}` | JWT | Metadata + status (`pending/extracting/chunking/embedding/ready/failed`) + `attempts` |
| GET | `/v1/documents/{id}/content` | JWT | Extracted text (`409` until `ready`) |
| GET | `/v1/documents/{id}/chunks` | JWT | Inspect how a document was chunked |
| DELETE | `/v1/documents/{id}` | JWT | Delete document + chunks + vectors + file |
| POST | `/v1/query` | JWT | RAG query over `ready` documents (rate limited) |
| POST | `/v1/webhooks` | JWT | Register an endpoint; returns the signing `secret` once |
| GET | `/v1/webhooks` | JWT | List your webhooks (no secrets) |
| DELETE | `/v1/webhooks/{id}` | JWT | Delete a webhook (cancels its pending deliveries) |
| GET | `/v1/webhooks/{id}/deliveries` | JWT | Delivery history: status, attempts, last error |
| POST | `/v1/webhooks/{id}/ping` | JWT | Send a signed `webhook.ping` test event |
| GET | `/v1/tenant` | JWT | Plan, document usage, and limits |
| GET | `/health` | — | Liveness check |

Webhook events: `document.ready`, `document.failed` (plus `webhook.ping`).

## What's deliberately *not* here yet

- Observability (metrics/tracing/logging), CI/CD, Kubernetes, a webhook sweeper, object storage → **Phase 5**
- Hybrid search (keyword + vector), re-ranking, usage-based billing → **Phase 6**
