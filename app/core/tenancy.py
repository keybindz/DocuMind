"""
Tenant context for Postgres Row-Level Security.

How it works:

1. Every RLS policy compares a row's tenant_id to the Postgres setting
   `app.current_tenant` (see the Phase 3 migration).
2. A request's tenant comes from its verified JWT. `set_tenant()` records it
   on the SQLAlchemy session (`session.info`).
3. An `after_begin` hook re-applies it at the start of EVERY transaction via
   `set_config(..., is_local => true)`. Transaction-scoped settings vanish at
   COMMIT/ROLLBACK, and our request handlers commit mid-request, so setting
   it once per request would silently drop isolation after the first commit.
4. If no tenant is set, the setting is empty and every policy evaluates to
   "no rows" -- the system fails closed.

The one deliberate escape hatch is the *system* context (`app.bypass_rls`),
used only by signup/login, which must look up users before any tenant is
known. `get_system_db` is the only way to get it, and a test
(`test_system_context_only_on_auth_routes`) asserts no other route can.

On non-Postgres dialects (the SQLite test database) all of this is a no-op.
"""
import uuid
from contextlib import contextmanager

from sqlalchemy import event, text
from sqlalchemy.orm import Session

_TENANT_KEY = "tenant_id"
_BYPASS_KEY = "bypass_rls"

_SET_TENANT_SQL = text("SELECT set_config('app.current_tenant', :v, true)")
_SET_BYPASS_SQL = text("SELECT set_config('app.bypass_rls', :v, true)")


def _apply(connection, info: dict) -> None:
    if connection.dialect.name != "postgresql":
        return
    tenant = info.get(_TENANT_KEY)
    connection.execute(_SET_TENANT_SQL, {"v": str(tenant) if tenant else ""})
    connection.execute(_SET_BYPASS_SQL, {"v": "on" if info.get(_BYPASS_KEY) else ""})


@event.listens_for(Session, "after_begin")
def _on_begin(session: Session, transaction, connection) -> None:
    _apply(connection, session.info)


def _reapply_if_in_transaction(session: Session) -> None:
    # If a transaction is already open (begun before the context was set),
    # after_begin has already fired, so apply the new context immediately.
    if session.in_transaction():
        _apply(session.connection(), session.info)


def set_tenant(session: Session, tenant_id: uuid.UUID) -> None:
    session.info[_TENANT_KEY] = tenant_id
    session.info.pop(_BYPASS_KEY, None)
    _reapply_if_in_transaction(session)


def set_system_context(session: Session) -> None:
    session.info[_BYPASS_KEY] = True
    session.info.pop(_TENANT_KEY, None)
    _reapply_if_in_transaction(session)


@contextmanager
def tenant_session(session_factory, tenant_id: uuid.UUID):
    """
    A short-lived session scoped to one tenant, for background workers. A
    worker has no HTTP request or JWT, so the tenant id travels in the job's
    arguments and is applied here -- which means workers run under Row-Level
    Security exactly like the API does, rather than as a privileged backdoor.
    """
    session = session_factory()
    try:
        set_tenant(session, tenant_id)
        yield session
    finally:
        session.close()
