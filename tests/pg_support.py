"""
Shared fixtures/helpers for tests that need a REAL Postgres (and the schema
built by running every Alembic migration against it).

Env vars (see README):
    RLS_TEST_ADMIN_URL  owner/superuser URL of a scratch database (schema is WIPED)
    RLS_TEST_APP_URL    URL of the restricted `documind_app` role for the same database
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

ADMIN_URL = os.environ.get("RLS_TEST_ADMIN_URL")
APP_URL = os.environ.get("RLS_TEST_APP_URL")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = ["webhook_deliveries", "webhooks", "idempotency_keys", "chunks", "documents", "users", "tenants"]


def _alembic(*args: str) -> None:
    env = {**os.environ, "DATABASE_URL": ADMIN_URL}
    env.pop("MIGRATION_DATABASE_URL", None)
    proc = subprocess.run(
        [sys.executable, "-m", "alembic", *args], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, f"alembic {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}"


def _grant_app_role(admin_engine) -> None:
    app_role = create_engine(APP_URL).url.username
    with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {app_role}"))
        conn.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {app_role}"))


@pytest.fixture(scope="module")
def admin_engine():
    engine = create_engine(ADMIN_URL)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    _alembic("upgrade", "head")  # also proves every migration runs on real Postgres
    _grant_app_role(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def app_engine(admin_engine):
    engine = create_engine(APP_URL)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_tables(admin_engine):
    with admin_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    yield


# ---------- helpers: raw SQL under an explicit RLS context ----------

def _ctx(conn, tenant=None, bypass=False):
    conn.execute(text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant) if tenant else ""})
    conn.execute(text("SELECT set_config('app.bypass_rls', :b, true)"), {"b": "on" if bypass else ""})


def rows(engine, sql, params=None, tenant=None, bypass=False):
    with engine.begin() as conn:
        _ctx(conn, tenant, bypass)
        return conn.execute(text(sql), params or {}).fetchall()


def execute(engine, sql, params=None, tenant=None, bypass=False) -> int:
    with engine.begin() as conn:
        _ctx(conn, tenant, bypass)
        return conn.execute(text(sql), params or {}).rowcount


