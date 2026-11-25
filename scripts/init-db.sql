-- Runs once, on first Postgres container start (docker-entrypoint-initdb.d).
--
-- Creates the restricted role the API connects as. It is deliberately NOT a
-- superuser and does NOT have BYPASSRLS: Postgres exempts superusers and
-- BYPASSRLS roles from Row-Level Security unconditionally, so connecting the
-- API as the owner/superuser would silently turn every RLS policy into a
-- no-op. The owner role (POSTGRES_USER) is used only to run migrations.

CREATE ROLE documind_app LOGIN PASSWORD 'documind_app'
    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;

GRANT CONNECT ON DATABASE documind TO documind_app;
GRANT USAGE ON SCHEMA public TO documind_app;

-- Tables the owner creates later (via Alembic) are automatically usable by
-- the app role: data access only, never DDL.
ALTER DEFAULT PRIVILEGES FOR ROLE documind IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO documind_app;

-- Scratch database for the integration tests (tests/test_rls_postgres.py).
-- The tests wipe and rebuild its schema, so it must never be the dev DB.
CREATE DATABASE documind_test OWNER documind;
GRANT CONNECT ON DATABASE documind_test TO documind_app;
