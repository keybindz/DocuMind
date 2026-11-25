"""
Database engine/session setup, plus a portable UUID column type.

The GUID type stores a native UUID on Postgres (dialect-optimized, indexable)
but falls back to CHAR(36) on SQLite. This lets the exact same models power
the production Postgres database *and* a fast in-memory SQLite database for
unit tests, without maintaining two schemas.
"""
import uuid
from typing import Generator

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.types import CHAR, TypeDecorator

from app.config import get_settings
from app.core import tenancy  # noqa: F401 - registers the RLS after_begin hook

settings = get_settings()

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


class Base(DeclarativeBase):
    pass


class GUID(TypeDecorator):
    """Platform-independent UUID type: native UUID on Postgres, CHAR(36) elsewhere."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            return str(uuid.UUID(str(value)))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(str(value))
        return value


def get_db() -> Generator:
    """FastAPI dependency: yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_system_db(db=Depends(get_db)):
    """
    A session in the *system* RLS context (bypasses tenant policies).

    Only for flows that must run before a tenant is known -- signup and
    login. Never use this in a route that serves tenant data; the isolation
    test suite fails if any non-auth route depends on it.
    """
    tenancy.set_system_context(db)
    return db
