from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.database import Base, engine
from app.routers import auth, documents, query, tenant, webhooks

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev/test convenience only. Real environments run `alembic upgrade head`
    # as a deploy step instead — see alembic/README in this repo.
    if settings.AUTO_CREATE_TABLES:
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Multi-tenant document intelligence platform — Phases 1-4 (auth, async ingestion, RAG, tenancy, webhooks).",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.ENV}


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(tenant.router)
app.include_router(webhooks.router)
