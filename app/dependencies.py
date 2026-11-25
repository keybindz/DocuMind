"""Reusable FastAPI dependencies for auth."""
import math
import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import InvalidTokenError, decode_access_token
from app.core.tenancy import set_tenant
from app.database import get_db
from app.core.plans import limits_for
from app.models import Tenant, User
from app.services.rate_limit import RateLimiter, get_rate_limiter

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
    except InvalidTokenError:
        raise credentials_exception

    try:
        user_id = uuid.UUID(payload["sub"])
        tenant_id = uuid.UUID(payload["tenant_id"])
    except (KeyError, ValueError):
        raise credentials_exception

    # Scope the DB session to the token's tenant BEFORE the first query, so
    # Row-Level Security applies to the user lookup itself. If the token's
    # tenant claim doesn't match the user's real tenant, RLS hides the row
    # and this 401s; the explicit check below is belt-and-suspenders.
    set_tenant(db, tenant_id)
    user = db.get(User, user_id)
    if user is None or user.tenant_id != tenant_id:
        raise credentials_exception
    return user


def rate_limit(action: str):
    """
    Per-tenant token-bucket rate limit, sized by the tenant's plan.
    `action` is "query" or "upload". Keyed by tenant (not user), so a tenant's
    whole organization shares one budget and noisy tenants can't starve others.
    """

    def dependency(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
        limiter: RateLimiter = Depends(get_rate_limiter),
    ) -> None:
        tenant = db.get(Tenant, current_user.tenant_id)
        limits = limits_for(tenant.plan if tenant else "free")
        spec = limits.query_bucket if action == "query" else limits.upload_bucket
        result = limiter.check(
            f"rl:{action}:{current_user.tenant_id}",
            capacity=spec.capacity,
            refill_per_second=spec.refill_per_minute / 60.0,
        )
        if not result.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {action}. Retry in {result.retry_after_seconds:.0f}s.",
                headers={"Retry-After": str(max(1, math.ceil(result.retry_after_seconds)))},
            )

    return dependency
