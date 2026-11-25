from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.plans import limits_for
from app.database import get_db
from app.dependencies import get_current_user
from app.models import Document, Tenant, User
from app.schemas import TenantUsageOut

router = APIRouter(prefix="/v1/tenant", tags=["tenant"])


@router.get("", response_model=TenantUsageOut)
def get_tenant_usage(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """The caller's tenant: plan, current usage, and the limits that apply."""
    tenant = db.get(Tenant, current_user.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    limits = limits_for(tenant.plan)
    count = db.query(Document).filter(Document.tenant_id == tenant.id).count()
    return TenantUsageOut(
        tenant_id=tenant.id,
        name=tenant.name,
        plan=tenant.plan,
        document_count=count,
        max_documents=limits.max_documents,
        query_rate_per_minute=limits.query_bucket.refill_per_minute,
        upload_rate_per_minute=limits.upload_bucket.refill_per_minute,
    )
