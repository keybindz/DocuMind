import logging
import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core import crypto
from app.core.plans import limits_for
from app.database import get_db
from app.dependencies import get_current_user
from app.models import Tenant, User, Webhook, WebhookDelivery
from app.schemas import DeliveryListOut, DeliveryOut, WebhookCreate, WebhookCreatedOut, WebhookOut
from app.services.task_queue import QueueUnavailable, TaskQueue, get_task_queue
from app.services.webhooks import WebhookURLError, create_ping_delivery, validate_webhook_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])


def _get_owned_webhook(db: Session, webhook_id: uuid.UUID, tenant_id: uuid.UUID) -> Webhook:
    hook = db.query(Webhook).filter(Webhook.id == webhook_id, Webhook.tenant_id == tenant_id).first()
    if hook is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")
    return hook


@router.post("", response_model=WebhookCreatedOut, status_code=status.HTTP_201_CREATED)
def create_webhook(
    payload: WebhookCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Registers an endpoint to receive signed event callbacks. The response
    contains the signing `secret` exactly once -- it is stored encrypted and
    never returned again.
    """
    try:
        validate_webhook_url(payload.url)
    except WebhookURLError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    tenant = db.get(Tenant, current_user.tenant_id)
    plan = tenant.plan if tenant else "free"
    limit = limits_for(plan).max_webhooks
    if db.query(Webhook).filter(Webhook.tenant_id == current_user.tenant_id).count() >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Webhook limit reached for plan '{plan}' ({limit}).",
        )

    secret = "whsec_" + secrets.token_urlsafe(32)
    hook = Webhook(
        tenant_id=current_user.tenant_id,
        url=payload.url,
        secret_encrypted=crypto.encrypt(secret),
        events=list(dict.fromkeys(payload.events)),
        is_active=True,
    )
    db.add(hook)
    db.commit()
    db.refresh(hook)
    return WebhookCreatedOut(
        id=hook.id, url=hook.url, events=hook.events, is_active=hook.is_active, created_at=hook.created_at, secret=secret
    )


@router.get("", response_model=list[WebhookOut])
def list_webhooks(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Webhook).filter(Webhook.tenant_id == current_user.tenant_id).order_by(Webhook.created_at).all()


@router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_webhook(
    webhook_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    hook = _get_owned_webhook(db, webhook_id, current_user.tenant_id)
    db.delete(hook)  # cascades to its delivery history; queued deliveries will find nothing and no-op
    db.commit()
    return None


@router.get("/{webhook_id}/deliveries", response_model=DeliveryListOut)
def list_deliveries(
    webhook_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delivery history: status, attempt count, and the last error -- for debugging a receiver."""
    _get_owned_webhook(db, webhook_id, current_user.tenant_id)
    q = db.query(WebhookDelivery).filter(
        WebhookDelivery.webhook_id == webhook_id, WebhookDelivery.tenant_id == current_user.tenant_id
    )
    total = q.count()
    items = q.order_by(WebhookDelivery.created_at.desc()).offset(offset).limit(limit).all()
    return DeliveryListOut(items=items, total=total, limit=limit, offset=offset)


@router.post("/{webhook_id}/ping", response_model=DeliveryOut, status_code=status.HTTP_202_ACCEPTED)
def ping_webhook(
    webhook_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    queue: TaskQueue = Depends(get_task_queue),
):
    """Sends a signed `webhook.ping` event so you can verify your receiver end to end."""
    hook = _get_owned_webhook(db, webhook_id, current_user.tenant_id)
    delivery = create_ping_delivery(db, hook)
    db.commit()
    db.refresh(delivery)
    try:
        queue.enqueue_webhook_delivery(delivery.id, current_user.tenant_id)
    except QueueUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing queue unavailable; try again shortly.",
            headers={"Retry-After": "30"},
        )
    db.refresh(delivery)
    return delivery
