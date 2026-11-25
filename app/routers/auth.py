from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.database import get_system_db
from app.models import Tenant, User, UserRole
from app.schemas import SignupRequest, SignupResponse, Token

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_system_db)):
    """
    Creates a brand-new tenant plus its first user (owner role).

    This is the only endpoint that creates a tenant — everything else
    operates within the tenant of the authenticated caller.
    """
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    tenant = Tenant(name=payload.tenant_name)
    db.add(tenant)
    db.flush()  # populate tenant.id before creating the user

    user = User(
        tenant_id=tenant.id,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=UserRole.owner,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user_id=user.id, tenant_id=tenant.id)
    return SignupResponse(user=user, token=Token(access_token=token))


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_system_db)):
    """
    OAuth2-password-flow-shaped login (form fields: username, password) so
    it's compatible with FastAPI's built-in /docs "Authorize" button and any
    standard OAuth2 client — `username` is the user's email.
    """
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user_id=user.id, tenant_id=user.tenant_id)
    return Token(access_token=token)
