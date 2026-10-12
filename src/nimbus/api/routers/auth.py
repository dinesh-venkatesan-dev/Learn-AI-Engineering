"""Sign up and sign in: POST /auth/register (JSON), POST /auth/token (a FORM, the OAuth2 password flow)."""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from nimbus.api.db import SessionDep
from nimbus.api.deps import SettingsDep
from nimbus.api.errors import ApiError, Unauthorized
from nimbus.api.models import User, UserPublic
from nimbus.api.security import CurrentUser, create_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class Registration(BaseModel):
  username: str = Field(min_length=3, max_length=40, pattern=r"^[a-z0-9_.-]+$")
  password: str = Field(min_length=10, max_length=200)


class Token(BaseModel):
  access_token: str
  token_type: str = "bearer"


class UsernameTaken(ApiError):
  status, code = 409, "username_taken"


@router.post("/register", status_code=201)
def register(body: Registration, session: SessionDep) -> UserPublic:
  user = User(username=body.username, password_hash=hash_password(body.password))  # the hash, never the password
  session.add(user)
  try:
    session.commit()
  except IntegrityError as error:
    session.rollback()
    raise UsernameTaken(f"The username {body.username!r} is taken.") from error
  session.refresh(user)
  return user


@router.post("/token")
def token(form: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDep, settings: SettingsDep) -> Token:
  user = session.exec(select(User).where(User.username == form.username)).first()
  if user is None or not verify_password(form.password, user.password_hash):
    raise Unauthorized("Wrong username or password.")  # the same answer for both: no help guessing
  return Token(access_token=create_token(user.username, settings.jwt_secret.get_secret_value(), settings.jwt_minutes))


@router.get("/me")
def me(user: CurrentUser) -> UserPublic:
  return user
