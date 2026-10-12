"""Users, passwords and access tokens (OAuth2 password flow with JWTs).

hash_password / verify_password   Argon2 via pwdlib: slow on purpose, salted; the password itself is never stored
create_token                      a signed JWT {"sub": username, "exp": ...}: anyone can read it, nobody can change it
CurrentUser                       the dependency for protected endpoints: missing, expired or tampered → 401
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from sqlmodel import select

from nimbus.api.db import SessionDep
from nimbus.api.deps import SettingsDep
from nimbus.api.errors import Unauthorized
from nimbus.api.models import User

ALGORITHM = "HS256"
password_hash = PasswordHash.recommended()  # Argon2
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")  # /docs' Authorize button speaks this flow


def hash_password(password: str) -> str:
  return password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
  return password_hash.verify(password, hashed)


def create_token(username: str, secret: str, minutes: int) -> str:
  claims = {"sub": username, "exp": datetime.now(UTC) + timedelta(minutes=minutes)}
  return jwt.encode(claims, secret, algorithm=ALGORITHM)


def username_from(token: str, secret: str) -> str:
  try:
    return jwt.decode(token, secret, algorithms=[ALGORITHM])["sub"]
  except jwt.PyJWTError as error:
    raise Unauthorized("Invalid or expired token.", headers={"WWW-Authenticate": "Bearer"}) from error


def current_user(token: Annotated[str, Depends(oauth2)], settings: SettingsDep, session: SessionDep) -> User:
  username = username_from(token, settings.jwt_secret.get_secret_value())
  user = session.exec(select(User).where(User.username == username)).first()
  if user is None:
    raise Unauthorized("Invalid or expired token.", headers={"WWW-Authenticate": "Bearer"})
  return user


CurrentUser = Annotated[User, Depends(current_user)]
