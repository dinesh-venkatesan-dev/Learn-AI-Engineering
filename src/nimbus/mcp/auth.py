"""Who is calling a Nimbus MCP server over HTTP, and what they may do (MCP Module 4).

Callers send a JWT from the identity provider (Auth0, Okta, Entra ID, ...). The server holds only the provider's
PUBLIC key and checks signature, issuer, audience and expiry on every request, before any tool runs.

  sub      who: a customer id (c-1042) or a member of staff (staff-anita)
  scopes   orders:read_all (support staff: any customer's orders), orders:refund (issue refunds)
  roles    manager (the daily sales report), in the "roles" claim

DevIdentityProvider plays the provider in labs and tests: it generates a key pair and signs tokens.
"""

import os

from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from fastmcp.server.dependencies import get_access_token

ISSUER = "https://auth.nimbus.example"
AUDIENCE = "nimbus-mcp"
PUBLIC_KEY_ENV = "NIMBUS_JWT_PUBLIC_KEY"


class DevIdentityProvider:
  """A stand-in identity provider: it holds the private key; servers get only public_key."""

  def __init__(self) -> None:
    self.keys = RSAKeyPair.generate()
    self.public_key = self.keys.public_key

  def token(
    self,
    subject: str,
    scopes: tuple[str, ...] = (),
    roles: tuple[str, ...] = (),
    audience: str = AUDIENCE,
    expires_in_seconds: int = 3600,
  ) -> str:
    return self.keys.create_token(
      subject=subject,
      issuer=ISSUER,
      audience=audience,
      scopes=list(scopes),
      expires_in_seconds=expires_in_seconds,
      additional_claims={"roles": list(roles)},
    )


def verifier(public_key: str | None = None) -> JWTVerifier:
  """The server side: trust tokens signed by the provider, issued FOR this server (audience), not expired."""
  return JWTVerifier(public_key=public_key or os.environ[PUBLIC_KEY_ENV], issuer=ISSUER, audience=AUDIENCE)


def caller() -> str:
  """Who is calling, from the verified token: never from a tool argument the model fills in."""
  return get_access_token().claims["sub"]


def caller_scopes() -> list[str]:
  return list(get_access_token().scopes)


def roles(claims: dict) -> list[str]:
  """Where this provider puts roles (every provider uses its own claim)."""
  return claims.get("roles", [])
