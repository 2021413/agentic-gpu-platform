"""Service authentication for the internal worker API (spec section 19).

The internal surface lets a caller inject capacity into the platform and take it
away again, so it is authenticated — as an *abstraction*. ``ServiceAuthenticator``
is what the routes depend on; the shared-token implementation below is only the
first one, and swapping it for mTLS or signed JWTs must not touch a route.

Two rules the implementation exists to enforce:

* no secret is ever hard-coded — the token is a constructor argument that
  ``bootstrap`` reads from configuration (``SERVICE_TOKEN``);
* token comparison is constant-time. ``==`` on bytes short-circuits on the first
  differing byte, which leaks the length of the shared prefix and makes a token
  guessable byte by byte over enough requests.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from interfaces.api.errors import ServiceAuthError

__all__ = [
    "ServiceAuthenticator",
    "ServicePrincipal",
    "SharedSecretServiceAuthenticator",
    "extract_service_token",
]


@dataclass(frozen=True, slots=True)
class ServicePrincipal:
    """Who the authenticated caller is, once credentials have been accepted."""

    name: str


@runtime_checkable
class ServiceAuthenticator(Protocol):
    """Decides whether a credential admits a caller to the internal API."""

    def authenticate(self, token: str | None) -> ServicePrincipal:
        """Return the principal, or raise ``ServiceAuthError``.

        Raising rather than returning ``None`` keeps the refusal reason — no
        credential versus a rejected one — from being flattened by the caller.
        """
        ...


class SharedSecretServiceAuthenticator:
    """One shared token for every worker. Adequate locally, minimal elsewhere.

    It gives no per-worker identity and no revocation short of rotating the
    secret for the whole fleet; that trade-off is acceptable precisely because
    the interface hides it.
    """

    def __init__(self, token: str, *, principal: str = "worker") -> None:
        if not token or not token.strip():
            # Refusing to start beats starting with authentication silently
            # disabled, which is how internal APIs end up world-writable.
            raise ValueError(
                "a non-empty service token is required; configure SERVICE_TOKEN "
                "instead of running the internal API unauthenticated"
            )
        self._expected = token.encode("utf-8")
        self._principal = principal

    def authenticate(self, token: str | None) -> ServicePrincipal:
        if token is None:
            raise ServiceAuthError.missing()
        if not secrets.compare_digest(token.encode("utf-8"), self._expected):
            raise ServiceAuthError.invalid()
        return ServicePrincipal(name=self._principal)


_BEARER_PREFIX = "bearer "


def extract_service_token(
    *, authorization: str | None = None, service_token_header: str | None = None
) -> str | None:
    """Pull the credential out of the two accepted headers.

    ``Authorization: Bearer <token>`` is the standard form; ``X-Service-Token``
    exists because some proxies and sidecars rewrite ``Authorization``. Pure
    protocol parsing: whether the value is *valid* is the authenticator's call,
    not this function's.
    """
    if authorization is not None:
        candidate = authorization.strip()
        if candidate.lower().startswith(_BEARER_PREFIX):
            token = candidate[len(_BEARER_PREFIX) :].strip()
            return token or None
        return None
    if service_token_header is not None:
        return service_token_header.strip() or None
    return None
