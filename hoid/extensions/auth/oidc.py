from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx

from hoid._optional import require as _require

from ._context import AuthContext

log = logging.getLogger(__name__)

try:
    import jwt
    from jwt import PyJWK
except ImportError:
    jwt = None  # type: ignore[assignment,misc]
    PyJWK = None  # type: ignore[assignment,misc]


class OIDCAuthProvider:
    """OIDC Authorization Code flow; exchanges a provider-issued code for an AuthContext."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        discovery_url: str,
        redirect_uri: str,
        roles_claim: str = "roles",
        scopes: list[str] | None = None,
        role_map: dict[str, set[str]] | None = None,
    ):
        _require("jwt", jwt)
        self._client_id = client_id
        self._client_secret = client_secret
        self._discovery_url = discovery_url
        self._redirect_uri = redirect_uri
        self._roles_claim = roles_claim
        self._scopes = scopes or ["openid", "email", "profile"]
        self._role_map: dict[str, set[str]] = role_map or {}
        self._config: dict | None = None

    @classmethod
    def from_env(
        cls,
        redirect_uri: str,
        role_map: dict[str, set[str]] | None = None,
    ) -> OIDCAuthProvider:
        "Instantiate from OIDC_CLIENT_ID, OIDC_CLIENT_SECRET, OIDC_DISCOVERY_URL env vars."
        return cls(
            client_id=os.environ["OIDC_CLIENT_ID"],
            client_secret=os.environ["OIDC_CLIENT_SECRET"],
            discovery_url=os.environ["OIDC_DISCOVERY_URL"],
            redirect_uri=redirect_uri,
            roles_claim=os.environ.get("OIDC_ROLES_CLAIM", "roles"),
            role_map=role_map,
        )

    async def _discovery(self) -> dict:
        if self._config is None:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self._discovery_url)
                resp.raise_for_status()
                self._config = resp.json()
        return self._config

    async def authorization_url(self, state: str) -> tuple[str, str]:
        "Build the provider authorization URL; returns (url, code_verifier) for PKCE (RFC 7636)."
        config = await self._discovery()
        # 96 random bytes → ~128-char verifier, well within the 43-128 char RFC range after base64url
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(96)).rstrip(b"=").decode()
        )
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "scope": " ".join(self._scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return config["authorization_endpoint"] + "?" + urlencode(params), code_verifier

    async def _fetch_signing_key(self, jwks_uri: str, kid: str | None) -> Any | None:
        "Fetch JWKS from uri and return the matching signing key, or None if not found."
        async with httpx.AsyncClient() as client:
            resp = await client.get(jwks_uri)
            if not resp.is_success:
                log.warning("JWKS fetch failed from %s: %s", jwks_uri, resp.status_code)
                return None
            jwks = resp.json()

        all_keys = jwks.get("keys", [])
        log.debug(
            "JWKS %s — keys: %s",
            jwks_uri,
            [
                {
                    "kid": k.get("kid"),
                    "x5t": k.get("x5t"),
                    "x5t#S256": k.get("x5t#S256"),
                }
                for k in all_keys
            ],
        )

        # some providers set the JWT kid to a cert thumbprint rather than the JWK kid field
        key_data = next(
            (
                k
                for k in all_keys
                if k.get("kid") == kid
                or k.get("x5t") == kid
                or k.get("x5t#S256") == kid
            ),
            None,
        )
        if key_data is None:
            return None

        try:
            return PyJWK.from_dict(key_data).key
        except Exception as exc:
            log.warning("key construction error for kid=%s: %s", kid, exc)
            return None

    async def exchange_code(self, code: str, code_verifier: str) -> dict | None:
        "Exchange an authorization code for decoded id_token claims, or None on failure."
        config = await self._discovery()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                config["token_endpoint"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self._redirect_uri,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code_verifier": code_verifier,
                },
            )
            if not resp.is_success:
                log.warning("token exchange failed: %s %s", resp.status_code, resp.text)
                return None
            tokens = resp.json()

        id_token = tokens.get("id_token")
        if not id_token:
            log.warning("token response contained no id_token")
            return None

        try:
            header = jwt.get_unverified_header(id_token)
            kid = header.get("kid")
            alg = header.get("alg", "RS256")
        except jwt.DecodeError as exc:
            log.warning("id_token header decode failed: %s", exc)
            return None

        log.debug("id_token header: kid=%s alg=%s", kid, alg)

        v2_jwks_uri = config["jwks_uri"]
        # providers that expose separate versioned key sets need both paths tried
        v1_jwks_uri = (
            v2_jwks_uri.replace("/v2.0/keys", "/keys")
            if "/v2.0/keys" in v2_jwks_uri
            else None
        )
        # apps migrated from SAML may sign with an app-specific cert exposed via ?appid=
        app_jwks_uri = f"{v2_jwks_uri}?appid={self._client_id}"

        signing_key = await self._fetch_signing_key(v2_jwks_uri, kid)
        if signing_key is None and v1_jwks_uri:
            log.debug("kid not found in v2.0 JWKS, trying v1.0 JWKS: %s", v1_jwks_uri)
            signing_key = await self._fetch_signing_key(v1_jwks_uri, kid)
        if signing_key is None:
            log.debug(
                "kid not found in tenant JWKS, trying app-specific JWKS: %s",
                app_jwks_uri,
            )
            signing_key = await self._fetch_signing_key(app_jwks_uri, kid)

        if signing_key is None:
            log.warning(
                "no JWKS key matched id_token kid=%s (checked v2, v1, and app-specific endpoints)",
                kid,
            )
            return None

        try:
            claims: dict = jwt.decode(
                id_token,
                signing_key,
                algorithms=["RS256", "ES256", "RS384", "ES384"],
                audience=self._client_id,
            )
        except jwt.InvalidTokenError as exc:
            log.warning("id_token validation failed: %s", exc)
            return None

        log.info("OIDC claims for %s: roles=%s", claims.get("email") or claims.get("sub"), claims.get(self._roles_claim))

        # Azure puts app role assignments in the access_token, not the id_token.
        # If the id_token has no roles claim, merge it from the access_token.
        if not claims.get(self._roles_claim):
            access_token = tokens.get("access_token")
            if access_token:
                try:
                    at_claims = jwt.decode(
                        access_token,
                        options={"verify_signature": False},
                        algorithms=["RS256", "ES256", "RS384", "ES384"],
                    )
                    at_roles = at_claims.get(self._roles_claim)
                    if at_roles:
                        log.debug(
                            "roles claim absent from id_token; merging from access_token: %s",
                            at_roles,
                        )
                        claims[self._roles_claim] = at_roles
                except jwt.DecodeError as exc:
                    log.debug("access_token decode skipped: %s", exc)

        return claims

    def claims_to_context(
        self,
        claims: dict,
        role_map: dict[str, set[str]] | None = None,
    ) -> AuthContext:
        # prefer email as user_id; sub is often an opaque UUID
        user_id = claims.get("email") or claims.get("sub", "unknown")
        raw = claims.get(self._roles_claim, [])
        roles: set[str] = (
            set(raw) if isinstance(raw, list) else ({raw} if raw else set())
        )
        # role_map lets callers assign roles that the token doesn't carry
        effective_map = role_map if role_map is not None else self._role_map
        if effective_map:
            roles |= effective_map.get(claims.get("sub", ""), set())
            roles |= effective_map.get(claims.get("email", ""), set())
        return AuthContext(user_id=user_id, roles=roles)

    async def resolve(self, credentials: dict) -> AuthContext | None:
        "Resolve OIDC authorization code credentials to an AuthContext."
        if credentials.get("type") != "oidc_code":
            return None
        code = credentials.get("code")
        code_verifier = credentials.get("code_verifier")
        if not code or not code_verifier:
            return None
        claims = await self.exchange_code(code, code_verifier)
        if claims is None:
            return None
        return self.claims_to_context(claims)
