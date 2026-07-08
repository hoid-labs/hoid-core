from __future__ import annotations

from ._context import AuthContext


class StaticAuthProvider:
    "Resolves credentials against a static in-process mapping; for development and testing only."

    def __init__(
        self,
        api_keys: dict[str, AuthContext] | None = None,
        users: dict[str, AuthContext] | None = None,
    ) -> None:
        """
        Args:
            api_keys: Map of raw key string to AuthContext, matched via `{"type": "api_key", "key": ...}`.
            users: Map of username string to AuthContext, matched via `{"type": "username", "username": ...}`.
        """
        self._api_keys: dict[str, AuthContext] = api_keys or {}
        self._users: dict[str, AuthContext] = users or {}

    async def resolve(self, credentials: dict) -> AuthContext | None:
        cred_type = credentials.get("type")
        if cred_type == "api_key":
            return self._api_keys.get(credentials.get("key", ""))
        if cred_type == "username":
            return self._users.get(credentials.get("username", ""))
        return None
