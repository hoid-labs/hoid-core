from typing import Protocol, runtime_checkable

from ._context import AuthContext
from .backends import FilePolicyBackend, MemoryPolicyBackend
from .oidc import OIDCAuthProvider
from .providers import StaticAuthProvider


@runtime_checkable
class PolicyBackend(Protocol):
    "Interface for any backend that resolves tool permissions for an AuthContext."

    def get_allowed_tools(self, context: AuthContext) -> set[str]: ...

    def is_allowed(self, tool_name: str, context: AuthContext) -> bool: ...


@runtime_checkable
class AuthProvider(Protocol):
    "Interface for resolving raw credentials into an AuthContext."

    async def resolve(self, credentials: dict) -> AuthContext | None: ...


class AuthGate:
    "Enforces tool-access policy at schema filtering and at execution time."

    def __init__(self, backend: PolicyBackend):
        """
        Args:
            backend: A PolicyBackend that resolves tool permissions for a given AuthContext.
        """
        self.backend = backend

    def filter_schemas(
        self, schemas: list[dict], context: AuthContext | None
    ) -> list[dict]:
        "Return only the JSON function schemas this context is allowed to call."
        if context is None:
            return schemas
        allowed = self.backend.get_allowed_tools(context)
        if "*" in allowed:
            return schemas
        return [s for s in schemas if s["function"]["name"] in allowed]

    def authorize(self, tool_name: str, context: AuthContext | None) -> bool:
        "Return True if this context may invoke the named tool."
        if context is None:
            return True
        return self.backend.is_allowed(tool_name, context)


__all__ = [
    "AuthContext",
    "AuthGate",
    "AuthProvider",
    "FilePolicyBackend",
    "MemoryPolicyBackend",
    "OIDCAuthProvider",
    "PolicyBackend",
    "StaticAuthProvider",
]
