from __future__ import annotations

import json
import pathlib

from ._context import AuthContext


class MemoryPolicyBackend:
    "In-process policy backend; same rules format as FilePolicyBackend but held in a dict."

    def __init__(self, policy: dict) -> None:
        """
        Args:
            policy: Dict with `roles` and `users` keys, matching the FilePolicyBackend JSON schema.
        """
        self._policy = policy

    def _role_tools(self, role: str) -> set[str]:
        return set(self._policy.get("roles", {}).get(role, {}).get("tools", []))

    def get_allowed_tools(self, context: AuthContext) -> set[str]:
        "Effective tool set: union of role grants + extra_tools − denied_tools."
        allowed: set[str] = set()
        for role in context.roles:
            allowed |= self._role_tools(role)
        user_entry = self._policy.get("users", {}).get(context.user_id, {})
        allowed |= set(user_entry.get("extra_tools", []))
        allowed -= set(user_entry.get("denied_tools", []))
        return allowed

    def is_allowed(self, tool_name: str, context: AuthContext) -> bool:
        allowed = self.get_allowed_tools(context)
        return "*" in allowed or tool_name in allowed


class FilePolicyBackend(MemoryPolicyBackend):
    "RBAC + per-user ACL policy loaded from a JSON file; extends MemoryPolicyBackend with disk persistence."

    def __init__(self, path: str | pathlib.Path) -> None:
        """
        Args:
            path: Path to a JSON policy file with `roles` and `users` top-level keys.
        """
        self._path = pathlib.Path(path)
        super().__init__(self._read())

    def _read(self) -> dict:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def reload(self) -> None:
        "Re-read the policy file from disk — call when the file changes at runtime."
        self._policy = self._read()
