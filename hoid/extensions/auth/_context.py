from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuthContext:
    "Resolved identity and role set for a single request."

    user_id: str
    roles: set[str] = field(default_factory=set)
    # reserved for future attribute-based policy evaluation
    attributes: dict[str, Any] = field(default_factory=dict)
