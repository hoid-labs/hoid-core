"""Backward-compat re-export. Prefer `hoid.core.observability` for new code."""
from hoid.core.observability import *  # noqa: F401, F403
from hoid.core.observability import (  # private names used by tests
    _attach_ctx,  # noqa: F401
    _context_var,  # noqa: F401
)
