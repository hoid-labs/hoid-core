"""Shared filter grammar for vector-store metadata.

Backends translate a normalized filter dict into their native query layer
(Qdrant ``Filter``, ``sqlite-vec`` ``json_extract`` WHERE, AWS S3 Vectors
``FilterExpression``, etc.). Operators and value-shape rules live here so
the grammar is provider-agnostic and adding a new backend does not require
re-defining the operator set.

Grammar:

    {key: value}                          shorthand for {key: {"$eq": value}}
    {key: {"$eq": v}}                     equality
    {key: {"$ne": v}}                     inequality
    {key: {"$gt": v}}                     strict greater-than
    {key: {"$gte": v}}                    greater-than-or-equal
    {key: {"$lt": v}}                     strict less-than
    {key: {"$lte": v}}                    less-than-or-equal
    {key: {"$in": [v1, v2, ...]}}         membership
    {key: {"$nin": [v1, v2, ...]}}        non-membership

Top-level keys are AND-composed (all conditions must match); this matches
both Qdrant and S3 Vectors semantics, so we never need OR or nested groups.
"""

from __future__ import annotations

SUPPORTED_OPERATORS: frozenset[str] = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
)


class FilterValidationError(ValueError):
    "Raised when a filter dict does not match the shared grammar."


def normalize_filter(filt: dict | None) -> dict | None:
    """Validate ``filt`` and normalize the shorthand ``{key: value}`` form.

    Returns the canonical ``{key: {op: value}}`` form so backends can
    assume a single shape. Returns ``None`` for an unset filter.
    """
    if filt is None:
        return None
    if not isinstance(filt, dict) or not filt:
        raise FilterValidationError("filter must be a non-empty dict when provided")
    out: dict = {}
    for key, cond in filt.items():
        if not isinstance(key, str) or not key:
            raise FilterValidationError(f"filter keys must be non-empty strings, got {key!r}")
        if not isinstance(cond, dict):
            # shorthand: value -> {"$eq": value}
            out[key] = {"$eq": cond}
            continue
        if not cond:
            raise FilterValidationError(f"filter value for {key!r} must be non-empty")
        for op in cond:
            if op not in SUPPORTED_OPERATORS:
                raise FilterValidationError(
                    f"Unsupported filter operator {op!r} on {key!r}; "
                    f"supported: {sorted(SUPPORTED_OPERATORS)}"
                )
        out[key] = dict(cond)
    return out
