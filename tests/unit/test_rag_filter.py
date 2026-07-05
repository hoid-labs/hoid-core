import pytest

from llm_framework.extensions.rag._filter import (
    SUPPORTED_OPERATORS,
    FilterValidationError,
    normalize_filter,
)


def test_none_returns_none():
    assert normalize_filter(None) is None


def test_shorthand_value_normalized_to_eq():
    assert normalize_filter({"status": "ready"}) == {"status": {"$eq": "ready"}}


def test_explicit_operators_pass_through():
    filt = {"published_at": {"$gte": "2026-07-01"}, "kind": {"$in": ["a", "b"]}}
    assert normalize_filter(filt) == filt


def test_unsupported_operator_rejected():
    with pytest.raises(FilterValidationError, match="Unsupported filter operator"):
        normalize_filter({"x": {"$regex": "abc"}})


def test_empty_dict_rejected():
    with pytest.raises(FilterValidationError, match="non-empty"):
        normalize_filter({})


def test_non_dict_value_rejected():
    with pytest.raises(FilterValidationError, match="non-empty"):
        normalize_filter({"x": {}})


def test_non_string_key_rejected():
    with pytest.raises(FilterValidationError, match="non-empty strings"):
        normalize_filter({"": {"$eq": 1}})
    with pytest.raises(FilterValidationError, match="non-empty strings"):
        normalize_filter({1: {"$eq": 1}})  # type: ignore[dict-item]


def test_unsupported_operator_lists_full_set():
    "Backends should rely on SUPPORTED_OPERATORS rather than re-validating."
    assert {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"} <= SUPPORTED_OPERATORS


def test_top_level_keys_are_and_composed_via_validation_only():
    "Two top-level keys must not interfere with each other; AND composition is the backend's job."
    filt = {"a": {"$eq": 1}, "b": {"$ne": 2}}
    out = normalize_filter(filt)
    assert set(out) == {"a", "b"}
