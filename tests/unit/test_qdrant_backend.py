"""Filter translation tests for the qdrant backend.

Tests the pure ``_to_qdrant_filter`` helper rather than instantiating a full
``QdrantBackend`` against a live cluster. The helper is the only provider-specific
piece the rest of the code calls; verifying its translation guarantees the
backend will issue a correct prefilter at search time.
"""

from __future__ import annotations

import pytest

try:
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchAny,
        MatchExcept,
        MatchValue,
        Range,
    )

    from hoid.extensions.rag.vector_store.qdrant import _to_qdrant_filter
except ImportError:
    _to_qdrant_filter = None  # type: ignore[assignment]

try:
    from hoid.extensions.rag.vector_store.qdrant import QdrantBackend
except ImportError:
    QdrantBackend = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    _to_qdrant_filter is None,
    reason="requires [qdrant] extra: uv pip install -e '.[qdrant]'",
)


def test_eq_becomes_match_value():
    cond = _to_qdrant_filter({"kind": {"$eq": "news"}})  # type: ignore[arg-type]
    assert isinstance(cond, Filter)
    assert len(cond.must) == 1
    fc = cond.must[0]
    assert isinstance(fc, FieldCondition)
    assert fc.key == "kind"
    assert fc.match == MatchValue(value="news")


def test_ne_becomes_match_except():
    cond = _to_qdrant_filter({"status": {"$ne": "deleted"}})  # type: ignore[arg-type]
    fc = cond.must[0]
    assert isinstance(fc, FieldCondition)
    assert fc.match == MatchExcept(**{"except": ["deleted"]})


def test_gt_gte_lt_lte_become_range():
    for op, key in [("$gt", "gt"), ("$gte", "gte"), ("$lt", "lt"), ("$lte", "lte")]:
        cond = _to_qdrant_filter({"score": {op: 0.5}})  # type: ignore[arg-type]
        fc = cond.must[0]
        assert isinstance(fc, FieldCondition)
        assert fc.range == Range(**{key: 0.5})  # type: ignore[arg-type]


def test_in_becomes_match_any():
    cond = _to_qdrant_filter({"tag": {"$in": ["a", "b", "c"]}})  # type: ignore[arg-type]
    fc = cond.must[0]
    assert isinstance(fc, FieldCondition)
    assert fc.match == MatchAny(any=["a", "b", "c"])


def test_nin_becomes_match_except_list():
    cond = _to_qdrant_filter({"tag": {"$nin": ["a", "b"]}})  # type: ignore[arg-type]
    fc = cond.must[0]
    assert isinstance(fc, FieldCondition)
    assert fc.match == MatchExcept(**{"except": ["a", "b"]})


def test_multiple_keys_are_and_composed():
    cond = _to_qdrant_filter(  # type: ignore[arg-type]
        {
            "epoch": {"$gte": 1717200000},
            "kind": {"$eq": "news"},
        }
    )
    assert len(cond.must) == 2
    keys = {fc.key for fc in cond.must if isinstance(fc, FieldCondition)}
    assert keys == {"epoch", "kind"}


@pytest.mark.skipif(
    QdrantBackend is None, reason="requires [qdrant] extra: uv pip install -e '.[qdrant]'"
)
async def test_update_payload_by_doc_id_uses_set_payload_with_doc_id_filter(monkeypatch):
    """Rewriting payload must hit Qdrant's ``set_payload`` with a doc_id filter and not re-upload points."""
    backend = QdrantBackend(collection_name="kb", vector_size=3)  # type: ignore[call-arg]
    backend._initialized = True  # avoid hitting _ensure_collection's collection_exists

    captured: dict = {}

    class _StubClient:
        async def set_payload(self, *, collection_name, payload, points):
            captured["collection_name"] = collection_name
            captured["payload"] = payload
            captured["points"] = points

    backend.db = _StubClient()  # type: ignore[assignment]

    await backend.update_payload_by_doc_id("doc-1", {"author": "Bob", "kind": "news"})

    assert captured["collection_name"] == "kb"
    assert captured["payload"] == {"author": "Bob", "kind": "news"}
    sel = captured["points"]
    assert isinstance(sel, Filter)
    assert len(sel.must) == 1
    fc = sel.must[0]
    assert isinstance(fc, FieldCondition)
    assert fc.key == "doc_id"
    assert fc.match == MatchValue(value="doc-1")
