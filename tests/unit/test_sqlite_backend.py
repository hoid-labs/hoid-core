import pprint

import pytest

try:
    from hoid.extensions.rag.vector_store.sqlite import SqliteVecBackend
except ImportError:
    SqliteVecBackend = None

pytestmark = pytest.mark.skipif(
    SqliteVecBackend is None,
    reason="requires [rag] extra: uv pip install -e '.[rag]'",
)

VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]
VEC_C = [0.0, 0.0, 1.0]
PAYLOAD_A = {"text": "alpha", "source": "test"}
PAYLOAD_B = {"text": "bravo", "source": "test"}
PAYLOAD_C = {"text": "charlie", "source": "test"}


@pytest.fixture
def backend():
    "Fresh in-memory backend for each test."
    return SqliteVecBackend(path=":memory:", vector_size=3)


async def _upsert_abc(b: SqliteVecBackend):
    await b.upsert(
        ids=["a", "b", "c"],
        vectors=[VEC_A, VEC_B, VEC_C],
        payloads=[PAYLOAD_A, PAYLOAD_B, PAYLOAD_C],
    )


async def test_nearest_neighbor_returned_first(backend):
    "Querying with VEC_A should rank 'alpha' highest."
    await _upsert_abc(backend)
    results = await backend.search(VEC_A, limit=3)
    print("search results (limit=3):")
    pprint.pprint(results)
    assert results[0]["text"] == "alpha"


async def test_limit_respected(backend):
    await _upsert_abc(backend)
    results = await backend.search(VEC_A, limit=1)
    print("search with limit=1:")
    pprint.pprint(results)
    assert len(results) == 1


async def test_all_payloads_present(backend):
    await _upsert_abc(backend)
    results = await backend.search(VEC_A, limit=10)
    texts = {r["text"] for r in results}
    print(f"all texts found: {sorted(texts)}")
    assert texts == {"alpha", "bravo", "charlie"}


async def test_upsert_replaces_on_same_id(backend):
    "Reinserting the same id must not create a duplicate row."
    await backend.upsert(["a"], [VEC_A], [PAYLOAD_A])
    await backend.upsert(["a"], [VEC_B], [{"text": "alpha-updated"}])
    results = await backend.search([0.0, 1.0, 0.0], limit=10)
    a_records = [r for r in results if r.get("text", "").startswith("alpha")]
    print("alpha records after re-upsert:")
    pprint.pprint(a_records)
    assert len(a_records) == 1
    assert a_records[0]["text"] == "alpha-updated"


async def test_empty_store_returns_empty(backend):
    results = await backend.search(VEC_A, limit=5)
    assert results == []


async def test_payload_roundtrip(backend):
    "Arbitrary payload keys survive serialisation."
    payload = {"text": "test", "score": 3.14, "tags": ["x", "y"], "nested": {"k": 1}}
    await backend.upsert(["x"], [VEC_A], [payload])
    results = await backend.search(VEC_A, limit=1)
    print("payload roundtrip:")
    pprint.pprint(results[0])
    assert results[0] == payload


async def test_limit_zero_returns_empty(backend):
    await _upsert_abc(backend)
    results = await backend.search(VEC_A, limit=0)
    assert results == []


async def test_upsert_multiple_then_delete_by_overwrite(backend):
    "A second upsert of the same IDs replaces all three entries without duplicates."
    await _upsert_abc(backend)
    await backend.upsert(
        ["a", "b", "c"],
        [VEC_C, VEC_C, VEC_C],
        [{"text": "a2"}, {"text": "b2"}, {"text": "c2"}],
    )
    results = await backend.search(VEC_C, limit=10)
    texts = {r["text"] for r in results}
    print(f"texts after double upsert: {sorted(texts)}  count: {len(results)}")
    assert texts == {"a2", "b2", "c2"}
    assert len(results) == 3


async def test_filter_eq_excludes_non_matches(backend):
    "An $eq filter on a metadata key returns only matching chunks."
    await backend.upsert(
        ["old", "new"],
        [VEC_A, VEC_A],
        [
            {"text": "old", "source": "old.pdf", "published_at": "2025-01-01"},
            {"text": "new", "source": "new.pdf", "published_at": "2026-07-05"},
        ],
    )
    results = await backend.search(
        VEC_A, limit=10, filter={"published_at": {"$eq": "2026-07-05"}}
    )
    print(f"$eq filter results: {[r['text'] for r in results]}")
    assert len(results) == 1
    assert results[0]["source"] == "new.pdf"


async def test_filter_gte_date_range(backend):
    "A $gte date-range filter is the canonical 'last week' prefilter."
    await backend.upsert(
        ["a", "b", "c"],
        [VEC_A, VEC_A, VEC_A],
        [
            {"text": "old", "source": "old.pdf", "published_at": "2025-01-01"},
            {"text": "this_week", "source": "this.pdf", "published_at": "2026-07-01"},
            {"text": "today", "source": "today.pdf", "published_at": "2026-07-05"},
        ],
    )
    results = await backend.search(
        VEC_A, limit=10, filter={"published_at": {"$gte": "2026-07-01"}}
    )
    texts = sorted(r["text"] for r in results)
    print(f"$gte filter results: {texts}")
    assert texts == ["this_week", "today"]


async def test_filter_in_and_nin(backend):
    "$in / $nin translate to SQL IN / NOT IN."
    await backend.upsert(
        ["a", "b", "c"],
        [VEC_A, VEC_A, VEC_A],
        [
            {"text": "x", "source": "x.pdf", "kind": "news"},
            {"text": "y", "source": "y.pdf", "kind": "blog"},
            {"text": "z", "source": "z.pdf", "kind": "paper"},
        ],
    )
    in_results = await backend.search(VEC_A, limit=10, filter={"kind": {"$in": ["news", "paper"]}})
    in_texts = sorted(r["text"] for r in in_results)
    print(f"$in results: {in_texts}")
    assert in_texts == ["x", "z"]

    nin_results = await backend.search(VEC_A, limit=10, filter={"kind": {"$nin": ["news"]}})
    nin_texts = sorted(r["text"] for r in nin_results)
    print(f"$nin results: {nin_texts}")
    assert nin_texts == ["y", "z"]


async def test_filter_combined_with_limit(backend):
    "Filter narrows candidates before the limit is applied; if filter excludes all, result is empty."
    await backend.upsert(
        ["a", "b"],
        [VEC_A, VEC_A],
        [
            {"text": "old", "source": "a.pdf", "published_at": "2020-01-01"},
            {"text": "new", "source": "b.pdf", "published_at": "2026-07-01"},
        ],
    )
    results = await backend.search(VEC_A, limit=1, filter={"published_at": {"$gte": "2026-01-01"}})
    assert len(results) == 1
    assert results[0]["text"] == "new"


async def test_filter_no_matches_returns_empty(backend):
    await backend.upsert(["a"], [VEC_A], [{"text": "x", "source": "a.pdf", "tag": "alpha"}])
    results = await backend.search(VEC_A, limit=10, filter={"tag": {"$eq": "beta"}})
    assert results == []


async def test_update_payload_by_doc_id_rewrites_metadata_in_place(backend):
    """Rewriting payload for a doc_id must update prefilter results without re-embedding."""
    await backend.upsert(
        ["a", "b", "c"],
        [VEC_A, VEC_A, VEC_A],
        [
            {"text": "doc1-a", "source": "a.pdf", "doc_id": "doc-1", "author": "Alice"},
            {"text": "doc1-b", "source": "a.pdf", "doc_id": "doc-1", "author": "Alice"},
            {"text": "doc2-a", "source": "b.pdf", "doc_id": "doc-2", "author": "Alice"},
        ],
    )

    # baseline: both docs match the existing author
    before = await backend.search(VEC_A, limit=10, filter={"author": {"$eq": "Alice"}})
    assert sorted(r["text"] for r in before) == ["doc1-a", "doc1-b", "doc2-a"]

    await backend.update_payload_by_doc_id("doc-1", {"author": "Bob", "kind": "news"})

    # doc-1 chunks now match Bob, doc-2 still matches Alice
    bob_hits = await backend.search(VEC_A, limit=10, filter={"author": {"$eq": "Bob"}})
    assert sorted(r["text"] for r in bob_hits) == ["doc1-a", "doc1-b"]

    alice_hits = await backend.search(VEC_A, limit=10, filter={"author": {"$eq": "Alice"}})
    assert sorted(r["text"] for r in alice_hits) == ["doc2-a"]

    # the new field is queryable, and built-in payload keys are preserved
    news_hits = await backend.search(VEC_A, limit=10, filter={"kind": {"$eq": "news"}})
    assert sorted(r["text"] for r in news_hits) == ["doc1-a", "doc1-b"]
    for hit in news_hits:
        assert hit["source"] == "a.pdf"
        assert hit["doc_id"] == "doc-1"


async def test_update_payload_by_doc_id_is_noop_for_missing_doc(backend):
    """An unknown doc_id must not raise or touch other chunks."""
    await backend.upsert(
        ["a"], [VEC_A], [{"text": "x", "source": "a.pdf", "doc_id": "doc-1", "k": "v"}]
    )
    await backend.update_payload_by_doc_id("doc-missing", {"k": "changed"})
    results = await backend.search(VEC_A, limit=10)
    assert results[0]["k"] == "v"
