"""End-to-end RAGStore tests exercising the metadata + filter contract.

Verifies that:
- ``RAGStore.ingest_file(metadata=...)`` writes the metadata to every chunk.
- ``RAGStore.search(filter=...)`` normalizes shorthand ``{key: value}`` to ``{$eq: value}``.
- Unsupported operators are rejected at the normalize boundary, before the backend.
"""

from __future__ import annotations

import pytest

try:
    from hoid.extensions.rag import RAGStore
    from hoid.extensions.rag.vector_store.sqlite import SqliteVecBackend
except ImportError:
    RAGStore = None  # type: ignore[assignment]
    SqliteVecBackend = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    SqliteVecBackend is None or RAGStore is None,
    reason="requires [rag] extra",
)

from hoid.extensions.rag._filter import (  # noqa: E402
    FilterValidationError,
    normalize_filter,
)


class FakeEmbedClient:
    "Minimal EmbeddingClient that returns a deterministic vector for any input."

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def embeddings(self, inputs: list[str]) -> list[list[float]]:
        # All inputs collapse to the same vector so cosine similarity is 1 between any two;
        # ordering is decided by filter prefilter or by row order on ties.
        return [[1.0, 0.0, 0.0] for _ in inputs]


async def _seed_three(backend: SqliteVecBackend):
    "Three chunks with ``epoch`` (numeric) ``published_at`` and string ``kind``."
    await backend.upsert(
        ["old", "newer", "newest"],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [
            {"text": "old_doc", "source": "old.pdf", "epoch": 1_700_000_000, "kind": "news"},
            {"text": "newer_doc", "source": "newer.pdf", "epoch": 1_730_000_000, "kind": "blog"},
            {"text": "newest_doc", "source": "newest.pdf", "epoch": 1_750_000_000, "kind": "paper"},
        ],
    )


async def test_search_normalizes_shorthand_filter():
    backend = SqliteVecBackend(path=":memory:", vector_size=3)
    await _seed_three(backend)
    store = RAGStore(
        llm_client=FakeEmbedClient(),  # type: ignore[arg-type]
        storage_backend=backend,
        allowed_base="/tmp",
    )
    hits = await store.search("anything", limit=5, filter={"epoch": 1_750_000_000})
    sources = [h.split("\n", 1)[0] for h in hits]
    print(f"shorthand-filtered hits: {sources}")
    assert sources == ["Source: newest.pdf"]


async def test_search_supports_gte_range():
    backend = SqliteVecBackend(path=":memory:", vector_size=3)
    await _seed_three(backend)
    store = RAGStore(
        llm_client=FakeEmbedClient(),  # type: ignore[arg-type]
        storage_backend=backend,
        allowed_base="/tmp",
    )
    hits = await store.search(
        "anything", limit=5, filter={"epoch": {"$gte": 1_730_000_000}}
    )
    sources = sorted(h.split("\n", 1)[0] for h in hits)
    assert sources == ["Source: newer.pdf", "Source: newest.pdf"]


async def test_search_supports_in_operator():
    backend = SqliteVecBackend(path=":memory:", vector_size=3)
    await _seed_three(backend)
    store = RAGStore(
        llm_client=FakeEmbedClient(),  # type: ignore[arg-type]
        storage_backend=backend,
        allowed_base="/tmp",
    )
    hits = await store.search(
        "anything", limit=5, filter={"kind": {"$in": ["news", "paper"]}}
    )
    sources = sorted(h.split("\n", 1)[0] for h in hits)
    assert sources == ["Source: newest.pdf", "Source: old.pdf"]


async def test_search_rejects_unsupported_operator_at_boundary():
    backend = SqliteVecBackend(path=":memory:", vector_size=3)
    await _seed_three(backend)
    store = RAGStore(
        llm_client=FakeEmbedClient(),  # type: ignore[arg-type]
        storage_backend=backend,
        allowed_base="/tmp",
    )
    with pytest.raises(FilterValidationError):
        await store.search("q", filter={"field": {"$regex": ".*"}})


async def test_normalize_is_pure():
    "The normalized dict is what the backend receives; this is the contract backends rely on."
    raw = {"epoch": 1_750_000_000, "kind": {"$in": ["news", "blog"]}}
    canonical = normalize_filter(raw)
    assert canonical == {
        "epoch": {"$eq": 1_750_000_000},
        "kind": {"$in": ["news", "blog"]},
    }
    # Original dict is not mutated.
    assert raw == {"epoch": 1_750_000_000, "kind": {"$in": ["news", "blog"]}}


async def test_ingest_writes_metadata_to_every_chunk(tmp_path):
    "ingest_file takes metadata and stores the fields on every chunk payload."
    backend = SqliteVecBackend(path=":memory:", vector_size=3)
    store = RAGStore(
        llm_client=FakeEmbedClient(),  # type: ignore[arg-type]
        storage_backend=backend,
        allowed_base=tmp_path,
    )
    # Content long enough that the default chunker produces more than one chunk
    file_path = tmp_path / "doc.txt"
    file_path.write_text(
        "alpha segment one. " * 200 + "\n\n" + "bravo segment two. " * 200
    )

    count = await store.ingest_file(
        file_path, metadata={"epoch": 1_750_000_000, "tag": "alpha"}
    )
    print(f"chunk count from ingest: {count}")
    assert count >= 2

    hits = await backend.search([1.0, 0.0, 0.0], limit=10)
    relevant = [h for h in hits if "alpha" in h.get("text", "") or "bravo" in h.get("text", "")]
    print(f"relevant hits count: {len(relevant)}")
    assert len(relevant) >= 2
    for chunk in relevant:
        assert chunk["epoch"] == 1_750_000_000
        assert chunk["tag"] == "alpha"
        assert chunk["doc_id"]
        assert chunk["source"] == "doc.txt"
