import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from llm_framework._env import load_env
from llm_framework._optional import require as _require
from llm_framework.core.llm import LLMClient
from llm_framework.core.observability import RAGEvent
from llm_framework.core.protocols import EmbeddingClientProtocol

from ._converter import to_markdown
from ._filter import FilterValidationError as FilterValidationError
from ._filter import normalize_filter

log = logging.getLogger(__name__)

try:
    from semantic_text_splitter import MarkdownSplitter
except ImportError:
    MarkdownSplitter = None  # type: ignore[misc,assignment]

try:
    from llm_framework.extensions.rag.vector_store.sqlite import SqliteVecBackend
except ImportError:
    SqliteVecBackend = None  # type: ignore[misc,assignment]

try:
    from llm_framework.extensions.rag.vector_store.qdrant import QdrantBackend
except ImportError:
    QdrantBackend = None  # type: ignore[misc,assignment]


@runtime_checkable
class BaseStorageBackend(Protocol):
    "Interface for vector database backends."

    async def upsert(
        self, ids: list[str], vectors: list[list[float]], payloads: list[dict]
    ): ...

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter: dict | None = None,
    ) -> list[dict]: ...

    async def update_payload_by_doc_id(
        self, doc_id: str, payload_update: dict
    ):
        """Merge ``payload_update`` into the payload of every chunk tagged with ``doc_id``.

        Used to rewrite per-document metadata without re-embedding: the chunk
        text and vectors are unchanged, only the JSON payload fields are
        updated in place. Built-in payload keys (``text``, ``source``, ``doc_id``)
        are preserved by the caller — ``payload_update`` is the new metadata
        dict, not the full replacement payload.
        """


def backend_from_env(collection: str | None = None) -> BaseStorageBackend:
    """Construct the configured vector backend from environment variables.

    Args:
        collection: Optional logical name for the vector namespace (collection or database).
            When provided, backends use it to isolate data — sqlite appends it as
            ``<SQLITE_PATH>/<collection>.db``; qdrant uses it as the collection name.
            When omitted, each backend falls back to its own env-var default.
    """
    name = os.getenv("VECTOR_BACKEND", "sqlite").lower()

    if name == "sqlite":
        _require("sqlite_vec", SqliteVecBackend)
        if collection:
            base = os.getenv("SQLITE_PATH", "./data/sqlite")
            path = str(Path(base) / f"{collection}.db")
        else:
            path = os.getenv("SQLITE_PATH", ":memory:")
        return SqliteVecBackend(
            path=path,
            vector_size=int(os.getenv("VECTOR_SIZE", "768")),
        )

    if name == "qdrant":
        _require("qdrant_client", QdrantBackend)
        return QdrantBackend(
            collection_name=collection
            or os.getenv("QDRANT_COLLECTION")
            or "knowledge_base",
            vector_size=int(os.getenv("VECTOR_SIZE", "768")),
            path=os.getenv("QDRANT_PATH"),
            url=os.getenv("QDRANT_URL"),
        )

    raise ValueError(f"Unknown VECTOR_BACKEND: '{name}'")


class RAGStore:
    "Retrieval-augmented generation store: ingest files and search by semantic similarity."

    def __init__(
        self,
        llm_client: EmbeddingClientProtocol,
        storage_backend: BaseStorageBackend,
        default_max_tokens: int = 300,
        embed_batch_size: int = 64,
        _owns_client: bool = False,
        on_rag: Any | None = None,
        allowed_base: str | Path | None = None,
    ):
        """
        Args:
            llm_client: An `LLMClient` instance used exclusively for embedding requests.
            storage_backend: A `BaseStorageBackend` implementation (e.g. `QdrantBackend`).
            default_max_tokens: Approximate token limit per chunk when ingesting files (default 300).
            embed_batch_size: Maximum strings per embedding API call; prevents exceeding provider limits (default 64).
            on_rag: Optional observability callback receiving `RAGEvent` on ingest/search.
            allowed_base: Root directory that `ingest_file` paths must reside under.
                Defaults to the current user's home directory. Override when running in
                containers where data lives outside the home directory (e.g. ``/app/data``).
        """
        self.llm = llm_client
        self.storage = storage_backend
        self.default_max_tokens = default_max_tokens
        self._embed_batch_size = embed_batch_size
        self._owns_client = _owns_client
        self.on_rag = on_rag
        self._allowed_base = Path(allowed_base).resolve() if allowed_base is not None else Path.home()
        _require("semantic_text_splitter", MarkdownSplitter)

    @classmethod
    def from_env(cls, storage_backend: BaseStorageBackend) -> "RAGStore":
        "Construct a RAGStore from env vars; must be used as an async context manager to avoid leaking the HTTP client."
        load_env()
        client = LLMClient(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.getenv("EMBED_MODEL", os.environ["LLM_MODEL"]),
            verify=os.getenv("CA_BUNDLE_PATH") or True,
        )
        return cls(client, storage_backend, _owns_client=True)

    # --- lifecycle ---

    async def __aenter__(self):
        if self._owns_client:
            await self.llm.__aenter__()
        return self

    async def __aexit__(self, *exc):
        if self._owns_client:
            await self.llm.__aexit__(*exc)

    # --- operations ---

    async def ingest_file(
        self,
        file_path: str | Path,
        max_tokens: int | None = None,
        metadata: dict | None = None,
    ):
        """Convert, chunk, embed, and store a file; returns the number of chunks ingested.

        Args:
            file_path: Path to the file to ingest.
            max_tokens: Optional override for the per-chunk token limit.
            metadata: Optional dict merged into every chunk payload (alongside the
                built-in ``text``, ``source``, and ``doc_id`` fields). Use it for
                attributes that callers may want to filter on later
                (e.g. ``{"published_at": "2026-07-01"}``).
        """
        path_obj = Path(file_path)

        # sandbox to allowed_base so library callers can't read arbitrary filesystem paths
        try:
            path_obj.resolve().relative_to(self._allowed_base)
        except ValueError:
            raise PermissionError(
                f"Path '{path_obj.resolve()}' is outside the allowed base directory '{self._allowed_base}'"
            ) from None

        if not path_obj.exists():
            raise FileNotFoundError(f"Cannot find file: {file_path}")

        try:
            text_content = to_markdown(path_obj)
        except Exception as exc:
            log.warning("Failed to extract text from %s: %s", file_path, exc)
            return 0

        if not text_content or not text_content.strip():
            return 0

        limit = max_tokens if max_tokens is not None else self.default_max_tokens
        overlap_limit = int(limit * 0.15)

        # char÷4 approximates tokens without a tokenizer
        splitter = MarkdownSplitter.from_callback(
            lambda text: len(text) // 4, limit, overlap=overlap_limit
        )
        chunks = splitter.chunks(text_content)

        if not chunks:
            return 0

        # batch calls to stay within provider per-request token limits
        vectors: list[list[float]] = []
        start = time.perf_counter()
        for i in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[i : i + self._embed_batch_size]
            vectors.extend(await self.llm.embeddings(batch))
        ingest_ms = (time.perf_counter() - start) * 1000.0
        # deterministic UUIDs make re-ingestion idempotent and satisfy Qdrant point id format
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path_obj)))
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path_obj.name}:{chunk}"))
            for chunk in chunks
        ]
        # doc_id lets search callers group hits by source document; metadata fields
        # are replicated to every chunk so backends can prefilter at vector-search time
        extra: dict[str, Any] = {"doc_id": doc_id}
        if metadata:
            extra.update(metadata)
        payloads = [{"text": chunk, "source": path_obj.name, **extra} for chunk in chunks]

        await self.storage.upsert(ids, vectors, payloads)
        if self.on_rag:
            await self.on_rag(
                RAGEvent(
                    op="ingest",
                    collection_id=getattr(self.storage, "collection_id", "default")
                    or "default",
                    doc_id=doc_id,
                    chunk_count=len(chunks),
                    latency_ms=ingest_ms,
                )
            )
        return len(chunks)

    async def search(
        self,
        query: str,
        limit: int = 3,
        filter: dict | None = None,
    ):
        """Return the top-k most semantically similar passages for a query string.

        Args:
            query: Natural-language query to embed and search.
            limit: Max number of passages to return.
            filter: Optional metadata filter dict using the shared grammar; see
                ``extensions/rag/_filter.py`` for supported operators. Backends
                translate this to a native prefilter where possible so semantic
                search only runs over matching documents.
        """
        normalized = normalize_filter(filter)
        start = time.perf_counter()
        query_vector = (await self.llm.embeddings([query]))[0]
        search_ms = (time.perf_counter() - start) * 1000.0
        payloads = await self.storage.search(query_vector, limit, filter=normalized)
        if self.on_rag:
            await self.on_rag(
                RAGEvent(
                    op="search",
                    collection_id=getattr(self.storage, "collection_id", "default")
                    or "default",
                    latency_ms=search_ms,
                )
            )
        return [
            f"Source: {payload.get('source', 'Unknown')}\n\n{payload.get('text', '')}"
            for payload in payloads
        ]
