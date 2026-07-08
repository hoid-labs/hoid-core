try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchAny,
        MatchExcept,
        MatchValue,
        PointStruct,
        Range,
        VectorParams,
    )
except ImportError as _e:
    raise ImportError(
        "QdrantBackend requires the [qdrant] extra: "
        "uv pip install 'hoid[qdrant]'"
    ) from _e


def _to_qdrant_filter(filt: dict) -> Filter:
    "Translate a normalized filter dict into a Qdrant ``Filter`` (AND-composed must clauses)."
    must: list = []
    for key, cond in filt.items():
        for op, val in cond.items():
            if op == "$eq":
                must.append(FieldCondition(key=key, match=MatchValue(value=val)))
            elif op == "$ne":
                # ``except`` is a Python keyword, so unpack via ** to reach the qdrant kwarg
                must.append(FieldCondition(key=key, match=MatchExcept(**{"except": [val]})))
            elif op in ("$gt", "$gte", "$lt", "$lte"):
                must.append(FieldCondition(key=key, range=Range(**{op.lstrip("$"): val})))
            elif op == "$in":
                must.append(FieldCondition(key=key, match=MatchAny(any=list(val))))
            elif op == "$nin":
                must.append(FieldCondition(key=key, match=MatchExcept(**{"except": list(val)})))
    return Filter(must=must)


class QdrantBackend:
    "Qdrant-based vector storage; supports in-memory, local file, and remote cluster modes."

    # --- construction ---

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        vector_size: int = 768,
        path: str | None = None,
        url: str | None = None,
    ):
        """
        Args:
            collection_name: Qdrant collection to use (default `knowledge_base`).
            vector_size: Dimensionality of embedding vectors; must match the embedding model output (default 768).
            path: Local file path for a persistent on-disk store. Omit for in-memory.
            url: Remote Qdrant cluster URL. Takes precedence over `path`.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._initialized = False

        if url:
            # remote cluster or cloud
            self.db = AsyncQdrantClient(url=url)
        elif path:
            # local file-backed store, persists across restarts
            self.db = AsyncQdrantClient(path=path)
        else:
            # in-memory, lost on process exit
            self.db = AsyncQdrantClient(location=":memory:")

    # --- internal ---

    async def _ensure_collection(self):
        # lazy because __init__ is sync but collection creation is async
        if not self._initialized:
            if not await self.db.collection_exists(self.collection_name):
                await self.db.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
            self._initialized = True

    # --- storage interface ---

    async def upsert(
        self, ids: list[str], vectors: list[list[float]], payloads: list[dict]
    ):
        await self._ensure_collection()
        points = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        await self.db.upsert(collection_name=self.collection_name, points=points)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter: dict | None = None,
    ) -> list[dict]:
        await self._ensure_collection()
        response = await self.db.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=_to_qdrant_filter(filter) if filter else None,
            limit=limit,
        )
        return [p for p in (hit.payload for hit in response.points) if p is not None]

    async def update_payload_by_doc_id(
        self, doc_id: str, payload_update: dict
    ) -> None:
        """Merge ``payload_update`` into every point tagged with ``doc_id``.

        Uses Qdrant's ``set_payload`` filter API to rewrite payload fields in
        place; vectors are preserved (no re-embedding). Built-in payload keys
        (``text``, ``source``, ``doc_id``) are never overwritten because the
        caller passes only the metadata dict.
        """
        await self._ensure_collection()
        await self.db.set_payload(
            collection_name=self.collection_name,
            payload=payload_update,
            points=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
