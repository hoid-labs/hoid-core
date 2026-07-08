import asyncio
import json
import sqlite3
from pathlib import Path

try:
    import sqlite_vec
    from sqlite_vec import serialize_float32
except ImportError as _e:
    raise ImportError(
        "SqliteVecBackend requires the [rag] extra: pip install 'hoid[rag]'"
    ) from _e


_SQLITE_OPS: dict[str, str] = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def _to_sqlite_clause(filt: dict) -> tuple[str, list]:
    "Translate a normalized filter dict into (WHERE clause SQL, bind params)."
    clauses: list[str] = []
    params: list = []
    for key, cond in filt.items():
        path = f"$.{key}"
        for op, val in cond.items():
            if op in _SQLITE_OPS:
                clauses.append(f"json_extract(payload, ?) {_SQLITE_OPS[op]} ?")
                params.extend([path, val])
            elif op == "$in":
                placeholders = ",".join("?" for _ in val)
                clauses.append(f"json_extract(payload, ?) IN ({placeholders})")
                params.append(path)
                params.extend(list(val))
            elif op == "$nin":
                placeholders = ",".join("?" for _ in val)
                clauses.append(f"json_extract(payload, ?) NOT IN ({placeholders})")
                params.append(path)
                params.extend(list(val))
    return " AND ".join(clauses), params


class SqliteVecBackend:
    "SQLite-based vector storage using sqlite-vec; no server required, file or in-memory."

    # --- construction ---

    def __init__(
        self,
        path: str = ":memory:",
        vector_size: int = 768,
    ):
        """
        Args:
            path: SQLite database path. Defaults to ``:memory:`` (lost on process exit).
                  Set to a file path for persistence across restarts.
            vector_size: Dimensionality of embedding vectors; must match the embedding model output (default 768).
        """
        self._path = path
        self._vector_size = vector_size
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        # lazy init: deferred because __init__ is sync and extension loading is also sync
        if self._conn is not None:
            return self._conn
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        # python-build-standalone (used by uv) enables load_extension; macOS system Python does not
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id      TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        conn.commit()
        self._conn = conn
        return conn

    # --- storage interface ---

    async def upsert(
        self, ids: list[str], vectors: list[list[float]], payloads: list[dict]
    ):
        rows = [
            (id_, serialize_float32(vec), json.dumps(payload))
            for id_, vec, payload in zip(ids, vectors, payloads, strict=True)
        ]
        await asyncio.to_thread(self._upsert_sync, rows)

    def _upsert_sync(self, rows: list[tuple]):
        conn = self._connect()
        conn.executemany(
            "INSERT OR REPLACE INTO chunks(id, embedding, payload) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filter: dict | None = None,
    ) -> list[dict]:
        return await asyncio.to_thread(self._search_sync, query_vector, limit, filter)

    def _search_sync(
        self,
        query_vector: list[float],
        limit: int,
        filter: dict | None,
    ) -> list[dict]:
        conn = self._connect()
        sql = "SELECT payload FROM chunks"
        params: list = []
        if filter:
            clause, filter_params = _to_sqlite_clause(filter)
            sql += f" WHERE {clause}"
            params.extend(filter_params)
        sql += " ORDER BY vec_distance_cosine(embedding, ?) LIMIT ?"
        params.extend([serialize_float32(query_vector), limit])
        rows = conn.execute(sql, params).fetchall()
        return [json.loads(row[0]) for row in rows]

    async def update_payload_by_doc_id(
        self, doc_id: str, payload_update: dict
    ) -> None:
        """Merge ``payload_update`` into the JSON payload of every chunk tagged with ``doc_id``.

        Embeddings are preserved (no re-embedding); only the payload column is
        rewritten. Built-in payload keys (``text``, ``source``, ``doc_id``) are
        never overwritten because the caller passes only the metadata dict.
        """
        return await asyncio.to_thread(self._update_payload_sync, doc_id, payload_update)

    def _update_payload_sync(self, doc_id: str, payload_update: dict) -> None:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, payload FROM chunks WHERE json_extract(payload, '$.doc_id') = ?",
            (doc_id,),
        ).fetchall()
        if not rows:
            return
        updates: list[tuple] = []
        for chunk_id, payload_json in rows:
            payload = json.loads(payload_json)
            payload.update(payload_update)
            updates.append((json.dumps(payload), chunk_id))
        conn.executemany("UPDATE chunks SET payload = ? WHERE id = ?", updates)
        conn.commit()
