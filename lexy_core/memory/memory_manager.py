"""
Lexy AI - MemoryManager.

ChromaDB-backed memory with optional BM25 (SQLite FTS5) hybrid search.

Layers
------
* **Vector store** – ChromaDB HTTP server (4 collections: ``facts``,
  ``solutions``, ``errors``, ``context``).
* **FTS5 store**  – ``aiosqlite`` table mirroring stored items for keyword
  search.
* **HybridSearch** – combines normalised vector + BM25 scores
  (70/30 by default).

Embeddings come from the LexyApp-owned ``EmbeddingClient`` (Jina v3 / Jina v5).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import chromadb
import structlog
from chromadb.api.client import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError as ChromaNotFoundError

from lexy_core.config import MemoryConfig
from lexy_core.project import DEFAULT_PROJECT_ID
from lexy_core.utils.logging import get_logger

log: structlog.BoundLogger = get_logger(module="memory_manager")


# Sentinel a caller passes to ``recall`` to explicitly skip project
# scoping ("I want hits from every project"). ``None`` already means
# "scope disabled", but the sentinel survives serialisation through
# plugin APIs where ``None`` might get coerced to a default.
CROSS_PROJECT_SCOPE = "__all__"


@dataclass
class MemoryItem:
    """One scored memory result."""

    id: str
    content: str
    collection: str
    metadata: dict[str, Any]
    score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    created_at: float = 0.0


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a dict of scores into [0, 1].

    Edge case when all values are equal (hi ≈ lo):
    * If the common value is > 0 → return **1.0** for every key. These
      are real matches that all scored equally (e.g. a single recall hit);
      treating them as "best" is correct.
    * If the common value is ≤ 0 → return **0.0**. These are genuine
      no-match results and should stay below the threshold.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        # All scores identical — are they real matches or zero-relevance?
        fallback = 1.0 if hi > 1e-9 else 0.0
        return {key: fallback for key in scores}
    return {key: (val - lo) / (hi - lo) for key, val in scores.items()}


# SQLite FTS5 MATCH syntax treats many punctuation characters as operators.
# We split the input into alphanumeric tokens, strip everything else, and
# feed FTS5 a safe OR-joined phrase query so any single term can match.
_FTS_TOKEN_RE = re.compile(r"[^\w\u00C0-\u024F]+", re.UNICODE)


def _sanitize_fts_query(query: str) -> str:
    """
    Turn arbitrary user text into a safe FTS5 MATCH expression.

    Returns an empty string when no usable tokens remain — callers must treat
    an empty result as "skip the FTS branch of the hybrid search".
    """
    tokens = [token for token in _FTS_TOKEN_RE.split(query) if len(token) >= 2]
    if not tokens:
        return ""
    # Wrap each token in double quotes, escape any existing quotes, OR them.
    quoted = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
    return " OR ".join(quoted)


def _meta_matches(metadata: Any, key: str, value: Any) -> bool:
    """Post-filter helper: True if ``metadata[key] == value`` after a safe
    unwrap. Used to scrub BM25 hits that ChromaDB's vector ``where`` clause
    can't filter (the FTS mirror only indexes project_id, so character_id
    filtering has to happen here).
    """
    if not isinstance(metadata, dict):
        return False
    return metadata.get(key) == value


class MemoryManager:
    """
    High-level facade for the four core collections + hybrid recall.

    Plugins use ``PluginAPI.memory_store/recall``; LexyApp wires this manager
    during startup. Embeddings are computed by the ``EmbeddingClient``.
    """

    def __init__(self, config: MemoryConfig, embedding_client: Any) -> None:
        self._config = config
        self._embedding = embedding_client
        self._client: ClientAPI | None = None
        self._collections: dict[str, Collection] = {}
        self._fts: aiosqlite.Connection | None = None
        # Optional session store. When wired, ``store()`` can resolve the
        # current ``project_id`` from a session metadata lookup whenever
        # the caller passes ``session_id`` but not ``project_id``.
        self._session_store: Any = None

    def set_session_store(self, session_store: Any) -> None:
        """Wire the :class:`SessionStore` for automatic project resolution."""
        self._session_store = session_store

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open ChromaDB connection, ensure collections, init FTS5 mirror."""
        log.info(
            "memory.connecting",
            host=self._config.chroma_host,
            port=self._config.chroma_port,
        )
        self._client = chromadb.HttpClient(
            host=self._config.chroma_host,
            port=self._config.chroma_port,
        )

        try:
            self._client.heartbeat()
        except Exception as exc:  # noqa: BLE001
            log.error("memory.chromadb_unreachable", error=str(exc))
            raise RuntimeError(
                f"ChromaDB not reachable at "
                f"{self._config.chroma_host}:{self._config.chroma_port}"
            ) from exc

        for name in self._config.collections:
            self._collections[name] = self._client.get_or_create_collection(name=name)
            log.info("memory.collection_ready", collection=name)

        await self._init_fts()
        log.info("memory.ready", collections=list(self._collections.keys()))

    async def _init_fts(self) -> None:
        """Create / open the FTS5 mirror used for BM25.

        The schema was extended with a ``project_id`` column in Phase 4.
        FTS5 virtual tables do **not** support ``ALTER TABLE ADD COLUMN``,
        so we rebuild the table if an older schema without the column is
        detected. All existing rows are migrated into the default project.
        """
        db_path = Path(self._config.fts_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts = await aiosqlite.connect(str(db_path))

        table_exists = False
        cursor = await self._fts.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='items_fts'"
        )
        row = await cursor.fetchone()
        await cursor.close()
        table_exists = row is not None

        has_project_column = False
        if table_exists:
            cursor = await self._fts.execute("PRAGMA table_info(items_fts)")
            cols = [r[1] for r in await cursor.fetchall()]
            await cursor.close()
            has_project_column = "project_id" in cols

        if table_exists and not has_project_column:
            log.warning(
                "memory.fts_migrating",
                reason="add project_id column",
                fallback_project=DEFAULT_PROJECT_ID,
            )
            # Rename old table, create new one with project_id, copy rows.
            await self._fts.execute(
                "ALTER TABLE items_fts RENAME TO items_fts_legacy"
            )
            await self._fts.execute(
                """
                CREATE VIRTUAL TABLE items_fts USING fts5(
                    id UNINDEXED,
                    collection UNINDEXED,
                    content,
                    created_at UNINDEXED,
                    project_id UNINDEXED
                )
                """
            )
            await self._fts.execute(
                """
                INSERT INTO items_fts(id, collection, content, created_at, project_id)
                SELECT id, collection, content, created_at, ?
                FROM items_fts_legacy
                """,
                (DEFAULT_PROJECT_ID,),
            )
            await self._fts.execute("DROP TABLE items_fts_legacy")
            await self._fts.commit()
            log.info("memory.fts_migration_complete")
        else:
            await self._fts.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                    id UNINDEXED,
                    collection UNINDEXED,
                    content,
                    created_at UNINDEXED,
                    project_id UNINDEXED
                )
                """
            )
            await self._fts.commit()

    async def shutdown(self) -> None:
        if self._fts is not None:
            await self._fts.close()
            self._fts = None
        self._client = None
        self._collections.clear()
        log.info("memory.shutdown")

    # ─── Wipe operations ───────────────────────────────────────────

    async def ensure_collection(self, name: str) -> None:
        """Idempotently register a (possibly new) ChromaDB collection.

        Phase 13 introduced per-RP-session collections (``rp__<id>``)
        that are created on demand instead of being declared up-front
        in config. This method is the entry point: it registers the
        name in the in-process cache so ``store`` / ``recall`` can
        target it like any built-in collection.
        """
        if self._client is None:
            raise RuntimeError("MemoryManager not initialised")
        if name in self._collections:
            return
        try:
            self._collections[name] = self._client.get_or_create_collection(
                name=name
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "memory.ensure_collection_failed",
                collection=name,
                error=str(exc),
            )
            raise
        log.info("memory.collection_ensured", collection=name)

    async def delete_collection(self, name: str) -> dict[str, int]:
        """Permanently delete a collection AND its FTS rows.

        Used by the RP session container's ``destroy()`` — when Mike
        deletes a session, the per-session collection vanishes
        completely (no recreate, unlike :meth:`wipe_collection`).

        Returns ``{"chroma": <count_before>, "fts": <rows_deleted>}``.
        """
        if self._client is None:
            raise RuntimeError("MemoryManager not initialised")
        prev_count = 0
        col = self._collections.get(name)
        if col is not None:
            try:
                prev_count = col.count()
            except Exception:  # noqa: BLE001
                prev_count = 0
        try:
            self._client.delete_collection(name=name)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "memory.delete_collection_chroma_failed",
                collection=name,
                error=str(exc),
            )
        self._collections.pop(name, None)

        fts_deleted = 0
        if self._fts is not None:
            cursor = await self._fts.execute(
                "DELETE FROM items_fts WHERE collection = ?",
                (name,),
            )
            fts_deleted = cursor.rowcount if cursor.rowcount is not None else 0
            await self._fts.commit()

        log.info(
            "memory.collection_deleted",
            collection=name,
            chroma_items=int(prev_count),
            fts_rows=int(fts_deleted),
        )
        return {"chroma": int(prev_count), "fts": int(fts_deleted)}

    async def wipe_collection(self, name: str) -> dict[str, int]:
        """
        Drop and recreate a single ChromaDB collection and delete its
        rows from the FTS5 mirror. Returns ``{"chroma": <prev_count>,
        "fts": <prev_fts_count>}``.
        """
        if self._client is None:
            raise RuntimeError("MemoryManager not initialised")
        if name not in self._collections:
            raise KeyError(f"unknown collection: {name}")

        collection = self._collections[name]
        try:
            prev_count = collection.count()
        except Exception:  # noqa: BLE001
            prev_count = 0

        # Drop + recreate — cleanest way to remove ALL items, schema,
        # metadata, and any lingering index state.
        try:
            self._client.delete_collection(name=name)
        except Exception as exc:  # noqa: BLE001
            log.warning("memory.wipe_delete_failed", collection=name, error=str(exc))
        self._collections[name] = self._client.get_or_create_collection(name=name)

        # Clear matching FTS rows
        fts_deleted = 0
        if self._fts is not None:
            cursor = await self._fts.execute(
                "DELETE FROM items_fts WHERE collection = ?",
                (name,),
            )
            fts_deleted = cursor.rowcount if cursor.rowcount is not None else 0
            await self._fts.commit()

        log.warning(
            "memory.wiped_collection",
            collection=name,
            chroma_items=int(prev_count),
            fts_rows=int(fts_deleted),
        )
        return {"chroma": int(prev_count), "fts": int(fts_deleted)}

    async def delete_last_for_session(
        self,
        session_id: str,
        collection: str = "context",
    ) -> int:
        """
        Remove the most recent memory item written for a given session.

        Used by the regenerate flow: LexyAgent._reflect() stores every
        turn as ``collection='context', metadata={'session_id': ..., ...}``.
        When the user regenerates, we drop that last auto-memorized entry
        so the fresh answer isn't polluted by the stale one.

        Returns the number of items actually removed (0 if nothing matched).
        """
        if self._client is None:
            return 0
        col = self._collections.get(collection)
        if col is None:
            return 0

        try:
            got = col.get(
                where={"session_id": session_id},
                include=["metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "memory.delete_last_failed",
                session_id=session_id,
                error=str(exc),
            )
            return 0

        ids: list[str] = list(got.get("ids") or [])
        metas: list[dict[str, Any]] = list(got.get("metadatas") or [])
        if not ids:
            return 0

        # Pick the most recent by metadata.created_at (falls back to
        # last id in the list if created_at is missing).
        scored = [
            (float(meta.get("created_at", 0) or 0), item_id)
            for item_id, meta in zip(ids, metas)
        ]
        scored.sort(reverse=True)
        target_id = scored[0][1]

        try:
            col.delete(ids=[target_id])
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "memory.delete_last_chroma_failed",
                id=target_id,
                error=str(exc),
            )
            return 0

        if self._fts is not None:
            try:
                await self._fts.execute(
                    "DELETE FROM items_fts WHERE id = ?", (target_id,)
                )
                await self._fts.commit()
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "memory.delete_last_fts_failed",
                    id=target_id,
                    error=str(exc),
                )

        log.info(
            "memory.deleted_last_for_session",
            session_id=session_id,
            id=target_id,
        )
        return 1

    async def wipe_all(self) -> dict[str, Any]:
        """
        Drop every configured collection and clear the entire FTS5
        mirror. Returns per-collection counts plus the total FTS row
        count dropped.
        """
        if self._client is None:
            raise RuntimeError("MemoryManager not initialised")

        totals: dict[str, Any] = {"collections": {}, "total_chroma": 0}

        for name in list(self._collections.keys()):
            try:
                counts = await self.wipe_collection(name)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "memory.wipe_collection_failed",
                    collection=name,
                    error=str(exc),
                )
                counts = {"chroma": 0, "fts": 0, "error": str(exc)}
            totals["collections"][name] = counts
            totals["total_chroma"] += int(counts.get("chroma", 0))

        # Belt-and-suspenders: clear the FTS table entirely in case any
        # stray rows exist outside the configured collection names.
        fts_total = 0
        if self._fts is not None:
            cursor = await self._fts.execute("DELETE FROM items_fts")
            fts_total = cursor.rowcount if cursor.rowcount is not None else 0
            await self._fts.commit()
            try:
                await self._fts.execute("VACUUM")
                await self._fts.commit()
            except Exception as exc:  # noqa: BLE001
                log.debug("memory.vacuum_skipped", error=str(exc))

        totals["total_fts"] = int(fts_total)
        log.warning(
            "memory.wiped_all",
            total_chroma=totals["total_chroma"],
            total_fts=totals["total_fts"],
        )
        return totals

    # ─── Helpers ────────────────────────────────────────────────────

    def _require_collection(self, name: str) -> Collection:
        collection = self._collections.get(name)
        if collection is None:
            raise KeyError(
                f"Unknown collection {name!r}. Known: {list(self._collections)}"
            )
        return collection

    def _refresh_collection(self, name: str) -> Collection | None:
        """Re-fetch a collection from ChromaDB and update the cache.

        Called when an operation raises :class:`ChromaNotFoundError` — this
        usually means the server-side collection was deleted (manual wipe,
        docker volume reset) while we kept a stale handle. ``get_or_create_collection``
        repairs the state: if the collection is truly gone, we create a
        fresh empty one under the same name; if it merely lost its cached
        uuid, we get a valid handle back.
        """
        if self._client is None:
            return None
        try:
            col = self._client.get_or_create_collection(name=name)
        except Exception as exc:  # noqa: BLE001
            log.error("memory.refresh_failed", collection=name, error=str(exc))
            return None
        self._collections[name] = col
        log.info("memory.collection_refreshed", collection=name)
        return col

    # ─── Store / Recall ─────────────────────────────────────────────

    async def store(
        self,
        text: str,
        collection: str = "facts",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store one item. Returns its ChromaDB id.

        Project-scoping rules
        ---------------------
        1. An explicit ``project_id`` in ``metadata`` always wins.
        2. Otherwise, if ``session_id`` is set and a session store is
           wired, look up the session's project.
        3. Otherwise, fall back to :data:`DEFAULT_PROJECT_ID`.

        The resolved project id is stored both in the ChromaDB metadata
        and in the FTS mirror so hybrid recall can scope by project.
        """
        if not text:
            return ""
        item_id = uuid.uuid4().hex
        meta = dict(metadata or {})
        meta.setdefault("created_at", time.time())
        meta.setdefault("collection", collection)

        if "project_id" not in meta:
            session_id = meta.get("session_id")
            resolved: str | None = None
            if session_id and self._session_store is not None:
                try:
                    sm = self._session_store.get_meta(session_id)
                    candidate = sm.get("project_id") if isinstance(sm, dict) else None
                    if candidate:
                        resolved = candidate
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "memory.project_lookup_failed",
                        session_id=session_id,
                        error=str(exc),
                    )
            meta["project_id"] = resolved or DEFAULT_PROJECT_ID

        embedding = await self._embed_one(text)

        col = self._require_collection(collection)
        col.add(
            ids=[item_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[meta],
        )

        if self._fts is not None:
            await self._fts.execute(
                "INSERT INTO items_fts(id, collection, content, created_at, project_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    item_id,
                    collection,
                    text,
                    meta["created_at"],
                    meta["project_id"],
                ),
            )
            await self._fts.commit()

        log.debug(
            "memory.stored",
            id=item_id,
            collection=collection,
            project_id=meta["project_id"],
        )
        return item_id

    async def recall(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 5,
        project_id: str | None = None,
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid recall (vector + BM25). ``collection=None`` searches all.

        ``project_id``:
            * ``None`` → no filter applied (legacy behaviour; every project
              is visible). Useful for plugins that deliberately want global
              recall.
            * a concrete project id → only items tagged with that id OR
              with no ``project_id`` at all (legacy items from before the
              scoping rollout are considered shared).
            * :data:`CROSS_PROJECT_SCOPE` (``"__all__"``) → same as ``None``.

        ``metadata_equals``:
            Optional exact-match filter on ChromaDB metadata fields (e.g.
            ``{"character_id": "luna123"}`` to fetch only Luna's memory).
            Applied on the vector-search path as a ``$and`` extension of
            the project-scope clause; for the FTS path we post-filter on
            the merged result because the FTS mirror only stores
            ``project_id``. That's a small precision loss for BM25 hits
            that's acceptable given the small hit-sets involved.
        """
        if not query.strip():
            return []
        targets = [collection] if collection else list(self._collections.keys())

        scope = project_id
        if scope == CROSS_PROJECT_SCOPE:
            scope = None

        # Fetch more candidates when we're post-filtering so the final
        # window after filtering isn't starved.
        overfetch = 1 + len(metadata_equals or {})
        fetch = limit * 3 * overfetch

        vector_hits = await self._vector_search(
            query, targets, fetch, scope, metadata_equals
        )
        bm25_hits = (
            await self._fts_search(query, targets, fetch, scope)
            if self._fts
            else []
        )

        merged = self._merge(vector_hits, bm25_hits, fetch)
        if metadata_equals:
            merged = [
                item
                for item in merged
                if all(
                    _meta_matches(item.get("metadata"), k, v)
                    for k, v in metadata_equals.items()
                )
            ]
        return merged[:limit]

    async def search_fts(
        self,
        query: str,
        limit: int = 10,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure BM25 search across the FTS mirror.

        When ``project_id`` is set, results are limited to items tagged
        with that project (or with no ``project_id`` at all, for legacy
        rows). Passing :data:`CROSS_PROJECT_SCOPE` disables scoping.
        """
        if self._fts is None or not query.strip():
            return []
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []

        scope = project_id
        if scope == CROSS_PROJECT_SCOPE:
            scope = None

        if scope:
            sql = (
                "SELECT id, collection, content, created_at, "
                "bm25(items_fts) AS rank "
                "FROM items_fts "
                "WHERE items_fts MATCH ? "
                "AND (project_id = ? OR project_id IS NULL OR project_id = '') "
                "ORDER BY rank "
                "LIMIT ?"
            )
            params: tuple[Any, ...] = (sanitized, scope, limit)
        else:
            sql = (
                "SELECT id, collection, content, created_at, "
                "bm25(items_fts) AS rank "
                "FROM items_fts "
                "WHERE items_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?"
            )
            params = (sanitized, limit)

        cursor = await self._fts.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": row[0],
                "collection": row[1],
                "content": row[2],
                "created_at": row[3],
                "bm25_score": -float(row[4]),  # bm25() returns negative; lower = better
            }
            for row in rows
        ]

    async def browse(
        self,
        collection: str = "facts",
        page: int = 1,
        limit: int = 25,
    ) -> tuple[list[dict[str, Any]], int]:
        """Plain pagination through one collection.

        Robust against stale collection handles: if the cached handle
        points at a deleted collection (e.g. after a manual ChromaDB
        wipe), we refresh it once and retry. This stops a wiped Chroma
        from crashing every request indefinitely.
        """
        col = self._require_collection(collection)
        offset = max(0, (page - 1) * limit)
        try:
            result = col.get(limit=limit, offset=offset)
            total = col.count()
        except ChromaNotFoundError:
            refreshed = self._refresh_collection(collection)
            if refreshed is None:
                return [], 0
            col = refreshed
            try:
                result = col.get(limit=limit, offset=offset)
                total = col.count()
            except Exception as retry_exc:  # noqa: BLE001
                log.warning(
                    "memory.browse_retry_failed",
                    collection=collection,
                    error=str(retry_exc),
                )
                return [], 0
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        items = [
            {"id": i, "content": d, "metadata": m or {}}
            for i, d, m in zip(ids, docs, metas)
        ]
        return items, total

    # ─── Internals ──────────────────────────────────────────────────

    async def _embed_one(self, text: str) -> list[float]:
        if self._embedding is None:
            raise RuntimeError("EmbeddingClient not configured")
        return await self._embedding.embed(text)

    async def _vector_search(
        self,
        query: str,
        collections: list[str],
        limit: int,
        project_id: str | None = None,
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._embedding is None:
            return []
        try:
            query_vec = await self._embedding.embed(query)
        except Exception as exc:  # noqa: BLE001
            log.error("memory.embed_error", error=str(exc))
            return []

        # When scoping, also keep "legacy" items (written before the scoping
        # rollout) which carry no ``project_id`` at all so we don't silently
        # lose prior knowledge of the user.
        scope_clause: dict[str, Any] | None = None
        if project_id:
            scope_clause = {
                "$or": [
                    {"project_id": project_id},
                    {"project_id": {"$exists": False}},
                ]
            }

        extra_clauses: list[dict[str, Any]] = []
        for key, value in (metadata_equals or {}).items():
            extra_clauses.append({key: value})

        where: dict[str, Any] | None
        if scope_clause and extra_clauses:
            where = {"$and": [scope_clause, *extra_clauses]}
        elif scope_clause:
            where = scope_clause
        elif len(extra_clauses) == 1:
            where = extra_clauses[0]
        elif extra_clauses:
            where = {"$and": extra_clauses}
        else:
            where = None

        results: list[dict[str, Any]] = []
        for name in collections:
            col = self._collections.get(name)
            if col is None:
                continue
            try:
                kwargs: dict[str, Any] = {
                    "query_embeddings": [query_vec],
                    "n_results": limit,
                }
                if where is not None:
                    kwargs["where"] = where
                response = col.query(**kwargs)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "memory.query_error",
                    collection=name,
                    error=str(exc),
                    project_id=project_id,
                )
                continue
            ids = (response.get("ids") or [[]])[0]
            docs = (response.get("documents") or [[]])[0]
            metas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]
            for idx, doc, meta, dist in zip(ids, docs, metas, distances):
                results.append(
                    {
                        "id": idx,
                        "collection": name,
                        "content": doc,
                        "metadata": meta or {},
                        # Convert distance → similarity (cosine)
                        "vector_score": 1.0 - float(dist),
                    }
                )
        return results

    async def _fts_search(
        self,
        query: str,
        collections: list[str],
        limit: int,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._fts is None:
            return []
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []
        placeholders = ",".join("?" for _ in collections)

        # Legacy rows may have project_id = NULL (pre-migration) OR the
        # empty string; both should remain visible under scoping, mirroring
        # the ``$exists=False`` behaviour of the ChromaDB path.
        if project_id:
            sql = (
                "SELECT id, collection, content, created_at, "
                "bm25(items_fts) AS rank "
                "FROM items_fts "
                f"WHERE items_fts MATCH ? AND collection IN ({placeholders}) "
                "AND (project_id = ? OR project_id IS NULL OR project_id = '') "
                "ORDER BY rank "
                "LIMIT ?"
            )
            params: tuple[Any, ...] = (
                sanitized,
                *collections,
                project_id,
                limit,
            )
        else:
            sql = (
                "SELECT id, collection, content, created_at, "
                "bm25(items_fts) AS rank "
                "FROM items_fts "
                f"WHERE items_fts MATCH ? AND collection IN ({placeholders}) "
                "ORDER BY rank "
                "LIMIT ?"
            )
            params = (sanitized, *collections, limit)

        cursor = await self._fts.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": row[0],
                "collection": row[1],
                "content": row[2],
                "metadata": {"created_at": row[3]},
                "bm25_score": -float(row[4]),
            }
            for row in rows
        ]

    def _merge(
        self,
        vector_hits: list[dict[str, Any]],
        bm25_hits: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        vector_map = {hit["id"]: hit for hit in vector_hits}
        bm25_map = {hit["id"]: hit for hit in bm25_hits}

        vec_scores = {hit["id"]: hit["vector_score"] for hit in vector_hits}
        bm25_scores = {hit["id"]: hit["bm25_score"] for hit in bm25_hits}

        norm_vec = _normalise(vec_scores)
        norm_bm25 = _normalise(bm25_scores)

        threshold = self._config.recall_threshold
        v_weight = self._config.vector_weight
        b_weight = self._config.bm25_weight

        combined: list[dict[str, Any]] = []
        for item_id in set(vec_scores) | set(bm25_scores):
            base = vector_map.get(item_id) or bm25_map.get(item_id) or {}
            v = norm_vec.get(item_id, 0.0)
            b = norm_bm25.get(item_id, 0.0)
            score = v * v_weight + b * b_weight
            if score < threshold:
                continue
            combined.append(
                {
                    "id": item_id,
                    "collection": base.get("collection", ""),
                    "content": base.get("content", ""),
                    "metadata": base.get("metadata", {}),
                    "score": score,
                    "vector_score": v,
                    "bm25_score": b,
                }
            )

        combined.sort(key=lambda h: h["score"], reverse=True)
        return combined[:limit]
