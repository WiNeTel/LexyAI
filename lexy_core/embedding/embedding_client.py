"""
Lexy AI - Embedding Client.

Wraps sentence-transformers (Jina v3 / Jina v5 / MiniLM …) with a small LRU
cache. Model load and inference are CPU/GPU bound; we offload them to a thread
via ``asyncio.to_thread`` (the only allowed off-loop primitive in v2 — used
because sentence-transformers has no native async API).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from lexy_core.config import EmbeddingConfig
from lexy_core.utils.logging import get_logger

log = get_logger(module="embedding_client")


class EmbeddingClient:
    """
    Sentence-transformers embedding client.

    Lazy model load, async ``embed`` / ``embed_batch``, deterministic
    fallback to CPU if the requested device fails.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._device: str = config.device
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._initialized: bool = False

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Load the model on the configured device, fall back to CPU."""
        try:
            self._model = await asyncio.to_thread(self._load, self._device)
            self._initialized = True
            log.info(
                "embedding.ready",
                model=self._config.model,
                device=self._device,
                dim=self._config.dimension,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "embedding.gpu_failed",
                device=self._device,
                error=str(exc),
            )

        try:
            self._device = "cpu"
            self._model = await asyncio.to_thread(self._load, self._device)
            self._initialized = True
            log.info("embedding.ready_cpu", model=self._config.model)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("embedding.cpu_failed", error=str(exc))
            self._initialized = False
            return False

    def _load(self, device: str) -> Any:
        """Synchronous model load (runs in a worker thread)."""
        from sentence_transformers import SentenceTransformer

        # trust_remote_code is only needed for a handful of models
        # (jinaai/jina-embeddings-v3, nomic-ai/*, some Chinese BGE variants).
        needs_remote = any(
            tag in self._config.model.lower()
            for tag in ("jinaai/", "nomic-ai/", "-remote")
        )
        return SentenceTransformer(
            self._config.model,
            device=device,
            trust_remote_code=needs_remote,
        )

    async def shutdown(self) -> None:
        self._model = None
        self._cache.clear()
        self._initialized = False
        log.info("embedding.shutdown")

    # ─── Inference ──────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Embed a single string. Cached."""
        if not self._initialized:
            raise RuntimeError("EmbeddingClient not initialised")

        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        vector = await asyncio.to_thread(self._encode_one, text)
        self._cache[text] = vector
        if len(self._cache) > self._config.cache_size:
            self._cache.popitem(last=False)
        return vector

    def _encode_one(self, text: str) -> list[float]:
        result = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return result.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Bypasses the cache."""
        if not self._initialized:
            raise RuntimeError("EmbeddingClient not initialised")
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_batch, texts)

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self._config.batch_size,
        )
        return [list(row) for row in result]
