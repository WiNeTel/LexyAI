"""
Lexy AI - LexyLLM Client.

Direct httpx client for llama.cpp's OpenAI-compatible ``/v1/chat/completions``
endpoint. We previously used LiteLLM, but that introduced:

* Silent ``api_base`` caching between calls (requests for E4B sometimes
  landed on the A4B endpoint).
* A proxy-server import path that required ``apscheduler`` +
  ``pydantic[email]`` just to compute telemetry keys nobody reads.
* INFO-level spam per request.

By talking to llama.cpp directly we get:

* **Guaranteed endpoint pinning** per brain via explicit ``base_url``.
* **Native Gemma 4 reasoning format** (``<|think|>`` system token to turn
  thinking on, ``<|channel>thought ... <channel|>`` block in the output).
* **No hidden telemetry**, **no INFO spam**, **no cache surprises**.

Features
--------
* Two-brain routing via ``brain="e4b" | "a4b" | "auto"``.
* Streaming with ``RepetitionDetector`` and the
  ``<|channel>thought ... <channel|>`` reasoning parser.
* Single source of truth: ``chat_stream_structured`` yields
  ``("reasoning", chunk)`` and ``("content", chunk)`` tuples.
* ``chat`` and ``chat_stream`` are thin wrappers on the structured stream.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from lexy_core.config import BrainConfig, LexyConfig
from lexy_core.llm.repetition import RepetitionDetector
from lexy_core.utils.logging import get_logger

log = get_logger(module="llm_client")


class LLMError(RuntimeError):
    """Raised when an LLM call fails after the stream has opened."""


# Gemma 4 reasoning tokens (source: huggingface.co/google/gemma-4-26B-A4B-it)
# * ``<|think|>`` at the start of the system prompt enables the thinking mode.
# * The model then emits ``<|channel>thought\n<reasoning>\n<channel|>`` before
#   the final answer.
_GEMMA_THINK_ENABLE = "<|think|>"
_GEMMA_REASON_OPEN = "<|channel>thought"
_GEMMA_REASON_CLOSE = "<channel|>"

# Legacy / fallback tag openers + closers we also recognise in case a
# particular fine-tune emits the older Qwen3 / DeepSeek style.
_LEGACY_OPEN = ("<think>", "<thinking>")
_LEGACY_CLOSE = ("</think>", "</thinking>")

# Longest possible tail we need to buffer while deciding if we're in
# reasoning mode. Used so we never hoard more text than necessary.
_MAX_OPEN_LEN = max(
    len(_GEMMA_REASON_OPEN),
    *(len(t) for t in _LEGACY_OPEN),
)
_MAX_CLOSE_LEN = max(
    len(_GEMMA_REASON_CLOSE),
    *(len(t) for t in _LEGACY_CLOSE),
)


class LexyLLM:
    """Two-brain LLM client talking to llama.cpp directly via httpx."""

    def __init__(self, config: LexyConfig) -> None:
        self._config = config
        self._default_brain = config.routing.default_brain or "e4b"
        # Keep a dedicated AsyncClient per brain — each one uses the brain's
        # own base_url so there is zero chance of cross-routing.
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._connected: bool = False

    # ─── Connection management ──────────────────────────────────────

    async def connect(self) -> bool:
        """Open one httpx client per active brain (respects profile) and probe /health."""
        ok_any = False
        active = self._config.active_brain_names()
        log.info(
            "llm.profile",
            profile=self._config.system.profile,
            active_brains=sorted(active),
            configured=sorted(self._config.brains.keys()),
        )
        for brain_name, brain_cfg in self._config.brains.items():
            if brain_name not in active:
                log.debug("llm.brain_skipped_by_profile", brain=brain_name)
                continue
            base = brain_cfg.endpoint.rstrip("/")  # e.g. http://127.0.0.1:5005/v1
            client = httpx.AsyncClient(
                base_url=base,
                timeout=httpx.Timeout(brain_cfg.timeout, connect=5.0),
                headers={
                    "Authorization": f"Bearer {brain_cfg.api_key}",
                    "Content-Type": "application/json",
                },
            )

            try:
                # /health lives at the root, not under /v1
                health_url = base.removesuffix("/v1") + "/health"
                resp = await client.get(health_url, headers={})
                healthy = resp.status_code < 500
            except httpx.HTTPError as exc:
                log.warning(
                    "llm.brain_unreachable",
                    brain=brain_name,
                    endpoint=base,
                    error=str(exc),
                )
                healthy = False
            if healthy:
                # Nur healthy Brains in self._clients halten — sonst
                # greift der _resolve()-Fallback nicht und Requests an
                # ein totes Brain (z.B. e4b im Qwen-Setup) crashen mit
                # "All connection attempts failed".
                self._clients[brain_name] = client
                ok_any = True
                log.info(
                    "llm.brain_ready",
                    brain=brain_name,
                    endpoint=base,
                    model=brain_cfg.model,
                    thinking=brain_cfg.thinking,
                )
            else:
                # Client wird nicht gespeichert -> sauber schliessen,
                # damit kein Socket-Leak entsteht.
                await client.aclose()
        self._connected = ok_any
        return ok_any

    async def disconnect(self) -> None:
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()
        self._connected = False
        log.info("llm.disconnected")

    # ─── Internal helpers ────────────────────────────────────────────

    def _resolve(self, brain: str) -> tuple[str, BrainConfig, httpx.AsyncClient]:
        if brain == "auto":
            brain = self._default_brain
        # Wenn connect() schon lief und das angefragte Brain damals als
        # unreachable markiert wurde (kein Eintrag in self._clients),
        # routen wir auf den default_brain um. So koennen E4B-Requests
        # weiterlaufen, wenn nur das Hauptmodell auf :5005 lebt.
        if (
            self._connected
            and brain not in self._clients
            and brain != self._default_brain
            and self._default_brain in self._clients
        ):
            log.info(
                "llm.brain_fallback",
                requested=brain,
                fallback=self._default_brain,
            )
            brain = self._default_brain
        cfg = self._config.get_brain(brain)
        client = self._clients.get(brain)
        if client is None:
            # Lazy-create if connect() was skipped (e.g. in tests).
            base = cfg.endpoint.rstrip("/")
            client = httpx.AsyncClient(
                base_url=base,
                timeout=httpx.Timeout(cfg.timeout, connect=5.0),
                headers={
                    "Authorization": f"Bearer {cfg.api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._clients[brain] = client
        return brain, cfg, client

    @staticmethod
    def _inject_thinking_token(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Ensure the system prompt starts with ``<|think|>`` so Gemma 4
        switches into reasoning mode. If the first message isn't a system
        message we prepend a new one carrying only the token.

        Multimodal note: a ``content`` field may be a list of content
        blocks (``[{"type":"text",...}, {"type":"image_url",...}]``)
        instead of a plain string. We only ever inject the think-token
        into the *system* message, which by convention is plain text —
        so the list-content path doesn't need special handling here.
        """
        if not messages:
            return [{"role": "system", "content": _GEMMA_THINK_ENABLE}]

        first = messages[0]
        if first.get("role") == "system":
            content = first.get("content", "") or ""
            # Defensive: if some caller put a list here, fall back to
            # prepending a fresh system message instead of mangling it.
            if isinstance(content, list):
                return [
                    {"role": "system", "content": _GEMMA_THINK_ENABLE},
                    *messages,
                ]
            if _GEMMA_THINK_ENABLE in content:
                return messages  # already enabled
            patched = dict(first)
            patched["content"] = f"{_GEMMA_THINK_ENABLE}\n{content}".strip()
            return [patched, *messages[1:]]

        return [
            {"role": "system", "content": _GEMMA_THINK_ENABLE},
            *messages,
        ]

    def _build_payload(
        self,
        brain_cfg: BrainConfig,
        messages: list[dict[str, Any]],
        stream: bool,
        **overrides: Any,
    ) -> dict[str, Any]:
        thinking = bool(overrides.get("thinking", brain_cfg.thinking))
        if thinking:
            messages = self._inject_thinking_token(messages)

        payload: dict[str, Any] = {
            "model": brain_cfg.model,
            "messages": messages,
            "stream": stream,
            "temperature": overrides.get("temperature", brain_cfg.temperature),
            "top_p": overrides.get("top_p", brain_cfg.top_p),
            "top_k": overrides.get("top_k", brain_cfg.top_k),
            "max_tokens": overrides.get("max_tokens", brain_cfg.max_tokens),
            "repeat_penalty": overrides.get("repeat_penalty", brain_cfg.repeat_penalty),
        }

        # Tell llama.cpp explicitly whether to use the model's reasoning
        # channel — in BOTH directions. Qwen3 defaults to thinking ON unless
        # told otherwise, so a ``thinking: false`` brain that sent nothing
        # still burned its token budget inside ``<think>`` and, with a small
        # ``max_tokens`` (e.g. a classifier call), returned EMPTY content
        # (finish_reason=length, all tokens in reasoning_content). Sending
        # ``enable_thinking: false`` makes ``thinking: false`` actually mean
        # no reasoning. Harmless for templates that ignore the kwarg (Gemma
        # uses the ``<|think|>`` system-token path above instead).
        payload.setdefault("chat_template_kwargs", {"enable_thinking": thinking})
        if thinking and brain_cfg.reasoning_budget is not None:
            payload["reasoning_budget"] = brain_cfg.reasoning_budget

        return payload

    # ─── Single-shot ─────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, Any]],
        brain: str = "auto",
        **overrides: Any,
    ) -> str:
        """Single-shot completion. Returns the final content (no reasoning)."""
        content_parts: list[str] = []
        async for kind, chunk in self.chat_stream_structured(
            messages=messages, brain=brain, **overrides
        ):
            if kind == "content":
                content_parts.append(chunk)
        return "".join(content_parts)

    # ─── Streaming (content only, legacy signature) ─────────────────

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        brain: str = "auto",
        **overrides: Any,
    ) -> AsyncIterator[str]:
        """Backward-compatible content-only stream."""
        async for kind, chunk in self.chat_stream_structured(
            messages=messages, brain=brain, **overrides
        ):
            if kind == "content":
                yield chunk

    # ─── Streaming (structured: reasoning + content) ────────────────

    async def chat_stream_structured(
        self,
        messages: list[dict[str, Any]],
        brain: str = "auto",
        **overrides: Any,
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Yields ``(kind, chunk)`` tuples where ``kind`` is ``"reasoning"`` or
        ``"content"``.

        Reasoning sources, in priority order:

        1. ``delta.reasoning_content`` / ``delta.reasoning`` — llama.cpp
           newer builds emit these when ``chat_template_kwargs.enable_thinking``
           is true.
        2. Gemma 4 native ``<|channel>thought ... <channel|>`` block inside
           the content stream.
        3. Legacy ``<think>...</think>`` / ``<thinking>...</thinking>`` tags.

        The parser is a single character-level state machine that handles
        tags split across chunks and never yields mixed segments.
        """
        brain_name, brain_cfg, client = self._resolve(brain)
        payload = self._build_payload(
            brain_cfg, messages, stream=True, **overrides
        )
        log.debug(
            "llm.chat_stream",
            brain=brain_name,
            endpoint=str(client.base_url),
            model=brain_cfg.model,
            thinking=bool(payload.get("chat_template_kwargs", {}).get("enable_thinking")),
            messages=len(messages),
        )

        detector = RepetitionDetector()

        # Streaming state machine
        in_reason = False
        tag_buffer = ""  # buffer for partial tag detection

        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMError(
                        f"LLM HTTP {response.status_code} ({brain_name}): "
                        f"{body.decode('utf-8', 'replace')[:400]}"
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    line = line.strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        log.debug(
                            "llm.malformed_sse_chunk",
                            brain=brain_name,
                            preview=line[:120],
                        )
                        continue

                    try:
                        delta = chunk["choices"][0]["delta"]
                    except (KeyError, IndexError):
                        continue

                    # Source 1: explicit reasoning fields
                    reasoning_piece = delta.get("reasoning_content") or delta.get(
                        "reasoning"
                    )
                    if reasoning_piece:
                        yield "reasoning", str(reasoning_piece)

                    content_piece = delta.get("content")
                    if not content_piece:
                        continue

                    # Source 2 + 3: inline tags. Char-level state machine.
                    for char in content_piece:
                        tag_buffer += char

                        if in_reason:
                            # Looking for a closer.
                            closed = False
                            for closer in (
                                _GEMMA_REASON_CLOSE,
                                *_LEGACY_CLOSE,
                            ):
                                if tag_buffer.endswith(closer):
                                    body = tag_buffer[: -len(closer)]
                                    if body:
                                        yield "reasoning", body
                                    tag_buffer = ""
                                    in_reason = False
                                    closed = True
                                    break
                            if closed:
                                continue
                            # Flush head when buffer grows past the longest closer
                            if len(tag_buffer) > _MAX_CLOSE_LEN:
                                flush = tag_buffer[:-_MAX_CLOSE_LEN]
                                if flush:
                                    yield "reasoning", flush
                                tag_buffer = tag_buffer[-_MAX_CLOSE_LEN:]
                            continue

                        # Not in a reason block — watch for an opener.
                        opener_found = False
                        for opener in (_GEMMA_REASON_OPEN, *_LEGACY_OPEN):
                            if tag_buffer.endswith(opener):
                                prefix = tag_buffer[: -len(opener)]
                                if prefix:
                                    if detector.check(prefix):
                                        log.warning(
                                            "llm.stream_aborted_repetition",
                                            brain=brain_name,
                                        )
                                        return
                                    yield "content", prefix
                                tag_buffer = ""
                                in_reason = True
                                opener_found = True
                                break
                        if opener_found:
                            continue

                        # Could we still be mid-opener? Hold on.
                        partial_match = any(
                            opener.startswith(tag_buffer)
                            for opener in (_GEMMA_REASON_OPEN, *_LEGACY_OPEN)
                        )
                        if partial_match:
                            continue

                        # No pending match — flush to content.
                        if detector.check(tag_buffer):
                            log.warning(
                                "llm.stream_aborted_repetition",
                                brain=brain_name,
                            )
                            return
                        yield "content", tag_buffer
                        tag_buffer = ""

        except httpx.HTTPError as exc:
            log.error(
                "llm.stream_failed",
                brain=brain_name,
                endpoint=str(client.base_url),
                error=str(exc),
            )
            raise LLMError(f"LLM stream failed ({brain_name}): {exc}") from exc

        # Flush anything left in the buffer at EOS.
        if tag_buffer:
            kind = "reasoning" if in_reason else "content"
            yield kind, tag_buffer
