"""
Lexy AI - PluginAPI.

The single facade plugins use to interact with the core. Each plugin gets its
own PluginAPI instance, which tracks every registration so ``cleanup()`` can
remove them all on disable.

Design rules
------------
* Plugins must NEVER touch the core directly. Only call PluginAPI methods.
* Every register_* call records the registration internally.
* ``cleanup()`` is idempotent and called from PluginLoader.disable_plugin().
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

import aiosqlite

from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.app import LexyApp
    from lexy_core.plugin_system.plugin_manifest import PluginManifest

log = get_logger(module="plugin_api")


HookCallback = Callable[[dict[str, Any]], Any | Awaitable[Any]]
EventCallback = Callable[[Any], Any | Awaitable[Any]]
ToolHandler = Callable[..., Any | Awaitable[Any]]
WSHandler = Callable[..., Any | Awaitable[Any]]


class PluginAPI:
    """
    Facade between a plugin and the LexyApp core.

    Sections:
    LLM, Memory, Solutions, Events, Hooks, WebSocket, Tools, Voice,
    Storage, Config, Plugins, Frontend, Cleanup.
    """

    def __init__(
        self,
        plugin_name: str,
        app: "LexyApp",
        manifest: "PluginManifest",
    ) -> None:
        self._name = plugin_name
        self._app = app
        self._manifest = manifest
        self._log = log.bind(plugin=plugin_name)

        # Tracked registrations for cleanup
        self._ws_handlers: list[str] = []
        self._hooks: list[tuple[str, HookCallback]] = []
        self._events: list[tuple[str, EventCallback]] = []
        self._frontend: list[str] = []
        self._tools: list[str] = []
        self._voice_providers: list[tuple[str, str]] = []  # (kind, provider_name)
        self._channels: list[str] = []
        self._db: aiosqlite.Connection | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def manifest(self) -> "PluginManifest":
        return self._manifest

    # ─── LLM ─────────────────────────────────────────────────────────

    async def llm_chat(
        self,
        messages: list[dict[str, str]],
        brain: str = "auto",
        **kwargs: Any,
    ) -> str:
        """Single-shot chat. ``brain`` may be ``"auto"``, ``"e4b"`` or ``"a4b"``."""
        if not self._app.llm:
            raise RuntimeError("LLM client not initialised")
        return await self._app.llm.chat(messages=messages, brain=brain, **kwargs)

    async def llm_chat_stream(
        self,
        messages: list[dict[str, str]],
        brain: str = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming chat. Yields content chunks."""
        if not self._app.llm:
            raise RuntimeError("LLM client not initialised")
        async for chunk in self._app.llm.chat_stream(
            messages=messages, brain=brain, **kwargs
        ):
            yield chunk

    async def llm_tool_chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        """Tool-oriented chat. Always uses E4B (fast brain) for tool calls."""
        return await self.llm_chat(messages, brain="e4b", **kwargs)

    def get_brain_context_size(self, brain: str = "auto") -> int:
        """Return the configured ``context_size`` for a brain (live read).

        Plugins that build large prompts (RP characters, long histories,
        multi-turn rounds) should query this instead of hardcoding a size
        — that way a change in ``routing.yaml`` is picked up without any
        plugin code change. ``brain="auto"`` resolves to the default brain.

        Falls back to 16384 when the brain is unknown rather than raising,
        so a misconfigured brain name can't crash a background round.
        """
        name = brain
        if name == "auto":
            name = (
                getattr(self._app.config.routing, "default_brain", None)
                or "e4b"
            )
        try:
            return int(self._app.config.get_brain(name).context_size)
        except (KeyError, AttributeError, TypeError):
            self._log.warning("plugin_api.unknown_brain", brain=name)
            return 16384

    # ─── Session ─────────────────────────────────────────────────────

    def get_session_history(
        self, session_id: str, limit: int = 8
    ) -> list[dict[str, str]]:
        """Return the last ``limit`` messages from a session.

        Plugins should use this instead of reaching into ``api._app``
        directly. Returns an empty list if the session store is
        unavailable or the session doesn't exist.
        """
        store = getattr(self._app, "session_store", None)
        if store is None:
            return []
        raw = store.get(session_id)
        if not raw:
            return []
        return list(raw[-limit:])

    # ─── Agent ───────────────────────────────────────────────────────

    async def agent_proactive(
        self,
        session_id: str,
        prompt: str,
        label: str = "",
    ) -> bool:
        """Ask the chat agent to speak unprompted in ``session_id``.

        The agent receives ``prompt`` as an internal trigger (no user
        message is stored) and its reply is appended to the session and
        broadcast to WS clients. Returns ``False`` if the agent is not
        available or the call fails — callers can then fall back to
        emitting a plain event.
        """
        agent = getattr(self._app, "agent", None)
        if agent is None or not hasattr(agent, "process_proactive"):
            self._log.warning("agent.proactive_unavailable")
            return False
        try:
            await agent.process_proactive(
                {
                    "session_id": session_id,
                    "text": prompt,
                    "label": label,
                    "from": self._name,
                    "internal": True,
                }
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._log.error("agent.proactive_failed", error=str(exc))
            return False

    # ─── Memory ──────────────────────────────────────────────────────

    async def memory_store(
        self,
        text: str,
        collection: str = "facts",
        metadata: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> str:
        """Store an item in a memory collection. Returns its id.

        If ``project_id`` is given it overrides whatever is in ``metadata``
        (or any auto-resolution via ``session_id``). Otherwise the manager
        falls back to the session-store lookup, then to the default project.
        """
        if not self._app.memory:
            self._log.warning("memory.store_skipped", reason="memory_not_ready")
            return ""
        meta = dict(metadata or {})
        if project_id is not None:
            meta["project_id"] = project_id
        return await self._app.memory.store(
            text=text, collection=collection, metadata=meta
        )

    async def memory_recall(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 5,
        project_id: str | None = None,
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid recall (vector + BM25). Returns scored memory items.

        ``project_id``:
            * ``None`` → no project scoping (every project visible).
            * a concrete project id → items tagged with that project, plus
              legacy items without a ``project_id`` tag.
            * ``"__all__"`` → explicit opt-out from scoping (same as ``None``).

        ``metadata_equals``:
            Optional exact-match filter on stored metadata (e.g.
            ``{"character_id": "luna123"}`` to fetch only Luna's own
            memory for per-character recall). Pushed into the vector
            ``where`` clause; FTS hits are post-filtered in Python.
        """
        if not self._app.memory:
            return []
        return await self._app.memory.recall(
            query=query,
            collection=collection,
            limit=limit,
            project_id=project_id,
            metadata_equals=metadata_equals,
        )

    async def memory_ensure_collection(self, name: str) -> None:
        """Idempotently register a memory collection by name.

        Phase 13 — used by the character_chat plugin's RP session
        registry to spin up per-session collections (``rp__<id>``) at
        attach time without requiring config changes.
        """
        if not self._app.memory:
            self._log.warning(
                "memory.ensure_skipped", reason="memory_not_ready"
            )
            return
        await self._app.memory.ensure_collection(name)

    async def memory_delete_collection(self, name: str) -> dict[str, int]:
        """Delete a memory collection (Chroma + FTS rows).

        Returns ``{"chroma": <count>, "fts": <rows>}`` describing what
        was removed. Used by the RP session container's ``destroy()``.
        """
        if not self._app.memory:
            self._log.warning(
                "memory.delete_collection_skipped",
                reason="memory_not_ready",
            )
            return {"chroma": 0, "fts": 0}
        return await self._app.memory.delete_collection(name)

    async def memory_search_fts(
        self,
        query: str,
        limit: int = 10,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure FTS5 (BM25) search; useful for keyword queries."""
        if not self._app.memory:
            return []
        return await self._app.memory.search_fts(
            query=query, limit=limit, project_id=project_id
        )

    # ─── Solutions ───────────────────────────────────────────────────

    async def solutions_store(self, problem: str, solution: str) -> str:
        if not self._app.memory:
            return ""
        return await self._app.memory.store(
            text=f"PROBLEM: {problem}\nSOLUTION: {solution}",
            collection="solutions",
            metadata={"problem": problem, "solution": solution},
        )

    async def solutions_recall(
        self, problem: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        if not self._app.memory:
            return []
        return await self._app.memory.recall(
            query=problem, collection="solutions", limit=limit
        )

    # ─── Events ──────────────────────────────────────────────────────

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> int:
        """Emit an event from this plugin (auto-source)."""
        return await self._app.event_bus.emit(event, data, source=self._name)

    def on_event(self, event: str, callback: EventCallback) -> None:
        """Subscribe to an event."""
        self._app.event_bus.on(event, callback, source=self._name)
        self._events.append((event, callback))

    def off_event(self, event: str, callback: EventCallback) -> None:
        """Unsubscribe a single listener."""
        self._app.event_bus.off(event, callback)

    # ─── Hooks ───────────────────────────────────────────────────────

    def register_hook(
        self,
        hook_name: str,
        callback: HookCallback,
        priority: int = 50,
    ) -> None:
        """Register a hook callback."""
        self._app.hooks.register(
            hook_name,
            callback,
            priority=priority,
            name=f"{self._name}.{getattr(callback, '__name__', 'cb')}",
            source=self._name,
        )
        self._hooks.append((hook_name, callback))

    # ─── WebSocket ───────────────────────────────────────────────────

    def register_ws_handler(self, msg_type: str, handler: WSHandler) -> None:
        """Register a handler for a WebSocket message type."""
        if self._app.ws_server is not None:
            self._app.ws_server.register_handler(msg_type, handler, source=self._name)
        self._ws_handlers.append(msg_type)

    async def ws_broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast a JSON message to all connected clients."""
        if self._app.ws_server is not None:
            await self._app.ws_server.broadcast(data)

    async def ws_send(self, client_id: str, data: dict[str, Any]) -> None:
        """Send a JSON message to a single client."""
        if self._app.ws_server is not None:
            await self._app.ws_server.send_to(client_id, data)

    # ─── Tools (LLM Function Calling) ────────────────────────────────

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        description: str,
        schema: dict[str, Any],
    ) -> None:
        """Register a tool the LLM can call."""
        if self._app.tool_registry is not None:
            self._app.tool_registry.register(
                name=name,
                handler=handler,
                description=description,
                schema=schema,
                source=self._name,
            )
        self._tools.append(name)

    def get_tool_caller(self) -> Any:
        return self._app.tool_caller

    def get_tool_registry(self) -> Any:
        return self._app.tool_registry

    async def call_tool(
        self, name: str, args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a registered tool by name and return a plain dict.

        Returns ``{"ok": bool, "data": Any, "error": str}``. Always returns a
        dict (never raises) so plugins running mini agent loops don't have to
        wrap every call in try/except.
        """
        registry = self._app.tool_registry
        if registry is None:
            return {"ok": False, "data": None, "error": "tool_registry_unavailable"}
        result = await registry.execute(name, dict(args or {}))
        return {
            "ok": bool(result.success),
            "data": result.data,
            "error": result.error,
        }

    def list_tool_names(self) -> list[str]:
        """Return the names of every currently registered tool."""
        registry = self._app.tool_registry
        if registry is None:
            return []
        return [schema["name"] for schema in registry.get_all_schemas()]

    # ─── Voice (NEW in v2) ──────────────────────────────────────────

    def register_voice_provider(self, kind: str, provider: Any) -> None:
        """
        Register an STT or TTS provider.

        Args:
            kind: ``"stt"`` or ``"tts"``.
            provider: Concrete provider implementing STTProvider or TTSProvider.
        """
        if kind not in ("stt", "tts"):
            raise ValueError(f"voice provider kind must be 'stt' or 'tts', got {kind!r}")
        if self._app.voice is None:
            raise RuntimeError("VoiceManager not initialised")
        self._app.voice.register_provider(kind=kind, name=self._name, provider=provider)
        self._voice_providers.append((kind, self._name))

    async def tts_speak(
        self, text: str, voice: str | None = None
    ) -> bytes:
        """Synthesize ``text`` to WAV bytes.

        ``voice`` is an optional provider-specific voice override (e.g.
        a CosyVoice speaker id for a named RP character). The active TTS
        provider uses it when supported and falls back to its default
        voice otherwise.
        """
        if self._app.voice is None:
            return b""
        return await self._app.voice.synthesize(text, voice=voice)

    async def stt_transcribe(self, audio_bytes: bytes) -> str:
        if self._app.voice is None:
            return ""
        return await self._app.voice.transcribe(audio_bytes)

    # ─── Channels ────────────────────────────────────────────────────

    def register_channel(self, channel: Any) -> None:
        """Register an external channel (WhatsApp/Discord/Telegram/…)."""
        if self._app.channel_router is None:
            raise RuntimeError("ChannelRouter not initialised")
        self._app.channel_router.register(channel)
        self._channels.append(channel.name)

    # ─── Storage ─────────────────────────────────────────────────────

    def get_data_path(self) -> Path:
        """Plugin-private writable data directory (created on demand)."""
        path = Path("data/plugins") / self._name
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def get_db(self) -> aiosqlite.Connection:
        """
        Plugin-private aiosqlite connection. Cached for the plugin lifetime.
        Connection is closed automatically by ``cleanup()``.
        """
        if self._db is None:
            db_path = self.get_data_path() / f"{self._name}.db"
            self._db = await aiosqlite.connect(str(db_path))
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA busy_timeout=5000")
            await self._db.commit()
        return self._db

    # ─── Config ──────────────────────────────────────────────────────

    def get_config(self) -> dict[str, Any]:
        """
        Merged config: ``manifest.config_defaults`` overlaid by the plugin's
        section in ``config/plugins.yaml`` (or the matching block in config.raw).
        """
        merged: dict[str, Any] = dict(self._manifest.config_defaults)
        plugin_overrides = self._app.plugin_overrides.get(self._name, {})
        merged.update(plugin_overrides)
        return merged

    # ─── Plugin lookup ───────────────────────────────────────────────

    def get_plugin(self, name: str) -> Any:
        """Look up another loaded plugin (or None)."""
        if self._app.plugin_loader is None:
            return None
        return self._app.plugin_loader.get_plugin(name)

    # ─── Frontend ────────────────────────────────────────────────────

    def register_frontend_module(
        self, module_id: str, config: dict[str, Any]
    ) -> None:
        if self._app.ws_server is not None:
            self._app.ws_server.register_frontend_module(
                module_id=module_id,
                config={**config, "plugin": self._name},
            )
        self._frontend.append(module_id)

    # ─── Cleanup ─────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        """
        Remove every registration this plugin made. Idempotent.
        Called by PluginLoader.disable_plugin().
        """
        # Events
        self._app.event_bus.off_all(self._name)
        self._events.clear()

        # Hooks
        self._app.hooks.unregister_all(self._name)
        self._hooks.clear()

        # WebSocket handlers
        if self._app.ws_server is not None:
            for msg_type in self._ws_handlers:
                self._app.ws_server.unregister_handler(msg_type, source=self._name)
        self._ws_handlers.clear()

        # Frontend modules
        if self._app.ws_server is not None:
            for module_id in self._frontend:
                self._app.ws_server.unregister_frontend_module(module_id)
        self._frontend.clear()

        # Tools
        if self._app.tool_registry is not None:
            self._app.tool_registry.unregister_all(self._name)
        self._tools.clear()

        # Voice providers
        if self._app.voice is not None:
            for kind, provider_name in self._voice_providers:
                self._app.voice.unregister_provider(kind=kind, name=provider_name)
        self._voice_providers.clear()

        # Channels
        if self._app.channel_router is not None:
            for channel_name in self._channels:
                self._app.channel_router.unregister(channel_name)
        self._channels.clear()

        # DB
        if self._db is not None:
            await self._db.close()
            self._db = None

        self._log.info("plugin.cleaned_up")
