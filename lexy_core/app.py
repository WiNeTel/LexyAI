"""
Lexy AI - LexyApp.

Central application object. Owns every subsystem (config, events, memory,
voice, agent, plugins, FastAPI gateway).

Startup order (see ``architecture/overview.md``):

1. Config + logging
2. EventBus + Hooks + Signals
3. EmbeddingClient
4. LLM client (LiteLLM)
5. ToolRegistry + ToolCaller
6. MemoryManager (optional — needs ChromaDB)
7. VoiceManager
8. ChannelRouter
9. WSServer (created, not started)
10. PluginLoader: discover → topo-sort → on_load → on_enable
11. LexyAgent
12. Built-in WS handlers (chat, tts, stt, signals, plugins)
13. Uvicorn / Gateway start
14. ``core.system_ready`` event
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from lexy_core.agent import LexyAgent, Persona, SessionStore, load_persona
from lexy_core.channels import ChannelRouter
from lexy_core.config import LexyConfig, load_config, set_config
from lexy_core.embedding import EmbeddingClient
from lexy_core.events import EventBus, HookManager, LexySignals, SystemState
from lexy_core.llm import LexyLLM
from lexy_core.memory import MemoryManager
from lexy_core.plugin_system import PluginLoader
from lexy_core.project import DEFAULT_PROJECT_ID, ProjectStore
from lexy_core.tools import ToolCaller, ToolRegistry
from lexy_core.utils.logging import configure_logging, get_logger
from lexy_core.voice import VoiceManager
from lexy_core.websocket import WSServer, build_app

log = get_logger(module="app")


class LexyApp:
    """The Lexy AI core application object."""

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        self._config_path = Path(config_path)
        self.config: LexyConfig = LexyConfig()  # placeholder until startup()
        self.plugin_overrides: dict[str, dict[str, Any]] = {}

        # Subsystems – populated in startup()
        self.event_bus: EventBus = EventBus()
        self.hooks: HookManager = HookManager()
        self.signals: LexySignals = LexySignals()
        self.session_store: SessionStore = SessionStore()
        self.project_store: ProjectStore = ProjectStore()
        self.persona: Persona = Persona()  # Loaded properly in startup()
        self.embedding: EmbeddingClient | None = None
        self.llm: LexyLLM | None = None
        self.memory: MemoryManager | None = None
        self.voice: VoiceManager | None = None
        self.tool_registry: ToolRegistry | None = None
        self.tool_caller: ToolCaller | None = None
        self.agent: LexyAgent | None = None
        self.plugin_loader: PluginLoader | None = None
        self.channel_router: ChannelRouter | None = None
        self.ws_server: WSServer | None = None
        self.fastapi: Any = None

        self._uvicorn_server: uvicorn.Server | None = None
        self._running: bool = False

    # ─── Startup / Shutdown ─────────────────────────────────────────

    async def startup(self) -> None:
        """Initialise every subsystem in the correct order."""
        # 1. Config + logging
        self.config = load_config(self._config_path)
        set_config(self.config)
        configure_logging(level=self.config.system.log_level)

        log.info(
            "app.starting",
            name=self.config.system.name,
            version=self.config.system.version,
        )

        # 1a. Persistent session store (survives server restart)
        self.session_store = SessionStore(
            max_messages=self.config.system.sessions_max_messages,
            persistent_path=self.config.system.sessions_path,
        )
        log.info(
            "app.session_store_loaded",
            path=self.config.system.sessions_path,
            sessions=len(self.session_store.sessions()),
        )

        # 1b. Persistent project registry + migrate orphan sessions into
        # the default project so the sidebar grouping always works.
        self.project_store = ProjectStore(
            persistent_path=self.config.system.projects_path,
        )
        self.project_store.get_default()
        migrated = 0
        for session_id, meta, _count in self.session_store.sessions_with_meta():
            project_id = meta.get("project_id")
            if not project_id:
                self.session_store.set_project(session_id, DEFAULT_PROJECT_ID)
                migrated += 1
        log.info(
            "app.project_store_loaded",
            path=self.config.system.projects_path,
            projects=len(self.project_store.list(include_archived=True)),
            migrated_sessions=migrated,
        )

        self._load_plugin_overrides()

        # 1b. Persona (separate file, user-editable, creates default on first run)
        self.persona = load_persona()
        log.info(
            "app.persona_loaded",
            name=self.persona.name,
            user_name=self.persona.user_name,
            prompt_length=len(self.persona.system_prompt),
        )

        # 2. Events / hooks / signals
        self.signals.update(system_state=SystemState.STARTING)
        log.info("app.events_ready")

        # 3. Embedding
        self.embedding = EmbeddingClient(self.config.embedding)
        embedding_ok = await self.embedding.initialize()
        if not embedding_ok:
            log.warning("app.embedding_unavailable")

        # 4. LLM
        self.llm = LexyLLM(self.config)
        await self.llm.connect()

        # 5. Tools
        self.tool_registry = ToolRegistry()
        self.tool_caller = ToolCaller(self.tool_registry)

        # 6. Memory (best effort — ChromaDB may be down)
        try:
            self.memory = MemoryManager(self.config.memory, self.embedding)
            # Wire the session store so ``memory.store()`` can resolve the
            # current project_id from a session lookup whenever a plugin
            # writes with just ``session_id`` in the metadata.
            self.memory.set_session_store(self.session_store)
            await self.memory.initialize()
        except Exception as exc:  # noqa: BLE001
            log.warning("app.memory_unavailable", error=str(exc))
            self.memory = None

        # 7. Voice
        self.voice = VoiceManager(self.config.voice)

        # 8. Channels
        self.channel_router = ChannelRouter(self)

        # 9. WS server (no listen yet)
        self.ws_server = WSServer(self)

        # 10. Plugins
        self.plugin_loader = PluginLoader(
            plugins_path=Path(self.config.plugins.path),
            app=self,
        )
        await self.plugin_loader.discover_and_load()
        log.info("app.plugins_loaded", count=self.plugin_loader.loaded_count)

        # 11. Agent
        self.agent = LexyAgent(self)

        # 12. Built-in WS handlers
        self._register_builtin_handlers()

        # 13. FastAPI app
        self.fastapi = build_app(self)

        # Ready
        self.signals.update(system_state=SystemState.READY)
        await self.event_bus.emit(
            "core.system_ready", {"version": self.config.system.version}
        )
        self._running = True
        log.info("app.ready")

    async def shutdown(self) -> None:
        """Reverse-order shutdown."""
        if not self._running:
            return
        log.info("app.shutdown")
        self._running = False
        self.signals.update(system_state=SystemState.SHUTDOWN)
        await self.event_bus.emit("core.system_shutdown", {})

        # Flush the session store first so nothing is lost if a plugin
        # explodes during its own shutdown.
        try:
            self.session_store.save()
        except Exception as exc:  # noqa: BLE001
            log.warning("app.session_store_save_failed", error=str(exc))

        if self.plugin_loader is not None:
            await self.plugin_loader.unload_all()
        if self.voice is not None:
            await self.voice.shutdown()
        if self.memory is not None:
            await self.memory.shutdown()
        if self.embedding is not None:
            await self.embedding.shutdown()
        if self.llm is not None:
            await self.llm.disconnect()

        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True

        log.info("app.shutdown_complete")

    # ─── Plugin overrides ───────────────────────────────────────────

    def _load_plugin_overrides(self) -> None:
        """Load ``config/plugins.yaml`` for per-plugin overrides."""
        path = Path("config/plugins.yaml")
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            self.plugin_overrides = {
                str(name): dict(opts) if isinstance(opts, dict) else {}
                for name, opts in data.items()
            }
            log.info(
                "app.plugin_overrides_loaded",
                plugins=list(self.plugin_overrides.keys()),
            )

    # ─── Built-in WebSocket handlers ────────────────────────────────

    def _register_builtin_handlers(self) -> None:
        """Wire chat / signals / plugins handlers if not already supplied."""
        assert self.ws_server is not None

        async def handle_chat(client: Any, message: dict[str, Any]) -> None:
            if self.agent is None:
                await client.send_json({"type": "error", "error": "agent not ready"})
                return
            text = message.get("text", "")
            session_id = message.get("session_id") or client.session_id
            async for chunk in self.agent.process_stream(
                text=text,
                session_id=session_id,
                user_id=client.user_id,
                brain=message.get("brain", "auto"),
            ):
                await client.send_json(chunk)

        async def handle_regenerate(client: Any, message: dict[str, Any]) -> None:
            """
            Drop the last assistant reply from SessionStore + context memory
            and stream a fresh generation of the same user turn.
            """
            if self.agent is None:
                await client.send_json({"type": "error", "error": "agent not ready"})
                return
            session_id = message.get("session_id") or client.session_id
            user_msg, _dropped = self.session_store.pop_last_pair(session_id)
            if user_msg is None:
                await client.send_json(
                    {"type": "error", "error": "no user turn to regenerate"}
                )
                return

            # Clean the last auto-memorized context item for this session
            if self.memory is not None:
                try:
                    await self.memory.delete_last_for_session(session_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "regenerate.memory_cleanup_failed",
                        session_id=session_id,
                        error=str(exc),
                    )

            await client.send_json({"type": "regenerating"})

            async for chunk in self.agent.process_stream(
                text=user_msg["content"],
                session_id=session_id,
                user_id=client.user_id,
                brain=message.get("brain", "auto"),
            ):
                await client.send_json(chunk)

        async def handle_signals(client: Any, message: dict[str, Any]) -> None:
            await client.send_json(
                {"type": "signals_snapshot", "signals": self.signals.get_snapshot()}
            )

        async def handle_plugins(client: Any, message: dict[str, Any]) -> None:
            plugins = (
                self.plugin_loader.get_plugin_info() if self.plugin_loader else []
            )
            await client.send_json({"type": "plugins_list", "plugins": plugins})

        async def handle_tts(client: Any, message: dict[str, Any]) -> None:
            if self.voice is None:
                await client.send_json({"type": "error", "error": "voice not ready"})
                return
            text = message.get("text", "")
            async for chunk in self.voice.synthesize_streaming(text):
                await client.send_bytes(chunk)
            await client.send_json({"type": "tts_done"})

        async def handle_stt_start(client: Any, message: dict[str, Any]) -> None:
            """Reset the binary buffer before the browser starts sending audio."""
            client.audio_buffer.clear()
            await client.send_json({"type": "stt_started"})

        async def handle_stt_end(client: Any, message: dict[str, Any]) -> None:
            """
            Finalise a microphone recording: hand the accumulated audio to
            the VoiceManager (→ ``voice_gemma4`` primary, ``voice_canary``
            fallback), then — if ``auto_chat`` is requested — route the
            transcript straight into the chat agent so the whole mic →
            answer cycle happens on one WS round-trip.
            """
            if self.voice is None or not self.voice.has_stt:
                log.warning(
                    "ws.stt_end.no_provider",
                    voice_exists=self.voice is not None,
                    has_stt=self.voice.has_stt if self.voice else False,
                )
                await client.send_json(
                    {"type": "error", "error": "stt not ready (no provider registered)"}
                )
                client.audio_buffer.clear()
                return

            audio_bytes = bytes(client.audio_buffer)
            client.audio_buffer.clear()
            log.info(
                "ws.stt_end.received",
                audio_bytes=len(audio_bytes),
                auto_chat=message.get("auto_chat"),
                session_id=message.get("session_id"),
            )
            if not audio_bytes:
                log.warning("ws.stt_end.empty_buffer")
                await client.send_json(
                    {"type": "stt_result", "text": "", "error": "empty audio buffer"}
                )
                return

            sample_rate = int(message.get("sample_rate") or self.config.voice.sample_rate)
            try:
                text = await self.voice.transcribe(audio_bytes, sample_rate=sample_rate)
            except Exception as exc:  # noqa: BLE001
                log.error("ws.stt_failed", error=str(exc))
                await client.send_json({"type": "error", "error": f"stt: {exc}"})
                return

            log.info(
                "ws.stt_end.transcribed",
                text_length=len(text),
                text_preview=text[:80] if text else "(empty)",
            )
            await client.send_json({"type": "stt_result", "text": text})

            auto_chat = bool(message.get("auto_chat"))
            has_text = bool(text.strip())
            has_agent = self.agent is not None

            if not auto_chat:
                log.debug("ws.stt_end.no_auto_chat")
                return
            if not has_text:
                log.warning(
                    "ws.stt_end.empty_transcript",
                    hint="STT returned empty text — check voice provider logs above.",
                )
                return
            if not has_agent:
                log.error("ws.stt_end.no_agent", hint="Agent is None — chat won't work.")
                return

            log.info("ws.stt_end.forwarding_to_agent", text_preview=text[:80])
            session_id = message.get("session_id") or client.session_id
            async for evt in self.agent.process_stream(
                text=text,
                session_id=session_id,
                user_id=client.user_id,
                brain=message.get("brain", "auto"),
            ):
                await client.send_json(evt)

        self.ws_server.register_handler("chat", handle_chat)
        self.ws_server.register_handler("regenerate", handle_regenerate)
        self.ws_server.register_handler("get_signals", handle_signals)
        self.ws_server.register_handler("get_plugins", handle_plugins)
        self.ws_server.register_handler("tts", handle_tts)
        self.ws_server.register_handler("stt_start", handle_stt_start)
        self.ws_server.register_handler("stt_end", handle_stt_end)

    # ─── Run loop ───────────────────────────────────────────────────

    async def serve(self) -> None:
        """Start uvicorn (HTTP[S] + WebSocket)."""
        if self.fastapi is None:
            raise RuntimeError("LexyApp.startup() must be called before serve()")

        ssl_kwargs: dict[str, Any] = {}
        if self.config.server.ssl_enabled:
            from lexy_core.utils.ssl_utils import ensure_cert

            cert_path, key_path = ensure_cert(
                self.config.server.ssl_certfile,
                self.config.server.ssl_keyfile,
            )
            ssl_kwargs = {
                "ssl_certfile": str(cert_path),
                "ssl_keyfile": str(key_path),
            }
            log.info(
                "app.ssl_enabled",
                cert=str(cert_path),
            )

        config = uvicorn.Config(
            self.fastapi,
            host=self.config.server.host,
            port=self.config.server.port,
            log_config=None,
            access_log=False,
            lifespan="off",
            **ssl_kwargs,
        )
        self._uvicorn_server = uvicorn.Server(config)
        log.info(
            "app.serve",
            host=self.config.server.host,
            port=self.config.server.port,
            scheme="https" if ssl_kwargs else "http",
        )
        await self._uvicorn_server.serve()

    async def run_forever(self) -> None:
        """Startup → serve → shutdown on SIGINT/SIGTERM."""
        await self.startup()

        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def _request_stop() -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):
                # Windows: signal handlers in asyncio aren't always supported
                pass

        serve_task = asyncio.create_task(self.serve(), name="lexy.serve")
        stop_task = asyncio.create_task(stop_event.wait(), name="lexy.stop")

        try:
            done, _pending = await asyncio.wait(
                {serve_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            await self.shutdown()
            for task in (serve_task, stop_task):
                if not task.done():
                    task.cancel()


async def main() -> None:
    """Entry point for ``python -m lexy_core``."""
    app = LexyApp()
    await app.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
