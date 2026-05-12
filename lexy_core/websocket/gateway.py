"""
Lexy AI - FastAPI Gateway.

Builds the FastAPI application:
* HTTP routes for chat / voice / memory / plugins.
* WebSocket route at ``/ws``.

The app keeps a reference to the LexyApp via ``app.state.lexy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lexy_core.uploads import UploadHandler
from lexy_core.uploads.handler import UploadValidationError
from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.app import LexyApp

log = get_logger(module="gateway")


# ─── Pydantic request/response models ────────────────────────────────────────


class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"
    user_id: str = "default"
    brain: str = "auto"


class ChatResponse(BaseModel):
    text: str
    tools_used: list[str] = Field(default_factory=list)
    brain: str = "e4b"


class MemoryStoreRequest(BaseModel):
    text: str
    collection: str = "facts"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecallRequest(BaseModel):
    query: str
    collection: str | None = None
    limit: int = 5


class TTSRequest(BaseModel):
    text: str


class BrainPatch(BaseModel):
    """Partial update for a single brain in config.brains."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repeat_penalty: float | None = None
    thinking: bool | None = None
    reasoning_budget: int | None = None


class RoutingPatch(BaseModel):
    default_brain: str | None = None


class SystemPatch(BaseModel):
    profile: str | None = None
    log_level: str | None = None


class SettingsPatch(BaseModel):
    """Live patch of runtime settings — only a subset is editable."""

    brains: dict[str, BrainPatch] = Field(default_factory=dict)
    routing: RoutingPatch | None = None
    system: SystemPatch | None = None


class VoiceConfigPatch(BaseModel):
    """Runtime tweaks for the active TTS provider (CosyVoice by default)."""

    speed: float | None = None
    default_instruct: str | None = None
    narrator_mode: str | None = None
    segment_pause_ms: int | None = None
    voice: str | None = None


class MemoryDeleteRequest(BaseModel):
    id: str
    collection: str = "facts"


class MemoryWipeRequest(BaseModel):
    """Destructive. ``confirm`` must be True — safety guard."""

    confirm: bool = False
    collections: bool = True      # wipe ChromaDB + FTS5 mirror
    sessions: bool = True         # reset SessionStore (conversation history)
    plugin_data: bool = False     # drop data/plugins/* (timers, thoughts, …)


class PersonaSectionsPatch(BaseModel):
    """Partial update of editable persona sections."""

    identity: str | None = None
    style: str | None = None
    rules: str | None = None


class PersonaPatch(BaseModel):
    """Partial update of the active persona."""

    name: str | None = None
    user_name: str | None = None
    language: str | None = None
    sections: PersonaSectionsPatch | None = None
    thinking_enabled: bool | None = None
    temperature_override: float | None = None
    tags: list[str] | None = None


class MessageEditRequest(BaseModel):
    content: str


class SessionRegisterRequest(BaseModel):
    """
    Register an empty session slot on the backend. Called by the frontend
    when the user starts a new chat so the session exists in
    ``data/sessions.json`` even before the first message is sent — this
    fixes the "in-progress chat disappears after restart" bug.
    """

    session_id: str
    project_id: str | None = None
    title: str | None = None


class SessionPatchRequest(BaseModel):
    """
    Update metadata on an existing session. Currently supports moving a
    session between projects, renaming its title, and switching its kind
    (Phase 9.12 — Chat / Roleplay split).
    """

    project_id: str | None = None
    title: str | None = None
    kind: str | None = None  # "chat" | "rp"


class RPSessionCreateRequest(BaseModel):
    """Phase 13 — create an RP session with its own container.

    The session_id is generated client-side (UUID hex). ``tracked_stats``
    is what Mike types in the session modal as semicolon-separated
    ``key=value`` pairs — we accept either the parsed dict or the raw
    string. ``scene`` is optional and seeds the container's session.json.
    """

    session_id: str
    title: str = ""
    scene: str = ""
    tracked_stats: dict[str, str] | str | None = None
    project_id: str | None = None


class ProjectCreateRequest(BaseModel):
    """Create a new project from the sidebar."""

    name: str
    description: str = ""
    color: str = "#7aa2f7"
    icon: str = ""
    persona_override: str = ""
    memory_scoped: bool = True


class ProjectUpdateRequest(BaseModel):
    """Partial update of an existing project."""

    name: str | None = None
    description: str | None = None
    color: str | None = None
    icon: str | None = None
    persona_override: str | None = None
    memory_scoped: bool | None = None


# ─── App factory ─────────────────────────────────────────────────────────────


def build_app(lexy: "LexyApp") -> FastAPI:
    """Construct the FastAPI app for an initialised LexyApp."""

    api = FastAPI(
        title=lexy.config.system.name,
        version=lexy.config.system.version,
        description="Lexy AI – Local-first AI assistant",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.state.lexy = lexy

    def _app(request: Request) -> "LexyApp":
        return request.app.state.lexy  # type: ignore[no-any-return]

    # ─── Health ──────────────────────────────────────────────────

    @api.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        services: dict[str, str] = {
            "llm": "ready" if lexy.llm and lexy.llm._connected else "down",
            "memory": "ready" if lexy.memory else "off",
            "voice": "ready" if lexy.voice and lexy.voice.has_tts else "off",
            "plugins": str(lexy.plugin_loader.loaded_count if lexy.plugin_loader else 0),
        }
        return {
            "status": lexy.signals.get("system_state").value,
            "version": lexy.config.system.version,
            "services": services,
        }

    # ─── Chat ────────────────────────────────────────────────────

    @api.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        app = _app(request)
        if app.agent is None:
            raise HTTPException(503, "Agent not initialised")
        result = await app.agent.process(
            text=req.text,
            session_id=req.session_id,
            user_id=req.user_id,
            brain=req.brain,
        )
        return ChatResponse(
            text=result.get("text", ""),
            tools_used=result.get("tools_used", []),
            brain=result.get("brain", "e4b"),
        )

    # ─── Memory ──────────────────────────────────────────────────

    @api.post("/api/v1/memory/store")
    async def memory_store(req: MemoryStoreRequest, request: Request) -> dict[str, str]:
        app = _app(request)
        if app.memory is None:
            raise HTTPException(503, "Memory not initialised")
        item_id = await app.memory.store(
            text=req.text, collection=req.collection, metadata=req.metadata
        )
        return {"id": item_id}

    @api.post("/api/v1/memory/recall")
    async def memory_recall(
        req: MemoryRecallRequest, request: Request
    ) -> dict[str, list[dict[str, Any]]]:
        app = _app(request)
        if app.memory is None:
            raise HTTPException(503, "Memory not initialised")
        results = await app.memory.recall(
            query=req.query, collection=req.collection, limit=req.limit
        )
        return {"results": results}

    @api.get("/api/v1/memory/browse")
    async def memory_browse(
        request: Request,
        collection: str = "facts",
        page: int = 1,
        limit: int = 25,
    ) -> dict[str, Any]:
        app = _app(request)
        if app.memory is None:
            raise HTTPException(503, "Memory not initialised")
        items, total = await app.memory.browse(
            collection=collection, page=page, limit=limit
        )
        return {"items": items, "total": total}

    # ─── Voice ───────────────────────────────────────────────────

    @api.post("/api/v1/voice/transcribe")
    async def voice_transcribe(request: Request) -> dict[str, str]:
        app = _app(request)
        if app.voice is None:
            raise HTTPException(503, "Voice not initialised")
        body = await request.body()
        text = await app.voice.transcribe(body)
        return {"text": text}

    @api.post("/api/v1/voice/synthesize")
    async def voice_synthesize(req: TTSRequest, request: Request) -> Response:
        app = _app(request)
        if app.voice is None:
            raise HTTPException(503, "Voice not initialised")
        audio = await app.voice.synthesize(req.text)
        return Response(content=audio, media_type="audio/wav")

    # ─── Plugins ─────────────────────────────────────────────────

    @api.get("/api/v1/plugins")
    async def list_plugins(request: Request) -> dict[str, Any]:
        app = _app(request)
        if app.plugin_loader is None:
            return {"plugins": []}
        return {"plugins": app.plugin_loader.get_plugin_info()}

    @api.post("/api/v1/plugins/{name}/disable")
    async def disable_plugin(name: str, request: Request) -> dict[str, str]:
        app = _app(request)
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        await app.plugin_loader.disable_plugin(name)
        return {"status": "disabled"}

    @api.post("/api/v1/plugins/{name}/enable")
    async def enable_plugin(name: str, request: Request) -> dict[str, Any]:
        """Enable a plugin.

        Returns a structured response so the frontend can show a useful
        toast instead of a bare 500. Two outcomes:

        * ``{"status": "enabled", "degraded": False}`` — plugin is fully
          operational.
        * ``{"status": "enabled", "degraded": True, "last_error": ...}``
          — plugin loaded but its provider couldn't be initialised
          (e.g. CosyVoice server unreachable). The plugin is still
          registered, only the provider is missing.

        Hard failures (manifest invalid, import explodes before
        ``on_load`` can swallow it) come back as ``422`` with a
        ``detail`` payload that the frontend renders verbatim — much
        more useful than ``HTTP 500``.
        """
        app = _app(request)
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        try:
            await app.plugin_loader._load_plugin(name)  # type: ignore[attr-defined]
            await app.plugin_loader._enable_plugin(name)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "plugin_enable_failed",
                    "plugin": name,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            ) from exc

        plugin = app.plugin_loader.get_plugin(name)
        last_error = ""
        degraded = False
        if plugin is not None and hasattr(plugin, "get_status"):
            try:
                status = plugin.get_status() or {}
                last_error = str(status.get("last_error") or "")
                # A plugin can declare itself "degraded" by exposing a
                # falsy ``*_active`` field. Otherwise we infer from
                # the presence of last_error.
                if last_error:
                    degraded = True
            except Exception as exc:  # noqa: BLE001
                last_error = f"get_status raised: {exc}"
                degraded = True
        return {
            "status": "enabled",
            "degraded": degraded,
            "last_error": last_error,
        }

    @api.get("/api/v1/plugins/{name}/status")
    async def plugin_status(name: str, request: Request) -> dict[str, Any]:
        """Return plugin runtime state.

        Plugins can implement ``get_status() -> dict`` to surface
        custom fields (e.g. ``server_url``, ``module_importable``,
        ``last_error``). The dashboard's green-dot and the plugin
        tab's offline-badge both read from here, so they stay in
        sync — fixes the Phase 9.10 inconsistency Mike flagged on
        the CosyVoice plugin.
        """
        app = _app(request)
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        manifest = app.plugin_loader.get_manifests().get(name)
        if manifest is None:
            raise HTTPException(404, f"unknown plugin: {name}")
        plugin = app.plugin_loader.get_plugin(name)
        loaded = plugin is not None
        enabled = bool(plugin.enabled) if plugin is not None else False
        custom: dict[str, Any] = {}
        if plugin is not None and hasattr(plugin, "get_status"):
            try:
                raw = plugin.get_status() or {}
                if isinstance(raw, dict):
                    custom = raw
            except Exception as exc:  # noqa: BLE001
                custom = {"last_error": f"get_status raised: {exc}"}
        return {
            "name": name,
            "loaded": loaded,
            "enabled": enabled,
            "version": manifest.version,
            "description": manifest.description,
            **custom,
        }

    @api.get("/api/v1/plugins/{name}/config")
    async def get_plugin_config(name: str, request: Request) -> dict[str, Any]:
        """
        Return a plugin's current config, split into:

        * ``defaults``  — keys straight from ``plugin.yaml#config_defaults``
        * ``overrides`` — user-supplied values from ``config/plugins.yaml``
        * ``effective`` — the merged dict that ``api.get_config()`` returns
        * ``description`` — one-line description from the manifest
        """
        app = _app(request)
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        manifest = app.plugin_loader.get_manifests().get(name)
        if manifest is None:
            raise HTTPException(404, f"unknown plugin: {name}")
        defaults = dict(manifest.config_defaults or {})
        overrides = dict(app.plugin_overrides.get(name, {}))
        effective: dict[str, Any] = {**defaults, **overrides}
        return {
            "name": name,
            "version": manifest.version,
            "description": manifest.description,
            "defaults": defaults,
            "overrides": overrides,
            "effective": effective,
        }

    # ─── character_chat: avatar upload + import ──────────────────────

    CHARACTER_AVATAR_DIR = Path("data/plugins/character_chat/avatars")
    CHARACTER_AVATAR_MAX_BYTES = 512 * 1024  # 512 KiB — keeps the GUI snappy
    _ALLOWED_AVATAR_EXT = {"png", "jpg", "jpeg", "webp", "gif"}

    @api.post("/api/v1/plugins/character_chat/avatars")
    async def upload_character_avatar(
        request: Request,
        character_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Upload or replace a character's avatar image.

        Persists the file to ``data/plugins/character_chat/avatars/`` as
        ``<character_id>.<ext>`` and patches the card's ``avatar`` field
        with the relative path. Served via the ``/avatars`` static mount.
        """
        app = _app(request)
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        plugin = app.plugin_loader.get_plugin("character_chat")
        if plugin is None:
            raise HTTPException(404, "character_chat plugin not loaded")
        store = getattr(plugin, "_store", None)
        if store is None:
            raise HTTPException(503, "character_chat store not ready")

        existing = await store.get(character_id)
        if existing is None:
            raise HTTPException(404, f"character not found: {character_id}")

        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in _ALLOWED_AVATAR_EXT:
            raise HTTPException(400, f"unsupported image type: {ext!r}")

        data = await file.read()
        if len(data) > CHARACTER_AVATAR_MAX_BYTES:
            raise HTTPException(
                413, f"avatar too large (max {CHARACTER_AVATAR_MAX_BYTES // 1024} KiB)"
            )
        if not data:
            raise HTTPException(400, "empty upload")

        CHARACTER_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        # Remove previous file of a different extension so we don't leave
        # orphans (e.g. swapping from .png to .webp).
        for stale in CHARACTER_AVATAR_DIR.glob(f"{character_id}.*"):
            try:
                stale.unlink()
            except OSError:
                pass
        avatar_path = CHARACTER_AVATAR_DIR / f"{character_id}.{ext}"
        avatar_path.write_bytes(data)

        # The card's ``avatar`` field holds the relative URL served by the
        # static mount below. That way the frontend only needs to render
        # <img src="/avatars/<file>"> without knowing about data paths.
        url = f"/avatars/{avatar_path.name}"
        try:
            updated = await store.update(character_id, avatar=url)
        except Exception as exc:  # noqa: BLE001
            log.error("character_chat.avatar_patch_failed", error=str(exc))
            raise HTTPException(500, f"avatar patch failed: {exc}") from exc

        # Broadcast so open tabs refresh their character list.
        await plugin._broadcast_character_event("character_updated", updated)

        return {
            "ok": True,
            "character_id": character_id,
            "avatar": url,
            "size": len(data),
            "content_type": file.content_type,
        }

    @api.post("/api/v1/plugins/character_chat/import")
    async def import_character_card(
        request: Request,
        file: UploadFile = File(...),
        color: str = Form(""),
        age_stage: str = Form("adult"),
    ) -> dict[str, Any]:
        """Import a Silly-Tavern character card.

        Accepts:
        * **JSON file** (``application/json`` or ``.json``) — v1 flat
          or v2 ``{spec,data}`` shape.
        * **PNG file** (``image/png`` or ``.png``) — Silly-Tavern PNG
          card with the character JSON in a ``chara`` tEXt chunk. The
          embedded image becomes the new character's avatar.

        Returns the persisted card. Broadcasts ``character_created``
        on success.
        """
        app = _app(request)
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        plugin = app.plugin_loader.get_plugin("character_chat")
        if plugin is None:
            raise HTTPException(404, "character_chat plugin not loaded")
        store = getattr(plugin, "_store", None)
        if store is None:
            raise HTTPException(503, "character_chat store not ready")

        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")

        try:
            saved = await store.import_silly_tavern_bytes(
                data,
                filename=file.filename or "",
                content_type=file.content_type or "",
                color=color or None,
                age_stage=age_stage,
                avatar_dir=CHARACTER_AVATAR_DIR,
            )
        except Exception as exc:  # noqa: BLE001 — user-facing error message
            raise HTTPException(400, f"import failed: {exc}") from exc

        # Broadcast so open tabs refresh their character list. Mirrors
        # the avatar-upload endpoint pattern.
        try:
            await plugin._broadcast_character_event("character_created", saved)
        except Exception:  # noqa: BLE001
            pass

        return {
            "ok": True,
            "character": {
                "id": saved.id,
                "name": saved.name,
                "avatar": saved.avatar,
                "age_stage": saved.age_stage,
            },
            "size": len(data),
            "content_type": file.content_type,
        }

    # ─── Lorebooks (Phase 9.11 — REST mirror of WS handlers) ─────────
    #
    # The character_chat plugin already exposes ``_tool_lorebook_*`` and
    # ``_tool_lore_entry_*`` plus the matching WS handlers (Phase 9.8).
    # The frontend editor uses REST instead of WS round-trips because
    # synchronous request/response is cleaner for plain CRUD.
    #
    # All routes go through the plugin's tool methods so the broadcast
    # events (`lorebook_created`, etc.) keep firing — open tabs that
    # subscribed via WS still see live updates after a REST mutation.

    def _character_chat_plugin(app: Any) -> Any:
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        plugin = app.plugin_loader.get_plugin("character_chat")
        if plugin is None:
            raise HTTPException(404, "character_chat plugin not loaded")
        if getattr(plugin, "_lore_store", None) is None:
            raise HTTPException(503, "lorebook store not ready")
        return plugin

    def _lore_unwrap(result: dict[str, Any], not_found_msg: str) -> dict[str, Any]:
        """Convert a tool result ``{ok, ...}`` into a REST response.

        ``ok=False`` from the tool layer is mostly user-input error
        (unknown scope, bad position, missing key on a non-always-on
        entry) — those become 400. Specifically "not found" results
        are remapped to 404 so the frontend can branch on it.
        """
        if not result.get("ok"):
            err = str(result.get("error") or "")
            if "not found" in err.lower():
                raise HTTPException(404, err or not_found_msg)
            raise HTTPException(400, err or "request failed")
        # Strip ``ok`` from the response — caller doesn't need it,
        # the HTTP status already says "success".
        out = dict(result)
        out.pop("ok", None)
        return out

    @api.get("/api/v1/plugins/character_chat/lorebooks")
    async def list_lorebooks(
        request: Request,
        scope: str | None = None,
        scope_id: str | None = None,
        enabled_only: bool = False,
    ) -> dict[str, Any]:
        """List lorebooks, optionally filtered by scope / scope_id.

        Query params:
        * ``scope`` — ``global`` | ``character`` | ``session``. Omit
          to list every book regardless of scope.
        * ``scope_id`` — character_id or session_id. Required when
          ``scope`` is ``character`` or ``session``; ignored otherwise.
        * ``enabled_only`` — set ``true`` to hide disabled books.
        """
        app = _app(request)
        plugin = _character_chat_plugin(app)
        result = await plugin._tool_lorebook_list(
            scope=scope, scope_id=scope_id, enabled_only=enabled_only,
        )
        return _lore_unwrap(result, "no lorebooks")

    @api.post("/api/v1/plugins/character_chat/lorebooks")
    async def create_lorebook(
        request: Request,
        body: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Create a lorebook.

        Body: ``{name, description?, scope?, scope_id?, token_budget?}``.
        ``scope=character|session`` requires a non-empty ``scope_id``.
        """
        app = _app(request)
        plugin = _character_chat_plugin(app)
        result = await plugin._tool_lorebook_create(
            name=str(body.get("name") or "").strip(),
            description=str(body.get("description") or ""),
            scope=str(body.get("scope") or "global"),
            scope_id=str(body.get("scope_id") or ""),
            token_budget=int(body.get("token_budget") or 1500),
        )
        return _lore_unwrap(result, "create failed")

    @api.patch("/api/v1/plugins/character_chat/lorebooks/{book_id}")
    async def patch_lorebook(
        book_id: str,
        request: Request,
        body: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Patch an existing lorebook. Only known fields are applied.

        Allowed fields: ``name``, ``description``, ``scope``,
        ``scope_id``, ``enabled``, ``token_budget``.
        """
        app = _app(request)
        plugin = _character_chat_plugin(app)
        # Strip unknown keys so the tool's allow-list is the source of
        # truth — we don't have to mirror it here.
        result = await plugin._tool_lorebook_update(id=book_id, **body)
        return _lore_unwrap(result, "lorebook not found")

    @api.delete("/api/v1/plugins/character_chat/lorebooks/{book_id}")
    async def delete_lorebook(
        book_id: str, request: Request,
    ) -> dict[str, Any]:
        """Delete a lorebook + all its entries (cascade)."""
        app = _app(request)
        plugin = _character_chat_plugin(app)
        result = await plugin._tool_lorebook_delete(id=book_id)
        if not result.get("ok"):
            raise HTTPException(404, "lorebook not found")
        return {"id": book_id}

    @api.get(
        "/api/v1/plugins/character_chat/lorebooks/{book_id}/entries"
    )
    async def list_lore_entries(
        book_id: str,
        request: Request,
        enabled_only: bool = False,
    ) -> dict[str, Any]:
        """List entries inside a lorebook, sorted by priority then name."""
        app = _app(request)
        plugin = _character_chat_plugin(app)
        result = await plugin._tool_lore_entry_list(
            lorebook_id=book_id, enabled_only=enabled_only,
        )
        return _lore_unwrap(result, "no entries")

    @api.post(
        "/api/v1/plugins/character_chat/lorebooks/{book_id}/entries"
    )
    async def create_lore_entry(
        book_id: str,
        request: Request,
        body: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Create a lore entry under the given book.

        Body: ``{name, keys?, content?, position?, priority?,
        always_on?, scan_depth?}``. Rule: at least one key OR
        ``always_on=True`` (otherwise the entry would never fire).
        """
        app = _app(request)
        plugin = _character_chat_plugin(app)
        result = await plugin._tool_lore_entry_create(
            lorebook_id=book_id,
            name=str(body.get("name") or "").strip(),
            keys=list(body.get("keys") or []),
            content=str(body.get("content") or ""),
            position=str(body.get("position") or "before_scenario"),
            priority=int(body.get("priority") or 100),
            always_on=bool(body.get("always_on", False)),
            scan_depth=int(body.get("scan_depth") or 4),
        )
        return _lore_unwrap(result, "create failed")

    @api.patch("/api/v1/plugins/character_chat/lore_entries/{entry_id}")
    async def patch_lore_entry(
        entry_id: str,
        request: Request,
        body: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Patch a lore entry. Allowed fields: name, keys, content,
        position, priority, always_on, scan_depth, enabled."""
        app = _app(request)
        plugin = _character_chat_plugin(app)
        result = await plugin._tool_lore_entry_update(id=entry_id, **body)
        return _lore_unwrap(result, "entry not found")

    @api.delete("/api/v1/plugins/character_chat/lore_entries/{entry_id}")
    async def delete_lore_entry(
        entry_id: str, request: Request,
    ) -> dict[str, Any]:
        """Delete a single lore entry."""
        app = _app(request)
        plugin = _character_chat_plugin(app)
        result = await plugin._tool_lore_entry_delete(id=entry_id)
        if not result.get("ok"):
            raise HTTPException(404, "entry not found")
        return {"id": entry_id}

    @api.get("/api/v1/plugins/character_chat/characters/{char_id}/prompt")
    async def get_character_prompt_preview(
        char_id: str, request: Request, session_id: str = "",
    ) -> dict[str, Any]:
        """Render the system prompt that would be sent to the LLM for
        this character — useful for debugging "why does Sandra still
        think she's wearing a Shirt".

        Mike's report: characters keep referring to clothes that
        aren't in their state anymore. The cause is usually one of:
          * persona text still mentions the clothes
          * example_dialog has stale clothing references (and the
            LLM uses the example as a few-shot anchor)
          * state.clothing wasn't actually saved
          * scenario text describes the wrong outfit
        This endpoint dumps the rendered prompt so we can see at a
        glance which source contains the stale text.

        Query param ``session_id`` (optional) — if given, the prompt
        includes the session's other characters under "## Andere
        Anwesende".
        """
        app = _app(request)
        plugin = _character_chat_plugin(app)
        store = getattr(plugin, "_store", None)
        if store is None:
            raise HTTPException(503, "character store not ready")
        card = await store.get(char_id)
        if card is None:
            raise HTTPException(404, f"character not found: {char_id}")

        other_characters = []
        if session_id:
            try:
                bound = await store.list_in_session(session_id)
                other_characters = [c for c in bound if c.id != char_id]
            except Exception:  # noqa: BLE001
                other_characters = []

        # Phase 13: when the session is an RP session with a container,
        # pull the live state for THIS character (overrides card.state)
        # and the session's tracked_stats list (drives the rules block).
        live_state: dict[str, str] | None = None
        tracked_stats: dict[str, str] | None = None
        scene_text = ""
        rp_registry = getattr(plugin, "_rp_registry", None)
        if session_id and rp_registry is not None and rp_registry.is_rp_session(
            session_id,
        ):
            try:
                container = await rp_registry.get(session_id)
                if container is not None:
                    live_state = await container.get_char_state(char_id)
                    tracked_stats = await container.get_tracked_stats()
                    meta = await container.get_meta()
                    scene_text = str(meta.get("scene", "") or "")
            except Exception:  # noqa: BLE001
                pass

        prompt = card.build_system_prompt(
            other_characters=other_characters,
            scene=scene_text,
            live_state=live_state,
            tracked_stats=tracked_stats,
        )
        return {
            "character_id": card.id,
            "character_name": card.name,
            "session_id": session_id,
            "prompt": prompt,
            "prompt_length": len(prompt),
            # Phase 13: session-live state if available, else legacy.
            "state": (live_state if live_state is not None else dict(card.state)),
            "tracked_stats": tracked_stats or {},
            "persona": card.persona,
            "scenario": card.scenario,
            "example_dialog": card.example_dialog,
        }

    @api.get(
        "/api/v1/plugins/character_chat/sessions/{session_id}/turns"
    )
    async def get_session_character_turns(
        session_id: str, request: Request, limit: int = 500,
    ) -> dict[str, Any]:
        """Return all character_turns for a session, chronologically.

        Phase 11 fix — Mike reported that resuming an RP session only
        showed user messages because ``/sessions/{id}/history`` reads
        from the agent's session_store (which has user/assistant rows
        only). Character bubbles persist in a SEPARATE table
        (``character_turns`` in ``data/plugins/character_chat/...``)
        so the frontend has to fetch them as a second step.

        Returns turns in chronological order with everything the UI
        needs to render an action bar (turn_id, character_id, content,
        round_id + trigger_text for interleaving with user messages).
        """
        app = _app(request)
        plugin = _character_chat_plugin(app)
        capped = max(1, min(2000, int(limit)))

        # Phase 13: RP sessions store their turns in a per-session
        # SQLite under ``data/rp_sessions/<id>/turns.db``. Try that
        # first; fall back to the legacy global ``character_turns``
        # table for any non-RP / pre-Phase-13 session.
        rp_registry = getattr(plugin, "_rp_registry", None)
        if rp_registry is not None and rp_registry.is_rp_session(session_id):
            container = await rp_registry.get(session_id)
            if container is not None:
                rows = await container.list_turns(limit=capped)
                return {
                    "session_id": session_id,
                    "turns": [
                        {
                            "turn_id": t.id,
                            "character_id": t.character_id,
                            "character_name": t.character_name,
                            "round_id": t.round_id,
                            "order": t.order_num,
                            "content": t.content,
                            "skipped": t.skipped,
                            "trigger_kind": t.trigger_kind,
                            "trigger_text": t.trigger_text,
                            "created_at": t.created_at,
                        }
                        for t in rows
                    ],
                }

        # Legacy / non-RP fallback.
        db = await plugin.api.get_db()
        cursor = await db.execute(
            "SELECT id, character_id, character_name, round_id, "
            "order_num, content, skipped, trigger_kind, trigger_text, "
            "created_at FROM character_turns "
            "WHERE session_id = ? "
            "ORDER BY created_at ASC, order_num ASC LIMIT ?",
            (session_id, capped),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return {
            "session_id": session_id,
            "turns": [
                {
                    "turn_id": r[0],
                    "character_id": r[1],
                    "character_name": r[2],
                    "round_id": r[3],
                    "order": int(r[4] or 0),
                    "content": r[5] or "",
                    "skipped": bool(r[6]),
                    "trigger_kind": r[7] or "",
                    "trigger_text": r[8] or "",
                    "created_at": float(r[9] or 0.0),
                }
                for r in rows
            ],
        }

    # ─── Skill packager (Phase 11 — agentskills.io) ────────────────
    #
    # Two endpoints flank the existing skill_writer WS handlers:
    # * POST /api/v1/plugins/skill_writer/skills/import (multipart .zip)
    # * GET  /api/v1/plugins/skill_writer/skills/{name}/export
    #
    # Both delegate to plugins.skill_writer.skill_packager which has
    # the path-traversal/zip-bomb/spec-validation guards. The plugin
    # itself stays in the loop so registry + WS broadcast stay in
    # sync — REST is just a different transport.

    def _skill_writer_plugin(app: Any) -> Any:
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        plugin = app.plugin_loader.get_plugin("skill_writer")
        if plugin is None:
            raise HTTPException(404, "skill_writer plugin not loaded")
        if (
            getattr(plugin, "_validator", None) is None
            or getattr(plugin, "_registry", None) is None
        ):
            raise HTTPException(503, "skill_writer not ready")
        return plugin

    @api.post("/api/v1/plugins/skill_writer/skills/import")
    async def import_skill(
        request: Request,
        file: UploadFile = File(...),
        overwrite: bool = Form(False),
    ) -> dict[str, Any]:
        """Import an agentskills.io-shaped skill from a .zip upload.

        Body (multipart):
        * ``file`` — the ZIP archive (top-level folder is the skill).
        * ``overwrite`` — set ``true`` to replace an existing skill of
          the same name. Default ``false`` returns 409 on conflict.

        Returns ``{ok, skill: SkillCardPublic, overwrote_existing}``.
        """
        from plugins.skill_writer.skill_packager import (
            SkillImportConflict,
            SkillPackageError,
            import_skill_zip,
        )

        app = _app(request)
        plugin = _skill_writer_plugin(app)
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")

        try:
            result = await import_skill_zip(
                data,
                dest_root=plugin._skills_path,
                validator=plugin._validator,
                overwrite=overwrite,
            )
        except SkillImportConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except SkillPackageError as exc:
            raise HTTPException(400, str(exc)) from exc

        # Sync the registry: insert (or update_metadata if overwriting)
        # so the imported skill shows up in list_skills immediately.
        existing = await plugin._registry.get(result.card.name)
        if existing is None:
            await plugin._registry.register(
                name=result.card.name,
                description=result.card.description,
                file_path=str(result.card.folder),
                source="imported",
                license=result.card.frontmatter.license,
                compatibility=result.card.frontmatter.compatibility,
                metadata=result.card.frontmatter.metadata,
                allowed_tools=result.card.frontmatter.allowed_tools,
                body_md=result.card.frontmatter.body,
            )
        else:
            await plugin._registry.update_metadata(
                result.card.name,
                description=result.card.description,
                license=result.card.frontmatter.license,
                compatibility=result.card.frontmatter.compatibility,
                metadata=result.card.frontmatter.metadata,
                allowed_tools=result.card.frontmatter.allowed_tools,
                body_md=result.card.frontmatter.body,
            )

        # Broadcast so any open Skills tab refreshes.
        try:
            await app.ws_server.broadcast({
                "type": "skill_imported",
                "name": result.card.name,
                "overwrote_existing": result.overwrote_existing,
            })
        except Exception:  # noqa: BLE001
            pass

        return {
            "ok": True,
            "overwrote_existing": result.overwrote_existing,
            "skill": result.card.to_public(),
        }

    @api.get("/api/v1/plugins/skill_writer/skills/{name}/export")
    async def export_skill(
        name: str, request: Request,
    ) -> Response:
        """Pack the named skill into a downloadable ZIP."""
        from plugins.skill_writer.skill_packager import (
            SkillPackageError,
            export_skill_zip,
        )

        app = _app(request)
        plugin = _skill_writer_plugin(app)
        entry = await plugin._registry.get(name)
        if entry is None:
            raise HTTPException(404, f"skill not found: {name}")

        from pathlib import Path as _Path
        folder = _Path(entry.file_path)
        if not folder.is_dir():
            raise HTTPException(
                404, f"skill folder missing on disk: {folder}"
            )

        try:
            zip_bytes = await export_skill_zip(folder)
        except SkillPackageError as exc:
            raise HTTPException(400, str(exc)) from exc

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{name}.zip"'
                ),
            },
        )

    # ─── Chat-attachment uploads ──────────────────────────────────────
    #
    # Four endpoints, one per kind. They all return a dict the frontend
    # passes verbatim into the next chat message's ``attachments`` array,
    # so the agent can build a multimodal user message (image_url block
    # for vision) or a doc-excerpt block. Storage is at
    # ``data/uploads/<session>/<upload_id>.<ext>`` and served via the
    # ``/uploads`` StaticFiles mount registered later in this function.

    async def _do_upload(
        request: Request, kind: str, file: UploadFile, session_id: str
    ) -> dict[str, Any]:
        app = _app(request)
        if file is None:
            raise HTTPException(400, "no file provided")
        if not session_id:
            raise HTTPException(400, "session_id required")
        try:
            handler = await UploadHandler.from_app(app)
            data = await file.read()
            if kind == "image":
                result = await handler.handle_image(
                    data=data,
                    filename=file.filename or "image",
                    mime=file.content_type or "",
                    session_id=session_id,
                )
            elif kind == "document":
                result = await handler.handle_document(
                    data=data,
                    filename=file.filename or "document",
                    mime=file.content_type or "",
                    session_id=session_id,
                )
            elif kind == "code":
                result = await handler.handle_code(
                    data=data,
                    filename=file.filename or "code.txt",
                    mime=file.content_type or "",
                    session_id=session_id,
                )
            elif kind == "audio":
                result = await handler.handle_audio(
                    data=data,
                    filename=file.filename or "audio",
                    mime=file.content_type or "",
                    session_id=session_id,
                )
            else:
                raise HTTPException(400, f"unknown upload kind: {kind!r}")
        except UploadValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("uploads.failed kind=%s error=%s", kind, exc)
            raise HTTPException(500, f"upload failed: {exc}") from exc
        return result.payload

    @api.post("/api/v1/uploads/image")
    async def upload_image(
        request: Request,
        session_id: str = Form("default"),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        return await _do_upload(request, "image", file, session_id)

    @api.post("/api/v1/uploads/document")
    async def upload_document(
        request: Request,
        session_id: str = Form("default"),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        return await _do_upload(request, "document", file, session_id)

    @api.post("/api/v1/uploads/code")
    async def upload_code(
        request: Request,
        session_id: str = Form("default"),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        return await _do_upload(request, "code", file, session_id)

    @api.post("/api/v1/uploads/audio")
    async def upload_audio(
        request: Request,
        session_id: str = Form("default"),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        return await _do_upload(request, "audio", file, session_id)

    @api.delete("/api/v1/uploads/{upload_id}")
    async def delete_upload(upload_id: str, request: Request) -> dict[str, Any]:
        app = _app(request)
        handler = await UploadHandler.from_app(app)
        ok = await handler.store.delete(upload_id)
        if not ok:
            raise HTTPException(404, "upload not found")
        return {"ok": True, "upload_id": upload_id}

    # NOTE: ``/api/v1/plugins/{name}/status`` is registered earlier (Phase
    # 9.10) — the older variant here used to return a different schema.
    # We keep the single registration to avoid FastAPI route shadowing.

    @api.patch("/api/v1/plugins/{name}/config")
    async def patch_plugin_config(
        name: str,
        request: Request,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update a plugin's config in memory AND persist it to
        ``config/plugins.yaml`` so the change survives restart.

        The patch is a flat ``{key: value}`` dict. Keys not present in the
        manifest's ``config_defaults`` are still accepted (plugins can
        define new runtime-only keys). Secrets should be passed via env
        vars, not via this endpoint — values are written to disk.
        """
        app = _app(request)
        if app.plugin_loader is None:
            raise HTTPException(503, "Plugin loader not initialised")
        if name not in app.plugin_loader.get_manifests():
            raise HTTPException(404, f"unknown plugin: {name}")

        if not isinstance(patch, dict):
            raise HTTPException(400, "patch body must be a JSON object")

        existing = dict(app.plugin_overrides.get(name, {}))
        existing.update(patch)
        app.plugin_overrides[name] = existing

        # Persist to config/plugins.yaml (writing the whole overrides map).
        import yaml as _yaml

        plugins_yaml = Path("config/plugins.yaml")
        try:
            plugins_yaml.parent.mkdir(parents=True, exist_ok=True)
            with open(plugins_yaml, "w", encoding="utf-8") as handle:
                _yaml.safe_dump(
                    app.plugin_overrides,
                    handle,
                    sort_keys=True,
                    allow_unicode=True,
                )
        except OSError as exc:
            log.error("plugin_config.write_failed", plugin=name, error=str(exc))
            raise HTTPException(500, f"write failed: {exc}") from exc

        # Try to apply the new config live. Plugins that override
        # ``on_config_changed`` pick up the patch without a restart.
        from lexy_core.plugin_system.base_plugin import BasePlugin  # local import

        applied_live = False
        plugin = app.plugin_loader.get_plugin(name)
        if plugin is not None:
            hook = getattr(plugin, "on_config_changed", None)
            if callable(hook):
                manifest = app.plugin_loader.get_manifests().get(name)
                merged: dict[str, Any] = dict(
                    (manifest.config_defaults if manifest else None) or {}
                )
                merged.update(app.plugin_overrides.get(name, {}))
                # Only call + advertise live if the plugin actually overrode the hook.
                overrides_hook = (
                    type(plugin).on_config_changed is not BasePlugin.on_config_changed
                )
                if overrides_hook:
                    try:
                        result = hook(merged)
                        if hasattr(result, "__await__"):
                            await result
                        applied_live = True
                    except Exception as exc:  # noqa: BLE001
                        log.error(
                            "plugin_config.on_config_changed_failed",
                            plugin=name,
                            error=str(exc),
                        )

        log.info(
            "plugin_config.updated",
            plugin=name,
            keys=list(patch.keys()),
            applied_live=applied_live,
        )
        return {
            "status": "ok",
            "restart_required": not applied_live,
            "applied_live": applied_live,
            "overrides": app.plugin_overrides.get(name, {}),
        }

    # ─── Settings (live, subset of config) ──────────────────────

    @api.get("/api/v1/settings")
    async def get_settings(request: Request) -> dict[str, Any]:
        app = _app(request)
        cfg = app.config
        return {
            "system": {
                "name": cfg.system.name,
                "version": cfg.system.version,
                "debug": cfg.system.debug,
                "log_level": cfg.system.log_level,
                "profile": cfg.system.profile,
                "active_brains": sorted(cfg.active_brain_names()),
            },
            "brains": {
                name: {
                    "model": brain.model,
                    "endpoint": brain.endpoint,
                    "context_size": brain.context_size,
                    "max_tokens": brain.max_tokens,
                    "temperature": brain.temperature,
                    "top_p": brain.top_p,
                    "top_k": brain.top_k,
                    "repeat_penalty": brain.repeat_penalty,
                    "thinking": brain.thinking,
                    "reasoning_budget": brain.reasoning_budget,
                    "timeout": brain.timeout,
                }
                for name, brain in cfg.brains.items()
            },
            "routing": {
                "default_brain": cfg.routing.default_brain,
                "rules": [
                    {"pattern": r.pattern, "brain": r.brain, "max_tokens": r.max_tokens}
                    for r in cfg.routing.rules
                ],
            },
            "voice": {
                "stt_enabled": cfg.voice.stt_enabled,
                "tts_enabled": cfg.voice.tts_enabled,
                "sample_rate": cfg.voice.sample_rate,
                "stt_primary": cfg.voice.stt.primary,
                "stt_fallback": cfg.voice.stt.fallback,
                "tts_primary": cfg.voice.tts.primary,
                "tts_endpoint": cfg.voice.tts.endpoint,
                "tts_voice": cfg.voice.tts.voice,
                "tts_speed": cfg.voice.tts.speed,
            },
            "memory": {
                "chroma_host": cfg.memory.chroma_host,
                "chroma_port": cfg.memory.chroma_port,
                "collections": cfg.memory.collections,
                "recall_limit": cfg.memory.recall_limit,
                "recall_threshold": cfg.memory.recall_threshold,
                "vector_weight": cfg.memory.vector_weight,
                "bm25_weight": cfg.memory.bm25_weight,
            },
            "embedding": {
                "model": cfg.embedding.model,
                "device": cfg.embedding.device,
                "dimension": cfg.embedding.dimension,
            },
            "plugins": {
                "path": cfg.plugins.path,
                "enabled": cfg.plugins.enabled,
                "disabled": cfg.plugins.disabled,
                "hot_reload": cfg.plugins.hot_reload,
            },
        }

    @api.patch("/api/v1/settings")
    async def patch_settings(
        patch: SettingsPatch, request: Request
    ) -> dict[str, Any]:
        """
        Runtime-patch a subset of the config. Changes apply **immediately**
        but are NOT written back to ``config/config.yaml`` — they live
        until restart. This is intended for tweaking thinking mode, brain
        temperature, default brain, etc. from the GUI without editing files.
        """
        app = _app(request)
        changed: dict[str, Any] = {}

        if patch.brains:
            for name, brain_patch in patch.brains.items():
                if name not in app.config.brains:
                    raise HTTPException(404, f"unknown brain: {name}")
                brain = app.config.brains[name]
                updates = brain_patch.model_dump(exclude_none=True)
                for field, value in updates.items():
                    setattr(brain, field, value)
                if updates:
                    changed.setdefault("brains", {})[name] = updates

        if patch.routing is not None:
            routing_updates = patch.routing.model_dump(exclude_none=True)
            for field, value in routing_updates.items():
                setattr(app.config.routing, field, value)
            if routing_updates:
                changed["routing"] = routing_updates
            # Rebuild the BrainRouter so the new default takes effect
            if app.agent is not None:
                from lexy_core.agent.router import BrainRouter

                app.agent._router = BrainRouter(app.config.routing)  # type: ignore[attr-defined]

        if patch.system is not None:
            sys_updates = patch.system.model_dump(exclude_none=True)
            for field, value in sys_updates.items():
                setattr(app.config.system, field, value)
            if sys_updates:
                changed["system"] = sys_updates
                if "profile" in sys_updates:
                    # Profile change requires a restart to rewire llama.cpp
                    # connections + plugin loadout. We log it loudly so the
                    # GUI can show a "restart required" toast.
                    log.warning(
                        "settings.profile_changed_restart_required",
                        new_profile=sys_updates["profile"],
                    )

        log.info("settings.patched", changed=changed)
        return {"status": "ok", "changed": changed}

    # ─── Voice (GUI-friendly endpoints) ──────────────────────────

    @api.get("/api/v1/voice/providers")
    async def voice_providers(request: Request) -> dict[str, Any]:
        app = _app(request)
        if app.voice is None:
            return {"stt": [], "tts": [], "active_stt": None, "active_tts": None}
        stt_providers = list(app.voice._stt_providers.keys())  # type: ignore[attr-defined]
        tts_providers = list(app.voice._tts_providers.keys())  # type: ignore[attr-defined]
        return {
            "stt": stt_providers,
            "tts": tts_providers,
            "active_stt": app.config.voice.stt.primary,
            "active_tts": app.config.voice.tts.primary,
            "has_stt": app.voice.has_stt,
            "has_tts": app.voice.has_tts,
        }

    @api.get("/api/v1/voice/config")
    async def get_voice_config(request: Request) -> dict[str, Any]:
        app = _app(request)
        if app.voice is None:
            return {}
        tts_provider = app.voice._select_tts()  # type: ignore[attr-defined]
        if tts_provider is None or not hasattr(tts_provider, "get_config"):
            return {}
        return tts_provider.get_config()  # type: ignore[no-any-return]

    @api.patch("/api/v1/voice/config")
    async def patch_voice_config(
        patch: VoiceConfigPatch, request: Request
    ) -> dict[str, Any]:
        app = _app(request)
        if app.voice is None:
            raise HTTPException(503, "Voice not initialised")
        tts_provider = app.voice._select_tts()  # type: ignore[attr-defined]
        if tts_provider is None or not hasattr(tts_provider, "update_config"):
            raise HTTPException(503, "TTS provider does not support live config")
        updates = patch.model_dump(exclude_none=True)
        tts_provider.update_config(updates)
        return tts_provider.get_config()  # type: ignore[no-any-return]

    # ─── Sessions (conversation history browser) ────────────────

    @api.get("/api/v1/sessions")
    async def list_sessions(
        request: Request,
        project_id: str | None = None,
        kind: str | None = None,
        all: bool = False,
    ) -> dict[str, Any]:
        """
        List all known sessions with a short preview for the GUI sidebar.

        When ``project_id`` is provided the listing is filtered to sessions
        belonging to that project. Sessions without an assigned project
        are treated as belonging to the default project (``"default"``),
        so requesting ``project_id=default`` also surfaces legacy
        unassigned sessions.

        Phase 9.12: ``?kind=chat|rp`` orthogonally filters the list by
        session kind (used by the new Chat / Rollenspiel tab split).
        Sessions without a stored kind are treated as ``"chat"`` so old
        sessions stay visible in the chat tab. ``?all=true`` bypasses
        BOTH project and kind filters.

        ``?all=true`` bypasses the project filter entirely so the user can
        see every session across every project — useful as an escape hatch
        when sessions appear "missing" because they were created under a
        different active project.

        For each session we return:
          * ``id``           – session id
          * ``messages``     – message count (non-system)
          * ``last_role``    – role of the last stored message (may be
            ``"user"`` if the session was interrupted before Lexy replied)
          * ``last_preview`` – first ~80 chars of the last message
          * ``last_user``    – first ~80 chars of the most recent user turn
          * ``title``        – short title derived from the first user
            message, or null for empty/unnamed sessions
          * ``project_id``   – project the session belongs to, or null
          * ``kind``         – ``"chat"`` or ``"rp"`` (Phase 9.12)
          * ``created_at``   – unix timestamp when the session was first
            registered (0.0 for legacy sessions migrated from v1)
          * ``updated_at``   – unix timestamp of the last mutation
          * ``in_progress``  – true when the last stored message is a
            user turn with no assistant reply yet (crash / unfinished)
        """
        app = _app(request)

        def _trim(text: str, limit: int = 80) -> str:
            text = " ".join(text.split())
            if len(text) <= limit:
                return text
            return text[: limit - 1].rstrip() + "…"

        wanted_project = project_id.strip() if isinstance(project_id, str) else None
        if wanted_project == "":
            wanted_project = None
        wanted_kind = kind.strip() if isinstance(kind, str) else None
        if wanted_kind == "":
            wanted_kind = None
        if wanted_kind is not None and wanted_kind not in ("chat", "rp"):
            raise HTTPException(
                400, f"unknown kind: {kind!r}; expected 'chat' or 'rp'"
            )
        # ``all=true`` short-circuits filtering entirely — same behaviour
        # as omitting project_id, but explicit so the frontend can ask
        # for "every session" without faking a missing parameter.
        if all:
            wanted_project = None
            wanted_kind = None

        out: list[dict[str, Any]] = []
        for session_id, meta, _count in app.session_store.sessions_with_meta():
            session_project = meta.get("project_id") or "default"
            if wanted_project is not None and session_project != wanted_project:
                continue
            session_kind = meta.get("kind") or "chat"
            if wanted_kind is not None and session_kind != wanted_kind:
                continue
            history = app.session_store.get(session_id)
            last = history[-1] if history else None
            last_user = next(
                (m for m in reversed(history) if m["role"] == "user"),
                None,
            )
            in_progress = bool(last and last["role"] == "user")

            out.append(
                {
                    "id": session_id,
                    "messages": len(history),
                    "last_role": last["role"] if last else None,
                    "last_preview": _trim(last["content"]) if last else "",
                    "last_user": _trim(last_user["content"]) if last_user else "",
                    "title": meta.get("title"),
                    "project_id": session_project,
                    "kind": session_kind,
                    "created_at": meta.get("created_at") or 0.0,
                    "updated_at": meta.get("updated_at") or 0.0,
                    "in_progress": in_progress,
                }
            )
        # Sort: most recently updated first, empty sessions last
        out.sort(
            key=lambda s: (
                s["messages"] == 0,
                -(s["updated_at"] or 0.0),
                s["id"],
            )
        )
        return {"sessions": out}

    @api.post("/api/v1/rp_sessions/register")
    async def register_rp_session(
        req: RPSessionCreateRequest, request: Request,
    ) -> dict[str, Any]:
        """Phase 13 entry point — create an RP session in one shot.

        Atomically:
          1. Registers the session with kind="rp" + optional title
             in the core session_store.
          2. Materialises the per-session container (folder +
             dedicated Chroma collection ``rp__<id>``) seeded with
             the user-defined ``tracked_stats``.

        After this returns, the session is ready to attach characters
        to and start chatting — the container guarantees a fresh,
        empty memory namespace.
        """
        app = _app(request)

        # Parse tracked_stats: accept dict or "key=val; key" string.
        if isinstance(req.tracked_stats, str):
            from plugins.character_chat.rp_session_store import (
                parse_stats_input,
            )
            stats = parse_stats_input(req.tracked_stats)
        elif isinstance(req.tracked_stats, dict):
            stats = {str(k): str(v) for k, v in req.tracked_stats.items()}
        else:
            stats = {}

        # Register the core session row.
        app.session_store.register_empty(
            session_id=req.session_id,
            project_id=req.project_id,
            title=req.title or None,
        )
        try:
            app.session_store.set_kind(req.session_id, "rp")
        except Exception:  # noqa: BLE001
            pass

        # Materialise the container.
        plugin = _character_chat_plugin(app)
        registry = getattr(plugin, "_rp_registry", None)
        if registry is None:
            raise HTTPException(
                503, "RP session registry not initialised (plugin not loaded?)",
            )
        container = await registry.get_or_create(
            req.session_id,
            title=req.title,
            scene=req.scene,
            tracked_stats=stats,
        )
        return {
            "status": "ok",
            "session_id": req.session_id,
            "kind": "rp",
            "tracked_stats": stats,
            "collection": container.collection,
        }

    @api.post("/api/v1/sessions/register")
    async def register_session(
        req: SessionRegisterRequest, request: Request
    ) -> dict[str, Any]:
        """
        Register a session slot up front. Called by the frontend when the
        user starts a new chat so the session persists even before the
        first message. Idempotent — calling twice with the same id just
        enriches optional metadata (project_id, title) if they were
        previously unset.
        """
        app = _app(request)
        created = app.session_store.register_empty(
            session_id=req.session_id,
            project_id=req.project_id,
            title=req.title,
        )
        return {
            "status": "ok",
            "created": created,
            "session_id": req.session_id,
        }

    @api.patch("/api/v1/sessions/{session_id}")
    async def patch_session(
        session_id: str,
        req: SessionPatchRequest,
        request: Request,
    ) -> dict[str, Any]:
        """
        Update session metadata — currently supports moving a session to
        a different project and overriding its title. The target project
        must exist; unknown project ids result in a 404.
        """
        app = _app(request)
        if app.session_store.get_meta(session_id) == {}:
            raise HTTPException(404, f"unknown session: {session_id}")

        changes: dict[str, Any] = {}
        prev_project_id: str | None = None
        if req.project_id is not None:
            if not app.project_store.exists(req.project_id):
                raise HTTPException(
                    404, f"unknown project: {req.project_id}"
                )
            prev_meta = app.session_store.get_meta(session_id) or {}
            prev_project_id = prev_meta.get("project_id")
            app.session_store.set_project(session_id, req.project_id)
            changes["project_id"] = req.project_id
        if req.title is not None:
            if app.session_store.set_title(session_id, req.title):
                changes["title"] = req.title
                # Phase 13: keep the RP container's session.json in
                # sync with the SessionStore title so the right name
                # shows up in the prompt-preview / sidebar regardless
                # of which surface the user edits.
                try:
                    plugin = _character_chat_plugin(app)
                    registry = getattr(plugin, "_rp_registry", None)
                    if registry is not None and registry.is_rp_session(
                        session_id,
                    ):
                        container = await registry.get(session_id)
                        if container is not None:
                            await container.update_meta(title=req.title)
                except Exception:  # noqa: BLE001
                    pass
        if req.kind is not None:
            try:
                if app.session_store.set_kind(session_id, req.kind):
                    changes["kind"] = req.kind
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

        if changes and app.ws_server is not None:
            try:
                await app.ws_server.broadcast(
                    {
                        "type": "session_updated",
                        "session_id": session_id,
                        **changes,
                    }
                )
            except Exception:  # noqa: BLE001
                pass

        # Emit an internal event when a session moves to a different project.
        # Plugins with per-session state (character_chat, etc.) listen so
        # they can react — e.g. disabling character_mode in the moved
        # session because characters are memory-scoped per project and a
        # cross-project move would desynchronise.
        if (
            "project_id" in changes
            and prev_project_id != changes["project_id"]
            and app.event_bus is not None
        ):
            try:
                await app.event_bus.emit(
                    "core.session_project_changed",
                    {
                        "session_id": session_id,
                        "from_project": prev_project_id,
                        "to_project": changes["project_id"],
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        log.info(
            "session.patched",
            session_id=session_id,
            changes=list(changes.keys()),
        )
        return {"status": "ok", "session_id": session_id, "changes": changes}

    @api.get("/api/v1/sessions/{session_id}/history")
    async def session_history(session_id: str, request: Request) -> dict[str, Any]:
        app = _app(request)
        history = app.session_store.get(session_id)
        # Return the session's metadata alongside the messages so the
        # frontend can sync ``activeProjectId`` on resume — otherwise a
        # session resumed from another project disappears from the list
        # the next time the user views it.
        meta = app.session_store.get_meta(session_id) or {}
        return {
            "session_id": session_id,
            "messages": history,
            "project_id": meta.get("project_id"),
            "title": meta.get("title"),
            # Phase 9.12: surfaces the session kind so the frontend can
            # snap the UI to the right tab when a session is resumed.
            "kind": meta.get("kind") or "chat",
            "created_at": meta.get("created_at") or 0.0,
            "updated_at": meta.get("updated_at") or 0.0,
        }

    @api.delete("/api/v1/sessions/{session_id}")
    async def clear_session(session_id: str, request: Request) -> dict[str, Any]:
        app = _app(request)
        # Phase 13: if this is an RP session, also destroy the
        # per-session container (folder + Chroma collection). This
        # is what makes "delete session = nothing left" actually true
        # — Mike's whole reason for the per-session-folder design.
        rp_destroyed = False
        try:
            plugin = _character_chat_plugin(app) if hasattr(
                app, "plugin_loader",
            ) else None
        except Exception:  # noqa: BLE001
            plugin = None
        if plugin is not None:
            registry = getattr(plugin, "_rp_registry", None)
            if registry is not None and registry.is_rp_session(session_id):
                try:
                    rp_destroyed = await registry.destroy(session_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "session.rp_container_destroy_failed",
                        session_id=session_id,
                        error=str(exc),
                    )
        dropped = app.session_store.clear(session_id)
        return {
            "status": "cleared",
            "dropped": dropped,
            "rp_container_destroyed": rp_destroyed,
        }

    @api.patch("/api/v1/sessions/{session_id}/messages/{index}")
    async def edit_session_message(
        session_id: str,
        index: int,
        req: MessageEditRequest,
        request: Request,
    ) -> dict[str, Any]:
        """
        Edit a message (user OR assistant) in a session's history.
        Updates both the SessionStore and — if this is an assistant
        message that was auto-memorized — the matching ``context``
        collection entry gets updated too.
        """
        app = _app(request)
        updated = app.session_store.replace_at(session_id, index, req.content)
        if updated is None:
            raise HTTPException(404, f"no message at index {index}")
        log.info(
            "session.message_edited",
            session_id=session_id,
            index=index,
            role=updated["role"],
            new_length=len(req.content),
        )
        return {"status": "ok", "message": updated}

    @api.delete("/api/v1/sessions/{session_id}/messages/{index}")
    async def delete_session_message(
        session_id: str,
        index: int,
        request: Request,
    ) -> dict[str, Any]:
        app = _app(request)
        dropped = app.session_store.delete_at(session_id, index)
        if dropped is None:
            raise HTTPException(404, f"no message at index {index}")
        log.info(
            "session.message_deleted",
            session_id=session_id,
            index=index,
            role=dropped["role"],
        )
        return {"status": "ok", "dropped": dropped}

    @api.post("/api/v1/sessions/{session_id}/regenerate")
    async def regenerate_last(
        session_id: str, request: Request
    ) -> dict[str, Any]:
        """
        Non-streaming regenerate: drops the last assistant reply from
        session history + context memory and re-runs the last user
        message through LexyAgent.process(). Returns the new reply.

        The GUI uses the streaming WebSocket ``regenerate`` handler
        instead; this HTTP path exists for scripts and tests.
        """
        app = _app(request)
        if app.agent is None:
            raise HTTPException(503, "Agent not initialised")

        user_msg, _assistant_msg = app.session_store.pop_last_pair(session_id)
        if user_msg is None:
            raise HTTPException(404, "no user turn to regenerate")

        if app.memory is not None:
            try:
                await app.memory.delete_last_for_session(session_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "regenerate.memory_cleanup_failed",
                    session_id=session_id,
                    error=str(exc),
                )

        result = await app.agent.process(
            text=user_msg["content"],
            session_id=session_id,
            user_id="default",
            brain="auto",
        )
        return {
            "status": "ok",
            "text": result.get("text", ""),
            "tools_used": result.get("tools_used", []),
            "brain": result.get("brain", "e4b"),
        }

    # ─── Projects (sidebar workspace partitions) ─────────────────

    @api.get("/api/v1/projects")
    async def list_projects(
        request: Request,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """
        Return every known project. ``include_archived=true`` also
        surfaces hidden projects (useful for a "manage" view).
        """
        app = _app(request)
        projects = app.project_store.list(include_archived=include_archived)
        return {"projects": [p.to_dict() for p in projects]}

    @api.get("/api/v1/projects/{project_id}")
    async def get_project(project_id: str, request: Request) -> dict[str, Any]:
        app = _app(request)
        project = app.project_store.get(project_id)
        if project is None:
            raise HTTPException(404, f"unknown project: {project_id}")
        return {"project": project.to_dict()}

    @api.post("/api/v1/projects")
    async def create_project(
        req: ProjectCreateRequest, request: Request
    ) -> dict[str, Any]:
        app = _app(request)
        try:
            project = app.project_store.create(
                name=req.name,
                description=req.description,
                color=req.color,
                icon=req.icon,
                persona_override=req.persona_override,
                memory_scoped=req.memory_scoped,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if app.ws_server is not None:
            try:
                await app.ws_server.broadcast(
                    {"type": "project_created", "project": project.to_dict()}
                )
            except Exception:  # noqa: BLE001
                pass
        return {"project": project.to_dict()}

    @api.patch("/api/v1/projects/{project_id}")
    async def update_project(
        project_id: str,
        req: ProjectUpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        app = _app(request)
        patch = req.model_dump(exclude_none=True)
        project = app.project_store.update(project_id, **patch)
        if project is None:
            raise HTTPException(404, f"unknown project: {project_id}")
        if app.ws_server is not None:
            try:
                await app.ws_server.broadcast(
                    {"type": "project_updated", "project": project.to_dict()}
                )
            except Exception:  # noqa: BLE001
                pass
        return {"project": project.to_dict()}

    @api.delete("/api/v1/projects/{project_id}")
    async def delete_project(
        project_id: str, request: Request
    ) -> dict[str, Any]:
        """
        Delete a project. Its sessions are not deleted — they are
        migrated to the default project so no chat history is lost.
        The default project itself cannot be deleted (HTTP 400).
        """
        app = _app(request)
        if not app.project_store.exists(project_id):
            raise HTTPException(404, f"unknown project: {project_id}")

        # Migrate sessions BEFORE dropping the project so the reassign
        # references a project_id that still exists.
        migrated = 0
        for session_id, meta, _count in app.session_store.sessions_with_meta():
            if meta.get("project_id") == project_id:
                app.session_store.set_project(session_id, "default")
                migrated += 1

        deleted, _snapshot = app.project_store.delete(project_id)
        if not deleted:
            # Most likely the default project — can't be removed.
            raise HTTPException(400, "cannot delete the default project")

        if app.ws_server is not None:
            try:
                await app.ws_server.broadcast(
                    {
                        "type": "project_deleted",
                        "id": project_id,
                        "migrated_sessions": migrated,
                    }
                )
            except Exception:  # noqa: BLE001
                pass

        log.info(
            "project.deleted",
            project_id=project_id,
            migrated_sessions=migrated,
        )
        return {
            "status": "deleted",
            "id": project_id,
            "migrated_sessions": migrated,
        }

    @api.post("/api/v1/projects/{project_id}/archive")
    async def archive_project(
        project_id: str, request: Request
    ) -> dict[str, Any]:
        app = _app(request)
        if not app.project_store.exists(project_id):
            raise HTTPException(404, f"unknown project: {project_id}")
        changed = app.project_store.archive(project_id)
        if changed and app.ws_server is not None:
            project = app.project_store.get(project_id)
            try:
                await app.ws_server.broadcast(
                    {
                        "type": "project_updated",
                        "project": project.to_dict() if project else None,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        return {"status": "archived" if changed else "unchanged", "id": project_id}

    @api.post("/api/v1/projects/{project_id}/unarchive")
    async def unarchive_project(
        project_id: str, request: Request
    ) -> dict[str, Any]:
        app = _app(request)
        if not app.project_store.exists(project_id):
            raise HTTPException(404, f"unknown project: {project_id}")
        changed = app.project_store.unarchive(project_id)
        if changed and app.ws_server is not None:
            project = app.project_store.get(project_id)
            try:
                await app.ws_server.broadcast(
                    {
                        "type": "project_updated",
                        "project": project.to_dict() if project else None,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        return {"status": "unarchived" if changed else "unchanged", "id": project_id}

    # ─── Persona (Lexy's personality + system prompt) ────────────

    @api.get("/api/v1/persona")
    async def get_persona(request: Request) -> dict[str, Any]:
        app = _app(request)
        data = app.persona.model_dump()
        # Include the fully assembled prompt so the GUI can show a
        # read-only preview of what the LLM actually receives.
        data["system_prompt"] = app.persona.assemble()
        return data

    @api.patch("/api/v1/persona")
    async def patch_persona(
        patch: PersonaPatch, request: Request
    ) -> dict[str, Any]:
        """
        Live-patch Lexy's persona. Changes apply **immediately** to the
        next chat turn AND are written back to ``config/persona.yaml``
        so they survive restart.

        Supports the **sectioned** persona model: ``sections.identity``,
        ``sections.style``, ``sections.rules`` are the user-editable
        parts. Protected sections (context, capabilities) live in code
        and cannot be patched.
        """
        app = _app(request)
        updates = patch.model_dump(exclude_none=True)
        if not updates:
            return {"status": "no-op", "persona": app.persona.model_dump()}

        # Deep-merge sections
        current = app.persona.model_dump()
        if "sections" in updates and isinstance(updates["sections"], dict):
            cur_sections = current.get("sections", {})
            for key, value in updates["sections"].items():
                if value is not None:
                    cur_sections[key] = value
            current["sections"] = cur_sections
            del updates["sections"]
        # Apply top-level scalar fields
        current.update(updates)

        from lexy_core.agent.persona import Persona, save_persona

        try:
            new_persona = Persona.model_validate(current)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"invalid persona: {exc}") from exc

        app.persona = new_persona
        save_persona(new_persona)
        log.info(
            "persona.patched",
            keys=list(patch.model_dump(exclude_none=True).keys()),
            name=new_persona.name,
            thinking=new_persona.thinking_enabled,
        )
        data = new_persona.model_dump()
        data["system_prompt"] = new_persona.assemble()
        return {
            "status": "ok",
            "persona": data,
        }

    @api.post("/api/v1/persona/reset")
    async def reset_persona_route(request: Request) -> dict[str, Any]:
        """Reset the persona back to the built-in default."""
        app = _app(request)
        from lexy_core.agent.persona import reset_persona

        new_persona = reset_persona()
        app.persona = new_persona
        log.info("persona.reset_via_api")
        return {
            "status": "ok",
            "persona": new_persona.model_dump(),
        }

    # ─── Memory wipe (nuclear option) ────────────────────────────

    @api.delete("/api/v1/memory/collection/{name}")
    async def memory_wipe_collection(
        name: str, request: Request
    ) -> dict[str, Any]:
        """Wipe a single ChromaDB collection + its FTS rows."""
        app = _app(request)
        if app.memory is None:
            raise HTTPException(503, "Memory not initialised")
        try:
            result = await app.memory.wipe_collection(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"wipe failed: {exc}") from exc
        return {"status": "wiped", "collection": name, **result}

    @api.post("/api/v1/memory/wipe")
    async def memory_wipe(
        req: MemoryWipeRequest, request: Request
    ) -> dict[str, Any]:
        """
        Nuke Lexy's memory across all layers. **Destructive** — requires
        ``confirm: true`` in the request body or 400 is returned.

        Depending on flags, this clears:

        * ``collections`` – every ChromaDB collection + FTS5 mirror
        * ``sessions``    – in-memory SessionStore (conversation history)
        * ``plugin_data`` – every file under ``data/plugins/*`` (scheduler
          timers, thinking thoughts, plugin-owned SQLite DBs). Lexy must
          be restarted afterwards because plugins hold open handles.
        """
        if not req.confirm:
            raise HTTPException(
                400,
                "Destructive operation. Set confirm: true in the request body.",
            )

        app = _app(request)
        result: dict[str, Any] = {"status": "wiped"}

        # 1. ChromaDB collections + FTS5 mirror
        if req.collections:
            if app.memory is None:
                result["collections"] = {"error": "memory not initialised"}
            else:
                try:
                    result["collections"] = await app.memory.wipe_all()
                except Exception as exc:  # noqa: BLE001
                    log.error("memory.wipe_collections_failed", error=str(exc))
                    result["collections"] = {"error": str(exc)}

        # 2. SessionStore — kill all conversation history
        if req.sessions:
            before = len(app.session_store.sessions())
            app.session_store.reset_all()
            result["sessions"] = {
                "dropped_sessions": before,
                "status": "cleared",
            }

        # 3. Plugin-owned data (opt-in, requires restart)
        if req.plugin_data:
            import shutil

            plugins_dir = Path("data/plugins")
            dropped: list[str] = []
            errors: list[str] = []
            if plugins_dir.exists():
                for child in plugins_dir.iterdir():
                    try:
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                        dropped.append(child.name)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{child.name}: {exc}")
            result["plugin_data"] = {
                "dropped": dropped,
                "errors": errors,
                "restart_required": True,
            }

        # Emit an event so other plugins know memory just got wiped.
        await app.event_bus.emit("core.memory_wiped", result)
        log.warning(
            "memory.wipe_complete",
            collections=req.collections,
            sessions=req.sessions,
            plugin_data=req.plugin_data,
        )
        return result

    # ─── Memory delete (for GUI browser) ─────────────────────────

    @api.post("/api/v1/memory/delete")
    async def memory_delete(
        req: MemoryDeleteRequest, request: Request
    ) -> dict[str, str]:
        app = _app(request)
        if app.memory is None:
            raise HTTPException(503, "Memory not initialised")
        col = app.memory._require_collection(req.collection)  # type: ignore[attr-defined]
        try:
            col.delete(ids=[req.id])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"delete failed: {exc}") from exc
        # Also drop from FTS mirror
        if app.memory._fts is not None:  # type: ignore[attr-defined]
            await app.memory._fts.execute(  # type: ignore[attr-defined]
                "DELETE FROM items_fts WHERE id = ?", (req.id,)
            )
            await app.memory._fts.commit()  # type: ignore[attr-defined]
        return {"status": "deleted"}

    # ─── WebSocket ───────────────────────────────────────────────

    @api.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        if lexy.ws_server is None:
            await websocket.close(code=1011)
            return
        await lexy.ws_server.handle(websocket)

    # ─── Static frontend ─────────────────────────────────────────

    # Serve plugin-owned character avatars at /avatars — mounted whether or
    # not the directory exists yet; the upload endpoint creates it on first
    # use. FastAPI will 404 gracefully for missing files.
    avatars_dir = Path("data/plugins/character_chat/avatars")
    avatars_dir.mkdir(parents=True, exist_ok=True)
    api.mount(
        "/avatars",
        StaticFiles(directory=str(avatars_dir)),
        name="character_avatars",
    )

    # Chat-attachment uploads — same pattern, served under /uploads/<id>.<ext>.
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    api.mount(
        "/uploads",
        StaticFiles(directory=str(uploads_dir)),
        name="chat_uploads",
    )

    # Serve frontend/static at /static and redirect / → /static/index.html
    static_dir = Path("frontend/static")
    if static_dir.is_dir():
        # Avatar assets (GLB/glTF + textures + backgrounds + animations)
        # live OUTSIDE the repo on Mike's machine and are exposed via
        # Windows junctions under frontend/static/avatar-world/assets/.
        # Starlette's StaticFiles refuses to serve files whose realpath
        # lives outside the configured directory (it does a `realpath`
        # + `commonpath` containment check), so a junction-pointed file
        # would 404 even though it exists.
        #
        # Workaround: mount each potentially-junctioned asset folder
        # directly at its own URL prefix, BEFORE the broader /static
        # mount. Starlette matches routes first-hit-wins, so the more
        # specific mount catches the request and serves the resolved
        # realpath. When the folder is a real directory (no junction),
        # realpath equals the original path and this mount is a no-op
        # duplicate of the /static route — also fine.
        avatar_asset_dirs = (
            "frontend/static/avatar-world/assets/models",
            "frontend/static/avatar-world/assets/backgrounds",
            "frontend/static/avatar-world/assets/animations",
            "frontend/static/avatar-world/assets/apartment",
        )
        for rel in avatar_asset_dirs:
            asset_path = Path(rel)
            if not asset_path.exists():
                continue
            real_path = asset_path.resolve()
            url_prefix = "/" + rel.replace("\\", "/")
            api.mount(
                url_prefix,
                StaticFiles(directory=str(real_path)),
                name="avatar_assets_" + asset_path.name,
            )
            log.info(
                "static.avatar_assets_mounted",
                url=url_prefix,
                real_path=str(real_path),
            )

        api.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        index_file = static_dir / "index.html"

        @api.get("/", include_in_schema=False)
        async def root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/static/index.html")

        @api.get("/static/index.html", include_in_schema=False)
        async def serve_index_no_cache() -> FileResponse:
            """Serve index.html with no-cache headers so a Lexy update
            always reaches the browser even when the user never
            hard-reloads. The HTML carries ``?v=<date>`` query strings
            on its asset references — those are cacheable; only the
            HTML itself needs to bypass the cache so the new version
            strings reach the client.
            """
            return FileResponse(
                str(index_file),
                media_type="text/html",
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )

        @api.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> Response:
            # Return the logo text if present, otherwise empty 204
            icon = static_dir / "favicon.ico"
            if icon.exists():
                return FileResponse(str(icon))
            return Response(status_code=204)

        log.info("gateway.frontend_mounted", dir=str(static_dir.resolve()))
    else:
        log.warning("gateway.frontend_missing", dir=str(static_dir.resolve()))

    return api
