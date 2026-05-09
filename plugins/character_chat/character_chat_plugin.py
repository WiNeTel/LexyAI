"""
character_chat — persistent RP characters with sequential group dynamics.

Phase 1+2 scope (this file):

* Character-card CRUD (spawn / update / archive / delete / import) exposed as
  both LLM tools and WebSocket handlers.
* A ``run_character_round`` tool/WS that runs one turn through
  :class:`GroupTurnOrchestrator` with sequential prompting — each character's
  prompt includes the previous speakers' turns in the same round so real
  group dynamics emerge (Decision #1 from the design review).
* Each character turn is persisted in a local ``character_turns`` table
  (full history for the UI) and mirrored into memory with a
  ``character_id`` metadata tag (strict isolation — Decision #4).
* WS broadcasts per turn as ``character_turn`` so the frontend can render
  avatars/badges inline.

Phase 3+ (not in this file):

* Auto-intercept of ``core.user_message`` for sessions that have characters
  attached — currently the round is only triggered on explicit tool/WS call.
* Per-character semantic recall when building the turn prompt.
* Scheduler-driven proactive pulses (needs scheduler plugin wiring).
* Frontend dashboard widget + session-sidebar character list.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from pydantic import ValidationError

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .character_card import AGE_STAGES, CharacterCard, CharacterCardError
from .character_store import CharacterStore
from .group_turn import (
    CharacterTurn,
    GroupTurnOrchestrator,
    GroupTurnRequest,
)
from .lorebook_engine import LorebookEngine
from .lorebook_store import (
    LorebookStore,
    SCOPE_CHARACTER,
    SCOPE_GLOBAL,
    SCOPE_SESSION,
    VALID_POSITIONS,
    VALID_SCOPES,
)
from .mention_parser import parse_nl_mentions
from .pulse_generator import PulseGenerator
from .rp_session_registry import RPSessionRegistry
from .rp_session_store import (
    MemoryBackend,
    RPSessionContainer,
    TurnRow,
    parse_stats_input,
)
from .state_updater import merge_state, parse_state_block


log = get_logger(module="character_chat")


# ─── Memory backend adapter (Phase 13) ───────────────────────────────


class _PluginAPIMemoryBackend:
    """Adapt :class:`PluginAPI` to the :class:`MemoryBackend` protocol.

    The RP session container only knows about ``MemoryBackend`` (so
    tests can fake it). The plugin wires this adapter so containers
    can store/recall through the standard PluginAPI without seeing
    the Manager directly.
    """

    def __init__(self, api: Any) -> None:
        self._api = api

    async def ensure_collection(self, name: str) -> None:
        await self._api.memory_ensure_collection(name)

    async def delete_collection(self, name: str) -> None:
        await self._api.memory_delete_collection(name)

    async def store(
        self,
        text: str,
        collection: str = "facts",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self._api.memory_store(
            text=text, collection=collection, metadata=metadata,
        )

    async def recall(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 5,
        project_id: str | None = None,
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._api.memory_recall(
            query=query,
            collection=collection,
            limit=limit,
            project_id=project_id,
            metadata_equals=metadata_equals,
        )


# ─── Tool schemas ─────────────────────────────────────────────────────────────

SPAWN_CHARACTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Name (z.B. 'Luna')"},
        "persona": {
            "type": "string",
            "description": "Persönlichkeit, Aussehen, Eigenheiten.",
        },
        "greeting": {"type": "string", "description": "Erste Nachricht."},
        "scenario": {"type": "string", "description": "Szene/Setting."},
        "example_dialog": {
            "type": "string",
            "description": "Beispiel-Dialog (optional, few-shot).",
        },
        "age_stage": {
            "type": "string",
            "enum": list(AGE_STAGES),
            "description": "Alter: baby/toddler/child/teen/adult.",
        },
        "color": {"type": "string", "description": "Hex-Farbe z.B. '#ff77cc'."},
        "voice": {
            "type": "string",
            "description": (
                "Optionaler TTS-Voice-Name (z.B. CosyVoice-Speaker-ID "
                "wie 'luna'). Leer = Default-Voice."
            ),
        },
        "relationships": {
            "type": "object",
            "description": (
                "Beziehungen zu anderen Charakteren: {other_id: 'Mutter', ...}"
            ),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tags für UI-Filter.",
        },
        "proactive_pulse_pattern": {
            "type": "string",
            "description": "Scheduler-Pattern (z.B. 'every 3h') für Pulses. Leer bei baby/toddler/child = Auto-Default.",
        },
        "session_id": {
            "type": "string",
            "description": (
                "Optional: Session-ID zum automatischen Attach + Pulse-Registrierung. "
                "Wenn gesetzt, wird der Charakter sofort in die Session eingebunden."
            ),
        },
    },
    "required": ["name"],
}

UPDATE_CHARACTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Character-ID."},
        "name": {"type": "string"},
        "persona": {"type": "string"},
        "greeting": {"type": "string"},
        "scenario": {"type": "string"},
        "example_dialog": {"type": "string"},
        "avatar": {"type": "string"},
        "color": {"type": "string"},
        "age_stage": {"type": "string", "enum": list(AGE_STAGES)},
        "voice": {"type": "string"},
        "relationships": {"type": "object"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "proactive_pulse_pattern": {"type": "string"},
        "proactive_pulse_prompt": {"type": "string"},
    },
    "required": ["id"],
}

ARCHIVE_CHARACTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"id": {"type": "string"}},
    "required": ["id"],
}

LIST_CHARACTERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_archived": {
            "type": "boolean",
            "description": "Archivierte mitliefern? (default false)",
        },
        "session_id": {
            "type": "string",
            "description": "Nur Charaktere in dieser Session.",
        },
    },
    "required": [],
}

ATTACH_CHARACTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "session_id": {"type": "string"},
    },
    "required": ["id", "session_id"],
}

IMPORT_CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "payload": {
            "type": "object",
            "description": (
                "Silly-Tavern JSON (v1 oder v2 'data'-nested). Genau "
                "EINE von ``payload`` oder ``png_b64`` muss gesetzt sein."
            ),
        },
        "png_b64": {
            "type": "string",
            "description": (
                "Base64-encoded PNG card mit eingebettetem "
                "'chara'-tEXt-Chunk (Silly-Tavern Standard). Das Bild "
                "selbst wird als Avatar des Charakters gespeichert."
            ),
        },
        "filename": {"type": "string"},
        "content_type": {"type": "string"},
        "color": {"type": "string"},
        "age_stage": {"type": "string", "enum": list(AGE_STAGES)},
    },
    "required": [],
}

RUN_ROUND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "user_message": {
            "type": "string",
            "description": "Optional — leer = proaktive Runde.",
        },
        "pulse_from_id": {
            "type": "string",
            "description": "Character-ID die den Pulse ausgelöst hat.",
        },
        "pulse_text": {
            "type": "string",
            "description": "Pulse-Text (z.B. '*schreit laut*').",
        },
        "scene": {"type": "string", "description": "Optionale Szenenbeschreibung."},
    },
    "required": ["session_id"],
}


# ─── Plugin ───────────────────────────────────────────────────────────────────


class CharacterChatPlugin(BasePlugin):
    """Silly-Tavern-lite for Lexy: persistent RP characters with group turns."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        # Config (populated in on_load)
        self._default_brain: str = "e4b"
        # Phase 13.2: 320 was too tight for German narrative RP — Mike
        # saw mid-sentence cut-offs and ``<state>`` blocks leaking into
        # the visible chat because the cut-off snipped the closing tag.
        # 4096 lets even verbose narrators finish their full action.
        self._max_tokens: int = 4096
        self._temperature: float = 0.8
        # Phase 13.2: 4 speakers per round produced an "ich auch / ich
        # auch / ich auch"-loop where every char repeated the same
        # sand-staring beat. 2 keeps the dynamic alive over time —
        # different chars react to different pulses.
        self._max_speakers: int = 2
        self._turn_selection: str = "autonomous"
        # Brain used for the cheap speaker-selection classifier. Independent
        # of ``default_brain`` so the big A4B brain stays free to do actual
        # in-character turns. Default e4b (Gemma 4 12B on :5006).
        self._speaker_selection_brain: str = "e4b"
        # Global RP style prompt — Mike's request "alle charaktere sollen
        # im gleichen Style schreiben". Injected MUST-priority into every
        # character turn's system prompt. Empty string disables.
        self._global_rp_style_prompt: str = ""
        # Force the orchestrator brain to confirm/refine speaker order
        # even when @-mentions or NL-mentions already cover everyone.
        # Off by default — adds one E4B call per round.
        self._always_call_orchestrator: bool = False
        # Smart pulse generation — replaces the static _DEFAULT_PULSES with
        # a tiny LLM call (E4B by default) that reads persona + state +
        # recent history. When False, the old static behaviour is exact.
        self._smart_pulses_enabled: bool = True
        self._pulse_generation_brain: str = "e4b"
        self._pulse_history_window: int = 6
        self._pulse_max_tokens: int = 600  # was 200 — Phase 13.2
        self._proactive_pulses_enabled: bool = True
        self._lexy_auto_reacts: bool = True
        self._memory_strict_isolation: bool = True
        # Context window knobs. 0 = read live from brain config.
        self._context_size_override: int = 0
        self._context_safety_margin: int = 256
        # Pulse debounce: when 4 babies each have their own timer, they
        # cascade (Sandra fires → others react → Sophie fires → others
        # react → ...). This cooldown ensures only ONE pulse-round per
        # session within a window. Key = session_id, value = timestamp
        # of the last pulse-round.
        self._pulse_cooldowns: dict[str, float] = {}
        self._pulse_cooldown_seconds: float = 600.0  # 10 min default
        # Sessions older than this in seconds skip pulse + sim ticks.
        # 0 = disabled (= pre-9.7 behaviour). Set in _apply_config.
        self._pulse_session_stale_seconds: float = 21600.0  # 6h default
        # Guard: ensure _rehydrate_pulse_timers runs exactly once.
        self._pulse_rehydrated: bool = False
        # Autonomous simulation: per-session recurring timer id + state.
        # Key = session_id, value = scheduler timer id. When the timer
        # fires, _run_autonomous_tick picks ONE speaker to react.
        self._simulation_timers: dict[str, str] = {}
        self._simulation_default_interval: int = 3  # minutes
        self._lexy_turn_probability: float = 0.3    # 0.0-1.0

        # Components (created in on_load)
        self._store: CharacterStore | None = None
        self._orchestrator: GroupTurnOrchestrator | None = None
        self._pulse_generator: PulseGenerator | None = None
        self._lore_store: LorebookStore | None = None
        self._lore_engine: LorebookEngine = LorebookEngine()
        # Phase 13: per-RP-session container registry. Created lazily
        # in ``on_load`` once the data dir is known. RP sessions own
        # their own folder + Chroma collection; recall/state writes
        # for those sessions go through the registry instead of the
        # global character_chat tables.
        self._rp_registry: RPSessionRegistry | None = None
        # Phase 13.2: skip-cooldown table. When a character returns an
        # empty / pass turn, we mark them with a small cooldown so the
        # next round's speaker selection avoids them — otherwise the
        # LLM-orchestrator keeps picking the same silent character and
        # produces "*Yara schweigt*" five rounds in a row.
        # Keyed: ``{session_id: {character_id: rounds_remaining}}``.
        self._skip_cooldowns: dict[str, dict[str, int]] = {}
        # Serialise round execution per session so two concurrent
        # run_round requests don't interleave their broadcasts.
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Scheduler-timer id per (character_id, session_id) so we can
        # update/cancel a character's pulse when the card is edited
        # or detached.
        self._pulse_timers: dict[tuple[str, str], str] = {}

    # ─── Lifecycle ───────────────────────────────────────────────────────

    async def on_load(self) -> None:
        self._apply_config(self.api.get_config())

        # SQLite schema
        db = await self.api.get_db()
        self._store = CharacterStore(db)
        await self._store.init_schema()
        # Lorebook store shares the character_chat DB — same connection,
        # separate tables. Init unconditionally (the table CREATE IF NOT
        # EXISTS is idempotent and cheap).
        self._lore_store = LorebookStore(db)
        await self._lore_store.init_schema()
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS character_turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                round_id TEXT NOT NULL,
                order_num INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                skipped INTEGER NOT NULL DEFAULT 0,
                trigger_kind TEXT NOT NULL DEFAULT 'user',
                trigger_text TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
            """
        )
        # Per-session opt-in: when ``character_mode = 1``, the
        # ``before_user_input`` hook intercepts the user message and runs
        # a character round instead of the normal agent flow.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS character_sessions (
                session_id TEXT PRIMARY KEY,
                character_mode INTEGER NOT NULL DEFAULT 0,
                scene TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_session ON "
            "character_turns(session_id, created_at)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_character ON "
            "character_turns(character_id, created_at)"
        )
        await db.commit()

        # Phase 13: per-RP-session container registry. Sessions live
        # in ``data/rp_sessions/<session_id>/`` (NOT under plugin-data,
        # because conceptually they're session-scoped artefacts that
        # the gateway also reads). Memory backend is the PluginAPI
        # adapter so containers route through standard memory ops.
        from pathlib import Path
        rp_root = Path("data/rp_sessions")
        rp_root.mkdir(parents=True, exist_ok=True)
        self._rp_registry = RPSessionRegistry(
            rp_root, _PluginAPIMemoryBackend(self.api),
        )
        # Phase 13 wipe-once: on first load after the upgrade, clear
        # all legacy RP data (character_turns / character_sessions /
        # source=character_chat memories) so Mike sees a guaranteed
        # empty state. Marker file prevents re-wipe on every restart.
        await self._maybe_wipe_legacy_rp_data(db)

        # Orchestrator (with per-character memory-recall wired in)
        self._rebuild_orchestrator()

        log.info("character_chat.loaded", brain=self._default_brain)

    def _rebuild_orchestrator(self) -> None:
        """(Re)create the orchestrator — called on load and on config change.

        The orchestrator receives a ``context_size_fn`` callback instead of
        a hardcoded value. Every turn queries this callback live, so a
        ``routing.yaml`` edit or a default-brain switch is picked up on
        the next turn without any plugin reload.
        """
        override = self._context_size_override

        def _ctx_fn() -> int:
            # Override wins if set — useful for testing or for forcing a
            # tighter budget than the brain reports.
            if override > 0:
                return override
            try:
                return self.api.get_brain_context_size(self._default_brain)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.ctx_size_fallback brain=%s error=%s",
                    self._default_brain,
                    exc,
                )
                return 16384

        self._orchestrator = GroupTurnOrchestrator(
            llm_chat=self.api.llm_chat,
            brain=self._default_brain,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            max_speakers_per_round=self._max_speakers,
            turn_selection=self._turn_selection,
            recall_fn=self._character_recall if self._memory_strict_isolation else None,
            recall_limit=3,
            context_size_fn=_ctx_fn,
            safety_margin_tokens=self._context_safety_margin,
            speaker_selection_brain=self._speaker_selection_brain,
            global_style_prompt=self._global_rp_style_prompt,
            always_call_orchestrator=self._always_call_orchestrator,
        )
        self._pulse_generator = PulseGenerator(
            llm_chat=self.api.llm_chat,
            brain=self._pulse_generation_brain,
            max_tokens=self._pulse_max_tokens,
            history_window=self._pulse_history_window,
        )

    async def _character_recall(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 3,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Per-character memory fetch with strict isolation.

        Phase 13: when ``session_id`` belongs to an RP session, recall
        is scoped to that session's dedicated Chroma collection — so
        Sandra in Session B sees nothing from Session A even if she's
        the same character. For non-RP sessions (or missing session_id)
        the legacy global ``context`` collection is used.
        """
        try:
            if session_id and self._rp_registry is not None:
                if self._rp_registry.is_rp_session(session_id):
                    container = await self._rp_registry.get(session_id)
                    if container is not None:
                        return await container.memory_recall(
                            query=query,
                            character_id=character_id,
                            limit=limit,
                        )
            # Legacy fallback: pre-Phase-13 global character recall.
            return await self.api.memory_recall(
                query=query,
                collection="context",
                limit=limit,
                metadata_equals={"character_id": character_id},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.recall_unavailable",
                character_id=character_id,
                session_id=session_id,
                error=str(exc),
            )
            return []

    async def _maybe_wipe_legacy_rp_data(
        self, db: Any,
    ) -> None:
        """Phase 13 wipe-once: drop legacy RP data on first load.

        Mike's Phase 13 ask: *"alle Memorys und sessions sollten
        gelöscht sein"*. We honour that by clearing every legacy
        artefact that pre-dates the RP container architecture.

        Marker file ``data/.phase13_wiped`` records the version that
        last ran; if the file shows an OLDER version (or doesn't
        exist), the wipe runs again. Phase 13.1 added scheduler-
        timer cleanup, so first-time installations and existing
        Phase-13 installs both end up cleaned up.
        """
        from pathlib import Path
        WIPE_VERSION = "13.1"
        marker = Path("data/.phase13_wiped")
        if marker.exists():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                last_version = str(payload.get("version") or "13.0")
            except Exception:  # noqa: BLE001
                last_version = "13.0"
            if last_version >= WIPE_VERSION:
                return
            log.warning(
                "character_chat.phase13_wipe_upgrading",
                from_version=last_version,
                to_version=WIPE_VERSION,
            )
        else:
            log.warning("character_chat.phase13_wipe_starting")
        # 1) character_turns + character_sessions: blow them away.
        deleted_turns = 0
        deleted_sess = 0
        try:
            cur = await db.execute("DELETE FROM character_turns")
            deleted_turns = cur.rowcount or 0
            cur = await db.execute("DELETE FROM character_sessions")
            deleted_sess = cur.rowcount or 0
            # Reset characters.state to '{}' so the legacy state column
            # doesn't keep injecting stale clothing/posture into prompts.
            await db.execute("UPDATE characters SET state = '{}'")
            # Detach all characters from sessions — active_sessions
            # was a global-shared list and is no longer authoritative.
            await db.execute("UPDATE characters SET active_sessions = '[]'")
            await db.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.phase13_db_wipe_failed", error=str(exc),
            )

        # 2) Drop every memory item tagged source=character_chat from
        #    the global "context" collection (and FTS mirror). We
        #    reach into Chroma directly here — no public delete-by-
        #    metadata helper exists yet, and adding one for a one-
        #    shot wipe would be over-engineering.
        deleted_mem = 0
        try:
            mem = self._app_memory()
            if mem is not None and getattr(mem, "_collections", None):
                col = mem._collections.get("context")
                if col is not None:
                    got = col.get(where={"source": "character_chat"})
                    ids = list(got.get("ids") or [])
                    if ids:
                        col.delete(ids=ids)
                        deleted_mem = len(ids)
                    # FTS mirror
                    fts = getattr(mem, "_fts", None)
                    if fts is not None and ids:
                        for chunk_start in range(0, len(ids), 500):
                            chunk = ids[chunk_start:chunk_start + 500]
                            placeholders = ",".join("?" * len(chunk))
                            await fts.execute(
                                f"DELETE FROM items_fts WHERE id IN ({placeholders})",
                                chunk,
                            )
                        await fts.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.phase13_mem_wipe_failed", error=str(exc),
            )

        # 3) Strip kind=rp sessions from the core session store.
        deleted_core_sessions = 0
        try:
            ss = self._app_session_store()
            if ss is not None and hasattr(ss, "delete_by_kind"):
                deleted_core_sessions = await ss.delete_by_kind("rp")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.phase13_session_wipe_failed",
                error=str(exc),
            )

        # 4) (NEW in 13.1) Kill every character_pulse:* and
        #    autonomous_sim:* timer in the scheduler. Mike reported
        #    that ghost timers from pre-Phase-13 sessions kept firing
        #    against now-dead session_ids — they survived the DB
        #    wipe because they live in the scheduler plugin's own
        #    table. Phase 13's clean-slate promise demands they go.
        cancelled_timers = 0
        try:
            cancelled_timers = await self._cancel_all_character_timers()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.phase13_timer_wipe_failed", error=str(exc),
            )

        marker.parent.mkdir(parents=True, exist_ok=True)
        import time as _time
        marker.write_text(
            json.dumps({
                "version": WIPE_VERSION,
                "wiped_at": _time.time(),
                "character_turns": deleted_turns,
                "character_sessions": deleted_sess,
                "memory_items": deleted_mem,
                "core_sessions": deleted_core_sessions,
                "timers_cancelled": cancelled_timers,
            }, indent=2),
            encoding="utf-8",
        )
        log.warning(
            "character_chat.phase13_wipe_complete",
            character_turns=deleted_turns,
            character_sessions=deleted_sess,
            memory_items=deleted_mem,
            core_sessions=deleted_core_sessions,
            timers_cancelled=cancelled_timers,
        )

    async def _cancel_all_character_timers(self) -> int:
        """Cancel every ``character_pulse:*`` and ``autonomous_sim:*``
        recurring timer in the scheduler.

        Used by the Phase-13 wipe to clear ghosts from pre-Phase-13
        sessions. Returns the number of timers cancelled.
        """
        try:
            listing = await self.api.call_tool("list_timers", {})
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.timer_list_failed", error=str(exc),
            )
            return 0
        cancelled = 0
        for t in (listing or {}).get("timers", []):
            label = str(t.get("label") or "")
            if not (
                label.startswith("character_pulse:")
                or label.startswith("autonomous_sim:")
            ):
                continue
            timer_id = t.get("id")
            if not timer_id:
                continue
            try:
                await self.api.call_tool(
                    "cancel_timer", {"id": timer_id},
                )
                cancelled += 1
            except Exception:  # noqa: BLE001
                pass
        # Reset our in-process bookkeeping — the timers we tracked are
        # all gone; future attaches will create fresh ones.
        self._pulse_timers.clear()
        self._simulation_timers.clear()
        return cancelled

    def _app_memory(self) -> Any:
        """Best-effort lookup of MemoryManager via the PluginAPI."""
        # PluginAPI doesn't expose memory directly; we access it via
        # the application held by the API. Wrapped in try/except since
        # the surface is private and may move in future refactors.
        try:
            return getattr(self.api, "_app").memory  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return None

    def _app_session_store(self) -> Any:
        try:
            return getattr(self.api, "_app").session_store  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return None

    def _tick_skip_cooldowns(self, session_id: str) -> set[str]:
        """Snapshot the chars on cooldown for THIS round, then tick.

        Phase 13.2: when a character is silent in round N, they're put
        on a 1-round cooldown. This call runs at the START of round
        N+1: every char in the cooldown table at that moment is
        excluded for this round, *then* their counter is decremented;
        whoever drops to ≤ 0 becomes eligible again from round N+2 on.
        That gives the contract Mike asked for: "skipped once → out
        for one round, then back in".
        """
        cooldowns = self._skip_cooldowns.get(session_id)
        if not cooldowns:
            return set()
        # Snapshot first — every char currently on cooldown is
        # excluded for this round, regardless of remaining count.
        excluded: set[str] = set(cooldowns.keys())
        # Then decrement and prune.
        for char_id in list(cooldowns.keys()):
            cooldowns[char_id] -= 1
            if cooldowns[char_id] <= 0:
                cooldowns.pop(char_id, None)
        if not cooldowns:
            self._skip_cooldowns.pop(session_id, None)
        return excluded

    def _record_skip_cooldowns(
        self, session_id: str, turns: list[Any],
    ) -> None:
        """After a round, mark every skipped character with a 1-round
        cooldown. Phase 13.2 — prevents the LLM-orchestrator from
        looping the same silent char.
        """
        if not turns:
            return
        for turn in turns:
            if not getattr(turn, "skipped", False):
                continue
            char_id = getattr(turn, "character_id", None)
            if not char_id:
                continue
            cooldowns = self._skip_cooldowns.setdefault(session_id, {})
            cooldowns[char_id] = 1

    async def _load_prior_turns_per_char(
        self,
        *,
        session_id: str,
        characters: list[CharacterCard],
        limit: int = 5,
    ) -> dict[str, list[str]]:
        """Phase 13.5 (B+D) — fetch each char's last N own turns for the
        cross-round repetition guard.

        Returns a mapping {character_id: [oldest_text, ..., newest_text]}.
        Empty values for chars without history. Best-effort: any DB
        error is logged but doesn't block the round.
        """
        out: dict[str, list[str]] = {}
        if not characters:
            return out
        try:
            container = await self._get_rp_container(session_id)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "character_chat.prior_turns_container_lookup_failed "
                "session=%s error=%s",
                session_id, str(exc),
            )
            container = None

        if container is not None:
            for c in characters:
                try:
                    rows = await container.list_turns(
                        character_id=c.id, limit=limit,
                    )
                    out[c.id] = [
                        r.content for r in rows
                        if r.content and not r.skipped
                    ]
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "character_chat.prior_turns_rp_failed "
                        "session=%s char=%s error=%s",
                        session_id, c.name, str(exc),
                    )
            return out

        # Non-RP fallback — query the legacy character_turns table.
        db = await self.api.get_db()
        for c in characters:
            try:
                cursor = await db.execute(
                    "SELECT content FROM character_turns "
                    "WHERE session_id = ? AND character_id = ? "
                    "AND skipped = 0 AND content != '' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (session_id, c.id, limit),
                )
                rows = await cursor.fetchall()
                # DB returns newest-first; reverse to oldest-first to
                # match the container's ASC order.
                out[c.id] = [str(r[0]) for r in reversed(rows)]
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "character_chat.prior_turns_legacy_failed "
                    "session=%s char=%s error=%s",
                    session_id, c.name, str(exc),
                )
        return out

    async def _is_rp_session(self, session_id: str) -> bool:
        """True if the given session is RP — checks both the registry
        (existing folder) AND the core session store's kind marker."""
        if not session_id:
            return False
        if self._rp_registry is not None and self._rp_registry.is_rp_session(
            session_id
        ):
            return True
        # Fallback: ask the session store. A session that was just
        # created with kind="rp" but doesn't have a folder yet still
        # qualifies — the next attach will materialise the folder.
        ss = self._app_session_store()
        if ss is None:
            return False
        try:
            meta = ss.get_meta(session_id) if hasattr(ss, "get_meta") else None
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(meta, dict):
            return False
        return str(meta.get("kind", "")) == "rp"

    async def _get_rp_container(
        self,
        session_id: str,
        *,
        title: str = "",
        scene: str = "",
        tracked_stats: dict[str, str] | None = None,
    ) -> RPSessionContainer | None:
        """Return the container for an RP session, creating if missing."""
        if not session_id or self._rp_registry is None:
            return None
        if not await self._is_rp_session(session_id):
            return None
        return await self._rp_registry.get_or_create(
            session_id,
            title=title,
            scene=scene,
            tracked_stats=tracked_stats,
        )

    async def on_config_changed(self, cfg: dict[str, Any]) -> None:
        self._apply_config(cfg)
        # Rebuild orchestrator with new params (cheap — it's just a dataclass-y
        # wrapper around the LLM callable).
        if self._orchestrator is not None:
            self._rebuild_orchestrator()
        log.info("character_chat.config_changed")

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        self._default_brain = str(cfg.get("default_brain", "e4b"))
        self._max_tokens = int(cfg.get("max_tokens_per_turn", 320))
        self._temperature = float(cfg.get("temperature", 0.8))
        self._max_speakers = int(cfg.get("max_speakers_per_round", 4))
        turn_sel = str(cfg.get("turn_selection", "autonomous"))
        if turn_sel not in ("autonomous", "round_robin"):
            turn_sel = "autonomous"
        self._turn_selection = turn_sel
        self._speaker_selection_brain = str(
            cfg.get("speaker_selection_brain", "e4b") or "e4b"
        )
        self._global_rp_style_prompt = str(
            cfg.get("global_rp_style_prompt") or ""
        ).strip()
        self._always_call_orchestrator = bool(
            cfg.get("always_call_orchestrator", False)
        )
        self._smart_pulses_enabled = bool(
            cfg.get("smart_pulses_enabled", True)
        )
        self._pulse_generation_brain = str(
            cfg.get("pulse_generation_brain", "e4b") or "e4b"
        )
        try:
            self._pulse_history_window = max(
                0, int(cfg.get("pulse_history_window", 6) or 6)
            )
        except (TypeError, ValueError):
            self._pulse_history_window = 6
        try:
            self._pulse_max_tokens = max(
                40, int(cfg.get("pulse_max_tokens", 200) or 200)
            )
        except (TypeError, ValueError):
            self._pulse_max_tokens = 200
        self._proactive_pulses_enabled = bool(
            cfg.get("proactive_pulses_enabled", True)
        )
        self._lexy_auto_reacts = bool(cfg.get("lexy_auto_reacts_to_pulses", True))
        self._memory_strict_isolation = bool(
            cfg.get("memory_strict_isolation", True)
        )
        # Pulse debounce: minimum seconds between two pulse-rounds per session.
        try:
            self._pulse_cooldown_seconds = float(
                cfg.get("pulse_cooldown_seconds", 600) or 600
            )
        except (TypeError, ValueError):
            self._pulse_cooldown_seconds = 600.0
        # Pulse staleness: skip pulse + sim if the session has been idle
        # for longer than this. 0 disables the check entirely.
        try:
            self._pulse_session_stale_seconds = max(
                0.0, float(cfg.get("pulse_session_stale_seconds", 21600) or 0)
            )
        except (TypeError, ValueError):
            self._pulse_session_stale_seconds = 21600.0

        # Autonomous simulation defaults.
        try:
            self._simulation_default_interval = max(
                1, min(15, int(cfg.get("simulation_default_interval_minutes", 3) or 3))
            )
        except (TypeError, ValueError):
            self._simulation_default_interval = 3
        try:
            self._lexy_turn_probability = max(
                0.0, min(1.0, float(cfg.get("lexy_turn_probability", 0.3) or 0.3))
            )
        except (TypeError, ValueError):
            self._lexy_turn_probability = 0.3

        # Context window overrides. Positive = force this size; 0 = auto
        # from brain. Safety margin reserves room for tokenizer overhead
        # and prompt framing.
        try:
            self._context_size_override = int(
                cfg.get("context_size_override", 0) or 0
            )
        except (TypeError, ValueError):
            self._context_size_override = 0
        try:
            self._context_safety_margin = int(
                cfg.get("context_safety_margin", 256) or 256
            )
        except (TypeError, ValueError):
            self._context_safety_margin = 256

    async def on_enable(self) -> None:
        # LLM Tools
        self.api.register_tool(
            name="spawn_character",
            handler=self._tool_spawn_character,
            description=(
                "Spawne einen neuen RP-Charakter mit Persona/Greeting/Alter. "
                "Wird persistent gespeichert und kann danach in Sessions "
                "attached werden."
            ),
            schema=SPAWN_CHARACTER_SCHEMA,
        )
        self.api.register_tool(
            name="update_character",
            handler=self._tool_update_character,
            description="Aktualisiere Felder eines existierenden Charakters.",
            schema=UPDATE_CHARACTER_SCHEMA,
        )
        self.api.register_tool(
            name="archive_character",
            handler=self._tool_archive_character,
            description="Archiviere einen Charakter (bleibt in DB, spricht nicht mehr).",
            schema=ARCHIVE_CHARACTER_SCHEMA,
        )
        self.api.register_tool(
            name="unarchive_character",
            handler=self._tool_unarchive_character,
            description="Reaktiviere einen archivierten Charakter.",
            schema=ARCHIVE_CHARACTER_SCHEMA,
        )
        self.api.register_tool(
            name="list_characters",
            handler=self._tool_list_characters,
            description="Liste alle Charaktere (optional: session_id / include_archived).",
            schema=LIST_CHARACTERS_SCHEMA,
        )
        self.api.register_tool(
            name="attach_character_to_session",
            handler=self._tool_attach_character,
            description="Hänge einen Charakter an eine Session an.",
            schema=ATTACH_CHARACTER_SCHEMA,
        )
        self.api.register_tool(
            name="detach_character_from_session",
            handler=self._tool_detach_character,
            description="Entferne einen Charakter aus einer Session.",
            schema=ATTACH_CHARACTER_SCHEMA,
        )
        self.api.register_tool(
            name="import_character_card",
            handler=self._tool_import_card,
            description=(
                "Importiere einen Silly-Tavern Character Card JSON "
                "(v1 flach oder v2 mit 'data' nested)."
            ),
            schema=IMPORT_CARD_SCHEMA,
        )
        self.api.register_tool(
            name="run_character_round",
            handler=self._tool_run_round,
            description=(
                "Starte eine Runde Charakter-Chat für eine Session. Der "
                "Orchestrator entscheidet welche Charaktere reagieren und "
                "in welcher Reihenfolge (sequentielles Prompting)."
            ),
            schema=RUN_ROUND_SCHEMA,
        )

        self.api.register_tool(
            name="set_character_mode",
            handler=self._tool_set_character_mode,
            description=(
                "Aktiviere/deaktiviere character_mode für eine Session. Bei "
                "aktiviertem Modus übernimmt der Charakter-Orchestrator "
                "vollständig — der normale Lexy-Agent antwortet nicht mehr."
            ),
            schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "scene": {"type": "string"},
                },
                "required": ["session_id", "enabled"],
            },
        )
        self.api.register_tool(
            name="start_simulation",
            handler=self._tool_start_simulation,
            description=(
                "Starte eine autonome Simulation in einer Session. Alle "
                "interval_minutes Minuten spricht EIN Charakter (oder Lexy) "
                "automatisch und reagiert auf den letzten Turn. Default: 3 min."
            ),
            schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "interval_minutes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 15,
                    },
                },
                "required": ["session_id"],
            },
        )
        self.api.register_tool(
            name="stop_simulation",
            handler=self._tool_stop_simulation,
            description="Stoppe die autonome Simulation einer Session.",
            schema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        )

        # ── Lorebook tools (Phase 9.8) ───────────────────────────────
        self.api.register_tool(
            name="lorebook_create",
            handler=self._tool_lorebook_create,
            description=(
                "Erstelle ein neues Lorebook. scope=global|character|session, "
                "scope_id ist die character_id (für character) oder "
                "session_id (für session) — leer für global."
            ),
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": ["global", "character", "session"],
                    },
                    "scope_id": {"type": "string"},
                    "token_budget": {"type": "integer"},
                },
                "required": ["name"],
            },
        )
        self.api.register_tool(
            name="lorebook_list",
            handler=self._tool_lorebook_list,
            description="Liste alle Lorebooks (optional gefiltert nach scope/scope_id).",
            schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string"},
                    "scope_id": {"type": "string"},
                    "enabled_only": {"type": "boolean"},
                },
            },
        )
        self.api.register_tool(
            name="lorebook_update",
            handler=self._tool_lorebook_update,
            description="Aktualisiere ein Lorebook (name/description/enabled/token_budget).",
            schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "token_budget": {"type": "integer"},
                },
                "required": ["id"],
            },
        )
        self.api.register_tool(
            name="lorebook_delete",
            handler=self._tool_lorebook_delete,
            description="Lösche ein Lorebook + alle seine Einträge.",
            schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        )
        self.api.register_tool(
            name="lore_entry_create",
            handler=self._tool_lore_entry_create,
            description=(
                "Erstelle einen Eintrag in einem Lorebook. keys = Liste "
                "von Trigger-Wörtern (case-insensitive Substring). "
                "always_on=true → feuert jede Runde ohne Trigger. "
                "position bestimmt wo im Prompt der Inhalt landet."
            ),
            schema={
                "type": "object",
                "properties": {
                    "lorebook_id": {"type": "string"},
                    "name": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "content": {"type": "string"},
                    "position": {
                        "type": "string",
                        "enum": list(VALID_POSITIONS),
                    },
                    "priority": {"type": "integer"},
                    "always_on": {"type": "boolean"},
                    "scan_depth": {"type": "integer"},
                },
                "required": ["lorebook_id", "name"],
            },
        )
        self.api.register_tool(
            name="lore_entry_list",
            handler=self._tool_lore_entry_list,
            description="Liste alle Einträge eines Lorebooks.",
            schema={
                "type": "object",
                "properties": {
                    "lorebook_id": {"type": "string"},
                    "enabled_only": {"type": "boolean"},
                },
                "required": ["lorebook_id"],
            },
        )
        self.api.register_tool(
            name="lore_entry_update",
            handler=self._tool_lore_entry_update,
            description="Aktualisiere einen Lore-Eintrag.",
            schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "content": {"type": "string"},
                    "position": {"type": "string"},
                    "priority": {"type": "integer"},
                    "always_on": {"type": "boolean"},
                    "scan_depth": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                },
                "required": ["id"],
            },
        )
        self.api.register_tool(
            name="lore_entry_delete",
            handler=self._tool_lore_entry_delete,
            description="Lösche einen Lore-Eintrag.",
            schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        )

        # WebSocket handlers
        self.api.register_ws_handler("character_list", self._ws_list_characters)
        self.api.register_ws_handler("character_create", self._ws_create_character)
        self.api.register_ws_handler("character_update", self._ws_update_character)
        self.api.register_ws_handler("character_archive", self._ws_archive_character)
        self.api.register_ws_handler(
            "character_unarchive", self._ws_unarchive_character
        )
        self.api.register_ws_handler("character_delete", self._ws_delete_character)
        self.api.register_ws_handler("character_attach", self._ws_attach_character)
        self.api.register_ws_handler("character_detach", self._ws_detach_character)
        self.api.register_ws_handler("character_turn_request", self._ws_run_round)
        self.api.register_ws_handler("character_history", self._ws_history)
        # Per-turn editing — Mike's audit point #3: at parity with how
        # Lexy's normal-chat bubbles can be edited / deleted /
        # regenerated. The frontend bubble action bar dispatches these.
        self.api.register_ws_handler(
            "character_turn_edit", self._ws_turn_edit
        )
        self.api.register_ws_handler(
            "character_turn_delete", self._ws_turn_delete
        )
        self.api.register_ws_handler(
            "character_turn_regenerate", self._ws_turn_regenerate
        )
        self.api.register_ws_handler("character_import", self._ws_import_card)
        self.api.register_ws_handler("character_session_get", self._ws_session_get)
        self.api.register_ws_handler("character_session_set", self._ws_session_set)
        self.api.register_ws_handler("simulation_start", self._ws_simulation_start)
        self.api.register_ws_handler("simulation_stop", self._ws_simulation_stop)
        self.api.register_ws_handler(
            "simulation_status_get", self._ws_simulation_status_get
        )
        # Lorebook WS handlers — frontend admin panel uses these.
        self.api.register_ws_handler(
            "lorebook_list", self._ws_lorebook_list,
        )
        self.api.register_ws_handler(
            "lorebook_create", self._ws_lorebook_create,
        )
        self.api.register_ws_handler(
            "lorebook_update", self._ws_lorebook_update,
        )
        self.api.register_ws_handler(
            "lorebook_delete", self._ws_lorebook_delete,
        )
        self.api.register_ws_handler(
            "lore_entry_list", self._ws_lore_entry_list,
        )
        self.api.register_ws_handler(
            "lore_entry_create", self._ws_lore_entry_create,
        )
        self.api.register_ws_handler(
            "lore_entry_update", self._ws_lore_entry_update,
        )
        self.api.register_ws_handler(
            "lore_entry_delete", self._ws_lore_entry_delete,
        )

        # Hook: intercept user messages before the normal agent runs them.
        # priority=30 runs before memory/routing hooks (default 50).
        self.api.register_hook(
            "before_user_input", self._hook_before_user_input, priority=30
        )

        # Event: proactive pulses come in as scheduler-triggered events with
        # action_type = "character_pulse". See _on_scheduler_triggered.
        self.api.on_event(
            "core.scheduler_triggered", self._on_scheduler_triggered
        )

        # Event: when a session is moved to a different project, clear
        # character_mode. Characters are memory-scoped per project (strict
        # isolation), so silently leaving character_mode on in a foreign
        # project would mix contexts in confusing ways. Safer to disable
        # and let the user re-enable explicitly if desired.
        self.api.on_event(
            "core.session_project_changed",
            self._on_session_project_changed,
        )

        # Hook: after Lexy responds in hybrid mode, fire the character round
        # so the characters react to both the user AND Lexy's response.
        self.api.register_hook(
            "after_response_send",
            self._hook_after_response_send,
            priority=30,
        )

        # Re-hydrate the in-memory pulse-timers dict from the scheduler's
        # persistent DB. Without this, a restart would leave pulse timers
        # running in the scheduler but the plugin couldn't cancel them
        # (the ``_pulse_timers`` dict starts empty on boot). We kick this
        # off in the background so on_enable itself stays fast — the
        # scheduler plugin may still be warming up at this exact instant.
        asyncio.create_task(self._rehydrate_pulse_timers())

        log.info("character_chat.enabled", max_speakers=self._max_speakers)

    async def on_disable(self) -> None:
        self._session_locks.clear()
        self._pulse_rehydrated = False  # allow re-rehydrate on re-enable
        self._pulse_cooldowns.clear()
        # Cancel all outstanding pulse timers — the scheduler plugin's
        # cleanup already removes them on shutdown, but we defensively
        # unregister so a disable/re-enable cycle doesn't double-book.
        await self._cancel_all_pulse_timers()
        # Phase 13: close every open RP session container so SQLite
        # handles drop cleanly on shutdown.
        if self._rp_registry is not None:
            try:
                await self._rp_registry.shutdown()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.rp_registry_shutdown_failed",
                    error=str(exc),
                )
        log.info("character_chat.disabled")

    # ─── Hook: before_user_input intercept ───────────────────────────────

    async def _hook_before_user_input(
        self, ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Intercept user messages for character-mode sessions.

        Three modes (``character_mode`` column in ``character_sessions``):

        * ``0`` — off. Lexy's normal agent handles everything. Characters
          only react via pulse timers.
        * ``1`` — characters only. Lexy is skipped; attached characters
          answer the user's message directly.
        * ``2`` — **hybrid**. Lexy answers first (normal agent flow),
          then a character round runs so characters react to both the
          user AND Lexy's response. Natural mode for a family RP where
          Lexy and her children all participate.
        """
        if self._store is None or self._orchestrator is None:
            return ctx

        session_id = str(ctx.get("session_id", "") or "")
        if not session_id:
            return ctx

        state = await self._get_session_state(session_id)
        mode = int(state.get("character_mode") or 0)

        if mode == 0:
            return ctx

        user_text = str(ctx.get("text", "") or "").strip()
        if not user_text:
            return ctx

        scene = str(state.get("scene") or "")

        if mode == 1:
            # Characters-only: skip Lexy entirely, fire character round.
            asyncio.create_task(
                self._run_round_safe(
                    session_id=session_id,
                    user_message=user_text,
                    scene=scene,
                ),
                name=f"character_chat.round.{session_id}",
            )
            ctx["skip_agent"] = True
            ctx["skip_reason"] = "character_mode"

        elif mode == 2:
            # Hybrid: let Lexy answer first (don't skip agent). Stash
            # the intent so after_response_send fires the character round
            # once Lexy has spoken.
            ctx["_character_hybrid_round"] = {
                "session_id": session_id,
                "user_message": user_text,
                "scene": scene,
            }

        return ctx

    async def _hook_after_response_send(
        self, ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Hybrid mode: fire a character round AFTER Lexy has spoken.

        The ``before_user_input`` hook stashed a ``_character_hybrid_round``
        dict in ctx when the session is in mode=2. Now that Lexy has
        produced her response (it's already in session_store + broadcast),
        we kick off a character round so the attached characters react to
        both the user's message AND Lexy's answer.

        This creates a natural conversation flow:
            Mike: "Wie geht es den Kindern?"
            Lexy: "Ich glaube Sandra hat Hunger..."
            Sandra: *schreit und greift nach Mama*
            Luna: *gurgelt zufrieden*
        """
        hybrid = ctx.get("_character_hybrid_round")
        if not hybrid:
            return ctx

        session_id = str(hybrid.get("session_id") or "")
        user_message = str(hybrid.get("user_message") or "")
        scene = str(hybrid.get("scene") or "")

        if not session_id:
            return ctx

        # Small delay so Lexy's response has time to arrive at the
        # frontend before the character turns start streaming.
        async def _delayed_round() -> None:
            await asyncio.sleep(0.5)
            await self._run_round_safe(
                session_id=session_id,
                user_message=user_message,
                scene=scene,
            )

        asyncio.create_task(
            _delayed_round(),
            name=f"character_chat.hybrid_round.{session_id}",
        )
        return ctx

    async def _run_round_safe(
        self,
        *,
        session_id: str,
        user_message: str = "",
        pulse_from_id: str = "",
        pulse_text: str = "",
        scene: str = "",
    ) -> None:
        """Wrap ``_tool_run_round`` with a safety net for background tasks.

        After a pulse-triggered round completes, if
        ``lexy_auto_reacts_to_pulses`` is enabled, Lexy (the main agent)
        is nudged via ``agent_proactive`` so she responds naturally to
        what happened — e.g. "Sandra weint" → Lexy goes "*nimmt Sandra
        hoch und wiegt sie*".
        """
        try:
            result = await self._tool_run_round(
                session_id=session_id,
                user_message=user_message,
                pulse_from_id=pulse_from_id,
                pulse_text=pulse_text,
                scene=scene,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character_chat.background_round_failed",
                session_id=session_id,
                error=str(exc),
            )
            await self.api.ws_broadcast(
                {
                    "type": "character_round_error",
                    "session_id": session_id,
                    "error": str(exc),
                }
            )
            return

        # Lexy auto-reacts to pulse-triggered rounds. This makes her
        # respond to baby-crying, toddler-tugging etc. without the user
        # having to type anything — the core agent speaks *as Lexy* in
        # the session, naturally reacting to what the characters did.
        # Phase 13: respect the session's character_mode. Mode 1 ("only
        # characters, Lexy stays silent") is Mike's autonomous-test
        # setting — Lexy butting in there breaks the immersion. Only
        # Mode 0 (chat-tab default) and Mode 2 (hybrid) get auto-react.
        sess_state_for_react = await self._get_session_state(session_id)
        sess_mode = int(sess_state_for_react.get("character_mode") or 0)
        if (
            self._lexy_auto_reacts
            and sess_mode != 1
            and pulse_from_id
            and result
            and result.get("ok")
        ):
            # Build a short summary of what happened in the round so the
            # agent prompt is grounded in the actual character turns.
            turns = result.get("turns") or []
            turn_summary = "; ".join(
                f"{t.get('character_name', '?')}: {(t.get('content') or '')[:120]}"
                for t in turns
                if not t.get("skipped") and t.get("content")
            )
            if not turn_summary:
                turn_summary = pulse_text or "Etwas ist passiert."

            # Fetch the pulse originator's name for a natural prompt.
            pulse_name = pulse_from_id
            if self._store:
                try:
                    card = await self._store.get(pulse_from_id)
                    if card:
                        pulse_name = card.name
                except Exception:  # noqa: BLE001
                    pass

            prompt = (
                f"{pulse_name} hat gerade etwas getan: {turn_summary}\n"
                "Reagiere natürlich und in-character als Lexy. "
                "Kurz (1-3 Sätze). Keine Meta-Kommentare."
            )
            try:
                await self.api.agent_proactive(
                    session_id=session_id,
                    prompt=prompt,
                    label=f"auto_react:{pulse_name}",
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "character_chat.auto_react_failed error=%s", exc
                )

    # ─── Event: scheduler-triggered pulses ───────────────────────────────

    async def _on_scheduler_triggered(self, event: Any) -> None:
        """Scheduler just fired a timer — check if it's one of ours.

        Character pulses are registered with the scheduler using
        ``action_type = "character_pulse"`` and payload
        ``{character_id, session_id, pulse_text}``. The scheduler emits
        ``core.scheduler_triggered`` with the decoded action on every fire.
        """
        if not self._proactive_pulses_enabled:
            return

        data: dict[str, Any] = {}
        if hasattr(event, "data") and isinstance(event.data, dict):
            data = event.data
        elif isinstance(event, dict):
            data = event
        else:
            return

        # Phase-3: the scheduler now emits both ``action_type`` and the raw
        # ``action`` dict alongside the legacy fields. Filter by action_type
        # first — we only care about character pulses.
        action_type = str(data.get("action_type", "") or "")
        action = data.get("action") or {}
        if isinstance(action, str):
            # Defensive: pre-3 scheduler versions may still send a JSON string.
            try:
                action = json.loads(action)
            except (TypeError, json.JSONDecodeError):
                return
        if not isinstance(action, dict):
            return

        # Accept either the outer ``action_type`` or the inner ``type`` key.
        kind = action_type or action.get("type", "")

        # Autonomous simulation tick: fire one speaker per tick.
        if kind == "autonomous_sim":
            sim_session = str(action.get("session_id") or "")
            if sim_session:
                if self._is_session_stale(sim_session):
                    log.debug(
                        "character_chat.sim_skipped_stale_session "
                        "session=%s threshold=%.0fs",
                        sim_session, self._pulse_session_stale_seconds,
                    )
                    return
                asyncio.create_task(
                    self._run_autonomous_tick(sim_session),
                    name=f"character_chat.sim_tick.{sim_session}",
                )
            return

        if kind != "character_pulse":
            return

        character_id = str(action.get("character_id", "") or "")
        session_id = str(action.get("session_id", "") or data.get("session_id", "") or "")
        pulse_text = str(action.get("pulse_text", "") or "")

        if not character_id or not session_id:
            return

        # Debounce: when multiple babies share a session, each has its
        # own timer. Without a cooldown, Sandra fires → others react →
        # 2 min later Sophie fires → others react → endless cascade.
        # We enforce a minimum gap between pulse-rounds per session.
        now = time.time()
        last_pulse = self._pulse_cooldowns.get(session_id, 0.0)
        if now - last_pulse < self._pulse_cooldown_seconds:
            log.debug(
                "character_chat.pulse_debounced character=%s session=%s "
                "cooldown_remaining=%.0fs",
                character_id,
                session_id,
                self._pulse_cooldown_seconds - (now - last_pulse),
            )
            return

        # Staleness guard: skip pulses for sessions Mike hasn't touched
        # in a while. Without this, a character with a 2h pulse pattern
        # would keep firing forever even though the user moved on days
        # ago — wasting LLM calls and producing ghost-replies for
        # sessions no one is reading. Resumes automatically when the
        # user comes back (any new user message updates session
        # ``meta.updated_at``).
        if self._is_session_stale(session_id):
            log.info(
                "character_chat.pulse_skipped_stale_session "
                "character=%s session=%s threshold=%.0fs",
                character_id, session_id, self._pulse_session_stale_seconds,
            )
            return

        self._pulse_cooldowns[session_id] = now

        # Resolve the card so we can enrich the pulse text. Fallback chain:
        #   1. ``pulse_text`` from the scheduler payload (legacy / manual)
        #   2. ``card.proactive_pulse_prompt`` (per-character override)
        #   3. LLM-generated text via PulseGenerator (smart_pulses_enabled)
        #   4. Static age-stage default (last resort, original behaviour)
        if self._store is not None and not pulse_text:
            card = await self._store.get(character_id)
            if card is not None:
                if card.proactive_pulse_prompt:
                    pulse_text = card.proactive_pulse_prompt
                elif self._smart_pulses_enabled and self._pulse_generator is not None:
                    others = await self._store.list_in_session(session_id)
                    history = self._load_session_history(session_id)
                    sess_state = await self._get_session_state(session_id)
                    scene = str(sess_state.get("scene") or "")
                    try:
                        pulse_text = await self._pulse_generator.generate(
                            character=card,
                            others_in_session=others,
                            recent_history=history,
                            scene=scene,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "character_chat.pulse_generator_failed character=%s error=%s",
                            card.name,
                            str(exc),
                        )
                        pulse_text = ""
                if not pulse_text:
                    pulse_text = _default_pulse(card.age_stage)

        if not pulse_text:
            pulse_text = "*bemerkt etwas und bewegt sich*"

        log.info(
            "character_chat.pulse_received",
            character_id=character_id,
            session_id=session_id,
        )

        asyncio.create_task(
            self._run_round_safe(
                session_id=session_id,
                user_message="",
                pulse_from_id=character_id,
                pulse_text=pulse_text,
            ),
            name=f"character_chat.pulse.{character_id}",
        )

    async def _run_autonomous_tick(self, session_id: str) -> None:
        """One tick of the autonomous simulation.

        Each tick picks EXACTLY one speaker:
        * With probability ``lexy_turn_probability``, Lexy speaks via
          ``api.agent_proactive`` — she reacts as herself to the scene.
        * Otherwise one character (LLM-selected) speaks a single turn,
          reacting to the last few messages.

        Shares the pulse cooldown so a pulse round that just fired won't
        trigger an immediate sim tick on top of it.
        """
        import random

        if not session_id or self._store is None:
            return

        # Read-only on the pulse cooldown: we respect a pulse round that
        # just fired (don't stack a sim tick on top of it), but we don't
        # write our own timestamp back. Our rate limit is ``interval_minutes``
        # from the user-facing start — writing here would self-block every
        # subsequent tick whenever the sim interval is shorter than
        # ``pulse_cooldown_seconds`` (e.g. 2 min interval vs. 10 min cooldown).
        now = time.time()
        last_round = self._pulse_cooldowns.get(session_id, 0.0)
        if now - last_round < self._pulse_cooldown_seconds:
            log.debug(
                "character_chat.sim_tick_debounced session=%s remaining=%.0fs",
                session_id,
                self._pulse_cooldown_seconds - (now - last_round),
            )
            return

        # Fetch session context once.
        state = await self._get_session_state(session_id)
        scene = str(state.get("scene") or "")

        try:
            characters = await self._store.list_in_session(session_id)
        except Exception:  # noqa: BLE001
            characters = []

        # Roll for Lexy's turn. Lexy always gets a chance, even without
        # any characters attached (in that case probability = 1).
        lexy_roll = random.random() < self._lexy_turn_probability or not characters

        if lexy_roll:
            prompt = (
                "In der Szene ist gerade etwas Zeit vergangen. "
                "Schau auf die letzten Nachrichten und reagiere natürlich "
                "als du selbst — eine kurze, alltägliche Bemerkung oder "
                "Handlung (1-2 Sätze). Keine Frage an Mike, kein "
                "Meta-Kommentar. Einfach weiterleben in der Szene."
            )
            if scene:
                prompt = f"Szene: {scene}\n\n" + prompt
            try:
                await self.api.agent_proactive(
                    session_id=session_id,
                    prompt=prompt,
                    label="autonomous_sim:lexy",
                )
                log.info(
                    "character_chat.sim_tick_lexy session=%s", session_id
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.sim_tick_lexy_failed session=%s error=%s",
                    session_id, exc,
                )
            return

        # Character turn: run a 1-speaker round, no pulse originator.
        # The orchestrator's LLM-picker selects who makes sense right now.
        if self._orchestrator is None:
            return
        original_max = self._orchestrator._max_speakers  # type: ignore[attr-defined]
        try:
            self._orchestrator._max_speakers = 1  # type: ignore[attr-defined]
            log.info(
                "character_chat.sim_tick_character session=%s candidates=%d",
                session_id, len(characters),
            )
            await self._run_round_safe(
                session_id=session_id,
                user_message="",
                pulse_from_id="",
                pulse_text="",
                scene=scene,
            )
        finally:
            self._orchestrator._max_speakers = original_max  # type: ignore[attr-defined]

    async def _on_session_project_changed(self, event: Any) -> None:
        """React when a session is moved to a different project.

        Characters are memory-scoped per project (strict isolation). When
        a session moves across a project boundary, silently keeping
        ``character_mode = 1`` would cause its characters' memory lookups
        to happen under a different project scope than they were
        recorded under — confusing for the user and for the characters
        themselves.

        Strategy: disable character_mode + detach all characters from the
        session. The user can re-enable explicitly after reviewing which
        characters belong in the new project. Pulse timers for that
        session are cancelled because the detach cascade already calls
        ``_cancel_pulse_timer``.
        """
        data: dict[str, Any] = {}
        if hasattr(event, "data") and isinstance(event.data, dict):
            data = event.data
        elif isinstance(event, dict):
            data = event
        else:
            return

        session_id = str(data.get("session_id", "") or "")
        if not session_id:
            return

        current = await self._get_session_state(session_id)
        if not current.get("character_mode"):
            # Session wasn't in character mode; nothing to sync.
            return

        log.info(
            "character_chat.session_project_changed_auto_detach session_id=%s "
            "from=%s to=%s",
            session_id,
            data.get("from_project"),
            data.get("to_project"),
        )

        # Detach all bound characters — this also cancels their pulse timers.
        if self._store is not None:
            try:
                bound = await self._store.list_in_session(session_id)
            except Exception:  # noqa: BLE001
                bound = []
            for card in bound:
                try:
                    await self._store.detach_from_session(card.id, session_id)
                    await self._cancel_pulse_timer(card.id, session_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "character_chat.cross_project_detach_failed "
                        "character=%s session=%s error=%s",
                        card.id,
                        session_id,
                        exc,
                    )

        # Flip character_mode off.
        await self._set_session_state(session_id, character_mode=False)

    # ─── Session character_mode bookkeeping ──────────────────────────────

    async def _get_session_state(self, session_id: str) -> dict[str, Any]:
        """Fetch the row from ``character_sessions`` (or defaults)."""
        db = await self.api.get_db()
        cursor = await db.execute(
            "SELECT character_mode, scene, updated_at "
            "FROM character_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return {"character_mode": 0, "scene": "", "updated_at": 0.0}
        return {
            "character_mode": int(row[0] or 0),
            "scene": str(row[1] or ""),
            "updated_at": float(row[2] or 0.0),
        }

    async def _set_session_state(
        self,
        session_id: str,
        *,
        character_mode: int | bool | None = None,
        scene: str | None = None,
    ) -> dict[str, Any]:
        """Update session character state.

        ``character_mode`` accepts:
        * ``0`` / ``False`` — off
        * ``1`` / ``True`` — characters only (legacy compat)
        * ``2`` — hybrid (Lexy + characters)
        """
        current = await self._get_session_state(session_id)
        if character_mode is not None:
            new_mode = int(character_mode)
        else:
            new_mode = int(current["character_mode"])
        new_scene = scene if scene is not None else current["scene"]
        db = await self.api.get_db()
        await db.execute(
            "INSERT INTO character_sessions (session_id, character_mode, scene, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "character_mode=excluded.character_mode, "
            "scene=excluded.scene, "
            "updated_at=excluded.updated_at",
            (session_id, new_mode, new_scene, time.time()),
        )
        await db.commit()
        await self.api.ws_broadcast(
            {
                "type": "character_session_mode",
                "session_id": session_id,
                "character_mode": new_mode,
                "scene": new_scene,
            }
        )
        return {
            "session_id": session_id,
            "character_mode": new_mode,
            "scene": new_scene,
        }

    # ─── Tool handlers ───────────────────────────────────────────────────

    async def _tool_spawn_character(self, **kwargs: Any) -> dict[str, Any]:
        """Create a character, optionally auto-attach + register pulses.

        When ``session_id`` is given, the character is immediately attached
        to that session and (for young age stages without an explicit
        pattern) a default proactive pulse is registered. This lets the LLM
        spawn 4 babies in a row without needing 12 separate tool calls.
        """
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}

        age_stage = str(kwargs.get("age_stage", "adult") or "adult")
        explicit_pattern = str(kwargs.get("proactive_pulse_pattern", "") or "")

        # Auto-default: babies/toddlers/children get a pulse pattern if the
        # caller didn't set one explicitly. Teens and adults stay quiet by
        # default (they can speak when addressed).
        pulse_pattern = explicit_pattern
        if not pulse_pattern:
            pulse_pattern = _DEFAULT_PULSE_PATTERNS.get(age_stage, "")

        try:
            card = CharacterCard(
                name=str(kwargs.get("name", "")),
                persona=str(kwargs.get("persona", "") or ""),
                greeting=str(kwargs.get("greeting", "") or ""),
                scenario=str(kwargs.get("scenario", "") or ""),
                example_dialog=str(kwargs.get("example_dialog", "") or ""),
                age_stage=age_stage,
                color=str(kwargs.get("color") or "#7aa2f7"),
                voice=str(kwargs.get("voice", "") or ""),
                relationships=dict(kwargs.get("relationships") or {}),
                tags=list(kwargs.get("tags") or []),
                proactive_pulse_pattern=pulse_pattern,
            )
        except (CharacterCardError, ValidationError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        try:
            saved = await self._store.create(card)
        except CharacterCardError as exc:
            return {"ok": False, "error": str(exc)}
        await self._broadcast_character_event("character_created", saved)

        # Auto-attach to session if requested. This saves the LLM from
        # having to make a separate attach_character call per baby.
        session_id = str(kwargs.get("session_id", "") or "")
        attached = False
        if session_id and self._store is not None:
            try:
                await self._store.attach_to_session(saved.id, session_id)
                attached = True
                await self.api.ws_broadcast(
                    {
                        "type": "character_attached",
                        "character_id": saved.id,
                        "session_id": session_id,
                    }
                )
                # Register proactive pulse timer if the card has a pattern.
                if saved.proactive_pulse_pattern and self._proactive_pulses_enabled:
                    await self._register_pulse_timer(saved, session_id)
                # Phase 9.12: same auto-tag as ``_tool_attach_character``.
                await self._maybe_tag_session_rp(session_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.spawn_auto_attach_failed",
                    character=saved.name,
                    session_id=session_id,
                    error=str(exc),
                )

        # Minimal result — multi-call chains (e.g. 4× spawn for quadruplets)
        # bloat the LLM context if we return full card JSON each time.
        return {
            "ok": True,
            "id": saved.id,
            "name": saved.name,
            "age_stage": saved.age_stage,
            "pulse_pattern": saved.proactive_pulse_pattern,
            "attached_to": session_id if attached else "",
        }

    async def _tool_update_character(
        self, id: str, **patch: Any
    ) -> dict[str, Any]:
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}
        try:
            updated = await self._store.update(id, **patch)
        except CharacterCardError as exc:
            return {"ok": False, "error": str(exc)}
        if updated is None:
            return {"ok": False, "error": "not_found"}
        await self._broadcast_character_event("character_updated", updated)
        return {"ok": True, "character": _card_to_public(updated)}

    async def _tool_archive_character(self, id: str) -> dict[str, Any]:
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}
        ok = await self._store.archive(id)
        if ok:
            card = await self._store.get(id)
            if card is not None:
                await self._broadcast_character_event("character_updated", card)
        return {"ok": ok}

    async def _tool_unarchive_character(self, id: str) -> dict[str, Any]:
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}
        ok = await self._store.unarchive(id)
        if ok:
            card = await self._store.get(id)
            if card is not None:
                await self._broadcast_character_event("character_updated", card)
        return {"ok": ok}

    async def _tool_list_characters(
        self,
        include_archived: bool = False,
        session_id: str = "",
    ) -> dict[str, Any]:
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}
        if session_id:
            cards = await self._store.list_in_session(
                session_id, include_archived=include_archived
            )
        else:
            cards = await self._store.list(include_archived=include_archived)
        return {
            "ok": True,
            "characters": [_card_to_public(c) for c in cards],
        }

    async def _tool_attach_character(
        self, id: str, session_id: str
    ) -> dict[str, Any]:
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}
        updated = await self._store.attach_to_session(id, session_id)
        if updated is None:
            return {"ok": False, "error": "not_found"}
        await self._broadcast_character_event("character_updated", updated)
        # Phase 9.12 — auto-tag the session as ``kind="rp"``. Lazy
        # migration: old chat-only sessions are upgraded the first time
        # someone attaches a character, so existing roleplay sessions
        # show up in the new RP tab without a one-shot migration script.
        # ``set_kind`` is idempotent (returns False on no-change) so we
        # only broadcast when it actually flips.
        await self._maybe_tag_session_rp(session_id)
        # Phase 13: ensure an RP container exists for this session and
        # snapshot the character's session-state from the session's
        # tracked_stats defaults. We never clobber existing live state
        # on re-attach (snapshot is a no-op if state is already set).
        try:
            container = await self._get_rp_container(session_id)
            if container is not None:
                await container.snapshot_template_for_char(id)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.rp_snapshot_failed",
                character_id=id,
                session_id=session_id,
                error=str(exc),
            )
        # If the card has a proactive-pulse pattern, register it with the
        # scheduler now (idempotent: duplicate attach simply re-registers).
        if updated.proactive_pulse_pattern:
            await self._register_pulse_timer(updated, session_id)
        return {"ok": True, "character": _card_to_public(updated)}

    async def _maybe_tag_session_rp(self, session_id: str) -> None:
        """Flip ``meta.kind`` to ``"rp"`` when a character first joins.

        Silent no-op if the session store can't be reached (tests that
        stub the api), if the session doesn't exist yet, or if the kind
        was already ``"rp"``. Broadcasts ``session_kind_changed`` only
        when an actual flip happened so stale tabs can re-route.
        """
        if not session_id:
            return
        session_store = getattr(self.api._app, "session_store", None)
        if session_store is None:
            return
        try:
            changed = session_store.set_kind(session_id, "rp")
        except (ValueError, AttributeError):
            return
        if changed:
            try:
                await self.api.ws_broadcast(
                    {
                        "type": "session_kind_changed",
                        "session_id": session_id,
                        "kind": "rp",
                    }
                )
            except Exception:  # noqa: BLE001
                pass

    async def _tool_detach_character(
        self, id: str, session_id: str
    ) -> dict[str, Any]:
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}
        updated = await self._store.detach_from_session(id, session_id)
        if updated is None:
            return {"ok": False, "error": "not_found"}
        await self._broadcast_character_event("character_updated", updated)
        await self._cancel_pulse_timer(id, session_id)
        return {"ok": True, "character": _card_to_public(updated)}

    async def _tool_set_character_mode(
        self,
        session_id: str,
        enabled: bool | int = True,
        mode: int | None = None,
        scene: str = "",
    ) -> dict[str, Any]:
        """Set character mode. Accepts:
        * ``mode=0/1/2`` (preferred)
        * ``enabled=True/False`` (legacy compat, maps to 1/0)
        """
        if not session_id:
            return {"ok": False, "error": "session_id_required"}
        if mode is not None:
            effective_mode = max(0, min(2, int(mode)))
        else:
            effective_mode = 1 if enabled else 0
        state = await self._set_session_state(
            session_id, character_mode=effective_mode, scene=scene
        )
        return {"ok": True, **state}

    # ─── Autonomous Simulation tools ─────────────────────────────────────

    async def _tool_start_simulation(
        self,
        session_id: str,
        interval_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Start the autonomous simulation for ``session_id``.

        Registers a recurring scheduler timer with ``action_type =
        "autonomous_sim"``. If a timer already exists for the session,
        it's cancelled first (idempotent).
        """
        if not session_id:
            return {"ok": False, "error": "session_id_required"}

        # Differentiate "not provided" (None) from explicit 0 (clamp to 1).
        interval_raw = (
            interval_minutes
            if interval_minutes is not None
            else self._simulation_default_interval
        )
        interval = max(1, min(15, int(interval_raw)))

        # Cancel any existing timer for this session first.
        old_timer = self._simulation_timers.pop(session_id, None)
        if old_timer:
            try:
                await self.api.call_tool("cancel_timer", {"id": old_timer})
            except Exception:  # noqa: BLE001
                pass

        try:
            result = await self.api.call_tool(
                "set_recurring",
                {
                    "label": f"autonomous_sim:{session_id[:8]}",
                    "pattern": f"every {interval}m",
                    "action_type": "autonomous_sim",
                    "action_payload": {
                        "session_id": session_id,
                        "interval_minutes": interval,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"scheduler_unavailable: {exc}"}

        if not result.get("ok"):
            return {"ok": False, "error": result.get("error") or "register_failed"}

        timer_id = str(result.get("data", {}).get("id") or "")
        if not timer_id:
            return {"ok": False, "error": "no_timer_id_returned"}

        self._simulation_timers[session_id] = timer_id
        log.info(
            "character_chat.sim_started session=%s interval=%dm timer=%s",
            session_id, interval, timer_id,
        )
        await self.api.ws_broadcast(
            {
                "type": "simulation_started",
                "session_id": session_id,
                "interval_minutes": interval,
                "timer_id": timer_id,
            }
        )
        return {
            "ok": True,
            "session_id": session_id,
            "interval_minutes": interval,
            "timer_id": timer_id,
        }

    async def _tool_stop_simulation(self, session_id: str) -> dict[str, Any]:
        """Stop the autonomous simulation for ``session_id``."""
        if not session_id:
            return {"ok": False, "error": "session_id_required"}

        timer_id = self._simulation_timers.pop(session_id, None)
        if not timer_id:
            # Nothing to stop — consider this a no-op success so the
            # frontend can safely re-click stop without error.
            return {"ok": True, "session_id": session_id, "was_running": False}

        try:
            await self.api.call_tool("cancel_timer", {"id": timer_id})
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.sim_stop_cancel_failed session=%s error=%s",
                session_id, exc,
            )

        log.info("character_chat.sim_stopped session=%s", session_id)
        await self.api.ws_broadcast(
            {"type": "simulation_stopped", "session_id": session_id}
        )
        return {"ok": True, "session_id": session_id, "was_running": True}

    async def _tool_simulation_status(
        self, session_id: str
    ) -> dict[str, Any]:
        """Return whether the sim is running for a session + its timer."""
        if not session_id:
            return {"ok": False, "error": "session_id_required"}
        timer_id = self._simulation_timers.get(session_id, "")
        return {
            "ok": True,
            "session_id": session_id,
            "running": bool(timer_id),
            "timer_id": timer_id,
        }

    async def _tool_import_card(
        self,
        payload: dict[str, Any] | None = None,
        png_b64: str = "",
        filename: str = "",
        content_type: str = "",
        color: str = "",
        age_stage: str = "adult",
    ) -> dict[str, Any]:
        """Import a Silly-Tavern character card.

        Three input forms — exactly one must be supplied:

        * ``payload`` — JSON dict (the v1 flat or v2 ``{spec,data}`` shape).
        * ``png_b64`` — base64-encoded PNG bytes carrying a ``chara``
          tEXt chunk (Silly-Tavern PNG card). The image itself is
          written to the avatar directory and bound to the new character.
        * (REST upload) — handled by the gateway endpoint, not here.
        """
        if self._store is None:
            return {"ok": False, "error": "store_not_ready"}
        # Resolve the avatar directory from the gateway's well-known
        # constant so PNG cards get their picture rendered next to
        # uploaded avatars. Lazy import avoids a hard dep cycle in
        # plugin-only tests.
        from pathlib import Path
        avatar_dir = Path("data/plugins/character_chat/avatars")

        try:
            if png_b64:
                import base64
                try:
                    raw = base64.b64decode(png_b64, validate=False)
                except Exception as exc:  # noqa: BLE001
                    return {
                        "ok": False,
                        "error": f"png_b64 not valid base64: {exc}",
                    }
                saved = await self._store.import_silly_tavern_bytes(
                    raw,
                    filename=filename or "card.png",
                    content_type=content_type or "image/png",
                    color=color or None,
                    age_stage=age_stage,
                    avatar_dir=avatar_dir,
                )
            elif payload is not None:
                saved = await self._store.import_silly_tavern(
                    payload, color=color or None, age_stage=age_stage,
                )
            else:
                return {
                    "ok": False,
                    "error": "either 'payload' (JSON dict) or "
                             "'png_b64' (base64 PNG) is required",
                }
        except CharacterCardError as exc:
            return {"ok": False, "error": str(exc)}
        await self._broadcast_character_event("character_created", saved)
        return {"ok": True, "character": _card_to_public(saved)}

    async def _tool_run_round(
        self,
        session_id: str,
        user_message: str = "",
        pulse_from_id: str = "",
        pulse_text: str = "",
        scene: str = "",
    ) -> dict[str, Any]:
        if self._store is None or self._orchestrator is None:
            return {"ok": False, "error": "plugin_not_ready"}

        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            characters = await self._store.list_in_session(session_id)
            if not characters:
                return {"ok": False, "error": "no_characters_in_session"}

            history = self._load_session_history(session_id)

            # Pulse-mention propagation: if the pulse text addresses another
            # character by name (NL detection — same parser used for user
            # messages), push them into ``extra_forced`` so they answer in
            # the SAME round as the pulse. Without this they'd only see the
            # pulse as background context and stay silent.
            extra_forced: list[str] = []
            if pulse_text and pulse_from_id:
                extra_forced = parse_nl_mentions(pulse_text, characters)
                # Strip self-mentions defensively.
                extra_forced = [
                    cid for cid in extra_forced if cid != pulse_from_id
                ]
                if extra_forced:
                    log.info(
                        "character_chat.pulse_mention_propagated from=%s to=%s",
                        pulse_from_id,
                        extra_forced,
                    )

            # Pre-resolve lorebook activations per speaker. We do this
            # here (in the plugin, which owns the LorebookStore) so the
            # orchestrator stays DB-agnostic. The engine is cheap — a
            # cache-friendly substring scan plus list assembly — so
            # running it once per character per round is fine.
            lore_by_speaker = await self._resolve_lore_per_speaker(
                characters=characters,
                session_id=session_id,
                history=history,
                user_message=user_message,
                pulse_text=pulse_text,
            )

            # Phase 13: pull each speaker's live state from the RP
            # container so the orchestrator's prompt builder sees the
            # CURRENT session-state, not the stale character.state.
            live_state_by_char: dict[str, dict[str, str]] = {}
            try:
                container = await self._get_rp_container(session_id)
                if container is not None:
                    for c in characters:
                        st = await container.get_char_state(c.id)
                        if st:
                            live_state_by_char[c.id] = st
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.rp_state_load_failed",
                    session_id=session_id,
                    error=str(exc),
                )

            # Phase 13.2: build the skip-cooldown exclusion set + tick
            # any existing cooldowns down by 1.
            excluded = self._tick_skip_cooldowns(session_id)

            # Phase 13.5 (B+D): pull each char's last 5 own turns so the
            # repetition guard catches self-repetition across rounds. RP
            # turns live in the per-session container; non-RP fall back
            # to the legacy character_turns table.
            prior_turns_by_char = await self._load_prior_turns_per_char(
                session_id=session_id,
                characters=characters,
                limit=5,
            )

            req = GroupTurnRequest(
                session_id=session_id,
                history=history,
                characters=characters,
                user_message=user_message,
                pulse_from_id=pulse_from_id,
                pulse_text=pulse_text,
                scene=scene,
                extra_forced=extra_forced,
                lore_by_speaker=lore_by_speaker,
                live_state_by_char=live_state_by_char,
                excluded_speaker_ids=excluded,
                prior_turns_by_char=prior_turns_by_char,
            )

            round_id = uuid.uuid4().hex[:12]
            trigger_kind, trigger_text = _describe_trigger(
                user_message, pulse_from_id, pulse_text
            )

            await self.api.ws_broadcast(
                {
                    "type": "character_round_start",
                    "session_id": session_id,
                    "round_id": round_id,
                    "trigger_kind": trigger_kind,
                    "trigger_text": trigger_text,
                }
            )

            result = await self._orchestrator.run_round(req)

            # Phase 13.2: anyone who skipped this round earns a 1-round
            # cooldown so the LLM-orchestrator doesn't keep picking
            # them silent again next round.
            self._record_skip_cooldowns(session_id, result.turns)

            await self._persist_and_broadcast_turns(
                session_id=session_id,
                round_id=round_id,
                trigger_kind=trigger_kind,
                trigger_text=trigger_text,
                turns=result.turns,
            )

            await self.api.ws_broadcast(
                {
                    "type": "character_round_done",
                    "session_id": session_id,
                    "round_id": round_id,
                    "turns": [_turn_to_public(t) for t in result.turns],
                }
            )

            return {
                "ok": True,
                "round_id": round_id,
                "speakers": result.speaker_order,
                "turns": [_turn_to_public(t) for t in result.turns],
            }

    # ─── WebSocket wrappers ──────────────────────────────────────────────
    # WS handlers receive ``(client, message)`` directly and must send their
    # own response via ``client.send_json``. The inner ``_tool_*`` methods
    # return plain dicts so they stay usable as LLM tools too.

    async def _ws_list_characters(self, client: Any, message: dict[str, Any]) -> None:
        result = await self._tool_list_characters(
            include_archived=bool(message.get("include_archived", False)),
            session_id=str(message.get("session_id", "") or ""),
        )
        await client.send_json({"type": "character_list", **result})

    async def _ws_create_character(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        payload = {k: v for k, v in message.items() if k != "type"}
        result = await self._tool_spawn_character(**payload)
        await client.send_json({"type": "character_created", **result})

    async def _ws_update_character(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        char_id = str(message.get("id", "") or "")
        if not char_id:
            await client.send_json(
                {"type": "character_updated", "ok": False, "error": "id_required"}
            )
            return
        patch = {k: v for k, v in message.items() if k not in ("type", "id")}
        result = await self._tool_update_character(id=char_id, **patch)
        await client.send_json({"type": "character_updated", **result})

    async def _ws_archive_character(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        char_id = str(message.get("id", "") or "")
        if not char_id:
            await client.send_json(
                {"type": "character_updated", "ok": False, "error": "id_required"}
            )
            return
        result = await self._tool_archive_character(id=char_id)
        # Archive re-uses character_updated so the UI's cache invalidation
        # is single-path (it refetches the list on that event).
        await client.send_json({"type": "character_updated", "id": char_id, **result})

    async def _ws_unarchive_character(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        char_id = str(message.get("id", "") or "")
        if not char_id:
            await client.send_json(
                {"type": "character_updated", "ok": False, "error": "id_required"}
            )
            return
        result = await self._tool_unarchive_character(id=char_id)
        await client.send_json({"type": "character_updated", "id": char_id, **result})

    async def _ws_delete_character(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        char_id = str(message.get("id", "") or "")
        if self._store is None or not char_id:
            await client.send_json(
                {"type": "character_deleted", "ok": False, "error": "id_required"}
            )
            return
        ok = await self._store.delete(char_id)
        if ok:
            await self.api.ws_broadcast(
                {"type": "character_deleted", "id": char_id}
            )
        else:
            await client.send_json(
                {"type": "character_deleted", "ok": False, "id": char_id}
            )

    async def _ws_attach_character(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        result = await self._tool_attach_character(
            id=str(message.get("id", "") or ""),
            session_id=str(message.get("session_id", "") or ""),
        )
        await client.send_json({"type": "character_updated", **result})

    async def _ws_detach_character(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        result = await self._tool_detach_character(
            id=str(message.get("id", "") or ""),
            session_id=str(message.get("session_id", "") or ""),
        )
        await client.send_json({"type": "character_updated", **result})

    async def _ws_import_card(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        payload = message.get("payload")
        if not isinstance(payload, dict):
            await client.send_json(
                {
                    "type": "character_created",
                    "ok": False,
                    "error": "payload_required",
                }
            )
            return
        result = await self._tool_import_card(
            payload=payload,
            color=str(message.get("color", "") or ""),
            age_stage=str(message.get("age_stage", "adult") or "adult"),
        )
        await client.send_json({"type": "character_created", **result})

    async def _ws_run_round(self, client: Any, message: dict[str, Any]) -> None:
        result = await self._tool_run_round(
            session_id=str(message.get("session_id", "") or ""),
            user_message=str(message.get("user_message", "") or ""),
            pulse_from_id=str(message.get("pulse_from_id", "") or ""),
            pulse_text=str(message.get("pulse_text", "") or ""),
            scene=str(message.get("scene", "") or ""),
        )
        await client.send_json({"type": "character_round_result", **result})

    async def _ws_session_get(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        session_id = str(message.get("session_id", "") or "")
        if not session_id:
            await client.send_json(
                {
                    "type": "character_session_get",
                    "ok": False,
                    "error": "session_id_required",
                }
            )
            return
        state = await self._get_session_state(session_id)
        await client.send_json(
            {
                "type": "character_session_get",
                "ok": True,
                "session_id": session_id,
                **state,
            }
        )

    async def _ws_session_set(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        # Support both the new ``mode`` int (0/1/2) and the legacy
        # ``enabled`` bool for backward compatibility.
        mode_raw = message.get("mode")
        if mode_raw is not None:
            mode_val = int(mode_raw)
        elif "enabled" in message:
            mode_val = 1 if message["enabled"] else 0
        elif "character_mode" in message:
            mode_val = int(message["character_mode"])
        else:
            mode_val = 0

        result = await self._tool_set_character_mode(
            session_id=str(message.get("session_id", "") or ""),
            mode=mode_val,
            scene=str(message.get("scene", "") or ""),
        )
        # The underlying _set_session_state already broadcasts
        # ``character_session_mode`` to every client. Reply to the caller
        # with an ack so it can resolve its promise if needed.
        await client.send_json({"type": "character_session_set", **result})

    async def _ws_simulation_start(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        session_id = str(message.get("session_id", "") or "")
        interval = message.get("interval_minutes")
        try:
            interval_int = int(interval) if interval is not None else None
        except (TypeError, ValueError):
            interval_int = None
        result = await self._tool_start_simulation(
            session_id=session_id, interval_minutes=interval_int
        )
        await client.send_json({"type": "simulation_started", **result})

    async def _ws_simulation_stop(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        session_id = str(message.get("session_id", "") or "")
        result = await self._tool_stop_simulation(session_id=session_id)
        await client.send_json({"type": "simulation_stopped", **result})

    async def _ws_simulation_status_get(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        session_id = str(message.get("session_id", "") or "")
        result = await self._tool_simulation_status(session_id=session_id)
        await client.send_json({"type": "simulation_status", **result})

    async def _ws_history(self, client: Any, message: dict[str, Any]) -> None:
        session_id = str(message.get("session_id", "") or "")
        limit = int(message.get("limit", 50))
        if not session_id:
            await client.send_json(
                {
                    "type": "character_history",
                    "ok": False,
                    "error": "session_id_required",
                }
            )
            return
        db = await self.api.get_db()
        cursor = await db.execute(
            "SELECT id, character_id, character_name, round_id, order_num, "
            "content, skipped, trigger_kind, trigger_text, created_at "
            "FROM character_turns WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, max(1, min(500, limit))),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        turns = [
            {
                "id": r[0],
                "character_id": r[1],
                "character_name": r[2],
                "round_id": r[3],
                "order": int(r[4] or 0),
                "content": r[5],
                "skipped": bool(r[6]),
                "trigger_kind": r[7],
                "trigger_text": r[8],
                "created_at": float(r[9]),
            }
            for r in rows
        ]
        turns.reverse()  # chronological for UI
        await client.send_json(
            {
                "type": "character_history",
                "ok": True,
                "session_id": session_id,
                "turns": turns,
            }
        )

    # ─── Per-turn edit / delete / regenerate ────────────────────────────
    #
    # Mike's audit point #3: at parity with Lexy's normal-chat bubbles,
    # character bubbles must support edit / delete / regenerate. All
    # three round-trip through these WS handlers; the frontend's
    # `appendCharacterTurn` action bar dispatches them.

    async def _fetch_turn_row(
        self, turn_id: str
    ) -> dict[str, Any] | None:
        """Read a single character_turns row by id.

        Phase 13: a turn for an RP session lives in that session's
        container, not in the global ``character_turns`` table. We
        try the legacy table first (for non-RP sessions), then fall
        back to scanning RP containers. Each container only opens
        its SQLite handle on demand so this stays cheap.
        """
        if not turn_id:
            return None
        # Legacy path
        db = await self.api.get_db()
        async with db.execute(
            "SELECT id, session_id, character_id, character_name, round_id, "
            "order_num, content, skipped, trigger_kind, trigger_text, "
            "created_at FROM character_turns WHERE id = ?",
            (turn_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            return {
                "id": row[0],
                "session_id": row[1],
                "character_id": row[2],
                "character_name": row[3],
                "round_id": row[4],
                "order": int(row[5] or 0),
                "content": row[6],
                "skipped": bool(row[7]),
                "trigger_kind": row[8],
                "trigger_text": row[9],
                "created_at": float(row[10]),
                "_in_rp_container": False,
            }
        # RP fallback: scan every known RP container.
        if self._rp_registry is None:
            return None
        try:
            session_ids = await self._rp_registry.list_session_ids()
        except Exception:  # noqa: BLE001
            return None
        for sid in session_ids:
            container = await self._rp_registry.get(sid)
            if container is None:
                continue
            t = await container.get_turn(turn_id)
            if t is not None:
                return {
                    "id": t.id,
                    "session_id": sid,
                    "character_id": t.character_id,
                    "character_name": t.character_name,
                    "round_id": t.round_id,
                    "order": t.order_num,
                    "content": t.content,
                    "skipped": t.skipped,
                    "trigger_kind": t.trigger_kind,
                    "trigger_text": t.trigger_text,
                    "created_at": t.created_at,
                    "_in_rp_container": True,
                }
        return None

    async def _ws_turn_edit(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Replace a turn's text. Used by the inline edit form."""
        turn_id = str(message.get("turn_id") or "")
        new_content = str(message.get("content") or "")
        if not turn_id or not new_content.strip():
            await client.send_json(
                {"type": "character_turn_edit_ack", "ok": False,
                 "error": "turn_id and non-empty content required"}
            )
            return
        existing = await self._fetch_turn_row(turn_id)
        if existing is None:
            await client.send_json(
                {"type": "character_turn_edit_ack", "ok": False,
                 "error": "turn not found"}
            )
            return
        if existing.get("_in_rp_container"):
            container = await self._get_rp_container(existing["session_id"])
            if container is not None:
                await container.update_turn_content(turn_id, new_content)
        else:
            db = await self.api.get_db()
            await db.execute(
                "UPDATE character_turns SET content = ?, skipped = 0 "
                "WHERE id = ?",
                (new_content, turn_id),
            )
            await db.commit()
        # Broadcast so all open tabs render the new text.
        await self.api.ws_broadcast(
            {
                "type": "character_turn_updated",
                "session_id": existing["session_id"],
                "round_id": existing["round_id"],
                "turn_id": turn_id,
                "character_id": existing["character_id"],
                "character_name": existing["character_name"],
                "content": new_content,
                "skipped": False,
            }
        )
        await client.send_json(
            {"type": "character_turn_edit_ack", "ok": True, "turn_id": turn_id}
        )

    async def _ws_turn_delete(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Remove a single character_turn row + broadcast deletion."""
        turn_id = str(message.get("turn_id") or "")
        if not turn_id:
            await client.send_json(
                {"type": "character_turn_delete_ack", "ok": False,
                 "error": "turn_id required"}
            )
            return
        existing = await self._fetch_turn_row(turn_id)
        if existing is None:
            await client.send_json(
                {"type": "character_turn_delete_ack", "ok": False,
                 "error": "turn not found"}
            )
            return
        if existing.get("_in_rp_container"):
            container = await self._get_rp_container(existing["session_id"])
            if container is not None:
                await container.delete_turn(turn_id)
        else:
            db = await self.api.get_db()
            await db.execute(
                "DELETE FROM character_turns WHERE id = ?", (turn_id,),
            )
            await db.commit()
        await self.api.ws_broadcast(
            {
                "type": "character_turn_deleted",
                "session_id": existing["session_id"],
                "round_id": existing["round_id"],
                "turn_id": turn_id,
            }
        )
        await client.send_json(
            {"type": "character_turn_delete_ack", "ok": True, "turn_id": turn_id}
        )

    async def _ws_turn_regenerate(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Drop a turn and re-run JUST that single character's response.

        Strategy: load the deleted turn's round context (other turns of
        the same round that came BEFORE it), then call
        ``_run_single_turn`` on the character with the round's original
        trigger. Result lands as a new ``character_turn`` broadcast +
        DB row, and the old turn is gone — UI replaces accordingly.
        """
        turn_id = str(message.get("turn_id") or "")
        if not turn_id or self._store is None or self._orchestrator is None:
            await client.send_json(
                {"type": "character_turn_regenerate_ack", "ok": False,
                 "error": "turn_id required / plugin not ready"}
            )
            return
        existing = await self._fetch_turn_row(turn_id)
        if existing is None:
            await client.send_json(
                {"type": "character_turn_regenerate_ack", "ok": False,
                 "error": "turn not found"}
            )
            return

        session_id = existing["session_id"]
        round_id = existing["round_id"]
        char_id = existing["character_id"]

        # Load the same character's card + the other speakers in the round
        # so the prompt builder can render "Reaktionen dieser Runde".
        card = await self._store.get(char_id)
        if card is None:
            await client.send_json(
                {"type": "character_turn_regenerate_ack", "ok": False,
                 "error": "character no longer exists"}
            )
            return

        # Phase 13: load siblings + delete old turn from the right
        # store. RP sessions live in their container; non-RP keep
        # using the legacy global ``character_turns`` table.
        in_container = bool(existing.get("_in_rp_container"))
        if in_container:
            container = await self._get_rp_container(session_id)
            if container is None:
                await client.send_json(
                    {"type": "character_turn_regenerate_ack", "ok": False,
                     "error": "rp container not available"}
                )
                return
            sibling_rows = await container.list_turns_for_round(round_id)
            prior_turns = [
                CharacterTurn(
                    character_id=tr.character_id,
                    character_name=tr.character_name,
                    content=tr.content,
                    skipped=tr.skipped,
                    order=tr.order_num,
                )
                for tr in sibling_rows
                if tr.id != turn_id and tr.order_num < int(existing["order"] or 0)
            ]
            await container.delete_turn(turn_id)
        else:
            db = await self.api.get_db()
            async with db.execute(
                "SELECT id, character_id, character_name, content, skipped, "
                "order_num FROM character_turns "
                "WHERE round_id = ? AND id != ? "
                "ORDER BY order_num ASC",
                (round_id, turn_id),
            ) as cur:
                siblings = list(await cur.fetchall())
            prior_turns = [
                CharacterTurn(
                    character_id=row[1], character_name=row[2],
                    content=row[3] or "", skipped=bool(row[4]),
                    order=int(row[5] or 0),
                )
                for row in siblings
                if int(row[5] or 0) < int(existing["order"] or 0)
            ]
            # Drop the old row + tell the UI it's gone before we generate
            # the replacement (so tabs show a brief "regenerating…" state).
            await db.execute(
                "DELETE FROM character_turns WHERE id = ?", (turn_id,),
            )
            await db.commit()
        await self.api.ws_broadcast(
            {
                "type": "character_turn_deleted",
                "session_id": session_id, "round_id": round_id,
                "turn_id": turn_id,
            }
        )

        # Reconstruct a minimal GroupTurnRequest with the original trigger
        # so the prompt builder picks the same "Impuls" / "User"-Variante.
        all_chars = await self._store.list_in_session(session_id)
        history = self._load_session_history(session_id)
        trigger_kind = (existing.get("trigger_kind") or "").lower()
        trigger_text = existing.get("trigger_text") or ""
        user_msg = ""
        pulse_from = ""
        pulse_text = ""
        if trigger_kind == "user":
            user_msg = trigger_text
        elif trigger_kind == "pulse":
            # trigger_text is "[char_id] *pulse text*" per _describe_trigger
            if trigger_text.startswith("["):
                close = trigger_text.find("]")
                if close > 0:
                    pulse_from = trigger_text[1:close].strip()
                    pulse_text = trigger_text[close + 1 :].strip()
        sess_state = await self._get_session_state(session_id)
        scene = str(sess_state.get("scene") or "")
        # Phase 13: same live_state injection as run_round.
        live_state_by_char: dict[str, dict[str, str]] = {}
        try:
            container = await self._get_rp_container(session_id)
            if container is not None:
                for c in all_chars:
                    st = await container.get_char_state(c.id)
                    if st:
                        live_state_by_char[c.id] = st
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.rp_state_load_failed_regen",
                session_id=session_id, error=str(exc),
            )
        req = GroupTurnRequest(
            session_id=session_id,
            history=history,
            characters=all_chars,
            user_message=user_msg,
            pulse_from_id=pulse_from,
            pulse_text=pulse_text,
            scene=scene,
            live_state_by_char=live_state_by_char,
        )
        try:
            new_turn = await self._orchestrator._run_single_turn(
                card=card,
                order=int(existing["order"] or 0),
                previous_turns=prior_turns,
                req=req,
                all_cards=all_chars,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character_chat.regenerate_failed turn_id=%s error=%s",
                turn_id, exc,
            )
            await client.send_json(
                {"type": "character_turn_regenerate_ack", "ok": False,
                 "error": f"regeneration failed: {exc}"}
            )
            return

        # Persist + broadcast as a normal turn so UI re-renders the bubble.
        await self._persist_and_broadcast_turns(
            session_id=session_id,
            round_id=round_id,
            trigger_kind=existing.get("trigger_kind") or "user",
            trigger_text=trigger_text,
            turns=[new_turn],
        )
        await client.send_json(
            {"type": "character_turn_regenerate_ack", "ok": True,
             "turn_id": turn_id, "session_id": session_id}
        )

    # ─── Lorebook tools (Phase 9.8) ─────────────────────────────────────

    def _lore(self) -> LorebookStore:
        if self._lore_store is None:
            raise RuntimeError("lorebook store not initialised")
        return self._lore_store

    async def _tool_lorebook_create(
        self,
        name: str,
        description: str = "",
        scope: str = "global",
        scope_id: str = "",
        token_budget: int = 1500,
    ) -> dict[str, Any]:
        try:
            book = await self._lore().create_lorebook(
                name=name, description=description, scope=scope,
                scope_id=scope_id, token_budget=int(token_budget),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        await self.api.ws_broadcast(
            {"type": "lorebook_created", "book": book.to_public()}
        )
        return {"ok": True, "book": book.to_public()}

    async def _tool_lorebook_list(
        self,
        scope: str | None = None,
        scope_id: str | None = None,
        enabled_only: bool = False,
    ) -> dict[str, Any]:
        try:
            books = await self._lore().list_lorebooks(
                scope=scope, scope_id=scope_id,
                enabled_only=bool(enabled_only),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "books": [b.to_public() for b in books]}

    async def _tool_lorebook_update(
        self, id: str, **patch: Any  # noqa: A002
    ) -> dict[str, Any]:
        try:
            book = await self._lore().update_lorebook(id, **patch)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if book is None:
            return {"ok": False, "error": "lorebook not found"}
        await self.api.ws_broadcast(
            {"type": "lorebook_updated", "book": book.to_public()}
        )
        return {"ok": True, "book": book.to_public()}

    async def _tool_lorebook_delete(self, id: str) -> dict[str, Any]:  # noqa: A002
        ok = await self._lore().delete_lorebook(id)
        if ok:
            await self.api.ws_broadcast(
                {"type": "lorebook_deleted", "id": id}
            )
        return {"ok": ok}

    async def _tool_lore_entry_create(
        self,
        lorebook_id: str,
        name: str,
        keys: list[str] | None = None,
        content: str = "",
        position: str = "before_scenario",
        priority: int = 100,
        always_on: bool = False,
        scan_depth: int = 4,
    ) -> dict[str, Any]:
        try:
            entry = await self._lore().create_entry(
                lorebook_id=lorebook_id, name=name,
                keys=list(keys or []), content=content,
                position=position, priority=int(priority),
                always_on=bool(always_on),
                scan_depth=int(scan_depth),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        await self.api.ws_broadcast(
            {"type": "lore_entry_created", "entry": entry.to_public()}
        )
        return {"ok": True, "entry": entry.to_public()}

    async def _tool_lore_entry_list(
        self,
        lorebook_id: str,
        enabled_only: bool = False,
    ) -> dict[str, Any]:
        try:
            entries = await self._lore().list_entries(
                lorebook_id=lorebook_id,
                enabled_only=bool(enabled_only),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "entries": [e.to_public() for e in entries],
        }

    async def _tool_lore_entry_update(
        self, id: str, **patch: Any  # noqa: A002
    ) -> dict[str, Any]:
        try:
            entry = await self._lore().update_entry(id, **patch)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if entry is None:
            return {"ok": False, "error": "lore entry not found"}
        await self.api.ws_broadcast(
            {"type": "lore_entry_updated", "entry": entry.to_public()}
        )
        return {"ok": True, "entry": entry.to_public()}

    async def _tool_lore_entry_delete(self, id: str) -> dict[str, Any]:  # noqa: A002
        ok = await self._lore().delete_entry(id)
        if ok:
            await self.api.ws_broadcast(
                {"type": "lore_entry_deleted", "id": id}
            )
        return {"ok": ok}

    # ─── Lorebook WS handlers ──────────────────────────────────────────

    async def _ws_lorebook_list(self, client: Any, message: dict[str, Any]) -> None:
        result = await self._tool_lorebook_list(
            scope=message.get("scope"),
            scope_id=message.get("scope_id"),
            enabled_only=bool(message.get("enabled_only", False)),
        )
        await client.send_json({"type": "lorebook_list", **result})

    async def _ws_lorebook_create(self, client: Any, message: dict[str, Any]) -> None:
        payload = {k: v for k, v in message.items() if k != "type"}
        result = await self._tool_lorebook_create(**payload)
        await client.send_json({"type": "lorebook_create_ack", **result})

    async def _ws_lorebook_update(self, client: Any, message: dict[str, Any]) -> None:
        book_id = str(message.get("id") or "")
        if not book_id:
            await client.send_json(
                {"type": "lorebook_update_ack", "ok": False, "error": "id required"}
            )
            return
        patch = {k: v for k, v in message.items() if k not in ("type", "id")}
        result = await self._tool_lorebook_update(id=book_id, **patch)
        await client.send_json({"type": "lorebook_update_ack", **result})

    async def _ws_lorebook_delete(self, client: Any, message: dict[str, Any]) -> None:
        book_id = str(message.get("id") or "")
        result = await self._tool_lorebook_delete(id=book_id)
        await client.send_json({"type": "lorebook_delete_ack", **result})

    async def _ws_lore_entry_list(self, client: Any, message: dict[str, Any]) -> None:
        result = await self._tool_lore_entry_list(
            lorebook_id=str(message.get("lorebook_id") or ""),
            enabled_only=bool(message.get("enabled_only", False)),
        )
        await client.send_json({"type": "lore_entry_list", **result})

    async def _ws_lore_entry_create(self, client: Any, message: dict[str, Any]) -> None:
        payload = {k: v for k, v in message.items() if k != "type"}
        result = await self._tool_lore_entry_create(**payload)
        await client.send_json({"type": "lore_entry_create_ack", **result})

    async def _ws_lore_entry_update(self, client: Any, message: dict[str, Any]) -> None:
        entry_id = str(message.get("id") or "")
        if not entry_id:
            await client.send_json(
                {"type": "lore_entry_update_ack", "ok": False, "error": "id required"}
            )
            return
        patch = {k: v for k, v in message.items() if k not in ("type", "id")}
        result = await self._tool_lore_entry_update(id=entry_id, **patch)
        await client.send_json({"type": "lore_entry_update_ack", **result})

    async def _ws_lore_entry_delete(self, client: Any, message: dict[str, Any]) -> None:
        entry_id = str(message.get("id") or "")
        result = await self._tool_lore_entry_delete(id=entry_id)
        await client.send_json({"type": "lore_entry_delete_ack", **result})

    # ─── Persistence of turns ────────────────────────────────────────────

    async def _persist_and_broadcast_turns(
        self,
        *,
        session_id: str,
        round_id: str,
        trigger_kind: str,
        trigger_text: str,
        turns: list[CharacterTurn],
    ) -> None:
        # Phase 13: when this is an RP session, we route ALL writes
        # (turns + state + memory) through the per-session container.
        # Non-RP sessions use the legacy global tables and global
        # ``context`` collection — unchanged.
        container: RPSessionContainer | None = None
        try:
            container = await self._get_rp_container(session_id)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.rp_container_lookup_failed",
                session_id=session_id,
                error=str(exc),
            )

        db = await self.api.get_db()
        now = time.time()
        for t in turns:
            turn_id = uuid.uuid4().hex[:12]

            # Strip any <state>...</state> block from the visible content
            # and apply the parsed state diff to the right scope.
            if not t.skipped and t.content:
                cleaned, state_updates = parse_state_block(t.content)
                if cleaned != t.content:
                    t.content = cleaned
                if state_updates:
                    try:
                        if container is not None:
                            # RP path: state lives in the session, not
                            # on the character. tracked_stats filtering
                            # happens inside update_char_state.
                            await container.update_char_state(
                                t.character_id, state_updates,
                            )
                            log.info(
                                "character_chat.state_updated_rp "
                                "character=%s updates=%s",
                                t.character_name,
                                state_updates,
                            )
                        else:
                            # Phase 13: non-RP sessions DROP state
                            # updates from LLM ``<state>`` blocks. The
                            # old behaviour wrote them to the global
                            # ``characters.state`` column, which
                            # contaminated the character's defaults
                            # for OTHER RP sessions (Mike's whole bug
                            # report). State now belongs strictly to
                            # the session that produced it.
                            log.debug(
                                "character_chat.state_update_dropped "
                                "(non-rp session) character=%s updates=%s",
                                t.character_name,
                                state_updates,
                            )
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "character_chat.state_update_failed "
                            "character=%s error=%s",
                            t.character_name,
                            str(exc),
                        )

            # Persist the turn. RP → container.turns.db.
            #                Non-RP → legacy global character_turns.
            if container is not None:
                try:
                    await container.append_turn(TurnRow(
                        id=turn_id,
                        character_id=t.character_id,
                        character_name=t.character_name,
                        round_id=round_id,
                        order_num=t.order,
                        content=t.content,
                        skipped=t.skipped,
                        trigger_kind=trigger_kind,
                        trigger_text=trigger_text,
                        created_at=now,
                    ))
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "character_chat.rp_turn_persist_failed",
                        session_id=session_id,
                        character=t.character_name,
                        error=str(exc),
                    )
            else:
                await db.execute(
                    "INSERT INTO character_turns (id, session_id, character_id, "
                    "character_name, round_id, order_num, content, skipped, "
                    "trigger_kind, trigger_text, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        turn_id,
                        session_id,
                        t.character_id,
                        t.character_name,
                        round_id,
                        t.order,
                        t.content,
                        1 if t.skipped else 0,
                        trigger_kind,
                        trigger_text,
                        now,
                    ),
                )

            # Broadcast per turn so the UI can render them streaming-style.
            await self.api.ws_broadcast(
                {
                    "type": "character_turn",
                    "session_id": session_id,
                    "round_id": round_id,
                    "turn_id": turn_id,
                    **_turn_to_public(t),
                }
            )

            # Per-character voice: if the card has a voice name set, TTS-
            # synthesise the turn text and broadcast the audio as a follow-
            # up message. Runs as a background task so it doesn't block the
            # remaining speakers in this round.
            if not t.skipped and t.content:
                asyncio.create_task(
                    self._speak_character_turn(
                        session_id=session_id,
                        round_id=round_id,
                        turn_id=turn_id,
                        character_id=t.character_id,
                        text=t.content,
                    ),
                    name=f"character_chat.tts.{turn_id}",
                )

            # Memory write — RP routes to the per-session collection,
            # non-RP uses the global ``context`` collection.
            if self._memory_strict_isolation and not t.skipped and t.content:
                try:
                    if container is not None:
                        await container.memory_write(
                            text=t.content,
                            character_id=t.character_id,
                            metadata={
                                "character_name": t.character_name,
                                "round_id": round_id,
                                "trigger_kind": trigger_kind,
                            },
                        )
                    else:
                        await self.api.memory_store(
                            text=t.content,
                            collection="context",
                            metadata={
                                "source": "character_chat",
                                "character_id": t.character_id,
                                "character_name": t.character_name,
                                "session_id": session_id,
                                "round_id": round_id,
                                "trigger_kind": trigger_kind,
                            },
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "character_chat.memory_store_failed",
                        character=t.character_name,
                        error=str(exc),
                    )
        if container is None:
            await db.commit()

    async def _speak_character_turn(
        self,
        *,
        session_id: str,
        round_id: str,
        turn_id: str,
        character_id: str,
        text: str,
    ) -> None:
        """Synthesise a character's turn text with their voice, broadcast audio.

        Skipped silently if:
        * the character has no ``voice`` configured (card.voice == "")
        * TTS isn't available (no provider loaded)
        * synthesis returns empty bytes

        Runs as a background task — failures are logged but don't affect
        the round's control flow.
        """
        if self._store is None:
            return
        try:
            card = await self._store.get(character_id)
        except Exception:  # noqa: BLE001
            return
        if card is None or not card.voice.strip():
            return

        try:
            audio = await self.api.tts_speak(text, voice=card.voice.strip())
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.tts_failed character=%s error=%s",
                card.name,
                exc,
            )
            return

        if not audio:
            return

        # Encode as base64 for inline delivery over the WS channel. The
        # frontend decodes and plays via HTMLAudioElement.
        import base64

        audio_b64 = base64.b64encode(audio).decode("ascii")
        await self.api.ws_broadcast(
            {
                "type": "character_turn_audio",
                "session_id": session_id,
                "round_id": round_id,
                "turn_id": turn_id,
                "character_id": character_id,
                "character_name": card.name,
                "voice": card.voice,
                "audio_b64": audio_b64,
                "mime": "audio/wav",
            }
        )

    async def _broadcast_character_event(
        self, event_type: str, card: CharacterCard
    ) -> None:
        await self.api.ws_broadcast(
            {
                "type": event_type,
                "character": _card_to_public(card),
            }
        )

    # ─── Scheduler: pulse timer management ───────────────────────────────

    async def _register_pulse_timer(
        self, card: CharacterCard, session_id: str
    ) -> str:
        """Register a recurring pulse with the scheduler plugin.

        Uses scheduler's ``set_recurring`` tool with
        ``action_type="character_pulse"`` and payload ``{character_id,
        session_id, pulse_text}``. Returns the scheduler's timer id (empty
        string if the scheduler is unavailable).

        Idempotent per (character_id, session_id): calling twice cancels
        the previous timer first.
        """
        if not self._proactive_pulses_enabled or not card.proactive_pulse_pattern:
            return ""
        scheduler = self.api.get_plugin("scheduler")
        if scheduler is None:
            log.warning("character_chat.scheduler_unavailable_for_pulse")
            return ""

        # Cancel any existing pulse for this pair first.
        await self._cancel_pulse_timer(card.id, session_id)

        pulse_text = card.proactive_pulse_prompt or _default_pulse(card.age_stage)

        try:
            result = await self.api.call_tool(
                "set_recurring",
                {
                    "label": f"character_pulse:{card.name}",
                    "pattern": card.proactive_pulse_pattern,
                    "action_type": "character_pulse",
                    "action_payload": {
                        "character_id": card.id,
                        "session_id": session_id,
                        "pulse_text": pulse_text,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character_chat.pulse_register_failed",
                character=card.name,
                error=str(exc),
            )
            return ""

        if not result.get("ok"):
            log.warning(
                "character_chat.pulse_register_rejected",
                character=card.name,
                error=result.get("error"),
            )
            return ""

        timer_id = str(result.get("data", {}).get("id", "") or "")
        if timer_id:
            self._pulse_timers[(card.id, session_id)] = timer_id
            log.info(
                "character_chat.pulse_registered",
                character=card.name,
                session_id=session_id,
                pattern=card.proactive_pulse_pattern,
                timer_id=timer_id,
            )
        return timer_id

    async def _cancel_pulse_timer(
        self, character_id: str, session_id: str
    ) -> bool:
        """Cancel the scheduler timer registered for this (character, session)."""
        key = (character_id, session_id)
        timer_id = self._pulse_timers.pop(key, None)
        if not timer_id:
            return False
        try:
            await self.api.call_tool("cancel_timer", {"id": timer_id})
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.pulse_cancel_failed",
                character_id=character_id,
                timer_id=timer_id,
                error=str(exc),
            )
            return False
        log.info(
            "character_chat.pulse_cancelled",
            character_id=character_id,
            session_id=session_id,
            timer_id=timer_id,
        )
        return True

    async def _cancel_all_pulse_timers(self) -> None:
        for (character_id, session_id) in list(self._pulse_timers.keys()):
            await self._cancel_pulse_timer(character_id, session_id)

    async def _rehydrate_pulse_timers(self) -> None:
        """Repopulate ``_pulse_timers`` from the scheduler DB and create
        missing timers for attached characters.

        Two-phase approach:
        1. **Rehydrate** — scan the scheduler DB for existing
           ``character_pulse`` timers and wire them back into the
           in-memory ``_pulse_timers`` dict. Cancel stale ones.
        2. **Ensure** — for every character currently attached to a
           session that has a pulse pattern but NO timer yet, register
           a new timer. This covers the case where timers were never
           created (code update, failed registration, etc.).
        """
        # Guard: only run once per plugin lifetime. Multiple on_enable calls
        # (hot-reload, re-connect) must not duplicate timers.
        if self._pulse_rehydrated:
            log.debug("character_chat.rehydrate_skipped already_done=True")
            return
        self._pulse_rehydrated = True

        # Small delay so the scheduler plugin's on_enable has a chance to
        # register its tools. This is a boot-time only cost.
        await asyncio.sleep(1.0)

        scheduler = self.api.get_plugin("scheduler")
        if scheduler is None:
            log.info("character_chat.rehydrate_skipped scheduler_absent=True")
            self._pulse_rehydrated = False  # allow retry on next enable
            return

        # ── Phase 1: Rehydrate existing timers from scheduler DB ─────
        try:
            result = await self.api.call_tool(
                "list_timers", {"include_inactive": False}
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("character_chat.rehydrate_list_failed error=%s", exc)
            result = {"ok": False}

        # Pre-9.6 bug: this code read ``data["items"]`` but the scheduler
        # tool returns ``data["timers"]``. As a result Phase 1 always saw
        # an empty list and Phase 2 happily registered a fresh timer for
        # every (char, session) pair on every restart — accumulating 18,
        # 45, 66 duplicates over time. We now read both keys defensively
        # so an older / customised scheduler still works.
        data: dict[str, Any] = (result.get("data") or {}) if result.get("ok") else {}
        items: list[dict[str, Any]] = list(
            data.get("timers")
            or data.get("items")
            or []
        )

        # Collect ALL character_pulse timers from the scheduler, keyed by
        # (character_id, session_id). If there are DUPLICATES for the same
        # pair (e.g. from repeated restarts), we keep only the newest and
        # cancel the rest — this prevents the "92 timers for 4 babies" bug.
        store = self._store
        by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}  # pair → [(timer_id, action_raw)]
        # Also rehydrate autonomous_sim timers keyed by session_id.
        sim_by_session: dict[str, list[str]] = {}
        for item in items:
            action_raw = item.get("action") or ""
            if not action_raw:
                continue
            try:
                action = json.loads(action_raw)
            except (TypeError, json.JSONDecodeError):
                continue

            action_kind = action.get("type")
            if action_kind == "autonomous_sim":
                sim_session = str(action.get("session_id") or "")
                sim_timer_id = str(item.get("id") or "")
                if sim_session and sim_timer_id:
                    sim_by_session.setdefault(sim_session, []).append(sim_timer_id)
                continue

            if action_kind != "character_pulse":
                continue

            # _encode_action flattens the payload into the action dict
            # (no nested "payload" key). Read directly from top level.
            character_id = str(action.get("character_id") or "")
            session_id = str(action.get("session_id") or "")
            timer_id = str(item.get("id") or "")
            if not character_id or not session_id or not timer_id:
                continue

            pair = (character_id, session_id)
            by_pair.setdefault(pair, []).append(timer_id)

        # Resolve known-alive sessions from the SessionStore so we can
        # cancel timers whose session is gone (Mike's "scheduler bloat"
        # came from stale session_ids from old experiments).
        known_sessions: set[str] = set()
        try:
            session_store = getattr(self.api._app, "session_store", None)
            if session_store is not None:
                known_sessions = set(session_store.sessions())
        except Exception:  # noqa: BLE001
            known_sessions = set()

        # Restore autonomous_sim timers — one per session, dedupe duplicates,
        # AND drop any timer whose session no longer exists.
        sim_restored = 0
        sim_cancelled = 0
        for sim_session, sim_ids in sim_by_session.items():
            session_alive = (
                not known_sessions or sim_session in known_sessions
            )
            if not session_alive:
                # Session is gone — cancel ALL sim timers for it.
                for dup in sim_ids:
                    try:
                        await self.api.call_tool("cancel_timer", {"id": dup})
                        sim_cancelled += 1
                    except Exception:  # noqa: BLE001
                        pass
                continue
            keeper = sim_ids[-1]
            for dup in sim_ids[:-1]:
                try:
                    await self.api.call_tool("cancel_timer", {"id": dup})
                    sim_cancelled += 1
                except Exception:  # noqa: BLE001
                    pass
            self._simulation_timers[sim_session] = keeper
            sim_restored += 1
        if sim_restored or sim_cancelled:
            log.info(
                "character_chat.sim_timers_rehydrated restored=%d cancelled=%d",
                sim_restored, sim_cancelled,
            )

        restored = 0
        stale: list[str] = []
        stale_reasons: dict[str, int] = {
            "dead_character": 0,
            "dead_session": 0,
            "duplicate": 0,
        }
        for pair, timer_ids in by_pair.items():
            character_id, session_id = pair
            # Check if the character is still alive.
            alive = False
            if store is not None:
                try:
                    card = await store.get(character_id)
                    alive = card is not None and not card.archived
                except Exception:  # noqa: BLE001
                    alive = False

            if not alive:
                # Cancel ALL timers for this dead character.
                stale.extend(timer_ids)
                stale_reasons["dead_character"] += len(timer_ids)
                continue

            # Cancel timers whose session no longer exists. We only
            # apply this when the SessionStore reported at least one
            # known session — otherwise a brand-new install would
            # cancel everything by mistake.
            if known_sessions and session_id not in known_sessions:
                stale.extend(timer_ids)
                stale_reasons["dead_session"] += len(timer_ids)
                continue

            # Keep the LAST timer, cancel all duplicates.
            keeper = timer_ids[-1]
            for dup in timer_ids[:-1]:
                stale.append(dup)
                stale_reasons["duplicate"] += 1
            self._pulse_timers[pair] = keeper
            restored += 1

        # Cancel stale + duplicate timers.
        for timer_id in stale:
            try:
                await self.api.call_tool("cancel_timer", {"id": timer_id})
            except Exception:  # noqa: BLE001
                pass

        log.info(
            "character_chat.pulse_timers_rehydrated restored=%d "
            "cancelled=%d (dead_char=%d dead_session=%d dup=%d)",
            restored,
            len(stale),
            stale_reasons["dead_character"],
            stale_reasons["dead_session"],
            stale_reasons["duplicate"],
        )

        # ── Phase 2: Create missing timers for attached characters ────
        # Characters can be attached to sessions and have a pulse pattern
        # in the DB, but no corresponding scheduler timer (e.g. the timer
        # was never created, or the scheduler was down, or a code update
        # added the pattern after the attach). We scan ALL non-archived
        # characters with a pattern and active_sessions, and register
        # timers for any pair that isn't in _pulse_timers yet.
        if store is None:
            return
        try:
            all_chars = await store.list(include_archived=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("character_chat.ensure_pulse_list_failed error=%s", exc)
            return

        created = 0
        pruned_sessions: list[tuple[str, str]] = []
        for card in all_chars:
            if not card.proactive_pulse_pattern:
                continue
            # Auto-prune ``active_sessions`` entries that reference
            # sessions which no longer exist. Otherwise Phase 2 would
            # keep registering fresh timers for dead sessions on every
            # restart — that's exactly the leak Mike saw (18 timers
            # for Lena across 2 sessions, multiplied by every previous
            # restart). A clean active_sessions list also prevents
            # similar leaks from any future feature that iterates over it.
            if known_sessions:
                live_sessions_for_card = [
                    s for s in card.active_sessions if s in known_sessions
                ]
                if len(live_sessions_for_card) != len(card.active_sessions):
                    dropped = [
                        s for s in card.active_sessions
                        if s not in known_sessions
                    ]
                    try:
                        await store.update(
                            card.id, active_sessions=live_sessions_for_card,
                        )
                        for s in dropped:
                            pruned_sessions.append((card.name, s))
                        card.active_sessions = live_sessions_for_card
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "character_chat.prune_active_sessions_failed "
                            "character=%s error=%s",
                            card.name, exc,
                        )
            for sid in card.active_sessions:
                if (card.id, sid) in self._pulse_timers:
                    continue  # already rehydrated from Phase 1
                # Skip dead sessions defensively (covered by the prune
                # above when ``known_sessions`` is populated, but the
                # prune is best-effort).
                if known_sessions and sid not in known_sessions:
                    continue
                try:
                    timer_id = await self._register_pulse_timer(card, sid)
                    if timer_id:
                        created += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "character_chat.ensure_pulse_register_failed "
                        "character=%s session=%s error=%s",
                        card.name,
                        sid,
                        exc,
                    )
        if created or pruned_sessions:
            log.info(
                "character_chat.pulse_timers_ensured created=%d "
                "pruned_sessions=%d",
                created, len(pruned_sessions),
            )
            if pruned_sessions:
                log.info(
                    "character_chat.pruned_stale_active_sessions: %s",
                    pruned_sessions[:20],
                )

    # ─── Helpers ─────────────────────────────────────────────────────────

    async def _resolve_lore_per_speaker(
        self,
        *,
        characters: list[CharacterCard],
        session_id: str,
        history: list[dict[str, Any]],
        user_message: str,
        pulse_text: str,
    ) -> dict[str, Any]:
        """Pre-compute :class:`ActivationResult` per character_id.

        Returns ``{char_id: ActivationResult}`` for every alive character
        in the round. The engine handles per-character filtering of
        scope=character books; we hand it the full visible list so a
        single DB read covers all speakers.
        """
        out: dict[str, Any] = {}
        if self._lore_store is None:
            return out
        # Pull all books visible to this session: global + session-scoped
        # for this session + character-scoped for ANY of the speakers.
        # We over-fetch slightly (more books than we'll filter to per
        # speaker) but that's still one DB round-trip total.
        try:
            all_books = await self._lore_store.list_lorebooks(enabled_only=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("character_chat.lore_list_failed error=%s", exc)
            return out
        if not all_books:
            return out

        speaker_ids = {c.id for c in characters}
        # Pre-filter to books we might use: global, character (in this round),
        # session (this session).
        candidate_books = [
            b for b in all_books
            if (
                b.scope == SCOPE_GLOBAL
                or (b.scope == SCOPE_CHARACTER and b.scope_id in speaker_ids)
                or (b.scope == SCOPE_SESSION and b.scope_id == session_id)
            )
        ]
        if not candidate_books:
            return out

        # Pre-fetch entries for every candidate book.
        entries_by_book: dict[str, list[Any]] = {}
        for book in candidate_books:
            try:
                entries_by_book[book.id] = await self._lore_store.list_entries(
                    lorebook_id=book.id, enabled_only=True,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.lore_entries_failed book=%s error=%s",
                    book.id, exc,
                )
                entries_by_book[book.id] = []

        for card in characters:
            try:
                result = self._lore_engine.activate(
                    speaker=card,
                    session_id=session_id,
                    history=history,
                    user_message=user_message,
                    pulse_text=pulse_text,
                    lorebooks=candidate_books,
                    entries=entries_by_book,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "character_chat.lore_activate_failed char=%s error=%s",
                    card.name, exc,
                )
                continue
            if result.by_position:
                out[card.id] = result
        return out

    def _is_session_stale(self, session_id: str) -> bool:
        """True iff the session hasn't been touched within the staleness
        window. Reads ``SessionStore.get_meta(session_id).updated_at``.

        A staleness threshold of 0 (config) disables the check entirely
        and this always returns False. Sessions the SessionStore doesn't
        know about (no meta) are treated as stale — there's no UI for
        them, so a pulse round would have no audience anyway.
        """
        if self._pulse_session_stale_seconds <= 0:
            return False
        try:
            session_store = getattr(self.api._app, "session_store", None)
            if session_store is None:
                return False  # without a store we can't decide → don't block
            meta = session_store.get_meta(session_id) or {}
        except Exception:  # noqa: BLE001
            return False
        updated_at = float(meta.get("updated_at") or 0.0)
        if updated_at <= 0:
            # Brand-new or unknown session — better to skip than to
            # spam pulses for something the user hasn't engaged with.
            return True
        return (time.time() - updated_at) > self._pulse_session_stale_seconds

    def _load_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Pull the last N messages from the core session store as history.

        Uses the proper PluginAPI method instead of reaching into
        ``api._app`` directly (which violates the encapsulation contract).
        """
        raw = self.api.get_session_history(session_id, limit=8)
        # Normalise to {"role", "name", "content"} for the orchestrator.
        out: list[dict[str, Any]] = []
        for m in raw or []:
            role = str(m.get("role", "user"))
            out.append(
                {
                    "role": role,
                    "name": "Mike" if role == "user" else "Lexy",
                    "content": str(m.get("content", "")),
                }
            )
        return out


# ─── Public serialisers ──────────────────────────────────────────────────────


def _card_to_public(card: CharacterCard) -> dict[str, Any]:
    """Return a JSON-safe view of a card for WS/REST responses."""
    return {
        "id": card.id,
        "name": card.name,
        "persona": card.persona,
        "greeting": card.greeting,
        "scenario": card.scenario,
        "example_dialog": card.example_dialog,
        "avatar": card.avatar,
        "color": card.color,
        "age_stage": card.age_stage,
        "voice": card.voice,
        "relationships": dict(card.relationships),
        "tags": list(card.tags),
        "active_sessions": list(card.active_sessions),
        "state": dict(card.state),
        "proactive_pulse_pattern": card.proactive_pulse_pattern,
        "proactive_pulse_prompt": card.proactive_pulse_prompt,
        "archived": card.archived,
        "created_at": card.created_at,
        "updated_at": card.updated_at,
    }


def _turn_to_public(turn: CharacterTurn) -> dict[str, Any]:
    return {
        "character_id": turn.character_id,
        "character_name": turn.character_name,
        "content": turn.content,
        "skipped": turn.skipped,
        "order": turn.order,
    }


def _describe_trigger(
    user_message: str, pulse_from_id: str, pulse_text: str
) -> tuple[str, str]:
    if pulse_from_id and pulse_text:
        return ("pulse", f"[{pulse_from_id}] {pulse_text}")
    if user_message:
        return ("user", user_message)
    return ("spontaneous", "")


_DEFAULT_PULSES: dict[str, str] = {
    "baby": "*schreit laut und sucht nach Mama*",
    "toddler": "*zieht an Mamas Ärmel* Mama, schau!",
    "child": "*platzt ins Zimmer* Ich hab was zu erzählen!",
    "teen": "*lehnt sich im Türrahmen an* ...darf ich was fragen?",
    "adult": "*sieht auf und sucht Blickkontakt*",
}

# Default proactive-pulse scheduler patterns per age stage. Young children
# are active more often (crying, hunger, diaper). Set automatically in
# spawn_character when the user doesn't specify a pattern explicitly.
_DEFAULT_PULSE_PATTERNS: dict[str, str] = {
    "baby": "every 30m",
    "toddler": "every 1h",
    "child": "every 2h",
}


def _default_pulse(age_stage: str) -> str:
    """Fallback pulse text if the card doesn't define one."""
    return _DEFAULT_PULSES.get(age_stage, "*bewegt sich*")
