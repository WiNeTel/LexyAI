"""
RP Director plugin.

A "Director" is a meta-character that conducts a structured setup
conversation with the user, then commits the result (scenario +
characters) into the existing ``character_chat`` plugin and switches the
session into character_mode.

Architecture
------------
The Director is **two things at once**:

1. **A persona** (``config/personas/director.yaml``) — identity, style,
   rules. Loaded once on enable and cached as a fully-assembled prompt.
2. **A plugin** (this file) — exposes 4 LLM tools, owns a SQLite state
   table per Director session, and registers a ``before_prompt_build``
   hook that *replaces* Lexy's persona for sessions in director-mode.

The session-aware hook is what makes the Director feel like a separate
character even though it shares Lexy's agent loop, tool registry, and
memory plumbing. When a session is not in director-mode the hook
no-ops and Lexy answers normally.

Tools
-----
- ``start_rp_setup`` — Lexy/the user enter director-mode.
- ``propose_scenario`` — Director records a scenario draft.
- ``propose_characters`` — Director records 1-N character drafts.
- ``commit_rp_setup`` — Director writes characters via
  ``spawn_character`` (character_chat) and toggles character_mode.
- ``cancel_rp_setup`` — abandon the setup, hand control back to Lexy.

WS handlers
-----------
- ``rp_director_start`` — Frontend slash command ``/rp [intent]``.
- ``rp_director_status`` — Frontend pull for the active state record.

Hook
----
- ``before_prompt_build`` (priority configurable, default 65) — when the
  current ``session_id`` is in an active Director state, replaces
  ``ctx["system_prompt_parts"][0]`` (Lexy's persona) with the Director's
  assembled prompt + state block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lexy_core.agent.persona import load_persona
from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .director_state import DirectorState
from .prompts import assemble_director_prompt, safe_json_dump_for_log

log = get_logger(module="rp_director")


# ─── Tool schemas ──────────────────────────────────────────────────────


START_RP_SETUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {
            "type": "string",
            "description": (
                "Die Session-ID, in der der Director uebernimmt. Optional "
                "— wird sonst aus dem aktuellen Tool-Call-Context gezogen."
            ),
        },
        "user_intent": {
            "type": "string",
            "description": (
                "Optionaler Wunsch des Users in einem Satz, z.B. "
                "'duesteres Sci-Fi mit Captain + 2 Crew'."
            ),
        },
    },
    "required": [],
}

AUTONOMY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["addressed_only", "proactive", "simulation"],
            "description": (
                "Wie agieren die Charaktere von selbst? "
                "'addressed_only' = Charaktere antworten nur wenn der User "
                "sie anspricht. 'proactive' = jeder Charakter meldet sich "
                "alle pulse_minutes von selbst. 'simulation' = alle "
                "simulation_interval_minutes spricht ein zufaelliger "
                "Charakter (oder Lexy)."
            ),
        },
        "pulse_minutes": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Bei mode='proactive': Intervall pro Charakter in Minuten. "
                "Default 30."
            ),
        },
        "simulation_interval_minutes": {
            "type": "integer",
            "minimum": 1,
            "maximum": 15,
            "description": (
                "Bei mode='simulation': Intervall in Minuten (1-15). "
                "Default 3."
            ),
        },
        "character_mode": {
            "type": "integer",
            "enum": [1, 2],
            "description": (
                "1 = Lexy bleibt aktiv (kann narratieren), Charaktere "
                "sprechen ueber den Group-Orchestrator. "
                "2 = nur Charaktere sprechen, Lexy schweigt komplett. "
                "Default 1."
            ),
        },
    },
    "required": [],
}

PROPOSE_SCENARIO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "scenario": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "mood": {"type": "string"},
                "hook": {"type": "string"},
                "rules": {"type": "string"},
                "scene_text": {
                    "type": "string",
                    "description": (
                        "1-3 Saetze die als character_chat-Scene committed "
                        "werden. Wird beim commit in set_character_mode "
                        "uebergeben."
                    ),
                },
                "autonomy": AUTONOMY_SCHEMA,
            },
            "required": [],
        },
    },
    "required": ["scenario"],
}

SET_RP_AUTONOMY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "mode": {
            "type": "string",
            "enum": ["addressed_only", "proactive", "simulation"],
        },
        "pulse_minutes": {"type": "integer", "minimum": 1},
        "simulation_interval_minutes": {
            "type": "integer", "minimum": 1, "maximum": 15
        },
        "character_mode": {"type": "integer", "enum": [1, 2]},
    },
    "required": ["mode"],
}

PROPOSE_CHARACTERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "persona": {"type": "string"},
                    "greeting": {"type": "string"},
                    "scenario": {"type": "string"},
                    "example_dialog": {"type": "string"},
                    "age_stage": {
                        "type": "string",
                        "enum": ["baby", "toddler", "child", "teen", "adult"],
                    },
                    "voice": {"type": "string"},
                    "color": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "relationships": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": (
                            "Andere Charakter-Namen → Beziehungslabel "
                            "('Schwester von ...', 'misstrauisch ggn ...')."
                        ),
                    },
                    "proactive_pulse_pattern": {"type": "string"},
                    "proactive_pulse_prompt": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    "required": ["characters"],
}

COMMIT_RP_SETUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
    },
    "required": [],
}

CANCEL_RP_SETUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
    },
    "required": [],
}


# ─── Plugin ────────────────────────────────────────────────────────────


class RPDirectorPlugin(BasePlugin):
    """Conversational RP-Setup Director."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._state: DirectorState | None = None
        self._persona_prompt: str = ""
        self._user_name: str = "Mike"

        # Config (set in on_load)
        self._persona_path: str = "config/personas/director.yaml"
        self._hook_priority: int = 65
        self._idle_timeout: float = 1800.0
        self._auto_activate_mode: bool = True

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def on_load(self) -> None:
        config = self.api.get_config()

        self._persona_path = str(
            config.get("persona_path", "config/personas/director.yaml")
        )
        self._hook_priority = int(config.get("hook_priority", 65))
        self._idle_timeout = float(config.get("idle_timeout_seconds", 1800))
        self._auto_activate_mode = bool(
            config.get("auto_activate_character_mode", True)
        )

        # Load + cache the Director persona prompt.
        self._reload_persona_prompt()

        # Carry the user_name from Lexy's main persona so placeholders in
        # the Director prompt match the same name the user sees elsewhere.
        main_persona = getattr(self.api._app, "persona", None)
        if main_persona is not None and getattr(main_persona, "user_name", ""):
            self._user_name = main_persona.user_name

        # Init state DB.
        db = await self.api.get_db()
        self._state = DirectorState(db)
        await self._state.init_table()

        # Best-effort: expire any sessions left in 'collecting' from a
        # previous run that crashed mid-setup.
        if self._idle_timeout > 0:
            try:
                expired = await self._state.expire_idle(self._idle_timeout)
                if expired:
                    log.info(
                        "rp_director.expired_stale_sessions",
                        count=len(expired),
                        ids=expired[:10],
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "rp_director.expire_stale_failed", error=str(exc)
                )

        log.info(
            "rp_director.loaded",
            persona_path=self._persona_path,
            persona_chars=len(self._persona_prompt),
            hook_priority=self._hook_priority,
        )

    async def on_enable(self) -> None:
        # Tools.
        self.api.register_tool(
            name="start_rp_setup",
            handler=self._tool_start_rp_setup,
            description=(
                "Aktiviere den RP-Director fuer die aktuelle Session. "
                "Der Director uebernimmt die naechste Antwort-Runde und "
                "fuehrt einen geleiteten Setup-Dialog (Scenario, Charaktere, "
                "Beziehungen) bis zum commit."
            ),
            schema=START_RP_SETUP_SCHEMA,
        )
        self.api.register_tool(
            name="propose_scenario",
            handler=self._tool_propose_scenario,
            description=(
                "(Director-Tool) Speichere einen Scenario-Vorschlag. "
                "Setting/Mood/Hook/Rules/scene_text. Mehrfachaufrufe "
                "ersetzen den vorherigen Vorschlag komplett."
            ),
            schema=PROPOSE_SCENARIO_SCHEMA,
        )
        self.api.register_tool(
            name="propose_characters",
            handler=self._tool_propose_characters,
            description=(
                "(Director-Tool) Speichere 1-N Charakter-Vorschlaege. "
                "Mehrfachaufrufe ersetzen die Liste komplett."
            ),
            schema=PROPOSE_CHARACTERS_SCHEMA,
        )
        self.api.register_tool(
            name="commit_rp_setup",
            handler=self._tool_commit_rp_setup,
            description=(
                "(Director-Tool) Schreibe die proposed Charaktere via "
                "spawn_character, aktiviere character_mode mit der Scene, "
                "und uebergib zurueck an Lexy. NUR nach expliziter "
                "Bestaetigung des Users aufrufen."
            ),
            schema=COMMIT_RP_SETUP_SCHEMA,
        )
        self.api.register_tool(
            name="cancel_rp_setup",
            handler=self._tool_cancel_rp_setup,
            description=(
                "(Director-Tool) Abbrechen des Setups. Schreibt nichts in "
                "character_chat, gibt Kontrolle an Lexy zurueck."
            ),
            schema=CANCEL_RP_SETUP_SCHEMA,
        )
        self.api.register_tool(
            name="set_rp_autonomy",
            handler=self._tool_set_rp_autonomy,
            description=(
                "Konfiguriere das Eigenleben der RP-Charaktere fuer eine "
                "Session. mode='addressed_only' (nur auf Ansprache), "
                "'proactive' (jeder Char alle pulse_minutes proaktiv), "
                "'simulation' (alle simulation_interval_minutes spricht ein "
                "zufaelliger Char). Optional character_mode=1|2. Funktioniert "
                "auch fuer bereits committete RPs — kann jederzeit aufgerufen "
                "werden, um Autonomie umzustellen."
            ),
            schema=SET_RP_AUTONOMY_SCHEMA,
        )

        # WebSocket handlers (frontend slash command + status pull).
        self.api.register_ws_handler(
            "rp_director_start", self._ws_rp_director_start
        )
        self.api.register_ws_handler(
            "rp_director_status", self._ws_rp_director_status
        )
        self.api.register_ws_handler(
            "rp_director_reload_persona", self._ws_reload_persona
        )

        # Hook: replace Lexy's persona for active Director sessions.
        self.api.register_hook(
            "before_prompt_build",
            self._hook_inject_director_prompt,
            priority=self._hook_priority,
        )

        log.info("rp_director.enabled")

    async def on_disable(self) -> None:
        # Plugin-owned cleanup; PluginAPI removes registrations.
        log.info("rp_director.disabled")

    # ─── Helpers ───────────────────────────────────────────────────────

    def _reload_persona_prompt(self) -> None:
        """Load and assemble the Director persona from disk."""
        try:
            persona = load_persona(Path(self._persona_path))
            self._persona_prompt = persona.assemble()
            log.debug(
                "rp_director.persona_loaded",
                path=self._persona_path,
                length=len(self._persona_prompt),
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "rp_director.persona_load_failed",
                path=self._persona_path,
                error=str(exc),
            )
            # Fallback so the plugin still works — the user gets a generic
            # but functional Director instead of a hard crash.
            self._persona_prompt = (
                "Du bist der Director, ein RP-Architekt. Hilf dem User, "
                "Scenario und Charaktere fuer ein RP zu entwerfen. Frage "
                "konkret nach, schlage Optionen vor, und rufe "
                "`commit_rp_setup` erst nach Bestaetigung auf."
            )

    def _resolve_session_id(self, supplied: str | None) -> str:
        """Pick the best session_id available to a tool call."""
        if supplied:
            return supplied
        # The agent doesn't pass session_id into tool kwargs by default.
        # Plugins that need it commonly take it as an explicit parameter,
        # but the LLM sometimes omits it. We fall back to "" and let the
        # caller decide what to do.
        return ""

    # ─── Tool: start_rp_setup ──────────────────────────────────────────

    async def _tool_start_rp_setup(
        self,
        session_id: str = "",
        user_intent: str = "",
    ) -> dict[str, Any]:
        if self._state is None:
            return {"ok": False, "error": "state_not_ready"}
        sid = self._resolve_session_id(session_id)
        if not sid:
            return {
                "ok": False,
                "error": "session_id_required",
                "hint": (
                    "Rufe das Tool mit explizitem session_id-Argument auf "
                    "(die aktuelle Session-ID steht im Frontend-Pill)."
                ),
            }
        record = await self._state.start(sid, user_intent=user_intent.strip())
        await self.api.ws_broadcast(
            {
                "type": "rp_director_started",
                "session_id": sid,
                "user_intent": user_intent.strip(),
            }
        )
        log.info(
            "rp_director.session_started",
            session_id=sid,
            intent=user_intent[:120],
        )
        return {
            "ok": True,
            "session_id": sid,
            "state": record["state"],
            "user_intent": record.get("user_intent", ""),
            "next": (
                "Der Director uebernimmt ab der naechsten Antwort. Stell "
                "ihm vor, was du dir vorstellst — er schlaegt dann ein "
                "konkretes Scenario vor."
            ),
        }

    # ─── Tool: propose_scenario ────────────────────────────────────────

    async def _tool_propose_scenario(
        self,
        scenario: dict[str, Any] | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        if self._state is None:
            return {"ok": False, "error": "state_not_ready"}
        if not isinstance(scenario, dict):
            return {"ok": False, "error": "scenario_must_be_object"}
        sid = self._resolve_session_id(session_id)
        if not sid:
            return {"ok": False, "error": "session_id_required"}
        if not await self._state.is_active(sid):
            return {
                "ok": False,
                "error": "no_active_director_session",
                "hint": "Rufe zuerst start_rp_setup auf.",
            }
        record = await self._state.set_scenario(sid, scenario)
        if record is None:
            return {"ok": False, "error": "could_not_save_scenario"}
        await self.api.ws_broadcast(
            {
                "type": "rp_director_scenario_proposed",
                "session_id": sid,
                "scenario": record.get("scenario"),
            }
        )
        log.info(
            "rp_director.scenario_proposed",
            session_id=sid,
            scenario=safe_json_dump_for_log(record.get("scenario")),
        )
        return {"ok": True, "scenario": record.get("scenario")}

    # ─── Tool: propose_characters ──────────────────────────────────────

    async def _tool_propose_characters(
        self,
        characters: list[dict[str, Any]] | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        if self._state is None:
            return {"ok": False, "error": "state_not_ready"}
        if not isinstance(characters, list) or not characters:
            return {"ok": False, "error": "characters_must_be_nonempty_list"}
        cleaned = [c for c in characters if isinstance(c, dict) and c.get("name")]
        if not cleaned:
            return {"ok": False, "error": "no_named_characters"}
        sid = self._resolve_session_id(session_id)
        if not sid:
            return {"ok": False, "error": "session_id_required"}
        if not await self._state.is_active(sid):
            return {
                "ok": False,
                "error": "no_active_director_session",
                "hint": "Rufe zuerst start_rp_setup auf.",
            }
        record = await self._state.set_characters(sid, cleaned)
        if record is None:
            return {"ok": False, "error": "could_not_save_characters"}
        await self.api.ws_broadcast(
            {
                "type": "rp_director_characters_proposed",
                "session_id": sid,
                "characters": record.get("characters"),
            }
        )
        log.info(
            "rp_director.characters_proposed",
            session_id=sid,
            count=len(cleaned),
            names=[c.get("name") for c in cleaned],
        )
        return {
            "ok": True,
            "count": len(record.get("characters") or []),
            "characters": record.get("characters"),
        }

    # ─── Tool: commit_rp_setup ─────────────────────────────────────────

    async def _tool_commit_rp_setup(
        self,
        session_id: str = "",
    ) -> dict[str, Any]:
        if self._state is None:
            return {"ok": False, "error": "state_not_ready"}
        sid = self._resolve_session_id(session_id)
        if not sid:
            return {"ok": False, "error": "session_id_required"}
        record = await self._state.get(sid)
        if record is None or record["state"] not in ("collecting", "proposing"):
            return {
                "ok": False,
                "error": "no_active_director_session",
                "hint": "Rufe zuerst start_rp_setup auf.",
            }

        characters = record.get("characters") or []
        scenario = record.get("scenario") or {}
        if not characters:
            return {
                "ok": False,
                "error": "no_characters_proposed",
                "hint": "Rufe propose_characters mit mindestens einem Char auf.",
            }

        # ── 1) Spawn each character via character_chat.spawn_character ──
        spawned: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        scene_text = (scenario.get("scene_text") or scenario.get("hook") or "").strip()

        for char in characters:
            payload: dict[str, Any] = {
                "name": str(char.get("name") or "").strip(),
                "persona": str(char.get("persona") or "").strip(),
                "greeting": str(char.get("greeting") or "").strip(),
                "scenario": str(char.get("scenario") or scene_text).strip(),
                "example_dialog": str(char.get("example_dialog") or "").strip(),
                "age_stage": str(char.get("age_stage") or "adult"),
                "voice": str(char.get("voice") or "").strip(),
                "color": str(char.get("color") or "").strip() or "#7aa2f7",
                "tags": list(char.get("tags") or []),
                "relationships": dict(char.get("relationships") or {}),
                "session_id": sid,  # auto-attach to the active session
            }
            pulse_pattern = str(char.get("proactive_pulse_pattern") or "").strip()
            if pulse_pattern:
                payload["proactive_pulse_pattern"] = pulse_pattern
            pulse_prompt = str(char.get("proactive_pulse_prompt") or "").strip()
            if pulse_prompt:
                payload["proactive_pulse_prompt"] = pulse_prompt

            spawn_result = await self.api.call_tool("spawn_character", payload)
            if spawn_result.get("ok") and isinstance(spawn_result.get("data"), dict):
                spawned.append(spawn_result["data"])
            else:
                failures.append(
                    {
                        "name": payload["name"],
                        "error": spawn_result.get("error")
                        or "spawn_character returned no data",
                    }
                )

        # If everything failed, abort and let the user retry.
        if not spawned:
            log.error(
                "rp_director.commit_failed_all_characters",
                session_id=sid,
                failures=failures,
            )
            return {
                "ok": False,
                "error": "all_character_spawns_failed",
                "failures": failures,
            }

        # ── 2) Resolve relationship references (name → character_id) ──
        # CharacterCard stores relationships keyed by ID; the Director
        # collected them keyed by NAME because that's what the LLM is
        # comfortable with. Translate now that we have the spawned IDs.
        name_to_id = {
            c.get("name"): c.get("id") for c in spawned if c.get("id")
        }
        relationship_updates: list[tuple[str, dict[str, str]]] = []
        for char, spawned_info in zip(characters, spawned):
            relationships_raw = char.get("relationships") or {}
            if not isinstance(relationships_raw, dict) or not relationships_raw:
                continue
            translated: dict[str, str] = {}
            for other_name, label in relationships_raw.items():
                other_id = name_to_id.get(other_name)
                if other_id:
                    translated[other_id] = str(label)
            if translated:
                relationship_updates.append(
                    (str(spawned_info.get("id") or ""), translated)
                )

        for char_id, rel in relationship_updates:
            if not char_id:
                continue
            try:
                await self.api.call_tool(
                    "update_character",
                    {"id": char_id, "relationships": rel},
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "rp_director.relationship_update_failed",
                    char_id=char_id,
                    error=str(exc),
                )

        # ── 3) Activate character_mode with the scene ──
        autonomy_cfg = scenario.get("autonomy") or {}
        # If the Director picked character_mode=2 in autonomy, honour that
        # in the initial set_character_mode call so Lexy stays silent from
        # the very first turn instead of needing a second mode-switch.
        explicit_mode = autonomy_cfg.get("character_mode") if isinstance(autonomy_cfg, dict) else None
        mode_result: dict[str, Any] = {"ok": False}
        if self._auto_activate_mode:
            mode_payload: dict[str, Any] = {
                "session_id": sid,
                "scene": scene_text,
            }
            if explicit_mode in (1, 2):
                mode_payload["mode"] = int(explicit_mode)
            else:
                mode_payload["enabled"] = True
            mode_result = await self.api.call_tool(
                "set_character_mode", mode_payload
            )

        # ── 3b) Apply autonomy config (pulses, simulation) ──
        autonomy_result: dict[str, Any] | None = None
        if isinstance(autonomy_cfg, dict) and autonomy_cfg.get("mode"):
            spawned_ids = [s.get("id") for s in spawned if s.get("id")]
            autonomy_result = await self._apply_autonomy(
                session_id=sid,
                autonomy=autonomy_cfg,
                character_ids=spawned_ids,
            )

        # ── 4) Mark Director state committed; broadcast handover ──
        await self._state.mark_committed(sid)
        await self.api.ws_broadcast(
            {
                "type": "rp_director_committed",
                "session_id": sid,
                "scenario": scenario,
                "characters": [
                    {
                        "id": s.get("id"),
                        "name": s.get("name"),
                        "age_stage": s.get("age_stage"),
                    }
                    for s in spawned
                ],
                "character_mode_set": bool(mode_result.get("ok")),
                "autonomy": autonomy_result,
                "failures": failures,
            }
        )
        log.info(
            "rp_director.committed",
            session_id=sid,
            spawned=len(spawned),
            failures=len(failures),
            mode_set=bool(mode_result.get("ok")),
            autonomy_mode=(autonomy_cfg or {}).get("mode") if isinstance(autonomy_cfg, dict) else None,
        )
        return {
            "ok": True,
            "session_id": sid,
            "spawned": spawned,
            "failures": failures,
            "scene": scene_text,
            "character_mode_set": bool(mode_result.get("ok")),
            "autonomy": autonomy_result,
            "handover": (
                "Buehne ist bereit. Lexy uebernimmt ab der naechsten "
                "User-Nachricht; die Charaktere sprechen via "
                "character_chat-Orchestrator."
            ),
        }

    # ─── Autonomy application (shared by commit + set_rp_autonomy) ────

    async def _apply_autonomy(
        self,
        session_id: str,
        autonomy: dict[str, Any],
        character_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply an autonomy configuration to a session.

        ``autonomy`` keys:
            - ``mode`` (required): "addressed_only" | "proactive" | "simulation"
            - ``pulse_minutes``: int (proactive only, default 30)
            - ``simulation_interval_minutes``: int (simulation only, default 3)
            - ``character_mode``: 1 | 2 (only honoured if explicitly set)

        ``character_ids`` is the explicit list of characters to configure.
        If None, the plugin pulls all currently attached characters of the
        session via ``list_characters``.

        Returns a dict suitable for tool-result reporting:
            ``{ok, mode, pulse_set: int, simulation_started: bool, errors[]}``
        """
        mode = str(autonomy.get("mode") or "").strip()
        if mode not in ("addressed_only", "proactive", "simulation"):
            return {"ok": False, "error": f"invalid_mode: {mode!r}"}

        # Resolve target character list if not given.
        if character_ids is None:
            list_result = await self.api.call_tool(
                "list_characters", {"session_id": session_id}
            )
            if list_result.get("ok") and isinstance(list_result.get("data"), dict):
                character_ids = [
                    c.get("id")
                    for c in list_result["data"].get("characters", [])
                    if isinstance(c, dict) and c.get("id")
                ]
            else:
                character_ids = []

        errors: list[dict[str, Any]] = []
        pulse_set_count = 0
        simulation_started = False

        # ── Step 1: clear-or-set per-character pulse pattern ────────
        if mode == "addressed_only":
            new_pattern = ""
        elif mode == "proactive":
            pulse_minutes = max(1, int(autonomy.get("pulse_minutes") or 30))
            new_pattern = f"every {pulse_minutes}m"
        else:  # simulation: per-character pulses are typically OFF
            new_pattern = ""

        for char_id in character_ids:
            update_result = await self.api.call_tool(
                "update_character",
                {"id": char_id, "proactive_pulse_pattern": new_pattern},
            )
            if not update_result.get("ok"):
                errors.append(
                    {"id": char_id, "step": "update_character",
                     "error": update_result.get("error")}
                )
                continue
            # Re-attach so the scheduler timer is (re-)registered or
            # cancelled. attach_character_to_session is idempotent for
            # session membership but always rebinds the pulse timer.
            attach_result = await self.api.call_tool(
                "attach_character_to_session",
                {"id": char_id, "session_id": session_id},
            )
            if attach_result.get("ok"):
                if new_pattern:
                    pulse_set_count += 1
            else:
                errors.append(
                    {"id": char_id, "step": "attach",
                     "error": attach_result.get("error")}
                )

        # ── Step 2: start or stop the session-wide simulation ────────
        if mode == "simulation":
            interval = int(autonomy.get("simulation_interval_minutes") or 3)
            sim_result = await self.api.call_tool(
                "start_simulation",
                {"session_id": session_id, "interval_minutes": interval},
            )
            if sim_result.get("ok"):
                simulation_started = True
            else:
                errors.append(
                    {"step": "start_simulation",
                     "error": sim_result.get("error")}
                )
        else:
            # Defensive: ensure no leftover simulation timer is running.
            await self.api.call_tool(
                "stop_simulation", {"session_id": session_id}
            )

        # ── Step 3: optionally update character_mode (1 vs 2) ────────
        char_mode = autonomy.get("character_mode")
        char_mode_set: int | None = None
        if char_mode in (1, 2):
            mode_result = await self.api.call_tool(
                "set_character_mode",
                {"session_id": session_id, "mode": int(char_mode)},
            )
            if mode_result.get("ok"):
                char_mode_set = int(char_mode)
            else:
                errors.append(
                    {"step": "set_character_mode",
                     "error": mode_result.get("error")}
                )

        log.info(
            "rp_director.autonomy_applied",
            session_id=session_id,
            mode=mode,
            pulse_set=pulse_set_count,
            simulation_started=simulation_started,
            character_mode=char_mode_set,
            errors=len(errors),
        )
        return {
            "ok": True,
            "mode": mode,
            "pulse_set": pulse_set_count,
            "simulation_started": simulation_started,
            "character_mode_set": char_mode_set,
            "characters_targeted": len(character_ids),
            "errors": errors,
        }

    # ─── Tool: set_rp_autonomy ─────────────────────────────────────────

    async def _tool_set_rp_autonomy(
        self,
        mode: str,
        session_id: str = "",
        pulse_minutes: int | None = None,
        simulation_interval_minutes: int | None = None,
        character_mode: int | None = None,
    ) -> dict[str, Any]:
        """Reconfigure autonomy for an existing (or just-committed) RP."""
        sid = self._resolve_session_id(session_id)
        if not sid:
            return {"ok": False, "error": "session_id_required"}
        autonomy: dict[str, Any] = {"mode": mode}
        if pulse_minutes is not None:
            autonomy["pulse_minutes"] = pulse_minutes
        if simulation_interval_minutes is not None:
            autonomy["simulation_interval_minutes"] = simulation_interval_minutes
        if character_mode is not None:
            autonomy["character_mode"] = character_mode
        result = await self._apply_autonomy(sid, autonomy, character_ids=None)
        await self.api.ws_broadcast(
            {"type": "rp_director_autonomy_changed",
             "session_id": sid, "result": result}
        )
        return result

    # ─── Tool: cancel_rp_setup ─────────────────────────────────────────

    async def _tool_cancel_rp_setup(
        self,
        session_id: str = "",
    ) -> dict[str, Any]:
        if self._state is None:
            return {"ok": False, "error": "state_not_ready"}
        sid = self._resolve_session_id(session_id)
        if not sid:
            return {"ok": False, "error": "session_id_required"}
        record = await self._state.get(sid)
        if record is None or record["state"] not in ("collecting", "proposing"):
            return {"ok": False, "error": "no_active_director_session"}
        await self._state.mark_cancelled(sid)
        await self.api.ws_broadcast(
            {"type": "rp_director_cancelled", "session_id": sid}
        )
        log.info("rp_director.cancelled", session_id=sid)
        return {"ok": True, "session_id": sid, "state": "cancelled"}

    # ─── WS: rp_director_start (slash command) ─────────────────────────

    async def _ws_rp_director_start(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        sid = str(message.get("session_id", "") or "")
        intent = str(message.get("user_intent", "") or "")
        if not sid:
            await client.send_json(
                {
                    "type": "rp_director_error",
                    "error": "session_id_required",
                }
            )
            return
        result = await self._tool_start_rp_setup(
            session_id=sid, user_intent=intent
        )
        await client.send_json({"type": "rp_director_start_ack", **result})

    async def _ws_rp_director_status(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        if self._state is None:
            await client.send_json(
                {"type": "rp_director_status", "error": "state_not_ready"}
            )
            return
        sid = str(message.get("session_id", "") or "")
        if sid:
            record = await self._state.get(sid)
            await client.send_json(
                {
                    "type": "rp_director_status",
                    "session_id": sid,
                    "record": record,
                }
            )
            return
        active = await self._state.list_active()
        await client.send_json(
            {"type": "rp_director_status", "active": active}
        )

    async def _ws_reload_persona(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        self._reload_persona_prompt()
        await client.send_json(
            {
                "type": "rp_director_persona_reloaded",
                "length": len(self._persona_prompt),
            }
        )

    # ─── Hook: before_prompt_build ─────────────────────────────────────

    async def _hook_inject_director_prompt(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        If the current session is in an active Director state, replace
        Lexy's persona slot in ``system_prompt_parts`` with the Director
        persona + state block.

        We replace ``system_prompt_parts[0]`` (the persona slot) instead
        of appending so the LLM sees ONE consistent voice. Date/time,
        tool prompt, and recalled memory (later slots) stay intact.
        """
        if self._state is None:
            return data

        session_id = str(data.get("session_id", "") or "")
        if not session_id:
            return data

        try:
            record = await self._state.get(session_id)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "rp_director.hook_state_lookup_failed",
                session_id=session_id,
                error=str(exc),
            )
            return data

        if record is None or record["state"] not in ("collecting", "proposing"):
            return data

        director_prompt = assemble_director_prompt(
            persona_prompt=self._persona_prompt,
            state_record=record,
            user_name=self._user_name,
        )

        parts = data.get("system_prompt_parts")
        if isinstance(parts, list) and parts:
            parts[0] = director_prompt
        else:
            data["system_prompt_parts"] = [director_prompt]

        log.debug(
            "rp_director.hook_replaced_persona",
            session_id=session_id,
            state=record["state"],
            prompt_chars=len(director_prompt),
        )
        return data
