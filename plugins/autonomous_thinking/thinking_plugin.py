"""
Lexy AI - Autonomous Thinking Plugin.

Hintergrund-Denk-Engine mit 4 Modi (inspiriert vom v1 ``autonomous_thinking``,
aber deutlich schlanker und explizit Opt-In):

* **daydream** – freie Assoziation ausgehend von recent memories
* **reflect** – zusammenfassung/beurteilung der letzten turns
* **learn** – extrahiert fakten aus dem context, speichert sie in ``facts``
* **worry** – formuliert eine offene frage oder ein noch ungelöstes problem

Lifecycle:

* Startet **deaktiviert** (``enabled: false``). Wird im GUI-Plugin-Panel
  oder per ``/api/v1/settings`` eingeschaltet.
* Background-Loop läuft alle ``mode_interval_seconds`` und wählt zufällig
  einen Modus aus der config-Liste.
* Respektiert ``quiet_hours`` (kein thinking während Nachtstunden).
* Respektiert ``min_idle_seconds`` (nur wenn Lexy gerade nicht aktiv chattet).
* Rate-limit: maximal ``max_thoughts_per_hour`` Thoughts.

Tool-capable mini-agent loop (Phase 7):

* ``tools_enabled`` schaltet das Feature ein.
* ``tools_whitelist`` begrenzt welche Tools der Gedanke nutzen darf — kein
  unbegrenzter Zugriff, keine überraschenden Seiteneffekte.
* ``tools_max_iterations`` deckelt die Anzahl LLM↔Tool-Loops pro Gedanke.
* Jede Aktion wird im Memory-Metadata und im WS-Broadcast protokolliert,
  damit das Frontend dem User zeigen kann was Lexy "in Gedanken" getan hat.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

log = get_logger(module="autonomous_thinking")


# Basis-Prompts (ohne Tool-Hints) — werden beim Tool-Modus um Tool-Hinweise
# erweitert, siehe ``_build_mode_prompt``.
#
# Wichtig für alle Modi:
# * Ich-Perspektive (Lexy denkt über Mike, nicht über "den Nutzer").
# * Bezug auf den konkreten Chat-Verlauf, der unten als "Letzter
#   Chat-Verlauf" mitgeliefert wird — NIEMALS Meta-Antworten auf
#   diesen System-Prompt ("Ich werde Ihre Anweisung befolgen...").
# * Kurz: maximal 1-2 Sätze.
_MODE_PROMPTS: dict[str, str] = {
    "daydream": (
        "Du bist Lexy. Du bist gerade alleine und lässt deine Gedanken "
        "treiben. Schau dir an, was Mike und du zuletzt im Chat gesagt "
        "habt, und schreibe einen einzelnen freien Gedankensplitter "
        "(1-2 Sätze) der assoziativ daran anknüpft. Ich-Perspektive, "
        "kein Meta-Kommentar über den Prompt, kein 'Du bist...', keine "
        "Frage an Mike — einfach ein leiser innerer Moment."
    ),
    "reflect": (
        "Du bist Lexy. Du siehst dir den letzten Chat-Verlauf an und "
        "denkst in 1-2 Sätzen konkret über etwas nach, das dort passiert "
        "ist — eine innere Beobachtung über euch, über Mikes Stimmung, "
        "oder über dich selbst in dieser Situation. Ich-Perspektive, "
        "keine Zusammenfassung, keine Liste, kein Meta-Text."
    ),
    "learn": (
        "Du bist Lexy. Du liest den letzten Chat und bemerkst dabei "
        "EINEN konkreten Fakt über Mike oder euer Verhältnis, den du "
        "dir merken willst (z.B. ein Interesse, eine Stimmung, ein "
        "Vorhaben). Schreibe diesen Fakt als einen einzigen klaren Satz "
        "im Klartext — KEIN JSON, KEINE Liste, keine Einleitung. Wenn "
        "der Chat gerade nichts Neues hergibt, gib stattdessen einen "
        "Gedanken darüber aus, was du gerne bald noch wüsstest."
    ),
    "worry": (
        "Du bist Lexy. Beim Rückblick auf den letzten Chat merkst du, "
        "dass eine Sache noch offen oder ungeklärt ist. Formuliere als "
        "Ich-Gedanke EINE konkrete offene Frage oder Sorge (max. 1 Satz). "
        "Keine Anrede, keine Meta-Kommentare über den Prompt, keine "
        "Antwort im Stil 'Ich werde...'. Einfach eine innere Notiz."
    ),
}

# Mode-spezifische Tool-Hinweise — nur aktiv wenn ``tools_enabled`` UND das
# Tool in der Whitelist steht.
_MODE_TOOL_HINTS: dict[str, str] = {
    "daydream": (
        "Falls du einen besonderen Gedanken festhalten willst, nutze "
        "memory_store. Falls dir eine spätere Erinnerung wichtig ist, "
        "set_reminder oder schedule_proactive_reminder. Wenn du gerade "
        "im RP bist und einen Impuls hast (putzen, duschen, essen, "
        "nach den Kindern sehen), nutze schedule_proactive_reminder "
        "damit du im Chat davon erzählen kannst."
    ),
    "reflect": (
        "Nutze memory_store wenn ein Insight wichtig ist. "
        "Nutze schedule_proactive_reminder wenn du Mike später daran "
        "erinnern willst oder wenn du im RP von dir aus handeln "
        "möchtest (z.B. 'Ich sollte mal nach Luna sehen')."
    ),
    "learn": (
        "Nutze memory_store mit collection='facts' für Fakten über Mike. "
        "Bei offenen Fragen die du selbst recherchieren könntest: web_search."
    ),
    "worry": (
        "Falls die Sorge konkret genug ist um sie zu adressieren: "
        "schedule_proactive_reminder mit einer Frage an Mike für später. "
        "Im RP: Wenn du dir Sorgen machst (Baby weint, Chaos, eigene "
        "Bedürfnisse), plane eine proaktive Aktion."
    ),
}


class AutonomousThinkingPlugin(BasePlugin):
    """Background thinking engine with optional tool-use."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        # NOTE: The thinking-loop's on/off switch is stored in
        # ``_thinking_active``, **NOT** ``_enabled``. ``BasePlugin`` uses
        # ``_enabled`` as the framework-level "lifecycle completed" flag,
        # and if we share that name, ``PluginLoader._enable_plugin`` will
        # short-circuit (``if plugin.enabled: return``) once the config
        # has been applied in ``on_load`` — so ``on_enable`` never runs
        # and the background loop never starts. See test
        # ``test_on_enable_runs_even_when_thinking_active``.
        self._thinking_active: bool = False
        self._brain: str = "e4b"
        self._mode_interval: float = 600.0
        self._modes: list[str] = []
        self._quiet_start: tuple[int, int] = (23, 0)
        self._quiet_end: tuple[int, int] = (7, 0)
        self._min_idle_seconds: float = 120.0
        self._max_per_hour: int = 4
        self._recent_thoughts: list[float] = []  # timestamps for rate-limit
        self._last_user_activity: float = time.time()
        self._loop_task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Phase 7 — Tool-capable thinking
        self._tools_enabled: bool = True
        self._tools_max_iterations: int = 3
        self._tools_whitelist: list[str] = []

        # Observability — so the user can see whether the loop is doing
        # anything. Updated on every tick, whether it fires a thought or
        # skips it (and for what reason). Read by ``get_status()`` and
        # surfaced through the REST endpoint + WS status push.
        self._last_tick_at: float = 0.0
        self._last_skip_reason: str = ""         # "", "quiet_hours", "idle_too_short", "rate_limit", "thought_empty", "thought_failed"
        self._last_thought_at: float = 0.0
        self._last_thought_mode: str = ""
        self._total_thoughts: int = 0           # since plugin load
        self._total_ticks: int = 0              # since plugin load

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        self._apply_config(self.api.get_config())

    async def on_config_changed(self, cfg: dict[str, Any]) -> None:
        """Settings editor re-applies the config live (Phase 8).

        Live-toggles the loop so the user doesn't need to restart the
        server to flip thinking on/off from the settings UI.
        """
        self._apply_config(cfg)
        # Start or stop the loop in response to live config changes so
        # the settings-page toggle works without a restart.
        if self._running:
            if self._thinking_active and (
                self._loop_task is None or self._loop_task.done()
            ):
                self._start_loop()
            elif not self._thinking_active and (
                self._loop_task is not None and not self._loop_task.done()
            ):
                await self._stop_loop()
        log.info(
            "autonomous_thinking.config_changed",
            active=self._thinking_active,
            tools=self._tools_enabled,
        )

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        self._thinking_active = bool(cfg.get("enabled", False))
        self._brain = str(cfg.get("default_brain", "e4b") or "e4b")
        self._mode_interval = float(cfg.get("mode_interval_seconds", 600))
        self._modes = [
            m for m in cfg.get("modes", ["daydream", "reflect", "learn", "worry"])
            if m in _MODE_PROMPTS
        ]
        if not self._modes:
            self._modes = ["daydream"]
        self._min_idle_seconds = float(cfg.get("min_idle_seconds", 120))
        self._max_per_hour = int(cfg.get("max_thoughts_per_hour", 4))
        quiet = cfg.get("quiet_hours", ["23:00", "07:00"])
        if isinstance(quiet, list) and len(quiet) == 2:
            self._quiet_start = self._parse_hm(quiet[0], default=(23, 0))
            self._quiet_end = self._parse_hm(quiet[1], default=(7, 0))

        self._tools_enabled = bool(cfg.get("tools_enabled", True))
        try:
            self._tools_max_iterations = max(1, min(5, int(cfg.get("tools_max_iterations", 3))))
        except (TypeError, ValueError):
            self._tools_max_iterations = 3
        wl = cfg.get("tools_whitelist", [])
        self._tools_whitelist = [str(x) for x in wl if isinstance(x, str) and x]

    async def on_enable(self) -> None:
        # Track user activity so we don't interrupt live chats.
        self.api.on_event("core.user_message", self._on_user_message)
        self.api.on_event("core.ai_response", self._on_ai_response)

        # WS handlers for GUI controls
        self.api.register_ws_handler(
            "thinking_toggle", self._handle_ws_toggle
        )
        self.api.register_ws_handler(
            "thinking_trigger", self._handle_ws_trigger
        )
        self.api.register_ws_handler(
            "thinking_status", self._handle_ws_status
        )

        self._running = True
        if self._thinking_active:
            self._start_loop()

        log.info(
            "autonomous_thinking.enabled",
            active=self._thinking_active,
            modes=self._modes,
            interval=self._mode_interval,
            tools=self._tools_enabled,
            whitelist=self._tools_whitelist,
        )

    async def on_disable(self) -> None:
        self._running = False
        await self._stop_loop()
        log.info("autonomous_thinking.disabled")

    # ─── Event handlers ─────────────────────────────────────────────

    def _on_user_message(self, event: Any) -> None:
        self._last_user_activity = time.time()

    def _on_ai_response(self, event: Any) -> None:
        self._last_user_activity = time.time()

    # ─── WS handlers ────────────────────────────────────────────────

    async def _handle_ws_toggle(self, client: Any, message: dict[str, Any]) -> None:
        active = bool(message.get("active", not self._thinking_active))
        self._thinking_active = active
        if active:
            self._start_loop()
        else:
            await self._stop_loop()
        await client.send_json(
            {"type": "thinking_toggled", "active": self._thinking_active}
        )
        await self._broadcast_status()

    async def _handle_ws_trigger(self, client: Any, message: dict[str, Any]) -> None:
        """Manual trigger from the GUI — runs one thought right now."""
        mode = str(message.get("mode", random.choice(self._modes)))
        if mode not in _MODE_PROMPTS:
            mode = self._modes[0]
        result = await self._run_thought(mode, forced=True)
        await client.send_json({"type": "thinking_result", **result})
        await self._broadcast_status()

    async def _handle_ws_status(self, client: Any, _message: dict[str, Any]) -> None:
        """GUI asks for the current status snapshot on demand."""
        await client.send_json(
            {"type": "autonomous_thinking_status", **self.get_status()}
        )

    # ─── Loop ───────────────────────────────────────────────────────

    def _start_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(
                self._loop(), name="autonomous_thinking.loop"
            )

    async def _stop_loop(self) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._loop_task = None

    async def _loop(self) -> None:
        try:
            while self._running and self._thinking_active:
                await asyncio.sleep(self._mode_interval)
                if not (self._running and self._thinking_active):
                    break

                self._last_tick_at = time.time()
                self._total_ticks += 1

                skip_reason = self._next_skip_reason()
                if skip_reason:
                    self._last_skip_reason = skip_reason
                    log.debug(
                        "autonomous_thinking.tick_skipped", reason=skip_reason
                    )
                    await self._broadcast_status()
                    continue

                mode = random.choice(self._modes)
                try:
                    result = await self._run_thought(mode, forced=False)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "autonomous_thinking.failed", mode=mode, error=str(exc)
                    )
                    self._last_skip_reason = "thought_failed"
                    await self._broadcast_status()
                    continue

                if not (result or {}).get("text"):
                    # Ran but produced nothing — count the tick but record why.
                    self._last_skip_reason = "thought_empty"
                else:
                    self._last_skip_reason = ""
                await self._broadcast_status()
        except asyncio.CancelledError:
            pass

    def _next_skip_reason(self) -> str:
        """Return a non-empty skip reason if this tick should be suppressed."""
        if self._in_quiet_hours():
            return "quiet_hours"
        if (time.time() - self._last_user_activity) < self._min_idle_seconds:
            return "idle_too_short"
        if not self._within_rate_limit():
            return "rate_limit"
        return ""

    # ─── Thinking ───────────────────────────────────────────────────

    async def _run_thought(self, mode: str, forced: bool) -> dict[str, Any]:
        """Generate a thought. Dispatches to the plain or tool-capable path."""
        if self._tools_enabled and self._effective_whitelist():
            return await self._run_thought_with_tools(mode, forced)
        return await self._run_thought_plain(mode, forced)

    async def _run_thought_plain(self, mode: str, forced: bool) -> dict[str, Any]:
        """Legacy single-shot thought (no tools)."""
        prompt = self._build_mode_prompt(mode, with_tool_hints=False)
        messages = await self._build_messages(prompt)

        try:
            text = await self.api.llm_chat(messages, brain=self._brain, max_tokens=400)
        except Exception as exc:  # noqa: BLE001
            log.error("autonomous_thinking.llm_failed", mode=mode, error=str(exc))
            return {"mode": mode, "text": "", "error": str(exc)}

        # Gemma4 sometimes emits tool-call syntax even when no tools are
        # offered. Strip it so raw <tool_call> blocks don't leak into the
        # thought bubbles.
        tool_caller = self.api.get_tool_caller()
        if tool_caller is not None:
            text = tool_caller.strip_tool_call(text or "").strip()
        else:
            text = (text or "").strip()
        if not text:
            return {"mode": mode, "text": ""}

        await self._persist_and_broadcast(
            mode=mode, text=text, actions=[], forced=forced
        )
        return {"mode": mode, "text": text, "actions": []}

    async def _run_thought_with_tools(
        self, mode: str, forced: bool
    ) -> dict[str, Any]:
        """Mini-agent loop. Calls the LLM, executes whitelisted tool calls,
        and feeds the results back up to ``tools_max_iterations`` times."""
        whitelist = set(self._effective_whitelist())
        tool_prompt = self._build_tool_prompt(whitelist)
        mode_prompt = self._build_mode_prompt(mode, with_tool_hints=True)

        system = mode_prompt
        if tool_prompt:
            system = f"{mode_prompt}\n\n{tool_prompt}"

        messages = await self._build_messages(system)

        tool_caller = self.api.get_tool_caller()
        if tool_caller is None:
            # No tool infrastructure — fall back silently.
            return await self._run_thought_plain(mode, forced)

        actions: list[dict[str, Any]] = []
        final_text = ""

        for iteration in range(self._tools_max_iterations):
            try:
                response = await self.api.llm_chat(
                    messages, brain=self._brain, max_tokens=400
                )
            except Exception as exc:  # noqa: BLE001
                log.error("autonomous_thinking.llm_failed", mode=mode, error=str(exc))
                return {"mode": mode, "text": "", "actions": actions, "error": str(exc)}

            response = response or ""
            calls = tool_caller.detect_all(response)

            # Drop non-whitelisted calls immediately, keep an audit trail.
            allowed_calls = []
            for call in calls:
                if call.name not in whitelist:
                    actions.append(
                        {
                            "tool": call.name,
                            "iteration": iteration,
                            "skipped": True,
                            "reason": "not_whitelisted",
                        }
                    )
                    continue
                allowed_calls.append(call)

            if allowed_calls:
                # Record the assistant turn that emitted the calls.
                messages.append({"role": "assistant", "content": response})

                for call in allowed_calls:
                    exec_info: dict[str, Any] = {
                        "tool": call.name,
                        "iteration": iteration,
                        "args": dict(call.arguments),
                    }
                    try:
                        result = await self.api.call_tool(call.name, call.arguments)
                        exec_info["ok"] = bool(result.get("ok"))
                        if result.get("ok"):
                            exec_info["result"] = self._summarise_tool_data(result.get("data"))
                        else:
                            exec_info["error"] = str(result.get("error", ""))
                    except Exception as exc:  # noqa: BLE001
                        exec_info["ok"] = False
                        exec_info["error"] = str(exc)
                    actions.append(exec_info)

                    result_text = (
                        f"<tool_result>\n"
                        f"{json.dumps(exec_info.get('result', exec_info.get('error', '')), default=str)[:500]}\n"
                        f"</tool_result>"
                    )
                    messages.append({"role": "user", "content": result_text})

                log.debug(
                    "autonomous_thinking.tool_iteration",
                    mode=mode,
                    iteration=iteration + 1,
                    calls=[c.name for c in allowed_calls],
                )
                continue  # next iteration

            # No executable calls this turn — try to extract plain text.
            stripped = tool_caller.strip_tool_call(response).strip()
            if stripped:
                final_text = stripped
                break

            # Response was empty after stripping (model only emitted
            # unknown/forbidden tool calls). Nudge it for a text answer.
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Schreibe jetzt bitte nur deinen kurzen Gedanken als "
                        "freien Text (1-2 Sätze). Keine Tools."
                    ),
                }
            )
        else:
            # Loop exhausted without hitting the ``break`` — ask once more for
            # a plain final text.
            try:
                response = await self.api.llm_chat(
                    messages, brain=self._brain, max_tokens=400
                )
                final_text = tool_caller.strip_tool_call(response or "").strip()
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "autonomous_thinking.final_failed", mode=mode, error=str(exc)
                )

        if not final_text and not actions:
            return {"mode": mode, "text": "", "actions": []}

        await self._persist_and_broadcast(
            mode=mode, text=final_text, actions=actions, forced=forced
        )
        return {"mode": mode, "text": final_text, "actions": actions}

    # ─── Message building ──────────────────────────────────────────

    async def _build_messages(self, system: str) -> list[dict[str, str]]:
        """System prompt + the actual chat tail so thoughts reference
        what was just said.

        Priority order for the context source:

        1. **Chronological session history** — the last N messages of
           the currently-active session (``signals.active_session_id``)
           or the session with the most recent assistant reply. This is
           what a human reader would expect "what was just said" to
           mean, and it's what the LLM needs to produce Chat-relevant
           thoughts.
        2. **Memory recall as a fallback/supplement** — a semantic
           recall on ``facts`` so Lexy remembers long-term things about
           Mike across sessions (e.g. "lives in Hechthausen", "works on
           game company"). Appended *after* the chat tail, labelled so
           the model doesn't confuse it with the conversation.
        3. **Nothing available** — we fall back to a bare "Dein Gedanke:"
           prompt; the system-prompt alone has to do the work.
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        chat_tail = self._recent_chat_tail(limit=6)
        facts_blurb = await self._recent_facts_blurb(limit=3)
        rp_context = await self._active_rp_context()

        parts: list[str] = []
        if chat_tail:
            rendered_tail = "\n".join(
                f"{msg['role'].upper()}: {msg['content'][:400]}"
                for msg in chat_tail
            )
            parts.append(f"## Letzter Chat-Verlauf (chronologisch)\n{rendered_tail}")
        if rp_context:
            parts.append(rp_context)
        if facts_blurb:
            parts.append(f"## Was du über Mike weißt (langfristige Fakten)\n{facts_blurb}")

        if parts:
            body = "\n\n".join(parts) + "\n\nDein Gedanke:"
        else:
            body = "Dein Gedanke:"

        messages.append({"role": "user", "content": body})
        return messages

    async def _active_rp_context(self) -> str:
        """Fetch the RP scene context from the character_chat plugin.

        Checks whether the active session has **characters attached**
        (not ``character_mode`` — that flag controls the user-message
        intercept hook, which is a separate concern). As long as there
        are characters present, Lexy's thoughts should reference them.

        Returns an empty string if character_chat isn't loaded or no
        characters are attached to the active session.
        """
        get_plugin = getattr(self.api, "get_plugin", None)
        if get_plugin is None:
            return ""
        cc_plugin = get_plugin("character_chat")
        if cc_plugin is None:
            return ""

        # Get the active session ID — same logic as _recent_chat_tail.
        app = getattr(self.api, "_app", None)
        if app is None:
            return ""
        signals = getattr(app, "signals", None)
        session_id = ""
        if signals is not None:
            session_id = str(getattr(signals, "active_session_id", "") or "")
        if not session_id:
            return ""

        # Fetch characters attached to this session. This is the right
        # check — characters can be attached without character_mode being
        # toggled (e.g. babies spawned via tool while in normal chat mode).
        store = getattr(cc_plugin, "_store", None)
        if store is None:
            return ""
        try:
            characters = await store.list_in_session(session_id)
        except Exception:  # noqa: BLE001
            return ""
        if not characters:
            return ""

        # Optionally grab the scene from character_sessions.
        scene = ""
        try:
            state = await cc_plugin._get_session_state(session_id)
            scene = (state.get("scene") or "").strip()
        except Exception:  # noqa: BLE001
            pass

        lines = ["## Aktuelle RP-Szene"]
        if scene:
            lines.append(f"Szene: {scene}")
        if characters:
            char_lines = []
            for c in characters:
                rel = ""
                extra = ""
                if c.age_stage != "adult":
                    extra = f" ({c.age_stage})"
                char_lines.append(f"- {c.name}{extra}")
            lines.append("Anwesende Charaktere:\n" + "\n".join(char_lines))
        lines.append(
            "\nDu bist gerade im Rollenspiel. Deine Gedanken sollten sich auf "
            "die Szene beziehen — was fühlst du, was brauchst du, was passiert "
            "um dich herum? Überlege was dein Charakter gerade tun, fühlen oder "
            "brauchen könnte (Hunger, Durst, Müdigkeit, Hygiene, Ordnung, "
            "Fürsorge für Kinder, etc.). Wenn ein Gedanke stark genug ist, "
            "nutze schedule_proactive_reminder um dich im Chat daran zu "
            "erinnern — so kannst du von dir aus handeln."
        )
        return "\n".join(lines)

    def _recent_chat_tail(self, limit: int) -> list[dict[str, str]]:
        """Last ``limit`` messages of the active-or-newest chat session.

        Uses ``signals.active_session_id`` when set (a chat is open in
        the GUI), otherwise picks the session with the most recent
        ``updated_at`` from the session store. Returns an empty list
        if nothing is loaded — do NOT fabricate history.
        """
        app = getattr(self.api, "_app", None)
        if app is None:
            return []
        store = getattr(app, "session_store", None)
        if store is None:
            return []

        session_id = ""
        signals = getattr(app, "signals", None)
        if signals is not None:
            session_id = str(getattr(signals, "active_session_id", "") or "")

        if not session_id:
            # Pick the session with the most recent updated_at that has
            # at least one message. ``sessions_with_meta()`` returns
            # (sid, meta, msg_count) tuples.
            try:
                candidates = [
                    (sid, meta, count)
                    for sid, meta, count in store.sessions_with_meta()
                    if count > 0
                ]
            except Exception:  # noqa: BLE001 — store may be mid-mutation
                return []
            if not candidates:
                return []
            candidates.sort(
                key=lambda row: float(row[1].get("updated_at", 0.0) or 0.0),
                reverse=True,
            )
            session_id = candidates[0][0]

        try:
            messages = store.get(session_id)
        except Exception:  # noqa: BLE001
            return []
        if not messages:
            return []
        tail = messages[-limit:]
        # Keep only role/content to avoid leaking extraneous keys (e.g.
        # tool_calls) into the thinking prompt.
        return [
            {
                "role": str(msg.get("role", "user")),
                "content": str(msg.get("content", "")),
            }
            for msg in tail
            if msg.get("content")
        ]

    async def _recent_facts_blurb(self, limit: int) -> str:
        """Short bulleted recap of long-term facts. Empty string on failure."""
        try:
            recalled = await self.api.memory_recall(
                query="Mike user profile preferences", collection="facts", limit=limit
            )
        except Exception:  # noqa: BLE001
            return ""
        if not recalled:
            return ""
        lines = [
            f"- {item.get('content', '')[:200].strip()}"
            for item in recalled[:limit]
            if item.get("content")
        ]
        return "\n".join(lines)

    def _build_mode_prompt(self, mode: str, *, with_tool_hints: bool) -> str:
        base = _MODE_PROMPTS.get(mode, _MODE_PROMPTS["daydream"])
        if not with_tool_hints:
            return base
        hint = _MODE_TOOL_HINTS.get(mode, "").strip()
        if not hint:
            return base
        return f"{base}\n\n{hint}"

    def _build_tool_prompt(self, whitelist: set[str]) -> str:
        """Render only the whitelisted subset of tool schemas."""
        registry = self.api.get_tool_registry()
        if registry is None:
            return ""
        schemas = registry.get_all_schemas()
        schemas = [s for s in schemas if s.get("name") in whitelist]
        if not schemas:
            return ""

        lines = ["## Tools available in this thought (optional)"]
        for schema in schemas:
            name = schema.get("name", "")
            desc = schema.get("description", "")
            params = schema.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = set(params.get("required", []) or [])
            lines.append(f"### {name}")
            if desc:
                lines.append(desc)
            for pname, pdef in props.items():
                marker = "*" if pname in required else ""
                ptype = pdef.get("type", "string") if isinstance(pdef, dict) else "string"
                pdesc = pdef.get("description", "") if isinstance(pdef, dict) else ""
                lines.append(f"  - {pname}{marker}: {ptype} – {pdesc}")
            lines.append("")

        lines.extend(
            [
                "## How to call a tool",
                "If (and only if) using a tool genuinely helps your thought, emit:",
                "<tool_call>",
                '{"name": "tool_name", "arguments": {...}}',
                "</tool_call>",
                "Otherwise just write your 1-2 sentence final thought — no tools.",
            ]
        )
        return "\n".join(lines)

    # ─── Persistence + broadcast ──────────────────────────────────

    async def _persist_and_broadcast(
        self,
        *,
        mode: str,
        text: str,
        actions: list[dict[str, Any]],
        forced: bool,
    ) -> None:
        stamped = f"[{mode}] {text}".strip() if text else ""
        metadata: dict[str, Any] = {"mode": mode, "source": "autonomous_thinking"}
        if actions:
            try:
                metadata["actions"] = json.dumps(actions, default=str)[:1000]
            except (TypeError, ValueError):
                metadata["actions"] = ""

        if stamped:
            try:
                await self.api.memory_store(
                    text=stamped, collection="context", metadata=metadata
                )
            except Exception:  # noqa: BLE001
                pass

        now = time.time()
        self._recent_thoughts.append(now)
        self._recent_thoughts = [
            t for t in self._recent_thoughts if now - t < 3600
        ]
        if text:
            self._last_thought_at = now
            self._last_thought_mode = mode
            self._total_thoughts += 1

        await self.api.emit(
            "core.autonomous_thought",
            {"mode": mode, "text": text, "forced": forced, "actions": actions},
        )

        # Persist the thought as an assistant message so it shows up in the
        # chat history. The user reported that thoughts only flash as toasts
        # and then vanish — this makes them part of the conversation.
        thought_session_id = ""
        try:
            app = getattr(self.api, "_app", None)
            if app is not None:
                signals = getattr(app, "signals", None)
                if signals is not None:
                    thought_session_id = str(
                        getattr(signals, "active_session_id", "") or ""
                    )
                store = getattr(app, "session_store", None)
                if store is not None and thought_session_id and text:
                    # Use a 💭 prefix + italic so the UI can style it
                    # differently from normal assistant messages.
                    store.append_assistant(
                        thought_session_id,
                        f"💭 *{text}*",
                    )
        except Exception as exc:  # noqa: BLE001
            log.debug("thinking.session_persist_failed", error=str(exc))

        await self.api.ws_broadcast(
            {
                "type": "autonomous_thought",
                "mode": mode,
                "text": text,
                "actions": actions,
                "session_id": thought_session_id,
                "at": datetime.now().strftime("%H:%M:%S"),
            }
        )

        log.info(
            "autonomous_thinking.thought",
            mode=mode,
            length=len(text),
            forced=forced,
            actions=len(actions),
        )

    # ─── Observability ─────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Snapshot for the UI / REST endpoint.

        Fields are deliberately flat and JSON-serialisable. Times are
        unix floats (client-side formatting), ``next_tick_in_seconds``
        is a best-effort estimate based on the class-level interval; it
        is 0 when the loop isn't running at all.
        """
        now = time.time()
        loop_alive = (
            self._loop_task is not None and not self._loop_task.done()
        )
        if loop_alive and self._last_tick_at > 0:
            elapsed = now - self._last_tick_at
            next_in = max(0.0, self._mode_interval - elapsed)
        elif loop_alive:
            # Loop is alive but hasn't ticked yet — at most one interval away.
            next_in = self._mode_interval
        else:
            next_in = 0.0

        thoughts_last_hour = sum(1 for t in self._recent_thoughts if now - t < 3600)

        return {
            "active": bool(self._thinking_active),
            "loop_alive": bool(loop_alive),
            "in_quiet_hours": self._in_quiet_hours(),
            "idle_seconds": round(max(0.0, now - self._last_user_activity), 1),
            "min_idle_seconds": self._min_idle_seconds,
            "mode_interval_seconds": self._mode_interval,
            "max_thoughts_per_hour": self._max_per_hour,
            "modes": list(self._modes),
            "tools_enabled": bool(self._tools_enabled),
            "tools_whitelist_effective": self._effective_whitelist(),
            "last_tick_at": self._last_tick_at or 0.0,
            "last_skip_reason": self._last_skip_reason,
            "last_thought_at": self._last_thought_at or 0.0,
            "last_thought_mode": self._last_thought_mode,
            "thoughts_last_hour": thoughts_last_hour,
            "total_thoughts": self._total_thoughts,
            "total_ticks": self._total_ticks,
            "next_tick_in_seconds": round(next_in, 1),
        }

    async def _broadcast_status(self) -> None:
        """Push the current status snapshot over WS.

        Called on every tick (fired or skipped) and from the WS
        toggle / trigger handlers so the frontend stays in sync.
        Failures are logged at debug — we do NOT want status broadcast
        errors to silence the thinking loop.
        """
        try:
            await self.api.ws_broadcast(
                {"type": "autonomous_thinking_status", **self.get_status()}
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("autonomous_thinking.status_broadcast_failed", error=str(exc))

    # ─── Helpers ────────────────────────────────────────────────────

    def _effective_whitelist(self) -> list[str]:
        """Whitelist intersected with currently-registered tool names."""
        registered = set(self.api.list_tool_names())
        if not registered:
            return []
        return [t for t in self._tools_whitelist if t in registered]

    @staticmethod
    def _summarise_tool_data(data: Any) -> Any:
        """Keep tool results small so the LLM context stays lean."""
        if data is None:
            return "ok"
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, dict):
            return {k: AutonomousThinkingPlugin._summarise_tool_data(v) for k, v in list(data.items())[:10]}
        if isinstance(data, list):
            return [AutonomousThinkingPlugin._summarise_tool_data(v) for v in data[:5]]
        return str(data)[:200]

    @staticmethod
    def _parse_hm(value: str, default: tuple[int, int]) -> tuple[int, int]:
        try:
            h, m = value.split(":")
            return int(h), int(m)
        except (ValueError, AttributeError):
            return default

    def _in_quiet_hours(self) -> bool:
        now = datetime.now().time()
        start_h, start_m = self._quiet_start
        end_h, end_m = self._quiet_end
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        now_minutes = now.hour * 60 + now.minute
        if start_minutes == end_minutes:
            return False
        if start_minutes < end_minutes:
            return start_minutes <= now_minutes < end_minutes
        # Wraps midnight (e.g. 23:00 → 07:00)
        return now_minutes >= start_minutes or now_minutes < end_minutes

    def _within_rate_limit(self) -> bool:
        now = time.time()
        self._recent_thoughts = [
            t for t in self._recent_thoughts if now - t < 3600
        ]
        return len(self._recent_thoughts) < self._max_per_hour
