"""
Lexy AI - LexyAgent (Think → Plan → Execute → Reflect).

Single entry point for chat turns. Coordinates:

* Hooks (``before_user_input`` → ``before_prompt_build`` → … → ``after_response_send``).
* BrainRouter selection (E4B / A4B).
* Memory auto-recall and auto-memorize.
* Tool detection + execution loop (max 4 iterations).
* RepetitionDetector via the LLM streaming layer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, AsyncIterator

from lexy_core.agent.router import BrainRouter
from lexy_core.agent.time_awareness import build_time_awareness_block
from lexy_core.utils.logging import get_logger


# Shared so both ``_plan`` and ``process_proactive`` can format the
# weekday without re-declaring it locally.
_WEEKDAYS_DE: tuple[str, ...] = (
    "Montag", "Dienstag", "Mittwoch", "Donnerstag",
    "Freitag", "Samstag", "Sonntag",
)

if TYPE_CHECKING:
    from lexy_core.app import LexyApp

log = get_logger(module="agent")

_MAX_TOOL_ITERATIONS = 4

# Streaming look-ahead: how many characters we buffer before deciding
# between "plain answer → stream live" and "tool call → buffer silently".
_LOOKAHEAD_BYTES = 80

# Regex matching any known tool-call opener. Used by process_stream to
# decide whether to forward buffered bytes live or hold them back.
_TOOL_OPENER_RE = re.compile(
    r"(?:"
    r"<tool_call>"                    # Lexy native
    r"|<\|tool_call"                  # ChatML / Qwen / Gemma 4
    r"|```tool_code"                  # Llama / Gemma tool fence
    r"|```json"                       # Generic JSON fence (may be tool)
    r'|\{\s*"name"\s*:'               # Bare JSON fallback
    r")",
    re.IGNORECASE,
)


def _has_tool_opener(text: str) -> bool:
    """Return True if ``text`` contains any known tool-call opener."""
    return bool(_TOOL_OPENER_RE.search(text))


# The agent's base system prompt now comes from ``LexyApp.persona`` at
# runtime so users can edit Lexy's personality in ``config/persona.yaml``
# or via the Settings GUI without touching code. This constant is kept
# as a fallback only — used if ``app.persona`` is None for some reason
# (shouldn't happen in practice since LexyApp always loads one).
_FALLBACK_SYSTEM_PROMPT = (
    "Du bist Lexy, eine lokale KI-Begleiterin. Rede natürlich auf Deutsch, "
    "sei ehrlich, sei du selbst."
)


class LexyAgent:
    """The orchestrator that turns one user message into one assistant reply."""

    def __init__(self, app: "LexyApp") -> None:
        self._app = app
        self._router = BrainRouter(app.config.routing)

    # ─── Public entry point ─────────────────────────────────────────

    async def process(
        self,
        text: str,
        session_id: str = "default",
        user_id: str = "default",
        brain: str = "auto",
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run the full Think→Plan→Execute→Reflect loop. Returns a result dict.

        ``attachments`` carries upload manifests from the
        :mod:`lexy_core.uploads` pipeline. Each item is the dict the
        upload handler returned, plus a ``data_url`` for images. The
        agent uses them to:

        * Build a multimodal user message (image_url blocks for vision)
        * Inject document/code excerpts into the prompt
        * Append a transcript line for audio
        * Force the multimodal brain when at least one image is present
        """
        log.info(
            "agent.process",
            session=session_id,
            user=user_id,
            length=len(text),
            attachments=len(attachments or []),
        )

        ctx: dict[str, Any] = {
            "text": text,
            "session_id": session_id,
            "user_id": user_id,
            "brain": brain,
            "tools_used": [],
            "attachments": list(attachments or []),
        }

        # Snapshot the session's previous ``updated_at`` BEFORE we append
        # the user message — otherwise the append would overwrite it with
        # ``now`` and we'd lose the gap. A missing / zero value means a
        # brand-new session; ``time_awareness`` treats that as "fresh".
        prev_meta = self._app.session_store.get_meta(session_id) or {}
        ctx["previous_interaction_at"] = float(
            prev_meta.get("updated_at") or 0.0
        )

        # before_user_input hook
        ctx = await self._app.hooks.execute_modifying("before_user_input", ctx)

        await self._app.event_bus.emit(
            "core.user_message",
            {"text": ctx["text"], "user_id": user_id, "session_id": session_id},
        )

        self._app.signals.update(
            ai_thinking=True,
            current_input=ctx["text"],
            active_session_id=session_id,
        )

        # Persist the user message NOW so a crash mid-response still
        # leaves a trace. History is crash-safe even if _execute raises.
        self._app.session_store.append_user(
            session_id=session_id,
            user_text=ctx["text"],
        )

        # Hook opt-out: a plugin (e.g. character_chat) may decide that
        # this session is managed by itself — in that case it sets
        # ``ctx["skip_agent"] = True`` during the ``before_user_input``
        # hook and produces its own response(s). We still keep the
        # session store + user_message event intact above so downstream
        # (dashboard history, memory auto-learn, etc.) behave normally.
        if ctx.get("skip_agent"):
            self._app.signals.update(ai_thinking=False)
            return {
                "text": "",
                "tools_used": [],
                "brain": ctx.get("brain", "character_mode"),
                "session_id": session_id,
                "skipped": True,
                "skip_reason": ctx.get("skip_reason", "character_mode"),
            }

        try:
            # Think
            recalled = await self._think(ctx)
            ctx["recalled"] = recalled

            # Plan: build messages + brain selection
            # NOTE: _plan reads conversation history; since we already
            # appended the user message above, we must filter it out so
            # it doesn't appear twice (once in history, once as the
            # current user turn).
            messages = await self._plan(ctx)

            # Execute: LLM + tool loop
            response_text = await self._execute(ctx, messages)
            ctx["response"] = response_text

            # Reflect: memory auto-memorize
            await self._reflect(ctx)

            # before_response_send hook
            ctx = await self._app.hooks.execute_modifying("before_response_send", ctx)
            response_text = ctx.get("response", response_text)

            # Persist the assistant reply (user message already stored above).
            self._app.session_store.append_assistant(
                session_id=session_id,
                assistant_text=response_text,
            )

            await self._app.event_bus.emit(
                "core.ai_response",
                {"text": response_text, "session_id": session_id},
            )

            return {
                "text": response_text,
                "tools_used": ctx.get("tools_used", []),
                "brain": ctx.get("brain", "e4b"),
                "session_id": session_id,
            }
        finally:
            self._app.signals.update(ai_thinking=False, current_response=ctx.get("response", ""))
            await self._app.hooks.execute_void("after_response_send", ctx)

    async def process_stream(
        self,
        text: str,
        session_id: str = "default",
        user_id: str = "default",
        brain: str = "auto",
        attachments: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Streaming variant with full tool-execution loop.

        Yields event dicts:

        * ``{"type": "chunk", "text": "..."}``            – assistant text chunk
        * ``{"type": "tool_call", "tool": name, "arguments": {...}}`` – tool triggered
        * ``{"type": "tool_result", "tool": name, "text": "..."}``    – tool result
        * ``{"type": "error",  "error": "..."}``          – fatal error
        * ``{"type": "done",   "tools_used": [...]}``     – turn finished

        Strategy
        --------
        Each iteration runs an LLM stream with a *smart look-ahead buffer*:
        the first 80 characters are held back; if they do not contain a
        tool-call opener (``<tool_call>``, ``<|tool_call``, ``` ```tool_code ```,
        ``` ```json ```, or a bare ``{"name":``), the buffer is flushed and
        subsequent chunks are yielded live. If a tool opener *is* detected,
        the whole response is buffered silently, tools are executed, and the
        next iteration runs against the updated message list.

        This gives instant streaming for plain answers and clean (non-raw)
        output for tool-using answers, up to ``_MAX_TOOL_ITERATIONS`` rounds.
        """
        ctx: dict[str, Any] = {
            "text": text,
            "session_id": session_id,
            "user_id": user_id,
            "brain": brain,
            "tools_used": [],
            "attachments": list(attachments or []),
        }

        # Snapshot previous updated_at BEFORE appending the user message,
        # so _plan can compute "time since last conversation". See process()
        # for the mirrored block in the non-streaming path.
        prev_meta = self._app.session_store.get_meta(session_id) or {}
        ctx["previous_interaction_at"] = float(
            prev_meta.get("updated_at") or 0.0
        )

        ctx = await self._app.hooks.execute_modifying("before_user_input", ctx)

        await self._app.event_bus.emit(
            "core.user_message",
            {"text": ctx["text"], "user_id": user_id, "session_id": session_id},
        )
        self._app.signals.update(
            ai_thinking=True,
            current_input=ctx["text"],
            active_session_id=session_id,
        )

        # Persist the user message NOW so a crash mid-stream still leaves
        # a trace. The assistant reply is appended only after the stream
        # completes successfully below.
        self._app.session_store.append_user(
            session_id=session_id,
            user_text=ctx["text"],
        )

        # Plugin-owned session? Skip the normal agent pipeline entirely.
        # See process() for the non-streaming mirror of this path.
        if ctx.get("skip_agent"):
            self._app.signals.update(ai_thinking=False)
            yield {
                "type": "done",
                "tools_used": [],
                "skipped": True,
                "skip_reason": ctx.get("skip_reason", "character_mode"),
            }
            return

        try:
            recalled = await self._think(ctx)
            ctx["recalled"] = recalled

            messages = await self._plan(ctx)
            brain_name = ctx.get("brain", "auto")

            if self._app.llm is None:
                yield {"type": "error", "error": "LLM not initialised"}
                return

            tools_used: list[str] = []
            final_text = ""

            # Persona-level thinking toggle overrides the per-brain setting.
            persona = getattr(self._app, "persona", None)
            thinking_enabled = (
                persona.thinking_enabled if persona is not None
                else self._app.config.get_brain(brain_name).thinking
            )

            # Inform the frontend which brain was picked so the UI can
            # badge the message (e4b vs a4b, complexity/rule/explicit).
            yield {
                "type": "brain",
                "brain": brain_name,
                "thinking": thinking_enabled,
            }

            # Build LLM overrides from persona settings
            _llm_overrides: dict[str, Any] = {}
            if persona is not None:
                _llm_overrides["thinking"] = persona.thinking_enabled

            for iteration in range(_MAX_TOOL_ITERATIONS):
                accumulated = ""
                committed = False

                async for event in self._stream_with_lookahead(
                    messages, brain_name, **_llm_overrides
                ):
                    kind = event["kind"]
                    if kind == "reasoning":
                        yield {"type": "reasoning", "text": event["text"]}
                    elif kind == "content":
                        yield {"type": "chunk", "text": event["text"]}
                    elif kind == "final":
                        accumulated = event["accumulated"]
                        committed = event["committed"]

                # Scenario A — live-streamed plain answer. Done with this turn.
                if committed:
                    final_text = accumulated
                    break

                # Scenario B — buffered silently. Check for tool calls.
                calls: list[Any] = []
                if self._app.tool_caller is not None:
                    calls = self._app.tool_caller.detect_all(accumulated)

                if not calls:
                    clean = (
                        self._app.tool_caller.strip_tool_call(accumulated)
                        if self._app.tool_caller is not None
                        else accumulated
                    )
                    if clean:
                        yield {"type": "chunk", "text": clean}
                    final_text = clean
                    break

                # Tool calls detected — execute them and loop.
                messages.append({"role": "assistant", "content": accumulated})
                for call in calls:
                    tools_used.append(call.name)
                    yield {
                        "type": "tool_call",
                        "tool": call.name,
                        "arguments": call.arguments,
                    }
                    result_text = await self._app.tool_caller.execute_and_format(call)
                    messages.append({"role": "user", "content": result_text})
                    yield {
                        "type": "tool_result",
                        "tool": call.name,
                        "text": result_text,
                    }
                    await self._app.event_bus.emit(
                        "core.tool_executed",
                        {"tool_name": call.name, "result": result_text},
                    )
            else:
                # Hit the iteration cap — make a final plain call.
                log.warning(
                    "agent.stream_tool_loop_max",
                    session=session_id,
                )
                tail = await self._app.llm.chat(messages, brain=brain_name)
                if self._app.tool_caller is not None:
                    tail = self._app.tool_caller.strip_tool_call(tail)
                if tail:
                    yield {"type": "chunk", "text": tail}
                final_text = tail

            ctx["response"] = final_text
            ctx["tools_used"] = tools_used

            ctx = await self._app.hooks.execute_modifying("before_response_send", ctx)
            await self._reflect(ctx)

            # Persist the assistant reply (user message already stored at
            # the start of the stream, so it survives mid-stream crashes).
            self._app.session_store.append_assistant(
                session_id=session_id,
                assistant_text=final_text,
            )

            await self._app.event_bus.emit(
                "core.ai_response",
                {"text": final_text, "session_id": session_id},
            )
            yield {"type": "done", "tools_used": tools_used}
        finally:
            self._app.signals.update(
                ai_thinking=False,
                current_response=ctx.get("response", ""),
            )
            await self._app.hooks.execute_void("after_response_send", ctx)

    # ─── Smart look-ahead streaming helper ──────────────────────────

    async def _stream_with_lookahead(
        self,
        messages: list[dict[str, str]],
        brain: str,
        **overrides: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator that yields a mix of live events and one final
        summary event for a single LLM streaming call:

        * ``{"kind": "reasoning", "text": "..."}`` — reasoning chunk (live)
        * ``{"kind": "content",   "text": "..."}`` — forwardable content chunk (live)
        * ``{"kind": "final", "accumulated": str, "streamed_count": int,
             "reasoning": str, "committed": bool}``
          — emitted exactly once at the end. ``committed=True`` means the
            look-ahead decided plain answer and streamed content chunks
            live; ``False`` means the response was buffered silently for
            tool detection.

        Content state machine
        ---------------------
        ``buffering``  – waiting for enough characters to decide
        ``streaming``  – plain answer, every chunk is forwarded live
        ``silent``     – tool-opener detected, buffer without forwarding
        """
        if self._app.llm is None:
            yield {
                "kind": "final",
                "accumulated": "",
                "streamed_count": 0,
                "reasoning": "",
                "committed": False,
            }
            return

        accumulated = ""
        streamed_count = 0
        reasoning_full = ""
        state = "buffering"
        pending_buffer = ""  # holds the first N chars until we commit to a state

        async for kind, chunk in self._app.llm.chat_stream_structured(
            messages, brain=brain, **overrides
        ):
            if kind == "reasoning":
                reasoning_full += chunk
                yield {"kind": "reasoning", "text": chunk}
                continue

            # kind == "content"
            accumulated += chunk

            if state == "streaming":
                streamed_count += 1
                yield {"kind": "content", "text": chunk}
                continue

            if state == "silent":
                continue

            # state == "buffering"
            pending_buffer += chunk
            if len(accumulated) >= _LOOKAHEAD_BYTES:
                if _has_tool_opener(accumulated):
                    state = "silent"
                else:
                    state = "streaming"
                    streamed_count += 1
                    yield {"kind": "content", "text": pending_buffer}
                pending_buffer = ""

        # Stream ended while still buffering — decide now.
        # IMPORTANT: short answers (< _LOOKAHEAD_BYTES) that are plain
        # content must *not* be live-streamed here, because the Agent's
        # outer loop also handles Scenario B ("buffered silently, no
        # tools, flush"). Yielding the pending_buffer a second time
        # would cause duplicate output. We just commit silently; the
        # outer loop will do the flush from `accumulated`.
        if state == "buffering":
            if _has_tool_opener(accumulated):
                state = "silent"
            # else: stay in buffering → committed=False → outer loop flushes

        committed = state == "streaming"
        yield {
            "kind": "final",
            "accumulated": accumulated,
            "streamed_count": streamed_count,
            "reasoning": reasoning_full,
            "committed": committed,
        }

    # ─── Pipeline stages ────────────────────────────────────────────

    async def _think(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Auto-recall memories relevant to the current user message.

        Also resolves the session's owning project and attaches it to
        ``ctx`` so ``_plan`` can append a per-project persona-override.
        Recall is scoped to that project when ``project.memory_scoped`` is
        true; otherwise we fall back to cross-project recall so projects
        that opt out (e.g. the default "Allgemein") still see everything.
        """
        session_id = ctx.get("session_id", "default")

        # Resolve current project from the session metadata.
        project_id: str | None = None
        project = None
        try:
            sm = self._app.session_store.get_meta(session_id)
            candidate = sm.get("project_id") if isinstance(sm, dict) else None
            if isinstance(candidate, str) and candidate:
                project_id = candidate
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "agent.project_lookup_failed",
                session=session_id,
                error=str(exc),
            )
        if project_id is None:
            project_id = "default"

        project_store = getattr(self._app, "project_store", None)
        if project_store is not None:
            try:
                project = project_store.get(project_id)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "agent.project_fetch_failed",
                    project_id=project_id,
                    error=str(exc),
                )

        ctx["project_id"] = project_id
        ctx["project"] = project

        if self._app.memory is None:
            return []

        # Scope recall by project if the project wants memory isolation.
        # When the project is missing or explicitly cross-scoped, we recall
        # everything (``scope=None``) so the user never accidentally loses
        # access to pre-existing memories.
        scope: str | None = None
        if project is not None and project.memory_scoped:
            scope = project_id

        try:
            return await self._app.memory.recall(
                query=ctx["text"],
                limit=5,
                project_id=scope,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("agent.recall_failed", error=str(exc))
            return []

    async def _plan(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the message list and decide on a brain."""
        attachments: list[dict[str, Any]] = ctx.get("attachments") or []
        has_images = any(
            (a or {}).get("kind") == "image" for a in attachments
        )
        brain, reason = self._router.route(
            ctx["text"], ctx.get("brain", "auto"), has_images=has_images
        )
        ctx["brain"] = brain
        await self._app.event_bus.emit(
            "core.brain_routed", {"brain": brain, "reason": reason},
        )

        # Pull the persona prompt from app state (user-editable). Fall
        # back to the tiny inline constant only if someone forgets to
        # load one — should never happen in practice.
        persona = getattr(self._app, "persona", None)
        persona_prompt = (
            persona.rendered_system_prompt() if persona is not None
            else _FALLBACK_SYSTEM_PROMPT
        )
        system_parts: list[str] = [persona_prompt]

        # Project-level persona override: a short, user-editable text
        # block that gets APPENDED to Lexy's base persona so she stays in
        # character but also knows she's operating inside a named project
        # ("Spielefirma", "Arbeit", …). Empty strings are skipped.
        project = ctx.get("project")
        if project is not None:
            override = (project.persona_override or "").strip()
            project_name = (project.name or "").strip()
            if override:
                header = f"## Projekt-Kontext: {project_name}" if project_name else "## Projekt-Kontext"
                system_parts.append(f"{header}\n{override}")

        # Inject current date + time so Lexy can answer "Wie spät ist
        # es?" and "Welcher Tag ist heute?" correctly.
        from datetime import datetime as _dt

        now = _dt.now()
        weekday = _WEEKDAYS_DE[now.weekday()]
        system_parts.append(
            f"Aktuelles Datum und Uhrzeit: {weekday}, "
            f"{now.strftime('%d.%m.%Y %H:%M:%S')} Uhr (lokal)."
        )

        # Zeitgefühl: how long has it been since the last chat turn in
        # this session? The block is empty for fresh sessions / mid-flow
        # moments at ordinary daytime — it only appears when there's
        # something interesting to notice (real gap or unusual time).
        prev_ts = float(ctx.get("previous_interaction_at") or 0.0)
        awareness_block = build_time_awareness_block(
            previous_ts=prev_ts,
            now_dt=now,
            weekday_de=weekday,
        )
        if awareness_block:
            system_parts.append(awareness_block)

        # Tool prompt
        if self._app.tool_caller is not None:
            tool_prompt = self._app.tool_caller.build_tool_prompt()
            if tool_prompt:
                system_parts.append(tool_prompt)

        # Recalled memory
        recalled: list[dict[str, Any]] = ctx.get("recalled", [])
        if recalled:
            mem_lines = ["Relevant memories:"]
            for item in recalled[:5]:
                mem_lines.append(f"- ({item.get('collection', '?')}) {item['content']}")
            system_parts.append("\n".join(mem_lines))

        ctx["system_prompt_parts"] = system_parts

        # before_prompt_build hook can mutate parts
        ctx = await self._app.hooks.execute_modifying("before_prompt_build", ctx)
        system_parts = ctx.get("system_prompt_parts", system_parts)

        # Conversation history from the session store.
        #
        # We appended the current user turn at the start of ``process()``
        # so it survives a crash. The last history entry is therefore the
        # message we are about to send as the current user turn — drop it
        # here to avoid sending it twice.
        history = self._app.session_store.get(ctx["session_id"])
        if (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == ctx["text"]
        ):
            history = history[:-1]

        # Defensive cleanup: strip any leftover tool-call markers from
        # assistant turns stored in history.
        if history and self._app.tool_caller is not None:
            cleaned_history: list[dict[str, str]] = []
            for msg in history:
                if msg.get("role") == "assistant":
                    clean = self._app.tool_caller.strip_tool_call(
                        msg.get("content", "")
                    )
                    if clean:
                        cleaned_history.append({"role": "assistant", "content": clean})
                else:
                    cleaned_history.append(msg)
            history = cleaned_history

        # Build the user message. If the turn carries image attachments
        # we switch to a multimodal content array; non-image attachments
        # (docs/code/audio) are folded into the text since the LLM only
        # needs the parsed excerpt + a hint where it came from.
        user_text = ctx["text"]
        attachment_text = _render_non_image_attachments(attachments)
        if attachment_text:
            user_text = f"{user_text}\n\n{attachment_text}" if user_text else attachment_text

        user_message: dict[str, Any]
        image_blocks = _build_image_blocks(attachments)
        if image_blocks:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text or "(siehe Anhang)"},
                    *image_blocks,
                ],
            }
        else:
            user_message = {"role": "user", "content": user_text}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            *history,
            user_message,
        ]
        ctx["messages"] = messages
        ctx["history_length"] = len(history)
        log.debug(
            "agent.plan",
            session=ctx.get("session_id"),
            history_msgs=len(history),
            brain=brain,
        )
        return messages

    async def _execute(
        self, ctx: dict[str, Any], messages: list[dict[str, str]]
    ) -> str:
        """LLM call + tool loop (max N iterations)."""
        if self._app.llm is None:
            raise RuntimeError("LLM client not initialised")
        if self._app.tool_caller is None:
            response = await self._app.llm.chat(messages, brain=ctx["brain"])
            return response

        tools_used: list[str] = []
        current_messages = list(messages)

        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await self._app.llm.chat(current_messages, brain=ctx["brain"])

            calls = self._app.tool_caller.detect_all(response)
            if not calls:
                ctx["tools_used"] = tools_used
                return self._app.tool_caller.strip_tool_call(response)

            current_messages.append({"role": "assistant", "content": response})

            for call in calls:
                tools_used.append(call.name)
                result_text = await self._app.tool_caller.execute_and_format(call)
                current_messages.append({"role": "user", "content": result_text})
                await self._app.event_bus.emit(
                    "core.tool_executed",
                    {"tool_name": call.name, "result": result_text},
                )

            log.debug(
                "agent.tool_iteration",
                iteration=iteration + 1,
                calls=[call.name for call in calls],
            )

        # Max iterations reached
        ctx["tools_used"] = tools_used
        log.warning("agent.tool_loop_max", session=ctx.get("session_id"))
        return await self._app.llm.chat(current_messages, brain=ctx["brain"])

    # ─── Proactive (scheduler-triggered) ────────────────────────────

    async def process_proactive(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make the agent speak unprompted in an existing session.

        Triggered by the scheduler (or any plugin via
        :meth:`PluginAPI.agent_proactive`). The ``payload`` carries the
        internal trigger; no user message is persisted. The generated
        assistant text is appended to the session and broadcast to all
        WebSocket clients so the GUI shows Lexy's message popping in.

        Expected keys in ``payload``:

        * ``session_id`` – session to speak into (required; falls back to
          ``"default"``)
        * ``text`` – internal trigger / prompt the scheduler handed over
        * ``label`` – optional human-readable label of the schedule entry
        * ``from`` – source name (e.g. ``"scheduler"``) for logs
        """
        session_id = str(payload.get("session_id") or "default")
        trigger_text = str(payload.get("text") or "").strip()
        label = str(payload.get("label") or "").strip()
        source = str(payload.get("from") or "scheduler")

        log.info(
            "agent.proactive",
            session=session_id,
            source=source,
            label=label,
            prompt_preview=trigger_text[:80],
        )

        if self._app.llm is None:
            log.warning("agent.proactive_no_llm")
            return {"ok": False, "reason": "llm_unavailable"}

        persona = getattr(self._app, "persona", None)
        persona_prompt = (
            persona.rendered_system_prompt() if persona is not None
            else _FALLBACK_SYSTEM_PROMPT
        )

        # Resolve project-context so Lexy still stays in-character for
        # the target project's persona override.
        project_prompt = ""
        try:
            sm = self._app.session_store.get_meta(session_id)
            project_id = sm.get("project_id") if isinstance(sm, dict) else None
            project_store = getattr(self._app, "project_store", None)
            if project_id and project_store is not None:
                project = project_store.get(project_id)
                if project is not None and (project.persona_override or "").strip():
                    header = (
                        f"## Projekt-Kontext: {project.name}"
                        if project.name else "## Projekt-Kontext"
                    )
                    project_prompt = f"{header}\n{project.persona_override.strip()}"
        except Exception as exc:  # noqa: BLE001
            log.debug("agent.proactive_project_lookup_failed", error=str(exc))

        trigger_line = (
            f"Du wirst vom {source} ausgelöst: {trigger_text}"
            if trigger_text
            else f"Du wirst vom {source} ausgelöst."
        )
        system_parts = [persona_prompt]
        if project_prompt:
            system_parts.append(project_prompt)

        # Time-awareness block: especially useful here because Lexy is the
        # one speaking — knowing that the last exchange was four hours ago
        # lets her pick a natural greeting instead of diving in cold.
        try:
            prev_ts = 0.0
            sm_meta = self._app.session_store.get_meta(session_id)
            if isinstance(sm_meta, dict):
                prev_ts = float(sm_meta.get("updated_at") or 0.0)
            from datetime import datetime as _dt

            now_dt = _dt.now()
            weekday_de = _WEEKDAYS_DE[now_dt.weekday()]
            awareness = build_time_awareness_block(
                previous_ts=prev_ts,
                now_dt=now_dt,
                weekday_de=weekday_de,
            )
            if awareness:
                system_parts.append(awareness)
        except Exception as exc:  # noqa: BLE001
            log.debug("agent.proactive_time_block_failed", error=str(exc))

        history = self._app.session_store.get(session_id)
        # Keep the last few turns for context so Lexy's proactive line
        # feels coherent with the ongoing chat.
        recent = history[-8:] if history else []

        # The trigger nudge goes as the FINAL message (role=user) so the
        # chat history always ends with a user turn. Two reasons:
        # 1. llama.cpp rejects ``enable_thinking`` + trailing assistant
        #    message with "Assistant response prefill is incompatible with
        #    enable_thinking." → without this, Lexy's proactive turns
        #    after a character-turn silently fail with 400.
        # 2. It's cleaner semantics: the trigger is what Lexy should react
        #    to *now*, not part of her standing persona.
        trigger_user_content = (
            f"[Interner Trigger von {source}] {trigger_line}\n\n"
            "Melde dich jetzt kurz und natürlich bei Mike, als wäre dir "
            "der Gedanke gerade gekommen. Maximal 1–2 Sätze auf Deutsch. "
            "Kein Meta-Kommentar dass du ein Reminder/Trigger bist."
        )

        messages = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            *recent,
            {"role": "user", "content": trigger_user_content},
        ]

        try:
            response = await self._app.llm.chat(messages, brain="e4b")
        except Exception as exc:  # noqa: BLE001
            log.error("agent.proactive_llm_failed", error=str(exc))
            return {"ok": False, "reason": str(exc)}

        if self._app.tool_caller is not None:
            response = self._app.tool_caller.strip_tool_call(response)
        response = response.strip()
        if not response:
            log.warning("agent.proactive_empty_response")
            return {"ok": False, "reason": "empty_response"}

        # Persist as an assistant message without a preceding user turn.
        self._app.session_store.append_assistant(
            session_id=session_id,
            assistant_text=response,
        )

        if self._app.ws_server is not None:
            await self._app.ws_server.broadcast({
                "type": "proactive_message",
                "session_id": session_id,
                "text": response,
                "from": source,
                "label": label,
            })

        await self._app.event_bus.emit(
            "core.proactive_message",
            {
                "session_id": session_id,
                "text": response,
                "from": source,
                "label": label,
            },
        )
        return {"ok": True, "text": response, "session_id": session_id}

    async def _reflect(self, ctx: dict[str, Any]) -> None:
        """Auto-memorize the (user, assistant) pair.

        The project id resolved in ``_think`` is written alongside the
        usual session/user metadata so follow-up ``recall`` calls in the
        same project can surface the exchange again.
        """
        if self._app.memory is None:
            return
        metadata: dict[str, Any] = {
            "session_id": ctx.get("session_id", "default"),
            "user_id": ctx.get("user_id", "default"),
            "brain": ctx.get("brain", "e4b"),
        }
        project_id = ctx.get("project_id")
        if isinstance(project_id, str) and project_id:
            metadata["project_id"] = project_id

        try:
            await self._app.memory.store(
                text=f"User: {ctx['text']}\nLexy: {ctx.get('response', '')}",
                collection="context",
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("agent.memorize_failed", error=str(exc))



# ─── Attachment helpers ───────────────────────────────────────────────


def _build_image_blocks(
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return ``image_url`` content blocks for multimodal LLM messages.

    Skips entries that don't look like images, or have no usable URL /
    data URL. The caller passes the result inline into the user-content
    array; if there are no images, the chat stays plain-text.
    """
    blocks: list[dict[str, Any]] = []
    for item in attachments or []:
        if not isinstance(item, dict) or item.get("kind") != "image":
            continue
        url = item.get("data_url") or item.get("url") or ""
        if not url:
            continue
        blocks.append(
            {"type": "image_url", "image_url": {"url": url}}
        )
    return blocks


def _render_non_image_attachments(
    attachments: list[dict[str, Any]],
) -> str:
    """Format doc/code/audio attachments as a text block for the prompt.

    Images are handled separately via ``_build_image_blocks`` and so are
    skipped here. Each non-image item gets one short labelled block; the
    LLM sees it as an inline reference like:

        [Datei] readme.md (document, 12 KB) — first 400 chars excerpt …
    """
    if not attachments:
        return ""
    lines: list[str] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind") or ""
        if kind == "image":
            continue
        filename = item.get("filename") or "(file)"
        size = item.get("size") or 0
        size_kb = max(1, size // 1024) if size else 0
        size_label = f"{size_kb} KB" if size else "?"
        if kind == "document":
            excerpt = (item.get("excerpt") or "").strip()
            chunks = item.get("chunks_indexed") or 0
            head = (
                f"[Datei] {filename} (document, {size_label}, "
                f"{chunks} chunks indexed)"
            )
            body = excerpt if excerpt else "(no extractable text)"
            lines.append(f"{head}\n{body}")
        elif kind == "code":
            language = item.get("language") or ""
            lang_str = f" {language}" if language else ""
            line_count = item.get("lines") or 0
            excerpt = (item.get("excerpt") or "").strip()
            head = (
                f"[Datei] {filename} (code{lang_str}, {line_count} lines)"
            )
            body = excerpt if excerpt else "(empty)"
            lines.append(f"{head}\n{body}")
        elif kind == "audio":
            transcript = (item.get("transcript") or "").strip()
            head = f"[Audio] {filename} ({size_label})"
            body = (
                f"Transcript: {transcript}"
                if transcript
                else "(no transcript available)"
            )
            lines.append(f"{head}\n{body}")
        else:
            lines.append(f"[Anhang] {filename} ({kind}, {size_label})")
    if not lines:
        return ""
    return "## Anhänge dieser Nachricht\n" + "\n\n".join(lines)
