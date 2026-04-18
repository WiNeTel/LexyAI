"""
Lexy AI - Expert Panel Plugin.

Multi-agent collaborative discussion system where 3-5 agents with different
perspectives discuss a topic in three phases:

1. **Analysis** -- each role gives an independent first assessment (parallel).
2. **Discussion** -- sequential rounds where each role responds to others.
3. **Synthesis** -- structured summary with consensus, dissent, action items.

Convergence detection can terminate discussion early when enough agreement
points have been reached.

Persistence: SQLite via ``api.get_db()`` (plugin-private).
Real-time updates: WebSocket broadcasts per agent message.
Memory integration: final synthesis stored in ``facts`` collection.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .convergence import ConvergenceDetector
from .panel_session import PanelSession
from .roles import ROLE_COLORS, ROLE_LABELS, ROLE_PROMPTS
from .synthesizer import PanelSynthesizer

log = get_logger(module="expert_panel")


# ─── Tool schemas ────────────────────────────────────────────────────────────

START_PANEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "Das Thema, das die Experten diskutieren sollen",
        },
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Liste der Rollen (analyst, critic, creative, pragmatist, synthesizer). "
                "Standard: alle 5."
            ),
        },
        "rounds": {
            "type": "integer",
            "description": "Anzahl Diskussionsrunden (1-5, Standard: 3)",
        },
        "brain": {
            "type": "string",
            "description": "Welches Brain (e4b/a4b, Standard: e4b)",
        },
    },
    "required": ["topic"],
}

PANEL_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "panel_id": {
            "type": "string",
            "description": "ID des Panels",
        },
    },
    "required": ["panel_id"],
}

PANEL_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "panel_id": {
            "type": "string",
            "description": "ID des Panels",
        },
    },
    "required": ["panel_id"],
}


# ─── Plugin ──────────────────────────────────────────────────────────────────


class ExpertPanelPlugin(BasePlugin):
    """Multi-agent collaborative discussion plugin."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._sessions: dict[str, PanelSession] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._convergence: ConvergenceDetector = ConvergenceDetector()
        self._synthesizer: PanelSynthesizer = PanelSynthesizer()

        # Config-Werte (gesetzt in on_load)
        self._default_roles: list[str] = []
        self._default_rounds: int = 3
        self._max_rounds: int = 5
        self._default_brain: str = "e4b"
        self._convergence_threshold: int = 3
        self._max_tokens: int = 400
        self._temperatures: dict[str, float] = {}

    # ─── Lifecycle ──────────────────────────────────────────────────────

    async def on_load(self) -> None:
        cfg = self.api.get_config()
        self._default_roles = [
            r for r in cfg.get("default_roles", list(ROLE_PROMPTS.keys()))
            if r in ROLE_PROMPTS
        ]
        if not self._default_roles:
            self._default_roles = list(ROLE_PROMPTS.keys())
        self._default_rounds = int(cfg.get("default_rounds", 3))
        self._max_rounds = int(cfg.get("max_rounds", 5))
        self._default_brain = str(cfg.get("default_brain", "e4b"))
        self._convergence_threshold = int(cfg.get("convergence_threshold", 3))
        self._max_tokens = int(cfg.get("max_tokens_per_response", 400))
        self._temperatures = cfg.get("temperature_overrides", {})

        # SQLite-Tabellen anlegen
        db = await self.api.get_db()
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS panels (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                roles TEXT NOT NULL,
                brain TEXT NOT NULL,
                rounds_planned INTEGER NOT NULL,
                rounds_completed INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                finished_at REAL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS panel_messages (
                id TEXT PRIMARY KEY,
                panel_id TEXT NOT NULL,
                role TEXT NOT NULL,
                phase TEXT NOT NULL,
                round_num INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (panel_id) REFERENCES panels(id)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS panel_results (
                panel_id TEXT PRIMARY KEY,
                consensus_points TEXT NOT NULL DEFAULT '[]',
                dissent_points TEXT NOT NULL DEFAULT '[]',
                action_items TEXT NOT NULL DEFAULT '[]',
                summary TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                FOREIGN KEY (panel_id) REFERENCES panels(id)
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_panel_messages_panel "
            "ON panel_messages(panel_id, round_num)"
        )
        await db.commit()
        log.info("expert_panel.tables_ready")

    async def on_enable(self) -> None:
        # LLM Tools
        self.api.register_tool(
            name="start_panel",
            handler=self._tool_start_panel,
            description=(
                "Starte eine Expertenpanel-Diskussion. 3-5 KI-Experten mit "
                "verschiedenen Perspektiven (Analyst, Kritiker, Kreativer, "
                "Pragmatiker, Synthesizer) diskutieren ein Thema in mehreren "
                "Runden und liefern eine strukturierte Synthese."
            ),
            schema=START_PANEL_SCHEMA,
        )
        self.api.register_tool(
            name="panel_status",
            handler=self._tool_panel_status,
            description="Zeige den aktuellen Status eines laufenden Expertenpanels.",
            schema=PANEL_STATUS_SCHEMA,
        )
        self.api.register_tool(
            name="panel_result",
            handler=self._tool_panel_result,
            description=(
                "Hole das Ergebnis eines abgeschlossenen Expertenpanels "
                "(Konsens, Dissens, Action Items, Zusammenfassung)."
            ),
            schema=PANEL_RESULT_SCHEMA,
        )

        # WebSocket Handlers
        self.api.register_ws_handler("panel_start", self._handle_ws_start)
        self.api.register_ws_handler("panel_status_request", self._handle_ws_status)
        self.api.register_ws_handler("panel_cancel", self._handle_ws_cancel)
        self.api.register_ws_handler("panel_list", self._handle_ws_list)

        log.info(
            "expert_panel.enabled",
            default_roles=self._default_roles,
            default_rounds=self._default_rounds,
        )

    async def on_disable(self) -> None:
        # Alle laufenden Panels abbrechen
        for panel_id, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if panel_id in self._sessions:
                self._sessions[panel_id].finish("cancelled")
                await self._persist_panel_status(panel_id, "cancelled")
        self._tasks.clear()
        self._sessions.clear()
        log.info("expert_panel.disabled")

    # ─── Tool handlers ──────────────────────────────────────────────────

    async def _tool_start_panel(
        self,
        topic: str,
        roles: list[str] | None = None,
        rounds: int | None = None,
        brain: str | None = None,
    ) -> dict[str, Any]:
        """Start a new expert panel discussion."""
        # Rollen validieren
        if roles:
            valid_roles = [r for r in roles if r in ROLE_PROMPTS]
            if len(valid_roles) < 2:
                return {"error": "Mindestens 2 gueltige Rollen erforderlich."}
        else:
            valid_roles = list(self._default_roles)

        # Runden clampen
        num_rounds = min(
            max(rounds or self._default_rounds, 1),
            self._max_rounds,
        )
        chosen_brain = brain if brain in ("e4b", "a4b") else self._default_brain

        panel_id = uuid.uuid4().hex[:12]
        session = PanelSession(
            panel_id=panel_id,
            topic=topic,
            roles=valid_roles,
            brain=chosen_brain,
            rounds_planned=num_rounds,
        )
        self._sessions[panel_id] = session

        # In DB persistieren
        db = await self.api.get_db()
        await db.execute(
            """
            INSERT INTO panels (id, topic, status, roles, brain, rounds_planned, rounds_completed, created_at)
            VALUES (?, ?, 'running', ?, ?, ?, 0, ?)
            """,
            (
                panel_id,
                topic,
                json.dumps(valid_roles),
                chosen_brain,
                num_rounds,
                session.created_at,
            ),
        )
        await db.commit()

        # Diskussion als Background-Task starten
        task = asyncio.create_task(
            self._run_panel(panel_id), name=f"expert_panel.{panel_id}"
        )
        self._tasks[panel_id] = task

        log.info(
            "expert_panel.started",
            panel_id=panel_id,
            topic=topic[:80],
            roles=valid_roles,
            rounds=num_rounds,
            brain=chosen_brain,
        )

        return {
            "panel_id": panel_id,
            "topic": topic,
            "roles": valid_roles,
            "role_labels": {r: ROLE_LABELS.get(r, r) for r in valid_roles},
            "rounds": num_rounds,
            "brain": chosen_brain,
            "status": "running",
        }

    async def _tool_panel_status(self, panel_id: str) -> dict[str, Any]:
        """Query current panel state and messages."""
        # Erst im Cache schauen
        if panel_id in self._sessions:
            session = self._sessions[panel_id]
            return {
                **session.to_status_dict(),
                "messages": [m.to_dict() for m in session.messages],
            }

        # Sonst aus der DB laden
        db = await self.api.get_db()
        cursor = await db.execute(
            "SELECT id, topic, status, roles, brain, rounds_planned, "
            "rounds_completed, created_at, finished_at FROM panels WHERE id = ?",
            (panel_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return {"error": f"Panel {panel_id!r} nicht gefunden."}

        # Nachrichten laden
        msg_cursor = await db.execute(
            "SELECT role, phase, round_num, content, created_at "
            "FROM panel_messages WHERE panel_id = ? ORDER BY created_at ASC",
            (panel_id,),
        )
        msg_rows = await msg_cursor.fetchall()
        await msg_cursor.close()

        messages = [
            {
                "role": mr[0],
                "phase": mr[1],
                "round": mr[2],
                "content": mr[3],
                "created_at": mr[4],
            }
            for mr in msg_rows
        ]

        return {
            "panel_id": row[0],
            "topic": row[1],
            "status": row[2],
            "roles": json.loads(row[3]),
            "brain": row[4],
            "rounds_planned": row[5],
            "current_round": row[6],
            "message_count": len(messages),
            "created_at": row[7],
            "finished_at": row[8],
            "messages": messages,
        }

    async def _tool_panel_result(self, panel_id: str) -> dict[str, Any]:
        """Query final panel results."""
        db = await self.api.get_db()
        cursor = await db.execute(
            "SELECT panel_id, consensus_points, dissent_points, action_items, "
            "summary, created_at FROM panel_results WHERE panel_id = ?",
            (panel_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            # Eventuell laeuft es noch
            if panel_id in self._sessions:
                session = self._sessions[panel_id]
                return {
                    "panel_id": panel_id,
                    "status": session.status,
                    "message": "Panel laeuft noch, Ergebnis noch nicht verfuegbar.",
                }
            return {"error": f"Kein Ergebnis fuer Panel {panel_id!r} gefunden."}

        return {
            "panel_id": row[0],
            "consensus_points": json.loads(row[1]),
            "dissent_points": json.loads(row[2]),
            "action_items": json.loads(row[3]),
            "summary": row[4],
            "created_at": row[5],
        }

    # ─── WebSocket handlers ─────────────────────────────────────────────

    async def _handle_ws_start(self, client: Any, message: dict[str, Any]) -> None:
        """Start a panel from the GUI."""
        topic = str(message.get("topic", ""))
        if not topic:
            await client.send_json({"type": "error", "error": "Kein Thema angegeben."})
            return
        roles = message.get("roles")
        rounds = message.get("rounds")
        brain = message.get("brain")
        result = await self._tool_start_panel(
            topic=topic,
            roles=roles,
            rounds=int(rounds) if rounds is not None else None,
            brain=str(brain) if brain else None,
        )
        await client.send_json({"type": "panel_started", **result})

    async def _handle_ws_status(self, client: Any, message: dict[str, Any]) -> None:
        """Return panel status to a single client."""
        panel_id = str(message.get("panel_id", ""))
        if not panel_id:
            await client.send_json({"type": "error", "error": "Keine panel_id angegeben."})
            return
        result = await self._tool_panel_status(panel_id)
        await client.send_json({"type": "panel_status", **result})

    async def _handle_ws_cancel(self, client: Any, message: dict[str, Any]) -> None:
        """Cancel a running panel."""
        panel_id = str(message.get("panel_id", ""))
        if not panel_id:
            await client.send_json({"type": "error", "error": "Keine panel_id angegeben."})
            return

        if panel_id not in self._tasks:
            await client.send_json({
                "type": "panel_cancelled",
                "panel_id": panel_id,
                "error": "Panel nicht aktiv oder nicht gefunden.",
            })
            return

        task = self._tasks[panel_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if panel_id in self._sessions:
            self._sessions[panel_id].finish("cancelled")
        await self._persist_panel_status(panel_id, "cancelled")

        await client.send_json({
            "type": "panel_cancelled",
            "panel_id": panel_id,
            "status": "cancelled",
        })
        await self.api.ws_broadcast({
            "type": "panel_done",
            "panel_id": panel_id,
            "status": "cancelled",
        })
        log.info("expert_panel.cancelled", panel_id=panel_id)

    async def _handle_ws_list(self, client: Any, message: dict[str, Any]) -> None:
        """List all panels (active + recent from DB)."""
        db = await self.api.get_db()
        cursor = await db.execute(
            "SELECT id, topic, status, roles, rounds_planned, rounds_completed, "
            "created_at, finished_at FROM panels ORDER BY created_at DESC LIMIT 20"
        )
        rows = await cursor.fetchall()
        await cursor.close()

        panels: list[dict[str, Any]] = []
        for row in rows:
            panels.append({
                "panel_id": row[0],
                "topic": row[1],
                "status": row[2],
                "roles": json.loads(row[3]),
                "rounds_planned": row[4],
                "rounds_completed": row[5],
                "created_at": row[6],
                "finished_at": row[7],
            })

        await client.send_json({"type": "panel_list", "panels": panels})

    # ─── Discussion Protocol ────────────────────────────────────────────

    async def _run_panel(self, panel_id: str) -> None:
        """
        Execute the full discussion protocol for a panel.

        Phase 1: Individual analysis (parallel)
        Phase 2: Discussion rounds (sequential, with convergence check)
        Phase 3: Synthesis (final summary)
        """
        session = self._sessions.get(panel_id)
        if session is None:
            log.error("expert_panel.session_not_found", panel_id=panel_id)
            return

        try:
            # ── Phase 1: Analysis (parallel) ─────────────────────────
            session.current_phase = "analysis"
            log.info("expert_panel.phase_analysis", panel_id=panel_id)

            await self.api.ws_broadcast({
                "type": "panel_phase",
                "panel_id": panel_id,
                "phase": "analysis",
            })

            analysis_tasks: list[asyncio.Task[str | None]] = []
            for role in session.roles:
                task = asyncio.create_task(
                    self._run_agent(
                        panel_id=panel_id,
                        role=role,
                        phase="analysis",
                        round_num=0,
                        topic=session.topic,
                        context="",
                    ),
                    name=f"panel.{panel_id}.analysis.{role}",
                )
                analysis_tasks.append(task)

            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    role = session.roles[i]
                    log.error(
                        "expert_panel.analysis_failed",
                        panel_id=panel_id,
                        role=role,
                        error=str(result),
                    )

            # ── Phase 2: Discussion (sequential rounds) ──────────────
            session.current_phase = "discussion"
            log.info("expert_panel.phase_discussion", panel_id=panel_id)

            await self.api.ws_broadcast({
                "type": "panel_phase",
                "panel_id": panel_id,
                "phase": "discussion",
            })

            for round_num in range(1, session.rounds_planned + 1):
                session.current_round = round_num
                context = self._format_discussion_context(session)

                for role in session.roles:
                    # Pruefe ob abgebrochen
                    if session.status == "cancelled":
                        return

                    await self._run_agent(
                        panel_id=panel_id,
                        role=role,
                        phase="discussion",
                        round_num=round_num,
                        topic=session.topic,
                        context=context,
                    )
                    # Kontext nach jedem Beitrag aktualisieren
                    context = self._format_discussion_context(session)

                # Runde abgeschlossen -- DB updaten
                await self._persist_rounds_completed(panel_id, round_num)

                # Konvergenz-Check nach jeder Runde
                all_messages = [m.to_dict() for m in session.messages]
                convergence = await self._convergence.check(
                    messages=all_messages,
                    roles=session.roles,
                    threshold=self._convergence_threshold,
                    api=self.api,
                    brain=session.brain,
                )

                await self.api.ws_broadcast({
                    "type": "panel_round_done",
                    "panel_id": panel_id,
                    "round": round_num,
                    "rounds_planned": session.rounds_planned,
                    "convergence": convergence,
                })

                log.info(
                    "expert_panel.round_done",
                    panel_id=panel_id,
                    round=round_num,
                    converged=convergence["converged"],
                    agreements=convergence["agreement_count"],
                )

                if convergence["converged"]:
                    log.info(
                        "expert_panel.converged_early",
                        panel_id=panel_id,
                        round=round_num,
                    )
                    break

            # ── Phase 3: Synthesis ───────────────────────────────────
            if session.status == "cancelled":
                return

            session.current_phase = "synthesis"
            session.status = "synthesizing"

            await self.api.ws_broadcast({
                "type": "panel_synthesizing",
                "panel_id": panel_id,
            })

            log.info("expert_panel.phase_synthesis", panel_id=panel_id)

            all_messages = [m.to_dict() for m in session.messages]
            result = await self._synthesizer.synthesize(
                messages=all_messages,
                topic=session.topic,
                api=self.api,
                brain=session.brain,
            )

            # In DB speichern
            await self._persist_result(panel_id, result)

            # In Memory speichern
            memory_text = (
                f"Expertenpanel zu: {session.topic}\n\n"
                f"Zusammenfassung: {result['summary']}\n\n"
                f"Konsens: {'; '.join(result['consensus_points'])}\n"
                f"Dissens: {'; '.join(result['dissent_points'])}\n"
                f"Actions: {'; '.join(result['action_items'])}"
            )
            try:
                await self.api.memory_store(
                    text=memory_text,
                    collection="facts",
                    metadata={
                        "source": "expert_panel",
                        "panel_id": panel_id,
                        "topic": session.topic,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("expert_panel.memory_store_failed", error=str(exc))

            # Session abschliessen
            session.finish("done")
            await self._persist_panel_status(panel_id, "done")

            await self.api.ws_broadcast({
                "type": "panel_done",
                "panel_id": panel_id,
                "status": "done",
                "result": result,
            })

            await self.api.emit(
                "core.expert_panel_done",
                {
                    "panel_id": panel_id,
                    "topic": session.topic,
                    "result": result,
                },
            )

            log.info(
                "expert_panel.done",
                panel_id=panel_id,
                total_messages=len(session.messages),
                rounds_completed=session.current_round,
            )

        except asyncio.CancelledError:
            session.finish("cancelled")
            await self._persist_panel_status(panel_id, "cancelled")
            log.info("expert_panel.cancelled_by_task", panel_id=panel_id)

        except Exception as exc:  # noqa: BLE001
            log.error("expert_panel.run_failed", panel_id=panel_id, error=str(exc))
            session.finish("cancelled")
            await self._persist_panel_status(panel_id, "cancelled")
            await self.api.ws_broadcast({
                "type": "panel_done",
                "panel_id": panel_id,
                "status": "error",
                "error": str(exc),
            })

        finally:
            # Task aus dem Tracking entfernen
            self._tasks.pop(panel_id, None)

    async def _run_agent(
        self,
        panel_id: str,
        role: str,
        phase: str,
        round_num: int,
        topic: str,
        context: str,
    ) -> str | None:
        """
        Run a single agent contribution.

        Builds role-specific messages, calls the LLM, persists the response,
        and broadcasts it via WebSocket.
        """
        session = self._sessions.get(panel_id)
        if session is None:
            return None

        role_prompt = ROLE_PROMPTS.get(role, "")
        temperature = self._temperatures.get(role, 0.5)

        # LLM-Nachrichten bauen
        llm_messages: list[dict[str, str]] = []

        if phase == "analysis":
            llm_messages = [
                {"role": "system", "content": role_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Thema: {topic}\n\n"
                        "Gib deine erste Einschaetzung. Sei praegnant und "
                        "fokussiert (maximal 3-4 Absaetze)."
                    ),
                },
            ]
        else:
            # Discussion: vorherige Beitraege als Kontext
            system_content = (
                f"{role_prompt}\n\n"
                f"Runde {round_num} von {session.rounds_planned}. "
                "Beziehe dich auf die Beitraege der anderen Experten. "
                "Entwickle deine Position weiter oder verteidige sie."
            )
            llm_messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        f"Thema: {topic}\n\n"
                        f"Bisherige Diskussion:\n{context}\n\n"
                        "Dein Beitrag:"
                    ),
                },
            ]

        try:
            response = await self.api.llm_chat(
                llm_messages,
                brain=session.brain,
                max_tokens=self._max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "expert_panel.agent_failed",
                panel_id=panel_id,
                role=role,
                phase=phase,
                round_num=round_num,
                error=str(exc),
            )
            response = f"[{ROLE_LABELS.get(role, role)}: Antwort fehlgeschlagen]"

        response = response.strip()
        if not response:
            response = f"[{ROLE_LABELS.get(role, role)}: Keine Antwort]"

        # Im Session-Objekt speichern
        session.add_message(
            role=role,
            phase=phase,
            round_num=round_num,
            content=response,
        )

        # In DB persistieren
        await self._persist_message(
            panel_id=panel_id,
            role=role,
            phase=phase,
            round_num=round_num,
            content=response,
        )

        # Per WebSocket broadcasten
        await self.api.ws_broadcast({
            "type": "panel_agent_message",
            "panel_id": panel_id,
            "role": role,
            "role_label": ROLE_LABELS.get(role, role),
            "role_color": ROLE_COLORS.get(role, "#888888"),
            "content": response,
            "phase": phase,
            "round": round_num,
        })

        log.debug(
            "expert_panel.agent_response",
            panel_id=panel_id,
            role=role,
            phase=phase,
            round_num=round_num,
            length=len(response),
        )

        return response

    # ─── Context formatting ─────────────────────────────────────────────

    @staticmethod
    def _format_discussion_context(session: PanelSession) -> str:
        """
        Format all previous messages as context for the next agent.

        Groups by phase and round for readability.
        """
        if not session.messages:
            return "(Noch keine Beitraege)"

        parts: list[str] = []
        current_phase = ""
        current_round = -1

        for msg in session.messages:
            # Phasen-/Runden-Header wenn noetig
            if msg.phase != current_phase or msg.round_num != current_round:
                current_phase = msg.phase
                current_round = msg.round_num
                if msg.phase == "analysis":
                    parts.append("--- Erste Einschaetzungen ---")
                else:
                    parts.append(f"--- Runde {msg.round_num} ---")

            label = ROLE_LABELS.get(msg.role, msg.role)
            parts.append(f"[{label}]: {msg.content}")

        return "\n\n".join(parts)

    # ─── DB persistence ─────────────────────────────────────────────────

    async def _persist_message(
        self,
        panel_id: str,
        role: str,
        phase: str,
        round_num: int,
        content: str,
    ) -> None:
        """Persist a single panel message to SQLite."""
        db = await self.api.get_db()
        msg_id = uuid.uuid4().hex[:12]
        await db.execute(
            """
            INSERT INTO panel_messages (id, panel_id, role, phase, round_num, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, panel_id, role, phase, round_num, content, time.time()),
        )
        await db.commit()

    async def _persist_panel_status(self, panel_id: str, status: str) -> None:
        """Update the panel status and finished_at in the DB."""
        db = await self.api.get_db()
        finished_at = time.time() if status in ("done", "cancelled") else None
        await db.execute(
            "UPDATE panels SET status = ?, finished_at = ? WHERE id = ?",
            (status, finished_at, panel_id),
        )
        await db.commit()

    async def _persist_rounds_completed(self, panel_id: str, rounds: int) -> None:
        """Update the rounds_completed counter."""
        db = await self.api.get_db()
        await db.execute(
            "UPDATE panels SET rounds_completed = ? WHERE id = ?",
            (rounds, panel_id),
        )
        await db.commit()

    async def _persist_result(self, panel_id: str, result: dict[str, Any]) -> None:
        """Persist final synthesis result to the DB."""
        db = await self.api.get_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO panel_results
                (panel_id, consensus_points, dissent_points, action_items, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                panel_id,
                json.dumps(result.get("consensus_points", []), ensure_ascii=False),
                json.dumps(result.get("dissent_points", []), ensure_ascii=False),
                json.dumps(result.get("action_items", []), ensure_ascii=False),
                result.get("summary", ""),
                time.time(),
            ),
        )
        await db.commit()
