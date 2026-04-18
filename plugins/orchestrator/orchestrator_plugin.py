"""
Lexy AI - Orchestrator Plugin.

Hybrid orchestrator with scheduler integration. Provides:

* **Task delegation** -- enqueue work items with priority and persona selection.
* **Agent management** -- persona-aware agent spawning via the AgentManager.
* **Orchestrator brain** -- fast LLM decisions for routing and delegation.
* **Persona registry** -- built-in + user-defined agent personas.
* **Scheduler integration** -- schedule agent tasks via ``core.scheduler_triggered``.
* **Automatic retries** -- failed tasks with ``retry_count < 2`` are requeued.
* **Prompt injection** -- running agents and queue status visible in system prompt.

Tools registered (9):
    delegate_task, ask_orchestrator, spawn_persona, list_running_agents,
    agent_status, stop_running_agent, list_personas, create_persona,
    schedule_agent_task.

Events consumed:
    agent.done, agent.error, agent.started, core.scheduler_triggered.

Hook:
    before_prompt_build (priority 55) -- inject agent/queue status.
"""

from __future__ import annotations

import json
import time
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .agent_pool import OrchestratorPool
from .orchestrator_brain import OrchestratorBrain
from .persona_registry import PersonaRegistry
from .task_queue import TaskQueue

log = get_logger(module="orchestrator_plugin")


# ─── Tool Schemas ─────────────────────────────────────────────────────────────


DELEGATE_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "Detaillierte Aufgabenbeschreibung fuer den Agent",
        },
        "persona": {
            "type": "string",
            "description": (
                "Persona-ID (z.B. 'researcher', 'coder', 'tutor'). "
                "Wird automatisch gewaehlt wenn nicht angegeben."
            ),
        },
        "brain": {
            "type": "string",
            "description": "LLM brain: 'e4b' (schnell) oder 'a4b' (komplex)",
        },
        "priority": {
            "type": "string",
            "description": "Prioritaet: 'low', 'normal' (Standard), 'high'",
        },
    },
    "required": ["task"],
}

ASK_ORCHESTRATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "Frage an den Orchestrator (z.B. Delegation, Planung)",
        },
        "context": {
            "type": "string",
            "description": "Zusaetzlicher Kontext fuer die Entscheidung",
        },
    },
    "required": ["question"],
}

SPAWN_PERSONA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "persona": {
            "type": "string",
            "description": "Persona-ID (z.B. 'researcher', 'coder', 'tutor')",
        },
        "task": {
            "type": "string",
            "description": "Aufgabe fuer den Agent",
        },
        "name": {
            "type": "string",
            "description": "Optionaler Name fuer den Agent (sonst Persona-Name)",
        },
    },
    "required": ["persona", "task"],
}

LIST_RUNNING_AGENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

AGENT_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Die Agent-ID",
        },
    },
    "required": ["agent_id"],
}

STOP_RUNNING_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Die Agent-ID zum Stoppen",
        },
    },
    "required": ["agent_id"],
}

LIST_PERSONAS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

CREATE_PERSONA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Eindeutige Persona-ID in snake_case (z.B. 'math_tutor')",
        },
        "name": {
            "type": "string",
            "description": "Anzeigename der Persona (z.B. 'Mathe-Tutor')",
        },
        "prompt": {
            "type": "string",
            "description": "System-Prompt fuer die Persona",
        },
        "brain": {
            "type": "string",
            "description": "LLM brain: 'e4b' oder 'a4b' (Standard: 'e4b')",
        },
        "temperature": {
            "type": "number",
            "description": "Temperature 0.0-1.0 (Standard: 0.6)",
        },
    },
    "required": ["id", "name", "prompt"],
}

SCHEDULE_AGENT_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "Die Aufgabe die der Agent ausfuehren soll",
        },
        "persona": {
            "type": "string",
            "description": "Persona-ID (optional, wird sonst auto-gewaehlt)",
        },
        "schedule_at": {
            "type": "string",
            "description": (
                "Wann die Aufgabe ausgefuehrt werden soll: "
                "'in 5min', '14:30', Anzahl Sekunden oder HH:MM Format"
            ),
        },
        "repeat_interval": {
            "type": "integer",
            "description": "Wiederholungsintervall in Minuten (optional)",
        },
        "label": {
            "type": "string",
            "description": "Optionales Label fuer den Scheduler-Eintrag",
        },
    },
    "required": ["task", "schedule_at"],
}


# ─── Plugin ───────────────────────────────────────────────────────────────────


class OrchestratorPlugin(BasePlugin):
    """
    Hybrid Orchestrator: task delegation, agent management, persona spawning.

    On load: initialises DB tables, PersonaRegistry, TaskQueue, OrchestratorBrain.
    On enable: connects to AgentManager (via skill_writer), registers 9 tools,
    WS handlers, event listeners, and the prompt-build hook.
    """

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._registry: PersonaRegistry = PersonaRegistry()
        self._queue: TaskQueue | None = None
        self._brain: OrchestratorBrain | None = None
        self._pool: OrchestratorPool | None = None

        # Config (set in on_load)
        self._max_agents: int = 5
        self._decision_brain: str = "e4b"
        self._decision_max_tokens: int = 200
        self._agent_default_brain: str = "e4b"
        self._agent_default_timeout: float = 300.0
        self._agent_max_iterations: int = 15

        # Mapping: scheduler label -> task info, fuer scheduled tasks
        self._scheduled_tasks: dict[str, dict[str, Any]] = {}

    # ─── Lifecycle ─────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        """Initialise DB tables, PersonaRegistry, TaskQueue, OrchestratorBrain."""
        config = self.api.get_config()

        # Config lesen
        self._max_agents = int(config.get("max_agents", 5))
        self._decision_brain = str(config.get("decision_brain", "e4b"))
        self._decision_max_tokens = int(config.get("decision_max_tokens", 200))
        self._agent_default_brain = str(config.get("agent_default_brain", "e4b"))
        self._agent_default_timeout = float(config.get("agent_default_timeout", 300.0))
        self._agent_max_iterations = int(config.get("agent_max_iterations", 15))

        db = await self.api.get_db()

        # Entscheidungs-Log Tabelle
        await db.execute(
            """CREATE TABLE IF NOT EXISTS orchestrator_decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                context     TEXT,
                created_at  REAL NOT NULL
            )"""
        )

        # Scheduled Tasks Mapping Tabelle
        await db.execute(
            """CREATE TABLE IF NOT EXISTS orchestrator_scheduled (
                label       TEXT PRIMARY KEY,
                task        TEXT NOT NULL,
                persona     TEXT,
                brain       TEXT NOT NULL DEFAULT 'e4b',
                created_at  REAL NOT NULL
            )"""
        )
        await db.commit()

        # Persona-Registry initialisieren
        personas_config = config.get("personas", {})
        if not isinstance(personas_config, dict):
            personas_config = {}
        await self._registry.init_db(db, personas_config)

        # Task-Queue initialisieren
        self._queue = TaskQueue(db)
        await self._queue.init_tables()

        # Brain initialisieren
        self._brain = OrchestratorBrain(
            api=self.api,
            brain=self._decision_brain,
            max_tokens=self._decision_max_tokens,
        )

        # Gespeicherte Scheduled-Task-Mappings laden
        async with db.execute(
            "SELECT label, task, persona, brain FROM orchestrator_scheduled"
        ) as cur:
            async for row in cur:
                self._scheduled_tasks[row[0]] = {
                    "task": row[1],
                    "persona": row[2],
                    "brain": row[3],
                }

        log.info(
            "orchestrator.loaded",
            max_agents=self._max_agents,
            personas=len(self._registry.list_all()),
            scheduled_mappings=len(self._scheduled_tasks),
        )

    async def on_enable(self) -> None:
        """
        Connect to AgentManager and register tools, handlers, events, hooks.
        """
        # AgentManager von skill_writer holen
        skill_writer = self.api.get_plugin("skill_writer")
        if skill_writer is None:
            log.error("orchestrator.skill_writer_not_found")
            return

        agent_manager = getattr(skill_writer, "_agent_manager", None)
        if agent_manager is None:
            log.error("orchestrator.agent_manager_not_available")
            return

        # Pool erstellen
        self._pool = OrchestratorPool(
            api=self.api,
            manager=agent_manager,
            registry=self._registry,
        )

        # ── 9 LLM-Tools registrieren ──────────────────────────────────

        self.api.register_tool(
            name="delegate_task",
            handler=self._tool_delegate_task,
            description=(
                "Delegiere eine Aufgabe an einen spezialisierten Agenten. "
                "Der Orchestrator waehlt automatisch die beste Persona wenn "
                "keine angegeben wird. Aufgaben werden priorisiert und "
                "in einer Queue verwaltet."
            ),
            schema=DELEGATE_TASK_SCHEMA,
        )

        self.api.register_tool(
            name="ask_orchestrator",
            handler=self._tool_ask_orchestrator,
            description=(
                "Stelle dem Orchestrator eine Frage zu Delegation, Planung "
                "oder laufenden Agents. Nutze dies fuer strategische "
                "Entscheidungen bevor du delegierst."
            ),
            schema=ASK_ORCHESTRATOR_SCHEMA,
        )

        self.api.register_tool(
            name="spawn_persona",
            handler=self._tool_spawn_persona,
            description=(
                "Starte einen Agent mit einer bestimmten Persona direkt "
                "(ohne Queue). Gut fuer sofortige Aufgaben."
            ),
            schema=SPAWN_PERSONA_SCHEMA,
        )

        self.api.register_tool(
            name="list_running_agents",
            handler=self._tool_list_running_agents,
            description="Liste alle laufenden und kuerzlich beendeten Agents.",
            schema=LIST_RUNNING_AGENTS_SCHEMA,
        )

        self.api.register_tool(
            name="agent_status",
            handler=self._tool_agent_status,
            description="Zeige detaillierten Status eines bestimmten Agents.",
            schema=AGENT_STATUS_SCHEMA,
        )

        self.api.register_tool(
            name="stop_running_agent",
            handler=self._tool_stop_running_agent,
            description="Stoppe einen laufenden Agent.",
            schema=STOP_RUNNING_AGENT_SCHEMA,
        )

        self.api.register_tool(
            name="list_personas",
            handler=self._tool_list_personas,
            description=(
                "Liste alle verfuegbaren Personas (built-in + benutzerdefiniert). "
                "Jede Persona hat einen spezialisierten System-Prompt."
            ),
            schema=LIST_PERSONAS_SCHEMA,
        )

        self.api.register_tool(
            name="create_persona",
            handler=self._tool_create_persona,
            description=(
                "Erstelle eine neue benutzerdefinierte Persona. "
                "Die Persona wird persistent gespeichert."
            ),
            schema=CREATE_PERSONA_SCHEMA,
        )

        self.api.register_tool(
            name="schedule_agent_task",
            handler=self._tool_schedule_agent_task,
            description=(
                "Plane eine Agent-Aufgabe zu einem bestimmten Zeitpunkt. "
                "Nutzt den Scheduler fuer Timer/Reminder und startet "
                "den Agent automatisch wenn die Zeit kommt."
            ),
            schema=SCHEDULE_AGENT_TASK_SCHEMA,
        )

        # ── WebSocket Handlers ────────────────────────────────────────

        self.api.register_ws_handler(
            "orchestrator_status", self._ws_orchestrator_status
        )
        self.api.register_ws_handler(
            "orchestrator_queue", self._ws_orchestrator_queue
        )
        self.api.register_ws_handler(
            "orchestrator_personas", self._ws_orchestrator_personas
        )

        # ── Events ────────────────────────────────────────────────────

        self.api.on_event("agent.done", self._on_agent_done)
        self.api.on_event("agent.error", self._on_agent_error)
        self.api.on_event("agent.started", self._on_agent_started)
        self.api.on_event("core.scheduler_triggered", self._on_scheduler_triggered)

        # ── Hook: Inject agent + queue status into system prompt ──────

        self.api.register_hook(
            "before_prompt_build",
            self._hook_inject_orchestrator_context,
            priority=55,
        )

        log.info("orchestrator.enabled")

    async def on_disable(self) -> None:
        """Release references. AgentManager belongs to skill_writer."""
        self._pool = None
        log.info("orchestrator.disabled")

    # ─── Tool: delegate_task ───────────────────────────────────────────────

    async def _tool_delegate_task(
        self,
        task: str,
        persona: str | None = None,
        brain: str | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """
        Delegate a task: auto-select persona if needed, enqueue, try to spawn.
        """
        if self._queue is None or self._brain is None or self._pool is None:
            return {"error": "Orchestrator nicht initialisiert"}

        # Persona auto-auswaehlen wenn nicht angegeben
        if not persona:
            personas_str = self._format_personas_for_brain()
            persona = await self._brain.select_persona(task, personas_str)
            # Validieren dass die gewaehlte Persona existiert
            if self._registry.get(persona) is None:
                persona = "researcher"  # Sicherer Fallback

        # Brain bestimmen (Persona-Default, ueberschreibbar)
        effective_brain = brain or self._agent_default_brain
        persona_obj = self._registry.get(persona)
        if persona_obj is not None and brain is None:
            effective_brain = persona_obj.brain

        # In Queue einfuegen
        task_id = await self._queue.enqueue(
            task=task,
            persona=persona,
            brain=effective_brain,
            priority=priority,
        )

        # Sofort versuchen zu dequeuen und zu spawnen
        spawn_result = await self._try_dequeue_and_spawn()

        # WS Broadcast
        await self.api.ws_broadcast({
            "type": "orchestrator_task_enqueued",
            "task_id": task_id,
            "task": task[:200],
            "persona": persona,
            "priority": priority,
            "spawn_result": spawn_result,
        })

        result: dict[str, Any] = {
            "task_id": task_id,
            "persona": persona,
            "brain": effective_brain,
            "priority": priority,
            "status": "queued",
        }

        if spawn_result and "agent_id" in spawn_result:
            result["agent_id"] = spawn_result["agent_id"]
            result["status"] = "spawned"

        return result

    # ─── Tool: ask_orchestrator ────────────────────────────────────────────

    async def _tool_ask_orchestrator(
        self,
        question: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Ask the orchestrator brain a question."""
        if self._brain is None or self._pool is None:
            return {"error": "Orchestrator nicht initialisiert"}

        running_str = self._format_running_agents()
        personas_str = self._format_personas_for_brain()

        answer = await self._brain.decide(
            question=question,
            context=context or "",
            running_agents=running_str,
            personas=personas_str,
        )

        # Entscheidung in DB speichern
        try:
            db = await self.api.get_db()
            await db.execute(
                "INSERT INTO orchestrator_decisions "
                "(question, answer, context, created_at) VALUES (?, ?, ?, ?)",
                (question[:500], answer[:500], (context or "")[:500], time.time()),
            )
            await db.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("orchestrator.decision_persist_failed", error=str(exc))

        return {"question": question, "answer": answer}

    # ─── Tool: spawn_persona ──────────────────────────────────────────────

    async def _tool_spawn_persona(
        self,
        persona: str,
        task: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Directly spawn an agent with a specific persona (bypasses queue)."""
        if self._pool is None:
            return {"error": "Orchestrator nicht initialisiert"}

        result = await self._pool.spawn_with_persona(
            task=task,
            persona_id=persona,
            name_override=name,
        )

        if "error" not in result:
            await self.api.ws_broadcast({
                "type": "orchestrator_agent_spawned",
                "agent_id": result.get("agent_id"),
                "persona": persona,
                "task": task[:200],
            })

        return result

    # ─── Tool: list_running_agents ────────────────────────────────────────

    async def _tool_list_running_agents(self) -> dict[str, Any]:
        """List all agents managed by the pool."""
        if self._pool is None:
            return {"error": "Orchestrator nicht initialisiert"}

        agents = self._pool.list_running()
        return {"count": len(agents), "agents": agents}

    # ─── Tool: agent_status ───────────────────────────────────────────────

    async def _tool_agent_status(self, agent_id: str) -> dict[str, Any]:
        """Get detailed status of a specific agent."""
        if self._pool is None:
            return {"error": "Orchestrator nicht initialisiert"}

        agent = self._pool.get(agent_id)
        if agent is None:
            return {"error": f"Agent '{agent_id}' nicht gefunden"}

        return {
            "agent_id": agent_id,
            "name": agent.name,
            "status": agent.status,
            "task": agent.task[:500],
            "iteration": agent._iteration,
            "max_iterations": agent._max_iterations,
            "tools_used": len(agent.results),
            "final_answer": agent._extract_final_answer()[:1000] if agent.status == "done" else None,
        }

    # ─── Tool: stop_running_agent ─────────────────────────────────────────

    async def _tool_stop_running_agent(self, agent_id: str) -> dict[str, Any]:
        """Stop a running agent."""
        if self._pool is None:
            return {"error": "Orchestrator nicht initialisiert"}

        stopped = await self._pool.stop(agent_id)
        if stopped:
            return {"status": "stopped", "agent_id": agent_id}
        return {"error": f"Agent '{agent_id}' nicht gefunden oder bereits beendet"}

    # ─── Tool: list_personas ──────────────────────────────────────────────

    async def _tool_list_personas(self) -> dict[str, Any]:
        """List all available personas."""
        personas = self._registry.list_all()
        return {"count": len(personas), "personas": personas}

    # ─── Tool: create_persona ─────────────────────────────────────────────

    async def _tool_create_persona(
        self,
        id: str,
        name: str,
        prompt: str,
        brain: str = "e4b",
        temperature: float = 0.6,
    ) -> dict[str, Any]:
        """Create a new user-defined persona."""
        # ID validieren
        clean_id = id.lower().replace(" ", "_").replace("-", "_")
        if not clean_id:
            return {"error": "Ungueltige Persona-ID"}

        # Pruefen ob built-in ueberschrieben wird
        existing = self._registry.get(clean_id)
        if existing is not None and existing.builtin:
            return {"error": f"Built-in Persona '{clean_id}' kann nicht ueberschrieben werden"}

        # Brain validieren
        if brain not in ("e4b", "a4b"):
            brain = "e4b"

        # Temperature clampen
        temperature = max(0.0, min(1.0, temperature))

        persona = await self._registry.register(
            id=clean_id,
            name=name,
            prompt=prompt,
            brain=brain,
            temperature=temperature,
        )

        await self.api.ws_broadcast({
            "type": "orchestrator_persona_created",
            "persona": persona.to_dict(),
        })

        return {
            "status": "created",
            "persona": persona.to_dict(),
        }

    # ─── Tool: schedule_agent_task ────────────────────────────────────────

    async def _tool_schedule_agent_task(
        self,
        task: str,
        schedule_at: str,
        persona: str | None = None,
        repeat_interval: int | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Schedule an agent task via the scheduler plugin."""
        # Scheduler-Plugin holen
        scheduler = self.api.get_plugin("scheduler")
        if scheduler is None:
            return {"error": "Scheduler-Plugin nicht verfuegbar"}

        # Label generieren
        schedule_label = label or f"orch:{task[:50]}"

        # Task-Info fuer spaeter speichern
        task_info: dict[str, Any] = {
            "task": task,
            "persona": persona,
            "brain": self._agent_default_brain,
        }

        # In DB persistieren
        try:
            db = await self.api.get_db()
            await db.execute(
                "INSERT OR REPLACE INTO orchestrator_scheduled "
                "(label, task, persona, brain, created_at) VALUES (?, ?, ?, ?, ?)",
                (schedule_label, task, persona, self._agent_default_brain, time.time()),
            )
            await db.commit()
        except Exception as exc:  # noqa: BLE001
            log.error("orchestrator.schedule_persist_failed", error=str(exc))
            return {"error": f"Konnte Schedule nicht speichern: {exc}"}

        self._scheduled_tasks[schedule_label] = task_info

        # Zeitpunkt parsen und Timer setzen
        result = await self._create_scheduler_entry(
            scheduler=scheduler,
            schedule_at=schedule_at,
            label=schedule_label,
            repeat_interval=repeat_interval,
        )

        if "error" in result:
            return result

        return {
            "status": "scheduled",
            "label": schedule_label,
            "task": task[:200],
            "persona": persona,
            "schedule_info": result,
        }

    async def _create_scheduler_entry(
        self,
        scheduler: Any,
        schedule_at: str,
        label: str,
        repeat_interval: int | None = None,
    ) -> dict[str, Any]:
        """
        Parse schedule_at and create a scheduler timer or reminder.

        Supports formats:
        - "in 5min" / "in 30s" -> timer
        - "14:30" -> reminder
        - Numeric seconds -> timer
        """
        schedule_at = schedule_at.strip().lower()

        # "in Xmin" / "in Xs" Format
        if schedule_at.startswith("in "):
            time_part = schedule_at[3:].strip()
            if time_part.endswith("min"):
                try:
                    minutes = int(time_part[:-3].strip())
                    return await scheduler._tool_set_timer(
                        label=label, minutes=minutes
                    )
                except (ValueError, TypeError):
                    return {"error": f"Ungueltiges Zeitformat: {schedule_at}"}
            elif time_part.endswith("s"):
                try:
                    seconds = int(time_part[:-1].strip())
                    return await scheduler._tool_set_timer(
                        label=label, seconds=seconds
                    )
                except (ValueError, TypeError):
                    return {"error": f"Ungueltiges Zeitformat: {schedule_at}"}
            else:
                # Versuche als Minuten zu parsen
                try:
                    minutes = int(time_part)
                    return await scheduler._tool_set_timer(
                        label=label, minutes=minutes
                    )
                except (ValueError, TypeError):
                    return {"error": f"Ungueltiges Zeitformat: {schedule_at}"}

        # HH:MM Format -> Reminder
        if ":" in schedule_at:
            try:
                return await scheduler._tool_set_reminder(
                    time=schedule_at, label=label
                )
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Reminder fehlgeschlagen: {exc}"}

        # Reine Zahl -> Sekunden
        try:
            seconds = int(schedule_at)
            return await scheduler._tool_set_timer(
                label=label, seconds=seconds
            )
        except (ValueError, TypeError):
            return {"error": f"Ungueltiges Zeitformat: {schedule_at}"}

    # ─── Event: agent.done ─────────────────────────────────────────────────

    async def _on_agent_done(self, event_data: Any) -> None:
        """Handle agent completion: update task queue, try next task."""
        if not isinstance(event_data, dict):
            return
        if self._queue is None or self._pool is None:
            return

        agent_id = event_data.get("agent_id", "")
        result_summary = event_data.get("result_summary", "")
        if not agent_id:
            return

        # Task in der Queue finden und als done markieren
        task_entry = await self._queue.find_by_agent(agent_id)
        if task_entry is not None:
            await self._queue.mark_done(task_entry["id"], result_summary)
            log.info(
                "orchestrator.task_done",
                task_id=task_entry["id"],
                agent_id=agent_id,
            )

            # WS Broadcast
            await self.api.ws_broadcast({
                "type": "orchestrator_task_done",
                "task_id": task_entry["id"],
                "agent_id": agent_id,
                "result_summary": result_summary[:300],
            })

        # Naechste Task aus der Queue starten
        await self._try_dequeue_and_spawn()

    # ─── Event: agent.error ────────────────────────────────────────────────

    async def _on_agent_error(self, event_data: Any) -> None:
        """Handle agent error: mark failed, retry if eligible."""
        if not isinstance(event_data, dict):
            return
        if self._queue is None or self._pool is None:
            return

        agent_id = event_data.get("agent_id", "")
        error_msg = event_data.get("error", "Unknown error")
        if not agent_id:
            return

        # Task finden und als failed markieren
        task_entry = await self._queue.find_by_agent(agent_id)
        if task_entry is not None:
            await self._queue.mark_failed(task_entry["id"], error_msg)
            log.warning(
                "orchestrator.task_failed",
                task_id=task_entry["id"],
                agent_id=agent_id,
                error=error_msg[:200],
            )

            # Retry pruefen (retry_count wird durch mark_failed inkrementiert)
            current_retries = task_entry.get("retry_count", 0)
            if current_retries < 1:
                # Retry mit alternativem Brain
                await self._queue.requeue(task_entry["id"])
                log.info(
                    "orchestrator.task_requeued",
                    task_id=task_entry["id"],
                    retry=current_retries + 1,
                )

            # WS Broadcast
            await self.api.ws_broadcast({
                "type": "orchestrator_task_error",
                "task_id": task_entry["id"],
                "agent_id": agent_id,
                "error": error_msg[:300],
                "will_retry": current_retries < 1,
            })

        # Naechste Task versuchen
        await self._try_dequeue_and_spawn()

    # ─── Event: agent.started ──────────────────────────────────────────────

    async def _on_agent_started(self, event_data: Any) -> None:
        """Handle agent started: log for tracking."""
        if not isinstance(event_data, dict):
            return

        agent_id = event_data.get("agent_id", "")
        agent_name = event_data.get("name", "")
        log.debug(
            "orchestrator.agent_started_observed",
            agent_id=agent_id,
            name=agent_name,
        )

    # ─── Event: core.scheduler_triggered ───────────────────────────────────

    async def _on_scheduler_triggered(self, event_data: Any) -> None:
        """Handle scheduler trigger: check if it's an orchestrator-scheduled task."""
        if not isinstance(event_data, dict):
            return
        if self._queue is None:
            return

        label = event_data.get("label", "")
        if not label.startswith("orch:"):
            return

        log.info("orchestrator.scheduler_triggered", label=label)

        # Task-Info aus dem Cache oder der DB laden
        task_info = self._scheduled_tasks.get(label)
        if task_info is None:
            # Versuche aus DB zu laden
            try:
                db = await self.api.get_db()
                async with db.execute(
                    "SELECT task, persona, brain FROM orchestrator_scheduled WHERE label = ?",
                    (label,),
                ) as cur:
                    row = await cur.fetchone()
                    if row is not None:
                        task_info = {
                            "task": row[0],
                            "persona": row[1],
                            "brain": row[2],
                        }
                        self._scheduled_tasks[label] = task_info
            except Exception as exc:  # noqa: BLE001
                log.error("orchestrator.scheduled_task_load_failed", error=str(exc))

        if task_info is None:
            log.warning("orchestrator.scheduled_task_not_found", label=label)
            return

        # Task enqueuen und spawnen
        task_id = await self._queue.enqueue(
            task=task_info["task"],
            persona=task_info.get("persona"),
            brain=task_info.get("brain", self._agent_default_brain),
            priority="normal",
            schedule_label=label,
        )

        spawn_result = await self._try_dequeue_and_spawn()

        await self.api.ws_broadcast({
            "type": "orchestrator_scheduled_task_started",
            "label": label,
            "task_id": task_id,
            "task": task_info["task"][:200],
            "spawn_result": spawn_result,
        })

    # ─── Hook: before_prompt_build (priority 55) ──────────────────────────

    async def _hook_inject_orchestrator_context(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Inject running agent status, queue info, and recent results
        into the system prompt so the main LLM is aware.
        """
        if self._pool is None or self._queue is None:
            return data

        sections: list[str] = []

        # Laufende Agents
        agents = self._pool.list_running()
        running = [a for a in agents if a.get("status") == "running"]
        if running:
            agent_lines = []
            for a in running:
                agent_lines.append(
                    f"- {a['name']} (id: {a['agent_id']}): "
                    f"{a['task'][:100]} "
                    f"[Iteration {a.get('iteration', 0)}/{a.get('max_iterations', 10)}]"
                )
            sections.append(
                "Laufende Agents:\n" + "\n".join(agent_lines)
            )

        # Kuerzlich beendete Tasks (letzte 3)
        recent = await self._queue.list_recent(limit=3)
        done_tasks = [t for t in recent if t["status"] == "done"]
        if done_tasks:
            done_lines = []
            for t in done_tasks[:3]:
                summary = t.get("result_summary") or "Kein Ergebnis"
                done_lines.append(
                    f"- {t['task'][:80]}: {summary[:100]}"
                )
            sections.append(
                "Kuerzlich erledigt:\n" + "\n".join(done_lines)
            )

        # Warteschlange
        pending = await self._queue.list_pending()
        if pending:
            pending_lines = []
            for p in pending[:5]:
                persona_str = f" [{p.get('persona', '?')}]" if p.get("persona") else ""
                pending_lines.append(
                    f"- {p['task'][:80]}{persona_str} ({p['status']})"
                )
            sections.append(
                "Warteschlange:\n" + "\n".join(pending_lines)
            )

        if not sections:
            return data

        context_block = (
            "\n## Hintergrund-Agents\n"
            + "\n\n".join(sections)
            + "\n\nNutze 'delegate_task' zum Delegieren, "
            "'list_running_agents' fuer Status, "
            "'agent_status' fuer Details.\n"
        )

        extra = data.get("extra_context", "")
        data["extra_context"] = extra + context_block
        return data

    # ─── WebSocket Handlers ────────────────────────────────────────────────

    async def _ws_orchestrator_status(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Return full orchestrator status."""
        agents: list[dict[str, Any]] = []
        if self._pool is not None:
            agents = self._pool.list_running()

        pending: list[dict[str, Any]] = []
        recent: list[dict[str, Any]] = []
        if self._queue is not None:
            pending = await self._queue.list_pending()
            recent = await self._queue.list_recent(limit=10)

        await client.send_json({
            "type": "orchestrator_status",
            "agents": agents,
            "pending_tasks": pending,
            "recent_tasks": recent,
            "personas": self._registry.list_all(),
        })

    async def _ws_orchestrator_queue(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Return task queue contents."""
        if self._queue is None:
            await client.send_json({
                "type": "orchestrator_queue",
                "error": "Queue nicht initialisiert",
            })
            return

        pending = await self._queue.list_pending()
        recent = await self._queue.list_recent(limit=20)
        await client.send_json({
            "type": "orchestrator_queue",
            "pending": pending,
            "recent": recent,
        })

    async def _ws_orchestrator_personas(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Return all personas."""
        await client.send_json({
            "type": "orchestrator_personas",
            "personas": self._registry.list_all(),
        })

    # ─── Internal Helpers ──────────────────────────────────────────────────

    async def _try_dequeue_and_spawn(self) -> dict[str, Any] | None:
        """
        Try to dequeue the next task and spawn an agent for it.

        Returns the spawn result, or None if nothing was dequeued.
        """
        if self._queue is None or self._pool is None:
            return None

        task_entry = await self._queue.dequeue()
        if task_entry is None:
            return None

        task_id = task_entry["id"]
        task_text = task_entry["task"]
        persona_id = task_entry.get("persona") or "researcher"
        brain = task_entry.get("brain", self._agent_default_brain)

        # Persona validieren
        if self._registry.get(persona_id) is None:
            persona_id = "researcher"

        # Spawnen
        result = await self._pool.spawn_with_persona(
            task=task_text,
            persona_id=persona_id,
        )

        if "error" in result:
            log.warning(
                "orchestrator.spawn_failed",
                task_id=task_id,
                error=result["error"],
            )
            return result

        agent_id = result.get("agent_id", "")
        await self._queue.mark_running(task_id, agent_id)

        log.info(
            "orchestrator.task_spawned",
            task_id=task_id,
            agent_id=agent_id,
            persona=persona_id,
        )

        return result

    def _format_running_agents(self) -> str:
        """Format running agents as a string for the brain."""
        if self._pool is None:
            return "Keine"

        agents = self._pool.list_running()
        running = [a for a in agents if a.get("status") == "running"]
        if not running:
            return "Keine"

        lines = []
        for a in running:
            lines.append(
                f"- {a['name']} ({a['agent_id']}): {a['task'][:80]}"
            )
        return "\n".join(lines)

    def _format_personas_for_brain(self) -> str:
        """Format personas as a compact string for brain prompts."""
        personas = self._registry.list_all()
        lines = []
        for p in personas:
            lines.append(f"- {p['id']}: {p['name']} (brain={p['brain']})")
        return "\n".join(lines)
