"""
Lexy AI - Orchestrator Agent Pool.

Thin wrapper around ``AgentManager`` (from the skill_writer plugin) that
adds persona-aware spawning. The pool builds a full system prompt from
the persona definition, injects the current date/time, and delegates
the actual agent lifecycle to the AgentManager.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from lexy_core.utils.logging import get_logger

from .persona_registry import Persona, PersonaRegistry

log = get_logger(module="orchestrator_pool")


class OrchestratorPool:
    """
    Agent pool with persona support.

    Wraps the existing ``AgentManager`` from the skill_writer plugin,
    adding persona-based system prompt construction.
    """

    def __init__(
        self,
        api: Any,
        manager: Any,
        registry: PersonaRegistry,
    ) -> None:
        self._api = api
        self._manager = manager   # AgentManager from skill_writer
        self._registry = registry  # PersonaRegistry

    async def spawn_with_persona(
        self,
        task: str,
        persona_id: str,
        name_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Spawn an agent using a persona's configuration.

        Args:
            task:          The task description for the agent.
            persona_id:    ID of the persona to use.
            name_override: Custom name (uses persona.name if None).

        Returns:
            Agent info dict from AgentManager.spawn(), or error dict.
        """
        persona = self._registry.get(persona_id)
        if persona is None:
            log.warning(
                "orchestrator_pool.persona_not_found",
                persona_id=persona_id,
            )
            return {"error": f"Persona '{persona_id}' nicht gefunden"}

        prompt = self._build_prompt(persona, task)
        agent_name = name_override or persona.name

        log.info(
            "orchestrator_pool.spawning",
            agent_name=agent_name,
            persona=persona_id,
            brain=persona.brain,
        )

        result = await self._manager.spawn(
            name=agent_name,
            task=task,
            system_prompt=prompt,
            brain=persona.brain,
        )
        return result

    @staticmethod
    def _build_prompt(persona: Persona, task: str) -> str:
        """
        Build a full system prompt for an agent from a persona.

        Includes:
        - The persona's base system prompt.
        - Current date/time (German weekday).
        - Generic tool/result instructions.
        """
        now = datetime.now()
        weekdays = [
            "Montag", "Dienstag", "Mittwoch", "Donnerstag",
            "Freitag", "Samstag", "Sonntag",
        ]

        parts = [
            persona.system_prompt,
            f"Aktuelles Datum: {weekdays[now.weekday()]}, "
            f"{now.strftime('%d.%m.%Y %H:%M')} Uhr.",
            "Du hast Zugriff auf alle verfuegbaren Tools. Nutze sie wenn noetig.",
            "Wenn du fertig bist, fasse dein Ergebnis zusammen.",
        ]
        return "\n\n".join(parts)

    def list_running(self) -> list[dict[str, Any]]:
        """List all agents (running, done, failed, etc.)."""
        return self._manager.list_agents()

    async def stop(self, agent_id: str) -> bool:
        """Stop a running agent by ID."""
        return await self._manager.stop(agent_id)

    def get(self, agent_id: str) -> Any:
        """Look up an agent by ID."""
        return self._manager.get_agent(agent_id)
