"""
Lexy AI - Orchestrator Brain.

Fast LLM decision-making for the orchestrator. Uses the E4B brain (fast,
cheap) by default to make quick routing and delegation decisions:

* ``decide()``        -- general orchestrator question / decision.
* ``select_persona()`` -- choose the best persona for a given task.

Keeps a small in-memory log of recent decisions for context.
"""

from __future__ import annotations

import time
from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="orchestrator_brain")


DECISION_PROMPT = """Du bist der Orchestrator von Lexy AI.
Entscheide schnell und praezise (max 2 Saetze).

Laufende Agents: {running_agents}
Verfuegbare Personas: {personas}
Letzte Entscheidungen: {recent_decisions}

Frage: {question}
Kontext: {context}"""


DELEGATION_PROMPT = """Waehle die beste Persona fuer diese Aufgabe.
Verfuegbare Personas: {personas}
Aufgabe: {task}

Antworte NUR mit der Persona-ID (z.B. "researcher"). Nichts anderes."""


class OrchestratorBrain:
    """
    Fast LLM-powered decision engine for the orchestrator.

    Uses short prompts and low max_tokens for quick responses.
    Maintains a rolling log of recent decisions for context.
    """

    def __init__(
        self,
        api: Any,
        brain: str = "e4b",
        max_tokens: int = 200,
    ) -> None:
        self._api = api
        self._brain = brain
        self._max_tokens = max_tokens
        # Rolling window of recent decisions (last 10 kept)
        self._decision_log: list[dict[str, Any]] = []

    async def decide(
        self,
        question: str,
        context: str,
        running_agents: str,
        personas: str,
    ) -> str:
        """
        Ask the brain a general orchestration question.

        Args:
            question:        The question to answer.
            context:         Additional context (user message, etc.).
            running_agents:  Formatted string of currently running agents.
            personas:        Formatted string of available personas.

        Returns:
            The brain's answer as a string.
        """
        recent = (
            "\n".join(
                f"- {d['question'][:60]} -> {d['answer'][:80]}"
                for d in self._decision_log[-5:]
            )
            or "Keine"
        )

        prompt = DECISION_PROMPT.format(
            running_agents=running_agents or "Keine",
            personas=personas,
            recent_decisions=recent,
            question=question,
            context=context or "Kein Kontext",
        )

        try:
            answer = await self._api.llm_chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                brain=self._brain,
                max_tokens=self._max_tokens,
            )
            answer = answer.strip()

            # Cache the decision
            self._decision_log.append({
                "question": question,
                "answer": answer,
                "at": time.time(),
            })
            if len(self._decision_log) > 20:
                self._decision_log = self._decision_log[-10:]

            log.debug(
                "orchestrator_brain.decided",
                question=question[:80],
                answer=answer[:100],
            )
            return answer

        except Exception as exc:  # noqa: BLE001
            log.error("orchestrator_brain.decide_failed", error=str(exc))
            return f"Entscheidung fehlgeschlagen: {exc}"

    async def select_persona(self, task: str, personas: str) -> str:
        """
        Select the best persona for a given task.

        Args:
            task:     The task description.
            personas: Formatted string of available persona IDs and names.

        Returns:
            The selected persona ID (e.g. ``"researcher"``).
            Falls back to ``"researcher"`` on error.
        """
        prompt = DELEGATION_PROMPT.format(personas=personas, task=task)

        try:
            answer = await self._api.llm_chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": task},
                ],
                brain=self._brain,
                max_tokens=50,
            )
            # Bereinigen: nur die ID extrahieren
            cleaned = answer.strip().lower().replace('"', "").replace("'", "")
            # Falls die Antwort Satzzeichen oder Zusatztext enthaelt,
            # nur das erste Wort nehmen
            cleaned = cleaned.split()[0] if cleaned else "researcher"

            log.debug(
                "orchestrator_brain.persona_selected",
                task=task[:80],
                persona=cleaned,
            )
            return cleaned

        except Exception as exc:  # noqa: BLE001
            log.error("orchestrator_brain.select_persona_failed", error=str(exc))
            return "researcher"
