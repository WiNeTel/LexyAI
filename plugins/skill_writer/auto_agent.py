"""
Lexy AI - Autonomous Agent System.

Provides ``AutoAgent`` — an independent agent with its own conversation,
tool access, and memory. Runs as an asyncio task alongside the main chat.

``AgentManager`` handles lifecycle, concurrency limits, and timeouts for
all running agents.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="auto_agent")


class AutoAgent:
    """
    Independent agent with own conversation, tools, and memory access.

    Each agent runs its own think-act loop: it sends messages to the LLM,
    detects tool calls in the response, executes them, and feeds the
    results back. The loop terminates when the LLM produces a response
    with no tool calls (i.e., the final answer), or when the iteration
    limit is reached.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        task: str,
        api: Any,
        brain: str = "e4b",
        max_iterations: int = 10,
        timeout: float = 120.0,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.task = task
        self._api = api
        self._brain = brain
        self._max_iterations = max_iterations
        self._timeout = timeout
        self.status: str = "idle"
        self.messages: list[dict[str, str]] = []
        self.results: list[dict[str, Any]] = []
        self._iteration: int = 0
        self._log = log.bind(agent_id=agent_id, agent_name=name)

    async def run(self) -> dict[str, Any]:
        """
        Execute the agent loop.

        The loop:
        1. Sends the conversation to the LLM.
        2. Checks for tool calls in the response.
        3. If tools found: execute them, append results, continue.
        4. If no tools: the agent is done.

        Returns:
            Summary dict with status, iteration count, tools used, result.
        """
        self.status = "running"
        await self._broadcast_status("started")
        self._log.info("auto_agent.started", task=self.task[:200])

        try:
            # Initiale Nachrichten aufbauen
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.task},
            ]

            for self._iteration in range(1, self._max_iterations + 1):
                self._log.debug(
                    "auto_agent.iteration",
                    iteration=self._iteration,
                    max=self._max_iterations,
                )

                # LLM aufrufen
                response = await self._api.llm_chat(
                    self.messages,
                    brain=self._brain,
                    max_tokens=1024,
                )
                self.messages.append({"role": "assistant", "content": response})

                # Tool-Calls pruefen
                tool_caller = self._api.get_tool_caller()
                if tool_caller is None:
                    self._log.debug("auto_agent.no_tool_caller")
                    break

                calls = tool_caller.detect_all(response)
                if not calls:
                    # Keine Tool-Calls: Agent ist fertig
                    self._log.debug("auto_agent.no_tool_calls_done")
                    break

                # Tool-Calls ausfuehren
                for call in calls:
                    self._log.info(
                        "auto_agent.tool_call",
                        tool=call.name,
                        iteration=self._iteration,
                    )
                    result_text = await tool_caller.execute_and_format(call)
                    self.messages.append({
                        "role": "user",
                        "content": result_text,
                    })
                    self.results.append({
                        "tool": call.name,
                        "args": call.arguments,
                        "result": result_text[:500],
                    })

                    # Fortschritt broadcasten
                    await self._broadcast_progress(
                        f"Tool: {call.name}",
                        self._iteration,
                    )

            # Finale Antwort extrahieren (letzter Assistant-Msg ohne Tool-Calls)
            final = self._extract_final_answer()

            # Ergebnis im shared Memory speichern
            if final:
                await self._api.memory_store(
                    text=(
                        f"[Agent:{self.name}] Aufgabe: {self.task[:200]}\n"
                        f"Ergebnis: {final[:500]}"
                    ),
                    collection="facts",
                    metadata={
                        "source": f"auto_agent_{self.agent_id}",
                        "agent_name": self.name,
                        "task": self.task[:200],
                    },
                )

            self.status = "done"
            result_summary = final[:500] if final else "No result"
            await self._broadcast_status("done", result_summary=result_summary)
            self._log.info(
                "auto_agent.done",
                iterations=self._iteration,
                tools_used=len(self.results),
            )
            return {
                "status": "done",
                "iterations": self._iteration,
                "tools_used": len(self.results),
                "result": final,
            }

        except asyncio.CancelledError:
            self.status = "cancelled"
            await self._broadcast_status("cancelled")
            self._log.warning("auto_agent.cancelled", iteration=self._iteration)
            return {"status": "cancelled", "iterations": self._iteration}

        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            await self._broadcast_status("error", error=str(exc))
            self._log.error(
                "auto_agent.failed",
                error=str(exc),
                iteration=self._iteration,
            )
            return {
                "status": "failed",
                "error": str(exc),
                "iterations": self._iteration,
            }

    def _extract_final_answer(self) -> str:
        """
        Get the last assistant message that doesn't contain tool calls.

        Walks the message history in reverse, stripping tool-call markup
        from assistant messages. Returns the first non-empty cleaned text.
        """
        tool_caller = self._api.get_tool_caller()
        for msg in reversed(self.messages):
            if msg["role"] != "assistant":
                continue
            if tool_caller is not None:
                clean = tool_caller.strip_tool_call(msg["content"])
                clean = tool_caller.strip_tool_result(clean).strip()
                if clean:
                    return clean
            else:
                content = msg["content"].strip()
                if content:
                    return content
        return ""

    def get_conversation(self) -> list[dict[str, str]]:
        """Return a copy of the full conversation history."""
        return list(self.messages)

    async def _broadcast_status(self, event: str, **extra: Any) -> None:
        """Broadcast agent status change via WebSocket and EventBus."""
        data: dict[str, Any] = {
            "type": f"agent_{event}",
            "agent_id": self.agent_id,
            "name": self.name,
            "task": self.task[:200],
            **extra,
        }
        await self._api.ws_broadcast(data)
        await self._api.emit(f"agent.{event}", data)

    async def _broadcast_progress(self, detail: str, iteration: int) -> None:
        """Broadcast iteration progress to WebSocket clients."""
        await self._api.ws_broadcast({
            "type": "agent_progress",
            "agent_id": self.agent_id,
            "name": self.name,
            "iteration": iteration,
            "max_iterations": self._max_iterations,
            "detail": detail,
        })


class AgentManager:
    """
    Manages all running auto-agents.

    Handles spawning, lifecycle, concurrency limits, and cleanup.
    Each agent runs as its own asyncio.Task with a timeout wrapper.
    """

    def __init__(
        self,
        api: Any,
        max_concurrent: int = 5,
        default_timeout: float = 120.0,
        default_max_iter: int = 10,
    ) -> None:
        self._api = api
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._default_max_iter = default_max_iter
        self._agents: dict[str, AutoAgent] = {}
        self._tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._log = log.bind(component="agent_manager")

    async def spawn(
        self,
        name: str,
        task: str,
        system_prompt: str | None = None,
        brain: str = "e4b",
    ) -> dict[str, Any]:
        """
        Spawn a new autonomous agent.

        Args:
            name:          Human-readable agent name.
            task:          The task description for the agent.
            system_prompt: Custom system prompt (auto-generated if None).
            brain:         LLM brain to use (``e4b`` or ``a4b``).

        Returns:
            Dict with agent_id, name, task, status. Contains ``error``
            key if the concurrency limit is reached.
        """
        running = self._running_agents()
        if len(running) >= self._max_concurrent:
            self._log.warning(
                "agent_manager.limit_reached",
                running=len(running),
                max=self._max_concurrent,
            )
            return {
                "error": (
                    f"Maximum {self._max_concurrent} concurrent agents "
                    f"reached ({len(running)} running)"
                ),
            }

        agent_id = uuid.uuid4().hex[:12]

        if not system_prompt:
            system_prompt = (
                f"Du bist ein autonomer Sub-Agent von Lexy AI. "
                f"Dein Name ist '{name}'.\n"
                f"Deine Aufgabe: {task}\n\n"
                "Arbeite Schritt fuer Schritt. Nutze die verfuegbaren Tools.\n"
                "Wenn du fertig bist, fasse dein Ergebnis zusammen."
            )

        agent = AutoAgent(
            agent_id=agent_id,
            name=name,
            system_prompt=system_prompt,
            task=task,
            api=self._api,
            brain=brain,
            max_iterations=self._default_max_iter,
            timeout=self._default_timeout,
        )
        self._agents[agent_id] = agent

        # Task mit Timeout starten
        timeout = self._default_timeout

        async def _run_with_timeout() -> dict[str, Any]:
            try:
                return await asyncio.wait_for(
                    agent.run(), timeout=timeout
                )
            except asyncio.TimeoutError:
                agent.status = "timeout"
                await agent._broadcast_status("error", error="Timeout reached")
                self._log.warning(
                    "agent_manager.timeout",
                    agent_id=agent_id,
                    timeout=timeout,
                )
                return {"status": "timeout"}

        task_obj = asyncio.create_task(
            _run_with_timeout(),
            name=f"auto_agent.{agent_id}",
        )
        self._tasks[agent_id] = task_obj

        self._log.info(
            "agent_manager.spawned",
            agent_id=agent_id,
            name=name,
            task=task[:200],
        )
        return {
            "agent_id": agent_id,
            "name": name,
            "task": task[:200],
            "status": "running",
        }

    async def stop(self, agent_id: str) -> bool:
        """
        Stop a running agent by cancelling its task.

        Returns:
            True if the agent was running and has been cancelled.
        """
        task = self._tasks.get(agent_id)
        if task is not None and not task.done():
            task.cancel()
            self._log.info("agent_manager.stopped", agent_id=agent_id)
            return True
        self._log.warning("agent_manager.stop_not_found", agent_id=agent_id)
        return False

    def get_agent(self, agent_id: str) -> AutoAgent | None:
        """Look up an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """
        List all agents (running, done, failed, etc.).

        Returns:
            List of agent summary dicts.
        """
        return [
            {
                "agent_id": aid,
                "name": agent.name,
                "task": agent.task[:200],
                "status": agent.status,
                "iteration": agent._iteration,
                "max_iterations": agent._max_iterations,
                "tools_used": len(agent.results),
            }
            for aid, agent in self._agents.items()
        ]

    def _running_agents(self) -> list[str]:
        """Return IDs of agents currently in 'running' status."""
        return [
            aid for aid, agent in self._agents.items()
            if agent.status == "running"
        ]

    async def cleanup(self) -> None:
        """
        Cancel all running tasks and clear all agent state.

        Called during plugin disable.
        """
        cancelled_count = 0
        for aid, task in self._tasks.items():
            if not task.done():
                task.cancel()
                cancelled_count += 1

        # Warten bis alle Tasks beendet sind
        if self._tasks:
            done, _ = await asyncio.wait(
                self._tasks.values(),
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED,
            )

        self._tasks.clear()
        self._agents.clear()
        self._log.info(
            "agent_manager.cleanup",
            cancelled=cancelled_count,
        )
