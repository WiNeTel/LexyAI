"""
Coder-Brain — Plan → Code → Test → Reflect loop with error-learning.

Inspired by the Agent-Zero monologue pattern. The brain is a
self-contained async state machine that:

1. **Plan**: asks the LLM (default brain ``a4b``) to break the user's
   task into a list of concrete steps.
2. **Step**: for each step, asks the LLM to emit a single tool call —
   one of the workspace_* tools registered by :class:`CoderPlugin`.
3. **Test**: when the step is a ``workspace_run``, the runner result is
   fed straight back as the next observation. Failure → reflect.
4. **Reflect**: on a non-zero exit (or an exception), pull similar
   past errors from :mod:`error_learning`, append them to the
   conversation, and let the LLM revise the plan. Up to
   ``max_retries_per_step`` retries before the step is marked failed.
5. **Done / Cancelled**: emit a final WS event, persist the success or
   failure to memory.

The brain doesn't speak HTTP — it calls a ``run_tool`` callable supplied
by the plugin so we can unit-test it with a fake.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from .error_learning import ErrorLearning, ErrorMemory


log = logging.getLogger(__name__)


# Type-aliases keep the public signatures readable.
LLMChat = Callable[..., Awaitable[str]]
ToolRunner = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
EventBroadcaster = Callable[[dict[str, Any]], Awaitable[None]]


TaskState = Literal[
    "pending", "planning", "running", "reflecting", "done", "failed", "cancelled"
]


@dataclass
class CoderStep:
    index: int
    description: str
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    retries: int = 0
    completed: bool = False

    def to_public(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "result_summary": _summarise_result(self.result),
            "error": self.error[:600] if self.error else "",
            "retries": self.retries,
            "completed": self.completed,
        }


@dataclass
class CoderTask:
    task_id: str
    description: str
    kind: str                          # skill / project / extension
    project: str = ""
    state: TaskState = "pending"
    plan: list[str] = field(default_factory=list)
    steps: list[CoderStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    last_error: str = ""
    session_id: str = ""

    def to_public(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "kind": self.kind,
            "project": self.project,
            "state": self.state,
            "plan": list(self.plan),
            "steps": [s.to_public() for s in self.steps],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "session_id": self.session_id,
        }


# ─── Brain ────────────────────────────────────────────────────────────


class CoderBrain:
    """Plan-Code-Test-Reflect loop. One brain instance per plugin."""

    _PLAN_SYSTEM_PROMPT = (
        "Du bist Lexys Coder-Brain. Deine Aufgabe: zerlege die Anfrage des "
        "Users in eine Liste konkreter Schritte. Jeder Schritt soll genau "
        "EINEN Workspace-Tool-Aufruf sein. Verfügbare Tools:\n"
        "{tools}\n\n"
        "Antworte AUSSCHLIESSLICH als JSON-Array von Strings. Beispiel:\n"
        '["Initialisiere Skill-Projekt foo", "Schreibe skill.py mit Sortier-Funktion", '
        '"Schreibe Test in tests/test_sort.py", "Führe Tests via workspace_run aus"]'
    )

    _STEP_SYSTEM_PROMPT = (
        "Du bist Lexys Coder-Brain. Du hast einen Plan und arbeitest Schritt {index} "
        "ab: \"{description}\". Wähle GENAU EIN passendes workspace_*-Tool und "
        "erzeuge ein JSON-Objekt mit den Feldern ``tool`` (string) und ``arguments`` "
        "(object). Keine Erklärung, kein Markdown — nur das JSON.\n\n"
        "## Verfügbare Tools\n{tools}\n\n"
        "## Bisherige Lehren\n{lessons}\n"
    )

    def __init__(
        self,
        *,
        llm_chat: LLMChat,
        tool_runner: ToolRunner,
        broadcast: EventBroadcaster,
        error_learning: ErrorLearning,
        brain: str = "a4b",
        max_steps: int = 12,
        max_retries_per_step: int = 3,
        history_window: int = 6,
    ) -> None:
        self._chat = llm_chat
        self._run_tool = tool_runner
        self._broadcast = broadcast
        self._learning = error_learning
        self._brain = brain
        self._max_steps = max(1, int(max_steps))
        self._max_retries = max(1, int(max_retries_per_step))
        self._history = max(0, int(history_window))
        self._tasks: dict[str, CoderTask] = {}
        self._runners: dict[str, asyncio.Task[None]] = {}

    # ─── Public API ──────────────────────────────────────────────────

    async def submit(
        self,
        *,
        description: str,
        kind: str = "skill",
        project: str = "",
        session_id: str = "",
        tool_catalog: str = "",
    ) -> str:
        """Queue a coder task. Returns the assigned ``task_id``."""
        task_id = uuid.uuid4().hex[:12]
        task = CoderTask(
            task_id=task_id,
            description=description.strip(),
            kind=kind,
            project=project.strip(),
            session_id=session_id,
        )
        self._tasks[task_id] = task
        runner = asyncio.create_task(
            self._run(task, tool_catalog),
            name=f"coder_brain.{task_id}",
        )
        self._runners[task_id] = runner
        return task_id

    def get(self, task_id: str) -> CoderTask | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[CoderTask]:
        return list(self._tasks.values())

    async def stop(self, task_id: str) -> bool:
        runner = self._runners.get(task_id)
        if runner is None or runner.done():
            return False
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass
        task = self._tasks.get(task_id)
        if task is not None and task.state not in ("done", "failed", "cancelled"):
            task.state = "cancelled"
            task.finished_at = time.time()
            await self._emit(task, "coder_cancelled")
        return True

    # ─── Loop ───────────────────────────────────────────────────────

    async def _run(self, task: CoderTask, tool_catalog: str) -> None:
        try:
            task.state = "planning"
            await self._emit(task, "coder_started")
            await self._plan(task, tool_catalog)
            # _plan sets state=failed on parse problems and emits its
            # own coder_error — bail out before entering the step loop
            # so an empty step list doesn't trip the else-branch into
            # "done" by accident.
            if task.state == "failed":
                return

            task.state = "running"
            for step in task.steps:
                await self._execute_step(task, step, tool_catalog)
                if task.state == "failed":
                    break
            else:
                task.state = "done"
                task.finished_at = time.time()
                await self._emit(task, "coder_done")
                await self._learning.remember_solution(
                    text=_summary_for_memory(task),
                    task_tag=_task_tag(task),
                    extras={"task_id": task.task_id, "kind": task.kind},
                )
        except asyncio.CancelledError:
            task.state = "cancelled"
            task.finished_at = time.time()
            await self._emit(task, "coder_cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("coder.brain_unhandled task=%s err=%s", task.task_id, exc)
            task.state = "failed"
            task.last_error = str(exc)
            task.finished_at = time.time()
            await self._emit(task, "coder_error")
        finally:
            # Drop the runner ref so the GC reclaims it; keep the task
            # snapshot around so coder_status still works after exit.
            self._runners.pop(task.task_id, None)

    async def _plan(self, task: CoderTask, tool_catalog: str) -> None:
        prompt = self._PLAN_SYSTEM_PROMPT.format(tools=tool_catalog or "(none)")
        user_msg = (
            f"Aufgabe: {task.description}\n"
            f"Kind: {task.kind}\n"
            f"Project: {task.project or '(neu anlegen)'}"
        )
        try:
            raw = await self._chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg},
                ],
                brain=self._brain,
                max_tokens=600,
                temperature=0.2,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            task.state = "failed"
            task.last_error = f"plan_failed: {exc}"
            task.finished_at = time.time()
            await self._emit(task, "coder_error")
            return

        plan = _parse_plan(raw)
        if not plan:
            task.state = "failed"
            task.last_error = "planner returned no usable steps"
            task.finished_at = time.time()
            await self._emit(task, "coder_error")
            return

        plan = plan[: self._max_steps]
        task.plan = plan
        task.steps = [
            CoderStep(index=i, description=desc) for i, desc in enumerate(plan)
        ]
        await self._emit(task, "coder_planned")

    async def _execute_step(
        self,
        task: CoderTask,
        step: CoderStep,
        tool_catalog: str,
    ) -> None:
        last_error = ""
        for attempt in range(self._max_retries):
            step.retries = attempt
            try:
                lessons = await self._format_lessons(
                    task=task, step=step, last_error=last_error,
                )
                raw = await self._chat(
                    messages=[
                        {
                            "role": "system",
                            "content": self._STEP_SYSTEM_PROMPT.format(
                                index=step.index + 1,
                                description=step.description,
                                tools=tool_catalog or "(none)",
                                lessons=lessons or "(keine bisher)",
                            ),
                        },
                        {
                            "role": "user",
                            "content": _step_user_msg(task, step, last_error),
                        },
                    ],
                    brain=self._brain,
                    max_tokens=400,
                    temperature=0.3,
                    thinking=False,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"chat_failed: {exc}"
                continue

            tool_call = _parse_tool_call(raw)
            if tool_call is None:
                last_error = f"unparseable tool-call: {raw[:200]}"
                continue
            step.tool = tool_call["tool"]
            step.arguments = dict(tool_call.get("arguments") or {})
            await self._emit(task, "coder_step_attempt")

            try:
                result = await self._run_tool(step.tool, step.arguments)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"tool_invocation_error: {exc}"
                continue

            step.result = result
            ok = bool(result.get("ok", True)) and not result.get("error")
            # If the tool was a subprocess run, prefer the run's actual
            # success flag (returncode == 0).
            run_payload = result.get("run") or result.get("data") or {}
            if isinstance(run_payload, dict) and "returncode" in run_payload:
                ok = ok and int(run_payload.get("returncode", 1)) == 0

            if ok:
                step.completed = True
                step.error = ""
                await self._emit(task, "coder_step_done")
                return

            # Failure path → record + reflect.
            err_text = (
                result.get("error")
                or (run_payload.get("stderr") if isinstance(run_payload, dict) else "")
                or "unknown error"
            )
            last_error = str(err_text)[:1500]
            step.error = last_error
            await self._learning.remember_failure(
                text=(
                    f"Task: {task.description}\n"
                    f"Step: {step.description}\n"
                    f"Tool: {step.tool}({json.dumps(step.arguments)[:200]})\n"
                    f"Error: {last_error}"
                ),
                task_tag=_task_tag(task),
                extras={
                    "task_id": task.task_id,
                    "step_index": step.index,
                    "tool": step.tool,
                },
            )
            await self._emit(task, "coder_step_retry")

        # All retries exhausted.
        task.state = "failed"
        task.last_error = (
            f"step {step.index} ({step.description!r}) failed after "
            f"{self._max_retries} attempts: {last_error}"
        )
        task.finished_at = time.time()
        await self._emit(task, "coder_error")

    # ─── Helpers ────────────────────────────────────────────────────

    async def _format_lessons(
        self,
        *,
        task: CoderTask,
        step: CoderStep,
        last_error: str,
    ) -> str:
        query = last_error or step.description or task.description
        memories = await self._learning.recall_similar(
            query=query, task_tag=_task_tag(task),
        )
        if not memories:
            return ""
        lines = []
        for i, mem in enumerate(memories, start=1):
            head = f"### Lehre {i}"
            block = mem.text[:600]
            if mem.solution:
                block += f"\n**Lösung damals**:\n{mem.solution[:400]}"
            lines.append(f"{head}\n{block}")
        return "\n\n".join(lines)

    async def _emit(self, task: CoderTask, event_type: str) -> None:
        try:
            await self._broadcast(
                {"type": event_type, **task.to_public()}
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("coder.brain_emit_failed err=%s", exc)


# ─── Module helpers ──────────────────────────────────────────────────


def _step_user_msg(task: CoderTask, step: CoderStep, last_error: str) -> str:
    parts = [f"Aktueller Schritt: {step.description}"]
    if task.project:
        parts.append(f"Project: {task.project} ({task.kind})")
    if step.retries > 0 and last_error:
        parts.append(
            f"Letzter Versuch ist fehlgeschlagen mit:\n{last_error[:800]}\n"
            "Korrigiere die Argumente."
        )
    parts.append('Antworte mit JSON: {"tool": "...", "arguments": {...}}')
    return "\n\n".join(parts)


def _parse_plan(raw: str) -> list[str]:
    """Best-effort plan extractor. Tolerates code-fences and prose."""
    text = (raw or "").strip()
    if not text:
        return []
    # Strip a leading code fence.
    if text.startswith("```"):
        # remove the first line and an optional closing fence.
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3]
    # Try strict JSON first.
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    # Fallback: numbered list ("1. foo\n2. bar").
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        # Strip leading "N." / "N)" prefixes.
        for prefix_pat in (r"^\d+[\.\)]\s+",):
            import re
            line = re.sub(prefix_pat, "", line)
        if line:
            out.append(line)
    return out


def _parse_tool_call(raw: str) -> dict[str, Any] | None:
    """Extract ``{tool, arguments}`` from an LLM string."""
    text = (raw or "").strip()
    if not text:
        return None
    # Strip code fences.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to find the first JSON object substring.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(parsed, dict):
        return None
    tool = parsed.get("tool") or parsed.get("name")
    if not isinstance(tool, str) or not tool.strip():
        return None
    args = parsed.get("arguments") or parsed.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return {"tool": tool.strip(), "arguments": args}


def _summarise_result(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return str(result)[:200]
    if "error" in result and result["error"]:
        return f"error: {str(result['error'])[:200]}"
    if "ok" in result:
        return "ok" if result["ok"] else "not_ok"
    return str(result)[:200]


def _task_tag(task: CoderTask) -> str:
    """Stable tag used as memory metadata key."""
    if task.project:
        return f"coder/{task.kind}/{task.project}"
    return f"coder/{task.kind}/{task.task_id}"


def _summary_for_memory(task: CoderTask) -> str:
    completed = sum(1 for s in task.steps if s.completed)
    return (
        f"Coder-Task `{task.description}` ({task.kind})\n"
        f"Steps: {completed}/{len(task.steps)} completed\n"
        f"Last action: "
        + (task.steps[-1].tool if task.steps else "(no steps)")
    )
