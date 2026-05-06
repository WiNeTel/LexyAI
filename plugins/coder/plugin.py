"""
Lexy AI — Coder plugin (self-coding workspace).

Composition root for the six modules in this package:

* :mod:`workspace_mgr` — path-safe filesystem operations
* :mod:`code_runner`   — subprocess executor with timeout
* :mod:`conda_env`     — venv / conda manager
* :mod:`git_committer` — auto-commit + log + revert
* :mod:`approval_gate` — async ask-Mike workflow
* :mod:`coder_brain`   — Plan-Code-Test-Reflect loop (optional)
* :mod:`error_learning`— ChromaDB-backed past-error recall

Tool surface (registered with the LLM):

| LOW (auto-approve)    | MED (modal)             | HIGH (always confirm) |
|-----------------------|-------------------------|-----------------------|
| workspace_list        | workspace_init_project  | workspace_run         |
| workspace_read        | workspace_write         | workspace_delete      |
| workspace_list_projects| workspace_apply_patch  | workspace_git_revert  |
| workspace_git_log     | workspace_create_env    |                       |
| workspace_git_diff    | workspace_install       |                       |

Plus the high-level multi-step entrypoints:

* coder_task     — start a Plan-Code-Test-Reflect run
* coder_status   — task progress + step log
* coder_stop     — cancel a running task
* coder_list_tasks — recent + active task overview
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from lexy_core.plugin_system import BasePlugin

from .approval_gate import (
    AUTO_APPROVE_LOW,
    ApprovalGate,
    RISK_HIGH,
    RISK_LOW,
    RISK_MED,
)
from .code_runner import CodeRunner, RunResult
from .coder_brain import CoderBrain, CoderTask
from .conda_env import CondaEnvManager, EnvInfo
from .error_learning import ErrorLearning
from .git_committer import GitCommitter, GitNotAvailable
from .workspace_mgr import (
    WorkspaceManager,
    WorkspaceNotFoundError,
    WorkspacePathError,
)


log = logging.getLogger(__name__)


# ─── Tool schemas ────────────────────────────────────────────────────


_PROJECT_REF: dict[str, Any] = {
    "kind": {
        "type": "string",
        "enum": ["skill", "project", "extension"],
        "description": "Type of workspace project: skill (runnable), project (Mike runs), extension (Lexy plugin).",
    },
    "name": {
        "type": "string",
        "description": "Project name (alphanumeric / dash / underscore, ≤64 chars).",
    },
}


WORKSPACE_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "rel_path": {"type": "string", "description": "Relative path inside the project. Default '' = project root."},
    },
    "required": ["kind", "name"],
}

WORKSPACE_READ_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "rel_path": {"type": "string"},
    },
    "required": ["kind", "name", "rel_path"],
}

WORKSPACE_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "rel_path": {"type": "string"},
        "content": {"type": "string"},
        "commit_message": {"type": "string", "description": "Optional. Override the default git-commit message."},
    },
    "required": ["kind", "name", "rel_path", "content"],
}

WORKSPACE_INIT_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "description": {"type": "string"},
    },
    "required": ["kind", "name"],
}

WORKSPACE_DELETE_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "rel_path": {"type": "string"},
    },
    "required": ["kind", "name", "rel_path"],
}

WORKSPACE_LIST_PROJECTS_SCHEMA = {"type": "object", "properties": {}}

WORKSPACE_RUN_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Skill name (kind is forced to 'skill' for safety)."},
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Arguments to pass after the python entry point.",
        },
        "entrypoint": {
            "type": "string",
            "description": "File to run (relative to skill dir). Defaults to 'skill.py'.",
        },
        "timeout_seconds": {"type": "number"},
    },
    "required": ["name"],
}

WORKSPACE_INSTALL_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "packages": {"type": "array", "items": {"type": "string"}},
        "upgrade": {"type": "boolean"},
    },
    "required": ["kind", "name", "packages"],
}

WORKSPACE_CREATE_ENV_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "python": {"type": "string", "description": "Optional path to a host python."},
    },
    "required": ["kind", "name"],
}

WORKSPACE_GIT_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "limit": {"type": "integer"},
    },
    "required": ["kind", "name"],
}

WORKSPACE_GIT_DIFF_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
    },
    "required": ["kind", "name"],
}

WORKSPACE_GIT_REVERT_SCHEMA = {
    "type": "object",
    "properties": {
        **_PROJECT_REF,
        "commit_ref": {"type": "string"},
    },
    "required": ["kind", "name", "commit_ref"],
}

CODER_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "kind": {"type": "string", "enum": ["skill", "project", "extension"]},
        "project": {"type": "string", "description": "Existing project to work inside, or empty to plan a new one."},
    },
    "required": ["description"],
}

CODER_STATUS_SCHEMA = {
    "type": "object",
    "properties": {"task_id": {"type": "string"}},
    "required": ["task_id"],
}

CODER_STOP_SCHEMA = CODER_STATUS_SCHEMA
CODER_LIST_TASKS_SCHEMA = {"type": "object", "properties": {}}

PUBLISH_EXTENSION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the extension under workspace/extensions/<name>/",
        },
    },
    "required": ["name"],
}

UNPUBLISH_EXTENSION_SCHEMA = PUBLISH_EXTENSION_SCHEMA


# Tool catalog string injected into the coder_brain prompt.
_TOOL_CATALOG = """\
- workspace_list({kind, name, rel_path?}) — list files in a project subdir.
- workspace_read({kind, name, rel_path}) — read a single file (text only).
- workspace_init_project({kind, name, description?}) — create a fresh project.
- workspace_write({kind, name, rel_path, content, commit_message?}) — write a file (auto-commits).
- workspace_delete({kind, name, rel_path}) — delete a single file (HIGH risk).
- workspace_list_projects({}) — list all known projects.
- workspace_create_env({kind, name, python?}) — create the project's .venv.
- workspace_install({kind, name, packages[], upgrade?}) — pip install into the env.
- workspace_run({name, args?, entrypoint?, timeout_seconds?}) — run a SKILL via subprocess.
- workspace_git_log({kind, name, limit?}) — last N commits.
- workspace_git_diff({kind, name}) — uncommitted changes vs HEAD.
- workspace_git_revert({kind, name, commit_ref}) — hard reset (HIGH risk).
"""


# ─── Plugin ──────────────────────────────────────────────────────────


class CoderPlugin(BasePlugin):
    """Self-coding workspace plugin. See module docstring for the surface."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._workspace: WorkspaceManager | None = None
        self._runner = CodeRunner()
        self._envs = CondaEnvManager(runner=self._runner)
        self._git = GitCommitter(runner=self._runner)
        self._gate: ApprovalGate | None = None
        self._learning: ErrorLearning | None = None
        self._brain: CoderBrain | None = None

        # Config (resolved in on_load)
        self._workspace_root: str = "workspace"
        self._approval_timeout: float = 180.0
        self._auto_low: bool = True
        self._run_default_timeout: float = 30.0
        self._run_max_output: int = 1_048_576
        self._run_default_python: str = "python"
        self._coder_brain_enabled: bool = True
        self._coder_brain_brain: str = "a4b"
        self._coder_max_steps: int = 12
        self._coder_max_retries: int = 3
        self._coder_history: int = 6
        self._error_learning_enabled: bool = True
        self._error_learning_recall: int = 3

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        cfg = self.api.get_config()
        self._workspace_root = str(cfg.get("workspace_root") or "workspace")
        self._approval_timeout = float(cfg.get("approval_timeout_seconds") or 180.0)
        self._auto_low = bool(cfg.get("auto_approve_low_risk", True))
        self._run_default_timeout = float(cfg.get("run_default_timeout_seconds") or 30.0)
        self._run_max_output = int(cfg.get("run_max_output_bytes") or 1_048_576)
        self._run_default_python = str(cfg.get("run_default_python") or "python")
        self._coder_brain_enabled = bool(cfg.get("coder_brain_enabled", True))
        self._coder_brain_brain = str(cfg.get("coder_brain_default_brain") or "a4b")
        self._coder_max_steps = int(cfg.get("coder_max_steps") or 12)
        self._coder_max_retries = int(cfg.get("coder_max_retries_per_step") or 3)
        self._coder_history = int(cfg.get("coder_history_window") or 6)
        self._error_learning_enabled = bool(cfg.get("error_learning_enabled", True))
        self._error_learning_recall = int(cfg.get("error_learning_recall_limit") or 3)

        # Resolve workspace dir relative to project root if it isn't absolute.
        ws_path = Path(self._workspace_root)
        if not ws_path.is_absolute():
            ws_path = Path.cwd() / ws_path
        self._workspace = WorkspaceManager(ws_path)
        await self._workspace.ensure_layout()

        # Re-tune the runner with config values.
        self._runner = CodeRunner(
            default_timeout=self._run_default_timeout,
            max_output_bytes=self._run_max_output,
        )
        self._envs = CondaEnvManager(runner=self._runner)
        self._git = GitCommitter(runner=self._runner)

        # Approval-Gate + audit DB.
        db = await self.api.get_db()
        self._gate = ApprovalGate(
            broadcast=self.api.ws_broadcast,
            default_timeout=self._approval_timeout,
            auto_approve_low=self._auto_low,
        )
        await self._gate.init_db(db)

        # Memory-driven error-learning.
        memory = getattr(self.api._app, "memory", None) if self._error_learning_enabled else None
        self._learning = ErrorLearning(memory, recall_limit=self._error_learning_recall)

        # Coder-Brain (high-level multi-step loop).
        if self._coder_brain_enabled:
            self._brain = CoderBrain(
                llm_chat=self.api.llm_chat,
                tool_runner=self._dispatch_tool,
                broadcast=self.api.ws_broadcast,
                error_learning=self._learning,
                brain=self._coder_brain_brain,
                max_steps=self._coder_max_steps,
                max_retries_per_step=self._coder_max_retries,
                history_window=self._coder_history,
            )

        log.info(
            "coder.loaded root=%s coder_brain=%s git=%s",
            ws_path, self._coder_brain_enabled, self._git.is_available(),
        )

    async def on_enable(self) -> None:
        api = self.api

        # ── LOW-risk tools ────────────────────────────────────────────
        api.register_tool(
            "workspace_list_projects",
            self._tool_list_projects,
            description="Liste alle Projekte im Workspace (skill / project / extension).",
            schema=WORKSPACE_LIST_PROJECTS_SCHEMA,
        )
        api.register_tool(
            "workspace_list",
            self._tool_list,
            description="Liste Dateien in einem Projekt-Ordner.",
            schema=WORKSPACE_LIST_SCHEMA,
        )
        api.register_tool(
            "workspace_read",
            self._tool_read,
            description="Lies eine Textdatei aus einem Projekt.",
            schema=WORKSPACE_READ_SCHEMA,
        )
        api.register_tool(
            "workspace_git_log",
            self._tool_git_log,
            description="Zeige die letzten Commits eines Projekts.",
            schema=WORKSPACE_GIT_LOG_SCHEMA,
        )
        api.register_tool(
            "workspace_git_diff",
            self._tool_git_diff,
            description="Zeige uncommittete Änderungen vs. HEAD.",
            schema=WORKSPACE_GIT_DIFF_SCHEMA,
        )

        # ── MED-risk tools (modal) ────────────────────────────────────
        api.register_tool(
            "workspace_init_project",
            self._tool_init_project,
            description="Lege ein neues Workspace-Projekt an (skill / project / extension).",
            schema=WORKSPACE_INIT_SCHEMA,
        )
        api.register_tool(
            "workspace_write",
            self._tool_write,
            description="Schreibe Inhalt in eine Datei. Auto-Commit nach Erfolg.",
            schema=WORKSPACE_WRITE_SCHEMA,
        )
        api.register_tool(
            "workspace_create_env",
            self._tool_create_env,
            description="Erzeuge ein .venv für ein Projekt.",
            schema=WORKSPACE_CREATE_ENV_SCHEMA,
        )
        api.register_tool(
            "workspace_install",
            self._tool_install,
            description="pip install Pakete in das Projekt-venv.",
            schema=WORKSPACE_INSTALL_SCHEMA,
        )

        # ── HIGH-risk tools ──────────────────────────────────────────
        api.register_tool(
            "workspace_run",
            self._tool_run,
            description="Führe einen SKILL via subprocess aus (NUR kind=skill, NIE projects).",
            schema=WORKSPACE_RUN_SCHEMA,
        )
        api.register_tool(
            "workspace_delete",
            self._tool_delete,
            description="Lösche eine Datei (kein Verzeichnis).",
            schema=WORKSPACE_DELETE_SCHEMA,
        )
        api.register_tool(
            "workspace_git_revert",
            self._tool_git_revert,
            description="Hard reset auf einen früheren Commit (zerstörerisch!).",
            schema=WORKSPACE_GIT_REVERT_SCHEMA,
        )

        # ── Phase 11: extension publish + reload ─────────────────────
        api.register_tool(
            "coder_publish_extension",
            self._tool_publish_extension,
            description=(
                "Veröffentliche eine Extension aus workspace/extensions/<name>/ "
                "ins laufende Lexy-System. Kopiert den Code nach plugins/<name>/ "
                "und (re)lädt das Plugin zur Laufzeit — ohne Backend-Restart. "
                "HIGH risk: führt fremden Code direkt im Lexy-Prozess aus."
            ),
            schema=PUBLISH_EXTENSION_SCHEMA,
        )
        api.register_tool(
            "coder_unpublish_extension",
            self._tool_unpublish_extension,
            description=(
                "Entferne eine Extension aus dem laufenden Lexy-System. "
                "Lädt sie ab und löscht plugins/<name>/. workspace/extensions/<name>/ "
                "bleibt unangetastet."
            ),
            schema=UNPUBLISH_EXTENSION_SCHEMA,
        )

        # ── Coder-Brain top-level tools ───────────────────────────────
        if self._brain is not None:
            api.register_tool(
                "coder_task",
                self._tool_coder_task,
                description="Starte einen Plan→Code→Test→Reflect Coder-Run. Liefert sofort task_id.",
                schema=CODER_TASK_SCHEMA,
            )
            api.register_tool(
                "coder_status",
                self._tool_coder_status,
                description="Status + Step-Log eines Coder-Runs.",
                schema=CODER_STATUS_SCHEMA,
            )
            api.register_tool(
                "coder_stop",
                self._tool_coder_stop,
                description="Brich einen laufenden Coder-Run ab.",
                schema=CODER_STOP_SCHEMA,
            )
            api.register_tool(
                "coder_list_tasks",
                self._tool_coder_list_tasks,
                description="Liste laufende + jüngste Coder-Runs.",
                schema=CODER_LIST_TASKS_SCHEMA,
            )

        # WebSocket handler — Mike's approval modal sends decisions here.
        api.register_ws_handler(
            "coder_approval_response", self._ws_approval_response,
        )
        api.register_ws_handler(
            "coder_status_query", self._ws_status_query,
        )

        log.info("coder.enabled tools=%d", 9 + (4 if self._brain else 0))

    async def on_disable(self) -> None:
        # Cancel running brain tasks so they don't keep churning after
        # the plugin is taken down.
        if self._brain is not None:
            for task in self._brain.list_all():
                if task.state in ("planning", "running", "reflecting"):
                    await self._brain.stop(task.task_id)
        log.info("coder.disabled")

    # ─── Internal helpers ───────────────────────────────────────────

    def _ws(self) -> WorkspaceManager:
        if self._workspace is None:
            raise RuntimeError("workspace not initialised")
        return self._workspace

    async def _ask(
        self,
        *,
        action: str,
        risk: str,
        payload: dict[str, Any],
        preview: str = "",
        session_id: str = "",
    ) -> bool:
        """Convenience: ask the gate, return True iff approved."""
        if self._gate is None:
            return False
        decision = await self._gate.request(
            action=action,
            risk=risk,
            payload=payload,
            preview=preview,
            session_id=session_id,
        )
        return decision.approved

    async def _dispatch_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Routes a tool name to its handler — used by the CoderBrain."""
        handler = {
            "workspace_list_projects": self._tool_list_projects,
            "workspace_list": self._tool_list,
            "workspace_read": self._tool_read,
            "workspace_init_project": self._tool_init_project,
            "workspace_write": self._tool_write,
            "workspace_create_env": self._tool_create_env,
            "workspace_install": self._tool_install,
            "workspace_run": self._tool_run,
            "workspace_delete": self._tool_delete,
            "workspace_git_log": self._tool_git_log,
            "workspace_git_diff": self._tool_git_diff,
            "workspace_git_revert": self._tool_git_revert,
        }.get(name)
        if handler is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            return await handler(**arguments)
        except TypeError as exc:
            return {"ok": False, "error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{name} failed: {exc}"}

    # ─── LOW-risk tool handlers ─────────────────────────────────────

    async def _tool_list_projects(self) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_list_projects", risk=RISK_LOW, payload={},
        ):
            return {"ok": False, "error": "denied"}
        projects = await self._ws().list_projects()
        return {"ok": True, "projects": [p.to_public() for p in projects]}

    async def _tool_list(
        self, kind: str, name: str, rel_path: str = ""
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_list", risk=RISK_LOW,
            payload={"kind": kind, "name": name, "rel_path": rel_path},
        ):
            return {"ok": False, "error": "denied"}
        try:
            entries = self._ws().list_files(kind=kind, name=name, rel_path=rel_path)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "entries": [e.to_public() for e in entries]}

    async def _tool_read(
        self, kind: str, name: str, rel_path: str
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_read", risk=RISK_LOW,
            payload={"kind": kind, "name": name, "rel_path": rel_path},
            preview=f"Read {kind}/{name}/{rel_path}",
        ):
            return {"ok": False, "error": "denied"}
        try:
            text = self._ws().read_file(kind=kind, name=name, rel_path=rel_path)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "content": text, "lines": text.count("\n") + (0 if text.endswith("\n") else 1)}

    async def _tool_git_log(
        self, kind: str, name: str, limit: int = 10
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_git_log", risk=RISK_LOW,
            payload={"kind": kind, "name": name, "limit": limit},
        ):
            return {"ok": False, "error": "denied"}
        try:
            project = self._ws().get_project(name, kind)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        try:
            entries = await self._git.log(project.root, limit=limit)
        except GitNotAvailable as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "commits": [c.to_public() for c in entries]}

    async def _tool_git_diff(
        self, kind: str, name: str
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_git_diff", risk=RISK_LOW,
            payload={"kind": kind, "name": name},
        ):
            return {"ok": False, "error": "denied"}
        try:
            project = self._ws().get_project(name, kind)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        try:
            diff = await self._git.diff(project.root)
        except GitNotAvailable as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "diff": diff}

    # ─── MED-risk tool handlers ────────────────────────────────────

    async def _tool_init_project(
        self, kind: str, name: str, description: str = ""
    ) -> dict[str, Any]:
        preview = f"Lege {kind}/{name} an"
        if not await self._ask(
            action="workspace_init_project", risk=RISK_MED,
            payload={"kind": kind, "name": name, "description": description},
            preview=preview,
        ):
            return {"ok": False, "error": "denied"}
        try:
            project = await self._ws().init_project(
                name=name, kind=kind, description=description,
            )
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        # Best-effort initial git commit. Skip silently if git missing.
        if self._git.is_available():
            try:
                await self._git.init(project.root)
                await self._git.add_and_commit(
                    project.root,
                    files=["-A"],
                    message=f"init: scaffold {kind}/{name}",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("coder.init_git_failed err=%s", exc)
        return {"ok": True, "project": project.to_public()}

    async def _tool_write(
        self,
        kind: str,
        name: str,
        rel_path: str,
        content: str,
        commit_message: str = "",
    ) -> dict[str, Any]:
        # Show a tiny diff preview if the file already exists, otherwise
        # show the first 800 chars of the new content.
        try:
            existing = self._ws().read_file(kind=kind, name=name, rel_path=rel_path)
        except (WorkspacePathError, WorkspaceNotFoundError):
            existing = None
        preview = _build_write_preview(rel_path, existing, content)
        if not await self._ask(
            action="workspace_write", risk=RISK_MED,
            payload={"kind": kind, "name": name, "rel_path": rel_path, "size": len(content)},
            preview=preview,
        ):
            return {"ok": False, "error": "denied"}
        try:
            path = await self._ws().write_file(
                kind=kind, name=name, rel_path=rel_path, content=content,
            )
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        commit_info = None
        if self._git.is_available():
            try:
                project = self._ws().get_project(name, kind)
                await self._git.init(project.root)
                rel_for_git = path.relative_to(project.root).as_posix()
                commit = await self._git.add_and_commit(
                    project.root,
                    files=[rel_for_git],
                    message=commit_message or f"lexy: edit {rel_for_git}",
                )
                if commit is not None:
                    commit_info = commit.to_public()
            except Exception as exc:  # noqa: BLE001
                log.warning("coder.write_git_failed err=%s", exc)
        return {
            "ok": True,
            "path": str(path),
            "size": len(content),
            "commit": commit_info,
        }

    async def _tool_create_env(
        self, kind: str, name: str, python: str = ""
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_create_env", risk=RISK_MED,
            payload={"kind": kind, "name": name, "python": python},
            preview=f"Erzeuge .venv in {kind}/{name}",
        ):
            return {"ok": False, "error": "denied"}
        try:
            project = self._ws().get_project(name, kind)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        info, run = await self._envs.create_venv(
            project.root, python=python or None,
        )
        return {
            "ok": run.ok or info.exists,
            "env": info.to_public(),
            "run": run.to_public(),
        }

    async def _tool_install(
        self,
        kind: str,
        name: str,
        packages: list[str],
        upgrade: bool = False,
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_install", risk=RISK_MED,
            payload={"kind": kind, "name": name, "packages": packages, "upgrade": upgrade},
            preview=f"pip install {' '.join(packages)} in {kind}/{name}",
        ):
            return {"ok": False, "error": "denied"}
        try:
            project = self._ws().get_project(name, kind)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        info = self._envs.venv_info(project.root)
        if not info.exists:
            return {"ok": False, "error": "no .venv — run workspace_create_env first"}
        run = await self._envs.pip_install(info, packages, upgrade=upgrade)
        return {"ok": run.ok, "run": run.to_public()}

    # ─── HIGH-risk tool handlers ───────────────────────────────────

    async def _tool_run(
        self,
        name: str,
        args: list[str] | None = None,
        entrypoint: str = "skill.py",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        # Skills only — Mike's rule: projects are HIS to run.
        kind = "skill"
        cmd_args = list(args or [])
        preview = f"python {entrypoint} {' '.join(cmd_args)}"
        if not await self._ask(
            action="workspace_run", risk=RISK_HIGH,
            payload={
                "kind": kind, "name": name,
                "entrypoint": entrypoint, "args": cmd_args,
                "timeout_seconds": timeout_seconds,
            },
            preview=preview,
        ):
            return {"ok": False, "error": "denied"}
        try:
            project = self._ws().get_project(name, kind)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        info = self._envs.venv_info(project.root)
        python_exe = str(info.python) if info.exists else self._run_default_python
        cmd = [python_exe, entrypoint, *cmd_args]
        run = await self._runner.run(
            cmd,
            cwd=str(project.root),
            timeout=timeout_seconds if timeout_seconds is not None else self._run_default_timeout,
        )
        return {"ok": run.ok, "run": run.to_public()}

    async def _tool_delete(
        self, kind: str, name: str, rel_path: str
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_delete", risk=RISK_HIGH,
            payload={"kind": kind, "name": name, "rel_path": rel_path},
            preview=f"DELETE {kind}/{name}/{rel_path}",
        ):
            return {"ok": False, "error": "denied"}
        try:
            removed = await self._ws().delete_file(
                kind=kind, name=name, rel_path=rel_path,
            )
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        if not removed:
            return {"ok": False, "error": "file not found"}
        commit_info = None
        if self._git.is_available():
            try:
                project = self._ws().get_project(name, kind)
                commit = await self._git.add_and_commit(
                    project.root,
                    files=[rel_path],
                    message=f"lexy: delete {rel_path}",
                )
                if commit is not None:
                    commit_info = commit.to_public()
            except Exception as exc:  # noqa: BLE001
                log.warning("coder.delete_git_failed err=%s", exc)
        return {"ok": True, "deleted": True, "commit": commit_info}

    async def _tool_git_revert(
        self, kind: str, name: str, commit_ref: str
    ) -> dict[str, Any]:
        if not await self._ask(
            action="workspace_git_revert", risk=RISK_HIGH,
            payload={"kind": kind, "name": name, "commit_ref": commit_ref},
            preview=f"git reset --hard {commit_ref} in {kind}/{name}",
        ):
            return {"ok": False, "error": "denied"}
        try:
            project = self._ws().get_project(name, kind)
        except (WorkspacePathError, WorkspaceNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        try:
            run = await self._git.revert(project.root, commit_ref=commit_ref)
        except GitNotAvailable as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": run.ok, "run": run.to_public()}

    # ─── Coder-Brain tools ─────────────────────────────────────────

    async def _tool_coder_task(
        self,
        description: str,
        kind: str = "skill",
        project: str = "",
    ) -> dict[str, Any]:
        if self._brain is None:
            return {"ok": False, "error": "coder_brain disabled in config"}
        task_id = await self._brain.submit(
            description=description,
            kind=kind,
            project=project,
            tool_catalog=_TOOL_CATALOG,
        )
        return {"ok": True, "task_id": task_id}

    async def _tool_coder_status(self, task_id: str) -> dict[str, Any]:
        if self._brain is None:
            return {"ok": False, "error": "coder_brain disabled"}
        task = self._brain.get(task_id)
        if task is None:
            return {"ok": False, "error": "task_id unknown"}
        return {"ok": True, "task": task.to_public()}

    async def _tool_coder_stop(self, task_id: str) -> dict[str, Any]:
        if self._brain is None:
            return {"ok": False, "error": "coder_brain disabled"}
        ok = await self._brain.stop(task_id)
        return {"ok": ok}

    async def _tool_coder_list_tasks(self) -> dict[str, Any]:
        if self._brain is None:
            return {"ok": False, "error": "coder_brain disabled"}
        return {
            "ok": True,
            "tasks": [t.to_public() for t in self._brain.list_all()],
        }

    # ─── Phase 11: publish / unpublish extension ──────────────────

    # Names that ship with Lexy core — refusing to overwrite them is the
    # cheap way to keep "publish my extension" from accidentally
    # replacing the orchestrator or character_chat plugin if Mike picks
    # the same name. Read fresh from disk each call so adding a new
    # core plugin doesn't require code changes here.
    def _core_plugin_names(self) -> set[str]:
        repo_plugins_root = Path("plugins")
        if not repo_plugins_root.is_dir():
            return set()
        return {
            entry.name
            for entry in repo_plugins_root.iterdir()
            if entry.is_dir() and (entry / "plugin.yaml").exists()
            # Anything not coming from workspace/extensions/ is "core"
            # for protection purposes — even sister extensions Mike
            # already published, so a re-publish doesn't blast over
            # changes he's been editing in plugins/ directly.
        }

    async def _tool_publish_extension(self, name: str) -> dict[str, Any]:
        if not name or not name.strip():
            return {"ok": False, "error": "name_required"}
        # Reuse the workspace name validator (alphanumeric/-/_, no
        # reserved Windows names, ≤64 chars).
        try:
            self._ws()._validate_name(name)
        except WorkspacePathError as exc:
            return {"ok": False, "error": str(exc)}

        ext_root = self._ws().kind_dir("extension") / name
        manifest_path = ext_root / "plugin.yaml"
        if not manifest_path.is_file():
            return {
                "ok": False,
                "error": f"no plugin.yaml under workspace/extensions/{name}/",
            }
        # Validate the manifest BEFORE asking for approval — bad YAML or a
        # missing entry shouldn't trigger a modal.
        try:
            from lexy_core.plugin_system.plugin_manifest import PluginManifest
            manifest = PluginManifest.from_yaml(manifest_path)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"invalid plugin.yaml: {exc}"}
        if manifest.name != name:
            return {
                "ok": False,
                "error": (
                    f"manifest name mismatch: directory '{name}' but "
                    f"plugin.yaml says '{manifest.name}'. Rename one."
                ),
            }

        target_root = Path("plugins") / name
        is_first_time = not target_root.exists()

        preview_lines = [
            f"Publish workspace/extensions/{name}/ → plugins/{name}/",
            f"manifest entry: {manifest.entry}",
            f"capabilities: {', '.join(manifest.capabilities) or '(none)'}",
            "" if is_first_time else "↑ overwrites existing plugin code (hot-reload)",
        ]
        if not await self._ask(
            action="coder_publish_extension",
            risk=RISK_HIGH,
            payload={"name": name, "first_time": is_first_time},
            preview="\n".join(line for line in preview_lines if line),
        ):
            return {"ok": False, "error": "denied"}

        loader = getattr(self.api._app, "plugin_loader", None)
        if loader is None:
            return {"ok": False, "error": "plugin_loader unavailable"}

        # Refuse to publish over a CORE plugin (one that wasn't created
        # via this tool). The ``_published.flag`` sentinel marks dirs we
        # control — its absence means the dir was hand-crafted or shipped.
        sentinel = target_root / "_published.flag"
        if target_root.exists() and not sentinel.exists():
            return {
                "ok": False,
                "error": (
                    f"plugins/{name}/ exists and is not marked as published "
                    "(missing _published.flag). Delete it manually if you "
                    "really want to overwrite a core plugin."
                ),
            }

        # If the plugin is currently loaded, take it down before clobbering
        # files — otherwise Windows file locks bite us on the .pyd / .dll.
        if loader.is_loaded(name):
            try:
                await loader.unload_plugin(name)
            except Exception as exc:  # noqa: BLE001
                log.warning("coder.unload_before_publish_failed err=%s", exc)

        import shutil
        try:
            if target_root.exists():
                shutil.rmtree(target_root)
            shutil.copytree(
                str(ext_root), str(target_root),
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache",
                    ".venv", "venv", ".mypy_cache",
                ),
            )
            sentinel.write_text(
                f"Published from workspace/extensions/{name}/\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            log.error("coder.publish_copy_failed err=%s", exc)
            return {"ok": False, "error": f"copy failed: {exc}"}

        # Hot-(re)load. ``reload_plugin`` auto-falls-through to
        # ``load_plugin`` when the plugin wasn't loaded before.
        ok = await loader.reload_plugin(name)
        await self.api.ws_broadcast(
            {
                "type": "coder_extension_published",
                "name": name,
                "first_time": is_first_time,
                "loaded": ok,
            }
        )
        return {
            "ok": ok,
            "name": name,
            "first_time": is_first_time,
            "loaded": ok,
            "tools_after": loader.get_plugin_info() if ok else None,
        }

    async def _tool_unpublish_extension(self, name: str) -> dict[str, Any]:
        if not name or not name.strip():
            return {"ok": False, "error": "name_required"}
        try:
            self._ws()._validate_name(name)
        except WorkspacePathError as exc:
            return {"ok": False, "error": str(exc)}

        target_root = Path("plugins") / name
        sentinel = target_root / "_published.flag"
        if not target_root.exists():
            return {"ok": False, "error": f"plugins/{name}/ does not exist"}
        if not sentinel.exists():
            return {
                "ok": False,
                "error": (
                    f"plugins/{name}/ is not marked as published — refusing "
                    "to remove a core plugin via this tool."
                ),
            }

        if not await self._ask(
            action="coder_unpublish_extension",
            risk=RISK_HIGH,
            payload={"name": name},
            preview=f"Unload + delete plugins/{name}/ (workspace copy stays).",
        ):
            return {"ok": False, "error": "denied"}

        loader = getattr(self.api._app, "plugin_loader", None)
        if loader is None:
            return {"ok": False, "error": "plugin_loader unavailable"}

        if loader.is_loaded(name):
            try:
                await loader.unload_plugin(name)
            except Exception as exc:  # noqa: BLE001
                log.warning("coder.unload_failed err=%s", exc)

        import shutil
        try:
            shutil.rmtree(str(target_root))
        except Exception as exc:  # noqa: BLE001
            log.error("coder.unpublish_rm_failed err=%s", exc)
            return {"ok": False, "error": f"rm failed: {exc}"}

        await self.api.ws_broadcast(
            {"type": "coder_extension_unpublished", "name": name}
        )
        return {"ok": True, "name": name}

    # ─── WS handlers ───────────────────────────────────────────────

    async def _ws_approval_response(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        if self._gate is None:
            return
        request_id = str(message.get("request_id") or "")
        decision = str(message.get("decision") or "")
        if not request_id or decision not in ("approve", "deny", "approve_session"):
            await client.send_json(
                {"type": "coder_approval_ack", "ok": False, "error": "bad request"}
            )
            return
        approved = decision in ("approve", "approve_session")
        reason = "approve_session" if decision == "approve_session" else (
            "user" if approved else "user_denied"
        )
        ok = self._gate.resolve(
            request_id=request_id, approved=approved, reason=reason,
        )
        if approved and decision == "approve_session":
            session_id = str(message.get("session_id") or "")
            action = str(message.get("action") or "")
            if session_id and action:
                self._gate.grant_session(session_id=session_id, action=action)
        await client.send_json(
            {"type": "coder_approval_ack", "ok": ok, "request_id": request_id}
        )

    async def _ws_status_query(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        if self._brain is None:
            await client.send_json(
                {"type": "coder_status", "ok": False, "error": "coder_brain disabled"}
            )
            return
        task_id = str(message.get("task_id") or "")
        if task_id:
            task = self._brain.get(task_id)
            await client.send_json(
                {
                    "type": "coder_status",
                    "ok": task is not None,
                    "task": task.to_public() if task is not None else None,
                }
            )
            return
        await client.send_json(
            {
                "type": "coder_status",
                "ok": True,
                "tasks": [t.to_public() for t in self._brain.list_all()],
            }
        )


# ─── Helpers ────────────────────────────────────────────────────────


def _build_write_preview(rel_path: str, existing: str | None, new: str) -> str:
    if existing is None:
        head = new[:800]
        return f"NEW {rel_path}\n{'-'*40}\n{head}{'... (truncated)' if len(new)>800 else ''}"
    if existing == new:
        return f"{rel_path}: identical (no-op)"
    # Tiny inline diff: show first 20 differing lines.
    a_lines = existing.splitlines()
    b_lines = new.splitlines()
    import difflib
    diff_lines = list(
        difflib.unified_diff(
            a_lines, b_lines,
            fromfile=f"{rel_path} (current)",
            tofile=f"{rel_path} (new)",
            lineterm="",
            n=2,
        )
    )
    body = "\n".join(diff_lines[:60])
    if len(diff_lines) > 60:
        body += "\n... (diff truncated)"
    return body
