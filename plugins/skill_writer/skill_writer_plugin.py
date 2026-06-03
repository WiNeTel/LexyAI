"""
Lexy AI - Skill Writer Plugin (Phase 11 — agentskills.io compliant).

Self-improvement system: Lexy writes, validates, and executes custom
skills, and spawns autonomous sub-agents for independent tasks.

Phase 11 brings the on-disk format onto the open agentskills.io
standard (originally Anthropic's "Agent Skills"; now adopted by
Claude Code, Cursor, OpenCode, GitHub Copilot, Goose, Letta, …).
A skill is now a *folder* with ``SKILL.md`` (YAML frontmatter +
markdown body) and an optional ``scripts/skill.py`` entry point —
identical layout to every other skills-compatible agent, so Mike's
skills round-trip seamlessly between Lexy and Cursor.

Features:

* **write_skill** — LLM writes a validated SKILL.md + scripts/skill.py.
* **run_skill** — Execute the primary script with sandboxed imports.
* **list_skills / delete_skill** — Manage the skill registry.
* **spawn_agent** — Launch an autonomous agent with own conversation loop.
* **list_agents / stop_agent / agent_result** — Manage running agents.
* **Skill proposals** — Auto-generated skills require approval before activation.
* **Usage tracking** — SQLite-backed success/failure counters per skill.
"""

from __future__ import annotations

import importlib.util
import asyncio
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .auto_agent import AgentManager
from .skill_curator import DEFAULT_MANAGED_SOURCES, SkillCurator
from .skill_executions import SkillExecutionLog, should_refine
from .skill_index import build_catalog_block
from .skill_loader import SkillLoaderError, load_skill_folder
from .skill_registry import SkillRegistry
from .skill_spec import SkillSpecError
from .skill_template import emit_skill_folder, sanitize_skill_name
from .skill_validator import SkillValidator
from .task_detector import TaskDetector, TaskSignal

log = get_logger(module="skill_writer_plugin")


# P4 — prompt the author brain uses to draft a reusable skill from a detected
# tool pattern. Output is strict JSON so it can be parsed + auto-registered.
_AUTO_SKILL_DRAFT_PROMPT = """\
Du bist Lexys Skill-Autor. Aus einer wiederkehrenden Aufgabe sollst du einen \
wiederverwendbaren Skill bauen.

Kontext:
- Auslöser: %REASON%
- Genutzte Tools (Reihenfolge): %TOOLS%
- Letzte User-Aufgabe: %REQUEST%

Schreibe einen kleinen Python-Skill, der diese Aufgabe automatisiert. Der Code \
ist NUR der Body einer Funktion `execute(api, **kwargs)` — keine Signatur, kein \
Header, keine Markdown-Fences. Verfügbar sind `api` (Lexy PluginAPI) und \
`kwargs`. Erlaubte Imports: json, re, datetime, math, collections, itertools, \
functools, pathlib, time, hashlib, base64. KEIN os/sys/subprocess/open.

Antworte AUSSCHLIESSLICH als JSON (kein weiterer Text):
{"name": "kebab-case-name", "description": "Was der Skill tut + wann nutzen", \
"code": "return {...}"}"""


# P5 — prompt the refine brain uses to patch a failing skill. Same strict-JSON
# contract as the draft prompt so the result can be parsed + re-registered.
_REFINE_SKILL_PROMPT = """\
Du bist Lexys Skill-Reparateur. Ein Skill schlägt wiederholt fehl. Repariere ihn.

Skill: %NAME%
Beschreibung: %DESCRIPTION%

Aktueller Code (Body von `execute(api, **kwargs)`):
%CODE%

Letzte Fehler:
%FAILURES%

Schreibe eine korrigierte Version. Der Code ist NUR der Body von \
`execute(api, **kwargs)` — keine Signatur, kein Header, keine Fences. Erlaubte \
Imports: json, re, datetime, math, collections, itertools, functools, pathlib, \
time, hashlib, base64. KEIN os/sys/subprocess/open.

Antworte AUSSCHLIESSLICH als JSON:
{"name": "%NAME%", "description": "ggf. verbessert", "code": "return {...}"}"""


# Anti-thrash: at most one self-refine attempt per skill per this window.
_REFINE_COOLDOWN_SECONDS = 300.0


def _parse_skill_draft(response: str) -> tuple[str, str, str] | None:
    """Parse the author brain's JSON draft → ``(name, description, code)``.

    Returns ``None`` when the output isn't usable (the auto-learn attempt is
    then quietly dropped — it's best-effort).
    """
    text = str(response or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    code = str(data.get("code") or "").strip()
    if not name or not description or not code:
        return None
    return (name, description, code)


# ── Tool Schemas ───────────────────────────────────────────────────────────


WRITE_SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Skill name in agentskills.io format: 1-64 chars, "
                "lowercase letters / digits / single hyphens "
                "(e.g. 'calc-bmi'). Must equal the on-disk folder name."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "1-1024 chars. Describes WHAT the skill does AND "
                "WHEN to use it (the agent reads this at discovery)."
            ),
        },
        "code": {
            "type": "string",
            "description": (
                "Python body for scripts/skill.py's execute() function. "
                "Must use 'api' parameter and return a dict. "
                "No imports of os/subprocess/sys allowed."
            ),
        },
        "body_md": {
            "type": "string",
            "description": (
                "(optional) Markdown body for SKILL.md — step-by-step "
                "instructions, examples, edge cases. Recommended to "
                "stay under ~500 lines. The agent loads this on "
                "activation but not at boot (progressive disclosure)."
            ),
        },
        "license": {
            "type": "string",
            "description": (
                "(optional) License name, e.g. 'Apache-2.0' or "
                "'Proprietary'. Lands as 'license:' in the frontmatter."
            ),
        },
        "compatibility": {
            "type": "string",
            "description": (
                "(optional) Environment requirements, max 500 chars. "
                "E.g. 'Requires Python 3.11+ and httpx'."
            ),
        },
        "tags": {
            "type": "string",
            "description": (
                "(optional) Comma-separated tags, persisted in "
                "metadata.tags."
            ),
        },
    },
    "required": ["name", "description", "code"],
}

LIST_SKILLS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "description": "Filter by status: active, disabled, failed (optional)",
        },
    },
}

RUN_SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill_name": {
            "type": "string",
            "description": "Name of the skill to execute",
        },
        "args": {
            "type": "string",
            "description": "JSON string with keyword arguments for the skill",
        },
    },
    "required": ["skill_name"],
}

DELETE_SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill_name": {
            "type": "string",
            "description": "Name of the skill to delete",
        },
    },
    "required": ["skill_name"],
}

SPAWN_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Human-readable agent name (e.g. 'Researcher')",
        },
        "task": {
            "type": "string",
            "description": "Detailed task description for the agent",
        },
        "system_prompt": {
            "type": "string",
            "description": "Custom system prompt (auto-generated if omitted)",
        },
        "brain": {
            "type": "string",
            "description": "LLM brain: 'e4b' (fast, default) or 'a4b' (complex)",
        },
    },
    "required": ["name", "task"],
}

LIST_AGENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

STOP_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "The agent ID to stop",
        },
    },
    "required": ["agent_id"],
}

AGENT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "The agent ID to get results from",
        },
    },
    "required": ["agent_id"],
}


# ── Plugin ─────────────────────────────────────────────────────────────────


class SkillWriterPlugin(BasePlugin):
    """
    Self-improvement plugin: skill authoring + autonomous agents.

    On load, sets up the SQLite tables, skill validator, and skill registry.
    On enable, registers 8 LLM tools, WS handlers, hooks, and event listeners.
    """

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._validator: SkillValidator | None = None
        self._registry: SkillRegistry | None = None
        self._agent_manager: AgentManager | None = None
        self._curator: SkillCurator | None = None
        self._skills_path: Path = Path("./data/skills")
        self._require_approval: bool = True
        self._auto_propose: bool = True
        self._inject_skill_catalog: bool = True
        self._catalog_inline_max: int = 12
        # P4 — autonomous skill learning
        self._detector: TaskDetector | None = None
        self._auto_learn_skills: bool = True
        self._skill_author_brain: str = "e4b"
        # P5 — self-refinement
        self._exec_log: SkillExecutionLog | None = None
        self._self_refine_skills: bool = True
        self._refine_after_failures: int = 3
        self._refine_brain: str = "e4b"
        self._refine_cooldown: dict[str, float] = {}

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        """
        Initialise resources: DB tables, validator, registry, disk scan.
        """
        config = self.api.get_config()

        # Pfade und Config
        self._skills_path = Path(config.get("skills_path", "./data/skills"))
        self._skills_path.mkdir(parents=True, exist_ok=True)
        self._require_approval = bool(config.get("require_approval", True))
        self._auto_propose = bool(config.get("auto_propose", True))
        self._inject_skill_catalog = bool(config.get("inject_skill_catalog", True))
        self._catalog_inline_max = int(config.get("catalog_inline_max", 12))
        self._auto_learn_skills = bool(config.get("auto_learn_skills", True))
        self._skill_author_brain = str(config.get("skill_author_brain", "e4b"))

        max_size = int(config.get("max_skill_size_bytes", 10000))
        allowed_imports = list(config.get("sandbox_allowed_imports", []))
        # __future__ und typing sind immer erlaubt
        for always_allowed in ("__future__", "typing"):
            if always_allowed not in allowed_imports:
                allowed_imports.append(always_allowed)

        self._validator = SkillValidator(
            allowed_imports=allowed_imports,
            max_size_bytes=max_size,
        )

        # DB + Registry
        db = await self.api.get_db()

        # Skill-Proposals Tabelle. Phase 11 erweitert um die
        # Frontmatter-Felder, sodass eine genehmigte Proposal direkt
        # einen kompletten ``data/skills/<name>/`` Folder produziert.
        # ``code`` enthält jetzt nur noch den Python-Body (für
        # scripts/skill.py) statt der ganzen .py-Datei mit Header.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_proposals (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                code        TEXT NOT NULL,
                tags        TEXT NOT NULL DEFAULT '',
                source      TEXT NOT NULL DEFAULT 'auto',
                status      TEXT NOT NULL DEFAULT 'pending',
                created_at  REAL NOT NULL,
                reviewed_at REAL,
                body_md       TEXT NOT NULL DEFAULT '',
                license       TEXT,
                compatibility TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                allowed_tools TEXT
            )
            """
        )
        # Idempotente Migration für DBs aus der Pre-Phase-11-Zeit.
        for col_name, col_decl in (
            ("body_md", "TEXT NOT NULL DEFAULT ''"),
            ("license", "TEXT"),
            ("compatibility", "TEXT"),
            ("metadata_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("allowed_tools", "TEXT"),
        ):
            try:
                await db.execute(
                    f"ALTER TABLE skill_proposals "
                    f"ADD COLUMN {col_name} {col_decl}"
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "duplicate column" not in msg and "already exists" not in msg:
                    raise
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_status "
            "ON skill_proposals(status)"
        )
        await db.commit()

        self._registry = SkillRegistry(db=db, skills_path=self._skills_path)
        await self._registry.init_tables()

        # P5 — skill execution log (for self-refinement error context).
        self._exec_log = SkillExecutionLog(db=db)
        await self._exec_log.init_tables()
        self._self_refine_skills = bool(config.get("self_refine_skills", True))
        self._refine_after_failures = int(config.get("refine_after_failures", 3))
        self._refine_brain = str(config.get("refine_brain", "e4b"))

        # Vorhandene Skills vom Disk einlesen
        scanned = await self._registry.scan_disk()
        if scanned > 0:
            log.info("skill_writer.scanned_disk", new_skills=scanned)

        log.info(
            "skill_writer.loaded",
            skills_path=str(self._skills_path),
            require_approval=self._require_approval,
        )

    async def on_enable(self) -> None:
        """Register tools, WS handlers, hooks, and event listeners."""
        config = self.api.get_config()

        # AgentManager erstellen
        self._agent_manager = AgentManager(
            api=self.api,
            max_concurrent=int(config.get("max_agents", 5)),
            default_timeout=float(config.get("agent_timeout", 120.0)),
            default_max_iter=int(config.get("agent_max_iterations", 10)),
        )

        # ── 8 LLM Tools registrieren ──

        self.api.register_tool(
            name="write_skill",
            handler=self._handle_write_skill,
            description=(
                "Schreibe einen neuen Skill (Python-Micro-Script). "
                "Der Code wird validiert und in data/skills/ gespeichert. "
                "Der Skill muss eine 'execute(api, **kwargs)' Funktion haben. "
                "Erlaubte Imports: json, re, datetime, math, collections, "
                "itertools, functools, pathlib, time, hashlib, base64."
            ),
            schema=WRITE_SKILL_SCHEMA,
        )

        self.api.register_tool(
            name="list_skills",
            handler=self._handle_list_skills,
            description="Liste alle registrierten Skills mit Status und Statistiken.",
            schema=LIST_SKILLS_SCHEMA,
        )

        self.api.register_tool(
            name="run_skill",
            handler=self._handle_run_skill,
            description=(
                "Fuehre einen registrierten Skill aus. "
                "Optionale Args als JSON-String."
            ),
            schema=RUN_SKILL_SCHEMA,
        )

        self.api.register_tool(
            name="delete_skill",
            handler=self._handle_delete_skill,
            description="Loesche einen Skill (Datei + Registry).",
            schema=DELETE_SKILL_SCHEMA,
        )

        self.api.register_tool(
            name="spawn_agent",
            handler=self._handle_spawn_agent,
            description=(
                "Starte einen autonomen Sub-Agent mit eigener Konversation. "
                "Der Agent arbeitet unabhaengig im Hintergrund und nutzt "
                "die gleichen Tools. Gut fuer Recherche, Analyse oder "
                "laengere Aufgaben."
            ),
            schema=SPAWN_AGENT_SCHEMA,
        )

        self.api.register_tool(
            name="list_agents",
            handler=self._handle_list_agents,
            description="Liste alle laufenden und abgeschlossenen Sub-Agents.",
            schema=LIST_AGENTS_SCHEMA,
        )

        self.api.register_tool(
            name="stop_agent",
            handler=self._handle_stop_agent,
            description="Stoppe einen laufenden Sub-Agent anhand seiner ID.",
            schema=STOP_AGENT_SCHEMA,
        )

        self.api.register_tool(
            name="agent_result",
            handler=self._handle_agent_result,
            description=(
                "Hole das Ergebnis eines Sub-Agents inkl. "
                "Konversationsverlauf und Tool-Nutzung."
            ),
            schema=AGENT_RESULT_SCHEMA,
        )

        # ── WebSocket Handlers ──

        self.api.register_ws_handler("skill_list", self._ws_skill_list)
        self.api.register_ws_handler("skill_approve", self._ws_skill_approve)
        self.api.register_ws_handler("skill_reject", self._ws_skill_reject)
        self.api.register_ws_handler("skill_proposals", self._ws_skill_proposals)
        self.api.register_ws_handler(
            "agent_list_request", self._ws_agent_list_request
        )

        # ── P3 — Skill Curator (lifecycle + low-success safety net) ──
        if self._registry is not None:
            self._curator = SkillCurator(
                registry=self._registry,
                skills_path=self._skills_path,
                api=self.api,
                stale_days=int(config.get("stale_days", 30)),
                archive_days=int(config.get("archive_days", 90)),
                min_success_rate=float(config.get("min_success_rate", 0.4)),
                min_runs=int(config.get("min_runs", 5)),
                interval_hours=float(config.get("curator_interval_hours", 24.0)),
                min_idle_minutes=float(config.get("curator_min_idle_minutes", 10.0)),
            )
            if bool(config.get("curator_enabled", True)):
                self._curator.start()
            self.api.register_ws_handler(
                "skill_curator_run", self._ws_curator_run
            )
            self.api.register_ws_handler(
                "skill_curator_status", self._ws_curator_status
            )
            self.api.register_ws_handler("skill_pin", self._ws_skill_pin)
            self.api.register_ws_handler("skill_restore", self._ws_skill_restore)
            self.api.on_event("core.user_message", self._on_user_activity)

        # ── Hook: Inject agent status into prompt context ──

        self.api.register_hook(
            "before_prompt_build",
            self._hook_inject_agent_context,
            priority=60,
        )

        # P6 — surface the skill catalogue into the system prompt so the
        # model knows which skills exist (progressive disclosure: names +
        # descriptions only). Uses ``system_prompt_parts`` — the channel
        # agent._plan actually reads.
        self.api.register_hook(
            "before_prompt_build",
            self._hook_inject_skill_catalog,
            priority=55,
        )

        # ── Events ──

        self.api.on_event("core.tool_error", self._on_tool_error)
        self.api.on_event("agent.done", self._on_agent_done)

        # P4 — autonomous skill learning. The detector watches each turn's
        # tool usage; a skill-worthy pattern is drafted + activated live.
        self._detector = TaskDetector(
            complex_threshold=int(config.get("complex_task_tool_threshold", 8)),
            repeat_threshold=int(config.get("repeat_threshold", 3)),
        )
        self.api.register_hook(
            "after_response_send",
            self._on_response_for_skill_learning,
            priority=50,
        )

        log.info("skill_writer.enabled")

    async def on_disable(self) -> None:
        """Cleanup: stop the curator loop and all agents."""
        if self._curator is not None:
            await self._curator.stop()
            self._curator = None
        if self._agent_manager is not None:
            await self._agent_manager.cleanup()
            self._agent_manager = None
        log.info("skill_writer.disabled")

    # ─── Tool: write_skill ──────────────────────────────────────────

    async def _handle_write_skill(
        self,
        name: str,
        description: str,
        code: str,
        body_md: str | None = None,
        license: str | None = None,
        compatibility: str | None = None,
        tags: str | None = None,
    ) -> dict[str, Any]:
        """Validate, write, and register a new skill (Phase 11 — folder).

        ``code`` lands as the body of ``scripts/skill.py``'s ``execute()``
        function; ``body_md`` (if provided) becomes the SKILL.md
        markdown body. Optional frontmatter fields land in metadata.

        When ``require_approval`` is set, the skill goes through the
        proposal table first — the genehmigte Proposal landet dann
        eins-zu-eins als Folder auf Disk.
        """
        if self._validator is None or self._registry is None:
            return {"error": "Skill writer not initialised"}

        clean_name = sanitize_skill_name(name)
        tag_list = (
            [t.strip() for t in tags.split(",") if t.strip()]
            if tags
            else []
        )
        metadata_dict: dict[str, str] = {}
        if tag_list:
            metadata_dict["tags"] = ",".join(tag_list)

        # Pre-flight Validierung: nur die Python-Quelle, bevor wir
        # was auf Disk schreiben. Bei Approval bleibt der Code so im
        # Proposal-Tabelle bis Mike approved.
        valid, error = self._validator.validate(code, expect_execute=False)
        # ``expect_execute=False`` weil ``code`` nur der Body ist;
        # die volle Funktion mit Signatur baut das Template.
        if not valid:
            # Wir tun denselben Check nochmal nach dem Folder-Build
            # gegen die fertige scripts/skill.py — aber meistens
            # schlägt's hier schon zu, mit klarer Meldung.
            log.warning(
                "skill_writer.validation_failed",
                name=clean_name,
                error=error,
            )
            return {"error": f"Validation failed: {error}"}

        # Approval-Modus: Proposal erstellen statt direkt speichern.
        if self._require_approval:
            proposal_id = await self._create_proposal(
                name=clean_name,
                description=description,
                code=code,
                tags=",".join(tag_list),
                source_tag="auto",
                body_md=body_md or "",
                license=license,
                compatibility=compatibility,
                metadata=metadata_dict,
            )
            log.info(
                "skill_writer.proposal_created",
                name=clean_name,
                proposal_id=proposal_id,
            )
            await self.api.ws_broadcast({
                "type": "skill_proposal_new",
                "proposal_id": proposal_id,
                "name": clean_name,
                "description": description,
            })
            return {
                "status": "proposal_created",
                "proposal_id": proposal_id,
                "name": clean_name,
                "message": (
                    f"Skill '{clean_name}' wurde als Vorschlag erstellt. "
                    "Warte auf Freigabe."
                ),
            }

        # Direkt speichern (require_approval = false)
        return await self._write_and_register(
            name=clean_name,
            description=description,
            code=code,
            body_md=body_md or "",
            license=license,
            compatibility=compatibility,
            metadata=metadata_dict,
            source_tag="auto",
        )

    async def _write_and_register(
        self,
        *,
        name: str,
        description: str,
        code: str,
        body_md: str = "",
        license: str | None = None,
        compatibility: str | None = None,
        metadata: dict[str, str] | None = None,
        source_tag: str = "auto",
    ) -> dict[str, Any]:
        """Emit the skill folder + register it in the DB.

        Phase 11: skill ist ein Folder. Wir bauen den Folder via
        :func:`emit_skill_folder`, validieren das Resultat, registrieren
        es und broadcasten das Event.
        """
        if self._registry is None or self._validator is None:
            return {"error": "Registry not initialised"}

        # Duplikat-Check (gegen Registry und gegen Disk-Folder).
        existing = await self._registry.get(name)
        if existing is not None:
            return {"error": f"Skill '{name}' already exists"}
        if (self._skills_path / name).exists():
            return {"error": f"Skill folder already exists on disk: {name}"}

        # Folder emittieren.
        try:
            folder = emit_skill_folder(
                name=name,
                description=description,
                code=code,
                target_root=self._skills_path,
                body_md=body_md or None,
                license=license,
                compatibility=compatibility,
                metadata=metadata,
            )
        except (SkillSpecError, FileExistsError) as exc:
            return {"error": f"Folder build failed: {exc}"}

        # Vollständige Folder-Validierung (Frontmatter + alle scripts).
        ok, err = self._validator.validate_folder(folder)
        if not ok:
            # Cleanup damit Disk + Registry konsistent bleiben.
            try:
                from .skill_template import _rm_tree
                _rm_tree(folder)
            except OSError:  # noqa: BLE001
                pass
            return {"error": f"Folder validation failed: {err}"}

        # Frontmatter aus dem fertigen Folder laden, damit die Registry
        # die Spec-konformen Felder direkt von der Source-of-truth nimmt.
        try:
            card = await load_skill_folder(folder)
        except (SkillLoaderError, SkillSpecError) as exc:
            return {"error": f"Loader failed after build: {exc}"}

        skill_id = await self._registry.register(
            name=card.name,
            description=card.description,
            file_path=str(card.folder),
            source=source_tag,
            license=card.frontmatter.license,
            compatibility=card.frontmatter.compatibility,
            metadata=card.frontmatter.metadata,
            allowed_tools=card.frontmatter.allowed_tools,
            body_md=card.frontmatter.body,
        )

        log.info("skill_writer.written", name=card.name, folder=str(card.folder))
        await self.api.emit(
            "skill.created",
            {"name": card.name, "id": skill_id, "source": source_tag},
        )

        return {
            "status": "created",
            "skill_id": skill_id,
            "name": card.name,
            "folder": str(card.folder),
        }

    # ─── Tool: list_skills ──────────────────────────────────────────

    async def _handle_list_skills(
        self,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List all registered skills (Phase 11 — includes spec fields)."""
        if self._registry is None:
            return {"error": "Registry not initialised"}

        entries = await self._registry.list_all(status=status)
        return {
            "count": len(entries),
            "skills": [
                {
                    "name": e.name,
                    "description": e.description,
                    "status": e.status,
                    "source": e.source,
                    "usage_count": e.usage_count,
                    "success_count": e.success_count,
                    "failure_count": e.failure_count,
                    "created_at": e.created_at,
                    # Phase 11 — agentskills.io frontmatter exposure
                    "license": e.license,
                    "compatibility": e.compatibility,
                    "metadata": dict(e.metadata),
                    "allowed_tools": e.allowed_tools,
                    "folder": e.file_path,
                }
                for e in entries
            ],
        }

    # ─── Tool: run_skill ────────────────────────────────────────────

    async def _handle_run_skill(
        self,
        skill_name: str,
        args: str | None = None,
    ) -> dict[str, Any]:
        """Load and execute a registered skill (Phase 11 — folder-based)."""
        if self._registry is None:
            return {"error": "Registry not initialised"}

        entry = await self._registry.get(skill_name)
        if entry is None:
            return {"error": f"Skill '{skill_name}' not found"}
        if entry.status != "active":
            return {"error": f"Skill '{skill_name}' is {entry.status}"}

        # Args parsen (JSON-String oder None)
        kwargs: dict[str, Any] = {}
        if args:
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    kwargs = parsed
                else:
                    return {"error": "args must be a JSON object"}
            except json.JSONDecodeError as exc:
                return {"error": f"Invalid args JSON: {exc}"}

        # Phase 11: ``entry.file_path`` zeigt auf den Folder. Wir laden
        # die SkillCard frisch vom Disk damit ``primary_script`` und
        # die Frontmatter-Validierung zur Run-Zeit nochmal greifen.
        folder = Path(entry.file_path)
        if not folder.is_dir():
            await self._registry.set_status(skill_name, "failed")
            return {"error": f"Skill folder not found: {folder}"}

        try:
            card = await load_skill_folder(folder)
        except (SkillLoaderError, SkillSpecError) as exc:
            await self._registry.set_status(skill_name, "failed")
            return {"error": f"Skill folder invalid: {exc}"}

        if card.primary_script is None:
            await self._registry.set_status(skill_name, "failed")
            return {
                "error": (
                    f"Skill '{skill_name}' has no executable script "
                    "(scripts/skill.py missing)"
                )
            }

        primary_path = card.primary_script_path()
        if primary_path is None or not primary_path.is_file():
            await self._registry.set_status(skill_name, "failed")
            return {"error": f"Primary script missing: {card.primary_script}"}

        # Modul-Name eindeutig pro Skill, sonst kollidieren Helper-
        # Imports zwischen Skills im Modul-Cache.
        module_name = f"lexy_skill_{skill_name.replace('-', '_')}"
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, str(primary_path)
            )
            if spec is None or spec.loader is None:
                return {
                    "error": f"Could not load skill module: {primary_path}"
                }
            module = importlib.util.module_from_spec(spec)
            # Cache-pinning damit Helper-Module die in scripts/ liegen
            # über ``import skill`` o.ä. den richtigen Namespace finden.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)  # type: ignore[union-attr]
            finally:
                sys.modules.pop(module_name, None)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "skill_writer.load_error",
                skill=skill_name,
                error=str(exc),
            )
            await self._registry.set_status(skill_name, "failed")
            return {"error": f"Failed to load skill: {exc}"}

        if not hasattr(module, "execute"):
            await self._registry.set_status(skill_name, "failed")
            return {"error": "Skill has no execute() function"}

        # Ausfuehren
        try:
            result = await module.execute(self.api, **kwargs)
            await self._registry.update_stats(skill_name, success=True)
            if self._exec_log is not None:
                await self._exec_log.record(skill_name, ok=True, args_json=args or "")
            log.info("skill_writer.run_ok", skill=skill_name)
            return {"skill": skill_name, "result": result}
        except Exception as exc:  # noqa: BLE001
            await self._registry.update_stats(skill_name, success=False)
            if self._exec_log is not None:
                await self._exec_log.record(
                    skill_name, ok=False, error=str(exc), args_json=args or ""
                )
            log.error(
                "skill_writer.run_error",
                skill=skill_name,
                error=str(exc),
            )
            # P5 — self-repair once the failure threshold is crossed.
            await self._maybe_refine_skill(skill_name)
            return {"skill": skill_name, "error": str(exc)}

    # ─── Tool: delete_skill ─────────────────────────────────────────

    async def _handle_delete_skill(
        self,
        skill_name: str,
    ) -> dict[str, Any]:
        """Delete a skill (folder + registry entry)."""
        if self._registry is None:
            return {"error": "Registry not initialised"}

        entry = await self._registry.get(skill_name)
        if entry is None:
            return {"error": f"Skill '{skill_name}' not found"}

        # Phase 11: file_path zeigt auf den Folder. Recursive remove.
        target = Path(entry.file_path)
        if target.exists():
            try:
                if target.is_dir():
                    from .skill_template import _rm_tree
                    _rm_tree(target)
                else:
                    # Defensive: ältere Einträge könnten noch auf
                    # eine .py-Datei zeigen (Pre-Phase-11).
                    target.unlink()
                log.info("skill_writer.deleted_from_disk", target=str(target))
            except OSError as exc:
                log.warning(
                    "skill_writer.delete_disk_failed",
                    target=str(target),
                    error=str(exc),
                )

        # Aus Registry entfernen
        await self._registry.delete(skill_name)

        await self.api.emit(
            "skill.deleted", {"name": skill_name, "id": entry.id}
        )
        return {"status": "deleted", "name": skill_name}

    # ─── Tool: spawn_agent ──────────────────────────────────────────

    async def _handle_spawn_agent(
        self,
        name: str,
        task: str,
        system_prompt: str | None = None,
        brain: str = "e4b",
    ) -> dict[str, Any]:
        """Spawn a new autonomous sub-agent."""
        if self._agent_manager is None:
            return {"error": "Agent manager not initialised"}

        result = await self._agent_manager.spawn(
            name=name,
            task=task,
            system_prompt=system_prompt,
            brain=brain,
        )
        return result

    # ─── Tool: list_agents ──────────────────────────────────────────

    async def _handle_list_agents(self) -> dict[str, Any]:
        """List all agents."""
        if self._agent_manager is None:
            return {"error": "Agent manager not initialised"}

        agents = self._agent_manager.list_agents()
        return {"count": len(agents), "agents": agents}

    # ─── Tool: stop_agent ───────────────────────────────────────────

    async def _handle_stop_agent(
        self,
        agent_id: str,
    ) -> dict[str, Any]:
        """Stop a running agent."""
        if self._agent_manager is None:
            return {"error": "Agent manager not initialised"}

        stopped = await self._agent_manager.stop(agent_id)
        if stopped:
            return {"status": "stopped", "agent_id": agent_id}
        return {"error": f"Agent '{agent_id}' not found or already finished"}

    # ─── Tool: agent_result ─────────────────────────────────────────

    async def _handle_agent_result(
        self,
        agent_id: str,
    ) -> dict[str, Any]:
        """Get the final result and conversation of an agent."""
        if self._agent_manager is None:
            return {"error": "Agent manager not initialised"}

        agent = self._agent_manager.get_agent(agent_id)
        if agent is None:
            return {"error": f"Agent '{agent_id}' not found"}

        # Konversation komprimieren (nur letzte N messages)
        conversation = agent.get_conversation()
        max_msgs = 20
        truncated = len(conversation) > max_msgs
        if truncated:
            conversation = conversation[-max_msgs:]

        return {
            "agent_id": agent_id,
            "name": agent.name,
            "status": agent.status,
            "task": agent.task[:500],
            "iterations": agent._iteration,
            "tools_used": [
                {"tool": r["tool"], "result_preview": r["result"][:200]}
                for r in agent.results
            ],
            "final_answer": agent._extract_final_answer()[:1000],
            "conversation": [
                {
                    "role": m["role"],
                    "content": m["content"][:300],
                }
                for m in conversation
            ],
            "truncated": truncated,
        }

    # ─── WebSocket Handlers ─────────────────────────────────────────

    async def _ws_skill_list(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Return full skill list."""
        result = await self._handle_list_skills(
            status=message.get("status"),
        )
        await client.send_json({"type": "skill_list", **result})

    # ─── P3 — Curator WS controls ───────────────────────────────────

    async def _on_user_activity(self, data: dict[str, Any]) -> None:
        """Reset the curator idle timer on every user message."""
        if self._curator is not None:
            self._curator.note_activity()

    async def _ws_curator_run(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Run a curator cycle. ``dry_run`` previews without mutating."""
        if self._curator is None:
            await client.send_json({"type": "error", "error": "curator disabled"})
            return
        dry_run = bool(message.get("dry_run", False))
        report = await self._curator.run(dry_run=dry_run)
        await client.send_json({"type": "skill_curator_result", **report})

    async def _ws_curator_status(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Current lifecycle-state summary."""
        if self._curator is None:
            await client.send_json({"type": "error", "error": "curator disabled"})
            return
        status = await self._curator.status()
        await client.send_json({"type": "skill_curator_status", **status})

    async def _ws_skill_pin(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Pin/unpin a skill so the curator leaves it alone."""
        name = str(message.get("name", ""))
        pinned = bool(message.get("pinned", True))
        if self._curator is None or not name:
            await client.send_json(
                {"type": "error", "error": "missing name or curator disabled"}
            )
            return
        ok = await self._curator.set_pinned(name, pinned)
        await client.send_json(
            {"type": "skill_pinned", "name": name, "pinned": pinned, "ok": ok}
        )

    async def _ws_skill_restore(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Restore an archived skill back to active."""
        name = str(message.get("name", ""))
        if self._curator is None or not name:
            await client.send_json(
                {"type": "error", "error": "missing name or curator disabled"}
            )
            return
        ok = await self._curator.restore(name)
        await client.send_json(
            {"type": "skill_restored", "name": name, "ok": ok}
        )

    async def _ws_skill_approve(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Approve a skill proposal."""
        proposal_id = str(message.get("proposal_id", ""))
        if not proposal_id:
            await client.send_json({
                "type": "error",
                "error": "missing proposal_id",
            })
            return

        result = await self._approve_proposal(proposal_id)
        await client.send_json({"type": "skill_approved", **result})

    async def _ws_skill_reject(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Reject a skill proposal."""
        proposal_id = str(message.get("proposal_id", ""))
        if not proposal_id:
            await client.send_json({
                "type": "error",
                "error": "missing proposal_id",
            })
            return

        result = await self._reject_proposal(proposal_id)
        await client.send_json({"type": "skill_rejected", **result})

    async def _ws_skill_proposals(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: List all pending skill proposals."""
        proposals = await self._list_proposals()
        await client.send_json({
            "type": "skill_proposals",
            "count": len(proposals),
            "proposals": proposals,
        })

    async def _ws_agent_list_request(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """WS: Return agent list."""
        result = await self._handle_list_agents()
        await client.send_json({"type": "agent_list", **result})

    # ─── Hook: Inject agent context ─────────────────────────────────

    async def _hook_inject_agent_context(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Hook into prompt building to inject info about running agents.

        Adds a section to the system prompt so the LLM knows which
        agents are active and what they're working on.
        """
        if self._agent_manager is None:
            return data

        agents = self._agent_manager.list_agents()
        running = [a for a in agents if a["status"] == "running"]

        if not running:
            return data

        agent_lines = []
        for a in running:
            agent_lines.append(
                f"- {a['name']} (id: {a['agent_id']}): "
                f"{a['task']} [Iteration {a['iteration']}/"
                f"{a['max_iterations']}]"
            )

        context_block = (
            "\n## Laufende Sub-Agents\n"
            + "\n".join(agent_lines)
            + "\n\nNutze 'agent_result' um Ergebnisse abzurufen "
            "oder 'stop_agent' zum Stoppen.\n"
        )

        # Kontext an extra_context anhaengen
        extra = data.get("extra_context", "")
        data["extra_context"] = extra + context_block
        return data

    async def _hook_inject_skill_catalog(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """before_prompt_build — append the active-skill catalogue.

        Writes to ``system_prompt_parts`` (the list agent._plan consumes),
        not the legacy ``extra_context`` channel.
        """
        if self._registry is None or not self._inject_skill_catalog:
            return data
        try:
            entries = await self._registry.list_all(status="active")
        except Exception as exc:  # noqa: BLE001
            log.debug("skill_writer.catalog_list_failed", error=str(exc))
            return data
        block = build_catalog_block(
            entries, inline_max=self._catalog_inline_max
        )
        if block:
            parts = data.setdefault("system_prompt_parts", [])
            if isinstance(parts, list):
                parts.append(block)
        return data

    # ─── P4 — Autonomous skill learning ─────────────────────────────

    async def _on_response_for_skill_learning(
        self, data: dict[str, Any]
    ) -> None:
        """after_response_send (void) — learn a skill from a worthy pattern.

        Fully autonomous: a detected pattern is drafted by the author brain
        and registered live (no approval gate). The curator's low-success
        net (P3) prunes any that turn out weak, and everything is recoverable.
        """
        if not self._auto_learn_skills or self._detector is None:
            return
        if not isinstance(data, dict):
            return
        signal = self._detector.record(data.get("tools_used") or [])
        if signal is None:
            return
        try:
            await self._auto_create_skill(signal, data)
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_writer.auto_learn_failed", error=str(exc))

    async def _auto_create_skill(
        self, signal: TaskSignal, ctx: dict[str, Any]
    ) -> None:
        """Draft + register a skill for a detected pattern (live)."""
        if self._registry is None:
            return
        request = str(ctx.get("text") or "").strip()[:500] or "(unbekannt)"
        prompt = (
            _AUTO_SKILL_DRAFT_PROMPT.replace("%REASON%", signal.reason)
            .replace("%TOOLS%", " > ".join(signal.tools))
            .replace("%REQUEST%", request)
        )
        try:
            response = await self.api.llm_chat(
                messages=[{"role": "user", "content": prompt}],
                brain=self._skill_author_brain,
                max_tokens=800,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_writer.auto_draft_failed", error=str(exc))
            return

        draft = _parse_skill_draft(response)
        if draft is None:
            log.info("skill_writer.auto_draft_unparseable", reason=signal.reason)
            return
        name, description, code = draft

        # Dedup: don't recreate a skill that already exists by name.
        if await self._registry.get(sanitize_skill_name(name)) is not None:
            log.info("skill_writer.auto_skill_exists", name=name)
            return

        result = await self._write_and_register(
            name=name,
            description=description,
            code=code,
            source_tag="auto_pattern",
        )
        if "error" in result:
            log.info(
                "skill_writer.auto_create_rejected",
                name=name,
                error=result["error"],
            )
            return
        log.info(
            "skill_writer.auto_created",
            name=result.get("name"),
            reason=signal.reason,
        )
        # Transparency: surface what Lexy just taught herself.
        try:
            await self.api.ws_broadcast(
                {
                    "type": "skill_auto_learned",
                    "name": result.get("name"),
                    "reason": signal.reason,
                }
            )
        except Exception:  # noqa: BLE001
            pass

    # ─── P5 — Self-refinement ───────────────────────────────────────

    async def _maybe_refine_skill(self, name: str) -> None:
        """Self-repair an auto-skill once it has failed enough times."""
        if (
            not self._self_refine_skills
            or self._registry is None
            or self._exec_log is None
        ):
            return
        entry = await self._registry.get(name)
        if entry is None or entry.source not in DEFAULT_MANAGED_SOURCES:
            return  # never autonomously rewrite the user's own skills
        if not should_refine(
            entry.failure_count, threshold=self._refine_after_failures
        ):
            return
        now = time.time()
        if now - self._refine_cooldown.get(name, 0.0) < _REFINE_COOLDOWN_SECONDS:
            return  # anti-thrash: one refine attempt per cooldown window
        self._refine_cooldown[name] = now
        try:
            await self._refine_skill(entry)
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_writer.refine_failed", name=name, error=str(exc))

    async def _refine_skill(self, entry: Any) -> None:
        """Draft a patched version, validate it, then swap it in live.

        The old version is recoverably archived to ``.archive/`` *before* the
        new one is written, and the new code is validated first so a failed
        draft never destroys a working skill.
        """
        if self._registry is None or self._validator is None or self._exec_log is None:
            return
        name = entry.name
        failures = await self._exec_log.recent_failures(name, limit=5)
        failure_text = "\n".join(f"- {f['error']}" for f in failures) or "(keine)"
        old_code = await self._read_skill_code(entry.file_path)

        brain = self._refine_brain
        if len(old_code) > 1500:  # size-gated escalation to the deep brain
            brain = "a4b"
        prompt = (
            _REFINE_SKILL_PROMPT.replace("%NAME%", name)
            .replace("%DESCRIPTION%", entry.description or "")
            .replace("%CODE%", old_code[:2000])
            .replace("%FAILURES%", failure_text[:1000])
        )
        try:
            response = await self.api.llm_chat(
                messages=[{"role": "user", "content": prompt}],
                brain=brain,
                max_tokens=900,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("skill_writer.refine_draft_failed", name=name, error=str(exc))
            return

        draft = _parse_skill_draft(response)
        if draft is None:
            log.info("skill_writer.refine_unparseable", name=name)
            return
        _, new_description, new_code = draft

        # Validate BEFORE touching the live skill — keep the old one intact on
        # a bad draft.
        valid, err = self._validator.validate(new_code, expect_execute=False)
        if not valid:
            log.info("skill_writer.refine_invalid", name=name, error=err)
            return

        new_version = (entry.version or 1) + 1
        old_id = entry.id

        # Archive the old folder (recoverable), then re-register fresh.
        old_folder = Path(entry.file_path)
        archive_root = self._skills_path / ".archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        dest = archive_root / f"{name}-v{entry.version}-{int(time.time())}"
        if old_folder.is_dir():
            try:
                await asyncio.to_thread(shutil.move, str(old_folder), str(dest))
            except OSError as exc:
                log.warning(
                    "skill_writer.refine_archive_failed", name=name, error=str(exc)
                )
                return
        await self._registry.delete(name)

        result = await self._write_and_register(
            name=name,
            description=new_description or entry.description,
            code=new_code,
            source_tag="self_refine",
        )
        if "error" in result:
            log.warning(
                "skill_writer.refine_write_failed", name=name, error=result["error"]
            )
            return
        await self._registry.set_version(name, new_version, supersedes=old_id)
        log.info("skill_writer.refined", name=name, version=new_version)
        try:
            await self.api.ws_broadcast(
                {"type": "skill_refined", "name": name, "version": new_version}
            )
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    async def _read_skill_code(folder: str) -> str:
        """Read a skill's ``scripts/skill.py`` (best-effort, for refine context)."""
        path = Path(folder) / "scripts" / "skill.py"

        def _read() -> str:
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return ""

        return await asyncio.to_thread(_read)

    # ─── Event Handlers ─────────────────────────────────────────────

    async def _on_tool_error(self, event_data: Any) -> None:
        """
        React to tool errors — potentially auto-propose a fix skill.

        When auto_propose is enabled and the same tool error occurs
        multiple times, Lexy can suggest writing a skill to handle it.
        """
        if not self._auto_propose:
            return

        if not isinstance(event_data, dict):
            return

        tool_name = event_data.get("tool", "")
        error_msg = event_data.get("error", "")
        if not tool_name or not error_msg:
            return

        log.debug(
            "skill_writer.tool_error_observed",
            tool=tool_name,
            error=error_msg[:200],
        )

        # Speichern als Beobachtung fuer spaetere Skill-Vorschlaege
        await self.api.memory_store(
            text=(
                f"[SkillWriter] Tool-Fehler beobachtet: "
                f"Tool={tool_name}, Error={error_msg[:300]}"
            ),
            collection="facts",
            metadata={
                "source": "skill_writer",
                "event": "tool_error",
                "tool": tool_name,
            },
        )

    async def _on_agent_done(self, event_data: Any) -> None:
        """Handle agent completion — log results and notify."""
        if not isinstance(event_data, dict):
            return

        agent_name = event_data.get("name", "unknown")
        agent_id = event_data.get("agent_id", "")

        log.info(
            "skill_writer.agent_completed",
            agent_name=agent_name,
            agent_id=agent_id,
        )

    # ─── Proposal Management ────────────────────────────────────────

    async def _create_proposal(
        self,
        name: str,
        description: str,
        code: str,
        tags: str,
        source_tag: str,
        *,
        body_md: str = "",
        license: str | None = None,
        compatibility: str | None = None,
        metadata: dict[str, str] | None = None,
        allowed_tools: str | None = None,
    ) -> str:
        """Create a pending skill proposal in the DB.

        Phase 11: persists the full agentskills.io frontmatter so the
        approval flow can rebuild the folder verbatim — no info loss
        between proposal and the final skill.
        """
        proposal_id = uuid.uuid4().hex[:12]
        now = time.time()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        db = await self.api.get_db()
        await db.execute(
            """
            INSERT INTO skill_proposals
                (id, name, description, code, tags, source, status,
                 created_at, body_md, license, compatibility,
                 metadata_json, allowed_tools)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal_id, name, description, code, tags, source_tag, now,
                body_md, license, compatibility, metadata_json, allowed_tools,
            ),
        )
        await db.commit()
        return proposal_id

    async def _approve_proposal(
        self, proposal_id: str
    ) -> dict[str, Any]:
        """Approve a proposal: emit folder + register it.

        Phase 11: includes the persisted frontmatter so approval
        produces the exact same folder layout the writer originally
        proposed.
        """
        db = await self.api.get_db()
        cursor = await db.execute(
            """
            SELECT id, name, description, code, tags, source,
                   body_md, license, compatibility, metadata_json,
                   allowed_tools
            FROM skill_proposals
            WHERE id = ? AND status = 'pending'
            """,
            (proposal_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()

        if row is None:
            return {"error": f"Proposal '{proposal_id}' not found or not pending"}

        (
            _pid, name, description, code, _tags, source_tag,
            body_md, license_val, compatibility_val, metadata_json,
            _allowed_tools,
        ) = row

        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
            if not isinstance(metadata, dict):
                metadata = {}
        except (TypeError, ValueError):
            metadata = {}

        # Folder bauen + registrieren.
        result = await self._write_and_register(
            name=name,
            description=description,
            code=code,
            body_md=body_md or "",
            license=license_val,
            compatibility=compatibility_val,
            metadata={str(k): str(v) for k, v in metadata.items()},
            source_tag=source_tag,
        )

        if "error" not in result:
            # Proposal als approved markieren
            now = time.time()
            await db.execute(
                "UPDATE skill_proposals SET status = 'approved', "
                "reviewed_at = ? WHERE id = ?",
                (now, proposal_id),
            )
            await db.commit()

            log.info(
                "skill_writer.proposal_approved",
                proposal_id=proposal_id,
                name=name,
            )

            await self.api.ws_broadcast({
                "type": "skill_approved_notification",
                "proposal_id": proposal_id,
                "name": name,
            })

        return result

    async def _reject_proposal(
        self, proposal_id: str
    ) -> dict[str, Any]:
        """Reject a pending skill proposal."""
        db = await self.api.get_db()
        now = time.time()
        cursor = await db.execute(
            "UPDATE skill_proposals SET status = 'rejected', "
            "reviewed_at = ? WHERE id = ? AND status = 'pending'",
            (now, proposal_id),
        )
        await db.commit()

        if cursor.rowcount == 0:
            return {"error": f"Proposal '{proposal_id}' not found or not pending"}

        log.info("skill_writer.proposal_rejected", proposal_id=proposal_id)
        return {"status": "rejected", "proposal_id": proposal_id}

    async def _list_proposals(
        self, status: str = "pending"
    ) -> list[dict[str, Any]]:
        """List skill proposals by status."""
        db = await self.api.get_db()
        cursor = await db.execute(
            """
            SELECT id, name, description, tags, source, status, created_at
            FROM skill_proposals
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (status,),
        )
        rows = await cursor.fetchall()
        await cursor.close()

        return [
            {
                "proposal_id": r[0],
                "name": r[1],
                "description": r[2],
                "tags": r[3],
                "source": r[4],
                "status": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]
