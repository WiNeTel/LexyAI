"""
Lexy AI - ProjectStore (JSON-on-disk).

Thread-safe project registry with atomic persistence. Mirrors the design
of :class:`SessionStore`: every mutation flushes to disk via
``temp + os.replace``, missing/corrupt files become an empty store, and
the default project is auto-bootstrapped on first access.

On-disk format (v1)::

    {
        "version": 1,
        "saved_at": 1712851200.0,
        "projects": {
            "default": {
                "id": "default",
                "name": "Allgemein",
                "description": "",
                "color": "#7aa2f7",
                "icon": "🏠",
                "persona_override": "",
                "memory_scoped": false,
                "is_default": true,
                "archived": false,
                "created_at": 1712851200.0,
                "updated_at": 1712851200.0
            },
            "<uuid>": { ... }
        }
    }
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from lexy_core.project.models import Project

log = structlog.get_logger(__name__)

_STORE_VERSION = 1

DEFAULT_PROJECT_ID: str = "default"
DEFAULT_PROJECT_NAME: str = "Allgemein"
DEFAULT_PROJECT_DESCRIPTION: str = (
    "Standard-Projekt — alle Sessions ohne explizite Zuordnung landen hier."
)
DEFAULT_PROJECT_ICON: str = "🏠"


class ProjectStore:
    """
    Thread-safe project registry with optional JSON-on-disk persistence.

    The default project (``"default"``) is created automatically on the
    first call to :meth:`get_default` or whenever the store is loaded
    from disk and finds no ``default`` entry.
    """

    def __init__(
        self,
        persistent_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._projects: dict[str, Project] = {}
        self._lock = threading.RLock()
        self._path: Path | None = (
            Path(persistent_path) if persistent_path else None
        )
        if self._path is not None:
            self.load()

    # ─── Persistence ────────────────────────────────────────────────

    @property
    def path(self) -> Path | None:
        return self._path

    def load(self) -> int:
        """
        Load projects from disk. Returns the number of restored projects.
        Missing or unreadable files result in an empty store + warning.
        """
        if self._path is None:
            return 0
        with self._lock:
            self._projects = {}
            if not self._path.exists():
                # First start — make sure the default exists so callers
                # can always rely on get_default().
                self._ensure_default_locked()
                return 0
            try:
                raw = self._path.read_text(encoding="utf-8")
                if not raw.strip():
                    self._ensure_default_locked()
                    return 0
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning(
                    "project_store.load_failed",
                    path=str(self._path),
                    error=str(exc),
                )
                self._ensure_default_locked()
                return 0

            projects_raw = (
                data.get("projects") if isinstance(data, dict) else None
            )
            if not isinstance(projects_raw, dict):
                log.warning(
                    "project_store.load_invalid_shape",
                    path=str(self._path),
                )
                self._ensure_default_locked()
                return 0

            restored = 0
            for project_id, raw_value in projects_raw.items():
                if not isinstance(project_id, str) or not isinstance(
                    raw_value, dict
                ):
                    continue
                try:
                    project = Project(**raw_value)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "project_store.invalid_entry",
                        project_id=project_id,
                        error=str(exc),
                    )
                    continue
                # Force the on-disk id to win over an inconsistent payload
                if project.id != project_id:
                    project = project.model_copy(update={"id": project_id})
                self._projects[project_id] = project
                restored += 1

            self._ensure_default_locked()

            log.info(
                "project_store.loaded",
                path=str(self._path),
                projects=restored,
            )
            return restored

    def _ensure_default_locked(self) -> Project:
        """
        Make sure a default project exists. Caller MUST hold ``self._lock``.
        Returns the (possibly newly created) default project.
        """
        existing = self._projects.get(DEFAULT_PROJECT_ID)
        if existing is not None:
            # Force is_default flag in case the file was hand-edited
            if not existing.is_default:
                existing = existing.model_copy(update={"is_default": True})
                self._projects[DEFAULT_PROJECT_ID] = existing
                self._save_locked()
            return existing
        now = time.time()
        default = Project(
            id=DEFAULT_PROJECT_ID,
            name=DEFAULT_PROJECT_NAME,
            description=DEFAULT_PROJECT_DESCRIPTION,
            color="#7aa2f7",
            icon=DEFAULT_PROJECT_ICON,
            persona_override="",
            memory_scoped=False,  # default project sees everything
            is_default=True,
            archived=False,
            created_at=now,
            updated_at=now,
        )
        self._projects[DEFAULT_PROJECT_ID] = default
        self._save_locked()
        log.info("project_store.default_bootstrapped")
        return default

    def save(self) -> bool:
        """Flush the store to disk. ``False`` if no path or write failed."""
        if self._path is None:
            return False
        with self._lock:
            return self._save_locked()

    def _save_locked(self) -> bool:
        """Inner writer — caller MUST hold the lock."""
        if self._path is None:
            return False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "version": _STORE_VERSION,
                "saved_at": time.time(),
                "projects": {
                    pid: project.to_dict()
                    for pid, project in self._projects.items()
                },
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
            return True
        except OSError as exc:
            log.warning(
                "project_store.save_failed",
                path=str(self._path),
                error=str(exc),
            )
            return False

    def _persist(self) -> None:
        """Flush if a path is configured. Caller must hold the lock."""
        if self._path is not None:
            self._save_locked()

    # ─── Reads ──────────────────────────────────────────────────────

    def get(self, project_id: str) -> Project | None:
        """Return the project (or ``None`` if missing/archived-hidden case)."""
        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                return None
            return project.model_copy()

    def get_default(self) -> Project:
        """Return the default project, creating it on demand."""
        with self._lock:
            return self._ensure_default_locked().model_copy()

    def list(self, include_archived: bool = False) -> list[Project]:
        """
        Return all known projects sorted by ``created_at`` (default first
        when present).
        """
        with self._lock:
            items = [
                project.model_copy()
                for project in self._projects.values()
                if include_archived or not project.archived
            ]
        items.sort(
            key=lambda p: (
                0 if p.is_default else 1,
                p.created_at,
                p.id,
            )
        )
        return items

    def exists(self, project_id: str) -> bool:
        with self._lock:
            return project_id in self._projects

    # ─── Mutations ──────────────────────────────────────────────────

    def create(
        self,
        name: str,
        description: str = "",
        color: str = "#7aa2f7",
        icon: str = "",
        persona_override: str = "",
        memory_scoped: bool = True,
    ) -> Project:
        """Create a new project with a fresh UUID-derived id."""
        project_id = self._new_id()
        now = time.time()
        project = Project(
            id=project_id,
            name=name,
            description=description,
            color=color,
            icon=icon,
            persona_override=persona_override,
            memory_scoped=memory_scoped,
            is_default=False,
            archived=False,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._projects[project_id] = project
            self._persist()
        log.info(
            "project_store.created",
            project_id=project_id,
            name=name,
        )
        return project.model_copy()

    def update(
        self,
        project_id: str,
        **patch: Any,
    ) -> Project | None:
        """
        Apply a partial update. Only known fields are accepted; ``id``,
        ``is_default`` and ``created_at`` are protected.
        """
        if not patch:
            return self.get(project_id)
        protected = {"id", "is_default", "created_at"}
        clean = {
            key: value
            for key, value in patch.items()
            if key not in protected and value is not None
        }
        if not clean:
            return self.get(project_id)
        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                return None
            try:
                updated = project.model_copy(update=clean)
                # Re-validate via a model rebuild so validators run on the
                # mutated fields (color, name, ...).
                updated = Project(**updated.model_dump())
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "project_store.update_invalid",
                    project_id=project_id,
                    error=str(exc),
                )
                return None
            updated = updated.model_copy(update={"updated_at": time.time()})
            # Default project's flag is sticky.
            if project_id == DEFAULT_PROJECT_ID:
                updated = updated.model_copy(update={"is_default": True})
            self._projects[project_id] = updated
            self._persist()
        log.info(
            "project_store.updated",
            project_id=project_id,
            fields=list(clean.keys()),
        )
        return updated.model_copy()

    def delete(self, project_id: str) -> tuple[bool, Project | None]:
        """
        Remove a project. The default project cannot be deleted. Returns
        ``(deleted, removed_project_snapshot)``. The session migration
        is the caller's responsibility (ProjectStore does not own
        session state).
        """
        if project_id == DEFAULT_PROJECT_ID:
            log.warning(
                "project_store.delete_blocked_default",
                project_id=project_id,
            )
            return False, None
        with self._lock:
            removed = self._projects.pop(project_id, None)
            if removed is None:
                return False, None
            self._persist()
        log.info("project_store.deleted", project_id=project_id)
        return True, removed.model_copy()

    def archive(self, project_id: str) -> bool:
        """Mark a non-default project as archived (hidden from sidebar)."""
        if project_id == DEFAULT_PROJECT_ID:
            return False
        with self._lock:
            project = self._projects.get(project_id)
            if project is None or project.archived:
                return False
            self._projects[project_id] = project.model_copy(
                update={"archived": True, "updated_at": time.time()}
            )
            self._persist()
        return True

    def unarchive(self, project_id: str) -> bool:
        with self._lock:
            project = self._projects.get(project_id)
            if project is None or not project.archived:
                return False
            self._projects[project_id] = project.model_copy(
                update={"archived": False, "updated_at": time.time()}
            )
            self._persist()
        return True

    # ─── Helpers ────────────────────────────────────────────────────

    def _new_id(self) -> str:
        """Return a fresh project id (UUID hex, 12 chars)."""
        for _ in range(8):
            candidate = uuid.uuid4().hex[:12]
            if (
                candidate != DEFAULT_PROJECT_ID
                and candidate not in self._projects
            ):
                return candidate
        # Astronomically unlikely fallback — full uuid
        return uuid.uuid4().hex
