"""
Lexy AI - Project subsystem.

Projects are top-level containers for sessions + memory. Each project has
its own persona-override, color, icon, and optional memory-scoping. The
default project ``"default"`` (display name "Allgemein") is auto-created
on first start and acts as the catch-all for orphan sessions.

Public API:
    * :class:`Project` — Pydantic model (see ``models.py``)
    * :class:`ProjectStore` — JSON-on-disk persistence (see ``store.py``)
"""

from __future__ import annotations

from lexy_core.project.models import Project
from lexy_core.project.store import (
    DEFAULT_PROJECT_DESCRIPTION,
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    ProjectStore,
)

__all__ = [
    "Project",
    "ProjectStore",
    "DEFAULT_PROJECT_ID",
    "DEFAULT_PROJECT_NAME",
    "DEFAULT_PROJECT_DESCRIPTION",
]
