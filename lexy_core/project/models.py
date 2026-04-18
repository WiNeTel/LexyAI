"""
Lexy AI - Project Pydantic model.

A Project is a top-level container the user creates from the sidebar.
Each project has its own scoped sessions, optional memory isolation,
optional persona-override, and visual identity (color + icon).

The reserved id ``"default"`` is the catch-all project ("Allgemein").
It always exists, cannot be deleted, and gets every orphan session
migrated into it on startup.
"""

from __future__ import annotations

import re
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator


_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class Project(BaseModel):
    """A user-defined workspace partition."""

    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    color: str = Field(default="#7aa2f7")
    icon: str = Field(default="", max_length=8)
    persona_override: str = Field(default="", max_length=4000)
    memory_scoped: bool = True
    is_default: bool = False
    archived: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("color")
    @classmethod
    def _validate_color(cls, value: str) -> str:
        """Accept only #RRGGBB hex codes; fall back to a sane default."""
        if not isinstance(value, str) or not _HEX_COLOR.match(value):
            return "#7aa2f7"
        return value.lower()

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("project name must not be empty after stripping")
        return cleaned

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot used for disk + websocket transport."""
        return self.model_dump(mode="json")
