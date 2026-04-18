"""
Lexy AI - Persona Registry.

Manages built-in and user-defined agent personas. Built-in personas come
from the plugin config (``plugin.yaml``), user-defined ones are persisted
in SQLite so they survive restarts.

Each persona carries a system prompt, preferred brain, and temperature,
which the ``OrchestratorPool`` uses when spawning an agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

log = get_logger(module="persona_registry")


@dataclass
class Persona:
    """Single persona definition."""

    id: str
    name: str
    system_prompt: str
    brain: str = "e4b"
    temperature: float = 0.6
    builtin: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API / WS responses (truncates long prompts)."""
        prompt_preview = self.system_prompt
        if len(prompt_preview) > 200:
            prompt_preview = prompt_preview[:200] + "..."
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": prompt_preview,
            "brain": self.brain,
            "temperature": self.temperature,
            "builtin": self.builtin,
        }


class PersonaRegistry:
    """
    Registry for agent personas (built-in + user-defined).

    Built-in personas are loaded from the plugin config on init.
    User-defined personas are persisted to SQLite.
    """

    def __init__(self) -> None:
        self._personas: dict[str, Persona] = {}
        self._db: aiosqlite.Connection | None = None

    async def init_db(
        self,
        db: aiosqlite.Connection,
        config_personas: dict[str, Any],
    ) -> None:
        """
        Create the personas table and load built-in + user-defined personas.

        Args:
            db:              The shared aiosqlite connection from ``api.get_db()``.
            config_personas: Persona definitions from ``plugin.yaml`` config_defaults.
        """
        self._db = db

        await db.execute(
            """CREATE TABLE IF NOT EXISTS personas (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                prompt      TEXT NOT NULL,
                brain       TEXT NOT NULL DEFAULT 'e4b',
                temperature REAL NOT NULL DEFAULT 0.6,
                builtin     INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            )"""
        )
        await db.commit()

        # Lade built-in Personas aus der Config
        for pid, cfg in config_personas.items():
            self._personas[pid] = Persona(
                id=pid,
                name=cfg.get("name", pid),
                system_prompt=cfg.get("prompt", ""),
                brain=cfg.get("brain", "e4b"),
                temperature=cfg.get("temperature", 0.6),
                builtin=True,
            )

        # Lade user-defined Personas aus der DB
        async with db.execute(
            "SELECT id, name, prompt, brain, temperature, created_at "
            "FROM personas WHERE builtin = 0"
        ) as cur:
            async for row in cur:
                self._personas[row[0]] = Persona(
                    id=row[0],
                    name=row[1],
                    system_prompt=row[2],
                    brain=row[3],
                    temperature=row[4],
                    builtin=False,
                    created_at=row[5],
                )

        builtin_count = sum(1 for p in self._personas.values() if p.builtin)
        user_count = sum(1 for p in self._personas.values() if not p.builtin)
        log.info(
            "persona_registry.loaded",
            builtin=builtin_count,
            user_defined=user_count,
        )

    def get(self, persona_id: str) -> Persona | None:
        """Look up a persona by ID. Returns None if not found."""
        return self._personas.get(persona_id)

    def list_all(self) -> list[dict[str, Any]]:
        """Return all personas as dicts, sorted built-in first then by name."""
        return [
            p.to_dict()
            for p in sorted(
                self._personas.values(),
                key=lambda p: (not p.builtin, p.name),
            )
        ]

    def list_ids(self) -> list[str]:
        """Return just the persona IDs."""
        return list(self._personas.keys())

    async def register(
        self,
        id: str,
        name: str,
        prompt: str,
        brain: str = "e4b",
        temperature: float = 0.6,
    ) -> Persona:
        """
        Register a new user-defined persona and persist to SQLite.

        If a persona with the same ID already exists, it is overwritten.
        Built-in personas cannot be overwritten via this method.
        """
        persona = Persona(
            id=id,
            name=name,
            system_prompt=prompt,
            brain=brain,
            temperature=temperature,
            builtin=False,
        )
        self._personas[id] = persona

        if self._db is not None:
            await self._db.execute(
                "INSERT OR REPLACE INTO personas "
                "(id, name, prompt, brain, temperature, builtin, created_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?)",
                (id, name, prompt, brain, temperature, persona.created_at),
            )
            await self._db.commit()

        log.info("persona_registry.registered", id=id, name=name, brain=brain)
        return persona

    async def delete(self, id: str) -> bool:
        """
        Delete a user-defined persona. Built-in personas cannot be deleted.

        Returns:
            True if deleted, False if not found or built-in.
        """
        persona = self._personas.get(id)
        if persona is None or persona.builtin:
            return False

        del self._personas[id]

        if self._db is not None:
            await self._db.execute(
                "DELETE FROM personas WHERE id = ? AND builtin = 0", (id,)
            )
            await self._db.commit()

        log.info("persona_registry.deleted", id=id)
        return True
