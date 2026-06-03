"""
Lexy AI — Skill catalog injection (Phase P6).

Until now the agent only learned which skills exist by *calling* the
``list_skills`` tool — nothing surfaced the catalog into the system prompt, so
the model had to guess that skills were available at all. This module builds a
compact, progressive-disclosure catalogue block (skill **name + description**
only — never the full body) that a ``before_prompt_build`` hook appends to
``system_prompt_parts``.

Kept as a pure function so it can be unit-tested without the plugin, the DB, or
a running agent.
"""

from __future__ import annotations

from typing import Any, Iterable

_HEADER = (
    "## Verfügbare Skills\n"
    "Nutze `run_skill` mit dem Namen, wenn ein Skill zur Aufgabe passt:"
)


def build_catalog_block(
    entries: Iterable[Any],
    *,
    inline_max: int = 12,
    max_desc_len: int = 160,
) -> str | None:
    """Format an active-skill catalogue for the system prompt.

    ``entries`` are :class:`SkillEntry`-like objects (need ``name``,
    ``description``, ``status``, ``state``, ``last_used_at``, ``created_at``).
    Only ``status == "active"`` and non-archived skills are listed. When there
    are more than ``inline_max``, the most-recently-used ones win (progressive
    disclosure — the catalogue stays bounded as it grows).

    Returns ``None`` when there's nothing to show, so the caller can skip the
    injection entirely.
    """
    active = [
        entry
        for entry in entries
        if getattr(entry, "status", "active") == "active"
        and getattr(entry, "state", "active") != "archived"
    ]
    if not active:
        return None

    truncated = len(active) > inline_max
    if truncated:
        active.sort(
            key=lambda e: (e.last_used_at or e.created_at or 0.0),
            reverse=True,
        )
        active = active[:inline_max]

    lines = [_HEADER]
    for entry in active:
        desc = (entry.description or "").strip().replace("\n", " ")
        if len(desc) > max_desc_len:
            desc = desc[: max_desc_len - 1].rstrip() + "…"
        lines.append(f"- **{entry.name}**: {desc}")
    if truncated:
        lines.append(
            f"_(weitere Skills via `list_skills`; die {inline_max} zuletzt "
            "genutzten sind oben.)_"
        )
    return "\n".join(lines)
