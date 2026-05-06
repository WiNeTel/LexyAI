"""
Lexy AI — agentskills.io skill folder loader (Phase 11).

Reads a folder layout that matches the official spec:

```
<skill-name>/
├── SKILL.md          # required (parsed by skill_spec)
├── scripts/          # optional executable code (Python only for now)
├── references/       # optional on-demand docs
├── assets/           # optional templates / data
└── ...               # any extra files
```

Returns a :class:`SkillCard` with parsed frontmatter + a manifest of
what's present in each subdir, so the registry and the runner can use
it without re-walking the disk.

The spec calls out **progressive disclosure**: at discovery time only
``frontmatter.name`` + ``frontmatter.description`` should ever leave
the loader (so the agent's startup context stays small). The body and
the file lists load eagerly here because we're already on disk and the
overhead is tiny — but ``SkillCard.discovery_dict()`` returns the
minimal view for the LLM-facing system-prompt injection.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexy_core.utils.logging import get_logger

from .skill_spec import (
    SkillFrontmatter,
    SkillFrontmatterError,
    parse_skill_md,
)

log = get_logger(module="skill_loader")

# Names per spec — case-sensitive. We document the recognised dirs
# in the spec table; everything else is just "extra files".
SCRIPTS_DIR = "scripts"
REFERENCES_DIR = "references"
ASSETS_DIR = "assets"

# Default executable inside scripts/. Spec doesn't mandate a name —
# we settle on ``skill.py`` because every existing Lexy skill template
# emits that, and it's an obvious convention. Skills that prefer
# something else can declare it via ``metadata.entry``; the loader
# resolves that if present.
PRIMARY_SCRIPT_DEFAULT = f"{SCRIPTS_DIR}/skill.py"


class SkillLoaderError(ValueError):
    """Raised when a folder isn't a valid skill folder."""


@dataclass
class SkillCard:
    """In-memory view of one on-disk skill folder.

    All paths inside this dataclass are absolute. The frontmatter has
    already been validated by :func:`parse_skill_md` at construction
    time, so consumers can trust it without re-checking.
    """

    folder: Path
    frontmatter: SkillFrontmatter

    has_scripts: bool = False
    has_references: bool = False
    has_assets: bool = False

    # Relative paths within ``scripts/`` (plus ``references/`` and
    # ``assets/`` for the listings). Useful for the UI showing the
    # bundle contents without forcing another disk walk.
    script_files: list[str] = field(default_factory=list)
    reference_files: list[str] = field(default_factory=list)
    asset_files: list[str] = field(default_factory=list)

    # Resolved primary entry point (relative to ``folder``). ``None``
    # for skills that ship docs/templates only.
    primary_script: str | None = None

    # ─── Convenience ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description

    def primary_script_path(self) -> Path | None:
        """Absolute path to the primary script, or ``None``."""
        if self.primary_script is None:
            return None
        return self.folder / self.primary_script

    def discovery_dict(self) -> dict[str, str]:
        """Minimal {name, description} view used at agent boot.

        Per spec this is what an agent should see during the
        *discovery* stage — full SKILL.md only loads on activation.
        """
        return {
            "name": self.frontmatter.name,
            "description": self.frontmatter.description,
        }

    def to_public(self) -> dict[str, Any]:
        """JSON-friendly dict for REST/WS responses."""
        out = self.frontmatter.to_public()
        out.update(
            {
                "folder": str(self.folder),
                "has_scripts": self.has_scripts,
                "has_references": self.has_references,
                "has_assets": self.has_assets,
                "script_files": list(self.script_files),
                "reference_files": list(self.reference_files),
                "asset_files": list(self.asset_files),
                "primary_script": self.primary_script,
                # Body is markdown — emit it separately so the UI can
                # render it lazily (progressive disclosure).
                "body": self.frontmatter.body,
            }
        )
        return out


# ─── Loaders ────────────────────────────────────────────────────────

async def load_skill_folder(path: Path) -> SkillCard:
    """Read and validate one skill folder from disk.

    Args:
        path: Path to the skill folder (must contain ``SKILL.md``).

    Returns:
        A fully-populated :class:`SkillCard`.

    Raises:
        SkillLoaderError: missing SKILL.md, parent name mismatch,
            unreadable folder, etc.
        SkillSpecError (or subclasses): frontmatter violates the spec.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise SkillLoaderError(f"not a directory: {folder}")

    skill_md_path = folder / "SKILL.md"
    if not skill_md_path.is_file():
        raise SkillLoaderError(f"SKILL.md missing in {folder}")

    # Run blocking I/O in a thread so async callers don't stall.
    text = await asyncio.to_thread(
        skill_md_path.read_text, encoding="utf-8"
    )

    # Spec rule: frontmatter.name must equal the parent dir name.
    # We pass the folder's name explicitly so parse_skill_md can
    # surface a clear error rather than silently accepting a mismatch.
    try:
        frontmatter = parse_skill_md(text, parent_dir_name=folder.name)
    except SkillFrontmatterError:
        raise
    # parse_skill_md raises subclasses of SkillSpecError; let them
    # propagate so callers can branch on the specific cause.

    # Walk the optional subdirs. ``iterdir`` ordering is OS-dependent;
    # we sort so listings are deterministic across machines.
    scripts_dir = folder / SCRIPTS_DIR
    references_dir = folder / REFERENCES_DIR
    assets_dir = folder / ASSETS_DIR

    script_files = await _list_files_relative(scripts_dir, prefix=SCRIPTS_DIR)
    reference_files = await _list_files_relative(
        references_dir, prefix=REFERENCES_DIR
    )
    asset_files = await _list_files_relative(assets_dir, prefix=ASSETS_DIR)

    primary_script = _resolve_primary_script(
        frontmatter=frontmatter,
        script_files=script_files,
    )

    card = SkillCard(
        folder=folder.resolve(),
        frontmatter=frontmatter,
        has_scripts=bool(script_files),
        has_references=bool(reference_files),
        has_assets=bool(asset_files),
        script_files=script_files,
        reference_files=reference_files,
        asset_files=asset_files,
        primary_script=primary_script,
    )
    log.debug(
        "skill_loader.loaded",
        name=card.name,
        folder=str(card.folder),
        scripts=len(script_files),
        references=len(reference_files),
        assets=len(asset_files),
        primary=primary_script,
    )
    return card


async def discover_skills(
    skills_root: Path, *, log_errors: bool = True
) -> list[SkillCard]:
    """Walk ``skills_root`` and return every valid skill found.

    Folders that fail validation are skipped with a warning log entry —
    the registry's ``scan_disk`` should surface them to the UI but the
    boot path keeps going so one broken skill can't take Lexy down.
    """
    root = Path(skills_root)
    if not root.is_dir():
        return []

    results: list[SkillCard] = []
    # Skip dotfiles + ``__pycache__``-style artefacts.
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        try:
            card = await load_skill_folder(entry)
        except (SkillLoaderError, ValueError) as exc:
            if log_errors:
                log.warning(
                    "skill_loader.skip_invalid",
                    folder=str(entry),
                    error=str(exc),
                )
            continue
        results.append(card)
    return results


# ─── Internals ──────────────────────────────────────────────────────

async def _list_files_relative(
    dir_path: Path, *, prefix: str
) -> list[str]:
    """List files inside ``dir_path`` recursively, returned as
    ``"prefix/relpath"`` so the SkillCard can store paths that work
    when joined with ``card.folder``.

    Returns ``[]`` if the directory doesn't exist (the optional dirs
    are, well, optional).
    """
    if not dir_path.is_dir():
        return []

    def _walk() -> list[str]:
        out: list[str] = []
        for p in sorted(dir_path.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name == "__pycache__":
                continue
            # Hidden parent? Skip.
            if any(part.startswith(".") for part in p.relative_to(dir_path).parts):
                continue
            rel = p.relative_to(dir_path).as_posix()
            out.append(f"{prefix}/{rel}")
        return out

    return await asyncio.to_thread(_walk)


def _resolve_primary_script(
    *,
    frontmatter: SkillFrontmatter,
    script_files: list[str],
) -> str | None:
    """Pick the entry-point script for ``run_skill``.

    Resolution order:
    1. ``frontmatter.metadata['entry']`` if set and the path exists in
       ``script_files`` (or matches one of the listed files).
    2. ``scripts/skill.py`` if present (Lexy's default convention).
    3. ``None`` — skill is docs/templates-only and can't be executed.
    """
    explicit = frontmatter.metadata.get("entry") if frontmatter.metadata else None
    if explicit:
        # Normalise: allow ``"scripts/foo.py"`` or just ``"foo.py"``.
        candidate = explicit if explicit.startswith(f"{SCRIPTS_DIR}/") else f"{SCRIPTS_DIR}/{explicit}"
        if candidate in script_files:
            return candidate
        # Author specified an entry but it doesn't exist on disk —
        # surface that so the registry can flag the skill as broken
        # rather than silently falling back to a different file.
        raise SkillLoaderError(
            f"metadata.entry={explicit!r} does not exist in scripts/"
        )
    if PRIMARY_SCRIPT_DEFAULT in script_files:
        return PRIMARY_SCRIPT_DEFAULT
    return None
