"""
Lexy AI — Skill ZIP import/export (Phase 11.B + 11.C).

agentskills.io's mandate is "a skill is a folder", and the ecosystem
has settled on ZIP files for distribution (Cursor, Claude Code, the
agentskills/agentskills repo all pass them around as ``.zip``). This
module is the ZIP I/O for that:

* :func:`import_skill_zip` extracts a ZIP into ``data/skills/<name>/``
  after spec-validation + AST-validation + path-traversal guard.
* :func:`export_skill_zip` serialises an existing skill folder back
  to bytes for download / distribution.

Round-trip: ``import_skill_zip(export_skill_zip(folder))`` lands an
identical folder (modulo timestamps). Tests pin that property because
the wire format is what other tools see.

Security:
* Path-traversal blocked — any entry containing ``..`` or absolute
  paths refuses to extract.
* ZIP-bomb heuristic — total uncompressed size capped at a
  configurable limit (default 8 MiB; well above any real skill,
  well below "this is malicious").
* Spec-validation runs BEFORE we touch the destination folder, so
  a bad import never overwrites disk.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lexy_core.utils.logging import get_logger

from .skill_loader import (
    SkillCard,
    SkillLoaderError,
    load_skill_folder,
)
from .skill_spec import SkillSpecError
from .skill_validator import SkillValidator

log = get_logger(module="skill_packager")


# ─── Errors ─────────────────────────────────────────────────────────

class SkillPackageError(ValueError):
    """Base class for any ZIP import/export failure."""


class SkillImportConflict(SkillPackageError):
    """Raised when a skill with the target name already exists."""


# ─── Limits (tuneable per-call) ─────────────────────────────────────

DEFAULT_MAX_ZIP_BYTES = 8 * 1024 * 1024       # 8 MiB on the wire
DEFAULT_MAX_UNCOMPRESSED = 16 * 1024 * 1024   # 16 MiB after extraction


# ─── Helpers ────────────────────────────────────────────────────────

def _is_safe_zip_name(name: str) -> bool:
    """True iff ``name`` from a ZIP entry is safe to extract.

    Blocks ``..``-traversal, absolute paths (``/foo`` or ``C:\\foo``),
    and weirdness like backslash mixed paths. We also reject names
    containing a NUL byte (some old extractors choke).
    """
    if not name:
        return False
    if "\x00" in name:
        return False
    if name.startswith(("/", "\\")):
        return False
    # Backslashes are valid in zip names per spec but we want POSIX-only.
    norm = name.replace("\\", "/")
    parts = norm.split("/")
    if any(part == ".." for part in parts):
        return False
    if any(p.startswith(":") or (len(p) >= 2 and p[1] == ":") for p in parts):
        # Windows-style absolute (C:foo) detected.
        return False
    return True


def _find_top_level_folder(names: list[str]) -> str | None:
    """Pick the single top-level folder common to every entry.

    Returns ``None`` if entries don't share one root (e.g. the ZIP
    contains files at the top level without a wrapper folder), which
    we reject — the spec mandates the skill folder itself BE the
    top-level entry.
    """
    if not names:
        return None
    tops: set[str] = set()
    for raw in names:
        norm = raw.replace("\\", "/").lstrip("/")
        if not norm:
            continue
        first = norm.split("/", 1)[0]
        if first:
            tops.add(first)
    if len(tops) != 1:
        return None
    return next(iter(tops))


# ─── Importer ───────────────────────────────────────────────────────

@dataclass
class ImportResult:
    card: SkillCard
    overwrote_existing: bool = False


async def import_skill_zip(
    data: bytes,
    *,
    dest_root: Path,
    validator: SkillValidator,
    overwrite: bool = False,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED,
) -> ImportResult:
    """Extract + validate a skill ZIP, install into ``dest_root``.

    Pipeline:
    1. Size cap on the raw ZIP bytes.
    2. ZIP open + path-traversal guard.
    3. ZIP-bomb heuristic on total uncompressed size.
    4. Extract into a temp dir.
    5. Locate the (single) top-level folder, parse its SKILL.md.
    6. AST-validate every Python file in ``scripts/``.
    7. Move into place at ``dest_root/<name>/`` (atomic-ish — temp
       dir is on the same filesystem as dest_root when possible).

    Raises:
        SkillPackageError: malformed ZIP, traversal, oversized, etc.
        SkillImportConflict: target exists and ``overwrite=False``.
        SkillSpecError: SKILL.md frontmatter invalid.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise SkillPackageError("ZIP data must be bytes")
    if len(data) > max_zip_bytes:
        raise SkillPackageError(
            f"ZIP too large: {len(data)} > {max_zip_bytes} bytes"
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(data), "r")
    except zipfile.BadZipFile as exc:
        raise SkillPackageError(f"not a valid ZIP file: {exc}") from exc

    with zf:
        names = zf.namelist()
        if not names:
            raise SkillPackageError("ZIP is empty")

        # Path-traversal + uncompressed-size guards.
        total_uncompressed = 0
        for info in zf.infolist():
            if not _is_safe_zip_name(info.filename):
                raise SkillPackageError(
                    f"unsafe ZIP entry: {info.filename!r}"
                )
            total_uncompressed += info.file_size
            if total_uncompressed > max_uncompressed_bytes:
                raise SkillPackageError(
                    f"ZIP exceeds uncompressed-size limit "
                    f"{max_uncompressed_bytes}"
                )

        top = _find_top_level_folder(names)
        if top is None:
            raise SkillPackageError(
                "ZIP must have exactly one top-level folder containing SKILL.md"
            )

        # Stage everything into a temp dir before promoting it. The
        # tempdir lives next to dest_root so the final move is a
        # rename (cheap + atomic on most filesystems).
        dest_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="skill_import_", dir=str(dest_root)
        ) as staging:
            staging_path = Path(staging)
            await asyncio.to_thread(zf.extractall, staging_path)
            extracted = staging_path / top
            if not extracted.is_dir():
                raise SkillPackageError(
                    "ZIP top-level entry is not a directory"
                )

            # Validate spec + scripts BEFORE we touch the dest_root.
            try:
                card = await load_skill_folder(extracted)
            except (SkillLoaderError, SkillSpecError) as exc:
                raise SkillPackageError(f"validation failed: {exc}") from exc

            ok, err = validator.validate_folder(extracted)
            if not ok:
                raise SkillPackageError(f"script validation failed: {err}")

            # The folder name on disk MUST match the frontmatter name
            # (the loader already enforces this). The dest folder name
            # is therefore card.name — not the random staging name.
            target = dest_root / card.name
            overwrote = target.exists()
            if overwrote:
                if not overwrite:
                    raise SkillImportConflict(
                        f"skill '{card.name}' already exists at {target}"
                    )
                # Wipe the existing folder. Use shutil — we're plugin
                # code, not skill code, so the import-restriction
                # doesn't apply.
                await asyncio.to_thread(shutil.rmtree, target)

            # If the staging dir has the right name we can rename it.
            # Otherwise: copy the validated content next to dest_root
            # under the proper name.
            if extracted.name == card.name:
                await asyncio.to_thread(shutil.move, str(extracted), str(target))
            else:
                await asyncio.to_thread(
                    shutil.copytree, str(extracted), str(target)
                )

            # Re-load from the final location so SkillCard.folder is
            # absolute + canonical for the registry.
            installed = await load_skill_folder(target)
            log.info(
                "skill_packager.imported",
                name=installed.name,
                folder=str(installed.folder),
                overwrote=overwrote,
            )
            return ImportResult(card=installed, overwrote_existing=overwrote)


# ─── Exporter ───────────────────────────────────────────────────────

# File / dir names we strip when packing — they're either
# Python-runtime artefacts or OS-noise that shouldn't ship with the
# skill.
_EXPORT_SKIP_DIRS = {"__pycache__"}
_EXPORT_SKIP_PREFIXES = (".",)
_EXPORT_SKIP_EXACT = {".DS_Store", "Thumbs.db", "desktop.ini"}


async def export_skill_zip(
    folder: Path,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    """Pack a skill folder into ZIP bytes ready for download.

    The folder itself is the top-level entry inside the ZIP, so a
    consumer just needs to ``unzip`` and they get the canonical
    ``<name>/`` layout. Hidden files, ``__pycache__``, and OS detritus
    are excluded.

    Returns the ZIP bytes (in-memory). For very large skills, callers
    can stream to disk instead — but in practice spec-compliant skills
    are well under a few MB.

    Raises :class:`SkillPackageError` if the folder doesn't look like
    a skill.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise SkillPackageError(f"not a directory: {folder}")
    if not (folder / "SKILL.md").is_file():
        raise SkillPackageError(f"SKILL.md missing in {folder}")

    def _build() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=compression) as zf:
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                # Skip noise + hidden files.
                rel = path.relative_to(folder)
                parts = rel.parts
                if any(p in _EXPORT_SKIP_DIRS for p in parts):
                    continue
                if any(
                    p.startswith(_EXPORT_SKIP_PREFIXES) for p in parts
                ):
                    continue
                if rel.name in _EXPORT_SKIP_EXACT:
                    continue
                # Top-level entry is the skill folder name itself.
                arcname = (Path(folder.name) / rel).as_posix()
                zf.write(path, arcname=arcname)
        return buf.getvalue()

    return await asyncio.to_thread(_build)
