"""
Phase 11 — Tests for the agentskills.io ZIP import/export pipeline.

These tests exercise:

* import-zip happy path (extraction → validation → install)
* path-traversal attacks (``../etc/passwd``) → blocked
* ZIP-bomb heuristic (uncompressed size cap)
* non-ZIP / corrupted bytes → SkillPackageError
* multiple top-level folders → reject (spec mandates exactly one)
* spec violation in SKILL.md → reject
* AST violation in scripts/ → reject
* overwrite=False on existing folder → SkillImportConflict
* overwrite=True replaces cleanly
* export-zip wraps the folder + skips __pycache__/dotfiles
* round-trip: ``import(export(folder))`` → identical SkillCard fields
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from plugins.skill_writer.skill_packager import (
    DEFAULT_MAX_UNCOMPRESSED,
    SkillImportConflict,
    SkillPackageError,
    export_skill_zip,
    import_skill_zip,
)
from plugins.skill_writer.skill_template import emit_skill_folder
from plugins.skill_writer.skill_validator import SkillValidator


# ─── Fixtures ────────────────────────────────────────────────────────

ALLOWED = [
    "json", "re", "datetime", "math", "collections", "pathlib",
    "__future__", "typing",
]


@pytest.fixture
def validator() -> SkillValidator:
    return SkillValidator(allowed_imports=ALLOWED)


@pytest.fixture
def sample_skill_folder(tmp_path: Path) -> Path:
    """Build one valid skill folder we can pack/unpack."""
    return emit_skill_folder(
        name="sample-skill",
        description="Sample skill for ZIP tests.",
        code='return {"ok": True}',
        target_root=tmp_path,
        license="MIT",
        metadata={"version": "1.0", "author": "test"},
    )


def _zip_bytes_from_folder(folder: Path) -> bytes:
    """Plain `zipfile`-based packer. We avoid using `export_skill_zip`
    here so import tests stay independent of the exporter."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            arcname = (Path(folder.name) / path.relative_to(folder)).as_posix()
            zf.write(path, arcname=arcname)
    return buf.getvalue()


# ─── Importer happy path ─────────────────────────────────────────────


class TestImportSkillZipHappyPath:
    @pytest.mark.asyncio
    async def test_imports_valid_skill(
        self,
        sample_skill_folder: Path,
        validator: SkillValidator,
        tmp_path: Path,
    ) -> None:
        zip_bytes = _zip_bytes_from_folder(sample_skill_folder)
        dest = tmp_path / "dest"

        result = await import_skill_zip(
            zip_bytes, dest_root=dest, validator=validator,
        )
        assert result.card.name == "sample-skill"
        assert result.overwrote_existing is False
        # Folder exists at dest/<name>/ with SKILL.md + scripts/skill.py
        installed = dest / "sample-skill"
        assert (installed / "SKILL.md").is_file()
        assert (installed / "scripts" / "skill.py").is_file()

    @pytest.mark.asyncio
    async def test_import_preserves_metadata(
        self,
        sample_skill_folder: Path,
        validator: SkillValidator,
        tmp_path: Path,
    ) -> None:
        zip_bytes = _zip_bytes_from_folder(sample_skill_folder)
        result = await import_skill_zip(
            zip_bytes, dest_root=tmp_path / "dest", validator=validator,
        )
        assert result.card.frontmatter.license == "MIT"
        assert result.card.frontmatter.metadata["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_overwrite_replaces_existing(
        self,
        sample_skill_folder: Path,
        validator: SkillValidator,
        tmp_path: Path,
    ) -> None:
        zip_bytes = _zip_bytes_from_folder(sample_skill_folder)
        dest = tmp_path / "dest"
        await import_skill_zip(
            zip_bytes, dest_root=dest, validator=validator,
        )
        # Re-import without overwrite → conflict
        with pytest.raises(SkillImportConflict):
            await import_skill_zip(
                zip_bytes, dest_root=dest, validator=validator,
            )
        # With overwrite=True → ok
        result = await import_skill_zip(
            zip_bytes, dest_root=dest, validator=validator, overwrite=True,
        )
        assert result.overwrote_existing is True


# ─── Importer error paths ────────────────────────────────────────────


class TestImportSkillZipErrors:
    @pytest.mark.asyncio
    async def test_corrupted_zip(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        with pytest.raises(SkillPackageError):
            await import_skill_zip(
                b"not a zip", dest_root=tmp_path, validator=validator,
            )

    @pytest.mark.asyncio
    async def test_empty_zip(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as _zf:
            pass  # no entries
        with pytest.raises(SkillPackageError):
            await import_skill_zip(
                buf.getvalue(), dest_root=tmp_path, validator=validator,
            )

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        """An entry escaping the skill folder via ``..`` must be rejected."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("evil-skill/../../etc/passwd", "root:x:0:0\n")
            zf.writestr(
                "evil-skill/SKILL.md",
                "---\nname: evil-skill\ndescription: D.\n---\n",
            )
        with pytest.raises(SkillPackageError, match="unsafe"):
            await import_skill_zip(
                buf.getvalue(), dest_root=tmp_path, validator=validator,
            )

    @pytest.mark.asyncio
    async def test_absolute_path_blocked(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("/abs/SKILL.md", "x")
        with pytest.raises(SkillPackageError, match="unsafe"):
            await import_skill_zip(
                buf.getvalue(), dest_root=tmp_path, validator=validator,
            )

    @pytest.mark.asyncio
    async def test_uncompressed_size_cap(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        """Reject ZIPs whose unpacked size exceeds the limit."""
        buf = io.BytesIO()
        big = b"a" * (1 * 1024 * 1024)  # 1 MiB compressible payload
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(5):  # 5 MiB of payload
                zf.writestr(f"big-skill/scripts/data_{i}.py", big)
            zf.writestr(
                "big-skill/SKILL.md",
                "---\nname: big-skill\ndescription: D.\n---\n",
            )
        with pytest.raises(SkillPackageError, match="uncompressed"):
            await import_skill_zip(
                buf.getvalue(),
                dest_root=tmp_path,
                validator=validator,
                max_uncompressed_bytes=2 * 1024 * 1024,  # 2 MiB cap
            )

    @pytest.mark.asyncio
    async def test_zip_too_large(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        """Reject ZIP bytes that are too large up front."""
        buf = b"x" * 1024
        with pytest.raises(SkillPackageError, match="too large"):
            await import_skill_zip(
                buf,
                dest_root=tmp_path,
                validator=validator,
                max_zip_bytes=512,
            )

    @pytest.mark.asyncio
    async def test_multiple_top_level_folders(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "first-skill/SKILL.md",
                "---\nname: first-skill\ndescription: D.\n---\n",
            )
            zf.writestr(
                "second-skill/SKILL.md",
                "---\nname: second-skill\ndescription: D.\n---\n",
            )
        with pytest.raises(SkillPackageError, match="exactly one"):
            await import_skill_zip(
                buf.getvalue(), dest_root=tmp_path, validator=validator,
            )

    @pytest.mark.asyncio
    async def test_invalid_frontmatter_rejected(
        self, validator: SkillValidator, tmp_path: Path,
    ) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "bad-skill/SKILL.md",
                "---\nname: BadSkill\ndescription: Demo.\n---\n",
            )
        with pytest.raises(SkillPackageError, match="validation"):
            await import_skill_zip(
                buf.getvalue(), dest_root=tmp_path, validator=validator,
            )

    @pytest.mark.asyncio
    async def test_ast_violation_in_scripts_rejected(
        self,
        sample_skill_folder: Path,
        validator: SkillValidator,
        tmp_path: Path,
    ) -> None:
        """Drop a forbidden import into scripts/ before zipping."""
        bad = sample_skill_folder / "scripts" / "skill.py"
        bad.write_text(
            "from typing import Any\n"
            "import subprocess\n\n"
            "async def execute(api: Any, **kwargs: Any) -> dict:\n"
            "    return {}\n",
            encoding="utf-8",
        )
        zip_bytes = _zip_bytes_from_folder(sample_skill_folder)
        with pytest.raises(SkillPackageError, match="validation"):
            await import_skill_zip(
                zip_bytes,
                dest_root=tmp_path / "dest",
                validator=validator,
            )


# ─── Exporter ────────────────────────────────────────────────────────


class TestExportSkillZip:
    @pytest.mark.asyncio
    async def test_export_creates_well_formed_zip(
        self, sample_skill_folder: Path,
    ) -> None:
        zip_bytes = await export_skill_zip(sample_skill_folder)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
        # Every entry sits under "sample-skill/" — that's the spec.
        assert all(n.startswith("sample-skill/") for n in names)
        # SKILL.md and scripts/skill.py are in there.
        assert "sample-skill/SKILL.md" in names
        assert "sample-skill/scripts/skill.py" in names

    @pytest.mark.asyncio
    async def test_export_excludes_pycache(
        self, sample_skill_folder: Path,
    ) -> None:
        # Drop a __pycache__ in scripts/ to simulate post-run state.
        cache = sample_skill_folder / "scripts" / "__pycache__"
        cache.mkdir()
        (cache / "skill.cpython-311.pyc").write_bytes(b"\x00\x01")
        # And a hidden file at the top.
        (sample_skill_folder / ".DS_Store").write_text("noise")

        zip_bytes = await export_skill_zip(sample_skill_folder)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
        assert not any("__pycache__" in n for n in names)
        assert not any(".DS_Store" in n for n in names)

    @pytest.mark.asyncio
    async def test_export_rejects_non_skill_folder(
        self, tmp_path: Path,
    ) -> None:
        not_a_skill = tmp_path / "random"
        not_a_skill.mkdir()
        with pytest.raises(SkillPackageError):
            await export_skill_zip(not_a_skill)

    @pytest.mark.asyncio
    async def test_export_rejects_missing_folder(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(SkillPackageError):
            await export_skill_zip(tmp_path / "missing")


# ─── Round-trip ─────────────────────────────────────────────────────


class TestRoundtrip:
    @pytest.mark.asyncio
    async def test_export_then_import_preserves_card(
        self,
        sample_skill_folder: Path,
        validator: SkillValidator,
        tmp_path: Path,
    ) -> None:
        zip_bytes = await export_skill_zip(sample_skill_folder)
        dest = tmp_path / "roundtrip"
        result = await import_skill_zip(
            zip_bytes, dest_root=dest, validator=validator,
        )
        # Frontmatter survives byte-for-byte
        original_text = (sample_skill_folder / "SKILL.md").read_text(
            encoding="utf-8"
        )
        roundtrip_text = (result.card.folder / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert original_text == roundtrip_text
        # And the script body too
        assert (
            (sample_skill_folder / "scripts" / "skill.py").read_text(encoding="utf-8")
            == (result.card.folder / "scripts" / "skill.py").read_text(encoding="utf-8")
        )

    @pytest.mark.asyncio
    async def test_round_trip_keeps_references_and_assets(
        self,
        sample_skill_folder: Path,
        validator: SkillValidator,
        tmp_path: Path,
    ) -> None:
        # Add references/ and assets/ to the source.
        refs = sample_skill_folder / "references"
        refs.mkdir()
        (refs / "REFERENCE.md").write_text("ref body", encoding="utf-8")
        assets = sample_skill_folder / "assets"
        assets.mkdir()
        (assets / "template.txt").write_text("tpl", encoding="utf-8")

        zip_bytes = await export_skill_zip(sample_skill_folder)
        dest = tmp_path / "roundtrip"
        result = await import_skill_zip(
            zip_bytes, dest_root=dest, validator=validator,
        )
        assert result.card.has_references is True
        assert result.card.has_assets is True
        assert "references/REFERENCE.md" in result.card.reference_files
        assert "assets/template.txt" in result.card.asset_files
