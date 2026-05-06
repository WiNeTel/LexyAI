"""
Phase 11 — Tests for the agentskills.io skill folder loader.

Validates the on-disk → SkillCard pipeline. Uses real tempdirs (per-
test cleanup) so we exercise the actual filesystem walking code path
that production uses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from plugins.skill_writer.skill_loader import (
    PRIMARY_SCRIPT_DEFAULT,
    SkillLoaderError,
    discover_skills,
    load_skill_folder,
)
from plugins.skill_writer.skill_spec import (
    SkillFrontmatterError,
    SkillNameError,
)


# ─── Helpers ─────────────────────────────────────────────────────────


def _write_skill_md(folder: Path, *, name: str, description: str = "Demo.") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody.\n",
        encoding="utf-8",
    )


def _write_skill_py(folder: Path, *, body: str = '    return {"ok": True}') -> None:
    scripts = folder / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "skill.py").write_text(
        "from typing import Any\n\n"
        "async def execute(api: Any, **kwargs: Any) -> dict[str, Any]:\n"
        f"{body}\n",
        encoding="utf-8",
    )


# ─── Happy path ──────────────────────────────────────────────────────


class TestLoadSkillFolderHappyPath:
    @pytest.mark.asyncio
    async def test_full_skill_with_scripts(self, tmp_path: Path) -> None:
        folder = tmp_path / "pdf-extract"
        _write_skill_md(folder, name="pdf-extract")
        _write_skill_py(folder)

        card = await load_skill_folder(folder)
        assert card.name == "pdf-extract"
        assert card.description == "Demo."
        assert card.has_scripts is True
        assert card.has_references is False
        assert card.has_assets is False
        assert card.primary_script == PRIMARY_SCRIPT_DEFAULT
        assert card.script_files == [PRIMARY_SCRIPT_DEFAULT]
        assert card.primary_script_path() == folder / "scripts" / "skill.py"

    @pytest.mark.asyncio
    async def test_docs_only_skill_has_no_primary(self, tmp_path: Path) -> None:
        """A skill without scripts/ is valid per spec — it just can't
        be ``run_skill``'d. Loader should not error."""
        folder = tmp_path / "docs-only"
        _write_skill_md(folder, name="docs-only")
        # No scripts/ subdir.

        card = await load_skill_folder(folder)
        assert card.has_scripts is False
        assert card.primary_script is None
        assert card.script_files == []

    @pytest.mark.asyncio
    async def test_with_references_and_assets(self, tmp_path: Path) -> None:
        folder = tmp_path / "full-skill"
        _write_skill_md(folder, name="full-skill")
        _write_skill_py(folder)
        (folder / "references").mkdir()
        (folder / "references" / "REFERENCE.md").write_text("ref", encoding="utf-8")
        (folder / "assets").mkdir()
        (folder / "assets" / "template.txt").write_text("tpl", encoding="utf-8")

        card = await load_skill_folder(folder)
        assert card.has_references is True
        assert card.has_assets is True
        assert "references/REFERENCE.md" in card.reference_files
        assert "assets/template.txt" in card.asset_files

    @pytest.mark.asyncio
    async def test_discovery_dict_minimal(self, tmp_path: Path) -> None:
        folder = tmp_path / "minimal"
        _write_skill_md(folder, name="minimal")
        card = await load_skill_folder(folder)
        # Discovery should ONLY return name + description (per spec —
        # ~100 tokens at agent boot).
        assert card.discovery_dict() == {
            "name": "minimal",
            "description": "Demo.",
        }


# ─── Error paths ─────────────────────────────────────────────────────


class TestLoadSkillFolderErrors:
    @pytest.mark.asyncio
    async def test_not_a_directory(self, tmp_path: Path) -> None:
        bad = tmp_path / "missing"
        with pytest.raises(SkillLoaderError):
            await load_skill_folder(bad)

    @pytest.mark.asyncio
    async def test_missing_skill_md(self, tmp_path: Path) -> None:
        folder = tmp_path / "no-md"
        folder.mkdir()
        with pytest.raises(SkillLoaderError):
            await load_skill_folder(folder)

    @pytest.mark.asyncio
    async def test_parent_dir_mismatch(self, tmp_path: Path) -> None:
        folder = tmp_path / "actual-name"
        _write_skill_md(folder, name="different-name")
        with pytest.raises(SkillNameError):
            await load_skill_folder(folder)

    @pytest.mark.asyncio
    async def test_invalid_yaml(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad-yaml"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: bad-yaml\nthis: is: broken\ndescription: D.\n---\n",
            encoding="utf-8",
        )
        with pytest.raises(SkillFrontmatterError):
            await load_skill_folder(folder)

    @pytest.mark.asyncio
    async def test_explicit_entry_must_exist(self, tmp_path: Path) -> None:
        folder = tmp_path / "missing-entry"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\n"
            "name: missing-entry\n"
            "description: D.\n"
            "metadata:\n"
            "  entry: scripts/nope.py\n"
            "---\n",
            encoding="utf-8",
        )
        scripts = folder / "scripts"
        scripts.mkdir()
        # Wrong file present, not the one declared.
        (scripts / "skill.py").write_text(
            "from typing import Any\nasync def execute(api: Any, **kwargs: Any) -> dict:\n    return {}\n",
            encoding="utf-8",
        )
        with pytest.raises(SkillLoaderError):
            await load_skill_folder(folder)


# ─── Discovery ───────────────────────────────────────────────────────


class TestDiscoverSkills:
    @pytest.mark.asyncio
    async def test_discovers_all_valid(self, tmp_path: Path) -> None:
        for name in ("alpha", "beta", "gamma"):
            f = tmp_path / name
            _write_skill_md(f, name=name)
            _write_skill_py(f)

        cards = await discover_skills(tmp_path)
        assert {c.name for c in cards} == {"alpha", "beta", "gamma"}

    @pytest.mark.asyncio
    async def test_skips_dotdirs_and_pycache(self, tmp_path: Path) -> None:
        """Hidden + __pycache__ dirs must not be treated as skills."""
        good = tmp_path / "real-skill"
        _write_skill_md(good, name="real-skill")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "__pycache__").mkdir()

        cards = await discover_skills(tmp_path)
        names = [c.name for c in cards]
        assert names == ["real-skill"]

    @pytest.mark.asyncio
    async def test_invalid_skills_are_skipped_not_fatal(
        self, tmp_path: Path,
    ) -> None:
        """One broken skill must not block discovery of the others."""
        good = tmp_path / "good-skill"
        _write_skill_md(good, name="good-skill")

        # Broken: parent name doesn't match frontmatter name.
        bad = tmp_path / "broken"
        _write_skill_md(bad, name="not-broken")

        cards = await discover_skills(tmp_path, log_errors=False)
        assert [c.name for c in cards] == ["good-skill"]

    @pytest.mark.asyncio
    async def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        result = await discover_skills(tmp_path / "nope")
        assert result == []
