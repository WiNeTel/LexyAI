"""Tests for the Skill Writer plugin (Phase 11 — agentskills.io aware).

Covers:
* :func:`emit_skill_folder` — produces a spec-compliant folder
* :func:`sanitize_skill_name` — slug-style naming per spec
* :class:`SkillValidator` — allowed/forbidden imports, calls,
  execute() check, size limit, ``validate_folder``
* :class:`SkillRegistry` CRUD (real aiosqlite, folder paths)
* :class:`AutoAgent` creation and field defaults
* :class:`AgentManager` spawn/stop/list

The pre-Phase-11 ``parse_skill_header`` / ``generate_skill_file`` API
is gone; equivalent coverage now lives in
``tests/test_skill_spec.py`` (frontmatter parser).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from plugins.skill_writer.skill_validator import (
    SkillValidator,
    FORBIDDEN_CALLS,
    FORBIDDEN_MODULES,
)
from plugins.skill_writer.skill_template import (
    emit_skill_folder,
    sanitize_skill_name,
)
from plugins.skill_writer.skill_spec import parse_skill_md
from plugins.skill_writer.skill_registry import SkillRegistry, SkillEntry
from plugins.skill_writer.auto_agent import AutoAgent, AgentManager


# ─── Allowed imports for the validator ────────────────────────────────────
#
# Mirrors what ``SkillWriterPlugin.on_load`` configures: the plugin
# auto-adds ``__future__`` and ``typing`` so the auto-generated skill
# template's ``from typing import Any`` line passes.

ALLOWED = [
    "json", "re", "datetime", "math", "collections", "pathlib",
    "__future__", "typing",
]


# ─── emit_skill_folder ───────────────────────────────────────────────────


class TestEmitSkillFolder:
    """The Phase-11 replacement for ``generate_skill_file``."""

    def test_creates_folder_with_skill_md_and_script(
        self, tmp_path: Path,
    ) -> None:
        folder = emit_skill_folder(
            name="hello-world",
            description="Says hello.",
            code='return {"hello": "world"}',
            target_root=tmp_path,
        )
        assert folder == tmp_path / "hello-world"
        assert (folder / "SKILL.md").is_file()
        assert (folder / "scripts" / "skill.py").is_file()

    def test_skill_md_is_valid_frontmatter(self, tmp_path: Path) -> None:
        folder = emit_skill_folder(
            name="round-trip",
            description="Round-trip test.",
            code='return {}',
            target_root=tmp_path,
        )
        text = (folder / "SKILL.md").read_text(encoding="utf-8")
        fm = parse_skill_md(text, parent_dir_name="round-trip")
        assert fm.name == "round-trip"
        assert fm.description == "Round-trip test."

    def test_script_is_valid_python(self, tmp_path: Path) -> None:
        folder = emit_skill_folder(
            name="syntax-ok",
            description="d",
            code='return {"x": 1}',
            target_root=tmp_path,
        )
        source = (folder / "scripts" / "skill.py").read_text(encoding="utf-8")
        compile(source, "skill.py", "exec")
        # The execute() body should be indented under the function.
        assert '    return {"x": 1}' in source

    def test_preserves_already_indented_code(self, tmp_path: Path) -> None:
        folder = emit_skill_folder(
            name="indented",
            description="d",
            code='    result = 1\n    return {"r": result}',
            target_root=tmp_path,
        )
        source = (folder / "scripts" / "skill.py").read_text(encoding="utf-8")
        assert "    result = 1" in source

    def test_empty_code_gets_default_body(self, tmp_path: Path) -> None:
        folder = emit_skill_folder(
            name="empty-body",
            description="d",
            code="",
            target_root=tmp_path,
        )
        source = (folder / "scripts" / "skill.py").read_text(encoding="utf-8")
        assert 'return {"status": "ok"}' in source

    def test_metadata_lands_in_frontmatter(self, tmp_path: Path) -> None:
        folder = emit_skill_folder(
            name="with-meta",
            description="d",
            code="return {}",
            target_root=tmp_path,
            license="MIT",
            metadata={"version": "2.0", "author": "lexy"},
        )
        text = (folder / "SKILL.md").read_text(encoding="utf-8")
        fm = parse_skill_md(text, parent_dir_name="with-meta")
        assert fm.license == "MIT"
        assert fm.metadata == {"version": "2.0", "author": "lexy"}

    def test_existing_folder_without_overwrite_raises(
        self, tmp_path: Path,
    ) -> None:
        emit_skill_folder(
            name="dup",
            description="d",
            code="return {}",
            target_root=tmp_path,
        )
        with pytest.raises(FileExistsError):
            emit_skill_folder(
                name="dup",
                description="d",
                code="return {}",
                target_root=tmp_path,
            )

    def test_overwrite_replaces_folder(self, tmp_path: Path) -> None:
        first = emit_skill_folder(
            name="dup-overwrite",
            description="first",
            code="return {}",
            target_root=tmp_path,
        )
        # Sanity: original SKILL.md says "first"
        assert "first" in (first / "SKILL.md").read_text(encoding="utf-8")
        emit_skill_folder(
            name="dup-overwrite",
            description="second",
            code="return {}",
            target_root=tmp_path,
            overwrite=True,
        )
        assert "second" in (first / "SKILL.md").read_text(encoding="utf-8")

    def test_invalid_name_raises_before_disk_write(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError):
            emit_skill_folder(
                name="UPPERCASE",
                description="d",
                code="return {}",
                target_root=tmp_path,
            )
        # Folder must NOT have been created.
        assert not (tmp_path / "UPPERCASE").exists()


class TestSanitizeSkillName:
    """Slug-style names per the agentskills.io spec."""

    def test_spaces_become_hyphens(self) -> None:
        assert sanitize_skill_name("My Skill!") == "my-skill"

    def test_underscores_become_hyphens(self) -> None:
        assert sanitize_skill_name("hello_world") == "hello-world"

    def test_consecutive_special_chars_collapse(self) -> None:
        assert sanitize_skill_name("a___b   c") == "a-b-c"

    def test_leading_trailing_special_chars_stripped(self) -> None:
        assert sanitize_skill_name("---foo---") == "foo"

    def test_empty_returns_default(self) -> None:
        assert sanitize_skill_name("") == "unnamed-skill"
        assert sanitize_skill_name("!!!") == "unnamed-skill"

    def test_already_valid(self) -> None:
        assert sanitize_skill_name("pdf-extract") == "pdf-extract"

    def test_truncates_to_64_chars(self) -> None:
        long = "x" * 100
        result = sanitize_skill_name(long)
        assert len(result) <= 64

    def test_uppercase_lowercased(self) -> None:
        assert sanitize_skill_name("UPPER") == "upper"


# ─── SkillValidator ───────────────────────────────────────────────────────


class TestSkillValidator:
    def setup_method(self) -> None:
        self.v = SkillValidator(allowed_imports=ALLOWED)

    def test_valid_skill(self) -> None:
        src = 'import json\n\nasync def execute(api, **kwargs):\n    return {"ok": True}\n'
        ok, err = self.v.validate(src)
        assert ok is True
        assert err == ""

    def test_rejects_syntax_error(self) -> None:
        src = "def broken(:\n    pass\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "Syntax error" in err

    def test_rejects_forbidden_import_os(self) -> None:
        src = "import os\n\nasync def execute(api, **kwargs):\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "os" in err

    def test_rejects_forbidden_import_subprocess(self) -> None:
        src = "import subprocess\n\nasync def execute(api, **kwargs):\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "subprocess" in err

    def test_rejects_forbidden_from_import(self) -> None:
        src = "from os.path import join\n\nasync def execute(api, **kwargs):\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "os" in err

    def test_rejects_eval_call(self) -> None:
        src = 'async def execute(api, **kwargs):\n    return eval("1+1")\n'
        ok, err = self.v.validate(src)
        assert ok is False
        assert "eval" in err

    def test_rejects_exec_call(self) -> None:
        src = 'async def execute(api, **kwargs):\n    exec("x=1")\n    return {}\n'
        ok, err = self.v.validate(src)
        assert ok is False
        assert "exec" in err

    def test_rejects_open_call(self) -> None:
        src = 'async def execute(api, **kwargs):\n    f = open("file")\n    return {}\n'
        ok, err = self.v.validate(src)
        assert ok is False
        assert "open" in err

    def test_rejects_missing_execute(self) -> None:
        src = "import json\n\ndef helper():\n    pass\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "execute" in err

    def test_rejects_sync_execute(self) -> None:
        src = "def execute(api, **kwargs):\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "async" in err

    def test_rejects_oversized_skill(self) -> None:
        v = SkillValidator(allowed_imports=ALLOWED, max_size_bytes=50)
        src = 'async def execute(api, **kwargs):\n    return {"x": "' + "a" * 100 + '"}\n'
        ok, err = v.validate(src)
        assert ok is False
        assert "too large" in err

    def test_allows_whitelisted_import(self) -> None:
        src = "import json\nimport re\nimport math\n\nasync def execute(api, **kwargs):\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is True

    def test_rejects_non_whitelisted_import(self) -> None:
        src = "import requests\n\nasync def execute(api, **kwargs):\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "requests" in err

    def test_rejects_getattr_call(self) -> None:
        src = 'async def execute(api, **kwargs):\n    return getattr(api, "x")\n'
        ok, err = self.v.validate(src)
        assert ok is False
        assert "getattr" in err

    def test_rejects_dunder_import(self) -> None:
        src = 'async def execute(api, **kwargs):\n    return __import__("os")\n'
        ok, err = self.v.validate(src)
        assert ok is False
        assert "__import__" in err

    def test_execute_needs_api_param(self) -> None:
        src = "async def execute():\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "positional argument" in err

    def test_rejects_relative_import(self) -> None:
        src = "from . import something\n\nasync def execute(api, **kwargs):\n    return {}\n"
        ok, err = self.v.validate(src)
        assert ok is False
        assert "Relative" in err


class TestForbiddenSets:
    def test_forbidden_calls_nonempty(self) -> None:
        assert len(FORBIDDEN_CALLS) >= 5
        assert "eval" in FORBIDDEN_CALLS
        assert "exec" in FORBIDDEN_CALLS
        assert "open" in FORBIDDEN_CALLS

    def test_forbidden_modules_nonempty(self) -> None:
        assert len(FORBIDDEN_MODULES) >= 5
        assert "os" in FORBIDDEN_MODULES
        assert "subprocess" in FORBIDDEN_MODULES
        assert "socket" in FORBIDDEN_MODULES


class TestValidateFolder:
    """Phase-11 method: validates the whole skill folder, not just one file."""

    def setup_method(self) -> None:
        self.v = SkillValidator(allowed_imports=ALLOWED)

    def test_emitted_skill_validates_clean(self, tmp_path: Path) -> None:
        folder = emit_skill_folder(
            name="ok-skill",
            description="d",
            code='return {"ok": True}',
            target_root=tmp_path,
        )
        ok, err = self.v.validate_folder(folder)
        assert ok is True, err

    def test_docs_only_skill_passes(self, tmp_path: Path) -> None:
        """Spec allows skills without scripts/."""
        folder = tmp_path / "docs-only"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: docs-only\ndescription: D.\n---\n", encoding="utf-8"
        )
        ok, err = self.v.validate_folder(folder)
        assert ok is True

    def test_helper_does_not_need_execute(self, tmp_path: Path) -> None:
        """Helper scripts in scripts/ don't need execute()."""
        folder = emit_skill_folder(
            name="with-helper",
            description="d",
            code="return {}",
            target_root=tmp_path,
        )
        # Drop a helper that's just utility functions.
        helper = folder / "scripts" / "utils.py"
        helper.write_text(
            "from typing import Any\n\n"
            "def add(x: int, y: int) -> int:\n"
            "    return x + y\n",
            encoding="utf-8",
        )
        ok, err = self.v.validate_folder(folder)
        assert ok is True, err

    def test_helper_with_forbidden_import_rejected(
        self, tmp_path: Path,
    ) -> None:
        folder = emit_skill_folder(
            name="bad-helper",
            description="d",
            code="return {}",
            target_root=tmp_path,
        )
        bad = folder / "scripts" / "evil.py"
        bad.write_text("import os\n", encoding="utf-8")
        ok, err = self.v.validate_folder(folder)
        assert ok is False
        assert "evil.py" in err and "os" in err

    def test_missing_primary_script_rejected(self, tmp_path: Path) -> None:
        folder = tmp_path / "no-primary"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: no-primary\ndescription: D.\n---\n", encoding="utf-8"
        )
        scripts = folder / "scripts"
        scripts.mkdir()
        # Only a helper, no skill.py — Phase-11 convention requires
        # the primary to exist when scripts/ is non-empty.
        (scripts / "utils.py").write_text(
            "def helper():\n    return 1\n", encoding="utf-8",
        )
        ok, err = self.v.validate_folder(folder)
        assert ok is False
        assert "primary" in err.lower()


# ─── SkillRegistry ────────────────────────────────────────────────────────


class TestSkillRegistry:
    """Phase 11: ``file_path`` now stores folder paths, plus new
    frontmatter columns (license, compatibility, metadata, body_md)
    persist."""

    @pytest.mark.asyncio
    async def test_register_and_get(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()

            skill_id = await registry.register(
                name="hello-world",
                description="Says hello",
                file_path=str(tmp_path / "hello-world"),
                source="manual",
            )
            assert len(skill_id) == 12

            entry = await registry.get("hello-world")
            assert entry is not None
            assert entry.name == "hello-world"
            assert entry.description == "Says hello"
            assert entry.status == "active"
            assert entry.usage_count == 0
            # New Phase-11 columns default sensibly when not provided
            assert entry.license is None
            assert entry.metadata == {}
            assert entry.body_md == ""

    @pytest.mark.asyncio
    async def test_register_with_frontmatter(self, tmp_path: Path) -> None:
        """Phase 11 columns round-trip cleanly."""
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            await registry.register(
                name="full-card",
                description="d",
                file_path=str(tmp_path / "full-card"),
                license="MIT",
                compatibility="Python 3.11+",
                metadata={"author": "lexy", "version": "1.0"},
                allowed_tools="Bash Read",
                body_md="# step 1\n",
            )
            entry = await registry.get("full-card")
            assert entry is not None
            assert entry.license == "MIT"
            assert entry.compatibility == "Python 3.11+"
            assert entry.metadata == {"author": "lexy", "version": "1.0"}
            assert entry.allowed_tools == "Bash Read"
            assert "step 1" in entry.body_md

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()

            assert await registry.get("nope") is None

    @pytest.mark.asyncio
    async def test_list_all(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()

            await registry.register("s1", "desc1", str(tmp_path / "s1"))
            await registry.register("s2", "desc2", str(tmp_path / "s2"))

            all_skills = await registry.list_all()
            assert len(all_skills) == 2

    @pytest.mark.asyncio
    async def test_list_all_with_status_filter(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()

            await registry.register("active-skill", "d", str(tmp_path / "a"))
            await registry.register("disabled-skill", "d", str(tmp_path / "d"))
            await registry.set_status("disabled-skill", "disabled")

            active = await registry.list_all(status="active")
            assert len(active) == 1
            assert active[0].name == "active-skill"

    @pytest.mark.asyncio
    async def test_update_stats_success(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            await registry.register("s1", "d", str(tmp_path / "s1"))

            await registry.update_stats("s1", success=True)
            entry = await registry.get("s1")
            assert entry is not None
            assert entry.usage_count == 1
            assert entry.success_count == 1
            assert entry.failure_count == 0
            assert entry.last_used_at is not None

    @pytest.mark.asyncio
    async def test_update_stats_failure(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            await registry.register("s1", "d", str(tmp_path / "s1"))

            await registry.update_stats("s1", success=False)
            entry = await registry.get("s1")
            assert entry is not None
            assert entry.usage_count == 1
            assert entry.success_count == 0
            assert entry.failure_count == 1

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            await registry.register("to-delete", "d", str(tmp_path / "to-delete"))

            deleted = await registry.delete("to-delete")
            assert deleted is True
            assert await registry.get("to-delete") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()

            deleted = await registry.delete("nope")
            assert deleted is False

    @pytest.mark.asyncio
    async def test_set_status(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            await registry.register("s1", "d", str(tmp_path / "s1"))

            await registry.set_status("s1", "failed")
            entry = await registry.get("s1")
            assert entry is not None
            assert entry.status == "failed"

    @pytest.mark.asyncio
    async def test_init_tables_idempotent_alter(self, tmp_path: Path) -> None:
        """Schema migration must be safe to run twice in a row.

        Pre-Phase-11 DBs only have the original 12 columns; running
        ``init_tables`` again should add the 5 new columns silently.
        """
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            # Second call must succeed — the ALTER would otherwise
            # raise "duplicate column" if not swallowed properly.
            await registry.init_tables()

    @pytest.mark.asyncio
    async def test_update_metadata(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            await registry.register(
                name="updateable",
                description="old",
                file_path=str(tmp_path / "updateable"),
            )
            ok = await registry.update_metadata(
                "updateable",
                description="new",
                license="Apache-2.0",
                metadata={"version": "2.0"},
            )
            assert ok is True
            entry = await registry.get("updateable")
            assert entry is not None
            assert entry.description == "new"
            assert entry.license == "Apache-2.0"
            assert entry.metadata == {"version": "2.0"}

    @pytest.mark.asyncio
    async def test_scan_disk_picks_up_folders(self, tmp_path: Path) -> None:
        """``scan_disk`` walks folders and registers ones not yet in DB."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        # Create a valid skill folder on disk.
        skill_dir = skills_root / "scanned-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: scanned-skill\ndescription: Demo.\n---\n",
            encoding="utf-8",
        )

        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, skills_root)
            await registry.init_tables()
            added = await registry.scan_disk()
            assert added == 1
            entry = await registry.get("scanned-skill")
            assert entry is not None
            assert entry.description == "Demo."
            # Second scan finds nothing new.
            again = await registry.scan_disk()
            assert again == 0


# ─── AutoAgent ────────────────────────────────────────────────────────────


class TestAutoAgent:
    def test_creation_defaults(self) -> None:
        api = MagicMock()
        agent = AutoAgent(
            agent_id="a1",
            name="test_agent",
            system_prompt="You are a helper.",
            task="Do something",
            api=api,
        )
        assert agent.agent_id == "a1"
        assert agent.name == "test_agent"
        assert agent.status == "idle"
        assert agent.messages == []
        assert agent.results == []
        assert agent._iteration == 0

    def test_custom_params(self) -> None:
        api = MagicMock()
        agent = AutoAgent(
            agent_id="a2",
            name="custom",
            system_prompt="prompt",
            task="task",
            api=api,
            brain="a4b",
            max_iterations=5,
            timeout=60.0,
        )
        assert agent._brain == "a4b"
        assert agent._max_iterations == 5
        assert agent._timeout == 60.0

    def test_get_conversation_returns_copy(self) -> None:
        api = MagicMock()
        agent = AutoAgent("a1", "test", "prompt", "task", api)
        agent.messages = [{"role": "system", "content": "hello"}]
        conv = agent.get_conversation()
        assert conv == agent.messages
        # Must be a copy
        conv.append({"role": "user", "content": "extra"})
        assert len(agent.messages) == 1


# ─── AgentManager ─────────────────────────────────────────────────────────


class TestAgentManager:
    def test_list_agents_empty(self) -> None:
        api = MagicMock()
        mgr = AgentManager(api, max_concurrent=3)
        assert mgr.list_agents() == []

    @pytest.mark.asyncio
    async def test_spawn_creates_agent(self) -> None:
        api = MagicMock()
        api.llm_chat = AsyncMock(return_value="Done. No tools needed.")
        api.get_tool_caller.return_value = None
        api.ws_broadcast = AsyncMock()
        api.emit = AsyncMock()
        api.memory_store = AsyncMock()

        mgr = AgentManager(api, max_concurrent=3)
        result = await mgr.spawn("helper", "Say hello")

        assert "agent_id" in result
        assert result["name"] == "helper"
        assert result["status"] == "running"

        # Give the task time to complete
        await asyncio.sleep(0.2)

        agents = mgr.list_agents()
        assert len(agents) == 1

        await mgr.cleanup()

    @pytest.mark.asyncio
    async def test_spawn_limit_reached(self) -> None:
        api = MagicMock()
        api.ws_broadcast = AsyncMock()
        api.emit = AsyncMock()
        api.get_tool_caller.return_value = None

        mgr = AgentManager(api, max_concurrent=1)

        # Use a long-running LLM to keep the agent "running"
        async def slow_chat(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(100)
            return "done"

        api.llm_chat = slow_chat
        first = await mgr.spawn("agent1", "task1")
        assert "agent_id" in first

        # Allow the first agent's task to start executing so status becomes "running"
        await asyncio.sleep(0.05)

        # Second should fail due to limit
        result = await mgr.spawn("agent2", "task2")
        assert "error" in result

        await mgr.cleanup()

    @pytest.mark.asyncio
    async def test_stop_agent(self) -> None:
        api = MagicMock()

        async def slow_chat(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(100)
            return "done"

        api.llm_chat = slow_chat
        api.ws_broadcast = AsyncMock()
        api.emit = AsyncMock()
        api.get_tool_caller.return_value = None

        mgr = AgentManager(api, max_concurrent=3)
        result = await mgr.spawn("stoppable", "long task")
        agent_id = result["agent_id"]

        stopped = await mgr.stop(agent_id)
        assert stopped is True

        # Stopping again returns False (task is done/cancelled)
        await asyncio.sleep(0.1)
        stopped2 = await mgr.stop(agent_id)
        assert stopped2 is False

        await mgr.cleanup()

    @pytest.mark.asyncio
    async def test_get_agent(self) -> None:
        api = MagicMock()
        api.llm_chat = AsyncMock(return_value="Result.")
        api.ws_broadcast = AsyncMock()
        api.emit = AsyncMock()
        api.memory_store = AsyncMock()
        api.get_tool_caller.return_value = None

        mgr = AgentManager(api, max_concurrent=3)
        result = await mgr.spawn("finder", "task")
        agent_id = result["agent_id"]

        agent = mgr.get_agent(agent_id)
        assert agent is not None
        assert agent.name == "finder"

        assert mgr.get_agent("nonexistent") is None

        await mgr.cleanup()
