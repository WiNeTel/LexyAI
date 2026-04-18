"""Tests for the Skill Writer plugin -- validator, template, registry, auto_agent.

Covers:
* parse_skill_header() and generate_skill_file()
* sanitize_skill_name()
* SkillValidator: allowed/forbidden imports, calls, execute() check, size limit
* SkillRegistry CRUD (real aiosqlite)
* AutoAgent creation and field defaults
* AgentManager spawn/stop/list
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
    parse_skill_header,
    generate_skill_file,
    sanitize_skill_name,
)
from plugins.skill_writer.skill_registry import SkillRegistry, SkillEntry
from plugins.skill_writer.auto_agent import AutoAgent, AgentManager


# ─── Allowed imports for the validator ────────────────────────────────────

ALLOWED = ["json", "re", "datetime", "math", "collections", "pathlib", "__future__"]


# ─── SkillTemplate ────────────────────────────────────────────────────────


class TestParseSkillHeader:
    def test_extracts_all_fields(self) -> None:
        src = (
            '"""\n'
            "Skill: my_skill\n"
            "Description: Does stuff\n"
            "Author: lexy_auto\n"
            "Version: 2.0\n"
            "Created: 2025-01-01T00:00:00Z\n"
            "Tags: web, search\n"
            '"""\n'
        )
        header = parse_skill_header(src)
        assert header["skill"] == "my_skill"
        assert header["description"] == "Does stuff"
        assert header["author"] == "lexy_auto"
        assert header["version"] == "2.0"
        assert header["created"] == "2025-01-01T00:00:00Z"
        assert header["tags"] == "web, search"

    def test_empty_source_returns_defaults(self) -> None:
        header = parse_skill_header("")
        assert header["skill"] == ""
        assert header["description"] == ""
        assert header["author"] == ""

    def test_no_docstring_returns_defaults(self) -> None:
        header = parse_skill_header("import json\n\nx = 1\n")
        assert header["skill"] == ""

    def test_partial_header(self) -> None:
        src = '"""\nSkill: partial_one\n"""\n'
        header = parse_skill_header(src)
        assert header["skill"] == "partial_one"
        assert header["description"] == ""

    def test_unclosed_docstring(self) -> None:
        src = '"""\nSkill: never_closed\nDescription: oops'
        header = parse_skill_header(src)
        assert header["skill"] == ""  # No closing quotes found


class TestGenerateSkillFile:
    def test_generates_valid_python(self) -> None:
        code = 'return {"hello": "world"}'
        result = generate_skill_file("test_skill", "A test skill", code)
        assert "Skill: test_skill" in result
        assert "Description: A test skill" in result
        assert "async def execute" in result
        # Must be valid Python
        compile(result, "skill.py", "exec")

    def test_auto_indents_code(self) -> None:
        code = 'return {"x": 1}'
        result = generate_skill_file("indent_test", "desc", code)
        # Body should be indented inside the function
        assert '    return {"x": 1}' in result

    def test_preserves_already_indented_code(self) -> None:
        code = '    result = api.llm_chat()\n    return {"ok": True}'
        result = generate_skill_file("pre_indented", "desc", code)
        assert "    result = api.llm_chat()" in result

    def test_tags_default_to_general(self) -> None:
        result = generate_skill_file("name", "desc", "return {}")
        assert "Tags: general" in result

    def test_custom_tags(self) -> None:
        result = generate_skill_file("name", "desc", "return {}", tags=["web", "ai"])
        assert "Tags: web, ai" in result

    def test_empty_code_gets_default(self) -> None:
        result = generate_skill_file("name", "desc", "")
        assert 'return {"status": "ok"}' in result


class TestSanitizeSkillName:
    def test_basic_sanitization(self) -> None:
        assert sanitize_skill_name("My Skill!") == "my_skill"

    def test_leading_digits_removed(self) -> None:
        assert sanitize_skill_name("123skill") == "skill"

    def test_double_underscores_collapsed(self) -> None:
        assert sanitize_skill_name("a__b___c") == "a_b_c"

    def test_empty_returns_unnamed(self) -> None:
        assert sanitize_skill_name("") == "unnamed_skill"
        assert sanitize_skill_name("!!!") == "unnamed_skill"

    def test_already_valid(self) -> None:
        assert sanitize_skill_name("web_search") == "web_search"


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


# ─── SkillRegistry ────────────────────────────────────────────────────────


class TestSkillRegistry:
    @pytest.mark.asyncio
    async def test_register_and_get(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()

            skill_id = await registry.register(
                name="hello_world",
                description="Says hello",
                file_path="/skills/hello_world.py",
                source="manual",
            )
            assert len(skill_id) == 12

            entry = await registry.get("hello_world")
            assert entry is not None
            assert entry.name == "hello_world"
            assert entry.description == "Says hello"
            assert entry.status == "active"
            assert entry.usage_count == 0

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

            await registry.register("s1", "desc1", "/s1.py")
            await registry.register("s2", "desc2", "/s2.py")

            all_skills = await registry.list_all()
            assert len(all_skills) == 2

    @pytest.mark.asyncio
    async def test_list_all_with_status_filter(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()

            await registry.register("active_skill", "d", "/a.py")
            await registry.register("disabled_skill", "d", "/d.py")
            await registry.set_status("disabled_skill", "disabled")

            active = await registry.list_all(status="active")
            assert len(active) == 1
            assert active[0].name == "active_skill"

    @pytest.mark.asyncio
    async def test_update_stats_success(self, tmp_path: Path) -> None:
        async with aiosqlite.connect(":memory:") as db:
            registry = SkillRegistry(db, tmp_path / "skills")
            await registry.init_tables()
            await registry.register("s1", "d", "/s1.py")

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
            await registry.register("s1", "d", "/s1.py")

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
            await registry.register("to_delete", "d", "/d.py")

            deleted = await registry.delete("to_delete")
            assert deleted is True
            assert await registry.get("to_delete") is None

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
            await registry.register("s1", "d", "/s1.py")

            await registry.set_status("s1", "failed")
            entry = await registry.get("s1")
            assert entry is not None
            assert entry.status == "failed"


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
