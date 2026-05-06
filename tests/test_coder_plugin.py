"""Tests for the self-coding (coder) plugin.

Five focus areas, each in its own class:

* WorkspaceManager — path-whitelist + init_project + read/write/list/delete
* CodeRunner — subprocess timeout, returncode, output truncation, spawn errors
* GitCommitter — init, add_and_commit (with a real local git when available)
* ApprovalGate — auto-low, session grants, request/resolve flow, timeout
* CoderBrain — Plan→Code→Test→Reflect with deterministic fakes for LLM + tools

The CoderBrain test in particular exercises the retry-on-failure path that
implements Mike's "learn from errors" requirement, using an
:class:`ErrorLearning` instance with an in-memory fake memory backend.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
import pytest_asyncio

from plugins.coder.approval_gate import ApprovalGate, RISK_HIGH, RISK_LOW, RISK_MED
from plugins.coder.code_runner import CodeRunner
from plugins.coder.coder_brain import (
    CoderBrain,
    _parse_plan,
    _parse_tool_call,
)
from plugins.coder.conda_env import CondaEnvManager
from plugins.coder.error_learning import ErrorLearning
from plugins.coder.git_committer import GitCommitter, GitNotAvailable
from plugins.coder.workspace_mgr import (
    WorkspaceManager,
    WorkspaceNotFoundError,
    WorkspacePathError,
)


# ─── Fakes ───────────────────────────────────────────────────────────


class _FakeMemory:
    """In-memory MemoryManager stand-in used by ErrorLearning tests."""

    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.solutions: list[dict[str, Any]] = []
        # Pre-recall hits the test seeds; key by collection name.
        self._recall_hits: dict[str, list[dict[str, Any]]] = {
            "errors": [],
            "solutions": [],
        }

    async def store(self, *, text: str, collection: str, metadata: dict[str, Any]) -> None:
        bucket = self.errors if collection == "errors" else self.solutions
        bucket.append({"text": text, "metadata": dict(metadata)})

    async def recall(
        self,
        *,
        query: str,
        collection: str,
        limit: int = 5,
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        hits = list(self._recall_hits.get(collection, []))
        if metadata_equals:
            hits = [
                h for h in hits
                if all(
                    (h.get("metadata") or {}).get(k) == v
                    for k, v in metadata_equals.items()
                )
            ]
        return hits[: max(1, limit)]


class _ScriptedLLM:
    """Async LLM stand-in: returns scripted strings in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def workspace(tmp_path: Path) -> WorkspaceManager:
    mgr = WorkspaceManager(tmp_path / "workspace")
    await mgr.ensure_layout()
    return mgr


@pytest_asyncio.fixture
async def gate_db(tmp_path: Path):
    db = await aiosqlite.connect(str(tmp_path / "approvals.db"))
    yield db
    await db.close()


# ─── WorkspaceManager ───────────────────────────────────────────────


class TestWorkspaceManager:
    @pytest.mark.asyncio
    async def test_init_skill_creates_template_files(
        self, workspace: WorkspaceManager
    ) -> None:
        info = await workspace.init_project(name="hello", kind="skill")
        assert info.runnable is True
        assert (info.root / "SKILL.md").exists()
        assert (info.root / "skill.py").exists()
        assert (info.root / "tests").is_dir()

    @pytest.mark.asyncio
    async def test_init_project_marks_non_runnable(
        self, workspace: WorkspaceManager
    ) -> None:
        info = await workspace.init_project(name="myproj", kind="project")
        assert info.runnable is False
        assert (info.root / "README.md").exists()
        assert (info.root / ".gitignore").exists()

    @pytest.mark.asyncio
    async def test_init_extension_uses_camelcase_class(
        self, workspace: WorkspaceManager
    ) -> None:
        info = await workspace.init_project(name="my_ext", kind="extension")
        py = (info.root / "plugin.py").read_text(encoding="utf-8")
        yaml = (info.root / "plugin.yaml").read_text(encoding="utf-8")
        assert "class MyExtPlugin(BasePlugin)" in py
        assert "entry: plugin.MyExtPlugin" in yaml

    @pytest.mark.asyncio
    async def test_invalid_name_rejected(
        self, workspace: WorkspaceManager
    ) -> None:
        with pytest.raises(WorkspacePathError):
            await workspace.init_project(name="../../etc", kind="skill")
        with pytest.raises(WorkspacePathError):
            await workspace.init_project(name="bad name", kind="skill")
        with pytest.raises(WorkspacePathError):
            # Reserved Windows name
            await workspace.init_project(name="con", kind="skill")

    @pytest.mark.asyncio
    async def test_double_init_rejected(self, workspace: WorkspaceManager) -> None:
        await workspace.init_project(name="dup", kind="skill")
        with pytest.raises(WorkspacePathError):
            await workspace.init_project(name="dup", kind="skill")

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(
        self, workspace: WorkspaceManager
    ) -> None:
        await workspace.init_project(name="hello", kind="skill")
        with pytest.raises(WorkspacePathError):
            workspace.resolve_inside(
                kind="skill", name="hello", rel_path="../../../etc/passwd"
            )
        with pytest.raises(WorkspacePathError):
            workspace.resolve_inside(
                kind="skill", name="hello", rel_path="..\\..\\..\\Windows\\System32",
            )

    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(
        self, workspace: WorkspaceManager
    ) -> None:
        await workspace.init_project(name="hello", kind="skill")
        await workspace.write_file(
            kind="skill", name="hello", rel_path="helpers/util.py",
            content="x = 1\n",
        )
        files = workspace.list_files(
            kind="skill", name="hello", rel_path="helpers"
        )
        assert any(f.name == "util.py" and not f.is_dir for f in files)
        text = workspace.read_file(
            kind="skill", name="hello", rel_path="helpers/util.py"
        )
        assert text == "x = 1\n"

    @pytest.mark.asyncio
    async def test_delete_file_works_but_not_directory(
        self, workspace: WorkspaceManager
    ) -> None:
        await workspace.init_project(name="hello", kind="skill")
        await workspace.write_file(
            kind="skill", name="hello", rel_path="tmp/ephemeral.txt",
            content="bye",
        )
        assert await workspace.delete_file(
            kind="skill", name="hello", rel_path="tmp/ephemeral.txt",
        ) is True
        with pytest.raises(WorkspacePathError):
            await workspace.delete_file(
                kind="skill", name="hello", rel_path="tmp",
            )

    @pytest.mark.asyncio
    async def test_read_too_large_rejected(
        self, workspace: WorkspaceManager
    ) -> None:
        await workspace.init_project(name="hello", kind="skill")
        await workspace.write_file(
            kind="skill", name="hello", rel_path="big.txt",
            content="A" * 10_000,
        )
        with pytest.raises(WorkspacePathError):
            workspace.read_file(
                kind="skill", name="hello", rel_path="big.txt",
                max_bytes=1024,
            )


# ─── CodeRunner ─────────────────────────────────────────────────────


class TestCodeRunner:
    @pytest.mark.asyncio
    async def test_simple_success(self, tmp_path: Path) -> None:
        runner = CodeRunner(default_timeout=5.0)
        result = await runner.run(
            [sys.executable, "-c", "print('hello')"],
            cwd=str(tmp_path),
        )
        assert result.ok
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert result.killed_reason == ""

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, tmp_path: Path) -> None:
        runner = CodeRunner(default_timeout=5.0)
        result = await runner.run(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            cwd=str(tmp_path),
        )
        assert not result.ok
        assert result.returncode == 7
        assert result.killed_reason == ""

    @pytest.mark.asyncio
    async def test_timeout_kills(self, tmp_path: Path) -> None:
        runner = CodeRunner(default_timeout=0.5)
        result = await runner.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=str(tmp_path),
            timeout=0.5,
        )
        assert result.killed_reason == "timeout"
        assert not result.ok
        # Should not take the full 60s.
        assert result.duration_s < 5.0

    @pytest.mark.asyncio
    async def test_spawn_error_caught(self, tmp_path: Path) -> None:
        runner = CodeRunner(default_timeout=2.0)
        result = await runner.run(
            ["nonexistent-binary-xyz-987"],
            cwd=str(tmp_path),
        )
        assert result.killed_reason == "spawn_error"
        assert result.returncode == -1

    @pytest.mark.asyncio
    async def test_stdout_truncated(self, tmp_path: Path) -> None:
        runner = CodeRunner(default_timeout=5.0, max_output_bytes=200)
        # Print 2000 bytes — well over the 200 cap.
        result = await runner.run(
            [
                sys.executable, "-c",
                "import sys; sys.stdout.write('A'*2000)",
            ],
            cwd=str(tmp_path),
        )
        assert result.truncated
        assert "[truncated" in result.stdout
        # Truncation kept the cap-sized prefix.
        assert result.stdout.startswith("A" * 200)

    @pytest.mark.asyncio
    async def test_error_summary_for_run_result(self, tmp_path: Path) -> None:
        runner = CodeRunner(default_timeout=2.0)
        timeout_result = await runner.run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(tmp_path), timeout=0.3,
        )
        assert "Timed out" in timeout_result.error_summary

        nonzero = await runner.run(
            [
                sys.executable, "-c",
                "import sys; sys.stderr.write('boom'); sys.exit(2)",
            ],
            cwd=str(tmp_path),
        )
        assert "Exited with code 2" in nonzero.error_summary
        assert "boom" in nonzero.error_summary


# ─── GitCommitter ───────────────────────────────────────────────────


class TestGitCommitter:
    """Real git is required — the suite skips when not available."""

    @pytest.mark.asyncio
    async def test_init_then_commit_then_log(self, tmp_path: Path) -> None:
        git = GitCommitter()
        if not git.is_available():
            pytest.skip("git not on PATH")
        repo = tmp_path / "rp"
        repo.mkdir()
        # init() creates the baseline commit; write the file *after*
        # so add_and_commit has actual changes to stage.
        await git.init(repo)
        (repo / "README.md").write_text("# hello\n", encoding="utf-8")
        commit = await git.add_and_commit(
            repo, files=["README.md"], message="add README",
        )
        assert commit is not None
        assert commit.short_sha
        log_entries = await git.log(repo, limit=10)
        assert len(log_entries) >= 1
        assert any("README" in e.subject or "init" in e.subject for e in log_entries)

    @pytest.mark.asyncio
    async def test_commit_returns_none_on_clean_tree(self, tmp_path: Path) -> None:
        git = GitCommitter()
        if not git.is_available():
            pytest.skip("git not on PATH")
        repo = tmp_path / "clean"
        repo.mkdir()
        await git.init(repo)
        # Nothing changed since init's baseline commit.
        result = await git.add_and_commit(
            repo, files=["-A"], message="should be skipped",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_diff_after_change(self, tmp_path: Path) -> None:
        git = GitCommitter()
        if not git.is_available():
            pytest.skip("git not on PATH")
        repo = tmp_path / "rp"
        repo.mkdir()
        (repo / "x.txt").write_text("first\n", encoding="utf-8")
        await git.init(repo)
        await git.add_and_commit(repo, files=["-A"], message="seed")
        (repo / "x.txt").write_text("second\n", encoding="utf-8")
        diff = await git.diff(repo)
        assert "second" in diff or "+second" in diff


# ─── ApprovalGate ───────────────────────────────────────────────────


class TestApprovalGate:
    @pytest.mark.asyncio
    async def test_auto_approve_low_risk(self, gate_db) -> None:
        broadcasts: list[dict[str, Any]] = []
        async def bc(payload):
            broadcasts.append(payload)

        gate = ApprovalGate(broadcast=bc, auto_approve_low=True)
        await gate.init_db(gate_db)
        decision = await gate.request(
            action="workspace_list", risk=RISK_LOW, payload={},
        )
        assert decision.approved is True
        assert decision.reason == "auto_low"
        assert broadcasts == []  # no UI roundtrip

    @pytest.mark.asyncio
    async def test_med_risk_waits_for_response(self, gate_db) -> None:
        sent: list[dict[str, Any]] = []
        async def bc(payload):
            sent.append(payload)

        gate = ApprovalGate(broadcast=bc, default_timeout=5.0)
        await gate.init_db(gate_db)

        async def respond_after_send():
            # Wait for the broadcast to land, then resolve.
            for _ in range(50):
                if sent:
                    break
                await asyncio.sleep(0.01)
            assert sent, "broadcast never fired"
            req_id = sent[0]["request_id"]
            gate.resolve(request_id=req_id, approved=True, reason="user")

        responder = asyncio.create_task(respond_after_send())
        decision = await gate.request(
            action="workspace_write", risk=RISK_MED, payload={"x": 1},
        )
        await responder
        assert decision.approved is True
        assert decision.reason == "user"

    @pytest.mark.asyncio
    async def test_timeout_yields_rejection(self, gate_db) -> None:
        async def bc(payload):
            return None
        gate = ApprovalGate(broadcast=bc, default_timeout=0.1)
        await gate.init_db(gate_db)
        decision = await gate.request(
            action="workspace_run", risk=RISK_HIGH,
            payload={}, timeout_seconds=0.1,
        )
        assert decision.approved is False
        assert decision.reason == "timeout"

    @pytest.mark.asyncio
    async def test_session_grant_skips_modal(self, gate_db) -> None:
        sent: list[dict[str, Any]] = []
        async def bc(payload):
            sent.append(payload)
        gate = ApprovalGate(broadcast=bc, default_timeout=5.0)
        await gate.init_db(gate_db)
        gate.grant_session(
            session_id="s1", action="workspace_write", ttl_seconds=60.0,
        )
        decision = await gate.request(
            action="workspace_write", risk=RISK_MED,
            session_id="s1", payload={},
        )
        assert decision.approved is True
        assert decision.reason == "auto_session"
        assert sent == []  # short-circuited

    @pytest.mark.asyncio
    async def test_high_risk_ignores_session_grant(self, gate_db) -> None:
        sent: list[dict[str, Any]] = []
        async def bc(payload):
            sent.append(payload)
        gate = ApprovalGate(broadcast=bc, default_timeout=0.1)
        await gate.init_db(gate_db)
        gate.grant_session(session_id="s1", action="workspace_run")
        decision = await gate.request(
            action="workspace_run", risk=RISK_HIGH,
            session_id="s1", payload={}, timeout_seconds=0.1,
        )
        # No session shortcut for HIGH risk → modal would have to confirm,
        # but we let it time out so the assertion is decisive.
        assert decision.approved is False
        assert decision.reason == "timeout"

    @pytest.mark.asyncio
    async def test_audit_log_recorded(self, gate_db) -> None:
        async def bc(payload):
            return None
        gate = ApprovalGate(broadcast=bc, auto_approve_low=True)
        await gate.init_db(gate_db)
        await gate.request(action="workspace_list", risk=RISK_LOW, payload={})
        async with gate_db.execute(
            "SELECT action, approved, reason FROM approvals"
        ) as cur:
            rows = list(await cur.fetchall())
        assert len(rows) == 1
        action, approved, reason = rows[0]
        assert action == "workspace_list"
        assert approved == 1
        assert reason == "auto_low"


# ─── ErrorLearning ──────────────────────────────────────────────────


class TestErrorLearning:
    @pytest.mark.asyncio
    async def test_remember_failure_stores_in_errors(self) -> None:
        mem = _FakeMemory()
        learning = ErrorLearning(mem)
        ok = await learning.remember_failure(
            text="ImportError: no module named foo",
            task_tag="coder/skill/foo",
        )
        assert ok
        assert len(mem.errors) == 1
        assert mem.errors[0]["metadata"]["kind"] == "code_runtime"

    @pytest.mark.asyncio
    async def test_recall_filters_by_task_tag(self) -> None:
        mem = _FakeMemory()
        mem._recall_hits["errors"] = [
            {"content": "matches", "metadata": {"task_tag": "coder/skill/foo"}},
            {"content": "off-topic", "metadata": {"task_tag": "coder/skill/bar"}},
        ]
        learning = ErrorLearning(mem)
        hits = await learning.recall_similar(
            query="something", task_tag="coder/skill/foo",
        )
        # The off-topic one is dropped because the tag doesn't match.
        assert len(hits) == 1
        assert hits[0].text == "matches"

    @pytest.mark.asyncio
    async def test_no_memory_no_recall(self) -> None:
        learning = ErrorLearning(None)
        hits = await learning.recall_similar(query="anything")
        assert hits == []
        ok = await learning.remember_failure(text="oops", task_tag="t")
        assert ok is False


# ─── CoderBrain (Plan→Code→Test→Reflect) ───────────────────────────


class TestCoderBrainParsers:
    def test_plan_parses_json_array(self) -> None:
        plan = _parse_plan('["a", "b", "c"]')
        assert plan == ["a", "b", "c"]

    def test_plan_parses_numbered_list(self) -> None:
        plan = _parse_plan("1. first\n2. second\n3. third")
        assert plan == ["first", "second", "third"]

    def test_plan_strips_code_fence(self) -> None:
        plan = _parse_plan('```json\n["x", "y"]\n```')
        assert plan == ["x", "y"]

    def test_tool_call_strict_json(self) -> None:
        call = _parse_tool_call('{"tool":"workspace_write","arguments":{"a":1}}')
        assert call == {"tool": "workspace_write", "arguments": {"a": 1}}

    def test_tool_call_extracts_object_substring(self) -> None:
        call = _parse_tool_call(
            'Here is the call: {"tool":"workspace_run","arguments":{"name":"foo"}} done.'
        )
        assert call is not None
        assert call["tool"] == "workspace_run"

    def test_tool_call_unparseable_returns_none(self) -> None:
        assert _parse_tool_call("not even close") is None
        assert _parse_tool_call("") is None


class TestCoderBrainLoop:
    @pytest.mark.asyncio
    async def test_happy_path_two_steps(self) -> None:
        # The LLM returns a 2-step plan, then one tool_call per step.
        # Both tools succeed → task ends with state=done.
        llm = _ScriptedLLM(
            [
                # Plan
                '["initialise project", "write skill.py"]',
                # Step 1 tool call
                '{"tool":"workspace_init_project","arguments":{"kind":"skill","name":"foo"}}',
                # Step 2 tool call
                '{"tool":"workspace_write","arguments":{"kind":"skill","name":"foo","rel_path":"skill.py","content":"print(1)"}}',
            ]
        )
        called: list[tuple[str, dict[str, Any]]] = []

        async def runner(name: str, args: dict[str, Any]) -> dict[str, Any]:
            called.append((name, dict(args)))
            return {"ok": True}

        events: list[dict[str, Any]] = []
        async def broadcast(payload: dict[str, Any]) -> None:
            events.append(payload)

        mem = _FakeMemory()
        brain = CoderBrain(
            llm_chat=llm,
            tool_runner=runner,
            broadcast=broadcast,
            error_learning=ErrorLearning(mem),
            brain="a4b",
            max_steps=12,
            max_retries_per_step=3,
        )
        task_id = await brain.submit(
            description="schreibe einen sortier-skill",
            kind="skill",
            tool_catalog="(tools)",
        )
        # Wait for the runner task to settle.
        for _ in range(200):
            task = brain.get(task_id)
            if task and task.state in ("done", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
        task = brain.get(task_id)
        assert task is not None, "task disappeared"
        assert task.state == "done", f"got {task.state}: {task.last_error}"
        assert len(task.steps) == 2
        assert all(s.completed for s in task.steps)
        assert called == [
            (
                "workspace_init_project",
                {"kind": "skill", "name": "foo"},
            ),
            (
                "workspace_write",
                {
                    "kind": "skill", "name": "foo",
                    "rel_path": "skill.py", "content": "print(1)",
                },
            ),
        ]
        # Success → solution recorded.
        assert any(
            evt.get("type") == "coder_done" for evt in events
        )

    @pytest.mark.asyncio
    async def test_retry_and_record_failure(self) -> None:
        # 1-step plan. First attempt fails, second succeeds. We expect
        # the brain to record the failure to ErrorLearning and end ``done``.
        llm = _ScriptedLLM(
            [
                '["run the thing"]',
                # First tool call (fails)
                '{"tool":"workspace_run","arguments":{"name":"foo"}}',
                # Retry: same tool call, args adjusted
                '{"tool":"workspace_run","arguments":{"name":"foo","args":["--ok"]}}',
            ]
        )
        attempts = {"n": 0}

        async def runner(name: str, args: dict[str, Any]) -> dict[str, Any]:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {
                    "ok": False,
                    "error": "ModuleNotFoundError: no module named foo",
                }
            return {"ok": True}

        async def broadcast(payload: dict[str, Any]) -> None:
            return None

        mem = _FakeMemory()
        brain = CoderBrain(
            llm_chat=llm,
            tool_runner=runner,
            broadcast=broadcast,
            error_learning=ErrorLearning(mem),
            max_steps=2,
            max_retries_per_step=3,
        )
        task_id = await brain.submit(
            description="run something", kind="skill",
            tool_catalog="(tools)",
        )
        for _ in range(200):
            task = brain.get(task_id)
            if task and task.state in ("done", "failed"):
                break
            await asyncio.sleep(0.01)
        task = brain.get(task_id)
        assert task is not None
        assert task.state == "done", f"got {task.state}: {task.last_error}"
        assert task.steps[0].retries == 1     # zero-indexed; second attempt
        assert task.steps[0].completed is True
        # Error-learning saw the failed first attempt.
        assert len(mem.errors) >= 1
        assert "ModuleNotFoundError" in mem.errors[0]["text"]

    @pytest.mark.asyncio
    async def test_unparseable_plan_marks_failed(self) -> None:
        llm = _ScriptedLLM(["this is not json or a list"])

        async def runner(name: str, args: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("runner should not be called on a plan failure")

        events: list[dict[str, Any]] = []
        async def broadcast(payload: dict[str, Any]) -> None:
            events.append(payload)

        mem = _FakeMemory()
        brain = CoderBrain(
            llm_chat=llm, tool_runner=runner, broadcast=broadcast,
            error_learning=ErrorLearning(mem),
        )
        # Plan-parser is best-effort: it accepts any non-empty list of
        # strings, including prose. So a one-line free-text response IS
        # treated as a single-step plan. Use an empty string to force
        # the failure path.
        llm = _ScriptedLLM([""])
        brain = CoderBrain(
            llm_chat=llm, tool_runner=runner, broadcast=broadcast,
            error_learning=ErrorLearning(mem),
        )
        task_id = await brain.submit(description="x", kind="skill", tool_catalog="")
        for _ in range(200):
            task = brain.get(task_id)
            if task and task.state in ("failed", "done"):
                break
            await asyncio.sleep(0.01)
        task = brain.get(task_id)
        assert task is not None
        assert task.state == "failed"
        assert any(evt.get("type") == "coder_error" for evt in events)

    @pytest.mark.asyncio
    async def test_stop_cancels_running_task(self) -> None:
        # Slow tool: stalls so we can cancel mid-step.
        llm = _ScriptedLLM(
            [
                '["wait a while"]',
                '{"tool":"sleep_forever","arguments":{}}',
            ]
        )

        async def runner(name: str, args: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(5.0)
            return {"ok": True}

        async def broadcast(payload: dict[str, Any]) -> None:
            return None

        brain = CoderBrain(
            llm_chat=llm, tool_runner=runner, broadcast=broadcast,
            error_learning=ErrorLearning(None),
            max_retries_per_step=1,
        )
        task_id = await brain.submit(
            description="x", kind="skill", tool_catalog="",
        )
        # Let it reach the stall.
        await asyncio.sleep(0.05)
        ok = await brain.stop(task_id)
        assert ok
        task = brain.get(task_id)
        assert task is not None
        assert task.state == "cancelled"


# ─── CondaEnvManager (lightweight; only happy paths) ────────────────


class TestCondaEnvManager:
    @pytest.mark.asyncio
    async def test_create_venv_real(self, tmp_path: Path) -> None:
        # Real venv creation — fast, ships with CPython.
        proj = tmp_path / "proj"
        proj.mkdir()
        mgr = CondaEnvManager()
        info, run = await mgr.create_venv(proj)
        assert run.ok
        assert info.exists
        assert info.python.exists()
        # Re-run is idempotent (no-op).
        info2, run2 = await mgr.create_venv(proj)
        assert info2.exists
        assert "already exists" in run2.stdout

    def test_conda_available_check(self) -> None:
        mgr = CondaEnvManager()
        # We don't assert True/False — we only assert it's a bool.
        assert isinstance(mgr.conda_available(), bool)
