"""
Git auto-committer for self-coding projects.

Every project under ``workspace/`` becomes a small Git repo so Mike has
a free undo history. We use ``git`` as a subprocess (we don't need a
proper Git library — the operations are minimal) and serialise per-repo
operations behind an asyncio.Lock to avoid concurrent commit races.

Operations:

* :meth:`init` — ``git init``, set local user.name / user.email so commits
  have a recognisable author.
* :meth:`add_and_commit` — stage ``files`` (relative paths), commit with
  the supplied message. Returns the SHA. Skips silently if there's
  nothing to commit (saves a noisy error log).
* :meth:`log` — last N entries as dicts ``{sha, author, ts, subject}``.
* :meth:`diff` — unstaged + staged diff as text (truncated).
* :meth:`revert` — hard-reset to a SHA (used by ``workspace_git_revert``).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .code_runner import CodeRunner, RunResult


log = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    sha: str
    short_sha: str
    subject: str
    author: str
    timestamp: float

    def to_public(self) -> dict[str, object]:
        return {
            "sha": self.sha,
            "short_sha": self.short_sha,
            "subject": self.subject,
            "author": self.author,
            "timestamp": self.timestamp,
        }


class GitNotAvailable(RuntimeError):
    """Raised when ``git`` isn't on PATH and we can't proceed."""


class GitCommitter:
    """Subprocess-driven git wrapper for one or more repos."""

    def __init__(
        self,
        *,
        runner: CodeRunner | None = None,
        author_name: str = "Lexy",
        author_email: str = "lexy@local",
    ) -> None:
        self._runner = runner or CodeRunner(default_timeout=60.0)
        self._author_name = author_name
        self._author_email = author_email
        self._locks: dict[str, asyncio.Lock] = {}
        self._available: bool | None = None

    # ─── Availability ────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._available is None:
            self._available = shutil.which("git") is not None
        return self._available

    def _require_git(self) -> None:
        if not self.is_available():
            raise GitNotAvailable(
                "git not found on PATH. Install git or skip git operations."
            )

    def _lock_for(self, repo: Path) -> asyncio.Lock:
        key = str(Path(repo).resolve())
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # ─── Operations ──────────────────────────────────────────────────

    async def init(self, repo: Path) -> RunResult:
        """``git init`` if needed, plus set local author identity."""
        self._require_git()
        repo = Path(repo)
        repo.mkdir(parents=True, exist_ok=True)
        async with self._lock_for(repo):
            git_dir = repo / ".git"
            if not git_dir.exists():
                init_result = await self._runner.run(
                    ["git", "init", "-q"], cwd=str(repo),
                )
                if not init_result.ok:
                    return init_result
            await self._runner.run(
                ["git", "config", "user.name", self._author_name],
                cwd=str(repo),
            )
            await self._runner.run(
                ["git", "config", "user.email", self._author_email],
                cwd=str(repo),
            )
            # Initial commit so a fresh repo has HEAD — needed before
            # log / diff / reset can work without the empty-tree dance.
            head_check = await self._runner.run(
                ["git", "rev-parse", "--verify", "HEAD"], cwd=str(repo),
            )
            if not head_check.ok:
                # Stage everything currently in the dir and commit it as
                # the baseline. Empty repos still produce a valid HEAD
                # via the --allow-empty fallback.
                await self._runner.run(
                    ["git", "add", "-A"], cwd=str(repo),
                )
                await self._runner.run(
                    [
                        "git", "commit", "-q",
                        "--allow-empty",
                        "-m", "init: Lexy workspace baseline",
                    ],
                    cwd=str(repo),
                )
        return RunResult(
            cmd=["git", "init"], cwd=str(repo), returncode=0,
            stdout="ok", stderr="", duration_s=0.0, started_at=time.time(),
        )

    async def add_and_commit(
        self,
        repo: Path,
        *,
        files: Sequence[str],
        message: str,
    ) -> CommitInfo | None:
        """Stage ``files`` (relative to repo) and commit. Returns the SHA.

        Returns ``None`` if there was nothing to commit (clean working
        tree). Files are added one at a time so an LLM-supplied path
        with weird characters can't sneak ``-A`` semantics in.
        """
        self._require_git()
        repo = Path(repo)
        if not files:
            files = ["-A"]  # caller asked for "everything in this commit"
        async with self._lock_for(repo):
            for f in files:
                if f == "-A":
                    add_result = await self._runner.run(
                        ["git", "add", "-A"], cwd=str(repo),
                    )
                else:
                    add_result = await self._runner.run(
                        ["git", "add", "--", f], cwd=str(repo),
                    )
                if not add_result.ok:
                    log.warning(
                        "coder.git_add_failed file=%s err=%s",
                        f, add_result.stderr[:200],
                    )
            # Bail early if nothing was actually staged. ``diff --cached
            # --quiet`` returns 1 if there *are* changes — counter-intuitive.
            check = await self._runner.run(
                ["git", "diff", "--cached", "--quiet"], cwd=str(repo),
            )
            if check.returncode == 0:
                return None  # nothing staged
            commit_result = await self._runner.run(
                [
                    "git", "commit", "-q", "-m",
                    (message or "lexy: auto-commit").splitlines()[0][:200],
                ],
                cwd=str(repo),
            )
            if not commit_result.ok:
                log.warning(
                    "coder.git_commit_failed err=%s",
                    commit_result.stderr[:200],
                )
                return None
            sha_result = await self._runner.run(
                ["git", "rev-parse", "HEAD"], cwd=str(repo),
            )
            sha = sha_result.stdout.strip()
            return CommitInfo(
                sha=sha,
                short_sha=sha[:8],
                subject=(message or "lexy: auto-commit").splitlines()[0],
                author=self._author_name,
                timestamp=time.time(),
            )

    async def log(self, repo: Path, *, limit: int = 10) -> list[CommitInfo]:
        self._require_git()
        repo = Path(repo)
        if not (repo / ".git").exists():
            return []
        # Use ``%x1f`` (unit-separator) as field delimiter — git's tab/null
        # handling differs across Windows builds; ASCII control chars are safe.
        pretty = "%H%x1f%h%x1f%s%x1f%an%x1f%at"
        result = await self._runner.run(
            ["git", "log", f"--pretty=format:{pretty}", f"-n", str(int(limit))],
            cwd=str(repo),
        )
        out: list[CommitInfo] = []
        if not result.ok:
            return out
        for line in result.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) != 5:
                continue
            sha, short, subject, author, ts = parts
            try:
                ts_f = float(ts)
            except ValueError:
                ts_f = 0.0
            out.append(
                CommitInfo(
                    sha=sha, short_sha=short, subject=subject,
                    author=author, timestamp=ts_f,
                )
            )
        return out

    async def diff(self, repo: Path, *, max_chars: int = 10_000) -> str:
        self._require_git()
        repo = Path(repo)
        if not (repo / ".git").exists():
            return ""
        # Combine staged + unstaged so the user sees everything.
        result = await self._runner.run(
            ["git", "diff", "HEAD"], cwd=str(repo),
        )
        out = result.stdout if result.ok else ""
        if len(out) > max_chars:
            out = out[:max_chars] + f"\n... [diff truncated at {max_chars} chars]"
        return out

    async def revert(
        self,
        repo: Path,
        *,
        commit_ref: str,
    ) -> RunResult:
        """Hard-revert to ``commit_ref``. Caller MUST gate via Approval."""
        self._require_git()
        return await self._runner.run(
            ["git", "reset", "--hard", commit_ref],
            cwd=str(repo),
        )
