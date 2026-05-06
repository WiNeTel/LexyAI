"""
Subprocess executor for Lexy's self-coding plugin.

Used by ``workspace_run`` (skills only — projects are never run by Lexy
per Mike's rule). Wraps :func:`asyncio.create_subprocess_exec` with:

* ``timeout`` — kills the process and reports ``killed_reason="timeout"``.
* Stdout/stderr capture with a hard byte limit so a runaway log doesn't
  fill RAM.
* Optional ``cwd`` and ``env`` (the plugin sets ``cwd`` to the project
  root and ``env_path`` to the project's venv-python).
* Cross-platform: prefers POSIX ``resource.setrlimit`` for hard limits,
  falls back to async polling on Windows.
* Returns a ``RunResult`` dataclass — never raises for runtime errors,
  only for unrecoverable spawn failures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Outcome of one subprocess invocation."""

    cmd: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    truncated: bool = False
    killed_reason: str = ""           # "timeout" / "spawn_error" / ""
    pid: int | None = None
    started_at: float = 0.0

    def to_public(self) -> dict[str, Any]:
        return {
            "cmd": list(self.cmd),
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": round(self.duration_s, 3),
            "truncated": self.truncated,
            "killed_reason": self.killed_reason,
            "pid": self.pid,
            "started_at": self.started_at,
        }

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.killed_reason

    @property
    def error_summary(self) -> str:
        """A one-paragraph summary suitable for memory_store(errors)."""
        if self.killed_reason == "timeout":
            return f"Timed out after {self.duration_s:.1f}s. Last stderr:\n{self.stderr[-400:]}"
        if self.killed_reason:
            return f"Killed: {self.killed_reason}. Stderr:\n{self.stderr[-400:]}"
        if self.returncode != 0:
            return (
                f"Exited with code {self.returncode}. Stderr tail:\n"
                f"{self.stderr[-600:]}"
            )
        return ""


# ─── Runner ──────────────────────────────────────────────────────────


class CodeRunner:
    """Spawn subprocesses with safety rails."""

    def __init__(
        self,
        *,
        default_timeout: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> None:
        self._default_timeout = float(default_timeout)
        self._max_output = max(1024, int(max_output_bytes))

    async def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin_data: bytes | None = None,
    ) -> RunResult:
        cmd_list = list(cmd)
        cwd_str = str(cwd)
        timeout_s = float(timeout if timeout is not None else self._default_timeout)
        merged_env = _merge_env(env)
        started_at = time.time()
        result = RunResult(
            cmd=cmd_list, cwd=cwd_str, returncode=-1,
            stdout="", stderr="", duration_s=0.0, started_at=started_at,
        )

        # ── Spawn ────────────────────────────────────────────────────
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                cwd=cwd_str,
                env=merged_env,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_apply_posix_limits if sys.platform != "win32" else None,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            result.killed_reason = "spawn_error"
            result.stderr = f"Failed to spawn: {exc}"
            result.duration_s = time.time() - started_at
            return result

        result.pid = proc.pid

        # ── Wait + capture ───────────────────────────────────────────
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            result.killed_reason = "timeout"
            await _kill_proc_async(proc)
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=2.0
                )
            except asyncio.TimeoutError:
                stdout, stderr = b"", b""
        except Exception as exc:  # noqa: BLE001
            result.killed_reason = f"runtime_error: {exc}"
            await _kill_proc_async(proc)
            stdout, stderr = b"", b""

        # ── Decode + truncate ────────────────────────────────────────
        stdout_truncated = len(stdout) > self._max_output
        stderr_truncated = len(stderr) > self._max_output
        out_text = _decode(stdout[: self._max_output] if stdout_truncated else stdout)
        err_text = _decode(stderr[: self._max_output] if stderr_truncated else stderr)
        if stdout_truncated:
            out_text += f"\n... [truncated at {self._max_output} bytes]"
        if stderr_truncated:
            err_text += f"\n... [truncated at {self._max_output} bytes]"

        result.returncode = int(proc.returncode if proc.returncode is not None else -1)
        result.stdout = out_text
        result.stderr = err_text
        result.truncated = stdout_truncated or stderr_truncated
        result.duration_s = time.time() - started_at
        return result


# ─── Helpers ────────────────────────────────────────────────────────


def _merge_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Start from os.environ so PATH+conda etc. work, then layer extras."""
    merged = dict(os.environ)
    if extra:
        for k, v in extra.items():
            merged[k] = "" if v is None else str(v)
    # Force unbuffered Python so streamed output is visible in real time.
    merged.setdefault("PYTHONUNBUFFERED", "1")
    return merged


def _decode(buf: bytes) -> str:
    """Permissive bytes-to-str. Mirrors text_parser.parse_text."""
    if not buf:
        return ""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return buf.decode(enc)
        except UnicodeDecodeError:
            continue
    return buf.decode("utf-8", errors="replace")


async def _kill_proc_async(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except Exception:  # noqa: BLE001
        pass


def _apply_posix_limits() -> None:
    """Best-effort hard limits. Only invoked on POSIX as preexec_fn.

    We cap CPU at 60s, RSS at 1 GB. These are *backups* — the wait_for
    timeout is the primary guard. Failing to set a limit (some sandboxes
    forbid it) is non-fatal: we fall back to the asyncio timeout alone.
    """
    try:
        import resource  # type: ignore[import-not-found]
        # 60s of CPU should be enough for a coding skill; longer means
        # something has gone wrong. Setting it as RLIMIT_CPU ensures the
        # kernel sends SIGXCPU even if asyncio.wait_for's timeout misses.
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        # 1 GB virtual memory.
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    except Exception:  # noqa: BLE001
        pass
