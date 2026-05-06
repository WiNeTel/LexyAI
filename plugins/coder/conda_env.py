"""
Lightweight venv / conda manager.

Per Mike's wishes, every coding project (skill or full project) gets its
own isolated environment under ``<project>/.venv/``. Conda is supported
when the user explicitly names a conda environment — otherwise we use
``python -m venv`` because it's standard, fast, and ships with CPython.

The module is *not* thread-safe; callers serialise via the plugin's
file-lock if needed. Operations return :class:`EnvInfo` so downstream
tooling (CodeRunner) can resolve the python executable without
re-implementing the same path arithmetic.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .code_runner import CodeRunner, RunResult


log = logging.getLogger(__name__)


@dataclass
class EnvInfo:
    kind: str               # "venv" | "conda"
    name: str               # human label (project name for venv, env name for conda)
    location: Path          # filesystem path to the env root
    python: Path            # path to the env's python interpreter
    exists: bool

    def to_public(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "location": str(self.location),
            "python": str(self.python),
            "exists": self.exists,
        }


# ─── Manager ─────────────────────────────────────────────────────────


class CondaEnvManager:
    """Manage venv / conda environments for self-coding projects."""

    def __init__(
        self,
        *,
        runner: CodeRunner | None = None,
        host_python: str | None = None,
    ) -> None:
        # Re-use the project-wide CodeRunner so all subprocess invocations
        # share timeout/output-cap semantics.
        self._runner = runner or CodeRunner(default_timeout=120.0)
        # Path to the host's python executable. When None, we use the
        # one running this process — that always works for venv -m.
        self._host_python = host_python or sys.executable

    # ─── venv (default) ──────────────────────────────────────────────

    def venv_info(self, project_dir: Path) -> EnvInfo:
        """Return the EnvInfo for the conventional ``<project>/.venv/``."""
        loc = Path(project_dir) / ".venv"
        return EnvInfo(
            kind="venv",
            name=Path(project_dir).name,
            location=loc,
            python=_venv_python(loc),
            exists=loc.is_dir() and _venv_python(loc).exists(),
        )

    async def create_venv(
        self,
        project_dir: Path,
        *,
        python: str | None = None,
    ) -> tuple[EnvInfo, RunResult]:
        """Create ``<project>/.venv/`` if it doesn't exist yet."""
        info = self.venv_info(project_dir)
        if info.exists:
            return info, RunResult(
                cmd=["(noop)"], cwd=str(project_dir), returncode=0,
                stdout="venv already exists", stderr="",
                duration_s=0.0, started_at=0.0,
            )

        host = python or self._host_python
        result = await self._runner.run(
            [host, "-m", "venv", str(info.location)],
            cwd=str(project_dir),
            timeout=180.0,
        )
        info = self.venv_info(project_dir)  # re-stat after creation
        return info, result

    # ─── conda (opt-in) ──────────────────────────────────────────────

    @staticmethod
    def conda_available() -> bool:
        """True iff ``conda`` is in PATH. Cheap shutil.which check."""
        return shutil.which("conda") is not None

    async def create_conda(
        self,
        env_name: str,
        *,
        python_version: str = "3.11",
        packages: Sequence[str] = (),
    ) -> tuple[EnvInfo, RunResult]:
        """Create a named conda environment.

        Returns ``(info, run_result)``. ``info.exists`` will be False if
        the create call failed — inspect ``run_result.stderr`` for why.
        """
        if not self.conda_available():
            return _conda_unavailable_info(env_name), RunResult(
                cmd=["conda"], cwd=".", returncode=-1,
                stdout="", stderr="conda not in PATH",
                duration_s=0.0, started_at=0.0, killed_reason="spawn_error",
            )
        cmd = [
            "conda", "create", "-n", env_name, "-y",
            f"python={python_version}",
            *packages,
        ]
        result = await self._runner.run(cmd, cwd=".", timeout=600.0)
        info = await self.conda_info(env_name)
        return info, result

    async def conda_info(self, env_name: str) -> EnvInfo:
        """Resolve a conda env by name. Best-effort — returns ``exists=False`` if not found."""
        if not self.conda_available():
            return _conda_unavailable_info(env_name)
        # Use ``conda env list --json`` so we don't have to parse the
        # human-readable table format. Returns a dict with "envs": [paths].
        result = await self._runner.run(
            ["conda", "env", "list", "--json"], cwd=".", timeout=30.0,
        )
        if not result.ok:
            return _conda_unavailable_info(env_name)
        import json
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return _conda_unavailable_info(env_name)
        for path_str in payload.get("envs", []):
            loc = Path(path_str)
            if loc.name == env_name:
                python = (
                    loc / "python.exe" if (loc / "python.exe").exists()
                    else loc / "bin" / "python"
                )
                return EnvInfo(
                    kind="conda",
                    name=env_name,
                    location=loc,
                    python=python,
                    exists=python.exists(),
                )
        return _conda_unavailable_info(env_name)

    # ─── pip install (works for both kinds) ──────────────────────────

    async def pip_install(
        self,
        env: EnvInfo,
        packages: Sequence[str],
        *,
        upgrade: bool = False,
    ) -> RunResult:
        """Install ``packages`` into the given env's python via pip."""
        if not env.exists:
            return RunResult(
                cmd=["pip"], cwd=str(env.location), returncode=-1,
                stdout="", stderr=f"env not ready: {env.location}",
                duration_s=0.0, started_at=0.0, killed_reason="spawn_error",
            )
        cmd = [str(env.python), "-m", "pip", "install"]
        if upgrade:
            cmd.append("-U")
        cmd.extend(packages)
        return await self._runner.run(cmd, cwd=str(env.location.parent), timeout=300.0)


# ─── Helpers ────────────────────────────────────────────────────────


def _venv_python(venv_root: Path) -> Path:
    """Resolve the python interpreter inside a venv, cross-platform."""
    win = venv_root / "Scripts" / "python.exe"
    if win.exists():
        return win
    posix = venv_root / "bin" / "python"
    if posix.exists():
        return posix
    # Even when neither exists (yet), prefer the platform-native path so
    # callers who pre-build the path get a sensible value.
    return win if sys.platform == "win32" else posix


def _conda_unavailable_info(env_name: str) -> EnvInfo:
    """Sentinel EnvInfo for the "conda env is not there" case."""
    return EnvInfo(
        kind="conda",
        name=env_name,
        location=Path(""),
        python=Path(""),
        exists=False,
    )
