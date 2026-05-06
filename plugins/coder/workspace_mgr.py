"""
Workspace-Manager — the only place that resolves on-disk paths for
Lexy's self-coding plugin.

Single rule:

    Every path Lexy touches must resolve to somewhere under
    ``<workspace_root>/{skills,projects,extensions}/<name>/...``.

If a tool call would resolve outside that, we raise
:class:`WorkspacePathError` *before* opening any file. The validation is
deliberately verbose — getting this wrong has the highest blast radius
of anything in the plugin (a path-traversal bug would let an LLM-driven
write trample arbitrary files on Mike's machine).

Project kinds and their constraints:

* ``skill`` — small scripts in a ``.venv``, runnable via subprocess.
* ``project`` — Mike's own projects, **not** runnable by Lexy
  (Mike's explicit rule: "I test those myself").
* ``extension`` — Lexy modifying her own plugins, Phase 4. Same
  filesystem layout as a normal LexyAI plugin.

Operations are deliberately small and synchronous (filesystem I/O is
fast). Each operation returns a dataclass so callers don't deal with
raw paths.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROJECT_KINDS: tuple[str, ...] = ("skill", "project", "extension")

# Names that would be confusing or break things on Windows.
_RESERVED_NAMES: frozenset[str] = frozenset(
    {"con", "prn", "aux", "nul", "com1", "com2", "com3", "lpt1", "lpt2"}
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


class WorkspacePathError(ValueError):
    """Raised when a path argument escapes the workspace whitelist."""


class WorkspaceNotFoundError(FileNotFoundError):
    """Raised when a referenced project / file does not exist."""


# ─── Models ──────────────────────────────────────────────────────────


@dataclass
class ProjectInfo:
    """A workspace project — skill / project / extension."""

    name: str
    kind: str           # one of PROJECT_KINDS
    root: Path
    created_at: float = field(default_factory=time.time)
    runnable: bool = False  # True for skills only

    def to_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": str(self.root),
            "runnable": self.runnable,
            "created_at": self.created_at,
        }


@dataclass
class FileEntry:
    """A directory entry inside the workspace (file or sub-folder)."""

    name: str
    is_dir: bool
    size: int = 0
    modified_at: float = 0.0

    def to_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified_at": self.modified_at,
        }


# ─── WorkspaceManager ────────────────────────────────────────────────


class WorkspaceManager:
    """Path-safe wrapper around the workspace directory tree."""

    # Per-kind subfolder name. The keys are stable across versions; if
    # you ever rename one, write a migration first.
    _KIND_DIR: dict[str, str] = {
        "skill": "skills",
        "project": "projects",
        "extension": "extensions",
    }

    def __init__(self, root: Path | str) -> None:
        # Resolve once at construction so symlinks at the root are
        # locked in — relative paths handed in later are joined to this
        # canonical version.
        self._root = Path(root).resolve()
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root

    # ─── Init ─────────────────────────────────────────────────────────

    async def ensure_layout(self) -> None:
        """Make sure the three subdirs exist. Idempotent."""
        self._root.mkdir(parents=True, exist_ok=True)
        for sub in self._KIND_DIR.values():
            (self._root / sub).mkdir(parents=True, exist_ok=True)

    # ─── Project-level ────────────────────────────────────────────────

    def kind_dir(self, kind: str) -> Path:
        if kind not in PROJECT_KINDS:
            raise WorkspacePathError(f"unknown project kind: {kind!r}")
        return self._root / self._KIND_DIR[kind]

    def project_path(self, name: str, kind: str) -> Path:
        """Compute the canonical project root, validating the name."""
        self._validate_name(name)
        return self.kind_dir(kind) / name

    async def init_project(
        self,
        *,
        name: str,
        kind: str,
        description: str = "",
    ) -> ProjectInfo:
        """Create a fresh project subtree.

        Layout per kind:

        * skill:     ``skills/<name>/{SKILL.md, skill.py, tests/}``
        * project:   ``projects/<name>/{README.md, .gitignore}``
        * extension: ``extensions/<name>/{plugin.yaml, plugin.py}``  (placeholder)
        """
        async with self._lock:
            project = self.project_path(name, kind)
            if project.exists():
                raise WorkspacePathError(
                    f"project already exists: {kind}/{name}"
                )
            project.mkdir(parents=True, exist_ok=False)

            if kind == "skill":
                (project / "tests").mkdir(parents=True, exist_ok=True)
                (project / "SKILL.md").write_text(
                    _DEFAULT_SKILL_MD.format(
                        name=name, description=description or "TODO"
                    ),
                    encoding="utf-8",
                )
                (project / "skill.py").write_text(
                    _DEFAULT_SKILL_PY.format(name=name),
                    encoding="utf-8",
                )
            elif kind == "project":
                (project / "README.md").write_text(
                    f"# {name}\n\n{description}\n",
                    encoding="utf-8",
                )
                (project / ".gitignore").write_text(
                    _DEFAULT_GITIGNORE, encoding="utf-8",
                )
            elif kind == "extension":
                # Capitalise the name into a CamelCase class name so the
                # entry stub matches the BasePlugin subclass we generate.
                class_name = _camelcase(name) + "Plugin"
                (project / "plugin.yaml").write_text(
                    _DEFAULT_EXTENSION_YAML.format(
                        name=name, class_name=class_name,
                    ),
                    encoding="utf-8",
                )
                (project / "plugin.py").write_text(
                    _DEFAULT_EXTENSION_PY.format(class_name=class_name),
                    encoding="utf-8",
                )

        return ProjectInfo(
            name=name,
            kind=kind,
            root=project,
            runnable=(kind == "skill"),
        )

    async def list_projects(self) -> list[ProjectInfo]:
        out: list[ProjectInfo] = []
        for kind, sub in self._KIND_DIR.items():
            kind_root = self._root / sub
            if not kind_root.is_dir():
                continue
            for entry in sorted(kind_root.iterdir(), key=lambda p: p.name.lower()):
                if not entry.is_dir():
                    continue
                if entry.name.startswith("."):
                    continue
                try:
                    created_at = entry.stat().st_ctime
                except OSError:
                    created_at = 0.0
                out.append(
                    ProjectInfo(
                        name=entry.name,
                        kind=kind,
                        root=entry,
                        created_at=created_at,
                        runnable=(kind == "skill"),
                    )
                )
        return out

    def get_project(self, name: str, kind: str) -> ProjectInfo:
        path = self.project_path(name, kind)
        if not path.is_dir():
            raise WorkspaceNotFoundError(
                f"project not found: {kind}/{name}"
            )
        return ProjectInfo(
            name=name,
            kind=kind,
            root=path,
            runnable=(kind == "skill"),
        )

    # ─── File-level ──────────────────────────────────────────────────

    def resolve_inside(self, *, kind: str, name: str, rel_path: str) -> Path:
        """Resolve ``<root>/<kind>/<name>/<rel_path>`` and verify it stays inside."""
        project_root = self.project_path(name, kind).resolve()
        if not project_root.exists():
            raise WorkspaceNotFoundError(f"project missing: {kind}/{name}")
        # Strip leading slashes / drive prefixes from the caller-supplied
        # path. We don't trust it.
        cleaned = (rel_path or "").lstrip("/\\").strip()
        if not cleaned or cleaned in (".", "./"):
            return project_root
        candidate = (project_root / cleaned).resolve()
        # Use os.path.commonpath so symlink-following is enforced via
        # resolve(); on Windows this also normalises slashes.
        try:
            common = Path(os.path.commonpath([candidate, project_root]))
        except ValueError:
            raise WorkspacePathError(
                f"path escapes project root: {rel_path!r}"
            )
        if common != project_root:
            raise WorkspacePathError(
                f"path escapes project root: {rel_path!r}"
            )
        return candidate

    def list_files(
        self,
        *,
        kind: str,
        name: str,
        rel_path: str = "",
    ) -> list[FileEntry]:
        path = self.resolve_inside(kind=kind, name=name, rel_path=rel_path)
        if not path.is_dir():
            raise WorkspaceNotFoundError(f"not a directory: {rel_path}")
        out: list[FileEntry] = []
        for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                stat = entry.stat()
            except OSError:
                continue
            out.append(
                FileEntry(
                    name=entry.name,
                    is_dir=entry.is_dir(),
                    size=int(stat.st_size) if entry.is_file() else 0,
                    modified_at=float(stat.st_mtime),
                )
            )
        return out

    def read_file(
        self,
        *,
        kind: str,
        name: str,
        rel_path: str,
        max_bytes: int = 1_048_576,
    ) -> str:
        path = self.resolve_inside(kind=kind, name=name, rel_path=rel_path)
        if not path.is_file():
            raise WorkspaceNotFoundError(f"not a file: {rel_path}")
        if path.stat().st_size > max_bytes:
            raise WorkspacePathError(
                f"file too large for read ({path.stat().st_size} > {max_bytes} bytes): "
                f"{rel_path}"
            )
        # Decode permissively. The same pattern Lexy uses everywhere else.
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")

    async def write_file(
        self,
        *,
        kind: str,
        name: str,
        rel_path: str,
        content: str,
        create_dirs: bool = True,
    ) -> Path:
        path = self.resolve_inside(kind=kind, name=name, rel_path=rel_path)
        if path.exists() and path.is_dir():
            raise WorkspacePathError(f"path is a directory: {rel_path}")
        async with self._lock:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return path

    async def delete_file(
        self,
        *,
        kind: str,
        name: str,
        rel_path: str,
    ) -> bool:
        path = self.resolve_inside(kind=kind, name=name, rel_path=rel_path)
        if not path.exists():
            return False
        async with self._lock:
            if path.is_file():
                path.unlink()
            else:
                # Refuse to nuke a directory recursively without an
                # explicit second tool call. Matches the Approval-Gate
                # contract: HIGH-risk operations should not chain
                # implicitly.
                raise WorkspacePathError(
                    f"refusing to recursively delete a directory: {rel_path}. "
                    "Delete files individually."
                )
        return True

    # ─── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not _NAME_RE.match(name):
            raise WorkspacePathError(
                f"invalid project name {name!r}: must match {_NAME_RE.pattern}"
            )
        if name.lower() in _RESERVED_NAMES:
            raise WorkspacePathError(
                f"project name reserved by the operating system: {name!r}"
            )


# ─── Templates (kept at module level, simple to swap) ────────────────


_DEFAULT_SKILL_MD = """\
---
name: {name}
description: {description}
---

# {name}

Beschreibe hier, was dieser Skill tut, welche Argumente er nimmt, und was er
zurückgibt.

## Run

```
python skill.py
```
"""


_DEFAULT_SKILL_PY = '''\
"""Skill entry point — runnable via ``python skill.py`` or workspace_run."""


def main() -> int:
    print("Hello from skill {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


_DEFAULT_GITIGNORE = """\
# venvs / build artefacts
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
.ruff_cache/

# editor cruft
.vscode/
.idea/
*.swp
.DS_Store
"""


_DEFAULT_EXTENSION_YAML = """\
name: {name}
version: 0.1.0
description: "Lexy-extension stub generated by the coder plugin. Edit and reload."
entry: plugin.{class_name}
requires: []
optional: []
capabilities: [tool]
"""


_DEFAULT_EXTENSION_PY = '''\
"""Auto-generated extension stub. Edit and reload via the plugin loader."""

from typing import Any

from lexy_core.plugin_system import BasePlugin


class {class_name}(BasePlugin):
    async def on_load(self) -> None:
        pass

    async def on_enable(self) -> None:
        pass

    async def on_disable(self) -> None:
        pass
'''


def _camelcase(name: str) -> str:
    """Convert ``foo_bar-baz`` → ``FooBarBaz``."""
    parts = re.split(r"[-_]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)
