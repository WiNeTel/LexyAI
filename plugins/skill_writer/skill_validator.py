"""
Lexy AI - Skill Validator.

Validates skill source code for safety before execution.
Performs static analysis using the ``ast`` module to block:

* Forbidden imports (``os``, ``subprocess``, ``sys``, etc.)
* Forbidden calls (``exec``, ``eval``, ``compile``, ``open``, etc.)
* Missing ``execute()`` entry point
* Syntax errors
* Oversized files

Phase 11 (agentskills.io) note: skills now live in folders rather than
flat ``.py`` files. The validator API is unchanged — callers feed it
the source of ``<folder>/scripts/skill.py`` (or the bytes thereof).
A new convenience method ``validate_folder()`` walks every ``.py`` in
``scripts/`` so importers can flag broken skills before they hit disk.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="skill_validator")

# Funktionsaufrufe die in Skills nicht erlaubt sind
FORBIDDEN_CALLS: frozenset[str] = frozenset({
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "breakpoint",
    "exit",
    "quit",
})

# Module die nicht importiert werden duerfen
FORBIDDEN_MODULES: frozenset[str] = frozenset({
    "os",
    "subprocess",
    "shutil",
    "sys",
    "importlib",
    "ctypes",
    "socket",
    "http",
    "ftplib",
    "smtplib",
    "multiprocessing",
    "threading",
    "signal",
    "pickle",
    "shelve",
    "tempfile",
    "webbrowser",
    "code",
    "codeop",
    "compileall",
    "py_compile",
})


class SkillValidator:
    """
    Validates skill source code before it can be registered or executed.

    Uses static AST analysis to detect unsafe patterns without executing
    the code. The allowed_imports whitelist controls which modules skills
    can import.
    """

    def __init__(
        self,
        allowed_imports: list[str],
        max_size_bytes: int = 10000,
    ) -> None:
        self._allowed_imports: set[str] = set(allowed_imports)
        self._max_size_bytes = max_size_bytes

    def validate(
        self, source: str, *, expect_execute: bool = True
    ) -> tuple[bool, str]:
        """
        Validate skill source code.

        Performs these checks in order:
        1. File size limit
        2. Syntax check (compilable Python)
        3. AST walk: forbidden imports
        4. AST walk: forbidden function calls
        5. ``async def execute(api, **kwargs)`` exists  *(only when
           ``expect_execute=True`` — helper modules in ``scripts/``
           don't need an execute() signature)*

        Args:
            source: Complete Python source code of the skill.
            expect_execute: Enforce the ``async def execute()`` entry-
                point signature. Default ``True`` (the primary script
                must have it). Set to ``False`` for helper modules
                that live alongside the primary in ``scripts/``.

        Returns:
            Tuple of ``(is_valid, error_message)``.
            On success: ``(True, "")``.
            On failure: ``(False, "human-readable error description")``.
        """
        # 1. Groessencheck
        source_bytes = len(source.encode("utf-8"))
        if source_bytes > self._max_size_bytes:
            return (
                False,
                f"Skill too large: {source_bytes} bytes "
                f"(max {self._max_size_bytes})",
            )

        # 2. Syntax-Check
        try:
            tree = compile(source, "skill.py", "exec", ast.PyCF_ONLY_AST)
        except SyntaxError as exc:
            line_info = f" (line {exc.lineno})" if exc.lineno else ""
            return False, f"Syntax error{line_info}: {exc.msg}"

        # 3. Import-Check
        import_err = self._check_imports(tree)
        if import_err is not None:
            return False, import_err

        # 4. Aufruf-Check
        call_err = self._check_calls(tree)
        if call_err is not None:
            return False, call_err

        # 5. execute()-Funktion vorhanden (only for primary script)
        if expect_execute:
            exec_err = self._check_execute_function(tree)
            if exec_err is not None:
                return False, exec_err

        log.debug("skill_validator.passed", size_bytes=source_bytes)
        return True, ""

    def _check_imports(self, tree: ast.AST) -> str | None:
        """
        Check all Import and ImportFrom nodes against the allowlist.

        Returns an error string if a forbidden import is found, else None.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    if module_root in FORBIDDEN_MODULES:
                        return (
                            f"Forbidden import: '{alias.name}' "
                            f"(module '{module_root}' is blocked)"
                        )
                    if module_root not in self._allowed_imports:
                        return (
                            f"Import not allowed: '{alias.name}' "
                            f"(module '{module_root}' not in allowlist)"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    return "Relative imports are not allowed in skills"
                module_root = node.module.split(".")[0]
                if module_root in FORBIDDEN_MODULES:
                    return (
                        f"Forbidden import: 'from {node.module}' "
                        f"(module '{module_root}' is blocked)"
                    )
                if module_root not in self._allowed_imports:
                    return (
                        f"Import not allowed: 'from {node.module}' "
                        f"(module '{module_root}' not in allowlist)"
                    )

        return None

    def _check_calls(self, tree: ast.AST) -> str | None:
        """
        Check for forbidden function calls in the AST.

        Catches both bare calls (``eval(x)``) and attribute calls
        where the attribute name is forbidden (``builtins.eval(x)``).

        Returns an error string if a forbidden call is found, else None.
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func

            # Einfacher Name: eval(), exec(), open()...
            if isinstance(func, ast.Name):
                if func.id in FORBIDDEN_CALLS:
                    return (
                        f"Forbidden call: '{func.id}()' "
                        f"is not allowed in skills"
                    )

            # Attribut-Zugriff: builtins.eval(), obj.__import__()...
            elif isinstance(func, ast.Attribute):
                if func.attr in FORBIDDEN_CALLS:
                    return (
                        f"Forbidden call: '.{func.attr}()' "
                        f"is not allowed in skills"
                    )

            # Subscript-basierte Aufrufe (z.B. getattr-Tricks) pruefen
            # wir konservativ: dunder-Methoden als Attribute blockieren
            if isinstance(func, ast.Attribute) and func.attr.startswith("__"):
                if func.attr not in ("__init__", "__str__", "__repr__"):
                    return (
                        f"Forbidden dunder call: '.{func.attr}()' "
                        f"is not allowed in skills"
                    )

        return None

    def _check_execute_function(self, tree: ast.AST) -> str | None:
        """
        Ensure ``async def execute(api, **kwargs)`` exists at module level.

        The function must be:
        - An ``async def`` (AsyncFunctionDef)
        - Named ``execute``
        - At the top level of the module

        Returns an error string if validation fails, else None.
        """
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute":
                # Pruefen ob mindestens ein Parameter (api) vorhanden ist
                args = node.args
                total_args = len(args.posonlyargs) + len(args.args)
                if total_args < 1:
                    return (
                        "execute() must accept at least one positional "
                        "argument (api)"
                    )
                return None

            # Auch synchrone execute() erkennen und hilfreiche Meldung geben
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                return (
                    "execute() must be async (use 'async def execute(...)' "
                    "instead of 'def execute(...)')"
                )

        return "Skill must define 'async def execute(api, **kwargs)' at module level"

    # ─── Phase 11: folder-level validation ──────────────────────────

    def validate_folder(
        self,
        folder: Path,
        *,
        primary_script: str | None = None,
    ) -> tuple[bool, str]:
        """Validate every ``.py`` in ``<folder>/scripts/``.

        The primary script (default ``scripts/skill.py`` per Lexy
        convention) MUST exist and pass full validation. Other Python
        files in ``scripts/`` are validated too — a broken helper
        would crash the skill at runtime, so we want to catch it at
        import time.

        Args:
            folder: Skill folder (the one containing ``SKILL.md``).
            primary_script: Relative path of the primary script.
                When ``None`` we look for ``scripts/skill.py``; the
                check passes if the skill bundles no scripts at all
                (docs-only skills are valid per spec).

        Returns:
            ``(True, "")`` if every script passes, otherwise
            ``(False, "<file>: <reason>")``.
        """
        scripts_dir = folder / "scripts"
        if not scripts_dir.is_dir():
            # Docs-only skill — fine per spec, no scripts to check.
            return True, ""

        py_files = sorted(p for p in scripts_dir.rglob("*.py") if p.is_file())
        if not py_files:
            return True, ""

        # If the caller specified a primary script (or default), it
        # must exist and be the first thing we check — gives the
        # cleanest error message when it's missing.
        primary = primary_script or "scripts/skill.py"
        primary_abs = folder / primary
        if not primary_abs.is_file():
            return False, f"primary script not found: {primary}"

        # Validate the primary first with the full check (execute
        # signature included), then helpers without that requirement.
        primary_resolved = primary_abs.resolve()
        for py_path in py_files:
            try:
                source = py_path.read_text(encoding="utf-8")
            except OSError as exc:
                return False, f"{py_path.name}: read failed: {exc}"

            is_primary = py_path.resolve() == primary_resolved
            ok, err = self.validate(source, expect_execute=is_primary)
            if not ok:
                rel = py_path.relative_to(folder).as_posix()
                return False, f"{rel}: {err}"

        return True, ""
