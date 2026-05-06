"""
Lexy AI — Skill folder generator (Phase 11 — agentskills.io compliant).

Replaces the pre-Phase-11 single-file template (``data/skills/<name>.py``
with a docstring header) by an *agentskills.io-shaped folder*:

```
<target_root>/<name>/
├── SKILL.md          # YAML frontmatter + markdown body
└── scripts/
    └── skill.py      # async def execute(api, **kwargs) -> dict
```

This module is consumed by:

* :class:`SkillWriterPlugin._tool_write_skill` — when the LLM proposes
  a new skill, we emit the folder before sending it off for AST
  validation and registry insertion.
* :class:`AutoAgent` — generates skills end-to-end without humans in
  the loop; same emit path.

The ``sanitize_skill_name`` helper still exists (some callers feed it
LLM-generated names that may contain spaces, capitals, underscores).
The output always passes :func:`skill_spec.validate_skill_name`.
"""

from __future__ import annotations

import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from .skill_spec import (
    SkillFrontmatter,
    render_skill_md,
    validate_skill_name,
)

# Default body for ``scripts/skill.py``. We emit a typed signature and
# a working `return` so the AST validator never trips on an empty
# function. Real skills overwrite the body via the ``code`` parameter.
_SKILL_PY_TEMPLATE = '''\
"""
Auto-generated entry point for the {name!r} skill.

Generated: {created}
Lexy plugin: skill_writer (Phase 11 / agentskills.io compliant).
"""
from __future__ import annotations
from typing import Any


async def execute(api: Any, **kwargs: Any) -> dict[str, Any]:
    """{description}

    Args:
{args_doc}
    """
{body}
'''

# Default Markdown body when the caller doesn't provide their own.
# Per spec we should keep this under 5000 tokens / 500 lines, so we
# stay deliberately terse — real skills should overwrite this from
# their LLM-generated documentation.
_DEFAULT_BODY_MD = """\
This skill was generated automatically by Lexy's `skill_writer` plugin.

Replace this body with step-by-step instructions for the agent that
will use the skill — the spec recommends sections like:

- **What it does**
- **When to use it**
- **Inputs** (kwargs the `execute()` function expects)
- **Examples**
- **Edge cases**

Keep it under ~500 lines; offload long reference material to
`references/REFERENCE.md` instead.
"""


def emit_skill_folder(
    *,
    name: str,
    description: str,
    code: str,
    target_root: Path | str,
    body_md: str | None = None,
    license: str | None = None,
    compatibility: str | None = None,
    metadata: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Create an agentskills.io-shaped folder on disk.

    Args:
        name: Skill name. Must already conform to the spec — call
            :func:`sanitize_skill_name` first if the source is
            user-or-LLM-supplied.
        description: 1-1024 chars. Spec mandates "what + when".
        code: Python body of ``async def execute(api, **kwargs)``.
            Indented at the column-0 boundary (we re-indent to 4
            spaces); whitespace around the block is trimmed.
        target_root: Parent directory under which ``<name>/`` is
            created. Created if missing.
        body_md: Markdown body of ``SKILL.md``. Defaults to a
            placeholder explaining how to replace it.
        license: Optional ``license:`` field for the frontmatter.
        compatibility: Optional ``compatibility:`` field.
        metadata: Optional ``metadata:`` map. Values are coerced to
            strings (per spec).
        overwrite: If ``True`` and the target folder exists, it's
            wiped before writing. Default ``False`` raises
            :class:`FileExistsError`.

    Returns:
        Path to the freshly-emitted skill folder.

    Raises:
        SkillSpecError: ``name``/``description``/``compatibility`` invalid.
        FileExistsError: target folder exists and ``overwrite=False``.
    """
    # Validate up front so callers get a clean error before any disk
    # writes happen.
    validate_skill_name(name)

    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    skill_folder = target_root / name

    if skill_folder.exists():
        if not overwrite:
            raise FileExistsError(
                f"skill folder already exists: {skill_folder}"
            )
        # Best-effort clean — caller asked for it.
        _rm_tree(skill_folder)

    skill_folder.mkdir(parents=True, exist_ok=False)

    # ── SKILL.md ────────────────────────────────────────────────
    fm = SkillFrontmatter(
        name=name,
        description=description,
        body=body_md or _DEFAULT_BODY_MD,
        license=license,
        compatibility=compatibility,
        metadata=dict(metadata) if metadata else {},
    )
    (skill_folder / "SKILL.md").write_text(
        render_skill_md(fm), encoding="utf-8"
    )

    # ── scripts/skill.py ────────────────────────────────────────
    scripts_dir = skill_folder / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=False)
    (scripts_dir / "skill.py").write_text(
        _build_skill_py(name=name, description=description, code=code),
        encoding="utf-8",
    )

    return skill_folder


def _build_skill_py(*, name: str, description: str, code: str) -> str:
    """Format a Python source file for ``scripts/skill.py``.

    Handles three quirks of LLM-generated code:
    * leading/trailing blank lines (stripped)
    * already-indented vs flush-left bodies (auto-indent to 4 spaces)
    * empty bodies (default to ``return {"status": "ok"}``)
    """
    code_lines = code.strip().splitlines()
    if code_lines:
        first = code_lines[0]
        if not first.startswith(("    ", "\t")):
            formatted_body = textwrap.indent("\n".join(code_lines), "    ")
        else:
            formatted_body = "\n".join(code_lines)
    else:
        formatted_body = '    return {"status": "ok"}'

    args_doc = "        **kwargs: Skill-specific arguments."
    created = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return _SKILL_PY_TEMPLATE.format(
        name=name,
        description=description,
        created=created,
        args_doc=args_doc,
        body=formatted_body,
    )


def _rm_tree(path: Path) -> None:
    """Pure-stdlib recursive delete — avoids importing shutil to keep
    the validator's allowlist tight.

    Skill code is forbidden from importing shutil; the *plugin* code
    has no such restriction, but keeping the surface small means we
    don't drag the pattern over to skill bodies via copy-paste.
    """
    for entry in sorted(path.rglob("*"), reverse=True):
        if entry.is_dir():
            entry.rmdir()
        else:
            entry.unlink()
    path.rmdir()


# ─── Name sanitisation ───────────────────────────────────────────────

def sanitize_skill_name(name: str) -> str:
    """Coerce a free-form name into a spec-valid skill name.

    Rules from agentskills.io: 1-64 chars, lowercase letters / digits
    / single hyphens, no leading/trailing hyphen. Existing callers
    sometimes pass spaces, underscores, capitals, or even numbers as
    the first character (which is fine — the spec only forbids
    leading hyphens, not leading digits).

    The output is guaranteed to pass
    :func:`skill_writer.skill_spec.validate_skill_name`.
    """
    # Lowercase + replace non-alphanumeric runs with a single hyphen.
    out = re.sub(r"[^a-z0-9]+", "-", (name or "").lower().strip())
    # Strip leading/trailing hyphens.
    out = out.strip("-")
    # Collapse any consecutive hyphens.
    out = re.sub(r"-{2,}", "-", out)
    # Truncate to 64 chars (spec limit).
    if len(out) > 64:
        out = out[:64].rstrip("-")
    if not out:
        out = "unnamed-skill"
    return out
