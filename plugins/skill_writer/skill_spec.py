"""
Lexy AI — agentskills.io SKILL.md frontmatter parser + validator (Phase 11).

Implements the open Agent Skills standard from Anthropic / agentskills.io.
A skill is a *folder* containing at minimum a ``SKILL.md`` file with YAML
frontmatter; this module is the canonical place where we parse and validate
that frontmatter so it's reusable across the loader, the importer, and the
exporter.

Spec reference: https://agentskills.io/specification

Required fields:
* ``name`` — 1–64 chars, ``[a-z0-9-]+``, no leading/trailing hyphen,
  no consecutive ``--``, **must equal the parent directory name**.
* ``description`` — 1–1024 chars, "what + when to use".

Optional fields:
* ``license`` — license name or reference to a bundled file.
* ``compatibility`` — max 500 chars, environment requirements.
* ``metadata`` — arbitrary str→str mapping (``"version": "1.0"``, etc.).
* ``allowed-tools`` — space-separated list, experimental in spec.

The body (Markdown after the closing ``---``) is treated as opaque —
no format restrictions per spec, but we expose it as ``frontmatter.body``
for consumers that want to render or transform it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

# ─── Constants from the spec ────────────────────────────────────────

NAME_MAX_LEN = 64
DESCRIPTION_MAX_LEN = 1024
COMPATIBILITY_MAX_LEN = 500

# Per spec: 1-64 chars, lowercase alphanumeric + hyphens, no leading
# or trailing hyphen, no consecutive ``--``. The two-branch alternation
# handles the single-character case (which the main pattern's anchored
# tail rejects).
NAME_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9]|-(?!-))*[a-z0-9]$|^[a-z0-9]$"
)


# ─── Errors ─────────────────────────────────────────────────────────

class SkillSpecError(ValueError):
    """Base class for any agentskills.io spec violation."""


class SkillNameError(SkillSpecError):
    """Raised when ``name`` violates the spec or doesn't match the parent dir."""


class SkillDescriptionError(SkillSpecError):
    """Raised when ``description`` is missing, empty, or too long."""


class SkillFrontmatterError(SkillSpecError):
    """Raised when the YAML frontmatter itself is malformed or missing."""


# ─── Dataclass ──────────────────────────────────────────────────────

@dataclass
class SkillFrontmatter:
    """Parsed + validated SKILL.md frontmatter plus the markdown body.

    Use :func:`parse_skill_md` to build an instance from on-disk text;
    use :func:`render_skill_md` for the inverse — emitting a SKILL.md
    string from a dataclass instance (useful for the exporter and
    auto-generated skills).
    """

    name: str
    description: str
    body: str = ""
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None

    def to_public(self) -> dict[str, Any]:
        """JSON-friendly dict for REST/WS responses.

        Mirrors the field names the spec uses (so external clients can
        round-trip without surprises). ``allowed-tools`` is hyphen-cased
        because that's how the spec writes it.
        """
        out: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.license is not None:
            out["license"] = self.license
        if self.compatibility is not None:
            out["compatibility"] = self.compatibility
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        if self.allowed_tools is not None:
            out["allowed-tools"] = self.allowed_tools
        return out


# ─── Validation helpers ─────────────────────────────────────────────

def validate_skill_name(name: str, *, parent_dir_name: str | None = None) -> None:
    """Raise :class:`SkillNameError` if ``name`` violates the spec.

    ``parent_dir_name`` is checked when given — the spec mandates that
    ``frontmatter.name`` equals the folder's name on disk.
    """
    if not isinstance(name, str):
        raise SkillNameError(f"name must be a string, got {type(name).__name__}")
    if not name:
        raise SkillNameError("name must not be empty")
    if len(name) > NAME_MAX_LEN:
        raise SkillNameError(
            f"name too long: {len(name)} > {NAME_MAX_LEN} chars"
        )
    # Cheaper rejections first (regex compile is amortised though)
    if name.startswith("-") or name.endswith("-"):
        raise SkillNameError(
            f"name {name!r} must not start or end with a hyphen"
        )
    if "--" in name:
        raise SkillNameError(
            f"name {name!r} must not contain consecutive hyphens"
        )
    if not NAME_RE.match(name):
        raise SkillNameError(
            f"name {name!r} is not a valid agentskills.io name "
            f"(allowed: lowercase a-z, 0-9, single hyphens)"
        )
    if parent_dir_name is not None and name != parent_dir_name:
        raise SkillNameError(
            f"name {name!r} does not match parent directory "
            f"{parent_dir_name!r} (spec requires equality)"
        )


def validate_skill_description(description: str) -> None:
    """Raise :class:`SkillDescriptionError` if ``description`` violates the spec."""
    if not isinstance(description, str):
        raise SkillDescriptionError(
            f"description must be a string, got {type(description).__name__}"
        )
    if not description.strip():
        raise SkillDescriptionError("description must not be empty")
    if len(description) > DESCRIPTION_MAX_LEN:
        raise SkillDescriptionError(
            f"description too long: {len(description)} > "
            f"{DESCRIPTION_MAX_LEN} chars"
        )


def validate_skill_compatibility(compat: str | None) -> None:
    """Raise :class:`SkillSpecError` if ``compatibility`` exceeds the spec limit."""
    if compat is None:
        return
    if not isinstance(compat, str):
        raise SkillSpecError(
            f"compatibility must be a string, got {type(compat).__name__}"
        )
    if len(compat) > COMPATIBILITY_MAX_LEN:
        raise SkillSpecError(
            f"compatibility too long: {len(compat)} > "
            f"{COMPATIBILITY_MAX_LEN} chars"
        )


# ─── Parser ─────────────────────────────────────────────────────────

# Frontmatter delimiter. Exactly three dashes on their own line, per
# the convention every frontmatter parser (Jekyll, Hugo, MkDocs, the
# official spec) follows. We accept Windows + Unix line endings.
_FRONT_RE = re.compile(
    r"\A---\s*\r?\n(?P<yaml>.*?)\r?\n---\s*\r?\n?(?P<body>.*)",
    re.DOTALL,
)


def parse_skill_md(
    text: str,
    *,
    parent_dir_name: str | None = None,
) -> SkillFrontmatter:
    """Parse + validate a ``SKILL.md`` file's contents.

    Args:
        text: Full UTF-8 text of the SKILL.md (frontmatter + body).
        parent_dir_name: When set, enforces ``frontmatter.name ==
            parent_dir_name`` per the spec.

    Returns:
        A validated :class:`SkillFrontmatter` instance.

    Raises:
        SkillFrontmatterError: missing or malformed YAML frontmatter.
        SkillNameError: ``name`` field invalid or mismatched with parent.
        SkillDescriptionError: ``description`` field invalid.
        SkillSpecError: any other spec violation.
    """
    if not isinstance(text, str):
        raise SkillFrontmatterError(
            f"SKILL.md content must be a string, got {type(text).__name__}"
        )

    match = _FRONT_RE.match(text)
    if match is None:
        raise SkillFrontmatterError(
            "SKILL.md must begin with a YAML frontmatter block "
            "delimited by '---' lines"
        )

    yaml_block = match.group("yaml")
    body = match.group("body")

    try:
        data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        raise SkillFrontmatterError(f"YAML parse failed: {exc}") from exc

    if not isinstance(data, dict):
        raise SkillFrontmatterError(
            f"frontmatter must parse to a mapping, got {type(data).__name__}"
        )

    # Required fields
    name = data.get("name")
    description = data.get("description")
    if name is None:
        raise SkillNameError("frontmatter missing required field: 'name'")
    if description is None:
        raise SkillDescriptionError(
            "frontmatter missing required field: 'description'"
        )

    validate_skill_name(name, parent_dir_name=parent_dir_name)
    validate_skill_description(description)

    # Optional fields — coerce to expected types where harmless.
    license_val = data.get("license")
    if license_val is not None and not isinstance(license_val, str):
        license_val = str(license_val)

    compat_val = data.get("compatibility")
    if compat_val is not None and not isinstance(compat_val, str):
        compat_val = str(compat_val)
    validate_skill_compatibility(compat_val)

    raw_meta = data.get("metadata") or {}
    if not isinstance(raw_meta, dict):
        raise SkillSpecError(
            f"metadata must be a mapping, got {type(raw_meta).__name__}"
        )
    # Spec says str → str. Coerce non-string values so a YAML number
    # like ``version: 1.0`` doesn't blow up (very common in the wild).
    metadata: dict[str, str] = {
        str(k): str(v) for k, v in raw_meta.items()
    }

    # ``allowed-tools`` uses a hyphen in YAML — read both spellings.
    allowed_tools = data.get("allowed-tools")
    if allowed_tools is None:
        allowed_tools = data.get("allowed_tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        # Some authors yaml-list it; flatten to space-separated string
        # so the field stays the spec's wire format.
        if isinstance(allowed_tools, list):
            allowed_tools = " ".join(str(x) for x in allowed_tools)
        else:
            allowed_tools = str(allowed_tools)

    return SkillFrontmatter(
        name=name,
        description=description,
        body=body or "",
        license=license_val,
        compatibility=compat_val,
        metadata=metadata,
        allowed_tools=allowed_tools,
    )


# ─── Renderer (inverse of the parser) ───────────────────────────────

def render_skill_md(fm: SkillFrontmatter) -> str:
    """Emit a SKILL.md text from a :class:`SkillFrontmatter` instance.

    The output is structured so :func:`parse_skill_md` can round-trip
    it byte-for-byte (modulo YAML key ordering, which we control here
    by writing fields in spec order). Validation runs again so callers
    can't sneak invalid frontmatter through the renderer.
    """
    validate_skill_name(fm.name)
    validate_skill_description(fm.description)
    validate_skill_compatibility(fm.compatibility)

    fm_dict: dict[str, Any] = {
        "name": fm.name,
        "description": fm.description,
    }
    if fm.license is not None:
        fm_dict["license"] = fm.license
    if fm.compatibility is not None:
        fm_dict["compatibility"] = fm.compatibility
    if fm.metadata:
        fm_dict["metadata"] = dict(fm.metadata)
    if fm.allowed_tools is not None:
        fm_dict["allowed-tools"] = fm.allowed_tools

    yaml_text = yaml.safe_dump(
        fm_dict,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip()

    body = (fm.body or "").lstrip("\n")
    if body and not body.endswith("\n"):
        body = body + "\n"

    return f"---\n{yaml_text}\n---\n{body}" if body else f"---\n{yaml_text}\n---\n"
