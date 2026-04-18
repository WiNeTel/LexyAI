"""
Lexy AI - Skill Template & Header Parsing.

Defines the standard file format for skills stored in ``data/skills/``.
Each skill file contains a structured docstring header with metadata,
followed by an ``async def execute(api, **kwargs)`` entry point.
"""

from __future__ import annotations

import re
import textwrap
from datetime import datetime, timezone
from typing import Any

SKILL_HEADER = '''\
"""
Skill: {name}
Description: {description}
Author: {author}
Version: {version}
Created: {created}
Tags: {tags}
"""
from __future__ import annotations
from typing import Any


async def execute(api: Any, **kwargs: Any) -> dict[str, Any]:
    """
    {description}

    Args:
{args_doc}
    """
{body}
'''

# Regex fuer Header-Felder im Docstring
_HEADER_RE = re.compile(
    r"^\s*(Skill|Description|Author|Version|Created|Tags)\s*:\s*(.+)$",
    re.MULTILINE,
)


def parse_skill_header(source: str) -> dict[str, str]:
    """
    Parse the docstring header of a skill file.

    Extracts Skill, Description, Author, Version, Created, Tags fields
    from the opening triple-quoted docstring.

    Returns:
        Dictionary mapping lowercase field names to their string values.
        Missing fields get empty string defaults.
    """
    result: dict[str, str] = {
        "skill": "",
        "description": "",
        "author": "",
        "version": "",
        "created": "",
        "tags": "",
    }

    # Extrahiere nur den Docstring-Block (erste """-Paar)
    doc_start = source.find('"""')
    if doc_start == -1:
        return result
    doc_end = source.find('"""', doc_start + 3)
    if doc_end == -1:
        return result

    docstring = source[doc_start + 3 : doc_end]

    for match in _HEADER_RE.finditer(docstring):
        field_name = match.group(1).lower()
        field_value = match.group(2).strip()
        if field_name in result:
            result[field_name] = field_value

    return result


def generate_skill_file(
    name: str,
    description: str,
    code: str,
    author: str = "lexy_auto",
    version: str = "1.0",
    tags: list[str] | None = None,
) -> str:
    """
    Generate a complete skill file from components.

    Args:
        name:        Skill name (snake_case preferred).
        description: One-line description of what the skill does.
        code:        The body of the ``execute()`` function (indented 4 spaces).
        author:      Author tag (default: ``lexy_auto``).
        version:     Version string (default: ``1.0``).
        tags:        Optional list of tag strings.

    Returns:
        Complete Python source code for the skill file.
    """
    tags_str = ", ".join(tags) if tags else "general"
    created = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Sicherstellen, dass der Code-Body korrekt eingerueckt ist
    code_lines = code.strip().splitlines()
    formatted_body = ""
    if code_lines:
        # Pruefen ob der Code bereits eingerueckt ist
        first_line = code_lines[0]
        if not first_line.startswith("    "):
            # Nicht eingerueckt: 4 spaces hinzufuegen
            formatted_body = textwrap.indent(
                "\n".join(code_lines), "    "
            )
        else:
            formatted_body = "\n".join(code_lines)
    else:
        formatted_body = '    return {"status": "ok"}'

    # Args-Dokumentation generieren (Basisdoku)
    args_doc = "        **kwargs: Skill-specific arguments."

    return SKILL_HEADER.format(
        name=name,
        description=description,
        author=author,
        version=version,
        created=created,
        tags=tags_str,
        args_doc=args_doc,
        body=formatted_body,
    )


def sanitize_skill_name(name: str) -> str:
    """
    Sanitize a skill name to a valid Python identifier.

    Converts to lowercase, replaces non-alphanumeric chars with underscores,
    strips leading digits and trailing underscores.
    """
    # Kleinbuchstaben, nur alphanumerisch und Unterstriche
    sanitized = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    # Fuehrende Ziffern entfernen
    sanitized = re.sub(r"^[0-9]+", "", sanitized)
    # Doppelte Unterstriche zusammenfassen
    sanitized = re.sub(r"_+", "_", sanitized)
    # Trailing Unterstriche entfernen
    sanitized = sanitized.strip("_")

    if not sanitized:
        sanitized = "unnamed_skill"

    return sanitized
