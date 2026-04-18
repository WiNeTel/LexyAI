"""
Lexy AI - DirtyJSON Parser.

Tolerant JSON parser used as a fallback when LLM output is not strictly valid
JSON. Tries (in order):

1. ``json.loads`` (strict).
2. ``dirtyjson.loads`` if installed.
3. A small set of heuristic fixups (trailing commas, single quotes, ``True`` →
   ``true``, etc.) followed by another ``json.loads``.

Returns ``None`` if every strategy fails. Never raises.
"""

from __future__ import annotations

import json
import re
from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="dirty_json")

try:
    import dirtyjson  # type: ignore[import-untyped]
    _HAS_DIRTYJSON = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_DIRTYJSON = False


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding ```json ... ``` fence if present."""
    fenced = re.match(
        r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$",
        text,
        re.DOTALL,
    )
    return fenced.group(1) if fenced else text


def _heuristic_fix(text: str) -> str:
    """Apply small fixups commonly seen in LLM output."""
    fixed = _strip_code_fence(text).strip()

    # True / False / None -> true / false / null
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)

    # Trailing commas before } or ]
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

    # Replace single-quoted strings with double quotes when safe
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')

    return fixed


def parse_dirty_json(text: str) -> Any | None:
    """
    Try to parse possibly broken JSON. Returns the parsed value or ``None``.

    The function never raises so callers can do ``data = parse_dirty_json(s)``.
    """
    if not text or not text.strip():
        return None

    candidate = _strip_code_fence(text).strip()

    # Strict
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # dirtyjson library
    if _HAS_DIRTYJSON:
        try:
            return dirtyjson.loads(candidate)  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            pass

    # Heuristic fixups
    try:
        return json.loads(_heuristic_fix(candidate))
    except json.JSONDecodeError as exc:
        log.debug("dirty_json.failed", error=str(exc), snippet=candidate[:120])
        return None
