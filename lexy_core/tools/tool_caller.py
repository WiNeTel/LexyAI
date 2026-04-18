"""
Lexy AI - ToolCaller.

Detects tool calls in LLM output, parses them (DirtyJSON fallback), executes
tools, and formats the results for the next turn.

Supported formats (in priority order)
-------------------------------------

1. **Lexy native** – what we ask for in the system prompt::

       <tool_call>
       {"name": "tool_name", "arguments": {"arg1": "value1"}}
       </tool_call>

2. **ChatML / Qwen** – the community standard::

       <|tool_call|>{"name": "...", "arguments": {...}}<|/tool_call|>

3. **Gemma 4 native** – Google's own function-calling channel::

       <|tool_call>{"name": "...", "arguments": {...}}

   Gemma 4 doesn't emit a closing tag, so we greedy-match a JSON object.

4. **Fenced tool_code block** – common with Gemma and Llama tool tuning::

       ```tool_code
       {"name": "...", "arguments": {...}}
       ```

5. **Fenced JSON block with a recognizable name/arguments shape**::

       ```json
       {"name": "...", "arguments": {...}}
       ```

6. **Bare JSON fallback** – any ``{"name": "...", "arguments": {...}}``
   object found in the text. Used when the model ignored the formatting
   instructions entirely.

All detected calls are validated against the ``ToolRegistry`` before being
returned; unknown tool names are dropped silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lexy_core.llm.dirty_json import parse_dirty_json
from lexy_core.tools.tool_registry import ToolRegistry
from lexy_core.utils.logging import get_logger

log = get_logger(module="tool_caller")

_TEMPLATE_PATH = Path("data/templates/tool_instructions.txt")

# ── Format patterns ─────────────────────────────────────────────────────────
# Each pattern captures the JSON payload in group 1 and the full match spans
# via match.span(0) so the caller can strip them from the final response.

# 1. <tool_call>...</tool_call>
_LEXY_TOOL_CALL = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# 2. <|tool_call|>...<|/tool_call|>  (ChatML / Qwen)
_CHATML_TOOL_CALL = re.compile(
    r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>",
    re.DOTALL,
)

# 3. <|tool_call>...  (Gemma 4 native — no closer; greedy JSON object)
#    We capture until we find a balanced-ish JSON object via the generic
#    _JSON_OBJECT_PATTERN below, applied on the slice after the opener.
_GEMMA_TOOL_OPENER = re.compile(r"<\|tool_call\|?>(?:call)?\s*", re.IGNORECASE)

# 4. ```tool_code ... ```
_TOOL_CODE_FENCE = re.compile(
    r"```(?:tool_code|tool)\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# 5. ```json ... ``` (only counted as a tool call if it has name+arguments)
_JSON_FENCE = re.compile(
    r"```(?:json)?\s*\n?(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)

# 6. Bare JSON object with a top-level "name" and "arguments" key.
#    Greedy-match a balanced object using a non-backtracking approach:
#    find `{"name"` start and scan until matching `}` with depth counter.
_NAME_ARGUMENTS_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{',
    re.DOTALL,
)

# Removal patterns — used by strip_tool_call() to scrub the assistant text
# before it's shown to the user. Order matters: more-specific first.
_STRIP_PATTERNS: list[re.Pattern[str]] = [
    _LEXY_TOOL_CALL,
    _CHATML_TOOL_CALL,
    _TOOL_CODE_FENCE,
    # Gemma-style openers + the object that follows them
    re.compile(r"<\|tool_call\|?>(?:call)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|tool_call\|?>", re.IGNORECASE),
    # Leftover special tokens Gemma sometimes emits around tool calls
    re.compile(r"<\|tool_code\|?>", re.IGNORECASE),
    re.compile(r"<end_of_turn>|<start_of_turn>\w*", re.IGNORECASE),
]
_TOOL_RESULT_PATTERN = re.compile(
    r"<tool_result>.*?</tool_result>",
    re.DOTALL,
)


@dataclass
class ToolCall:
    """A tool call detected in LLM output."""

    name: str
    arguments: dict[str, Any]
    raw_text: str = ""


# ── Helpers ─────────────────────────────────────────────────────────────────


def _extract_balanced_object(text: str, start: int) -> tuple[str, int] | None:
    """
    Extract a balanced ``{...}`` JSON object starting at ``text[start]``.

    Returns ``(object_text, end_index)`` or ``None`` if nothing balanced
    could be found. Strings are parsed aware of escape sequences so a brace
    inside a quoted value doesn't throw off the depth counter.
    """
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], idx + 1
    return None


def _parse_json_loose(raw: str) -> dict[str, Any] | None:
    """``json.loads`` → ``dirty_json`` fallback. Returns a dict or None."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = parse_dirty_json(raw)
    if not isinstance(data, dict):
        return None
    return data


# ── Public class ────────────────────────────────────────────────────────────


class ToolCaller:
    """Build tool prompts, detect calls, execute, and format results."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ─── Prompt building ────────────────────────────────────────────

    def build_tool_prompt(self) -> str:
        """Build the tool description block for the system prompt."""
        if not self._registry.has_tools():
            return ""

        schemas = self._registry.get_all_schemas()

        tool_lines: list[str] = []
        for schema in schemas:
            name = schema["name"]
            desc = schema.get("description", "")
            params = schema.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = params.get("required", []) or []

            tool_lines.append(f"### {name}")
            tool_lines.append(desc)
            if props:
                tool_lines.append("Parameters:")
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "string")
                    pdesc = pdef.get("description", "")
                    marker = "*" if pname in required else ""
                    tool_lines.append(f"  - {pname}{marker}: {ptype} – {pdesc}")
            tool_lines.append("")

        tool_list = "\n".join(tool_lines)

        if _TEMPLATE_PATH.exists():
            template = _TEMPLATE_PATH.read_text(encoding="utf-8").strip()
            return template.replace("{tool_list}", tool_list)

        return "\n".join(
            [
                "## Tools available to you",
                "",
                tool_list,
                "## How to call a tool",
                "",
                "When you need a tool, respond with EXACTLY this block and nothing else:",
                "",
                "<tool_call>",
                '{"name": "tool_name", "arguments": {"param1": "value1"}}',
                "</tool_call>",
                "",
                "Rules:",
                "- Output ONLY the <tool_call> block (no explanation before or after).",
                "- The JSON must be valid: double quotes, no trailing commas.",
                "- Wait for the <tool_result> block before answering the user.",
                "- Only call a tool when the user's request actually needs it.",
                "- After receiving the tool result, reply to the user in plain text.",
                "",
                'Example: for "weather in Hamburg" you would emit:',
                "<tool_call>",
                '{"name": "get_weather", "arguments": {"location": "Hamburg"}}',
                "</tool_call>",
            ]
        )

    # ─── Detection ──────────────────────────────────────────────────

    def detect_tool_call(self, text: str) -> ToolCall | None:
        """Return the first detected tool call, or None."""
        calls = self.detect_all(text)
        return calls[0] if calls else None

    def detect_all(self, text: str) -> list[ToolCall]:
        """
        Detect every tool call in the text across all supported formats.

        The same JSON object may be matched by multiple patterns (e.g. both
        the Gemma-native opener and the bare-JSON fallback match the same
        underlying object). We dedupe two ways:

        1. **Span overlap** — a candidate that fully contains a span already
           seen is a duplicate (bare-JSON fallback hit inside a Gemma opener).
        2. **(name, frozen arguments)** — identical calls are kept only once
           (a model may literally emit the same call twice in one turn).
        """
        calls: list[ToolCall] = []
        seen_spans: list[tuple[int, int]] = []
        seen_signatures: set[tuple[str, str]] = set()

        for name, args, start, end in self._iter_candidates(text):
            # Drop candidates whose span is fully inside a previous match.
            contained = any(s <= start and end <= e for s, e in seen_spans)
            if contained:
                continue

            if self._registry.get_tool(name) is None:
                log.warning("tool_caller.unknown_tool", tool=name)
                continue

            try:
                signature = (
                    name,
                    json.dumps(args, sort_keys=True, default=str),
                )
            except (TypeError, ValueError):
                signature = (name, repr(args))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            seen_spans.append((start, end))

            calls.append(
                ToolCall(
                    name=name,
                    arguments=args,
                    raw_text=text[start:end],
                )
            )

        return calls

    def _iter_candidates(
        self, text: str
    ) -> Iterable[tuple[str, dict[str, Any], int, int]]:
        """
        Yield ``(name, arguments, start, end)`` for every possible tool call
        found in ``text`` — validation happens in ``detect_all``.
        """
        # 1. Lexy native
        for match in _LEXY_TOOL_CALL.finditer(text):
            data = _parse_json_loose(match.group(1))
            if data and "name" in data:
                yield (
                    str(data["name"]),
                    data.get("arguments") or {},
                    match.start(),
                    match.end(),
                )

        # 2. ChatML / Qwen
        for match in _CHATML_TOOL_CALL.finditer(text):
            data = _parse_json_loose(match.group(1))
            if data and "name" in data:
                yield (
                    str(data["name"]),
                    data.get("arguments") or {},
                    match.start(),
                    match.end(),
                )

        # 3. Gemma 4 native: <|tool_call>... plus a balanced JSON object
        for opener in _GEMMA_TOOL_OPENER.finditer(text):
            obj_start = text.find("{", opener.end())
            if obj_start == -1 or obj_start - opener.end() > 20:
                continue
            extracted = _extract_balanced_object(text, obj_start)
            if extracted is None:
                continue
            obj_text, obj_end = extracted
            data = _parse_json_loose(obj_text)
            if data and "name" in data:
                yield (
                    str(data["name"]),
                    data.get("arguments") or {},
                    opener.start(),
                    obj_end,
                )

        # 4. ```tool_code ... ```
        for match in _TOOL_CODE_FENCE.finditer(text):
            data = _parse_json_loose(match.group(1))
            if data and "name" in data:
                yield (
                    str(data["name"]),
                    data.get("arguments") or {},
                    match.start(),
                    match.end(),
                )

        # 5. ```json ... ``` — only if the object has name+arguments shape
        for match in _JSON_FENCE.finditer(text):
            payload = match.group(1)
            if '"name"' not in payload or '"arguments"' not in payload:
                continue
            data = _parse_json_loose(payload)
            if data and "name" in data and "arguments" in data:
                yield (
                    str(data["name"]),
                    data.get("arguments") or {},
                    match.start(),
                    match.end(),
                )

        # 6. Bare JSON {"name":..., "arguments":...} fallback
        for marker in _NAME_ARGUMENTS_RE.finditer(text):
            extracted = _extract_balanced_object(text, marker.start())
            if extracted is None:
                continue
            obj_text, obj_end = extracted
            data = _parse_json_loose(obj_text)
            if data and "name" in data and "arguments" in data:
                yield (
                    str(data["name"]),
                    data.get("arguments") or {},
                    marker.start(),
                    obj_end,
                )

    # ─── Execution ──────────────────────────────────────────────────

    async def execute_and_format(self, call: ToolCall) -> str:
        """Execute the tool and wrap the result in <tool_result>."""
        log.info("tool_caller.executing", tool=call.name, args=call.arguments)
        result = await self._registry.execute(call.name, call.arguments)
        return f"<tool_result>\n{result.to_text()}\n</tool_result>"

    # ─── Helpers ────────────────────────────────────────────────────

    def strip_tool_call(self, text: str) -> str:
        """Remove every known tool-call wrapper from ``text``."""
        stripped = text
        for pattern in _STRIP_PATTERNS:
            stripped = pattern.sub("", stripped)
        # Also strip the bare-JSON fallback matches
        for marker in list(_NAME_ARGUMENTS_RE.finditer(stripped)):
            extracted = _extract_balanced_object(stripped, marker.start())
            if extracted is None:
                continue
            obj_text, _ = extracted
            stripped = stripped.replace(obj_text, "", 1)
        return stripped.strip()

    def strip_tool_result(self, text: str) -> str:
        return _TOOL_RESULT_PATTERN.sub("", text).strip()
