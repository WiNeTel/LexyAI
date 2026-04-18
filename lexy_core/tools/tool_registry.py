"""
Lexy AI - Tool Registry.

Central registry for tools that plugins expose to the LLM.
The system prompt receives the tool schemas; the LLM emits ``<tool_call>``
blocks which the ToolCaller parses and executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from lexy_core.utils.logging import get_logger

log = get_logger(module="tool_registry")

ToolHandler = Callable[..., Any | Awaitable[Any]]


@dataclass
class ToolDefinition:
    """One registered tool."""

    name: str
    handler: ToolHandler
    schema: dict[str, Any]
    description: str
    source: str  # plugin name or "core"

    def to_schema_dict(self) -> dict[str, Any]:
        """Schema dict for system-prompt injection."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema,
        }


@dataclass
class ToolResult:
    """Result of one tool execution."""

    success: bool
    data: Any = None
    error: str = ""

    def to_text(self) -> str:
        """Render the result for the LLM context."""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, dict):
            parts: list[str] = []
            for key, value in self.data.items():
                if isinstance(value, list):
                    parts.append(f"{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            parts.append(
                                "  - "
                                + ", ".join(f"{k}: {v}" for k, v in item.items())
                            )
                        else:
                            parts.append(f"  - {item}")
                else:
                    parts.append(f"{key}: {value}")
            return "\n".join(parts)
        return str(self.data) if self.data is not None else "OK"


class ToolRegistry:
    """Holds tool definitions and dispatches executions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ─── Registration ───────────────────────────────────────────────

    def register(
        self,
        name: str,
        handler: ToolHandler,
        schema: dict[str, Any],
        description: str = "",
        source: str = "core",
    ) -> None:
        """Register a new tool."""
        if name in self._tools:
            log.warning(
                "tool_registry.overwrite",
                tool=name,
                old_source=self._tools[name].source,
                new_source=source,
            )
        self._tools[name] = ToolDefinition(
            name=name,
            handler=handler,
            schema=schema,
            description=description,
            source=source,
        )
        log.info("tool_registry.registered", tool=name, source=source)

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            log.info("tool_registry.unregistered", tool=name)
            return True
        return False

    def unregister_all(self, source: str) -> int:
        to_remove = [
            name for name, tool in self._tools.items() if tool.source == source
        ]
        for name in to_remove:
            del self._tools[name]
        if to_remove:
            log.info("tool_registry.cleaned", source=source, count=len(to_remove))
        return len(to_remove)

    # ─── Lookup ─────────────────────────────────────────────────────

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_all_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema_dict() for tool in self._tools.values()]

    def has_tools(self) -> bool:
        return bool(self._tools)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    # ─── Execution ──────────────────────────────────────────────────

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")

        try:
            handler_result = tool.handler(**args)
            if hasattr(handler_result, "__await__"):
                handler_result = await handler_result  # type: ignore[assignment]
            return ToolResult(success=True, data=handler_result)
        except TypeError as exc:
            return ToolResult(
                success=False, error=f"Bad parameters for '{name}': {exc}"
            )
        except Exception as exc:  # noqa: BLE001 — exposed to LLM
            log.error("tool_registry.exec_error", tool=name, error=str(exc))
            return ToolResult(success=False, error=str(exc))
