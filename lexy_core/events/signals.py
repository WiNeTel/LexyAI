"""
Lexy AI - LexySignals (thread-safe shared state).

Tiny RLock-protected dataclass tracking core runtime state. Plugins keep their
own state; this is for cross-cutting flags only (system_state, ai_thinking, …).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SystemState(str, Enum):
    """High-level lifecycle state of LexyApp."""

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class LexySignals:
    """
    Thread-safe shared state container. Use ``update()`` / ``get()`` /
    ``get_snapshot()`` from any thread.
    """

    # System
    system_state: SystemState = SystemState.STARTING
    terminate: bool = False
    debug_mode: bool = False

    # Audio / interaction
    user_speaking: bool = False
    ai_thinking: bool = False
    ai_speaking: bool = False

    # Context
    active_session_id: str = ""
    current_input: str = ""
    current_response: str = ""

    # Timing
    last_interaction: datetime = field(default_factory=datetime.now)

    # Internal — not exposed via update()
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def update(self, **kwargs: Any) -> None:
        """Thread-safe bulk update of public attributes."""
        with self._lock:
            for key, value in kwargs.items():
                if key.startswith("_"):
                    continue
                if hasattr(self, key):
                    setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe attribute lookup."""
        with self._lock:
            return getattr(self, key, default)

    def get_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the public state."""
        with self._lock:
            return {
                "system_state": self.system_state.value,
                "terminate": self.terminate,
                "debug_mode": self.debug_mode,
                "user_speaking": self.user_speaking,
                "ai_thinking": self.ai_thinking,
                "ai_speaking": self.ai_speaking,
                "active_session_id": self.active_session_id,
                "current_input": self.current_input,
                "current_response": self.current_response,
                "last_interaction": self.last_interaction.isoformat(),
            }

    def is_ready(self) -> bool:
        """True if startup completed and no shutdown was requested."""
        return self.system_state == SystemState.READY and not self.terminate

    def is_busy(self) -> bool:
        """True while the agent or voice pipeline is active."""
        return self.ai_thinking or self.ai_speaking
