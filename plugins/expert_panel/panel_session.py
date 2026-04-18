"""
Lexy AI - Expert Panel Session.

State machine for a single panel discussion. Tracks topic, participating roles,
current phase (analysis -> discussion -> synthesis), round counter, and all
messages produced by the panel agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PanelMessage:
    """Single contribution from one panel agent."""

    role: str
    phase: str
    round_num: int
    content: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "phase": self.phase,
            "round": self.round_num,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass
class PanelSession:
    """
    Mutable state for one running expert panel.

    Lifecycle: running -> synthesizing -> done   (happy path)
               running -> cancelled               (user abort)
    """

    panel_id: str
    topic: str
    roles: list[str]
    brain: str
    rounds_planned: int
    status: str = "running"          # running | synthesizing | done | cancelled
    current_phase: str = "analysis"  # analysis | discussion | synthesis
    current_round: int = 0
    messages: list[PanelMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add_message(
        self,
        role: str,
        phase: str,
        round_num: int,
        content: str,
    ) -> PanelMessage:
        """Append a new message and return it."""
        msg = PanelMessage(
            role=role,
            phase=phase,
            round_num=round_num,
            content=content,
        )
        self.messages.append(msg)
        return msg

    def get_messages_for_round(self, round_num: int) -> list[PanelMessage]:
        """Return all messages from a specific round."""
        return [m for m in self.messages if m.round_num == round_num]

    def get_messages_by_role(self, role: str) -> list[PanelMessage]:
        """Return all messages from a specific role."""
        return [m for m in self.messages if m.role == role]

    def finish(self, status: str = "done") -> None:
        """Mark the panel as finished."""
        self.status = status
        self.finished_at = time.time()

    def to_status_dict(self) -> dict[str, Any]:
        """Compact status representation for tools and WS."""
        return {
            "panel_id": self.panel_id,
            "topic": self.topic,
            "status": self.status,
            "phase": self.current_phase,
            "roles": self.roles,
            "brain": self.brain,
            "rounds_planned": self.rounds_planned,
            "current_round": self.current_round,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }
