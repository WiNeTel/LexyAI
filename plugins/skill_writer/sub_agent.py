"""
Lexy AI - Lightweight sub-agent for quick one-shot tasks.

Re-exports AutoAgent and AgentManager for backward compatibility.
Sub-agents are just AutoAgents with lighter default system prompts.
"""

from __future__ import annotations

from .auto_agent import AgentManager, AutoAgent

__all__ = ["AutoAgent", "AgentManager"]
