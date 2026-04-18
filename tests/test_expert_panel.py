"""Tests for the Expert Panel plugin internals.

Covers:
* ROLE_PROMPTS / ROLE_COLORS / ROLE_LABELS contain all 5 roles
* PanelSession and PanelMessage dataclasses
* ConvergenceDetector with mocked LLM
* PanelSynthesizer with mocked LLM
* Panel message tracking and filtering
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.expert_panel.roles import ROLE_PROMPTS, ROLE_COLORS, ROLE_LABELS
from plugins.expert_panel.panel_session import PanelMessage, PanelSession
from plugins.expert_panel.convergence import ConvergenceDetector
from plugins.expert_panel.synthesizer import PanelSynthesizer


# ─── Roles ────────────────────────────────────────────────────────────────


EXPECTED_ROLES = {"analyst", "critic", "creative", "pragmatist", "synthesizer"}


class TestRoles:
    def test_all_five_roles_defined(self) -> None:
        assert set(ROLE_PROMPTS.keys()) == EXPECTED_ROLES

    def test_all_roles_have_colors(self) -> None:
        for role in ROLE_PROMPTS:
            assert role in ROLE_COLORS, f"Missing color for role '{role}'"
            assert ROLE_COLORS[role].startswith("#")

    def test_all_roles_have_labels(self) -> None:
        for role in ROLE_PROMPTS:
            assert role in ROLE_LABELS, f"Missing label for role '{role}'"
            assert len(ROLE_LABELS[role]) > 0

    def test_prompts_are_nonempty_strings(self) -> None:
        for role, prompt in ROLE_PROMPTS.items():
            assert isinstance(prompt, str)
            assert len(prompt) > 50, f"Prompt for '{role}' too short"

    def test_colors_are_valid_hex(self) -> None:
        for role, color in ROLE_COLORS.items():
            assert color.startswith("#"), f"{role} color missing '#'"
            assert len(color) == 7, f"{role} color not 7 chars: {color}"


# ─── PanelMessage ─────────────────────────────────────────────────────────


class TestPanelMessage:
    def test_to_dict(self) -> None:
        msg = PanelMessage(
            role="analyst",
            phase="analysis",
            round_num=1,
            content="My analysis...",
            created_at=1000.0,
        )
        d = msg.to_dict()
        assert d["role"] == "analyst"
        assert d["phase"] == "analysis"
        assert d["round"] == 1
        assert d["content"] == "My analysis..."
        assert d["created_at"] == 1000.0

    def test_default_created_at(self) -> None:
        msg = PanelMessage(role="critic", phase="discussion", round_num=2, content="x")
        assert msg.created_at > 0


# ─── PanelSession ─────────────────────────────────────────────────────────


class TestPanelSession:
    def test_initial_state(self) -> None:
        session = PanelSession(
            panel_id="p1",
            topic="Test topic",
            roles=["analyst", "critic"],
            brain="e4b",
            rounds_planned=3,
        )
        assert session.status == "running"
        assert session.current_phase == "analysis"
        assert session.current_round == 0
        assert session.messages == []
        assert session.finished_at is None

    def test_add_message(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="t", roles=["analyst"], brain="e4b", rounds_planned=2
        )
        msg = session.add_message("analyst", "analysis", 1, "My point.")
        assert isinstance(msg, PanelMessage)
        assert len(session.messages) == 1
        assert session.messages[0].content == "My point."

    def test_get_messages_for_round(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="t", roles=["analyst", "critic"],
            brain="e4b", rounds_planned=3,
        )
        session.add_message("analyst", "analysis", 1, "Round 1 analysis")
        session.add_message("critic", "analysis", 1, "Round 1 critique")
        session.add_message("analyst", "discussion", 2, "Round 2 analysis")

        round1 = session.get_messages_for_round(1)
        assert len(round1) == 2
        round2 = session.get_messages_for_round(2)
        assert len(round2) == 1

    def test_get_messages_by_role(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="t", roles=["analyst", "critic"],
            brain="e4b", rounds_planned=2,
        )
        session.add_message("analyst", "analysis", 1, "a1")
        session.add_message("critic", "analysis", 1, "c1")
        session.add_message("analyst", "discussion", 2, "a2")

        analyst_msgs = session.get_messages_by_role("analyst")
        assert len(analyst_msgs) == 2
        critic_msgs = session.get_messages_by_role("critic")
        assert len(critic_msgs) == 1

    def test_finish_done(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="t", roles=["analyst"], brain="e4b", rounds_planned=1
        )
        assert session.finished_at is None
        session.finish("done")
        assert session.status == "done"
        assert session.finished_at is not None

    def test_finish_cancelled(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="t", roles=["analyst"], brain="e4b", rounds_planned=1
        )
        session.finish("cancelled")
        assert session.status == "cancelled"
        assert session.finished_at is not None

    def test_to_status_dict(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="My Topic", roles=["analyst", "critic"],
            brain="a4b", rounds_planned=3,
        )
        session.add_message("analyst", "analysis", 1, "msg")
        d = session.to_status_dict()
        assert d["panel_id"] == "p1"
        assert d["topic"] == "My Topic"
        assert d["status"] == "running"
        assert d["roles"] == ["analyst", "critic"]
        assert d["brain"] == "a4b"
        assert d["rounds_planned"] == 3
        assert d["message_count"] == 1

    def test_empty_round_returns_empty_list(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="t", roles=["analyst"], brain="e4b", rounds_planned=1
        )
        assert session.get_messages_for_round(99) == []

    def test_empty_role_returns_empty_list(self) -> None:
        session = PanelSession(
            panel_id="p1", topic="t", roles=["analyst"], brain="e4b", rounds_planned=1
        )
        assert session.get_messages_by_role("nonexistent") == []


# ─── ConvergenceDetector ─────────────────────────────────────────────────


class TestConvergenceDetector:
    def setup_method(self) -> None:
        self.detector = ConvergenceDetector()

    @pytest.mark.asyncio
    async def test_empty_messages_no_convergence(self) -> None:
        api = MagicMock()
        result = await self.detector.check([], ["analyst", "critic"], 2, api)
        assert result["converged"] is False
        assert result["agreements"] == []
        assert result["agreement_count"] == 0

    @pytest.mark.asyncio
    async def test_convergence_detected(self) -> None:
        api = MagicMock()
        llm_response = json.dumps({
            "agreements": [
                {"point": "Python is great", "agreeing_roles": ["analyst", "critic"]},
                {"point": "Testing matters", "agreeing_roles": ["analyst", "pragmatist"]},
                {"point": "Docs are important", "agreeing_roles": ["critic", "creative"]},
            ]
        })
        api.llm_chat = AsyncMock(return_value=llm_response)

        messages = [
            {"role": "analyst", "phase": "discussion", "round": 1, "content": "Analysis..."},
            {"role": "critic", "phase": "discussion", "round": 1, "content": "Critique..."},
        ]
        roles = ["analyst", "critic", "creative", "pragmatist"]
        result = await self.detector.check(messages, roles, 2, api)

        assert result["converged"] is True
        assert result["agreement_count"] == 3

    @pytest.mark.asyncio
    async def test_below_threshold_no_convergence(self) -> None:
        api = MagicMock()
        llm_response = json.dumps({
            "agreements": [
                {"point": "One agreement", "agreeing_roles": ["analyst", "critic"]},
            ]
        })
        api.llm_chat = AsyncMock(return_value=llm_response)

        messages = [
            {"role": "analyst", "phase": "discussion", "round": 1, "content": "..."},
        ]
        roles = ["analyst", "critic"]
        result = await self.detector.check(messages, roles, 3, api)

        assert result["converged"] is False
        assert result["agreement_count"] == 1

    @pytest.mark.asyncio
    async def test_llm_error_returns_no_convergence(self) -> None:
        api = MagicMock()
        api.llm_chat = AsyncMock(side_effect=RuntimeError("LLM error"))

        messages = [{"role": "analyst", "phase": "discussion", "round": 1, "content": "..."}]
        result = await self.detector.check(messages, ["analyst", "critic"], 1, api)
        assert result["converged"] is False

    def test_parse_agreements_markdown_fences(self) -> None:
        raw = '```json\n{"agreements": [{"point": "P1", "agreeing_roles": ["analyst", "critic"]}]}\n```'
        result = ConvergenceDetector._parse_agreements(raw, ["analyst", "critic"])
        assert len(result) == 1
        assert result[0]["point"] == "P1"

    def test_parse_agreements_invalid_json(self) -> None:
        result = ConvergenceDetector._parse_agreements("not json at all", ["analyst"])
        assert result == []

    def test_parse_agreements_filters_invalid_roles(self) -> None:
        raw = json.dumps({
            "agreements": [
                {"point": "P1", "agreeing_roles": ["analyst", "unknown_role", "critic"]},
            ]
        })
        result = ConvergenceDetector._parse_agreements(raw, ["analyst", "critic"])
        assert len(result) == 1
        assert result[0]["agreeing_roles"] == ["analyst", "critic"]

    def test_parse_agreements_needs_two_roles(self) -> None:
        raw = json.dumps({
            "agreements": [
                {"point": "Only one role", "agreeing_roles": ["analyst"]},
            ]
        })
        result = ConvergenceDetector._parse_agreements(raw, ["analyst", "critic"])
        assert result == []

    def test_parse_agreements_max_five(self) -> None:
        raw = json.dumps({
            "agreements": [
                {"point": f"P{i}", "agreeing_roles": ["analyst", "critic"]}
                for i in range(10)
            ]
        })
        result = ConvergenceDetector._parse_agreements(raw, ["analyst", "critic"])
        assert len(result) == 5


# ─── PanelSynthesizer ────────────────────────────────────────────────────


class TestPanelSynthesizer:
    def setup_method(self) -> None:
        self.synthesizer = PanelSynthesizer()

    @pytest.mark.asyncio
    async def test_empty_messages_returns_placeholder(self) -> None:
        api = MagicMock()
        result = await self.synthesizer.synthesize([], "topic", api)
        assert "Keine Diskussion" in result["summary"]
        assert result["consensus_points"] == []
        assert result["dissent_points"] == []
        assert result["action_items"] == []

    @pytest.mark.asyncio
    async def test_successful_synthesis(self) -> None:
        api = MagicMock()
        llm_response = json.dumps({
            "summary": "The panel discussed X and concluded Y.",
            "consensus_points": ["Point A", "Point B"],
            "dissent_points": ["Disagreement on C"],
            "action_items": ["Implement D", "Research E"],
        })
        api.llm_chat = AsyncMock(return_value=llm_response)

        messages = [
            {"role": "analyst", "phase": "discussion", "round": 1, "content": "analysis text"},
            {"role": "critic", "phase": "discussion", "round": 1, "content": "critique text"},
        ]
        result = await self.synthesizer.synthesize(messages, "Test Topic", api)

        assert result["summary"] == "The panel discussed X and concluded Y."
        assert len(result["consensus_points"]) == 2
        assert len(result["dissent_points"]) == 1
        assert len(result["action_items"]) == 2

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self) -> None:
        api = MagicMock()
        api.llm_chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        messages = [{"role": "analyst", "phase": "analysis", "round": 1, "content": "text"}]
        result = await self.synthesizer.synthesize(messages, "topic", api)
        assert "fehlgeschlagen" in result["summary"]

    def test_parse_synthesis_with_markdown_fences(self) -> None:
        raw = '```json\n{"summary": "S", "consensus_points": ["A"], "dissent_points": [], "action_items": ["B"]}\n```'
        result = PanelSynthesizer._parse_synthesis(raw)
        assert result["summary"] == "S"
        assert result["consensus_points"] == ["A"]
        assert result["action_items"] == ["B"]

    def test_parse_synthesis_invalid_json_fallback(self) -> None:
        result = PanelSynthesizer._parse_synthesis("This is just plain text, no JSON.")
        # Falls back using text as summary
        assert "plain text" in result["summary"]
        assert result["consensus_points"] == []

    def test_parse_synthesis_max_5_items(self) -> None:
        raw = json.dumps({
            "summary": "S",
            "consensus_points": [f"p{i}" for i in range(10)],
            "dissent_points": [f"d{i}" for i in range(10)],
            "action_items": [f"a{i}" for i in range(10)],
        })
        result = PanelSynthesizer._parse_synthesis(raw)
        assert len(result["consensus_points"]) <= 5
        assert len(result["dissent_points"]) <= 5
        assert len(result["action_items"]) <= 5

    def test_parse_synthesis_no_json_at_all(self) -> None:
        result = PanelSynthesizer._parse_synthesis("")
        assert result["consensus_points"] == []
        assert result["dissent_points"] == []
        assert result["action_items"] == []
