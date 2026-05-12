"""Tests for the avatar_world plugin — pure components.

The plugin's heavy lifting lives in five small modules that don't need a
running ``LexyApp``:

* :class:`EmotionEngine` — bump / force / decay state machine.
* :class:`IdleTimer` — coarse active/tired/sleeping stages.
* :mod:`time_of_day` — hour → bucket → background mapping.
* :func:`suggest_outfit` — time-of-day × idle-stage routing.
* :class:`EventMapper` — backend-event → avatar-reaction translator.

Wiring with the real PluginAPI (event subscription, WS broadcast, tick
loop) is covered by manual verification.
"""

from __future__ import annotations

from typing import Any

import pytest

from plugins.avatar_world.emotion_engine import EmotionEngine
from plugins.avatar_world.event_mapper import EventMapper
from plugins.avatar_world.idle_timer import IdleStage, IdleTimer
from plugins.avatar_world.outfit_router import suggest_outfit
from plugins.avatar_world.state import AvatarState
from plugins.avatar_world.time_of_day import bucket_for_hour, pick_background


# ─── EmotionEngine ─────────────────────────────────────────────────────


class TestEmotionEngine:
    def test_bump_sets_emotion_and_intensity(self) -> None:
        state = AvatarState()
        engine = EmotionEngine(state=state, decay_seconds=10.0)

        emo = engine.bump("happy", 0.6)

        assert emo.name == "happy"
        assert emo.intensity == pytest.approx(0.6, abs=0.01)
        assert state.emotion.name == "happy"

    def test_bump_same_emotion_adds_intensity_capped_at_one(self) -> None:
        state = AvatarState()
        engine = EmotionEngine(state=state, decay_seconds=10.0)

        engine.bump("happy", 0.6)
        engine.bump("happy", 0.7)

        assert state.emotion.intensity == pytest.approx(1.0, abs=0.01)

    def test_bump_different_emotion_replaces(self) -> None:
        state = AvatarState()
        engine = EmotionEngine(state=state, decay_seconds=10.0)

        engine.bump("happy", 0.8)
        engine.bump("thinking", 0.3)

        # Replace, don't blend — Phase 1 has no mixing.
        assert state.emotion.name == "thinking"
        assert state.emotion.intensity == pytest.approx(0.3, abs=0.01)

    def test_force_overrides_immediately(self) -> None:
        state = AvatarState()
        engine = EmotionEngine(state=state, decay_seconds=10.0)
        engine.bump("happy", 0.9)

        emo = engine.force("tired", 0.4)

        assert emo.name == "tired"
        assert emo.intensity == pytest.approx(0.4, abs=0.01)

    def test_decay_reduces_intensity_linearly(self) -> None:
        state = AvatarState()
        engine = EmotionEngine(state=state, decay_seconds=10.0)
        engine.bump("happy", 1.0)

        engine.decay(5.0)  # halfway through a 10s window

        assert state.emotion.name == "happy"
        assert state.emotion.intensity == pytest.approx(0.5, abs=0.05)

    def test_decay_below_floor_returns_to_neutral(self) -> None:
        state = AvatarState()
        engine = EmotionEngine(state=state, decay_seconds=10.0)
        engine.bump("happy", 0.1)

        engine.decay(8.0)  # decay almost all the way

        assert state.emotion.name == "neutral"
        assert state.emotion.intensity == 0.0

    def test_decay_on_neutral_is_noop(self) -> None:
        state = AvatarState()
        engine = EmotionEngine(state=state, decay_seconds=10.0)

        result = engine.decay(2.0)

        assert result is None
        assert state.emotion.name == "neutral"

    def test_on_change_callback_fires(self) -> None:
        state = AvatarState()
        seen: list[str] = []

        engine = EmotionEngine(
            state=state,
            decay_seconds=10.0,
            on_change=lambda emo: seen.append(emo.name),
        )
        engine.bump("happy", 0.5)
        engine.bump("happy", 0.0)  # no change → still on the floor side

        assert "happy" in seen


# ─── IdleTimer ─────────────────────────────────────────────────────────


class TestIdleTimer:
    def test_starts_active(self) -> None:
        timer = IdleTimer(tired_after_s=60, sleep_after_s=180)
        assert timer.stage == IdleStage.ACTIVE

    def test_tired_after_first_threshold(self) -> None:
        timer = IdleTimer(tired_after_s=60, sleep_after_s=180)
        last_seen = 1_000_000.0
        # 90 seconds later — past tired (60s), before sleep (180s)
        stage = timer.tick(last_seen, now=last_seen + 90)
        assert stage == IdleStage.TIRED

    def test_sleeping_after_second_threshold(self) -> None:
        timer = IdleTimer(tired_after_s=60, sleep_after_s=180)
        last_seen = 1_000_000.0
        stage = timer.tick(last_seen, now=last_seen + 300)
        assert stage == IdleStage.SLEEPING

    def test_mark_seen_resets(self) -> None:
        timer = IdleTimer(tired_after_s=60, sleep_after_s=180)
        timer.tick(1_000_000.0, now=1_000_500.0)
        assert timer.stage == IdleStage.SLEEPING

        timer.mark_seen()

        assert timer.stage == IdleStage.ACTIVE

    def test_sleep_threshold_enforced_at_least_tired(self) -> None:
        # Defensive: a misconfig where sleep < tired must not break the FSM.
        timer = IdleTimer(tired_after_s=200, sleep_after_s=100)
        stage = timer.tick(1_000_000.0, now=1_000_300.0)
        assert stage == IdleStage.SLEEPING  # 300s > 200s


# ─── time_of_day ──────────────────────────────────────────────────────


class TestTimeOfDay:
    @pytest.mark.parametrize(
        "hour,expected",
        [
            (0, "night"),
            (4, "night"),
            (6, "morning"),
            (8, "morning"),
            (9, "midday"),
            (11, "midday"),
            (12, "afternoon"),
            (17, "afternoon"),
            (18, "evening"),
            (21, "evening"),
            (22, "night"),
            (23, "night"),
        ],
    )
    def test_bucket_for_hour(self, hour: int, expected: str) -> None:
        assert bucket_for_hour(hour) == expected

    def test_pick_background_uses_preferred_when_available(self) -> None:
        available = ["city_morning", "city_day", "city_evening", "city_night"]
        assert pick_background("morning", available) == "city_morning"
        assert pick_background("night", available) == "city_night"

    def test_pick_background_falls_back_to_first_if_preferred_missing(self) -> None:
        available = ["forest", "mountain"]
        # No 'city_morning' in list — fall back to first allowed.
        assert pick_background("morning", available) == "forest"

    def test_pick_background_empty_list_returns_empty(self) -> None:
        assert pick_background("midday", []) == ""


# ─── OutfitRouter ─────────────────────────────────────────────────────


class TestOutfitRouter:
    def test_night_sleeping_picks_pyjama(self) -> None:
        outfit = suggest_outfit(
            bucket="night",
            idle_stage=IdleStage.SLEEPING,
            available=["casual", "pyjama"],
        )
        assert outfit == "pyjama"

    def test_night_awake_stays_casual(self) -> None:
        # Late evening but Lexy hasn't conked out yet — she's not in pyjamas.
        outfit = suggest_outfit(
            bucket="night",
            idle_stage=IdleStage.ACTIVE,
            available=["casual", "pyjama"],
        )
        assert outfit == "casual"

    def test_evening_active_is_casual(self) -> None:
        outfit = suggest_outfit(
            bucket="evening",
            idle_stage=IdleStage.ACTIVE,
            available=["casual", "business", "pyjama"],
        )
        assert outfit == "casual"

    def test_missing_pyjama_falls_back_to_casual(self) -> None:
        # User pruned 'pyjama' from the whitelist — router must not crash.
        outfit = suggest_outfit(
            bucket="night",
            idle_stage=IdleStage.SLEEPING,
            available=["casual"],
        )
        assert outfit == "casual"


# ─── EventMapper ──────────────────────────────────────────────────────


class _StubWSPublisher:
    """Records every send call so a test can assert on it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def send(self, msg_type: str, payload: dict[str, Any]) -> None:
        self.calls.append((msg_type, payload))

    async def send_state(self, snapshot: dict[str, Any]) -> None:
        self.calls.append(("avatar.state", snapshot))

    async def send_emotion(
        self, name: str, intensity: float, ramp_ms: int = 600
    ) -> None:
        self.calls.append(
            ("avatar.emotion", {"name": name, "intensity": intensity})
        )

    async def send_activity(
        self, activity_id: str, transition: str = "cut"
    ) -> None:
        self.calls.append(
            ("avatar.activity", {"id": activity_id, "transition": transition})
        )

    async def send_speaking(self, state: str, stream_id: str = "") -> None:
        self.calls.append(
            ("avatar.speaking", {"state": state, "stream_id": stream_id})
        )

    async def send_outfit(self, outfit: str, reason: str = "") -> None:
        self.calls.append(
            ("avatar.outfit", {"outfit": outfit, "reason": reason})
        )

    async def send_attention(self, look_at: str) -> None:
        self.calls.append(("avatar.attention", {"look_at": look_at}))

    async def send_background(
        self, background_id: str, fade_ms: int = 2000
    ) -> None:
        self.calls.append(("avatar.background", {"id": background_id}))

    async def send_view_mode(self, mode: str) -> None:
        self.calls.append(("avatar.view_mode", {"mode": mode}))


def _fresh_mapper() -> tuple[EventMapper, AvatarState, _StubWSPublisher]:
    state = AvatarState()
    ws = _StubWSPublisher()
    emotion = EmotionEngine(state=state, decay_seconds=30.0)
    idle = IdleTimer(tired_after_s=600, sleep_after_s=1800)
    mapper = EventMapper(state=state, emotion=emotion, ws=ws, idle=idle)
    return mapper, state, ws


class TestEventMapper:
    @pytest.mark.asyncio
    async def test_user_message_turns_attention_to_camera(self) -> None:
        mapper, state, ws = _fresh_mapper()

        await mapper.apply("core.user_message", {"text": "hi"})

        types = [c[0] for c in ws.calls]
        assert "avatar.attention" in types
        attention_call = next(c for c in ws.calls if c[0] == "avatar.attention")
        assert attention_call[1]["look_at"] == "camera"
        assert state.gaze_target == "camera"
        assert state.emotion.name == "surprised"

    @pytest.mark.asyncio
    async def test_user_message_resets_idle(self) -> None:
        mapper, state, ws = _fresh_mapper()
        state.last_user_seen_at = 0.0  # ancient

        await mapper.apply("core.user_message", {"text": "hi"})

        assert state.last_user_seen_at > 0.0

    @pytest.mark.asyncio
    async def test_brain_routed_triggers_thinking(self) -> None:
        mapper, state, ws = _fresh_mapper()

        await mapper.apply("core.brain_routed", {"brain": "a4b"})

        assert state.emotion.name == "thinking"
        assert state.gaze_target == "screen"

    @pytest.mark.asyncio
    async def test_ai_response_triggers_happy(self) -> None:
        mapper, state, ws = _fresh_mapper()

        await mapper.apply("core.ai_response", {"text": "antwort"})

        assert state.emotion.name == "happy"
        assert any(c[0] == "avatar.emotion" for c in ws.calls)

    @pytest.mark.asyncio
    async def test_unknown_event_is_silent(self) -> None:
        mapper, state, ws = _fresh_mapper()

        await mapper.apply("plugin.foo.bar", {"hello": "world"})

        assert ws.calls == []
        assert state.emotion.name == "neutral"


# ─── State snapshot ───────────────────────────────────────────────────


class TestAvatarStateSnapshot:
    def test_snapshot_is_json_friendly(self) -> None:
        state = AvatarState()
        snap = state.snapshot()

        # All values are simple types — encode roundtrip proves it.
        import json

        encoded = json.dumps(snap)
        decoded = json.loads(encoded)

        assert decoded["emotion"]["name"] == "neutral"
        assert decoded["activity"] == "sit_desk"
        assert decoded["outfit"] == "casual"
        assert "intensity" in decoded["emotion"]
