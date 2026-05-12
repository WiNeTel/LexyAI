"""
Avatar World — Plugin entry point.

Subscribes to backend events, maintains an AvatarState, and broadcasts
``avatar.*`` WS frames whenever something interesting happens. A small
background ticker also drives idle progression, time-of-day buckets, and
the emotion decay.

Frontend (Babylon) is the rendering side; this plugin only ever speaks
in terms of state transitions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from lexy_core.events.event_bus import Event
from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger
from plugins.avatar_world.emotion_engine import EmotionEngine
from plugins.avatar_world.event_mapper import EventMapper
from plugins.avatar_world.idle_timer import IdleStage, IdleTimer
from plugins.avatar_world.outfit_router import suggest_outfit
from plugins.avatar_world.state import AvatarState
from plugins.avatar_world.time_of_day import current_bucket, pick_background
from plugins.avatar_world.ws_publisher import WSPublisher

log = get_logger(module="avatar_world")


class AvatarWorldPlugin(BasePlugin):
    """3D avatar + apartment state plugin."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        # Config-driven knobs (populated in on_load).
        self._available_outfits: list[str] = ["casual"]
        self._available_backgrounds: list[str] = ["city_day"]
        self._tick_interval_s: float = 5.0

        # Wired in on_load — declared here so type-checkers see them.
        self._state: AvatarState | None = None
        self._emotion: EmotionEngine | None = None
        self._idle: IdleTimer | None = None
        self._ws: WSPublisher | None = None
        self._mapper: EventMapper | None = None

        self._tick_task: asyncio.Task | None = None

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        cfg = self.api.get_config()

        self._available_outfits = list(
            cfg.get("available_outfits", ["casual"]) or ["casual"]
        )
        self._available_backgrounds = list(
            cfg.get("available_backgrounds", ["city_day"]) or ["city_day"]
        )
        self._tick_interval_s = float(cfg.get("tick_interval_s", 5.0))
        default_outfit = str(cfg.get("default_outfit", "casual"))
        default_activity = str(cfg.get("default_activity", "sit_desk"))
        default_view_mode = str(cfg.get("default_view_mode", "ambient"))
        emotion_decay_s = float(cfg.get("emotion_decay_s", 30.0))
        idle_tired_after_s = float(cfg.get("idle_tired_after_s", 600.0))
        idle_sleep_after_s = float(cfg.get("idle_sleep_after_s", 1800.0))

        # Build initial state from config.
        bucket = current_bucket()
        self._state = AvatarState(
            outfit=default_outfit if default_outfit in self._available_outfits
            else (self._available_outfits[0] if self._available_outfits else "casual"),
            activity=default_activity,  # type: ignore[arg-type]
            view_mode_hint=default_view_mode,  # type: ignore[arg-type]
            time_of_day=bucket,
            background_id=pick_background(bucket, self._available_backgrounds),
        )

        self._emotion = EmotionEngine(
            state=self._state,
            decay_seconds=emotion_decay_s,
        )
        self._idle = IdleTimer(
            tired_after_s=idle_tired_after_s,
            sleep_after_s=idle_sleep_after_s,
        )
        self._ws = WSPublisher(self.api)
        self._mapper = EventMapper(
            state=self._state,
            emotion=self._emotion,
            ws=self._ws,
            idle=self._idle,
        )

        log.info(
            "avatar_world.loaded",
            outfit=self._state.outfit,
            background=self._state.background_id,
            bucket=bucket,
            outfits=self._available_outfits,
        )

    async def on_enable(self) -> None:
        # Subscribe to the core.* bus — the EventBus delivers everything
        # under that wildcard, the mapper picks what to react to.
        self.api.on_event("core.*", self._on_core_event)

        # Frontend → backend control channel (outfit pick, view mode, etc.)
        self.api.register_ws_handler("avatar.request", self._ws_avatar_request)

        # Send the initial snapshot so freshly-connected clients have
        # something to render before any event fires.
        assert self._state is not None and self._ws is not None
        await self._ws.send_state(self._state.snapshot())

        # Start the background sweeper (idle / time-of-day / decay).
        self._tick_task = asyncio.create_task(self._tick_loop())

        log.info("avatar_world.enabled")

    async def on_disable(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                log.warning("avatar_world.tick_task_stop_error", error=str(exc))
            self._tick_task = None
        log.info("avatar_world.disabled")

    # ─── Event subscribers ──────────────────────────────────────────

    async def _on_core_event(self, event: Event) -> None:
        """Hook into the core.* event stream and forward to the mapper."""
        if self._mapper is None:
            return
        await self._mapper.apply(event.name, dict(event.data or {}))

    # ─── WS handlers ────────────────────────────────────────────────

    async def _ws_avatar_request(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Handle frontend-initiated overrides (manual outfit, view-mode, …)."""
        if self._state is None or self._ws is None or self._emotion is None:
            await client.send_json(
                {"type": "avatar.error", "error": "plugin_not_ready"}
            )
            return

        payload = message.get("payload") or {}
        action = str(payload.get("action") or "").strip()

        if action == "set_outfit":
            outfit = str(payload.get("outfit") or "").strip()
            if outfit and outfit in self._available_outfits:
                self._state.outfit = outfit
                await self._ws.send_outfit(outfit, reason="user_request")
            else:
                await client.send_json(
                    {
                        "type": "avatar.error",
                        "error": f"unknown_outfit:{outfit!r}",
                    }
                )
            return

        if action == "set_view_mode":
            mode = str(payload.get("mode") or "").strip()
            if mode in ("conversation", "ambient", "pip"):
                self._state.view_mode_hint = mode  # type: ignore[assignment]
                await self._ws.send_view_mode(mode)
            else:
                await client.send_json(
                    {
                        "type": "avatar.error",
                        "error": f"unknown_view_mode:{mode!r}",
                    }
                )
            return

        if action == "force_emotion":
            name = str(payload.get("name") or "").strip()
            intensity = float(payload.get("intensity") or 0.0)
            allowed = {"neutral", "happy", "thinking", "surprised", "tired"}
            if name not in allowed:
                await client.send_json(
                    {
                        "type": "avatar.error",
                        "error": f"unknown_emotion:{name!r}",
                    }
                )
                return
            emo = self._emotion.force(name, intensity)  # type: ignore[arg-type]
            await self._ws.send_emotion(emo.name, emo.intensity)
            return

        if action == "get_state":
            # Frontend asks for a fresh snapshot (e.g. after reconnect).
            await client.send_json(
                {"type": "avatar.state", "payload": self._state.snapshot()}
            )
            return

        await client.send_json(
            {"type": "avatar.error", "error": f"unknown_action:{action!r}"}
        )

    # ─── Background sweeper ─────────────────────────────────────────

    async def _tick_loop(self) -> None:
        """Periodically advance idle, time-of-day, and emotion decay."""
        assert self._state is not None
        assert self._emotion is not None
        assert self._idle is not None
        assert self._ws is not None

        last_stage: IdleStage = self._idle.stage
        last_bucket = self._state.time_of_day

        try:
            while True:
                await asyncio.sleep(self._tick_interval_s)

                # ── 1) Emotion decay ────────────────────────────────
                emo_change = self._emotion.decay(self._tick_interval_s)
                if emo_change is not None:
                    await self._ws.send_emotion(
                        emo_change.name, emo_change.intensity
                    )

                # ── 2) Idle progression ─────────────────────────────
                stage = self._idle.tick(self._state.last_user_seen_at)
                if stage != last_stage:
                    await self._apply_idle_transition(stage)
                    last_stage = stage

                # ── 3) Time-of-day cycle ────────────────────────────
                bucket = current_bucket()
                if bucket != last_bucket:
                    last_bucket = bucket
                    self._state.time_of_day = bucket
                    new_bg = pick_background(
                        bucket, self._available_backgrounds
                    )
                    if new_bg and new_bg != self._state.background_id:
                        self._state.background_id = new_bg
                        await self._ws.send_background(new_bg)
                    # Re-evaluate outfit on bucket change too (might be
                    # evening now and we want to suggest pyjama later if
                    # idle moves us to SLEEPING).
                    await self._maybe_update_outfit()
        except asyncio.CancelledError:
            # Normal shutdown path — propagate so caller can await us.
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("avatar_world.tick_loop_error", error=str(exc))

    async def _apply_idle_transition(self, stage: IdleStage) -> None:
        """Reflect a coarse idle stage change in emotion + activity."""
        assert self._state is not None
        assert self._emotion is not None
        assert self._ws is not None

        now = datetime.now().timestamp()
        if stage == IdleStage.ACTIVE:
            # User came back — reset to neutral, sit at the desk, look at
            # them. Don't crank emotion to 1.0; the user-message event
            # already did the surprised bump.
            self._state.activity = "sit_desk"
            self._state.gaze_target = "camera"
            await self._ws.send_activity("sit_desk", transition="cut")
            await self._ws.send_attention("camera")
            await self._maybe_update_outfit()
            return

        if stage == IdleStage.TIRED:
            emo = self._emotion.bump("tired", 0.5)
            await self._ws.send_emotion(emo.name, emo.intensity)
            return

        if stage == IdleStage.SLEEPING:
            self._state.activity = "sleep_couch"
            self._state.gaze_target = "none"
            emo = self._emotion.force("tired", 0.9)
            await self._ws.send_activity("sleep_couch", transition="walk")
            await self._ws.send_attention("ambient")
            await self._ws.send_emotion(emo.name, emo.intensity)
            await self._maybe_update_outfit()
        # mark_seen / event-driven resets are NOT done here — they're the
        # caller's job (the user-message handler resets the idle clock).
        _ = now

    async def _maybe_update_outfit(self) -> None:
        """Suggest a new outfit if context changed; push only on diff."""
        assert self._state is not None and self._ws is not None
        assert self._idle is not None

        suggested = suggest_outfit(
            bucket=self._state.time_of_day,
            idle_stage=self._idle.stage,
            available=self._available_outfits,
            fallback=self._state.outfit,
        )
        if suggested and suggested != self._state.outfit:
            self._state.outfit = suggested
            await self._ws.send_outfit(
                suggested, reason=f"auto:{self._state.time_of_day}/{self._idle.stage.value}"
            )
