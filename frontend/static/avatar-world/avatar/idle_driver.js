/*
 * Avatar World — Idle / liveliness driver.
 *
 * Without this Lexy looks frozen between emotion changes. The driver
 * runs every frame and adds the small movements that sell "alive":
 *
 *   - **Blink** — eyeBlinkLeft/Right pulse every 3-6s with a 120ms
 *     close/open envelope. Skipped while the emotion is `tired` because
 *     the tired preset already half-closes the eyes.
 *   - **Breathing** — a low-frequency sine (~4s period) breathes the
 *     placeholder/avatar root by a few millimetres on Y, plus tiny
 *     scale on the chest if we ever rig that bone. Off in `sleep`
 *     activity (we already drove the avatar onto the couch).
 *   - **Gaze saccades** — random tiny look-around movements via the
 *     eyeLookIn/Out/Up/Down shapes. Less than 30 px-equivalent — it
 *     should be subliminal, not theatrical.
 *   - **Mikro-Bewegung** — the whole avatar root drifts by a few
 *     millimetres along X over many seconds to dodge the "statue" look.
 *
 * All shape writes go through `emotion_driver.setShape`, which adds
 * them onto the emotion-driven base layer.
 */
(() => {
    "use strict";

    const SHAPES = (window.LexyAvatar && window.LexyAvatar.morphs)
        ? window.LexyAvatar.morphs.SHAPES
        : null;

    const refs = {
        scene: null,
        handle: null,
        placeholder: null,
    };

    // Driver state — kept module-local so re-init is clean.
    let enabled = true;
    let nextBlinkAt = 0;
    let blinkPhase = "idle";       // idle | closing | opening
    let blinkStartedAt = 0;

    let breathPhase = 0;           // accumulated radians for sine
    let gazeNextSaccadeAt = 0;
    let gazeTarget = { in: 0, up: 0, out: 0, down: 0 };
    let gazeFrom = { in: 0, up: 0, out: 0, down: 0 };
    let gazeRampStartedAt = 0;
    let gazeRampDuration = 400;

    let activityHint = "sit_desk";   // updated by index.js when avatar.activity arrives

    function _rand(min, max) {
        return min + Math.random() * (max - min);
    }

    function _scheduleNextBlink(now) {
        // Pick a random gap. Slight gaussian-ish distribution by averaging
        // two uniforms — less robotic than pure uniform.
        const gap = (_rand(2.4, 5.5) + _rand(2.4, 5.5)) / 2;
        nextBlinkAt = now + gap * 1000;
    }

    function _scheduleNextSaccade(now) {
        gazeNextSaccadeAt = now + _rand(1800, 4500);
    }

    function init(opts) {
        refs.scene = (opts && opts.scene) || null;
        refs.handle = (opts && opts.handle) || null;
        refs.placeholder = (opts && opts.placeholder) || null;

        const now = performance.now();
        _scheduleNextBlink(now);
        _scheduleNextSaccade(now);
        blinkPhase = "idle";
        breathPhase = Math.random() * Math.PI * 2;
    }

    function setHandle(handle) {
        refs.handle = handle || null;
    }

    function setEnabled(flag) {
        enabled = !!flag;
        // When we disable, relax any additive shape we'd been driving.
        if (!enabled && window.LexyAvatar && window.LexyAvatar.emotion) {
            const emo = window.LexyAvatar.emotion;
            if (SHAPES) {
                emo.clearAdditive(SHAPES.eyeBlinkL);
                emo.clearAdditive(SHAPES.eyeBlinkR);
                for (const s of [SHAPES.eyeLookInL, SHAPES.eyeLookInR,
                                  SHAPES.eyeLookOutL, SHAPES.eyeLookOutR,
                                  SHAPES.eyeLookUpL,  SHAPES.eyeLookUpR,
                                  SHAPES.eyeLookDownL, SHAPES.eyeLookDownR]) {
                    if (s) emo.clearAdditive(s);
                }
            }
        }
    }

    function setActivity(id) {
        activityHint = id || activityHint;
    }

    function _drive(shape, value) {
        if (!shape) return;
        if (!window.LexyAvatar || !window.LexyAvatar.emotion) return;
        window.LexyAvatar.emotion.setShape(shape, value);
    }

    function _tickBlink(now) {
        if (!SHAPES) return;
        if (activityHint === "sleep_couch") {
            // Don't blink in your sleep.
            _drive(SHAPES.eyeBlinkL, 0);
            _drive(SHAPES.eyeBlinkR, 0);
            return;
        }
        if (blinkPhase === "idle") {
            if (now >= nextBlinkAt) {
                blinkPhase = "closing";
                blinkStartedAt = now;
            }
            return;
        }
        const dt = now - blinkStartedAt;
        const CLOSE_MS = 90;
        const OPEN_MS = 140;
        if (blinkPhase === "closing") {
            const t = Math.min(1, dt / CLOSE_MS);
            _drive(SHAPES.eyeBlinkL, t);
            _drive(SHAPES.eyeBlinkR, t);
            if (t >= 1) {
                blinkPhase = "opening";
                blinkStartedAt = now;
            }
        } else if (blinkPhase === "opening") {
            const t = Math.min(1, dt / OPEN_MS);
            _drive(SHAPES.eyeBlinkL, 1 - t);
            _drive(SHAPES.eyeBlinkR, 1 - t);
            if (t >= 1) {
                blinkPhase = "idle";
                _drive(SHAPES.eyeBlinkL, 0);
                _drive(SHAPES.eyeBlinkR, 0);
                _scheduleNextBlink(now);
            }
        }
    }

    function _tickBreathing(_now, dtMs) {
        // 4-second period — quiet, easy to miss, exactly the point.
        breathPhase += (dtMs / 1000) * (Math.PI * 2 / 4.0);
        const breath = Math.sin(breathPhase) * 0.5 + 0.5;   // 0..1
        const yOffset = 0.008 * (breath - 0.5);             // ±4mm

        const node = (refs.handle && refs.handle.root) || refs.placeholder;
        if (node && node.position) {
            // Only nudge the Y axis — anything else looks wrong.
            const baseY = node._lexyBreathBaseY;
            if (baseY === undefined) {
                node._lexyBreathBaseY = node.position.y;
                return;
            }
            node.position.y = baseY + yOffset;
        }
    }

    function _tickGaze(now) {
        if (!SHAPES) return;
        if (activityHint === "sleep_couch") {
            // No gaze movement while sleeping — eyes are closed.
            return;
        }
        if (now >= gazeNextSaccadeAt) {
            // Pick a tiny new direction. Keep weights low so the eyes
            // shift, they don't roll.
            gazeFrom = { ...gazeTarget };
            gazeTarget = {
                in:  Math.random() < 0.5 ? _rand(0, 0.15) : 0,
                out: Math.random() < 0.5 ? _rand(0, 0.15) : 0,
                up:  Math.random() < 0.5 ? _rand(0, 0.10) : 0,
                down: Math.random() < 0.5 ? _rand(0, 0.10) : 0,
            };
            gazeRampStartedAt = now;
            gazeRampDuration = _rand(220, 380);
            _scheduleNextSaccade(now + _rand(900, 2400));
        }

        const t = Math.min(1, (now - gazeRampStartedAt) / gazeRampDuration);
        const cur = {
            in:  gazeFrom.in  + (gazeTarget.in  - gazeFrom.in)  * t,
            out: gazeFrom.out + (gazeTarget.out - gazeFrom.out) * t,
            up:  gazeFrom.up  + (gazeTarget.up  - gazeFrom.up)  * t,
            down: gazeFrom.down + (gazeTarget.down - gazeFrom.down) * t,
        };
        // Mirror left/right — both eyes look together.
        _drive(SHAPES.eyeLookInL,  cur.in);
        _drive(SHAPES.eyeLookInR,  cur.in);
        _drive(SHAPES.eyeLookOutL, cur.out);
        _drive(SHAPES.eyeLookOutR, cur.out);
        _drive(SHAPES.eyeLookUpL,  cur.up);
        _drive(SHAPES.eyeLookUpR,  cur.up);
        _drive(SHAPES.eyeLookDownL, cur.down);
        _drive(SHAPES.eyeLookDownR, cur.down);
    }

    function tick(now, dtMs) {
        if (!enabled) return;
        _tickBlink(now);
        _tickBreathing(now, dtMs);
        _tickGaze(now);
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.idle = {
        init,
        setHandle,
        setEnabled,
        setActivity,
        tick,
    };
})();
