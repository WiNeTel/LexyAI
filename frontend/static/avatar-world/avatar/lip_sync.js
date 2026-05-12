/*
 * Avatar World — Lip-Sync driver (Phase 1B).
 *
 * Drives the avatar's mouth from the live TTS audio without requiring
 * phoneme timings from the backend. Two layers:
 *
 *   1. **Speaking lifecycle** — onSpeakStart / onSpeakEnd toggle the
 *      glowing aura placeholder (Phase 0 affordance) and the "active"
 *      flag. The actual envelope is computed in tick().
 *
 *   2. **Energy envelope** — every animation frame we read the
 *      AudioContext analyser exposed by app.js (window.Lexy.audio),
 *      compute a smoothed RMS, and lerp the avatar's `jawOpen` morph
 *      target between currentJaw and targetJaw. A tiny `mouthFunnel`
 *      modulation adds a touch of vowel variation.
 *
 * Safeties:
 *   - `setEnabled(false)` — silences everything. Called by app.js when
 *     the user turns TTS off. Any in-flight speaking ramps back down
 *     to closed-mouth and the aura is hidden.
 *   - Watchdog — if onSpeakStart fired but no audio level arrived for
 *     > IDLE_TIMEOUT_MS the driver auto-ends. Covers the "TTS started
 *     but the WAV stream stalled" edge case.
 *   - No-audio-context — when `window.Lexy.audio.analyser` is missing
 *     (user never enabled TTS, browser blocked AudioContext) the
 *     analyser path is skipped and the aura still gives visible
 *     feedback if onSpeakStart was called explicitly.
 *
 * Public surface:
 *   - init({ scene, placeholder, speakAura, handle })
 *   - setEnabled(flag)
 *   - onSpeakStart(payload), onSpeakEnd(payload)
 *   - onAudioLevel(level)   — optional direct push (used by tests)
 *   - tick(now, dtMs)        — called from index.js animation loop
 */
(() => {
    "use strict";

    const SHAPES = (window.LexyAvatar && window.LexyAvatar.morphs)
        ? window.LexyAvatar.morphs.SHAPES
        : null;

    // How long without a fresh audio level before we assume TTS stopped
    // and auto-call onSpeakEnd. Slightly longer than a typical inter-
    // chunk gap so we don't flicker the mouth.
    const IDLE_TIMEOUT_MS = 800;
    // Smoothing factor for the RMS → target jawOpen mapping. Higher =
    // more responsive but jittery; lower = smoother but laggy.
    const SMOOTH_ALPHA = 0.45;
    // Empirical scale: typical CosyVoice WAV chunks land around RMS
    // 0.08–0.14 → push the jaw open by ~0.5–0.9.
    const RMS_TO_JAW = 6.0;
    // Cap so a louder-than-expected clip doesn't slam the jaw fully open.
    const MAX_JAW = 0.85;

    const refs = {
        scene: null,
        placeholder: null,
        speakAura: null,
        handle: null,
    };

    // Module state.
    let enabled = true;            // master switch — false silences everything
    let speaking = false;          // true between onSpeakStart and onSpeakEnd
    let lastLevelAt = 0;           // performance.now() of last audio level read
    let smoothedRMS = 0;
    let currentJaw = 0;
    let targetJaw = 0;
    let auraPulseStartedAt = 0;
    let warnedAnalyserMissing = false;

    function init(opts) {
        refs.scene = (opts && opts.scene) || null;
        refs.placeholder = (opts && opts.placeholder) || null;
        refs.speakAura = (opts && opts.speakAura) || null;
        refs.handle = (opts && opts.handle) || null;
        smoothedRMS = 0;
        currentJaw = 0;
        targetJaw = 0;
        speaking = false;
    }

    function setHandle(handle) {
        refs.handle = handle || null;
    }

    function setEnabled(flag) {
        const wasEnabled = enabled;
        enabled = !!flag;
        if (wasEnabled && !enabled) {
            // Hard reset — close the mouth, hide the aura.
            _silence();
        }
    }

    function _silence() {
        speaking = false;
        targetJaw = 0;
        smoothedRMS = 0;
        if (refs.speakAura) {
            refs.speakAura.scaling.x = refs.speakAura.scaling.y = refs.speakAura.scaling.z = 1.0;
            refs.speakAura.setEnabled(false);
        }
        if (SHAPES && window.LexyAvatar && window.LexyAvatar.emotion) {
            // Clear the additive jaw immediately so the rendered face
            // closes within the next frame, not after the lerp.
            window.LexyAvatar.emotion.clearAdditive(SHAPES.jawOpen);
            window.LexyAvatar.emotion.clearAdditive(SHAPES.mouthFunnel);
            window.LexyAvatar.emotion.clearAdditive(SHAPES.mouthStretchL);
            window.LexyAvatar.emotion.clearAdditive(SHAPES.mouthStretchR);
        }
        currentJaw = 0;
    }

    function onSpeakStart(payload) {
        if (!enabled) return;
        speaking = true;
        lastLevelAt = performance.now();
        auraPulseStartedAt = performance.now();
        if (refs.speakAura) refs.speakAura.setEnabled(true);
    }

    function onSpeakEnd(payload) {
        if (!speaking) return;
        speaking = false;
        targetJaw = 0;
        if (refs.speakAura) refs.speakAura.setEnabled(false);
    }

    function onAudioLevel(level) {
        // Direct entry point (used by tests + manual pushes). The tick
        // loop pulls levels from the analyser by default.
        if (!enabled) return;
        const r = Math.max(0, Math.min(1, Number(level) || 0));
        smoothedRMS = smoothedRMS + (r - smoothedRMS) * SMOOTH_ALPHA;
        targetJaw = Math.min(MAX_JAW, smoothedRMS * RMS_TO_JAW);
        lastLevelAt = performance.now();
        if (!speaking) {
            // First level after silence → auto-start speaking lifecycle.
            speaking = true;
            auraPulseStartedAt = lastLevelAt;
            if (refs.speakAura) refs.speakAura.setEnabled(true);
        }
    }

    function _readAnalyserRMS() {
        const audio = window.Lexy && window.Lexy.audio;
        if (!audio || !audio.analyser) {
            if (!warnedAnalyserMissing) {
                // Logged once — when TTS has never been enabled there's
                // no AudioContext to tap, and that's totally fine.
                warnedAnalyserMissing = true;
                console.debug("lip_sync: no audio analyser yet — running aura-only");
            }
            return null;
        }
        warnedAnalyserMissing = false;
        const analyser = audio.analyser;
        const bins = analyser.frequencyBinCount;
        // Reuse a buffer between frames to avoid GC churn.
        if (!_readAnalyserRMS._buf || _readAnalyserRMS._buf.length !== bins) {
            _readAnalyserRMS._buf = new Uint8Array(bins);
        }
        const buf = _readAnalyserRMS._buf;
        analyser.getByteFrequencyData(buf);
        let sum = 0;
        // Focus on the vocal band (0–2 kHz) — the high bins are mostly
        // sibilants/noise and inflate the value when the voice is calm.
        const vocalBins = Math.min(bins, 32);
        for (let i = 0; i < vocalBins; i++) {
            const v = buf[i] / 255;
            sum += v * v;
        }
        return Math.sqrt(sum / vocalBins);
    }

    function tick(now, _dtMs) {
        if (!enabled) {
            currentJaw = 0;
            return;
        }

        // 1) Pull a fresh audio level when speaking.
        if (speaking) {
            const rms = _readAnalyserRMS();
            if (rms !== null && rms > 0.0001) {
                onAudioLevel(rms);
            }
            // Watchdog: end the speaking phase if no level updated us
            // in a while. Only triggers when we DO have an analyser —
            // otherwise the aura-only mode keeps running until the
            // backend sends an explicit avatar.speaking{end}.
            const audioPresent = window.Lexy && window.Lexy.audio && window.Lexy.audio.analyser;
            if (audioPresent && (now - lastLevelAt > IDLE_TIMEOUT_MS)) {
                onSpeakEnd();
            }
        } else {
            targetJaw = 0;
        }

        // 2) Smooth jaw toward target.
        currentJaw = currentJaw + (targetJaw - currentJaw) * 0.5;
        if (currentJaw < 0.001) currentJaw = 0;

        // 3) Push to the morph (additive over emotion).
        if (SHAPES && window.LexyAvatar && window.LexyAvatar.emotion) {
            window.LexyAvatar.emotion.setShape(SHAPES.jawOpen, currentJaw);
            // Subtle lip stretch — looks more natural than pure jawOpen.
            const stretch = currentJaw * 0.35;
            window.LexyAvatar.emotion.setShape(SHAPES.mouthStretchL, stretch);
            window.LexyAvatar.emotion.setShape(SHAPES.mouthStretchR, stretch);
            // Mouth funnel modulates with a tiny sine for vowel variety.
            const funnel = currentJaw * 0.20 * (0.5 + 0.5 * Math.sin(now / 110));
            window.LexyAvatar.emotion.setShape(SHAPES.mouthFunnel, funnel);
        }

        // 4) Phase-0 aura — still useful when no morphs are loaded.
        if (refs.speakAura && refs.speakAura.isEnabled()) {
            const t = (now - auraPulseStartedAt) / 1000;
            // Larger aura when louder, plus a tiny shimmer.
            const base = 0.7 + currentJaw * 0.5;
            const s = base + 0.05 * Math.sin(t * 14);
            refs.speakAura.scaling.x = s;
            refs.speakAura.scaling.y = s;
            refs.speakAura.scaling.z = s;
        }
    }

    function snapshot() {
        return {
            enabled,
            speaking,
            currentJaw,
            smoothedRMS,
            analyserAvailable: !!(window.Lexy && window.Lexy.audio && window.Lexy.audio.analyser),
        };
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.lipSync = {
        init,
        setHandle,
        setEnabled,
        onSpeakStart,
        onSpeakEnd,
        onAudioLevel,
        tick,
        snapshot,
    };
})();
