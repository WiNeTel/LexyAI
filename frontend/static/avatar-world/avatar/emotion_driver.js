/*
 * Avatar World — Emotion driver.
 *
 * Phase 1A driver. Two operating modes:
 *
 *   1. **Morph mode** — when a real GLB with ARKit blendshapes is loaded
 *      (via avatar.loader) the driver lerps the relevant morph-target
 *      weights from 0 → preset[shape] * intensity over `RAMP_MS`. The
 *      previous emotion's shapes fade back to 0 simultaneously.
 *
 *   2. **Placeholder mode** — no GLB available, only the capsule
 *      mesh exists. We tint its diffuse colour by emotion as a visible
 *      stand-in. Lets the rest of the wiring be verified before a model
 *      is dropped in.
 *
 * Public surface:
 *   - init({ scene, placeholder, handle })
 *   - apply({ name, intensity })
 *   - setShape(targetName, weight) — used by idle/lip-sync drivers.
 *   - getShape(targetName) → current weight (or 0 if missing).
 *   - snapshot() → { name, intensity, mode }
 *   - hasMorphs() → boolean
 *
 * The driver keeps `additiveShapes` separate from the emotion-driven
 * `baseShapes`. The final weight pushed onto the morph target is
 * `baseShape + additiveShape`, clamped to [0, 1]. This lets the idle
 * driver layer a blink on top of the current emotion without fighting
 * for the same target value.
 */
(() => {
    "use strict";

    const RAMP_MS = 600;

    const refs = {
        scene: null,
        placeholder: null,
        handle: null,        // from avatar.loader.loadAvatar()
    };

    let currentEmotion = { name: "neutral", intensity: 0.0 };
    let targetEmotion = { name: "neutral", intensity: 0.0 };
    let rampStartedAt = 0;
    let rampFrom = { name: "neutral", intensity: 0.0 };

    // Pre-summed view of every shape an emotion preset touches so we
    // can lerp the entire set in one tick, instead of dictionary-lookup
    // per shape per frame.
    const baseShapes = new Map();        // shapeName → current emotion-driven value
    const additiveShapes = new Map();    // shapeName → idle/lip-sync overlay

    function _lerp(a, b, t) { return a + (b - a) * t; }

    function _ease(t) {
        // Smoothstep — gives a nicer feel than a flat lerp.
        return t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
    }

    function init(opts) {
        refs.scene = (opts && opts.scene) || null;
        refs.placeholder = (opts && opts.placeholder) || null;
        refs.handle = (opts && opts.handle) || null;
        // Reset state on a fresh init.
        currentEmotion = { name: "neutral", intensity: 0.0 };
        targetEmotion = { name: "neutral", intensity: 0.0 };
        baseShapes.clear();
        additiveShapes.clear();
    }

    function setHandle(handle) {
        refs.handle = handle || null;
    }

    function hasMorphs() {
        return !!(refs.handle && refs.handle.morphTargets && refs.handle.morphTargets.size > 0);
    }

    function _writeTarget(name, weight) {
        const targets = refs.handle && refs.handle.morphTargets
            ? refs.handle.morphTargets.get(name)
            : null;
        const clamped = Math.max(0, Math.min(1, weight));
        if (targets) {
            for (const t of targets) {
                // Babylon's morph target uses .influence in [0..1].
                t.influence = clamped;
            }
            return;
        }
        // No morph target with that name. If the bone-animator
        // recognises this ARKit shape (jawOpen, eyeLook*, eyeBlink),
        // route it there so MakeHuman / Mixamo rigs still react.
        const ba = window.LexyAvatar && window.LexyAvatar.boneAnimator;
        if (ba && ba.canHandle && ba.canHandle(name)) {
            ba.setShape(name, clamped);
        }
    }

    function _flushTarget(name) {
        // Write the combined (base + additive) value to the GLB.
        const base = baseShapes.get(name) || 0;
        const add = additiveShapes.get(name) || 0;
        _writeTarget(name, base + add);
    }

    function _placeholderColour(name, intensity) {
        if (!refs.placeholder || !refs.placeholder.material) return;
        const presets = window.LexyAvatar && window.LexyAvatar.morphs
            ? window.LexyAvatar.morphs.EMOTION_COLORS
            : null;
        if (!presets) return;
        const base = presets.neutral;
        const target = presets[name] || presets.neutral;
        const c = refs.placeholder.material.diffuseColor;
        if (!c) return;
        const t = Math.max(0, Math.min(1, intensity));
        c.r = _lerp(base[0], target[0], t);
        c.g = _lerp(base[1], target[1], t);
        c.b = _lerp(base[2], target[2], t);
    }

    function apply(payload) {
        const name = (payload && payload.name) || "neutral";
        const intensity = Math.max(
            0, Math.min(1, Number(payload && payload.intensity) || 0)
        );

        rampFrom = { ...currentEmotion };
        targetEmotion = { name, intensity };
        rampStartedAt = performance.now();
    }

    function tick(_dtMs) {
        // 1) Advance the ramp toward targetEmotion.
        if (currentEmotion.name !== targetEmotion.name
            || Math.abs(currentEmotion.intensity - targetEmotion.intensity) > 0.005) {
            const dt = performance.now() - rampStartedAt;
            const t = _ease(Math.min(1, dt / RAMP_MS));
            if (t >= 1) {
                currentEmotion = { ...targetEmotion };
            } else if (rampFrom.name === targetEmotion.name) {
                // Same emotion, just intensity changing — simple lerp.
                currentEmotion = {
                    name: targetEmotion.name,
                    intensity: _lerp(rampFrom.intensity, targetEmotion.intensity, t),
                };
            } else {
                // Emotion swap: blend fromIntensity → 0 then 0 → toIntensity.
                // Phase-1 approximation: fade from at (1-t), to at t.
                currentEmotion = {
                    name: targetEmotion.name,
                    intensity: _lerp(0, targetEmotion.intensity, t),
                };
                // Drive the *previous* emotion's shapes down toward 0 too:
                _applyEmotionToBase(rampFrom.name, _lerp(rampFrom.intensity, 0, t));
            }
        }

        // 2) Update baseShapes from currentEmotion + flush.
        _applyEmotionToBase(currentEmotion.name, currentEmotion.intensity);

        // 3) Placeholder colour (no-op if GLB took over).
        _placeholderColour(currentEmotion.name, currentEmotion.intensity);

        // 4) Push every touched target.
        // Even when the GLB has no morph targets we still flush — the
        // bone-animator fallback (jawOpen, eyeLook*, eyeBlink) is wired
        // through _writeTarget so it picks up the additive shapes.
        const keys = new Set([...baseShapes.keys(), ...additiveShapes.keys()]);
        for (const name of keys) _flushTarget(name);

        // 5) Bone-only emotion overlay — small head tilt per emotion.
        // Skipped silently when morphs are available (we don't want
        // to double-dip with both a smile-morph and a head-droop).
        if (!hasMorphs()) {
            const ba = window.LexyAvatar && window.LexyAvatar.boneAnimator;
            if (ba && ba.hasBones && ba.hasBones()) {
                ba.applyEmotionExpression(
                    currentEmotion.name, currentEmotion.intensity
                );
            }
        }
    }

    function _applyEmotionToBase(emotionName, intensity) {
        const presets = window.LexyAvatar && window.LexyAvatar.morphs
            ? window.LexyAvatar.morphs.EMOTION_PRESETS
            : null;
        if (!presets) return;
        // First, zero out any base shape that was used by the previous
        // emotion but isn't part of the new one — so a switch from
        // "thinking" (browDown) to "happy" (smile) actually relaxes the
        // brow instead of leaving it stuck.
        baseShapes.forEach((_v, key) => baseShapes.set(key, 0));

        const preset = presets[emotionName] || presets.neutral;
        for (const [shape, weightAt1] of Object.entries(preset)) {
            baseShapes.set(shape, weightAt1 * intensity);
        }
    }

    // ── Public shape setters (used by idle / lip-sync) ─────────────

    function setShape(name, weight) {
        // Additive layer — clamped to [0, 1]; combined with the emotion-
        // driven base in the tick loop.
        const clamped = Math.max(0, Math.min(1, Number(weight) || 0));
        additiveShapes.set(name, clamped);
        if (hasMorphs()) _flushTarget(name);
    }

    function getShape(name) {
        const base = baseShapes.get(name) || 0;
        const add = additiveShapes.get(name) || 0;
        return base + add;
    }

    function clearAdditive(name) {
        if (additiveShapes.has(name)) {
            additiveShapes.set(name, 0);
            if (hasMorphs()) _flushTarget(name);
        }
    }

    function snapshot() {
        return {
            name: currentEmotion.name,
            intensity: currentEmotion.intensity,
            mode: hasMorphs() ? "morph" : "placeholder",
        };
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.emotion = {
        init,
        setHandle,
        apply,
        tick,
        setShape,
        getShape,
        clearAdditive,
        snapshot,
        hasMorphs,
    };
})();
