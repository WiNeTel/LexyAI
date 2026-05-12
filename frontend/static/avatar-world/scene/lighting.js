/*
 * Avatar World — Time-of-day lighting presets.
 *
 * Five buckets matching the backend's `time_of_day` enum. Each preset
 * sets a hemispheric (ambient) light, a directional "sun"/lamp key
 * light, and a "lamp" point light that switches on for evening/night.
 *
 * Public surface:
 *   - init({ scene, hemi, key, lamp })   // light refs from apartment.js
 *   - applyBucket(bucket, durationMs)    // lerp toward the preset
 */
(() => {
    "use strict";

    // [r, g, b, intensity] for each light. Tuned so the same avatar
    // looks readable across all buckets — no preset cranks anyone to
    // black or pure white.
    const PRESETS = {
        morning: {
            hemi:   { rgb: [1.00, 0.92, 0.80], intensity: 0.65 },
            key:    { rgb: [1.00, 0.86, 0.62], intensity: 0.85, direction: [-0.6, -1.0, -0.3] },
            lamp:   { intensity: 0.0 },
        },
        midday: {
            hemi:   { rgb: [1.00, 0.98, 0.95], intensity: 0.85 },
            key:    { rgb: [1.00, 1.00, 0.98], intensity: 0.95, direction: [-0.3, -1.2, -0.2] },
            lamp:   { intensity: 0.0 },
        },
        afternoon: {
            hemi:   { rgb: [1.00, 0.96, 0.88], intensity: 0.78 },
            key:    { rgb: [1.00, 0.92, 0.78], intensity: 0.90, direction: [-0.8, -1.0, -0.3] },
            lamp:   { intensity: 0.0 },
        },
        evening: {
            hemi:   { rgb: [0.70, 0.65, 0.80], intensity: 0.40 },
            key:    { rgb: [0.95, 0.62, 0.45], intensity: 0.55, direction: [-1.0, -0.6, -0.4] },
            lamp:   { rgb: [1.00, 0.78, 0.50], intensity: 0.75 },
        },
        night: {
            hemi:   { rgb: [0.30, 0.36, 0.55], intensity: 0.22 },
            key:    { rgb: [0.35, 0.45, 0.70], intensity: 0.25, direction: [-0.3, -1.0, -0.3] },
            lamp:   { rgb: [1.00, 0.85, 0.62], intensity: 0.85 },
        },
    };

    const refs = { scene: null, hemi: null, key: null, lamp: null };

    let currentBucket = "midday";
    let animation = null;   // { from, to, startedAt, durationMs }

    function _color(rgb) {
        const B = window.BABYLON;
        return new B.Color3(rgb[0], rgb[1], rgb[2]);
    }

    function _lerp(a, b, t) { return a + (b - a) * t; }

    function _snapshot() {
        if (!refs.hemi || !refs.key) return null;
        const lampIntensity = refs.lamp ? refs.lamp.intensity : 0;
        return {
            hemi: {
                rgb: [refs.hemi.diffuse.r, refs.hemi.diffuse.g, refs.hemi.diffuse.b],
                intensity: refs.hemi.intensity,
            },
            key: {
                rgb: [refs.key.diffuse.r, refs.key.diffuse.g, refs.key.diffuse.b],
                intensity: refs.key.intensity,
                direction: refs.key.direction
                    ? [refs.key.direction.x, refs.key.direction.y, refs.key.direction.z]
                    : [-0.3, -1.0, -0.2],
            },
            lamp: { intensity: lampIntensity },
        };
    }

    function _applyImmediate(preset) {
        if (refs.hemi) {
            refs.hemi.diffuse = _color(preset.hemi.rgb);
            refs.hemi.groundColor = _color([
                preset.hemi.rgb[0] * 0.6,
                preset.hemi.rgb[1] * 0.6,
                preset.hemi.rgb[2] * 0.7,
            ]);
            refs.hemi.intensity = preset.hemi.intensity;
        }
        if (refs.key) {
            refs.key.diffuse = _color(preset.key.rgb);
            refs.key.intensity = preset.key.intensity;
            if (preset.key.direction && refs.key.direction) {
                const d = preset.key.direction;
                refs.key.direction.set(d[0], d[1], d[2]);
            }
        }
        if (refs.lamp) {
            if (preset.lamp.rgb) refs.lamp.diffuse = _color(preset.lamp.rgb);
            refs.lamp.intensity = preset.lamp.intensity;
            // Lamp is "off" when intensity drops below a tiny epsilon.
            if (preset.lamp.intensity > 0.01) refs.lamp.setEnabled(true);
            else if (refs.lamp.isEnabled()) refs.lamp.setEnabled(false);
        }
    }

    function init(opts) {
        refs.scene = (opts && opts.scene) || null;
        refs.hemi = (opts && opts.hemi) || null;
        refs.key = (opts && opts.key) || null;
        refs.lamp = (opts && opts.lamp) || null;
        // Apply the default bucket immediately so the first frame
        // is already lit correctly.
        _applyImmediate(PRESETS[currentBucket] || PRESETS.midday);
    }

    function applyBucket(bucket, durationMs) {
        if (!PRESETS[bucket]) return;
        if (bucket === currentBucket && animation == null) return;
        const ms = (durationMs == null) ? 1800 : Math.max(0, Number(durationMs));
        const from = _snapshot();
        if (!from) {
            _applyImmediate(PRESETS[bucket]);
            currentBucket = bucket;
            return;
        }
        animation = {
            from,
            to: PRESETS[bucket],
            startedAt: performance.now(),
            durationMs: ms,
            target: bucket,
        };
        // If we want an instant change, just apply now.
        if (ms === 0) {
            _applyImmediate(PRESETS[bucket]);
            currentBucket = bucket;
            animation = null;
        }
    }

    function tick() {
        if (!animation) return;
        const t = Math.min(
            1,
            (performance.now() - animation.startedAt) / animation.durationMs,
        );
        const from = animation.from;
        const to = animation.to;
        if (refs.hemi) {
            refs.hemi.diffuse.r = _lerp(from.hemi.rgb[0], to.hemi.rgb[0], t);
            refs.hemi.diffuse.g = _lerp(from.hemi.rgb[1], to.hemi.rgb[1], t);
            refs.hemi.diffuse.b = _lerp(from.hemi.rgb[2], to.hemi.rgb[2], t);
            refs.hemi.intensity = _lerp(from.hemi.intensity, to.hemi.intensity, t);
        }
        if (refs.key) {
            refs.key.diffuse.r = _lerp(from.key.rgb[0], to.key.rgb[0], t);
            refs.key.diffuse.g = _lerp(from.key.rgb[1], to.key.rgb[1], t);
            refs.key.diffuse.b = _lerp(from.key.rgb[2], to.key.rgb[2], t);
            refs.key.intensity = _lerp(from.key.intensity, to.key.intensity, t);
        }
        if (refs.lamp) {
            const lampI = _lerp(from.lamp.intensity, to.lamp.intensity, t);
            refs.lamp.intensity = lampI;
            if (lampI > 0.01 && !refs.lamp.isEnabled()) refs.lamp.setEnabled(true);
            if (lampI <= 0.01 && refs.lamp.isEnabled()) refs.lamp.setEnabled(false);
            if (to.lamp.rgb) {
                refs.lamp.diffuse.r = _lerp(from.hemi.rgb[0], to.lamp.rgb[0], t);
                refs.lamp.diffuse.g = _lerp(from.hemi.rgb[1], to.lamp.rgb[1], t);
                refs.lamp.diffuse.b = _lerp(from.hemi.rgb[2], to.lamp.rgb[2], t);
            }
        }
        if (t >= 1) {
            currentBucket = animation.target;
            animation = null;
        }
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.lighting = {
        init,
        applyBucket,
        tick,
        currentBucket: () => currentBucket,
    };
})();
