/*
 * Avatar World — Background (Skybox / window backplate) manager.
 *
 * Phase 1C builds an outdoors plate visible through the apartment
 * window. Two strategies coexist:
 *
 *   1. **Image plate** — a textured plane parented to the window
 *      frame. Fast, always works, used when a JPG is dropped into
 *      `assets/backgrounds/<id>.jpg`.
 *
 *   2. **Procedural sky** — a Babylon `SkyMaterial` on a large box;
 *      kicks in when an id maps to a known procedural preset (e.g.
 *      `proc_sunny`). Phase 1C ships one procedural preset for
 *      missing images; richer skies stay for later.
 *
 * Crossfade between presets is done by ramping the plate's alpha
 * from 1 → 0 on the old plate while the new one fades in.
 *
 * Public surface:
 *   - init({ scene, mountNode })
 *   - setBackground(id, fadeMs)
 *   - currentId() → string
 */
(() => {
    "use strict";

    const ASSET_BASE = "/static/avatar-world/assets/backgrounds/";

    const refs = {
        scene: null,
        mountNode: null,
        activePlate: null,
        fadeTimer: null,
    };

    let currentId = "";

    // Procedural fallbacks per id when the JPG is missing. RGB pairs
    // describe a top→bottom gradient on a stand-in skybox.
    const FALLBACK_GRADIENTS = {
        city_morning:  [[0.95, 0.78, 0.58], [0.85, 0.55, 0.40]],
        city_day:      [[0.55, 0.78, 0.95], [0.85, 0.92, 0.98]],
        city_evening:  [[0.32, 0.20, 0.45], [0.95, 0.55, 0.30]],
        city_night:    [[0.04, 0.06, 0.16], [0.10, 0.13, 0.24]],
        forest:        [[0.55, 0.78, 0.55], [0.20, 0.42, 0.22]],
        mountain:      [[0.75, 0.85, 0.95], [0.35, 0.45, 0.55]],
        rain:          [[0.35, 0.40, 0.48], [0.20, 0.25, 0.32]],
    };

    function init(opts) {
        refs.scene = (opts && opts.scene) || null;
        refs.mountNode = (opts && opts.mountNode) || null;
    }

    function _makePlateWithImage(scene, url) {
        const B = window.BABYLON;
        const plane = B.MeshBuilder.CreatePlane(
            "bg-plate",
            { width: 4.0, height: 2.4 },
            scene,
        );
        plane.position = new B.Vector3(0, 1.6, -3.0);
        plane.parent = refs.mountNode || null;

        const mat = new B.StandardMaterial("bg-plate-mat", scene);
        const tex = new B.Texture(url, scene, true, false);
        tex.onLoadObservable.addOnce(() => {
            tex.uOffset = 0; tex.vOffset = 0;
        });
        mat.diffuseTexture = tex;
        mat.emissiveTexture = tex;     // glow a bit so it's visible even
        mat.disableLighting = true;    // when the room is dark
        mat.alpha = 0.0;
        plane.material = mat;
        return plane;
    }

    function _makePlateWithGradient(scene, id) {
        const B = window.BABYLON;
        const gradient = FALLBACK_GRADIENTS[id] || FALLBACK_GRADIENTS.city_day;
        const plane = B.MeshBuilder.CreatePlane(
            "bg-plate-grad",
            { width: 4.0, height: 2.4 },
            scene,
        );
        plane.position = new B.Vector3(0, 1.6, -3.0);
        plane.parent = refs.mountNode || null;

        const mat = new B.StandardMaterial("bg-plate-grad-mat", scene);
        // Build a procedural top→bottom gradient via a dynamic texture.
        const dt = new B.DynamicTexture("bg-grad-dt", { width: 4, height: 256 }, scene, false);
        const ctx = dt.getContext();
        const grad = ctx.createLinearGradient(0, 0, 0, 256);
        const top = gradient[0];
        const bot = gradient[1];
        grad.addColorStop(0, `rgb(${(top[0] * 255) | 0},${(top[1] * 255) | 0},${(top[2] * 255) | 0})`);
        grad.addColorStop(1, `rgb(${(bot[0] * 255) | 0},${(bot[1] * 255) | 0},${(bot[2] * 255) | 0})`);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 4, 256);
        dt.update();
        mat.diffuseTexture = dt;
        mat.emissiveTexture = dt;
        mat.disableLighting = true;
        mat.alpha = 0.0;
        plane.material = mat;
        return plane;
    }

    function _fadeOut(plate, durationMs) {
        if (!plate || !plate.material) return;
        const start = performance.now();
        const startAlpha = plate.material.alpha;
        const ms = Math.max(1, Number(durationMs) || 0);
        function step() {
            const t = Math.min(1, (performance.now() - start) / ms);
            plate.material.alpha = startAlpha * (1 - t);
            if (t < 1) {
                requestAnimationFrame(step);
            } else if (plate.dispose) {
                try { plate.dispose(); } catch (_) { /* ignore */ }
            }
        }
        requestAnimationFrame(step);
    }

    function _fadeIn(plate, durationMs) {
        if (!plate || !plate.material) return;
        const start = performance.now();
        const ms = Math.max(1, Number(durationMs) || 0);
        function step() {
            const t = Math.min(1, (performance.now() - start) / ms);
            plate.material.alpha = t;
            if (t < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    function setBackground(id, fadeMs) {
        if (!refs.scene || !window.BABYLON) return;
        const target = String(id || "");
        if (!target || target === currentId) return;

        const fade = (fadeMs == null) ? 1800 : Math.max(0, Number(fadeMs));
        const url = ASSET_BASE + target + ".jpg";

        // Try the image first; on 404 the Babylon texture loader logs
        // a network error and shows transparent — we still keep the
        // gradient fallback as a separate mesh underneath.
        const fallback = _makePlateWithGradient(refs.scene, target);
        _fadeIn(fallback, fade);

        const img = _makePlateWithImage(refs.scene, url);
        // Move the image plate slightly forward so it occludes the
        // gradient when (and only if) it actually loaded.
        img.position.z = -2.99;
        if (img.material && img.material.diffuseTexture) {
            img.material.diffuseTexture.onLoadObservable.addOnce(() => {
                _fadeIn(img, fade);
            });
        }

        // Fade out the previous plate(s).
        if (refs.activePlate) {
            _fadeOut(refs.activePlate.image, fade);
            _fadeOut(refs.activePlate.gradient, fade);
        }

        refs.activePlate = { image: img, gradient: fallback };
        currentId = target;
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.background = {
        init,
        setBackground,
        currentId: () => currentId,
    };
})();
