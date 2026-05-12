/*
 * Avatar World — Frontend bootstrap.
 *
 * Order of operations:
 *   1. Wait for DOM ready.
 *   2. Find the <canvas id="lexy-avatar-canvas">. If missing → bail.
 *   3. Wait briefly for BABYLON to appear. If still absent → show a
 *      placeholder hint in the canvas and stop. Lexy keeps working.
 *   4. Create the Babylon scene (Phase 0 placeholder).
 *   5. Try to load the configured avatar GLB. On 404/no-file we keep
 *      the placeholder capsule visible.
 *   6. Init each driver (emotion / idle / lip-sync) with whatever
 *      handle we ended up with (real GLB or null).
 *   7. Register every avatar.* WS topic.
 *   8. Start the per-frame tick loop via scene.onBeforeRenderObservable.
 *   9. Ask the backend for an initial state snapshot.
 */
(() => {
    "use strict";

    const CANVAS_ID = "lexy-avatar-canvas";
    const BABYLON_WAIT_MS = 3000;
    const BABYLON_POLL_MS = 100;

    function waitForBabylon() {
        return new Promise((resolve) => {
            if (typeof window.BABYLON !== "undefined") {
                resolve(true);
                return;
            }
            const deadline = performance.now() + BABYLON_WAIT_MS;
            const tick = () => {
                if (typeof window.BABYLON !== "undefined") {
                    resolve(true);
                    return;
                }
                if (performance.now() > deadline) {
                    resolve(false);
                    return;
                }
                setTimeout(tick, BABYLON_POLL_MS);
            };
            setTimeout(tick, BABYLON_POLL_MS);
        });
    }

    function showPlaceholderHint(canvas) {
        const ctx = canvas.getContext && canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = canvas.clientWidth || 320;
        canvas.height = canvas.clientHeight || 200;
        ctx.fillStyle = "rgba(8, 10, 16, 0.85)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#7ac1ff";
        ctx.font = "14px ui-monospace, monospace";
        ctx.fillText("Avatar World: BABYLON not loaded.", 16, 30);
        ctx.fillStyle = "#aaa";
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText("Drop babylon.js into /static/vendor/ and reload.", 16, 52);
        ctx.fillText("See frontend/static/avatar-world/assets/README.md", 16, 70);
    }

    async function _tryLoadAvatar(scene) {
        try {
            // loadAvatarAuto walks a fallback chain: lexy_base.glb →
            // lexy_base.gltf → base_female1.gltf → base_female1.glb.
            // First success wins; null on miss-everywhere.
            return await window.LexyAvatar.loader.loadAvatarAuto(scene, {
                position: { x: 0, y: 0, z: 0 },
                scale: 1.0,
            });
        } catch (err) {
            console.warn("avatar-world: loadAvatar threw", err);
            return null;
        }
    }

    function _wireWS(scene, refs) {
        const net = window.LexyAvatar.net;

        net.register("avatar.state", (payload) => {
            if (!payload) return;
            if (payload.emotion) window.LexyAvatar.emotion.apply(payload.emotion);
            if (payload.activity && window.LexyAvatar.idle) {
                window.LexyAvatar.idle.setActivity(payload.activity);
            }
            if (payload.background_id && window.LexyAvatar.background) {
                window.LexyAvatar.background.setBackground(payload.background_id, 0);
            }
            if (payload.time_of_day && window.LexyAvatar.lighting) {
                window.LexyAvatar.lighting.applyBucket(payload.time_of_day);
            }
            // Surface the snapshot to anything else listening.
            refs.lastState = payload;
        });

        net.register("avatar.emotion", (payload) => {
            window.LexyAvatar.emotion.apply(payload);
        });

        net.register("avatar.speaking", (payload) => {
            if (payload && payload.state === "start") {
                window.LexyAvatar.lipSync.onSpeakStart(payload);
            } else {
                window.LexyAvatar.lipSync.onSpeakEnd(payload);
            }
        });

        net.register("avatar.attention", (payload) => {
            // Phase 1A: idle gaze randomisation is enough. Phase 2 will
            // bias the gaze ramp toward (camera | screen | window).
            if (window.LexyAvatar.idle) {
                window.LexyAvatar.idle.setActivity(refs.lastActivity || "sit_desk");
            }
            console.debug("avatar.attention", payload && payload.look_at);
        });

        net.register("avatar.activity", (payload) => {
            const id = payload && payload.id;
            refs.lastActivity = id || refs.lastActivity;
            if (window.LexyAvatar.idle) {
                window.LexyAvatar.idle.setActivity(id || "sit_desk");
            }
        });

        net.register("avatar.outfit", (payload) => {
            console.debug("avatar.outfit", payload && payload.outfit);
        });

        net.register("avatar.background", (payload) => {
            const id = payload && payload.id;
            if (id && window.LexyAvatar.background) {
                window.LexyAvatar.background.setBackground(id, payload.fade_ms || 2000);
            }
        });

        net.register("avatar.view_mode", (payload) => {
            if (window.LexyAvatar.viewMode && payload && payload.mode) {
                window.LexyAvatar.viewMode.set(payload.mode);
            }
        });
    }

    function _startTickLoop(scene) {
        let lastNow = performance.now();
        scene.onBeforeRenderObservable.add(() => {
            const now = performance.now();
            const dtMs = Math.max(1, now - lastNow);
            lastNow = now;
            // emotion writes additive shapes into morph-or-bone targets;
            // then lip-sync stacks jawOpen on top; idle adds blinks /
            // gaze / breathing. Finally the bone-animator and lighting
            // tick to flush their per-frame lerps.
            if (window.LexyAvatar.emotion) window.LexyAvatar.emotion.tick(dtMs);
            if (window.LexyAvatar.idle)    window.LexyAvatar.idle.tick(now, dtMs);
            if (window.LexyAvatar.lipSync && window.LexyAvatar.lipSync.tick) {
                window.LexyAvatar.lipSync.tick(now, dtMs);
            }
            if (window.LexyAvatar.boneAnimator && window.LexyAvatar.boneAnimator.tick) {
                window.LexyAvatar.boneAnimator.tick(dtMs);
            }
            if (window.LexyAvatar.lighting && window.LexyAvatar.lighting.tick) {
                window.LexyAvatar.lighting.tick();
            }
        });
    }

    async function boot() {
        const canvas = document.getElementById(CANVAS_ID);
        if (!canvas) {
            console.warn("avatar-world: no #" + CANVAS_ID + " on the page — skipping");
            return;
        }

        const ok = await waitForBabylon();
        if (!ok) {
            console.warn(
                "avatar-world: BABYLON is missing — see "
                + "frontend/static/avatar-world/assets/README.md"
            );
            showPlaceholderHint(canvas);
            return;
        }

        const sceneRefs = window.LexyAvatar.scene.createScene(canvas);
        if (!sceneRefs) {
            showPlaceholderHint(canvas);
            return;
        }

        // Try to swap the placeholder for a real GLB. The loader is
        // forgiving — if the file isn't dropped in yet we keep the
        // capsule and the drivers still work via placeholder mode.
        const handle = await _tryLoadAvatar(sceneRefs.scene);
        if (handle) {
            // Hide the capsule but keep it around for the colour-tint
            // fallback (useful while a model lacks an ARKit set).
            if (sceneRefs.placeholder) {
                sceneRefs.placeholder.setEnabled(false);
            }
            const morphCount = handle.morphTargets ? handle.morphTargets.size : 0;
            const boneCount = handle.bones ? handle.bones.size : 0;
            console.info(
                `avatar-world: GLB loaded — morphs=${morphCount} bones=${boneCount}`
            );
            // Initialise the bone-animator with the loaded skeleton so
            // emotion_driver can route ARKit shapes onto bones when
            // there are no morph targets (MakeHuman / Mixamo case).
            if (window.LexyAvatar.boneAnimator) {
                window.LexyAvatar.boneAnimator.init(handle, sceneRefs.scene);
            }
        } else {
            console.info("avatar-world: running in placeholder mode");
        }

        // Boot drivers with whatever we ended up with.
        window.LexyAvatar.emotion.init({
            scene: sceneRefs.scene,
            placeholder: sceneRefs.placeholder,
            handle,
        });
        window.LexyAvatar.idle.init({
            scene: sceneRefs.scene,
            placeholder: sceneRefs.placeholder,
            handle,
        });
        window.LexyAvatar.lipSync.init({
            scene: sceneRefs.scene,
            placeholder: sceneRefs.placeholder,
            speakAura: sceneRefs.speakAura,
            handle,
        });
        if (window.LexyAvatar.outfit) {
            window.LexyAvatar.outfit.init(handle);
        }
        if (window.LexyAvatar.background) {
            window.LexyAvatar.background.init({
                scene: sceneRefs.scene,
                mountNode: sceneRefs.windowAnchor,
            });
        }
        if (window.LexyAvatar.lighting) {
            window.LexyAvatar.lighting.init({
                scene: sceneRefs.scene,
                hemi: sceneRefs.lights && sceneRefs.lights.hemi,
                key:  sceneRefs.lights && sceneRefs.lights.key,
                lamp: sceneRefs.lights && sceneRefs.lights.lamp,
            });
        }

        const refs = { lastState: null, lastActivity: "sit_desk" };
        _wireWS(sceneRefs.scene, refs);
        _startTickLoop(sceneRefs.scene);

        // Ask the backend for the canonical snapshot once the socket
        // is up — we poll briefly because app.js opens the WS on its
        // own clock and we'd rather be a little late than miss it.
        const ensureSnapshot = () => {
            if (window.LexyAvatar.net.send("get_state")) return;
            setTimeout(ensureSnapshot, 250);
        };
        ensureSnapshot();

        console.info("avatar-world: ready");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
