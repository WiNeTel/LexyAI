/*
 * Avatar World — GLB loader.
 *
 * Tries to load the configured avatar GLB from /static/avatar-world/assets/
 * models/. The loader is forgiving — when the file isn't there yet (no
 * Ready Player Me export dropped in) it returns `null` and the scene
 * falls back to the placeholder capsule. This is intentional: we want
 * the avatar layer to work end-to-end even before assets exist.
 *
 * Public surface:
 *   - loadAvatar(scene, { url, position, scale }) → Promise<AvatarHandle | null>
 *
 * AvatarHandle:
 *   - root            transform node containing every loaded mesh
 *   - meshes          list of imported meshes
 *   - skeleton        first skeleton (Ready Player Me has exactly one)
 *   - morphTargets    Map<targetName, BABYLON.MorphTarget>  (drives ARKit shapes)
 *   - bones           Map<boneName, BABYLON.TransformNode>  (eyes / head for gaze)
 *   - dispose()       tear-down helper
 */
(() => {
    "use strict";

    const DEFAULT_URL = "/static/avatar-world/assets/models/lexy_base.glb";
    // Fallback chain tried by `loadAvatarAuto`. First hit wins. The
    // chain lets the user drop in a Ready Player Me / Avaturn GLB
    // without editing code — and falls back to whatever MakeHuman /
    // Mixamo / Avaturn export is already in the folder.
    const FALLBACK_URLS = [
        DEFAULT_URL,
        "/static/avatar-world/assets/models/lexy_base.gltf",
        "/static/avatar-world/assets/models/base_female1.gltf",
        "/static/avatar-world/assets/models/base_female1.glb",
    ];

    function _collectMorphTargets(meshes) {
        // Ready Player Me wraps every blendshape on the Wolf3D_Head /
        // Wolf3D_Teeth / Wolf3D_*-named meshes. We merge them into a
        // single name→target map; if two meshes share a target name
        // (eyeBlinkLeft on Head + Lashes) the *first* wins for setting
        // and we drive the duplicate via a parallel reference.
        const map = new Map();
        for (const mesh of meshes || []) {
            const mtm = mesh && mesh.morphTargetManager;
            if (!mtm) continue;
            for (let i = 0; i < mtm.numTargets; i++) {
                const target = mtm.getTarget(i);
                if (!target || !target.name) continue;
                if (!map.has(target.name)) {
                    map.set(target.name, []);
                }
                map.get(target.name).push(target);
            }
        }
        return map;
    }

    function _collectBones(skeletons) {
        const map = new Map();
        for (const skel of skeletons || []) {
            for (const bone of skel.bones || []) {
                if (bone && bone.name && !map.has(bone.name)) {
                    map.set(bone.name, bone);
                }
            }
        }
        return map;
    }

    async function loadAvatar(scene, opts = {}) {
        if (!scene || typeof window.BABYLON === "undefined") {
            console.warn("avatar.loader: BABYLON unavailable");
            return null;
        }
        const B = window.BABYLON;
        if (!B.SceneLoader || typeof B.SceneLoader.ImportMeshAsync !== "function") {
            console.warn("avatar.loader: BABYLON.SceneLoader.ImportMeshAsync "
                + "missing — babylonjs.loaders.min.js failed to load. Check "
                + "<script> tags in index.html and the network tab.");
            return null;
        }
        // Babylon's GLB plugin registers itself when the loaders bundle is
        // loaded — bail loudly if it's missing instead of silently rendering
        // an empty room.
        if (typeof B.GLTFFileLoader === "undefined" && B.SceneLoaderFlags) {
            console.warn("avatar.loader: GLTFFileLoader plugin not registered — "
                + "loaders bundle is loaded but the GLB extension isn't. "
                + "Check the browser console for earlier errors.");
        }

        const url = opts.url || DEFAULT_URL;
        const lastSlash = url.lastIndexOf("/");
        const rootUrl = url.substring(0, lastSlash + 1);
        const sceneFile = url.substring(lastSlash + 1);
        console.debug("avatar.loader: attempting " + url);

        let result;
        try {
            result = await B.SceneLoader.ImportMeshAsync(
                null, rootUrl, sceneFile, scene
            );
        } catch (err) {
            // Silent on 404 — assets are user-supplied. Log everything
            // else so a corrupted GLB or path typo is visible.
            const msg = String(err && err.message || err);
            if (/404|not.found|failed to fetch/i.test(msg)) {
                console.info("avatar.loader: " + url + " not found (404)");
            } else {
                console.warn("avatar.loader: " + url + " load failed — " + msg, err);
            }
            return null;
        }

        const meshes = result.meshes || [];
        if (meshes.length === 0) {
            console.warn("avatar.loader: GLB has no meshes — bailing");
            return null;
        }

        // Group everything under a single root for easy translate/scale.
        const root = new B.TransformNode("avatar-root", scene);
        const rootMesh = meshes.find((m) => m && m.name === "__root__") || meshes[0];
        rootMesh.parent = root;

        if (opts.position) {
            root.position.set(opts.position.x || 0, opts.position.y || 0, opts.position.z || 0);
        }
        if (opts.scale) {
            const s = Number(opts.scale) || 1.0;
            root.scaling.set(s, s, s);
        }

        const morphTargets = _collectMorphTargets(meshes);
        const bones = _collectBones(result.skeletons);

        if (morphTargets.size === 0) {
            console.warn(
                "avatar.loader: GLB has no morph targets — emotion/lip-sync "
                + "will only drive placeholder color. Re-export the avatar "
                + "with ARKit blendshapes enabled."
            );
        }

        // Auto-start every animation group that came with the asset
        // (idle / walk / pose loops). CesiumMan, Michelle, Soldier etc.
        // each ship one walk/idle clip; without this they pose-frozen
        // and the user reads it as "the avatar doesn't move". Looping
        // is on because none of these clips were authored for
        // one-shot playback — they're all idle cycles.
        const animationGroups = result.animationGroups || [];
        for (const ag of animationGroups) {
            try {
                ag.start(true);  // true = loop
            } catch (err) {
                console.warn("avatar.loader: failed to start animation",
                    ag && ag.name, err);
            }
        }
        if (animationGroups.length > 0) {
            console.info(
                "avatar.loader: started " + animationGroups.length
                + " animation group(s): "
                + animationGroups.map((a) => a && a.name).join(", ")
            );
        }

        return {
            root,
            meshes,
            skeleton: (result.skeletons || [])[0] || null,
            morphTargets,
            bones,
            animationGroups,
            dispose() {
                for (const ag of animationGroups) {
                    try { ag.stop(); ag.dispose(); } catch (_) { /* ignore */ }
                }
                for (const mesh of meshes) {
                    if (mesh && typeof mesh.dispose === "function") {
                        try { mesh.dispose(false, true); } catch (_) { /* ignore */ }
                    }
                }
                if (root && typeof root.dispose === "function") {
                    root.dispose();
                }
            },
        };
    }

    async function loadAvatarAuto(scene, opts = {}) {
        // Try the configured fallback chain until something loads.
        const candidates = (opts.urls && opts.urls.length)
            ? opts.urls
            : FALLBACK_URLS;
        console.info("avatar.loader: trying candidates", candidates);
        for (const url of candidates) {
            const handle = await loadAvatar(scene, { ...opts, url });
            if (handle) {
                console.info("avatar.loader: loaded", url);
                return handle;
            }
        }
        console.warn(
            "avatar.loader: ALL candidates failed. "
            + "Check Network tab for the actual responses."
        );
        return null;
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.loader = {
        loadAvatar,
        loadAvatarAuto,
        DEFAULT_URL,
        FALLBACK_URLS,
    };
})();
