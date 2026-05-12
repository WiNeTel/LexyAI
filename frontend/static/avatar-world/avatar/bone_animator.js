/*
 * Avatar World — Bone-based animator (Phase 1A fallback).
 *
 * Used when a loaded GLB has a skeleton but no ARKit blendshapes —
 * typical of MakeHuman / Mixamo / Auto-Rig exports. The animator maps
 * a subset of ARKit shape names onto bone rotations so the rest of the
 * stack (emotion_driver, idle_driver, lip_sync) stays unchanged:
 *
 *   jawOpen                  → jaw bone, rotate around X (mouth opens)
 *   eyeLookIn/Out{L,R}       → eye.L / eye.R bone, rotate around Y
 *   eyeLookUp/Down{L,R}      → eye.L / eye.R bone, rotate around X
 *   eyeBlinkLeft/Right       → no clean bone-only blink — we shrink
 *                              the eye-mesh on Y as a stand-in
 *   mouthSmile / brow* etc.  → no bone mapping → silently ignored
 *
 * Plus a separate emotion overlay that nudges the head bone for the
 * 5 emotion presets (tiny tilt for thinking, lift for surprised, droop
 * for tired …). emotion_driver calls applyEmotionExpression().
 *
 * Per-tick the animator lerps every modified bone from its current
 * rotation toward the target. Lerp is fast (alpha 0.4) — the per-frame
 * smoothing already gives a soft feel.
 *
 * Public surface:
 *   - init(handle, scene)
 *   - setShape(name, weight)
 *   - applyEmotionExpression(name, intensity)
 *   - tick(dtMs)
 *   - canHandle(name) → bool
 *   - hasBones() → bool
 */
(() => {
    "use strict";

    const SHAPES = (window.LexyAvatar && window.LexyAvatar.morphs)
        ? window.LexyAvatar.morphs.SHAPES
        : null;

    // Magnitudes — in radians at weight=1.0. Tuned to feel "present"
    // without crossing into uncanny territory.
    const JAW_OPEN_MAX_RAD       = Math.PI / 9;     // ~20°
    const EYE_LOOK_HORIZ_MAX_RAD = Math.PI / 10;    // ~18°
    const EYE_LOOK_VERT_MAX_RAD  = Math.PI / 14;    // ~12.8°
    const HEAD_TILT_MAX_RAD      = Math.PI / 22;    // ~8°
    // Smoothing factor toward target rotation per frame.
    const LERP_ALPHA = 0.4;

    // Common bone-name aliases — different exporters spell the same
    // bone different ways. The resolver tries each variant in order.
    const BONE_ALIASES = {
        jaw:   ["jaw", "Jaw", "jaw_M", "DEF-jaw"],
        eyeL:  ["eye.L", "eyeL", "eye_L", "leftEye", "LeftEye", "Eye.L"],
        eyeR:  ["eye.R", "eyeR", "eye_R", "rightEye", "RightEye", "Eye.R"],
        head:  ["head", "Head", "head_M", "DEF-head"],
        spine: ["spine04", "spine03", "spine02", "spine", "Spine", "chest"],
    };

    const refs = {
        scene: null,
        handle: null,
    };

    // Bone references resolved once on init.
    const bones = {
        jaw: null,
        eyeL: null,
        eyeR: null,
        head: null,
        spine: null,
    };

    // Rest-pose rotations (Euler). Captured on init so we can drive
    // bones as "rest + delta" without losing the original facing.
    const rest = {
        jaw: null,
        eyeL: null,
        eyeR: null,
        head: null,
    };

    // Targets per "channel" (axis-resolved deltas relative to rest).
    const targetDelta = {
        jaw:  { x: 0, y: 0, z: 0 },
        eyeL: { x: 0, y: 0, z: 0 },
        eyeR: { x: 0, y: 0, z: 0 },
        head: { x: 0, y: 0, z: 0 },
    };
    const currentDelta = {
        jaw:  { x: 0, y: 0, z: 0 },
        eyeL: { x: 0, y: 0, z: 0 },
        eyeR: { x: 0, y: 0, z: 0 },
        head: { x: 0, y: 0, z: 0 },
    };

    // Eye-mesh refs for the blink fallback — scaling the mesh on Y is
    // an ugly but visible stand-in for an eyelid blendshape.
    let eyeBlinkAmount = 0;       // 0 = open, 1 = closed
    const eyeMeshes = [];

    function _findBone(skeleton, candidates) {
        if (!skeleton || !skeleton.bones) return null;
        for (const name of candidates) {
            // Try exact match first.
            for (const b of skeleton.bones) {
                if (b && b.name === name) return b;
            }
        }
        // Looser: case-insensitive contains.
        const lower = candidates.map((c) => c.toLowerCase());
        for (const b of skeleton.bones) {
            if (!b || !b.name) continue;
            const ln = b.name.toLowerCase();
            for (const cand of lower) {
                if (ln === cand || ln.includes(cand)) return b;
            }
        }
        return null;
    }

    function _captureRest(bone) {
        if (!bone) return null;
        try {
            const r = bone.getRotation(window.BABYLON.Space.LOCAL);
            return { x: r.x, y: r.y, z: r.z };
        } catch (_) {
            return { x: 0, y: 0, z: 0 };
        }
    }

    function init(handle, scene) {
        refs.scene = scene || null;
        refs.handle = handle || null;
        const skel = handle && handle.skeleton;
        if (!skel) {
            bones.jaw = bones.eyeL = bones.eyeR = bones.head = bones.spine = null;
            return;
        }

        bones.jaw   = _findBone(skel, BONE_ALIASES.jaw);
        bones.eyeL  = _findBone(skel, BONE_ALIASES.eyeL);
        bones.eyeR  = _findBone(skel, BONE_ALIASES.eyeR);
        bones.head  = _findBone(skel, BONE_ALIASES.head);
        bones.spine = _findBone(skel, BONE_ALIASES.spine);

        rest.jaw  = _captureRest(bones.jaw);
        rest.eyeL = _captureRest(bones.eyeL);
        rest.eyeR = _captureRest(bones.eyeR);
        rest.head = _captureRest(bones.head);

        // Find eye meshes for blink fallback. Many MakeHuman exports
        // separate the eyeballs into their own mesh ("brown_eye").
        eyeMeshes.length = 0;
        for (const mesh of (handle.meshes || [])) {
            if (!mesh || !mesh.name) continue;
            const ln = mesh.name.toLowerCase();
            if (ln.includes("eye") && !ln.includes("brow") && !ln.includes("lash")) {
                eyeMeshes.push(mesh);
            }
        }

        console.info(
            "bone_animator: bones found "
            + `jaw=${!!bones.jaw} eyeL=${!!bones.eyeL} eyeR=${!!bones.eyeR} `
            + `head=${!!bones.head} spine=${!!bones.spine} eyeMeshes=${eyeMeshes.length}`
        );
    }

    function hasBones() {
        return !!(bones.jaw || bones.eyeL || bones.eyeR || bones.head);
    }

    // ARKit shape names we know how to fake. emotion_driver uses this
    // to decide whether to route a shape to morphs or bones.
    function canHandle(name) {
        if (!SHAPES) return false;
        return [
            SHAPES.jawOpen,
            SHAPES.eyeLookInL,  SHAPES.eyeLookInR,
            SHAPES.eyeLookOutL, SHAPES.eyeLookOutR,
            SHAPES.eyeLookUpL,  SHAPES.eyeLookUpR,
            SHAPES.eyeLookDownL, SHAPES.eyeLookDownR,
            SHAPES.eyeBlinkL,    SHAPES.eyeBlinkR,
        ].includes(name);
    }

    function setShape(name, weight) {
        if (!SHAPES) return;
        const w = Math.max(0, Math.min(1, Number(weight) || 0));

        if (name === SHAPES.jawOpen) {
            targetDelta.jaw.x = w * JAW_OPEN_MAX_RAD;
            return;
        }
        if (name === SHAPES.eyeLookInL)  { targetDelta.eyeL.y =  w * EYE_LOOK_HORIZ_MAX_RAD; return; }
        if (name === SHAPES.eyeLookOutL) { targetDelta.eyeL.y = -w * EYE_LOOK_HORIZ_MAX_RAD; return; }
        if (name === SHAPES.eyeLookInR)  { targetDelta.eyeR.y = -w * EYE_LOOK_HORIZ_MAX_RAD; return; }
        if (name === SHAPES.eyeLookOutR) { targetDelta.eyeR.y =  w * EYE_LOOK_HORIZ_MAX_RAD; return; }
        if (name === SHAPES.eyeLookUpL)  { targetDelta.eyeL.x = -w * EYE_LOOK_VERT_MAX_RAD; return; }
        if (name === SHAPES.eyeLookUpR)  { targetDelta.eyeR.x = -w * EYE_LOOK_VERT_MAX_RAD; return; }
        if (name === SHAPES.eyeLookDownL){ targetDelta.eyeL.x =  w * EYE_LOOK_VERT_MAX_RAD; return; }
        if (name === SHAPES.eyeLookDownR){ targetDelta.eyeR.x =  w * EYE_LOOK_VERT_MAX_RAD; return; }
        if (name === SHAPES.eyeBlinkL || name === SHAPES.eyeBlinkR) {
            // Combine both blink shapes into one — closing one eye only
            // with a bone-fallback looks worse than not blinking at all.
            // Use the max of L and R as the close amount.
            eyeBlinkAmount = Math.max(eyeBlinkAmount, w);
            return;
        }
    }

    // Per-emotion head-bone overlay. Values describe rest-relative
    // rotation deltas at intensity = 1.0.
    const EMOTION_HEAD_DELTA = {
        neutral:   { x: 0,    y: 0,    z: 0 },
        happy:     { x: -0.04, y: 0,   z: 0.03 },   // tiny head bounce/lift
        thinking:  { x:  0.08, y: 0.04, z: 0 },     // chin down + slight turn
        surprised: { x: -0.12, y: 0,   z: 0 },      // chin up
        tired:     { x:  0.12, y: 0,   z: 0.04 },   // chin down + tilt
    };

    function applyEmotionExpression(name, intensity) {
        const delta = EMOTION_HEAD_DELTA[name] || EMOTION_HEAD_DELTA.neutral;
        const t = Math.max(0, Math.min(1, Number(intensity) || 0));
        targetDelta.head.x = delta.x * t * HEAD_TILT_MAX_RAD / 0.12;
        targetDelta.head.y = delta.y * t * HEAD_TILT_MAX_RAD / 0.12;
        targetDelta.head.z = delta.z * t * HEAD_TILT_MAX_RAD / 0.12;
    }

    function _lerpDelta(current, target) {
        current.x += (target.x - current.x) * LERP_ALPHA;
        current.y += (target.y - current.y) * LERP_ALPHA;
        current.z += (target.z - current.z) * LERP_ALPHA;
    }

    function _writeBone(bone, restRot, delta) {
        if (!bone || !restRot) return;
        const B = window.BABYLON;
        if (!B) return;
        try {
            bone.setRotation(
                new B.Vector3(
                    restRot.x + delta.x,
                    restRot.y + delta.y,
                    restRot.z + delta.z,
                ),
                B.Space.LOCAL,
            );
        } catch (_) {
            // Some bones can't accept setRotation on certain rigs —
            // skip silently rather than spam the console.
        }
    }

    function _applyEyeBlink() {
        // Scale the eye mesh down on Y to fake an eyelid closing. Crude
        // but visible — and only triggers in bone-only mode where no
        // proper blendshape exists.
        if (eyeMeshes.length === 0) return;
        const yScale = 1.0 - eyeBlinkAmount * 0.85;
        for (const m of eyeMeshes) {
            if (m && m.scaling) {
                m.scaling.y = yScale;
            }
        }
    }

    function tick(_dtMs) {
        if (!hasBones() && eyeMeshes.length === 0) return;

        _lerpDelta(currentDelta.jaw, targetDelta.jaw);
        _lerpDelta(currentDelta.eyeL, targetDelta.eyeL);
        _lerpDelta(currentDelta.eyeR, targetDelta.eyeR);
        _lerpDelta(currentDelta.head, targetDelta.head);

        _writeBone(bones.jaw,  rest.jaw,  currentDelta.jaw);
        _writeBone(bones.eyeL, rest.eyeL, currentDelta.eyeL);
        _writeBone(bones.eyeR, rest.eyeR, currentDelta.eyeR);
        _writeBone(bones.head, rest.head, currentDelta.head);

        _applyEyeBlink();

        // Decay the blink amount on its own — `setShape` keeps writing
        // it while a blink-shape is held high; once the additive layer
        // releases it (idle_driver moves to "opening"), we ease back to 0.
        eyeBlinkAmount = eyeBlinkAmount * 0.6;
    }

    function clearAll() {
        targetDelta.jaw.x = targetDelta.jaw.y = targetDelta.jaw.z = 0;
        targetDelta.eyeL.x = targetDelta.eyeL.y = targetDelta.eyeL.z = 0;
        targetDelta.eyeR.x = targetDelta.eyeR.y = targetDelta.eyeR.z = 0;
        targetDelta.head.x = targetDelta.head.y = targetDelta.head.z = 0;
        eyeBlinkAmount = 0;
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.boneAnimator = {
        init,
        setShape,
        applyEmotionExpression,
        tick,
        clearAll,
        canHandle,
        hasBones,
    };
})();
