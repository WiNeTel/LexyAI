/*
 * Avatar World — ARKit Morph-Target map and emotion presets.
 *
 * Centralises the shape names used by the rest of the avatar layer so
 * we don't sprinkle string literals across three files. ARKit's 52-shape
 * set is the de-facto standard for Ready Player Me / Apple FaceKit /
 * MetaHuman exports, so most realistic GLB models from those pipelines
 * carry these target names verbatim.
 *
 * The emotion presets describe how each emotion drives the shapes at
 * intensity = 1.0. The driver lerps from 0 → preset[shape] * intensity.
 */
(() => {
    "use strict";

    // ── ARKit 52-shape names (subset we actually drive) ─────────────
    const SHAPES = {
        // Eyes
        eyeBlinkL: "eyeBlinkLeft",
        eyeBlinkR: "eyeBlinkRight",
        eyeWideL:  "eyeWideLeft",
        eyeWideR:  "eyeWideRight",
        eyeSquintL: "eyeSquintLeft",
        eyeSquintR: "eyeSquintRight",
        eyeLookUpL: "eyeLookUpLeft",
        eyeLookUpR: "eyeLookUpRight",
        eyeLookDownL: "eyeLookDownLeft",
        eyeLookDownR: "eyeLookDownRight",
        eyeLookInL: "eyeLookInLeft",
        eyeLookInR: "eyeLookInRight",
        eyeLookOutL: "eyeLookOutLeft",
        eyeLookOutR: "eyeLookOutRight",
        // Brows
        browDownL:  "browDownLeft",
        browDownR:  "browDownRight",
        browInnerUp: "browInnerUp",
        browOuterUpL: "browOuterUpLeft",
        browOuterUpR: "browOuterUpRight",
        // Mouth — open / shape
        jawOpen:     "jawOpen",
        jawForward:  "jawForward",
        mouthClose:  "mouthClose",
        mouthFunnel: "mouthFunnel",
        mouthPucker: "mouthPucker",
        mouthShrugUpper: "mouthShrugUpper",
        mouthShrugLower: "mouthShrugLower",
        // Mouth — smiles / frowns
        mouthSmileL: "mouthSmileLeft",
        mouthSmileR: "mouthSmileRight",
        mouthFrownL: "mouthFrownLeft",
        mouthFrownR: "mouthFrownRight",
        mouthStretchL: "mouthStretchLeft",
        mouthStretchR: "mouthStretchRight",
        mouthDimpleL: "mouthDimpleLeft",
        mouthDimpleR: "mouthDimpleRight",
        // Cheeks / nose
        cheekPuff: "cheekPuff",
        cheekSquintL: "cheekSquintLeft",
        cheekSquintR: "cheekSquintRight",
        noseSneerL: "noseSneerLeft",
        noseSneerR: "noseSneerRight",
    };

    // ── Emotion presets: shape → weight at full intensity ──────────
    // Phase 1 keeps the table small. Add more nuance over time without
    // touching the driver code.
    const EMOTION_PRESETS = {
        neutral: {},

        happy: {
            [SHAPES.mouthSmileL]: 0.85,
            [SHAPES.mouthSmileR]: 0.85,
            [SHAPES.cheekSquintL]: 0.4,
            [SHAPES.cheekSquintR]: 0.4,
            [SHAPES.eyeSquintL]: 0.25,
            [SHAPES.eyeSquintR]: 0.25,
            [SHAPES.browInnerUp]: 0.15,
        },

        thinking: {
            [SHAPES.browDownL]: 0.55,
            [SHAPES.browDownR]: 0.55,
            [SHAPES.mouthPucker]: 0.20,
            [SHAPES.mouthDimpleL]: 0.30,
            [SHAPES.mouthDimpleR]: 0.30,
            [SHAPES.eyeSquintL]: 0.15,
            [SHAPES.eyeSquintR]: 0.15,
        },

        surprised: {
            [SHAPES.eyeWideL]: 0.95,
            [SHAPES.eyeWideR]: 0.95,
            [SHAPES.jawOpen]: 0.40,
            [SHAPES.browOuterUpL]: 0.70,
            [SHAPES.browOuterUpR]: 0.70,
            [SHAPES.browInnerUp]: 0.40,
            [SHAPES.mouthFunnel]: 0.15,
        },

        tired: {
            [SHAPES.eyeBlinkL]: 0.55,
            [SHAPES.eyeBlinkR]: 0.55,
            [SHAPES.mouthFrownL]: 0.30,
            [SHAPES.mouthFrownR]: 0.30,
            [SHAPES.browDownL]: 0.20,
            [SHAPES.browDownR]: 0.20,
            [SHAPES.mouthShrugLower]: 0.15,
        },
    };

    // Colour-tint per emotion — used only by the placeholder capsule so
    // Phase 0 still has visible feedback before a GLB is loaded.
    const EMOTION_COLORS = {
        neutral:   [0.55, 0.35, 0.25],
        happy:     [0.95, 0.65, 0.45],
        thinking:  [0.55, 0.50, 0.85],
        surprised: [1.00, 0.85, 0.55],
        tired:     [0.45, 0.45, 0.50],
    };

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.morphs = {
        SHAPES,
        EMOTION_PRESETS,
        EMOTION_COLORS,
    };
})();
