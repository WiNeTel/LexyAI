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

    // ── MMD (MikuMikuDance) → ARKit alias table ────────────────────
    //
    // Genshin-Style / VRoid / MMD models ship blendshapes named in
    // Japanese instead of ARKit. The loader uses this table to alias
    // every ARKit shape name to whatever MMD target the asset
    // actually has — emotion_driver and lip_sync don't need to know
    // which naming convention the GLB used.
    //
    // First match wins. The alias list is forgiving: if none of the
    // aliases match, the shape is silently dropped (Bone-fallback
    // takes over for jaw / eye / head).
    const MMD_ALIASES = {
        // ── Mouth — visemes / lip-sync ───────────────────────────
        [SHAPES.jawOpen]:        ["あ", "あ２", "笑い"],
        [SHAPES.mouthClose]:     ["ん"],
        [SHAPES.mouthFunnel]:    ["お", "う"],
        [SHAPES.mouthPucker]:    ["う", "ω"],
        [SHAPES.mouthStretchL]:  ["口横広げ", "い", "い１"],
        [SHAPES.mouthStretchR]:  ["口横広げ", "い", "い１"],

        // ── Mouth — smiles / frowns ──────────────────────────────
        [SHAPES.mouthSmileL]:    ["にやり", "にっこり", "なごみ左", "なごみ", "口角上げ", "にこり左", "にこり"],
        [SHAPES.mouthSmileR]:    ["にやり", "にっこり", "なごみ右", "なごみ", "口角上げ", "にこり右", "にこり"],
        [SHAPES.mouthFrownL]:    ["困る左", "困る", "困る２左", "困る２"],
        [SHAPES.mouthFrownR]:    ["困る右", "困る", "困る２右", "困る２"],
        [SHAPES.mouthDimpleL]:   ["なごみ左", "なごみ"],
        [SHAPES.mouthDimpleR]:   ["なごみ右", "なごみ"],

        // ── Eyes — blinks ────────────────────────────────────────
        [SHAPES.eyeBlinkL]:      ["ウィンク", "ウィンク２", "まばたき"],
        [SHAPES.eyeBlinkR]:      ["ウィンク右", "ウィンク２右", "まばたき"],

        // ── Eyes — wide / squint ─────────────────────────────────
        [SHAPES.eyeWideL]:       ["瞳大", "びっくり"],
        [SHAPES.eyeWideR]:       ["瞳大", "びっくり"],
        [SHAPES.eyeSquintL]:     ["瞳小", "じと目", "ジト目"],
        [SHAPES.eyeSquintR]:     ["瞳小", "じと目", "ジト目"],

        // ── Eyes — gaze direction ────────────────────────────────
        [SHAPES.eyeLookUpL]:     ["上左", "上"],
        [SHAPES.eyeLookUpR]:     ["上右", "上"],
        [SHAPES.eyeLookDownL]:   ["下左", "下"],
        [SHAPES.eyeLookDownR]:   ["下右", "下"],
        [SHAPES.eyeLookInL]:     ["前左", "前"],
        [SHAPES.eyeLookInR]:     ["前右", "前"],
        [SHAPES.eyeLookOutL]:    ["前左", "前"],
        [SHAPES.eyeLookOutR]:    ["前右", "前"],

        // ── Brows ────────────────────────────────────────────────
        [SHAPES.browDownL]:      ["怒り左", "怒り目", "怒り"],
        [SHAPES.browDownR]:      ["怒り右", "怒り目", "怒り"],
        [SHAPES.browInnerUp]:    ["困る", "真面目"],
        [SHAPES.browOuterUpL]:   ["喜び", "眼角上"],
        [SHAPES.browOuterUpR]:   ["喜び", "眼角上"],

        // ── Cheeks / nose ────────────────────────────────────────
        [SHAPES.cheekPuff]:      ["ぷく", "ω"],
        [SHAPES.cheekSquintL]:   ["なごみ左", "なごみ"],
        [SHAPES.cheekSquintR]:   ["なごみ右", "なごみ"],
    };

    // Bonus: MMD emotion-mesh overlays. Layered ON TOP of normal
    // emotions to deepen the expression — purely additive.
    //   照れ (blush) lights up on strong "happy"
    //   汗 (sweat)  on intense "thinking"
    //   涙 (tear)   on intense "tired"
    //   //// (embarrassed sweat) reserved for future "shy" emotion
    const MMD_EMOTION_OVERLAY = {
        happy:    { "照れ": 0.5 },
        thinking: { "汗":   0.4 },
        tired:    { "涙":   0.3 },
    };

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.morphs = {
        SHAPES,
        EMOTION_PRESETS,
        EMOTION_COLORS,
        MMD_ALIASES,
        MMD_EMOTION_OVERLAY,
    };
})();
