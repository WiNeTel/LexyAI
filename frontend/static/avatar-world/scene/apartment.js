/*
 * Avatar World — Apartment Scene (Phase 1C).
 *
 * Builds the apartment from Babylon primitives (no GLB needed). A small
 * single-room layout big enough to read on a PiP canvas:
 *
 *   - Floor + Backwall + Sidewall
 *   - Desk + Monitor + Chair in the right-back corner
 *   - Couch + side table + stand lamp on the left
 *   - Bookshelf on the back wall
 *   - Window in the centre-back (acts as anchor for background.js)
 *   - Bathroom door on the back-left
 *   - Plant by the window
 *
 * The avatar (placeholder capsule or loaded GLB) stands in front of
 * the desk so the camera framing reads naturally.
 *
 * Public surface:
 *   - createScene(canvas) → { engine, scene, camera, placeholder,
 *                              speakAura, windowAnchor,
 *                              lights: { hemi, key, lamp } }
 */
(() => {
    "use strict";

    function _mat(scene, name, rgb, opts = {}) {
        const B = window.BABYLON;
        const m = new B.StandardMaterial(name, scene);
        m.diffuseColor = new B.Color3(rgb[0], rgb[1], rgb[2]);
        if (opts.spec) m.specularColor = new B.Color3(opts.spec, opts.spec, opts.spec);
        else m.specularColor = new B.Color3(0.05, 0.05, 0.05);
        if (opts.emissive) {
            m.emissiveColor = new B.Color3(opts.emissive[0], opts.emissive[1], opts.emissive[2]);
            m.disableLighting = !!opts.disableLighting;
        }
        return m;
    }

    function _buildRoom(scene) {
        const B = window.BABYLON;
        const wallMat = _mat(scene, "wall", [0.30, 0.32, 0.42]);
        const floorMat = _mat(scene, "floor", [0.22, 0.18, 0.16]);

        const floor = B.MeshBuilder.CreateGround("floor", { width: 6, height: 6 }, scene);
        floor.material = floorMat;

        const backWall = B.MeshBuilder.CreateBox("backwall", { width: 6, height: 3.0, depth: 0.1 }, scene);
        backWall.position = new B.Vector3(0, 1.5, -3.0);
        backWall.material = wallMat;

        const leftWall = B.MeshBuilder.CreateBox("leftwall", { width: 0.1, height: 3.0, depth: 6.0 }, scene);
        leftWall.position = new B.Vector3(-3.0, 1.5, 0);
        leftWall.material = wallMat;

        return { floor, backWall, leftWall };
    }

    function _buildWindow(scene) {
        const B = window.BABYLON;
        // The window is a hole-shaped frame on the back wall plus an
        // anchor node for the background plate.
        const frameMat = _mat(scene, "frame", [0.18, 0.16, 0.14], { spec: 0.12 });
        const frame = new B.TransformNode("window-frame", scene);
        frame.position = new B.Vector3(0, 1.6, -2.95);

        const top = B.MeshBuilder.CreateBox("window-top", { width: 2.2, height: 0.08, depth: 0.04 }, scene);
        top.position.y = 0.6; top.parent = frame; top.material = frameMat;
        const bot = B.MeshBuilder.CreateBox("window-bot", { width: 2.2, height: 0.08, depth: 0.04 }, scene);
        bot.position.y = -0.6; bot.parent = frame; bot.material = frameMat;
        const left = B.MeshBuilder.CreateBox("window-left", { width: 0.08, height: 1.28, depth: 0.04 }, scene);
        left.position.x = -1.06; left.parent = frame; left.material = frameMat;
        const right = B.MeshBuilder.CreateBox("window-right", { width: 0.08, height: 1.28, depth: 0.04 }, scene);
        right.position.x = 1.06; right.parent = frame; right.material = frameMat;
        const cross = B.MeshBuilder.CreateBox("window-cross", { width: 0.04, height: 1.28, depth: 0.04 }, scene);
        cross.parent = frame; cross.material = frameMat;

        return { anchor: frame };
    }

    function _buildDesk(scene) {
        const B = window.BABYLON;
        const woodMat = _mat(scene, "desk-wood", [0.42, 0.27, 0.18], { spec: 0.10 });
        const screenMat = _mat(scene, "monitor", [0.04, 0.05, 0.08], {
            emissive: [0.18, 0.32, 0.45],
            disableLighting: true,
        });
        const top = B.MeshBuilder.CreateBox("desk-top",
            { width: 1.6, height: 0.05, depth: 0.7 }, scene);
        top.position = new B.Vector3(1.5, 0.75, -1.8);
        top.material = woodMat;

        const leg1 = B.MeshBuilder.CreateBox("desk-leg1", { width: 0.05, height: 0.75, depth: 0.05 }, scene);
        leg1.position = new B.Vector3(1.5 + 0.75, 0.375, -1.8 - 0.32); leg1.material = woodMat;
        const leg2 = B.MeshBuilder.CreateBox("desk-leg2", { width: 0.05, height: 0.75, depth: 0.05 }, scene);
        leg2.position = new B.Vector3(1.5 - 0.75, 0.375, -1.8 - 0.32); leg2.material = woodMat;
        const leg3 = B.MeshBuilder.CreateBox("desk-leg3", { width: 0.05, height: 0.75, depth: 0.05 }, scene);
        leg3.position = new B.Vector3(1.5 + 0.75, 0.375, -1.8 + 0.32); leg3.material = woodMat;
        const leg4 = B.MeshBuilder.CreateBox("desk-leg4", { width: 0.05, height: 0.75, depth: 0.05 }, scene);
        leg4.position = new B.Vector3(1.5 - 0.75, 0.375, -1.8 + 0.32); leg4.material = woodMat;

        const monitor = B.MeshBuilder.CreateBox("monitor",
            { width: 0.65, height: 0.42, depth: 0.05 }, scene);
        monitor.position = new B.Vector3(1.5, 1.10, -2.0);
        monitor.material = screenMat;

        const stand = B.MeshBuilder.CreateBox("monitor-stand",
            { width: 0.12, height: 0.20, depth: 0.04 }, scene);
        stand.position = new B.Vector3(1.5, 0.88, -2.0);
        stand.material = _mat(scene, "stand", [0.10, 0.10, 0.12]);

        // Chair — simple box + back.
        const chairMat = _mat(scene, "chair", [0.16, 0.18, 0.22]);
        const seat = B.MeshBuilder.CreateBox("chair-seat",
            { width: 0.45, height: 0.05, depth: 0.45 }, scene);
        seat.position = new B.Vector3(1.5, 0.45, -1.20);
        seat.material = chairMat;
        const back = B.MeshBuilder.CreateBox("chair-back",
            { width: 0.45, height: 0.55, depth: 0.05 }, scene);
        back.position = new B.Vector3(1.5, 0.75, -0.99);
        back.material = chairMat;

        return { top, monitor };
    }

    function _buildCouch(scene) {
        const B = window.BABYLON;
        const couchMat = _mat(scene, "couch", [0.40, 0.22, 0.30]);
        const baseSeat = B.MeshBuilder.CreateBox("couch-base",
            { width: 1.6, height: 0.35, depth: 0.7 }, scene);
        baseSeat.position = new B.Vector3(-1.6, 0.20, 0.4);
        baseSeat.material = couchMat;

        const back = B.MeshBuilder.CreateBox("couch-back",
            { width: 1.6, height: 0.55, depth: 0.15 }, scene);
        back.position = new B.Vector3(-1.6, 0.55, 0.05);
        back.material = couchMat;

        const armL = B.MeshBuilder.CreateBox("couch-arm-l",
            { width: 0.12, height: 0.45, depth: 0.7 }, scene);
        armL.position = new B.Vector3(-1.6 + 0.86, 0.30, 0.4);
        armL.material = couchMat;
        const armR = B.MeshBuilder.CreateBox("couch-arm-r",
            { width: 0.12, height: 0.45, depth: 0.7 }, scene);
        armR.position = new B.Vector3(-1.6 - 0.86, 0.30, 0.4);
        armR.material = couchMat;

        // Side table next to the couch.
        const tableMat = _mat(scene, "side-table", [0.32, 0.22, 0.16]);
        const tableTop = B.MeshBuilder.CreateBox("side-table",
            { width: 0.4, height: 0.04, depth: 0.4 }, scene);
        tableTop.position = new B.Vector3(-2.6, 0.6, 0.4);
        tableTop.material = tableMat;
        const tablePole = B.MeshBuilder.CreateCylinder("side-table-pole",
            { diameter: 0.05, height: 0.6 }, scene);
        tablePole.position = new B.Vector3(-2.6, 0.3, 0.4);
        tablePole.material = tableMat;

        return { baseSeat };
    }

    function _buildLamp(scene) {
        const B = window.BABYLON;
        // Stand lamp behind the couch — corresponds to the lighting.js
        // "lamp" preset that switches on for evening / night.
        const lampMat = _mat(scene, "lamp-shade", [0.95, 0.85, 0.60], {
            emissive: [0.80, 0.55, 0.30],
            disableLighting: true,
        });
        const poleMat = _mat(scene, "lamp-pole", [0.12, 0.10, 0.10]);
        const pole = B.MeshBuilder.CreateCylinder("lamp-pole",
            { diameter: 0.05, height: 1.6 }, scene);
        pole.position = new B.Vector3(-2.5, 0.8, -0.4);
        pole.material = poleMat;

        const shade = B.MeshBuilder.CreateCylinder("lamp-shade",
            { diameterTop: 0.18, diameterBottom: 0.28, height: 0.30 }, scene);
        shade.position = new B.Vector3(-2.5, 1.62, -0.4);
        shade.material = lampMat;

        // Point light placed inside the shade. Disabled by default —
        // lighting.js enables it for evening/night buckets.
        const lamp = new B.PointLight(
            "lamp",
            new B.Vector3(-2.5, 1.55, -0.4),
            scene,
        );
        lamp.diffuse = new B.Color3(1.0, 0.78, 0.50);
        lamp.specular = new B.Color3(0.4, 0.30, 0.20);
        lamp.intensity = 0.0;
        lamp.setEnabled(false);

        return { lamp, shade };
    }

    function _buildBookshelfAndCorkboard(scene) {
        const B = window.BABYLON;
        // Bookshelf on the back wall — 4 shelves with coloured "books"
        // (boxes) so it reads from any camera angle.
        const woodMat = _mat(scene, "shelf-wood", [0.30, 0.20, 0.14], { spec: 0.10 });
        const cluster = new B.TransformNode("bookshelf", scene);
        cluster.position = new B.Vector3(-1.5, 1.5, -2.85);
        const frame = B.MeshBuilder.CreateBox("shelf-frame", {
            width: 1.1, height: 1.4, depth: 0.25,
        }, scene);
        frame.material = woodMat;
        frame.parent = cluster;

        const bookColours = [
            [0.55, 0.30, 0.25], [0.30, 0.45, 0.55], [0.85, 0.70, 0.40],
            [0.40, 0.55, 0.35], [0.65, 0.40, 0.60], [0.30, 0.30, 0.50],
        ];
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 6; col++) {
                const colour = bookColours[(row * 6 + col) % bookColours.length];
                const book = B.MeshBuilder.CreateBox("book", {
                    width: 0.13, height: 0.30, depth: 0.20,
                }, scene);
                book.position = new B.Vector3(
                    -0.45 + col * 0.16,
                    0.35 - row * 0.42,
                    0.04,
                );
                book.material = _mat(scene, `book-${row}-${col}`, colour, { spec: 0.05 });
                book.parent = cluster;
            }
        }

        // Corkboard on the side wall (left) — placeholder for "facts"
        // sticky notes in a later phase.
        const board = B.MeshBuilder.CreateBox("corkboard",
            { width: 0.05, height: 0.9, depth: 1.0 }, scene);
        board.position = new B.Vector3(-2.95, 1.6, -1.2);
        board.material = _mat(scene, "cork", [0.55, 0.40, 0.22], { spec: 0.05 });
        return { cluster, board };
    }

    function _buildPlant(scene) {
        const B = window.BABYLON;
        const pot = B.MeshBuilder.CreateCylinder("plant-pot",
            { diameterTop: 0.30, diameterBottom: 0.24, height: 0.30 }, scene);
        pot.position = new B.Vector3(0.9, 0.15, -2.7);
        pot.material = _mat(scene, "plant-pot", [0.45, 0.30, 0.22], { spec: 0.05 });

        const leaves = B.MeshBuilder.CreateSphere("plant-leaves",
            { diameter: 0.55, segments: 12 }, scene);
        leaves.position = new B.Vector3(0.9, 0.55, -2.7);
        leaves.scaling.y = 1.4;
        leaves.material = _mat(scene, "plant-leaves", [0.28, 0.45, 0.22]);
        return { pot, leaves };
    }

    function _buildBathroomDoor(scene) {
        const B = window.BABYLON;
        const doorMat = _mat(scene, "door", [0.42, 0.30, 0.22], { spec: 0.10 });
        const knobMat = _mat(scene, "knob", [0.75, 0.65, 0.30], {
            emissive: [0.05, 0.05, 0.02],
        });
        const door = B.MeshBuilder.CreateBox("bathroom-door",
            { width: 0.9, height: 2.0, depth: 0.06 }, scene);
        door.position = new B.Vector3(-2.4, 1.0, -2.92);
        door.material = doorMat;

        const knob = B.MeshBuilder.CreateSphere("bathroom-knob",
            { diameter: 0.07, segments: 10 }, scene);
        knob.position = new B.Vector3(-2.0, 0.95, -2.90);
        knob.material = knobMat;
        return { door, knob };
    }

    function _buildPlaceholderAvatar(scene) {
        const B = window.BABYLON;
        const body = B.MeshBuilder.CreateCapsule(
            "lexy-placeholder",
            { height: 1.7, radius: 0.32 },
            scene,
        );
        body.position.y = 0.85;
        body.material = _mat(scene, "bodyMat", [0.55, 0.35, 0.25]);

        const aura = B.MeshBuilder.CreateSphere("speak-aura", { diameter: 0.12 }, scene);
        aura.position.y = 1.92;
        aura.material = _mat(scene, "auraMat", [0.0, 0.0, 0.0], {
            emissive: [0.40, 0.70, 1.00],
            disableLighting: true,
        });
        aura.setEnabled(false);
        return { body, aura };
    }

    function createScene(canvas) {
        if (typeof window.BABYLON === "undefined") {
            console.warn("avatar-world: BABYLON is not loaded");
            return null;
        }
        const B = window.BABYLON;

        const engine = new B.Engine(canvas, true, {
            preserveDrawingBuffer: true,
            stencil: true,
            antialias: true,
        });
        const scene = new B.Scene(engine);
        scene.clearColor = new B.Color4(0.04, 0.05, 0.07, 1.0);
        scene.imageProcessingConfiguration.vignetteEnabled = true;
        scene.imageProcessingConfiguration.vignetteWeight = 2.2;

        // Camera framed toward the desk wall (back of the room).
        // Babylon's ArcRotateCamera convention: alpha=PI/2 puts the
        // camera at z=+radius (in front of the room), beta tilts down.
        // Earlier this was -PI/2.1 which positioned the camera BEHIND
        // the back wall — Mike saw the apartment from outside, looking
        // at the back of the wall the desk hangs on.
        const camera = new B.ArcRotateCamera(
            "cam",
            Math.PI / 2,        // front view, looking toward -Z (the back wall)
            Math.PI / 2.5,      // ~72°, slightly above horizontal
            5.8,
            new B.Vector3(0, 1.2, -1.0),   // look at the avatar / desk area
            scene,
        );
        camera.attachControl(canvas, true);
        camera.lowerRadiusLimit = 3.0;
        camera.upperRadiusLimit = 10.0;
        camera.lowerBetaLimit = Math.PI / 4;
        camera.upperBetaLimit = (Math.PI * 5) / 12;
        // Constrain horizontal pan so the user can't spin behind the
        // apartment's missing front wall (we only modelled back + left).
        camera.lowerAlphaLimit = Math.PI / 3;
        camera.upperAlphaLimit = (Math.PI * 2) / 3;
        camera.wheelDeltaPercentage = 0.01;

        // Lights — tuned in lighting.js based on time_of_day.
        const hemi = new B.HemisphericLight("hemi", new B.Vector3(0, 1, 0), scene);
        hemi.intensity = 0.85;
        hemi.diffuse = new B.Color3(1, 1, 0.96);
        hemi.groundColor = new B.Color3(0.4, 0.4, 0.45);

        const key = new B.DirectionalLight("key", new B.Vector3(-0.3, -1.2, -0.2), scene);
        key.intensity = 0.95;
        key.diffuse = new B.Color3(1, 1, 0.98);

        _buildRoom(scene);
        const win = _buildWindow(scene);
        _buildDesk(scene);
        _buildCouch(scene);
        const lampSet = _buildLamp(scene);
        _buildBookshelfAndCorkboard(scene);
        _buildPlant(scene);
        _buildBathroomDoor(scene);
        const placeholder = _buildPlaceholderAvatar(scene);

        window.addEventListener("resize", () => engine.resize());
        engine.runRenderLoop(() => scene.render());

        return {
            engine,
            scene,
            camera,
            placeholder: placeholder.body,
            speakAura: placeholder.aura,
            windowAnchor: win.anchor,
            lights: { hemi, key, lamp: lampSet.lamp },
        };
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.scene = { createScene };
})();
