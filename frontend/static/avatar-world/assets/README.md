# Avatar World — Asset Drop Zone

This folder holds the runtime assets the BabylonJS scene loads. They are
intentionally *not* checked into the repo — they're either user-generated
(Ready Player Me avatar) or large binary files (Mixamo FBX/GLB, Polyhaven
HDRis). Each subfolder has its own conventions.

## Required for Phase 1A — Avatar steht

### `models/`
- `lexy_base.glb` — Ready Player Me avatar export. Recommended settings:
  * **Half-body** or **Full-body** export
  * **ARKit blendshapes** ENABLED (visemes + facial expressions)
  * **Apose** rig (default RPM pose)
- `outfit_casual.glb`, `outfit_sport.glb`, `outfit_business.glb`, `outfit_pyjama.glb`
  — same base body, alternate outfits. For Phase 1A a single `lexy_base.glb`
  is enough; outfit swap comes in Phase 1D.

### `apartment/`
- `room_phase_a.glb` — single-room scene (Desk + Couch + Window + Bathroom door).
  Until we have one, the placeholder floor in `scene/apartment.js` stands in.

### `backgrounds/`
- `city_morning.jpg`, `city_day.jpg`, `city_evening.jpg`, `city_night.jpg`
- Optionally `forest.jpg`, `mountain.jpg`, `rain.jpg`
- 4K JPGs are fine for Phase 1; HDRi/EXR for Phase 2 if we want IBL.

## Required for Phase 1B — Lexy spricht

### `animations/`
- `idle_sit.glb`, `idle_stand.glb`, `typing.glb`, `reading.glb`,
  `walking.glb`, `sleeping.glb` — Mixamo download with skin enabled.

## Babylon engine

The scene loader expects `window.BABYLON` to be available globally. By
default `index.html` pulls the three scripts from `cdn.babylonjs.com`:

```html
<script src="https://cdn.babylonjs.com/babylon.js" defer></script>
<script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js" defer></script>
<script src="https://cdn.babylonjs.com/gui/babylon.gui.min.js" defer></script>
```

For offline / air-gapped setups, run the bundled helper:

```cmd
scripts\vendor_babylon.bat
```

It downloads the three files into `frontend/static/vendor/babylon/` and
prints the script tags you can paste into `index.html` in place of the
CDN ones. The avatar-world bootstrap detects either case (it just waits
for `window.BABYLON`).

## Licensing reminders

- Ready Player Me assets: free for personal/commercial use under their TOS.
- Mixamo: free, requires an Adobe account; redistribute the FBX is fine.
- Polyhaven: CC0.
- **Avoid HoYoverse-style models from third-party stores** — most have
  educational-only licenses that don't fit a long-running personal AI.
