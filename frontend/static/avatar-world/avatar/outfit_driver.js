/*
 * Avatar World — Outfit / accessory toggle.
 *
 * Genshin/MMD-style GLBs (DLP3D, VRoid) ship the avatar's body, head
 * and accessories as SEPARATE meshes inside the same GLB. This driver
 * finds those meshes by name and exposes a simple
 * `setPart(name, enabled)` API so the UI / backend can flip them on
 * and off without a full GLB reload.
 *
 * What counts as a "part" is matched fuzzy on the mesh / node name:
 *
 *   skirt / outskirt           — typically the lower outfit
 *   collar                      — neck accessory
 *   socks / stockings           — leg accessory
 *   hairband / hair-band / band — hair accessory
 *   ...plus a generic "accessories" toggle that hides all of them at once
 *
 * If the loaded asset is structured differently (single combined mesh
 * for body + clothes, or different naming) the driver silently has
 * nothing to toggle. The UI then hides the menu — see view_mode.js.
 *
 * Public surface:
 *   - init(handle)
 *   - listParts() → [{key, label, enabled, count}]
 *   - setPart(key, enabled)
 *   - togglePart(key)
 *   - hasParts() → boolean
 */
(() => {
    "use strict";

    // Part definitions. Each maps a stable key to one or more
    // case-insensitive name-fragments. A mesh matches when *any*
    // fragment appears as a substring of its name OR its parent node's
    // name. Multiple meshes can belong to a single part — for example a
    // shoe pair on one mesh, both socks on another.
    const PART_RULES = [
        { key: "skirt",     label: "Rock",         frags: ["skirt", "outskirt", "dress"] },
        { key: "collar",    label: "Kragen",       frags: ["collar", "necktie", "tie"] },
        { key: "socks",     label: "Socken",       frags: ["socks", "stocking"] },
        { key: "hairband",  label: "Haarband",     frags: ["hair-band", "hairband", "headband"] },
        { key: "shoes",     label: "Schuhe",       frags: ["shoe", "boot"] },
        { key: "gloves",    label: "Handschuhe",   frags: ["glove", "gauntlet"] },
    ];

    let handleRef = null;
    // partKey -> array of mesh refs that participate in that part.
    const partMeshes = new Map();
    // partKey -> bool (currently enabled).
    const partState = new Map();

    function _matchesFragments(name, frags) {
        if (!name) return false;
        const n = name.toLowerCase();
        for (const f of frags) {
            if (n.includes(f.toLowerCase())) return true;
        }
        return false;
    }

    function _meshIdentifiers(mesh) {
        // The mesh name is sometimes generic ("Grok-Ms.Ani.003") while
        // its parent node carries the meaningful label ("P1-OutSkirt").
        // We check both.
        const out = [];
        if (mesh && mesh.name) out.push(mesh.name);
        if (mesh && mesh.parent && mesh.parent.name) out.push(mesh.parent.name);
        return out;
    }

    function init(handle) {
        handleRef = handle || null;
        partMeshes.clear();
        partState.clear();
        if (!handle || !handle.meshes) return;

        for (const rule of PART_RULES) {
            const matches = [];
            for (const mesh of handle.meshes) {
                const idents = _meshIdentifiers(mesh);
                for (const ident of idents) {
                    if (_matchesFragments(ident, rule.frags)) {
                        matches.push(mesh);
                        break;
                    }
                }
            }
            if (matches.length > 0) {
                partMeshes.set(rule.key, matches);
                partState.set(rule.key, true);
            }
        }

        if (partMeshes.size > 0) {
            const summary = [...partMeshes.entries()]
                .map(([k, ms]) => `${k}(${ms.length})`)
                .join(", ");
            console.info("outfit_driver: " + partMeshes.size
                + " toggleable parts: " + summary);
        } else {
            console.info("outfit_driver: no toggleable parts detected on this asset");
        }
    }

    function hasParts() {
        return partMeshes.size > 0;
    }

    function listParts() {
        const out = [];
        for (const rule of PART_RULES) {
            const meshes = partMeshes.get(rule.key);
            if (!meshes) continue;
            out.push({
                key: rule.key,
                label: rule.label,
                enabled: partState.get(rule.key) !== false,
                count: meshes.length,
            });
        }
        return out;
    }

    function setPart(key, enabled) {
        const meshes = partMeshes.get(key);
        if (!meshes) return false;
        const flag = !!enabled;
        for (const m of meshes) {
            if (m && typeof m.setEnabled === "function") {
                m.setEnabled(flag);
            }
        }
        partState.set(key, flag);
        return true;
    }

    function togglePart(key) {
        const current = partState.get(key);
        return setPart(key, current === false);
    }

    function setAll(enabled) {
        for (const key of partMeshes.keys()) {
            setPart(key, enabled);
        }
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.outfit = {
        init,
        listParts,
        setPart,
        togglePart,
        setAll,
        hasParts,
    };
})();
