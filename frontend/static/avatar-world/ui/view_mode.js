/*
 * Avatar World — View mode toggle + drag.
 *
 * Three modes — `conversation` (big centered), `ambient` (medium), `pip`
 * (small floating). Stored on the host element via `data-view-mode`;
 * CSS does the layout.
 *
 * Two fixes vs Phase 0:
 *
 *   1. **Idempotent set()** — the WS round-trip
 *      (UI click → backend → `avatar.view_mode` broadcast →
 *      frontend handler → set() again → WS again …) was causing an
 *      endless ping-pong loop that retriggered the CSS transition every
 *      frame, which read as "wackelt hin und her" in the PiP. Now set()
 *      returns immediately when the mode already matches.
 *
 *   2. **Draggable in PiP mode** — header-bar drag updates inline
 *      `left/top` so the user can park the PiP anywhere. Drag is
 *      disabled in ambient/conversation (they're centered/half-screen
 *      and have their own anchors). Switching back to PiP keeps the
 *      last drag position thanks to localStorage.
 */
(() => {
    "use strict";

    const HOST_ID = "lexy-avatar-pip";
    const STORAGE_KEY_MODE = "lexy.avatar.view_mode";
    const STORAGE_KEY_POS = "lexy.avatar.pip_pos";

    function _host() {
        return document.getElementById(HOST_ID);
    }

    function _currentMode(host) {
        return (host && host.dataset && host.dataset.viewMode) || "";
    }

    // ── Drag state ─────────────────────────────────────────────────
    const drag = {
        active: false,
        offsetX: 0,
        offsetY: 0,
    };

    function _applyPipPosition(host, left, top) {
        // Switch from right/bottom (CSS default) to left/top so JS can
        // pin the PiP anywhere. Inline-style wins over the stylesheet.
        host.style.left = left + "px";
        host.style.top = top + "px";
        host.style.right = "auto";
        host.style.bottom = "auto";
    }

    function _restorePipPositionFromStorage(host) {
        let pos = null;
        try {
            const raw = localStorage.getItem(STORAGE_KEY_POS);
            if (raw) pos = JSON.parse(raw);
        } catch { /* corrupt entry — ignore */ }
        if (pos && Number.isFinite(pos.left) && Number.isFinite(pos.top)) {
            _applyPipPosition(host, pos.left, pos.top);
        } else {
            // Default: bottom-right corner like the stylesheet says.
            host.style.left = "";
            host.style.top = "";
            host.style.right = "";
            host.style.bottom = "";
        }
    }

    function _clearInlinePosition(host) {
        host.style.left = "";
        host.style.top = "";
        host.style.right = "";
        host.style.bottom = "";
    }

    function _onDragStart(ev) {
        const host = _host();
        if (!host || _currentMode(host) !== "pip") return;
        if (ev.target && ev.target.closest("button")) return;   // let buttons work
        drag.active = true;
        const rect = host.getBoundingClientRect();
        drag.offsetX = ev.clientX - rect.left;
        drag.offsetY = ev.clientY - rect.top;
        ev.preventDefault();
    }

    function _onDragMove(ev) {
        if (!drag.active) return;
        const host = _host();
        if (!host) return;
        const rect = host.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        let left = ev.clientX - drag.offsetX;
        let top = ev.clientY - drag.offsetY;
        // Keep at least 16 px of the PiP on-screen so it can't be
        // dragged out of reach.
        const minLeft = 16 - w + 80;
        const maxLeft = window.innerWidth - 80;
        const minTop = 8;
        const maxTop = window.innerHeight - 24;
        if (left < minLeft) left = minLeft;
        if (left > maxLeft) left = maxLeft;
        if (top < minTop) top = minTop;
        if (top > maxTop) top = maxTop;
        _applyPipPosition(host, left, top);
    }

    function _onDragEnd() {
        if (!drag.active) return;
        drag.active = false;
        const host = _host();
        if (!host) return;
        // Persist so a reload keeps the user's preferred corner.
        try {
            const left = parseFloat(host.style.left) || 0;
            const top = parseFloat(host.style.top) || 0;
            localStorage.setItem(
                STORAGE_KEY_POS,
                JSON.stringify({ left, top }),
            );
        } catch { /* storage full / private mode — ignore */ }
    }

    // ── Mode switching ─────────────────────────────────────────────

    function set(mode) {
        const allowed = ["conversation", "ambient", "pip"];
        if (!allowed.includes(mode)) return;
        const host = _host();
        if (!host) return;

        // Idempotent — no-op if the mode already matches. Stops the WS
        // ping-pong loop (set → WS broadcast → set …) that was triggering
        // the CSS transition every frame.
        if (_currentMode(host) === mode) return;

        host.dataset.viewMode = mode;
        try { localStorage.setItem(STORAGE_KEY_MODE, mode); } catch { /* ignore */ }

        // Switch positioning style based on mode.
        if (mode === "pip") {
            _restorePipPositionFromStorage(host);
        } else {
            _clearInlinePosition(host);
        }

        const titleEl = host.querySelector(".lexy-avatar-title");
        if (titleEl) titleEl.textContent = `Lexy — ${mode}`;

        if (window.LexyAvatar && window.LexyAvatar.net) {
            window.LexyAvatar.net.send("set_view_mode", { mode });
        }
        // Single resize after the CSS transition settles. Idempotent
        // upstream means this fires at most once per real mode change.
        setTimeout(() => {
            window.dispatchEvent(new Event("resize"));
        }, 260);
    }

    function hide() {
        const host = _host();
        if (!host) return;
        host.hidden = true;
    }

    function show() {
        const host = _host();
        if (!host) return;
        host.hidden = false;
        setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
    }

    function init() {
        const host = _host();
        if (!host) return;

        const stored = (() => {
            try { return localStorage.getItem(STORAGE_KEY_MODE); } catch { return null; }
        })();
        // Force-apply the stored mode even if it equals our default —
        // we still need to wire the dataset attribute and restore the
        // PiP position from localStorage.
        const initialMode = stored || "pip";
        host.dataset.viewMode = initialMode;
        if (initialMode === "pip") _restorePipPositionFromStorage(host);
        const titleEl = host.querySelector(".lexy-avatar-title");
        if (titleEl) titleEl.textContent = `Lexy — ${initialMode}`;

        // Header click handlers for the four buttons. Use a delegated
        // listener so we don't have to re-bind when the header re-renders.
        host.querySelectorAll("[data-view-mode-btn]").forEach((btn) => {
            btn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                const m = btn.getAttribute("data-view-mode-btn");
                if (m === "close") hide();
                else set(m);
            });
        });

        // Drag wiring — the whole header bar is the grab handle (minus
        // buttons; those still work). Document-level move/up listeners
        // so the drag survives the cursor leaving the header.
        const header = host.querySelector(".lexy-avatar-header");
        if (header) header.addEventListener("mousedown", _onDragStart);
        document.addEventListener("mousemove", _onDragMove);
        document.addEventListener("mouseup", _onDragEnd);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.viewMode = { set, hide, show };
})();
