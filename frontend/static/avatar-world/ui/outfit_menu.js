/*
 * Avatar World — Outfit/Accessory toggle menu.
 *
 * Tiny popover that lists every part the outfit_driver detected on the
 * loaded GLB, with a checkbox per part. Anchored to the 👕 button in
 * the PiP header — opens on click, closes on outside-click or another
 * header button.
 *
 * Renders nothing when the avatar has zero toggleable parts (e.g.
 * Michelle / Soldier / single-mesh exports).
 */
(() => {
    "use strict";

    const HOST_ID = "lexy-avatar-pip";
    const BUTTON_ID = "lexy-avatar-outfit-btn";
    const MENU_ID = "lexy-avatar-outfit-menu";

    let menuEl = null;
    let buttonEl = null;
    let outsideListener = null;

    function _host() {
        return document.getElementById(HOST_ID);
    }

    function _build(host) {
        // Menu panel.
        menuEl = document.createElement("div");
        menuEl.id = MENU_ID;
        menuEl.className = "lexy-avatar-outfit-menu";
        menuEl.hidden = true;
        host.appendChild(menuEl);

        // Button — only inserted if the header has a controls list. We
        // sneak it in before the close button so existing layout stays
        // the same.
        const controls = host.querySelector(".lexy-avatar-controls");
        if (controls) {
            buttonEl = document.createElement("button");
            buttonEl.id = BUTTON_ID;
            buttonEl.title = "Outfit / Accessoires";
            buttonEl.textContent = "👕";
            const closeBtn = controls.querySelector("[data-view-mode-btn='close']");
            if (closeBtn) {
                controls.insertBefore(buttonEl, closeBtn);
            } else {
                controls.appendChild(buttonEl);
            }
            buttonEl.addEventListener("click", (ev) => {
                ev.stopPropagation();
                toggle();
            });
        }
    }

    function _render() {
        if (!menuEl) return;
        const drv = window.LexyAvatar && window.LexyAvatar.outfit;
        if (!drv || !drv.hasParts || !drv.hasParts()) {
            menuEl.innerHTML = "<div class='lexy-avatar-outfit-empty'>"
                + "Kein Outfit-Toggle für diesen Avatar."
                + "</div>";
            return;
        }
        const parts = drv.listParts();
        let html = "<div class='lexy-avatar-outfit-title'>Outfit</div>";
        for (const part of parts) {
            const checked = part.enabled ? "checked" : "";
            html += "<label class='lexy-avatar-outfit-row'>"
                + "<input type='checkbox' data-key='" + part.key + "' " + checked + ">"
                + "<span>" + part.label + "</span>"
                + "<span class='lexy-avatar-outfit-count'>"
                + part.count + " mesh" + (part.count === 1 ? "" : "es")
                + "</span>"
                + "</label>";
        }
        html += "<div class='lexy-avatar-outfit-actions'>"
            + "<button data-action='all-on'>Alles an</button>"
            + "<button data-action='all-off'>Alles aus</button>"
            + "</div>";
        menuEl.innerHTML = html;
    }

    function _onChange(ev) {
        if (ev.target && ev.target.tagName === "INPUT") {
            const drv = window.LexyAvatar && window.LexyAvatar.outfit;
            if (drv) drv.setPart(ev.target.dataset.key, ev.target.checked);
        }
    }

    function _onClick(ev) {
        const drv = window.LexyAvatar && window.LexyAvatar.outfit;
        if (!drv) return;
        const action = ev.target && ev.target.dataset && ev.target.dataset.action;
        if (action === "all-on") {
            drv.setAll(true);
            _render();
        } else if (action === "all-off") {
            drv.setAll(false);
            _render();
        }
    }

    function show() {
        if (!menuEl) return;
        _render();
        menuEl.hidden = false;
        // Outside-click handler — only installed while open.
        outsideListener = (ev) => {
            if (!menuEl) return;
            if (menuEl.contains(ev.target)) return;
            if (buttonEl && buttonEl.contains(ev.target)) return;
            hide();
        };
        // Defer one tick so the click that opened us doesn't close us.
        setTimeout(() => document.addEventListener("click", outsideListener), 0);
    }

    function hide() {
        if (!menuEl) return;
        menuEl.hidden = true;
        if (outsideListener) {
            document.removeEventListener("click", outsideListener);
            outsideListener = null;
        }
    }

    function toggle() {
        if (!menuEl) return;
        if (menuEl.hidden) show();
        else hide();
    }

    function init() {
        const host = _host();
        if (!host) return;
        if (menuEl) return;          // already initialised
        _build(host);
        menuEl.addEventListener("change", _onChange);
        menuEl.addEventListener("click", _onClick);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.outfitMenu = { show, hide, toggle, render: _render };
})();
