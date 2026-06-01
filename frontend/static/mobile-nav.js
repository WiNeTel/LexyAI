/*
 * mobile-nav.js — Hamburger / Drawer-Steuerung für schmale Screens.
 *
 * Auf dem Desktop ist die Sidebar eine normale Grid-Spalte. Ab ≤820px
 * (siehe style.css) wird sie zu einem von links einschiebbaren Drawer.
 * Dieses Skript öffnet/schließt den Drawer über die Klasse
 * ``nav-open`` auf <body>. Reines Vanilla-JS, keine Abhängigkeiten;
 * läuft nach app.js und fasst dessen Tab-Logik nicht an, sondern hängt
 * sich nur additiv an die vorhandenen .tab-btn-Buttons.
 */
(function () {
    "use strict";

    function init() {
        var body = document.body;
        var hamburger = document.getElementById("nav-hamburger");
        var backdrop = document.getElementById("nav-backdrop");
        var sidebar = document.getElementById("sidebar");

        // Ohne Hamburger (z.B. anderes Markup) bricht das Skript still ab.
        if (!hamburger || !sidebar) return;

        function isOpen() {
            return body.classList.contains("nav-open");
        }

        function open() {
            body.classList.add("nav-open");
            hamburger.setAttribute("aria-expanded", "true");
        }

        function close() {
            body.classList.remove("nav-open");
            hamburger.setAttribute("aria-expanded", "false");
        }

        function toggle() {
            if (isOpen()) close();
            else open();
        }

        hamburger.addEventListener("click", function (ev) {
            ev.stopPropagation();
            toggle();
        });

        if (backdrop) {
            backdrop.addEventListener("click", close);
        }

        // Tab-Wechsel schließt den Drawer, damit der gewählte Tab
        // sofort sichtbar ist. Delegiert über die Tab-Navigation.
        var tabnav = document.getElementById("tabnav");
        if (tabnav) {
            tabnav.addEventListener("click", function (ev) {
                if (ev.target.closest(".tab-btn")) close();
            });
        }

        // Escape schließt den offenen Drawer.
        document.addEventListener("keydown", function (ev) {
            if (ev.key === "Escape" && isOpen()) close();
        });

        // Wird das Fenster über den Breakpoint hinaus vergrößert (z.B.
        // Tablet-Drehung), darf kein "offener" Zustand zurückbleiben,
        // sonst klebt der Drawer-Overlay auf dem Desktop-Layout.
        var mq = window.matchMedia("(min-width: 821px)");
        function onChange(e) {
            if (e.matches) close();
        }
        if (mq.addEventListener) mq.addEventListener("change", onChange);
        else if (mq.addListener) mq.addListener(onChange); // ältere Safari/iOS
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
