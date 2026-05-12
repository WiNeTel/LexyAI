/*
 * Avatar World — WebSocket listener.
 *
 * The avatar layer doesn't own its own WebSocket — it reuses the main
 * connection in app.js to avoid a second client session. app.js's
 * `handleMessage` default-case forwards every `avatar.*` frame here via
 * `window.LexyAvatar.onWS(data)`.
 *
 * This module dispatches incoming messages to the driver subsystems
 * (emotion / activity / outfit / attention / lip-sync / background /
 * state) registered via `register(type, handler)`.
 */
(() => {
    "use strict";

    const handlers = new Map();
    const seen = [];        // tiny ring of recent frames for debugging
    const SEEN_MAX = 50;

    function register(type, handler) {
        if (typeof handler !== "function") return;
        if (!handlers.has(type)) handlers.set(type, []);
        handlers.get(type).push(handler);
    }

    function dispatch(data) {
        if (!data || typeof data !== "object" || !data.type) return;
        const type = String(data.type);
        if (!type.startsWith("avatar.")) return;

        // Debug ring — kept small so devtools doesn't drown.
        seen.push({ t: Date.now(), type, payload: data.payload });
        if (seen.length > SEEN_MAX) seen.shift();

        const list = handlers.get(type) || [];
        for (const h of list) {
            try {
                h(data.payload || {});
            } catch (err) {
                console.warn("avatar.dispatch failed for", type, err);
            }
        }
    }

    function send(action, extras = {}) {
        // Sends a frontend → backend control frame on the main WS, if
        // available. Silently no-ops when the socket isn't open.
        const ws = window.Lexy && window.Lexy.ws;
        if (!ws || ws.readyState !== 1) return false;
        ws.send(JSON.stringify({
            type: "avatar.request",
            payload: { action, ...extras },
        }));
        return true;
    }

    // Public surface — app.js calls .onWS() when an avatar.* frame
    // arrives, the rest is for drivers + console debugging.
    window.LexyAvatar = window.LexyAvatar || {};
    window.LexyAvatar.net = {
        register,
        dispatch,
        send,
        get _recent() { return seen.slice(-20); },
    };
    window.LexyAvatar.onWS = dispatch;
})();
