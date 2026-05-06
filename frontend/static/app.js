/*
 * Lexy AI — Full GUI frontend
 * Vanilla JS, no build step. Talks to the FastAPI backend on the same origin.
 *
 * Features:
 *   - Tab navigation (Chat / Memory / Plugins / Voice / Sessions / Settings)
 *   - WebSocket chat with reasoning + tool-call + brain events
 *   - TTS audio playback via WebAudio API (gapless scheduling)
 *   - Voice visualizer (frequency-domain analyser on the AudioContext)
 *   - STT microphone via MediaRecorder → WS binary → voice_gemma4
 *   - Memory browser with search / store / delete
 *   - Plugin manager with enable / disable
 *   - Voice settings for CosyVoice (speed, voice, narrator mode, test button)
 *   - Session history browser
 *   - System settings with live brain-config patch
 *   - Toast notifications for scheduler triggers and autonomous thoughts
 */

(() => {
    const $ = (id) => document.getElementById(id);
    const q = (sel, root = document) => root.querySelector(sel);
    const qa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

    // ── DOM refs ────────────────────────────────────────────────────
    const chatWindow = $("chat-window");
    const input = $("input");
    const sendBtn = $("send");
    const brainSelect = $("brain");
    const ttsToggle = $("tts-toggle");
    const micBtn = $("mic-btn");
    const clearChatBtn = $("clear-chat-btn");
    const refreshBtn = $("refresh-btn");
    const newSessionBtn = $("new-session-btn");
    const sessionPill = $("session-pill");
    const msgTemplate = $("msg-template");
    const uploadImageBtn = $("upload-image-btn");
    const uploadFileBtn = $("upload-file-btn");
    const uploadImageInput = $("upload-image-input");
    const uploadFileInput = $("upload-file-input");
    const composerAttachments = $("composer-attachments");

    // ── Phase 9.12: Rollenspiel-Tab DOM refs ──────────────────────
    // Eigener Composer + Window für die RP-Sessions, sodass der
    // Chat-Tab und der RP-Tab parallel je eine aktive Session
    // haben können (state.chatSessionId / state.rpSessionId).
    const rpChatWindow = $("rp-chat-window");
    const rpInput = $("rp-input");
    const rpSend = $("rp-send");
    const rpBrain = $("rp-brain");
    const rpTtsToggle = $("rp-tts-toggle");
    const rpMicBtn = $("rp-mic-btn");
    const rpClearBtn = $("rp-clear-btn");
    const rpNewSessionBtn = $("rp-new-session-btn");
    const rpSessionPill = $("rp-session-pill");
    const rpUploadImageBtn = $("rp-upload-image-btn");
    const rpUploadFileBtn = $("rp-upload-file-btn");
    const rpUploadImageInput = $("rp-upload-image-input");
    const rpUploadFileInput = $("rp-upload-file-input");
    const rpComposerAttachments = $("rp-composer-attachments");

    // Returns the DOM/state pair for whichever chat-context the user
    // is currently looking at (Chat tab vs Rollenspiel tab). Used by
    // sendMessage(), the streaming bubble code, and the upload UI so
    // each tab works on its own session/composer/window without
    // duplicating the entire send pipeline.
    function getActiveChatContext() {
        if (state.activeTab === "rp") {
            return {
                kind: "rp",
                input: rpInput,
                window: rpChatWindow,
                send: rpSend,
                brain: rpBrain,
                pill: rpSessionPill,
                clearBtn: rpClearBtn,
                attachmentsEl: rpComposerAttachments,
                sessionId: state.rpSessionId,
            };
        }
        return {
            kind: "chat",
            input: input,
            window: chatWindow,
            send: sendBtn,
            brain: brainSelect,
            pill: sessionPill,
            clearBtn: clearChatBtn,
            attachmentsEl: composerAttachments,
            sessionId: state.chatSessionId,
        };
    }

    const brandVersion = $("brand-version");
    const sysState = $("sys-state");
    const sysSession = $("sys-session");
    const sysWs = $("sys-ws");
    const sigThinking = $("sig-thinking");
    const sigSpeaking = $("sig-speaking");
    const servicesList = $("services-list");

    const arcReactorCanvas = $("arc-reactor");
    const toastContainer = $("toast-container");

    // ── AudioPlayer with analyser for visualization ────────────────
    class AudioPlayer {
        constructor(onSpeakingChange) {
            this.ctx = null;
            this.analyser = null;
            this.destination = null;
            this.nextStartTime = 0;
            this.pending = 0;
            this.activeSources = [];
            this.speaking = false;
            this.onSpeakingChange = onSpeakingChange || (() => {});
        }

        ensureContext() {
            if (this.ctx === null || this.ctx.state === "closed") {
                const Ctx = window.AudioContext || window.webkitAudioContext;
                if (!Ctx) return null;
                this.ctx = new Ctx();
                this.analyser = this.ctx.createAnalyser();
                this.analyser.fftSize = 256;
                this.analyser.smoothingTimeConstant = 0.75;
                this.analyser.connect(this.ctx.destination);
                this.destination = this.analyser;
                this.nextStartTime = 0;
            }
            if (this.ctx.state === "suspended") {
                this.ctx.resume().catch(() => {});
            }
            return this.ctx;
        }

        _setSpeaking(flag) {
            if (this.speaking !== flag) {
                this.speaking = flag;
                try { this.onSpeakingChange(flag); } catch (e) { /* noop */ }
            }
        }

        async enqueue(arrayBuffer) {
            const ctx = this.ensureContext();
            if (!ctx) return;

            this.pending += 1;
            const copy = arrayBuffer.slice(0);
            let audioBuffer;
            try {
                audioBuffer = await ctx.decodeAudioData(copy);
            } catch (err) {
                console.warn("tts decode failed", err);
                this.pending -= 1;
                if (this.pending === 0 && this.activeSources.length === 0) {
                    this._setSpeaking(false);
                }
                return;
            }

            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.destination || ctx.destination);

            const now = ctx.currentTime;
            const startTime = Math.max(now + 0.02, this.nextStartTime);
            source.start(startTime);
            this.nextStartTime = startTime + audioBuffer.duration;
            this.activeSources.push(source);
            this._setSpeaking(true);

            source.onended = () => {
                const idx = this.activeSources.indexOf(source);
                if (idx !== -1) this.activeSources.splice(idx, 1);
                this.pending -= 1;
                if (this.pending === 0 && this.activeSources.length === 0) {
                    this._setSpeaking(false);
                }
            };
        }

        stop() {
            for (const source of this.activeSources) {
                try { source.stop(); } catch (e) { /* already stopped */ }
            }
            this.activeSources = [];
            this.pending = 0;
            this.nextStartTime = 0;
            this._setSpeaking(false);
        }
    }

    // ── Arc Reactor Visualizer — Jarvis-style 3D rings + spectrum ──
    // A circular visualizer with three concentric rings:
    //   - Outer ring: frequency-domain spectrum bars radiating outward
    //   - Middle ring: rotating tick marks (constant) for "core spinning"
    //   - Inner ring: pulsing glow that tracks overall amplitude
    // Idle mode (no TTS speaking) runs a calm breathing pulse. When
    // `ai_thinking` is set the outer ring gets a cyan hue.
    class ArcReactorVisualizer {
        constructor(canvas, audioPlayer) {
            this.canvas = canvas;
            this.ctx2d = canvas.getContext("2d");
            this.audio = audioPlayer;
            this.running = false;
            this.dataArray = null;
            this.raf = 0;
            this.angle = 0;
            this.idlePhase = 0;
            this.thinking = false;
        }

        start() {
            if (this.running) return;
            this.running = true;
            this.loop();
        }
        stop() {
            this.running = false;
            if (this.raf) cancelAnimationFrame(this.raf);
        }

        setThinking(flag) { this.thinking = !!flag; }

        resize() {
            const parent = this.canvas.parentElement;
            if (!parent) return;
            const rect = parent.getBoundingClientRect();
            const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
            const cssWidth = Math.min(rect.width - 40, 520);
            const cssHeight = 160;
            this.canvas.style.width = cssWidth + "px";
            this.canvas.style.height = cssHeight + "px";
            this.canvas.width = Math.floor(cssWidth * dpr);
            this.canvas.height = Math.floor(cssHeight * dpr);
            this.ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        loop() {
            if (!this.running) return;
            this.render();
            this.angle += 0.012;
            this.idlePhase += 0.04;
            this.raf = requestAnimationFrame(() => this.loop());
        }

        render() {
            const cssWidth = parseFloat(this.canvas.style.width) || this.canvas.width;
            const cssHeight = parseFloat(this.canvas.style.height) || this.canvas.height;
            const ctx = this.ctx2d;
            ctx.clearRect(0, 0, cssWidth, cssHeight);

            const cx = cssWidth / 2;
            const cy = cssHeight / 2;
            const radius = Math.min(cssWidth, cssHeight) * 0.42;

            // Read analyser data if available
            const analyser = this.audio.analyser;
            let freqData = null;
            let amplitude = 0;
            if (analyser) {
                if (!this.dataArray || this.dataArray.length !== analyser.frequencyBinCount) {
                    this.dataArray = new Uint8Array(analyser.frequencyBinCount);
                }
                analyser.getByteFrequencyData(this.dataArray);
                freqData = this.dataArray;
                let sum = 0;
                for (let i = 0; i < freqData.length; i++) sum += freqData[i];
                amplitude = (sum / freqData.length) / 255;
            }

            const speaking = this.audio.speaking;
            const baseHue = this.thinking ? "#7c9eff" : "#ff7a1a";
            const hotHue = this.thinking ? "#a8c1ff" : "#ffb060";

            // ── Background halo
            const haloGradient = ctx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 1.4);
            haloGradient.addColorStop(0, this.thinking ? "rgba(124,158,255,0.18)" : "rgba(255,122,26,0.18)");
            haloGradient.addColorStop(1, "rgba(0,0,0,0)");
            ctx.fillStyle = haloGradient;
            ctx.beginPath();
            ctx.arc(cx, cy, radius * 1.4, 0, Math.PI * 2);
            ctx.fill();

            // ── Outer ring: frequency bars radiating outward
            const bars = 64;
            const step = freqData ? Math.max(1, Math.floor(freqData.length / bars)) : 1;
            for (let i = 0; i < bars; i++) {
                const theta = (i / bars) * Math.PI * 2 - Math.PI / 2;
                let v = 0;
                if (freqData) {
                    let s = 0;
                    for (let j = 0; j < step; j++) s += freqData[i * step + j] || 0;
                    v = s / step / 255;
                } else {
                    v = 0.15 + 0.1 * Math.sin(this.idlePhase + i * 0.25);
                }
                const inner = radius * 0.78;
                const outer = inner + v * radius * 0.55 + 2;
                const x1 = cx + Math.cos(theta) * inner;
                const y1 = cy + Math.sin(theta) * inner;
                const x2 = cx + Math.cos(theta) * outer;
                const y2 = cy + Math.sin(theta) * outer;

                ctx.strokeStyle = speaking || this.thinking ? hotHue : baseHue;
                ctx.globalAlpha = 0.4 + v * 0.6;
                ctx.lineWidth = 2.2;
                ctx.lineCap = "round";
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }
            ctx.globalAlpha = 1;

            // ── Middle ring: rotating tick marks
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(this.angle);
            const tickCount = 32;
            for (let i = 0; i < tickCount; i++) {
                const theta = (i / tickCount) * Math.PI * 2;
                const isMajor = i % 4 === 0;
                const innerR = radius * 0.55;
                const outerR = radius * (isMajor ? 0.68 : 0.62);
                const x1 = Math.cos(theta) * innerR;
                const y1 = Math.sin(theta) * innerR;
                const x2 = Math.cos(theta) * outerR;
                const y2 = Math.sin(theta) * outerR;
                ctx.strokeStyle = baseHue;
                ctx.globalAlpha = isMajor ? 0.55 : 0.25;
                ctx.lineWidth = isMajor ? 1.8 : 1.0;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }
            ctx.globalAlpha = 1;
            ctx.restore();

            // ── Inner ring: pulsing core
            const idlePulse = 0.5 + 0.5 * Math.sin(this.idlePhase * 0.8);
            const corePulse = speaking ? amplitude : idlePulse * 0.35;
            const coreRadius = radius * (0.32 + corePulse * 0.08);
            const coreGradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreRadius);
            coreGradient.addColorStop(0, this.thinking ? "#d9e4ff" : "#ffe3cc");
            coreGradient.addColorStop(0.4, hotHue);
            coreGradient.addColorStop(1, this.thinking ? "rgba(124,158,255,0)" : "rgba(255,122,26,0)");
            ctx.fillStyle = coreGradient;
            ctx.beginPath();
            ctx.arc(cx, cy, coreRadius, 0, Math.PI * 2);
            ctx.fill();

            // ── Thin border ring on top of the core
            ctx.strokeStyle = baseHue;
            ctx.globalAlpha = 0.9;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, radius * 0.4, 0, Math.PI * 2);
            ctx.stroke();
            ctx.globalAlpha = 1;
        }
    }

    // ── Mic recorder ───────────────────────────────────────────────
    class MicRecorder {
        constructor() {
            this.stream = null;
            this.recorder = null;
            this.chunks = [];
            this.active = false;
            this.mimeType = this._pickMimeType();
        }

        _pickMimeType() {
            if (typeof MediaRecorder === "undefined") return "";
            const candidates = [
                "audio/webm;codecs=opus",
                "audio/webm",
                "audio/ogg;codecs=opus",
                "audio/ogg",
                "audio/mp4",
            ];
            for (const mime of candidates) {
                if (MediaRecorder.isTypeSupported(mime)) return mime;
            }
            return "";
        }

        async start() {
            if (typeof navigator === "undefined" || !navigator.mediaDevices) {
                throw new Error("getUserMedia not available");
            }
            if (this.active) return;
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const opts = this.mimeType ? { mimeType: this.mimeType } : {};
            this.recorder = new MediaRecorder(this.stream, opts);
            this.chunks = [];
            this.recorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) this.chunks.push(e.data);
            };
            this.recorder.start(250);
            this.active = true;
        }

        async stop() {
            if (!this.active) return null;
            this.active = false;
            return new Promise((resolve) => {
                this.recorder.onstop = () => {
                    const blob = new Blob(this.chunks, {
                        type: this.mimeType || "audio/webm",
                    });
                    if (this.stream) {
                        for (const track of this.stream.getTracks()) track.stop();
                        this.stream = null;
                    }
                    this.recorder = null;
                    resolve(blob);
                };
                this.recorder.stop();
            });
        }
    }

    // ── State ──────────────────────────────────────────────────────
    const state = {
        ws: null,
        // Phase 9.12: ``sessionId`` always points to the *active tab*'s
        // session. The two per-tab ids below are remembered so a tab
        // switch restores the right session without going to the
        // backend. ``sessionId`` syncs from one of them on every
        // ``switchTab(chat|rp)`` call.
        sessionId: localStorage.getItem("lexy_last_session") || null,
        chatSessionId: localStorage.getItem("lexy_last_chat_session")
            || localStorage.getItem("lexy_last_session")
            || null,
        rpSessionId: localStorage.getItem("lexy_last_rp_session") || null,
        clientId: null,
        currentAssistantBubble: null,
        currentReasoningBubble: null,
        currentBrain: null,
        currentAssistantText: "",
        ttsEnabled: false,
        sending: false,
        activeTab: "chat",
        audio: null,
        visualizer: null,
        mic: new MicRecorder(),
        micActive: false,
        // Upload manifests waiting to be sent with the next chat message.
        // Each entry: {kind, upload_id, filename, size, mime, url, data_url?, ...}
        pendingAttachments: [],
        dashboardLayout: [],
        dashboardWidgets: {},
        dashboardEditing: false,
        dashboardClockInterval: null,
        // ── Projects ─────────────────────────────────────────────
        projects: [],
        projectsById: {},
        activeProjectId: localStorage.getItem("lexy.activeProjectId") || "default",
        projectsModalSelectedId: null,
        // ── Scheduler ─────────────────────────────────────────────
        schedulerTimers: [],
        schedulerTab: "active", // "active" | "recurring" | "inactive"
    };
    state.audio = new AudioPlayer((speaking) => {
        if (sigSpeaking) {
            sigSpeaking.textContent = speaking ? "true" : "false";
            sigSpeaking.style.color = speaking ? "var(--accent)" : "var(--text-dim)";
        }
    });
    state.visualizer = new ArcReactorVisualizer(arcReactorCanvas, state.audio);
    // The arc reactor runs constantly (idle breathing pulse when silent).
    state.visualizer.start();

    // ── Helpers ────────────────────────────────────────────────────
    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function toast(title, body, timeoutMs = 6000) {
        const el = document.createElement("div");
        el.className = "toast";
        el.innerHTML = `<div class="toast-title">${escapeHtml(title)}</div><div class="toast-body">${escapeHtml(body)}</div>`;
        toastContainer.appendChild(el);
        setTimeout(() => el.remove(), timeoutMs);
    }

    function setSending(busy) {
        sendBtn.disabled = busy;
        sendBtn.textContent = busy ? "…" : "Senden";
        // HoloMat: screen outline state + sounds
        if (window.LexyHolo) {
            if (busy) {
                window.LexyHolo.outline.setState("thinking");
                window.LexyHolo.avatar.setMood("thinking");
            } else {
                window.LexyHolo.outline.setState(null);
                window.LexyHolo.avatar.setMood("idle");
            }
        }
    }

    function appendMessage(role, text = "", extraClass = "") {
        const node = msgTemplate.content.cloneNode(true);
        const wrapper = node.querySelector(".msg");
        wrapper.classList.add(role);
        if (extraClass) wrapper.classList.add(extraClass);
        wrapper.querySelector(".role").textContent = role;
        wrapper.querySelector(".bubble").textContent = text;

        // Add message-actions only for user/assistant bubbles (not system,
        // not tool, not error). We attach the action bar here; the actual
        // history index is wired up later by `finalizeTurnMessages()` once
        // the backend confirms the pair has landed in SessionStore.
        if (role === "user" || role === "assistant") {
            const actions = buildMessageActions(wrapper, role);
            wrapper.appendChild(actions);
        }

        // Phase 9.12: render into whichever tab the user's looking at.
        // The streaming flow always lands in the *active* tab — the
        // user just clicked Send there. Cross-tab notifications
        // (character_pulse from a stale RP-session while the user is
        // in chat) are routed separately by their explicit session_id.
        const targetWindow = (state.activeTab === "rp" && rpChatWindow)
            ? rpChatWindow
            : chatWindow;
        targetWindow.appendChild(wrapper);
        targetWindow.scrollTop = targetWindow.scrollHeight;
        return wrapper.querySelector(".bubble");
    }

    function systemNote(text) { appendMessage("system", text); }
    function showError(text) { appendMessage("assistant", text, "error"); }

    // ── Message action bar (edit/delete/regenerate) ────────────────
    function buildMessageActions(wrapper, role) {
        const actions = document.createElement("div");
        actions.className = "msg-actions";

        const editBtn = document.createElement("button");
        editBtn.className = "msg-action";
        editBtn.textContent = "✎ edit";
        editBtn.title = "Edit this message";
        editBtn.addEventListener("click", () => startMessageEdit(wrapper));
        actions.appendChild(editBtn);

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "msg-action danger";
        deleteBtn.textContent = "✕ delete";
        deleteBtn.title = "Delete this message";
        deleteBtn.addEventListener("click", () => deleteMessage(wrapper));
        actions.appendChild(deleteBtn);

        if (role === "assistant") {
            const regenBtn = document.createElement("button");
            regenBtn.className = "msg-action primary";
            regenBtn.textContent = "↻ regenerate";
            regenBtn.title = "Drop this reply and run the last user message again";
            regenBtn.addEventListener("click", () => regenerateMessage(wrapper));
            actions.appendChild(regenBtn);
        }

        return actions;
    }

    // The chat window is a flat list of .msg bubbles (plus system /
    // reasoning / tool inserts). To find the history index of a given
    // bubble we walk the whole window and count .msg.user / .msg.assistant
    // bubbles up to the target — that's exactly the SessionStore index.
    function historyIndexOf(wrapper) {
        let idx = -1;
        let i = 0;
        for (const el of chatWindow.querySelectorAll(".msg.user, .msg.assistant")) {
            if (el === wrapper) {
                idx = i;
                break;
            }
            i += 1;
        }
        return idx;
    }

    function startMessageEdit(wrapper) {
        if (wrapper.classList.contains("editing")) return;
        const bubble = wrapper.querySelector(".bubble");
        if (!bubble) return;
        const currentText = bubble.textContent;
        wrapper.classList.add("editing");

        const form = document.createElement("div");
        form.className = "edit-form bubble";
        const textarea = document.createElement("textarea");
        textarea.className = "edit-textarea";
        textarea.value = currentText;
        textarea.rows = Math.max(3, Math.min(20, currentText.split("\n").length + 1));

        const buttons = document.createElement("div");
        buttons.className = "edit-buttons";
        const cancelBtn = document.createElement("button");
        cancelBtn.className = "btn";
        cancelBtn.textContent = "Cancel";
        const saveBtn = document.createElement("button");
        saveBtn.className = "btn primary";
        saveBtn.textContent = "Save";
        buttons.appendChild(cancelBtn);
        buttons.appendChild(saveBtn);

        form.appendChild(textarea);
        form.appendChild(buttons);

        bubble.replaceWith(form);
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);

        const finish = (newBubble) => {
            wrapper.classList.remove("editing");
            form.replaceWith(newBubble);
        };

        cancelBtn.addEventListener("click", () => {
            const restore = document.createElement("div");
            restore.className = "bubble";
            restore.textContent = currentText;
            finish(restore);
        });

        saveBtn.addEventListener("click", async () => {
            const newText = textarea.value;
            const idx = historyIndexOf(wrapper);
            if (idx < 0) {
                toast("Error", "Can't locate message in history");
                return;
            }
            try {
                const resp = await fetch(
                    `/api/v1/sessions/${encodeURIComponent(state.sessionId)}/messages/${idx}`,
                    {
                        method: "PATCH",
                        headers: { "content-type": "application/json" },
                        body: JSON.stringify({ content: newText }),
                    },
                );
                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(`HTTP ${resp.status}: ${err}`);
                }
                const updated = document.createElement("div");
                updated.className = "bubble";
                updated.textContent = newText;
                finish(updated);
                toast("Message updated", `history[${idx}] saved`);
            } catch (err) {
                toast("Error", err.message);
            }
        });

        textarea.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                saveBtn.click();
            }
            if (e.key === "Escape") {
                e.preventDefault();
                cancelBtn.click();
            }
        });
    }

    async function deleteMessage(wrapper) {
        const idx = historyIndexOf(wrapper);
        if (idx < 0) {
            wrapper.remove();
            return;
        }
        if (!confirm("Diese Nachricht wirklich löschen?")) return;
        try {
            const resp = await fetch(
                `/api/v1/sessions/${encodeURIComponent(state.sessionId)}/messages/${idx}`,
                { method: "DELETE" },
            );
            if (!resp.ok && resp.status !== 404) {
                throw new Error(`HTTP ${resp.status}`);
            }
            wrapper.remove();
            toast("Message deleted", `history[${idx}] dropped`);
        } catch (err) {
            toast("Error", err.message);
        }
    }

    function regenerateMessage(wrapper) {
        if (state.sending) {
            toast("Busy", "Warte bis die aktuelle Antwort fertig ist");
            return;
        }
        // Find the user bubble right before this assistant bubble and
        // remove everything downstream (this assistant bubble, any
        // trailing tool/reasoning markers, tool-chips etc.) so the new
        // stream can cleanly replace it.
        let node = wrapper.previousSibling;
        // Collect all siblings from `wrapper` onwards
        const toRemove = [wrapper];
        let after = wrapper.nextSibling;
        while (after) {
            toRemove.push(after);
            after = after.nextSibling;
        }
        for (const el of toRemove) {
            if (el.parentNode) el.parentNode.removeChild(el);
        }

        // Ask the backend to drop the last pair + rerun
        state.audio.stop();
        state.currentAssistantBubble = null;
        state.currentReasoningBubble = null;
        state.currentAssistantText = "";
        state.sending = true;
        setSending(true);

        if (!wsSend({
            type: "regenerate",
            session_id: state.sessionId,
            brain: brainSelect.value,
        })) {
            showError("WebSocket not connected.");
            state.sending = false;
            setSending(false);
        }
    }

    function appendToolIndicator(toolName, args) {
        const wrapper = document.createElement("div");
        wrapper.className = "msg tool";
        const label = document.createElement("div");
        label.className = "role";
        label.textContent = "tool call";
        const bubble = document.createElement("div");
        bubble.className = "bubble tool-bubble";
        const argsPreview = Object.entries(args || {})
            .map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(", ");
        bubble.innerHTML = `<span class="tool-name">🔧 ${escapeHtml(toolName)}</span>` +
            (argsPreview ? `<span class="tool-args">${escapeHtml(argsPreview)}</span>` : "");
        wrapper.appendChild(label);
        wrapper.appendChild(bubble);
        chatWindow.appendChild(wrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function appendToolResult(toolName, text) {
        const clean = String(text || "")
            .replace(/^<tool_result>\s*/i, "")
            .replace(/\s*<\/tool_result>$/i, "")
            .trim();
        if (!clean) return;
        const wrapper = document.createElement("div");
        wrapper.className = "msg tool";
        const label = document.createElement("div");
        label.className = "role";
        label.textContent = "tool result";
        const bubble = document.createElement("div");
        bubble.className = "bubble tool-result-bubble";
        bubble.textContent = clean.length > 280 ? clean.slice(0, 280) + "…" : clean;
        wrapper.appendChild(label);
        wrapper.appendChild(bubble);
        chatWindow.appendChild(wrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function appendBrainBadge(brain, thinking) {
        const wrapper = document.createElement("div");
        wrapper.className = "msg brain-badge";
        const bubble = document.createElement("div");
        bubble.className = "brain-badge-bubble";
        let icon, label;
        if (brain === "a4b") {
            icon = "🧠"; label = "A4B (deep)";
        } else if (brain === "multi") {
            icon = "🌀"; label = "MULTI (4B multimodal)";
        } else {
            icon = "⚡"; label = "E4B (fast)";
        }
        const suffix = thinking ? " · thinking" : "";
        bubble.textContent = `${icon} ${label}${suffix}`;
        wrapper.appendChild(bubble);
        chatWindow.appendChild(wrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function appendReasoningBubble() {
        const wrapper = document.createElement("div");
        wrapper.className = "msg reasoning";
        const label = document.createElement("div");
        label.className = "role";
        label.textContent = "reasoning";
        const details = document.createElement("details");
        details.className = "bubble reasoning-bubble";
        details.open = true;
        const summary = document.createElement("summary");
        summary.textContent = "💭 Denkt nach…";
        const body = document.createElement("div");
        body.className = "reasoning-body";
        details.appendChild(summary);
        details.appendChild(body);
        wrapper.appendChild(label);
        wrapper.appendChild(details);
        chatWindow.appendChild(wrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        details.dataset.finalized = "false";
        return wrapper;
    }

    // ── Tab navigation ─────────────────────────────────────────────
    function switchTab(name) {
        const prev = state.activeTab;
        state.activeTab = name;
        qa(".tab-btn").forEach((btn) =>
            btn.classList.toggle("active", btn.dataset.tab === name)
        );
        qa(".tab-panel").forEach((panel) =>
            panel.classList.toggle("active", panel.dataset.tab === name)
        );
        // Phase 9.12: Wenn der User zwischen Chat- und RP-Tab wechselt,
        // muss state.sessionId auf die zur Tab passende Session
        // umschalten und der Backend-WS auf diese Session "abonnieren"
        // (das Backend filtert WS-Broadcasts nicht; das macht der
        // Frontend-Dispatcher anhand session_id, aber der Send-Path
        // braucht trotzdem die richtige id).
        if (name === "chat" && prev !== "chat") {
            if (state.chatSessionId) {
                state.sessionId = state.chatSessionId;
                _syncSessionUI("chat");
            }
        }
        if (name === "rp" && prev !== "rp") {
            // RP-Session lazy laden — wenn Mike noch keine hat, legt
            // loadRoleplay() implizit eine an.
            if (state.rpSessionId) {
                state.sessionId = state.rpSessionId;
                _syncSessionUI("rp");
            }
        }
        if (name === "dashboard") loadDashboard();
        if (name === "memory") loadMemoryBrowse();
        if (name === "plugins") loadPlugins();
        if (name === "voice") loadVoiceConfig();
        if (name === "sessions") loadSessions();
        if (name === "scheduler") loadScheduler();
        if (name === "characters") loadCharacters();
        if (name === "rp") loadRoleplay();
        if (name === "lorebooks") loadLorebooks();
        if (name === "settings") loadSettings();
        if (name === "chat") state.visualizer.resize();
    }

    function _syncSessionUI(kind) {
        // Update the session-pill of the active tab + the small
        // header-id in the sidebar's "Session" line.
        const pill = kind === "rp" ? rpSessionPill : sessionPill;
        if (pill && state.sessionId) {
            pill.textContent = state.sessionId.slice(0, 8) + "…";
            pill.title = state.sessionId;
        }
        if (sysSession && state.sessionId) {
            sysSession.textContent = state.sessionId.slice(0, 8) + "…";
        }
    }
    qa(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // ── WebSocket ──────────────────────────────────────────────────
    function connect() {
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${proto}//${window.location.host}/ws`;
        const ws = new WebSocket(url);
        ws.binaryType = "blob";
        state.ws = ws;
        sysWs.textContent = "connecting…";

        ws.onopen = () => {
            sysWs.textContent = "open";
            sysWs.style.color = "var(--ok)";
            ws.send(JSON.stringify({ type: "get_signals" }));
            ws.send(JSON.stringify({ type: "get_plugins" }));
        };

        ws.onclose = () => {
            sysWs.textContent = "closed – reconnecting…";
            sysWs.style.color = "var(--err)";
            state.ws = null;
            setTimeout(connect, 2000);
        };

        ws.onerror = () => {
            sysWs.textContent = "error";
            sysWs.style.color = "var(--err)";
        };

        ws.onmessage = (event) => {
            if (event.data instanceof Blob) {
                event.data.arrayBuffer()
                    .then((buf) => state.audio.enqueue(buf))
                    .catch((err) => console.warn("tts blob read failed", err));
                return;
            }
            if (event.data instanceof ArrayBuffer) {
                state.audio.enqueue(event.data);
                return;
            }
            let data;
            try { data = JSON.parse(event.data); } catch { return; }
            handleMessage(data);
        };
    }

    function wsSend(obj) {
        if (state.ws && state.ws.readyState === 1) {
            state.ws.send(JSON.stringify(obj));
            return true;
        }
        return false;
    }

    // Bridge for sub-IIFEs (coder approval modal, future panels) that
    // need to send raw WS frames without re-importing the whole chat
    // state. Set once after the IIFE so subsequent sub-modules can rely
    // on it.
    window._lexySendWs = wsSend;
    // Mirror state.sessionId via a getter so sub-IIFEs don't snapshot
    // a stale value at load-time.
    if (!Object.getOwnPropertyDescriptor(window, "_lexySessionId")) {
        Object.defineProperty(window, "_lexySessionId", {
            get() { return state.sessionId || ""; },
        });
    }

    function handleMessage(data) {
        switch (data.type) {
            case "welcome":
                state.clientId = data.client_id;
                // Restore last session from localStorage, fall back to
                // server-assigned ID for brand-new visitors.
                if (!state.sessionId) {
                    state.sessionId = data.session_id;
                }
                updateSessionPill();
                brandVersion.textContent = `v${data.version}`;
                systemNote(`Connected. session=${state.sessionId}`);

                // Auto-restore previous conversation so the chat isn't
                // empty after a page reload or server restart.
                if (state.sessionId && chatWindow.children.length === 0) {
                    _autoRestoreSession(state.sessionId);
                }
                break;

            case "signals_snapshot": {
                const s = data.signals || {};
                sysState.textContent = s.system_state || "—";
                sysState.style.color = s.system_state === "ready" ? "var(--ok)" : "var(--warn)";
                sigThinking.textContent = s.ai_thinking ? "true" : "false";
                sigThinking.style.color = s.ai_thinking
                    ? "var(--reason)"
                    : "var(--text-dim)";
                state.visualizer.setThinking(!!s.ai_thinking);
                break;
            }

            case "plugins_list":
                break;

            case "brain":
                state.currentBrain = data.brain || "e4b";
                appendBrainBadge(state.currentBrain, Boolean(data.thinking));
                break;

            case "reasoning": {
                if (state.currentReasoningBubble === null) {
                    state.currentReasoningBubble = appendReasoningBubble();
                }
                const rb = state.currentReasoningBubble.querySelector(".reasoning-body");
                if (rb) rb.textContent += data.text || "";
                chatWindow.scrollTop = chatWindow.scrollHeight;
                break;
            }

            case "chunk":
                if (state.currentReasoningBubble !== null) {
                    const details = state.currentReasoningBubble.querySelector("details");
                    if (details && details.dataset.finalized !== "true") {
                        details.dataset.finalized = "true";
                        details.open = false;
                        const sum = details.querySelector("summary");
                        if (sum) sum.textContent = "💭 Reasoning (anzeigen)";
                    }
                    state.currentReasoningBubble = null;
                }
                if (state.currentAssistantBubble === null) {
                    state.currentAssistantBubble = appendMessage("assistant", "");
                }
                state.currentAssistantBubble.textContent += data.text || "";
                state.currentAssistantText += data.text || "";
                chatWindow.scrollTop = chatWindow.scrollHeight;
                break;

            case "tool_call":
                appendToolIndicator(data.tool, data.arguments || {});
                state.currentAssistantBubble = null;
                state.currentReasoningBubble = null;
                break;

            case "tool_result":
                appendToolResult(data.tool, data.text || "");
                break;

            case "done":
                if (state.currentAssistantBubble && Array.isArray(data.tools_used) && data.tools_used.length > 0) {
                    const chips = document.createElement("div");
                    chips.className = "tool-chips";
                    const unique = [...new Set(data.tools_used)];
                    for (const tool of unique) {
                        const chip = document.createElement("span");
                        chip.className = "tool-chip";
                        chip.textContent = `🔧 ${tool}`;
                        chips.appendChild(chip);
                    }
                    state.currentAssistantBubble.parentElement.appendChild(chips);
                }
                if (state.ttsEnabled && state.currentAssistantText.trim()) {
                    wsSend({ type: "tts", text: state.currentAssistantText });
                }
                state.currentAssistantBubble = null;
                state.currentReasoningBubble = null;
                state.currentAssistantText = "";
                state.currentBrain = null;
                state.sending = false;
                setSending(false);
                if (window.LexyHolo) window.LexyHolo.sound.play("receive");
                break;

            case "tts_done":
                break;

            case "regenerating":
                // Backend confirmed it popped the last pair + cleaned
                // memory. Nothing to render — chunks will follow next.
                break;

            case "stt_started":
                break;

            case "stt_result":
                if (data.text) {
                    input.value = data.text;
                    systemNote(`🎤 ${data.text}`);
                } else if (data.error) {
                    showError(`STT: ${data.error}`);
                }
                break;

            case "scheduler_triggered":
                toast(`⏰ ${data.kind || "Timer"}`, `${data.label || "(ohne Label)"} · ${data.fired_at || ""}`, 10000);
                if (window._lexyScheduler) window._lexyScheduler.onTriggered(data);
                break;

            case "scheduler_list":
                if (window._lexyScheduler) window._lexyScheduler.onList(data);
                break;

            case "scheduler_created":
                if (window._lexyScheduler) window._lexyScheduler.onCreated(data);
                break;

            case "scheduler_cancelled":
                if (window._lexyScheduler) window._lexyScheduler.onCancelled(data);
                break;

            case "scheduler_updated":
                if (window._lexyScheduler) window._lexyScheduler.onUpdated(data);
                break;

            case "proactive_message":
                if (window._lexyScheduler) window._lexyScheduler.onProactive(data);
                break;

            case "agent_task_spawned":
                toast("🤖 Agent-Task", `${data.label || "(ohne Label)"} gestartet · ${data.persona || ""}`, 8000);
                break;

            case "agent_task_done":
                toast("🤖 Agent-Task fertig", `${data.label || "(ohne Label)"}`, 10000);
                if (data.output) {
                    systemNote(`Agent-Task "${data.label || ""}":\n${data.output}`);
                }
                break;

            case "autonomous_thought": {
                const actions = Array.isArray(data.actions) ? data.actions : [];
                let body = data.text || "";
                if (actions.length) {
                    const lines = actions.map((a) => {
                        const icon = a.skipped ? "⛔" : (a.ok === false ? "⚠️" : "🔧");
                        const name = a.tool || "?";
                        const arg = a.args ? ` ${JSON.stringify(a.args).slice(0, 80)}` : "";
                        return `${icon} ${name}${arg}`;
                    });
                    body = `${body}\n\nAktionen:\n${lines.join("\n")}`.trim();
                }
                toast(`💭 ${data.mode || "thought"}`, body, actions.length ? 12000 : 8000);
                // Also render the thought as a chat bubble so it persists
                // in the visible conversation — toasts vanish after a few
                // seconds. Only for the active session.
                if (data.session_id === state.sessionId && data.text) {
                    appendMessage("thought", `💭 *${data.text}*`);
                }
                break;
            }

            case "thinking_toggled":
                if (window._lexyThinking) window._lexyThinking.onToggled(data.active);
                break;

            case "thinking_result":
                if (window._lexyThinking) window._lexyThinking.onResult(data.mode, data.text);
                break;

            case "dashboard_widgets": {
                // Backend sends an array of {widget_id, title, data, ...}.
                // Flatten into a dict keyed by widget_id so the renderer can
                // look up data by id. Tolerate a dict payload too.
                const incoming = data.widgets;
                const dict = {};
                if (Array.isArray(incoming)) {
                    for (const w of incoming) {
                        if (w && w.widget_id) dict[w.widget_id] = w.data || {};
                    }
                } else if (incoming && typeof incoming === "object") {
                    Object.assign(dict, incoming);
                }
                state.dashboardWidgets = dict;
                renderDashboardGrid(state.dashboardLayout, state.dashboardWidgets);
                break;
            }

            case "dashboard_widget_update":
                if (data.widget_id && state.dashboardWidgets) {
                    state.dashboardWidgets[data.widget_id] = data.data;
                    const container = document.querySelector(`.dashboard-widget[data-widget-id="${data.widget_id}"] .widget-body`);
                    if (container) renderWidget(data.widget_id, data.data, container);
                }
                break;

            case "dashboard_layout":
                state.dashboardLayout = data.layout || [];
                renderDashboardGrid(state.dashboardLayout, state.dashboardWidgets);
                break;

            case "dashboard_layout_saved":
                toast("Dashboard", "Layout saved");
                break;

            // ── Expert Panel messages ─────────────────────────────
            case "panel_started":
                toast("Expert Panel", `Panel "${data.topic}" gestartet (${(data.roles||[]).length} Rollen)`);
                break;
            case "panel_agent_speaking":
                toast("Panel", `${data.role || "?"} denkt nach…`, 3000);
                break;
            case "panel_agent_message":
                _renderPanelMessage(data);
                break;
            case "panel_round_done":
                toast("Panel", `Runde ${data.round} abgeschlossen${data.convergence?.converged ? " — Konsens erreicht!" : ""}`, 5000);
                break;
            case "panel_synthesizing":
                toast("Panel", "Synthese läuft…", 3000);
                break;
            case "panel_done":
                toast("Expert Panel", "Panel abgeschlossen!", 8000);
                _renderPanelResult(data);
                break;
            case "panel_error":
                toast("Panel Error", data.error || "Unknown", 6000);
                break;

            // ── Auto-Agent messages ──────────────────────────────────
            case "agent_started":
                toast("Agent", `"${data.name}" gestartet: ${(data.task||"").slice(0,80)}`, 6000);
                break;
            case "agent_progress":
                toast("Agent", `${data.name}: ${data.detail} (${data.iteration}/${data.max_iterations})`, 4000);
                break;
            case "agent_message":
                toast("Agent", `${data.name}: ${(data.message||"").slice(0,120)}`, 6000);
                break;
            case "agent_done":
                toast("Agent", `"${data.name}" fertig!`, 8000);
                break;
            case "agent_error":
                toast("Agent Error", `${data.name}: ${data.error||"Unknown"}`, 6000);
                break;

            // ── Knowledge Acquisition messages ───────────────────────
            case "knowledge_job_progress":
                toast("Knowledge", `${data.topic||""}: ${data.pages_processed}/${data.pages_found} Seiten, ${data.chunks_stored} Chunks`, 4000);
                break;
            case "knowledge_job_done":
                toast("Knowledge", `Recherche abgeschlossen: ${data.chunks_stored||0} Chunks gespeichert`, 8000);
                break;
            case "knowledge_job_error":
                toast("Knowledge Error", data.error || "Unknown", 6000);
                break;

            // ── YouTube messages ──────────────────────────────────────
            case "youtube_play":
                _openYouTubePlayer(data);
                break;

            // ── Spotify messages ──────────────────────────────────────
            case "spotify_now_playing":
                toast("Spotify", `${data.track||"?"} — ${data.artist||"?"}`, 5000);
                break;

            // ── MCP messages ─────────────────────────────────────────
            case "mcp_server_connected":
                toast("MCP", `Server "${data.name}" verbunden (${data.tools_count||0} Tools)`, 5000);
                break;
            case "mcp_server_disconnected":
                toast("MCP", `Server "${data.name}" getrennt`, 4000);
                break;
            case "mcp_server_error":
                toast("MCP Error", `${data.name}: ${data.error||"Unknown"}`, 6000);
                break;

            // ── Orchestrator messages ────────────────────────────────
            case "orchestrator_task_delegated":
                toast("Orchestrator", `Task delegiert: ${(data.task||"").slice(0,80)}`, 5000);
                break;
            case "orchestrator_task_started":
                toast("Orchestrator", `Agent "${data.persona||data.name||"?"}" gestartet`, 4000);
                break;
            case "orchestrator_task_done":
                toast("Orchestrator", `Agent fertig: ${(data.result_summary||"").slice(0,100)}`, 8000);
                break;
            case "orchestrator_task_error":
                toast("Orchestrator", `Agent gescheitert: ${(data.error||"").slice(0,100)}`, 6000);
                break;
            case "orchestrator_decision":
                toast("Orchestrator", (data.answer||"").slice(0,150), 6000);
                break;
            case "orchestrator_scheduled":
                toast("Orchestrator", `Task geplant: ${(data.label||"").slice(0,80)}`, 5000);
                break;

            case "error":
                showError(data.error || "Unknown error");
                state.currentAssistantBubble = null;
                state.currentReasoningBubble = null;
                state.currentAssistantText = "";
                state.sending = false;
                setSending(false);
                if (window.LexyHolo) window.LexyHolo.sound.play("error");
                break;

            case "project_created":
            case "project_updated":
            case "project_deleted":
            case "project_archived":
            case "project_unarchived":
                // Re-fetch on any project mutation so every tab stays
                // consistent. Cheap (one HTTP roundtrip, small payload).
                loadProjects().then(() => {
                    if (state.activeTab === "sessions") loadSessions();
                }).catch((err) => console.warn("project refresh failed:", err));
                break;

            case "session_updated":
                // A session was moved or renamed — refresh the list if
                // we're looking at it.
                if (state.activeTab === "sessions") loadSessions();
                break;

            case "session_kind_changed":
                // Phase 9.12: backend tagged a session as kind=rp (most
                // commonly because a character was just attached). If
                // the user is currently in the *chat* tab and the
                // session in question is their active chat session,
                // gently hint that the conversation has moved over.
                if (data && data.kind === "rp" && data.session_id) {
                    if (data.session_id === state.chatSessionId) {
                        // Bookkeeping: chat-tab no longer owns this id.
                        state.chatSessionId = null;
                        try { localStorage.removeItem("lexy_last_chat_session"); } catch (_e) {}
                    }
                    if (data.session_id === state.sessionId &&
                        state.activeTab === "chat") {
                        toast(
                            "Session ist jetzt RP",
                            "Charakter angeheftet — wechsle in den 🎬 Rollenspiel-Tab.",
                        );
                        // Move it over to the RP tab's memory so
                        // switchTab("rp") can pick it up.
                        state.rpSessionId = data.session_id;
                        try { localStorage.setItem("lexy_last_rp_session", data.session_id); } catch (_e) {}
                    } else if (data.session_id === state.rpSessionId) {
                        // Already in RP — nothing to do, but make sure
                        // the localStorage anchor is fresh.
                        try { localStorage.setItem("lexy_last_rp_session", data.session_id); } catch (_e) {}
                    }
                    if (state.activeTab === "sessions") loadSessions();
                }
                break;

            // ── Character chat ──
            case "character_list":
                if (window._lexyCharacters) window._lexyCharacters.onList(data);
                break;
            case "character_created":
                if (window._lexyCharacters) window._lexyCharacters.onCreated(data);
                break;
            case "character_updated":
                if (window._lexyCharacters) window._lexyCharacters.onUpdated(data);
                break;
            case "character_deleted":
                if (window._lexyCharacters) window._lexyCharacters.onDeleted(data);
                break;
            case "character_session_mode":
                if (window._lexyCharacters) window._lexyCharacters.onSessionMode(data);
                break;

            case "simulation_started":
                if (data && data.session_id === state.sessionId) {
                    updateSimUI(true, data.interval_minutes);
                    if (data.ok !== false) {
                        toast("🎬 Simulation", `läuft alle ${data.interval_minutes || "?"} Minuten`, 4000);
                    }
                }
                break;
            case "simulation_stopped":
                if (data && data.session_id === state.sessionId) {
                    updateSimUI(false, null);
                    if (data.was_running) toast("🎬 Simulation", "gestoppt", 3000);
                }
                break;
            case "simulation_status":
                if (data && data.session_id === state.sessionId) {
                    updateSimUI(!!data.running, null);
                }
                break;
            case "character_session_get":
                if (window._lexyCharacters) window._lexyCharacters.onSessionGet(data);
                break;
            case "character_round_start":
                if (window._lexyCharacters) window._lexyCharacters.onRoundStart(data);
                break;
            case "character_turn":
                if (window._lexyCharacters) window._lexyCharacters.onTurn(data);
                break;
            case "character_turn_updated":
                if (window._lexyCharacters) window._lexyCharacters.onTurnUpdated(data);
                break;
            case "character_turn_deleted":
                if (window._lexyCharacters) window._lexyCharacters.onTurnDeleted(data);
                break;
            case "character_turn_edit_ack":
            case "character_turn_delete_ack":
            case "character_turn_regenerate_ack":
                if (data && data.ok === false) {
                    showError(`Turn-Action: ${data.error || "Fehler"}`);
                }
                break;
            case "character_turn_audio":
                if (window._lexyCharacters) window._lexyCharacters.onTurnAudio(data);
                break;
            case "character_round_done":
                if (window._lexyCharacters) window._lexyCharacters.onRoundDone(data);
                break;
            case "character_round_error":
                if (window._lexyCharacters) window._lexyCharacters.onRoundError(data);
                break;

            case "rp_director_start_ack":
                if (data && data.ok === false) {
                    showError(`RP-Director: ${data.error || "konnte nicht starten"}`);
                }
                break;
            case "rp_director_started":
                // Echo broadcast from the plugin — user already sees the
                // intent note from handleSlashCommand. Stay quiet here.
                break;
            case "rp_director_scenario_proposed":
                if (data && data.scenario) {
                    const s = data.scenario;
                    const bits = [];
                    if (s.setting) bits.push(`Setting: ${s.setting}`);
                    if (s.mood) bits.push(`Stimmung: ${s.mood}`);
                    toast("🎬 Director", bits.join(" · ") || "Scenario-Vorschlag aktualisiert", 4000);
                }
                break;
            case "rp_director_characters_proposed":
                if (data && Array.isArray(data.characters)) {
                    const names = data.characters.map((c) => c && c.name).filter(Boolean).join(", ");
                    toast("🎬 Director", `Charaktere: ${names || data.characters.length}`, 4000);
                }
                break;
            case "rp_director_committed":
                if (data) {
                    const count = (data.characters || []).length;
                    systemNote(`🎬 RP-Setup committed — ${count} Charakter${count === 1 ? "" : "e"} aktiv. Lexy übernimmt.`);
                }
                break;
            case "rp_director_cancelled":
                systemNote("🎬 RP-Setup abgebrochen.");
                break;
            case "rp_director_error":
                showError(`RP-Director: ${(data && data.error) || "Fehler"}`);
                break;

            // ── Deep-Research console (Phase 12) ─────────────────────
            case "deep_research_started":
            case "deep_research_planned":
            case "deep_research_searching":
            case "deep_research_subquery_done":
            case "deep_research_subquery_progress":
            case "deep_research_synthesising":
            case "deep_research_done":
            case "deep_research_error":
            case "deep_research_cancelled":
                if (window._lexyResearch) window._lexyResearch.onEvent(data);
                break;

            // ── Coder approvals + console (Phase 10.B+C) ─────────────
            case "coder_approval_request":
                if (window._lexyCoder) window._lexyCoder.onApprovalRequest(data);
                break;
            case "coder_approval_ack":
                // Server acknowledges — we already cleared the modal locally on click.
                break;
            case "coder_started":
            case "coder_planned":
            case "coder_step_attempt":
            case "coder_step_done":
            case "coder_step_retry":
            case "coder_done":
            case "coder_error":
            case "coder_cancelled":
                if (window._lexyCoder) window._lexyCoder.onTaskEvent(data);
                break;

            default:
                break;
        }
    }

    // ── Slash commands ─────────────────────────────────────────────
    // Lightweight prefix interception. Anything that doesn't match
    // returns false and falls through to the normal chat path.
    function handleSlashCommand(text) {
        if (!text.startsWith("/")) return false;
        const space = text.indexOf(" ");
        const cmd = (space === -1 ? text : text.slice(0, space)).toLowerCase();
        const rest = space === -1 ? "" : text.slice(space + 1).trim();
        if (cmd === "/rp") {
            if (!state.sessionId) {
                showError("Keine aktive Session — neu starten.");
                return true;
            }
            if (!wsSend({
                type: "rp_director_start",
                session_id: state.sessionId,
                user_intent: rest,
            })) {
                showError("WebSocket not connected.");
                return true;
            }
            appendMessage("user", text);
            input.value = "";
            const intentNote = rest ? ` (Wunsch: ${rest})` : "";
            systemNote(`🎬 RP-Director aktiviert${intentNote}. Beschreib einfach, was du dir vorstellst.`);
            return true;
        }
        return false;
    }

    // ── Upload handling ────────────────────────────────────────────
    //
    // Two buttons in the composer toolbar trigger hidden <input type=file>
    // elements. Picked files are POSTed to the right /api/v1/uploads/<kind>
    // endpoint and the manifest is stashed in state.pendingAttachments.
    // sendMessage() picks up the array, includes it in the chat WS frame,
    // and the bubble-renderer shows previews/links inline.

    function inferUploadKind(file) {
        // Image files are obvious from the MIME type.
        const mime = (file.type || "").toLowerCase();
        if (mime.startsWith("image/")) return "image";
        if (mime.startsWith("audio/")) return "audio";
        // Treat common code/config extensions as code so the LLM gets a
        // language hint. Everything else falls through to "document".
        const name = (file.name || "").toLowerCase();
        const codeExts = [
            "py","pyi","js","mjs","cjs","jsx","ts","tsx","go","rs","java",
            "kt","kts","swift","c","h","cpp","cc","hpp","cs","rb","php",
            "sh","bash","zsh","ps1","lua","r","jl","ex","exs","erl","hs",
            "ml","fs","vue","svelte","css","scss","sass","less","sql",
            "graphql","gql","proto","tf","hcl","json","yaml","yml","toml",
            "xml","makefile","dockerfile","mk",
        ];
        const ext = name.includes(".") ? name.split(".").pop() : name;
        if (codeExts.includes(ext)) return "code";
        return "document";
    }

    async function uploadFile(file, forcedKind = null) {
        const kind = forcedKind || inferUploadKind(file);
        const form = new FormData();
        form.append("session_id", state.sessionId || "default");
        form.append("file", file);
        const resp = await fetch(`/api/v1/uploads/${kind}`, {
            method: "POST",
            body: form,
        });
        if (!resp.ok) {
            let detail = await resp.text();
            try { detail = JSON.parse(detail).detail || detail; } catch (_e) {}
            throw new Error(`Upload (${kind}) failed: ${detail}`);
        }
        const manifest = await resp.json();
        // Backend doesn't echo `kind` in every legacy field — make sure it's set.
        if (!manifest.kind) manifest.kind = kind;
        return manifest;
    }

    async function handleFilePick(file, forcedKind = null) {
        if (!file) return;
        const btn = forcedKind === "image" ? uploadImageBtn : uploadFileBtn;
        if (btn) btn.dataset.busy = "true";
        try {
            const manifest = await uploadFile(file, forcedKind);
            state.pendingAttachments.push(manifest);
            renderPendingAttachments();
            if (window.LexyHolo && window.LexyHolo.sound) {
                window.LexyHolo.sound.play("send");
            }
        } catch (err) {
            console.error("upload failed", err);
            showError(String(err && err.message ? err.message : err));
        } finally {
            if (btn) btn.dataset.busy = "false";
        }
    }

    function renderPendingAttachments() {
        if (!composerAttachments) return;
        if (!state.pendingAttachments.length) {
            composerAttachments.replaceChildren();
            composerAttachments.hidden = true;
            return;
        }
        composerAttachments.hidden = false;
        composerAttachments.replaceChildren();
        for (const att of state.pendingAttachments) {
            composerAttachments.appendChild(buildAttachmentChip(att));
        }
    }

    function buildAttachmentChip(att) {
        const chip = document.createElement("div");
        chip.className = "attachment-chip";

        const isImage = att.kind === "image";
        if (isImage && (att.data_url || att.url)) {
            const img = document.createElement("img");
            img.className = "chip-thumb";
            img.src = att.data_url || att.url;
            img.alt = att.filename || "image";
            chip.appendChild(img);
        } else {
            const icon = document.createElement("div");
            icon.className = "chip-icon";
            icon.textContent = ({
                document: "📄",
                code: "📜",
                audio: "🎵",
                image: "🖼",
            }[att.kind] || "📎");
            chip.appendChild(icon);
        }

        const name = document.createElement("span");
        name.className = "chip-name";
        name.textContent = att.filename || `${att.kind || "file"}`;
        name.title = att.filename || "";
        chip.appendChild(name);

        const meta = document.createElement("span");
        meta.className = "chip-meta";
        const sizeKb = att.size ? Math.max(1, Math.round(att.size / 1024)) : 0;
        const parts = [];
        if (sizeKb) parts.push(`${sizeKb} KB`);
        if (att.kind === "code" && att.language) parts.push(att.language);
        if (att.kind === "document" && typeof att.chunks_indexed === "number") {
            parts.push(`${att.chunks_indexed} chunks`);
        }
        if (att.kind === "audio" && att.transcript) {
            parts.push(`${att.transcript.length} chars`);
        }
        meta.textContent = parts.join(" · ");
        chip.appendChild(meta);

        const remove = document.createElement("button");
        remove.className = "chip-remove";
        remove.type = "button";
        remove.textContent = "✕";
        remove.title = "Anhang entfernen";
        remove.addEventListener("click", () => {
            const idx = state.pendingAttachments.indexOf(att);
            if (idx !== -1) {
                state.pendingAttachments.splice(idx, 1);
                renderPendingAttachments();
                // Best-effort server-side cleanup; ignore errors.
                if (att.upload_id) {
                    fetch(`/api/v1/uploads/${att.upload_id}`, { method: "DELETE" })
                        .catch(() => {});
                }
            }
        });
        chip.appendChild(remove);
        return chip;
    }

    function clearPendingAttachments() {
        state.pendingAttachments.length = 0;
        renderPendingAttachments();
    }

    function renderUserMessageWithAttachments(text, attachments) {
        // Show the user message bubble plus a per-attachment row inside it.
        // We don't reuse appendMessage's plain textContent because we need
        // to inject <img> / file-link nodes safely.
        const bubble = appendMessage("user", text || "");
        if (!bubble) return;
        for (const att of attachments) {
            if (att.kind === "image") {
                const img = document.createElement("img");
                img.className = "bubble-image";
                img.src = att.url || att.data_url || "";
                img.alt = att.filename || "image";
                bubble.appendChild(img);
            } else {
                const link = document.createElement("a");
                link.className = "bubble-file";
                link.href = att.url || "#";
                link.target = "_blank";
                link.rel = "noopener";
                link.textContent = `📎 ${att.filename || att.kind} `;
                const meta = document.createElement("span");
                meta.style.color = "var(--muted)";
                const sizeKb = att.size ? Math.max(1, Math.round(att.size / 1024)) : 0;
                meta.textContent = sizeKb ? `(${sizeKb} KB)` : "";
                link.appendChild(meta);
                bubble.appendChild(link);
            }
        }
    }

    if (uploadImageBtn && uploadImageInput) {
        uploadImageBtn.addEventListener("click", () => uploadImageInput.click());
        uploadImageInput.addEventListener("change", async () => {
            const file = uploadImageInput.files && uploadImageInput.files[0];
            uploadImageInput.value = "";
            if (file) await handleFilePick(file, "image");
        });
    }
    if (uploadFileBtn && uploadFileInput) {
        uploadFileBtn.addEventListener("click", () => uploadFileInput.click());
        uploadFileInput.addEventListener("change", async () => {
            const file = uploadFileInput.files && uploadFileInput.files[0];
            uploadFileInput.value = "";
            if (file) await handleFilePick(file);
        });
    }

    // Drag-and-drop directly onto the composer.
    if (input) {
        const onDrag = (ev) => { ev.preventDefault(); ev.stopPropagation(); };
        input.addEventListener("dragenter", onDrag);
        input.addEventListener("dragover", onDrag);
        input.addEventListener("drop", async (ev) => {
            ev.preventDefault();
            const files = ev.dataTransfer && ev.dataTransfer.files;
            if (!files || !files.length) return;
            for (const file of files) {
                await handleFilePick(file);
            }
        });
    }

    // ── Chat input ─────────────────────────────────────────────────
    // Phase 9.12: ``sendMessage()`` works on whichever tab is active —
    // Chat or Rollenspiel. The pre-Phase-9.12 version hard-coded
    // ``input``, ``brainSelect``, ``state.sessionId``; we now resolve
    // those from ``getActiveChatContext()`` so the same handler serves
    // both tabs without duplicating the streaming pipeline.
    function sendMessage() {
        const ctx = getActiveChatContext();
        const text = ctx.input.value.trim();
        const hasAttachments = state.pendingAttachments.length > 0;
        // Allow sending with empty text if attachments exist — common when
        // dropping an image and saying "describe this" implicitly.
        if (!text && !hasAttachments) return;
        if (state.sending) return;
        if (handleSlashCommand(text)) return;
        const attachments = state.pendingAttachments.slice();
        // Make sure ``state.sessionId`` is in sync with the active tab
        // before we send. Defensive: switchTab() already does this, but
        // some entry-points (initial load, hot resume) bypass it.
        const targetSessionId = ctx.sessionId || state.sessionId;
        if (!wsSend({
            type: "chat",
            text,
            session_id: targetSessionId,
            brain: ctx.brain.value,
            attachments,
        })) {
            showError("WebSocket not connected.");
            return;
        }
        state.audio.stop();
        renderUserMessageWithAttachments(text, attachments);
        ctx.input.value = "";
        clearPendingAttachments();
        state.currentAssistantText = "";
        state.sending = true;
        setSending(true);
        if (window.LexyHolo) window.LexyHolo.sound.play("send");
    }

    sendBtn.addEventListener("click", sendMessage);
    input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            sendMessage();
        }
    });

    // Phase 9.12: same handler for the RP composer.
    if (rpSend) rpSend.addEventListener("click", sendMessage);
    if (rpInput) rpInput.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            sendMessage();
        }
    });

    clearChatBtn.addEventListener("click", () => {
        chatWindow.innerHTML = "";
        systemNote("Chat cleared.");
    });
    if (rpClearBtn) rpClearBtn.addEventListener("click", () => {
        if (rpChatWindow) rpChatWindow.innerHTML = "";
        systemNote("RP-Window cleared.");
    });

    if (newSessionBtn) {
        newSessionBtn.addEventListener("click", newSession);
    }
    // Phase 9.12: RP-Tab hat seinen eigenen "Neue Session"-Button.
    // Beide rufen ``newSession()`` — die Funktion liest ``state.activeTab``
    // und legt eine Chat- bzw. RP-Session an.
    if (rpNewSessionBtn) {
        rpNewSessionBtn.addEventListener("click", newSession);
    }

    function _copySessionToClipboard() {
        if (!state.sessionId) return;
        navigator.clipboard.writeText(state.sessionId).then(() => {
            toast("Copied", state.sessionId, 3000);
        }).catch(() => {
            systemNote(`Session ID: ${state.sessionId}`);
        });
    }
    if (sessionPill) sessionPill.addEventListener("click", _copySessionToClipboard);
    if (rpSessionPill) rpSessionPill.addEventListener("click", _copySessionToClipboard);

    // RP TTS toggle mirrors the chat TTS toggle.
    if (rpTtsToggle) rpTtsToggle.addEventListener("click", () => {
        state.ttsEnabled = !state.ttsEnabled;
        const label = state.ttsEnabled ? "TTS on" : "TTS off";
        rpTtsToggle.dataset.active = state.ttsEnabled ? "true" : "false";
        rpTtsToggle.textContent = label;
        if (ttsToggle) {
            ttsToggle.dataset.active = state.ttsEnabled ? "true" : "false";
            ttsToggle.textContent = label;
        }
        if (state.ttsEnabled) state.audio.ensureContext();
        else state.audio.stop();
    });

    // ── loadRoleplay (Phase 9.12) ──────────────────────────────────
    async function loadRoleplay() {
        // First time the user opens the RP tab we want SOME session
        // to be active. Prefer the remembered ``rpSessionId``; if
        // that's missing or doesn't exist on the server anymore,
        // fall back to the most-recently-updated kind=rp session;
        // failing that, create a fresh one.
        if (state.rpSessionId) {
            // Best-effort verify the session still exists. If it does,
            // hop on it without re-rendering history (we trust the
            // local UI). If the user wants to re-load, they click
            // resume from the Sessions tab.
            state.sessionId = state.rpSessionId;
            _syncSessionUI("rp");
            return;
        }
        try {
            const resp = await fetch("/api/v1/sessions?kind=rp");
            if (resp.ok) {
                const body = await resp.json();
                const sessions = body.sessions || [];
                if (sessions.length > 0) {
                    // sessions are pre-sorted by updated_at desc
                    state.rpSessionId = sessions[0].id;
                    state.sessionId = sessions[0].id;
                    try { localStorage.setItem("lexy_last_rp_session", sessions[0].id); } catch (_e) {}
                    _syncSessionUI("rp");
                    return;
                }
            }
        } catch (err) {
            console.warn("loadRoleplay: failed to list rp sessions:", err);
        }
        // No RP session exists yet → create one.
        await newSession();
    }

    refreshBtn.addEventListener("click", () => {
        loadHealth();
        wsSend({ type: "get_signals" });
        if (state.activeTab === "plugins") loadPlugins();
        if (state.activeTab === "memory") loadMemoryBrowse();
        if (state.activeTab === "voice") loadVoiceConfig();
        if (state.activeTab === "sessions") loadSessions();
        if (state.activeTab === "scheduler") loadScheduler();
        if (state.activeTab === "settings") loadSettings();
    });

    ttsToggle.addEventListener("click", () => {
        state.ttsEnabled = !state.ttsEnabled;
        ttsToggle.dataset.active = state.ttsEnabled ? "true" : "false";
        ttsToggle.textContent = state.ttsEnabled ? "TTS on" : "TTS off";
        if (state.ttsEnabled) {
            state.audio.ensureContext();
        } else {
            state.audio.stop();
        }
    });

    // ── Mic (hold-to-talk) ─────────────────────────────────────────
    async function startMic() {
        if (state.micActive) return;
        try {
            await state.mic.start();
        } catch (err) {
            showError(`Mic: ${err.message}`);
            return;
        }
        state.micActive = true;
        micBtn.dataset.active = "true";
        micBtn.textContent = "🔴";
        wsSend({ type: "stt_start" });
    }

    async function stopMic() {
        if (!state.micActive) return;
        state.micActive = false;
        micBtn.dataset.active = "false";
        micBtn.textContent = "🎤";
        let blob;
        try {
            blob = await state.mic.stop();
        } catch (err) {
            showError(`Mic stop: ${err.message}`);
            return;
        }
        if (!blob || blob.size === 0) {
            showError("No audio captured.");
            return;
        }
        if (state.ws && state.ws.readyState === 1) {
            state.ws.send(await blob.arrayBuffer());
            state.ws.send(
                JSON.stringify({
                    type: "stt_end",
                    auto_chat: true,
                    session_id: state.sessionId,
                    brain: brainSelect.value,
                })
            );
            state.sending = true;
            setSending(true);
        }
    }

    micBtn.addEventListener("mousedown", startMic);
    micBtn.addEventListener("mouseup", stopMic);
    micBtn.addEventListener("mouseleave", () => { if (state.micActive) stopMic(); });
    micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startMic(); });
    micBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopMic(); });

    // ── Health / services ──────────────────────────────────────────
    async function loadHealth() {
        try {
            const resp = await fetch("/api/v1/health");
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const services = data.services || {};
            servicesList.innerHTML = "";
            for (const [key, value] of Object.entries(services)) {
                const li = document.createElement("li");
                let dot = "err";
                if (value === "ready") dot = "ok";
                else if (value === "off") dot = "warn";
                else if (/^\d+$/.test(value)) dot = value === "0" ? "warn" : "ok";
                li.innerHTML = `<span><span class="dot ${dot}"></span>${key}</span><strong>${value}</strong>`;
                servicesList.appendChild(li);
            }
            sysState.textContent = data.status || "—";
        } catch (exc) {
            servicesList.innerHTML = `<li class="muted">error: ${exc.message}</li>`;
        }
    }

    // ── Memory tab ─────────────────────────────────────────────────
    const memoryCollection = $("memory-collection");
    const memorySearchInput = $("memory-search-input");
    const memorySearchBtn = $("memory-search-btn");
    const memoryStoreInput = $("memory-store-input");
    const memoryStoreBtn = $("memory-store-btn");
    const memoryResults = $("memory-results");
    const memoryRefreshBtn = $("memory-refresh-btn");

    async function loadMemoryBrowse() {
        const col = memoryCollection.value;
        memoryResults.innerHTML = `<div class="muted">loading…</div>`;
        try {
            const resp = await fetch(`/api/v1/memory/browse?collection=${col}&limit=50`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            renderMemoryItems(data.items || [], data.total || 0, col);
        } catch (err) {
            memoryResults.innerHTML = `<div class="muted">error: ${err.message}</div>`;
        }
    }

    async function memorySearch() {
        const col = memoryCollection.value;
        const query = memorySearchInput.value.trim();
        if (!query) return loadMemoryBrowse();
        memoryResults.innerHTML = `<div class="muted">searching…</div>`;
        try {
            const resp = await fetch("/api/v1/memory/recall", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ query, collection: col, limit: 20 }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            renderMemoryItems(data.results || [], (data.results || []).length, col, true);
        } catch (err) {
            memoryResults.innerHTML = `<div class="muted">error: ${err.message}</div>`;
        }
    }

    async function memoryStore() {
        const text = memoryStoreInput.value.trim();
        const col = memoryCollection.value;
        if (!text) return;
        try {
            const resp = await fetch("/api/v1/memory/store", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ text, collection: col, metadata: {} }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            memoryStoreInput.value = "";
            toast("Memory stored", `Saved to ${col}`);
            await loadMemoryBrowse();
        } catch (err) {
            toast("Error", err.message);
        }
    }

    async function memoryDelete(id, col) {
        if (!confirm(`Delete item ${id} from ${col}?`)) return;
        try {
            const resp = await fetch("/api/v1/memory/delete", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ id, collection: col }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            toast("Deleted", id);
            await loadMemoryBrowse();
        } catch (err) {
            toast("Error", err.message);
        }
    }

    function renderMemoryItems(items, total, col, isSearch = false) {
        if (items.length === 0) {
            memoryResults.innerHTML = `<div class="muted">(empty)</div>`;
            return;
        }
        memoryResults.innerHTML = "";
        const header = document.createElement("div");
        header.className = "muted";
        header.textContent = isSearch
            ? `${items.length} result(s) for "${memorySearchInput.value}"`
            : `${items.length} of ${total} items in ${col}`;
        memoryResults.appendChild(header);

        for (const item of items) {
            const div = document.createElement("div");
            div.className = "mem-item";
            const id = item.id || "";
            const score = item.score;
            const content = item.content || "";
            const meta = item.metadata || {};
            const created = meta.created_at
                ? new Date((meta.created_at || 0) * 1000).toLocaleString()
                : "";
            div.innerHTML = `
                <div class="mem-item-head">
                    <span>${escapeHtml(id.slice(0, 12))} · ${escapeHtml(created)}</span>
                    ${score != null ? `<span class="mem-score">${score.toFixed(3)}</span>` : ""}
                </div>
                <div class="mem-item-content">${escapeHtml(content)}</div>
                <div class="mem-item-actions">
                    <button class="btn" data-action="delete">Delete</button>
                </div>
            `;
            div.querySelector('[data-action="delete"]').addEventListener("click", () =>
                memoryDelete(id, col)
            );
            memoryResults.appendChild(div);
        }
    }

    memoryRefreshBtn.addEventListener("click", loadMemoryBrowse);
    memoryCollection.addEventListener("change", loadMemoryBrowse);
    memorySearchBtn.addEventListener("click", memorySearch);
    memorySearchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") memorySearch();
    });
    memoryStoreBtn.addEventListener("click", memoryStore);
    memoryStoreInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") memoryStore();
    });

    // ── Memory wipe (collection + all) ─────────────────────────────
    const memoryWipeCollectionBtn = $("memory-wipe-collection-btn");
    const memoryWipeAllBtn = $("memory-wipe-all-btn");
    const memoryWipeModal = $("memory-wipe-modal");
    const memoryWipeClose = $("memory-wipe-close");
    const memoryWipeCancel = $("memory-wipe-cancel");
    const memoryWipeConfirm = $("memory-wipe-confirm");
    const wipeConfirmInput = $("wipe-confirm-input");
    const wipeOptCollections = $("wipe-opt-collections");
    const wipeOptSessions = $("wipe-opt-sessions");
    const wipeOptPluginData = $("wipe-opt-plugin-data");

    async function memoryWipeCollection() {
        const col = memoryCollection.value;
        if (!confirm(`Collection "${col}" wirklich komplett leeren?`)) return;
        try {
            const resp = await fetch(
                `/api/v1/memory/collection/${encodeURIComponent(col)}`,
                { method: "DELETE" },
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            toast(
                `Collection ${col} gewiped`,
                `ChromaDB: ${data.chroma} items · FTS: ${data.fts} rows`,
                7000,
            );
            await loadMemoryBrowse();
        } catch (err) {
            toast("Error", err.message);
        }
    }

    function openMemoryWipeModal() {
        memoryWipeModal.hidden = false;
        wipeConfirmInput.value = "";
        memoryWipeConfirm.disabled = true;
        wipeOptCollections.checked = true;
        wipeOptSessions.checked = true;
        wipeOptPluginData.checked = false;
        setTimeout(() => wipeConfirmInput.focus(), 50);
    }

    function closeMemoryWipeModal() {
        memoryWipeModal.hidden = true;
    }

    wipeConfirmInput.addEventListener("input", () => {
        memoryWipeConfirm.disabled = wipeConfirmInput.value.trim() !== "LÖSCHEN";
    });

    async function executeWipe() {
        if (wipeConfirmInput.value.trim() !== "LÖSCHEN") return;
        const body = {
            confirm: true,
            collections: wipeOptCollections.checked,
            sessions: wipeOptSessions.checked,
            plugin_data: wipeOptPluginData.checked,
        };
        memoryWipeConfirm.disabled = true;
        memoryWipeConfirm.textContent = "Lösche…";
        try {
            const resp = await fetch("/api/v1/memory/wipe", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!resp.ok) {
                const err = await resp.text();
                throw new Error(`HTTP ${resp.status}: ${err}`);
            }
            const data = await resp.json();

            const parts = [];
            if (data.collections && data.collections.total_chroma !== undefined) {
                parts.push(`${data.collections.total_chroma} vectors`);
                parts.push(`${data.collections.total_fts} fts rows`);
            }
            if (data.sessions && data.sessions.dropped_sessions !== undefined) {
                parts.push(`${data.sessions.dropped_sessions} sessions`);
            }
            if (data.plugin_data && data.plugin_data.dropped) {
                parts.push(`${data.plugin_data.dropped.length} plugin dirs`);
            }
            toast(
                "Memory gewiped",
                parts.join(" · ") || "alles leer",
                9000,
            );

            if (body.plugin_data) {
                toast(
                    "⚠ Restart required",
                    "Plugin data dropped — restart Lexy to reload plugins",
                    15000,
                );
            }

            closeMemoryWipeModal();
            if (state.activeTab === "memory") {
                await loadMemoryBrowse();
            }
        } catch (err) {
            toast("Error", err.message);
        } finally {
            memoryWipeConfirm.disabled = false;
            memoryWipeConfirm.textContent = "Jetzt löschen";
        }
    }

    memoryWipeCollectionBtn.addEventListener("click", memoryWipeCollection);
    memoryWipeAllBtn.addEventListener("click", openMemoryWipeModal);
    memoryWipeClose.addEventListener("click", closeMemoryWipeModal);
    memoryWipeCancel.addEventListener("click", closeMemoryWipeModal);
    memoryWipeConfirm.addEventListener("click", executeWipe);
    memoryWipeModal.addEventListener("click", (ev) => {
        if (ev.target === memoryWipeModal) closeMemoryWipeModal();
    });
    wipeConfirmInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !memoryWipeConfirm.disabled) executeWipe();
    });

    // ── Plugins tab ────────────────────────────────────────────────
    const pluginsGrid = $("plugins-grid");
    const pluginsRefreshBtn = $("plugins-refresh-btn");

    async function loadPlugins() {
        pluginsGrid.innerHTML = `<div class="muted">loading…</div>`;
        try {
            const resp = await fetch("/api/v1/plugins");
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            renderPluginCards(data.plugins || []);
        } catch (err) {
            pluginsGrid.innerHTML = `<div class="muted">error: ${err.message}</div>`;
        }
    }

    function renderPluginCards(plugins) {
        if (plugins.length === 0) {
            pluginsGrid.innerHTML = `<div class="muted">no plugins loaded</div>`;
            return;
        }
        pluginsGrid.innerHTML = "";
        for (const p of plugins) {
            const card = document.createElement("div");
            card.className = `plugin-card ${p.enabled ? "enabled" : "disabled"}`;
            card.innerHTML = `
                <h4>
                    <span>${escapeHtml(p.name)}</span>
                    <span class="plugin-version">v${escapeHtml(p.version || "?")}</span>
                </h4>
                <p>${escapeHtml(p.description || "")}</p>
                <div class="plugin-footer">
                    <span class="plugin-status">
                        <span class="dot ${p.enabled ? "ok" : "warn"}"></span>
                        ${p.enabled ? "enabled" : p.loaded ? "loaded" : "off"}
                    </span>
                    <div style="display:flex;gap:6px;">
                        <button class="btn plugin-config-btn" data-action="config">Config</button>
                        <button class="btn" data-action="toggle">
                            ${p.enabled ? "Disable" : "Enable"}
                        </button>
                    </div>
                </div>
            `;
            card.querySelector('[data-action="toggle"]').addEventListener("click", async () => {
                const action = p.enabled ? "disable" : "enable";
                try {
                    const resp = await fetch(`/api/v1/plugins/${p.name}/${action}`, { method: "POST" });
                    let body = null;
                    try { body = await resp.json(); } catch { /* non-json error */ }
                    if (!resp.ok) {
                        // Phase 9.10: backend returns 422 with
                        // {detail:{code,plugin,error}} for hard
                        // failures. Show that instead of "HTTP 500".
                        const detail = body && body.detail;
                        const msg = detail && (detail.error || detail.code)
                            ? `${detail.error || detail.code}`
                            : `HTTP ${resp.status}`;
                        throw new Error(msg);
                    }
                    if (action === "enable" && body && body.degraded) {
                        // Plugin loaded but its provider couldn't be
                        // initialised (e.g. CosyVoice server down).
                        // Don't claim success — show the reason.
                        toast(
                            `${p.name} enabled (degraded)`,
                            body.last_error || "provider not available",
                        );
                    } else {
                        toast(`Plugin ${action}d`, p.name);
                    }
                    await loadPlugins();
                } catch (err) {
                    toast(`${action} failed: ${p.name}`, err.message);
                }
            });
            card.querySelector('[data-action="config"]').addEventListener("click", () => {
                openPluginConfig(p.name);
            });
            pluginsGrid.appendChild(card);
        }
    }

    // ── Plugin Config Modal ────────────────────────────────────────
    const pluginConfigModal = $("plugin-config-modal");
    const pluginConfigTitle = $("plugin-config-title");
    const pluginConfigDesc = $("plugin-config-desc");
    const pluginConfigForm = $("plugin-config-form");
    const pluginConfigCancel = $("plugin-config-cancel");
    const pluginConfigClose = $("plugin-config-close");
    const pluginConfigSave = $("plugin-config-save");
    let pluginConfigCurrent = null;

    async function openPluginConfig(pluginName) {
        pluginConfigModal.hidden = false;
        pluginConfigTitle.textContent = `Plugin Config · ${pluginName}`;
        pluginConfigDesc.textContent = "loading…";
        pluginConfigForm.innerHTML = "";
        try {
            const resp = await fetch(`/api/v1/plugins/${pluginName}/config`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            pluginConfigCurrent = { name: pluginName, ...data };
            pluginConfigDesc.textContent = data.description || "(no description)";
            renderPluginConfigForm(data.defaults || {}, data.effective || {});
        } catch (err) {
            pluginConfigDesc.textContent = `error: ${err.message}`;
            pluginConfigCurrent = null;
        }
    }

    function renderPluginConfigForm(defaults, effective) {
        pluginConfigForm.innerHTML = "";
        const allKeys = new Set([
            ...Object.keys(defaults || {}),
            ...Object.keys(effective || {}),
        ]);
        if (allKeys.size === 0) {
            pluginConfigForm.innerHTML = `<div class="muted">This plugin declares no config keys.</div>`;
            return;
        }
        for (const key of Array.from(allKeys).sort()) {
            const row = document.createElement("div");
            row.className = "setting-row";
            const label = document.createElement("label");
            label.textContent = key;
            const currentValue = effective[key] !== undefined ? effective[key] : defaults[key];
            const input = buildInputForValue(key, currentValue);
            row.appendChild(label);
            row.appendChild(input);
            pluginConfigForm.appendChild(row);
        }
    }

    function buildInputForValue(key, value) {
        if (typeof value === "boolean") {
            const input = document.createElement("input");
            input.type = "checkbox";
            input.checked = value;
            input.dataset.configKey = key;
            input.dataset.configType = "boolean";
            return input;
        }
        if (typeof value === "number") {
            const input = document.createElement("input");
            input.type = "number";
            input.value = String(value);
            input.step = Number.isInteger(value) ? "1" : "any";
            input.dataset.configKey = key;
            input.dataset.configType = "number";
            return input;
        }
        if (Array.isArray(value)) {
            const input = document.createElement("input");
            input.type = "text";
            input.value = value.join(", ");
            input.placeholder = "comma-separated";
            input.dataset.configKey = key;
            input.dataset.configType = "array";
            return input;
        }
        // string or null/undefined
        const input = document.createElement("input");
        input.type = "text";
        input.value = value == null ? "" : String(value);
        input.dataset.configKey = key;
        input.dataset.configType = "string";
        return input;
    }

    function closePluginConfig() {
        pluginConfigModal.hidden = true;
        pluginConfigCurrent = null;
        pluginConfigForm.innerHTML = "";
    }

    async function savePluginConfig() {
        if (!pluginConfigCurrent) return;
        const patch = {};
        for (const el of pluginConfigForm.querySelectorAll("[data-config-key]")) {
            const key = el.dataset.configKey;
            const type = el.dataset.configType;
            if (type === "boolean") {
                patch[key] = el.checked;
            } else if (type === "number") {
                const n = Number(el.value);
                if (Number.isFinite(n)) patch[key] = n;
            } else if (type === "array") {
                patch[key] = el.value
                    .split(",")
                    .map((s) => s.trim())
                    .filter(Boolean);
            } else {
                patch[key] = el.value;
            }
        }
        try {
            const resp = await fetch(
                `/api/v1/plugins/${pluginConfigCurrent.name}/config`,
                {
                    method: "PATCH",
                    headers: { "content-type": "application/json" },
                    body: JSON.stringify(patch),
                },
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const note = data.applied_live
                ? "Applied live ✓"
                : data.restart_required
                    ? "Saved — restart required to apply"
                    : "Saved";
            toast(`Plugin · ${pluginConfigCurrent.name}`, note);
            closePluginConfig();
        } catch (err) {
            toast("Error", err.message);
        }
    }

    pluginConfigCancel.addEventListener("click", closePluginConfig);
    pluginConfigClose.addEventListener("click", closePluginConfig);
    pluginConfigSave.addEventListener("click", savePluginConfig);
    pluginConfigModal.addEventListener("click", (ev) => {
        if (ev.target === pluginConfigModal) closePluginConfig();
    });

    pluginsRefreshBtn.addEventListener("click", loadPlugins);

    // ── Voice tab ──────────────────────────────────────────────────
    const voiceProviders = $("voice-providers");
    const voiceSpeed = $("voice-speed");
    const voiceSpeedVal = $("voice-speed-val");
    const voiceVoice = $("voice-voice");
    const voiceNarratorMode = $("voice-narrator-mode");
    const voiceSegmentPause = $("voice-segment-pause");
    const voiceDefaultInstruct = $("voice-default-instruct");
    const voiceSaveBtn = $("voice-save-btn");
    const voiceTestBtn = $("voice-test-btn");
    const voiceRefreshBtn = $("voice-refresh-btn");

    async function loadVoiceConfig() {
        voiceProviders.innerHTML = `<div class="muted">loading…</div>`;
        try {
            const providers = await (await fetch("/api/v1/voice/providers")).json();
            voiceProviders.innerHTML = `
                <div class="kv"><span>STT providers</span><strong>${(providers.stt || []).join(", ") || "none"}</strong></div>
                <div class="kv"><span>TTS providers</span><strong>${(providers.tts || []).join(", ") || "none"}</strong></div>
                <div class="kv"><span>Active STT</span><strong>${providers.active_stt || "—"}</strong></div>
                <div class="kv"><span>Active TTS</span><strong>${providers.active_tts || "—"}</strong></div>
            `;

            const cfg = await (await fetch("/api/v1/voice/config")).json();
            if (cfg && Object.keys(cfg).length > 0) {
                voiceSpeed.value = cfg.speed != null ? cfg.speed : 1.0;
                voiceSpeedVal.textContent = Number(voiceSpeed.value).toFixed(2);
                voiceSegmentPause.value = cfg.segment_pause_ms || 80;
                voiceNarratorMode.value = cfg.narrator_mode || "full";
                voiceDefaultInstruct.value = cfg.default_instruct || "";
                voiceVoice.innerHTML = "";
                const voicesList = cfg.voices || [];
                if (voicesList.length > 0) {
                    for (const v of voicesList) {
                        const opt = document.createElement("option");
                        opt.value = v;
                        opt.textContent = v;
                        voiceVoice.appendChild(opt);
                    }
                } else if (cfg.voice) {
                    const opt = document.createElement("option");
                    opt.value = cfg.voice;
                    opt.textContent = cfg.voice;
                    voiceVoice.appendChild(opt);
                }
                voiceVoice.value = cfg.voice || voicesList[0] || "";
            }
        } catch (err) {
            voiceProviders.innerHTML = `<div class="muted">error: ${err.message}</div>`;
        }
    }

    voiceSpeed.addEventListener("input", () => {
        voiceSpeedVal.textContent = Number(voiceSpeed.value).toFixed(2);
    });

    voiceSaveBtn.addEventListener("click", async () => {
        const patch = {
            speed: parseFloat(voiceSpeed.value),
            narrator_mode: voiceNarratorMode.value,
            segment_pause_ms: parseInt(voiceSegmentPause.value, 10),
            default_instruct: voiceDefaultInstruct.value,
        };
        if (voiceVoice.value) patch.voice = voiceVoice.value;
        try {
            const resp = await fetch("/api/v1/voice/config", {
                method: "PATCH",
                headers: { "content-type": "application/json" },
                body: JSON.stringify(patch),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            toast("Voice saved", "CosyVoice config updated");
        } catch (err) {
            toast("Error", err.message);
        }
    });

    voiceTestBtn.addEventListener("click", () => {
        if (!state.ttsEnabled) ttsToggle.click();
        wsSend({
            type: "tts",
            text: "Hallo Mike, das ist ein Test der Sprachausgabe mit den aktuellen Einstellungen.",
        });
    });

    voiceRefreshBtn.addEventListener("click", loadVoiceConfig);

    // ── Sessions tab ───────────────────────────────────────────────
    const sessionsList = $("sessions-list");
    const sessionsHistory = $("sessions-history");
    const sessionsRefreshBtn = $("sessions-refresh-btn");

    async function _autoRestoreSession(sessionId) {
        // Silently restore the last session's messages into the chat
        // window so the user sees their conversation after a reload.
        try {
            const resp = await fetch(
                `/api/v1/sessions/${encodeURIComponent(sessionId)}/history`,
            );
            if (!resp.ok) return; // session gone — start fresh
            const data = await resp.json();
            const messages = data.messages || [];
            if (messages.length === 0) return;

            for (const m of messages) {
                if (m.role === "user" || m.role === "assistant") {
                    appendMessage(m.role, m.content || "");
                }
            }
            systemNote(`Session wiederhergestellt: ${sessionId} (${messages.length} Nachrichten)`);
        } catch {
            // Network error during restore — no big deal, user starts fresh
        }
    }

    async function loadSessions() {
        sessionsList.innerHTML = `<div class="muted">loading…</div>`;
        sessionsHistory.innerHTML = `<div class="muted">Select a session to see its history.</div>`;
        try {
            // ``state.sessionsShowAll`` is the per-tab toggle. When on, we
            // bypass the project filter so Mike can see every session
            // across every project — escape hatch for "where did my
            // session go?" moments.
            const showAll = !!state.sessionsShowAll;
            // Phase 9.12: optional kind filter (chat / rp / all). The
            // dropdown writes to ``state.sessionsKindFilter``; missing
            // means "all".
            const kindFilter = state.sessionsKindFilter || "all";
            const params = new URLSearchParams();
            if (showAll) {
                params.set("all", "true");
            } else if (state.activeProjectId) {
                params.set("project_id", state.activeProjectId);
            }
            if (kindFilter !== "all") {
                params.set("kind", kindFilter);
            }
            const url = `/api/v1/sessions${params.toString() ? "?" + params : ""}`;
            const data = await (await fetch(url)).json();
            let items = data.sessions || [];

            // Empty-result helper: if the active project shows nothing,
            // peek at "all" to find out whether sessions exist at all
            // and offer Mike a one-click fix instead of a dead-end
            // empty list.
            let crossProjectHint = "";
            if (items.length === 0 && !showAll) {
                try {
                    const allResp = await fetch("/api/v1/sessions?all=true");
                    if (allResp.ok) {
                        const allData = await allResp.json();
                        const everywhere = allData.sessions || [];
                        if (everywhere.length > 0) {
                            // Sessions exist in OTHER projects. Render
                            // them right away (effectively turning the
                            // toggle on for this view) and prepend a
                            // banner explaining the switch — no dead
                            // end, no scavenger hunt for the toggle.
                            items = everywhere;
                            const projName =
                                (state.projectsById[state.activeProjectId]
                                    && state.projectsById[state.activeProjectId].name)
                                || state.activeProjectId
                                || "(this project)";
                            crossProjectHint = (
                                `<div class="session-cross-hint">` +
                                `📂 Im Projekt <b>${escapeHtml(projName)}</b> ` +
                                `gibt es keine Sessions — zeige dir hier ` +
                                `alle aus allen Projekten an. ` +
                                `Klick die Session an, dann wechselt Lexy ` +
                                `automatisch ins richtige Projekt.</div>`
                            );
                        }
                    }
                } catch (_e) { /* best-effort fallback */ }
            }

            if (items.length === 0) {
                const hint = showAll
                    ? "no sessions yet"
                    : "no sessions yet — start a chat to create one";
                sessionsList.innerHTML = `<div class="muted">${hint}</div>`;
                return;
            }
            sessionsList.innerHTML = crossProjectHint;
            for (const s of items) {
                const div = document.createElement("div");
                div.className = "session-item";
                if (s.in_progress) div.classList.add("session-in-progress");
                const isActive = s.id === state.sessionId;
                // Tint the left border with the session's project color so
                // users can tell projects apart at a glance.
                const projectOfSession = state.projectsById[s.project_id || "default"];
                if (projectOfSession && projectOfSession.color) {
                    div.style.setProperty("--session-proj-color", projectOfSession.color);
                }
                const lastRole = s.last_role || "";
                const preview = s.last_preview || "(empty)";
                const userHint = s.last_user && s.last_user !== preview
                    ? `<div class="spreview spreview-user">🙂 ${escapeHtml(s.last_user)}</div>`
                    : "";
                const roleBadge = lastRole
                    ? `<span class="sid-role sid-role-${lastRole}">${lastRole}</span>`
                    : "";
                const inProgressBadge = s.in_progress
                    ? '<span class="sid-role sid-role-progress">⏳ in progress</span>'
                    : "";
                // Phase 9.12: ein kleines Tab-Badge je Session, damit
                // Mike auf den ersten Blick sieht, ob die Session Chat
                // oder RP ist.
                const kindBadge = s.kind === "rp"
                    ? '<span class="sid-role sid-role-rp" title="Rollenspiel-Session">🎬 RP</span>'
                    : '<span class="sid-role sid-role-chat" title="Chat-Session">💬</span>';
                // Prefer the derived title over the raw session id when present
                const headline = s.title
                    ? `<div class="sid sid-title">${escapeHtml(s.title)}</div>
                       <div class="sid-sub mono">${escapeHtml(s.id)}${isActive ? ' · active' : ''}</div>`
                    : `<div class="sid">${escapeHtml(s.id)}${isActive ? ' <span class="sid-active">· active</span>' : ''}</div>`;
                div.innerHTML = `
                    ${headline}
                    <div class="smeta">${kindBadge} ${s.messages} messages ${roleBadge}${inProgressBadge}</div>
                    <div class="spreview">${lastRole === "assistant" ? "🤖" : "🙂"} ${escapeHtml(preview)}</div>
                    ${userHint}
                    <div class="session-actions">
                        <button class="msg-action primary session-resume-btn">↻ Resume</button>
                        <button class="msg-action danger session-delete-btn">🗑 Delete</button>
                    </div>
                `;
                div.addEventListener("click", (ev) => {
                    if (ev.target.closest("button")) return;
                    loadSessionHistory(s.id, div);
                });
                div.addEventListener("contextmenu", (ev) => {
                    ev.preventDefault();
                    openSessionContextMenu(ev.clientX, ev.clientY, s.id);
                });
                div.querySelector(".session-resume-btn").addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    resumeSession(s.id);
                });
                div.querySelector(".session-delete-btn").addEventListener("click", async (ev) => {
                    ev.stopPropagation();
                    if (!confirm(`Session ${s.id} wirklich löschen?`)) return;
                    try {
                        const resp = await fetch(
                            `/api/v1/sessions/${encodeURIComponent(s.id)}`,
                            { method: "DELETE" },
                        );
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        toast("Session deleted", s.id);
                        await loadSessions();
                    } catch (err) {
                        toast("Error", err.message);
                    }
                });
                sessionsList.appendChild(div);
            }
        } catch (err) {
            sessionsList.innerHTML = `<div class="muted">error: ${err.message}</div>`;
        }
    }

    // ── Session switching (New / Resume) ───────────────────────────

    function updateSessionPill() {
        // Persist active session so we can restore it after page reload
        if (state.sessionId) {
            localStorage.setItem("lexy_last_session", state.sessionId);
        }
        if (!sessionPill) return;
        const short = state.sessionId
            ? state.sessionId.length > 22
                ? state.sessionId.slice(0, 20) + "…"
                : state.sessionId
            : "—";
        sessionPill.textContent = short;
        sessionPill.title = `Active session: ${state.sessionId || "(none)"} — click to copy`;
        if (sysSession) sysSession.textContent = state.sessionId || "—";
    }

    function generateSessionId() {
        // Friendly timestamp + short random suffix: sess-2026-04-11-1345-ab12
        const d = new Date();
        const pad = (n) => String(n).padStart(2, "0");
        const stamp =
            `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
            `-${pad(d.getHours())}${pad(d.getMinutes())}`;
        const rand = Math.random().toString(36).slice(2, 6);
        return `sess-${stamp}-${rand}`;
    }

    async function newSession() {
        if (state.sending) {
            toast("Busy", "Warte bis die aktuelle Antwort fertig ist");
            return;
        }
        // Phase 9.12: Welche Session wir gerade erneuern hängt vom
        // aktiven Tab ab. Im RP-Tab erzeugen wir eine RP-Session und
        // hinterlegen sie auch als ``state.rpSessionId`` (und PATCHen
        // sie kurz nach Registrieren auf ``kind=rp``); im Chat-Tab
        // bleibt es eine normale Chat-Session.
        const isRP = state.activeTab === "rp";
        state.audio.stop();
        state.sessionId = generateSessionId();
        state.currentAssistantBubble = null;
        state.currentReasoningBubble = null;
        state.currentAssistantText = "";
        const targetWindow = isRP && rpChatWindow ? rpChatWindow : chatWindow;
        targetWindow.innerHTML = "";
        if (isRP) {
            state.rpSessionId = state.sessionId;
            try { localStorage.setItem("lexy_last_rp_session", state.sessionId); } catch (_e) {}
        } else {
            state.chatSessionId = state.sessionId;
            try { localStorage.setItem("lexy_last_chat_session", state.sessionId); } catch (_e) {}
        }
        updateSessionPill();
        _syncSessionUI(isRP ? "rp" : "chat");
        systemNote(`Neue ${isRP ? "RP-" : ""}Session gestartet: ${state.sessionId}`);
        // Persist the session on the backend right away so it survives
        // a restart even if the user never sends a message. Non-blocking
        // best-effort — if it fails we log and continue.
        registerSessionWithBackend(state.sessionId).then(() => {
            // Phase 9.12: für RP-Sessions kind sofort setzen, damit
            // die Session in der RP-Filter-Liste auftaucht — auch
            // wenn der User nie einen Charakter anheftet.
            if (isRP) {
                fetch(`/api/v1/sessions/${encodeURIComponent(state.sessionId)}`, {
                    method: "PATCH",
                    headers: { "content-type": "application/json" },
                    body: JSON.stringify({ kind: "rp" }),
                }).catch((err) => console.warn("session kind PATCH failed:", err));
            }
        }).catch((err) => {
            console.warn("session register failed:", err);
        });
    }

    async function registerSessionWithBackend(sessionId, extra = {}) {
        if (!sessionId) return;
        // Tag every newly-created session with the currently active
        // project up front so the session-list filter doesn't hide it
        // the moment the user switches projects. ``extra`` overrides
        // win — the caller can pass an explicit project_id (or null
        // to deliberately leave it unassigned).
        const payload = { session_id: sessionId };
        if (state.activeProjectId) {
            payload.project_id = state.activeProjectId;
        }
        Object.assign(payload, extra);
        const resp = await fetch("/api/v1/sessions/register", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(`register HTTP ${resp.status}`);
        return resp.json();
    }

    async function resumeSession(sessionId) {
        if (state.sending) {
            toast("Busy", "Warte bis die aktuelle Antwort fertig ist");
            return;
        }
        try {
            const resp = await fetch(
                `/api/v1/sessions/${encodeURIComponent(sessionId)}/history`,
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const messages = data.messages || [];
            // ``data.project_id`` is the project the session belongs to.
            // If it doesn't match the current active project, switch the
            // active project so the sessions list and the chat agree —
            // otherwise Mike resumes a session and then immediately
            // can't find it again because the active project filter is
            // still pointing at a different bucket.
            const sessionProject = data.project_id || "default";
            if (sessionProject !== state.activeProjectId) {
                state.activeProjectId = sessionProject;
                localStorage.setItem("lexy.activeProjectId", sessionProject);
                if (typeof renderProjectsPanel === "function") {
                    try { renderProjectsPanel(); } catch (_e) {}
                }
            }

            // Phase 9.12: Wenn die Session ``kind=rp`` hat, schalten wir
            // auf den Rollenspiel-Tab um. Sonst bleibt's der Chat-Tab.
            const sessionKind = data.kind || "chat";
            const targetTab = sessionKind === "rp" ? "rp" : "chat";
            const targetWindow = sessionKind === "rp" && rpChatWindow
                ? rpChatWindow
                : chatWindow;

            // Swap state + UI
            state.audio.stop();
            state.sessionId = sessionId;
            if (sessionKind === "rp") {
                state.rpSessionId = sessionId;
                try { localStorage.setItem("lexy_last_rp_session", sessionId); } catch (_e) {}
            } else {
                state.chatSessionId = sessionId;
                try { localStorage.setItem("lexy_last_chat_session", sessionId); } catch (_e) {}
            }
            state.currentAssistantBubble = null;
            state.currentReasoningBubble = null;
            state.currentAssistantText = "";
            targetWindow.innerHTML = "";
            updateSessionPill();
            _syncSessionUI(targetTab);

            systemNote(`Session geladen: ${sessionId} (${messages.length} Nachrichten)`);

            // Re-render each message as a normal bubble. We're already
            // on the right tab so appendMessage() routes to the right
            // window automatically.
            switchTab(targetTab);
            for (const m of messages) {
                if (m.role === "user" || m.role === "assistant") {
                    appendMessage(m.role, m.content || "");
                }
            }

            toast("Session loaded", `${messages.length} messages (${sessionKind})`);
        } catch (err) {
            toast("Error", err.message);
        }
    }

    async function loadSessionHistory(sessionId, activeEl) {
        qa(".session-item").forEach((el) => el.classList.remove("active"));
        if (activeEl) activeEl.classList.add("active");
        sessionsHistory.innerHTML = `<div class="muted">loading…</div>`;
        try {
            const data = await (await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}/history`)).json();
            const msgs = data.messages || [];
            if (msgs.length === 0) {
                sessionsHistory.innerHTML = `<div class="muted">(empty)</div>`;
                return;
            }
            sessionsHistory.innerHTML = "";
            for (const m of msgs) {
                const div = document.createElement("div");
                div.className = `history-msg ${m.role}`;
                div.innerHTML = `<div class="hrole">${escapeHtml(m.role)}</div>${escapeHtml(m.content || "")}`;
                sessionsHistory.appendChild(div);
            }
        } catch (err) {
            sessionsHistory.innerHTML = `<div class="muted">error: ${err.message}</div>`;
        }
    }

    sessionsRefreshBtn.addEventListener("click", loadSessions);

    // "Alle Projekte" toggle — escape hatch for "where did my session go?"
    // moments. Starts off; persists the user's choice across reloads.
    const sessionsShowAllBtn = $("sessions-show-all-btn");
    if (sessionsShowAllBtn) {
        const persisted = localStorage.getItem("lexy.sessions.showAll") === "1";
        state.sessionsShowAll = persisted;
        sessionsShowAllBtn.dataset.active = persisted ? "true" : "false";
        sessionsShowAllBtn.addEventListener("click", () => {
            state.sessionsShowAll = !state.sessionsShowAll;
            sessionsShowAllBtn.dataset.active = state.sessionsShowAll ? "true" : "false";
            localStorage.setItem(
                "lexy.sessions.showAll",
                state.sessionsShowAll ? "1" : "0",
            );
            if (state.activeTab === "sessions") loadSessions();
        });
    }

    // Phase 9.12: kind filter dropdown wires into ``state.sessionsKindFilter``
    // and re-loads. Persisted across reloads so Mike's choice sticks.
    const sessionsKindFilter = $("sessions-kind-filter");
    if (sessionsKindFilter) {
        const persisted = localStorage.getItem("lexy.sessions.kindFilter") || "all";
        state.sessionsKindFilter = persisted;
        sessionsKindFilter.value = persisted;
        sessionsKindFilter.addEventListener("change", () => {
            state.sessionsKindFilter = sessionsKindFilter.value;
            localStorage.setItem(
                "lexy.sessions.kindFilter",
                state.sessionsKindFilter,
            );
            if (state.activeTab === "sessions") loadSessions();
        });
    }

    // ── Settings tab ───────────────────────────────────────────────
    const routingDefaultBrain = $("routing-default-brain");
    const routingSaveBtn = $("routing-save-btn");
    const brainsGrid = $("brains-grid");
    const memoryInfo = $("memory-info");
    const embeddingInfo = $("embedding-info");
    const settingsRefreshBtn = $("settings-refresh-btn");
    const brainRowTemplate = $("brain-row-template");
    const systemProfile = $("system-profile");
    const profileSaveBtn = $("profile-save-btn");
    const profileActiveBrains = $("profile-active-brains");
    const personaName = $("persona-name");
    const personaUserName = $("persona-user-name");
    const personaLanguage = $("persona-language");
    const personaIdentity = $("persona-identity");
    const personaStyle = $("persona-style");
    const personaRules = $("persona-rules");
    const personaThinking = $("persona-thinking");
    const personaThinkingLabel = $("persona-thinking-label");
    const personaSaveBtn = $("persona-save-btn");
    const personaResetBtn = $("persona-reset-btn");

    // Keep thinking label in sync with checkbox
    if (personaThinking) {
        personaThinking.addEventListener("change", () => {
            personaThinkingLabel.textContent = personaThinking.checked ? "On" : "Off";
        });
    }

    async function loadPersona() {
        try {
            const data = await (await fetch("/api/v1/persona")).json();
            personaName.value = data.name || "";
            personaUserName.value = data.user_name || "";
            personaLanguage.value = data.language || "";
            // Sectioned persona
            const sec = data.sections || {};
            personaIdentity.value = sec.identity || "";
            personaStyle.value = sec.style || "";
            personaRules.value = sec.rules || "";
            personaThinking.checked = !!data.thinking_enabled;
            personaThinkingLabel.textContent = data.thinking_enabled ? "On" : "Off";
            // Store snapshot for diff-based saves
            _personaSnapshot = {
                name: data.name || "",
                user_name: data.user_name || "",
                language: data.language || "",
                identity: sec.identity || "",
                style: sec.style || "",
                rules: sec.rules || "",
                thinking_enabled: !!data.thinking_enabled,
            };
        } catch (err) {
            toast("Error", `persona: ${err.message}`);
        }
    }

    // Snapshot of the persona as loaded from the server. Used to detect
    // which fields the user actually changed so we only PATCH those.
    let _personaSnapshot = {};

    async function savePersona() {
        const current = {
            name: personaName.value,
            user_name: personaUserName.value,
            language: personaLanguage.value,
            identity: personaIdentity.value,
            style: personaStyle.value,
            rules: personaRules.value,
            thinking_enabled: personaThinking.checked,
        };

        // Build the PATCH payload — only changed fields
        const patch = {};
        const sectionsPatch = {};

        for (const [key, value] of Object.entries(current)) {
            if (["identity", "style", "rules"].includes(key)) {
                if (value !== _personaSnapshot[key]) {
                    sectionsPatch[key] = value;
                }
            } else {
                if (value !== _personaSnapshot[key]) {
                    patch[key] = value;
                }
            }
        }
        if (Object.keys(sectionsPatch).length > 0) {
            patch.sections = sectionsPatch;
        }
        if (Object.keys(patch).length === 0) {
            toast("No changes", "Nothing to save");
            return;
        }
        try {
            const resp = await fetch("/api/v1/persona", {
                method: "PATCH",
                headers: { "content-type": "application/json" },
                body: JSON.stringify(patch),
            });
            if (!resp.ok) {
                const errBody = await resp.text();
                throw new Error(`HTTP ${resp.status}: ${errBody}`);
            }
            // Update snapshot
            Object.assign(_personaSnapshot, current);
            toast("Personality saved", "Live — applies to the next turn", 6000);
        } catch (err) {
            toast("Error", err.message);
        }
    }

    async function resetPersona() {
        if (!confirm("Reset Lexy's personality to the default? Your current edits will be overwritten.")) return;
        try {
            const resp = await fetch("/api/v1/persona/reset", { method: "POST" });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            await loadPersona();
            toast("Personality reset", "Defaults restored");
        } catch (err) {
            toast("Error", err.message);
        }
    }

    personaSaveBtn.addEventListener("click", savePersona);
    personaResetBtn.addEventListener("click", resetPersona);

    async function loadSettings() {
        brainsGrid.innerHTML = `<div class="muted">loading…</div>`;
        await loadPersona();
        try {
            const data = await (await fetch("/api/v1/settings")).json();

            if (data.system) {
                if (data.system.profile) systemProfile.value = data.system.profile;
                profileActiveBrains.textContent = (data.system.active_brains || []).join(", ") || "—";
            }

            if (data.routing) {
                routingDefaultBrain.value = data.routing.default_brain || "a4b";
            }

            brainsGrid.innerHTML = "";
            for (const [name, brain] of Object.entries(data.brains || {})) {
                const node = brainRowTemplate.content.cloneNode(true);
                const card = node.querySelector(".brain-card");
                q(".brain-name", card).textContent = name;
                q(".brain-model", card).textContent = brain.model;
                q(".brain-endpoint", card).textContent = brain.endpoint;
                q(".brain-temp", card).value = brain.temperature;
                q(".brain-top-p", card).value = brain.top_p;
                q(".brain-max-tokens", card).value = brain.max_tokens;
                q(".brain-thinking", card).checked = !!brain.thinking;
                q(".brain-reasoning-budget", card).value = brain.reasoning_budget || 0;
                q(".brain-save-btn", card).addEventListener("click", async () => {
                    const patch = {
                        brains: {
                            [name]: {
                                temperature: parseFloat(q(".brain-temp", card).value),
                                top_p: parseFloat(q(".brain-top-p", card).value),
                                max_tokens: parseInt(q(".brain-max-tokens", card).value, 10),
                                thinking: q(".brain-thinking", card).checked,
                                reasoning_budget: parseInt(q(".brain-reasoning-budget", card).value, 10) || null,
                            },
                        },
                    };
                    try {
                        const resp = await fetch("/api/v1/settings", {
                            method: "PATCH",
                            headers: { "content-type": "application/json" },
                            body: JSON.stringify(patch),
                        });
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        toast(`Brain ${name} saved`, "live");
                    } catch (err) {
                        toast("Error", err.message);
                    }
                });
                brainsGrid.appendChild(node);
            }

            if (data.memory) {
                memoryInfo.innerHTML = `
                    <div class="kv"><span>ChromaDB</span><strong>${escapeHtml(data.memory.chroma_host)}:${data.memory.chroma_port}</strong></div>
                    <div class="kv"><span>Collections</span><strong>${(data.memory.collections || []).join(", ")}</strong></div>
                    <div class="kv"><span>Recall limit</span><strong>${data.memory.recall_limit}</strong></div>
                    <div class="kv"><span>Vector weight</span><strong>${data.memory.vector_weight}</strong></div>
                    <div class="kv"><span>BM25 weight</span><strong>${data.memory.bm25_weight}</strong></div>
                `;
            }
            if (data.embedding) {
                embeddingInfo.innerHTML = `
                    <div class="kv"><span>Model</span><strong>${escapeHtml(data.embedding.model)}</strong></div>
                    <div class="kv"><span>Device</span><strong>${escapeHtml(data.embedding.device)}</strong></div>
                    <div class="kv"><span>Dimension</span><strong>${data.embedding.dimension}</strong></div>
                `;
            }
        } catch (err) {
            brainsGrid.innerHTML = `<div class="muted">error: ${err.message}</div>`;
        }

        // Thinking + channel cards are wired below; call the extras hook
        // so they refresh every time the Settings tab is (re)opened.
        if (typeof window._lexySettingsExtras === "function") {
            try {
                await window._lexySettingsExtras();
            } catch (err) {
                console.warn("settings extras failed", err);
            }
        }
    }

    routingSaveBtn.addEventListener("click", async () => {
        try {
            const resp = await fetch("/api/v1/settings", {
                method: "PATCH",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ routing: { default_brain: routingDefaultBrain.value } }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            toast("Routing saved", `default_brain = ${routingDefaultBrain.value}`);
        } catch (err) {
            toast("Error", err.message);
        }
    });

    profileSaveBtn.addEventListener("click", async () => {
        try {
            const resp = await fetch("/api/v1/settings", {
                method: "PATCH",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ system: { profile: systemProfile.value } }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            toast(
                "Profile saved",
                `profile = ${systemProfile.value} · restart Lexy to apply`,
                9000
            );
        } catch (err) {
            toast("Error", err.message);
        }
    });

    settingsRefreshBtn.addEventListener("click", loadSettings);

    // ── Autonomous Thinking quick controls ─────────────────────────
    const thinkingStatus = $("thinking-status");
    const thinkingToggleBtn = $("thinking-toggle-btn");
    const thinkingTriggerBtn = $("thinking-trigger-btn");
    const thinkingTriggerMode = $("thinking-trigger-mode");
    let thinkingActive = false;

    function renderThinkingStatus() {
        thinkingStatus.textContent = thinkingActive ? "ACTIVE 🟢" : "idle 🌙";
    }
    renderThinkingStatus();

    async function loadThinkingStatus() {
        // There's no GET endpoint — read the plugin's effective config
        // which has the `enabled` flag the user last persisted.
        try {
            const data = await (
                await fetch("/api/v1/plugins/autonomous_thinking/config")
            ).json();
            if (data && data.effective) {
                thinkingActive = !!data.effective.enabled;
                renderThinkingStatus();
            }
        } catch (err) {
            thinkingStatus.textContent = `error: ${err.message}`;
        }
    }

    thinkingToggleBtn.addEventListener("click", () => {
        const next = !thinkingActive;
        if (!wsSend({ type: "thinking_toggle", active: next })) {
            toast("Error", "WebSocket not connected");
            return;
        }
        // Optimistic update — server will echo back thinking_toggled
        thinkingActive = next;
        renderThinkingStatus();
        toast(
            "Autonomous thinking",
            next ? "active (background loop running)" : "paused",
            4500
        );
    });

    thinkingTriggerBtn.addEventListener("click", () => {
        const mode = thinkingTriggerMode.value;
        if (!wsSend({ type: "thinking_trigger", mode })) {
            toast("Error", "WebSocket not connected");
            return;
        }
        toast("Thinking…", `mode = ${mode}`, 3000);
    });

    // ── Settings-Editor (Phase 8): Autonomous Thinking ─────────────
    const thinkingInterval = $("thinking-interval");
    const thinkingIntervalVal = $("thinking-interval-val");
    const thinkingMinIdle = $("thinking-min-idle");
    const thinkingMinIdleVal = $("thinking-min-idle-val");
    const thinkingMaxPerHour = $("thinking-max-per-hour");
    const thinkingQuietFrom = $("thinking-quiet-from");
    const thinkingQuietTo = $("thinking-quiet-to");
    const thinkingToolsEnabled = $("thinking-tools-enabled");
    const thinkingToolsMaxIter = $("thinking-tools-max-iter");
    const thinkingToolsWhitelist = $("thinking-tools-whitelist");
    const thinkingSaveBtn = $("thinking-save-btn");
    const thinkingResetBtn = $("thinking-reset-btn");
    const thinkingSaveHint = $("thinking-save-hint");
    // Track defaults so the Reset button can revert without a server round-trip
    let thinkingDefaults = null;

    function applyThinkingSettings(cfg, defaults) {
        // Live sliders update their readout
        if (thinkingInterval) {
            thinkingInterval.value = cfg.mode_interval_seconds || 600;
            thinkingIntervalVal.textContent = thinkingInterval.value + "s";
        }
        if (thinkingMinIdle) {
            thinkingMinIdle.value = cfg.min_idle_seconds || 120;
            thinkingMinIdleVal.textContent = thinkingMinIdle.value + "s";
        }
        if (thinkingMaxPerHour) thinkingMaxPerHour.value = cfg.max_thoughts_per_hour || 4;
        const quiet = Array.isArray(cfg.quiet_hours) && cfg.quiet_hours.length === 2
            ? cfg.quiet_hours
            : ["23:00", "07:00"];
        if (thinkingQuietFrom) thinkingQuietFrom.value = quiet[0];
        if (thinkingQuietTo) thinkingQuietTo.value = quiet[1];
        // Modes checkboxes
        const modes = Array.isArray(cfg.modes) ? cfg.modes : [];
        document.querySelectorAll("#thinking-modes input[type=checkbox]").forEach((cb) => {
            cb.checked = modes.includes(cb.value);
        });
        // Tool use
        if (thinkingToolsEnabled) thinkingToolsEnabled.checked = !!cfg.tools_enabled;
        if (thinkingToolsMaxIter) thinkingToolsMaxIter.value = cfg.tools_max_iterations || 3;
        if (thinkingToolsWhitelist) {
            thinkingToolsWhitelist.value = Array.isArray(cfg.tools_whitelist)
                ? cfg.tools_whitelist.join(", ")
                : "";
        }
        if (defaults) thinkingDefaults = defaults;
    }

    async function loadThinkingSettings() {
        try {
            const data = await (
                await fetch("/api/v1/plugins/autonomous_thinking/config")
            ).json();
            applyThinkingSettings(data.effective || {}, data.defaults || {});
        } catch (err) {
            if (thinkingSaveHint) thinkingSaveHint.textContent = `load error: ${err.message}`;
        }
    }

    function buildThinkingPatch() {
        const quietFrom = thinkingQuietFrom ? thinkingQuietFrom.value : "";
        const quietTo = thinkingQuietTo ? thinkingQuietTo.value : "";
        const modes = Array.from(
            document.querySelectorAll("#thinking-modes input[type=checkbox]:checked")
        ).map((cb) => cb.value);
        const whitelist = (thinkingToolsWhitelist ? thinkingToolsWhitelist.value : "")
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
        return {
            mode_interval_seconds: Number(thinkingInterval.value),
            min_idle_seconds: Number(thinkingMinIdle.value),
            max_thoughts_per_hour: Number(thinkingMaxPerHour.value),
            quiet_hours: quietFrom && quietTo ? [quietFrom, quietTo] : undefined,
            modes,
            tools_enabled: !!(thinkingToolsEnabled && thinkingToolsEnabled.checked),
            tools_max_iterations: Number(thinkingToolsMaxIter.value),
            tools_whitelist: whitelist,
        };
    }

    if (thinkingInterval) thinkingInterval.addEventListener("input", () => {
        thinkingIntervalVal.textContent = thinkingInterval.value + "s";
    });
    if (thinkingMinIdle) thinkingMinIdle.addEventListener("input", () => {
        thinkingMinIdleVal.textContent = thinkingMinIdle.value + "s";
    });

    if (thinkingSaveBtn) thinkingSaveBtn.addEventListener("click", async () => {
        const patch = buildThinkingPatch();
        // Drop undefined fields (FastAPI endpoint tolerates partial patches)
        for (const k of Object.keys(patch)) if (patch[k] === undefined) delete patch[k];
        thinkingSaveHint.textContent = "saving…";
        try {
            const resp = await fetch(
                "/api/v1/plugins/autonomous_thinking/config",
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(patch),
                }
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            thinkingSaveHint.textContent = data.applied_live
                ? "saved ✓ (live)"
                : "saved ✓ (restart needed)";
            setTimeout(() => { thinkingSaveHint.textContent = ""; }, 2500);
        } catch (err) {
            thinkingSaveHint.textContent = `error: ${err.message}`;
        }
    });

    if (thinkingResetBtn) thinkingResetBtn.addEventListener("click", () => {
        if (thinkingDefaults) applyThinkingSettings(thinkingDefaults, thinkingDefaults);
    });

    // ── Settings-Editor (Phase 8): Scheduler ───────────────────────
    const schedCheckInterval = $("scheduler-check-interval");
    const schedMaxActive = $("scheduler-max-active");
    const schedImpulsesEnabled = $("scheduler-impulses-enabled");
    const schedImpulseMin = $("scheduler-impulse-min");
    const schedImpulseMax = $("scheduler-impulse-max");
    const schedSaveBtn = $("scheduler-save-btn");
    const schedResetBtn = $("scheduler-reset-btn");
    const schedSaveHint = $("scheduler-save-hint");
    let schedulerDefaults = null;

    function applySchedulerSettings(cfg, defaults) {
        if (schedCheckInterval) schedCheckInterval.value = cfg.check_interval || 5.0;
        if (schedMaxActive) schedMaxActive.value = cfg.max_active_timers || 50;
        if (schedImpulsesEnabled) schedImpulsesEnabled.checked = cfg.enable_impulses !== false;
        if (schedImpulseMin) schedImpulseMin.value = cfg.impulse_min_hour != null ? cfg.impulse_min_hour : 8;
        if (schedImpulseMax) schedImpulseMax.value = cfg.impulse_max_hour != null ? cfg.impulse_max_hour : 22;
        if (defaults) schedulerDefaults = defaults;
    }

    async function loadSchedulerSettings() {
        try {
            const data = await (
                await fetch("/api/v1/plugins/scheduler/config")
            ).json();
            applySchedulerSettings(data.effective || {}, data.defaults || {});
        } catch (err) {
            if (schedSaveHint) schedSaveHint.textContent = `load error: ${err.message}`;
        }
    }

    if (schedSaveBtn) schedSaveBtn.addEventListener("click", async () => {
        const patch = {
            check_interval: Number(schedCheckInterval.value),
            max_active_timers: Number(schedMaxActive.value),
            enable_impulses: !!(schedImpulsesEnabled && schedImpulsesEnabled.checked),
            impulse_min_hour: Number(schedImpulseMin.value),
            impulse_max_hour: Number(schedImpulseMax.value),
        };
        schedSaveHint.textContent = "saving…";
        try {
            const resp = await fetch("/api/v1/plugins/scheduler/config", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(patch),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            schedSaveHint.textContent = data.applied_live
                ? "saved ✓ (live)"
                : "saved ✓ (restart needed)";
            setTimeout(() => { schedSaveHint.textContent = ""; }, 2500);
        } catch (err) {
            schedSaveHint.textContent = `error: ${err.message}`;
        }
    });

    if (schedResetBtn) schedResetBtn.addEventListener("click", () => {
        if (schedulerDefaults) applySchedulerSettings(schedulerDefaults, schedulerDefaults);
    });

    // ── Channel config ─────────────────────────────────────────────
    const channelsGrid = $("channels-grid");

    async function loadChannelsConfig() {
        const channels = ["whatsapp", "discord", "telegram"];
        for (const ch of channels) {
            const pluginName = `channel_${ch}`;
            try {
                const resp = await fetch(
                    `/api/v1/plugins/${pluginName}/config`
                );
                if (!resp.ok) {
                    $(`channel-${ch}-status`).textContent = "not loaded";
                    continue;
                }
                const data = await resp.json();
                const eff = data.effective || {};
                $(`channel-${ch}-status`).textContent = "ready (restart needed after save)";

                if (ch === "whatsapp") {
                    $("channel-whatsapp-bridge").value = eff.bridge_url || "";
                    $("channel-whatsapp-key").value = eff.api_key || "";
                    const contacts = Array.isArray(eff.allowed_contacts)
                        ? eff.allowed_contacts.join(", ")
                        : "";
                    $("channel-whatsapp-contacts").value = contacts;
                } else if (ch === "discord") {
                    $("channel-discord-token-env").value = eff.token_env || "LEXY_DISCORD_TOKEN";
                    $("channel-discord-prefix").value = eff.command_prefix || "!lexy";
                } else if (ch === "telegram") {
                    $("channel-telegram-token-env").value = eff.token_env || "LEXY_TELEGRAM_TOKEN";
                    const allowed = Array.isArray(eff.allowed_users)
                        ? eff.allowed_users.join(",")
                        : "";
                    $("channel-telegram-users").value = allowed;
                }
            } catch (err) {
                $(`channel-${ch}-status`).textContent = `error: ${err.message}`;
            }
        }
    }

    async function saveChannelConfig(channel) {
        const pluginName = `channel_${channel}`;
        let patch = {};
        if (channel === "whatsapp") {
            const contactsRaw = $("channel-whatsapp-contacts").value.trim();
            const contacts = contactsRaw
                ? contactsRaw.split(/[\s,]+/).filter(Boolean)
                : [];
            patch = {
                bridge_url: $("channel-whatsapp-bridge").value.trim(),
                api_key: $("channel-whatsapp-key").value.trim(),
                allowed_contacts: contacts,
            };
        } else if (channel === "discord") {
            patch = {
                token_env: $("channel-discord-token-env").value.trim(),
                command_prefix: $("channel-discord-prefix").value.trim() || "!lexy",
            };
        } else if (channel === "telegram") {
            const usersRaw = $("channel-telegram-users").value.trim();
            const users = usersRaw
                ? usersRaw.split(/[\s,]+/).map((x) => parseInt(x, 10)).filter((x) => !isNaN(x))
                : [];
            patch = {
                token_env: $("channel-telegram-token-env").value.trim(),
                allowed_users: users,
            };
        }
        try {
            const resp = await fetch(
                `/api/v1/plugins/${pluginName}/config`,
                {
                    method: "PATCH",
                    headers: { "content-type": "application/json" },
                    body: JSON.stringify(patch),
                }
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            toast(
                `${channel} saved`,
                "Restart Lexy to reload the channel plugin",
                9000
            );
        } catch (err) {
            toast("Error", err.message);
        }
    }

    channelsGrid.addEventListener("click", (ev) => {
        const btn = ev.target.closest("[data-channel-save]");
        if (!btn) return;
        saveChannelConfig(btn.dataset.channelSave);
    });

    // Exposed so handleMessage (switch) can forward thinking_toggled
    // and thinking_result events back into our UI state.
    window._lexyThinking = {
        onToggled(active) {
            thinkingActive = !!active;
            renderThinkingStatus();
        },
        onResult(mode, text) {
            toast(`💭 ${mode || "thought"}`, text || "(empty)", 9000);
        },
    };
    window._lexySettingsExtras = async function () {
        await loadThinkingStatus();
        await loadThinkingSettings();
        await loadSchedulerSettings();
        await loadChannelsConfig();
    };

    // ── HoloMat: Avatar Toggle + Sound Toggle ──────────────────────
    const avatarToggleBtn = $("avatar-toggle-btn");
    const soundToggleBtn = $("sound-toggle-btn");
    const soundStatus = $("sound-status");
    let avatarVisible = false;
    let soundEnabled = localStorage.getItem("lexy-sound") !== "off";

    if (avatarToggleBtn) {
        avatarToggleBtn.addEventListener("click", () => {
            avatarVisible = !avatarVisible;
            if (window.LexyHolo) {
                if (avatarVisible) {
                    window.LexyHolo.avatar.show();
                } else {
                    window.LexyHolo.avatar.hide();
                }
            }
            if (window.LexyHolo) window.LexyHolo.sound.play("click");
        });
    }

    if (soundToggleBtn) {
        soundStatus.textContent = soundEnabled ? "on" : "off";
        if (window.LexyHolo) window.LexyHolo.sound.setEnabled(soundEnabled);
        soundToggleBtn.addEventListener("click", () => {
            soundEnabled = !soundEnabled;
            localStorage.setItem("lexy-sound", soundEnabled ? "on" : "off");
            soundStatus.textContent = soundEnabled ? "on" : "off";
            if (window.LexyHolo) window.LexyHolo.sound.setEnabled(soundEnabled);
        });
    }

    // Wire speaking state into outline + avatar when TTS audio plays
    const _origAudioPlay = state.audio && state.audio.play;
    if (state.audio && state.audio._context) {
        // AudioPlayer already uses Web Audio API — we hook into the analyser
        const checkAudioLevel = () => {
            if (!state.audio._analyser) { requestAnimationFrame(checkAudioLevel); return; }
            const data = new Uint8Array(state.audio._analyser.frequencyBinCount);
            state.audio._analyser.getByteFrequencyData(data);
            const avg = data.reduce((a, b) => a + b, 0) / data.length / 255;
            if (window.LexyHolo) {
                window.LexyHolo.avatar.setAudioLevel(avg);
                if (avg > 0.02) {
                    window.LexyHolo.outline.setState("speaking");
                    window.LexyHolo.rings.setLevel(avg);
                }
            }
            if (state.audio._playing) requestAnimationFrame(checkAudioLevel);
            else {
                if (window.LexyHolo) {
                    window.LexyHolo.outline.setState(null);
                    window.LexyHolo.avatar.setAudioLevel(0);
                    window.LexyHolo.rings.setActive(false);
                }
            }
        };
        // Monkey-patch audio.play to trigger the level loop
        const origPlay = state.audio.play.bind(state.audio);
        state.audio.play = function (arrayBuffer) {
            if (window.LexyHolo) {
                window.LexyHolo.rings.setActive(true);
                window.LexyHolo.outline.setState("speaking");
                window.LexyHolo.avatar.setMood("speaking");
            }
            const result = origPlay(arrayBuffer);
            requestAnimationFrame(checkAudioLevel);
            return result;
        };
    }

    // ── Dashboard ───────────────────────────────────────────────────
    const dashboardGrid = $("dashboard-grid");
    const dashboardEditBtn = $("dashboard-edit-btn");
    const dashboardRefreshBtn = $("dashboard-refresh-btn");

    function loadDashboard() {
        wsSend({ type: "get_dashboard_layout" });
        wsSend({ type: "get_dashboard_widgets" });
    }

    function renderDashboardGrid(layout, widgetData) {
        if (!dashboardGrid) return;
        // Clear any running clock interval
        if (state.dashboardClockInterval) {
            clearInterval(state.dashboardClockInterval);
            state.dashboardClockInterval = null;
        }
        dashboardGrid.innerHTML = "";

        // Default layout if none provided
        const items = (layout && layout.length > 0) ? layout : [
            { id: "clock",         col: "1 / 2",   row: "1 / 2"   },
            { id: "weather",       col: "2 / 3",   row: "1 / 2"   },
            { id: "system_status", col: "3 / 5",   row: "1 / 2"   },
            { id: "memory_stats",  col: "1 / 3",   row: "2 / 3"   },
            { id: "sessions",      col: "3 / 4",   row: "2 / 3"   },
            { id: "thoughts",      col: "4 / 5",   row: "2 / 3"   },
            { id: "notes",         col: "1 / 3",   row: "3 / 4"   },
            { id: "search",        col: "3 / 5",   row: "3 / 4"   },
        ];

        for (const item of items) {
            const card = document.createElement("div");
            card.className = "dashboard-widget";
            card.dataset.widgetId = item.id;
            if (item.col) card.style.gridColumn = item.col;
            if (item.row) card.style.gridRow = item.row;

            const header = document.createElement("div");
            header.className = "widget-header";
            const title = document.createElement("span");
            title.className = "widget-title";
            title.textContent = item.id.replace(/_/g, " ");
            const refreshBtn2 = document.createElement("button");
            refreshBtn2.className = "widget-refresh-btn";
            refreshBtn2.textContent = "↻";
            refreshBtn2.title = "Refresh widget";
            refreshBtn2.addEventListener("click", () => {
                wsSend({ type: "get_dashboard_widget", widget_id: item.id });
            });
            header.appendChild(title);
            header.appendChild(refreshBtn2);

            const body = document.createElement("div");
            body.className = "widget-body";

            card.appendChild(header);
            card.appendChild(body);
            dashboardGrid.appendChild(card);

            const data = (widgetData && widgetData[item.id]) || {};
            renderWidget(item.id, data, body);
        }
    }

    function renderWidget(widgetId, data, container) {
        switch (widgetId) {
            case "clock":         renderClockWidget(data, container); break;
            case "weather":       renderWeatherWidget(data, container); break;
            case "memory_stats":  renderMemoryStatsWidget(data, container); break;
            case "system_status": renderSystemStatusWidget(data, container); break;
            case "sessions":      renderSessionsWidget(data, container); break;
            case "thoughts":      renderThoughtsWidget(data, container); break;
            case "notes":         renderNotesWidget(data, container); break;
            case "search":        renderSearchWidget(data, container); break;
            default:
                container.innerHTML = `<div class="muted">Unknown widget: ${escapeHtml(widgetId)}</div>`;
        }
    }

    function renderClockWidget(_data, container) {
        container.innerHTML = `
            <div class="widget-clock-time" id="dashboard-clock-time"></div>
            <div class="widget-clock-date" id="dashboard-clock-date"></div>`;

        function updateClock() {
            const now = new Date();
            const timeEl = $("dashboard-clock-time");
            const dateEl = $("dashboard-clock-date");
            if (timeEl) {
                timeEl.textContent = now.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
            }
            if (dateEl) {
                dateEl.textContent = now.toLocaleDateString("de-DE", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
            }
        }

        updateClock();
        if (state.dashboardClockInterval) clearInterval(state.dashboardClockInterval);
        state.dashboardClockInterval = setInterval(updateClock, 1000);
    }

    // Backend returns icon as a key like "clear" / "rain" — map to emoji here.
    const _WEATHER_ICON_MAP = {
        clear: "☀️",
        partly_cloudy: "⛅️",
        overcast: "☁️",
        fog: "🌫️",
        drizzle: "🌦️",
        rain: "🌧️",
        snow: "❄️",
        thunderstorm: "⛈️",
        unknown: "🌤️",
    };

    function renderWeatherWidget(data, container) {
        if (data && data.available === false) {
            const err = data.error ? ` (${data.error})` : "";
            container.innerHTML = `<div class="muted">Wetter nicht verfügbar${escapeHtml(err)}</div>`;
            return;
        }
        const temp = data.temperature != null ? `${data.temperature}°C` : "—";
        const cond = data.condition || "Keine Daten";
        const iconKey = typeof data.icon === "string" ? data.icon : "unknown";
        const icon = _WEATHER_ICON_MAP[iconKey] || _WEATHER_ICON_MAP.unknown;
        const humidity = data.humidity != null ? `${data.humidity}%` : "—";
        // Backend field is `wind_speed`; keep `wind` as fallback for old payloads.
        const windVal = data.wind_speed != null ? data.wind_speed : data.wind;
        const wind = windVal != null ? `${windVal} km/h` : "—";
        const loc = data.location ? `<div class="muted" style="text-align:center;font-size:11px">${escapeHtml(data.location)}</div>` : "";
        container.innerHTML = `
            <div style="text-align:center;font-size:32px;margin-bottom:4px">${icon}</div>
            <div class="widget-weather-temp">${escapeHtml(temp)}</div>
            <div class="widget-weather-condition">${escapeHtml(cond)}</div>
            ${loc}
            <div class="widget-weather-details">
                <span>💧 ${escapeHtml(humidity)}</span>
                <span>💨 ${escapeHtml(wind)}</span>
            </div>`;
    }

    function renderMemoryStatsWidget(data, container) {
        if (data && data.available === false) {
            container.innerHTML = `<div class="muted">Memory nicht verfügbar</div>`;
            return;
        }
        const collections = data.collections || {};
        const entries = Object.entries(collections);
        if (entries.length === 0) {
            container.innerHTML = `<div class="muted">Keine Collections</div>`;
            return;
        }
        const maxCount = Math.max(1, ...Object.values(collections));
        let html = "";
        for (const [name, count] of entries) {
            const pct = Math.round((Number(count) / maxCount) * 100);
            html += `
                <div class="widget-memory-bar">
                    <span class="widget-memory-label">${escapeHtml(name)}</span>
                    <div class="widget-memory-fill">
                        <div class="widget-memory-fill-inner" style="width:${pct}%"></div>
                    </div>
                    <span class="widget-memory-count">${count}</span>
                </div>`;
        }
        const total = data.total != null ? data.total : Object.values(collections).reduce((a, b) => a + Number(b), 0);
        const fts = data.fts_count != null ? data.fts_count : null;
        const footer = `<div class="muted" style="margin-top:6px;font-size:11px">Σ ${total}${fts != null ? ` · FTS ${fts}` : ""}</div>`;
        container.innerHTML = html + footer;
    }

    function renderSystemStatusWidget(data, container) {
        const services = data.services || [];
        const plugins = data.plugins_loaded != null ? data.plugins_loaded : null;
        const uptime = data.uptime_seconds != null ? data.uptime_seconds : null;

        if (services.length === 0) {
            container.innerHTML = `<div class="muted">Keine Services</div>`;
            return;
        }
        let html = "";
        for (const svc of services) {
            // Backend emits status as a string: "up" | "down" | "timeout" | "error".
            // Fall back to `svc.up` (older shape) so new/old payloads both render.
            const status = typeof svc.status === "string" ? svc.status : (svc.up ? "up" : "down");
            const dotClass = status === "up" ? "up" : "down";
            const statusLabel = status !== "up" ? `<span class="muted" style="font-size:10px;margin-left:4px">(${escapeHtml(status)})</span>` : "";
            html += `
                <div class="widget-service-row">
                    <span class="widget-service-dot ${dotClass}"></span>
                    <span class="widget-service-name">${escapeHtml(svc.name || "—")}</span>
                    <span class="widget-service-host">${escapeHtml(svc.host || "")}</span>
                    ${statusLabel}
                </div>`;
        }
        if (plugins != null || uptime != null) {
            const parts = [];
            if (plugins != null) parts.push(`${plugins} Plugins`);
            if (uptime != null) parts.push(`Uptime ${_formatUptime(uptime)}`);
            html += `<div class="muted" style="margin-top:6px;font-size:11px">${parts.join(" · ")}</div>`;
        }
        container.innerHTML = html;
    }

    function _formatUptime(seconds) {
        const s = Math.floor(Number(seconds) || 0);
        if (s < 60) return `${s}s`;
        const m = Math.floor(s / 60);
        if (m < 60) return `${m}m`;
        const h = Math.floor(m / 60);
        return `${h}h ${m % 60}m`;
    }

    function renderSessionsWidget(data, container) {
        // Backend shape: {active_count, sessions: [{id, messages, last_snippet}], total_messages}
        // Keep support for legacy {count, active} shape as fallback.
        const activeCount = data.active_count != null ? data.active_count
            : (data.count != null ? data.count : 0);
        const totalMessages = data.total_messages != null ? data.total_messages
            : (data.active != null ? data.active : 0);
        const sessions = Array.isArray(data.sessions) ? data.sessions : [];

        let topHtml = "";
        for (const s of sessions.slice(0, 3)) {
            const snippet = (s.last_snippet || "").slice(0, 40);
            topHtml += `
                <div class="widget-session-row">
                    <span class="widget-session-id">${escapeHtml(String(s.id || "").slice(0, 8))}</span>
                    <span class="widget-session-msgcount">${escapeHtml(String(s.messages || 0))}</span>
                    <span class="widget-session-snippet">${escapeHtml(snippet)}</span>
                </div>`;
        }
        const listBlock = topHtml ? `<div class="widget-session-list">${topHtml}</div>` : "";

        container.innerHTML = `
            <div class="widget-session-count">${escapeHtml(String(activeCount))}</div>
            <div class="widget-session-label">Sessions · ${escapeHtml(String(totalMessages))} Nachrichten</div>
            ${listBlock}`;
    }

    function renderThoughtsWidget(data, container) {
        const thoughts = Array.isArray(data.thoughts) ? data.thoughts : [];
        if (thoughts.length === 0) {
            const enabled = data.enabled !== false;
            const msg = enabled
                ? "Keine Gedanken bisher"
                : "Autonomes Denken deaktiviert";
            container.innerHTML = `<div class="muted">${msg}</div>`;
            return;
        }
        let html = "";
        for (const t of thoughts) {
            // Backend emits `at` (HH:MM). Keep `time` as fallback for old payloads.
            const when = t.at || t.time || "";
            html += `
                <div class="widget-thought">
                    <div class="widget-thought-mode">${escapeHtml(t.mode || "thought")}</div>
                    <div class="widget-thought-text">${escapeHtml(t.text || "")}</div>
                    <div class="widget-thought-time">${escapeHtml(when)}</div>
                </div>`;
        }
        container.innerHTML = html;
    }

    function renderNotesWidget(data, container) {
        const notes = data.notes || [];
        let listHtml = "";
        for (let i = 0; i < notes.length; i++) {
            listHtml += `
                <div class="widget-note-item">
                    <span class="widget-note-text">${escapeHtml(notes[i].text || "")}</span>
                    <button class="widget-note-delete" data-note-idx="${i}" title="Delete">✕</button>
                </div>`;
        }
        container.innerHTML = `
            <div style="display:flex;gap:6px;margin-bottom:8px">
                <textarea class="widget-notes-input" rows="2" placeholder="Notiz schreiben…"></textarea>
                <button class="btn" style="align-self:flex-end;padding:6px 10px;font-size:11px" data-action="save-note">+</button>
            </div>
            <div class="widget-notes-list">${listHtml}</div>`;

        const saveBtn = container.querySelector("[data-action='save-note']");
        const textarea = container.querySelector(".widget-notes-input");
        if (saveBtn && textarea) {
            saveBtn.addEventListener("click", () => {
                const text = textarea.value.trim();
                if (!text) return;
                wsSend({ type: "dashboard_note_add", text });
                textarea.value = "";
            });
        }
        container.querySelectorAll(".widget-note-delete").forEach((btn) => {
            btn.addEventListener("click", () => {
                const idx = parseInt(btn.dataset.noteIdx, 10);
                wsSend({ type: "dashboard_note_delete", index: idx });
            });
        });
    }

    function renderSearchWidget(_data, container) {
        container.innerHTML = `
            <input type="text" class="widget-search-input" placeholder="Memory durchsuchen…" />
            <div class="widget-search-results"></div>`;

        const input2 = container.querySelector(".widget-search-input");
        const resultsDiv = container.querySelector(".widget-search-results");
        let debounce = null;
        input2.addEventListener("input", () => {
            clearTimeout(debounce);
            debounce = setTimeout(() => {
                const query = input2.value.trim();
                if (query.length < 2) { resultsDiv.innerHTML = ""; return; }
                wsSend({ type: "dashboard_search", query });
            }, 350);
        });

        // Listen for search results via a one-time handler on state
        state._dashboardSearchResults = resultsDiv;
    }

    function enableDashboardEdit() {
        state.dashboardEditing = !state.dashboardEditing;
        const widgets = qa(".dashboard-widget");
        widgets.forEach((w) => w.classList.toggle("editing", state.dashboardEditing));
        if (dashboardEditBtn) {
            dashboardEditBtn.textContent = state.dashboardEditing ? "💾 Save Layout" : "✏️ Edit Layout";
        }
        if (!state.dashboardEditing) {
            saveDashboardLayout();
        }
    }

    function saveDashboardLayout() {
        const widgets = qa(".dashboard-widget");
        const layout = [];
        for (const w of widgets) {
            layout.push({
                id: w.dataset.widgetId,
                col: w.style.gridColumn || "",
                row: w.style.gridRow || "",
            });
        }
        wsSend({ type: "save_dashboard_layout", layout });
    }

    if (dashboardEditBtn) {
        dashboardEditBtn.addEventListener("click", enableDashboardEdit);
    }
    if (dashboardRefreshBtn) {
        dashboardRefreshBtn.addEventListener("click", loadDashboard);
    }

    // ── Expert Panel rendering ──────────────────────────────────────
    const PANEL_COLORS = {
        analyst: "#3b82f6", critic: "#ef4444", creative: "#a855f7",
        pragmatist: "#22c55e", synthesizer: "#f59e0b",
    };
    const PANEL_LABELS = {
        analyst: "Analyst", critic: "Kritiker", creative: "Kreativer",
        pragmatist: "Pragmatiker", synthesizer: "Synthesizer",
    };
    state.panelMessages = [];

    function _renderPanelMessage(data) {
        state.panelMessages.push(data);
        const view = $("dashboard-panel-view");
        if (!view) return;
        view.hidden = false;
        const msg = document.createElement("div");
        msg.className = "panel-msg";
        const color = PANEL_COLORS[data.role] || "#888";
        const label = PANEL_LABELS[data.role] || data.role;
        const initial = label.charAt(0).toUpperCase();
        msg.innerHTML = `
            <div class="panel-avatar" style="background:${color}">${initial}</div>
            <div class="panel-msg-body">
                <div class="panel-msg-header">
                    <strong style="color:${color}">${escapeHtml(label)}</strong>
                    <span class="mono" style="font-size:10px;color:var(--text-dim)">
                        ${data.phase || ""} ${data.round ? "R" + data.round : ""}
                    </span>
                </div>
                <div class="panel-msg-text">${escapeHtml(data.content || "")}</div>
            </div>`;
        view.appendChild(msg);
        view.scrollTop = view.scrollHeight;
    }

    function _renderPanelResult(data) {
        const view = $("dashboard-panel-view");
        if (!view) return;
        const result = data.result || {};
        const box = document.createElement("div");
        box.className = "panel-result-box";
        let html = `<h4 style="color:var(--accent);margin:0 0 8px">Ergebnis</h4>`;
        if (result.summary) html += `<p>${escapeHtml(result.summary)}</p>`;
        if (result.consensus_points?.length) {
            html += `<h5>Konsens</h5><ul>${result.consensus_points.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`;
        }
        if (result.dissent_points?.length) {
            html += `<h5>Dissens</h5><ul>${result.dissent_points.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`;
        }
        if (result.action_items?.length) {
            html += `<h5>Action Items</h5><ul>${result.action_items.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`;
        }
        box.innerHTML = html;
        view.appendChild(box);
        view.scrollTop = view.scrollHeight;
    }

    // ── YouTube Player ───────────────────────────────────────────────
    function _openYouTubePlayer(data) {
        let overlay = $("youtube-overlay");
        if (!overlay) {
            overlay = document.createElement("div");
            overlay.id = "youtube-overlay";
            overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9000;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px";
            overlay.innerHTML = `
                <div style="display:flex;justify-content:space-between;width:100%;max-width:800px;margin-bottom:8px">
                    <span id="yt-title" style="color:#fff;font-size:14px"></span>
                    <button id="yt-close" class="btn" style="background:var(--err);color:#fff">Close</button>
                </div>
                <iframe id="yt-frame" width="800" height="450" frameborder="0"
                    allow="accelerometer;autoplay;clipboard-write;encrypted-media;gyroscope;picture-in-picture"
                    allowfullscreen style="border-radius:8px;max-width:100%"></iframe>`;
            document.body.appendChild(overlay);
            $("yt-close").addEventListener("click", () => overlay.style.display = "none");
            overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.style.display = "none"; });
        }
        const frame = $("yt-frame");
        const title = $("yt-title");
        if (frame) frame.src = data.embed_url || `https://www.youtube.com/embed/${data.video_id}?autoplay=1`;
        if (title) title.textContent = data.title || "";
        overlay.style.display = "flex";
    }

    // ── Periodic signal polling ────────────────────────────────────
    setInterval(() => wsSend({ type: "get_signals" }), 3000);
    window.addEventListener("resize", () => state.visualizer.resize());

    // ── Projects ────────────────────────────────────────────────────
    //
    // Projects let the user group sessions, isolate memory and append a
    // per-project persona override. The active project is persisted in
    // localStorage so it survives reloads. All mutations go through the
    // REST endpoints; the backend broadcasts ``project_*`` WS events so
    // other tabs / devices stay in sync (see handleMessage above).

    const projectBtn = $("project-dropdown-btn");
    const projectDropdown = $("project-dropdown");
    const projColorSwatch = $("proj-color");
    const projIconSpan = $("proj-icon");
    const projNameSpan = $("proj-name");
    const projectsModal = $("projects-modal");
    const projectsModalClose = $("projects-modal-close");
    const projectsModalList = $("projects-modal-list");
    const projectsModalForm = $("projects-modal-form");
    const projFormName = $("proj-form-name");
    const projFormDescription = $("proj-form-description");
    const projFormColor = $("proj-form-color");
    const projFormIcon = $("proj-form-icon");
    const projFormScoped = $("proj-form-scoped");
    const projFormOverride = $("proj-form-override");
    const projFormSave = $("proj-form-save");
    const projFormReset = $("proj-form-reset");
    const projFormArchive = $("proj-form-archive");
    const projFormUnarchive = $("proj-form-unarchive");
    const projFormDelete = $("proj-form-delete");
    const projFormHint = $("proj-form-hint");
    const sessionCtxMenu = $("session-context-menu");
    const sessionCtxList = $("scm-list");

    async function apiProjects(includeArchived = false) {
        const url = "/api/v1/projects" + (includeArchived ? "?include_archived=true" : "");
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return (await resp.json()).projects || [];
    }

    async function apiCreateProject(payload) {
        const resp = await fetch("/api/v1/projects", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        return (await resp.json()).project;
    }

    async function apiUpdateProject(id, patch) {
        const resp = await fetch(`/api/v1/projects/${encodeURIComponent(id)}`, {
            method: "PATCH",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(patch),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        return (await resp.json()).project;
    }

    async function apiDeleteProject(id) {
        const resp = await fetch(`/api/v1/projects/${encodeURIComponent(id)}`, {
            method: "DELETE",
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        return await resp.json();
    }

    async function apiArchiveProject(id) {
        const resp = await fetch(`/api/v1/projects/${encodeURIComponent(id)}/archive`, {
            method: "POST",
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    }

    async function apiUnarchiveProject(id) {
        const resp = await fetch(`/api/v1/projects/${encodeURIComponent(id)}/unarchive`, {
            method: "POST",
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    }

    async function apiMoveSession(sessionId, projectId) {
        const resp = await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, {
            method: "PATCH",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ project_id: projectId }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
        return await resp.json();
    }

    async function loadProjects() {
        try {
            const projects = await apiProjects(true);
            state.projects = projects;
            state.projectsById = {};
            for (const p of projects) state.projectsById[p.id] = p;
            if (!state.projectsById[state.activeProjectId]) {
                state.activeProjectId = "default";
                localStorage.setItem("lexy.activeProjectId", "default");
            }
            renderActiveProjectBar();
            renderProjectsDropdown();
            if (!projectsModal.hidden) renderProjectsModalList();
        } catch (err) {
            console.warn("loadProjects failed:", err);
        }
    }

    function renderActiveProjectBar() {
        const p = state.projectsById[state.activeProjectId]
            || state.projectsById["default"];
        if (!p) return;
        projColorSwatch.style.background = p.color;
        projNameSpan.textContent = p.name;
        projIconSpan.textContent = p.icon || (p.is_default ? "🏠" : "📁");
    }

    function renderProjectsDropdown() {
        projectDropdown.innerHTML = "";
        const visible = state.projects.filter((p) => !p.archived);
        for (const p of visible) {
            const item = document.createElement("div");
            item.className = "proj-item" + (p.id === state.activeProjectId ? " active" : "");
            item.innerHTML = `
                <span class="proj-color" style="background:${p.color}"></span>
                <span class="proj-icon">${escapeHtml(p.icon || (p.is_default ? "🏠" : "📁"))}</span>
                <span class="proj-name">${escapeHtml(p.name)}</span>
                ${p.id === state.activeProjectId ? '<span>✓</span>' : ''}
            `;
            item.addEventListener("click", () => {
                setActiveProject(p.id);
                closeProjectDropdown();
            });
            projectDropdown.appendChild(item);
        }
        const sep = document.createElement("div");
        sep.className = "proj-sep";
        projectDropdown.appendChild(sep);

        const newBtn = document.createElement("div");
        newBtn.className = "proj-action";
        newBtn.textContent = "➕ Neues Projekt";
        newBtn.addEventListener("click", () => {
            closeProjectDropdown();
            openProjectsModal({ create: true });
        });
        projectDropdown.appendChild(newBtn);

        const mgrBtn = document.createElement("div");
        mgrBtn.className = "proj-action";
        mgrBtn.textContent = "⚙ Projekte verwalten";
        mgrBtn.addEventListener("click", () => {
            closeProjectDropdown();
            openProjectsModal();
        });
        projectDropdown.appendChild(mgrBtn);
    }

    function setActiveProject(id) {
        if (state.activeProjectId === id) return;
        state.activeProjectId = id;
        localStorage.setItem("lexy.activeProjectId", id);
        renderActiveProjectBar();
        renderProjectsDropdown();
        if (state.activeTab === "sessions") loadSessions();
        const p = state.projectsById[id];
        if (p) toast("Projekt aktiv", `${p.icon || ""} ${p.name}`.trim(), 3000);
    }

    function openProjectDropdown() {
        projectDropdown.hidden = false;
        projectBtn.dataset.open = "true";
    }

    function closeProjectDropdown() {
        projectDropdown.hidden = true;
        projectBtn.dataset.open = "false";
    }

    projectBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (projectDropdown.hidden) openProjectDropdown();
        else closeProjectDropdown();
    });
    document.addEventListener("click", (ev) => {
        if (!projectDropdown.hidden && !projectBtn.contains(ev.target)
            && !projectDropdown.contains(ev.target)) {
            closeProjectDropdown();
        }
    });

    // ── Projects manage modal ──────────────────────────────────────
    function openProjectsModal({ create = false, id = null } = {}) {
        projectsModal.hidden = false;
        renderProjectsModalList();
        if (create) {
            resetProjectForm();
        } else if (id) {
            selectProjectInModal(id);
        } else {
            // Preselect the currently active project
            selectProjectInModal(state.activeProjectId);
        }
    }

    function closeProjectsModal() {
        projectsModal.hidden = true;
    }

    function renderProjectsModalList() {
        projectsModalList.innerHTML = "";
        const sorted = [...state.projects].sort((a, b) => {
            if (a.is_default !== b.is_default) return a.is_default ? -1 : 1;
            if (a.archived !== b.archived) return a.archived ? 1 : -1;
            return a.name.localeCompare(b.name);
        });
        for (const p of sorted) {
            const entry = document.createElement("div");
            entry.className = "pm-entry";
            if (p.archived) entry.classList.add("archived");
            if (state.projectsModalSelectedId === p.id) entry.classList.add("selected");
            entry.innerHTML = `
                <span class="proj-color" style="background:${p.color}"></span>
                <span>${escapeHtml(p.icon || (p.is_default ? "🏠" : "📁"))}</span>
                <span class="proj-name">${escapeHtml(p.name)}</span>
                ${p.archived ? '<span class="muted" style="font-size:10px">archiviert</span>' : ''}
                ${p.is_default ? '<span class="muted" style="font-size:10px">default</span>' : ''}
            `;
            entry.addEventListener("click", () => selectProjectInModal(p.id));
            projectsModalList.appendChild(entry);
        }
    }

    function resetProjectForm() {
        state.projectsModalSelectedId = null;
        projFormName.value = "";
        projFormDescription.value = "";
        projFormColor.value = "#7aa2f7";
        projFormIcon.value = "";
        projFormScoped.checked = true;
        projFormOverride.value = "";
        projFormSave.textContent = "Anlegen";
        projFormArchive.hidden = true;
        projFormUnarchive.hidden = true;
        projFormDelete.hidden = true;
        projFormHint.textContent = "Neues Projekt anlegen.";
        renderProjectsModalList();
    }

    function selectProjectInModal(id) {
        const p = state.projectsById[id];
        if (!p) {
            resetProjectForm();
            return;
        }
        state.projectsModalSelectedId = id;
        projFormName.value = p.name;
        projFormDescription.value = p.description || "";
        projFormColor.value = p.color || "#7aa2f7";
        projFormIcon.value = p.icon || "";
        projFormScoped.checked = !!p.memory_scoped;
        projFormOverride.value = p.persona_override || "";
        projFormSave.textContent = "Speichern";
        projFormArchive.hidden = p.is_default || p.archived;
        projFormUnarchive.hidden = !p.archived;
        projFormDelete.hidden = p.is_default;
        if (p.is_default) {
            projFormHint.textContent = "Standardprojekt — kann nicht gelöscht werden, Memory ist projekt-übergreifend.";
        } else if (p.archived) {
            projFormHint.textContent = "Archiviert — nicht in der Sidebar sichtbar.";
        } else {
            projFormHint.textContent = "";
        }
        renderProjectsModalList();
    }

    async function saveProjectFromForm(ev) {
        ev?.preventDefault();
        const name = projFormName.value.trim();
        if (!name) {
            toast("Fehler", "Name darf nicht leer sein.");
            return;
        }
        const payload = {
            name,
            description: projFormDescription.value,
            color: projFormColor.value,
            icon: projFormIcon.value,
            memory_scoped: projFormScoped.checked,
            persona_override: projFormOverride.value,
        };
        try {
            if (state.projectsModalSelectedId) {
                const updated = await apiUpdateProject(state.projectsModalSelectedId, payload);
                toast("Gespeichert", updated.name, 3000);
                await loadProjects();
                selectProjectInModal(updated.id);
            } else {
                const created = await apiCreateProject(payload);
                toast("Angelegt", created.name, 3000);
                await loadProjects();
                selectProjectInModal(created.id);
            }
        } catch (err) {
            toast("Fehler", err.message);
        }
    }

    async function handleArchiveClick() {
        const id = state.projectsModalSelectedId;
        if (!id) return;
        try {
            await apiArchiveProject(id);
            toast("Archiviert", state.projectsById[id]?.name || id, 3000);
            await loadProjects();
            selectProjectInModal(id);
        } catch (err) {
            toast("Fehler", err.message);
        }
    }

    async function handleUnarchiveClick() {
        const id = state.projectsModalSelectedId;
        if (!id) return;
        try {
            await apiUnarchiveProject(id);
            toast("Wiederhergestellt", state.projectsById[id]?.name || id, 3000);
            await loadProjects();
            selectProjectInModal(id);
        } catch (err) {
            toast("Fehler", err.message);
        }
    }

    async function handleDeleteClick() {
        const id = state.projectsModalSelectedId;
        if (!id) return;
        const p = state.projectsById[id];
        if (!p) return;
        if (!confirm(
            `Projekt "${p.name}" löschen?\n\nAlle Sessions werden nach "Allgemein" verschoben.`
        )) return;
        try {
            const resp = await apiDeleteProject(id);
            toast("Gelöscht", `${p.name} · ${resp.migrated_sessions || 0} Sessions migriert`, 6000);
            if (state.activeProjectId === id) {
                state.activeProjectId = "default";
                localStorage.setItem("lexy.activeProjectId", "default");
            }
            await loadProjects();
            resetProjectForm();
        } catch (err) {
            toast("Fehler", err.message);
        }
    }

    projectsModalForm.addEventListener("submit", saveProjectFromForm);
    projFormReset.addEventListener("click", resetProjectForm);
    projFormArchive.addEventListener("click", handleArchiveClick);
    projFormUnarchive.addEventListener("click", handleUnarchiveClick);
    projFormDelete.addEventListener("click", handleDeleteClick);
    projectsModalClose.addEventListener("click", closeProjectsModal);
    projectsModal.addEventListener("click", (ev) => {
        if (ev.target === projectsModal) closeProjectsModal();
    });

    // ── Session context menu (right-click → move to project) ───────
    function openSessionContextMenu(x, y, sessionId) {
        sessionCtxList.innerHTML = "";
        const visible = state.projects.filter((p) => !p.archived);
        for (const p of visible) {
            const item = document.createElement("div");
            item.className = "scm-item";
            item.innerHTML = `
                <span class="proj-color" style="background:${p.color}"></span>
                <span>${escapeHtml(p.icon || (p.is_default ? "🏠" : "📁"))}</span>
                <span>${escapeHtml(p.name)}</span>
            `;
            item.addEventListener("click", async () => {
                closeSessionContextMenu();
                try {
                    await apiMoveSession(sessionId, p.id);
                    toast("Session verschoben", p.name, 3000);
                    if (state.activeTab === "sessions") loadSessions();
                } catch (err) {
                    toast("Fehler", err.message);
                }
            });
            sessionCtxList.appendChild(item);
        }
        sessionCtxMenu.hidden = false;
        // Position the menu so it stays inside the viewport.
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        sessionCtxMenu.style.left = "0px";
        sessionCtxMenu.style.top = "0px";
        const rect = sessionCtxMenu.getBoundingClientRect();
        const left = Math.min(x, vw - rect.width - 8);
        const top = Math.min(y, vh - rect.height - 8);
        sessionCtxMenu.style.left = `${left}px`;
        sessionCtxMenu.style.top = `${top}px`;
    }

    function closeSessionContextMenu() {
        sessionCtxMenu.hidden = true;
    }

    document.addEventListener("click", (ev) => {
        if (!sessionCtxMenu.hidden && !sessionCtxMenu.contains(ev.target)) {
            closeSessionContextMenu();
        }
    });
    window.addEventListener("resize", closeSessionContextMenu);
    document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") {
            closeSessionContextMenu();
            if (!projectDropdown.hidden) closeProjectDropdown();
        }
    });

    // ── SchedulerManager ───────────────────────────────────────────
    //
    // Tab UI + creator pane for the Phase-6 Active Scheduler. Runs
    // entirely over the existing WS (message types scheduler_list,
    // scheduler_create, scheduler_cancel, scheduler_update). The
    // scheduler plugin pushes scheduler_triggered / scheduler_created /
    // proactive_message / agent_task_spawned broadcasts that we relay
    // into the tab and toast system.

    const SCHED_MODE_FIELDS = {
        timer: (v) => `
            <div class="setting-row">
                <label>Minuten</label>
                <input type="number" id="sched-minutes" min="0" value="${v.minutes ?? 5}" />
            </div>
            <div class="setting-row">
                <label>Sekunden</label>
                <input type="number" id="sched-seconds" min="0" value="${v.seconds ?? 0}" />
            </div>`,
        reminder: (v) => `
            <div class="setting-row">
                <label>Uhrzeit (HH:MM)</label>
                <input type="text" id="sched-time" placeholder="09:00" value="${v.time ?? ""}" />
            </div>
            <div class="setting-row">
                <label><input type="checkbox" id="sched-tomorrow" ${v.tomorrow ? "checked" : ""}/> Morgen</label>
            </div>`,
        recurring: (v) => `
            <div class="setting-row">
                <label>Pattern</label>
                <input type="text" id="sched-pattern" placeholder="daily 09:00" value="${v.pattern ?? ""}" />
            </div>
            <div class="setting-row">
                <label>Aktion</label>
                <select id="sched-action-type">
                    <option value="notify" ${v.action_type === "notify" ? "selected" : ""}>Notify (Toast)</option>
                    <option value="proactive_chat" ${v.action_type === "proactive_chat" ? "selected" : ""}>Proactive Chat</option>
                    <option value="agent_task" ${v.action_type === "agent_task" ? "selected" : ""}>Agent Task</option>
                </select>
            </div>
            <div id="sched-action-payload"></div>`,
        proactive_chat: (v) => `
            <div class="setting-row">
                <label>Zeit / Pattern</label>
                <input type="text" id="sched-time-or-pattern" placeholder="09:00 oder daily 09:00" value="${v.time_or_pattern ?? ""}" />
            </div>
            <div class="setting-row">
                <label>Nachricht</label>
                <textarea id="sched-message" rows="3" placeholder="Was soll Lexy sich überlegen?">${v.message ?? ""}</textarea>
            </div>
            <div class="setting-row">
                <label>Session-ID (leer = aktuelle)</label>
                <input type="text" id="sched-session-id" placeholder="leer für aktuelle" value="${v.session_id ?? ""}" />
            </div>
            <div class="setting-row">
                <label><input type="checkbox" id="sched-tomorrow" ${v.tomorrow ? "checked" : ""}/> Morgen (nur für HH:MM)</label>
            </div>`,
        agent_task: (v) => `
            <div class="setting-row">
                <label>Zeit / Pattern</label>
                <input type="text" id="sched-time-or-pattern" placeholder="08:00 oder daily 08:00" value="${v.time_or_pattern ?? ""}" />
            </div>
            <div class="setting-row">
                <label>Persona</label>
                <input type="text" id="sched-persona" placeholder="researcher" value="${v.persona ?? "default"}" />
            </div>
            <div class="setting-row">
                <label>Task</label>
                <textarea id="sched-task" rows="3" placeholder="Was soll der Agent tun?">${v.task ?? ""}</textarea>
            </div>
            <div class="setting-row">
                <label>Report-to-Session (leer = aktuelle)</label>
                <input type="text" id="sched-report-session" placeholder="leer für aktuelle" value="${v.report_to_session ?? ""}" />
            </div>
            <div class="setting-row">
                <label><input type="checkbox" id="sched-tomorrow" ${v.tomorrow ? "checked" : ""}/> Morgen (nur für HH:MM)</label>
            </div>`,
    };

    const schedModeSelect = document.getElementById("sched-mode");
    const schedLabelInput = document.getElementById("sched-label");
    const schedFields = document.getElementById("sched-fields");
    const schedCreateBtn = document.getElementById("sched-create-btn");
    const schedFeedback = document.getElementById("sched-feedback");
    const schedListEl = document.getElementById("sched-list");
    const schedRefreshBtn = document.getElementById("sched-refresh-btn");

    function renderSchedulerForm() {
        const mode = schedModeSelect.value;
        const builder = SCHED_MODE_FIELDS[mode] || SCHED_MODE_FIELDS.timer;
        schedFields.innerHTML = builder({});
    }

    function schedulerSetFeedback(text, kind = "") {
        schedFeedback.textContent = text || "";
        schedFeedback.className = "sched-feedback" + (kind ? " " + kind : "");
    }

    function buildSchedulerPayload() {
        const mode = schedModeSelect.value;
        const label = (schedLabelInput.value || "").trim();
        const payload = { type: "scheduler_create", mode, label };
        if (mode === "timer") {
            payload.minutes = Number(document.getElementById("sched-minutes").value || 0);
            payload.seconds = Number(document.getElementById("sched-seconds").value || 0);
        } else if (mode === "reminder") {
            payload.time = (document.getElementById("sched-time").value || "").trim();
            payload.tomorrow = document.getElementById("sched-tomorrow").checked;
        } else if (mode === "recurring") {
            payload.pattern = (document.getElementById("sched-pattern").value || "").trim();
            payload.action_type = document.getElementById("sched-action-type").value;
        } else if (mode === "proactive_chat") {
            payload.time_or_pattern = (document.getElementById("sched-time-or-pattern").value || "").trim();
            payload.message = (document.getElementById("sched-message").value || "").trim();
            const sid = (document.getElementById("sched-session-id").value || "").trim();
            payload.session_id = sid || state.sessionId || "";
            payload.tomorrow = document.getElementById("sched-tomorrow").checked;
        } else if (mode === "agent_task") {
            payload.time_or_pattern = (document.getElementById("sched-time-or-pattern").value || "").trim();
            payload.persona = (document.getElementById("sched-persona").value || "default").trim();
            payload.task = (document.getElementById("sched-task").value || "").trim();
            const sid = (document.getElementById("sched-report-session").value || "").trim();
            payload.report_to_session = sid || state.sessionId || "";
            payload.tomorrow = document.getElementById("sched-tomorrow").checked;
        }
        return payload;
    }

    function validateSchedulerPayload(p) {
        if (!p.label) return "Label fehlt";
        if (p.mode === "timer") {
            if ((p.minutes || 0) + (p.seconds || 0) <= 0)
                return "Minuten oder Sekunden > 0";
        } else if (p.mode === "reminder") {
            if (!p.time || !/^\d{1,2}:\d{2}$/.test(p.time))
                return "Zeit im Format HH:MM";
        } else if (p.mode === "recurring") {
            if (!p.pattern) return "Pattern fehlt";
        } else if (p.mode === "proactive_chat") {
            if (!p.time_or_pattern) return "Zeit/Pattern fehlt";
            if (!p.message) return "Nachricht fehlt";
        } else if (p.mode === "agent_task") {
            if (!p.time_or_pattern) return "Zeit/Pattern fehlt";
            if (!p.task) return "Task fehlt";
        }
        return null;
    }

    async function createScheduler() {
        const payload = buildSchedulerPayload();
        const err = validateSchedulerPayload(payload);
        if (err) {
            schedulerSetFeedback(err, "error");
            return;
        }
        schedulerSetFeedback("Wird erstellt…", "");
        wsSend(payload);
    }

    function renderSchedulerList() {
        const tab = state.schedulerTab;
        const all = state.schedulerTimers || [];
        let filtered;
        if (tab === "active") {
            filtered = all.filter((t) => !t.kind === false || (t.active !== false && !t.fired && !t.pattern));
            // Active = one-shot timers/reminders/proactive_chat/agent_task
            filtered = all.filter((t) =>
                t.active !== false && !t.fired && !(t.pattern || "")
            );
        } else if (tab === "recurring") {
            filtered = all.filter((t) => !!(t.pattern || ""));
        } else {
            filtered = all.filter((t) => t.fired || t.active === false);
        }

        if (!filtered.length) {
            schedListEl.innerHTML =
                `<div class="muted">Keine Einträge in "${tab}".</div>`;
            return;
        }
        const rows = filtered.map((t) => {
            const kindBadge = `<span class="sched-kind">${escapeHtml(t.kind || "?")}</span>`;
            const pattern = t.pattern
                ? `<code>${escapeHtml(t.pattern)}</code> · `
                : "";
            const fireInfo = t.fires_at
                ? `fires at ${escapeHtml(t.fires_at)}`
                : "";
            const status = t.active === false
                ? ' · <em>pausiert</em>'
                : t.fired
                    ? ' · <em>fertig</em>'
                    : "";
            const pauseBtn = (t.active === false)
                ? `<button class="btn" data-sched-act="resume" data-id="${escapeHtml(t.id)}">Fortsetzen</button>`
                : `<button class="btn" data-sched-act="pause" data-id="${escapeHtml(t.id)}">Pause</button>`;
            return `
                <div class="sched-entry">
                    <div>
                        <div class="sched-entry-header">${escapeHtml(t.label || "(ohne Label)")}</div>
                        <div class="sched-entry-meta">${kindBadge}${pattern}${fireInfo}${status}</div>
                    </div>
                    <div class="sched-entry-actions">
                        ${pauseBtn}
                        <button class="btn" data-sched-act="cancel" data-id="${escapeHtml(t.id)}">Abbrechen</button>
                    </div>
                </div>`;
        });
        schedListEl.innerHTML = rows.join("");

        qa("[data-sched-act]", schedListEl).forEach((btn) => {
            btn.addEventListener("click", () => {
                const id = btn.dataset.id;
                const act = btn.dataset.schedAct;
                if (act === "cancel") {
                    wsSend({ type: "scheduler_cancel", id });
                } else if (act === "pause") {
                    wsSend({ type: "scheduler_update", id, active: false });
                } else if (act === "resume") {
                    wsSend({ type: "scheduler_update", id, active: true });
                }
            });
        });
    }

    function loadScheduler() {
        // Refresh both active and inactive so the tabs always have data.
        wsSend({ type: "scheduler_list", include_inactive: true });
    }

    qa(".sched-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            state.schedulerTab = btn.dataset.schedTab;
            qa(".sched-tab").forEach((b) =>
                b.classList.toggle("active", b.dataset.schedTab === state.schedulerTab)
            );
            renderSchedulerList();
        });
    });

    if (schedModeSelect) {
        schedModeSelect.addEventListener("change", renderSchedulerForm);
        renderSchedulerForm();
    }
    if (schedCreateBtn) {
        schedCreateBtn.addEventListener("click", createScheduler);
    }
    if (schedRefreshBtn) {
        schedRefreshBtn.addEventListener("click", loadScheduler);
    }

    // Expose scheduler helpers for the WS handler below.
    window._lexyScheduler = {
        onList: (data) => {
            state.schedulerTimers = data.timers || [];
            if (state.activeTab === "scheduler") renderSchedulerList();
        },
        onCreated: (data) => {
            if (data.error) {
                schedulerSetFeedback(`Fehler: ${data.error}`, "error");
                return;
            }
            schedulerSetFeedback("Erstellt ✓", "ok");
            loadScheduler();
        },
        onCancelled: () => {
            loadScheduler();
        },
        onUpdated: () => {
            loadScheduler();
        },
        onTriggered: (data) => {
            // Existing toast handler in the main WS switch still fires;
            // we just refresh the list so the state reflects fired/rescheduled.
            if (state.activeTab === "scheduler") loadScheduler();
        },
        onProactive: (data) => {
            // Render proactive messages as a proper Lexy chat bubble —
            // the old systemNote was invisible and easy to miss. This is
            // especially important for auto-reactions to baby pulses
            // where Lexy needs to visibly respond.
            if (data && data.text && data.session_id === state.sessionId) {
                appendMessage("assistant", data.text);
            }
            if (data && data.text) {
                toast("💬 Lexy", data.text, 5000);
            }
        },
    };

    // ═══ Character Chat (Silly-Tavern-lite) ═══════════════════════════

    const charactersState = {
        list: [],
        selectedId: null,
        sessionState: { character_mode: false, scene: "" },
        roundsInFlight: new Set(),
    };

    const charsListEl = $("characters-list");
    const charsForm = $("characters-form");
    const charFormId = $("char-form-id");
    const charFormName = $("char-form-name");
    const charFormPersona = $("char-form-persona");
    const charFormScenario = $("char-form-scenario");
    const charFormGreeting = $("char-form-greeting");
    const charFormExample = $("char-form-example");
    const charFormColor = $("char-form-color");
    const charFormAge = $("char-form-age");
    const charFormVoice = $("char-form-voice");
    const charFormTags = $("char-form-tags");
    const charFormPulsePattern = $("char-form-pulse-pattern");
    const charFormPulsePrompt = $("char-form-pulse-prompt");
    const charFormAvatarImg = $("char-form-avatar-img");
    const charFormAvatarBtn = $("char-form-avatar-btn");
    const charFormAvatarFile = $("char-form-avatar-file");
    const charFormSave = $("char-form-save");
    const charFormReset = $("char-form-reset");
    const charFormAttach = $("char-form-attach");
    const charFormDetach = $("char-form-detach");
    const charFormArchive = $("char-form-archive");
    const charFormUnarchive = $("char-form-unarchive");
    const charFormDelete = $("char-form-delete");
    const charFormHint = $("char-form-hint");

    const charsNewBtn = $("chars-new-btn");
    const charsImportBtn = $("chars-import-btn");
    const charsImportFile = $("chars-import-file");
    const charsShowArchived = $("chars-show-archived");
    const charsRefreshBtn = $("chars-refresh-btn");

    // Phase 9.12: Charakter-Bar + 🎭-Rollen-Button leben jetzt im
    // Rollenspiel-Tab. Die Variablen heißen aus Legacy-Gründen weiter
    // ``chatCharacters*`` — Rename wäre eine 200+-Stellen-Diff und der
    // Name ist nur noch für interne Wiring sichtbar (im UI sind die
    // Buttons mit ``rp-`` gelabelt).
    const chatCharactersBtn = $("rp-characters-btn");
    const chatCharacterBar = $("rp-character-bar");
    const charSessionModal = $("character-session-modal");
    const charSessionClose = $("char-session-close");
    const charSessionDone = $("char-session-done");
    const charSessionModeSelect = $("char-session-mode-select");
    const charSessionModeHint = $("char-session-mode-hint");
    const _MODE_HINTS = {
        "0": "Lexy antwortet normal. Charaktere reagieren nur via Pulse-Timer.",
        "1": "Nur Charaktere antworten auf deine Nachrichten. Lexy ist still.",
        "2": "Lexy antwortet zuerst, dann reagieren die Charaktere — natürlicher Familien-RP-Flow.",
    };
    const charSessionScene = $("char-session-scene");
    const charSessionAttachedList = $("char-session-attached-list");
    const charSessionAvailableList = $("char-session-available-list");
    const charSessionSimToggle = $("char-session-sim-toggle");
    const charSessionSimInterval = $("char-session-sim-interval");
    const charSessionSimStatus = $("char-session-sim-status");
    let simRunning = false;  // tracked per open-modal session

    function charsHint(msg, kind) {
        if (!charFormHint) return;
        charFormHint.textContent = msg || "";
        charFormHint.className = "muted characters-form-hint" + (kind ? " " + kind : "");
        if (msg) setTimeout(() => { if (charFormHint.textContent === msg) charsHint(""); }, 3500);
    }

    async function loadCharacters() {
        if (!charsListEl) return;
        try {
            const includeArchived = charsShowArchived && charsShowArchived.checked;
            const qs = includeArchived ? "?include_archived=true" : "";
            const resp = await fetch(`/api/v1/plugins/character_chat/characters${qs}`, {
                method: "GET",
            }).catch(() => null);
            // Fallback: use WS. The REST endpoint above isn't implemented — we
            // talk over WS because that's where all character_chat handlers live.
        } catch (err) { /* ignore */ }
        // Always go via WS (authoritative):
        wsSend({
            type: "character_list",
            include_archived: !!(charsShowArchived && charsShowArchived.checked),
        });
    }

    function renderCharactersList() {
        if (!charsListEl) return;
        const list = charactersState.list;
        if (!list.length) {
            charsListEl.innerHTML = '<div class="muted">Noch keine Charaktere — klick "+ Neu" oder importiere eine Silly-Tavern Card.</div>';
            return;
        }
        charsListEl.innerHTML = "";
        for (const c of list) {
            const row = document.createElement("div");
            row.className = "char-card" + (c.archived ? " archived" : "") + (c.id === charactersState.selectedId ? " active" : "");
            row.style.setProperty("--char-color", c.color || "#7aa2f7");
            row.style.borderLeftColor = c.color || "#7aa2f7";
            row.addEventListener("click", () => selectCharacter(c.id));

            const avatar = document.createElement("div");
            avatar.className = "char-avatar";
            if (c.avatar) {
                const img = document.createElement("img");
                img.src = c.avatar;
                img.alt = c.name;
                avatar.appendChild(img);
            } else {
                const ph = document.createElement("span");
                ph.className = "placeholder";
                ph.textContent = (c.name || "?").slice(0, 1).toUpperCase();
                avatar.appendChild(ph);
            }
            row.appendChild(avatar);

            const meta = document.createElement("div");
            meta.className = "char-meta";
            const name = document.createElement("div");
            name.className = "char-name";
            name.textContent = c.name;
            if (c.age_stage && c.age_stage !== "adult") {
                const badge = document.createElement("span");
                badge.className = "char-badge";
                badge.textContent = c.age_stage;
                name.appendChild(badge);
            }
            if (c.archived) {
                const badge = document.createElement("span");
                badge.className = "char-badge";
                badge.textContent = "archived";
                name.appendChild(badge);
            }
            meta.appendChild(name);
            const blurb = document.createElement("div");
            blurb.className = "char-blurb";
            blurb.textContent = (c.persona || "").slice(0, 90);
            meta.appendChild(blurb);
            row.appendChild(meta);

            charsListEl.appendChild(row);
        }
    }

    function selectCharacter(id) {
        charactersState.selectedId = id;
        const c = charactersState.list.find((x) => x.id === id);
        if (!c) return;
        if (charsForm) charsForm.hidden = false;
        charFormId.value = c.id;
        charFormName.value = c.name || "";
        charFormPersona.value = c.persona || "";
        charFormScenario.value = c.scenario || "";
        charFormGreeting.value = c.greeting || "";
        charFormExample.value = c.example_dialog || "";
        charFormColor.value = c.color || "#7aa2f7";
        charFormAge.value = c.age_stage || "adult";
        if (charFormVoice) charFormVoice.value = c.voice || "";
        charFormTags.value = (c.tags || []).join(", ");
        charFormPulsePattern.value = c.proactive_pulse_pattern || "";
        charFormPulsePrompt.value = c.proactive_pulse_prompt || "";
        if (c.avatar) {
            charFormAvatarImg.src = c.avatar;
            charFormAvatarImg.style.display = "";
        } else {
            charFormAvatarImg.removeAttribute("src");
            charFormAvatarImg.style.display = "none";
        }
        // Toggle buttons based on state
        const isAttachedHere = !!(state.sessionId && (c.active_sessions || []).includes(state.sessionId));
        charFormAttach.hidden = isAttachedHere || c.archived;
        charFormDetach.hidden = !isAttachedHere;
        charFormArchive.hidden = c.archived;
        charFormUnarchive.hidden = !c.archived;
        charFormDelete.hidden = false;
        renderCharactersList();
    }

    function resetCharacterForm() {
        charactersState.selectedId = null;
        if (charsForm) charsForm.hidden = false;
        charFormId.value = "";
        charFormName.value = "";
        charFormPersona.value = "";
        charFormScenario.value = "";
        charFormGreeting.value = "";
        charFormExample.value = "";
        charFormColor.value = "#7aa2f7";
        charFormAge.value = "adult";
        if (charFormVoice) charFormVoice.value = "";
        charFormTags.value = "";
        charFormPulsePattern.value = "";
        charFormPulsePrompt.value = "";
        charFormAvatarImg.removeAttribute("src");
        charFormAvatarImg.style.display = "none";
        charFormAttach.hidden = true;
        charFormDetach.hidden = true;
        charFormArchive.hidden = true;
        charFormUnarchive.hidden = true;
        charFormDelete.hidden = true;
        renderCharactersList();
    }

    async function submitCharacterForm(e) {
        e.preventDefault();
        const id = charFormId.value.trim();
        const payload = {
            name: charFormName.value.trim(),
            persona: charFormPersona.value,
            scenario: charFormScenario.value,
            greeting: charFormGreeting.value,
            example_dialog: charFormExample.value,
            color: charFormColor.value,
            age_stage: charFormAge.value,
            voice: charFormVoice ? charFormVoice.value.trim() : "",
            tags: charFormTags.value.split(",").map((t) => t.trim()).filter(Boolean),
            proactive_pulse_pattern: charFormPulsePattern.value.trim(),
            proactive_pulse_prompt: charFormPulsePrompt.value.trim(),
        };
        if (!payload.name) { charsHint("Name fehlt", "error"); return; }
        if (id) {
            wsSend({ type: "character_update", id, ...payload });
            charsHint("Gespeichert ✓", "ok");
        } else {
            wsSend({ type: "character_create", ...payload });
            charsHint("Erstellt ✓", "ok");
        }
    }

    if (charsForm) charsForm.addEventListener("submit", submitCharacterForm);
    if (charsNewBtn) charsNewBtn.addEventListener("click", resetCharacterForm);
    if (charFormReset) charFormReset.addEventListener("click", resetCharacterForm);
    if (charsRefreshBtn) charsRefreshBtn.addEventListener("click", loadCharacters);
    if (charsShowArchived) charsShowArchived.addEventListener("change", loadCharacters);

    if (charFormAttach) charFormAttach.addEventListener("click", () => {
        const id = charFormId.value;
        if (!id || !state.sessionId) return;
        wsSend({ type: "character_attach", id, session_id: state.sessionId });
        charsHint("Angehängt ✓", "ok");
    });
    if (charFormDetach) charFormDetach.addEventListener("click", () => {
        const id = charFormId.value;
        if (!id || !state.sessionId) return;
        wsSend({ type: "character_detach", id, session_id: state.sessionId });
        charsHint("Entfernt ✓", "ok");
    });
    if (charFormArchive) charFormArchive.addEventListener("click", () => {
        const id = charFormId.value;
        if (!id) return;
        wsSend({ type: "character_archive", id });
        charsHint("Archiviert ✓", "ok");
    });
    if (charFormUnarchive) charFormUnarchive.addEventListener("click", () => {
        const id = charFormId.value;
        if (!id) return;
        wsSend({ type: "character_unarchive", id });
        charsHint("Wiederhergestellt ✓", "ok");
    });
    if (charFormDelete) charFormDelete.addEventListener("click", () => {
        const id = charFormId.value;
        if (!id) return;
        if (!confirm("Diesen Charakter wirklich dauerhaft löschen? Erinnerungen werden NICHT gelöscht (aber bleiben isoliert unter diesem character_id).")) return;
        wsSend({ type: "character_delete", id });
        resetCharacterForm();
    });

    // Avatar upload
    if (charFormAvatarBtn) {
        charFormAvatarBtn.addEventListener("click", () => {
            if (!charFormId.value) {
                charsHint("Speichere den Charakter zuerst, dann kannst du den Avatar hochladen.", "error");
                return;
            }
            charFormAvatarFile.click();
        });
    }
    if (charFormAvatarFile) {
        charFormAvatarFile.addEventListener("change", async () => {
            const file = charFormAvatarFile.files && charFormAvatarFile.files[0];
            const id = charFormId.value;
            if (!file || !id) return;
            const fd = new FormData();
            fd.append("character_id", id);
            fd.append("file", file);
            try {
                const resp = await fetch("/api/v1/plugins/character_chat/avatars", {
                    method: "POST",
                    body: fd,
                });
                const data = await resp.json();
                if (!resp.ok || !data.ok) throw new Error(data.detail || data.error || "upload failed");
                charFormAvatarImg.src = data.avatar + "?t=" + Date.now();
                charFormAvatarImg.style.display = "";
                charsHint("Avatar geladen ✓", "ok");
            } catch (err) {
                charsHint("Avatar-Upload fehlgeschlagen: " + err.message, "error");
            } finally {
                charFormAvatarFile.value = "";
            }
        });
    }

    // Silly-Tavern import — supports JSON cards AND PNG cards
    // (cards with the chara tEXt chunk). The REST endpoint
    // ``/api/v1/plugins/character_chat/import`` auto-detects the format
    // by magic bytes / extension / MIME — so the frontend just needs to
    // hand the file over via multipart/form-data.
    if (charsImportBtn) charsImportBtn.addEventListener("click", () => {
        if (charsImportFile) charsImportFile.click();
    });
    if (charsImportFile) {
        charsImportFile.addEventListener("change", async () => {
            const file = charsImportFile.files && charsImportFile.files[0];
            if (!file) return;
            charsImportBtn.disabled = true;
            charsImportBtn.textContent = "…";
            try {
                const form = new FormData();
                form.append("file", file);
                const resp = await fetch(
                    "/api/v1/plugins/character_chat/import",
                    { method: "POST", body: form },
                );
                if (!resp.ok) {
                    let detail = await resp.text();
                    try { detail = JSON.parse(detail).detail || detail; }
                    catch (_e) {}
                    throw new Error(`HTTP ${resp.status}: ${detail}`);
                }
                const data = await resp.json();
                const name = (data.character && data.character.name) || "(?)";
                charsHint(`Card "${name}" importiert.`, "ok");
                // Server broadcasts character_created → list refreshes
                // automatically, no manual reload needed.
            } catch (err) {
                charsHint("Import fehlgeschlagen: " + (err.message || err), "error");
            } finally {
                charsImportBtn.disabled = false;
                charsImportBtn.textContent = "📥 Import";
                charsImportFile.value = "";
            }
        });
    }

    // ─── Lorebooks tab (Phase 9.11) ─────────────────────────────────
    //
    // Two-pane editor over the character_chat plugin's lorebook REST
    // endpoints. Local state lives on ``state.lore``: list of books,
    // active book id, list of entries for that book, plus the live
    // scope filter. We refresh the list on demand (tab switch, manual
    // Refresh button, after every CRUD mutation). The plugin also
    // broadcasts ``lorebook_*`` / ``lore_entry_*`` events over WS, but
    // for the editor itself the ack from the REST call is enough — we
    // refresh locally rather than waiting for the broadcast.

    state.lore = {
        books: [],
        activeBookId: null,
        entries: [],
        scope: "all",      // matches the scope filter dropdown
        scopeId: "",       // character_id / session_id when scope != all/global
    };

    const loreBooksPane = $("lore-books-pane");
    const loreEntriesPane = $("lore-entries-pane");
    const loreScopeFilter = $("lore-scope-filter");
    const loreScopeIdFilter = $("lore-scope-id-filter");
    const loreNewBookBtn = $("lore-new-book-btn");
    const loreRefreshBtn = $("lore-refresh-btn");

    // Book modal refs
    const loreBookModal = $("lore-book-modal");
    const loreBookForm = $("lore-book-form");
    const loreBookId = $("lore-book-id");
    const loreBookName = $("lore-book-name");
    const loreBookDescription = $("lore-book-description");
    const loreBookScope = $("lore-book-scope");
    const loreBookScopeIdRow = $("lore-book-scope-id-row");
    const loreBookScopeIdLabel = $("lore-book-scope-id-label");
    const loreBookScopeId = $("lore-book-scope-id");
    const loreBookTokenBudget = $("lore-book-token-budget");
    const loreBookEnabled = $("lore-book-enabled");
    const loreBookDelete = $("lore-book-delete");
    const loreBookCancel = $("lore-book-cancel");
    const loreBookModalClose = $("lore-book-modal-close");
    const loreBookModalTitle = $("lore-book-modal-title");

    // Entry modal refs
    const loreEntryModal = $("lore-entry-modal");
    const loreEntryForm = $("lore-entry-form");
    const loreEntryId = $("lore-entry-id");
    const loreEntryBookId = $("lore-entry-book-id");
    const loreEntryName = $("lore-entry-name");
    const loreEntryKeys = $("lore-entry-keys");
    const loreEntryAlwaysOn = $("lore-entry-always-on");
    const loreEntryContent = $("lore-entry-content");
    const loreEntryPosition = $("lore-entry-position");
    const loreEntryPriority = $("lore-entry-priority");
    const loreEntryScanDepth = $("lore-entry-scan-depth");
    const loreEntryEnabled = $("lore-entry-enabled");
    const loreEntryDelete = $("lore-entry-delete");
    const loreEntryCancel = $("lore-entry-cancel");
    const loreEntryModalClose = $("lore-entry-modal-close");
    const loreEntryModalTitle = $("lore-entry-modal-title");

    // ── Top-level loader ──────────────────────────────────────────
    async function loadLorebooks() {
        if (!loreBooksPane) return;
        loreBooksPane.innerHTML = '<div class="muted">loading…</div>';
        // Build query based on scope filter.
        const params = new URLSearchParams();
        if (state.lore.scope && state.lore.scope !== "all") {
            params.set("scope", state.lore.scope);
            if (state.lore.scopeId) params.set("scope_id", state.lore.scopeId);
        }
        const qs = params.toString() ? `?${params}` : "";
        try {
            const resp = await fetch(`/api/v1/plugins/character_chat/lorebooks${qs}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const body = await resp.json();
            state.lore.books = body.books || [];
            renderLoreBooksList();
            // If the active book disappeared (filtered out, deleted), clear it.
            if (state.lore.activeBookId &&
                !state.lore.books.find(b => b.id === state.lore.activeBookId)) {
                state.lore.activeBookId = null;
                state.lore.entries = [];
                renderLoreEntries();
            } else if (state.lore.activeBookId) {
                // Refresh entries for the still-active book.
                await loadLoreEntries(state.lore.activeBookId);
            }
        } catch (err) {
            loreBooksPane.innerHTML = `<div class="muted">Fehler: ${err.message}</div>`;
        }
    }

    function renderLoreBooksList() {
        if (!loreBooksPane) return;
        loreBooksPane.innerHTML = "";
        if (!state.lore.books.length) {
            loreBooksPane.innerHTML =
                '<div class="muted">Keine Lorebooks. Klick „+ Neues Buch" um eines anzulegen.</div>';
            return;
        }
        for (const book of state.lore.books) {
            const card = document.createElement("div");
            card.className = `lore-book-card scope-${book.scope}`;
            if (book.id === state.lore.activeBookId) card.classList.add("active");
            if (!book.enabled) card.classList.add("disabled");
            const scopeLabel = scopeLabelFor(book);
            card.innerHTML = `
                <div class="lore-book-name">
                    ${escapeHtml(book.name)}
                    ${book.enabled ? "" : '<span class="muted">(off)</span>'}
                </div>
                <div class="lore-book-meta">
                    <span class="lore-scope-badge scope-${book.scope}">${book.scope}</span>
                    ${scopeLabel ? `<span>${escapeHtml(scopeLabel)}</span>` : ""}
                    <span>budget: ${book.token_budget}</span>
                </div>
                ${book.description ? `<div class="lore-book-desc">${escapeHtml(book.description)}</div>` : ""}
            `;
            card.addEventListener("click", (e) => {
                // Right-click / shift-click → edit book; plain click → load entries.
                if (e.shiftKey || e.button === 2) {
                    openBookModalForEdit(book);
                } else {
                    state.lore.activeBookId = book.id;
                    renderLoreBooksList();
                    loadLoreEntries(book.id);
                }
            });
            card.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                openBookModalForEdit(book);
            });
            loreBooksPane.appendChild(card);
        }
    }

    function scopeLabelFor(book) {
        if (book.scope === "global") return "alle Sessions";
        if (book.scope === "character") {
            const c = (state.characters || []).find(x => x.id === book.scope_id);
            return c ? `→ ${c.name}` : `→ ${book.scope_id.slice(0, 8)}…`;
        }
        if (book.scope === "session") {
            return `→ ${book.scope_id.slice(0, 8)}…`;
        }
        return "";
    }

    // ── Entry list (right pane) ───────────────────────────────────
    async function loadLoreEntries(bookId) {
        if (!loreEntriesPane) return;
        loreEntriesPane.innerHTML = '<div class="muted">loading entries…</div>';
        try {
            const resp = await fetch(
                `/api/v1/plugins/character_chat/lorebooks/${encodeURIComponent(bookId)}/entries`
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const body = await resp.json();
            state.lore.entries = body.entries || [];
            renderLoreEntries();
        } catch (err) {
            loreEntriesPane.innerHTML =
                `<div class="muted">Fehler beim Laden: ${err.message}</div>`;
        }
    }

    function renderLoreEntries() {
        if (!loreEntriesPane) return;
        const book = state.lore.books.find(b => b.id === state.lore.activeBookId);
        if (!book) {
            loreEntriesPane.innerHTML =
                '<div class="lore-empty muted">Wähle links ein Lorebook um seine Einträge zu sehen.</div>';
            return;
        }
        const head = document.createElement("div");
        head.className = "lore-entries-header";
        head.innerHTML = `
            <div>
                <div class="lore-current-book-name">${escapeHtml(book.name)}</div>
                <div class="lore-current-book-meta">
                    <span class="lore-scope-badge scope-${book.scope}">${book.scope}</span>
                    ${book.description ? escapeHtml(book.description) : "<span class=\"muted\">keine Beschreibung</span>"}
                </div>
            </div>
            <div style="display:flex;gap:6px;">
                <button class="btn" id="lore-edit-book-btn">📝 Buch bearbeiten</button>
                <button class="btn primary" id="lore-new-entry-btn">+ Neuer Eintrag</button>
            </div>
        `;
        loreEntriesPane.innerHTML = "";
        loreEntriesPane.appendChild(head);

        $("lore-edit-book-btn").addEventListener("click", () => openBookModalForEdit(book));
        $("lore-new-entry-btn").addEventListener("click", () => openEntryModalForNew(book.id));

        if (!state.lore.entries.length) {
            const empty = document.createElement("div");
            empty.className = "muted";
            empty.style.padding = "30px 10px";
            empty.style.textAlign = "center";
            empty.textContent = "Noch keine Einträge — leg den ersten an.";
            loreEntriesPane.appendChild(empty);
            return;
        }

        const table = document.createElement("table");
        table.className = "lore-entries-table";
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Trigger</th>
                    <th>Position</th>
                    <th>Prio</th>
                    <th>Inhalt</th>
                </tr>
            </thead>
            <tbody></tbody>
        `;
        const tbody = table.querySelector("tbody");
        for (const entry of state.lore.entries) {
            const row = document.createElement("tr");
            row.className = "entry-row";
            if (!entry.enabled) row.classList.add("disabled");
            const keysHtml = entry.always_on
                ? '<span class="lore-key-chip always-on">⏰ always-on</span>'
                : (entry.keys || []).map(k =>
                    `<span class="lore-key-chip">${escapeHtml(k)}</span>`
                  ).join(" ") || '<span class="muted">— (kein Trigger)</span>';
            const preview = (entry.content || "")
                .replace(/\s+/g, " ")
                .slice(0, 80);
            row.innerHTML = `
                <td><strong>${escapeHtml(entry.name)}</strong>${entry.enabled ? "" : ' <span class="muted">(off)</span>'}</td>
                <td>${keysHtml}</td>
                <td><span class="lore-position-pill">${entry.position}</span></td>
                <td>${entry.priority}</td>
                <td class="muted">${escapeHtml(preview)}${entry.content && entry.content.length > 80 ? "…" : ""}</td>
            `;
            row.addEventListener("click", () => openEntryModalForEdit(entry));
            tbody.appendChild(row);
        }
        loreEntriesPane.appendChild(table);
    }

    // ── Book modal ────────────────────────────────────────────────
    function openBookModalForNew() {
        loreBookForm.reset();
        loreBookId.value = "";
        loreBookEnabled.checked = true;
        loreBookTokenBudget.value = 1500;
        loreBookScope.value = "global";
        loreBookDelete.hidden = true;
        loreBookModalTitle.textContent = "Neues Lorebook";
        updateBookScopeIdRow();
        loreBookModal.hidden = false;
        loreBookName.focus();
    }

    function openBookModalForEdit(book) {
        loreBookForm.reset();
        loreBookId.value = book.id;
        loreBookName.value = book.name;
        loreBookDescription.value = book.description || "";
        loreBookScope.value = book.scope;
        loreBookTokenBudget.value = book.token_budget;
        loreBookEnabled.checked = !!book.enabled;
        loreBookDelete.hidden = false;
        loreBookModalTitle.textContent = `Lorebook · ${book.name}`;
        updateBookScopeIdRow(book.scope_id);
        loreBookModal.hidden = false;
        loreBookName.focus();
    }

    function updateBookScopeIdRow(currentValue = "") {
        const scope = loreBookScope.value;
        if (scope === "global") {
            loreBookScopeIdRow.hidden = true;
            return;
        }
        loreBookScopeIdRow.hidden = false;
        loreBookScopeIdLabel.textContent = scope === "character" ? "Charakter" : "Session";
        loreBookScopeId.innerHTML = "";
        if (scope === "character") {
            const chars = state.characters || [];
            if (!chars.length) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "(keine Charaktere — bitte erst einen anlegen)";
                loreBookScopeId.appendChild(opt);
            } else {
                for (const c of chars) {
                    const opt = document.createElement("option");
                    opt.value = c.id;
                    opt.textContent = c.name;
                    if (c.id === currentValue) opt.selected = true;
                    loreBookScopeId.appendChild(opt);
                }
            }
        } else {
            // Session: free-form input in a single option (we let the user
            // paste IDs from the Sessions tab — the active session is the
            // default if available).
            const opt = document.createElement("option");
            opt.value = currentValue || state.sessionId || "";
            opt.textContent = currentValue
                ? currentValue
                : (state.sessionId ? `aktive Session (${state.sessionId.slice(0, 8)}…)` : "—");
            loreBookScopeId.appendChild(opt);
        }
    }

    if (loreBookScope) loreBookScope.addEventListener("change", () => updateBookScopeIdRow());
    if (loreBookCancel) loreBookCancel.addEventListener("click", () => loreBookModal.hidden = true);
    if (loreBookModalClose) loreBookModalClose.addEventListener("click", () => loreBookModal.hidden = true);

    if (loreBookForm) loreBookForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            name: loreBookName.value.trim(),
            description: loreBookDescription.value,
            scope: loreBookScope.value,
            scope_id: loreBookScope.value === "global" ? "" : (loreBookScopeId.value || ""),
            token_budget: parseInt(loreBookTokenBudget.value, 10) || 1500,
            enabled: loreBookEnabled.checked,
        };
        const id = loreBookId.value;
        try {
            const resp = id
                ? await fetch(
                    `/api/v1/plugins/character_chat/lorebooks/${encodeURIComponent(id)}`,
                    { method: "PATCH", headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(payload) }
                  )
                : await fetch(
                    "/api/v1/plugins/character_chat/lorebooks",
                    { method: "POST", headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(payload) }
                  );
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }
            const body = await resp.json();
            loreBookModal.hidden = true;
            // Make the just-created book the active one.
            if (!id && body.book) state.lore.activeBookId = body.book.id;
            await loadLorebooks();
            toast(id ? "Lorebook aktualisiert" : "Lorebook erstellt", payload.name);
        } catch (err) {
            toast("Fehler", err.message);
        }
    });

    if (loreBookDelete) loreBookDelete.addEventListener("click", async () => {
        const id = loreBookId.value;
        if (!id) return;
        const book = state.lore.books.find(b => b.id === id);
        if (!confirm(
            `Lorebook „${book?.name || id}" wirklich löschen?\n` +
            "Alle Einträge werden mit gelöscht (cascade). Nicht rückgängig machbar."
        )) return;
        try {
            const resp = await fetch(
                `/api/v1/plugins/character_chat/lorebooks/${encodeURIComponent(id)}`,
                { method: "DELETE" }
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            loreBookModal.hidden = true;
            if (state.lore.activeBookId === id) {
                state.lore.activeBookId = null;
                state.lore.entries = [];
            }
            await loadLorebooks();
            toast("Lorebook gelöscht", book?.name || id);
        } catch (err) {
            toast("Fehler", err.message);
        }
    });

    // ── Entry modal ───────────────────────────────────────────────
    function openEntryModalForNew(bookId) {
        loreEntryForm.reset();
        loreEntryId.value = "";
        loreEntryBookId.value = bookId;
        loreEntryEnabled.checked = true;
        loreEntryAlwaysOn.checked = false;
        loreEntryPosition.value = "before_scenario";
        loreEntryPriority.value = 100;
        loreEntryScanDepth.value = 4;
        loreEntryDelete.hidden = true;
        loreEntryModalTitle.textContent = "Neuer Eintrag";
        loreEntryModal.hidden = false;
        loreEntryName.focus();
    }

    function openEntryModalForEdit(entry) {
        loreEntryForm.reset();
        loreEntryId.value = entry.id;
        loreEntryBookId.value = entry.lorebook_id;
        loreEntryName.value = entry.name;
        loreEntryKeys.value = (entry.keys || []).join(", ");
        loreEntryAlwaysOn.checked = !!entry.always_on;
        loreEntryContent.value = entry.content || "";
        loreEntryPosition.value = entry.position;
        loreEntryPriority.value = entry.priority;
        loreEntryScanDepth.value = entry.scan_depth;
        loreEntryEnabled.checked = !!entry.enabled;
        loreEntryDelete.hidden = false;
        loreEntryModalTitle.textContent = `Eintrag · ${entry.name}`;
        loreEntryModal.hidden = false;
        loreEntryName.focus();
    }

    if (loreEntryCancel) loreEntryCancel.addEventListener("click", () => loreEntryModal.hidden = true);
    if (loreEntryModalClose) loreEntryModalClose.addEventListener("click", () => loreEntryModal.hidden = true);

    if (loreEntryForm) loreEntryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const keys = (loreEntryKeys.value || "")
            .split(",")
            .map(k => k.trim())
            .filter(Boolean);
        if (!keys.length && !loreEntryAlwaysOn.checked) {
            toast("Eintrag braucht Trigger", "Mindestens ein Key oder Always-On.");
            return;
        }
        const payload = {
            name: loreEntryName.value.trim(),
            keys: keys,
            content: loreEntryContent.value,
            position: loreEntryPosition.value,
            priority: parseInt(loreEntryPriority.value, 10) || 100,
            always_on: loreEntryAlwaysOn.checked,
            scan_depth: parseInt(loreEntryScanDepth.value, 10) || 4,
            enabled: loreEntryEnabled.checked,
        };
        const id = loreEntryId.value;
        const bookId = loreEntryBookId.value;
        try {
            const resp = id
                ? await fetch(
                    `/api/v1/plugins/character_chat/lore_entries/${encodeURIComponent(id)}`,
                    { method: "PATCH", headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(payload) }
                  )
                : await fetch(
                    `/api/v1/plugins/character_chat/lorebooks/${encodeURIComponent(bookId)}/entries`,
                    { method: "POST", headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(payload) }
                  );
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }
            loreEntryModal.hidden = true;
            await loadLoreEntries(bookId);
            toast(id ? "Eintrag aktualisiert" : "Eintrag erstellt", payload.name);
        } catch (err) {
            toast("Fehler", err.message);
        }
    });

    if (loreEntryDelete) loreEntryDelete.addEventListener("click", async () => {
        const id = loreEntryId.value;
        const bookId = loreEntryBookId.value;
        if (!id) return;
        const entry = state.lore.entries.find(en => en.id === id);
        if (!confirm(`Eintrag „${entry?.name || id}" wirklich löschen?`)) return;
        try {
            const resp = await fetch(
                `/api/v1/plugins/character_chat/lore_entries/${encodeURIComponent(id)}`,
                { method: "DELETE" }
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            loreEntryModal.hidden = true;
            await loadLoreEntries(bookId);
            toast("Eintrag gelöscht", entry?.name || id);
        } catch (err) {
            toast("Fehler", err.message);
        }
    });

    // ── Scope filter + buttons ────────────────────────────────────
    if (loreScopeFilter) loreScopeFilter.addEventListener("change", () => {
        state.lore.scope = loreScopeFilter.value;
        // Show / hide the scope-id select based on the chosen scope.
        if (state.lore.scope === "character") {
            populateLoreScopeIdFilter("character");
            loreScopeIdFilter.hidden = false;
        } else if (state.lore.scope === "session") {
            populateLoreScopeIdFilter("session");
            loreScopeIdFilter.hidden = false;
        } else {
            loreScopeIdFilter.hidden = true;
            state.lore.scopeId = "";
        }
        loadLorebooks();
    });

    if (loreScopeIdFilter) loreScopeIdFilter.addEventListener("change", () => {
        state.lore.scopeId = loreScopeIdFilter.value;
        loadLorebooks();
    });

    function populateLoreScopeIdFilter(kind) {
        loreScopeIdFilter.innerHTML = "";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = kind === "character" ? "(alle Charaktere)" : "(aktuelle Session)";
        loreScopeIdFilter.appendChild(blank);
        if (kind === "character") {
            for (const c of state.characters || []) {
                const opt = document.createElement("option");
                opt.value = c.id;
                opt.textContent = c.name;
                loreScopeIdFilter.appendChild(opt);
            }
        } else if (kind === "session" && state.sessionId) {
            const opt = document.createElement("option");
            opt.value = state.sessionId;
            opt.textContent = `aktive (${state.sessionId.slice(0, 8)}…)`;
            opt.selected = true;
            loreScopeIdFilter.appendChild(opt);
            state.lore.scopeId = state.sessionId;
        }
    }

    if (loreNewBookBtn) loreNewBookBtn.addEventListener("click", openBookModalForNew);
    if (loreRefreshBtn) loreRefreshBtn.addEventListener("click", loadLorebooks);

    // ─── Character session modal (chat toolbar) ─────────────────────

    function openCharacterSessionModal() {
        if (!charSessionModal || !state.sessionId) return;
        // Fetch session state + refresh character list to know what's attached.
        wsSend({ type: "character_session_get", session_id: state.sessionId });
        wsSend({ type: "simulation_status_get", session_id: state.sessionId });
        loadCharacters();
        charSessionModal.hidden = false;
    }

    function closeCharacterSessionModal() {
        if (charSessionModal) charSessionModal.hidden = true;
    }

    function renderCharacterSessionModal() {
        if (!charSessionAttachedList || !charSessionAvailableList) return;
        if (!state.sessionId) return;
        if (charSessionModeSelect) {
            charSessionModeSelect.value = String(charactersState.sessionState.character_mode || 0);
            if (charSessionModeHint) charSessionModeHint.textContent = _MODE_HINTS[charSessionModeSelect.value] || "";
        }
        charSessionScene.value = charactersState.sessionState.scene || "";

        const attached = charactersState.list.filter(
            (c) => !c.archived && (c.active_sessions || []).includes(state.sessionId)
        );
        const available = charactersState.list.filter(
            (c) => !c.archived && !(c.active_sessions || []).includes(state.sessionId)
        );

        const renderRow = (c, attachedMode) => {
            const row = document.createElement("div");
            row.className = "char-session-row";
            row.style.setProperty("--char-color", c.color || "#7aa2f7");
            const avatar = document.createElement("div");
            avatar.className = "char-avatar";
            avatar.style.width = "24px";
            avatar.style.height = "24px";
            if (c.avatar) {
                const img = document.createElement("img");
                img.src = c.avatar;
                avatar.appendChild(img);
            } else {
                avatar.textContent = (c.name || "?").slice(0, 1).toUpperCase();
            }
            row.appendChild(avatar);
            const name = document.createElement("div");
            name.className = "name";
            name.textContent = c.name;
            row.appendChild(name);
            if (c.age_stage && c.age_stage !== "adult") {
                const age = document.createElement("span");
                age.className = "age";
                age.textContent = c.age_stage;
                row.appendChild(age);
            }
            const btn = document.createElement("button");
            btn.className = "btn";
            btn.textContent = attachedMode ? "Entfernen" : "Anhängen";
            btn.addEventListener("click", () => {
                wsSend({
                    type: attachedMode ? "character_detach" : "character_attach",
                    id: c.id,
                    session_id: state.sessionId,
                });
            });
            row.appendChild(btn);
            return row;
        };

        charSessionAttachedList.innerHTML = "";
        if (!attached.length) {
            charSessionAttachedList.innerHTML = '<div class="muted">Keine Charaktere in dieser Session.</div>';
        } else {
            for (const c of attached) charSessionAttachedList.appendChild(renderRow(c, true));
        }

        charSessionAvailableList.innerHTML = "";
        if (!available.length) {
            charSessionAvailableList.innerHTML = '<div class="muted">Alle sind schon hier.</div>';
        } else {
            for (const c of available) charSessionAvailableList.appendChild(renderRow(c, false));
        }
    }

    if (chatCharactersBtn) chatCharactersBtn.addEventListener("click", openCharacterSessionModal);
    if (charSessionClose) charSessionClose.addEventListener("click", closeCharacterSessionModal);
    if (charSessionDone) charSessionDone.addEventListener("click", closeCharacterSessionModal);

    if (charSessionModeSelect) {
        charSessionModeSelect.addEventListener("change", () => {
            const mode = parseInt(charSessionModeSelect.value, 10) || 0;
            if (charSessionModeHint) charSessionModeHint.textContent = _MODE_HINTS[String(mode)] || "";
            wsSend({
                type: "character_session_set",
                session_id: state.sessionId,
                mode,
                scene: charSessionScene.value,
            });
        });
    }
    if (charSessionScene) {
        let sceneTimer = null;
        charSessionScene.addEventListener("input", () => {
            clearTimeout(sceneTimer);
            sceneTimer = setTimeout(() => {
                const mode = charSessionModeSelect ? parseInt(charSessionModeSelect.value, 10) || 0 : 0;
                wsSend({
                    type: "character_session_set",
                    session_id: state.sessionId,
                    mode,
                    scene: charSessionScene.value,
                });
            }, 400);
        });
    }

    function updateSimUI(running, intervalMinutes) {
        simRunning = !!running;
        if (charSessionSimToggle) {
            charSessionSimToggle.textContent = simRunning ? "⏸ Stop" : "▶ Start";
            charSessionSimToggle.classList.toggle("running", simRunning);
        }
        if (charSessionSimStatus) {
            charSessionSimStatus.textContent = simRunning
                ? `läuft (alle ${intervalMinutes || "?"} Min)`
                : "gestoppt";
            charSessionSimStatus.classList.toggle("running", simRunning);
        }
        if (charSessionSimInterval && intervalMinutes) {
            charSessionSimInterval.value = intervalMinutes;
        }
    }

    if (charSessionSimToggle) {
        charSessionSimToggle.addEventListener("click", () => {
            if (!state.sessionId) {
                toast("Simulation", "Keine aktive Session", 4000);
                return;
            }
            if (simRunning) {
                wsSend({
                    type: "simulation_stop",
                    session_id: state.sessionId,
                });
            } else {
                const interval = parseInt(
                    (charSessionSimInterval && charSessionSimInterval.value) || "3",
                    10
                );
                wsSend({
                    type: "simulation_start",
                    session_id: state.sessionId,
                    interval_minutes: interval,
                });
            }
        });
    }

    // ─── Chat character bar (under header, always-visible in chat) ──

    function updateChatCharacterBar() {
        if (!chatCharacterBar) return;
        const sid = state.sessionId;
        const attached = charactersState.list.filter(
            (c) => !c.archived && (c.active_sessions || []).includes(sid)
        );
        const mode = parseInt(charactersState.sessionState.character_mode || 0, 10);
        if (!attached.length && mode === 0) {
            chatCharacterBar.hidden = true;
            chatCharacterBar.innerHTML = "";
            return;
        }
        chatCharacterBar.hidden = false;
        chatCharacterBar.innerHTML = "";
        if (mode > 0) {
            const tag = document.createElement("span");
            tag.className = "cbar-mode";
            const labels = { 1: "🎭 Nur Charaktere", 2: "🎭 Hybrid (Lexy + Charaktere)" };
            tag.textContent = labels[mode] || "🎭 Character-Mode";
            chatCharacterBar.appendChild(tag);
        }
        for (const c of attached) {
            const chip = document.createElement("span");
            chip.className = "cbar-chip";
            const dot = document.createElement("span");
            dot.className = "dot";
            dot.style.background = c.color || "#7aa2f7";
            chip.appendChild(dot);
            chip.appendChild(document.createTextNode(c.name));
            chatCharacterBar.appendChild(chip);
        }
    }

    // ─── Character turn rendering ──────────────────────────────────

    function appendCharacterTurn(turn) {
        // Phase 9.12: Charakter-Turns gehören eigentlich immer in den
        // Rollenspiel-Tab. Falls aus Legacy-Gründen ein Char-Turn für
        // eine Chat-Session reinkommt (alter Sim-Modus, Pre-9.12-Turn-
        // Replay), zeigen wir ihn als Fallback im chat-Tab. Routing
        // anhand turn.session_id stellt sicher, dass ein Pulse, der
        // von einer Hintergrund-Session kommt, im richtigen Window
        // landet — auch wenn der User gerade einen anderen Tab offen
        // hat.
        let targetWindow = rpChatWindow;
        if (turn.session_id && turn.session_id === state.chatSessionId) {
            targetWindow = chatWindow;
        } else if (!turn.session_id) {
            // Kein session_id im Payload (alte Backends) → aktuelle
            // Tab-Wahl als Best-Guess nehmen.
            targetWindow = (state.activeTab === "rp" && rpChatWindow)
                ? rpChatWindow
                : chatWindow;
        }
        if (!targetWindow) return;
        const card = charactersState.list.find((c) => c.id === turn.character_id);
        const wrapper = document.createElement("div");
        wrapper.className = "msg character" + (turn.skipped ? " skipped" : "");
        wrapper.style.setProperty("--char-color", (card && card.color) || "#7aa2f7");
        // Tag the wrapper with the turn_id so the WS update/delete
        // broadcasts can find and patch this exact bubble later.
        if (turn.turn_id) wrapper.dataset.turnId = turn.turn_id;
        if (turn.character_id) wrapper.dataset.characterId = turn.character_id;

        const avatar = document.createElement("div");
        avatar.className = "char-bubble-avatar";
        if (card && card.avatar) {
            const img = document.createElement("img");
            img.src = card.avatar;
            avatar.appendChild(img);
        } else {
            avatar.textContent = (turn.character_name || "?").slice(0, 1).toUpperCase();
        }
        wrapper.appendChild(avatar);

        const body = document.createElement("div");
        body.className = "char-bubble-body";
        const nameRow = document.createElement("div");
        nameRow.className = "char-bubble-name";
        nameRow.textContent = turn.character_name || "?";
        body.appendChild(nameRow);
        const text = document.createElement("div");
        text.className = "char-bubble-text";
        text.textContent = turn.skipped
            ? `*${turn.character_name} schweigt*`
            : (turn.content || "");
        body.appendChild(text);

        // Action bar — at parity with normal-chat bubbles. Edit / delete /
        // regenerate dispatch via WS to the character_chat plugin's
        // _ws_turn_* handlers.
        if (turn.turn_id) {
            const actions = buildCharacterTurnActions(wrapper, turn);
            body.appendChild(actions);
        }
        wrapper.appendChild(body);

        targetWindow.appendChild(wrapper);
        targetWindow.scrollTop = targetWindow.scrollHeight;
    }

    function buildCharacterTurnActions(wrapper, turn) {
        const bar = document.createElement("div");
        bar.className = "msg-actions char-msg-actions";

        const editBtn = document.createElement("button");
        editBtn.className = "msg-action";
        editBtn.textContent = "✎ Edit";
        editBtn.title = "Diesen Turn-Text bearbeiten";
        editBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            startEditCharacterTurn(wrapper, turn);
        });
        bar.appendChild(editBtn);

        const regenBtn = document.createElement("button");
        regenBtn.className = "msg-action primary";
        regenBtn.textContent = "↻ Regenerate";
        regenBtn.title = "Diesen Turn neu generieren (denselben Trigger)";
        regenBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            if (!confirm(`${turn.character_name}'s Antwort neu generieren?`)) return;
            wsSend({ type: "character_turn_regenerate", turn_id: turn.turn_id });
            regenBtn.disabled = true;
            regenBtn.textContent = "…";
        });
        bar.appendChild(regenBtn);

        const delBtn = document.createElement("button");
        delBtn.className = "msg-action danger";
        delBtn.textContent = "🗑 Delete";
        delBtn.title = "Diesen Turn löschen";
        delBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            if (!confirm(`Turn von ${turn.character_name} löschen?`)) return;
            wsSend({ type: "character_turn_delete", turn_id: turn.turn_id });
        });
        bar.appendChild(delBtn);
        return bar;
    }

    function startEditCharacterTurn(wrapper, turn) {
        const textEl = wrapper.querySelector(".char-bubble-text");
        if (!textEl) return;
        const original = turn.skipped ? "" : (turn.content || "");
        const ta = document.createElement("textarea");
        ta.className = "char-edit-textarea";
        ta.value = original;
        ta.rows = Math.min(12, Math.max(2, original.split("\n").length + 1));
        textEl.replaceWith(ta);
        ta.focus();
        ta.select();

        const oldBar = wrapper.querySelector(".char-msg-actions");
        const newBar = document.createElement("div");
        newBar.className = "msg-actions char-msg-actions";
        const saveBtn = document.createElement("button");
        saveBtn.className = "msg-action primary";
        saveBtn.textContent = "Speichern";
        const cancelBtn = document.createElement("button");
        cancelBtn.className = "msg-action";
        cancelBtn.textContent = "Abbrechen";
        newBar.appendChild(saveBtn);
        newBar.appendChild(cancelBtn);
        if (oldBar) oldBar.replaceWith(newBar);

        const restore = (newContent) => {
            const restoredText = document.createElement("div");
            restoredText.className = "char-bubble-text";
            restoredText.textContent = newContent || (turn.skipped ? `*${turn.character_name} schweigt*` : "");
            ta.replaceWith(restoredText);
            const restoredBar = buildCharacterTurnActions(wrapper, {
                ...turn, content: newContent, skipped: false,
            });
            newBar.replaceWith(restoredBar);
        };

        saveBtn.addEventListener("click", () => {
            const newContent = ta.value.trim();
            if (!newContent) {
                toast("Edit", "Inhalt darf nicht leer sein");
                return;
            }
            wsSend({
                type: "character_turn_edit",
                turn_id: turn.turn_id,
                content: newContent,
            });
            restore(newContent);
            // The server broadcast will arrive shortly and patch via
            // onTurnUpdated (idempotent — same content already applied).
        });
        cancelBtn.addEventListener("click", () => {
            restore(original);
        });
    }

    // ─── WS dispatch for character_chat messages (hook into existing switch) ──

    window._lexyCharacters = {
        onList: (data) => {
            charactersState.list = data.characters || [];
            if (state.activeTab === "characters") renderCharactersList();
            if (charSessionModal && !charSessionModal.hidden) renderCharacterSessionModal();
            updateChatCharacterBar();
        },
        onCreated: (data) => {
            if (data.error) { charsHint("Fehler: " + data.error, "error"); return; }
            // Server always re-broadcasts the full list on any character event, so
            // just refresh.
            loadCharacters();
        },
        onUpdated: (data) => {
            if (data.error) { charsHint("Fehler: " + data.error, "error"); return; }
            loadCharacters();
        },
        onDeleted: () => { loadCharacters(); },
        onSessionMode: (data) => {
            if (data && data.session_id === state.sessionId) {
                charactersState.sessionState = {
                    character_mode: !!data.character_mode,
                    scene: data.scene || "",
                };
                if (charSessionModal && !charSessionModal.hidden) renderCharacterSessionModal();
                updateChatCharacterBar();
            }
        },
        onSessionGet: (data) => {
            if (data && data.ok && data.session_id === state.sessionId) {
                charactersState.sessionState = {
                    character_mode: !!data.character_mode,
                    scene: data.scene || "",
                };
                renderCharacterSessionModal();
                updateChatCharacterBar();
            }
        },
        onRoundStart: (data) => {
            if (data.session_id !== state.sessionId) return;
            charactersState.roundsInFlight.add(data.round_id);
            // Pulse-triggered rounds no longer render a separate
            // "pulse [id] *schreit*" system message — the baby character
            // turns that follow already show what happened, and Lexy's
            // auto-reaction provides the response. The old system note
            // was redundant noise.
        },
        onTurn: (data) => {
            if (data.session_id !== state.sessionId) return;
            appendCharacterTurn(data);
        },
        onTurnUpdated: (data) => {
            if (!data || !data.turn_id) return;
            const wrapper = chatWindow && chatWindow.querySelector(
                `.msg.character[data-turn-id="${CSS.escape(data.turn_id)}"]`
            );
            if (!wrapper) return;
            const textEl = wrapper.querySelector(".char-bubble-text");
            if (textEl) {
                textEl.textContent = data.skipped
                    ? `*${data.character_name || "?"} schweigt*`
                    : (data.content || "");
            }
        },
        onTurnDeleted: (data) => {
            if (!data || !data.turn_id) return;
            const wrapper = chatWindow && chatWindow.querySelector(
                `.msg.character[data-turn-id="${CSS.escape(data.turn_id)}"]`
            );
            if (wrapper) wrapper.remove();
        },
        onTurnAudio: (data) => {
            // Per-character voice: backend sends us the turn text already
            // rendered as WAV bytes base64-encoded. Only play when the
            // message is for this session, and stop any currently-playing
            // Lexy TTS so voices don't overlap.
            if (data.session_id !== state.sessionId) return;
            if (!data.audio_b64) return;
            try {
                const mime = data.mime || "audio/wav";
                const src = "data:" + mime + ";base64," + data.audio_b64;
                // Reuse the global audio pipeline if present, else a detached
                // audio element — either way the user hears the voice.
                if (state.audio && typeof state.audio.playUrl === "function") {
                    state.audio.playUrl(src);
                } else {
                    const audio = new Audio(src);
                    audio.play().catch(() => {});
                }
            } catch (err) {
                console.warn("character_turn_audio.play_failed", err);
            }
        },
        onRoundDone: (data) => {
            if (data.session_id !== state.sessionId) return;
            charactersState.roundsInFlight.delete(data.round_id);
        },
        onRoundError: (data) => {
            if (data.session_id !== state.sessionId) return;
            showError("Character round error: " + (data.error || "unknown"));
        },
    };

    function escapeHtml(s) {
        return String(s || "").replace(/[&<>"]/g, (c) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
        }[c]));
    }

    // ── Init ───────────────────────────────────────────────────────
    // The arc reactor lives in the DOM immediately; a short timeout
    // lets the tab layout settle before we size the canvas.
    setTimeout(() => state.visualizer.resize(), 50);
    connect();
    loadHealth();
    loadProjects();
    systemNote("Willkommen bei Lexy AI. Klick TTS on oder halte das Mic-Icon.");
})();


/* ─── Coder UI (approval modal + live task console, Phase 10.B+C) ──── */
(() => {
    const $ = (id) => document.getElementById(id);

    const overlay     = $("coder-approval-overlay");
    const modalRisk   = $("coder-approval-risk");
    const modalTitle  = $("coder-approval-title");
    const modalAction = $("coder-approval-action");
    const modalPreview= $("coder-approval-preview");
    const modalPayload= $("coder-approval-payload");
    const btnApprove  = $("coder-approve-btn");
    const btnDeny     = $("coder-deny-btn");
    const btnSession  = $("coder-approve-session-btn");
    const consoleEl   = $("coder-console");
    const consoleBody = $("coder-console-body");
    const consoleClose= $("coder-console-close");

    // No coder UI on this page (e.g. older index.html) — bail safely.
    if (!overlay || !consoleEl) {
        window._lexyCoder = { onApprovalRequest: () => {}, onTaskEvent: () => {} };
        return;
    }

    // ── Modal state ────────────────────────────────────────────────
    let pendingRequest = null;

    function getActiveSessionId() {
        try {
            return window._lexySessionId
                || localStorage.getItem("lexy_last_session")
                || "";
        } catch (_e) { return ""; }
    }

    function wsSendCoder(payload) {
        // We don't have direct access to the chat IIFE's wsSend helper, so
        // we open a tiny shortcut via window-level event the chat module
        // listens for. The chat IIFE registered `window.LexyHolo` etc., but
        // not its ws — so we use a CustomEvent and register the listener
        // back in the chat IIFE if needed. For now, dispatch directly via
        // the global ws if the chat IIFE exposed it.
        if (window._lexySendWs) {
            return window._lexySendWs(payload);
        }
        console.warn("coder: no ws sender exposed; approval ignored");
        return false;
    }

    function showModal(req) {
        pendingRequest = req || null;
        if (!req) return;
        const risk = (req.risk || "med").toLowerCase();
        modalRisk.dataset.risk = risk;
        modalRisk.textContent = risk.toUpperCase();
        modalTitle.textContent = "Lexy fragt nach Genehmigung…";
        modalAction.textContent = req.action || "(unknown)";
        modalPreview.textContent = req.preview || "(no preview)";
        try {
            modalPayload.textContent = JSON.stringify(req.payload || {}, null, 2);
        } catch (_e) {
            modalPayload.textContent = String(req.payload || "");
        }
        // For HIGH risk, hide the session-approve shortcut so Mike can't
        // accidentally rubber-stamp dangerous actions.
        btnSession.hidden = (risk === "high");
        overlay.hidden = false;
    }

    function closeModal() {
        overlay.hidden = true;
        pendingRequest = null;
    }

    function decide(decision) {
        if (!pendingRequest) {
            closeModal();
            return;
        }
        wsSendCoder({
            type: "coder_approval_response",
            request_id: pendingRequest.request_id,
            decision,
            session_id: getActiveSessionId(),
            action: pendingRequest.action,
        });
        closeModal();
    }

    btnApprove.addEventListener("click", () => decide("approve"));
    btnDeny.addEventListener("click", () => decide("deny"));
    btnSession.addEventListener("click", () => decide("approve_session"));
    overlay.addEventListener("click", (ev) => {
        // Click outside the modal panel = treat as deny (safer default).
        if (ev.target === overlay) decide("deny");
    });
    document.addEventListener("keydown", (ev) => {
        if (overlay.hidden) return;
        if (ev.key === "Escape") decide("deny");
        if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) decide("approve");
    });

    // ── Console state ──────────────────────────────────────────────
    const tasks = new Map();   // task_id → DOM element

    function showConsole() {
        consoleEl.hidden = false;
    }
    consoleClose.addEventListener("click", () => {
        consoleEl.hidden = true;
    });

    function renderTask(task) {
        let card = tasks.get(task.task_id);
        if (!card) {
            card = document.createElement("div");
            card.className = "coder-task-card";
            card.dataset.taskId = task.task_id;
            consoleBody.prepend(card);
            tasks.set(task.task_id, card);
        }
        const stateBadge = `<span class="coder-task-state" data-state="${task.state}">${task.state}</span>`;
        const desc = (task.description || "").slice(0, 80) || "(no description)";
        const stepsHtml = (task.steps || []).map((s, idx) => {
            const completed = s.completed ? "true" : "false";
            const isCurrent = !s.completed && (task.steps || []).slice(0, idx).every(p => p.completed);
            const cls = isCurrent && task.state === "running" ? "coder-task-step current" : "coder-task-step";
            const retryHint = s.retries > 0 ? ` <span style="color:var(--muted)">(retry ${s.retries})</span>` : "";
            return `<div class="${cls}" data-completed="${completed}">${escapeHtml(s.description.slice(0, 70))}${retryHint}</div>`;
        }).join("");
        const errorHtml = task.last_error
            ? `<div class="coder-task-error" title="${escapeHtml(task.last_error)}">${escapeHtml(task.last_error.slice(0, 90))}</div>`
            : "";
        card.innerHTML = `
            <div class="coder-task-head">
                ${stateBadge}
                <span class="coder-task-desc" title="${escapeHtml(task.description || "")}">${escapeHtml(desc)}</span>
            </div>
            ${stepsHtml ? `<div class="coder-task-steps">${stepsHtml}</div>` : ""}
            ${errorHtml}
        `;
        showConsole();
    }

    function escapeHtml(str) {
        return String(str || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    // ── Public API for the chat module ─────────────────────────────
    window._lexyCoder = {
        onApprovalRequest(req) {
            if (!req) return;
            showModal(req);
        },
        onTaskEvent(task) {
            if (!task || !task.task_id) return;
            renderTask(task);
        },
    };
})();


/* ─── Deep-Research UI (live progress + markdown report, Phase 12) ─── */
(() => {
    const $ = (id) => document.getElementById(id);

    const consoleEl   = $("research-console");
    const consoleBody = $("research-console-body");
    const consoleClose= $("research-console-close");

    if (!consoleEl || !consoleBody) {
        window._lexyResearch = { onEvent: () => {} };
        return;
    }

    consoleClose.addEventListener("click", () => { consoleEl.hidden = true; });

    const tasks = new Map(); // research_id → DOM card

    function showConsole() { consoleEl.hidden = false; }

    function escapeHtml(str) {
        return String(str ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    /**
     * Bare-bones markdown → HTML renderer for the report bubble. Covers
     * the subset the synthesiser actually emits (headings, links, lists,
     * inline code). Anything else falls through as escaped text — we
     * deliberately don't pull in marked.js for this.
     */
    function renderMarkdown(md) {
        if (!md) return "";
        const lines = md.split(/\r?\n/);
        const out = [];
        let inList = false;
        const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };

        for (let raw of lines) {
            let line = raw;
            const headingMatch = line.match(/^(#{1,3})\s+(.*)$/);
            if (headingMatch) {
                closeList();
                const level = headingMatch[1].length;
                out.push(`<h${level}>${inlineMd(headingMatch[2])}</h${level}>`);
                continue;
            }
            const liMatch = line.match(/^[-*]\s+(.*)$/);
            if (liMatch) {
                if (!inList) { out.push("<ul>"); inList = true; }
                out.push(`<li>${inlineMd(liMatch[1])}</li>`);
                continue;
            }
            // Numbered "[1] Title — URL" sources lines render as a link.
            const sourceMatch = line.match(/^\[(\d+)\]\s+(.+?)\s+—\s+(https?:\/\/\S+)$/);
            if (sourceMatch) {
                closeList();
                const [, n, title, url] = sourceMatch;
                out.push(
                    `<div>[${n}] <a href="${escapeHtml(url)}" target="_blank" rel="noopener">` +
                    `${escapeHtml(title)}</a></div>`,
                );
                continue;
            }
            closeList();
            if (line.trim() === "") {
                out.push("");
            } else {
                out.push(`<p>${inlineMd(line)}</p>`);
            }
        }
        closeList();
        return out.join("\n");
    }

    function inlineMd(text) {
        // Citations like [1] or [1, 2] stay as-is but get a CSS hook later.
        let out = escapeHtml(text);
        // Bold + italics (very loose).
        out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        out = out.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
        // Inline code.
        out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
        // Auto-link bare http(s) URLs.
        out = out.replace(
            /(https?:\/\/[^\s<>"']+)/g,
            (m) => `<a href="${m}" target="_blank" rel="noopener">${m}</a>`,
        );
        return out;
    }

    function renderTask(task) {
        let card = tasks.get(task.research_id);
        if (!card) {
            card = document.createElement("div");
            card.className = "research-task-card";
            card.dataset.researchId = task.research_id;
            consoleBody.prepend(card);
            tasks.set(task.research_id, card);
        }

        const state = task.state || "pending";
        const progress = task.progress || {};
        const subqueriesHtml = (task.subqueries || task.plan || []).map((sq, idx) => {
            // task.subqueries is the structured list (with state); task.plan is just strings.
            const desc = typeof sq === "string" ? sq : (sq.query || `step ${idx+1}`);
            const completed = typeof sq === "object" && sq.completed === true;
            const sources = typeof sq === "object" && Array.isArray(sq.sources)
                ? sq.sources.length
                : 0;
            const fetched = typeof sq === "object" && Array.isArray(sq.sources)
                ? sq.sources.filter(s => s.fetched).length
                : 0;
            const marker = completed ? "✓" : (state === "searching" ? "→" : "·");
            const countText = sources
                ? ` ${fetched}/${sources}`
                : "";
            return `
                <div class="research-task-subquery" data-completed="${completed}">
                    <span class="sq-marker">${marker}</span>
                    <span class="sq-text" title="${escapeHtml(desc)}">${escapeHtml(desc)}</span>
                    <span class="sq-count">${escapeHtml(countText)}</span>
                </div>
            `;
        }).join("");

        const meta = [];
        if (progress.sources_fetched != null) {
            meta.push(`${progress.sources_fetched}/${progress.sources_total || "?"} Quellen`);
        }
        if (progress.sources_relevant != null && progress.sources_relevant > 0) {
            meta.push(`${progress.sources_relevant} relevant`);
        }
        const metaHtml = meta.length
            ? `<div class="research-task-meta">${escapeHtml(meta.join(" · "))}</div>`
            : "";

        const errorHtml = task.last_error
            ? `<div class="research-task-error" title="${escapeHtml(task.last_error)}">${escapeHtml(task.last_error.slice(0,160))}</div>`
            : "";

        const reportHtml = (state === "done" && task.report)
            ? `<div class="research-task-report">${renderMarkdown(task.report)}</div>`
            : "";

        card.innerHTML = `
            <div class="research-task-head">
                <span class="research-task-state" data-state="${state}">${state}</span>
                <span class="research-task-topic" title="${escapeHtml(task.topic || "")}">${escapeHtml(task.topic || "")}</span>
            </div>
            ${metaHtml}
            ${subqueriesHtml ? `<div class="research-task-subqueries">${subqueriesHtml}</div>` : ""}
            ${errorHtml}
            ${reportHtml}
        `;
        showConsole();
    }

    window._lexyResearch = {
        onEvent(data) {
            if (!data || !data.research_id) return;
            renderTask(data);
        },
    };
})();
