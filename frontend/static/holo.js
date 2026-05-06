/**
 * Lexy AI — HoloMat Visual Layer
 *
 * Boot Sequence · Screen Outline · Voice Rings · Sound Design
 * Three.js 3D Scene · Sleep/Wake · Theme Switcher
 *
 * Loaded BEFORE app.js. Provides global window.LexyHolo for app.js to
 * interact with (e.g. set outline state, trigger sounds, etc.).
 */

"use strict";

window.LexyHolo = (function () {
    // ═══════════════════════════════════════════════════════════════════
    //  SOUND DESIGN SYSTEM
    // ═══════════════════════════════════════════════════════════════════

    const SoundDesign = (() => {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        let enabled = true;

        // Programmatic sounds — no external files needed
        function play(type) {
            if (!enabled) return;
            // Unlock context on first interaction
            if (ctx.state === "suspended") ctx.resume();

            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            const t = ctx.currentTime;

            switch (type) {
                case "boot-start":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(200, t);
                    osc.frequency.exponentialRampToValueAtTime(800, t + 0.3);
                    gain.gain.setValueAtTime(0.15, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
                    osc.start(t); osc.stop(t + 0.5);
                    break;

                case "boot-step":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(600 + Math.random() * 200, t);
                    gain.gain.setValueAtTime(0.06, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
                    osc.start(t); osc.stop(t + 0.12);
                    break;

                case "boot-complete":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(400, t);
                    osc.frequency.setValueAtTime(600, t + 0.15);
                    osc.frequency.setValueAtTime(800, t + 0.3);
                    gain.gain.setValueAtTime(0.12, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.6);
                    osc.start(t); osc.stop(t + 0.6);
                    break;

                case "click":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(900, t);
                    gain.gain.setValueAtTime(0.04, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
                    osc.start(t); osc.stop(t + 0.06);
                    break;

                case "hover":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(1200, t);
                    gain.gain.setValueAtTime(0.02, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
                    osc.start(t); osc.stop(t + 0.04);
                    break;

                case "send":
                    osc.type = "triangle";
                    osc.frequency.setValueAtTime(500, t);
                    osc.frequency.exponentialRampToValueAtTime(1000, t + 0.15);
                    gain.gain.setValueAtTime(0.08, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
                    osc.start(t); osc.stop(t + 0.25);
                    break;

                case "receive":
                    osc.type = "triangle";
                    osc.frequency.setValueAtTime(800, t);
                    osc.frequency.exponentialRampToValueAtTime(500, t + 0.2);
                    gain.gain.setValueAtTime(0.06, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
                    osc.start(t); osc.stop(t + 0.3);
                    break;

                case "error":
                    osc.type = "sawtooth";
                    osc.frequency.setValueAtTime(200, t);
                    osc.frequency.setValueAtTime(150, t + 0.1);
                    gain.gain.setValueAtTime(0.1, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
                    osc.start(t); osc.stop(t + 0.3);
                    break;

                case "success":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(523, t);      // C5
                    osc.frequency.setValueAtTime(659, t + 0.1); // E5
                    osc.frequency.setValueAtTime(784, t + 0.2); // G5
                    gain.gain.setValueAtTime(0.08, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
                    osc.start(t); osc.stop(t + 0.4);
                    break;

                case "wake":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(300, t);
                    osc.frequency.exponentialRampToValueAtTime(600, t + 0.5);
                    gain.gain.setValueAtTime(0.1, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.8);
                    osc.start(t); osc.stop(t + 0.8);
                    break;

                case "sleep":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(500, t);
                    osc.frequency.exponentialRampToValueAtTime(200, t + 0.8);
                    gain.gain.setValueAtTime(0.06, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 1.0);
                    osc.start(t); osc.stop(t + 1.0);
                    break;

                case "thought":
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(440, t);
                    osc.frequency.exponentialRampToValueAtTime(550, t + 0.3);
                    gain.gain.setValueAtTime(0.05, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
                    osc.start(t); osc.stop(t + 0.5);
                    break;
            }
        }

        return { play, setEnabled: (v) => { enabled = v; } };
    })();

    // ═══════════════════════════════════════════════════════════════════
    //  BOOT SEQUENCE
    // ═══════════════════════════════════════════════════════════════════

    const Boot = (() => {
        const overlay = document.getElementById("boot-overlay");
        if (!overlay) return { skip: () => {} };

        const phase1 = overlay.querySelector(".boot-phase1");
        const phase2 = overlay.querySelector(".boot-transition");
        const phase3 = overlay.querySelector(".boot-status-screen");
        const btn = overlay.querySelector(".boot-btn");
        const progressBar = overlay.querySelector(".boot-progress-bar");
        const logEl = overlay.querySelector(".boot-log");

        const bootMessages = [
            { text: "Initializing Lexy Core v2.0.0 ...", cls: "highlight", delay: 100 },
            { text: "Loading config/config.yaml .......... [OK]", cls: "ok", delay: 200 },
            { text: "structlog configured ................ [OK]", cls: "ok", delay: 100 },
            { text: "EventBus + HookManager .............. [OK]", cls: "ok", delay: 150 },
            { text: "Embedding client (BAAI/bge-m3) ...... loading", cls: "", delay: 400 },
            { text: "  dimension=1024  device=cpu ........ [OK]", cls: "ok", delay: 100 },
            { text: "LLM Client connecting ...............", cls: "", delay: 200 },
            { text: "  brain a4b  127.0.0.1:5005 ........ [OK]", cls: "ok", delay: 200 },
            { text: "  brain e4b  127.0.0.1:5006 ........ [OK]", cls: "ok", delay: 150 },
            { text: "ToolRegistry + ToolCaller ........... [OK]", cls: "ok", delay: 100 },
            { text: "MemoryManager (ChromaDB) ............ connecting", cls: "", delay: 300 },
            { text: "  4 collections loaded .............. [OK]", cls: "ok", delay: 100 },
            { text: "VoiceManager (STT+TTS) .............. [OK]", cls: "ok", delay: 150 },
            { text: "PluginLoader: 9 plugins discovered", cls: "highlight", delay: 200 },
            { text: "  voice_cosyvoice ................... [OK]", cls: "ok", delay: 100 },
            { text: "  voice_gemma4 ...................... [OK]", cls: "ok", delay: 100 },
            { text: "  scheduler ......................... [OK]", cls: "ok", delay: 100 },
            { text: "  autonomous_thinking ............... [OK]", cls: "ok", delay: 80 },
            { text: "  web_crawler ....................... [OK]", cls: "ok", delay: 80 },
            { text: "  channel_whatsapp .................. [OK]", cls: "ok", delay: 80 },
            { text: "LexyAgent initialized ............... [OK]", cls: "ok", delay: 200 },
            { text: "WebSocket handlers registered ....... [OK]", cls: "ok", delay: 100 },
            { text: "FastAPI Gateway on 0.0.0.0:8765 .... [OK]", cls: "ok", delay: 150 },
            { text: "SessionStore: loaded sessions ....... [OK]", cls: "ok", delay: 100 },
            { text: "", cls: "", delay: 200 },
            { text: ">>> LEXY AI SYSTEM READY <<<", cls: "highlight", delay: 0 },
        ];

        function startBoot() {
            SoundDesign.play("boot-start");
            phase1.style.display = "none";
            phase2.classList.add("active");

            // Phase 2 → Phase 3
            setTimeout(() => {
                phase2.classList.remove("active");
                phase2.style.display = "none";
                phase3.classList.add("active");
                runBootLog();
            }, 1600);
        }

        async function runBootLog() {
            const total = bootMessages.length;
            for (let i = 0; i < total; i++) {
                const msg = bootMessages[i];
                if (msg.text) {
                    const line = document.createElement("div");
                    line.className = msg.cls || "";
                    line.textContent = msg.text;
                    logEl.appendChild(line);
                    logEl.scrollTop = logEl.scrollHeight;
                    SoundDesign.play("boot-step");
                }
                progressBar.style.width = `${((i + 1) / total) * 100}%`;
                await sleep(msg.delay);
            }
            SoundDesign.play("boot-complete");
            await sleep(600);
            overlay.classList.add("fade-out");
            setTimeout(() => {
                overlay.classList.add("hidden");
                ScreenOutline.build();
            }, 900);
        }

        if (btn) {
            btn.addEventListener("click", startBoot);
        }

        function skip() {
            overlay.classList.add("hidden");
            // ScreenOutline is declared LATER in this file. When the
            // browser hits this point on a returning visit (sessionStorage
            // already set) the const is still in its TDZ and a direct
            // reference would throw "can't access lexical declaration
            // 'ScreenOutline' before initialization". Defer the call to
            // a microtask — by the time it fires, the rest of the module
            // (including ScreenOutline) has been initialised.
            queueMicrotask(() => {
                try { ScreenOutline.build(); } catch (_e) {}
            });
        }

        // Check if boot was already seen this session
        if (sessionStorage.getItem("lexy-booted")) {
            skip();
        }

        return { startBoot, skip };
    })();

    // ═══════════════════════════════════════════════════════════════════
    //  SCREEN OUTLINE
    // ═══════════════════════════════════════════════════════════════════

    const ScreenOutline = (() => {
        const el = document.getElementById("screen-outline");
        if (!el) return { build: () => {}, setState: () => {} };

        const edges = el.querySelectorAll(".outline-edge");
        const particles = el.querySelectorAll(".outline-particle");

        function build() {
            el.classList.add("active");
            edges.forEach((e) => e.classList.add("build"));
            setTimeout(() => {
                particles.forEach((p) => p.classList.add("traveling"));
            }, 1300); // After all edges have built
            sessionStorage.setItem("lexy-booted", "1");
        }

        function setState(state) {
            el.classList.remove("thinking", "speaking", "idle");
            if (state) el.classList.add(state);
        }

        return { build, setState };
    })();

    // ═══════════════════════════════════════════════════════════════════
    //  VOICE RINGS
    // ═══════════════════════════════════════════════════════════════════

    const VoiceRings = (() => {
        const container = document.getElementById("voice-rings-container");
        if (!container) return { setActive: () => {}, setLevel: () => {} };

        const RING_COUNT = 5;
        const rings = [];

        for (let i = 0; i < RING_COUNT; i++) {
            const ring = document.createElement("div");
            ring.className = "voice-ring";
            container.appendChild(ring);
            rings.push(ring);
        }

        function setActive(active) {
            if (active) {
                container.classList.add("active");
                rings.forEach((r) => r.classList.add("active"));
            } else {
                container.classList.remove("active");
                rings.forEach((r) => r.classList.remove("active"));
            }
        }

        function setLevel(level) {
            // level 0..1
            const clamped = Math.max(0, Math.min(1, level));
            rings.forEach((ring, i) => {
                const baseSize = 40 + i * 30;
                const dynamic = clamped * i * 8;
                const size = baseSize + dynamic;
                ring.style.width = size + "px";
                ring.style.height = size + "px";
                ring.style.opacity = clamped > 0.01 ? (0.6 - i * 0.08) : 0;
                ring.style.boxShadow = `0 0 ${10 + clamped * 15}px var(--holo-glow)`;
            });
        }

        return { setActive, setLevel };
    })();

    // ═══════════════════════════════════════════════════════════════════
    //  THREE.JS 3D AVATAR STAGE
    // ═══════════════════════════════════════════════════════════════════

    const AvatarStage = (() => {
        const container = document.getElementById("avatar-stage");
        if (!container || typeof THREE === "undefined") {
            return { show: () => {}, hide: () => {}, setMood: () => {}, animate: () => {} };
        }

        let scene, camera, renderer, avatarGroup, particles;
        let animationId = null;
        let audioLevel = 0;
        let currentMood = "idle"; // idle, thinking, speaking, listening

        function init() {
            scene = new THREE.Scene();

            camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
            camera.position.set(0, 1.2, 3);
            camera.lookAt(0, 0.8, 0);

            renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setClearColor(0x000000, 0);
            container.appendChild(renderer.domElement);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404060, 0.6);
            scene.add(ambientLight);

            const accentColor = getComputedStyle(document.documentElement)
                .getPropertyValue("--holo-accent").trim() || "#00d4ff";
            const accentHex = parseInt(accentColor.replace("#", ""), 16);

            const pointLight = new THREE.PointLight(accentHex, 1.5, 10);
            pointLight.position.set(2, 3, 2);
            scene.add(pointLight);

            const rimLight = new THREE.PointLight(accentHex, 0.8, 10);
            rimLight.position.set(-2, 2, -1);
            scene.add(rimLight);

            // Avatar group — placeholder holographic figure
            avatarGroup = new THREE.Group();
            scene.add(avatarGroup);

            // Core orb (glowing sphere)
            const orbGeo = new THREE.SphereGeometry(0.3, 32, 32);
            const orbMat = new THREE.MeshPhongMaterial({
                color: accentHex,
                emissive: accentHex,
                emissiveIntensity: 0.4,
                transparent: true,
                opacity: 0.8,
                wireframe: false,
            });
            const orb = new THREE.Mesh(orbGeo, orbMat);
            orb.position.y = 1.0;
            orb.name = "core-orb";
            avatarGroup.add(orb);

            // Orbiting rings
            for (let i = 0; i < 3; i++) {
                const ringGeo = new THREE.TorusGeometry(0.5 + i * 0.15, 0.008, 8, 64);
                const ringMat = new THREE.MeshBasicMaterial({
                    color: accentHex,
                    transparent: true,
                    opacity: 0.5 - i * 0.12,
                });
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.position.y = 1.0;
                ring.rotation.x = Math.PI / 2 + i * 0.3;
                ring.name = `orbit-ring-${i}`;
                avatarGroup.add(ring);
            }

            // Floating particles around the avatar
            const particleCount = 60;
            const particleGeo = new THREE.BufferGeometry();
            const positions = new Float32Array(particleCount * 3);
            for (let i = 0; i < particleCount; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.random() * Math.PI;
                const r = 0.8 + Math.random() * 0.6;
                positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
                positions[i * 3 + 1] = 1.0 + r * Math.cos(phi) * 0.6;
                positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
            }
            particleGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            const particleMat = new THREE.PointsMaterial({
                color: accentHex,
                size: 0.03,
                transparent: true,
                opacity: 0.6,
                blending: THREE.AdditiveBlending,
            });
            particles = new THREE.Points(particleGeo, particleMat);
            avatarGroup.add(particles);

            // Ground disc (holographic base)
            const discGeo = new THREE.CircleGeometry(0.8, 64);
            const discMat = new THREE.MeshBasicMaterial({
                color: accentHex,
                transparent: true,
                opacity: 0.1,
                side: THREE.DoubleSide,
            });
            const disc = new THREE.Mesh(discGeo, discMat);
            disc.rotation.x = -Math.PI / 2;
            disc.position.y = 0.01;
            avatarGroup.add(disc);

            // Handle resize
            window.addEventListener("resize", () => {
                if (!container.clientWidth) return;
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });

            animate();
        }

        function animate() {
            animationId = requestAnimationFrame(animate);
            const time = Date.now() * 0.001;

            if (avatarGroup) {
                // Core orb breathing
                const orb = avatarGroup.getObjectByName("core-orb");
                if (orb) {
                    const breathScale = 1 + Math.sin(time * 1.5) * 0.08;
                    const audioBoost = 1 + audioLevel * 0.3;
                    orb.scale.setScalar(breathScale * audioBoost);
                    orb.material.emissiveIntensity = 0.3 + audioLevel * 0.5 + Math.sin(time * 2) * 0.1;
                }

                // Orbiting rings rotation
                for (let i = 0; i < 3; i++) {
                    const ring = avatarGroup.getObjectByName(`orbit-ring-${i}`);
                    if (ring) {
                        ring.rotation.z = time * (0.3 + i * 0.15) * (i % 2 === 0 ? 1 : -1);
                        ring.rotation.x = Math.PI / 2 + i * 0.3 + Math.sin(time * 0.5 + i) * 0.2;

                        // Speaking: rings expand; Thinking: rings pulse
                        if (currentMood === "speaking") {
                            ring.scale.setScalar(1 + audioLevel * 0.4);
                        } else if (currentMood === "thinking") {
                            ring.scale.setScalar(1 + Math.sin(time * 3 + i) * 0.15);
                        } else {
                            ring.scale.setScalar(1);
                        }
                    }
                }

                // Particles drift
                if (particles) {
                    particles.rotation.y = time * 0.1;
                    particles.material.opacity = 0.4 + audioLevel * 0.3;
                }
            }

            renderer.render(scene, camera);
        }

        function show() {
            container.classList.add("visible");
            if (!renderer) init();
        }

        function hide() {
            container.classList.remove("visible");
        }

        function setMood(mood) {
            currentMood = mood;
        }

        function setAudioLevel(level) {
            audioLevel = Math.max(0, Math.min(1, level));
        }

        return { show, hide, setMood, setAudioLevel, init };
    })();

    // ═══════════════════════════════════════════════════════════════════
    //  SLEEP / WAKE
    // ═══════════════════════════════════════════════════════════════════

    const SleepWake = (() => {
        const SLEEP_TIMEOUT = 5 * 60 * 1000; // 5 minutes
        let timer = null;
        let sleeping = false;
        const clockEl = document.querySelector(".sleep-clock");

        function resetTimer() {
            if (sleeping) wake();
            clearTimeout(timer);
            timer = setTimeout(goSleep, SLEEP_TIMEOUT);
        }

        function goSleep() {
            if (sleeping) return;
            sleeping = true;
            document.body.classList.add("sleeping");
            SoundDesign.play("sleep");
            ScreenOutline.setState("idle");
            updateClock();
        }

        function wake() {
            if (!sleeping) return;
            sleeping = false;
            document.body.classList.remove("sleeping");
            SoundDesign.play("wake");
        }

        function updateClock() {
            if (!clockEl) return;
            function tick() {
                if (!sleeping) return;
                const now = new Date();
                clockEl.textContent =
                    String(now.getHours()).padStart(2, "0") + ":" +
                    String(now.getMinutes()).padStart(2, "0");
                setTimeout(tick, 10000);
            }
            tick();
        }

        // Activity listeners
        ["mousemove", "mousedown", "keydown", "touchstart", "scroll"].forEach((evt) => {
            document.addEventListener(evt, resetTimer, { passive: true });
        });
        resetTimer();

        return { wake, goSleep, isAsleep: () => sleeping };
    })();

    // ═══════════════════════════════════════════════════════════════════
    //  THEME SWITCHER
    // ═══════════════════════════════════════════════════════════════════

    const Theme = (() => {
        const STORAGE_KEY = "lexy-theme";
        const DEFAULT = "cyber-blue";

        function apply(name) {
            document.documentElement.setAttribute("data-theme", name);
            localStorage.setItem(STORAGE_KEY, name);
            // Update accent in existing CSS vars so legacy styles match
            const style = getComputedStyle(document.documentElement);
            const accent = style.getPropertyValue("--holo-accent").trim();
            if (accent) {
                document.documentElement.style.setProperty("--accent", accent);
                document.documentElement.style.setProperty("--accent-dim", accent);
                document.documentElement.style.setProperty("--accent-ghost",
                    `rgba(${style.getPropertyValue("--holo-accent-rgb").trim()}, 0.1)`);
            }
            // Update swatches
            document.querySelectorAll(".theme-swatch").forEach((sw) => {
                sw.classList.toggle("active", sw.dataset.theme === name);
            });
        }

        function init() {
            const saved = localStorage.getItem(STORAGE_KEY) || DEFAULT;
            apply(saved);
            // Wire swatches
            document.querySelectorAll(".theme-swatch").forEach((sw) => {
                sw.addEventListener("click", () => {
                    apply(sw.dataset.theme);
                    SoundDesign.play("click");
                });
            });
        }

        return { apply, init, current: () => localStorage.getItem(STORAGE_KEY) || DEFAULT };
    })();

    // Init theme immediately
    Theme.init();

    // ═══════════════════════════════════════════════════════════════════
    //  UTILITY
    // ═══════════════════════════════════════════════════════════════════

    function sleep(ms) {
        return new Promise((r) => setTimeout(r, ms));
    }

    // ═══════════════════════════════════════════════════════════════════
    //  PUBLIC API
    // ═══════════════════════════════════════════════════════════════════

    return {
        sound: SoundDesign,
        boot: Boot,
        outline: ScreenOutline,
        rings: VoiceRings,
        avatar: AvatarStage,
        sleep: SleepWake,
        theme: Theme,
    };
})();
