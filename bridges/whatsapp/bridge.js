/**
 * Lexy AI — WhatsApp Bridge (Baileys)
 *
 * Connects to WhatsApp via @whiskeysockets/baileys (multi-device),
 * buffers inbound messages and exposes three REST endpoints for the
 * Lexy `channel_whatsapp` plugin:
 *
 *   GET  /health             → { status: "ok", connected: true/false }
 *   GET  /inbound            → [ {jid, text, id, timestamp, from_me} ]
 *                               (drains the buffer — Lexy polls every 2 s)
 *   POST /send  {jid, text}  → { status: "sent" }
 *
 * Authentication: every request must carry `X-API-Key: <shared secret>`.
 *
 * On first start the bridge prints a QR code in the terminal. Scan it
 * with the WhatsApp app on Lexy's phone (049 176 211 05176) under
 * "Linked Devices → Link a Device". The session is persisted in
 * ./auth_state/ so subsequent starts reconnect automatically.
 *
 * Usage:
 *   cd bridges/whatsapp
 *   npm install
 *   node bridge.js            (or: npm start)
 *   node bridge.js --verbose  (or: npm run dev)
 */

"use strict";

const {
  default: makeWASocket,
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
} = require("@whiskeysockets/baileys");
const express = require("express");
const pino = require("pino");
const qrcode = require("qrcode-terminal");
const path = require("path");

// ─── Config ──────────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.BRIDGE_PORT || "3000", 10);
const API_KEY = process.env.BRIDGE_API_KEY || "lexy-secret";
const AUTH_DIR = process.env.BRIDGE_AUTH_DIR || path.join(__dirname, "auth_state");
const VERBOSE = process.argv.includes("--verbose");
const MAX_BUFFER = 500; // max queued inbound messages before oldest are dropped
const LEXY_PHONE = "4917621105176"; // Lexy's WhatsApp number (without leading 0)

// ─── Logger ──────────────────────────────────────────────────────────────────

const logger = pino({
  level: VERBOSE ? "debug" : "warn",
  transport: { target: "pino/file", options: { destination: 1 } },
});

// ─── State ───────────────────────────────────────────────────────────────────

let sock = null;
let connected = false;
const inboundBuffer = [];

// ─── Baileys connection ──────────────────────────────────────────────────────

async function startSocket() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  sock = makeWASocket({
    version,
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    logger,
    printQRInTerminal: false, // we handle QR ourselves for nicer output
    // Reduce noise: don't download media, don't sync full history
    syncFullHistory: false,
    markOnlineOnConnect: true,
    generateHighQualityLinkPreview: false,
  });

  // ── QR code ─────────────────────────────────────────────────────

  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      console.log("\n╔══════════════════════════════════════════════════╗");
      console.log("║  Scan this QR code with Lexy's WhatsApp phone   ║");
      console.log("║  (Linked Devices → Link a Device)                ║");
      console.log("╚══════════════════════════════════════════════════╝\n");
      qrcode.generate(qr, { small: true });
      console.log("");
    }

    if (connection === "open") {
      connected = true;
      console.log(`[bridge] ✓ Connected to WhatsApp (Lexy: +${LEXY_PHONE})`);
      console.log(`[bridge] ✓ Listening on http://127.0.0.1:${PORT}`);
      console.log(`[bridge] ✓ API key: ${API_KEY.slice(0, 4)}***`);
    }

    if (connection === "close") {
      connected = false;
      const statusCode =
        lastDisconnect?.error?.output?.statusCode ?? 0;
      const shouldReconnect = statusCode !== DisconnectReason.loggedOut;

      if (statusCode === DisconnectReason.loggedOut) {
        console.log("[bridge] ✗ Logged out. Delete auth_state/ and restart to re-link.");
      } else {
        console.log(
          `[bridge] ⚠ Connection closed (code ${statusCode}), reconnecting…`
        );
        if (shouldReconnect) {
          setTimeout(() => startSocket(), 3000);
        }
      }
    }
  });

  // ── Credentials persistence ─────────────────────────────────────

  sock.ev.on("creds.update", saveCreds);

  // ── Inbound messages ────────────────────────────────────────────

  sock.ev.on("messages.upsert", ({ messages: msgs, type }) => {
    // type "notify" = real-time messages, "append" = history sync
    if (type !== "notify") return;

    for (const msg of msgs) {
      // Skip status broadcasts
      if (msg.key.remoteJid === "status@broadcast") continue;

      // Extract text (regular message or extended text with URL preview)
      const text =
        msg.message?.conversation ||
        msg.message?.extendedTextMessage?.text ||
        "";

      if (!text) continue; // skip media-only messages for now

      const entry = {
        jid: msg.key.remoteJid,
        from_me: !!msg.key.fromMe,
        text,
        id: msg.key.id || "",
        timestamp: msg.messageTimestamp
          ? Number(msg.messageTimestamp)
          : Math.floor(Date.now() / 1000),
        pushName: msg.pushName || "",
      };

      if (VERBOSE) {
        const dir = entry.from_me ? "→" : "←";
        console.log(
          `[bridge] ${dir} ${entry.jid}: ${entry.text.slice(0, 80)}`
        );
      }

      inboundBuffer.push(entry);

      // Cap buffer size
      while (inboundBuffer.length > MAX_BUFFER) {
        inboundBuffer.shift();
      }
    }
  });

  return sock;
}

// ─── Express REST server ─────────────────────────────────────────────────────

const app = express();
app.use(express.json({ limit: "1mb" }));

// API key middleware
app.use((req, res, next) => {
  // /health is public so monitoring tools can hit it without a key
  if (req.path === "/health") return next();

  const key = req.headers["x-api-key"];
  if (key !== API_KEY) {
    return res.status(401).json({ error: "invalid or missing X-API-Key" });
  }
  next();
});

// ── GET /health ───────────────────────────────────────────────────

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    connected,
    phone: LEXY_PHONE,
    buffered: inboundBuffer.length,
    uptime: Math.floor(process.uptime()),
  });
});

// ── GET /inbound ──────────────────────────────────────────────────

app.get("/inbound", (_req, res) => {
  // Drain: return all buffered messages and clear the buffer
  const messages = inboundBuffer.splice(0, inboundBuffer.length);
  res.json(messages);
});

// ── POST /send ────────────────────────────────────────────────────

app.post("/send", async (req, res) => {
  const { jid, text } = req.body || {};
  if (!jid || !text) {
    return res.status(400).json({ error: "jid and text are required" });
  }
  if (!sock || !connected) {
    return res.status(503).json({ error: "not connected to WhatsApp" });
  }
  try {
    await sock.sendMessage(jid, { text });
    if (VERBOSE) {
      console.log(`[bridge] → ${jid}: ${text.slice(0, 80)}`);
    }
    res.json({ status: "sent", jid });
  } catch (err) {
    console.error(`[bridge] send error: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── 404 catch-all ─────────────────────────────────────────────────

app.use((_req, res) => {
  res.status(404).json({ error: "not found" });
});

// ─── Start ───────────────────────────────────────────────────────────────────

async function main() {
  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║        Lexy AI — WhatsApp Bridge (Baileys)       ║");
  console.log("╚══════════════════════════════════════════════════╝");
  console.log(`  Phone:    +${LEXY_PHONE}`);
  console.log(`  Port:     ${PORT}`);
  console.log(`  Auth dir: ${AUTH_DIR}`);
  console.log(`  Verbose:  ${VERBOSE}`);
  console.log("");

  app.listen(PORT, "127.0.0.1", () => {
    console.log(`[bridge] HTTP server listening on http://127.0.0.1:${PORT}`);
  });

  await startSocket();
}

main().catch((err) => {
  console.error("[bridge] Fatal:", err);
  process.exit(1);
});
