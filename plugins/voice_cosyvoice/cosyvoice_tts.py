"""
Lexy AI - CosyVoice 3 TTS Provider.

Remote TTS via CosyVoice 3 REST API on a dedicated server.
Model: Fun-CosyVoice3-0.5B (multilingual).

Features:
* Text cleanup (markdown, emojis, special chars)
* Narrator detection (``*actions in asterisks*``)
* Emotion mapping (narrator text → instruct prompt)
* Segment-based streaming (one WAV per segment)
"""

from __future__ import annotations

import asyncio
import io
import re
import wave
from typing import AsyncGenerator

import httpx

from lexy_core.utils.logging import get_logger
from lexy_core.voice.tts_base import TTSProvider

log = get_logger(module="cosyvoice_tts")

# ─── Emotion mapping (narrator keyword → instruct prompt) ──────────────────

_EMOTION_MAP: dict[str, str] = {
    # German
    "traurig": "Please speak sadly and softly.",
    "weint": "Please speak sadly, almost crying.",
    "seufzt": "Please speak with a tired sigh.",
    "lacht": "Please speak happily with laughter.",
    "kichert": "Please speak playfully with a giggle.",
    "freut": "Please speak happily and with excitement.",
    "froh": "Please speak happily and with excitement.",
    "gluecklich": "Please speak happily.",
    "wuetend": "Please speak angrily.",
    "aergerlich": "Please speak angrily.",
    "aengstlich": "Please speak nervously and with fear.",
    "nervoes": "Please speak nervously.",
    "fluestert": "Please speak in a whispering voice.",
    "leise": "Please speak very softly.",
    "laut": "Please speak loudly with energy.",
    "schreit": "Please speak loudly, almost shouting.",
    "erschoepft": "Please speak tiredly and slowly.",
    "muede": "Please speak tiredly and slowly.",
    "aufgeregt": "Please speak with great enthusiasm.",
    "begeistert": "Please speak with great enthusiasm.",
    "nachdenklich": "Please speak thoughtfully and slowly.",
    "ueberrascht": "Please speak with surprise.",
    "erstaunt": "Please speak with surprise.",
    "sanft": "Please speak warmly and softly.",
    "liebevoll": "Please speak warmly and lovingly.",
    "schuechtern": "Please speak shyly and quietly.",
    "genervt": "Please speak with annoyance.",
    "sarkastisch": "Please speak sarcastically.",
    "ernst": "Please speak seriously.",
    "verzweifelt": "Please speak desperately.",
    "erschrocken": "Please speak with shock and surprise.",
    "verlegen": "Please speak shyly and embarrassed.",
    "verwirrt": "Please speak with confusion.",
    "cool": "Please speak confidently and relaxed.",
    "zwinkert": "Please speak playfully.",
    "zustimmend": "Please speak approvingly.",
    # English
    "sad": "Please speak sadly and softly.",
    "happy": "Please speak happily.",
    "angry": "Please speak angrily.",
    "whisper": "Please speak in a whispering voice.",
    "excited": "Please speak with great enthusiasm.",
    "tired": "Please speak tiredly and slowly.",
    "scared": "Please speak nervously and with fear.",
    "laughs": "Please speak happily with laughter.",
    "sighs": "Please speak with a tired sigh.",
}

_EMOJI_TO_EMOTION: dict[str, str] = {
    "\U0001F60A": "*freut sich*",
    "\U0001F604": "*lacht*",
    "\U0001F602": "*lacht*",
    "\U0001F622": "*traurig*",
    "\U0001F625": "*traurig*",
    "\U0001F62D": "*weint*",
    "\U0001F620": "*wuetend*",
    "\U0001F621": "*wuetend*",
    "\U0001F914": "*nachdenklich*",
    "\U0001F631": "*erschrocken*",
    "\U0001F60D": "*liebevoll*",
    "\U0001F917": "*liebevoll*",
    "\U0001F61E": "*traurig*",
    "\U0001F633": "*verlegen*",
    "\U0001F609": "*zwinkert*",
    "\U0001F60E": "*cool*",
    "\U0001F615": "*verwirrt*",
    "\U00002764": "*liebevoll*",
    "\U0001F525": "*aufgeregt*",
    "\U0001F389": "*begeistert*",
    "\U0001F31E": "*freut sich*",
    "\U00002728": "*begeistert*",
    "\U0001F44D": "*zustimmend*",
}

# ─── Regex patterns ─────────────────────────────────────────────────────────

_RE_CODEBLOCK = re.compile(r"```[\s\S]*?```")
_RE_INLINE_CODE = re.compile(r"`[^`]+`")
_RE_URL = re.compile(r"https?://\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_HTML = re.compile(r"<[^>]+>")
_RE_IMG_LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_RE_STRIKETHROUGH = re.compile(r"~~(.*?)~~")
_RE_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_RE_LIST_BULLET = re.compile(r"^\s*[-\u2022]\s+", re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r"^\s*>\s*", re.MULTILINE)
_RE_LIST_NUM = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_RE_EMOJI = re.compile(
    r"[\U0001F600-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\U0000200D]+"
)
_RE_MULTI_DOTS = re.compile(r"\.{2,}")
_RE_STUTTER = re.compile(
    r"\b([A-Za-z\u00C4\u00D6\u00DC\u00E4\u00F6\u00FC])-\1",
    re.IGNORECASE,
)
_RE_DASH_PAUSE = re.compile(r"\s*[\u2014\u2013-]\s*(?=[A-Z\u00C4\u00D6\u00DC])")
_RE_SPECIAL_CHARS = re.compile(r"[*_#~^+=<>{}\[\]()\\/@&$%\"'\xb4`\xb0\xa7]")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_MULTI_PUNCT = re.compile(r"([.!?])\1+")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
_RE_NARRATOR = re.compile(r"\*([^*]+)\*")


# ─── Text processing helpers ───────────────────────────────────────────────


def _clean_text(raw: str) -> str:
    """Clean LLM output for TTS (markdown, code, URLs, emojis)."""
    text = raw
    for emoji_char, marker in _EMOJI_TO_EMOTION.items():
        text = text.replace(emoji_char, f" {marker} ")
    text = _RE_CODEBLOCK.sub(" ", text)
    text = _RE_INLINE_CODE.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_IMG_LINK.sub(" ", text)
    text = _RE_MD_LINK.sub(r"\1", text)
    text = re.sub(r"\*{2,}", "*", text)
    text = text.replace("__", "_")
    text = _RE_STRIKETHROUGH.sub(r"\1", text)
    text = _RE_MULTI_DOTS.sub(".", text)
    text = _RE_HEADER.sub("", text)
    text = _RE_LIST_BULLET.sub("", text)
    text = _RE_BLOCKQUOTE.sub("", text)
    text = _RE_LIST_NUM.sub("", text)
    text = text.replace("|", " ")
    if text.count("*") % 2 != 0:
        text += "*"
    return text


def _clean_segment(text: str) -> str:
    """Final cleanup of one TTS segment."""
    text = _RE_EMOJI.sub(" ", text)
    text = _RE_STUTTER.sub(r"\1", text)
    text = _RE_DASH_PAUSE.sub(", ", text)
    text = _RE_MULTI_DOTS.sub(".", text)
    text = _RE_SPECIAL_CHARS.sub(" ", text)
    text = text.replace("LEXY", "Lexy")
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_PUNCT.sub(r"\1", text)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def _detect_emotion(narrator_text: str) -> str | None:
    """Return an instruct prompt when the narrator contains an emotion keyword."""
    lower = narrator_text.lower()
    for keyword, instruct in _EMOTION_MAP.items():
        if keyword in lower:
            return instruct
    return None


def _split_narrator(
    text: str,
    default_voice: str,
    narrator_voice: str,
    narrator_mode: str = "full",
) -> list[tuple[str, str, str | None]]:
    """
    Split a cleaned text into ``(segment, voice, instruct)`` triples.

    narrator_mode:
        "full"         – speak narrator + detect emotion (default)
        "emotion_only" – skip narrator text, carry emotion to next segment
        "disabled"     – ignore narrator markers entirely
    """
    segments: list[tuple[str, str, str | None]] = []

    if narrator_mode == "disabled":
        cleaned = _clean_segment(text)
        if cleaned:
            segments.append((cleaned, default_voice, None))
        return segments

    last_end = 0
    last_instruct: str | None = None

    for match in _RE_NARRATOR.finditer(text):
        before = text[last_end:match.start()].strip()
        if before:
            cleaned = _clean_segment(before)
            if cleaned:
                segments.append((cleaned, default_voice, last_instruct))
                last_instruct = None

        narrator_raw = match.group(1).strip()
        if narrator_raw:
            instruct = _detect_emotion(narrator_raw)
            if narrator_mode == "full":
                cleaned = _clean_segment(narrator_raw)
                if cleaned:
                    segments.append((cleaned, narrator_voice, instruct))
            last_instruct = instruct

        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        cleaned = _clean_segment(remaining)
        if cleaned:
            segments.append((cleaned, default_voice, last_instruct))

    if not segments:
        cleaned = _clean_segment(text)
        if cleaned:
            segments.append((cleaned, default_voice, None))

    return segments


def _make_silence(
    sample_rate: int, channels: int, sampwidth: int, duration_ms: int
) -> bytes:
    num_frames = int(sample_rate * duration_ms / 1000)
    return b"\x00" * (num_frames * channels * sampwidth)


def _append_silence(wav_data: bytes, pause_ms: int) -> bytes:
    if pause_ms <= 0 or len(wav_data) < 44:
        return wav_data
    try:
        with wave.open(io.BytesIO(wav_data), "rb") as handle:
            params = handle.getparams()
            frames = handle.readframes(handle.getnframes())
    except Exception:  # noqa: BLE001
        return wav_data

    silence = _make_silence(
        params.framerate, params.nchannels, params.sampwidth, pause_ms
    )
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setparams(params)
        handle.writeframes(frames + silence)
    return out.getvalue()


def _concat_wav(chunks: list[bytes], pause_ms: int = 0) -> bytes:
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]

    all_frames = b""
    params = None
    silence = b""

    for chunk in chunks:
        if len(chunk) < 44:
            continue
        try:
            with wave.open(io.BytesIO(chunk), "rb") as handle:
                if params is None:
                    params = handle.getparams()
                    if pause_ms > 0:
                        silence = _make_silence(
                            params.framerate,
                            params.nchannels,
                            params.sampwidth,
                            pause_ms,
                        )
                if all_frames and silence:
                    all_frames += silence
                all_frames += handle.readframes(handle.getnframes())
        except Exception:  # noqa: BLE001
            continue

    if params is None or not all_frames:
        return chunks[0]

    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setparams(params)
        handle.writeframes(all_frames)
    return out.getvalue()


# ─── Provider ───────────────────────────────────────────────────────────────


class CosyVoiceTTS(TTSProvider):
    """TTS provider that talks to a remote CosyVoice 3 REST server."""

    name = "voice_cosyvoice"

    def __init__(
        self,
        server_url: str = "http://172.20.0.245:5500",
        voice: str = "referenz_mio",
        narrator_voice: str = "referenz_mio",
        speed: float = 1.0,
        timeout: float = 30.0,
        retries: int = 2,
        streaming: bool = True,
        default_instruct: str = "",
        narrator_mode: str = "full",
        segment_pause_ms: int = 80,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._voice = voice
        self._narrator_voice = narrator_voice
        self._speed = speed
        self._timeout = timeout
        self._retries = retries
        self._streaming = streaming
        self._default_instruct = default_instruct
        self._narrator_mode = narrator_mode
        self._segment_pause_ms = segment_pause_ms
        self._client: httpx.AsyncClient | None = None
        self._available: bool = False
        self._instruct_supported: bool = False
        self._voices: list[str] = []

    # ─── Lifecycle ──────────────────────────────────────────────

    async def initialize(self) -> bool:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=5.0)
        )
        try:
            resp = await self._client.get(f"{self._server_url}/")
        except httpx.ConnectError:
            log.warning("cosyvoice.unreachable", url=self._server_url)
            return False
        except Exception as exc:  # noqa: BLE001
            log.error("cosyvoice.connect_error", error=str(exc))
            return False

        if resp.status_code != 200:
            log.warning("cosyvoice.bad_status", status=resp.status_code)
            return False

        data = resp.json()
        if data.get("status") != "ok":
            return False

        self._available = True
        self._instruct_supported = bool(data.get("instruct_supported", False))
        self._voices = list(data.get("voices", []))
        log.info(
            "cosyvoice.ready",
            model=data.get("model", "unknown"),
            voice=self._voice,
            narrator_voice=self._narrator_voice,
            speed=self._speed,
            voices=self._voices,
            instruct_supported=self._instruct_supported,
        )
        return True

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._available = False
        log.info("cosyvoice.shutdown")

    # ─── Synthesis ──────────────────────────────────────────────

    @property
    def supports_streaming(self) -> bool:
        return self._streaming

    def _prepare_segments(
        self, text: str, voice_override: str | None = None
    ) -> list[tuple[str, str, str | None]]:
        """Split ``text`` into (segment, voice, instruct) tuples.

        If ``voice_override`` is supplied (e.g. a per-character voice
        from an RP card), it replaces the default speaker voice — but
        the narrator voice still handles narrator blocks. This keeps
        the Narrator+Emotion pipeline intact while letting characters
        speak in their own voice.
        """
        return _split_narrator(
            _clean_text(text),
            voice_override or self._voice,
            self._narrator_voice,
            self._narrator_mode,
        )

    async def _tts_request(
        self, text: str, voice: str, instruct: str | None = None
    ) -> bytes:
        assert self._client is not None
        effective_instruct = ""
        if instruct:
            effective_instruct = instruct
        if self._default_instruct:
            effective_instruct = f"{effective_instruct} {self._default_instruct}".strip()

        payload: dict[str, object] = {
            "text": text,
            "voice": voice,
            "speed": self._speed,
        }
        if effective_instruct and self._instruct_supported:
            payload["instruct"] = effective_instruct

        for attempt in range(1 + self._retries):
            try:
                resp = await self._client.post(
                    f"{self._server_url}/tts", json=payload
                )
                resp.raise_for_status()
                wav_data = resp.content
                if self._segment_pause_ms > 0 and wav_data:
                    wav_data = _append_silence(wav_data, self._segment_pause_ms)
                return wav_data
            except httpx.TimeoutException:
                if attempt < self._retries:
                    log.warning(
                        "cosyvoice.timeout",
                        attempt=attempt + 1,
                        total=1 + self._retries,
                    )
                    await asyncio.sleep(0.5)
                else:
                    log.error("cosyvoice.timeout_final", text=text[:80])
            except httpx.HTTPStatusError as exc:
                log.error(
                    "cosyvoice.http_error",
                    status=exc.response.status_code,
                    text=text[:80],
                )
                break
            except Exception as exc:  # noqa: BLE001
                log.error("cosyvoice.error", error=str(exc))
                break
        return b""

    async def synthesize(
        self, text: str, voice: str | None = None
    ) -> bytes:
        if self._client is None or not self._available:
            return b""

        # Warn (once per unknown voice) if the caller asked for a voice
        # the server doesn't know about — we still pass it through in case
        # the voices list is stale, but the user should see the hint.
        if voice and self._voices and voice not in self._voices:
            log.debug(
                "cosyvoice.voice_not_in_list",
                requested=voice,
                known=self._voices,
            )

        segments = self._prepare_segments(text, voice_override=voice)
        if not segments:
            return b""
        if len(segments) == 1:
            seg_text, seg_voice, instruct = segments[0]
            return await self._tts_request(seg_text, seg_voice, instruct)

        chunks: list[bytes] = []
        for seg_text, seg_voice, instruct in segments:
            audio = await self._tts_request(seg_text, seg_voice, instruct)
            if audio:
                chunks.append(audio)
        return _concat_wav(chunks, self._segment_pause_ms)

    async def synthesize_streaming(
        self, text: str, voice: str | None = None
    ) -> AsyncGenerator[bytes, None]:
        if self._client is None or not self._available:
            return
        for seg_text, seg_voice, instruct in self._prepare_segments(
            text, voice_override=voice
        ):
            audio = await self._tts_request(seg_text, seg_voice, instruct)
            if audio:
                yield audio

    # ─── Config ─────────────────────────────────────────────────

    def get_config(self) -> dict[str, object]:
        return {
            "speed": self._speed,
            "default_instruct": self._default_instruct,
            "narrator_mode": self._narrator_mode,
            "segment_pause_ms": self._segment_pause_ms,
            "voice": self._voice,
            "narrator_voice": self._narrator_voice,
            "streaming": self._streaming,
            "instruct_supported": self._instruct_supported,
            "voices": self._voices,
        }

    def update_config(self, data: dict[str, object]) -> None:
        if "speed" in data:
            self._speed = max(0.5, min(2.0, float(data["speed"])))  # type: ignore[arg-type]
        if "default_instruct" in data:
            self._default_instruct = str(data["default_instruct"])
        if "narrator_mode" in data and data["narrator_mode"] in (
            "full",
            "emotion_only",
            "disabled",
        ):
            self._narrator_mode = str(data["narrator_mode"])
        if "segment_pause_ms" in data:
            self._segment_pause_ms = max(0, min(500, int(data["segment_pause_ms"])))  # type: ignore[arg-type]
