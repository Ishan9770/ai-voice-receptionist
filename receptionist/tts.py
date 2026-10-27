"""Text-to-speech. Primary: edge-tts (free, online, good Hindi/English-India voices).
Fallback: Windows built-in voices via pyttsx3 (offline, lower quality)."""
from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
import wave

import numpy as np

from .config import Config

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_MARKDOWN = re.compile(r"[*_#`~>|]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF]")


class TTSError(Exception):
    """Friendly, user-displayable speech-synthesis failure."""


def clean_for_speech(text: str) -> str:
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


class Synthesizer:
    SYNTH_TIMEOUT = 20

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def pick_voice(self, text: str) -> str:
        return self.cfg.tts_voice_hi if has_devanagari(text) else self.cfg.tts_voice_en

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (int16 mono samples, sample_rate)."""
        text = clean_for_speech(text)
        if not text:
            raise TTSError("Nothing to say.")
        try:
            return self._edge(text)
        except Exception as edge_err:
            if not self.cfg.tts_fallback_offline:
                raise TTSError(f"edge-tts failed ({type(edge_err).__name__}). Check your internet connection.") from None
            try:
                return self._offline(text)
            except Exception as off_err:
                raise TTSError(
                    f"Both voices failed. edge-tts: {type(edge_err).__name__}; "
                    f"offline: {type(off_err).__name__}: {off_err}"
                ) from None

    # --- edge-tts -> MP3 -> PCM ---
    def _edge(self, text: str) -> tuple[np.ndarray, int]:
        import edge_tts
        import miniaudio

        async def run() -> bytes:
            comm = edge_tts.Communicate(text, self.pick_voice(text), rate=self.cfg.tts_rate)
            buf = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf += chunk["data"]
            return bytes(buf)

        mp3 = asyncio.run(asyncio.wait_for(run(), timeout=self.SYNTH_TIMEOUT))
        if not mp3:
            raise RuntimeError("no audio returned")
        sr = 24000
        decoded = miniaudio.decode(
            mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=sr
        )
        return np.frombuffer(decoded.samples, dtype=np.int16).copy(), sr

    # --- Windows SAPI via pyttsx3 -> WAV -> PCM ---
    def _offline(self, text: str) -> tuple[np.ndarray, int]:
        import pyttsx3

        engine = pyttsx3.init()  # fresh engine per call avoids run-loop issues
        if has_devanagari(text):
            for v in engine.getProperty("voices"):
                if "hindi" in (v.name or "").lower() or "hi-in" in (v.id or "").lower():
                    engine.setProperty("voice", v.id)
                    break
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            with wave.open(path, "rb") as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
            if width != 2 or not raw:
                raise RuntimeError("unexpected WAV output")
            data = np.frombuffer(raw, dtype=np.int16)
            if ch > 1:
                data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
            return data.copy(), sr
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
