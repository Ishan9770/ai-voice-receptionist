"""Speech-to-text using faster-whisper (local, CPU). Handles Hindi, English, Hinglish."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import Config

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Gentle hints that bias Whisper's spelling style (Devanagari Hindi with English words kept in Latin).
INITIAL_PROMPTS = {
    "hi": "नमस्ते, कैसे हैं आप? Meeting कल हो सकती है, message दे दीजिए।",
    "en": None,
}


class STTError(Exception):
    """Friendly, user-displayable speech-recognition failure."""


@dataclass
class Transcript:
    text: str
    language: str | None


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise STTError("faster-whisper is not installed. Run: pip install -r requirements.txt") from None
        try:
            self.model = WhisperModel(
                cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type
            )
        except Exception as e:
            raise STTError(
                f"Could not load Whisper model '{cfg.whisper_model}'. The first run needs internet to "
                f"download it (then it is cached). Details: {type(e).__name__}: {e}"
            ) from None

    def _pick_language(self, audio: np.ndarray) -> str | None:
        """auto mode: choose only between Hindi and English (avoids Urdu/other mis-detections)."""
        mode = self.cfg.whisper_language
        if mode in ("hi", "en"):
            return mode
        try:
            _lang, _prob, all_probs = self.model.detect_language(audio)
            probs = dict(all_probs)
            return "hi" if probs.get("hi", 0.0) > probs.get("en", 0.0) else "en"
        except Exception:
            return None  # let Whisper decide

    def transcribe(self, pcm: np.ndarray) -> Transcript:
        audio = pcm.astype(np.float32) / 32768.0
        try:
            lang = self._pick_language(audio)
            segments, info = self.model.transcribe(
                audio,
                language=lang,
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                temperature=0.0,
                initial_prompt=INITIAL_PROMPTS.get(lang or ""),
            )
            parts = []
            for seg in segments:
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    continue  # probably noise
                parts.append(seg.text.strip())
        except Exception as e:
            raise STTError(f"Speech recognition failed: {type(e).__name__}: {e}") from None
        return Transcript(" ".join(p for p in parts if p).strip(), lang or getattr(info, "language", None))
