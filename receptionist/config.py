"""Settings loaded from .env / environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_KEYS = {"your_api_key_here", "changeme", ""}


class ConfigError(Exception):
    """Raised for missing or invalid configuration."""


def _get(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _float(name: str, default: float) -> float:
    raw = _get(name, str(default))
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None


def _int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from None


def _bool(name: str, default: bool) -> bool:
    return _get(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


def _device(name: str):
    """Audio device: empty -> None (default), digits -> index, else name substring."""
    raw = _get(name)
    if not raw:
        return None
    return int(raw) if raw.isdigit() else raw


@dataclass(frozen=True)
class Config:
    owner_name: str
    assistant_name: str
    assistant_gender: str
    greeting: str
    owner_info_file: Path

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout: float
    max_history_turns: int

    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str

    tts_voice_en: str
    tts_voice_hi: str
    tts_rate: str
    tts_fallback_offline: bool

    input_device: object
    output_device: object

    vad_threshold: float
    vad_min_rms: float
    vad_noise_mult: float
    silence_end_ms: int
    min_speech_ms: int
    max_utterance_s: int

    barge_in: bool

    @property
    def llm_is_local(self) -> bool:
        url = self.llm_base_url.lower()
        return "localhost" in url or "127.0.0.1" in url

    @classmethod
    def load(cls) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")

        info_path = Path(_get("OWNER_INFO_FILE", "owner_info.txt"))
        if not info_path.is_absolute():
            info_path = PROJECT_ROOT / info_path

        language = _get("WHISPER_LANGUAGE", "auto").lower()
        if language not in {"auto", "hi", "en"}:
            raise ConfigError("WHISPER_LANGUAGE must be auto, hi or en")
        gender = _get("ASSISTANT_GENDER", "female").lower()
        if gender not in {"female", "male"}:
            raise ConfigError("ASSISTANT_GENDER must be female or male")

        cfg = cls(
            owner_name=_get("OWNER_NAME", "the owner"),
            assistant_name=_get("ASSISTANT_NAME", "Aria"),
            assistant_gender=gender,
            greeting=_get("GREETING"),
            owner_info_file=info_path,
            llm_base_url=_get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            llm_api_key=_get("LLM_API_KEY"),
            llm_model=_get("LLM_MODEL", "llama-3.3-70b-versatile"),
            llm_temperature=_float("LLM_TEMPERATURE", 0.4),
            llm_max_tokens=_int("LLM_MAX_TOKENS", 150),
            llm_timeout=_float("LLM_TIMEOUT_SECONDS", 20),
            max_history_turns=max(1, _int("MAX_HISTORY_TURNS", 12)),
            whisper_model=_get("WHISPER_MODEL", "small"),
            whisper_device=_get("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=_get("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_language=language,
            tts_voice_en=_get("TTS_VOICE_EN", "en-IN-NeerjaNeural"),
            tts_voice_hi=_get("TTS_VOICE_HI", "hi-IN-SwaraNeural"),
            tts_rate=_get("TTS_RATE", "+0%"),
            tts_fallback_offline=_bool("TTS_FALLBACK_OFFLINE", True),
            input_device=_device("INPUT_DEVICE"),
            output_device=_device("OUTPUT_DEVICE"),
            vad_threshold=_float("VAD_THRESHOLD", 0),
            vad_min_rms=_float("VAD_MIN_RMS", 200),
            vad_noise_mult=_float("VAD_NOISE_MULT", 3.5),
            silence_end_ms=_int("SILENCE_END_MS", 900),
            min_speech_ms=_int("MIN_SPEECH_MS", 300),
            max_utterance_s=_int("MAX_UTTERANCE_SECONDS", 30),
            barge_in=_bool("BARGE_IN", False),
        )
        return cfg

    def validate_llm(self) -> None:
        """Checked separately so --test-mic / --list-devices work without an API key."""
        if not self.llm_is_local and self.llm_api_key in PLACEHOLDER_KEYS:
            raise ConfigError(
                "LLM_API_KEY is missing. Copy .env.example to .env and put your key in it."
            )
