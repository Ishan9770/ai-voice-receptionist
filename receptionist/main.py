"""Entry point: mic -> Whisper -> LLM -> TTS -> speaker, in a loop."""
from __future__ import annotations

import argparse
import sys

from .audio_io import AudioError, AudioIO, QuitRequested, list_devices, run_mic_test
from .config import Config, ConfigError
from .conversation import Conversation
from .llm import LLMError, create_llm
from .persona import build_greeting, build_system_prompt, load_owner_info
from .stt import STTError, Transcriber
from .tts import Synthesizer, TTSError

LLM_FALLBACK_REPLY = "Sorry, I'm having a technical problem right now. Could you please try again in a moment?"

BANNER = """
==============================================================
 Local AI voice receptionist  (Step 1 prototype)
--------------------------------------------------------------
 Runs on this PC : microphone, speech detection, Whisper (STT)
 Needs internet  : LLM API call, edge-tts voice (+ one-time
                   Whisper model download)
 Keys while running:
   S or Space  stop the assistant talking and listen again
   Q           quit        (Ctrl+C also works)
 Tip: use headphones to avoid the mic hearing the speaker.
==============================================================
"""


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def speak(text: str, tts: Synthesizer, audio: AudioIO):
    """Synthesize + play. Returns PlayResult or None if speech was impossible."""
    try:
        samples, sr = tts.synthesize(text)
    except TTSError as e:
        print(f"  [voice unavailable: {e}] (text shown above only)")
        return None
    print("  (speaking... press S to stop)")
    return audio.play(samples, sr)


def run() -> int:
    cfg = Config.load()
    cfg.validate_llm()

    print(BANNER)
    audio = AudioIO(cfg)
    print("Opening microphone and calibrating background noise (stay quiet for a second)...")
    audio.open()
    print(f"Microphone OK. Noise level {audio.noise_floor:.0f}, speech threshold {audio.threshold:.0f}.")

    print(f"Loading speech recognition model '{cfg.whisper_model}' (first run downloads it)...")
    stt = Transcriber(cfg)
    llm = create_llm(cfg)
    tts = Synthesizer(cfg)

    owner_info, found = load_owner_info(cfg)
    if not found:
        print(f"Note: no owner info found at {cfg.owner_info_file}; the assistant will know nothing about "
              f"{cfg.owner_name} beyond taking messages.")
    convo = Conversation(build_system_prompt(cfg, owner_info), cfg.max_history_turns)

    greeting = build_greeting(cfg)
    print(f"\nAssistant: {greeting}")
    result = speak(greeting, tts, audio)
    preroll = result.preroll if result else None

    try:
        while True:
            print("\nListening... (speak now)")
            pcm = audio.listen(preroll)
            preroll = None
            print("  Recognizing...")
            try:
                heard = stt.transcribe(pcm)
            except STTError as e:
                print(f"  [speech recognition problem: {e}]")
                continue
            if not heard.text:
                print("  (Could not make out any speech - please try again.)")
                continue
            print(f"You [{heard.language or '?'}]: {heard.text}")

            try:
                reply = convo.respond(heard.text, llm)
            except LLMError as e:
                print(f"  [LLM problem: {e}]")
                reply = LLM_FALLBACK_REPLY
            print(f"Assistant: {reply}")
            result = speak(reply, tts, audio)
            preroll = result.preroll if result else None
    except QuitRequested:
        print("\nQ pressed - goodbye.")
    except KeyboardInterrupt:
        print("\nCtrl+C - goodbye.")
    finally:
        audio.close()
    return 0


def main() -> None:
    _setup_console()
    parser = argparse.ArgumentParser(prog="receptionist", description="Local AI voice receptionist")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    parser.add_argument("--test-mic", action="store_true", help="show a live mic level meter and exit")
    args = parser.parse_args()
    try:
        if args.list_devices:
            print(list_devices())
            return
        if args.test_mic:
            run_mic_test(Config.load())
            return
        sys.exit(run())
    except (ConfigError, AudioError, STTError, LLMError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
