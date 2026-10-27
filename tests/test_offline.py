"""Offline tests: no microphone, speakers, network, or model downloads needed.
Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receptionist import audio_io
from receptionist.audio_io import BLOCK_SIZE, SAMPLE_RATE, AudioIO
from receptionist.config import Config, ConfigError
from receptionist.conversation import Conversation
from receptionist.llm import LLMError, LLMProvider
from receptionist.persona import build_greeting, build_system_prompt
from receptionist.tts import Synthesizer, clean_for_speech, has_devanagari


def make_cfg(**env):
    base = {"LLM_API_KEY": "k", "OWNER_NAME": "Asha", "OWNER_INFO_FILE": "nonexistent.txt"}
    base.update(env)
    with mock.patch.dict(os.environ, base, clear=True), mock.patch("receptionist.config.load_dotenv"):
        return Config.load()


def tone(ms, amp):
    t = np.arange(SAMPLE_RATE * ms // 1000) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def blocks(signal):
    n = len(signal) // BLOCK_SIZE
    return [signal[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] for i in range(n)]


class FakeLLM(LLMProvider):
    def __init__(self, fail=False):
        self.fail, self.seen = fail, []

    def generate(self, messages):
        self.seen.append(messages)
        if self.fail:
            raise LLMError("boom")
        return f"reply{len(self.seen)}"


class ConfigTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        cfg = make_cfg(WHISPER_MODEL="base", BARGE_IN="true", INPUT_DEVICE="2")
        self.assertEqual(cfg.whisper_model, "base")
        self.assertTrue(cfg.barge_in)
        self.assertEqual(cfg.input_device, 2)
        self.assertIsNone(cfg.output_device)

    def test_missing_key_rejected(self):
        cfg = make_cfg(LLM_API_KEY="your_api_key_here")
        with self.assertRaises(ConfigError):
            cfg.validate_llm()

    def test_local_llm_needs_no_key(self):
        cfg = make_cfg(LLM_API_KEY="", LLM_BASE_URL="http://localhost:11434/v1")
        cfg.validate_llm()

    def test_bad_values(self):
        with self.assertRaises(ConfigError):
            make_cfg(LLM_TEMPERATURE="hot")
        with self.assertRaises(ConfigError):
            make_cfg(WHISPER_LANGUAGE="fr")


class PersonaTests(unittest.TestCase):
    def test_prompt_and_greeting_disclose_ai(self):
        cfg = make_cfg()
        self.assertIn("AI assistant", build_greeting(cfg))
        prompt = build_system_prompt(cfg, "- Asha is a designer.")
        for needle in ("Asha is a designer", "never a human", "Never promise", "pass the message"):
            self.assertIn(needle, prompt)

    def test_gender_grammar(self):
        self.assertIn("masculine", build_system_prompt(make_cfg(ASSISTANT_GENDER="male"), "x"))


class ConversationTests(unittest.TestCase):
    def test_context_is_carried(self):
        llm, c = FakeLLM(), Conversation("SYS", 12)
        c.respond("hello", llm)
        c.respond("again", llm)
        roles = [m["role"] for m in llm.seen[1]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_trimming(self):
        llm, c = FakeLLM(), Conversation("SYS", 2)
        for i in range(6):
            c.respond(f"m{i}", llm)
        self.assertEqual(c.turns, 2)
        self.assertEqual(llm.seen[-1][0]["content"], "SYS")

    def test_failure_leaves_history_untouched(self):
        c = Conversation("SYS", 5)
        with self.assertRaises(LLMError):
            c.respond("hi", FakeLLM(fail=True))
        self.assertEqual(c.turns, 0)


class TextHelperTests(unittest.TestCase):
    def test_clean_for_speech(self):
        self.assertEqual(clean_for_speech("**Hello** there!  \n- item `x`"), "Hello there! item x")

    def test_voice_selection(self):
        s = Synthesizer(make_cfg())
        self.assertEqual(s.pick_voice("Hello"), "en-IN-NeerjaNeural")
        self.assertEqual(s.pick_voice("आपका message मैं पहुँचा दूँगी"), "hi-IN-SwaraNeural")
        self.assertTrue(has_devanagari("नमस्ते"))


class ListenTests(unittest.TestCase):
    """Feed synthetic audio through the real end-of-speech detector."""

    def setUp(self):
        self.audio = AudioIO(make_cfg(VAD_THRESHOLD="1000", SILENCE_END_MS="600", MIN_SPEECH_MS="300"))
        self.audio.threshold = 1000
        # listen() normally discards stale mic audio on entry; keep our pre-fed test audio.
        patcher = mock.patch.object(self.audio, "_flush")
        patcher.start()
        self.addCleanup(patcher.stop)

    def feed(self, signal):
        for b in blocks(signal):
            self.audio._q.put(b)

    def test_detects_utterance_after_silence(self):
        self.feed(np.concatenate([tone(500, 50), tone(1000, 8000), tone(900, 50)]))
        pcm = self.audio.listen()
        self.assertGreater(len(pcm), SAMPLE_RATE * 1.0)       # speech + preroll
        self.assertLess(len(pcm), SAMPLE_RATE * 2.2)           # stopped soon after silence

    def test_short_blip_is_ignored_then_real_speech_found(self):
        self.feed(np.concatenate([tone(100, 8000), tone(900, 50), tone(900, 8000), tone(900, 50)]))
        pcm = self.audio.listen()
        self.assertGreater(len(pcm), SAMPLE_RATE * 0.8)
        self.assertGreater(float(np.abs(pcm).max()), 7000)

    def test_quiet_noise_never_triggers(self):
        self.feed(tone(2000, 100))
        with mock.patch.object(audio_io, "poll_key", side_effect=[None] * 40 + ["q"]):
            with self.assertRaises(audio_io.QuitRequested):
                self.audio.listen()

    def test_q_quits(self):
        with mock.patch.object(audio_io, "poll_key", return_value="q"):
            with self.assertRaises(audio_io.QuitRequested):
                self.audio.listen()

    def test_no_audio_at_all_is_an_error(self):
        with mock.patch.object(audio_io, "poll_key", return_value=None):
            with mock.patch.object(self.audio, "next_block", return_value=None):
                with self.assertRaises(audio_io.AudioError):
                    self.audio.listen()


class DeviceErrorTests(unittest.TestCase):
    def test_mic_open_failure_gives_friendly_help(self):
        class FakeSD:
            class PortAudioError(Exception):
                pass

            @staticmethod
            def query_devices(*a, **k):
                return {}

            @staticmethod
            def InputStream(*a, **k):
                raise FakeSD.PortAudioError("Error opening InputStream: Invalid device [PaErrorCode -9996]")

        with mock.patch.object(audio_io, "_sd", return_value=FakeSD):
            with self.assertRaises(audio_io.AudioError) as ctx:
                AudioIO(make_cfg()).open()
        msg = str(ctx.exception)
        self.assertIn("Microphone access", msg)
        self.assertIn("--list-devices", msg)

    def test_no_default_input_device(self):
        class FakeSD:
            @staticmethod
            def query_devices(*a, **k):
                raise ValueError("No input device matching")

        with mock.patch.object(audio_io, "_sd", return_value=FakeSD):
            with self.assertRaises(audio_io.AudioError):
                AudioIO(make_cfg()).open()

    def test_missing_audio_library_is_friendly(self):
        # In this Linux sandbox PortAudio really is missing, so this is a genuine failure path.
        with mock.patch.dict(sys.modules, {"sounddevice": None}):
            with self.assertRaises(audio_io.AudioError):
                audio_io._sd()


class MainLoopTests(unittest.TestCase):
    """Whole loop with fake mic / STT / LLM / TTS / speaker."""

    def run_loop(self, transcripts, llm, tts_fails=False):
        from receptionist import main as m
        from receptionist.stt import Transcript
        from receptionist.tts import TTSError

        spoken, shown = [], []

        class FakeAudio:
            noise_floor, threshold = 10.0, 200.0
            def __init__(self, cfg): self.n = 0
            def open(self, calibrate=True): pass
            def close(self): shown.append("closed")
            def listen(self, preroll=None):
                if self.n >= len(transcripts):
                    raise audio_io.QuitRequested
                self.n += 1
                return np.zeros(10, dtype=np.int16)
            def play(self, samples, sr):
                return audio_io.PlayResult(False)

        class FakeSTT:
            def __init__(self, cfg): self.i = 0
            def transcribe(self, pcm):
                t = transcripts[self.i]; self.i += 1
                return Transcript(t, "en")

        class FakeTTS:
            def __init__(self, cfg): pass
            def synthesize(self, text):
                spoken.append(text)
                if tts_fails:
                    raise TTSError("offline")
                return np.zeros(10, dtype=np.int16), 24000

        cfg = make_cfg()
        with mock.patch.object(m.Config, "load", return_value=cfg), \
             mock.patch.object(m, "AudioIO", FakeAudio), mock.patch.object(m, "Transcriber", FakeSTT), \
             mock.patch.object(m, "Synthesizer", FakeTTS), mock.patch.object(m, "create_llm", return_value=llm):
            rc = m.run()
        return rc, spoken, shown

    def test_multi_turn_with_empty_speech(self):
        llm = FakeLLM()
        rc, spoken, shown = self.run_loop(["hello", "", "नमस्ते, message देना है"], llm)
        self.assertEqual(rc, 0)
        self.assertIn("AI assistant", spoken[0])           # greeting discloses AI
        self.assertEqual(spoken[1:], ["reply1", "reply2"])  # empty speech skipped, 2 LLM turns
        self.assertEqual(len(llm.seen[1]), 4)              # 2nd call includes the 1st exchange
        self.assertIn("closed", shown)                     # mic released on exit

    def test_llm_failure_speaks_fallback_and_continues(self):
        rc, spoken, _ = self.run_loop(["hi"], FakeLLM(fail=True))
        self.assertEqual(rc, 0)
        self.assertIn("technical problem", spoken[-1])

    def test_tts_failure_does_not_crash(self):
        rc, _, _ = self.run_loop(["hi"], FakeLLM(), tts_fails=True)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
