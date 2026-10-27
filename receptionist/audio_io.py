"""Microphone capture with end-of-speech detection, and interruptible playback.

One microphone stream stays open for the whole session. Blocks go into a queue.
Speech detection is a simple energy (RMS) gate calibrated to background noise.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config

SAMPLE_RATE = 16000          # what Whisper expects
BLOCK_MS = 30
BLOCK_SIZE = SAMPLE_RATE * BLOCK_MS // 1000
PREROLL_BLOCKS = 10          # ~300 ms kept from before speech starts
START_BLOCKS = 3             # ~90 ms of loud audio to count as speech starting
BARGE_IN_BLOCKS = 8          # ~240 ms of loud audio while speaking = interruption

try:  # Windows-only keyboard polling; harmless elsewhere
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


class AudioError(Exception):
    """Friendly, user-displayable audio failure."""


class QuitRequested(Exception):
    """User pressed Q."""


def poll_key() -> str | None:
    """Non-blocking single key press (lowercase), or None."""
    if msvcrt is None or not msvcrt.kbhit():
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):  # arrow/function key prefix
        msvcrt.getwch()
        return None
    return ch.lower()


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2))) if block.size else 0.0


def _sd():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as e:
        raise AudioError(
            "Could not load the 'sounddevice' audio library "
            f"({e}). Run: pip install -r requirements.txt"
        ) from None


MIC_HELP = (
    "Could not use the microphone.\n"
    "  - Windows: Settings > Privacy & security > Microphone: turn on 'Microphone access' "
    "and 'Let desktop apps access your microphone'.\n"
    "  - Check that a microphone is plugged in and set as the default input device.\n"
    "  - Close other apps that may be using it (Zoom, Teams, etc.).\n"
    "  - Run: python -m receptionist --list-devices, then set INPUT_DEVICE in .env."
)


def list_devices() -> str:
    sd = _sd()
    try:
        return str(sd.query_devices())
    except Exception as e:
        raise AudioError(f"Could not list audio devices: {e}") from None


@dataclass
class PlayResult:
    interrupted: bool
    preroll: list = field(default_factory=list)  # mic blocks captured during a voice barge-in


class AudioIO:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._q: queue.Queue = queue.Queue()
        self._stream = None
        self.threshold = cfg.vad_threshold if cfg.vad_threshold > 0 else cfg.vad_min_rms
        self.noise_floor = 0.0

    # ---------- microphone ----------
    def open(self, calibrate: bool = True) -> None:
        sd = _sd()
        try:
            sd.query_devices(self.cfg.input_device, "input")  # raises if none / invalid
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=BLOCK_SIZE,
                device=self.cfg.input_device, callback=self._on_audio,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise AudioError(f"{MIC_HELP}\n  (details: {e})") from None
        if calibrate:
            self._calibrate()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _on_audio(self, indata, frames, time_info, status) -> None:
        self._q.put(indata[:, 0].copy())

    def _flush(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _calibrate(self, blocks: int = 20) -> None:
        self._flush()
        levels = []
        try:
            for _ in range(blocks):
                levels.append(rms(self._q.get(timeout=2.0)))
        except queue.Empty:
            raise AudioError(
                "The microphone opened but no audio is arriving. Check that it is not muted "
                "and is the right INPUT_DEVICE."
            ) from None
        self.noise_floor = float(np.median(levels))
        if self.cfg.vad_threshold <= 0:
            self.threshold = max(self.cfg.vad_min_rms, self.noise_floor * self.cfg.vad_noise_mult)

    def next_block(self, timeout: float = 0.1) -> np.ndarray | None:
        """One mic block, or None on timeout. Raises AudioError if the stream died."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            if self._stream is not None and not self._stream.active:
                raise AudioError("The microphone stopped (device unplugged or disabled?).") from None
            return None

    def listen(self, preroll: list | None = None) -> np.ndarray:
        """Block until one full utterance is captured. Returns int16 mono samples.

        Raises QuitRequested on Q, AudioError if the mic fails.
        """
        cfg = self.cfg
        if not preroll:
            self._flush()
        end_blocks = max(1, cfg.silence_end_ms // BLOCK_MS)
        max_blocks = cfg.max_utterance_s * 1000 // BLOCK_MS
        min_voiced = max(1, cfg.min_speech_ms // BLOCK_MS)

        ring: deque = deque(preroll or [], maxlen=PREROLL_BLOCKS + (len(preroll) if preroll else 0))
        frames: list = []
        speaking = False
        loud_run = silence = voiced = idle_timeouts = 0

        while True:
            if poll_key() == "q":
                raise QuitRequested
            block = self.next_block()
            if block is None:
                idle_timeouts += 1
                if idle_timeouts > 50:  # ~5 s with no audio at all
                    raise AudioError("No audio is arriving from the microphone.")
                continue
            idle_timeouts = 0
            loud = rms(block) > self.threshold

            if not speaking:
                ring.append(block)
                loud_run = loud_run + 1 if loud else 0
                if loud_run >= START_BLOCKS:
                    speaking, frames = True, list(ring)
                    silence, voiced = 0, loud_run
            else:
                frames.append(block)
                if loud:
                    silence, voiced = 0, voiced + 1
                else:
                    silence += 1
                if silence >= end_blocks or len(frames) >= max_blocks:
                    if voiced >= min_voiced:
                        return np.concatenate(frames)
                    speaking, frames, loud_run = False, [], 0  # too short: noise blip

    # ---------- speaker ----------
    def play(self, samples: np.ndarray, sample_rate: int) -> PlayResult:
        """Play int16 mono audio. S/Space stops it; Q quits; optional voice barge-in."""
        sd = _sd()
        cfg = self.cfg
        samples = np.ascontiguousarray(samples, dtype=np.int16)
        pos = 0
        done = threading.Event()

        def callback(outdata, frames, time_info, status):
            nonlocal pos
            chunk = samples[pos:pos + frames]
            n = len(chunk)
            outdata[:n, 0] = chunk
            if n < frames:
                outdata[n:, 0] = 0
                raise sd.CallbackStop
            pos += n

        try:
            stream = sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="int16",
                device=cfg.output_device, callback=callback, finished_callback=done.set,
            )
        except Exception as e:
            raise AudioError(f"Could not open the speaker/output device: {e}") from None

        interrupted, barged, quit_flag = False, False, False
        recent, loud_run = deque(maxlen=BARGE_IN_BLOCKS + PREROLL_BLOCKS), 0
        self._flush()
        with stream:
            while not done.wait(0.03):
                key = poll_key()
                if key == "q":
                    interrupted = quit_flag = True
                    break
                if key in ("s", " "):
                    interrupted = True
                    break
                # drain mic blocks; only act on them if voice barge-in is enabled
                while True:
                    try:
                        b = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if cfg.barge_in:
                        recent.append(b)
                        loud_run = loud_run + 1 if rms(b) > self.threshold * 2 else 0
                if cfg.barge_in and loud_run >= BARGE_IN_BLOCKS:
                    interrupted = barged = True
                    break
            if interrupted:
                stream.abort()
        if quit_flag:
            raise QuitRequested
        if barged:
            return PlayResult(True, list(recent))
        time.sleep(0.2)  # let room echo die down, then discard anything heard meanwhile
        self._flush()
        return PlayResult(interrupted)


def run_mic_test(cfg: Config, seconds: int = 8) -> None:
    """Show a live level meter so you can see what threshold suits your mic."""
    audio = AudioIO(cfg)
    audio.open()
    print(f"Noise floor: {audio.noise_floor:.0f}   Speech threshold: {audio.threshold:.0f}")
    print(f"Speak normally for {seconds} seconds. '#' bars = level; '*' = counts as speech.\n")
    end = time.time() + seconds
    try:
        while time.time() < end:
            b = audio.next_block(0.2)
            if b is None:
                continue
            level = rms(b)
            bar = "#" * min(60, int(level / 100))
            print(f"{level:7.0f} {'*' if level > audio.threshold else ' '} {bar}")
    finally:
        audio.close()
    print("\nIf '*' never appears while you talk, set a lower VAD_THRESHOLD in .env.")
    print("If it appears while you are silent, set a higher one.")
