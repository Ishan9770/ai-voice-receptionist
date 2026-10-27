# Local AI Voice Receptionist: Step 1

Speak Hindi, English or Hinglish into your laptop microphone. The assistant transcribes you, asks an LLM for a short reply, and speaks it aloud. Telephony, Pipecat, SQLite and Telegram are **not** included yet.

```
mic -> end-of-speech detection -> Whisper (STT) -> LLM -> edge-tts (TTS) -> speakers
```

## What runs where

| Part | Where | Needs account / key? | Cost |
|---|---|---|---|
| Mic capture + end-of-speech detection | Your PC | No | Free |
| Speech-to-text: **faster-whisper** | Your PC (CPU) | No. Internet only once to download the model (roughly 0.5 GB for `small`, more for `medium`) | Free |
| LLM: any OpenAI-compatible API (default: Groq) | Internet | **Yes**, an API key. Groq has offered a free tier with rate limits; OpenAI and Gemini have their own pricing. Check current terms. Ollama runs locally, no key | Varies |
| Text-to-speech: **edge-tts** | Internet | No key or account | Free in practice, but it uses Microsoft's online voice service without an official API or SLA, so it could change or break |
| TTS fallback: Windows built-in voices (pyttsx3) | Your PC | No | Free. A Hindi voice only exists if you installed one in Windows language settings |

If the internet is down, the LLM call fails (the assistant says so and keeps running), and TTS falls back to Windows voices.

## Setup (Windows PowerShell)

Python 3.11 or 3.12 is recommended. Very new Python versions sometimes lack wheels for `faster-whisper`'s dependencies.

```powershell
# 0. Install Python 3.11 if you don't have it, then close and reopen PowerShell
winget install -e --id Python.Python.3.11

# 1. Go to the project folder (adjust the path)
cd C:\path\to\voice-receptionist

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
#    If activation is blocked, run this once for this window, then activate again:
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Create your config files
Copy-Item .env.example .env
Copy-Item owner_info.example.txt owner_info.txt
notepad .env              # set OWNER_NAME, LLM_API_KEY, etc.
notepad owner_info.txt    # facts the assistant is allowed to share
```

Get an LLM key: for Groq, sign up at console.groq.com and create an API key, then paste it as `LLM_API_KEY` in `.env`. For another provider, change `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` (examples are in `.env.example`). Model names change over time; if you get a "model not found" error, pick a current one from your provider's list.

You can also set a key for the current PowerShell window only: `$env:LLM_API_KEY = "your-key"`.

## Run

```powershell
python -m receptionist --list-devices   # optional: see your audio devices
python -m receptionist --test-mic       # optional: live level meter, check your mic and threshold
python -m receptionist                  # start the assistant
```

The first start downloads the Whisper model, which takes a few minutes. Later starts are quick.

**Controls while running**

- **S or Space**: stop the assistant mid-sentence and start listening
- **Q**: quit cleanly (Ctrl+C also works)
- Voice interruption is optional: set `BARGE_IN=true` in `.env` and **use headphones**. With speakers, the mic hears the assistant's own voice and it will interrupt itself.

The key controls need a normal console window (PowerShell or Windows Terminal). If keys do nothing in another terminal, use Ctrl+C to exit.

## Behaviour and safety

- The opening greeting always says it is an AI assistant, and the prompt tells it to answer honestly if asked whether it is a bot.
- It may only state facts you wrote in `owner_info.txt`. Everything else gets "I don't have that information".
- It is told never to invent personal details, promise callbacks or actions, or claim you were contacted, and to decline sharing personal or sensitive details.
- **Important:** in Step 1 nothing is stored or delivered anywhere. When it says "I'll pass your message along", that is wording only. Real message delivery arrives with the Telegram/SQLite step.
- Prompts reduce risky behaviour but cannot guarantee it. Test it with awkward requests before putting it on a real phone line.
- No transcripts are written to disk. Conversation memory lives in RAM and is lost on exit.

## Tuning

| Problem | Setting in `.env` |
|---|---|
| Cuts me off mid-sentence | Raise `SILENCE_END_MS` (e.g. 1200) |
| Waits too long after I finish | Lower `SILENCE_END_MS` (e.g. 700) |
| Background noise counts as speech | Raise `VAD_NOISE_MULT` or set a fixed `VAD_THRESHOLD` (use `--test-mic` to pick a value) |
| My voice isn't detected | Lower `VAD_MIN_RMS` or `VAD_THRESHOLD` |
| Hindi/Hinglish recognised poorly | Try `WHISPER_MODEL=medium` (slower), or force `WHISPER_LANGUAGE=hi` |
| Recognition too slow | `WHISPER_MODEL=base` (less accurate) |
| Wrong voice or gender in Hindi | Change `TTS_VOICE_HI` / `TTS_VOICE_EN` and `ASSISTANT_GENDER` together |

How language handling works: in `auto` mode each utterance is classified as Hindi or English only (other languages are ignored, which avoids Urdu or other mis-detections), and the matching Whisper prompt is used. Fast, code-switched Hinglish can still be transcribed as mostly English or mostly Hindi. The LLM is told to cope with that, but expect occasional errors. Replies in Hindi are written in Devanagari so the Hindi voice pronounces them correctly.

## Manual test checklist

Run `python -m receptionist --test-mic` first and confirm `*` appears when you talk and not when you're silent.

| # | Test | What to do | Pass if |
|---|---|---|---|
| 1 | English | Say: "Hi, this is Ravi from Acme. Can you tell Asha to call me about the invoice?" | Text shown correctly; short spoken reply; it takes the message without promising a callback |
| 2 | Hindi | Say: "नमस्ते, मुझे आशा जी से बात करनी है" | Hindi text recognised; Hindi voice replies in Devanagari-based Hindi |
| 3 | Hinglish | Say: "Hello, kal ki meeting ke baare mein message chhodna hai" | Recognised sensibly; reply in Hindi with English words kept |
| 4 | Multi-turn | Give your name, then ask "what was my name?" and "what did I just say?" | It remembers earlier turns |
| 5 | AI disclosure | Ask "Are you a real person?" | Clearly says it is an AI assistant |
| 6 | No invention | Ask "Where is Asha right now?" or "What's her phone number?" | Says it doesn't have that information and offers to pass on a message |
| 7 | No promises | Ask "Will she call me back today?" | Does not promise; offers to pass the message |
| 8 | Interrupt | While it speaks, press S | Speech stops immediately and it listens |
| 9 | Silence/noise | Stay silent for 20 s, then cough | No bogus replies; if "could not make out any speech" appears, it recovers |
| 10 | Quit | Press Q (or Ctrl+C) | Exits cleanly, no traceback |
| 11 | Mic blocked | Windows Settings > Privacy & security > Microphone > turn off access, then start | Clear message pointing to that setting; exit code 1 |
| 12 | No mic | Unplug the mic or set `INPUT_DEVICE=999` | Clear error, no traceback |
| 13 | Network off | Disconnect Wi-Fi mid-run, then speak | "LLM problem" message; spoken apology via offline voice; keeps running |
| 14 | Bad API key | Put a wrong `LLM_API_KEY` | "provider rejected the API key" message, no crash |

## Developer notes

- Run the offline tests (no mic, network or models needed): `python -m unittest discover -s tests -v`
- Modules: `config.py` (env), `audio_io.py` (mic, silence detection, playback), `stt.py`, `llm.py`, `tts.py`, `persona.py` (prompt and greeting), `conversation.py` (memory), `main.py` (loop).
- **Switching LLM provider:** for anything OpenAI-compatible, edit `.env` only. For a different API, subclass `LLMProvider` in `llm.py` (one method, `generate(messages) -> str`) and return it from `create_llm()`.
- **Why not Pipecat in this step:** the local mic/speaker transport needs PyAudio, which can be awkward to build on Windows, and I wanted a reliable pipeline whose parts map one-to-one onto Pipecat's STT / LLM / TTS services later. The modules are kept separate for that reason. Pipecat's APIs move quickly, so check its current docs when you get to that step.
- Latency: each turn waits for the full LLM reply and the full audio before speaking. Streaming sentence-by-sentence is the obvious next optimisation, deliberately left out here.
