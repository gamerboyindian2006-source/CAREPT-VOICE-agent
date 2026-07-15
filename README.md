# CAREPT Voice Agent

The **voice layer** for CarePT: takes text from any brain model and speaks it in a smooth, confident, compassionate human voice — fully local, no cloud APIs, no cost, private by default.

Powered by [Kokoro](https://github.com/thewh1teagle/kokoro-onnx), the best open-source neural TTS (close to ElevenLabs quality), running on-device.

## What makes it sound human

- **Warm, unhurried delivery** — a carefully chosen voice at a slightly slower pace reads as composed and caring, not robotic
- **Breathing pauses** — short silences between sentences and longer ones between paragraphs, like a real speaker
- **Formatting-aware** — markdown, bullets, links, and code from LLM output are cleaned so nothing is read as "asterisk asterisk"
- **Instant start** — long passages are synthesized sentence-by-sentence in a pipeline, so speech begins almost immediately

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download the Kokoro voice model (~340 MB, one time)
python download_models.py
```

## Use it

**From the command line:**

```bash
python -m voice_agent.speak "Hi Sarah, I looked over your results and there's nothing to worry about."
```

**Piped from any brain model:**

```bash
your_brain_process | python -m voice_agent.speak
```

**As a local HTTP endpoint** (best for connecting your own model):

```bash
python -m voice_agent.server
```

Then from anywhere:

```bash
# Speak on this machine's speakers
curl -X POST http://127.0.0.1:8756/speak \
  -H 'Content-Type: application/json' \
  -d '{"text": "Your results look good. Let me walk you through them."}'

# Or get the audio back as a WAV file instead
curl -X POST http://127.0.0.1:8756/speak \
  -d '{"text": "Hello there.", "return": "wav"}' -o hello.wav
```

**From Python:**

```python
from voice_agent.presenter import Presenter

presenter = Presenter()
presenter.speak("I'm here with you. Let's take this one step at a time.")
```

## Tuning the delivery

Everything lives in `voice_agent/config.py`:

- `TTS_VOICE` — `af_heart` (default, warm female), `af_bella`, `af_sarah`, `am_michael` (calm male), `bm_george` (British male), and more
- `TTS_SPEED` — `0.95` default; lower = more deliberate, higher = more energetic
- `SENTENCE_PAUSE_SEC` / `PARAGRAPH_PAUSE_SEC` — the breathing room between thoughts

## Project layout

```
voice_agent/
├── presenter.py  # the core: text cleanup, pacing, pipelined synthesis
├── tts.py        # Kokoro speech engine
├── audio.py      # speaker playback
├── speak.py      # CLI / stdin interface
├── server.py     # local HTTP endpoint for the brain model
└── config.py     # all knobs in one place
```
