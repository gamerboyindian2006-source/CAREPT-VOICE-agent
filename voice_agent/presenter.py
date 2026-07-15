"""The voice presenter: takes text from any brain model and speaks it
smoothly, confidently, and compassionately.

What makes the delivery feel human:
- markdown/formatting is stripped so nothing reads as "asterisk asterisk"
- text is split into sentences and synthesized in a pipeline, so speech
  starts almost immediately even for long passages
- short breathing pauses are inserted between sentences and paragraphs
- a warm voice at a slightly unhurried pace (see config.py)
"""

import queue
import re
import threading

import numpy as np

from . import audio, config
from .tts import TextToSpeech

_DONE = object()

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

# Formatting that should never be spoken aloud.
_MARKDOWN_PATTERNS = [
    (re.compile(r"```.*?```", re.DOTALL), " "),          # code blocks
    (re.compile(r"`([^`]*)`"), r"\1"),                    # inline code
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),           # images
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),        # links -> label
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),        # headers
    (re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}"), r"\1"),   # bold/italic
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),      # bullet markers
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),      # numbered lists
    (re.compile(r"[|#>~]"), " "),                          # leftover syntax
]


def clean_for_speech(text: str) -> str:
    """Strip formatting so the text sounds like something a person would say."""
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into speakable chunks, marking paragraph breaks."""
    chunks = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.replace("\n", " ").strip()
        if not paragraph:
            continue
        for sentence in SENTENCE_END.split(paragraph):
            if sentence.strip():
                chunks.append((sentence.strip(), False))
        if chunks:
            chunks[-1] = (chunks[-1][0], True)  # pause longer after paragraph
    return chunks


class Presenter:
    """Pipelines TTS: synthesizes upcoming sentences while the current one plays."""

    def __init__(self, tts: TextToSpeech | None = None):
        self.tts = tts or TextToSpeech()
        self._lock = threading.Lock()

    def _render(self, text: str) -> list[tuple[np.ndarray, int]]:
        clips = []
        for sentence, paragraph_end in split_sentences(clean_for_speech(text)):
            samples, rate = self.tts.synthesize(sentence)
            pause = config.PARAGRAPH_PAUSE_SEC if paragraph_end else config.SENTENCE_PAUSE_SEC
            silence = np.zeros(int(rate * pause), dtype=samples.dtype)
            clips.append((np.concatenate([samples, silence]), rate))
        return clips

    def render_wav(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize the whole passage and return (samples, sample_rate)."""
        clips = self._render(text)
        if not clips:
            return np.zeros(0, dtype=np.float32), 24000
        rate = clips[0][1]
        return np.concatenate([c[0] for c in clips]), rate

    def speak(self, text: str):
        """Speak text aloud. Synthesis of sentence N+1 overlaps playback of N."""
        with self._lock:  # one utterance at a time — voices don't talk over themselves
            clip_queue: queue.Queue = queue.Queue(maxsize=2)

            def synth_worker():
                try:
                    for clip in self._render(text):
                        clip_queue.put(clip)
                finally:
                    clip_queue.put(_DONE)

            threading.Thread(target=synth_worker, daemon=True).start()
            while True:
                clip = clip_queue.get()
                if clip is _DONE:
                    return
                samples, rate = clip
                audio.play(samples, rate)
