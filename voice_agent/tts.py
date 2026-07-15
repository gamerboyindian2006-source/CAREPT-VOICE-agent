"""Text-to-speech using Kokoro (ONNX)."""

import numpy as np
from kokoro_onnx import Kokoro

from . import config


class TextToSpeech:
    def __init__(self):
        if not config.KOKORO_MODEL_PATH.exists() or not config.KOKORO_VOICES_PATH.exists():
            raise FileNotFoundError(
                "Kokoro model files not found. Run: python download_models.py"
            )
        self.kokoro = Kokoro(str(config.KOKORO_MODEL_PATH), str(config.KOKORO_VOICES_PATH))

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        samples, sample_rate = self.kokoro.create(
            text, voice=config.TTS_VOICE, speed=config.TTS_SPEED, lang="en-us"
        )
        return samples, sample_rate
