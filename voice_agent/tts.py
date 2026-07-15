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

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
    ) -> tuple[np.ndarray, int]:
        samples, sample_rate = self.kokoro.create(
            text,
            voice=voice or config.TTS_VOICE,
            speed=speed or config.TTS_SPEED,
            lang=lang or "en-us",
        )
        return samples, sample_rate
