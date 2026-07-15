"""Speaker playback helpers."""

import numpy as np
import sounddevice as sd


def play(samples: np.ndarray, sample_rate: int):
    """Play audio and block until finished."""
    sd.play(samples, sample_rate)
    sd.wait()


def stop():
    sd.stop()
