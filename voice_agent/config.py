"""Central configuration for the voice presenter."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# --- Text-to-speech (Kokoro) ---
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

# The delivery: warm and unhurried reads as confident + compassionate.
# Other voices worth trying: af_bella (brighter), am_michael (male, calm),
# bm_george (male, British), af_sarah (softer).
TTS_VOICE = "af_heart"
TTS_SPEED = 0.95  # slightly slower than neutral — sounds composed, not rushed

# Silence inserted between sentences (seconds). Real speakers breathe;
# a small pause makes the delivery feel human instead of machine-gun TTS.
SENTENCE_PAUSE_SEC = 0.35
PARAGRAPH_PAUSE_SEC = 0.7

# --- HTTP server ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8756
