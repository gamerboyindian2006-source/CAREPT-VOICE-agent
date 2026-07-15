"""Local HTTP voice endpoint — any brain model can POST text here to speak.

    python -m voice_agent.server

Endpoints:
    POST /speak  {"text": "..."}          -> speaks on this machine's speakers
    POST /speak  {"text": "...", "return": "wav"} -> returns audio/wav bytes
    GET  /health                          -> {"status": "ok"}

Optional /speak fields:
    "language": "en" | "es" | "fr"  -> picks the matching Kokoro voice
    "voice":    any Kokoro voice name (overrides language)
    "speed":    playback speed multiplier (default from config)
"""

import io
import json
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import config
from .presenter import Presenter

presenter: Presenter | None = None


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        # The CarePT web app (any origin: file://, localhost, or the hosted
        # demo) fetches audio from this local server, so allow cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "voice": config.TTS_VOICE})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/speak":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            text = data.get("text", "").strip()
        except (json.JSONDecodeError, ValueError):
            self._json(400, {"error": "body must be JSON with a 'text' field"})
            return
        if not text:
            self._json(400, {"error": "'text' is required"})
            return

        lang_cfg = config.LANGUAGE_VOICES.get(str(data.get("language", "en")).lower()[:2])
        lang_cfg = lang_cfg or config.LANGUAGE_VOICES["en"]
        voice_opts = {
            "voice": data.get("voice") or lang_cfg["voice"],
            "lang": lang_cfg["lang"],
        }
        try:
            if data.get("speed"):
                voice_opts["speed"] = float(data["speed"])
        except (TypeError, ValueError):
            pass

        if data.get("return") == "wav":
            samples, rate = presenter.render_wav(text, **voice_opts)
            body = to_wav_bytes(samples, rate)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            # Speak in the background so the brain isn't blocked mid-conversation.
            threading.Thread(
                target=presenter.speak, args=(text,), kwargs=voice_opts, daemon=True
            ).start()
            self._json(200, {"status": "speaking"})

    def log_message(self, fmt, *args):
        print(f"[server] {fmt % args}")


def main():
    global presenter
    print("Loading Kokoro voice model...")
    presenter = Presenter()
    addr = (config.SERVER_HOST, config.SERVER_PORT)
    print(f"Voice endpoint ready at http://{addr[0]}:{addr[1]}/speak")
    ThreadingHTTPServer(addr, Handler).serve_forever()


if __name__ == "__main__":
    main()
