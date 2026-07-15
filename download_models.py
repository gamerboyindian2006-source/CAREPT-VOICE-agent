"""Download the Kokoro TTS model files (~340 MB total) into ./models/."""

import sys
from pathlib import Path

import requests

MODELS_DIR = Path(__file__).resolve().parent / "models"

FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def download(name: str, url: str):
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"✓ {name} already downloaded")
        return
    print(f"Downloading {name} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {pct}% ({done // (1 << 20)} MB)", end="", flush=True)
        tmp.rename(dest)
        print(f"\r✓ {name} downloaded          ")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in FILES.items():
        download(name, url)
    print("\nAll model files ready.")


if __name__ == "__main__":
    sys.exit(main())
