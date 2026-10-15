"""Audio transcription via Groq's Whisper endpoint.

Reads GROQ_API_KEY from the environment. Voice notes are downsampled to
16 kHz mono FLAC (Groq's recommended input) before upload to keep them
small and compatible. Audio is sent to Groq for transcription only.
"""
import os
import subprocess
import tempfile

import requests

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"
KEY_FILE = os.path.expanduser("~/.config/groq/api_key")


def _get_api_key() -> str:
    """Return the Groq API key from GROQ_API_KEY env, or the key file fallback."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    if os.path.isfile(KEY_FILE):
        with open(KEY_FILE) as f:
            return f.read().strip()
    raise RuntimeError(
        "No Groq API key found. Set GROQ_API_KEY or write the key to "
        f"{KEY_FILE}"
    )


def _to_flac_16k_mono(input_file: str) -> str:
    """Convert any audio file to 16 kHz mono FLAC in a temp file. Returns its path."""
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Audio file not found: {input_file}")

    tmp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-i", input_file,
        "-ar", "16000",   # 16 kHz
        "-ac", "1",       # mono
        "-c:a", "flac",
        "-y", tmp.name,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, check=True)
        return tmp.name
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr}")


def transcribe_file(audio_path: str, language: str | None = None,
                    model: str = DEFAULT_MODEL) -> str:
    """Transcribe a local audio file with Groq Whisper. Returns the transcript text.

    Args:
        audio_path: Path to the source audio (e.g. a downloaded .ogg voice note).
        language: Optional ISO-639-1 hint (e.g. "es"); None lets Whisper auto-detect.
        model: Groq transcription model.
    """
    api_key = _get_api_key()

    flac_path = _to_flac_16k_mono(audio_path)
    try:
        with open(flac_path, "rb") as f:
            data = {"model": model, "response_format": "text"}
            if language:
                data["language"] = language
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(flac_path), f, "audio/flac")},
                data=data,
                timeout=120,
            )
        resp.raise_for_status()
        return resp.text.strip()
    finally:
        if os.path.exists(flac_path):
            os.unlink(flac_path)
