"""
Conversational Voice Assistant — KittenTTS + Groq

An interactive terminal voice assistant that:
- Uses a Groq LLM to generate spoken-friendly responses
- Synthesizes replies with KittenTTS (nano model, Bella voice)
- Plays audio directly in the terminal — no files saved
- Types each word letter-by-letter in sync with the audio playback
- Maintains full conversation history for multi-turn context

Prerequisites:
    pip install kittentts groq sounddevice python-dotenv

Set your Groq API key (https://console.groq.com):
    export GROQ_API_KEY=your_key_here
    # or create a .env file with: GROQ_API_KEY=your_key_here

Usage:
    python example_conversation.py
    Type your message and press Enter. Type 'q' to quit.
"""

import os
import re
import sys
import time
import warnings
import threading
import contextlib
import io
import numpy as np
import sounddevice as sd
from groq import Groq
from kittentts import KittenTTS
from kittentts.preprocess import chunk_text

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

# ── Terminal colors ────────────────────────────────────────────────────────────
RESET      = "\033[0m"
BOLD       = "\033[1m"
DIM        = "\033[2m"
CYAN       = "\033[96m"
YELLOW     = "\033[93m"
GREEN      = "\033[92m"
MAGENTA    = "\033[95m"
WHITE      = "\033[97m"
CLEAR_LINE = "\033[2K\r"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a voice assistant. Your answers will be spoken aloud by a TTS engine "
    "with a short context limit. Keep responses concise and conversational — "
    "2 to 4 sentences max unless the user explicitly asks for a detailed or long answer. "
    "No markdown, no bullet points, no headers. Plain natural speech only."
)

# ── Available Groq models ──────────────────────────────────────────────────────
MODELS = [
    ("1", "llama-3.3-70b-versatile", "Llama 3.3 70B"),
    ("2", "llama-3.1-8b-instant",    "Llama 3.1 8B  (fast)"),
    ("3", "gemma2-9b-it",            "Gemma 2 9B"),
    ("4", "mixtral-8x7b-32768",      "Mixtral 8x7B"),
]
DEFAULT_MODEL_ID   = "llama-3.1-8b-instant"
DEFAULT_MODEL_NAME = "Llama 3.1 8B  (fast)"

VOICE      = "Bella"
TTS_MODEL  = "KittenML/kitten-tts-nano-0.8"
TTS_SPEED  = 1.8
SAMPLE_RATE = 24000


def strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s*',     '',    text)
    text = re.sub(r'^\s*[\*\-]\s+', '',    text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+',  '',    text, flags=re.MULTILINE)
    text = re.sub(r'`+.*?`+',       '',    text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\n',     ' ', text)
    return text.strip()


def spinner(label: str, stop_event: threading.Event, color: str = CYAN) -> None:
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"{CLEAR_LINE}{color}{frames[i % len(frames)]} {label}{RESET}")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1
    sys.stdout.write(CLEAR_LINE)
    sys.stdout.flush()


def run_with_spinner(label: str, fn, color: str = CYAN):
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=(label, stop, color))
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join()
    return result


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set. Export it or add it to a .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)

    print(f"\n{BOLD}{CYAN}  ✦ Voice Assistant{RESET}  {DIM}powered by Groq + KittenTTS{RESET}")
    print(f"{DIM}  ─────────────────────────────────────────{RESET}")

    print(f"\n{BOLD}  Select model:{RESET}")
    for key, _, name in MODELS:
        print(f"  {DIM}[{key}]{RESET} {name}")
    print(f"  {DIM}[enter]{RESET} default ({DEFAULT_MODEL_NAME})\n")

    choice = input(f"  {CYAN}›{RESET} ").strip()
    model_id   = next((m for k, m, _ in MODELS if k == choice), DEFAULT_MODEL_ID)
    model_name = next((n for k, _, n in MODELS if k == choice), DEFAULT_MODEL_NAME)

    print(f"\n{DIM}  Loading voice model...{RESET}", end="", flush=True)
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tts = KittenTTS(TTS_MODEL)
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr

    print(f"{CLEAR_LINE}{DIM}  Voice ready — {VOICE} / nano{RESET}")
    print(f"{DIM}  ─────────────────────────────────────────{RESET}")
    print(f"{DIM}  Type your message. Press {RESET}{BOLD}q{RESET}{DIM} to quit.{RESET}\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input(f"  {BOLD}{WHITE}You › {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{DIM}  Bye.{RESET}\n")
            break

        if user_input.lower() in ("q", "quit", "exit"):
            print(f"\n{DIM}  Bye.{RESET}\n")
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        def call_llm():
            return client.chat.completions.create(model=model_id, messages=history)

        response   = run_with_spinner(f"{model_name} thinking…", call_llm, YELLOW)
        answer     = response.choices[0].message.content
        history.append({"role": "assistant", "content": answer})

        clean  = strip_markdown(answer)
        chunks = list(chunk_text(clean))

        def synthesize_all():
            result = []
            with contextlib.redirect_stdout(io.StringIO()):
                for chunk in chunks:
                    result.append(tts.generate(text=chunk, voice=VOICE, speed=TTS_SPEED))
            return result

        audio_chunks = run_with_spinner("Preparing voice…", synthesize_all, GREEN)

        # Play audio while typing letters in sync
        print(f"\n  {MAGENTA}{VOICE} › {RESET}", end="", flush=True)
        for chunk, audio in zip(chunks, audio_chunks):
            audio_mono = audio.flatten().astype(np.float32)
            duration   = len(audio_mono) / SAMPLE_RATE
            delay      = duration / max(len(chunk), 1)
            sd.play(audio_mono, samplerate=SAMPLE_RATE)
            for char in chunk:
                sys.stdout.write(f"{MAGENTA}{char}{RESET}")
                sys.stdout.flush()
                time.sleep(delay)
            sd.wait()

        print(f"\n{DIM}  ─────────────────────────────────────────{RESET}\n")


if __name__ == "__main__":
    main()
