# KittenTTS-RS

Ultra-lightweight text-to-speech inference in Rust using ONNX Runtime and eSpeak-NG.

This is a Rust port of the KittenTTS project.

## Features

- **Fast & Efficient**: Low-latency synthesis via ONNX Runtime (`ort`).
- **High Quality**: Accurate pronunciation using IPA phonemes via eSpeak-NG.
- **Self-Contained**: eSpeak-NG is built-in as a submodule—no system installation required.
- **Configurable**: Multiple voices, speed control, and text preprocessing.

## Prerequisites

- **Rust**: [Install Rust](https://www.rust-lang.org/tools/install)
- **ONNX Model**: A directory containing `config.json`, `.onnx` model, and `voices.npz`.

## Running Locally

```bash
# download model weights (onnx)
git clone https://huggingface.co/KittenML/kitten-tts-mini-0.8

cargo run -- \
  --text "Hello, I am KittenTTS-RS!" \
  --model-dir "../kitten-tts-mini-0.8" \
  --voice "Luna" \
  --output "output.wav"
```

### Options

- `--text`: Text to synthesize.
- `--model-dir`: Directory containing ONNX model files.
- `--voice`: Voice name (e.g., "Luna", "Leo").
- `--speed`: Speech speed (default: 1.0).
- `--output`: Output file path (default: `output.wav`).
- `--no-clean`: Skip text preprocessing.

---

*Coded with ❤️ by Antigravity.*
