# KittenTTS Project Guide

This document provides essential information for AI coding agents working on the KittenTTS project.

## Project Overview

KittenTTS is an open-source, ultra-lightweight text-to-speech (TTS) model designed for CPU-optimized, high-quality voice synthesis without requiring a GPU. The project provides both a Python library and a web interface.

**Key Characteristics:**
- Model sizes range from 15M to 80M parameters
- ONNX-based inference for cross-platform compatibility
- Models downloaded from Hugging Face at runtime (not bundled)
- 8 distinct voices with speed control support
- Target: Real-time speech synthesis on consumer hardware

**Available Models:**
| Model | Params | Size | HuggingFace Repo |
|-------|--------|------|------------------|
| kitten-tts-mini | 80M | 80MB | KittenML/kitten-tts-mini-0.8 |
| kitten-tts-micro | 40M | 41MB | KittenML/kitten-tts-micro-0.8 |
| kitten-tts-nano | 15M | 56MB | KittenML/kitten-tts-nano-0.8-fp32 |
| kitten-tts-nano-int8 | 15M | 19MB | KittenML/kitten-tts-nano-0.8-int8 |

**Available Voices:** Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo (4 male, 4 female)

## Technology Stack

**Core Dependencies:**
- Python 3.8+ (recommended 3.12)
- `onnxruntime` - Model inference engine
- `phonemizer` + `espeak-ng` - Text-to-phoneme conversion
- `misaki[en]` - English text processing
- `spacy` - NLP processing
- `soundfile` - Audio I/O
- `huggingface_hub` - Model downloading

**WebUI Dependencies:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - Form parsing

**Build System:**
- `setuptools` with `pyproject.toml` (primary) and legacy `setup.py`
- `ruff` for linting (cache directory `.ruff_cache/` present)

## Project Structure

```
.
├── kittentts/                 # Core library package
│   ├── __init__.py           # Package exports (KittenTTS, get_model)
│   ├── __index__.py          # Legacy exports
│   ├── get_model.py          # Model download & main KittenTTS class
│   ├── onnx_model.py         # ONNX inference engine (KittenTTS_1_Onnx)
│   └── preprocess.py         # Text preprocessing pipeline
│
├── webui/                     # Web interface
│   ├── __init__.py
│   ├── server.py             # FastAPI application & endpoints
│   ├── templates/
│   │   └── index.html        # Main web interface
│   └── static/
│       ├── style.css         # UI styling
│       ├── app.js            # Frontend JavaScript
│       └── favicon.svg       # Branding icon
│
├── pyproject.toml            # Modern Python packaging config
├── setup.py                  # Legacy packaging (keep in sync)
├── requirements.txt          # Base dependencies
├── MANIFEST.in               # Package distribution includes
├── Dockerfile                # Container build
├── run_webui.py              # WebUI entry point
└── example.py                # Usage example
```

## Key Module Details

### 1. `kittentts/get_model.py`
- **KittenTTS** class: Main user-facing API
  - `__init__(model_name, cache_dir)` - Downloads model from HF if needed
  - `generate(text, voice, speed)` - Returns numpy array of audio
  - `generate_to_file(text, output_path, ...)` - Saves to WAV file
  - `available_voices` property - Lists supported voices
- **download_from_huggingface()** - Downloads config, model ONNX, and voice embeddings

### 2. `kittentts/onnx_model.py`
- **KittenTTS_1_Onnx** class: Low-level ONNX inference
  - Loads ONNX model and voice embeddings (NPZ format)
  - Uses EspeakBackend for phonemization (language: "en-us")
  - **TextCleaner** class: Maps phonemes to token IDs
  - **chunk_text()**: Splits long text at sentence/word boundaries (400 char limit)
  - Handles speed adjustments via voice-specific priors
- **StreamingTTS** class: Sentence-level streaming for real-time TTS
  - Buffers incoming text and yields audio when complete sentences are detected
  - `add_text(text)`: Add text chunk, yields audio for complete sentences
  - `flush()`: Synthesize any remaining buffered text
  - `reset()`: Clear buffer without generating audio
  - `buffered_text` property: View current buffered text

### 3. `kittentts/preprocess.py`
- **TextPreprocessor** class: Comprehensive text normalization
  - Number-to-words conversion (integers, floats, ordinals, fractions)
  - Currency expansion ($, €, £, ¥, ₹, ₩, ₿)
  - Time format expansion (3:30pm → "three thirty pm")
  - Unit expansion (km, kg, GB, °C, etc.)
  - Scientific notation, Roman numerals, phone numbers, IP addresses
  - Model name normalization (GPT-3 → "GPT 3")
  - HTML/URL/email removal, contraction expansion
  - Configurable pipeline via constructor flags

### 4. `webui/server.py`
- FastAPI application with CORS enabled
- Endpoints:
  - `GET /` - Serves HTML template
  - `GET /api/models` - Returns model metadata
  - `GET /api/voices` - Returns voice metadata
  - `POST /api/generate` - Generates speech (returns base64 WAV)
  - `GET /api/health` - Health check with loaded models
  - `POST /api/stream/start` - Start a streaming TTS session
  - `POST /api/stream/chunk` - Add text to streaming session, get audio for complete sentences
  - `DELETE /api/stream/end/{session_id}` - End streaming session
- **Model lazy-loading**: Models loaded on first request and cached
- **Streaming sessions**: In-memory session cache for streaming TTS

## Build and Installation

**Development Installation:**
```bash
pip install -e .
# Or with WebUI support:
pip install -e . fastapi uvicorn python-multipart
```

**Building Wheel:**
```bash
python -m build
```

**Docker Build:**
```bash
docker build -t kittentts-webui .
docker run -d -p 7860:7860 -v ~/.cache/huggingface:/root/.cache/huggingface kittentts-webui
```

## Running the Application

**Python API:**
```python
from kittentts import KittenTTS
import soundfile as sf

model = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = model.generate("Hello world", voice="Jasper", speed=1.0)
sf.write("output.wav", audio, 24000)
```

**WebUI:**
```bash
python run_webui.py --host 0.0.0.0 --port 7860
```

**Streaming TTS (for LLM integration):**
```python
from kittentts import KittenTTS, StreamingTTS
import soundfile as sf

# Initialize model
model = KittenTTS("KittenML/kitten-tts-mini-0.8")

# Create a streaming instance
streamer = model.create_streamer(voice="Jasper", speed=1.0)

# Simulate streaming from an LLM
llm_tokens = ["Hello", " there", "! How", " are", " you", " today", "?"]

for token in llm_tokens:
    # add_text() yields audio chunks when complete sentences are detected
    for audio_chunk in streamer.add_text(token):
        sf.write("chunk.wav", audio_chunk, 24000)
        # Or play immediately for real-time output

# Don't forget to flush remaining buffered text
for audio_chunk in streamer.flush():
    sf.write("final_chunk.wav", audio_chunk, 24000)
```

**Streaming via Web API:**
```python
import requests
import json

BASE_URL = "http://localhost:7860"

# Start a streaming session
response = requests.post(f"{BASE_URL}/api/stream/start?model=kitten-tts-nano&voice=Jasper&speed=1.0")
session_id = response.json()["session_id"]

# Stream text chunks
for token in ["Hello", " there", "! How", " are", " you", "?"]:
    response = requests.post(
        f"{BASE_URL}/api/stream/chunk?session_id={session_id}",
        json={"text": token, "flush": False}
    )
    result = response.json()
    for audio_base64 in result["audio_chunks"]:
        # Decode and play audio
        pass

# Flush remaining text and end session
response = requests.post(
    f"{BASE_URL}/api/stream/chunk?session_id={session_id}",
    json={"text": "", "flush": True}
)
requests.delete(f"{BASE_URL}/api/stream/end/{session_id}")
```

## Development Conventions

**Code Style:**
- Project uses `ruff` for linting (evidenced by `.ruff_cache/`)
- Follow PEP 8 conventions
- Use type hints where appropriate (FastAPI models use Pydantic)

**Text Processing Order:**
When modifying `preprocess.py`, maintain the processing order in `TextPreprocessor.process()`:
1. Unicode normalization
2. Content removal (HTML, URLs, emails)
3. Contraction expansion
4. IP addresses (before decimal normalization)
5. Currency/percentages/scientific notation
6. Time, ordinals, units, fractions, decades
7. Phone numbers (before ranges)
8. Ranges, model names, Roman numerals
9. Generic number replacement
10. Final cleanup (accents, punctuation, lowercase)

**Voice Aliases:**
The WebUI uses friendly names (Bella, Jasper, etc.) that map to internal voice IDs (expr-voice-2-f, expr-voice-2-m, etc.). Maintain this mapping in both `webui/server.py` and model configs.

## Testing

**Current State:** No test suite is currently present in the repository.

**Recommended Testing Approach:**
- Add unit tests for `TextPreprocessor` with various input cases
- Test ONNX model inference with dummy inputs
- Integration tests for HuggingFace model downloading
- WebUI API endpoint testing with `TestClient` from FastAPI

## Deployment Considerations

**System Requirements:**
- Python 3.12 recommended (3.8 minimum)
- `espeak-ng` system package required (installed in Dockerfile)
- HuggingFace cache directory should be persisted for faster restarts
- Models are downloaded on-demand (~80MB per model variant)

**Security:**
- WebUI runs with CORS allow-all (`["*"]`) - configure appropriately for production
- No authentication implemented in default WebUI
- Input validation present for speed range (0.25-3.0) and empty text

**Environment Variables:**
- `PYTHONUNBUFFERED=1` set in Docker
- HF cache location follows HuggingFace hub defaults (`~/.cache/huggingface`)

## Common Tasks

**Adding a New Voice:**
1. Add voice embeddings to model's voices.npz on HuggingFace
2. Update `available_voices` in `onnx_model.py`
3. Add voice alias mapping in `webui/server.py`
4. Update voice metadata in `VOICES` list in `server.py`

**Adding a New Model:**
1. Upload ONNX model and config to HuggingFace
2. Add entry to `MODELS` dict in `webui/server.py`
3. Add metadata to `MODEL_INFO` list
4. Ensure config.json has correct `type`, `model_file`, `voices` keys

**Modifying Text Preprocessing:**
1. Add new regex pattern near other `_RE_*` definitions
2. Create expansion function with docstring and examples
3. Add config flag to `TextPreprocessor.__init__`
4. Insert call in `process()` method at appropriate position
5. Add test case in `if __name__ == "__main__"` block

## License

Apache License 2.0 - See LICENSE file for details.
