# Kitten TTS

<p align="center">
  <img width="607" height="255" alt="Kitten TTS" src="https://github.com/user-attachments/assets/f4646722-ba78-4b25-8a65-81bacee0d4f6" />
</p>

[![Hugging Face Demo](https://img.shields.io/badge/Demo-Hugging%20Face%20Spaces-orange)](https://huggingface.co/spaces/KittenML/KittenTTS-Demo)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white)](https://discord.com/invite/VJ86W4SURW)
[![Website](https://img.shields.io/badge/Website-kittenml.com-blue)](https://kittenml.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

> **New:** Kitten TTS v0.8 is out -- 15M, 40M, and 80M parameter models now available.

Kitten TTS is an open-source, lightweight text-to-speech library built on ONNX. With models ranging from 15M to 80M parameters (25-80 MB on disk), it delivers high-quality voice synthesis on CPU without requiring a GPU.

> **Status:** Developer preview -- APIs may change between releases.

**Commercial support is available.** For integration assistance, custom voices, or enterprise licensing, [contact us](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview).

## Table of Contents

- [Features](#features)
- [Available Models](#available-models)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [System Requirements](#system-requirements)
- [Roadmap](#roadmap)
- [Commercial Support](#commercial-support)
- [Community and Support](#community-and-support)
- [License](#license)

## Features

- **Ultra-lightweight** -- Model sizes from 25 MB (int8) to 80 MB, suitable for edge deployment
- **CPU-optimized** -- ONNX-based inference runs efficiently without a GPU
- **8 built-in voices** -- Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, and Leo
- **Adjustable speech speed** -- Control playback rate via the `speed` parameter
- **Text preprocessing** -- Built-in pipeline handles numbers, currencies, units, and more
- **24 kHz output** -- High-quality audio at a standard sample rate

## Available Models

| Model | Parameters | Size | Download |
|---|---|---|---|
| kitten-tts-mini | 80M | 80 MB | [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) |
| kitten-tts-micro | 40M | 41 MB | [KittenML/kitten-tts-micro-0.8](https://huggingface.co/KittenML/kitten-tts-micro-0.8) |
| kitten-tts-nano | 15M | 56 MB | [KittenML/kitten-tts-nano-0.8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) |
| kitten-tts-nano (int8) | 15M | 25 MB | [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) |

> **Note:** Some users have reported issues with the `kitten-tts-nano-0.8-int8` model. If you encounter problems, please [open an issue](https://github.com/KittenML/KittenTTS/issues).

## Demo

https://github.com/user-attachments/assets/d80120f2-c751-407e-a166-068dd1dd9e8d

## Quick Start

### Prerequisites

- Python 3.8 or later
- pip

### Installation

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl
```

### Basic Usage

```python
from kittentts import KittenTTS

model = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = model.generate("This high-quality TTS model runs without a GPU.", voice="Jasper")

import soundfile as sf
sf.write("output.wav", audio, 24000)
```

### Advanced Usage

```python
# Adjust speech speed (default: 1.0)
audio = model.generate("Hello, world.", voice="Luna", speed=1.2)

# Save directly to a file
model.generate_to_file("Hello, world.", "output.wav", voice="Bruno", speed=0.9)

# List available voices
print(model.available_voices)
# ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
```

## API Reference

### `KittenTTS(model_name, cache_dir=None)`

Load a model from Hugging Face Hub.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"KittenML/kitten-tts-nano-0.8"` | Hugging Face repository ID |
| `cache_dir` | `str` | `None` | Local directory for caching downloaded model files |

### `model.generate(text, voice, speed, clean_text)`

Synthesize speech from text, returning a NumPy array of audio samples at 24 kHz.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | -- | Input text to synthesize |
| `voice` | `str` | `"expr-voice-5-m"` | Voice name (see available voices) |
| `speed` | `float` | `1.0` | Speech speed multiplier |
| `clean_text` | `bool` | `False` | Preprocess text (expand numbers, currencies, etc.) |

### `model.generate_to_file(text, output_path, voice, speed, sample_rate, clean_text)`

Synthesize speech and write directly to an audio file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | -- | Input text to synthesize |
| `output_path` | `str` | -- | Path to save the audio file |
| `voice` | `str` | `"expr-voice-5-m"` | Voice name |
| `speed` | `float` | `1.0` | Speech speed multiplier |
| `sample_rate` | `int` | `24000` | Audio sample rate in Hz |
| `clean_text` | `bool` | `True` | Preprocess text (expand numbers, currencies, etc.) |

### `model.available_voices`

Returns a list of available voice names: `['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']`

## System Requirements

- **Operating system:** Linux, macOS, or Windows
- **Python:** 3.8 or later
- **Hardware:** Runs on CPU; no GPU required
- **Disk space:** 25-80 MB depending on model variant

A virtual environment (conda, venv, or similar) is recommended to avoid dependency conflicts.

## Roadmap

- [ ] Release optimized inference engine
- [ ] Release mobile SDK
- [ ] Release higher quality TTS models
- [ ] Release multilingual TTS
- [ ] Release KittenASR
- [ ] Need anything else? [Let us know](https://github.com/KittenML/KittenTTS/issues)

## Commercial Support

We offer commercial support for teams integrating Kitten TTS into their products. This includes integration assistance, custom voice development, and enterprise licensing.

[Contact us](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview) or email info@stellonlabs.com to discuss your requirements.

## Community and Support

- **Discord:** [Join the community](https://discord.com/invite/VJ86W4SURW)
- **Website:** [kittenml.com](https://kittenml.com)
- **Custom support:** [Request form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
- **Email:** info@stellonlabs.com
- **Issues:** [GitHub Issues](https://github.com/KittenML/KittenTTS/issues)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
