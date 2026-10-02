# Lightweight ONNX models

The original Kitten TTS models. For KittenTTS 2 see [api.md](api.md).

The original Kitten TTS models, 15M to 80M parameters (25-80 MB on disk), built on ONNX. They run
efficiently on CPU without a GPU, which makes them the right choice for edge deployment and for
anything that cannot carry a PyTorch stack. They offer eight voices and adjustable speech speed,
and no voice cloning or expression control.

# Installation

Already covered by `pip install kittenml`, which includes the ONNX runtime. Python 3.8 or later,
and no GPU required.

Note that the default install also pulls PyTorch for KittenTTS 2, which is large relative to
these models. Installing `kittenml` with `--no-deps` alongside `onnxruntime`, `espeakng_loader`,
`phonemizer`, `soundfile`, `numpy` and `huggingface_hub` is enough to run them on their own.

# Basic usage

```python
from kittenml import KittenTTS

model = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = model.generate("This high-quality TTS model runs without a GPU.", voice="Jasper")

import soundfile as sf
sf.write("output.wav", audio, 24000)
```

# Advanced usage

```python
# Adjust speech speed (default: 1.0)
audio = model.generate("Hello, world.", voice="Luna", speed=1.2)

# Save directly to a file
model.generate_to_file("Hello, world.", "output.wav", voice="Bruno", speed=0.9)

# List available voices
print(model.available_voices)
# ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
```

# Using with GPU

```
pip install -r requirements_gpu.txt
```

```python
m = KittenTTS("KittenML/kitten-tts-mini-0.8", backend="cuda")
```

Check out `example_onnx_0.8.py`

# ONNX API Reference

# `KittenTTS(model_name, cache_dir=None, backend=None)`

Load a model from Hugging Face Hub.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"KittenML/kitten-tts-nano-0.8"` | Hugging Face repository ID |
| `cache_dir` | `str` | `None` | Local directory for caching downloaded model files |
| `backend` | `str` | `None` | ONNX execution provider: `"cpu"`, `"cuda"`, `"amd_gpu"` |

# `model.generate(text, voice, speed, clean_text)`

Synthesize speech from text, returning a NumPy array of audio samples at 24 kHz.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | -- | Input text to synthesize |
| `voice` | `str` | `"expr-voice-5-m"` | Voice name (see available voices) |
| `speed` | `float` | `1.0` | Speech speed multiplier |
| `clean_text` | `bool` | `False` | Preprocess text (expand numbers, currencies, etc.) |

# `model.generate_to_file(text, output_path, voice, speed, sample_rate, clean_text)`

Synthesize speech and write directly to an audio file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | -- | Input text to synthesize |
| `output_path` | `str` | -- | Path to save the audio file |
| `voice` | `str` | `"expr-voice-5-m"` | Voice name |
| `speed` | `float` | `1.0` | Speech speed multiplier |
| `sample_rate` | `int` | `24000` | Audio sample rate in Hz |
| `clean_text` | `bool` | `True` | Preprocess text (expand numbers, currencies, etc.) |

# `model.available_voices`

Returns a list of available voice names: `['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']`
