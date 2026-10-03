# Kitten TTS

<p align="center">
  <img width="607" alt="Kitten TTS" src="https://raw.githubusercontent.com/KittenML/KittenTTS/main/assets/banner.png" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/KittenML/KittenTTS-Demo"><img src="https://img.shields.io/badge/Demo-Hugging%20Face%20Spaces-orange" alt="Hugging Face Demo"></a>
  <a href="https://discord.com/invite/VJ86W4SURW"><img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://kittenml.com"><img src="https://img.shields.io/badge/Website-kittenml.com-blue" alt="Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>

> ## **New:** Free Kitten TTS API available at  [https://platform.kittenml.com](https://platform.kittenml.com/)

Kitten TTS is an open-source text-to-speech library. Its flagship model, **KittenTTS 2**, is a
1.7B-parameter 1-bit speech language model with in-context voice cloning and expression control: give it
five seconds of anyone's voice and it speaks your text in that voice. It runs realtime on a CPU!

The library also ships the original [**lightweight legacy models**](docs/onnx-models.md),
15M-80M parameters, which run on CPU without a GPU. Both families load through the same
`KittenTTS(...)` constructor.

**Commercial support is available.** For integration assistance, custom voices, or enterprise licensing, [contact us](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview).

## Table of Contents

- [Features](#features)
- [Available Models](#available-models)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Voice cloning](#voice-cloning)
- [Expression controls](#expression-controls)
- [Running on CPU](#running-on-cpu)
- [Long text and streaming](#long-text-and-streaming)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Commercial Support](#commercial-support)
- [Community and Support](#community-and-support)
- [License](#license)

## Features

- **Voice cloning** -- Clone any speaker from 5-30 seconds of audio, no fine-tuning
- **47 built-in voices** -- Including the eight from KittenTTS 0.8 and nine non-English
- **Multilingual** -- 20 languages: English, Arabic, Chinese, French, German, Hindi, Italian, Portuguese, Russian, Spanish, Japanese, Korean, Turkish, Dutch, Swedish, Danish, Finnish, Swahili, Greek, Hebrew
- **Expression control** -- `[emotion]` tags, inline `<event>` tags, and `(((emphasis)))` spans
- **Decoding presets** -- Trade stability against expressiveness per request
- **Long-form text** -- Sentence-aware chunking with seamless joins
- **Text preprocessing** -- Numbers, currencies, dates, units and abbreviations expanded automatically
- **24 kHz output** -- High-quality audio at a standard sample rate
- **Runs without a GPU**
- **Optimized C++ inference for CPU** -- Our fork of [llama.cpp](https://github.com/KittenML/kitten-tts-2-cpp)
## Available Models

**KittenTTS 2** -- speech language model:

| Model | Parameters | Size | Voices | Download |
|---|---|---|---|---|
| kitten-tts-2 | 1.7B | 506 MiB | 47 + cloning | [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2) |

**Lightweight ONNX** -- runs on CPU, no GPU required:

| Model | Parameters | Size | Download |
|---|---|---|---|
| kitten-tts-mini | 80M | 80 MB | [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) |
| kitten-tts-micro | 40M | 41 MB | [KittenML/kitten-tts-micro-0.8](https://huggingface.co/KittenML/kitten-tts-micro-0.8) |
| kitten-tts-nano | 15M | 56 MB | [KittenML/kitten-tts-nano-0.8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) |
| kitten-tts-nano (int8) | 15M | 25 MB | [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) |


## Demo


https://github.com/user-attachments/assets/c42c236b-7b7d-41d6-944b-7527a5f30626








### Try it online

Try Kitten TTS directly in your browser on [KittenML Platform](https://platform.kittenml.com).

## Quick Start

### Prerequisites

- Python 3.9 or later
- A CUDA GPU with roughly 8 GB free, or a CPU with about 6 GB of RAM
- About 1 GB of disk space for the model, or 506 MiB with the smaller weights

### Installation

```bash
pip install kittenml
```

That is the whole install. KittenTTS 2 is the default model, voice cloning is included, and no
Hugging Face login is needed -- every weight the model uses ships in its own repository. It also
pulls the small ONNX runtime, so the [lightweight models](docs/onnx-models.md) work from the
same install.

### Basic usage

```python
from kittenml import KittenTTS
import soundfile as sf

m = KittenTTS("KittenML/kitten-tts-2")

audio = m.generate("One day, a little girl named Lily found a needle in her room.",
                   voice="Bruno")
sf.write("output.wav", audio, m.sample_rate)
```

`m.available_voices` lists all 47 built-in voices, described in
[voices and expression](docs/voices-and-expression.md). Bella, Jasper, Luna, Bruno, Rosie, Hugo,
Kiki and Leo are the same speakers as in KittenTTS 0.8, so code written against the ONNX models
keeps working.

The weights come in two sizes. The default is 947 MiB and lossless; `weights="emb4"` is 506 MiB
because it quantises the token embedding, which costs a little quality. Only the one you ask for
is downloaded.

```python
m = KittenTTS("KittenML/kitten-tts-2", weights="emb4")   # half the download
```

```python
# Trade stability against expressiveness
audio = m.generate("Hello, world.", voice="Luna", preset="expressive")

# Save directly to a file
m.generate_to_file("Hello, world.", "output.wav", voice="Bruno")
```

## Voice cloning

Pass `reference=` instead of `voice=` — same method, the recording just replaces the built-in
speaker. Give it 5-30 seconds of a single speaker. The transcript is part of the prompt, but you
do not have to type it; Whisper fills it in when omitted.

```python
audio = m.generate("This is my own voice, cloned.", reference="my_voice.wav")

# Supplying the transcript skips the Whisper pass
audio = m.generate("This is my own voice.", reference="my_voice.wav",
                   reference_text="what is actually said in the clip")
```

The reference feeds the model by two independent routes -- a speaker embedding through the
model's projection head, and the clip itself as codec tokens in the prompt -- so identity
survives even when one route is weak.

Measured on the built-in voices: a generated clip scores 0.49-0.72 speaker similarity against its
own reference and 0.01-0.18 against the other 37, and cloning an unseen recording scores 0.81
against that recording.

## Expression controls

> **Beta.** Emotion control steers delivery rather than guaranteeing it, and the effect varies
> by voice and by sentence.

```python
audio = m.generate(
    "[joyful] We actually won the grant <laugh> I can (((hardly))) believe it!",
    voice="Kiki",
    preset="expressive",
)
```

A leading `[emotion]` tag, inline `<event>` tags and `(((emphasis)))` spans reach the model as
markup rather than being spoken, and automatically enable its expression conditioning. Ten
emotions and ten vocal events are recognised -- see
[voices and expression](docs/voices-and-expression.md) for the full lists and what is not
covered.

## Running on CPU

KittenTTS 2 runs on CPU out of the box — `device` is auto-detected — but the fastest way is
[kitten-tts-2-cpp](https://github.com/KittenML/kitten-tts-2-cpp), our llama.cpp fork. It reads
the GGUF weights in the model repository's `cpp/` directory.


## Long text and streaming

Long input is split on sentence boundaries and synthesized chunk by chunk, then joined with
silence trimming and short edge fades so the seams are inaudible. This is automatic: the model is
reliable on short inputs but truncates or drifts into repetition when asked for a whole script in
one pass.

To start playing before the whole thing is ready, stream it:

```python
for chunk in m.generate_stream(long_text, voice="Luna"):
    play(chunk)          # each chunk is a numpy array at m.sample_rate
```

It takes the same arguments as `generate`, so `reference=` streams a cloned voice too.

Streaming is **chunk-level, not token-level**: a chunk is generated and vocoded in full before it
is yielded, so the first chunk still costs its own generation time. On an A100, a 936-character
passage yielded its first 21 s of audio after 17 s and finished 54 s of audio in 42 s of wall
clock -- so playback keeps ahead of generation, but there is a real initial delay.

Two consequences worth knowing:

- **Short text does not stream.** Input that fits in one chunk (under roughly 380 characters,
  and short trailing pieces get merged into their neighbour) yields exactly one chunk, so
  `generate_stream` behaves like `generate`.
- **Chunks are yielded raw.** `generate` post-processes the seams -- trimming each segment's edge
  silence, adding short fades and one consistent pause -- which a streaming caller cannot do
  without waiting for the next chunk. Concatenating streamed chunks directly gives slightly
  rougher joins than `generate` on the same text.

## Documentation

| | |
|---|---|
| [API reference](docs/api.md) | Every argument to `generate`, streaming, and the advanced knobs |
| [Voices and expression](docs/voices-and-expression.md) | The 47 voices, emotion and vocal-event tags, the ten languages |
| [Decoders](docs/decoders.md) | How audio is decoded, and the smaller quantised decoders |
| [Text normalization](docs/text-normalization.md) | How written text becomes spoken text |
| [Architecture](docs/architecture.md) | What the model is, package layout, vendored components |
| [Lightweight ONNX models](docs/onnx-models.md) | The CPU models, 15M-80M parameters, and their API |

## System Requirements

**KittenTTS 2**

- **Operating system:** Linux, Windows or Mac
- **Python:** 3.9 or later


A virtual environment (conda, venv, or similar) is recommended to avoid dependency conflicts.


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

This project is licensed under the [Apache License 2.0](LICENSE). That covers the code in this
repository.

**The models are licensed separately and their terms may differ.** Each model repository carries
its own licensing, so check the one you intend to use before relying on it — do not assume the
code's license extends to the weights.

KittenTTS 2 is released under the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md)
