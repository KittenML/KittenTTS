# Running with Docker

The image runs a basic OpenAI-compatible speech API on one NVIDIA GPU: speech as a
complete file or streamed, and voice cloning. Its routes follow the
[hosted KittenML API](https://docs.kittenml.com/tts/generate-speech).

Use a Linux host with Docker, an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The host driver must meet the [vLLM GPU requirements](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/).

## Start the server

Build the image from the root of a KittenTTS checkout. The final `.` is that
directory, which the build reads the package from:

```sh
git clone https://github.com/KittenML/KittenTTS.git
cd KittenTTS
docker build -f kittenml/kittentts2/docker/Dockerfile -t kittenml .
```

Then start the server, from any directory:

```sh
docker run --rm --gpus all --ipc=host -p 8000:8000 -v kittenml-cache:/cache kittenml
```

The first start downloads the model and compiles GPU kernels, which takes a few
minutes. The named volume keeps both, so later starts are faster. The server is
ready when it prints `Ready`. Model files are not included in the image.

## Generate speech

```sh
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kitten-tts-2-latest", "input": "Hello from Kitten TTS.",
       "voice": "Bruno", "response_format": "wav", "stream": false}' \
  --output speech.wav
```

`"stream": false` returns the complete file. Without it, audio is streamed as each
sentence chunk is generated; `"stream_format": "sse"` sends Base64
`speech.audio.delta` events and a final `speech.audio.done`. `response_format` is
`mp3` (the default), `wav`, `pcm`, `flac` or `opus`; `flac` and `opus` are sent once
complete. `GET /v1/voices` lists the 47 built-in voices.

The OpenAI SDK works with any API key:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
        model="kitten-tts-2-latest", voice="Bruno", input="Hello there.",
        response_format="mp3") as response:
    response.stream_to_file("speech.mp3")
```

## Clone a voice

Upload 2-30 seconds of one speaker as `reference_audio` to clone it for one request:

```sh
curl http://localhost:8000/v1/audio/speech \
  -F model=kitten-tts-2-latest \
  -F input="This sentence uses the uploaded voice." \
  -F reference_audio=@reference.wav \
  -F response_format=wav -F stream=false \
  --output cloned.wav
```

Or save the voice once and reuse its ID as `"voice": {"id": "voice_..."}`:

```sh
curl http://localhost:8000/v1/audio/voices \
  -F name="My voice" \
  -F audio_sample=@reference.wav
```

Saved voices are kept in the cache volume and deleted with
`DELETE /v1/audio/voices/{voice_id}`. The clip is transcribed with Whisper, which
needs more GPU memory; add `-F reference_transcript="..."` with the exact words
spoken to skip it.

## Options

Pass these with `-e`:

| Variable | Default | Meaning |
|---|---|---|
| `KITTENML_BACKEND` | `vllm` | `torch` uses the PyTorch backend |
| `KITTENML_VLLM_OPTIONS` | | `vllm_options` as JSON, e.g. `{"gpu_memory_utilization": 0.7}` on an 8 GB GPU |

The server has no authentication, so do not expose its port publicly. Unlike the
hosted API, it supports only `speed` 1.0, uses the voice names from `/v1/voices`,
and has no consent resources.

To run your own script instead, run this from the directory that contains it:

```sh
docker run --rm --gpus all --ipc=host -v kittenml-cache:/cache -v "$PWD:/workspace" \
  kittenml python my_script.py
```
