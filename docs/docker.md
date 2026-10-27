# Running with Docker

The image runs a basic OpenAI-compatible speech API on one NVIDIA GPU: speech as a
complete file or streamed, and voice cloning. Its routes follow the
[hosted KittenML API](https://docs.kittenml.com/tts/generate-speech).

## Requirements

- Linux with Git, [Docker](https://docs.docker.com/engine/install/) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- An NVIDIA GPU with compute capability 7.5 or newer and at least 6 GB of VRAM.
  RTX 20 / GTX 16 use rounded float16: [supported, not optimal](vllm.md#older-gpus-rtx-20-and-gtx-16-series).
- NVIDIA driver 580 or newer for the image's CUDA 13.0 runtime.
- About 25 GB of free disk space for the image and cached models.

For AMD GPUs or V100, use the separate [AMD](vllm.md#amd-gpus-rocm) or
[V100](vllm.md#v100-cuda-126-source-build) instructions; this image does not support them.

## Set up the host

Check whether Docker can already use the GPU:

```sh
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

If this fails, complete the Docker and NVIDIA Container Toolkit setup linked above, then retry.

## Start the server

Clone the repository and build the image:

```sh
git clone https://github.com/KittenML/KittenTTS.git
cd KittenTTS
docker build -f kittenml/kittentts2/docker/Dockerfile -t kittenml .
```

Then start the server, from any directory:

```sh
docker run --rm --gpus all --ipc=host -p 8000:8000 -v kittenml-cache:/cache \
  -e KITTENML_VLLM_OPTIONS='{"quantization": "w4a16"}' \
  kittenml
```

The `-e` line selects [4-bit language-model weights](vllm.md#4-bit-weights).
For bf16 on RTX 30 or newer with more than 8 GB, omit that line. On smaller cards,
4-bit is selected automatically. Put any other [options](#options) in the same JSON.

The first start downloads the model and compiles GPU kernels, which takes a few
minutes. The named volume keeps both, so later starts are faster. When the server prints
`Ready`, use a second terminal for the requests below.

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
complete. `GET /v1/voices` lists the built-in voices (115 in the current model) and your
saved ones.

The OpenAI SDK works with any API key. Install it into a virtual environment
(on Ubuntu: `sudo apt-get install -y python3-venv`, then `python3 -m venv ~/openai-env`,
`source ~/openai-env/bin/activate`, `pip install openai`), save this as `openai_speech.py` and
run `python openai_speech.py`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
        model="kitten-tts-2-latest", voice="Bruno", input="Hello there.",
        response_format="mp3") as response:
    response.stream_to_file("speech.mp3")
```

It prints nothing; when it returns, `speech.mp3` is in the current directory.

## Clone a voice

Upload 5-30 seconds of one speaker as `reference_audio` to clone it for one request, as with
the Python package (the server refuses clips shorter than 2 or longer than 30 seconds). To try
it without a recording, make `reference.wav` with a built-in voice first:

```sh
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kitten-tts-2-latest", "voice": "Luna", "response_format": "wav", "stream": false,
       "input": "I am recording this short clip so that my voice can be cloned. It only needs a few seconds of clear speech."}' \
  --output reference.wav
```

Then clone it:

```sh
curl http://localhost:8000/v1/audio/speech \
  -F model=kitten-tts-2-latest \
  -F input="This sentence uses the uploaded voice." \
  -F reference_audio=@reference.wav \
  -F response_format=wav -F stream=false \
  --output cloned.wav
```

Whisper transcribes the clip automatically and runs on the CPU if GPU memory is low.
For faster cloning, add `-F reference_transcript="..."` with the exact words spoken to skip it.

Or save the voice once:

```sh
curl http://localhost:8000/v1/audio/voices \
  -F name="My voice" \
  -F audio_sample=@reference.wav
```

It prints the new voice's `id`, such as `voice_3f2a...`. Use it as `"voice": {"id": "..."}`;
replace `VOICE_ID` below with that id:

```sh
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kitten-tts-2-latest", "input": "Hello again.",
       "voice": {"id": "VOICE_ID"}, "response_format": "wav", "stream": false}' \
  --output saved.wav
```

Saved voices are kept in the cache volume. Delete one with:

```sh
curl -X DELETE http://localhost:8000/v1/audio/voices/VOICE_ID
```

It prints `{"id":"voice_...","object":"audio.voice","deleted":true}`.

## GPUs with 8 GB or less

Settings adjust automatically to available memory; explicit options are preserved.
Limited memory can reduce concurrency and the maximum request length. See
[Small GPUs](vllm.md#small-gpus) for requirements and tuning.

## Options

Pass these with `-e`:

| Variable | Default | Meaning |
|---|---|---|
| `KITTENML_BACKEND` | `vllm` | `torch` uses the PyTorch backend |
| `KITTENML_VLLM_OPTIONS` | | `vllm_options` as JSON, e.g. `{"quantization": "w4a16"}`; see [Options](vllm.md#options) |

4-bit weights leave more room for vLLM's cache. To reduce its reserved GPU memory,
also set `gpu_memory_utilization` in the same JSON.

The server has no authentication, so do not expose its port publicly. Unlike the
hosted API, it supports only `speed` 1.0, uses the voice names from `/v1/voices`,
and has no consent resources.

To run the [README's `vllm_hello.py`](../README.md#running-with-vllm), stop the server with
Ctrl+C to free GPU memory, then run this from the folder containing the script:

```sh
docker run --rm --gpus all --ipc=host -v kittenml-cache:/cache -v "$PWD:/workspace" \
  kittenml python vllm_hello.py
```

`$PWD:/workspace` mounts the current folder into the container; `vllm_output.wav` is saved there.
