# Running with MLX (Apple silicon)

The optional `mlx` backend runs KittenTTS 2's language model on a Mac's GPU
through [MLX](https://github.com/ml-explore/mlx). It is the fast way to run
KittenTTS 2 on an Apple-silicon Mac: the default PyTorch backend runs the model
on the CPU there, below realtime.

```bash
pip install "kittenml[mlx]"
```

```python
from kittenml import KittenTTS

m = KittenTTS("KittenML/kitten-tts-2", backend="mlx")
m.generate_to_file("One day, a little girl named Lily found a needle in her room.",
                   "output.wav", voice="Bruno")
```

`backend="auto"` is the portable spelling: MLX on an Apple-silicon Mac with
`mlx` installed, the PyTorch backend everywhere else. It never selects vLLM,
which has memory settings of its own to choose.

Voices, cloning, expression tags, decode presets, chunking and streaming are the
existing implementations and behave identically. Only the language model moves.

## Why it is faster

Generating audio is one forward pass per 1/25 s of speech, each pass reading
every weight once, so the model is bound by memory bandwidth rather than
arithmetic. Two things change:

- **The weights stay packed.** The published checkpoint's linear weights are
  ternary — inside every 128-wide group each value is exactly one of
  `{-scale, 0, +scale}` — and the repository ships them at 1.6 bits each. Both
  other backends expand them to bf16 before the first token, spending 16 bits to
  say one of three things. MLX's 4-bit quantised format can represent those three
  values *exactly*, by taking its scale from the group and its zero point at
  `-scale`, which puts `-scale`, `0` and `+scale` at codes 0, 1 and 2. So the
  1.4B ternary parameters are read as 4 bits each, and the weights being
  multiplied are bit-for-bit the published ones. Nothing is rounded, and there is
  no quantisation step at load time — the trits on disk already *are* the codes,
  so loading is bit shuffling and takes about a second.
- **It runs on the GPU.** Unified memory means the weights are not copied
  anywhere to get there, and audio decoding runs on the same GPU through MPS.

The result is roughly 1.4 GB read per token instead of 3.5 GB, at GPU bandwidth
instead of CPU bandwidth.

## What is and is not identical

The weights are exact, and the decode pipeline — the run and repetition
penalties, temperature, top-k, top-p, min-p, in that order — is a port of the
same one, checked against it in `tests/test_mlx.py`.

Generated audio is still not sample-identical to the PyTorch backend's, for the
same reason two PyTorch runs with different attention kernels are not: the model
computes in bfloat16, whose rounding depends on the order of operations. The
logit differences are of the size bfloat16 rounding produces — the same size as
between MLX in bfloat16 and MLX in float32 — and a seed therefore reproduces a
generation within a backend, not across backends. Speaker similarity against the
reference is unchanged.

## Requirements and limits

- An Apple-silicon Mac. `backend="mlx"` raises on Intel Macs and elsewhere.
- `weights="packed"`, which is the default. `weights="emb4"` is a different
  packing with a quantised embedding and is refused rather than silently
  expanded; `weights="full"` is not the packed file this backend reads.
- The first generation of a process also compiles the Metal kernels, which costs
  about a second on top; the numbers below are steady state.
- `device=` selects only where audio decoding, speaker embedding and
  transcription run, since the language model is always on the GPU. It defaults
  to `"mps"`, which decodes several times faster than the CPU. Pass
  `device="cpu"` if a PyTorch build's MPS backend is missing an op s3gen needs.
- One request at a time, like the PyTorch backend. Long text still uses
  KittenTTS's sentence chunking, and `generate_stream` yields completed chunks.
- Voice cloning without a supplied `reference_text` still loads Whisper through
  transformers, which runs on the CPU.

## Benchmarks

Measured on an Apple M4 Max with 12 GB of memory, macOS 15.6, Python 3.13, PyTorch
2.14.1 and MLX 0.32.4, using `weights="packed"`, the default decoder and the stable
preset. These are median end-to-end synthesis times from five runs after one warmup
per voice (three runs for PyTorch, which is slow enough that the medians were already
stable). Short, medium and long inputs contain 61, 205 and 522 characters; the long
input is split into two chunks. Model initialization and file writes are excluded.

The two backends sample differently, so each produces a slightly different length of
audio from the same text — hence the realtime factors rather than a bare time ratio.

| Input / voice | PyTorch CPU | MLX | PyTorch | MLX | Speedup |
|---|---:|---:|---:|---:|---:|
| Short / Bruno | 9.23 s | 0.86 s | 0.41x realtime | 4.76x realtime | 11.6x |
| Short / Bella | 13.06 s | 1.12 s | 0.39x realtime | 5.07x realtime | 13.0x |
| Medium / Bruno | 23.45 s | 2.55 s | 0.53x realtime | 4.77x realtime | 9.0x |
| Medium / Bella | 30.25 s | 3.31 s | 0.57x realtime | 5.21x realtime | 9.1x |
| Long / Bruno | 54.98 s | 6.14 s | 0.54x realtime | 4.68x realtime | 8.7x |
| Long / Bella | 71.21 s | 8.42 s | 0.55x realtime | 4.91x realtime | 8.9x |

Of the 0.86 s short run, the language model takes about 0.68 s — roughly 150 tokens
per second, against the codec's 25 tokens per second of speech — and audio decoding
about 0.16 s. Decoding on the CPU instead (`device="cpu"`) costs about 1.0 s, which
is why MPS is the default for it.

This machine is a virtual machine, which costs some memory bandwidth; a Mac running
the same model natively should be at least as fast.
