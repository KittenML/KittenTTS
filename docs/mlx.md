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
