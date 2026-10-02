# Architecture

KittenTTS 2 is not a phoneme-to-waveform model. It is a language model over a vocabulary shared
between text and audio: it reads your text and writes S3 audio codec tokens, which the s3gen
vocoder turns into 24 kHz speech. Voice identity enters through a speaker embedding projected into
the first position's hidden state, plus the reference clip's own codec tokens in the prompt.

# Package layout

```
kittenml/
  get_model.py        KittenTTS(...) -- reads config.json, builds the right backend
  repo.py             repository resolution shared by both backends
  preprocess.py       text normalization shared by both backends
  kittentts2/            the KittenTTS 2 speech language model
    model.py            KittenTTS2: generate, clone, streaming
    prompt.py           reference prefix and target segment assembly
    logits.py           decode-time sampling and runaway penalties
    speaker.py          spk_proj head, pyannote embedding, Whisper transcription
    text.py             expression markup, normalization, chunking, joining
    tokens.py           the shared text/audio token layout
    vocoder.py          reference encoding and token -> waveform
    s3gen.py            vendored S3 tokenizer + s3gen vocoder
  kittentts_legacy/      the ONNX models
    model.py            KittenTTSOnnx: generate, speed control
    onnx_model.py       the ONNX session and phonemizer
```

Neither backend is imported until it is the one selected, so a KittenTTS 2 install never needs
espeak and an ONNX install never needs torch.

# The model repository

A KittenTTS 2 repository is assembled by internal tooling from a training bundle, and holds
everything inference needs:

```
config.json     token layout, decode presets, voice and decoder indexes
lm/             the speech language model, plus its spk_proj head
speaker/        the speaker-embedding model and its license
voices/         reference clips, transcripts, and precomputed embeddings
decoders/       optional quantised decoders
```

Precomputing the per-voice embeddings is what lets a built-in voice skip the embedding model
entirely at run time. Nothing in the repository is gated, so installing KittenTTS 2 needs no
Hugging Face credentials.

Only the paths `config.json` names are downloaded, so a repository can carry artifacts for other
runtimes without every install paying for them.

### Packed language-model weights

The LM's linear weights are ternary: within each 128-wide group every value is exactly one of
`{-scale, 0, +scale}`. Stored as bf16 that spends 16 bits per weight to express one of three
states, so `ternary.py` packs them five trits to a byte — 1.6 bits each, log2(3) rounded up to a
byte boundary — keeping every group's scale at full bf16 precision. The token embedding, layer
norms and speaker head are not ternary and are stored unchanged.

That takes the checkpoint from 3.47 GB to 0.95 GB **losslessly**: unpacking reproduces the
original tensors bit-for-bit, and generated audio is bit-identical. `config.json` advertises it as
`lm_packed`, and the loader prefers it when present.

See `example.py` for a runnable script, and `example_cloning.py`, `example_expression.py`, `example_streaming.py` and `example_advanced.py` for the rest of the API.
