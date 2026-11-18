# KittenTTS 2 nano, pico and mini

Small multilingual models. Each repository ships every model three ways: ONNX graphs with int8 weights (the
default) and with int4 weights (about half the size), both running on CPU with ONNX Runtime and no PyTorch, and
the float32 weights for PyTorch (`backend="torch"`, CPU or GPU).

| Repository | Parameters | int8 size per model | int4 size per model | Models | Languages |
|---|---|---|---|---|---|
| `KittenML/kitten-tts-2-mini` | 82M | 82.7 MB | 47.5 MB | 1 | 16: en de es it pt ru hi fr da fi ja ko nl sv sw tr |
| `KittenML/kitten-tts-2-nano` | 17M | 17.4 MB | 8.8 MB | 8 (one per language) | en de es it pt ru hi fr |
| `KittenML/kitten-tts-2-pico` | 7.2M | 8.0 MB | 4.0 MB | 8 (one per language) | en de es it pt ru hi fr |

Only the files of the languages, the backend and the weight format you use are downloaded.

## Usage

```python
from kittenml import KittenTTS
import soundfile as sf

m = KittenTTS("KittenML/kitten-tts-2-nano")          # or a local directory laid out like the repository
print(m.languages)                                   # ['en', 'de', ...]
print(m.available_voices)                            # {'en': ['Eleanor', 'Ruth', ...], 'de': ['Anna', ...], ...}
for v in m.voices("de"):                             # name and description of each voice
    print(v["name"], v["description"])               # Anna Female 1, ...

audio = m.generate("Hello there.")                                  # English, first voice
audio = m.generate("Guten Morgen.", language="de")                  # German, first voice (Anna)
audio = m.generate("Guten Morgen.", voice="Paul")                    # a German voice: German
audio = m.generate("Guten Morgen.", voice="paul")                    # names are case-insensitive
audio = m.generate("Hello there.", voice="Eleanor", speed=1.1)
sf.write("out.wav", audio, m.sample_rate)                           # 24 kHz

for chunk in m.generate_stream(long_text, language="fr"):           # chunk by chunk
    ...
m.phonemize("Hello there.")                                          # what the model will say
```

| Argument | Default | |
|---|---|---|
| `voice` | the language's first voice | a name from `available_voices` in any case (`"Anna"`, `"anna"`) or a voice's description |
| `language` | the voice's language, else `"en"` | a code from `languages` |
| `speed` | `1.0` | larger is faster |

| `KittenTTS(..., backend=, device=, weights=)` | | |
|---|---|---|
| `backend=None` / `"onnx"` | default | ONNX graph in ONNX Runtime; `device="cuda"` / `"amd_gpu"` picks the CUDA / ROCm provider (needs `onnxruntime-gpu`) |
| `weights="int8"` | default | the int8 ONNX graphs |
| `weights="int4"` | | the int4 ONNX graphs, about half the size; needs `onnxruntime>=1.20` |
| `backend="torch"` | | float32 weights in PyTorch; `device="cpu"` / `"cuda"` (default: CUDA if available) |
| `backend="cuda"` / `"amd_gpu"` / `"cpu"` | | ONNX on that device (the 0.8 models' convention) |

```python
m = KittenTTS("KittenML/kitten-tts-2-pico", weights="int4")     # 4 MB per language
m = KittenTTS("KittenML/kitten-tts-2-nano", backend="torch", device="cuda")
audio = m.generate("Hello there.", voice="Eleanor")
```

The PyTorch backend gives the same samples as the training code's inference (same seed); the ONNX int8 and
int4 backends match it within the quantization error (see below).

## Text front end

The phonemes are produced exactly as in training:

- **English**: text normalisation (numbers, dates, times, units, ...), then the misaki lexicon, then CMUdict,
  then a 0.75M-parameter BART grapheme-to-phoneme model for the words neither knows. The lexicon and the
  BART ship in the model repository (`g2p/`); spaCy's `en_core_web_sm` is downloaded on first use.
- **Other languages**: espeak-ng (through `espeakng_loader`) with misaki's espeak post-processing.
- **Japanese** (mini): misaki's Japanese G2P. Install with `pip install "kittenml[ja]"`.

Long text is split at sentence ends into chunks of at most 350 phonemes, joined with 0.15 s of silence.

## Quantization

In the int8 graphs (the default) all weights (convolutions, LSTMs, matrix multiplications, embeddings) are stored as int8, symmetric, one
scale per output channel, and expanded by `DequantizeLinear`; the arithmetic stays float32. Dynamic int8
*activation* quantization (`MatMulInteger` / `ConvInteger`, as in the 0.8 models) was measured and rejected:
on these vocoders it gives audible spectral error and non-finite audio on part of the inputs.

The int8 weights come from quantization-aware distillation from the fp32 model: first the text side and
prosody predictor (durations, F0, energy), then the decoder, each trained through the exact int8 rounding the
graph uses, so the shipped weights are the ones that were trained. Against the fp32 model, on held-out text
and the same synthesis noise, the int8 models have the same Whisper character error rate, a speaker-embedding
similarity of about 0.999, and a median pitch difference of about 0 cents; see each model card for the
per-model numbers.

### int4

The int4 graphs (`weights="int4"`) store weights as int4, symmetric, in blocks of 128 along each layer's input
axis with one fp16 scale per block (`DequantizeLinear` with `block_size`, ONNX opset 21, so
`onnxruntime>=1.20`). About 1% of the weights, the ones the output is most sensitive to, stay int8, and the
small float parameters (biases, norms) are fp16; the arithmetic stays float32. They are trained the same way,
by quantization-aware distillation from the fp32 model, and have the same inputs and outputs as the int8
graphs.

On 60 held-out sentences per language (Whisper large-v3 character error rate, fp32 in brackets):

| | en | de | es | it | pt | ru | hi | fr |
|---|---|---|---|---|---|---|---|---|
| pico int4 (4.0 MB) | 1.17 (0.86) | 1.64 (1.78) | 0.98 (0.91) | 0.94 (0.66) | 4.08 (4.10) | 5.79 (5.87) | 10.72 (10.70) | 2.13 (2.05) |
| nano int4 (8.8 MB) | 1.08 (1.08) | 1.28 (1.63) | 0.89 (0.83) | 0.69 (0.61) | 3.93 (3.87) | 6.48 (6.50) | 13.21 (13.13) | 2.13 (1.94) |

Speaker-embedding similarity to fp32 is 0.996 to 0.999, and 95 to 99% of the phoneme durations are identical
to fp32's.

## Repository layout

```
config.json                   "type": "KITTEN2_TINY", phonemizer version, languages -> ONNX graphs by weight
                              format ("model_files": {"int8", "int4"}), PyTorch weight and voice files,
                              model architectures, voices (key = lowercase name, name, description)
vocab.json                    phoneme -> token id
kitten_tts_2_<family>_<lang>.onnx          (mini: kitten_tts_2_mini.onnx) int8 ONNX graph
kitten_tts_2_<family>_<lang>_int4.onnx     (mini: kitten_tts_2_mini_int4.onnx) int4 ONNX graph
kitten_tts_2_<family>_<lang>.safetensors   (mini: kitten_tts_2_mini.safetensors) float32 PyTorch weights
voices_<lang>.npz             one float32 [510, 256] pack per voice, stored under the voice key
g2p/                          the English CMUdict lexicon and BART G2P
```

The ONNX graphs and the PyTorch model: `input_ids [1, T] int64, style [1, 256] float32, speed [1] float32 -> waveform [N] float32,
durations [T] int64` (600 samples per duration frame).
