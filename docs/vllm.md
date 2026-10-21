# Running with vLLM

The optional `vllm` backend runs KittenTTS 2's language model with vLLM 0.31.0 on an NVIDIA GPU;
audio decoding, voice cloning and text preparation stay the same. It needs Linux and a GPU with
compute capability 8.0 or newer (Ampere or later). GPUs with compute capability 7.5 can use a
float16 fallback that is supported but not the optimal path
([Older GPUs](#older-gpus-compute-capability-75)); older GPUs are refused before any download.

Install and select it as shown in the [README](../README.md#running-with-vllm). That install gets
CUDA 13.0 builds of vLLM and PyTorch, which need a driver for CUDA 13.0
([GPU requirements](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/)). If
`nvidia-smi` reports a lower CUDA version, install the CUDA 12.9 builds with
[uv](https://docs.astral.sh/uv/) instead (checked with driver 570.195, which reports CUDA 12.8):

```bash
curl -LO https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
uv pip install --torch-backend cu129 "kittenml[vllm]" ./vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

## Options

| Option | Default | What it is for |
|---|---|---|
| `gpu_memory_utilization` | `0.5` | Share of the GPU's memory vLLM may use for the language model and its cache. Audio decoding needs memory outside this share. |
| `max_model_len` | `4096` | Longest request in tokens: the voice's reference prompt and the text, plus `max_new_tokens` (1000 by default). A request that does not fit raises an error. |
| `max_num_seqs` | `8` | Most requests generated at the same time. |
| `kv_cache_memory_bytes` | not set | A fixed cache size in bytes, used instead of sizing the cache from `gpu_memory_utilization`. |
| `enforce_eager` | `False` | Turns off CUDA graphs. Keep it off: with it on, 4-bit can be slower than the default bf16. |
| `quantization` | not set | `"w4a16"` for 4-bit weights; see [4-bit weights](#4-bit-weights). |
| `dtype` | `"bfloat16"` | Precision of the language model. `"float16"` is a fallback for GPUs without bfloat16; see [Older GPUs](#older-gpus-compute-capability-75). |

Pass these as `vllm_options` when constructing the model, for example
`vllm_options={"max_model_len": 2048}`; unknown option names are refused before any download. Select
the GPU with `CUDA_VISIBLE_DEVICES`; set `VLLM_LOGGING_LEVEL=INFO` before starting Python to see
routine engine logs. Long text uses the default backend's sentence chunking, and `generate_stream`
yields each audio chunk once it is complete. The same seed can give different audio across backends
and with or without 4-bit weights. Start-up is slow (see [Benchmarks](#benchmarks)), so reuse the
loaded model.

## Low GPU memory

- Reduce `max_model_len`, keeping room for the prompt plus `max_new_tokens`: a shorter context
  needs a smaller cache.
- If startup reports no memory available for the cache, raise `gpu_memory_utilization`, leaving room
  for audio decoding, or set `kv_cache_memory_bytes`. The cache needs 112 KiB per token; a cache of
  exactly `max_model_len` tokens is rejected, so leave some margin.
- Use [4-bit weights](#4-bit-weights); the smaller `weights="packed"` download alone does not
  lower GPU memory.
- Cloning without `reference_text` also loads Whisper large-v3 on the GPU to transcribe the clip;
  pass `reference_text` to skip it.

## 4-bit weights

Add `"quantization": "w4a16"` to `vllm_options` to use about 2 GB less GPU memory for the language
model's weights (1.29 instead of 3.24 GiB). The model's weights stay exactly the same, and measured
audio quality was the same as without it. With default settings vLLM still takes half the GPU and
gives the saved memory to its cache; to lower total use, also set `kv_cache_memory_bytes` or a lower
`gpu_memory_utilization`. It works with `weights="packed"` (the default) or `"full"`, and also
with `weights="emb4"`. That one is lossy: its embedding is stored in 4 bits (at export, perplexity
13.60 against 13.50), and it only makes the download smaller, not GPU memory, because the
embedding is expanded to bf16 on the GPU either way. `"emb4"` is not available with the float16
fallback below. KittenTTS checks that vLLM runs the 4-bit layers
with the kernel they were verified with, and refuses settings that would change it.

The biggest gain is with one request at a time: the language model ran about 1.4x faster on an RTX
PRO 6000 (96 GB) and 1.8x faster on an RTX A2000 (6 GB). With many concurrent requests the gain is
small (1.02x with 50 at once on the RTX PRO 6000). On the RTX A2000, bf16 does not start with
default `vllm_options` but 4-bit does; with `kv_cache_memory_bytes` 512 MiB and `max_num_seqs` 1 for
both, five sentences one at a time:

| RTX A2000 (6 GB) | bf16 (default) | 4-bit |
|---|---:|---:|
| Peak GPU memory, whole card | 6,116 MiB | 4,096 MiB |
| Median time per sentence | 2.65 s | 1.73 s |
| Language-model speed | 54.5 tokens/s | 97.2 tokens/s |

## Older GPUs (compute capability 7.5)

GPUs without bfloat16, such as the GeForce RTX 20 and GTX 16 series, can run the language model in
float16 instead, with or without 4-bit weights:

```python
m = KittenTTS("KittenML/kitten-tts-2", backend="vllm",
              vllm_options={"dtype": "float16", "quantization": "w4a16"})
```

**This is a fallback: it is supported, but it is not the optimal path.** Float16 is used only when
you ask for it; on GPUs with compute capability 8.0 or newer, keep the default (bfloat16). Only an
RTX 2070 (8 GB) was measured.

- **Precision change.** The model is published in bfloat16 and overflows in plain float16, so
  KittenTTS scales its internal values down by an exact power of two. Rounding follows float16
  instead of bfloat16. Given the same text and tokens, the next-token probabilities differ from
  bfloat16's by 0.0085 nats on average (KL divergence; 4-bit weights in bfloat16 differ by
  0.0047). Word error rates on 36 requests (8 languages, long text, cloning) were 0.9% in
  bfloat16 and 0.8-0.9% in float16. No listening test was done.
- **Not the fastest attention kernel.** vLLM's FlashAttention 2 needs compute capability 8.0, so
  these GPUs use a general kernel; startup logs `Cannot use FA version 2 ...`, which is expected.
  bfloat16 cannot run on these GPUs, so there is no same-card speed comparison.
- **More memory.** Float16 keeps a separate copy of the output layer: 618 MiB more than bfloat16.

| RTX 2070 (8 GB), one request at a time | float16 + 4-bit, default settings | float16, `kv_cache_memory_bytes` 1 GiB |
|---|---:|---:|
| Language-model speed | 108 tokens/s | 74 tokens/s |
| Synthesis time per second of audio | 0.30 s | 0.41 s |
| First audio when streaming (median) | 1.33 s | 1.80 s |
| Peak GPU memory, whole card | 6,574 MiB | 8,024 MiB |
| Startup with warm caches | 55 s | 46 s |

Measured with `generate()` on the same 36 texts (29 of them for streaming), one request at a time.

## Benchmarks

One RTX 3060 (12 GB), driver 570.195.03 (CUDA 12.8), Ubuntu 24.04, Python 3.12, nothing else on the
GPU. PyTorch GPU is the default backend (PyTorch 2.14.1 CUDA 12.6 build, Transformers 5.20.2); the
vLLM columns use vLLM 0.31.0 (CUDA 12.9 build, PyTorch 2.13.0, Transformers 5.17.0) with default
`vllm_options`: "vLLM bf16" is without quantization, "vLLM 4-bit" adds `"quantization": "w4a16"`
([4-bit weights](#4-bit-weights)). Each time is the median of five `generate()` calls after two
warmups (`weights="packed"`, default decoder, stable preset, model loading excluded) on 61, 226 and
680 characters of text (the last in two chunks).

| Input / voice | PyTorch GPU | vLLM bf16 | vLLM 4-bit | Speedup bf16 vs PyTorch | Speedup 4-bit vs PyTorch |
|---|---:|---:|---:|---:|---:|
| Short / Bruno | 4.44 s | 1.72 s | 1.18 s | 2.59x | 3.77x |
| Short / Bella | 6.14 s | 2.67 s | 1.62 s | 2.30x | 3.78x |
| Medium / Bruno | 14.74 s | 5.12 s | 3.43 s | 2.88x | 4.30x |
| Medium / Bella | 20.63 s | 7.01 s | 4.63 s | 2.94x | 4.46x |
| Long / Bruno | 45.20 s | 16.10 s | 10.38 s | 2.81x | 4.35x |
| Long / Bella | 53.15 s | 18.96 s | 12.24 s | 2.80x | 4.34x |

| RTX 3060 (12 GB) | PyTorch GPU | vLLM bf16 | vLLM 4-bit |
|---|---:|---:|---:|
| Synthesis time per second of audio | 1.17-1.19 s | 0.41-0.45 s | 0.26-0.30 s |
| Peak GPU memory, whole card | 6,901 MiB | 9,026 MiB | 9,890 MiB |
| Start-up with files already downloaded (very first start) | 9.8 s (11.2 s) | 44.0 s (145.4 s) | 50.3 s (140.7 s) |

Per second of audio the speedups are 2.64-2.85x (bf16) and 3.94-4.51x (4-bit). Both vLLM columns
give vLLM the same share of the card, so 4-bit gets a larger cache (37,120 tokens instead of
18,512); its higher whole-card peak was in the main process, which decodes the audio, not in
vLLM's own process. Whole-card peaks include about 374 MiB the GPU reports as used when idle. A
full repeat of every column was within -0.9% to +3.4%.
