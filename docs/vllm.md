# Running with vLLM

The optional `vllm` backend runs KittenTTS 2's language model on an NVIDIA CUDA
GPU. Audio decoding, voice cloning and text preparation use the existing
KittenTTS implementations. The default backend remains PyTorch and already
supports CUDA.

Install the optional backend and select `backend="vllm"` as shown in the
[README](../README.md#running-with-vllm). This backend requires Linux and uses
vLLM 0.31.0; see its [GPU requirements](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/)
for compatible NVIDIA drivers.
It loads the selected published weights directly; no manual export
or additional language-model checkpoint is required. Packed weights are expanded in
memory, so their smaller download size does not reduce GPU memory use.

The engine defaults to a 4,096-token context, up to eight active sequences, and
half the GPU memory, leaving space for audio decoding. In the README example,
pass `vllm_options` when constructing the model to change these settings:

```python
m = KittenTTS("KittenML/kitten-tts-2", backend="vllm",
              vllm_options={"gpu_memory_utilization": 0.5,
                            "max_model_len": 4096})
```

`gpu_memory_utilization` is a fraction of the GPU's total VRAM. It sets vLLM's
budget; the audio decoder and other GPU processes need additional memory.

If VRAM is limited, reduce `max_model_len` to lower cache requirements. If startup
reports no available memory for cache blocks, increase `gpu_memory_utilization`
to give vLLM a larger share of VRAM, leaving space for audio decoding and other
GPU processes.

Routine engine logs are quiet by default; warnings and errors remain visible.
Set `VLLM_LOGGING_LEVEL=INFO` before starting Python to see startup diagnostics.

The context must fit the reference prompt and the requested generation budget.
Long text still uses KittenTTS's sentence chunking. `generate_stream` yields
completed audio chunks, just like the default backend.

`vllm_options` also accepts `max_num_seqs` and `enforce_eager`. Select a GPU with
`CUDA_VISIBLE_DEVICES` before starting Python. Sampling implementations differ
between the backends, so a matching seed does not guarantee identical audio.

## Benchmarks

Measured on an RTX 3060 (12 GB), Linux, Python 3.12 and NVIDIA driver 580.95.05,
using `weights="packed"`, the default decoder and the stable preset. These are
median end-to-end synthesis times from five runs after two warmups per voice.
Short, medium and long inputs contain 61, 226 and 680 characters; the long input
is split into two chunks. Model initialization and file writes are excluded.

| Input / voice | PyTorch GPU | vLLM | Speedup |
|---|---:|---:|---:|
| Short / Bruno | 6.01 s | 2.05 s | 2.93x |
| Short / Bella | 6.98 s | 2.54 s | 2.75x |
| Medium / Bruno | 22.93 s | 5.49 s | 4.18x |
| Medium / Bella | 27.95 s | 8.26 s | 3.39x |
| Long / Bruno | 56.77 s | 17.56 s | 3.23x |
| Long / Bella | 79.24 s | 24.04 s | 3.30x |

The PyTorch column uses unchanged upstream code in a fresh normal installation
(PyTorch 2.14.1, Transformers 5.18.0). With versions matched to the vLLM install
(PyTorch 2.13.0, Transformers 5.17.0, vLLM 0.31.0), the measured speedup was
3.05-4.49x. Generated audio lengths differ between samplers; vLLM's synthesis
time was 0.44-0.48 seconds per second of output audio in these tests.

vLLM used about 8.5 GiB of GPU memory versus 5.8 GiB for the normal installation.
Cloning without `reference_text` also loads Whisper; those tests used up to
11.5 GiB. Supplying the transcript skips that model.
Initialization with cached downloads took 51 seconds versus 10 seconds; the
first vLLM startup took about 162 seconds. Reuse the loaded model across calls.
