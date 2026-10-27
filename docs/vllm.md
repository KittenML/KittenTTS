# Running with vLLM

The optional `vllm` backend runs KittenTTS 2's language model on the GPU. Voices, cloning,
expression controls and audio decoding use the same API. These instructions use vLLM 0.31.0.

Use the [README example](../README.md#running-with-vllm) on Linux or NVIDIA WSL2 with
NVIDIA driver 580 or newer, compute capability 7.5 or newer and at least 6 GB of VRAM.
vLLM also needs a C compiler and Python development headers
(`gcc` and `python3-dev` on Ubuntu); see its [requirements](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/).
For other setups, expand the relevant section below. Native Windows is not supported by vLLM.

The first start downloads weights and compiles kernels. Reuse the loaded model across requests.
See [benchmarks](#benchmarks) for performance comparisons.

## Options

Pass options when constructing the model, for example `vllm_options={"max_model_len": 2048}`.

| Option | Default | What it is for |
|---|---|---|
| `gpu_memory_utilization` | `0.5` | Share of GPU memory for the language model and cache; leave room for audio decoding. |
| `max_model_len` | `4096` | Maximum tokens for the reference prompt, input text and generated audio together. |
| `max_num_seqs` | `8` | Maximum concurrent requests. |
| `max_num_batched_tokens` | vLLM's default | Tokens processed per step; lower values use less working memory. |
| `kv_cache_memory_bytes` | not set | Fixed cache size instead of sizing it from `gpu_memory_utilization`. |
| `enforce_eager` | `False` | Disable CUDA graphs; normally leave this off. |
| `quantization` | not set; `"w4a16"` on RTX 20 / GTX 16 | Use `"w4a16"` for 4-bit weights, or `None` for dense weights. |
| `dtype` | `"bfloat16"`; `"float16"` on compute capability 7.x | Language-model precision. `"float32"` needs more memory and cannot use 4-bit weights. |

Unknown options raise an error. Select a GPU with `CUDA_VISIBLE_DEVICES`; set
`VLLM_LOGGING_LEVEL=INFO` to show engine logs.

## Small GPUs

On GPUs with 8 GB or less, and on compute capability 7.x GPUs, KittenTTS adjusts unset options
if the defaults do not fit. It can select 4-bit weights, reduce the cache or shorten the
context, and logs the chosen settings. Explicit options are preserved; setting
`gpu_memory_utilization` or `kv_cache_memory_bytes` disables automatic fitting.

This supports 6–8 GB GPUs with enough free memory, but can limit concurrent requests and
maximum request length. If even the smallest settings do not fit, loading stops with the
required free-memory amount. 4 GB cards are insufficient for this vLLM path.

For voice cloning, pass `reference_text` to skip transcription. Otherwise Whisper uses the
CPU when GPU memory is low, which takes longer and needs additional host RAM.

## 4-bit weights

Set `vllm_options={"quantization": "w4a16"}` to reduce language-model weight memory.
With bf16, the weight values are preserved; float16 also rounds them. Generated audio can
differ because the kernels differ.

By default, vLLM gives the saved memory to its cache. To reduce total memory use, also lower
`gpu_memory_utilization` or set `kv_cache_memory_bytes`. The separate `weights="emb4"` option
reduces download size, not GPU memory, and cannot be used with float16.

## Older NVIDIA drivers

<details>
<summary>CUDA 12.9 installation for drivers older than 580</summary>

For GPUs with compute capability 7.5 or newer, use the CUDA 12.9 builds if your driver
supports CUDA 12 but not CUDA 13. Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
and the compiler prerequisites above, then open a new shell with no active Python environment:

```bash
uv venv --seed --python 3.12 ~/kitten-env-cu129
source ~/kitten-env-cu129/bin/activate
uv pip install --torch-backend cu129 "kittenml[vllm]" \
  https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
uv pip uninstall torchcodec
```

The last line removes an unused optional decoder that can conflict with this CUDA build.
Run the README's `vllm_hello.py` in this environment. Activate `~/kitten-env-cu129` again
in each new terminal.

</details>

## Older GPUs (RTX 20 and GTX 16 series)

**Supported, not optimal.** These GPUs have no native bfloat16, so KittenTTS selects rounded
float16 with 4-bit weights. Use the normal install or the older-driver instructions above;
no extra precision option is needed. Audio can differ from newer GPUs.

At least 6 GB of VRAM is needed. To use dense float16, set `{"quantization": None}`;
`{"dtype": "float32"}` is an alternative that needs more memory. GPUs below compute
capability 7.0, including GTX 10, cannot use this backend; use `device="cpu"` with the
default backend instead.

## V100 (CUDA 12.6 source build)

<details>
<summary>Build vLLM for V100</summary>

**Supported, not optimal.** V100 uses rounded float16 with dense weights. It cannot use
bfloat16 or the 4-bit kernel. The standard CUDA 12.9/13 wheels and KittenTTS Docker image
cannot run on it.

Use Linux x86_64 with Python 3.12, Git, GCC/G++ and Python development headers. Install the
[CUDA 12.6 development toolkit](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html)
first; `nvcc --version` must report 12.6. Then, in a new shell:

```sh
python3 -m venv ~/kitten-env-v100
source ~/kitten-env-v100/bin/activate
python -m pip install --upgrade pip uv
nvcc --version
uv pip install --torch-backend cu126 torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0
git clone --depth 1 --branch v0.31.0 https://github.com/vllm-project/vllm.git ~/vllm-v100
cd ~/vllm-v100
uv pip install --torch-backend cu126 -r requirements/build/cuda.txt
TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=8 NVCC_THREADS=2 VLLM_USE_PRECOMPILED_RUST=1 uv pip install --torch-backend cu126 --no-build-isolation .
uv pip install --torch-backend cu126 kittenml
uv pip uninstall torchcodec
cd ~
```

Activate `~/kitten-env-v100` in each new terminal. Save this as `v100_hello.py` and run
`python v100_hello.py`; it leaves out the README example's unsupported 4-bit option:

```python
from kittenml import KittenTTS
import soundfile as sf

if __name__ == "__main__":
    model = KittenTTS("KittenML/kitten-tts-2", backend="vllm")
    audio = model.generate("Hello from KittenTTS!", voice="Bruno")
    sf.write("v100_output.wav", audio, model.sample_rate)
    print("Saved v100_output.wav")
```

</details>

## AMD GPUs (ROCm)

Radeon RX 7000 (RX 7700 and up) and RX 9000 series use a separate ROCm installation.
The WSL2 application path is tested; native Linux remains untested. RX 7600 and older
Radeon GPUs are not supported by this vLLM build.

<details>
<summary>Windows 11 with WSL2</summary>

Use Ubuntu 24.04 in WSL2 with Python 3.12, a C compiler and Python development headers.
Follow [AMD's WSL2 setup](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
for the Windows driver, and [ROCDXG setup](https://github.com/ROCm/librocdxg/blob/v1.2.2/README.md#quickstart)
for ROCm and its WSL2 bridge. Use **ROCm 7.2.4 and librocdxg 1.2.2** with the package versions
below. Do not install `amdgpu-dkms` inside WSL2. Windows-side setup was not part of our testing.

Once `rocminfo` lists your GPU, create a fresh environment in Ubuntu:

```sh
python3 -m venv ~/kitten-env-rocm
source ~/kitten-env-rocm/bin/activate
pip install uv
uv pip install "kittenml[vllm]" --extra-index-url https://wheels.vllm.ai/rocm/0.31.0/rocm723
```

Save the README's `vllm_hello.py`. In each terminal, activate the environment and set the
WSL2 compatibility variables before running it:

```sh
source ~/kitten-env-rocm/bin/activate
export HSA_ENABLE_DXG_DETECTION=1
export ROCPROFILER_REGISTER_ENABLED=0
python vllm_hello.py
```

</details>

<details>
<summary>Native Linux (untested)</summary>

Set up the host with [AMD's Linux driver instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/native_linux/install-radeon.html)
and [Docker](https://docs.docker.com/engine/install/). The GPU must be accessible through
`/dev/kfd` and `/dev/dri`. Start vLLM's ROCm container, then install KittenTTS inside it:

```bash
docker run -it --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash vllm/vllm-openai-rocm:v0.31.0
pip install "kittenml[vllm]"
```

Use `backend="vllm"` in your script. These device flags are for native Linux, not WSL2.

</details>

## Output differences

The same seed does not guarantee identical audio across GPU types, backends or precision
settings. Cloning can also differ when CPU transcription uses a different Whisper model;
provide `reference_text` to skip that step.

## Benchmarks

Times vary with hardware, settings and text; startup is excluded unless stated.

### PyTorch and vLLM

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

### Older NVIDIA GPUs

These modes used the same model revision and 72 requests: three voices, six texts, two seeds,
streaming and complete audio. RTF is generation time divided by audio length; lower is faster.
First audio is the median for short streaming requests. Memory is whole-card peak use.

| GPU and mode | Word error rate | Median RTF | First audio | Peak GPU memory |
|---|---:|---:|---:|---:|
| RTX 3060 12 GB, bf16 reference | 0.25% | 0.45 | 1.52 s | 10,466 MiB |
| RTX 2060 12 GB, rounded float16 | 0.25% | 0.48 | 1.60 s | 10,144 MiB |
| RTX 2060 12 GB, rounded float16 + 4-bit (chosen) | 0.25% | 0.33 | 1.15 s | 9,672 MiB |
| RTX 2060 12 GB, float32 | 0.25% | 0.79 | 2.55 s | 11,640 MiB |
| GTX 1660 Ti 6 GB, rounded float16 + 4-bit | 0.25% | 1.12 | 3.75 s | 5,608 MiB |

The float32 RTX 2060 run used `gpu_memory_utilization=0.74` and
`max_num_batched_tokens=2048`. The GTX 1660 Ti used automatic memory fitting.

| V100 32 GB mode | Word error rate | Median RTF | First audio | Peak GPU memory |
|---|---:|---:|---:|---:|
| Rounded float16 (chosen) | 0.25% | 0.25 | 0.82 s | 20,747 MiB |
| Float32 | 0.25% | 0.36 | 1.21 s | 20,953 MiB |

V100 used the CUDA 12.6 source build and default memory settings. All modes in these two
tables had eight word errors in 3,248 words, no detected dropped sentences, and no silent or
non-finite audio. These are transcription measurements; human listening remains pending.

### AMD Radeon under WSL2

Each mode used 72 requests: three voices, six texts, two seeds, streaming and complete audio.
The RX 7900 XTX and RX 9070 XT used vLLM's ROCm image under WSL2; the RX 9060 XT and RX 7800 XT
used the WSL2 Python environment. These are not native Linux results.

| GPU | bf16: seconds per second of audio | 4-bit: seconds per second of audio | Word error rate: bf16 / 4-bit |
|---|---:|---:|---:|
| RX 7900 XTX | 0.24 | 0.18 | 0.31% / 0.31% |
| RX 9070 XT | 0.28 | 0.21 | 0.31% / 0.31% |
| RX 9060 XT | 0.47 | 0.32 | 0.25% / 0.62% |
| RX 7800 XT | 0.36 | 0.26 | 0.31% / 0.31% |

No request lost words on these cards. The RX 7900 XTX was measured before the long-text
decoder fix; the RX 9070 XT includes it. Transcription scores do not establish identical
sound across GPUs or replace listening checks.
