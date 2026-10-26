"""Optional GPU language-model backend; waveform decoding stays in KittenTTS."""

from dataclasses import asdict
import atexit
import contextlib
import importlib.util
import logging
import math
import os
import sys
import weakref

import torch

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _engine_environment(forced=None):
    # KittenTTS already applies its warpers. Use vLLM's native sampler by default
    # so a pip installation needs no nvcc for FlashInfer's sampling JIT. The
    # worker inherits these settings, including quiet startup defaults. Restore
    # the caller's os.environ afterwards and respect explicit overrides.
    # `forced` values replace the caller's while the engine is created. vLLM reads
    # them when it builds the engine; in single-process mode
    # (VLLM_ENABLE_V1_MULTIPROCESSING=0) it then keeps its own frozen copy of the
    # environment for the rest of the process, which restoring os.environ does
    # not change (VLLMBackend drops that copy before creating each engine).
    defaults = {"VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_LOGGING_LEVEL": "WARNING", "TQDM_DISABLE": "1"}
    forced = dict(forced or {})
    previous = {key: os.environ.get(key) for key in {**defaults, **forced}}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.update(forced)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_environment(device=None, dtype=None):
    """Check the platform and GPU; `dtype` is vllm_options['dtype'] as given."""
    if sys.platform != "linux":
        raise RuntimeError("the vLLM backend requires Linux and an NVIDIA CUDA GPU "
                           "(or an AMD GPU with ROCm)")
    if importlib.util.find_spec("vllm") is None:
        raise ImportError('install the optional backend with pip install "kittenml[vllm]"')
    selected = torch.device(device or "cuda")
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the vLLM backend requires an NVIDIA CUDA GPU (or an AMD GPU with ROCm)")
    if selected.index not in (None, 0):
        raise ValueError("select the GPU with CUDA_VISIBLE_DEVICES and use device='cuda'")
    from .rocm import check_amd_gpu, is_rocm
    if is_rocm():
        # AMD GPUs with a ROCm build of PyTorch and vLLM (rocm.py); the capability
        # numbers below are NVIDIA's.
        check_amd_gpu(dtype)
        check_torchcodec()
        return
    major, minor = torch.cuda.get_device_capability()
    if major < 7:
        raise RuntimeError(f"this GPU (SM{major}{minor}) is not supported by the vLLM backend, "
                           "which needs compute capability 7.0 or newer; use the default "
                           "backend on CPU (omit backend='vllm' and set device='cpu')")
    if major == 7 and dtype == "bfloat16":
        raise ValueError(f"SM{major}{minor} has no native bfloat16; use dtype='float16' "
                         "or 'float32', or leave dtype unset")
    if (major, minor) < (7, 5):
        # CUDA 12.9/13 PyTorch wheels omit Volta. Check before downloading weights;
        # a compatible CUDA 12.6 build is needed by both PyTorch and vLLM.
        arches = torch.cuda.get_arch_list()
        ptx = [int(a.removeprefix("compute_")) for a in arches
               if a.startswith("compute_") and a.removeprefix("compute_").isdigit()]
        if f"sm_{major}{minor}" not in arches and not any(a <= major * 10 + minor for a in ptx):
            raise RuntimeError(
                f"this PyTorch build does not support SM{major}{minor}; V100 needs PyTorch "
                "and vLLM built with CUDA 12.6 and SM70 support. See docs/vllm.md. "
                "The standard CUDA 12.9/13 wheels cannot run on V100")
    check_torchcodec()
    return major, minor


def options_for_gpu(options, capability):
    """Choose the measured CC 7.x path; preserve every explicit option and modern GPU path."""
    if not isinstance(capability, tuple) or capability[0] != 7:
        return options
    chosen = dict(options or {})
    if chosen.get("dtype") is None:
        chosen["dtype"] = "float16"
    if capability < (7, 5) and chosen.get("quantization") == "w4a16":
        raise ValueError("W4A16 requires compute capability 7.5 or newer; on V100 use "
                         "quantization=None")
    if capability >= (7, 5) and chosen["dtype"] == "float16" and "quantization" not in chosen:
        try:
            check_w4a16_environment()
        except RuntimeError:
            pass  # Preserve the caller's kernel overrides; dense float16 remains available.
        else:
            # All three modes tied on WER/drops; this was the fastest and fit 6 GB.
            chosen["quantization"] = "w4a16"
    return chosen


def check_torchcodec():
    """Refuse an installed torchcodec that would stop vLLM at start-up.

    vLLM imports torchcodec's audio decoder when it starts and carries on if that
    fails with ImportError or RuntimeError (no FFmpeg). A torchcodec built for a
    different CUDA version than PyTorch fails with OSError instead once FFmpeg is
    installed (PyPI's torchcodec 0.17.0 is a CUDA 13 build, so this happens with the
    CUDA 12.9 builds of vLLM and PyTorch), and every engine start fails on it.
    KittenTTS does not use torchcodec.
    """
    if importlib.util.find_spec("torchcodec") is None:
        return
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
    except (ImportError, RuntimeError):
        return
    except OSError as error:
        reason = (str(error).strip().splitlines() or [type(error).__name__])[0][:300]
        raise RuntimeError(
            f"torchcodec is installed but cannot load ({reason}). It is probably built for "
            "a different CUDA version than PyTorch, and vLLM stops on this when it starts. "
            "KittenTTS does not use torchcodec: remove it with `pip uninstall torchcodec` "
            "(or `uv pip uninstall torchcodec`)") from error


# Engine settings KittenTTS chooses for vLLM, and their defaults.
_ENGINE_DEFAULTS = dict(gpu_memory_utilization=0.5, max_model_len=4096,
                        max_num_seqs=8, enforce_eager=False)
# Options KittenTTS interprets itself. None keeps the default behaviour
# (for max_num_batched_tokens: vLLM's own default).
_KITTEN_OPTIONS = ("quantization", "dtype", "kv_cache_memory_bytes", "max_num_batched_tokens")
_QUANTIZATIONS = (None, "w4a16")
_DTYPES = {None: None, "bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}
# W4A16 runs on vLLM's Marlin kernel, the one whose exact output was verified.
# vLLM tries Machete first on Hopper (SM90) GPUs, so it is disabled for the
# engine; the loader then checks that every quantized layer uses Marlin.
_W4A16_KERNEL = "MarlinLinearKernel"
_W4A16_DISABLED_KERNELS = ("MacheteLinearKernel",)


def resolve_options(options):
    """Split `vllm_options` into engine keyword arguments and KittenTTS choices."""
    options = dict(options or {})
    unknown = set(options) - _ENGINE_DEFAULTS.keys() - set(_KITTEN_OPTIONS)
    if unknown:
        raise ValueError(f"unknown vllm_options: {sorted(unknown)}")
    kitten = {key: options.pop(key, None) for key in _KITTEN_OPTIONS}
    if kitten["quantization"] not in _QUANTIZATIONS:
        raise ValueError(f"vllm_options['quantization'] must be None or 'w4a16', "
                         f"not {kitten['quantization']!r}")
    if kitten["dtype"] not in _DTYPES:
        raise ValueError(f"vllm_options['dtype'] must be None, 'bfloat16', 'float16' or 'float32', "
                         f"not {kitten['dtype']!r}")
    for key in ("kv_cache_memory_bytes", "max_num_batched_tokens"):
        value = kitten[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                  or value <= 0):
            raise ValueError(f"vllm_options['{key}'] must be a positive integer")
    engine = dict(_ENGINE_DEFAULTS)
    engine.update(options)
    for key in ("kv_cache_memory_bytes", "max_num_batched_tokens"):
        if kitten[key] is not None:
            engine[key] = kitten[key]
    return engine, kitten


def check_options(options, weights):
    """Validate `vllm_options` before anything is downloaded or loaded."""
    defaults, kitten = resolve_options(options)
    if kitten["quantization"] == "w4a16":
        if kitten["dtype"] == "float32":
            raise ValueError("W4A16 needs bfloat16 or float16 activations, not float32")
        check_w4a16_environment()
    if weights == "emb4" and kitten["dtype"] == "float16":
        # The float16 rescale was measured on the exact embedding (fp16.py).
        raise ValueError("vllm_options['dtype']='float16' cannot be combined with "
                         "weights='emb4'; use weights='packed' or 'full'")
    return defaults, kitten


def check_w4a16_environment(environ=None):
    """Refuse settings that would change what W4A16 computes or which kernel runs it."""
    environ = os.environ if environ is None else environ
    refuse = "vllm_options['quantization']='w4a16' refuses to start while "
    if "VLLM_MARLIN_INPUT_DTYPE" in environ:
        raise RuntimeError(refuse + "VLLM_MARLIN_INPUT_DTYPE is set: it quantizes activations "
                           "to int8/fp8, turning W4A16 into W4A8. Unset it to run exact W4A16")
    if environ.get("VLLM_MARLIN_USE_ATOMIC_ADD") == "1":  # vLLM reads only "1" as on
        raise RuntimeError(refuse + "VLLM_MARLIN_USE_ATOMIC_ADD=1: it would let Marlin reduce "
                           "with unordered atomic adds. Unset it to run exact W4A16")
    disabled = environ.get("VLLM_DISABLED_KERNELS", "").split(",")
    kernel = _w4a16_kernel()
    if kernel in disabled:
        raise RuntimeError(refuse + f"VLLM_DISABLED_KERNELS names {kernel}, the only "
                           "kernel exact W4A16 is verified with. Remove it from the list")


def w4a16_engine_environment(environ=None):
    """VLLM_DISABLED_KERNELS for the engine: the caller's list plus Machete
    (on AMD GPUs: plus the RDNA3 kernel, rocm.py)."""
    environ = os.environ if environ is None else environ
    disabled = [k for k in environ.get("VLLM_DISABLED_KERNELS", "").split(",") if k]
    disabled += [k for k in _w4a16_disabled_kernels() if k not in disabled]
    return {"VLLM_DISABLED_KERNELS": ",".join(disabled)}


def _w4a16_kernel():
    """The kernel exact W4A16 is verified with on this platform."""
    from . import rocm
    return rocm.W4A16_KERNEL if rocm.is_rocm() else _W4A16_KERNEL


def _w4a16_disabled_kernels():
    from . import rocm
    return rocm.W4A16_DISABLED_KERNELS if rocm.is_rocm() else _W4A16_DISABLED_KERNELS


def _main_process_tensor(name):
    """Only these LM tensors are used outside vLLM: prompt embeddings and spk_proj."""
    return name == "model.embed_tokens.weight" or name.startswith("spk_proj.")


def quantization_args(quantization):
    """Engine arguments that select exact W4A16 (compressed-tensors, Marlin)."""
    if quantization is None:
        return {}
    from .w4a16 import quantization_config
    return {"quantization": "compressed-tensors",
            "hf_overrides": {"quantization_config": quantization_config()}}


def engine_args(quantization, float16):
    """quantization_args, plus the float16 rescale's model-config changes when asked."""
    args = quantization_args(quantization)
    if float16:
        from .fp16 import hf_overrides
        args["hf_overrides"] = {**args.get("hf_overrides", {}), **hf_overrides()}
    return args


# ── Fitting vLLM to the GPU ──────────────────────────────────────────────────
# With the defaults vLLM takes half the GPU, and this process needs memory next to
# it for audio decoding. On cards with 8 GB or less that does not fit, so there the
# settings the caller did not give are chosen from the free memory instead. Larger
# GPUs keep their defaults unless the CC 7.x path requests fitting; callers who
# set the memory share themselves keep it, with a warning when it is too small. The byte
# counts below were measured with vLLM 0.31.0 for KittenTTS 2 on RTX 4060 desktop
# and laptop GPUs (8 GB).
MiB = 1 << 20
# Cards with 8 GB or less (CUDA reports 7.6-7.8 GiB for 8 GB cards, 9.6 GiB or
# more for 10 GB ones). CC 7.x can also request fitting on larger cards.
_SMALL_GPU_TOTAL_BYTES = 9 << 30
# This process beside vLLM: the audio decoder (s3gen, 1,030 MiB in float32) and its
# working memory for the longest chunk, 1,000 tokens (2,824 MiB reserved at the
# peak with the default decoder; it grows faster than the chunk's length).
_MAIN_PROCESS_BYTES = 2816 * MiB
# vLLM's engine process: its CUDA context (about 125 MiB), taken before vLLM sizes
# its share.
_ENGINE_CONTEXT_BYTES = 256 * MiB
# Inside vLLM's share, besides weights and KV cache: memory outside PyTorch and
# CUDA graphs, plus the activation peak, which grows with max_num_batched_tokens
# (measured 0.15 GiB at 2,048 and 0.41 GiB at vLLM's default of 8,192).
_ENGINE_FIXED_BYTES = 256 * MiB
_ACTIVATION_BYTES_PER_TOKEN = 56 * 1024
# vLLM's default max_num_batched_tokens for the LLM class on GPUs under 70 GB.
_VLLM_BATCHED_TOKENS = 8192
# On the smallest GPUs: prefill in steps of this many tokens (a typical prompt
# still fits in one), which lowers the activation peak by 0.26 GiB.
_SMALL_GPU_BATCHED_TOKENS = 2048
# KV cache on small GPUs: room for this many full-length requests if it fits, and
# at least one (vLLM refuses a cache smaller than max_model_len) plus two blocks.
_KV_TARGET_REQUESTS = 4
_KV_MARGIN_TOKENS = 32
# The shortest context a small GPU may fall back to: a long reference prompt plus
# a full chunk still fits.
_MIN_MODEL_LEN = 2048


def lm_memory(lm_config, quantization=None, float16=False, float32=False):
    """(weight bytes on the GPU, KV-cache bytes per token) for a Qwen3 language model."""
    from ..ternary import GROUP_SIZE

    element_bytes = 4 if float32 else 2
    if float32 and quantization == "w4a16":
        raise ValueError("W4A16 needs bfloat16 or float16 activations, not float32")
    hidden, layers = lm_config.hidden_size, lm_config.num_hidden_layers
    heads, kv_heads = lm_config.num_attention_heads, lm_config.num_key_value_heads
    head_dim = getattr(lm_config, "head_dim", None) or hidden // heads
    projections = layers * (2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
                            + 3 * hidden * lm_config.intermediate_size)
    embedding = lm_config.vocab_size * hidden
    if quantization == "w4a16":   # 4-bit codes plus one bf16 scale per group
        projection_bytes = projections // 2 + projections // GROUP_SIZE * 2
    else:
        projection_bytes = projections * element_bytes
    weight_bytes = element_bytes * embedding + projection_bytes
    if float16 or not getattr(lm_config, "tie_word_embeddings", True):
        weight_bytes += element_bytes * embedding   # a separate output head
    return weight_bytes, 2 * layers * kv_heads * head_dim * element_bytes


def _engine_bytes(weights, batched_tokens, float32=False):
    """Everything in vLLM's share except the KV cache."""
    # Float32 also doubles the activation storage (0.25 GiB measured at 2,048
    # tokens on RTX 2060, including vLLM's profiling overhead).
    activation_bytes = _ACTIVATION_BYTES_PER_TOKEN * (2 if float32 else 1)
    return weights + _ENGINE_FIXED_BYTES + activation_bytes * batched_tokens


def fit_to_gpu(options, free_bytes, total_bytes, lm_config, float16=False, w4a16=True,
               force=False):
    """`vllm_options` with what the caller left out chosen for this GPU's free memory.

    Returns (options, report). The options come back unchanged (report["fitted"]
    False) when they fit: vLLM's share leaves this process room for audio decoding
    and holds a KV cache for one full request. They also come back unchanged on
    GPUs with more than 8 GB (unless `force=True`), and when the caller set the memory share
    (`gpu_memory_utilization` or `kv_cache_memory_bytes`); if they do not fit there,
    report["suggested"] holds settings that would. Otherwise, on a card with 8 GB or
    less or with `force=True`, the options the caller did not give (even None counts as given) are
    chosen: 4-bit weights (`w4a16`, when allowed) and a share sized for the
    weights plus a KV cache of up to four full requests; then prefill in 2,048-token
    steps and a shorter `max_model_len` (not below 2,048), only if one full request
    would not fit without them. Raises RuntimeError when not even that fits,
    saying how much free memory is needed.
    """
    given = dict(options or {})
    engine, kitten = resolve_options(given)
    room = free_bytes - _ENGINE_CONTEXT_BYTES - _MAIN_PROCESS_BYTES
    report = {"free_mib": free_bytes // MiB, "total_mib": total_bytes // MiB,
              "room_for_vllm_mib": max(room, 0) // MiB, "fitted": False}
    float32 = kitten["dtype"] == "float32"
    w4a16 = w4a16 and not float32
    weights, kv_token = lm_memory(lm_config, kitten["quantization"], float16, float32)
    fixed = _engine_bytes(weights, engine.get("max_num_batched_tokens", _VLLM_BATCHED_TOKENS),
                          float32)
    share = int(engine["gpu_memory_utilization"] * total_bytes)
    kv = kitten["kv_cache_memory_bytes"] or share - fixed
    used = share if kitten["kv_cache_memory_bytes"] is None else kv + fixed
    one_request = (engine["max_model_len"] + _KV_MARGIN_TOKENS) * kv_token
    if used <= room and kv >= one_request:
        return given, report
    report["decoding_room_mib"] = max(free_bytes - _ENGINE_CONTEXT_BYTES - used, 0) // MiB
    report["cache_short"] = kv < one_request
    own_share = "gpu_memory_utilization" in given or kitten["kv_cache_memory_bytes"] is not None
    if (total_bytes > _SMALL_GPU_TOTAL_BYTES and not force) or own_share:
        # Not fitted: a larger GPU keeps vLLM's defaults, and a share the caller set
        # stays theirs. Only suggest what would fit.
        mine = {k: v for k, v in given.items()
                if k not in ("gpu_memory_utilization", "kv_cache_memory_bytes")}
        try:
            report["suggested"] = _choose(mine, free_bytes, total_bytes, lm_config, float16,
                                          w4a16, {})[0]
        except RuntimeError:
            report["suggested"] = None
        return given, report
    fitted, chosen_report = _choose(given, free_bytes, total_bytes, lm_config, float16, w4a16,
                                    report)
    report.update(chosen_report, fitted=True,
                  chosen={k: v for k, v in fitted.items() if k not in given})
    return fitted, report


def _choose(given, free_bytes, total_bytes, lm_config, float16, w4a16, report):
    """The small-GPU choices for the options not in `given` (see fit_to_gpu)."""
    fitted, report = dict(given), dict(report)
    room = free_bytes - _ENGINE_CONTEXT_BYTES - _MAIN_PROCESS_BYTES
    if "quantization" not in given and w4a16:
        fitted["quantization"] = "w4a16"
    engine, kitten = resolve_options(fitted)
    float32 = kitten["dtype"] == "float32"
    weights, kv_token = lm_memory(lm_config, kitten["quantization"], float16,
                                  float32)
    one_request = (engine["max_model_len"] + _KV_MARGIN_TOKENS) * kv_token
    # vLLM sizes its share as ceil(total * utilization), so a utilization rounded
    # down to 1e-6 loses at most total / 10^6 bytes of the planned cache.
    step = total_bytes / 1e6
    if "max_num_batched_tokens" not in given and \
            room - _engine_bytes(weights, _VLLM_BATCHED_TOKENS, float32) - step < one_request:
        # vLLM's default step is kept when a full request fits with it: the step size
        # selects which compiled kernels run, so keeping it gives the same output as
        # quantization="w4a16" with the defaults (same speed either way, measured).
        fitted["max_num_batched_tokens"] = _SMALL_GPU_BATCHED_TOKENS
        engine, kitten = resolve_options(fitted)
    fixed = _engine_bytes(weights, engine.get("max_num_batched_tokens", _VLLM_BATCHED_TOKENS),
                          float32)
    available = room - fixed
    max_len = engine["max_model_len"]
    if "max_model_len" not in given:
        while (max_len > _MIN_MODEL_LEN
               and (max_len + _KV_MARGIN_TOKENS) * kv_token > available - step):
            max_len = max(_MIN_MODEL_LEN, max_len - 512)
        if max_len != engine["max_model_len"]:
            fitted["max_model_len"] = max_len
    one_request = (max_len + _KV_MARGIN_TOKENS) * kv_token
    if available - step < one_request:
        needed = free_bytes - available + one_request + step
        raise RuntimeError(
            f"not enough free GPU memory for the vLLM backend: about {needed // MiB:,.0f} MiB "
            f"free is needed and {free_bytes // MiB:,} MiB is free. Close other programs "
            "using the GPU, use a GPU with more memory, or run the default backend on CPU "
            "(omit backend='vllm' and set device='cpu')")
    kv = min(available, _KV_TARGET_REQUESTS * (max_len + _KV_MARGIN_TOKENS) * kv_token)
    # Rounded down so vLLM's share never exceeds what was measured as free.
    fitted["gpu_memory_utilization"] = math.floor((fixed + kv) / total_bytes * 1e6) / 1e6
    report["planned_kv_cache_tokens"] = int(kv // kv_token)
    report["full_requests"] = int(kv // one_request)
    return fitted, report


def _stop_engine(backend_ref):
    """Stop a VLLMBackend's engine process at interpreter exit (see VLLMBackend)."""
    engine = getattr(backend_ref(), "engine", None)
    if engine is None:
        return
    try:
        engine.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001 - nothing to report at exit
        pass


def _free_gpu_memory():
    """(free, total) bytes on the current GPU, as this process sees them."""
    return torch.cuda.mem_get_info()


class VLLMBackend:
    def __init__(self, owner, options=None):
        options = dict(options or {})
        capability = validate_environment(owner.device, options.get("dtype"))
        options = options_for_gpu(options, capability)
        from transformers import AutoConfig, AutoTokenizer
        from ..speaker import load_speaker_head

        check_options(options, owner.weights)
        lm_dir = owner.repo_dir / owner.config.get("lm_dir", "lm")
        config = AutoConfig.from_pretrained(str(lm_dir))
        if config.model_type != "qwen3":
            raise ValueError("the vLLM backend currently supports Qwen3 KittenTTS 2 checkpoints")
        # Explicit dtypes have known weight/cache sizes; preserve the existing
        # behavior for custom model configurations without an explicit dtype.
        half = owner.config.get("dtype", "bf16") == "bf16" or options.get("dtype") is not None
        legacy = isinstance(capability, tuple) and capability[0] == 7
        if legacy:
            options, self.auto_fit = self._fit(options, config, half,
                                               legacy_capability=capability)
        else:
            options, self.auto_fit = self._fit(options, config, half)
        defaults, kitten = check_options(options, owner.weights)
        quantization = kitten["quantization"]
        forced_environment = w4a16_engine_environment() if quantization == "w4a16" else None
        default_dtype = (torch.bfloat16 if owner.config.get("dtype", "bf16") == "bf16"
                         else torch.float32)
        dtype = _DTYPES[kitten["dtype"]] or default_dtype
        # CC 7.x defaults to the rescaled model; modern GPUs keep their original dtype.
        float16 = dtype is torch.float16
        if quantization == "w4a16" and dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"W4A16 needs bfloat16 or float16 activations, not {dtype}")
        if legacy:
            mode = "rounded float16" if float16 else "float32"
            if quantization == "w4a16":
                mode += " with 4-bit weights"
            # Preflight runs twice; warn only here, once the actual mode is known.
            logger.warning(
                "KittenTTS: SM%s%s has no native bfloat16; using %s. This is supported, "
                "not optimal: audio can differ from RTX 30 and newer. See docs/vllm.md "
                "for measured quality, speed and memory", *capability, mode)
        self.max_model_len = defaults["max_model_len"]
        self.token_map = owner.token_map
        owner.dtype = dtype
        owner.tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
        # vLLM's own loader reads the full LM; this process only needs the
        # embedding and the speaker head, so do not expand the rest here.
        state = owner._load_lm_weights(lm_dir, include=_main_process_tensor)
        if float16:
            # Prompt embeddings at the engine's residual scale, built exactly as
            # the engine's own embedding table; the speaker head runs in float32.
            from .fp16 import ScaledSpeakerHead, scale_embedding
            self.embedding = scale_embedding(state["model.embed_tokens.weight"])
            self.spk_proj = ScaledSpeakerHead(load_speaker_head(
                config.hidden_size, str(lm_dir), owner.device, torch.float32,
                spk_dim=owner.config.get("speaker_embedding_dim", 512), state=state))
        else:
            self.embedding = state["model.embed_tokens.weight"].to(owner.dtype)
            self.spk_proj = load_speaker_head(config.hidden_size, str(lm_dir),
                                              owner.device, owner.dtype,
                                              spk_dim=owner.config.get("speaker_embedding_dim",
                                                                       512),
                                              state=state)
        del state
        loader_config = {"repo_dir": str(owner.repo_dir), "config": owner.config,
                         "weights": owner.weights}
        if quantization is not None:
            loader_config["quantization"] = quantization
        if float16:
            loader_config["float16_rescale"] = True
        with _engine_environment(forced_environment):
            import vllm.envs
            from vllm import LLM
            from vllm.config import LoggingConfig

            # After an engine has started in this process (single-process mode),
            # vLLM answers from a frozen copy of the environment; drop it so this
            # engine sees the settings checked above.
            vllm.envs.disable_envs_cache()
            self.engine = LLM(
                model=str(lm_dir), tokenizer=str(lm_dir), dtype=owner.dtype,
                load_format="kittenml",
                model_loader_extra_config=loader_config,
                enable_prompt_embeds=True, enable_prefix_caching=False,
                logits_processors=["kittenml.kittentts2.vllm.logits:KittenLogitsProcessor"],
                logging_config=LoggingConfig(), disable_log_stats=True,
                **engine_args(quantization, float16), **defaults)
        # vLLM stops its engine process from a weakref finalizer at exit. If Python's
        # multiprocessing exit handler runs first, it waits for that process, which is
        # still waiting for work, so a finished script never returns. Exit handlers run
        # last-registered first, so this one stops the engine before either of them.
        atexit.register(_stop_engine, weakref.ref(self))

    @staticmethod
    def _fit(options, lm_config, half=True, legacy_capability=None):
        """fit_to_gpu with this GPU's free memory, and its warnings.

        The caller's options are returned unchanged if the free memory cannot be
        read or a custom model's dtype is unknown. Explicit float32 also sizes its
        weights, cache and activations at four bytes per element.
        """
        if not half:
            return options, {"fitted": False, "skipped": "not a 16-bit model"}
        try:
            free, total = _free_gpu_memory()
        except Exception as error:  # noqa: BLE001 - fitting is an aid, not a requirement
            return options, {"fitted": False, "error": repr(error)[:200]}
        float16 = options.get("dtype") == "float16"
        # 4-bit is chosen automatically only where its kernel was verified (Marlin on NVIDIA,
        # vLLM's RDNA kernel on Radeon RX 7000 and RX 9000, rocm.py), and only when the
        # environment would not refuse it.
        from . import rocm
        w4a16 = not rocm.is_rocm() or rocm.has_w4a16_kernel()
        if legacy_capability is not None and legacy_capability < (7, 5):
            w4a16 = False
        try:
            check_w4a16_environment()
        except RuntimeError:
            w4a16 = False
        fitted, report = fit_to_gpu(options, free, total, lm_config, float16=float16,
                                    w4a16=w4a16, force=legacy_capability is not None)
        if report["fitted"]:
            requests = report["full_requests"]
            limits = [f"the cache holds about {report['planned_kv_cache_tokens']:,} tokens "
                      f"({requests} full-length request{'s' if requests != 1 else ''} at a time)"]
            if "max_model_len" in report["chosen"]:
                limits.append(f"the context is {report['chosen']['max_model_len']:,} tokens, "
                              "so longer requests raise an error")
            logger.warning(
                "KittenTTS: %s MiB of GPU memory free; vLLM's defaults do not fit next to audio "
                "decoding, so using %s (set these in vllm_options to choose them yourself). This "
                "is supported, but not the optimal path: %s", f"{free // MiB:,}",
                ", ".join(f"{k}={v!r}" for k, v in report["chosen"].items()), "; ".join(limits))
        elif "suggested" in report:
            reason = ("the cache cannot hold one full-length request" if report["cache_short"]
                      else f"about {report['decoding_room_mib']:,} MiB of GPU memory is left for "
                           f"audio decoding, which needs up to {_MAIN_PROCESS_BYTES // MiB:,} MiB")
            hint = (f" Settings that fit the free memory now: vllm_options={report['suggested']!r}"
                    if report["suggested"] else "")
            logger.warning("KittenTTS: with these vLLM settings %s; nothing was changed.%s",
                           reason, hint)
        return fitted, report

    def prepare_request(self, prompt, speaker_embedding, settings, max_new_tokens, advanced):
        from vllm import SamplingParams

        if len(prompt) + int(max_new_tokens) > self.max_model_len:
            raise ValueError("prompt and token budget exceed vllm_options['max_model_len']; "
                             "increase it or shorten the reference/token budget")
        with torch.inference_mode():
            embeds = self.embedding[torch.tensor(prompt, dtype=torch.long)].clone()
            embeds[0] = self.spk_proj(speaker_embedding).squeeze(0).cpu()
        sampling = {k: settings[k] for k in ("temperature", "top_k", "top_p")}
        sampling["min_p"] = settings.get("min_p", 0.0)
        sampling.update({k: advanced[k] for k in ("repetition_penalty", "repetition_window",
                                                  "token_run_penalty", "token_run_grace")})
        seed = advanced.get("seed")
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        params = SamplingParams(
            temperature=1.0, top_k=-1, top_p=1.0, min_p=0.0,
            repetition_penalty=1.0, max_tokens=int(max_new_tokens),
            stop_token_ids=[self.token_map.speech_end_id, self.token_map.stop_id],
            ignore_eos=True, detokenize=False, seed=int(seed),
            extra_args={"kittenml": {"prompt": prompt, "token_map": asdict(self.token_map),
                                      "sampling": sampling}})
        return {"prompt_embeds": embeds}, params

    def generate_tokens(self, prompt, speaker_embedding, settings, max_new_tokens, advanced):
        request, params = self.prepare_request(prompt, speaker_embedding, settings,
                                               max_new_tokens, advanced)
        output = self.engine.generate([request], [params], use_tqdm=False)[0]
        low, high = self.token_map.audio_id_base, self.token_map.audio_id_end
        return [t - low for t in output.outputs[0].token_ids if low <= t < high]
