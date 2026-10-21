"""Optional GPU language-model backend; waveform decoding stays in KittenTTS."""

from dataclasses import asdict
import contextlib
import importlib.util
import os
import sys

import torch


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
        raise RuntimeError("the vLLM backend requires Linux and an NVIDIA CUDA GPU")
    if importlib.util.find_spec("vllm") is None:
        raise ImportError('install the optional backend with pip install "kittenml[vllm]"')
    selected = torch.device(device or "cuda")
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the vLLM backend requires an NVIDIA CUDA GPU")
    if selected.index not in (None, 0):
        raise ValueError("select the GPU with CUDA_VISIBLE_DEVICES and use device='cuda'")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (7, 5):
        raise RuntimeError(f"this GPU (SM{major}{minor}) is not supported by the vLLM backend, "
                           "which needs compute capability 7.5 or newer")
    if major < 8 and dtype != "float16":
        raise RuntimeError(f"this GPU (SM{major}{minor}) has no bfloat16 support, which "
                           f"KittenTTS's language model runs in by default. {_FLOAT16_OPT_IN}")


# Engine settings KittenTTS chooses for vLLM, and their defaults.
_ENGINE_DEFAULTS = dict(gpu_memory_utilization=0.5, max_model_len=4096,
                        max_num_seqs=8, enforce_eager=False)
# Options KittenTTS interprets itself. None keeps the default behaviour.
_KITTEN_OPTIONS = ("quantization", "dtype", "kv_cache_memory_bytes")
_QUANTIZATIONS = (None, "w4a16")
_DTYPES = {None: None, "bfloat16": torch.bfloat16, "float16": torch.float16}
# W4A16 runs on vLLM's Marlin kernel, the one whose exact output was verified.
# vLLM tries Machete first on Hopper (SM90) GPUs, so it is disabled for the
# engine; the loader then checks that every quantized layer uses Marlin.
_W4A16_KERNEL = "MarlinLinearKernel"
_W4A16_DISABLED_KERNELS = ("MacheteLinearKernel",)
# GPUs without bfloat16 (compute capability 7.5) can opt in to float16, a
# rescaled fallback with a precision change (fp16.py).
_FLOAT16_OPT_IN = ("To run on it anyway, opt in to the float16 fallback with "
                   "vllm_options={'dtype': 'float16'}: it is supported but not the optimal "
                   "path (a precision change from bfloat16; see the vLLM backend docs)")


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
        raise ValueError(f"vllm_options['dtype'] must be None, 'bfloat16' or 'float16', "
                         f"not {kitten['dtype']!r}")
    memory = kitten["kv_cache_memory_bytes"]
    if memory is not None and (isinstance(memory, bool) or not isinstance(memory, int)
                               or memory <= 0):
        raise ValueError("vllm_options['kv_cache_memory_bytes'] must be a positive integer")
    engine = dict(_ENGINE_DEFAULTS)
    engine.update(options)
    if memory is not None:
        engine["kv_cache_memory_bytes"] = memory
    return engine, kitten


def check_options(options, weights):
    """Validate `vllm_options` before anything is downloaded or loaded."""
    defaults, kitten = resolve_options(options)
    if kitten["quantization"] == "w4a16":
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
    if _W4A16_KERNEL in disabled:
        raise RuntimeError(refuse + f"VLLM_DISABLED_KERNELS names {_W4A16_KERNEL}, the only "
                           "kernel exact W4A16 is verified with. Remove it from the list")


def w4a16_engine_environment(environ=None):
    """VLLM_DISABLED_KERNELS for the engine: the caller's list plus Machete."""
    environ = os.environ if environ is None else environ
    disabled = [k for k in environ.get("VLLM_DISABLED_KERNELS", "").split(",") if k]
    disabled += [k for k in _W4A16_DISABLED_KERNELS if k not in disabled]
    return {"VLLM_DISABLED_KERNELS": ",".join(disabled)}


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


class VLLMBackend:
    def __init__(self, owner, options=None):
        validate_environment(owner.device, dict(options or {}).get("dtype"))
        from transformers import AutoConfig, AutoTokenizer
        from ..speaker import load_speaker_head

        defaults, kitten = check_options(options, owner.weights)
        quantization = kitten["quantization"]
        forced_environment = w4a16_engine_environment() if quantization == "w4a16" else None
        default_dtype = (torch.bfloat16 if owner.config.get("dtype", "bf16") == "bf16"
                         else torch.float32)
        dtype = _DTYPES[kitten["dtype"]] or default_dtype
        # Only an explicit opt-in runs the rescaled float16 model (fp16.py).
        float16 = dtype is torch.float16
        if quantization == "w4a16" and dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"W4A16 needs bfloat16 or float16 activations, not {dtype}")
        self.max_model_len = defaults["max_model_len"]
        self.token_map = owner.token_map
        lm_dir = owner.repo_dir / owner.config.get("lm_dir", "lm")
        config = AutoConfig.from_pretrained(str(lm_dir))
        if config.model_type != "qwen3":
            raise ValueError("the vLLM backend currently supports Qwen3 KittenTTS 2 checkpoints")
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
