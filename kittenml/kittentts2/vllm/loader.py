"""vLLM plugin: selected KittenTTS weights, without an intermediate export."""

from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from ..weights import iter_lm_weights


class KittenModelLoader(BaseModelLoader):
    def download_model(self, model_config):
        # KittenTTS resolves the selected files before constructing the engine.
        pass

    def load_weights(self, model, model_config):
        options = self.load_config.model_loader_extra_config
        quantization = options.get("quantization")
        # The opt-in float16 fallback for GPUs without bfloat16 (fp16.py).
        float16 = options.get("float16_rescale", False)
        if float16:
            _check_float16_model(model, model_config)
        if quantization == "w4a16":
            # vLLM chose each layer's kernel when it built the model; check before loading.
            _check_w4a16_kernels(model, model_config, float16)
            # Exact 4-bit weights straight from the trits, one tensor at a time.
            from .w4a16 import iter_w4a16
            weights = iter_w4a16(options["repo_dir"], options["config"], options["weights"])
        elif quantization is None:
            state = iter_lm_weights(options["repo_dir"], options["config"], options["weights"])
            weights = ((name, tensor) for name, tensor in state
                       if not name.startswith("spk_proj."))
        else:
            raise ValueError(f"unknown KittenTTS quantization {quantization!r}")
        if float16:
            from .fp16 import rescale_weights
            weights = rescale_weights(weights)
        loaded = model.load_weights(weights)
        missing = set(dict(model.named_parameters())) - loaded - _kv_cache_scale_names(model)
        if missing:
            raise ValueError(f"weights missing from the checkpoint: {sorted(missing)[:5]}")
        if float16:
            from .fp16 import scale_norm_epsilons
            config = model_config.hf_config
            scale_norm_epsilons(model, config.num_hidden_layers, config.rms_norm_eps)


def _check_float16_model(model, model_config):
    """The float16 rescale needs a float16 engine with its own, unscaled output head."""
    import torch

    refuse = "float16 rescale: refusing to load: "
    if model_config.dtype != torch.float16:
        raise RuntimeError(refuse + f"the engine dtype is {model_config.dtype}, not float16")
    head = getattr(getattr(model, "lm_head", None), "weight", None)
    embedding = getattr(getattr(getattr(model, "model", None), "embed_tokens", None),
                        "weight", None)
    if getattr(model_config.hf_config, "tie_word_embeddings", True) or head is None \
            or head is embedding:
        raise RuntimeError(refuse + "the output head is tied to the scaled embedding")


def _quantized_layers(model):
    """(name, module, kernel) for every linear layer vLLM built with a mixed-precision kernel."""
    for name, module in model.named_modules():
        kernel = getattr(getattr(module, "scheme", None), "kernel", None)
        if kernel is not None:
            yield name, module, kernel


def _check_w4a16_kernels(model, model_config, float16=False):
    """Every quantized linear layer must run exact W4A16 on Marlin.

    vLLM picks the mixed-precision kernel per GPU (Machete on Hopper, Marlin
    elsewhere, others when a kernel is disabled). Exact W4A16 was verified with
    Marlin only, so refuse to load if vLLM chose anything else. This runs in the
    engine process: VLLM_MARLIN_INPUT_DTYPE keeps the Marlin kernel but quantizes
    its activations to 8 bits (W4A8), which shows in the kernel's activation type
    (in vLLM 0.31.0 that is the only way the setting acts).
    """
    import torch
    from vllm.model_executor.kernels.linear import MarlinLinearKernel

    refuse = "W4A16: refusing to load: "
    # bfloat16, or float16 in the opt-in float16 mode (its group scales are rescaled).
    expected = torch.float16 if float16 else torch.bfloat16
    if model_config.dtype != expected:
        raise RuntimeError(refuse + f"the engine dtype is {model_config.dtype}, "
                           f"not {str(expected).removeprefix('torch.')}")
    chosen, activations = {}, {}
    for name, _, kernel in _quantized_layers(model):
        chosen.setdefault(type(kernel).__name__, []).append(name)
        act_type = getattr(getattr(kernel, "config", None), "act_type", None)
        activations.setdefault(str(act_type), []).append(name)
    if not chosen:
        raise RuntimeError("W4A16: vLLM created no quantized linear layers")
    other = {kind: names for kind, names in chosen.items()
             if kind != MarlinLinearKernel.__name__}
    if other:
        layers = sum(map(len, other.values()))
        raise RuntimeError(f"W4A16: vLLM chose {sorted(other)} for {layers} linear layers, but "
                           "exact W4A16 is verified only with MarlinLinearKernel (check "
                           "VLLM_DISABLED_KERNELS; with VLLM_ENABLE_V1_MULTIPROCESSING=0 vLLM "
                           "may still use the value from the first engine in this process)")
    wrong = {act: names for act, names in activations.items() if act != str(model_config.dtype)}
    if wrong:
        layers = sum(map(len, wrong.values()))
        raise RuntimeError(refuse + f"{layers} linear layers would quantize their activations to "
                           f"{sorted(wrong)} instead of computing in {model_config.dtype} (check "
                           "VLLM_MARLIN_INPUT_DTYPE)")


def _kv_cache_scale_names(model):
    """Attention KV-cache scales a quantization method adds, which the checkpoint omits.

    With quantization="compressed-tensors", every attention layer gets q/k/v scale
    and zero-point parameters for a quantized KV cache. KittenTTS keeps the KV cache
    in the model dtype, so they keep vLLM's defaults (as with vLLM's own loader);
    every other parameter must still come from the checkpoint.
    """
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    names = set()
    for prefix, module in model.named_modules():
        if isinstance(getattr(module, "quant_method", None), BaseKVCacheMethod):
            names.update(f"{prefix}.{name}" if prefix else name
                         for name, _ in module.named_parameters(recurse=False))
    return names


def register():
    from vllm.model_executor.model_loader import register_model_loader
    register_model_loader("kittenml")(KittenModelLoader)
