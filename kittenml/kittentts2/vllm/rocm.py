"""AMD GPUs (ROCm) for the vLLM backend: the GPU check, and which kernel runs 4-bit weights.

On NVIDIA GPUs vLLM runs KittenTTS's 4-bit (W4A16) layers with its Marlin kernel. Marlin is
written with NVIDIA-only instructions (tensor-core ``mma.sync``, ``ldmatrix`` and ``cp.async`` in
inline PTX), so vLLM does not build it for ROCm. On AMD GPUs vLLM 0.31.0 tries, in this order:

- ``RDNA3W4A16LinearKernel`` (gfx1100, Radeon RX 7900 only). It splits every dot product into
  256-wide slices and adds each slice's float32 result into the bfloat16 output with an atomic
  add, so a sum is rounded to bfloat16 once per slice (8 to 24 times for this model) and in an
  order that changes from run to run. KittenTTS disables it.
- ``RDNAHybridW4A16LinearKernel`` (gfx11 and gfx12: Radeon RX 7000 and RX 9000). For up to 5
  rows (decoding), a HIP kernel keeps one float32 sum per output over the whole row and adds the
  lanes of a wave in a fixed order; for more rows (prefill), a Triton kernel that also sums in
  float32. KittenTTS allows this one, after the check below.
- Triton, Conch and Exllama kernels: not checked, so 4-bit is refused (e.g. on Instinct GPUs).

``probe_w4a16`` runs in vLLM's engine process once the kernel has repacked its weights, on two
layers (one attention, one MLP projection), through both of the kernel's paths (1 and 64 rows):

1. One-hot rows: each output is a single weight times 1 + 2^-7 (a value 8-bit activation formats
   cannot hold), so it must equal that product rounded once, bit for bit. A wrong unpacking or
   group scale, or quantized activations, change it.
2. Random rows, run twice: the results must be bit-identical (a fixed summation order), and
   within one unit in the last place plus 2^-14 of sum(|w*x|) of the exact sum (float32
   accumulation; rounding the running sum to bfloat16 misses this by several units).
"""

import torch

# The kernel KittenTTS allows for 4-bit weights on AMD GPUs, and the one it disables.
W4A16_KERNEL = "RDNAHybridW4A16LinearKernel"
W4A16_DISABLED_KERNELS = ("RDNA3W4A16LinearKernel",)
# One attention and one MLP projection (inputs 2,048 and 6,144 wide).
PROBE_LAYERS = ("model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj")
# vLLM's hybrid kernel decodes up to 5 rows with its HIP kernel and prefills more with Triton.
_PROBE_ROWS = (1, 64)
_ONE_HOT = 1 + 2 ** -7
_TOLERANCE = 2 ** -14


def is_rocm():
    """True when PyTorch is a ROCm (AMD) build; the GPU is then still called "cuda"."""
    return bool(getattr(torch.version, "hip", None))


def check_amd_gpu(dtype=None):
    """Refuse AMD GPUs the installed ROCm build of PyTorch (and so vLLM) has no code for."""
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    built = sorted({name.split(":")[0] for name in torch.cuda.get_arch_list()})
    if built and arch not in built:
        raise RuntimeError(f"this AMD GPU ({arch}) is not supported by the installed ROCm build "
                           f"of PyTorch and vLLM, which has code for {', '.join(built)} "
                           "(for Radeon: RX 7900 and RX 9000 series)")
    if dtype == "float16":
        raise RuntimeError("vllm_options['dtype']='float16' is not supported on AMD GPUs; "
                           "the AMD GPUs that vLLM supports have bfloat16, so leave dtype out")


def has_w4a16_kernel():
    """True on the AMD GPUs the allowed 4-bit kernel covers: gfx11 and gfx12 (Radeon RX 7000
    and RX 9000 series). gfx1250 is excluded: vLLM counts it with its data-center GPUs
    (platforms/rocm.py, _ON_CDNA) and its RDNAHybrid kernel refuses it."""
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001 - no GPU to ask: no automatic 4-bit
        return False
    return arch.startswith(("gfx11", "gfx12")) and not arch.startswith("gfx1250")


def keep_references(weights, references):
    """Pass the weights through, keeping a CPU copy of the probed layers' 4-bit tensors."""
    wanted = {f"{layer}.{part}" for layer in PROBE_LAYERS
              for part in ("weight_packed", "weight_scale")}
    for name, tensor in weights:
        if name in wanted:
            references[name] = tensor.detach().to("cpu", copy=True)
        yield name, tensor


def dequantize(packed, scale):
    """compressed-tensors uint4b8 words and group scales -> the float32 weight they encode."""
    from .w4a16 import GROUP_SIZE, NUM_BITS, OFFSET, PACK_FACTOR

    words = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(PACK_FACTOR, dtype=torch.int64) * NUM_BITS
    codes = (words.unsqueeze(-1) >> shifts) & ((1 << NUM_BITS) - 1)
    signs = codes.reshape(packed.shape[0], -1) - OFFSET
    # Exact: every code here times its group's scale is a product of two small floats.
    return signs.to(torch.float32) * scale.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=1)


def _ulp(values, dtype):
    """One unit in the last place of `dtype` at each value's magnitude (0 for 0)."""
    _, exponent = torch.frexp(values)
    ulp = torch.finfo(dtype).eps * torch.exp2((exponent - 1).to(values.dtype))
    return torch.where(values == 0, torch.zeros_like(ulp), ulp)


def probe_w4a16(model, references, dtype):
    """Raise RuntimeError unless the 4-bit kernel computes the probed layers as described above."""
    from .w4a16 import bit_equal

    modules = dict(model.named_modules())
    generator = torch.Generator().manual_seed(0)
    report = {}
    for layer in PROBE_LAYERS:
        module = modules[layer]
        kernel = type(module.scheme.kernel).__name__
        weight = dequantize(references[f"{layer}.weight_packed"],
                            references[f"{layer}.weight_scale"])
        out_features, in_features = weight.shape
        device = next(module.parameters()).device

        def run(x):
            with torch.inference_mode():
                return module.scheme.apply_weights(module, x.to(device), None).to("cpu")

        def refuse(what):
            raise RuntimeError(f"W4A16: refusing to load: {kernel} {what} for {layer}; 4-bit "
                               "weights are not available on this GPU: use bf16 weights with "
                               "vllm_options={'quantization': None}")

        rows = max(_PROBE_ROWS)
        picked = torch.linspace(0, in_features - 1, rows).round().long()
        one_hot = torch.zeros(rows, in_features, dtype=dtype)
        one_hot[torch.arange(rows), picked] = _ONE_HOT
        expected = (weight[:, picked].T * _ONE_HOT).to(dtype)
        random = torch.randn(rows, in_features, generator=generator).to(dtype)
        exact = random.double() @ weight.double().T
        bound = random.double().abs() @ weight.double().abs().T
        worst = 0.0
        for count in _PROBE_ROWS:
            got = run(one_hot[:count])
            if not bit_equal(got, expected[:count]):
                wrong = int((got.float() != expected[:count].float()).sum())
                refuse(f"does not reproduce the 4-bit weights exactly ({wrong} of "
                       f"{got.numel()} one-hot outputs differ, {count} rows)")
            got = run(random[:count])
            if not bit_equal(got, run(random[:count])):
                refuse(f"gives different results for the same input ({count} rows)")
            error = (got.double() - exact[:count]).abs()
            allowed = _ulp(exact[:count], dtype) + _TOLERANCE * bound[:count]
            if bool((error > allowed).any()):
                over = int((error > allowed).sum())
                refuse(f"does not sum in float32 ({over} of {error.numel()} outputs are further "
                       f"from the exact result than one unit in the last place, {count} rows)")
            worst = max(worst, float((error / allowed.clamp_min(1e-30)).max()))
        report[layer] = {"kernel": kernel, "shape": [out_features, in_features],
                         "worst_error_vs_allowed": round(worst, 4)}
    return report
