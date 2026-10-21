"""Exact 4-bit (W4A16) weights for the vLLM backend.

The language model's 196 linear projections are ternary: every 128-wide input
group of every output row holds only {-s, 0, +s}, with one bf16 scale per group
(see ternary.py). That is exactly what a symmetric 4-bit, group-128 weight can
represent: compressed-tensors' ``uint4b8`` type stores ``q = sign + 8``, so the
codes are 7, 8 and 9, and dequantising gives ``(q - 8) * s``, the original bf16
weight bit for bit. The group scales are copied through unchanged.

vLLM then runs these layers with its Marlin mixed-precision kernel: 4-bit
weights, bf16 activations (the backend and loader keep vLLM from choosing
another kernel or 8-bit activations). Everything that is not ternary stays at
its stored precision: the token embedding and the tied output head, all norms,
and the KV cache. ``spk_proj`` runs outside vLLM and is dropped here.

With the lossy ``emb4`` packing the projections are still exact (its TL2 body
holds the same trits); only its 4-bit embedding, and so the tied output head,
differs from the published bf16 model.

In the opt-in float16 mode (fp16.py) the same codes are used; the group scales
of v/o/up/down are multiplied by that mode's powers of two and stored in
float16, which must be exact.

vLLM fuses q/k/v into ``qkv_proj`` and gate/up into ``gate_up_proj``
(``Qwen3ForCausalLM.packed_modules_mapping``), so the 196 source projections
(7 per layer) become 4 quantized modules per layer: 112 for the 28-layer model.

This module does not import vLLM, so it can be tested on any machine.
"""

import copy
import json
import re

import torch

from .. import ternary
from ..weights import plain_shards, weights_path

NUM_BITS = 4
GROUP_SIZE = ternary.GROUP_SIZE  # 128, as in the checkpoint
OFFSET = 1 << (NUM_BITS - 1)  # 8: uint4b8 stores sign + 8
PACK_FACTOR = 32 // NUM_BITS  # 8 codes per int32

# The compressed-tensors config vLLM reads from hf_config.quantization_config.
# Targets match vLLM's linear classes by name (QKVParallelLinear,
# MergedColumnParallelLinear, RowParallelLinear); the vocabulary embedding is
# not a Linear and stays unquantized, and the tied head is ignored explicitly.
QUANTIZATION_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": NUM_BITS,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": GROUP_SIZE,
                "dynamic": False,
                "actorder": None,
            },
            "input_activations": None,
        },
    },
    "ignore": ["lm_head"],
}

# The checkpoint's ternary projections, by name. Anything else passes through.
_PROJECTION = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$")

def quantization_config():
    """A fresh copy: vLLM adds keys (head counts) to the dict it is given."""
    return copy.deepcopy(QUANTIZATION_CONFIG)


def is_projection(name):
    """True for the q/k/v/o/gate/up/down weights that become 4-bit."""
    return _PROJECTION.match(name) is not None


def pack_int4(signs):
    """int8 values in [-8, 7] -> int32 words, compressed-tensors' bit order.

    Matches ``compressed_tensors.compressors.pack_quantized.helpers.pack_to_int32``
    with ``num_bits=4, packed_dim=1``: each value is shifted to unsigned
    (+8), and column ``8*j + i`` of a row occupies bits ``4*i`` to ``4*i + 3`` of
    word ``j``, lowest column in the lowest bits.
    """
    if signs.dtype != torch.int8 or signs.dim() != 2:
        raise ValueError("expected a 2-D int8 tensor")
    rows, cols = signs.shape
    if cols % PACK_FACTOR:
        raise ValueError(f"input width {cols} is not a multiple of {PACK_FACTOR}")
    codes = (signs.to(torch.int64) + OFFSET).reshape(rows, cols // PACK_FACTOR, PACK_FACTOR)
    if codes.min() < 0 or codes.max() >= 1 << NUM_BITS:
        raise ValueError("values do not fit in 4 bits")
    shifts = torch.arange(PACK_FACTOR, dtype=torch.int64) * NUM_BITS
    words = (codes << shifts).sum(dim=2)
    # Reinterpret the unsigned 32-bit words as int32 without relying on overflow.
    words = torch.where(words >= 1 << 31, words - (1 << 32), words)
    return words.to(torch.int32).contiguous()


def bit_equal(a, b):
    """Same dtype, shape and bit pattern. Unlike torch.equal, -0.0 differs from +0.0."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if not a.is_floating_point():
        return torch.equal(a, b)
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(bits), b.contiguous().view(bits))


def _quantized(name, signs, scale):
    """Yield the three compressed-tensors parameters for one projection."""
    rows, cols = signs.shape
    if scale.shape != (rows, cols // GROUP_SIZE):
        raise ValueError(f"{name}: scale shape {tuple(scale.shape)} does not match "
                         f"[{rows}, {cols // GROUP_SIZE}]")
    if scale.dtype != torch.bfloat16:
        raise ValueError(f"{name}: group scales are {scale.dtype}, not bfloat16")
    base = name.removesuffix(".weight")
    yield f"{base}.weight_packed", pack_int4(signs)
    yield f"{base}.weight_scale", scale.contiguous()
    yield f"{base}.weight_shape", torch.tensor([rows, cols], dtype=torch.int64)


def _from_dense(name, tensor):
    """A dense projection that must be exactly ternary, as 4-bit parameters."""
    if tensor.dim() != 2 or tensor.shape[1] % GROUP_SIZE:
        raise ValueError(f"{name}: shape {tuple(tensor.shape)} cannot be grouped by {GROUP_SIZE}")
    trits, scale = ternary._to_trits(tensor, GROUP_SIZE)
    rebuilt = ternary._from_trits(trits, scale, tensor.shape, GROUP_SIZE, tensor.dtype)
    if not bit_equal(rebuilt, tensor):
        raise ValueError(f"{name} is not exactly ternary in {GROUP_SIZE}-wide groups; "
                         "refusing to quantize it to W4A16, which would be lossy")
    signs = (trits.to(torch.int8) - 1).reshape(tensor.shape)
    yield from _quantized(name, signs, scale)


def _iter_packed(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "kittenml-ternary-1.58":
            raise ValueError(f"not a packed ternary checkpoint: {metadata.get('format')!r}")
        if int(metadata["group_size"]) != GROUP_SIZE:
            raise ValueError(f"packed group size {metadata['group_size']} is not {GROUP_SIZE}")
        packed = json.loads(metadata["packed"])
        for name in handle.keys():
            if "::" in name or name.startswith("spk_proj."):
                continue
            tensor = handle.get_tensor(name)
            if is_projection(name):
                yield from _from_dense(name, tensor)
            else:
                yield name, tensor
        for name, spec in packed.items():
            if name.startswith("spk_proj."):
                continue
            shape = torch.Size(spec["shape"])
            trits = ternary._unpack_trits(handle.get_tensor(f"{name}::codes"), shape.numel())
            scale = handle.get_tensor(f"{name}::scale")
            if not is_projection(name):
                # Not a layer vLLM quantizes: hand it over dense, exactly as unpacked.
                yield name, ternary._from_trits(trits, scale, shape, GROUP_SIZE,
                                                getattr(torch, spec["dtype"]))
                continue
            signs = (trits.to(torch.int8) - 1).reshape(shape)
            yield from _quantized(name, signs, scale)


def _iter_emb4(repo_dir, config):
    from ..weights import iter_lm_weights

    for name, tensor in iter_lm_weights(repo_dir, config, "emb4"):
        if name.startswith("spk_proj."):
            continue
        if is_projection(name):
            yield from _from_dense(name, tensor)
        else:
            yield name, tensor


def _iter_full(lm_dir, shards):
    from safetensors import safe_open

    if not shards:
        raise FileNotFoundError(f"no plain model weights in {lm_dir}")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                if name.startswith("spk_proj."):
                    continue
                tensor = handle.get_tensor(name)
                if is_projection(name):
                    yield from _from_dense(name, tensor)
                else:
                    yield name, tensor


def iter_w4a16(repo_dir, config, variant):
    """Yield (name, tensor) pairs for vLLM's compressed-tensors W4A16 layers.

    One tensor at a time, so the expanded bf16 state dict is never built. (The
    bf16 files are memory-mapped, so with ``variant="full"`` the pages read so
    far still count towards the process's resident memory.)

    - ``variant="packed"`` reads the 1.58-bit file and converts trits directly.
    - ``variant="full"`` reads the bf16 files and checks every projection is
      exactly ternary, raising otherwise.
    - ``variant="emb4"`` unpacks the TL2 body (exactly ternary, checked like
      ``"full"``) and keeps its 4-bit embedding, which is lossy; only the
      embedding and the tied output head differ from the bf16 model.

    Projections become ``weight_packed`` (int32), ``weight_scale`` (the stored
    bf16 scales, unchanged) and ``weight_shape`` (int64 [2]). Other tensors pass
    through; ``spk_proj.*`` is dropped because it runs outside vLLM.
    """
    if variant == "emb4":
        return _iter_emb4(repo_dir, config)
    path = weights_path(repo_dir, config, variant)
    if path is None:
        return _iter_full(*plain_shards(repo_dir, config))
    if not path.is_file():
        raise FileNotFoundError(f"{variant!r} weights not found: {path}")
    return _iter_packed(path)
