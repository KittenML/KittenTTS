"""Lossless 1.58-bit packing for the language model's ternary weights.

The checkpoint's linear weights are ternary: within every 128-wide group, each
value is exactly one of {-scale, 0, +scale}. Stored as bf16 that costs 16 bits per
weight to say one of three things, so the file is 3.47 GB when the information in
it fits in about 0.95 GB.

Packing is therefore **exact, not approximate**. Each weight becomes a trit, five
trits pack into one byte (3^5 = 243 < 256, so 8/5 = 1.6 bits per weight — the
"1.58 bit" of log2(3) rounded up to a whole byte boundary), and each group keeps
its scale at full bf16 precision. Unpacking reproduces the original tensor
bit-for-bit.

Tensors that are not ternary — the token embedding, the layer norms, the speaker
projection — are stored unchanged at their original dtype. They are only 0.325B of
the 1.735B parameters but, being dense, they dominate the packed file.

    lm/model.safetensors           3.47 GB   bf16 throughout
    lm/model-ternary.safetensors   0.95 GB   same weights, 3.6x smaller
"""

import json

import torch

# The group width the checkpoint was quantised with. Each group carries one
# scale, so this has to match how the weights were produced.
GROUP_SIZE = 128

# Trits per byte. 3**5 = 243 fits in a byte; 3**6 = 729 does not.
TRITS_PER_BYTE = 5

_POWERS = torch.tensor([3 ** i for i in range(TRITS_PER_BYTE)], dtype=torch.int32)


def is_ternary(tensor, group_size=GROUP_SIZE):
    """True when every `group_size`-wide group holds only {-s, 0, +s} for some s.

    Checked exactly. A tensor that is merely close to ternary is not packable, and
    silently treating it as such would lose accuracy.
    """
    if tensor.dim() != 2 or tensor.shape[1] % group_size:
        return False
    grouped = tensor.float().reshape(tensor.shape[0], -1, group_size)
    magnitude = grouped.abs()
    scale = magnitude.amax(dim=2, keepdim=True)
    nonzero = grouped != 0
    return bool(torch.all(~nonzero | (magnitude == scale)))


def pack_state_dict(state_dict, group_size=GROUP_SIZE):
    """Split a state dict into packed ternary tensors and untouched dense ones.

    Returns (tensors, metadata) ready to hand to `safetensors.torch.save_file`.
    """
    tensors, packed_keys = {}, {}
    for name, tensor in state_dict.items():
        if not is_ternary(tensor, group_size):
            tensors[name] = tensor.contiguous()
            continue
        codes, scale = _to_trits(tensor, group_size)
        tensors[f"{name}::codes"] = _pack_trits(codes)
        tensors[f"{name}::scale"] = scale
        packed_keys[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
    metadata = {
        "format": "kittenml-ternary-1.58",
        "group_size": str(group_size),
        "packed": json.dumps(packed_keys),
    }
    return tensors, metadata


def unpack_state_dict(tensors, metadata):
    """Rebuild the original state dict from `pack_state_dict`'s output."""
    if metadata.get("format") != "kittenml-ternary-1.58":
        raise ValueError(f"not a packed ternary checkpoint: {metadata.get('format')!r}")
    group_size = int(metadata["group_size"])
    packed_keys = json.loads(metadata["packed"])

    state = {k: v for k, v in tensors.items() if "::" not in k}
    for name, spec in packed_keys.items():
        shape = torch.Size(spec["shape"])
        trits = _unpack_trits(tensors[f"{name}::codes"], shape.numel())
        scale = tensors[f"{name}::scale"]
        state[name] = _from_trits(trits, scale, shape, group_size,
                                  getattr(torch, spec["dtype"]))
    return state


def _to_trits(tensor, group_size):
    """Ternary weights -> trits in {0, 1, 2} for {-s, 0, +s}, plus per-group scales."""
    grouped = tensor.reshape(tensor.shape[0], -1, group_size)
    scale = grouped.abs().amax(dim=2, keepdim=True)
    # sign() gives -1/0/+1; shift to 0/1/2 so it fits an unsigned base-3 digit.
    trits = (grouped.sign().to(torch.int8) + 1).reshape(-1)
    return trits, scale.squeeze(2).contiguous()


def _from_trits(trits, scale, shape, group_size, dtype):
    signs = (trits.to(torch.int8) - 1).reshape(shape[0], -1, group_size)
    scale = scale.reshape(shape[0], -1, 1)
    return (signs.to(scale.dtype) * scale).reshape(shape).to(dtype).contiguous()


def _pack_trits(trits):
    """Five base-3 digits per byte."""
    padding = (-trits.numel()) % TRITS_PER_BYTE
    if padding:
        trits = torch.cat([trits, torch.zeros(padding, dtype=trits.dtype)])
    digits = trits.to(torch.int32).reshape(-1, TRITS_PER_BYTE)
    return (digits * _POWERS).sum(dim=1).to(torch.uint8).contiguous()


def _unpack_trits(packed, numel):
    values = packed.to(torch.int32).unsqueeze(1)
    digits = torch.div(values, _POWERS, rounding_mode="floor") % 3
    return digits.reshape(-1)[:numel]
