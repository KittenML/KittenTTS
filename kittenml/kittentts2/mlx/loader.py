"""The published ternary weights, loaded straight into MLX's 4-bit kernels.

The checkpoint's linear weights are ternary: inside every 128-wide group each
value is exactly one of {-scale, 0, +scale}, stored as five base-3 trits per byte
(see `ternary.py`). MLX multiplies by a quantised weight as

    w = scales * code + biases

so a ternary group is representable *exactly* in MLX's 4-bit affine format by
taking `scales = scale` and `biases = -scale`, which puts the three values the
weight can take at codes 0, 1 and 2:

    code 0 -> -scale      code 1 -> 0      code 2 -> +scale

The trits on disk already use that numbering — `ternary.py` writes `sign + 1` —
so a trit *is* its MLX code and conversion is pure bit shuffling: unpack base-3,
repack as nibbles. No dequantisation, no re-quantisation, nothing rounded. The
weights MLX multiplies by are bit-for-bit the weights the published checkpoint
holds, at 4 bits each instead of the 16 the unpacked tensor would need.

`mx.quantize` is not usable here: it fits its own min/max affine grid, whose
16 levels straddle zero for a symmetric tensor, so a ternary weight's exact zero
would land between two codes.
"""

import json

import mlx.core as mx

# How `ternary.py` writes the codes: five base-3 digits in one byte, least
# significant digit first.
TRITS_PER_BYTE = 5
_POWERS = [3 ** i for i in range(TRITS_PER_BYTE)]

# How MLX reads 4-bit codes: eight nibbles per uint32 word, lowest nibble first.
BITS = 4
CODES_PER_WORD = 32 // BITS

PACKED_FORMAT = "kittenml-ternary-1.58"
# MLX's quantised kernels exist for these group widths only.
MLX_GROUP_SIZES = (32, 64, 128)


def load_lm_weights(path, dtype=mx.bfloat16):
    """Read a packed ternary checkpoint as MLX tensors.

    Returns `(dense, quantised, group_size)`, where `dense` holds the tensors the
    checkpoint stores unpacked — the token embedding, the norms, the speaker head
    — and `quantised` maps each ternary weight's name to the
    `(codes, scales, biases)` triple `mx.quantized_matmul` takes.
    """
    tensors, metadata = mx.load(str(path), return_metadata=True)
    if metadata.get("format") != PACKED_FORMAT:
        raise ValueError(f"not a packed ternary checkpoint: {metadata.get('format')!r}. "
                         "The MLX backend needs the repository's packed weights "
                         "(weights='packed').")
    group_size = int(metadata["group_size"])
    if group_size not in MLX_GROUP_SIZES:
        raise ValueError(f"the checkpoint is quantised in groups of {group_size}, and "
                         f"MLX's 4-bit kernels support {MLX_GROUP_SIZES}")

    dense = {name: tensor.astype(dtype) if tensor.dtype != dtype else tensor
             for name, tensor in tensors.items() if "::" not in name}
    quantised = {}
    for name, spec in json.loads(metadata["packed"]).items():
        rows, columns = spec["shape"]
        scales = tensors[f"{name}::scale"].astype(dtype)
        codes = _to_mlx_codes(tensors[f"{name}::codes"], rows, columns)
        quantised[name] = (codes, scales, -scales)
        # One tensor at a time: the unpacked trits are eight times the size of the
        # nibbles they become, and holding every layer's at once would cost more
        # memory than loading the model unpacked in the first place.
        mx.eval(quantised[name])
        del tensors[f"{name}::codes"], tensors[f"{name}::scale"]
    return dense, quantised, group_size


def _to_mlx_codes(packed, rows, columns):
    """Base-3 bytes -> the uint32 nibble words `mx.quantized_matmul` reads."""
    powers = mx.array(_POWERS, mx.uint32)
    trits = (packed.astype(mx.uint32).reshape(-1, 1) // powers) % 3
    # The packing pads the final byte to a whole five trits; drop that tail.
    trits = trits.reshape(-1)[:rows * columns].reshape(rows, -1, CODES_PER_WORD)
    shifts = mx.arange(0, 32, BITS, dtype=mx.uint32)
    return mx.sum(mx.left_shift(trits, shifts), axis=-1).astype(mx.uint32)
