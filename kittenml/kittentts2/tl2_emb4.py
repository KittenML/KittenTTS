"""Reader for the TL2 + 4-bit-embedding checkpoint.

A third packing of the same language model, smaller than `ternary.py`'s at the
cost of being lossy in one place:

    model.safetensors          3469 MB   bf16 throughout
    model-ternary.safetensors   954 MB   lossless, embedding still bf16
    model-tl2-emb4.safetensors  469 MB   lossless body, 4-bit embedding

The body uses TL2 rather than this package's own packing — a different layout for
the same ternary codeword, 1.8125 bits per weight including scales, and equally
exact. The saving is almost entirely the embedding: 324M parameters that
`ternary.py` has to leave at bf16 because they are not ternary, quantised here to
about 4.2 bits each.

**That embedding step is lossy**, unlike everything else this package ships:
L2 relative error 0.1184 against the bf16 source, measured by the exporter.
Reported quality cost at the time of export was perplexity 13.60 against 13.50 on
h3, and 62.85 against 61.98 on vox_plain. Small, but not nothing — which is why
this is opt-in and `ternary.py`'s lossless packing stays the default.

Both unpackers are ports of the exporter's own reader, so the weights this
produces are the weights it was verified against.
"""

import json
import math

import numpy as np
import torch

# TL2 geometry: one block covers 32 output rows x 128 inputs in 928 bytes.
GROUP = 128
TL2_ROWS = 32
TL2_GROUP_BYTES = 928


def unpack_tl2(buf, out_features, in_features):
    """uint8 [out/32, in/128, 928] -> bf16 [out, in], the exact ternary codeword.

    Each block holds 32 fp16 group scales, then four sub-blocks of ten
    balanced-base-3 triples plus a two-bit tail. A triple stores three trits in
    one value, which is what gets the body under two bits per weight.
    """
    rows, groups = out_features // TL2_ROWS, in_features // GROUP
    if buf.shape != (rows, groups, TL2_GROUP_BYTES):
        raise ValueError(f"expected {(rows, groups, TL2_GROUP_BYTES)}, got {buf.shape}")

    gamma = buf[:, :, :TL2_ROWS * 2].copy().view(np.float16).transpose(0, 2, 1)
    trits = np.empty((rows, TL2_ROWS, groups, GROUP), dtype=np.int8)
    cursor = TL2_ROWS * 2

    for sub in range(4):
        segment = buf[:, :, cursor:cursor + 200].reshape(rows, groups, 10, 20)
        cursor += 200
        packed = np.ascontiguousarray(segment[..., :16].transpose(0, 3, 1, 2))
        magnitude = np.empty((rows, TL2_ROWS, groups, 10), dtype=np.uint8)
        magnitude[:, 0::2] = packed & 0x0F                  # even rows: low nibble
        magnitude[:, 1::2] = packed >> 4                    # odd rows: high nibble

        signs = np.ascontiguousarray(segment[..., 16:].transpose(0, 3, 1, 2))
        negative = np.unpackbits(signs, axis=1, count=TL2_ROWS,
                                 bitorder="little").astype(bool)
        balanced = magnitude.astype(np.int16)
        np.negative(balanced, out=balanced, where=negative)

        n = balanced + 13                                   # balanced base 3 -> 0..26
        block = trits[:, :, :, sub * 32:(sub + 1) * 32]
        triple = np.empty((rows, TL2_ROWS, groups, 10, 3), dtype=np.int8)
        triple[..., 0] = (n // 9) - 1
        triple[..., 1] = ((n // 3) % 3) - 1
        triple[..., 2] = (n % 3) - 1
        block[..., :30] = triple.reshape(rows, TL2_ROWS, groups, 30)

        tail_bytes = np.ascontiguousarray(buf[:, :, cursor:cursor + 16].transpose(0, 2, 1))
        cursor += 16
        tail = np.empty((rows, TL2_ROWS, groups), dtype=np.uint8)
        tail[:, 0::2] = tail_bytes & 0x0F
        tail[:, 1::2] = tail_bytes >> 4
        block[..., 30] = (tail & 0x03).astype(np.int8) - 1
        block[..., 31] = ((tail >> 2) & 0x03).astype(np.int8) - 1

    values = torch.from_numpy(trits.reshape(out_features, groups, GROUP)).to(torch.bfloat16)
    scales = torch.from_numpy(
        np.ascontiguousarray(gamma.reshape(out_features, groups))).to(torch.bfloat16)
    return (values * scales.unsqueeze(-1)).reshape(out_features, in_features)


def _hadamard(n):
    matrix = torch.tensor([[1.0]])
    for _ in range(int(round(math.log2(n)))):
        matrix = torch.cat([torch.cat([matrix, matrix], 1),
                            torch.cat([matrix, -matrix], 1)], 0)
    return matrix / math.sqrt(n)


def _unpack_codes(buf, codebook_size, count):
    bits = int(math.ceil(math.log2(codebook_size)))
    if bits == 8:
        return buf[:count].astype(np.int64)
    per_byte = 8 // bits
    out = np.empty(buf.size * per_byte, dtype=np.uint8)
    for j in range(per_byte):
        out[j::per_byte] = (buf >> (bits * j)) & ((1 << bits) - 1)
    return out[:count].astype(np.int64)


def dequant_embedding(get_tensor, metadata, chunk=1 << 15):
    """4-bit codes -> a bf16 embedding matrix.

    Rows are split into frequency tiers with their own codebook sizes — audio
    tokens get a 256-entry book, rare text rows only 16 — then rotated by a
    Hadamard matrix and coded per column. Reversing it means looking up each
    row's centres, rotating back, and undoing the row and column scales.
    """
    vocab, dim = int(metadata["V"]), int(metadata["D"])
    codebook_sizes = json.loads(metadata["Ks"])
    tier_map = get_tensor("emb4/tier_map").numpy()
    hadamard = _hadamard(dim)

    weights = torch.empty(vocab, dim, dtype=torch.float32)
    for tier, codebook_size in enumerate(codebook_sizes):
        rows = np.flatnonzero(tier_map == tier)             # ascending == packing order
        codes = _unpack_codes(get_tensor(f"emb4/t{tier}_codes").numpy(),
                              codebook_size, rows.size * dim).reshape(rows.size, dim)
        centres = get_tensor(f"emb4/t{tier}_centers").to(torch.float32).numpy()
        row_scale = get_tensor(f"emb4/t{tier}_s").to(torch.float32).numpy()
        col_scale = torch.from_numpy(
            get_tensor(f"emb4/t{tier}_t").to(torch.float32).numpy())
        centres_t = np.ascontiguousarray(centres.T)
        # Chunked: the full rotation would otherwise materialise 158k x 2048 floats.
        for start in range(0, rows.size, chunk):
            stop = min(start + chunk, rows.size)
            rotated = np.take_along_axis(centres_t, codes[start:stop].T, axis=1).T
            restored = torch.from_numpy(np.ascontiguousarray(rotated)) @ hadamard.T
            weights[rows[start:stop]] = (restored * col_scale[None, :]
                                         * torch.from_numpy(row_scale[start:stop])[:, None])
    return weights.to(torch.bfloat16)


def load_state_dict(path):
    """Rebuild the full bf16 state dict from a TL2 + emb4 checkpoint."""
    from safetensors import safe_open

    state = {}
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        if metadata.get("schema") != "kitten_ternary_tl2_emb4_v1":
            raise ValueError(f"unexpected schema {metadata.get('schema')!r} in {path}")
        embedding_metadata = json.loads(metadata.get("embedding_metadata", "{}"))
        for key in handle.keys():
            if key.startswith("tl2/"):
                buf = handle.get_tensor(key).numpy()
                rows, groups, _ = buf.shape
                state[key[4:]] = unpack_tl2(buf, rows * TL2_ROWS, groups * GROUP)
            elif key.startswith("fp/"):
                state[key[3:]] = handle.get_tensor(key)
        state["model.embed_tokens.weight"] = dequant_embedding(
            handle.get_tensor, embedding_metadata)
    return state
