"""Ternary quantization-aware training for the 196 body linears.

The checkpoint's q/k/v/o/gate/up/down weights are already folded ternary -- every
128-wide group holds exactly {-s, 0, +s} (`ternary.py`). A fine-tune therefore
cannot just train them as dense bf16 and hope: the export has to be ternary
again, and a model trained dense then quantized once at the end loses far more
than one trained through the quantizer. So: keep a full-precision *shadow*
weight, quantize it in the forward pass, let the gradient pass through the
rounding unchanged (straight-through estimator), and update the shadow.

What decides whether that works at all is the **scale rule**, because QAT starts
with shadow = the released weights. If the forward quantizer does not map those
weights to themselves, step 0 moves every weight in the model before a single
gradient is applied:

    absmax   s = max|W|    a fixed point on the released weights, bit-exactly.
                           Shadow = W and step 0 changes nothing.
    absmean  s = mean|W|   NOT a fixed point as-is. On ternary input
                           mean|W| = f * max|W|, where f is the group's nonzero
                           fraction, so the rule shrinks every surviving weight
                           by f -- measured 0.61 to 0.98 across groups, no sign
                           flips, no new zeros, just a uniformly detuned model
                           across all 196 layers. `shadow_from_released`
                           cancels it by pre-dividing by f; the `clamp(-1, 1)`
                           is what absorbs the resulting 1/f > 1 ratios, and is
                           load-bearing, not defensive -- at f = 0.61 the
                           unclamped round() returns 2.

Which rule the original training used is **not identifiable from the
checkpoint** -- the measured 15% zero fraction matches neither the ~31% absmean
leaves on Gaussian weights nor the ~83% absmax leaves (`indic/bench/
ternary_check.py`). Both are offered, neither is assumed.

The long-horizon risk of absmax is the reason absmean exists at all. Its scale
is the group maximum, and `round(w/s)` is zero whenever |w| < s/2 -- so one
weight that drifts to twice the magnitude of its 127 neighbours zeroes the
entire group, and because the group is then all zeros its gradient signal is
whatever the STE passes through a dead group. Nothing recovers it on its own.
absmean's scale tracks the bulk of the group rather than its extreme, so an
outlier raises the threshold only in proportion to its share of the mean.
Resuming on absmax is free and correct at step 0; over a long run, watch
`collapsed_group_fraction` and switch if it climbs.

    from indic.train.qat import wrap_ternary, export_packed
    wrap_ternary(model, rule="absmax")          # 196 linears, shadow = W
    ...                                          # train
    export_packed(model, "lm/model-ternary.safetensors")
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import load_kittentts2

_ternary = load_kittentts2("ternary")
# Imported rather than restated: if the packing ever moves off 128-wide groups,
# QAT has to move with it or the export stops being packable.
GROUP_SIZE = _ternary.GROUP_SIZE

RULES = ("absmax", "absmean")

# The seven projections per layer that the checkpoint stores ternary; 7 x 28 =
# 196. Matched on the dotted suffix because a bare "k_proj" is also a substring
# of "spk_proj", the dense speaker head, which is not ternary at all.
TARGET_SUFFIXES = (".q_proj", ".k_proj", ".v_proj", ".o_proj",
                   ".gate_proj", ".up_proj", ".down_proj")

# Guards division by an all-zero group's scale. Small enough never to perturb a
# real scale (the smallest observed is ~1e-3), large enough that 0/EPS is 0 and
# not a NaN.
EPS = 1e-12


def quantize(weight, rule, group=GROUP_SIZE):
    """One ternary forward pass: the quantizer as the training graph applies it.

    Computed in float32 whatever the shadow's dtype, because under `absmean` the
    scale is a 128-term sum and a bf16 accumulation of it loses enough precision
    to move weights (measured: a bf16 shadow misses the released values by up to
    1.1e-3, against 6e-8 in float32).
    """
    if rule not in RULES:
        raise ValueError(f"unknown ternary rule {rule!r}; choose from {list(RULES)}")
    if weight.dim() != 2 or weight.shape[1] % group:
        raise ValueError(f"shape {tuple(weight.shape)} is not groupable at {group}")

    grouped = weight.float().reshape(weight.shape[0], -1, group)
    magnitude = grouped.abs()
    scale = (magnitude.amax(dim=2, keepdim=True) if rule == "absmax"
             else magnitude.mean(dim=2, keepdim=True)).clamp(min=EPS)
    return ((grouped / scale).round().clamp(-1, 1) * scale).reshape(weight.shape)


def quantize_ste(weight, rule, group=GROUP_SIZE):
    """`quantize`, with the gradient routed straight through the rounding.

    round() has zero derivative almost everywhere, so without this the 196
    shadow weights would never receive a gradient and QAT would be a very
    expensive way to run frozen inference.
    """
    return weight + (quantize(weight, rule, group) - weight).detach()


def shadow_from_released(weight, rule, group=GROUP_SIZE):
    """The shadow that makes step 0 reproduce `weight` exactly, under `rule`.

    Requires `weight` to be the released ternary tensor -- the `absmean` rescale
    divides by each group's nonzero fraction, which is only the right correction
    when every nonzero in the group already shares one magnitude. Applying it to
    a dense tensor would be meaningless, so that is refused rather than guessed.
    """
    if rule not in RULES:
        raise ValueError(f"unknown ternary rule {rule!r}; choose from {list(RULES)}")
    if not _ternary.is_ternary(weight, group):
        raise ValueError(
            "shadow_from_released needs an already-ternary weight; this tensor is "
            "dense. Use wrap_ternary(..., resume=False) to start QAT from it.")
    if rule == "absmax":
        # W/max|W| is already in {-1, 0, +1} per group, so quantize(W) == W.
        return weight.float().clone()

    grouped = weight.float().reshape(weight.shape[0], -1, group)
    fraction = (grouped != 0).to(torch.float32).mean(dim=2, keepdim=True)
    # An all-zero group has nothing to rescale and 0/EPS is 0, so it survives
    # the clamp unchanged rather than becoming NaN.
    return (grouped / fraction.clamp(min=EPS)).reshape(weight.shape)


def collapsed_group_fraction(weight, rule, group=GROUP_SIZE):
    """Fraction of groups this rule would quantize to all zeros.

    The absmax failure mode, made measurable: a group whose maximum has run away
    from its 127 neighbours quantizes to nothing and stops carrying signal. Log
    it per-run; on the released weights under absmax it is 0.
    """
    quantized = quantize(weight, rule, group)
    grouped = quantized.reshape(weight.shape[0], -1, group)
    return float((grouped == 0).all(dim=2).to(torch.float32).mean())


class TernaryLinear(nn.Module):
    """An `nn.Linear` whose weight is ternarized on every forward pass.

    `weight` is the float32 shadow and is what the optimizer sees; the ternary
    tensor is derived, never stored. The parameter keeps the name `weight` so
    that `state_dict()` on a wrapped model produces exactly the keys the
    released checkpoint uses -- which is what lets `export_packed` hand the
    result straight to `ternary.pack_state_dict`.
    """

    def __init__(self, linear, rule, group=GROUP_SIZE, resume=True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rule = rule
        self.group = group
        released = linear.weight.detach()
        shadow = (shadow_from_released(released, rule, group) if resume
                  else released.float().clone())
        self.weight = nn.Parameter(shadow)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())

    def forward(self, x):
        # Cast to the activation dtype rather than the reverse: under bf16
        # autocast the activations are bf16 and the shadow is float32, and
        # pulling the activations up to float32 would silently undo the mixed
        # precision for a third of the model's FLOPs.
        return F.linear(x, quantize_ste(self.weight, self.rule, self.group).to(x.dtype),
                        self.bias)

    def quantized_weight(self, dtype=torch.bfloat16):
        """The ternary tensor this layer would export, detached."""
        with torch.no_grad():
            return quantize(self.weight, self.rule, self.group).to(dtype)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, rule={self.rule}, group={self.group}")


def target_linears(model, suffixes=TARGET_SUFFIXES):
    """Names of the linears QAT should wrap, in module order."""
    return [name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.endswith(suffixes)]


def wrap_ternary(model, rule="absmax", group=GROUP_SIZE, resume=True,
                 suffixes=TARGET_SUFFIXES):
    """Replace every target linear in place with its QAT equivalent.

    `resume=True` is the normal path and asserts that what it is wrapping really
    is the released ternary weight. `resume=False` is for a dense or
    randomly-initialised model -- a toy config in the self-check, or a first QAT
    pass over a model that was never quantized -- where there is no fixed point
    to preserve.

    Returns the wrapped names, so a caller can assert it got all 196 rather than
    silently training a model where half the body was missed.
    """
    names = target_linears(model, suffixes)
    for name in names:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attribute = name.rsplit(".", 1)[-1]
        linear = getattr(parent, attribute)
        setattr(parent, attribute, TernaryLinear(linear, rule, group, resume=resume))
    return names


def unwrap_ternary(model, dtype=torch.bfloat16):
    """Fold every `TernaryLinear` back to a plain `nn.Linear` holding its ternary weight.

    For evaluating mid-run through the ordinary inference path, which knows
    nothing about shadows. Destructive: the shadow is discarded, so take a
    checkpoint first if the run continues.
    """
    names = [name for name, module in model.named_modules()
             if isinstance(module, TernaryLinear)]
    for name in names:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attribute = name.rsplit(".", 1)[-1]
        wrapped = getattr(parent, attribute)
        linear = nn.Linear(wrapped.in_features, wrapped.out_features,
                           bias=wrapped.bias is not None, dtype=dtype)
        with torch.no_grad():
            linear.weight.copy_(wrapped.quantized_weight(dtype))
            if wrapped.bias is not None:
                linear.bias.copy_(wrapped.bias.to(dtype))
        setattr(parent, attribute, linear)
    return names


def ternary_state_dict(model, dtype=torch.bfloat16):
    """The model's state dict with every shadow replaced by its ternary export.

    Keys are unchanged from the released checkpoint, and `lm_head.weight` is
    dropped: it is tied to the embedding, so storing it would double the largest
    tensor in the file and `model.py` already expects it to be absent.
    """
    quantized = {name: module.quantized_weight(dtype)
                 for name, module in model.named_modules()
                 if isinstance(module, TernaryLinear)}
    state = {}
    for key, tensor in model.state_dict().items():
        if key == "lm_head.weight":
            continue
        shadow_owner = key[:-len(".weight")] if key.endswith(".weight") else None
        state[key] = (quantized[shadow_owner] if shadow_owner in quantized
                      else tensor.detach().to(dtype))
    return state


def export_packed(model, path, dtype=torch.bfloat16):
    """Write a packed-ternary checkpoint the existing loaders can read.

    Goes through `ternary.pack_state_dict`, so the result is the same
    `kittenml-ternary-1.58` format `weights.load_lm_weights` and the llama.cpp
    fork already consume -- a fine-tune that produced its own format would be a
    model nothing can run.

    Raises if any of the 196 came out non-ternary, because `pack_state_dict`
    silently stores a non-ternary tensor dense instead: the file would load, the
    model would work, and it would be three times the size with no indication
    why.
    """
    from safetensors.torch import save_file

    state = ternary_state_dict(model, dtype)
    expected = {name for name, module in model.named_modules()
                if isinstance(module, TernaryLinear)}
    tensors, metadata = _ternary.pack_state_dict(state)
    packed = set(json.loads(metadata["packed"]))
    missed = sorted({f"{name}.weight" for name in expected} - packed)
    if missed:
        raise ValueError(f"{len(missed)} quantized weights did not pack as ternary, "
                         f"e.g. {missed[:3]}")
    save_file(tensors, str(path), metadata=metadata)
    return {"path": str(path), "packed_tensors": len(packed),
            "dense_tensors": len(state) - len(packed)}


def load_released_state(path):
    """Unpack a whole packed-ternary checkpoint to its bf16 tensors.

    `weights.load_lm_weights` does the same thing, but it is a relative import
    inside `kittenml.kittentts2`, so reaching it means importing the package --
    which is the dependency this module exists to avoid. The public unpacker is
    the same one either way.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}
    return _ternary.unpack_state_dict(tensors, metadata)


def load_released_tensors(path, keys):
    """Unpack just `keys` from a packed-ternary checkpoint.

    The released file is 0.95 GB and unpacks to 3.47 GB; the fixed-point check
    needs two tensors out of it. Filtering the metadata rather than reaching into
    `ternary.py`'s private helpers keeps this on the public API, so it cannot
    drift from however the real loader unpacks.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        packed = json.loads(metadata["packed"])
        unknown = [k for k in keys if k not in packed]
        if unknown:
            raise KeyError(f"not packed ternary tensors in {path}: {unknown}")
        metadata["packed"] = json.dumps({k: packed[k] for k in keys})
        tensors = {}
        for key in keys:
            tensors[f"{key}::codes"] = handle.get_tensor(f"{key}::codes")
            tensors[f"{key}::scale"] = handle.get_tensor(f"{key}::scale")
    return _ternary.unpack_state_dict(tensors, metadata)
