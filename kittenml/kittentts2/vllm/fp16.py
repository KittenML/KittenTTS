"""Opt-in float16 mode for GPUs without bfloat16 (compute capability 7.5).

``vllm_options={"dtype": "float16"}`` is a fallback with a precision change:
the language model was trained and is published in bfloat16, whose range
float16 (largest value 65,504) does not have. Measured in bfloat16 over a
multilingual, long-form and cloning corpus, the residual stream (the running
sum every layer adds to) reaches 69,376 after layer 9 and 3,957,760 at the
final norm, and the MLP's inner activation (the ``down_proj`` input) reaches
154,624. Run as is in float16 these overflow and the model emits no speech.

The fix rescales the model exactly instead of changing what it computes:

- Every read of the residual stream goes through an RMSNorm, and
  ``RMSNorm(c * x)`` with epsilon ``c**2 * eps`` equals ``RMSNorm(x)`` with
  ``eps``. So the whole stream can be carried at ``2**-9`` of its value: the
  token embeddings and the speaker projection (the only inputs to the stream)
  are multiplied by ``2**-9``, so are the outputs of every attention and MLP
  branch, and the epsilon of the 57 norms that read the stream is multiplied
  by ``2**-18``. The attention and MLP inputs are normed, so they see the
  same values as in bfloat16.
- Each branch's ``2**-9`` is split over two weights, so that the activation
  inside the branch is scaled too: ``v_proj`` x ``2**-3`` and ``o_proj`` x
  ``2**-6`` (attention is linear in V), ``up_proj`` x ``2**-5`` and
  ``down_proj`` x ``2**-4`` (silu(gate) * up is linear in up). Multiplying by
  a power of two only changes the exponent, and every scaled weight of the
  published checkpoint is still exactly representable in float16 (checked at
  load time; the loader refuses otherwise).
- The output head shares the embedding matrix in the checkpoint. It is
  untied here and keeps the unscaled values, so the logits are not scaled
  either (the final norm already removes the stream's scale).

What does change is float16 rounding: activations and weights are stored
with float16's precision instead of bfloat16's, and the scaled embedding
(``2**-9`` of values around 0.03) partly falls into float16's subnormal
range, where 1.35% of its elements are rounded (worst error 4.6e-4 of a
token's RMS, a quarter of bfloat16's own rounding step). The untied head
costs one extra copy of the embedding (158,241 x 2,048, 618 MiB).

The scales were chosen from those measurements: ``2**-9`` leaves 8.5x
headroom below 65,504 at the largest value seen, and the MLP's inner
activation peaks at 4,832 (13.6x). They are specific to the published
checkpoint; another checkpoint needs new measurements.

This module does not import vLLM, so it can be tested on any machine.
"""

import re

import torch

# The residual stream is carried at 2**-RESIDUAL_SHIFT of its bfloat16 value.
RESIDUAL_SHIFT = 9
# Weight -> power-of-two shift (multiply by 2**-shift). Each branch's two
# shifts add up to RESIDUAL_SHIFT, so its output joins the stream at the
# stream's scale.
PROJECTION_SHIFTS = {"self_attn.v_proj": 3, "self_attn.o_proj": 6,
                     "mlp.up_proj": 5, "mlp.down_proj": 4}
# RMSNorm(c x; c^2 eps) == RMSNorm(x; eps).
EPS_SHIFT = 2 * RESIDUAL_SHIFT

EMBEDDING = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"

# The norms that read the residual stream. q_norm/k_norm read projections of
# an already normed input and keep their epsilon.
_RESIDUAL_NORM = re.compile(
    r"^model\.(?:layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|norm)$")
# A scaled projection, dense (".weight") or as W4A16 group scales (".weight_scale").
_SCALED = re.compile(r"^model\.layers\.\d+\.(self_attn\.(?:v|o)_proj|mlp\.(?:up|down)_proj)"
                     r"\.(weight|weight_scale)$")
# Every projection weight (dense) or W4A16 group scale: converted to float16
# here, so a lossy conversion is refused rather than left to vLLM's copy.
_PROJECTION = re.compile(r"^model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|"
                         r"mlp\.(?:gate|up|down)_proj)\.(?:weight|weight_scale)$")

# Written into the engine's model config (hf_overrides). vLLM keys its
# torch.compile cache on that config, so a float16 engine never reuses graphs
# compiled with another epsilon, and a change to this plan gets new ones.
PLAN = (f"residual 2^-{RESIDUAL_SHIFT}; "
        + ", ".join(f"{name.split('.')[-1]} 2^-{shift}"
                    for name, shift in PROJECTION_SHIFTS.items())
        + f"; residual-norm eps x 2^-{EPS_SHIFT}; untied unscaled lm_head")
PLAN_KEY = "kittenml_float16_rescale"

if any(shift != RESIDUAL_SHIFT for shift in (
        PROJECTION_SHIFTS["self_attn.v_proj"] + PROJECTION_SHIFTS["self_attn.o_proj"],
        PROJECTION_SHIFTS["mlp.up_proj"] + PROJECTION_SHIFTS["mlp.down_proj"])):
    raise AssertionError("each branch must be scaled by exactly 2**-RESIDUAL_SHIFT")


def hf_overrides():
    """Model-config changes for the engine: an untied head, and the plan."""
    return {"tie_word_embeddings": False, PLAN_KEY: PLAN}


def shift_of(name):
    """The power-of-two shift applied to tensor `name` (0 when it is not scaled)."""
    if name == EMBEDDING:
        return RESIDUAL_SHIFT
    match = _SCALED.match(name)
    return PROJECTION_SHIFTS[match.group(1)] if match else 0


def scaled(tensor, shift):
    """``tensor * 2**-shift`` in float16, rounded once (the scaling itself is exact)."""
    return (tensor.to(torch.float32) * 2.0 ** -shift).to(torch.float16)


def _scaled_and_exact(tensor, shift):
    """(``scaled(tensor, shift)``, whether that float16 value is exact)."""
    target = tensor.to(torch.float32) * 2.0 ** -shift
    result = target.to(torch.float16)
    # Compare bits so a -0.0 that became +0.0 (or the reverse) is not exact.
    exact = torch.equal(result.to(torch.float32).view(torch.int32), target.view(torch.int32))
    return result, exact


def is_exact(tensor, shift):
    """True when ``tensor * 2**-shift`` is exactly representable in float16."""
    return _scaled_and_exact(tensor, shift)[1]


def exact(name, tensor, shift):
    """`scaled`, refusing a weight float16 cannot hold exactly at this scale."""
    result, ok = _scaled_and_exact(tensor, shift)
    if not ok:
        raise ValueError(f"{name} x 2^-{shift} is not exact in float16; the float16 rescale "
                         "was measured for the published checkpoint and does not fit this one")
    return result


def scale_embedding(weight):
    """Token embeddings as the float16 engine stores them (also used for prompts)."""
    return scaled(weight, RESIDUAL_SHIFT)


def rescale_weights(weights):
    """Rescale (name, tensor) pairs, dense or W4A16, for the float16 engine.

    - ``model.embed_tokens.weight`` becomes the scaled embedding, and an
      unscaled float16 copy is added as ``lm_head.weight`` (the untied head);
    - every projection weight (dense) or W4A16 group scale becomes float16,
      with its shift for v/o/up/down, and must be exact;
    - everything else (norm weights, W4A16 codes and shapes) passes through
      unchanged, and vLLM converts floating-point tensors to float16 itself.
    """
    for name, tensor in weights:
        if name == LM_HEAD:
            raise ValueError("the float16 rescale expects a checkpoint whose output head is "
                             "tied to the embedding, not a separate lm_head.weight")
        if name == EMBEDDING:
            yield name, scale_embedding(tensor)
            yield LM_HEAD, tensor.to(torch.float16)
        elif _PROJECTION.match(name):
            yield name, exact(name, tensor, shift_of(name))
        else:
            yield name, tensor


class ScaledSpeakerHead(torch.nn.Module):
    """The speaker projection, run in float32, its output scaled and rounded to float16."""

    def __init__(self, head):
        super().__init__()
        self.head = head.float()

    def forward(self, speaker_embedding):
        return scaled(self.head(speaker_embedding.float()), RESIDUAL_SHIFT)


def residual_norms(model):
    """(name, module) for every RMSNorm that reads the residual stream."""
    return [(name, module) for name, module in model.named_modules()
            if _RESIDUAL_NORM.match(name)]


def scale_norm_epsilons(model, layers, eps):
    """Set the residual norms' epsilon to ``eps * 2**-EPS_SHIFT``; returns how many.

    Runs in the engine after the weights are loaded and before the model is
    compiled or captured. Setting (not multiplying) keeps a second call harmless.
    """
    norms = residual_norms(model)
    if len(norms) != 2 * layers + 1:
        raise RuntimeError(f"float16 rescale: expected {2 * layers + 1} residual-stream norms, "
                           f"found {len(norms)}")
    target = eps * 2.0 ** -EPS_SHIFT
    for name, module in norms:
        if module.variance_epsilon not in (eps, target):
            raise RuntimeError(f"float16 rescale: {name} has epsilon {module.variance_epsilon}, "
                               f"not the model's rms_norm_eps {eps}")
        module.variance_epsilon = target
    return len(norms)
