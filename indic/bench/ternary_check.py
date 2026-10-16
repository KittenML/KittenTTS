"""Work out which ternary quantizer produced the checkpoint, and whether QAT can
resume from it.

This is the gate on the whole training plan. The body's linear weights are ternary
(`ternary.py`), so any fine-tune has to end ternary again, which means
quantization-aware training: keep a full-precision shadow weight, quantize it in
the forward pass through a straight-through estimator, update the shadow.

The subtlety that decides whether that works is the **scale rule**. QAT starts
with shadow = the released weights. If the forward quantizer does not map those
weights to themselves, then step 0 of training perturbs every weight in the model
before a single gradient is applied, and the run starts from a damaged checkpoint
at a much worse loss. The rule must be a fixed point on the released weights.

Two candidate rules, per 128-wide group:

    absmean (BitNet b1.58):  s = mean|W|   Q = clamp(round(W/s), -1, 1) * s
    absmax:                  s = max|W|    Q = clamp(round(W/s), -1, 1) * s

The stored scale in `ternary.py:_to_trits` is `amax`, but that proves nothing on
its own: once a tensor is ternary its only nonzero magnitude *is* its scale, so
amax recovers whatever rule produced it. The same degeneracy sinks the
fixed-point test as a way of *naming* the rule -- see `verdict` -- so the only
discriminating measurement is **sparsity**. Under absmean, a roughly Gaussian
weight zeroes out everywhere |w| < 0.5*mean|w|, which is about 31% of entries.
Under absmax, almost everything falls below half the group maximum, so about 83%
would be zero -- a network that sparse would not work. The zero fraction is what
tells us the rule; the fixed-point test answers the separate question of whether
QAT can resume without perturbing anything.

    python -m indic.bench.ternary_check
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

GROUP = 128

# What each rule leaves zero on a roughly Gaussian tensor at this group width.
# Measured, not assumed: 10 draws of a 4096x2048 standard normal, quantized both
# ways. These are the yardsticks the checkpoint's own sparsity is read against.
GAUSSIAN_ZEROS = {"absmean": 0.31, "absmax": 0.83}


def quantize(weight, rule, group=GROUP):
    """One ternary quantization pass, as a QAT forward would do it."""
    import torch

    grouped = weight.float().reshape(weight.shape[0], -1, group)
    if rule == "absmean":
        scale = grouped.abs().mean(dim=2, keepdim=True)
    elif rule == "absmax":
        scale = grouped.abs().amax(dim=2, keepdim=True)
    elif rule == "absmedian":
        scale = grouped.abs().median(dim=2, keepdim=True).values
    else:
        raise ValueError(rule)
    scale = scale.clamp(min=1e-12)
    return ((grouped / scale).round().clamp(-1, 1) * scale).reshape(weight.shape)


def analyse(weight, group=GROUP):
    """Sparsity, scale consistency and fixed-point behaviour for one tensor."""
    import torch

    grouped = weight.float().reshape(weight.shape[0], -1, group)
    magnitude = grouped.abs()
    amax = magnitude.amax(dim=2, keepdim=True)
    nonzero = grouped != 0

    zero_fraction = 1.0 - nonzero.float().mean().item()
    # Exactly ternary means every nonzero in a group has the same magnitude.
    is_ternary = bool(torch.all(~nonzero | (magnitude == amax)))
    amean = magnitude.mean(dim=2, keepdim=True)
    # mean|W| / nonzero fraction recovers the stored scale. That is an identity
    # for any ternary tensor, not evidence for any rule -- what it actually
    # checks is the group width, since at the wrong width a group holds two
    # different magnitudes and the ratio stops closing.
    frac_nonzero_per_group = nonzero.float().mean(dim=2, keepdim=True)
    implied = (amean / frac_nonzero_per_group.clamp(min=1e-12))
    scale_matches_amax = torch.allclose(implied, amax, rtol=1e-2)

    out = {"zero_fraction": zero_fraction, "exactly_ternary": is_ternary,
           "implied_absmean_scale_equals_stored": bool(scale_matches_amax)}
    for rule in ("absmean", "absmax"):
        requantized = quantize(weight, rule, group)
        same = torch.equal(requantized, weight.float())
        drift = (requantized - weight.float()).abs().mean().item()
        out[f"{rule}_is_fixed_point"] = bool(same)
        out[f"{rule}_mean_abs_drift"] = drift
    return out


def load_state(path):
    """Unpack the released packed-ternary checkpoint to its bf16 tensors."""
    import importlib.util

    from safetensors import safe_open

    # Imported by file path rather than as `kittenml.kittentts2.ternary`, because
    # that package's __init__ pulls in the whole TTS stack (diffusers, librosa,
    # the normalizer). This script only needs the unpacker, and keeping it free of
    # those imports means the QAT check runs on a bare machine.
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_ternary", root / "kittenml" / "kittentts2" / "ternary.py")
    ternary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ternary)
    unpack_state_dict = ternary.unpack_state_dict

    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}
    return unpack_state_dict(tensors, metadata)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="path to lm/model-ternary.safetensors")
    ap.add_argument("--layers", type=int, default=3, help="layers to sample")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    import torch

    path = args.checkpoint
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("KittenML/kitten-tts-2", "lm/model-ternary.safetensors")

    print(f"checkpoint {path}\nunpacking ...", flush=True)
    state = load_state(path)

    # Dotted on purpose. A bare "k_proj" is also a substring of "spk_proj", the
    # dense speaker projection, which is not ternary at all; left unanchored it
    # lands on the wrong side of the parameter split and vanishes from the list
    # of big dense tensors, where it is the second largest.
    ternary_keys = [k for k in state
                    if k.endswith(".weight") and state[k].dim() == 2
                    and any(p in k for p in (".q_proj", ".k_proj", ".v_proj", ".o_proj",
                                             ".gate_proj", ".up_proj", ".down_proj"))]
    print(f"{len(ternary_keys)} candidate linear weights\n")

    sampled = [k for k in ternary_keys
               if any(f"layers.{i}." in k for i in range(args.layers))]
    rows = {}
    for key in sampled:
        rows[key] = analyse(state[key])

    print(f"{'tensor':<46}{'zeros':>8}{'ternary':>9}{'absmean fp':>12}{'absmax fp':>11}"
          f"{'absmean drift':>15}")
    for key, r in rows.items():
        print(f"{key[-45:]:<46}{r['zero_fraction']:>7.1%}"
              f"{str(r['exactly_ternary']):>9}"
              f"{str(r['absmean_is_fixed_point']):>12}"
              f"{str(r['absmax_is_fixed_point']):>11}"
              f"{r['absmean_mean_abs_drift']:>15.4f}")

    zeros = [r["zero_fraction"] for r in rows.values()]
    print(f"\nzero fraction: min {min(zeros):.1%}  mean {sum(zeros)/len(zeros):.1%}  "
          f"max {max(zeros):.1%}")
    verdict(rows)

    # The dense tensors are what a vocab swap and a QAT run actually resize or keep
    # in full precision, so size them here too.
    dense = {k: v for k, v in state.items() if k not in ternary_keys}
    dense_params = sum(v.numel() for v in dense.values())
    ternary_params = sum(state[k].numel() for k in ternary_keys)
    print(f"\nternary parameters {ternary_params/1e9:.3f}B   "
          f"full-precision {dense_params/1e9:.3f}B")
    for key in sorted(dense, key=lambda k: -dense[k].numel())[:4]:
        print(f"  {key:<50}{tuple(dense[key].shape)}  {dense[key].numel()/1e6:>8.1f}M")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"checkpoint": str(path), "tensors": rows,
             "ternary_params": ternary_params, "dense_params": dense_params}, indent=2))
        print(f"\nwrote {args.out}")


def verdict(rows):
    """Name the rule from the sparsity, then say what QAT can resume with."""
    zeros = [r["zero_fraction"] for r in rows.values()]
    observed = sum(zeros) / len(zeros)
    absmean_fp = all(r["absmean_is_fixed_point"] for r in rows.values())
    absmax_fp = all(r["absmax_is_fixed_point"] for r in rows.values())
    drift = sum(r["absmean_mean_abs_drift"] for r in rows.values()) / len(rows)

    print("\n== Verdict ==")
    # The fixed-point columns look decisive and are not. On an already-ternary
    # tensor absmax is a fixed point by construction -- the single nonzero
    # magnitude in a group *is* amax, so W/amax already sits in {-1, 0, +1} --
    # and absmean is never one, because mean|W| is exactly (nonzero fraction) *
    # amax, so the rule shrinks every surviving weight by that fraction. Both
    # results hold whatever rule wrote the file. Only sparsity discriminates.
    print(f"  fixed point   absmean {absmean_fp}   absmax {absmax_fp}   "
          "(both tautological on\n                ternary input -- they name no rule)")
    print(f"  zero fraction {observed:.1%}   vs  absmean {GAUSSIAN_ZEROS['absmean']:.0%} / "
          f"absmax {GAUSSIAN_ZEROS['absmax']:.0%} on Gaussian weights")

    # Wide brackets, because the latent weights a real QAT run quantizes are
    # only roughly Gaussian; anything outside them is a different rule.
    if abs(observed - GAUSSIAN_ZEROS["absmean"]) <= 0.10:
        print("\n  absmean (BitNet b1.58) is the rule: this is the sparsity that\n"
              "  thresholding at half of mean|W| leaves behind.")
    elif abs(observed - GAUSSIAN_ZEROS["absmax"]) <= 0.15:
        print("\n  absmax is the rule: only a threshold at half the group maximum\n"
              "  zeroes a tensor that thoroughly.")
    else:
        print(f"\n  NEITHER rule explains the sparsity. {observed:.1%} zeros is nowhere near\n"
              f"  the {GAUSSIAN_ZEROS['absmax']:.0%} absmax would leave, which rules absmax out flatly, but it\n"
              f"  is also only about half the {GAUSSIAN_ZEROS['absmean']:.0%} absmean leaves on Gaussian weights.\n"
              "  Either the scale is roughly 0.5*mean|W| rather than mean|W|, or the\n"
              "  latent weights that were quantized were far from Gaussian -- which\n"
              "  is what converged straight-through training tends to produce, since\n"
              "  it drives latent mass off the threshold. Consistent with absmean,\n"
              "  not evidence for it: the rule is not identified from this file\n"
              "  alone, so do not hardcode absmean on the strength of it.")

    # Whether QAT can resume is a separate question, and the fixed-point columns
    # do answer that one -- a tautology is still true.
    print("\n  Resuming: a shadow initialised from the released weights is bit-exact\n"
          "  under an absmax forward, so step 0 perturbs nothing. Under absmean it\n"
          f"  moves every weight by {drift:.4f} on average: no sign flips and no new\n"
          "  zeros, but each group uniformly shrunk by its own nonzero fraction, so\n"
          "  the run would start from a quietly detuned model.")


if __name__ == "__main__":
    main()
