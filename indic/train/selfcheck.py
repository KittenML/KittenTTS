"""Prove the training pipeline works before there is any data to train it on.

There is no Indic corpus yet and no GPU budget spent, so the training code in
this package has never processed a real example. That is the normal situation at
this point in a project, and it is also how training pipelines ship with a mask
off by one, a frozen parameter that was meant to move, or a speaker vector that
never reaches the graph -- none of which a loss curve reports. Every one of
those is checkable today, on a 2-layer model over fabricated examples, in under
a minute on a CPU. This suite is that proof, and it is the actual deliverable;
the loop is only worth having because this can be run against it.

    python -m indic.train.selfcheck --device cpu --out indic/data/train_selfcheck.json
    python -m indic.train.selfcheck --full --checkpoint .../lm --device cuda

Each check prints PASS or FAIL, the run exits non-zero if any failed, and
`--out` writes the numbers as JSON.

Two rules shape how the checks are written, both learned the hard way:

  * **A check that cannot fail is worse than no check.** It reports green and
    retires the question. The ternary fixed-point test was exactly this -- on an
    already-ternary tensor `absmax` reproduces the input by algebra, whatever
    rule wrote the file, and the "confirmation" meant nothing. So the assertions
    here are paired with negative controls: a deliberately wrong variant that
    the check must *reject*. If a negative control passes, the check fails.
  * **Say what was not verified.** A skipped assertion is reported as skipped,
    with the reason, never folded into a pass.

One check the task asked for cannot be written as stated, and the honest version
is in `check_loss_mask`: "randomising tokens at non-target positions leaves the
loss unchanged" is false for any causal LM, because a non-target position before
the target span is context and changing it changes the prediction legitimately.
What *is* exactly invariant is the padding, and what independently pins the mask
is locating the target span in the token stream rather than trusting the
arithmetic that built it. Both are checked.
"""

import argparse
import copy
import glob
import itertools
import json
import pathlib
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.train import data, loop, qat

SPEAKER_DIM = 512

# Thresholds. Each one is a claim about what the check proves, so they are named
# and justified rather than inlined.
#
# A 2-layer model with 0.35M body parameters memorising 2 fabricated sequences of
# ~30 supervised tokens should reach a loss indistinguishable from zero; anything
# above this means the gradient is not actually reaching the parameters that
# produce those logits.
OVERFIT_LOSS = 0.05
# Held to the same bar through the straight-through estimator. The forward is
# ternary there, so the shadow has to find a ternary configuration that fits
# rather than an arbitrary one, and it was not obvious in advance that it could
# -- but it does, to 1.0e-4 against the dense run's 9.9e-5. Keeping the harder
# threshold is deliberate: a ratio-based bar would also pass a straight-through
# estimator that leaked only a trickle of gradient to the shadow weights.
OVERFIT_QAT_LOSS = OVERFIT_LOSS
# `absmean` on a float32 shadow is exact only up to the rounding of a 128-term
# sum. Measured on the released weights: 6.0e-8 absolute, 2.0e-6 relative to the
# group scale. The bar is set an order of magnitude above that and four orders
# below the error any of the negative controls produces.
FIXED_POINT_RELATIVE = 1e-5
# A speaker embedding that reaches the graph moves the loss by far more than
# this; one that does not moves it by exactly 0.
SPEAKER_LOSS_DELTA = 1e-4


class Result:
    """One check's verdict, with the numbers that justify it."""

    def __init__(self, name, passed, detail=None, note="", skipped=False):
        self.name = name
        self.passed = bool(passed)
        self.detail = detail or {}
        self.note = note
        self.skipped = bool(skipped)

    def as_dict(self):
        return {"check": self.name, "passed": self.passed, "skipped": self.skipped,
                "note": self.note, **self.detail}


# ── fixtures ──────────────────────────────────────────────────────────────────
def released_checkpoint(explicit=None):
    """The released `lm/` directory, if one is on this machine.

    Looked up in the Hugging Face cache rather than downloaded: a self-check that
    pulls a gigabyte before it can report is one nobody runs.
    """
    if explicit:
        return pathlib.Path(explicit)
    hits = glob.glob(str(pathlib.Path.home() / ".cache/huggingface/hub/"
                         "models--KittenML--kitten-tts-2/snapshots/*/lm"))
    return pathlib.Path(sorted(hits)[-1]) if hits else None


def toy_model(device, mode="full", qat_rule=None, seed=0):
    """A fresh 2-layer model with its speaker head, built identically every time."""
    loop.seed_everything(seed)
    token_map = data.tiny_token_map()
    model = loop.toy_model(token_map, dtype=torch.float32, seed=seed)
    loop.attach_speaker_head(model, speaker_dim=SPEAKER_DIM, dtype=torch.float32)
    if qat_rule is not None:
        # resume=False: a randomly initialised toy weight is not ternary, so
        # there is no released fixed point to preserve here.
        qat.wrap_ternary(model, rule=qat_rule, resume=False)
    if mode == "lora":
        model = loop.apply_lora(model, rank=8, alpha=16, dropout=0.0)
    model = model.to(device)
    loop.set_trainable(model, mode)
    return model, token_map


def toy_batch(token_map, indices=(0, 1), seed=0):
    dataset = data.SyntheticDataset(token_map, size=64, speaker_dim=SPEAKER_DIM, seed=seed)
    return data.Collator()([dataset[i] for i in indices])


def train_args(**overrides):
    """A `loop` argument namespace, parsed by `loop` itself.

    Built through `parse_args` rather than by hand so the suite exercises the
    same defaults and the same `--amp auto` resolution a real run gets.
    """
    args = loop.parse_args(["--toy", "--device", overrides.pop("device", "cpu")])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_loop(model, batches, args, start_step=0, optimizer=None, scheduler=None):
    """Drive `loop.run_steps` -- the real one, not a reimplementation."""
    optimizer = optimizer or loop.build_optimizer(model, args)
    scheduler = scheduler or loop.build_scheduler(optimizer, args)
    history = loop.run_steps(model, batches, optimizer, scheduler, args,
                             torch.device(args.device), start_step=start_step)
    return history, optimizer, scheduler


# ── a. overfit one batch ──────────────────────────────────────────────────────
def check_overfit(ctx):
    """The canonical proof that a training loop works.

    One fixed batch, repeated. A correct loop drives the loss on it to zero,
    because the model is free to memorise it; a loop with a detached graph, a
    dead optimizer group, a schedule that never leaves warmup or labels that do
    not line up with logits cannot, whatever else looks right.

    Run twice, once dense and once through the ternary straight-through
    estimator, and both must reach it. The second run is what shows gradient
    reaching the shadow weights rather than stopping at `round()`: a quantized
    forward that received no gradient would sit at its initial loss, and one that
    received a fraction of it would stall well short.
    """
    device = torch.device(ctx["device"])
    detail = {}
    passed = True

    for label, rule in (("dense", None), ("qat_absmax", "absmax")):
        model, token_map = toy_model(device, mode="full", qat_rule=rule, seed=1)
        batch = toy_batch(token_map, indices=(3, 7), seed=1)
        args = train_args(device=ctx["device"], steps=ctx["overfit_steps"],
                          lr=3e-3, lr_embed=3e-3, warmup=5, weight_decay=0.0,
                          log_every=0)
        history, _, _ = run_loop(model, itertools.repeat(batch), args)
        detail[f"{label}_first_loss"] = history[0]
        detail[f"{label}_final_loss"] = history[-1]
        detail[f"{label}_steps"] = len(history)
        threshold = OVERFIT_LOSS if rule is None else OVERFIT_QAT_LOSS
        detail[f"{label}_threshold"] = threshold
        detail[f"{label}_loss_ratio"] = history[-1] / max(history[0], 1e-9)
        ok = history[-1] < threshold
        detail[f"{label}_passed"] = ok
        passed = passed and ok

    # Negative control: the same loop with every parameter frozen but the
    # optimizer still stepping must NOT drive the loss down. If it does, the
    # losses being reported are not the losses being optimised.
    model, token_map = toy_model(device, mode="full", seed=1)
    batch = toy_batch(token_map, indices=(3, 7), seed=1)
    args = train_args(device=ctx["device"], steps=5, lr=0.0, lr_embed=0.0,
                      warmup=0, weight_decay=0.0, log_every=0)
    history, _, _ = run_loop(model, itertools.repeat(batch), args)
    control_flat = history[-1] == history[0]
    detail["zero_lr_control_delta"] = history[-1] - history[0]
    detail["zero_lr_control_flat"] = control_flat
    passed = passed and control_flat

    note = ("both the dense and the ternary-STE run memorise the batch; a zero-LR "
            "control does not move at all")
    return Result("overfit_one_batch", passed, detail, note)


# ── b. loss masking ───────────────────────────────────────────────────────────
def check_loss_mask(ctx):
    """Three claims about the label mask, none of which is an identity.

    1. **The mask is where the tokens say it is.** `build_example` computes the
       supervised span from the prompt's length; `data.supervised_span` finds it
       by locating `<speech_start>` and `<speech_end>` in the finished sequence.
       Two independent derivations, compared. An off-by-one in either shows up
       here, and neither is computed from the other.
    2. **Padding cannot touch the loss.** In a ragged batch the short row's pad
       positions sit at indices the long row uses for real tokens. Scrambling
       every pad id must leave the loss *bit-identical*: causal attention never
       looks forward at them and their labels are -100. If the mask has leaked
       one position past the end of a row, this is where it surfaces.
    3. **The target does touch the loss.** Rebuilding the batch with different
       target audio tokens must change it. This is what fails when the mask is
       empty -- a bug that otherwise reports a suspiciously stable loss rather
       than an error.

    The negative control for (2): scrambling a *supervised* position's input id
    must change the loss. Without it, "bit-identical" would also be satisfied by
    a forward pass that ignored its input entirely.
    """
    device = torch.device(ctx["device"])
    model, token_map = ctx["model_factory"](device, "embed")
    model.eval()
    detail = {}

    dataset = data.SyntheticDataset(token_map, size=64, speaker_dim=SPEAKER_DIM, seed=5)
    # Deliberately ragged: the whole point of (2) is pads that sit beside real
    # tokens of another row.
    items = sorted((dataset[i] for i in range(8)), key=lambda x: len(x["input_ids"]))
    batch = data.Collator()([items[0], items[-1]])
    detail["row_lengths"] = [int(m.sum()) for m in batch["attention_mask"]]
    detail["pad_positions"] = int((batch["attention_mask"] == 0).sum())

    # 1. the two derivations of the mask
    agreements = []
    for row in range(batch["input_ids"].shape[0]):
        length = int(batch["attention_mask"][row].sum())
        ids = batch["input_ids"][row, :length]
        from_labels = (batch["labels"][row, :length] != data.IGNORE_INDEX).nonzero().flatten()
        span = data.supervised_span(ids, token_map)
        expected = torch.arange(span[0], span[1] + 1) if span else torch.tensor([])
        agreements.append(bool(torch.equal(from_labels, expected)))
    detail["mask_matches_token_scan"] = all(agreements)
    detail["rows_checked"] = len(agreements)

    with torch.no_grad():
        base = float(loop.forward_batch(model, batch, device).loss)
    detail["baseline_loss"] = base

    # 2. pads scrambled, loss must be bit-identical
    generator = torch.Generator().manual_seed(11)
    padded = copy.deepcopy(batch)
    pad_cells = padded["attention_mask"] == 0
    padded["input_ids"][pad_cells] = torch.randint(
        0, token_map.speech_start_id, (int(pad_cells.sum()),), generator=generator)
    with torch.no_grad():
        pad_loss = float(loop.forward_batch(model, padded, device).loss)
    detail["pad_scrambled_loss"] = pad_loss
    detail["pad_scrambled_delta"] = pad_loss - base
    pad_invariant = pad_loss == base

    # negative control for (2): a supervised position must matter
    control = copy.deepcopy(batch)
    row = 0
    span = data.supervised_span(control["input_ids"][row][:detail["row_lengths"][0]], token_map)
    control["input_ids"][row, span[0]] = token_map.audio_id_base + (
        (int(control["input_ids"][row, span[0]]) - token_map.audio_id_base + 1)
        % token_map.num_audio_tokens)
    with torch.no_grad():
        control_loss = float(loop.forward_batch(model, control, device).loss)
    detail["supervised_token_control_delta"] = control_loss - base
    control_moves = control_loss != base

    # 3. a different target must change the loss
    other = data.Collator()([dataset[20], dataset[21]])
    with torch.no_grad():
        other_loss = float(loop.forward_batch(model, other, device).loss)
    detail["different_target_loss"] = other_loss
    target_matters = other_loss != base

    detail.update({"pad_invariant": pad_invariant,
                   "supervised_control_moves_loss": control_moves,
                   "target_changes_loss": target_matters})
    passed = (all(agreements) and pad_invariant and control_moves and target_matters)
    return Result("loss_masking", passed, detail,
                  "mask located two independent ways; padding bit-invariant; "
                  "supervised positions provably do move the loss")


# ── c. gradient flow ──────────────────────────────────────────────────────────
def check_grad_flow(ctx):
    """Trainable parameters get gradients; frozen ones get none. Per mode.

    `requires_grad=True` and "receives a gradient" are different claims: a
    parameter the forward pass never reaches is trainable and immovable at once,
    and the optimizer will report it among its parameter counts either way.

    One optimizer step is taken before gradients are measured, because LoRA's B
    matrix is initialised to zero -- at step 0 the gradient with respect to A is
    exactly zero by construction, which is correct behaviour that would read as
    a dead parameter.
    """
    device = torch.device(ctx["device"])
    detail = {}
    passed = True
    modes = ctx["modes"]

    for mode in modes:
        try:
            model, token_map = ctx["model_factory"](device, mode)
        except ImportError as error:
            detail[mode] = {"skipped": f"{type(error).__name__}: {error}"}
            continue
        batch = toy_batch(token_map, indices=(2, 9), seed=3)
        args = train_args(device=ctx["device"], steps=1, lr=1e-3, lr_embed=1e-3,
                          warmup=0, log_every=0)
        run_loop(model, itertools.repeat(batch), args)

        model.zero_grad(set_to_none=True)
        output = loop.forward_batch(model, batch, device)
        output.loss.backward()

        trainable_zero, frozen_with_grad = [], []
        n_trainable = n_frozen = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                n_trainable += 1
                if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0:
                    trainable_zero.append(name)
            else:
                n_frozen += 1
                if parameter.grad is not None and float(parameter.grad.abs().sum()) != 0.0:
                    frozen_with_grad.append(name)

        speaker_grad = float(model.get_submodule(
            [n for n, _ in model.named_modules() if n.endswith("spk_proj.0")][0]
        ).weight.grad.abs().sum())
        ok = not trainable_zero and not frozen_with_grad and speaker_grad > 0
        detail[mode] = {"trainable_tensors": n_trainable, "frozen_tensors": n_frozen,
                        "trainable_without_grad": trainable_zero[:5],
                        "frozen_with_grad": frozen_with_grad[:5],
                        "spk_proj_grad_abs_sum": speaker_grad, "passed": ok}
        passed = passed and ok

    return Result("gradient_flow", passed, detail,
                  "measured after one optimizer step so LoRA's zero-initialised "
                  "B matrix does not read as a dead parameter")


# ── d. QAT fixed point ────────────────────────────────────────────────────────
def check_qat_fixed_point(ctx):
    """Step 0 of QAT must reproduce the released weights, under either rule.

    This is the check that was previously got wrong, so what it does *not* claim
    is worth stating: it does not identify which rule produced the checkpoint.
    `absmax` reproducing a ternary tensor is an algebraic identity and names no
    rule (`indic/bench/ternary_check.py`). What it genuinely tests is the
    implementation in `qat.py` -- the grouping axis, the group width, the
    `clamp(-1, 1)` and the shadow's dtype -- each of which is pinned by a
    negative control that must come out wrong:

        absmean on an unrescaled shadow   detuned, ~0.03 mean drift
        absmean without the clamp         1/f exceeds 1.5 at f < 0.667 and
                                          round() returns 2
        absmean on a bf16 shadow          the 128-term sum loses precision
        absmax grouped along rows         groups then span several scales

    If any negative control reproduces the weights, the positive result proves
    nothing and this check fails.
    """
    detail = {}
    checkpoint = ctx["checkpoint"]
    packed = checkpoint / "model-ternary.safetensors" if checkpoint else None
    if packed is not None and packed.is_file():
        keys = ["model.layers.0.self_attn.q_proj.weight",
                "model.layers.13.mlp.down_proj.weight"]
        tensors = qat.load_released_tensors(packed, keys)
        detail["source"] = str(packed)
    else:
        tensors = {"synthetic.ternary": _synthetic_ternary()}
        detail["source"] = "synthetic"
        detail["skipped_assertion"] = (
            "released weights not cached: the fixed point was verified on a "
            "fabricated ternary tensor, which exercises the implementation but "
            "not the checkpoint's own group statistics (f range, scale spread)")

    passed = True
    for name, weight in tensors.items():
        row = {"shape": list(weight.shape), "dtype": str(weight.dtype)}
        reference = weight.float()
        scale = reference.abs().reshape(reference.shape[0], -1, qat.GROUP_SIZE).amax(dim=2)
        row["max_group_scale"] = float(scale.max())
        fraction = (reference != 0).float().reshape(
            reference.shape[0], -1, qat.GROUP_SIZE).mean(dim=2)
        row["nonzero_fraction_min"] = float(fraction.min())
        row["nonzero_fraction_max"] = float(fraction.max())

        for rule in qat.RULES:
            shadow = qat.shadow_from_released(weight, rule)
            quantized = qat.quantize(shadow, rule)
            drift = float((quantized - reference).abs().max())
            relative = drift / max(float(scale.max()), 1e-30)
            exact = bool(torch.equal(quantized, reference))
            signs_exact = bool(torch.equal(torch.sign(quantized), torch.sign(reference)))
            ok = (exact or relative < FIXED_POINT_RELATIVE) and signs_exact
            row[f"{rule}_bit_exact"] = exact
            row[f"{rule}_max_abs_drift"] = drift
            row[f"{rule}_relative_drift"] = relative
            row[f"{rule}_sign_pattern_exact"] = signs_exact
            row[f"{rule}_passed"] = ok
            passed = passed and ok
            row[f"{rule}_collapsed_groups"] = qat.collapsed_group_fraction(shadow, rule)

        controls = _fixed_point_controls(weight)
        row["negative_controls"] = controls
        # Every control must be wrong. One that reproduces the weights would mean
        # the positive result above is an identity rather than a measurement.
        controls_rejected = all(v["reproduces"] is False for v in controls.values())
        row["negative_controls_all_rejected"] = controls_rejected
        passed = passed and controls_rejected
        detail[name] = row

    return Result("qat_fixed_point", passed, detail,
                  "tests the implementation, not which rule trained the checkpoint")


def _synthetic_ternary(rows=256, cols=256, seed=7):
    """A ternary tensor with the released weights' group statistics.

    Per-group scale and nonzero fraction drawn to match what was measured on the
    checkpoint (f between 0.6 and 1.0), because the `clamp(-1, 1)` only matters
    below f = 2/3 and a uniformly dense fabrication would never reach there.
    """
    generator = torch.Generator().manual_seed(seed)
    groups = cols // qat.GROUP_SIZE
    scale = (torch.rand(rows, groups, 1, generator=generator) * 0.05 + 0.005)
    keep = torch.rand(rows, groups, qat.GROUP_SIZE, generator=generator)
    threshold = torch.rand(rows, groups, 1, generator=generator) * 0.4
    signs = torch.where(torch.rand(rows, groups, qat.GROUP_SIZE, generator=generator) < 0.5,
                        -1.0, 1.0)
    trits = torch.where(keep < threshold, torch.zeros_like(signs), signs)
    return (trits * scale).reshape(rows, cols).to(torch.bfloat16)


def _fixed_point_controls(weight):
    """Deliberately wrong variants of the step-0 forward. All must be wrong."""
    reference = weight.float()
    group = qat.GROUP_SIZE
    grouped = reference.reshape(reference.shape[0], -1, group)
    fraction = (grouped != 0).float().mean(dim=2, keepdim=True).clamp(min=qat.EPS)
    shadow = grouped / fraction

    controls = {}

    unrescaled = qat.quantize(weight, "absmean")
    controls["absmean_without_rescale"] = _control(unrescaled, reference)

    scale = shadow.abs().mean(dim=2, keepdim=True).clamp(min=qat.EPS)
    controls["absmean_without_clamp"] = _control(
        ((shadow / scale).round() * scale).reshape(reference.shape), reference)

    bf16_shadow = shadow.to(torch.bfloat16).float()
    bf16_scale = bf16_shadow.abs().mean(dim=2, keepdim=True).clamp(min=qat.EPS)
    controls["absmean_bf16_shadow"] = _control(
        ((bf16_shadow / bf16_scale).round().clamp(-1, 1) * bf16_scale).reshape(
            reference.shape), reference)

    # Grouping along rows instead of along input features. Only meaningful when
    # the tensor has enough rows to regroup, which the released ones do.
    if reference.shape[0] % group == 0:
        transposed = reference.t().contiguous()
        regrouped = qat.quantize(transposed, "absmax").t().contiguous()
        controls["absmax_grouped_along_rows"] = _control(regrouped, reference)

    return controls


def _control(candidate, reference):
    drift = float((candidate - reference).abs().max())
    return {"reproduces": bool(torch.equal(candidate, reference)),
            "max_abs_drift": drift}


# ── e. pack round trip ────────────────────────────────────────────────────────
def check_pack_roundtrip(ctx):
    """Quantize -> pack -> unpack must return the identical tensor.

    Two halves. The tensor half goes through `ternary.pack_state_dict` directly,
    including the released weights when they are on the machine. The model half
    runs the real export path end to end: wrap a model in QAT, `export_packed`
    it, and read the file back with the same unpacker inference uses. That
    second half is what would catch an export that silently stored a weight
    dense -- the file still loads, the model still runs, and it is three times
    the size for no stated reason.
    """
    from indic.train import load_kittentts2

    ternary = load_kittentts2("ternary")
    detail = {}
    tensors = {"synthetic": _synthetic_ternary()}
    checkpoint = ctx["checkpoint"]
    packed_file = checkpoint / "model-ternary.safetensors" if checkpoint else None
    if packed_file is not None and packed_file.is_file():
        tensors.update(qat.load_released_tensors(
            packed_file, ["model.layers.0.self_attn.o_proj.weight"]))
    else:
        detail["skipped_assertion"] = "released weights not cached: synthetic tensors only"

    passed = True
    for name, tensor in tensors.items():
        blobs, metadata = ternary.pack_state_dict({name: tensor})
        restored = ternary.unpack_state_dict(blobs, metadata)[name]
        exact = bool(torch.equal(restored, tensor))
        detail[f"tensor::{name}"] = {
            "exact": exact, "packed": f"{name}::codes" in blobs,
            "bytes_in": tensor.numel() * tensor.element_size(),
            "bytes_out": int(blobs[f"{name}::codes"].nbytes
                             + blobs[f"{name}::scale"].nbytes)
            if f"{name}::codes" in blobs else None}
        passed = passed and exact and f"{name}::codes" in blobs

    device = torch.device(ctx["device"])
    model, _ = toy_model(device, mode="full", qat_rule="absmax", seed=2)
    expected = qat.ternary_state_dict(model)
    with tempfile.TemporaryDirectory() as directory:
        # Laid out as a repository, so the last read below can go through the
        # inference package's own loader rather than a stand-in for it.
        repo = pathlib.Path(directory)
        (repo / "lm").mkdir()
        path = repo / "lm" / "model-ternary.safetensors"
        summary = qat.export_packed(model, path)
        restored = qat.load_released_state(path)
        mismatched = [k for k, v in expected.items()
                      if k not in restored or not torch.equal(restored[k], v)]
        detail["inference_loader"] = _inference_loader_probe(repo, expected)
    detail["export"] = {**summary, "tensors": len(expected),
                        "mismatched": mismatched[:5],
                        "missing": sorted(set(expected) - set(restored))[:5]}
    export_ok = not mismatched and set(expected) == set(restored) and \
        summary["packed_tensors"] == len([n for n, m in model.named_modules()
                                          if isinstance(m, qat.TernaryLinear)]) and \
        detail["inference_loader"].get("exact") is not False
    detail["export"]["passed"] = export_ok
    passed = passed and export_ok

    return Result("pack_roundtrip", passed, detail,
                  "export goes through ternary.py, so the result loads in the "
                  "existing inference path")


def _inference_loader_probe(repo, expected):
    """Read the exported file back with `weights.load_lm_weights` itself.

    The claim the export has to support is "the existing inference path and the
    llama.cpp fork keep working", and only the real loader can settle the first
    half of it. Skipped, not failed, when the inference package is not
    importable -- the training box deliberately does not need it.
    """
    try:
        from kittenml.kittentts2.weights import load_lm_weights
    except Exception as error:
        return {"skipped": f"inference package not importable: {type(error).__name__}"}

    config = {"lm_dir": "lm", "lm_packed": "lm/model-ternary.safetensors"}
    state = load_lm_weights(repo, config, "packed")
    mismatched = [k for k, v in expected.items()
                  if k not in state or not torch.equal(state[k], v)]
    return {"tensors": len(state), "mismatched": mismatched[:3],
            "exact": not mismatched and set(state) == set(expected)}


# ── f. determinism ────────────────────────────────────────────────────────────
def check_determinism(ctx):
    """Same seed, same losses. Exactly, not approximately.

    Without this every other check in the suite is unfalsifiable: a run whose
    loss moves between invocations cannot be compared to anything, and the
    resume check below is comparing two runs.

    The negative control is a different seed, which must produce a *different*
    first loss -- otherwise "identical" would also be satisfied by a loop whose
    loss does not depend on initialisation at all.
    """
    device = torch.device(ctx["device"])
    histories = []
    for seed in (4, 4, 5):
        model, token_map = toy_model(device, mode="full", seed=seed)
        dataset = data.SyntheticDataset(token_map, size=32, speaker_dim=SPEAKER_DIM,
                                        seed=seed)
        collator = data.Collator()
        batches = (collator([dataset[(2 * i) % 32], dataset[(2 * i + 1) % 32]])
                   for i in itertools.count())
        args = train_args(device=ctx["device"], steps=6, lr=1e-3, lr_embed=1e-3,
                          warmup=2, log_every=0)
        history, _, _ = run_loop(model, batches, args)
        histories.append(history)

    identical = histories[0] == histories[1]
    different = histories[0][0] != histories[2][0]
    detail = {"run_a": histories[0], "run_b": histories[1],
              "other_seed_first_loss": histories[2][0],
              "same_seed_identical": identical,
              "different_seed_differs": different}
    return Result("determinism", identical and different, detail,
                  "exact equality, and a different seed proves the comparison "
                  "is not vacuous")


# ── g. checkpoint resume ──────────────────────────────────────────────────────
def check_resume(ctx):
    """A run interrupted and resumed must continue, not restart.

    Eight steps straight through, against four steps, a save, a fresh model and
    optimizer, a load, and four more. The second halves must match step for
    step. Anything the save forgets -- the optimizer moments, the scheduler's
    step counter, the RNG streams -- shows up as a divergence at the seam, which
    in a real run is invisible because nobody has the uninterrupted curve to
    compare against.

    The batch stream is a function of the step index, so what is being compared
    is the training state and not the data order.
    """
    device = torch.device(ctx["device"])
    total, cut = 8, 4

    def stream(token_map, start):
        dataset = data.SyntheticDataset(token_map, size=32, speaker_dim=SPEAKER_DIM, seed=6)
        collator = data.Collator()
        for step in itertools.count(start):
            yield collator([dataset[(2 * step) % 32], dataset[(2 * step + 1) % 32]])

    horizon = train_args(device=ctx["device"], steps=total, lr=1e-3, lr_embed=2e-3,
                         warmup=2, log_every=0)

    model, token_map = toy_model(device, mode="full", seed=8)
    straight, _, _ = run_loop(model, stream(token_map, 0), horizon)

    # The same run, stopped at `cut`. The scheduler keeps the full horizon --
    # rebuilding it against the shortened run would put the resumed half on a
    # different cosine and the comparison would fail for the wrong reason.
    stop_early = train_args(device=ctx["device"], steps=cut, lr=1e-3, lr_embed=2e-3,
                            warmup=2, log_every=0)
    model, token_map = toy_model(device, mode="full", seed=8)
    optimizer = loop.build_optimizer(model, horizon)
    scheduler = loop.build_scheduler(optimizer, horizon)
    first_half = loop.run_steps(model, stream(token_map, 0), optimizer, scheduler,
                                stop_early, device, start_step=0)

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "mid.pt"
        loop.save_checkpoint(path, model, optimizer, scheduler, cut, horizon)
        # A fresh model at a *different* seed, so a resume that failed to load
        # would diverge immediately rather than coincidentally agreeing.
        fresh, token_map = toy_model(device, mode="full", seed=99)
        fresh_optimizer = loop.build_optimizer(fresh, horizon)
        fresh_scheduler = loop.build_scheduler(fresh_optimizer, horizon)
        step = loop.load_checkpoint(path, fresh, fresh_optimizer, fresh_scheduler)
    resumed = loop.run_steps(fresh, stream(token_map, step), fresh_optimizer,
                             fresh_scheduler, horizon, device, start_step=step)

    detail = {"straight_through": straight, "first_half": first_half,
              "resumed_from": step, "resumed_tail": resumed,
              "expected_tail": straight[step:],
              "first_half_identical": first_half == straight[:cut],
              "tail_identical": resumed == straight[step:]}
    passed = first_half == straight[:cut] and resumed == straight[step:]
    return Result("checkpoint_resume", passed, detail,
                  "optimizer moments, scheduler position and RNG all restored")


# ── h. speaker conditioning ───────────────────────────────────────────────────
def check_speaker(ctx):
    """The speaker vector has to be in the training graph, not merely available.

    `speaker.py` injects it through a forward *pre-hook* that fires only at
    prefill, and the research notes flag it as an open question whether a
    teacher-forced training forward counts. It does: with no cache and no
    `cache_position`, `past_key_values` is None and the hook takes the prefill
    branch. That is asserted here directly, by reading the `inputs_embeds` the
    hook hands to `Qwen3Model` and comparing position 0 against `spk_proj`'s own
    output.

    What is *not* safe is how the embedding is passed. The inference hook reads
    `input_ids` from keyword arguments only, so two perfectly ordinary calls
    skip injection with no error at all:

        model(ids, labels=...)                  input_ids positional
        model(inputs_embeds=emb, labels=...)    what a trainer naturally does

    Both are measured below against the real `kittenml.kittentts2.speaker` hook,
    and the assertion is exact rather than approximate: their loss must equal the
    loss of a forward with *no speaker at all*, bit for bit, and must differ from
    the correctly-conditioned one. That is what makes this check go red if anyone
    later switches the loop to either call.

    `loop.attach_speaker_head` closes both -- the positional one by reading
    `args[0]`, the `inputs_embeds` one by refusing the call -- and both fixes are
    asserted here too, so removing either turns this check red as well.

    Structure checks on the sequence itself (marker order, prefix identity,
    role-token layout) belong to `indic/bench/teacher_forced_loss.py
    --self-test` and are not repeated here; this check is about the training
    graph.
    """
    device = torch.device(ctx["device"])
    model, token_map = ctx["model_factory"](device, "embed")
    batch = toy_batch(token_map, indices=(1, 4), seed=12)
    detail = {}

    captured = {}

    def capture(module, args, kwargs):
        captured["inputs_embeds"] = kwargs.get("inputs_embeds")
        captured["input_ids"] = kwargs.get("input_ids")
        return args, kwargs

    inner = model.get_submodule(
        [n for n, m in model.named_modules() if type(m).__name__ == "Qwen3Model"][0])
    handle = inner.register_forward_pre_hook(capture, with_kwargs=True)
    model.train()
    output = loop.forward_batch(model, batch, device)
    output.loss.backward()
    handle.remove()

    embeds = captured.get("inputs_embeds")
    detail["model_received_inputs_embeds"] = embeds is not None
    detail["model_received_input_ids"] = captured.get("input_ids") is not None
    if embeds is None:
        return Result("speaker_conditioning", False, detail,
                      "the pre-hook did not convert input_ids to inputs_embeds")

    head = model.get_submodule([n for n, _ in model.named_modules()
                                if n.endswith("spk_proj")][0])
    # The head's dtype, not the batch's: the released checkpoint is bf16 and
    # handing its Linear a float32 vector raises rather than casting. The hook
    # does the same cast (`embedding.to(embeds.dtype)`), so this reproduces its
    # arithmetic exactly instead of approximating it.
    head_dtype = next(head.parameters()).dtype
    # bf16 carries about three decimal digits, so a float32 tolerance would
    # reject a correct injection on the released weights.
    tolerance = 1e-6 if head_dtype == torch.float32 else 3e-3
    with torch.no_grad():
        projected = head(batch["speaker_embedding"].to(device=device, dtype=head_dtype))
        start_embedding = model.get_input_embeddings()(
            batch["input_ids"][:, :1].to(device)).squeeze(1)
    actual = embeds[:, 0, :].detach().float()
    detail["dtype"] = str(head_dtype)
    detail["position_zero_vs_spk_proj"] = float((actual - projected.float()).abs().max())
    detail["position_zero_vs_start_token"] = float(
        (actual - start_embedding.float()).abs().max())
    position_zero = detail["position_zero_vs_spk_proj"] <= tolerance
    detail["position_zero_is_spk_proj"] = bool(position_zero)
    # Prefill was taken: if it had not been, position 0 would hold the [START]
    # token's embedding instead.
    detail["position_zero_is_start_token"] = bool(
        detail["position_zero_vs_start_token"] <= tolerance)

    speaker_grad = float(head[0].weight.grad.abs().sum())
    detail["spk_proj_grad_abs_sum"] = speaker_grad

    with torch.no_grad():
        base = float(loop.forward_batch(model, batch, device).loss)
        swapped = copy.deepcopy(batch)
        swapped["speaker_embedding"] = torch.flip(swapped["speaker_embedding"], dims=[0])
        flipped = float(loop.forward_batch(model, swapped, device).loss)
        zeroed = copy.deepcopy(batch)
        zeroed["speaker_embedding"] = torch.zeros_like(zeroed["speaker_embedding"])
        zero_loss = float(loop.forward_batch(model, zeroed, device).loss)
    detail["loss_speakers"] = base
    detail["loss_speakers_swapped"] = flipped
    detail["loss_speaker_zero"] = zero_loss
    detail["swap_delta"] = abs(flipped - base)
    detail["zero_delta"] = abs(zero_loss - base)

    detail.update(_guard_probe(model, batch, device))
    detail["inference_hook"] = _inference_hook_probe(device)

    hook = detail["inference_hook"]
    # Skipped only when the inference package is not installed; a present probe
    # that fails to reproduce the trap means the trap is gone or the probe is
    # broken, and either way the fix above needs rereading.
    hook_ok = "skipped" in hook or hook.get("both_bypasses_reproduced") is True
    passed = (position_zero and not detail["position_zero_is_start_token"]
              and speaker_grad > 0
              and detail["swap_delta"] > SPEAKER_LOSS_DELTA
              and detail["zero_delta"] > SPEAKER_LOSS_DELTA
              and detail["missing_speaker_raises"]
              and detail["inputs_embeds_raises"]
              and detail["positional_call_matches_keyword"]
              and hook_ok)
    return Result("speaker_conditioning", passed, detail,
                  "prefill branch confirmed for teacher-forced training; "
                  "positional-call hazard measured against the inference hook")


def _guard_probe(model, batch, device):
    """The two ways the fixed hook has to behave, on the training model itself.

    A positional call must now give *exactly* the keyword call's loss -- the
    fix is reading `args[0]`, and a loss that merely came close would mean
    something else changed. An `inputs_embeds` call must raise, because there
    are no ids for the hook to overwrite and no way to condition the batch.
    """
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    speaker = batch["speaker_embedding"].to(device)
    out = {}

    with torch.no_grad():
        keyword = float(model(input_ids=ids, attention_mask=mask, labels=labels,
                              speaker_embedding=speaker, use_cache=False).loss)
        positional = float(model(ids, attention_mask=mask, labels=labels,
                                 speaker_embedding=speaker, use_cache=False).loss)
    out["keyword_call_loss"] = keyword
    out["positional_call_loss"] = positional
    out["positional_call_matches_keyword"] = positional == keyword

    for name, call in (("missing_speaker_raises",
                        lambda: model(input_ids=ids, attention_mask=mask,
                                      labels=labels, use_cache=False)),
                       ("inputs_embeds_raises",
                        lambda: model(inputs_embeds=model.get_input_embeddings()(ids),
                                      attention_mask=mask, labels=labels,
                                      speaker_embedding=speaker, use_cache=False))):
        raised = False
        try:
            with torch.no_grad():
                call()
        except Exception as error:
            raised = "speaker conditioning" in str(error)
        out[name] = raised
    return out


def _inference_hook_probe(device):
    """Reproduce the silent skip in `speaker.py`'s own hook, to justify the fix.

    Runs the real `attach_speaker_head` -- not a copy of it -- on a toy model and
    calls it the way a training loop naturally would, with `input_ids`
    positional. The result is the evidence that `loop.attach_speaker_head` is a
    fix rather than a gratuitous divergence.
    """
    try:
        from kittenml.kittentts2.speaker import attach_speaker_head
    except Exception as error:
        return {"skipped": f"inference package not importable: {type(error).__name__}"}

    loop.seed_everything(0)
    token_map = data.tiny_token_map()
    model = loop.toy_model(token_map, dtype=torch.float32, seed=0)
    hidden = int(model.config.hidden_size)
    state = {"spk_proj.0.weight": torch.randn(hidden, SPEAKER_DIM) * 0.02,
             "spk_proj.0.bias": torch.zeros(hidden),
             "spk_proj.1.weight": torch.ones(hidden),
             "spk_proj.1.bias": torch.zeros(hidden)}
    attach_speaker_head(model, None, device, torch.float32, spk_dim=SPEAKER_DIM,
                        state=state)
    # attach_speaker_head returns the head in eval(); a gradient needs it live.
    model.spk_proj.train()
    for parameter in model.spk_proj.parameters():
        parameter.requires_grad_(True)
    model = model.to(device)
    batch = toy_batch(token_map, indices=(1, 4), seed=12)
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    speaker = batch["speaker_embedding"].to(device)
    model._pending_speaker_embedding = speaker

    keyword = model(input_ids=ids, attention_mask=mask, labels=labels,
                    speaker_embedding=speaker, use_cache=False)
    keyword.loss.backward()
    grad = model.spk_proj[0].weight.grad
    keyword_grad = 0.0 if grad is None else float(grad.abs().sum())
    keyword_loss = float(keyword.loss.detach())

    # The reference the two bypasses have to equal: position 0 left holding the
    # [START] token's own embedding, which is what "no speaker conditioning at
    # all" means. Built by handing the model embeddings directly, so the hook
    # has no ids to read and provably does nothing.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        plain = model.get_input_embeddings()(ids)
        unconditioned = float(model(inputs_embeds=plain, attention_mask=mask,
                                    labels=labels, use_cache=False).loss)

    model.zero_grad(set_to_none=True)
    positional = model(ids, attention_mask=mask, labels=labels, use_cache=False)
    positional.loss.backward()
    grad = model.spk_proj[0].weight.grad
    positional_grad = 0.0 if grad is None else float(grad.abs().sum())
    positional_loss = float(positional.loss.detach())

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        embeds_call = float(model(inputs_embeds=plain, attention_mask=mask,
                                  labels=labels, speaker_embedding=speaker,
                                  use_cache=False).loss)

    positional_bypassed = positional_loss == unconditioned and positional_grad == 0.0
    embeds_bypassed = embeds_call == unconditioned
    return {"keyword_call_loss": keyword_loss,
            "keyword_call_spk_grad": keyword_grad,
            "unconditioned_loss": unconditioned,
            "positional_call_loss": positional_loss,
            "positional_call_spk_grad": positional_grad,
            "inputs_embeds_call_loss": embeds_call,
            "keyword_call_differs_from_unconditioned": keyword_loss != unconditioned,
            "positional_call_is_unconditioned": positional_bypassed,
            "inputs_embeds_call_is_unconditioned": embeds_bypassed,
            "both_bypasses_reproduced": bool(positional_bypassed and embeds_bypassed
                                             and keyword_loss != unconditioned),
            "loss_difference": abs(positional_loss - keyword_loss)}


# ── driver ────────────────────────────────────────────────────────────────────
CHECKS = [
    ("overfit_one_batch", check_overfit),
    ("loss_masking", check_loss_mask),
    ("gradient_flow", check_grad_flow),
    ("qat_fixed_point", check_qat_fixed_point),
    ("pack_roundtrip", check_pack_roundtrip),
    ("determinism", check_determinism),
    ("checkpoint_resume", check_resume),
    ("speaker_conditioning", check_speaker),
]

# Checks that run many optimizer steps. On the released 28-layer model that is
# hours of CPU for a property that is about the loop, not the checkpoint, so
# --full runs the single-forward checks only and says so.
MULTI_STEP = {"overfit_one_batch", "determinism", "checkpoint_resume"}


def full_model_factory(checkpoint, device, dtype):
    """Build the released 28-layer model once and re-freeze it per mode."""
    cache = {}

    def factory(_device, mode):
        if "model" not in cache:
            cache["model"] = loop.released_model(checkpoint, dtype=dtype).to(device)
            config = json.loads((pathlib.Path(checkpoint).parent / "config.json").read_text())
            cache["token_map"] = data.TokenMap.from_config(config)
        model = cache["model"]
        if mode != "embed":
            raise ImportError(
                f"--full runs {mode!r} nowhere but a large GPU: LoRA would re-wrap "
                "the released model and 'full' holds 1.7B gradients. The toy suite "
                "covers both modes.")
        loop.set_trainable(model, mode)
        model.zero_grad(set_to_none=True)
        return model, cache["token_map"]

    return factory


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--full", action="store_true",
                    help="run the single-forward checks against the released 28-layer model")
    ap.add_argument("--checkpoint", help="path to the repository's lm/ directory")
    ap.add_argument("--full-dtype", default="auto", choices=["auto", "fp32", "bf16"],
                    help="dtype for the released model under --full; bf16 halves the "
                         "7 GB a float32 copy needs on a CPU")
    ap.add_argument("--only", action="append", default=[],
                    help="run only these checks, repeatable")
    ap.add_argument("--overfit-steps", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads; the toy shapes are too small to use "
                         "a whole machine and oversubscribing makes the suite slower")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    checkpoint = released_checkpoint(args.checkpoint)
    context = {
        "device": args.device,
        "checkpoint": checkpoint,
        "overfit_steps": args.overfit_steps,
        "modes": ["embed", "lora", "full"],
        "model_factory": lambda device, mode: toy_model(device, mode=mode),
        "full": args.full,
    }
    if args.full:
        if checkpoint is None:
            raise SystemExit("--full needs the released checkpoint; pass --checkpoint")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(
            args.full_dtype, torch.float32 if args.device == "cpu" else torch.bfloat16)
        context["model_factory"] = full_model_factory(checkpoint, args.device, dtype)
        context["modes"] = ["embed"]

    selected = [(name, fn) for name, fn in CHECKS
                if not args.only or name in args.only]
    results = []
    for name, check in selected:
        if args.full and name in MULTI_STEP:
            results.append(Result(name, True, {}, skipped=True,
                                  note="skipped under --full: many optimizer steps on "
                                       "the 28-layer model, and the property is about "
                                       "the loop rather than the checkpoint"))
            continue
        started = time.time()
        try:
            result = check(context)
        except Exception as error:                          # a crash is a failure
            import traceback
            result = Result(name, False, {"exception": traceback.format_exc()[-2000:]},
                            note=f"{type(error).__name__}: {error}")
        result.detail["seconds"] = round(time.time() - started, 2)
        results.append(result)
        print(f"{'SKIP' if result.skipped else 'PASS' if result.passed else 'FAIL':>4}  "
              f"{result.name:<24} {result.detail['seconds']:>6.2f}s  {result.note}")

    report(results)
    payload = {"device": args.device, "full": args.full,
               "checkpoint": str(checkpoint) if checkpoint else None,
               "results": [r.as_dict() for r in results]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}")

    failed = [r.name for r in results if not r.passed and not r.skipped]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    return 0


def report(results):
    print("\n== Self-check ==")
    for result in results:
        status = "skip" if result.skipped else "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}")
        for key, value in _flatten(result.detail):
            print(f"      {key:<46} {value}")
        if result.note:
            print(f"      -- {result.note}")


def _flatten(detail, prefix="", depth=0):
    """Nested detail as one line per leaf.

    Printed flat because a check's numbers are the evidence for its verdict, and
    a dict truncated to fit a terminal line hides exactly the sub-result that
    explains a failure.
    """
    for key, value in detail.items():
        if key == "seconds":
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict) and depth < 2:
            yield from _flatten(value, f"{name}.", depth + 1)
        elif isinstance(value, float):
            yield name, f"{value:.6g}"
        elif isinstance(value, list) and value and isinstance(value[0], float):
            yield name, "[" + ", ".join(f"{v:.6g}" for v in value) + "]"
        else:
            yield name, str(value)[:160]


if __name__ == "__main__":
    sys.exit(main())
