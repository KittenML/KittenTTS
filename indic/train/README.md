# indic/train

Training for the Indic adaptation of KittenTTS 2, and the suite that proves the
pipeline is correct before there is a corpus to train it on.

There is no Indic data yet and no GPU time spent, so none of this code has seen
a real example. The self-check is what stands in for that: it runs a 2-layer
model over fabricated examples on a CPU in about a minute and asserts the
properties that a loss curve would not have reported anyway.

```
python -m indic.train.selfcheck --device cpu --out indic/data/train_selfcheck.json
python -m indic.train.loop --toy --steps 20 --device cpu --mode full
```

`selfcheck` prints PASS/FAIL per check, exits non-zero if any failed, and writes
the numbers as JSON. Last run on CPU: all eight pass
(`indic/data/train_selfcheck.json`), and under `--full --full-dtype bf16` the
two single-forward checks pass against the released 28-layer checkpoint
(`indic/data/train_selfcheck_full.json`).

## The modules

| file | what it is |
|---|---|
| `qat.py` | ternary quantization-aware training: per-128-group `absmax`/`absmean` forward with a straight-through estimator, shadow initialisation that makes step 0 a fixed point, wrap/unwrap of the 196 body linears, and an export that packs back through `kittenml/kittentts2/ternary.py` |
| `data.py` | the training example, assembled by `prompt.py` itself; loss on target audio tokens and `<speech_end>` only; `SyntheticDataset`, a `ShardDataset` stub, and a right-padding collator |
| `loop.py` | bf16, gradient checkpointing, accumulation, AdamW (8-bit optional), warmup + cosine with a separate embedding LR, save/resume, JSONL logging, seeding; single-GPU and `torchrun` unchanged; modes `embed`, `lora`, `full` |
| `selfcheck.py` | the suite below |

## What each check proves

| check | the claim | how it could fail |
|---|---|---|
| `overfit_one_batch` | the loop optimises what it reports | a detached graph, an empty optimizer group, a schedule stuck in warmup, labels not aligned to logits. Run dense and through the ternary STE; both must reach ~0, which is what shows gradient reaching the shadow weights rather than stopping at `round()`. Negative control: the same loop at LR 0 must not move the loss at all |
| `loss_masking` | the mask is where the tokens say it is, padding cannot touch the loss, and the target can | an off-by-one between the prompt length and the supervised span. The mask is derived twice — once from arithmetic in `build_example`, once by locating `<speech_start>`/`<speech_end>` in the finished sequence — and compared. Padding is scrambled in a ragged batch and the loss must be *bit-identical*. Negative control: scrambling a supervised token must change it |
| `gradient_flow` | trainable parameters receive gradients and frozen ones do not, per mode | a parameter marked trainable that the graph never reaches, or a freeze that missed. Measured after one optimizer step, because LoRA's zero-initialised `B` makes `dL/dA` exactly zero at step 0 by construction |
| `qat_fixed_point` | shadow initialisation reproduces the released weights under either rule | the grouping axis, the group width, the missing `clamp(-1, 1)`, a bf16 shadow. Four negative controls must all come out *wrong*; if any reproduces the weights the check fails, because the positive result would then be an algebraic identity rather than a measurement |
| `pack_roundtrip` | quantize → pack → unpack is bit-exact, and the export loads | a weight that silently stored dense (the file still works, three times the size). Runs on synthetic tensors, on the released tensors when cached, and end-to-end through `export_packed` |
| `determinism` | same seed, same losses, exactly | anything that makes the rest of the suite unfalsifiable. Negative control: a different seed must give a different loss |
| `checkpoint_resume` | an interrupted run continues rather than restarts | optimizer moments, scheduler position or RNG streams missing from the save. Eight steps straight through against four + save + fresh model + load + four, compared step for step. The fresh model is built at a *different* seed so a failed load diverges instead of coincidentally agreeing |
| `speaker_conditioning` | the speaker vector is in the training graph, and neither call convention can bypass it | `speaker.py` injects through a prefill-only forward pre-hook. Position 0 is read off the `inputs_embeds` handed to `Qwen3Model` and compared against `spk_proj`'s output. Both silent bypasses are then measured against the real inference hook and must reproduce the *unconditioned* loss bit for bit, and both fixes are asserted on the training model. Verified falsifiable: removing either fix turns the check red |

`--full` swaps the toy model for the released 28-layer checkpoint on the
single-forward checks and skips the multi-step ones, which are about the loop
rather than the checkpoint. `--full-dtype bf16` halves the 7 GB a float32 copy
needs; `gradient_flow` under `--full` covers `embed` only, since `lora` would
re-wrap the released model and `full` holds 1.7B gradients. `qat_fixed_point`
and `pack_roundtrip` already use the released weights whenever they are in the
Hugging Face cache, with no flag.

## The speaker hook

A teacher-forced training forward **does** take the prefill branch: with no KV
cache and no `cache_position`, `past_key_values` is `None` and the hook injects.
`spk_proj` receives gradient and changing the speaker embedding changes the
loss, both asserted.

The hazard is the call convention. `speaker.py`'s hook reads `input_ids` from
keyword arguments only, so two ordinary calls skip injection with no error:

```
model(ids, labels=...)                  # input_ids positional
model(inputs_embeds=emb, labels=...)    # what a trainer naturally does
```

Measured on a toy forward against the real hook: the correctly conditioned loss
is 5.41624 with a `spk_proj` gradient of 75.4, and **both** bypasses give
5.40856 — bit-identical to a forward with no speaker at all — with a gradient of
exactly 0. Because position 0's `[START]` id is a placeholder whose embedding is
*meant* to be overwritten, such a run trains on a different input sequence than
it will generate from.

`loop.attach_speaker_head` closes both: it reads `args[0]` as well, and its
second pre-hook refuses a training forward unless the first one set an
injected flag — the flag, not the presence of `inputs_embeds`, because that
presence is exactly what the second bypass produces. `loop.forward_batch` is the
single call path that stays on the safe side. `selfcheck` reproduces both
original failures against the real inference hook, so the divergence stays
justified rather than becoming folklore.

Sequence *structure* — marker order, prefix identity, role-token layout — is
covered by `indic/bench/teacher_forced_loss.py --self-test` and is not
duplicated here.

## Padding side

Right, and it is not interchangeable: the speaker vector is written over
sequence position 0 unconditionally, so left padding would overwrite a pad token
while `[START]` sat at a per-row offset. Batched *generation* would want the
opposite; the inference path sidesteps it by running one sequence at a time.

## Ternary rules

`absmax` is a bit-exact fixed point on the released weights, so QAT resumes with
zero perturbation at step 0. Its long-horizon risk is that the scale is the
group maximum: one weight that drifts to twice its 127 neighbours pushes all of
them below the `s/2` rounding threshold and the whole group goes to zero, with
nothing to recover it. `absmean` tracks the bulk of the group instead, but is
not a fixed point as released — on ternary input `mean|W| = f · max|W|`, so it
shrinks every surviving weight by that group's nonzero fraction (measured 0.61
to 0.98). `shadow_from_released` cancels that by pre-dividing by `f`.

Which rule produced the checkpoint is **not identifiable from the file**
(`indic/bench/ternary_check.py`), so neither is hardcoded. Log
`collapsed_group_fraction` over a long run and switch if it climbs.

## What is still unproven

The suite proves the plumbing. It cannot prove anything about speech, because
the synthetic examples are noise with the right shape. Still open until there is
data and a GPU:

- **That the sequence format matches what the checkpoint was trained on.** The
  layout comes from `prompt.py`, so it matches *inference*; whether training
  used the same one is a teacher-forced loss measurement on English audio
  (step 1 of `docs/indic/training.md`), not something a synthetic batch can
  answer. In particular `<final_seg>`, the `{emo: 1}` control and the
  reference/no-reference ratio are all unknown.
- **That QAT converges.** Step 0 is a verified fixed point and the STE passes
  gradient; that the body stays trainable through a ternary forward over
  thousands of steps is not shown here, and neither is whether `absmax` collapses
  groups at scale.
- **Loss scale and hyperparameters.** The LRs in `loop.py` are the research
  doc's estimates. Nothing here has calibrated them.
- **The released model under training steps.** `--full --full-dtype bf16` has
  been run on CPU for the two single-forward checks and both pass against the
  real 28-layer checkpoint — `indic/data/train_selfcheck_full.json`. The
  multi-step checks (`overfit`, `determinism`, `resume`) are not practical there
  on a CPU, and `gradient_flow` under `--full` was not run because an AdamW step
  over the 324M-row embedding did not fit alongside the model in this box's
  15 GB. Nothing here has taken an optimizer step on the real weights.
- **Multi-GPU.** The `torchrun` path has been run: 2 ranks on CPU over gloo, in
  `embed` and in `full` (QAT + gradient checkpointing), both producing a loss.
  NCCL, multi-node and anything at the released model's size are untested.
- **`ShardDataset`.** A stub on purpose — writing a reader against an imagined
  schema is how the reader and the shard writer end up disagreeing.
