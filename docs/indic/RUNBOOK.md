# Runbook: first day on the cluster

Ordered so that the cheapest things that could invalidate the plan run first.
Every step is independent of the training loop, so none of it is blocked on code
that does not exist yet.

Target assumed: **Slurm, Ampere or newer (A100 / H100 / L40S).** Compute nodes are
assumed to have **no outbound network** — if they do, the staging step is still
worth running to avoid rate-limiting an array job against the Hub.

---

## 0. Before anything — on the LOGIN node

```bash
git clone <this repo> && cd kindic
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r indic/requirements-hpc.txt
```

Pick the torch build that matches the cluster's CUDA, **not** the one pinned on
the dev laptop (that wheel is `cu128`, built for a Blackwell consumer card, and is
wrong for an A100).

```bash
.venv/bin/python -m indic.hpc.capability          # what am I actually on?
.venv/bin/python -m indic.hpc.stage_cache --all   # pull everything, login node only
```

### Gated repositories — do this first, it will otherwise stall day 1

`HF_TOKEN` is **not optional**, and the terms must be accepted once per repo with
that same account. Approval is automatic, but a bare download returns 401 until
it is done, and the failure surfaces only at download time — on a compute node
with no network, hours in.

| Repo | Kind | Needed for |
|---|---|---|
| `ai4bharat/indic-conformer-600m-multilingual` | model | **every Indic CER number** |
| `ai4bharat/IndicF5` | model | the main cloning baseline |
| `ai4bharat/indicvoices_r` | dataset | speaker variety for cloning |
| `ai4bharat/Rasa` | dataset | expressive / emotion data |

Ungated and needing nothing: `google/fleurs`, `openai/whisper-large-v3`,
`microsoft/wavlm-base-plus-sv`, `KittenML/kitten-tts-2`, `sarvamai/sarvam-1`.

Note that **FLORES and FLORES+ are both gated**, which is why the harness measures
on FLEURS instead — same sentences, open repo, and it ships the audio too.

Then export on every compute job:

```bash
export HF_HOME=/shared/path/hf          # must be on a filesystem the nodes see
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=$SLURM_TMPDIR             # node-local scratch, not the shared FS
```

---

### One environment trap

`torchaudio` 2.11 routes `save`/`load` through `torchcodec`, which is not a
dependency of this project. All audio I/O in `indic/` goes through `soundfile`
plus `torchaudio.functional.resample` (which still works, and matches the
resampler `speaker.py` uses for speaker embeddings — a different resampler
shifts the embedding). Do not "fix" a torchaudio I/O error by adding torchcodec
without checking it against that.

---

## 1. The two go/no-go gates — run these before anything else

Both are cheap, both can kill the plan, and neither needs training code.

```bash
# Can the S3 codec represent Tamil and Telugu at all?
.venv/bin/python -m indic.bench.codec_roundtrip --clips 32 --device cuda \
    --out indic/data/codec_roundtrip.json

# Does our training sequence format match what the checkpoint was trained on?
.venv/bin/python -m indic.bench.teacher_forced_loss --device cuda \
    --out indic/data/teacher_forced.json
```

**How to read the codec gate.** The verdict is each language's *excess over
English*, never an absolute number — a recognizer's own error rate on Telugu is
several times its rate on Hindi, so a 20% reconstruction CER means nothing until
you know the original scored 17% through the same model. Log-mel distance is the
cross-check that distinguishes "the codec dropped this language" from "the
recognizer is bad at this language"; an ASR alone cannot tell those apart.

- *If Tamil/Telugu excess is close to English's* → codec is fine, proceed.
- *If it is several times English's* → **stop.** The ceiling is fixed before the
  LM is touched. The answer is a different codec, not more training.

**How to read the format gate.** A low loss alone proves nothing — it must beat
the negative controls (shuffled target audio, audio from a different utterance,
reference dropped, speaker embedding replaced with noise, role tokens swapped)
*decisively*. If it does not, our reconstruction of the training format is wrong,
and training it would optimise a mis-specified objective.

---

## 2. Prove the training loop — still no data needed

```bash
.venv/bin/python -m indic.train.selfcheck --out indic/data/train_selfcheck.json
.venv/bin/python -m indic.train.selfcheck --full --device cuda   # real 28-layer weights
```

The check that matters most is **overfit-one-batch**: if the loop cannot drive a
single synthetic batch to near-zero loss, it is broken, and every later loss curve
would be fiction. Second most: the **loss mask** test — randomising tokens at
non-target positions must leave the loss *exactly* unchanged.

---

## 3. Rebuild the vocabulary-swapped checkpoints

```bash
.venv/bin/python -m indic.vocab.build_sarvam_vocab \
    --out $SCRATCH/kitten-tts-2-sarvam-rescaled --init mean_rescaled
.venv/bin/python -m indic.vocab.verify_surgery \
    --checkpoint $SCRATCH/kitten-tts-2-sarvam-rescaled
```

CPU, minutes, ~3 GB out. Build the `mean` and `random` arms too — they are the
ablations that tell you whether the initialisation mattered. **Do not skip
`verify_surgery`:** an audio-block misalignment of one row produces a checkpoint
that loads, runs, and emits noise.

---

## 4. Size the run for the cluster's actual GPU

```bash
.venv/bin/python -m indic.bench.fit_config --vocab 74752 --reserve 4 \
    --out indic/data/fit_a100.json
```

On an 80 GB A100 this will recommend something far larger than the laptop's
batch 8 @ seq 512. The fitted relation is linear in **tokens per step** and
indifferent to how you split that between batch and sequence length:

```
reserved_GB = 3.897 + 0.001357 x tokens_per_step   (measured on 11.5 GB; re-fit per card)
```

---

## 5. Baseline sweep — the C0 control arm

Embarrassingly parallel, ideal for a Slurm array. Score stock KittenTTS 2 and
IndicF5 on FLEURS hi/bn/ta/te/mr, and **always report the ground-truth
recording's CER through the same recognizer** as the floor. A CER without its
floor is not comparable across languages.

```bash
sbatch indic/hpc/slurm/eval_array.sbatch
```

---

## 6. Start the data pipeline — the real long pole

Begin this on day 1 and let it run underneath everything else. Order:
resample to 24 kHz → NFC + Indic number normalization → ASR-filter clips whose
CER against the transcript exceeds ~10–15% → precompute S3 tokens (25 Hz),
speaker embeddings, and text ids → shard, holding out speakers for evaluation.

Starter mix (~800–900 h): SYSPIN for hi/bn/mr/te, Rasa and IndicTTS for all five,
IndicVoices-R filtered by quality for speaker variety, plus **10–20% real English
recordings** (LibriTTS-R or similar) to prevent forgetting. English is
re-tokenized under Sarvam too, so it has to be relearned — and under §4b of the
license, KittenTTS output must never enter the mix.

---

## 7. Only then, train

Phase A (embedding warm-up, body frozen) → Phase B (joint, QAT or LoRA). Resume
QAT with the **absmax** forward for zero step-0 damage, or absmean with the `W/f`
per-group shadow rescale. Do **not** assume absmean was the original rule; the
checkpoint's 15% sparsity matches neither textbook rule, and the question is open.

---

## The one command that checks all of it

```bash
.venv/bin/python -m indic.preflight              # run everything, in order
.venv/bin/python -m indic.overnight --deadline-hours 8   # keep sweeping unattended
```

`indic/data/PREFLIGHT_REPORT.md` is the output. A **FAIL is a finding**, not a
crash — that is the entire point of running it before spending an allocation.
