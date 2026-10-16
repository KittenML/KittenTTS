# `indic/` — the Indic adaptation harness

Everything needed to decide whether KittenTTS 2 can be adapted to Indian
languages, and to prove the training pipeline is correct before any allocation is
spent on it. Nothing here modifies the shipped library except the danda fixes in
`kittenml/kittentts2/text.py`.

## The one command

```bash
.venv/bin/python -m indic.preflight
```

Runs every check in dependency order, caches results, and writes
`indic/data/PREFLIGHT_REPORT.md`. A check that FAILS is a finding about the
project, not a crash — `codec_roundtrip` failing on Telugu would mean the codec
cannot represent Telugu, which is exactly what we want to learn before training
rather than after.

To keep it running unattended until it settles:

```bash
.venv/bin/python -m indic.overnight --deadline-hours 8
```

It re-sweeps as missing scripts appear, never re-runs a green check, defers GPU
checks while another process holds the card, and retries a failure once before
accepting it as the answer.

## What each piece answers

| Module | Question |
|---|---|
| `bench/tokenizer_fertility.py` | How much more does Indic text cost under Qwen3 than Sarvam-1? |
| `bench/ternary_check.py` | Which quantizer wrote the weights, and can QAT resume without damaging them? |
| `bench/codec_roundtrip.py` | Can the S3 codec represent Tamil and Telugu at all? **Go/no-go.** |
| `bench/teacher_forced_loss.py` | Does our training sequence format match what the checkpoint was trained on? **Go/no-go.** |
| `bench/fit_config.py` | What is the largest training configuration that fits a given GPU? |
| `vocab/build_sarvam_vocab.py` | Rebuild the checkpoint around a different text tokenizer. |
| `vocab/verify_surgery.py` | Is the rebuilt checkpoint actually correct, not merely written? |
| `train/selfcheck.py` | Does the training loop provably work, on synthetic data? |
| `eval/` | CER, speaker similarity, test sets, and the scoring driver. |
| `hpc/` | Capability detection, offline cache staging, Slurm templates. |
| `text/indic_text.py` | Danda-aware splitting, Indic digits, NFC. |

## Two things worth knowing before reading the code

**The vocabulary is one shared table.** Text ids and audio codec ids index the
same `embed_tokens`, which is tied to `lm_head`. Swapping the text tokenizer is
therefore surgery on the matrix that both reads text and writes audio — see the
diagram in `vocab/build_sarvam_vocab.py`. Getting the audio block's new base
address wrong by one row produces a model that loads, runs, and emits noise.
`vocab/verify_surgery.py` exists specifically to catch that.

**A check that cannot fail is worse than no check.** An early version of
`ternary_check.py` asked whether re-quantization was a fixed point — which is an
algebraic identity on already-ternary weights, so it "confirmed" a conclusion that
the sparsity measurement then contradicted. When adding a check, ask what input
would make it fail.

## Rebuilding the swapped checkpoints

```bash
# the main arm: new rows = mean of the Qwen rows for the same string, rescaled
python -m indic.vocab.build_sarvam_vocab --out indic/data/kitten-tts-2-sarvam-rescaled \
    --init mean_rescaled
# ablations
python -m indic.vocab.build_sarvam_vocab --out indic/data/kitten-tts-2-sarvam --init mean
python -m indic.vocab.build_sarvam_vocab --out indic/data/kitten-tts-2-sarvam-random --init random
python -m indic.vocab.verify_surgery --checkpoint indic/data/kitten-tts-2-sarvam-rescaled
```

Each is ~3 GB and gitignored: they are reproducible from the source repo plus the
command, so they are artifacts, not source. The `vocab_surgery.json` written
beside each one records which init produced that table and is worth keeping.

## Licensing, before anything ships

The `sarvamai/sarvam-1` tokenizer is under Sarvam's **non-commercial** license.
That is fine for the experiment and not fine for a product. If the Sarvam arm
wins, retrain a SentencePiece tokenizer of the same size on our own Indic and
English text — every script here takes `--tokenizer`, so the pipeline is
unchanged. The KittenTTS 2 weights are under the Stellon Labs Community License,
which permits fine-tunes and derivative releases but requires attribution, a
NOTICE recording what was changed, and terminates above $1M revenue or funding.
