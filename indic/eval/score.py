"""Score a finished synthesis run: ASR, CER/WER, the ASR floor, speaker similarity.

Reads what `run_eval.py` wrote and produces one row per utterance — a table, not a
verdict. Kept separate from synthesis because the two need different machines and
happen at different times, and because re-scoring an old run with a better
recognizer must not mean re-synthesizing it.

    python -m indic.eval.score --run runs/kitten2-A-hi
    python -m indic.eval.score --run runs/kitten2-A-hi --speaker-sim --format parquet

**The floor.** Every row that has a ground-truth recording carries `floor_cer`: the
human reading of the same sentence, through the same recognizer, with the same
normalization. `cer_above_floor` is the difference, and it is the only one of the
three that can be compared across languages. The recognizer is worse at Telugu than
at Hindi before any TTS is involved, so a raw Telugu CER two points higher than
Hindi's may be a better model, not a worse one. Floor transcripts are cached: they
depend on the test set and the recognizer, never on the model under test, so five
arms share one pass over the ground truth.

Set C has no floor. It is generated text, nobody recorded it, and its references
hold digits where the recognizer writes words — so its CER level is inflated and
means nothing on its own. What means something there is the gap between a pair's
ASCII-digit and native-digit halves, which the summary prints separately.

Shardable the same way as synthesis, writing `scores.<i>.csv` per shard:

    python -m indic.eval.score --run runs/kitten2-A-hi --shard $SLURM_ARRAY_TASK_ID \
        --num-shards 4
    python -m indic.eval.score --run runs/kitten2-A-hi --summary-only
"""

import argparse
import csv
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.eval import asr as asr_backends
from indic.eval import metrics
from indic.eval.testsets import cache_root

# Fixed and explicit, so every shard's CSV has the same shape and `--resume` can
# append to a file written by an earlier invocation.
COLUMNS = [
    "run_id", "model", "model_revision", "git_commit", "git_dirty", "text_fixes",
    "testset", "lang", "split", "seed", "record_seed", "id", "category", "variant",
    "pair_id",
    "text", "spoken_text", "ref_text", "hypothesis", "cer", "wer",
    "ref_chars", "ref_words", "floor_hypothesis", "floor_cer", "floor_wer",
    "cer_above_floor", "speaker_similarity", "audio_seconds", "synth_seconds",
    "error", "wav", "truth_wav", "ref_wav",
    "asr_backend", "asr_model", "asr_decoding", "speaker_embedder",
]


def load_run(run_dir):
    """Every meta row in a run directory, with its shard's provenance attached.

    Rows are deduplicated by record id: re-running a shard rewrites its meta file,
    but a run re-sharded from 4 to 8 leaves both generations behind, and scoring a
    record twice would weight it twice in the corpus rate.
    """
    run_dir = pathlib.Path(run_dir)
    provenance = {}
    for path in sorted(run_dir.glob("run.*.json")):
        run = json.loads(path.read_text())
        provenance[run["run_id"]] = run

    rows, seen = [], set()
    for path in sorted(run_dir.glob("meta.*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rows.append({**provenance.get(row.get("run_id"), {}), **row})
    if not rows:
        raise FileNotFoundError(f"no meta.*.jsonl rows in {run_dir}")
    return rows, provenance


class FloorCache:
    """Ground-truth transcripts, keyed by recording and recognizer.

    Worth caching because the floor is a property of the test set and the ASR, not
    of the model under test — without this, scoring five arms transcribes the same
    human recordings five times. Each shard owns its own file and reads everyone
    else's, which avoids two processes writing one file.
    """

    def __init__(self, backend, lang, testset, shard=0, cache_dir=None):
        self.directory = (cache_root(cache_dir) / "floor" /
                          f"{backend['backend']}-{backend['asr_decoding']}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{lang}-{testset}.{shard}.json"
        self.entries = {}
        for path in self.directory.glob(f"{lang}-{testset}.*.json"):
            try:
                self.entries.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        self.own = dict(self.entries)

    def get(self, wav_path, transcribe):
        if wav_path not in self.entries:
            self.entries[wav_path] = transcribe(wav_path)
            self.own[wav_path] = self.entries[wav_path]
        return self.entries[wav_path]

    def save(self):
        self.path.write_text(json.dumps(self.own, ensure_ascii=False, indent=1))


def score_row(row, recognizer, floor, embedder=None):
    """One utterance: transcribe, measure, and attach the floor it is read against."""
    lang = row["lang"]
    wav = row.get("wav")
    # A record that failed to synthesize, or whose wav never landed, is a total
    # miss. Dropping it instead would quietly reward a model for the sentences it
    # could not say.
    hypothesis = ("" if row.get("error") or not (wav and pathlib.Path(wav).exists())
                  else recognizer.transcribe(wav, lang))

    reference = row["ref_text"]
    char_errors, ref_chars = metrics.error_counts(reference, hypothesis, "char")
    word_errors, ref_words = metrics.error_counts(reference, hypothesis, "word")

    scored = {
        **{column: row.get(column) for column in COLUMNS},
        "hypothesis": hypothesis,
        "cer": metrics.rate(char_errors, ref_chars, hypothesis),
        "wer": metrics.rate(word_errors, ref_words, hypothesis),
        "ref_chars": ref_chars, "ref_words": ref_words,
        "floor_hypothesis": None, "floor_cer": None, "floor_wer": None,
        "cer_above_floor": None, "speaker_similarity": None,
        **recognizer.describe(),
    }

    truth = row.get("truth_wav")
    if truth and pathlib.Path(truth).exists():
        floor_hypothesis = floor.get(truth, lambda p: recognizer.transcribe(p, lang))
        scored["floor_hypothesis"] = floor_hypothesis
        scored["floor_cer"] = metrics.cer(reference, floor_hypothesis)
        scored["floor_wer"] = metrics.wer(reference, floor_hypothesis)
        scored["cer_above_floor"] = scored["cer"] - scored["floor_cer"]

    # Gated on there being audio, not on the ASR having heard words in it: a clip
    # the recognizer could make nothing of still has a voice, and whether that voice
    # is the reference speaker's is exactly what one wants to know about it.
    if embedder and row.get("ref_wav") and wav and pathlib.Path(wav).exists():
        scored["speaker_similarity"] = metrics.speaker_similarity(
            wav, row["ref_wav"], embedder)
        scored.update(embedder.describe())
    return scored


def write_rows(path, rows, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def write_parquet(path, rows):
    import pandas

    pandas.DataFrame(rows, columns=COLUMNS).to_parquet(path, index=False)


def summarize(rows):
    """Corpus rates, always with the floor they stand on."""
    if not rows:
        print("nothing scored")
        return

    def corpus(subset, key="cer"):
        pairs = [(r["ref_text"], r["hypothesis" if key == "cer" else "floor_hypothesis"])
                 for r in subset
                 if key == "cer" or r.get("floor_hypothesis") is not None]
        return metrics.corpus_rate(pairs) if pairs else None

    testsets_seen = sorted({r["testset"] for r in rows})
    print(f"\n{len(rows)} utterances   model {rows[0]['model']}   "
          f"asr {rows[0]['asr_model']} ({rows[0]['asr_decoding']})   "
          f"text_fixes {rows[0]['text_fixes']}")

    for testset in testsets_seen:
        subset = [r for r in rows if r["testset"] == testset]
        with_floor = [r for r in subset if r.get("floor_hypothesis") is not None]
        cer = corpus(subset)
        print(f"\n== set {testset} ==")
        if with_floor:
            # Both rates over the *same* utterances. Subtracting a floor measured on
            # three recordings from a CER measured on seven is not a difference of
            # anything, and a mixed set is exactly where that would go unnoticed.
            floor_cer = corpus(with_floor, "floor")
            paired_cer = corpus(with_floor)
            print(f"  corpus CER {paired_cer:.4f}   ASR floor {floor_cer:.4f}   "
                  f"above floor {paired_cer - floor_cer:+.4f}   "
                  f"(over the {len(with_floor)} utterance(s) with a ground-truth "
                  "recording)")
            if len(with_floor) != len(subset):
                print(f"  corpus CER {cer:.4f} over all {len(subset)}; the other "
                      f"{len(subset) - len(with_floor)} have no floor, so that number "
                      "is not comparable across languages")
        else:
            # Never a CER on its own: without the floor the reader cannot tell a bad
            # model from a hard language, and set C's references make the level
            # meaningless anyway.
            print(f"  corpus CER {cer:.4f}   ASR floor n/a — no ground-truth "
                  "recordings in this set, so this level is not comparable with set A "
                  "or across languages")

        by_category = sorted({r["category"] for r in subset})
        if len(by_category) > 1:
            print(f"  {'category':<18}{'n':>5}{'CER':>9}{'floor':>9}")
            for category in by_category:
                items = [r for r in subset if r["category"] == category]
                floor_items = [r for r in items if r.get("floor_hypothesis") is not None]
                # Only when the whole category has a floor, for the same reason.
                floor_cer = (corpus(floor_items, "floor")
                             if len(floor_items) == len(items) and floor_items else None)
                print(f"  {category:<18}{len(items):>5}{corpus(items):>9.4f}"
                      + (f"{floor_cer:>9.4f}" if floor_cer is not None else f"{'-':>9}"))

        _summarize_digit_pairs(subset)

    missing = [r for r in rows if r.get("error")]
    if missing:
        print(f"\n{len(missing)} record(s) had no audio and were counted as total "
              f"misses, e.g. {missing[0]['id']}: {missing[0]['error']}")


def _summarize_digit_pairs(rows):
    """ASCII vs native digits, which is the only reading set C's numbers support."""
    ascii_rows = {r["pair_id"]: r for r in rows if r.get("variant") == "ascii_digits"}
    native_rows = {r["pair_id"]: r for r in rows if r.get("variant") == "native_digits"}
    shared = sorted(set(ascii_rows) & set(native_rows))
    if not shared:
        return
    ascii_cer = metrics.corpus_rate(
        [(ascii_rows[p]["ref_text"], ascii_rows[p]["hypothesis"]) for p in shared])
    native_cer = metrics.corpus_rate(
        [(native_rows[p]["ref_text"], native_rows[p]["hypothesis"]) for p in shared])
    print(f"  digit form      {len(shared)} pairs   ASCII {ascii_cer:.4f}   "
          f"native {native_cer:.4f}   gap {native_cer - ascii_cer:+.4f}")
    print("  (levels inflated by digit-vs-word references; only the gap is a quantity)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=pathlib.Path, required=True,
                    help="an output directory written by run_eval.py")
    ap.add_argument("--asr", default=None,
                    help="indicconformer / whisper; default depends on the language")
    ap.add_argument("--decoding", default="rnnt", choices=("rnnt", "ctc"),
                    help="IndicConformer decoder; the two differ by points of CER")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: <run>/scores.<shard>.csv")
    ap.add_argument("--format", default="csv", choices=("csv", "parquet"))
    ap.add_argument("--speaker-sim", action="store_true")
    ap.add_argument("--speaker-embedder", default="wavlm", choices=("wavlm", "kitten"))
    ap.add_argument("--kitten-speaker-weights", default=None,
                    help="speaker/model.safetensors, for --speaker-embedder kitten")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in the output file")
    ap.add_argument("--summary-only", action="store_true",
                    help="re-print the summary from every scores.*.csv in the run")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    if args.summary_only:
        rows = []
        for path in sorted(glob.glob(str(args.run / "scores.*.csv"))):
            with open(path, newline="", encoding="utf-8") as handle:
                rows.extend(_retype(row) for row in csv.DictReader(handle))
        summarize(rows)
        return

    rows, _ = load_run(args.run)
    rows = rows[args.shard::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]

    if args.resume and args.format != "csv":
        ap.error("--resume reads the existing rows back, which only the CSV writer does")
    out = args.out or args.run / f"scores.{args.shard}.{args.format}"
    done = set()
    if args.resume and out.exists() and args.format == "csv":
        with open(out, newline="", encoding="utf-8") as handle:
            done = {row["id"] for row in csv.DictReader(handle)}
        rows = [row for row in rows if row["id"] not in done]

    langs = {row["lang"] for row in rows}
    if len(langs) > 1:
        raise ValueError(f"run mixes languages {sorted(langs)}; score one at a time "
                         "so the recognizer and its floor stay fixed")
    lang = next(iter(langs)) if langs else None
    if lang is None:
        print("nothing left to score")
        return

    backend = args.asr or asr_backends.default_backend(lang)
    kwargs = {"device": args.device, "hf_token": args.hf_token,
              "cache_dir": args.cache_dir}
    if backend == "indicconformer":
        kwargs["decoding"] = args.decoding
    recognizer = asr_backends.get_asr(backend, **kwargs)

    embedder = None
    if args.speaker_sim:
        if args.speaker_embedder == "kitten":
            if not args.kitten_speaker_weights:
                ap.error("--speaker-embedder kitten needs --kitten-speaker-weights")
            embedder = metrics.KittenSpeakerEmbedder(args.kitten_speaker_weights,
                                                     args.device)
        else:
            embedder = metrics.WavLMSpeakerEmbedder(args.device,
                                                    cache_dir=args.cache_dir)

    testset = rows[0]["testset"]
    floor = FloorCache(recognizer.describe(), lang, testset, args.shard, args.cache_dir)
    print(f"scoring {len(rows)} utterances  lang {lang}  asr {backend}  -> {out}")

    scored = []
    for index, row in enumerate(rows, start=1):
        scored.append(score_row(row, recognizer, floor, embedder))
        if index % 25 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    floor.save()

    if args.format == "parquet":
        write_parquet(out, scored)
    else:
        write_rows(out, scored, append=bool(done))
    print(f"wrote {out}")
    summarize(scored)


def _retype(row):
    """CSV reads everything back as a string; the summary needs the numbers."""
    for column in ("cer", "wer", "floor_cer", "floor_wer", "cer_above_floor",
                   "speaker_similarity", "audio_seconds", "synth_seconds"):
        row[column] = float(row[column]) if row.get(column) not in (None, "") else None
    for column in ("ref_chars", "ref_words"):
        row[column] = int(row[column]) if row.get(column) not in (None, "") else None
    for column in ("variant", "pair_id", "error", "floor_hypothesis"):
        row[column] = row.get(column) or None
    return row


if __name__ == "__main__":
    main()
