"""Can the S3 codec carry Indic speech at all?

KittenTTS 2 never emits a waveform. It emits S3 codec tokens, and everything
audible is whatever s3gen can rebuild from them. That codec -- CosyVoice's
S3TokenizerV2, 6561 tokens at 25 Hz -- was trained without a single Indic
language (docs/indic/finetuning.md). If it cannot carry Tamil or Telugu through a
round trip, the ceiling on this whole effort is fixed before the language model
is touched: no amount of LM fine-tuning recovers information the codec discarded,
and the answer is a different codec, not more training.

So: real FLEURS recordings -> S3 tokens -> s3gen -> waveform, scored against the
original. Two independent scores, because each covers the other's blind spot.

  * **CER through an ASR**, which is what intelligibility means, but only as a
    *delta*. A recognizer's own error rate on Telugu is several times its rate on
    Hindi, so a reconstruction CER of 20% says nothing until you know the
    original recording scored 17% through the same model. The floor is measured
    on the same clip, through the same recognizer, and subtracted.
  * **Log-mel distance and duration preservation**, which need no ASR. They are
    the verdict on a machine with no network or no access to the gated Indic
    recognizer, and they are also the only metrics that distinguish "the codec
    dropped this language" from "the recognizer is bad at this language" --
    an ASR cannot tell those apart, which is the entire reason both exist here.

English rides through every step as the control. It is in the codec's training
distribution, so it calibrates what "good" looks like *on this pipeline*, which
is the only thing that makes the Indic numbers readable. The verdict is each
language's excess over English, never an absolute threshold: a Tamil delta-CER
of 8 points is a disaster if English scores 1 and unremarkable if English scores
7, and nothing but the control says which.

    python -m indic.bench.codec_roundtrip --clips 20 --out indic/data/codec_roundtrip.json

Exits non-zero when a language is decisively worse than English, so a Slurm job
can gate the training plan on it.
"""

import argparse
import json
import math
import pathlib
import sys
import tarfile
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.bench.tokenizer_fertility import FLEURS_REPO, parallel_corpus
from indic.langs import ALL, LANGUAGES

ROOT = pathlib.Path(__file__).resolve().parents[2]
# Under .cache/ because .gitignore already covers it and a dev split is 240 MB of
# audio per language -- not something to leave loose in the working tree.
CACHE_DIR = ROOT / ".cache" / "fleurs_audio"
AUDIO_DIR = ROOT / ".cache" / "codec_roundtrip"

# S3 tokenizes at 16 kHz and s3gen writes 24 kHz, so "the same clip" is two
# different arrays and every comparison has to say which rate it is at.
S3_RATE = 16000
OUTPUT_RATE = 24000
TOKEN_RATE = 25

# vocoder.S3Codec.decode appends three silence tokens as streaming lookahead, so
# the reconstruction is always ~120 ms longer than the tokens account for. Trimmed
# before the spectral comparison; left in the written wav, which is what a listener
# would actually hear.
DECODE_PAD_TOKENS = 3

# A language fails when its reconstruction is this much worse than English's. The
# CER tolerance is in absolute character-error points, because that is what the
# metric means. The spectral one is a *ratio*: the MCD here is on s3gen's own
# clamped log-mel scale (see `spectral_distance`), and an absolute dB threshold on
# a scale nobody has calibrated would be a number pulled from the air.
CER_EXCESS_TOLERANCE = 0.05
MCD_RATIO_TOLERANCE = 1.3


# ── FLEURS audio ──────────────────────────────────────────────────────────────
def audio_index(code, split):
    """{sentence_id: wav filename} for one FLEURS split.

    `tokenizer_fertility.load_fleurs` deliberately returns only text, and keeps the
    *first* row for each sentence id because FLEURS records every sentence several
    times with different speakers. A round trip needs a specific recording, so it
    needs the filename column as well -- and it has to keep the same first-row-wins
    rule, or the clip and the transcript it is scored against come from different
    rows and the CER measures the mismatch rather than the codec.
    """
    import csv

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FLEURS_REPO, f"data/{LANGUAGES[code].fleurs}/{split}.tsv",
                           repo_type="dataset")
    index = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2 and row[0] not in index:
                index[row[0]] = row[1]
    return index


def extract_clips(code, split, filenames, cache_dir=CACHE_DIR):
    """Pull just the wanted wavs out of the split's tarball, once.

    FLEURS ships a split's audio as one gzipped tar. The archive is downloaded
    whole -- the Hub caches it, and there is no ranged read into a gzip stream --
    but unpacked lazily in a single sequential pass, because seeking is exactly
    what a gzip stream cannot do: reopening it per clip re-inflates the entire
    240 MB archive each time.
    """
    from huggingface_hub import hf_hub_download

    out_dir = pathlib.Path(cache_dir) / LANGUAGES[code].fleurs / split
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name: out_dir / name for name in filenames
              if not (out_dir / name).is_file()}
    if wanted:
        archive = hf_hub_download(
            FLEURS_REPO, f"data/{LANGUAGES[code].fleurs}/audio/{split}.tar.gz",
            repo_type="dataset")
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                # Read through extractfile and written to a path built here, never
                # tar.extract: the member names come out of a downloaded archive,
                # and constructing the destination ourselves is what stops a
                # crafted name from writing outside the cache directory.
                name = pathlib.PurePosixPath(member.name).name
                if not (member.isfile() and name in wanted):
                    continue
                with tar.extractfile(member) as source:
                    wanted.pop(name).write_bytes(source.read())
                if not wanted:
                    break
        if wanted:
            raise FileNotFoundError(
                f"{split}.tar.gz for {code} does not contain {sorted(wanted)[:3]} -- "
                "the TSV and the audio archive disagree")
    return [out_dir / name for name in filenames]


def fleurs_clips(codes, split, n_clips, cache_dir=CACHE_DIR):
    """{lang: [{id, path, text}]}, the same sentences in every language.

    Parallel on purpose, the same way fertility is measured parallel: if Tamil is
    scored on Tamil news and English on English weather reports, the CER gap
    between them is partly the sentences. Here only the language varies.
    """
    ids, corpora = parallel_corpus(codes, split, limit=n_clips)
    clips = {}
    for code in codes:
        index = audio_index(code, split)
        text_by_id = dict(zip(ids, corpora[code]))
        usable = [i for i in ids if i in index]
        paths = extract_clips(code, split, [index[i] for i in usable], cache_dir)
        clips[code] = [{"id": i, "path": path, "text": text_by_id[i]}
                       for i, path in zip(usable, paths)]
    return clips


# ── signal metrics ────────────────────────────────────────────────────────────
def log_mel(wav, rate=OUTPUT_RATE):
    """Log-mel exactly as s3gen computes it.

    Its own analysis, not a fresh one: a different n_fft or mel basis would put
    our choice of front end into the number alongside the codec's losses.
    """
    import logging

    import torch

    from kittenml.kittentts2.s3gen import mel_spectrogram

    # s3gen warns whenever its input leaves [-1, 1], which is advice for feeding
    # its vocoder and says nothing about using its analysis front end. A decoded
    # clip overshoots routinely, and two lines per clip would bury the table.
    s3gen_logger = logging.getLogger("kittenml.kittentts2.s3gen")
    previous = s3gen_logger.level
    s3gen_logger.setLevel(logging.ERROR)
    try:
        with torch.no_grad():
            return mel_spectrogram(torch.as_tensor(wav).float(), sampling_rate=rate)[0]
    finally:
        s3gen_logger.setLevel(previous)


def spectral_distance(original, reconstruction, max_shift=8):
    """Log-mel L1 and mel cepstral distortion between two 24 kHz waveforms.

    s3gen is not sample-synchronous with its input -- the token grid, the 20 ms
    trim fade and the decoder's own latency can slide the output by a frame or
    two -- and an L1 taken at a constant offset reports that slide, not the codec.
    So the best alignment within +/-`max_shift` frames (160 ms) is searched and
    reported; a shift at the edge of that window is itself a finding.

    The MCD here is computed over s3gen's clamped log-mel rather than over a
    classical spectral envelope, so it is on its own scale and must not be
    compared with MCD figures from the vocoder literature. It is only ever read
    against the English control measured the same way, which is the point.
    """
    import numpy as np
    import torch
    from scipy.fft import dct

    a, b = log_mel(original), log_mel(reconstruction)
    best = None
    for shift in range(-max_shift, max_shift + 1):
        x = a[:, max(0, -shift):]
        y = b[:, max(0, shift):]
        n = min(x.shape[1], y.shape[1])
        if n < 10:
            continue
        l1 = float(torch.abs(x[:, :n] - y[:, :n]).mean())
        if best is None or l1 < best[0]:
            best = (l1, shift, n)
    if best is None:
        return {"log_mel_l1": None, "mcd_db": None, "frame_shift": None,
                "frames_compared": 0}
    l1, shift, n = best

    # MCD over coefficients 1-13: c0 is overall loudness, which s3gen does not
    # preserve and nobody hears as a codec artefact, so including it would score
    # gain rather than spectral shape.
    x = a[:, max(0, -shift):][:, :n].numpy()
    y = b[:, max(0, shift):][:, :n].numpy()
    cx = dct(x, type=2, axis=0, norm="ortho")[1:14]
    cy = dct(y, type=2, axis=0, norm="ortho")[1:14]
    mcd = (10.0 / math.log(10)) * np.sqrt(2.0 * ((cx - cy) ** 2).sum(axis=0))
    return {"log_mel_l1": l1, "mcd_db": float(mcd.mean()), "frame_shift": shift,
            "frames_compared": int(n)}


# ── intelligibility ───────────────────────────────────────────────────────────
def normalize_for_cer(text):
    """NFC, casefolded, punctuation and symbols dropped, whitespace collapsed.

    Applied identically to the reference and to both hypotheses. The absolute CER
    depends on this choice -- a danda scored as a character costs a point on every
    Hindi sentence -- but the delta that the verdict reads does not, because the
    same normalization sits under both halves of the subtraction.
    """
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    kept = [ch for ch in text if unicodedata.category(ch)[0] not in ("P", "S")]
    return " ".join("".join(kept).split())


def character_error_rate(reference, hypothesis):
    """Levenshtein distance over characters, divided by the reference length.

    Written out rather than pulled from jiwer: it is fifteen lines, both scripts
    in `indic/bench` are meant to run on a bare machine, and a scoring dependency
    that `indic/eval` may later standardise differently is not worth adding here.
    """
    ref, hyp = normalize_for_cer(reference), normalize_for_cer(hypothesis)
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def load_recognizers(codes, backend, device):
    """{lang: recognizer}, or None when ASR is unavailable.

    Deliberately non-fatal. `indic/eval/asr.py` is written by a different strand
    of this work and its Indic backend is a gated download, so a machine that has
    neither must still produce the spectral verdict rather than exiting with a
    traceback and no measurement at all.
    """
    if backend == "none":
        return None, "disabled by --asr none"
    try:
        from indic.eval.asr import default_backend, get_asr
    except ImportError as exc:
        return None, f"indic.eval.asr unavailable ({exc})"

    instances, per_language = {}, {}
    for code in codes:
        name = default_backend(code) if backend == "auto" else backend
        if name not in instances:
            instances[name] = get_asr(name, device=device)
        per_language[code] = instances[name]
    return per_language, None


# ── the round trip ────────────────────────────────────────────────────────────
def roundtrip(codec, wav_path):
    """One clip through encode and decode, plus everything the metrics need.

    The decoder is conditioned on the clip being reconstructed, not on some other
    reference. That is the point: it hands s3gen the speaker for free so that what
    is left to measure is only what the token stream itself carries.
    """
    import librosa

    tokens = codec.encode_reference(wav_path)
    original24, _ = librosa.load(str(wav_path), sr=OUTPUT_RATE, mono=True)
    if not tokens:
        return {"tokens": [], "original": original24, "reconstruction": None}
    reconstruction = codec.decode(codec.reference_conditioning(wav_path), tokens)
    return {"tokens": tokens, "original": original24, "reconstruction": reconstruction}


def score_clip(clip, result, recognizer, lang, audio_dir):
    """Every number for one clip: spectral, duration, and CER for both arms."""
    import soundfile as sf

    original, reconstruction = result["original"], result["reconstruction"]
    row = {"id": clip["id"], "n_tokens": len(result["tokens"]),
           "original_s": len(original) / OUTPUT_RATE}
    if reconstruction is None:
        row.update({"reconstruction_s": 0.0, "failed": "codec produced no audio"})
        return row

    row["reconstruction_s"] = len(reconstruction) / OUTPUT_RATE
    # The decoder's three pad tokens are subtracted before the ratio: they are a
    # fixed 120 ms that would otherwise dominate the number on a short clip and
    # report the same constant for every language. What is left is the question
    # worth asking -- whether s3gen holds the time axis.
    row["duration_ratio"] = ((row["reconstruction_s"] - DECODE_PAD_TOKENS / TOKEN_RATE)
                             / max(row["original_s"], 1e-6))
    # The token count is what the LM will have to produce, so a codec that
    # silently stretched time would make every duration target in training wrong.
    row["token_seconds"] = len(result["tokens"]) / TOKEN_RATE

    paths = {}
    if audio_dir is not None:
        out = pathlib.Path(audio_dir) / lang
        out.mkdir(parents=True, exist_ok=True)
        paths = {"original": out / f"{clip['id']}.orig.wav",
                 "reconstruction": out / f"{clip['id']}.recon.wav"}
        sf.write(str(paths["original"]), original, OUTPUT_RATE)
        sf.write(str(paths["reconstruction"]), reconstruction, OUTPUT_RATE)
        row["wav"] = {k: str(v) for k, v in paths.items()}

    trim = min(DECODE_PAD_TOKENS * OUTPUT_RATE // TOKEN_RATE, len(reconstruction) // 2)
    row.update(spectral_distance(original, reconstruction[:len(reconstruction) - trim]))

    if recognizer is not None:
        for arm, audio in (("original", original), ("reconstruction", reconstruction)):
            # The written file when there is one, so the recognizer reads exactly
            # the bytes a human auditing this run would hear; otherwise the array,
            # which `eval.asr.load_audio` takes as already being at its own rate.
            source = str(paths[arm]) if paths else _to_asr_rate(audio)
            try:
                heard = recognizer.transcribe(source, lang)
            except Exception as exc:                      # gated model, missing token
                row["asr_error"] = f"{type(exc).__name__}: {exc}"
                break
            row[f"cer_{arm}"] = character_error_rate(clip["text"], heard)
            row[f"heard_{arm}"] = heard
    # Paired per clip, not as a difference of two language means: a clip whose
    # reconstruction arm failed would otherwise contribute its original CER to one
    # mean and nothing to the other, and the subtraction would silently compare
    # two different sets of sentences.
    if row.get("cer_original") is not None and row.get("cer_reconstruction") is not None:
        row["cer_delta"] = row["cer_reconstruction"] - row["cer_original"]
    return row


def _to_asr_rate(wav):
    """24 kHz waveform -> 16 kHz.

    `eval.asr.load_audio` takes an array as already being at its own rate, by
    design -- it refuses to guess -- so the resample has to happen here.
    """
    import torch
    import torchaudio

    return torchaudio.functional.resample(
        torch.as_tensor(wav).float(), OUTPUT_RATE, S3_RATE).numpy()


# ── report ────────────────────────────────────────────────────────────────────
def summarize(rows):
    """Per-language means over the clips that produced each number."""
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    summary = {key: mean(key) for key in
               ("mcd_db", "log_mel_l1", "duration_ratio", "cer_original",
                "cer_reconstruction", "cer_delta", "frame_shift")}
    summary["n_clips"] = len(rows)
    summary["n_failed"] = sum(1 for r in rows if r.get("failed"))
    return summary


def report(summaries, args):
    """The table, then the verdict. Returns the languages that failed."""
    control = summaries.get("en", {})
    print(f"\n{'':<6}{'clips':>6}{'MCD':>8}{'MCD/en':>8}{'logmel':>8}{'dur':>7}"
          f"{'CERorig':>9}{'CERrec':>9}{'delta':>8}{'d-d(en)':>9}")
    failed = []
    for code in ALL:
        s = summaries.get(code)
        if not s:
            continue
        mcd_ratio = _ratio(s["mcd_db"], control.get("mcd_db"))
        cer_excess = _excess(s["cer_delta"], control.get("cer_delta"))
        print(f"{code:<6}{s['n_clips']:>6}{_fmt(s['mcd_db'], '{:.2f}'):>8}"
              f"{_fmt(mcd_ratio, '{:.2f}x'):>8}"
              f"{_fmt(s['log_mel_l1'], '{:.3f}'):>8}"
              f"{_fmt(s['duration_ratio'], '{:.2f}'):>7}"
              f"{_fmt(s['cer_original'], '{:.1%}'):>9}"
              f"{_fmt(s['cer_reconstruction'], '{:.1%}'):>9}"
              f"{_fmt(s['cer_delta'], '{:+.1%}'):>8}"
              f"{_fmt(cer_excess, '{:+.1%}'):>9}")
        if code == "en":
            continue
        # Both are checked, neither as a fallback for the other: a language can
        # reconstruct with a clean spectrum and still come out unintelligible, and
        # it can score a low CER on an audibly mangled reconstruction because the
        # recognizer guessed the sentence from its own language model.
        if cer_excess is not None and cer_excess > args.cer_excess:
            failed.append((code, f"CER delta is {cer_excess:+.1%} worse than English's"))
        if mcd_ratio is not None and mcd_ratio > args.mcd_ratio:
            failed.append((code, f"MCD is {mcd_ratio:.2f}x English's"))

    print("\n== Verdict ==")
    if control.get("cer_delta") is None:
        print("  No ASR in this run -- the verdict is spectral only. MCD and duration\n"
              "  catch a codec that mangles a language's phonetics; they cannot catch a\n"
              "  reconstruction that is clean but says a different word. Re-run with\n"
              "  --asr once indic/eval/asr.py has its weights to close that gap.")
    if not control:
        print("  INCONCLUSIVE: English was not measured, so nothing calibrates these\n"
              "  numbers. Every threshold here is an excess over the control, and the\n"
              "  raw MCD scale is s3gen's own -- it means nothing on its own.")
        return [("en", "control missing")]
    measured = [c for c in summaries if c != "en" and summaries[c]["n_clips"]]
    print(f"  Control: English reconstructs at MCD {_fmt(control['mcd_db'], '{:.2f}')}"
          + (f", CER delta {control['cer_delta']:+.1%}."
             if control.get("cer_delta") is not None else "."))
    if not measured:
        print("  INCONCLUSIVE: only the control ran. Nothing was asked about Indic.")
        return []
    if not failed:
        print(f"  PASS. {', '.join(measured)} reconstructed within the control's own\n"
              "  margin, so the S3 codec represents them about as well as it does\n"
              "  English. Keep it frozen and train the LM.")
    else:
        print("  FAIL. These languages lose decisively more than English does:")
        for code, why in failed:
            print(f"    {code}: {why}")
        print("  The codec is the ceiling for them. Listen to the written wavs before\n"
              "  acting on this -- then either fine-tune s3gen (docs/indic/training.md\n"
              "  step 6) or change codec. LM training will not fix it.")
    return failed


def _excess(value, control):
    return None if value is None or control is None else value - control


def _ratio(value, control):
    return None if not value or not control else value / control


def _fmt(value, spec):
    return "-" if value is None else spec.format(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default=",".join(ALL),
                    help="comma-separated; en is the control and should stay in")
    ap.add_argument("--clips", type=int, default=20, help="clips per language")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--asr", default="auto",
                    choices=["auto", "none", "whisper", "indicconformer"],
                    help="auto picks per language (see indic/eval/asr.py)")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=CACHE_DIR)
    ap.add_argument("--audio-dir", type=pathlib.Path, default=AUDIO_DIR,
                    help="where the original/reconstruction pairs are written")
    ap.add_argument("--cer-excess", type=float, default=CER_EXCESS_TOLERANCE,
                    help="fail a language whose CER delta exceeds English's by this")
    ap.add_argument("--mcd-ratio", type=float, default=MCD_RATIO_TOLERANCE,
                    help="fail a language whose MCD is more than this times English's")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    codes = [c.strip() for c in args.langs.split(",") if c.strip()]
    unknown = [c for c in codes if c not in LANGUAGES]
    if unknown:
        raise SystemExit(f"unknown language codes: {unknown}")

    from kittenml.kittentts2.vocoder import S3Codec

    print(f"fetching {args.clips} parallel FLEURS/{args.split} clips per language ...",
          flush=True)
    clips = fleurs_clips(codes, args.split, args.clips, args.cache_dir)
    recognizers, asr_note = load_recognizers(codes, args.asr, args.device)
    if asr_note:
        print(f"ASR: {asr_note} -- reporting spectral metrics only")

    print(f"loading s3gen on {args.device} ...", flush=True)
    codec = S3Codec(device=args.device)

    results, summaries = {}, {}
    for code in codes:
        rows = []
        for clip in clips[code]:
            rows.append(score_clip(clip, roundtrip(codec, clip["path"]),
                                   recognizers.get(code) if recognizers else None,
                                   code, args.audio_dir))
            print(f"  {code} {len(rows)}/{len(clips[code])}", end="\r", flush=True)
        print(f"  {code}: {len(rows)} clips scored" + " " * 20)
        results[code] = rows
        summaries[code] = summarize(rows)

    failed = report(summaries, args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            # Which recognizer produced each language's CER, in its own words. A
            # CER is only comparable with another one taken through the same
            # backend and the same decoding, so the provenance travels with it.
            {"config": {"split": args.split, "clips": args.clips, "asr": args.asr,
                        "asr_note": asr_note, "device": args.device,
                        "asr_backends": {c: r.describe()
                                         for c, r in (recognizers or {}).items()},
                        "cer_excess": args.cer_excess, "mcd_ratio": args.mcd_ratio},
             "summary": summaries, "clips": results,
             "failed": [{"lang": c, "reason": w} for c, w in failed]},
            indent=2, ensure_ascii=False, default=str))
        print(f"\nwrote {args.out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
