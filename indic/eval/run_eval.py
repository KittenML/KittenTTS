"""Synthesize a test set with one model, and write down enough to trust the result.

This is the generation half of the eval; `score.py` is the other half. They are
separate because they run at different times on different machines — synthesis
wants the GPU, scoring wants the ASR models, and the comparison that matters is
between arms recorded weeks apart. Which is also why this writes so much metadata:
a `wav/` directory with no record of which checkpoint, which text fixes, which
seed and which library versions produced it is not a result, it is a sound effect.

    python -m indic.eval.run_eval --model KittenML/kitten-tts-2 --testset A --lang hi \
        --out runs/kitten2-A-hi
    python -m indic.eval.run_eval --model ai4bharat/IndicF5 --testset C --lang ta \
        --reference-from fleurs --out runs/indicf5-C-ta

Resumable: a record whose wav already exists is skipped, and wavs are renamed into
place so a killed job never leaves one that looks finished. Shardable for a Slurm
array:

    python -m indic.eval.run_eval ... --shard $SLURM_ARRAY_TASK_ID --num-shards 8

Each shard writes its own `run.<i>.json` and `meta.<i>.jsonl` into the same output
directory; nothing is appended to by two processes at once, and `score.py` reads
whatever shards exist.

`--text-fixes indic` applies `text/indic_text.py` to the text before it reaches the
model, which is the A/B that module was written for. It covers NFC, Indic digits
and the doubled-terminator bug; it does *not* fix the sentence splitter, which
lives inside `text.py`'s chunker and cannot be patched from out here. So a long-input
regression is still expected under `--text-fixes indic`, and the flag is recorded in
`run.json` because two runs that differ on it are not comparable.
"""

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import time
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.eval import testsets
from indic.langs import LANGUAGES
from indic.text.indic_text import normalize_indic

# Recorded in every run.json. Anything whose version can move a waveform belongs
# here; a missing package is reported as such rather than omitted, because "torch
# was absent" and "nobody wrote it down" are different stories later.
TRACKED_PACKAGES = ("torch", "torchaudio", "transformers", "numpy", "soundfile",
                    "huggingface_hub", "kittenml", "vllm")


def git_state(root):
    """The commit this harness ran at, and whether the tree was dirty.

    A dirty tree is not an error — it is the normal state of a working day — but a
    number taken from one cannot be reproduced from the commit alone, and that has
    to be visible in the output rather than remembered.
    """
    def run(*args):
        try:
            return subprocess.run(args, cwd=str(root), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    return {"git_commit": run("git", "rev-parse", "HEAD"),
            "git_dirty": bool(run("git", "status", "--porcelain"))}


def library_versions():
    import importlib.metadata as metadata

    versions = {"python": sys.version.split()[0]}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_revision(spec, hf_token=None):
    """The Hub commit a model id resolves to, when it is a Hub id at all.

    A tag like `main` moves. Recording the sha is what lets a result from six weeks
    ago be re-synthesized from the same weights rather than from whatever is there
    now.
    """
    if pathlib.Path(spec).exists():
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(spec, token=hf_token).sha
    except Exception:
        return None


def seed_for(record_id, base_seed):
    """A per-record seed that does not depend on the order records were generated in.

    CRC32 of the id rather than `hash()`, which is salted per process: with `hash()`
    a resumed shard would re-synthesize an interrupted record with a different seed
    than a fresh run would, and the two runs' audio would differ for no reason
    anyone could reconstruct.
    """
    return (base_seed + zlib.crc32(record_id.encode("utf-8"))) % (2 ** 31 - 1)


def seed_everything(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


class KittenSynthesizer:
    """A KittenTTS repository or local checkpoint directory."""

    def __init__(self, spec, device=None, voice=None, preset=None, weights=None,
                 cache_dir=None, hf_token=None):
        from kittenml import KittenTTS

        self.spec = spec
        self.voice = voice
        self.preset = preset
        self.model = KittenTTS(spec, device=device, cache_dir=cache_dir,
                               weights=weights, hf_token=hf_token)

    @property
    def sample_rate(self):
        return self.model.sample_rate

    def synthesize(self, record, seed):
        kwargs = {"advanced": {"seed": seed}}
        if self.preset:
            kwargs["preset"] = self.preset
        if record.ref_wav:
            # The transcript is passed, never left to be discovered: the cloning
            # path's own transcriber asks Whisper for English regardless of the
            # clip's language (speaker.py), which would put an English prompt in
            # front of a Hindi reference.
            kwargs["reference"] = record.ref_wav
            kwargs["reference_text"] = record.ref_wav_text
        elif self.voice:
            kwargs["voice"] = self.voice
        return self.model.generate(record.text, **kwargs)

    def describe(self):
        return {"synthesizer": "kitten", "voice": self.voice, "preset": self.preset}


class IndicF5Synthesizer:
    """ai4bharat/IndicF5, the Indic TTS baseline to beat.

    Clone-only: it has no built-in voices, so every record needs a reference clip
    and its transcript. On a set that carries none, pass `--reference-from fleurs`.
    Gated like the rest of AI4Bharat's repositories — accept the terms and export
    HF_TOKEN.
    """

    SAMPLE_RATE = 24000

    def __init__(self, spec, device=None, cache_dir=None, hf_token=None):
        from transformers import AutoModel

        self.spec = spec
        model = AutoModel.from_pretrained(
            spec, trust_remote_code=True, token=hf_token,
            **({"cache_dir": cache_dir} if cache_dir else {}))
        self.model = model.to(device) if device else model

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    def synthesize(self, record, seed):
        import numpy as np

        if not record.ref_wav or not record.ref_wav_text:
            raise ValueError(
                f"{record.id} has no reference clip, and IndicF5 can only speak by "
                "cloning — build the set with --reference-from fleurs")
        del seed                            # no seed argument; seeded globally
        audio = self.model(record.text, ref_audio_path=record.ref_wav,
                           ref_text=record.ref_wav_text)
        audio = np.asarray(audio)
        # The repo's own example checks this: the model returns int16 on some paths
        # and float on others, and writing int16 as float is a wav of silence.
        return (audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16
                else audio.astype(np.float32))

    def describe(self):
        return {"synthesizer": "indicf5"}


def build_synthesizer(kind, spec, args, hf_token=None):
    if kind == "auto":
        kind = "indicf5" if "indicf5" in spec.lower() else "kitten"
    if kind == "indicf5":
        return IndicF5Synthesizer(spec, args.device, args.cache_dir, hf_token)
    return KittenSynthesizer(spec, args.device, args.voice, args.preset,
                             args.weights, args.cache_dir, hf_token)


def prepare_text(text, text_fixes):
    return normalize_indic(text) if text_fixes == "indic" else text


def write_wav(path, audio, sample_rate):
    """Write through a temporary name, so an existing wav is always a complete one.

    That is what makes `--resume` safe: a wav that exists has been fully written,
    and a job killed mid-write leaves a `.partial` nothing looks at. The format is
    named explicitly because the temporary has no `.wav` extension to infer it
    from; the subtype is libsndfile's default, matching `generate_to_file`.
    """
    import soundfile as sf

    partial = path.with_name(path.name + ".partial")
    sf.write(str(partial), audio, sample_rate, format="WAV")
    partial.rename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="a KittenTTS repo id or local directory, or ai4bharat/IndicF5")
    ap.add_argument("--synthesizer", default="auto", choices=("auto", "kitten", "indicf5"))
    ap.add_argument("--testset", default="A", choices=testsets.TESTSETS)
    ap.add_argument("--lang", required=True, choices=sorted(LANGUAGES))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None, help="default: cuda when available")
    ap.add_argument("--text-fixes", default="none", choices=("none", "indic"),
                    help="apply indic/text/indic_text.py before synthesis")
    ap.add_argument("--voice", default=None, help="built-in voice for records with no reference")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--weights", default=None, help="KittenTTS weight variant")
    ap.add_argument("--reference-from", default="none", choices=("none", "fleurs"),
                    help="give reference-less records a FLEURS clone reference")
    ap.add_argument("--reference-wav", default=None)
    ap.add_argument("--reference-text", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop on the first synthesis failure instead of recording it")
    ap.add_argument("--dry-run", action="store_true", help="list the work, synthesize nothing")
    args = ap.parse_args()

    if not 0 <= args.shard < args.num_shards:
        ap.error(f"--shard must be in [0, {args.num_shards})")

    reference = None
    if args.reference_wav:
        reference = (args.reference_wav, args.reference_text)
    elif args.reference_from == "fleurs":
        reference = testsets.fleurs_reference_pool(
            args.lang, args.split, args.cache_dir, size=1)[0]

    records = testsets.build(args.testset, args.lang, args.split, args.limit or None,
                             args.cache_dir, reference)
    # Strided, not blocked: the test sets are ordered (set A by sentence id, set C by
    # category), so contiguous blocks would hand one shard every long input and
    # another every short one, and the array job would wait on the slow shard.
    shard = records[args.shard::args.num_shards]

    run_id = f"{_slug(args.model)}-{args.testset}-{args.lang}-{args.text_fixes}-s{args.seed}"
    root = pathlib.Path(__file__).resolve().parents[2]
    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"{run_id}\n  {len(records)} records, shard {args.shard}/{args.num_shards} "
          f"-> {len(shard)}\n  out {args.out}")
    if args.dry_run:
        for record in shard[:10]:
            print(f"    {record.id:<36}{record.text[:60]}")
        return

    synthesizer = build_synthesizer(args.synthesizer, args.model, args, args.hf_token)
    run = {"run_id": run_id, "model": args.model,
           "model_revision": model_revision(args.model, args.hf_token),
           "synthesizer": args.synthesizer, "testset": args.testset, "lang": args.lang,
           "split": args.split, "limit": args.limit, "seed": args.seed,
           "text_fixes": args.text_fixes, "device": args.device,
           "sample_rate": synthesizer.sample_rate,
           "shard": args.shard, "num_shards": args.num_shards,
           "n_records_total": len(records), "n_records_shard": len(shard),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "argv": sys.argv, "versions": library_versions(),
           **synthesizer.describe(), **git_state(root)}
    (args.out / f"run.{args.shard}.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False))

    meta_path = args.out / f"meta.{args.shard}.jsonl"
    generated = skipped = failed = 0
    with open(meta_path, "w", encoding="utf-8") as meta:
        for record in shard:
            wav_path = wav_dir / f"{record.id}.wav"
            seed = seed_for(record.id, args.seed)
            spoken = prepare_text(record.text, args.text_fixes)
            row = {**record.to_dict(), "run_id": run_id, "model": args.model,
                   "spoken_text": spoken, "record_seed": seed,
                   "wav": str(wav_path), "sample_rate": synthesizer.sample_rate,
                   "error": None, "synth_seconds": None, "audio_seconds": None}

            if wav_path.exists():
                skipped += 1
            else:
                seed_everything(seed)
                started = time.time()
                try:
                    audio = synthesizer.synthesize(
                        dataclasses.replace(record, text=spoken), seed)
                    write_wav(wav_path, audio, synthesizer.sample_rate)
                    row["synth_seconds"] = round(time.time() - started, 3)
                    row["audio_seconds"] = round(len(audio) / synthesizer.sample_rate, 3)
                    generated += 1
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    # One unsynthesizable record should not cost the other 399.
                    # It is written down as a failure so the scorer counts it
                    # against the model rather than quietly dropping it.
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    failed += 1
                    print(f"  FAILED {record.id}: {row['error']}", flush=True)

            meta.write(json.dumps(row, ensure_ascii=False) + "\n")
            meta.flush()

    print(f"\n{generated} generated, {skipped} already present, {failed} failed")
    print(f"wrote {meta_path}")
    if failed:
        print("a failed record has no wav; score.py counts it as a total miss")


def _slug(spec):
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(spec)).strip("-")


if __name__ == "__main__":
    main()
