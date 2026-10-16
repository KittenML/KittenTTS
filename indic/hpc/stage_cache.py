"""Pre-fetch every Hugging Face artifact on a login node, so compute nodes need no network.

On most clusters the compute nodes have no route off the machine. Nothing in this
stack says so politely: `hf_hub_download` hangs until its timeout and then raises
somewhere deep inside a model constructor, an hour into an array job, on the task
that happened to be first to need the vocoder. Worse, the failure is *per task* --
a 300-task sweep can get 280 of its artifacts from a warm cache and fail the other
20 for one missing file.

So the artifacts are staged once, from somewhere with a network, into a directory
that the compute nodes can read; then the staging is verified under the same
offline conditions the job will see. What gets staged is derived from the code
that will later ask for it, not from a hand-kept list:

    config.json of the TTS repo    ->  kittenml/get_model.py::_kitten2_files
    s3gen vocoder weights          ->  kittentts2/vocoder.py (CHATTERBOX_REPO)
    Sarvam-1 tokenizer             ->  indic/vocab/build_sarvam_vocab.py
    FLEURS transcripts             ->  indic/bench/tokenizer_fertility.py
    ASR + speaker scorers          ->  docs/indic/evaluation.md (flag-gated)
    baseline TTS systems           ->  docs/indic/evaluation.md (flag-gated)

A manifest of exactly which files were staged, at which revision and what size,
is written next to the cache. Verification reads that manifest rather than
guessing, so "the cache is complete" means complete *for this sweep* and a
re-stage at a different revision is caught instead of silently mixing arms.

    # on a login node, with a network
    python -m indic.hpc.stage_cache download --hf-home /scratch/shared/hf --asr

    # anywhere, including inside the job, as a precondition
    python -m indic.hpc.stage_cache verify --hf-home /scratch/shared/hf

    # the export lines a compute node needs
    eval "$(python -m indic.hpc.stage_cache env --hf-home /scratch/shared/hf)"

Environment on compute nodes
----------------------------
`HF_HOME` points at the staged tree. The hub cache is `$HF_HOME/hub`, which is why
this tool takes `--hf-home` and appends `hub` itself: pass the two different
things and the node will look in an empty directory and try to download.

`HF_HUB_OFFLINE=1` makes `huggingface_hub` resolve from the cache and raise
immediately instead of waiting out a TCP timeout it will never satisfy. Without
it, a node with no route spends its wall clock in `connect()`.

`TRANSFORMERS_OFFLINE=1` is the same switch for the code paths in `transformers`
that still consult the Hub for their own reasons (auto-class resolution,
`trust_remote_code` lookups). Both are needed; neither covers the other.

`HF_DATASETS_OFFLINE=1` likewise for `datasets`, which the eval harness uses to
read FLEURS audio.

`TMPDIR` matters more than it looks. On many clusters `/tmp` is a small tmpfs
carved out of the node's RAM, shared by every job on the node, and a few hundred
megabytes of extracted FLEURS audio in it either fills it or counts against the
job's memory limit and gets the job killed for OOM with no mention of /tmp. Point
it at node-local scratch. The same argument applies to the compile caches that
default under `$HOME`: on a shared filesystem, 300 array tasks JIT-compiling into
one Triton cache directory race each other and produce corrupted entries, so they
belong on node-local scratch too, one per task.
"""

import argparse
import dataclasses
import fnmatch
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.langs import ALL, LANGUAGES

MANIFEST_NAME = "indic-stage-manifest.json"

# The TTS checkpoint itself. Its file list is not hardcoded here: `_kitten2_files`
# in kittenml/get_model.py decides which weight variant and decoder a given config
# actually points at, and staging anything else would either miss a file the model
# loads or pull the ~1.2 GB llama.cpp bundle that the Python path never opens.
KITTEN_REPO = "KittenML/kitten-tts-2"

# The s3gen vocoder, fetched by S3Codec.__init__ on every single construction of
# KittenTTS2. It lives in a different repo from the checkpoint, which is the file
# most likely to be forgotten when staging by hand.
VOCODER_REPO = "ResembleAI/chatterbox-turbo"
VOCODER_FILE = "s3gen_meanflow.safetensors"

SARVAM_REPO = "sarvamai/sarvam-1"
# Tokenizer files only. sarvam-1 is a 2B language model and none of its weights
# are used — `build_sarvam_vocab.py` reads the vocabulary, not the model.
SARVAM_PATTERNS = ("config.json", "tokenizer.json", "tokenizer.model",
                   "tokenizer_config.json", "special_tokens_map.json")

FLEURS_REPO = "google/fleurs"

# Scoring and baseline repositories. The first three match the MODEL_ID constants
# the harness actually loads (`indic/eval/asr.py`, `indic/eval/metrics.py`), which
# is where to look when one of these stops being right; `docs/indic/evaluation.md`
# is the plan these came from. Extend with --extra-repo rather than editing this
# table. A wrong id or a stale pattern fails at download time on the login node,
# which is the right place for it to fail.
SCORING_REPOS = {
    # ONNX encoder plus a per-language joint/decoder head, so the whole repo is
    # wanted; it carries no duplicate-format weights to filter.
    "indicconformer": ("ai4bharat/indic-conformer-600m-multilingual", ()),
    # Also the transcriber KittenTTS2 loads when a clone call omits the reference
    # transcript — staging it is what makes offline voice cloning work at all.
    "whisper": ("openai/whisper-large-v3", ()),
    # The third ASR cross-check in docs/indic/evaluation.md. `indic/eval/asr.py`
    # does not load it today, so this is staged ahead of a scorer that may want
    # it; drop the entry if that never happens. 3,600 files, because it ships one
    # adapter per language — only the base model and the adapters for the
    # languages being scored are taken, which is 40x less than the whole repo.
    "mms": ("facebook/mms-1b-all", "per-language"),
    "wavlm-sv": ("microsoft/wavlm-base-plus-sv", ()),
}
BASELINE_REPOS = {
    "indicf5": "ai4bharat/IndicF5",
    "indic-parler-tts": "ai4bharat/indic-parler-tts",
}

# Weights for frameworks this stack does not use. Staged repos routinely carry
# two or three copies of the same model, and on whisper-large-v3 the copies are
# most of the download.
FOREIGN_FRAMEWORK = ("*.h5", "*.msgpack", "*.ot", "*.tflite", "tf_model*",
                     "flax_model*", "*.fp32-*", "*.index.fp32.json")

GROUPS = ("core", "fleurs-audio", "asr", "baselines")


@dataclasses.dataclass
class Artifact:
    """One repository, and the subset of it this pipeline needs."""
    name: str
    repo_id: str
    repo_type: str
    group: str
    why: str
    patterns: tuple = ()
    ignore: tuple = ()
    # An offline load worth attempting during verification, because file presence
    # is not the same as usability: a tokenizer.json can be staged and still fail
    # to build if transformers wants a sibling file nobody listed.
    loads_as: str = ""


def plan(groups, weights=None, decoder=None, languages=ALL, splits=("dev",),
         hf_home=None, token=None, extra=()):
    """The artifacts to stage, given which groups were asked for.

    Reads the TTS repo's config.json first (a few kB) because the rest of that
    repo's file list is a function of it.
    """
    artifacts = []
    if "core" in groups:
        patterns = _kitten_patterns(weights, decoder, hf_home, token)
        artifacts += [
            Artifact("kitten-tts-2", KITTEN_REPO, "model", "core",
                     "the checkpoint, its tokenizer, voices and speaker head",
                     tuple(patterns), loads_as="kitten"),
            Artifact("s3gen", VOCODER_REPO, "model", "core",
                     "the vocoder every generation runs through",
                     (VOCODER_FILE,), loads_as="safetensors"),
            Artifact("sarvam-1-tokenizer", SARVAM_REPO, "model", "core",
                     "the replacement text vocabulary under test",
                     SARVAM_PATTERNS, loads_as="tokenizer"),
            Artifact("fleurs-text", FLEURS_REPO, "dataset", "core",
                     "parallel transcripts for fertility and eval text",
                     tuple(f"data/{LANGUAGES[c].fleurs}/{s}.tsv"
                           for c in languages for s in splits)),
        ]
    if "fleurs-audio" in groups:
        artifacts.append(
            Artifact("fleurs-audio", FLEURS_REPO, "dataset", "fleurs-audio",
                     "ground-truth recordings: the CER floor, and clone references",
                     tuple(f"data/{LANGUAGES[c].fleurs}/audio/{s}.tar.gz"
                           for c in languages for s in splits)))
    if "asr" in groups:
        for name, (repo, patterns) in SCORING_REPOS.items():
            if patterns == "per-language":
                # MMS names its adapters and vocabularies by ISO-639-3, which is the
                # first field of the FLORES code `langs.py` already carries. Named
                # one by one rather than globbed, so a renamed file is an error
                # here instead of a quietly smaller download.
                iso3 = [LANGUAGES[c].flores.split("_")[0] for c in languages]
                patterns = ("config.json", "preprocessor_config.json",
                            "special_tokens_map.json", "tokenizer_config.json",
                            "vocab.json", "model.safetensors",
                            *(f"adapter.{code}.safetensors" for code in iso3),
                            *(f"vocabs/{code}.txt" for code in iso3))
            artifacts.append(Artifact(name, repo, "model", "asr",
                                      "scoring model (docs/indic/evaluation.md)",
                                      tuple(patterns), FOREIGN_FRAMEWORK))
    if "baselines" in groups:
        artifacts += [Artifact(name, repo, "model", "baselines",
                               "baseline TTS system (docs/indic/evaluation.md)",
                               (), FOREIGN_FRAMEWORK)
                      for name, repo in BASELINE_REPOS.items()]
    for spec in extra:
        name, _, repo = spec.partition("=")
        artifacts.append(Artifact(name or repo, repo or name, "model", "extra",
                                  "requested with --extra-repo", (), FOREIGN_FRAMEWORK))
    return artifacts


def _kitten_patterns(weights, decoder, hf_home, token):
    """Which files of the TTS repo this configuration needs.

    Delegated to the dispatcher that will later do the download for real, so the
    staged set cannot drift from the loaded set. Imported here rather than at
    module scope so that `verify` on a bare node does not need the package's
    dependency tree.
    """
    from huggingface_hub import hf_hub_download

    try:
        from kittenml.get_model import _kitten2_files
    except ImportError as error:
        raise SystemExit(
            f"cannot import kittenml.get_model ({error}). Run this from the repo "
            f"root, or install the package, so the staged file set matches what "
            f"the model will ask for.")

    config_path = hf_hub_download(KITTEN_REPO, "config.json", token=token,
                                  cache_dir=hub_cache(hf_home))
    config = json.loads(pathlib.Path(config_path).read_text())
    return _kitten2_files(config, decoder, weights)


def hub_cache(hf_home):
    """`$HF_HOME/hub`, which is where huggingface_hub actually keeps repositories.

    Taking HF_HOME and deriving this is deliberate: the operator exports HF_HOME on
    the compute node, and a tool that staged into HF_HOME itself would leave that
    node looking one directory too high at an empty tree.
    """
    return str(pathlib.Path(hf_home).expanduser() / "hub") if hf_home else None


def manifest_path(hf_home):
    return pathlib.Path(hf_home).expanduser() / MANIFEST_NAME


def download(artifacts, hf_home, token=None, max_workers=4):
    """Fetch every artifact and record what landed. Returns the manifest dict."""
    from huggingface_hub import HfApi, snapshot_download

    cache = hub_cache(hf_home)
    pathlib.Path(hf_home).expanduser().mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    entries = []

    for artifact in artifacts:
        print(f"\n== {artifact.name}  ({artifact.repo_id}, {artifact.repo_type})")
        print(f"   {artifact.why}")
        available = api.list_repo_files(artifact.repo_id, repo_type=artifact.repo_type)
        # A named file that is not there is an error: the repository reorganised,
        # and staging the rest produces a cache that looks fine until the job tries
        # to load it. A *glob* that matches nothing is only a warning, because a
        # glob is how an optional set of sidecar files gets asked for.
        unmatched = [p for p in artifact.patterns if not fnmatch.filter(available, p)]
        literals = [p for p in unmatched if not set(p) & set("*?[")]
        if literals:
            raise SystemExit(
                f"{artifact.repo_id}: these files are not in the repository: "
                f"{literals}. It has {len(available)} files, e.g. {available[:5]}. "
                f"The layout changed, or the pattern is wrong — fix it rather than "
                f"staging the whole repository.")
        for pattern in unmatched:
            print(f"   ! {pattern} matched nothing")
        wanted = _select(artifact.patterns, artifact.ignore, available)

        print(f"   {len(wanted)} files")
        local = snapshot_download(
            artifact.repo_id, repo_type=artifact.repo_type, cache_dir=cache,
            token=token, max_workers=max_workers,
            allow_patterns=list(artifact.patterns) or None,
            ignore_patterns=list(artifact.ignore) or None)
        revision = pathlib.Path(local).name
        files = []
        for name in wanted:
            path = pathlib.Path(local) / name
            if not path.exists():
                raise SystemExit(f"{artifact.repo_id}: {name} did not land in {local}")
            files.append({"name": name, "size": path.stat().st_size})
        total = sum(f["size"] for f in files)
        print(f"   {total / 1e9:.2f} GB at revision {revision[:12]}")
        entries.append({**dataclasses.asdict(artifact), "revision": revision,
                        "files": files, "bytes": total})

    manifest = {
        "hf_home": str(pathlib.Path(hf_home).expanduser()),
        "staged_by": os.environ.get("USER", ""),
        "artifacts": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }
    manifest_path(hf_home).write_text(json.dumps(manifest, indent=2))
    return manifest


def _select(patterns, ignore, available):
    """The repository files a pattern/ignore pair selects, in repository order.

    Mirrors `snapshot_download`'s own allow/ignore semantics so the manifest
    records what the download actually produced. No patterns means the whole
    repository minus the ignores.
    """
    if patterns:
        selected = set()
        for pattern in patterns:
            selected.update(fnmatch.filter(available, pattern))
    else:
        selected = set(available)
    for pattern in ignore:
        selected.difference_update(fnmatch.filter(available, pattern))
    return [name for name in available if name in selected]


def verify(hf_home, check_loads=True):
    """Resolve every manifest entry from the cache alone. Returns a list of problems.

    Uses `local_files_only=True`, which is the same resolution path `HF_HUB_OFFLINE=1`
    forces, and pins the revision the manifest recorded so a cache that was re-staged
    at a different commit reports as wrong rather than as fine.
    """
    # `_CACHED_NO_EXIST` is the sentinel for "the Hub was asked and said 404, and
    # that answer is cached". It is re-exported from the package root in
    # huggingface_hub 0.x and 2.x alike; the submodule it lives in has moved.
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    path = manifest_path(hf_home)
    if not path.is_file():
        return [f"no manifest at {path} — this cache was never staged by this tool"]

    manifest = json.loads(path.read_text())
    cache = hub_cache(hf_home)
    problems = []
    for entry in manifest["artifacts"]:
        for record in entry["files"]:
            resolved = try_to_load_from_cache(
                entry["repo_id"], record["name"], cache_dir=cache,
                revision=entry["revision"], repo_type=entry["repo_type"])
            if resolved is _CACHED_NO_EXIST:
                problems.append(f"{entry['repo_id']}:{record['name']} is cached as "
                                f"known-missing (a 404 was stored, not the file)")
            elif resolved is None:
                problems.append(f"{entry['repo_id']}:{record['name']} not in the cache "
                                f"at revision {entry['revision'][:12]}")
            elif not os.path.exists(resolved):
                problems.append(f"{entry['repo_id']}:{record['name']} resolves to "
                                f"{resolved}, which does not exist (dangling symlink: "
                                f"the blob was deleted or never copied)")
            else:
                size = os.path.getsize(resolved)
                if size != record["size"]:
                    problems.append(
                        f"{entry['repo_id']}:{record['name']} is {size} bytes, staged "
                        f"as {record['size']} — truncated or replaced")
        if check_loads and entry.get("loads_as") and not problems:
            problems += _check_load(entry, cache)
    return problems


def _check_load(entry, cache):
    """Construct what the pipeline will construct, offline.

    File presence is necessary and not sufficient: a tokenizer resolves sibling
    files of its own, and a safetensors file that copied badly opens as a header
    error rather than a short read.
    """
    try:
        if entry["loads_as"] == "tokenizer":
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(entry["repo_id"], cache_dir=cache,
                                          local_files_only=True)
        elif entry["loads_as"] == "kitten":
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(entry["repo_id"], subfolder="lm",
                                          cache_dir=cache, local_files_only=True)
        elif entry["loads_as"] == "safetensors":
            from huggingface_hub import try_to_load_from_cache
            from safetensors import safe_open
            path = try_to_load_from_cache(entry["repo_id"], entry["files"][0]["name"],
                                          cache_dir=cache, revision=entry["revision"],
                                          repo_type=entry["repo_type"])
            with safe_open(str(path), framework="pt") as handle:
                handle.keys()
    except Exception as error:
        return [f"{entry['repo_id']}: staged files are present but will not load "
                f"offline — {type(error).__name__}: {error}"]
    return []


def offline_env(hf_home):
    """The variables a compute node must have, as (name, value) pairs.

    See the module docstring for why each one is here. Returned rather than set,
    because a job script should export them before Python starts: `transformers`
    reads TRANSFORMERS_OFFLINE at import time, and setting it afterwards is too
    late.
    """
    home = str(pathlib.Path(hf_home).expanduser())
    return [
        ("HF_HOME", home),
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_DATASETS_OFFLINE", "1"),
        # Telemetry is an outbound request that offline mode does not cover on
        # every version; on a node with no route it is a timeout per process.
        ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["download", "verify", "env", "plan"])
    ap.add_argument("--hf-home", required=True,
                    help="the directory to export as HF_HOME on compute nodes; "
                         "repositories land in <HF_HOME>/hub")
    ap.add_argument("--asr", action="store_true",
                    help="also stage the ASR and speaker-similarity scorers "
                         "(several GB; needed by the scoring job and by cloning "
                         "without a reference transcript)")
    ap.add_argument("--baselines", action="store_true",
                    help="also stage the baseline TTS systems")
    ap.add_argument("--fleurs-audio", action="store_true",
                    help="also stage FLEURS audio archives (large; needed for the "
                         "ground-truth CER floor)")
    ap.add_argument("--weights", default=None,
                    help="weight variant to stage (default: the repo's own default)")
    ap.add_argument("--decoder", default=None, help="decoder variant to stage")
    ap.add_argument("--langs", default=",".join(ALL),
                    help=f"comma-separated language codes from {ALL}")
    ap.add_argument("--splits", default="dev",
                    help="comma-separated FLEURS splits, e.g. dev,test")
    ap.add_argument("--extra-repo", action="append", default=[], metavar="NAME=REPO",
                    help="stage another repository in full, repeatable")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="Hugging Face token for gated repositories")
    ap.add_argument("--no-load-check", action="store_true",
                    help="verify file presence only, skipping the offline construction")
    ap.add_argument("--in-offline-child", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.action == "env":
        for name, value in offline_env(args.hf_home):
            print(f"export {name}={value}")
        return 0

    groups = ["core"]
    groups += ["asr"] if args.asr else []
    groups += ["baselines"] if args.baselines else []
    groups += ["fleurs-audio"] if getattr(args, "fleurs_audio") else []

    if args.action == "verify":
        # Re-run under the real offline switches rather than trusting
        # local_files_only to cover every library: `datasets` and `transformers`
        # each have their own paths out to the network, and a check that passes
        # with a network available proves nothing about a node without one.
        if not args.in_offline_child:
            env = {**os.environ, **dict(offline_env(args.hf_home))}
            return subprocess.call([sys.executable, "-m", "indic.hpc.stage_cache",
                                    "verify", "--hf-home", args.hf_home,
                                    "--in-offline-child",
                                    *(["--no-load-check"] if args.no_load_check else [])],
                                   env=env)
        problems = verify(args.hf_home, check_loads=not args.no_load_check)
        if problems:
            print(f"CACHE INCOMPLETE — {len(problems)} problem(s) under "
                  f"HF_HUB_OFFLINE=1:\n", file=sys.stderr)
            for problem in problems:
                print(f"  X {problem}", file=sys.stderr)
            print(f"\nRe-stage on a node with a network:\n"
                  f"  python -m indic.hpc.stage_cache download --hf-home {args.hf_home}",
                  file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path(args.hf_home).read_text())
        print(f"cache OK under HF_HUB_OFFLINE=1: "
              f"{len(manifest['artifacts'])} repositories, "
              f"{sum(len(a['files']) for a in manifest['artifacts'])} files, "
              f"{manifest['total_bytes'] / 1e9:.2f} GB")
        return 0

    artifacts = plan(groups, weights=args.weights, decoder=args.decoder,
                     languages=args.langs.split(","), splits=tuple(args.splits.split(",")),
                     hf_home=args.hf_home, token=args.token, extra=args.extra_repo)

    if args.action == "plan":
        for artifact in artifacts:
            print(f"{artifact.group:<13} {artifact.name:<22} {artifact.repo_id}")
            print(f"{'':<13} {artifact.why}")
            for pattern in artifact.patterns or ("(whole repository)",):
                print(f"{'':<13}   {pattern}")
            if artifact.ignore:
                print(f"{'':<13}   minus {', '.join(artifact.ignore)}")
        return 0

    manifest = download(artifacts, args.hf_home, token=args.token)
    print(f"\nstaged {manifest['total_bytes'] / 1e9:.2f} GB into "
          f"{hub_cache(args.hf_home)}")
    print(f"manifest: {manifest_path(args.hf_home)}")
    print("\nexport these on compute nodes:")
    for name, value in offline_env(args.hf_home):
        print(f"  export {name}={value}")
    print("\nthen confirm with:")
    print(f"  python -m indic.hpc.stage_cache verify --hf-home {args.hf_home}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
