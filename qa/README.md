# Platform QA

`.github/workflows/platform-qa.yml` installs this checkout of `kittenml` on GitHub-hosted
runners — Linux, Windows and macOS, x86_64 and ARM, Intel, AMD and Apple CPUs — on every
supported Python version. It runs each model and posts one report to the pull request.

It runs on pull requests and pushes to `main` that touch the package, its dependencies or
this folder. You can also start it from the Actions tab.

macOS runs every KittenTTS 2 test only on pushes to `main` and manual runs. GitHub's M1
runner is about 200× slower than realtime for KittenTTS 2, so on a pull request it only
checks that KittenTTS 2 loads and speaks one line.

## What the report shows

The pull request gets one short comment for each commit:

- **Platforms:** one row per platform and one column per Python version, with the CPUs the
  runners drew and how long the jobs took. Jobs over `report.slow_job_minutes` get 🐢.
- **Tests:** one row per README example (install and import; each model speaks the sample
  text; streaming, speed, `generate_to_file`, expression tags, both kinds of voice cloning,
  `weights="emb4"`) and one column per platform. A failure names the Python versions it
  failed on. — means that test does not run on that platform for this trigger.
- **Speed:** each model's real-time factor (generation time ÷ audio length) per platform.
  🐢 means slower than realtime.
- **Failures:** one line each, with a link to the job's log and the log's tail.

The run summary has everything above plus every job's numbers: load time, first and warm
generation times, peak RAM, what Whisper heard, every test's result and an audio download.

The run fails when a supported platform fails, a job produces no result, or a model's WER
is above `asr.fail_above`.

## Changing what is tested

Edit [`config.toml`](config.toml). The workflow needs no changes.

| To… | Do this |
|---|---|
| Add or drop a Python version | Edit `pythons` on a `[[target]]` |
| Add a platform | Add a `[[target]]` with a [runner label](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners) |
| Mark a platform as known-unsupported | `expect = "install-fails"` and a `reason` |
| Report a platform without failing the run | `gating = false` |
| Add a model | Add a `[models.<key>]` and list the key in a target's `models` |
| Change one model on one platform | `overrides = { tts2 = { weights = "emb4", warm_runs = 0 } }` |
| Run a platform only on PRs, or only on `main` | `events = ["pull_request"]` or `events = ["push", "workflow_dispatch"]` |
| Change when a job counts as slow | `[report] slow_job_minutes` |
| Stop a known, tracked bug from failing every run | a `[[known_issue]]` with `cpu`, `runner`, `model`, `reason` |
| Change the spoken text or voice | `[sample]`, or `text = "..."` on a target |
| Change the WER thresholds or ASR model | `[asr]` |

`python qa/plan.py` checks the config and prints the jobs it expands to. The Plan job runs
it too, along with `python -m unittest discover -s qa/tests`.

## Narrowing a manual run

**Actions → Platform QA → Run workflow** takes comma-separated filters: `targets` (matches
name or runner, e.g. `Linux, macos`), `pythons` (e.g. `3.12`) and `models` (e.g.
`nano,tts2`). `source` set to a pip requirement such as `kittenml==0.9.3` tests a PyPI
release instead of the branch.

## Running one platform locally

```bash
python qa/plan.py --out plan.json          # QA_TARGETS / QA_PYTHONS / QA_MODELS filter it
python -c "import json; json.dump(json.load(open('plan.json'))['jobs'][0], open('spec.json', 'w'))"
python qa/run_target.py --spec spec.json --out qa-out
python qa/report.py qa-out report && open report/summary.md
```

Keep the virtualenv path short: espeak-ng exits the whole process when its data path is
160 characters or longer.

## Pull requests from forks

GitHub gives fork pull requests a read-only token, so the report job cannot comment on
them. `platform-qa-comment.yml` runs after the workflow, in the base repository, and posts
the report. It only works once it is on the default branch.
