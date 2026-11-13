# Platform QA

`.github/workflows/platform-qa.yml` installs this checkout of `kittenml` on GitHub-hosted
runners (Linux, Windows and macOS; x86_64 and ARM; Intel, AMD and Apple CPUs) on every
Python in `[matrix]`, runs the README's examples for each model, and posts one report to
the pull request.

It assumes nothing about what should work. It answers two questions:

- **What works where?** Every platform runs every Python version, and the report shows
  ✅ or ❌ for each, with pip's or the test's own error for every ❌. That is the list of
  platforms and versions to add support for.
- **Did this change break anything?** Each run is compared with the latest finished run
  on `main` (or this branch's previous run when `main` has none). The run fails only when
  a test that works there stops working here. Something that does not work on `main`
  either is listed, not failed; something that starts working is marked **new**.

A job GitHub never ran (no runner free) is shown as "no result" and not counted. The first
run, with nothing to compare with, only reports.

It runs on pull requests and pushes to `main` that touch the package, its dependencies or
this folder. You can also start it from the Actions tab.

## How a job runs

1. `pip install` this checkout, as the README says. If that fails, the job reports
   ❌ install with pip's error and runs nothing else.
2. Each test is one README example in its own Python process: every model speaks the
   sample text, and each check in `config.toml` (streaming, `generate_to_file`, voice
   cloning, …) runs on its own. One that crashes or stalls cannot take the others down.
3. Each test is stopped after `[limits] step_minutes` (10 min), the install after
   `install_minutes` (20 min: on Linux pip unpacks about 4 GB of CUDA libraries). A test
   that crashes or times out is run once more; if it passes then, it is reported as flaky.
4. Whisper transcribes every clip; a WER above `[asr] fail_above` fails that test.
5. No test starts when the job is near its time limit, so every job reports.

## What the report shows

The pull request gets one comment for each commit, laid out like the React Native SDK's:

- **Summary:** the commit, how many jobs pass every test, and what it was compared with.
- **Broke in This PR:** only when something broke, one row per test with a log link.
- **Platform Status:** one row per platform, one column per Python version: ✅, ❌ with how
  many tests pass, or ❌ install. The CPUs the runners drew and the job times (🐢 slow).
- **What Does Not Work:** one row per reason, with every platform and Python it affects.
- **KittenTTS 0.8 (ONNX)** and **KittenTTS 2:** one row per platform and CPU, one column per
  test, then real-time factor, peak RAM and WER.
- **Notes:** what started working, what was flaky, and what did not run.

The run summary adds every job's numbers: load and generation time, RTF, peak RAM, what
Whisper heard, and the log of every failure.

## Changing what is tested

Edit [`config.toml`](config.toml). The workflow needs no changes.

| To… | Do this |
|---|---|
| Add or drop a Python version | `[matrix] pythons`, or `pythons` on one `[[target]]` |
| Add a platform | Add a `[[target]]` with a [runner label](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners) |
| Add a model | Add a `[models.<key>]`; every platform runs it (`models` on a target narrows that) |
| Change one model on one platform | `overrides = { tts2 = { weights = "emb4", warm_runs = 0 } }` (the report marks it) |
| Run a platform only on PRs, or only on `main` | `events = ["pull_request"]` or `events = ["push", "workflow_dispatch"]` |
| Change when a job counts as slow | `[report] slow_job_minutes` |
| Change the time limits | `[limits] step_minutes`, `job_minutes`, or `timeout_minutes` on a target |
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
