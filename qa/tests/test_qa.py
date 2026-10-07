"""Tests for the QA scripts themselves. Run: python -m unittest discover -s qa/tests"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

QA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, QA)

import plan  # noqa: E402
import run_target  # noqa: E402
import report  # noqa: E402
from qa_common import FAILED, NO_RESULT, PASSED, classify, normalize_words, wer  # noqa: E402

ASR = {"enabled": True, "model": "openai/whisper-small.en", "fail_above": 0.5}
NANO = {"key": "nano", "label": "Nano", "repo": "KittenML/kitten-tts-nano-0.8", "checks": []}
TTS2 = {"key": "tts2", "label": "KittenTTS 2", "repo": "KittenML/kitten-tts-2", "checks": ["stream", "clone"]}

# Tests use this, not qa/config.toml, so editing the real config never breaks them.
FIXTURE = """
[sample]
text = "Hello there."
voice = "Bruno"

[matrix]
pythons = ["3.11", "3.12"]

[models.small]
repo = "KittenML/kitten-tts-nano-0.8"
checks = ["stream"]

[models.big]
repo = "KittenML/kitten-tts-2"
checks = ["clone"]

[[target]]
name = "Big runner"
runner = "ubuntu-24.04"

[[target]]
name = "Small runner"
runner = "macos-15"
pythons = ["3.12"]
models = ["big"]
text = "Short."
timeout_minutes = 90
overrides = { big = { weights = "emb4", warm_runs = 0 } }

[[target]]
name = "Main only"
runner = "windows-2025"
events = ["push"]
"""


def spec(**kw):
    s = {"id": "linux-x64-py3.12", "name": "Linux x64", "runner": "ubuntu-24.04", "python": "3.12",
         "text": "Hello there.", "voice": "Bruno", "models": [NANO], "asr": ASR, "limits": {"step_minutes": 10}}
    s.update(kw)
    s["id"] = kw.get("id") or f"{s['name'].lower().replace(' ', '-')}-py{s['python']}"
    return s


def test(key="nano", title="Nano · Speak", status="pass", **kw):
    """One test row as run_target.py writes it."""
    t = {"key": key, "title": title, "status": status, "secs": 5.0, "load_s": 1.0, "gen_s": 0.4, "audio_s": 4.0,
         "rtf": 0.1, "peak_rss_mb": 900, "wer": 0.0}
    t.update(kw)
    return t


def result(s=None, ok=True, tests=None, cpu="AMD EPYC 7763 64-Core Processor", **install):
    """A result.json as run_target.py writes it: no tests when nothing installs."""
    inst = {"ok": ok, "secs": 90.0, "versions": {"kittenml": "0.9.3"}, "import_ok": ok}
    inst.update(install)
    r = {"spec": s or spec(), "env": {"cpu": cpu, "cpu_count": 4, "ram_gb": 15.6}, "install": inst,
         "tests": tests if tests is not None else ([test()] if ok or "without" in install else [])}
    return r


def tts2_tests(**by_key):
    """KittenTTS 2's three tests, all passing unless given."""
    rows = []
    for key, title in (("tts2", "KittenTTS 2 · Speak"), ("tts2:stream", "KittenTTS 2 · Stream"),
                       ("tts2:clone", "KittenTTS 2 · Clone")):
        rows.append(test(key, title, **by_key.get(key, {})))
    return rows


class Classify(unittest.TestCase):
    def test_works(self):
        self.assertEqual(classify(result()), (PASSED, []))

    def test_failed_test(self):
        status, reasons = classify(result(tests=[test(status="fail", error="ValueError: boom")]))
        self.assertEqual((status, reasons), (FAILED, ["Nano · Speak: ValueError: boom"]))

    def test_install_failure_says_why(self):
        r = result(ok=False, error="ERROR: No matching distribution found for torch>=2.6",
                   reason="pip finds no torch>=2.6 for this platform and Python")
        self.assertEqual(classify(r), (FAILED, ["install: pip finds no torch>=2.6 for this platform and Python"]))

    def test_wer_above_limit_fails(self):
        status, reasons = classify(result(tests=[test(wer=0.8, transcript="something else")]))
        self.assertEqual(status, FAILED)
        self.assertIn("Whisper heard “something else” (WER 80%)", reasons[0])

    def test_job_without_install_record_is_no_result(self):
        self.assertEqual(classify({"spec": spec()})[0], NO_RESULT)


class Runner(unittest.TestCase):
    def test_crash_reasons(self):
        self.assertEqual(run_target.crash_reason(3221225501), "crashed: illegal CPU instruction (0xC000001D)")
        self.assertEqual(run_target.crash_reason(-11), "crashed: segmentation fault (SIGSEGV)")
        self.assertIn("out of memory", run_target.crash_reason(-9))
        log = "Warning: unauthenticated\nError processing file 'phontab': No such file or directory.\n"
        self.assertEqual(run_target.crash_reason(1, log),
                         "exited with code 1: Error processing file 'phontab': No such file or directory.")

    def test_install_errors(self):
        self.assertEqual(run_target.NO_BUILD.findall("ERROR: No matching distribution found for torch>=2.6"),
                         ["torch>=2.6"])
        self.assertEqual(run_target.PY_REFUSED.search(
            "ERROR: Package 'kittenml' requires a different Python: 3.9.25 not in '>=3.10'").group(1), ">=3.10")

    def test_one_test_per_readme_example(self):
        tests = run_target.plan_tests(spec(models=[NANO, TTS2]))
        self.assertEqual([t["key"] for t in tests], ["nano", "tts2", "tts2:stream", "tts2:clone"])
        self.assertEqual(tests[3]["title"], "KittenTTS 2 · Clone")

    def test_wer(self):
        for said in ("KittenTTS rocks", "Kitten TTS rocks", "Kitten T.T.S. rocks", "kitten-tts rocks"):
            self.assertEqual(normalize_words(said), ["kitten", "tts", "rocks"], said)
        w, edits, n = wer("one two three four", "one two tree four")
        self.assertEqual((edits, n, w), (1, 4, 0.25))


class Plan(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def expand(self, path, **env):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return [json.loads(j["spec"]) for j in plan.expand(plan.load(path))]
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_repo_config_is_valid_and_expands(self):
        jobs = plan.expand(plan.load(os.path.join(QA, "config.toml")))
        self.assertTrue(jobs)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs), "job ids must be unique")

    def test_matrix_overrides_and_events(self):
        specs = self.expand(self.write(FIXTURE), GITHUB_EVENT_NAME="pull_request")
        self.assertEqual([(s["name"], s["python"]) for s in specs],
                         [("Big runner", "3.11"), ("Big runner", "3.12"), ("Small runner", "3.12")])
        big, small = specs[1], specs[2]
        self.assertEqual([m["key"] for m in big["models"]], ["small", "big"])      # every model by default
        self.assertEqual(small["models"][0]["weights"], "emb4")
        self.assertEqual(small["text"], "Short.")
        self.assertNotIn("weights", big["models"][1])
        self.assertEqual(big["limits"], {"step_minutes": 10, "job_minutes": 45})
        self.assertEqual(small["limits"]["job_minutes"], 90)                      # the runner's own deadline
        pushed = self.expand(self.write(FIXTURE), GITHUB_EVENT_NAME="push")
        self.assertIn("Main only", {s["name"] for s in pushed})

    def test_filters(self):
        specs = self.expand(self.write(FIXTURE), QA_TARGETS="big", QA_PYTHONS="3.12", QA_MODELS="small")
        self.assertEqual([(s["name"], s["python"], [m["key"] for m in s["models"]]) for s in specs],
                         [("Big runner", "3.12", ["small"])])

    def test_invalid_config_is_rejected_with_reasons(self):
        path = self.write('[sample]\ntext="x"\nvoice="Bruno"\n[limits]\nstep_minute=1\n'
                          '[models.nano]\nrepo="KittenML/kitten-tts-nano-0.8"\nchecks=["clone", "bogus"]\n'
                          '[[target]]\nname="A"\nrunner="ubuntu-24.04"\nmodels=["nano", "missing"]\nevents=["nightly"]\n')
        with self.assertRaises(SystemExit) as e:
            plan.load(path)
        for bit in ("'clone' only works with KittenTTS 2", "unknown check 'bogus'", "unknown model 'missing'",
                    "unknown event 'nightly'", "limits: unknown setting 'step_minute'"):
            self.assertIn(bit, str(e.exception))


class Report(unittest.TestCase):
    def run_report(self, results, baseline=None, planned=None, jobs=None):
        d = tempfile.mkdtemp()
        for name, rs in (("results", results), ("baseline", baseline or [])):
            os.makedirs(os.path.join(d, name))
            for i, r in enumerate(rs):
                os.makedirs(os.path.join(d, name, str(i)))
                with open(os.path.join(d, name, str(i), "result.json"), "w") as f:
                    json.dump(r, f)
        if baseline is not None:
            with open(os.path.join(d, "baseline", "about.json"), "w") as f:
                json.dump({"run_id": 1, "label": "main"}, f)
        with open(os.path.join(d, "plan.json"), "w") as f:
            json.dump({"report": {"slow_job_minutes": 30}, "jobs": planned or [r["spec"] for r in results]}, f)
        args = [sys.executable, os.path.join(QA, "report.py"), os.path.join(d, "results"), os.path.join(d, "out"),
                "--plan", os.path.join(d, "plan.json"), "--baseline", os.path.join(d, "baseline")]
        if jobs is not None:
            with open(os.path.join(d, "jobs.json"), "w") as f:
                json.dump(jobs, f)
            args += ["--jobs", os.path.join(d, "jobs.json")]
        # Without the runner's GITHUB_* variables, so the report reads the same locally and in CI.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
        subprocess.run(args, check=True, capture_output=True, env=env)
        gate = subprocess.run([sys.executable, os.path.join(QA, "report.py"), "--gate",
                               os.path.join(d, "out", "summary.json")], capture_output=True, text=True)
        with open(os.path.join(d, "out", "pr-comment.md")) as f:
            md = f.read()
        with open(os.path.join(d, "out", "summary.md")) as f:
            self.summary = f.read()
        return md, gate.returncode

    def test_first_run_reports_without_failing(self):
        broken = result(spec(python="3.13"), tests=[test(status="fail", error="ValueError: boom")])
        md, code = self.run_report([result(), broken])
        self.assertEqual(code, 0)
        self.assertIn("✅ **Report only**: there is no earlier run to compare with yet.", md)
        self.assertIn("| Results | 1 pass every test · 1 pass some · 0 cannot install |", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ | ❌ 1/2 |", md)
        self.assertIn("| Nano: Speak | ValueError: boom | Linux x64 · 3.13 |", md)

    def test_layout(self):
        s = spec(models=[NANO, TTS2])
        md, code = self.run_report([result(s, tests=[test()] + tts2_tests())], baseline=[])
        self.assertEqual(code, 0)
        self.assertTrue(md.startswith("# KittenTTS Python Platform Report\n\n"))
        for heading in ("## Summary", "## Platform Status", "## KittenTTS 0.8 (ONNX)", "## KittenTTS 2"):
            self.assertIn(heading, md)
        self.assertIn("| Platform | CPU | Speak | Avg RTF | Worst RTF | WER |", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ | 0.10 | 0.10 Nano | 0% |", md)
        self.assertIn("| Platform | CPU | Speak | Stream | Clone | RTF | Peak RAM | WER |", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ | ✅ | ✅ | 0.10 | 0.9 GB | 0% |", md)
        self.assertNotIn("## What Does Not Work", md)
        self.assertNotIn("## Job Details", md)
        self.assertIn("## Job Details", self.summary)

    def test_something_that_worked_on_main_and_breaks_fails_the_run(self):
        now = [result(tests=[test(status="fail", error="ValueError: bad audio", trace="Traceback ...")])]
        jobs = [{"name": "Linux x64 · py3.12", "html_url": "https://example.test/1",
                 "started_at": "2026-10-05T10:00:00Z", "completed_at": "2026-10-05T10:20:00Z"}]
        md, code = self.run_report(now, baseline=[result()], jobs=jobs)
        self.assertEqual(code, 1)
        self.assertIn("❌ **1 test broke** compared with main.", md)
        self.assertIn("| Linux x64 | 3.12 | AMD EPYC 7763 | Nano · Speak | ValueError: bad audio | passed in 5 s | "
                      "[log](https://example.test/1) |", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ❌ 1/2 new | 20 min |", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ❌ new |", md)

    def test_something_that_also_fails_on_main_is_listed_not_failed(self):
        no_torch = spec(name="macOS Intel", runner="macos-15-intel")
        why = "pip finds no torch>=2.6 for this platform and Python"
        md, code = self.run_report([result(no_torch, ok=False, reason=why)],
                                   baseline=[result(no_torch, ok=False, reason=why)])
        self.assertEqual(code, 0)
        self.assertIn("✅ **Nothing broke** compared with main.", md)
        self.assertIn("| macOS Intel | Intel | ❌ install |", md.replace("AMD EPYC 7763", "Intel"))
        self.assertIn(f"| Install | {why} | macOS Intel · 3.12 |", md)

    def test_the_same_reason_on_several_platforms_is_one_row(self):
        why = "pip finds no torch>=2.6 for this platform and Python"
        jobs = [result(spec(name=n, python=p), ok=False, reason=why)
                for n in ("Linux x64", "Windows x64") for p in ("3.14", "3.15")]
        jobs += [result(spec(name="macOS Intel", python="3.15"), ok=False, reason=why)]
        jobs += [result(spec(name="macOS Intel", python="3.14"))]
        md, _ = self.run_report(jobs)
        self.assertIn(f"| Install | {why} | Linux x64, Windows x64 · every Python<br>macOS Intel · 3.15 |", md)

    def test_partial_install_shows_what_works_without_the_missing_package(self):
        s = spec(name="macOS Intel", models=[NANO, TTS2])
        no_torch = "ModuleNotFoundError: No module named 'torch'"
        r = result(s, ok=False, reason="pip finds no torch>=2.6 for this platform and Python",
                   without=["torch", "torchaudio"], import_ok=True,
                   tests=[test()] + tts2_tests(**{k: {"status": "fail", "error": no_torch}
                                                  for k in ("tts2", "tts2:stream", "tts2:clone")}))
        md, code = self.run_report([r])
        self.assertIn("| macOS Intel | AMD EPYC 7763 | ❌ 1/5 |", md)
        self.assertIn("| macOS Intel ¹ | AMD EPYC 7763 | ✅ |", md)
        self.assertIn("| macOS Intel ¹ | AMD EPYC 7763 | ❌ | ❌ | ❌ |", md)
        self.assertIn("¹ `pip install kittenml` fails on macOS Intel, so kittenml was installed without torch, "
                      "torchaudio to see what works without them.", md)
        self.assertIn(f"| KittenTTS 2: every test | {no_torch} | macOS Intel · 3.12 |", md)

    def test_platform_settings_are_flagged(self):
        s = spec(name="macOS Apple Silicon", models=[TTS2], overrides={"tts2": {"weights": "emb4"}})
        md, _ = self.run_report([result(s, tests=tts2_tests())])
        self.assertIn("| macOS Apple Silicon ² | AMD EPYC 7763 | ✅ | ✅ | ✅ |", md)
        self.assertIn("² macOS Apple Silicon runs these with `weights='emb4'` (`overrides` in qa/config.toml).", md)

    def test_failure_on_a_cpu_main_never_drew_is_reported_not_failed(self):
        crash = test(status="crash", error="crashed: illegal CPU instruction (0xC000001D)")
        md, code = self.run_report([result(tests=[crash], cpu="INTEL(R) XEON(R) PLATINUM 8573C")],
                                   baseline=[result()])
        self.assertEqual(code, 0)
        self.assertIn("**Not counted as broken**: Linux x64 · 3.12: Nano Speak on Intel Xeon Platinum 8573C, "
                      "which the baseline never drew", md)

    def test_an_install_that_breaks_fails_even_on_a_new_cpu(self):
        md, code = self.run_report([result(ok=False, reason="ERROR: bad dependency", cpu="Some new CPU")],
                                   baseline=[result()])
        self.assertEqual(code, 1)

    def test_something_that_starts_working_is_marked(self):
        md, code = self.run_report([result()], baseline=[result(ok=False, reason="no torch")])
        self.assertEqual(code, 0)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ new |", md)
        self.assertIn("**Works now**, did not in the baseline: Linux x64 · 3.12: Install", md)

    def test_missing_job_that_worked_on_main_fails(self):
        md, code = self.run_report([], baseline=[result()], planned=[spec()])
        self.assertEqual(code, 1)
        self.assertIn("| Linux x64 | ubuntu-24.04 | ❌ no result new |", md)

    def test_timeouts_count_only_when_main_ran_clean_and_fast(self):
        s = spec(models=[TTS2])
        stalled = {"tts2:stream": {"status": "timeout", "error": "took longer than 10 min, twice"}}
        now = result(s, tests=tts2_tests(**stalled))
        md, code = self.run_report([now], baseline=[result(s, tests=tts2_tests())])
        self.assertEqual(code, 1)                                  # main streamed in 5 s
        md, code = self.run_report([now], baseline=[result(s, tests=tts2_tests(**{"tts2:stream": {"secs": 400.0}}))])
        self.assertEqual(code, 0)                                  # 7 min on main: a slow runner
        self.assertIn("Linux x64 · 3.12: KittenTTS 2 Stream timed out; it took 7 min in the baseline too", md)
        other = result(spec(models=[TTS2], python="3.13"), tests=tts2_tests(**{"tts2:clone": {"status": "timeout"}}))
        md, code = self.run_report([now], baseline=[result(s, tests=tts2_tests()), other])
        self.assertEqual(code, 0)                                  # main stalled on this platform too
        self.assertIn("tests on this platform timed out in the baseline too", md)
        self.assertIn("| KittenTTS 2: Stream | took longer than 10 min, twice | Linux x64 · 3.12 |", md)

    def test_skipped_tests_never_count(self):
        s = spec(models=[TTS2])
        now = result(s, tests=tts2_tests(**{"tts2:clone": {"status": "skipped", "error": "not run"}}))
        md, code = self.run_report([now], baseline=[result(s, tests=tts2_tests())])
        self.assertEqual(code, 0)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ | ✅ | — |", md)

    def test_python_versions_in_cells(self):
        jobs = [result(spec(python=p), tests=[test(status="fail" if p in ("3.10", "3.11", "3.13") else "pass")])
                for p in ("3.10", "3.11", "3.12", "3.13")]
        md, _ = self.run_report(jobs)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ❌ 3.10–3.11, 3.13 |", md)

    def test_wer_failure(self):
        md, code = self.run_report([result(tests=[test(wer=0.9, transcript="something else")])],
                                   baseline=[result()])
        self.assertEqual(code, 1)
        self.assertIn("Whisper heard “something else” (WER 90%)", md)

    def test_flaky_is_listed_not_failed(self):
        flaky = test(flaky="crashed: segmentation fault (SIGSEGV) the first time; passed when run again")
        md, code = self.run_report([result(tests=[flaky])], baseline=[result()])
        self.assertEqual(code, 0)
        self.assertIn("**Flaky**, failed and then passed when run again: Linux x64 · 3.12: Nano Speak "
                      "(crashed: segmentation fault (SIGSEGV) the first time; passed when run again)", md)

    def test_logs_cannot_break_the_summary(self):
        crash = test(status="crash", error="crashed", log_tail="```\nKilled")
        self.run_report([result(tests=[crash])], baseline=[result()])
        self.assertIn("````\n```\nKilled\n````", self.summary)

    def test_comment_stays_under_github_limit(self):
        models = [{"key": f"m{j}", "label": f"M{j}", "repo": "KittenML/x", "checks": []} for j in range(5)]

        def job(i, status):
            return result(spec(name=f"Platform {i}", models=models),
                          tests=[test(f"m{j}", f"M{j} · Speak", status=status, error="x" * 300) for j in range(5)])
        md, code = self.run_report([job(i, "fail") for i in range(40)], baseline=[job(i, "pass") for i in range(40)])
        self.assertLessEqual(len(md), report.COMMENT_LIMIT)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
