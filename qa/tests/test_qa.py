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
from qa_common import (CHANGED, FAILED, NO_RESULT, PASSED, UNSUPPORTED,  # noqa: E402
                       classify, normalize_words, wer)

ASR = {"enabled": True, "model": "openai/whisper-small.en", "warn_above": 0.15, "fail_above": 0.5}


# Tests use this, not qa/config.toml, so editing the real config never breaks them.
FIXTURE = """
[sample]
text = "Hello there."
voice = "Bruno"

[models.small]
repo = "KittenML/kitten-tts-nano-0.8"
checks = ["stream"]

[models.big]
repo = "KittenML/kitten-tts-2"
checks = ["clone"]

[[target]]
name = "Big runner"
runner = "ubuntu-24.04"
pythons = ["3.11", "3.12"]
models = ["small", "big"]

[[target]]
name = "Small runner"
runner = "macos-15"
pythons = ["3.12"]
models = ["big"]
text = "Short."
overrides = { big = { weights = "emb4", warm_runs = 0 } }
"""


def spec(**kw):
    s = {"id": "linux-x64-py3.12", "name": "Linux x64", "runner": "ubuntu-24.04", "python": "3.12",
         "expect": "works", "gating": True, "reason": "", "text": "Hello there.", "voice": "Bruno",
         "models": [{"key": "nano", "label": "Nano", "checks": []}], "asr": ASR}
    s.update(kw)
    return s


def model(label="Nano", status="pass", **kw):
    m = {"key": label.lower(), "label": label, "status": status, "load_s": 1.0, "first_s": 1.2,
         "warm_s": [0.5, 0.4, 0.6], "audio_s": 4.0, "wav": f"audio/{label.lower()}.wav",
         "peak_rss_mb": 900, "checks": []}
    m.update(kw)
    return m


def result(s=None, ok=True, models=None, asr_rows=None, **install):
    """A result.json as run_target.py writes it: nothing past the install when that fails."""
    inst = {"ok": ok, "refused": False, "secs": 90.0, "versions": {"kittenml": "0.9.3"}}
    inst.update(install)
    r = {"spec": s or spec(), "env": {"cpu": "AMD EPYC 7763", "cpu_count": 4, "ram_gb": 15.6}, "install": inst}
    if ok:
        r["package"] = {"checks": [{"name": "import", "status": "pass"}]}
        r["models"] = models if models is not None else [model()]
    if asr_rows is not None:
        r["asr"] = {"status": "done", "rows": asr_rows}
    return r


class Classify(unittest.TestCase):
    def test_passing_platform(self):
        self.assertEqual(classify(result())[0], PASSED)

    def test_failed_model_fails_the_run(self):
        status, reasons, failing = classify(result(models=[model(status="fail", error="boom")]))
        self.assertEqual((status, failing), (FAILED, True))
        self.assertIn("boom", reasons[0])

    def test_crashed_model_fails(self):
        self.assertEqual(classify(result(models=[model(status="crash")]))[0], FAILED)

    def test_install_failure_on_supported_platform_fails(self):
        self.assertEqual(classify(result(ok=False, error="ERROR: no torch"))[:3:2], (FAILED, True))

    def test_expected_install_failure_is_unsupported_not_failing(self):
        s = spec(expect="install-fails")
        self.assertEqual(classify(result(s, ok=False))[0::2], (UNSUPPORTED, False))

    def test_platform_that_starts_installing_is_flagged_not_failed(self):
        s = spec(expect="install-fails")
        self.assertEqual(classify(result(s))[0::2], (CHANGED, False))

    def test_refused_python(self):
        s = spec(expect="refused", python="3.9")
        self.assertEqual(classify(result(s, ok=False, refused=True))[0::2], (UNSUPPORTED, False))
        self.assertEqual(classify(result(s, ok=True))[0::2], (FAILED, True))

    def test_non_gating_failure_is_reported_but_does_not_fail(self):
        s = spec(gating=False)
        self.assertEqual(classify(result(s, ok=False))[0::2], (FAILED, False))

    def test_wer_above_fail_threshold_fails(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.8, "transcript": "nope"}]
        self.assertEqual(classify(result(asr_rows=rows))[0], FAILED)

    def test_wer_between_thresholds_only_flags(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.3, "transcript": "close"}]
        self.assertEqual(classify(result(asr_rows=rows))[0], PASSED)

    def test_crashed_package_checks_fail(self):
        r = result()
        r["package"] = {"status": "crash", "error": "process exited with code 1 before reporting"}
        status, reasons, failing = classify(r)
        self.assertEqual((status, failing), (FAILED, True))
        self.assertIn("package checks", reasons[0])

    def test_crashed_transcription_fails(self):
        r = result()
        r["asr"] = {"status": "timeout", "error": "timed out after 20 min"}
        self.assertEqual(classify(r)[0], FAILED)

    def test_job_without_install_record_is_no_result(self):
        self.assertEqual(classify({"spec": spec()})[0], NO_RESULT)


class ExitReason(unittest.TestCase):
    def test_native_crashes_are_named(self):
        self.assertIn("0xC000001D, illegal CPU instruction", run_target.exit_reason(3221225501))
        self.assertIn("0xC0000005, access violation", run_target.exit_reason(-1073741819))
        self.assertIn("segmentation fault", run_target.exit_reason(-11))
        self.assertIn("out of memory", run_target.exit_reason(-9))
        self.assertEqual(run_target.exit_reason(1), "exited with code 1")


class Wer(unittest.TestCase):
    def test_kitten_tts_spellings_match(self):
        for said in ("KittenTTS rocks", "Kitten TTS rocks", "Kitten T.T.S. rocks", "kitten-tts rocks"):
            self.assertEqual(normalize_words(said), ["kitten", "tts", "rocks"], said)

    def test_case_and_punctuation_do_not_count(self):
        self.assertEqual(wer("Speed, quality, and consistency.", "speed quality and consistency")[0], 0)

    def test_one_substitution(self):
        w, edits, n = wer("one two three four", "one two tree four")
        self.assertEqual((edits, n), (1, 4))
        self.assertAlmostEqual(w, 0.25)


class Plan(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def test_repo_config_is_valid_and_expands(self):
        cfg = plan.load(os.path.join(QA, "config.toml"))
        jobs = plan.expand(cfg)
        self.assertTrue(jobs)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs), "job ids must be unique")
        for j in jobs:
            s = json.loads(j["spec"])
            self.assertEqual(s["id"], j["id"])
            self.assertGreater(j["timeout"], 0)

    def test_overrides_apply_to_one_target(self):
        specs = [json.loads(j["spec"]) for j in plan.expand(plan.load(self.write(FIXTURE)))]
        small = next(s for s in specs if s["name"] == "Small runner")
        big = next(s for s in specs if s["name"] == "Big runner" and s["python"] == "3.12")
        self.assertEqual(next(m for m in small["models"] if m["key"] == "big")["weights"], "emb4")
        self.assertEqual(small["text"], "Short.")
        self.assertNotIn("weights", next(m for m in big["models"] if m["key"] == "big"))
        self.assertEqual(big["text"], "Hello there.")

    def test_known_issues_reach_only_their_runner(self):
        path = self.write(FIXTURE + '\n[[known_issue]]\ncpu = "8573C"\nrunner = "macos-15"\nmodel = "big"\nreason = "r"\n')
        specs = {json.loads(j["spec"])["name"]: json.loads(j["spec"]) for j in plan.expand(plan.load(path))}
        self.assertEqual([k["model"] for k in specs["Small runner"]["known_issues"]], ["big"])
        self.assertEqual(specs["Big runner"]["known_issues"], [])

    def test_invalid_config_is_rejected_with_reasons(self):
        path = self.write('[sample]\ntext="x"\nvoice="Bruno"\n'
                          '[models.nano]\nrepo="KittenML/kitten-tts-nano-0.8"\nchecks=["clone", "bogus"]\n'
                          '[[target]]\nname="A"\nrunner="ubuntu-24.04"\npythons=["3.12"]\n'
                          'models=["nano", "missing"]\nexpect="maybe"\n')
        with self.assertRaises(SystemExit) as e:
            plan.load(path)
        msg = str(e.exception)
        for bit in ("'clone' only works with KittenTTS 2", "unknown check 'bogus'",
                    "unknown model 'missing'", "expect must be one of"):
            self.assertIn(bit, msg)

    def test_filters(self):
        cfg = plan.load(self.write(FIXTURE))
        env = {"QA_TARGETS": "big", "QA_PYTHONS": "3.12", "QA_MODELS": "small"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            jobs = plan.expand(cfg)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.assertEqual([(j["name"], j["python"]) for j in jobs], [("Big runner", "3.12")])
        self.assertEqual([m["key"] for m in json.loads(jobs[0]["spec"])["models"]], ["small"])


class Events(unittest.TestCase):
    def test_targets_run_only_for_their_events(self):
        cfg = plan.load(os.path.join(QA, "config.toml"))
        cfg = {**cfg, "target": [
            {"name": "PR only", "runner": "macos-15", "pythons": ["3.12"], "models": [], "events": ["pull_request"]},
            {"name": "Main only", "runner": "macos-15", "pythons": ["3.12"], "models": [], "events": ["push"]},
            {"name": "Always", "runner": "ubuntu-24.04", "pythons": ["3.12"], "models": []}]}
        old = os.environ.get("GITHUB_EVENT_NAME")
        try:
            for event, want in (("pull_request", ["PR only", "Always"]), ("push", ["Main only", "Always"])):
                os.environ["GITHUB_EVENT_NAME"] = event
                self.assertEqual([j["name"] for j in plan.expand(cfg)], want)
        finally:
            os.environ.pop("GITHUB_EVENT_NAME", None) if old is None else os.environ.__setitem__("GITHUB_EVENT_NAME", old)


class Report(unittest.TestCase):
    def run_report(self, results, planned=None, jobs=None, slow_minutes=30):
        d = tempfile.mkdtemp()
        for i, r in enumerate(results):
            os.makedirs(os.path.join(d, "results", str(i)))
            with open(os.path.join(d, "results", str(i), "result.json"), "w") as f:
                json.dump(r, f)
        plan_path = os.path.join(d, "plan.json")
        with open(plan_path, "w") as f:
            json.dump({"report": {"slow_job_minutes": slow_minutes},
                       "jobs": planned or [r["spec"] for r in results]}, f)
        args = [sys.executable, os.path.join(QA, "report.py"), os.path.join(d, "results"),
                os.path.join(d, "out"), "--plan", plan_path]
        if jobs is not None:
            with open(os.path.join(d, "jobs.json"), "w") as f:
                json.dump(jobs, f)
            args += ["--jobs", os.path.join(d, "jobs.json")]
        subprocess.run(args, check=True, capture_output=True)
        gate = subprocess.run([sys.executable, os.path.join(QA, "report.py"), "--gate",
                               os.path.join(d, "out", "summary.json")], capture_output=True, text=True)
        with open(os.path.join(d, "out", "pr-comment.md")) as f:
            md = f.read()
        with open(os.path.join(d, "out", "summary.md")) as f:
            self.summary = f.read()
        return md, gate.returncode, gate.stdout

    def row(self, md, first_cell):
        return next(l for l in md.splitlines() if l.startswith(f"| {first_cell} |"))

    def test_all_pass(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.0, "transcript": "Hello there."}]
        md, code, _ = self.run_report([result(asr_rows=rows)])
        self.assertEqual(code, 0)
        self.assertIn("## ✅ All 1 supported platform jobs passed", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ |", md)
        self.assertIn("| Test | Linux x64<br>AMD EPYC 7763 |", md)       # one column per platform and CPU
        self.assertIn("| Nano: speaks the sample text | ✅ |", md)
        self.assertIn("| Nano RTF | 0.10 |", md)             # best warm run 0.4 s for 4 s of audio
        self.assertIn("### KittenTTS 0.8 (ONNX models)", md)
        self.assertNotIn("### KittenTTS 2", md)               # no KittenTTS 2 in this run
        self.assertNotIn("### Failures", md)
        self.assertIn("The clip transcribes word for word.", self.summary)   # details live in the summary
        self.assertNotIn("### Every job", md)          # per-job details: summary only

    def test_tests_table_has_a_row_per_check_and_dash_where_not_run(self):
        tts2_spec = {"key": "tts2", "label": "KittenTTS 2", "repo": "KittenML/kitten-tts-2"}
        s_full = spec(models=[{**tts2_spec, "checks": ["stream", "clone"]}])
        s_quick = spec(id="mac", name="macOS Apple Silicon", runner="macos-15", models=[{**tts2_spec, "checks": []}])
        tts2 = model("KittenTTS 2", key="tts2", wav="audio/tts2.wav", status="fail", error="failed checks: clone",
                     checks=[{"name": "stream", "status": "pass"}, {"name": "clone", "status": "fail", "error": "boom"}])
        mac = result(s_quick, models=[model("KittenTTS 2", key="tts2", wav="audio/tts2.wav")])
        mac["env"]["cpu"] = "Apple M1 (Virtual)"
        md, code, _ = self.run_report([result(s_full, models=[tts2]), mac])
        self.assertEqual(code, 1)
        self.assertIn("### KittenTTS 2", md)
        self.assertIn("| Test | Linux x64<br>AMD EPYC 7763 | macOS Apple Silicon<br>Apple M1 (Virtual) |", md)
        self.assertEqual(self.row(md, "KittenTTS 2: streaming (`generate_stream`)"),
                         "| KittenTTS 2: streaming (`generate_stream`) | ✅ | — |")
        self.assertEqual(self.row(md, "KittenTTS 2: voice cloning with a transcript"),
                         "| KittenTTS 2: voice cloning with a transcript | ❌ | — |")
        self.assertIn("KittenTTS 2 · voice cloning with a transcript: boom", md)

    def test_platforms_table_has_a_column_per_python(self):
        a = result(spec(id="a", python="3.11"))
        b = result(spec(id="b", python="3.12"), models=[model(status="fail", error="bad")])
        md, code, _ = self.run_report([a, b])
        self.assertEqual(code, 1)
        self.assertIn("| Platform | CPUs | py3.11 | py3.12 | Job time |", md)
        self.assertTrue(self.row(md, "Linux x64").startswith("| Linux x64 | AMD EPYC 7763 | ✅ | ❌ |"))
        self.assertIn("## ❌ 1 of 2 supported platform jobs failed", md)

    def test_slow_jobs_are_flagged_and_failures_link_their_log(self):
        r = result(models=[model(status="fail", error="ValueError: bad audio", trace="Traceback...")])
        jobs = [{"name": "Linux x64 · py3.12", "html_url": "https://example.test/job/1",
                 "started_at": "2026-10-05T10:00:00Z", "completed_at": "2026-10-05T10:45:00Z"}]
        md, code, _ = self.run_report([r], jobs=jobs, slow_minutes=30)
        self.assertEqual(code, 1)
        self.assertIn("🐢 45 min", md)
        self.assertIn("🐢 slower than 30 min: Linux x64 · py3.12 took 45 min", md)
        self.assertIn("**Linux x64 · py3.12** · Nano: ValueError: bad audio · [log](https://example.test/job/1)", md)

    def test_cpu_names_are_shortened(self):
        for raw, short in (("AMD EPYC 7763 64-Core Processor", "AMD EPYC 7763"),
                           ("INTEL(R) XEON(R) PLATINUM 8573C", "Intel Xeon Platinum 8573C"),
                           ("Intel(R) Xeon(R) 6973P-C", "Intel Xeon 6973P-C"),
                           ("Apple M1 (Virtual)", "Apple M1 (Virtual)"),
                           ("Neoverse-N2", "Neoverse-N2")):
            self.assertEqual(report.short_cpu(raw), short)

    def test_missing_job_fails_the_gate(self):
        ok = result()
        missing = spec(id="mac-py3.12", name="macOS", runner="macos-15")
        md, code, out = self.run_report([ok], planned=[ok["spec"], missing])
        self.assertEqual(code, 1)
        self.assertIn("**macOS · py3.12**: no result", md)
        self.assertIn("FAILED macOS · py3.12", out)

    def test_wer_failure_marks_the_model_failed(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.9, "transcript": "something else"}]
        md, code, _ = self.run_report([result(asr_rows=rows)])
        self.assertEqual(code, 1)
        self.assertIn("| Nano: speaks the sample text | ❌ |", md)
        self.assertIn("Nano: WER 90.0% — Whisper heard “something else”", md)
        self.assertIn("| Nano | Failed (WER) |", self.summary)

    def test_non_gating_failure_is_labelled_and_passes_the_gate(self):
        s = spec(id="next", name="Linux x64 · next Python", python="3.15", gating=False, reason="pre-release")
        md, code, _ = self.run_report([result(), result(s, ok=False, error="ERROR: no torch for 3.15")])
        self.assertEqual(code, 0)
        self.assertTrue(self.row(md, "Linux x64").startswith("| Linux x64 | AMD EPYC 7763 | ✅ | ❌ |"))
        self.assertIn("❌ does not fail the run: Linux x64 · next Python · py3.15 (pre-release)", md)
        self.assertIn("| Install and import | ✅ |", md)     # a pre-release Python that cannot install stays out of Tests

    def test_known_issue_is_reported_but_does_not_fail(self):
        known = [{"cpu": "8573C", "model": "nano", "reason": "illegal instruction on this CPU"}]
        crashed = {"key": "nano", "label": "Nano", "status": "crash", "error": "exited with code 3221225501"}
        r = result(spec(known_issues=known), models=[crashed])   # what run_target.py records for a crash
        r["env"]["cpu"] = "INTEL(R) XEON(R) PLATINUM 8573C"
        md, code, _ = self.run_report([r])
        self.assertEqual(code, 0)
        self.assertTrue(self.row(md, "Linux x64").startswith("| Linux x64 | Intel Xeon Platinum 8573C | ❌ |"))
        self.assertIn("| Nano: speaks the sample text | ❌ |", md)
        self.assertIn("❌ known issue (does not fail the run): Linux x64 · py3.12 on Intel Xeon Platinum 8573C "
                      "(illegal instruction on this CPU)", md)
        self.assertNotIn("### Failures", md)

    def test_known_issue_only_covers_its_cpu(self):
        known = [{"cpu": "8573C", "model": "nano", "reason": "x"}]
        r = result(spec(known_issues=known), models=[model(status="crash", error="boom")])   # AMD EPYC 7763
        md, code, _ = self.run_report([r])
        self.assertEqual(code, 1)

    def test_a_timed_out_step_shows_a_timer_and_skips_the_rest(self):
        s = spec(models=[{"key": "tts2", "label": "KittenTTS 2", "repo": "KittenML/kitten-tts-2",
                          "checks": ["stream", "clone"]}])
        timed = {"key": "tts2", "label": "KittenTTS 2", "status": "timeout", "first_s": 30.0, "audio_s": 3.0,
                 "warm_s": [], "wav": "audio/tts2.wav", "error": "stream took longer than 10 min",
                 "checks": [{"name": "stream", "status": "timeout", "error": "took longer than 10 min"},
                            {"name": "clone", "status": "skipped", "error": "not run: an earlier step timed out"}]}
        md, code, _ = self.run_report([result(s, models=[timed])])
        self.assertEqual(code, 1)
        self.assertIn("| KittenTTS 2: speaks the sample text | ✅ |", md)
        self.assertIn("| KittenTTS 2: streaming (`generate_stream`) | ⏱️ |", md)
        self.assertIn("| KittenTTS 2: voice cloning with a transcript | ⏱️ |", md)
        self.assertIn("streaming (`generate_stream`): took longer than 10 min", md)
        self.assertNotIn("not run: an earlier step timed out", md)

    def test_crashed_transcription_is_shown_with_its_log(self):
        r = result()
        r["asr"] = {"status": "crash", "error": "process exited with code -9 before reporting",
                    "log_tail": "Killed"}
        md, code, _ = self.run_report([r])
        self.assertEqual(code, 1)
        self.assertIn("WER transcription: process exited with code -9", md)
        self.assertIn("````\nKilled\n````", md)
        self.assertTrue(md.rstrip().endswith("</details>"))      # the log block did not swallow the footer

    def test_unsupported_platforms_are_listed_and_pass(self):
        s = spec(id="mac-intel", name="macOS Intel", runner="macos-15-intel", expect="install-fails",
                 reason="no torch wheels")
        md, code, _ = self.run_report([result(), result(s, ok=False, error="ERROR: No matching distribution")])
        self.assertEqual(code, 0)
        self.assertIn("| macOS Intel | AMD EPYC 7763 | ❌ | — |", md)
        self.assertIn("| Install and import | ✅ | ❌ |", md)          # its column is all red crosses
        self.assertIn("| Nano RTF | 0.10 | — |", md)                  # and no speed: nothing ran
        self.assertIn("❌ not supported, as expected (does not fail the run): macOS Intel (no torch wheels)", md)

    def test_comment_stays_under_github_limit(self):
        big = [result(spec(id=f"p{i}", name=f"Platform {i}"),
                      models=[model(label=f"M{j}", status="fail", error="x" * 300, trace="t" * 5000)
                              for j in range(5)]) for i in range(40)]
        md, code, _ = self.run_report(big)
        self.assertLessEqual(len(md), report.COMMENT_LIMIT)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
