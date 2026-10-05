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
overrides = { big = { weights = "emb4", warm_runs = 0 } }

[[target]]
name = "Main only"
runner = "windows-2025"
events = ["push"]
"""


def spec(**kw):
    s = {"id": "linux-x64-py3.12", "name": "Linux x64", "runner": "ubuntu-24.04", "python": "3.12",
         "text": "Hello there.", "voice": "Bruno", "models": [NANO], "asr": ASR}
    s.update(kw)
    return s


def model(label="Nano", status="pass", **kw):
    m = {"key": label.lower(), "label": label, "status": status, "load_s": 1.0, "first_s": 1.2,
         "warm_s": [0.5, 0.4, 0.6], "audio_s": 4.0, "wav": f"audio/{label.lower()}.wav",
         "peak_rss_mb": 900, "checks": []}
    m.update(kw)
    return m


def result(s=None, ok=True, models=None, asr_rows=None, cpu="AMD EPYC 7763 64-Core Processor", **install):
    """A result.json as run_target.py writes it: nothing past the install when that fails."""
    inst = {"ok": ok, "secs": 90.0, "versions": {"kittenml": "0.9.3"}}
    inst.update(install)
    r = {"spec": s or spec(), "env": {"cpu": cpu, "cpu_count": 4, "ram_gb": 15.6}, "install": inst}
    if ok:
        r["package"] = {"checks": [{"name": "import", "status": "pass"}]}
        r["models"] = models if models is not None else [model()]
    if asr_rows is not None:
        r["asr"] = {"status": "done", "rows": asr_rows}
    return r


class Classify(unittest.TestCase):
    def test_works(self):
        self.assertEqual(classify(result()), (PASSED, []))

    def test_failed_model(self):
        status, reasons = classify(result(models=[model(status="fail", error="boom")]))
        self.assertEqual(status, FAILED)
        self.assertIn("boom", reasons[0])

    def test_install_failure_says_why(self):
        self.assertEqual(classify(result(ok=False, error="ERROR: no torch")), (FAILED, ["install: ERROR: no torch"]))

    def test_wer_above_limit_fails(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.8}]
        self.assertEqual(classify(result(asr_rows=rows))[0], FAILED)

    def test_crashed_package_checks_and_transcription_fail(self):
        r = result()
        r["package"] = {"status": "crash", "error": "exited"}
        self.assertEqual(classify(r)[0], FAILED)
        r = result()
        r["asr"] = {"status": "timeout", "error": "slow"}
        self.assertEqual(classify(r)[0], FAILED)

    def test_job_without_install_record_is_no_result(self):
        self.assertEqual(classify({"spec": spec()})[0], NO_RESULT)


class Helpers(unittest.TestCase):
    def test_exit_codes(self):
        self.assertIn("0xC000001D, illegal CPU instruction", run_target.exit_reason(3221225501))
        self.assertIn("segmentation fault", run_target.exit_reason(-11))
        self.assertIn("out of memory", run_target.exit_reason(-9))

    def test_wer(self):
        for said in ("KittenTTS rocks", "Kitten TTS rocks", "Kitten T.T.S. rocks", "kitten-tts rocks"):
            self.assertEqual(normalize_words(said), ["kitten", "tts", "rocks"], said)
        w, edits, n = wer("one two three four", "one two tree four")
        self.assertEqual((edits, n, w), (1, 4, 0.25))

    def test_cpu_names_and_python_ranges(self):
        self.assertEqual(report.short_cpu("INTEL(R) XEON(R) PLATINUM 8573C"), "Intel Xeon Platinum 8573C")
        self.assertEqual(report.short_cpu("AMD EPYC 7763 64-Core Processor"), "AMD EPYC 7763")
        everyone = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14", "3.15"]
        self.assertEqual(report.py_list(everyone, everyone), "py3.9–3.15")
        self.assertEqual(report.py_list(["3.9", "3.11", "3.12"], everyone), "py3.9, py3.11–3.12")


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
        self.assertEqual(small["limits"]["step_minutes"], 10)
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
        subprocess.run(args, check=True, capture_output=True)
        gate = subprocess.run([sys.executable, os.path.join(QA, "report.py"), "--gate",
                               os.path.join(d, "out", "summary.json")], capture_output=True, text=True)
        with open(os.path.join(d, "out", "pr-comment.md")) as f:
            md = f.read()
        with open(os.path.join(d, "out", "summary.md")) as f:
            self.summary = f.read()
        return md, gate.returncode

    def test_first_run_reports_without_failing(self):
        broken = result(spec(id="linux-x64-py3.13", python="3.13"), models=[model(status="fail", error="boom")])
        md, code = self.run_report([result(), broken])
        self.assertEqual(code, 0)
        self.assertIn("## ✅ Report only: there is no earlier run to compare with yet", md)
        self.assertIn("1 of 2 platform and Python combinations work fully", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ | ❌ |", md)
        self.assertIn("- Linux x64 on AMD EPYC 7763 py3.13: Nano: boom", md)

    def test_layout(self):
        s = spec(models=[NANO, TTS2])
        tts2 = model("KittenTTS 2", key="tts2", wav="audio/tts2.wav", checks=[
            {"name": "stream", "status": "pass"}, {"name": "clone", "status": "pass"}])
        md, code = self.run_report([result(s, models=[model(), tts2])], baseline=[])
        self.assertEqual(code, 0)
        self.assertIn("### KittenTTS 0.8 (ONNX models)", md)
        self.assertIn("### KittenTTS 2", md)
        self.assertIn("| Test | Linux x64<br>AMD EPYC 7763 |", md)
        self.assertIn("| Install and import | ✅ |", md)
        self.assertIn("| KittenTTS 2: streaming (`generate_stream`) | ✅ |", md)
        self.assertIn("| Nano RTF | 0.10 |", md)
        self.assertIn("| Peak RAM | 0.9 GB |", md)
        self.assertNotIn("### Every job", md)
        self.assertIn("### Every job", self.summary)

    def test_something_that_worked_on_main_and_breaks_fails_the_run(self):
        # A model that fails to generate has no generation time, as run_target.py records it.
        now = [result(models=[model(status="fail", first_s=None, error="ValueError: bad audio", trace="Traceback ...")])]
        md, code = self.run_report(now, baseline=[result()])
        self.assertEqual(code, 1)
        self.assertIn("## ❌ 1 test broke compared with main", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ❌ new |", md)
        self.assertIn("| Nano: speaks the sample text | ❌ new |", md)
        self.assertIn("**Linux x64 · py3.12** · Nano: speaks the sample text: ValueError: bad audio", md)

    def test_something_that_also_fails_on_main_is_listed_not_failed(self):
        no_torch = spec(id="macos-intel-py3.12", name="macOS Intel", runner="macos-15-intel")
        error = "ERROR: No matching distribution found for torch>=2.6"
        md, code = self.run_report([result(no_torch, ok=False, error=error)],
                                   baseline=[result(no_torch, ok=False, error=error)])
        self.assertEqual(code, 0)
        self.assertIn("## ✅ Nothing broke compared with main", md)
        self.assertIn("**Does not work** (on the baseline too, so it does not fail the run):", md)
        self.assertIn(f"- macOS Intel py3.12: install: {error}", md)

    def test_failure_on_a_cpu_main_never_drew_is_reported_not_failed(self):
        crash = {"key": "nano", "label": "Nano", "status": "crash", "error": "illegal CPU instruction"}
        md, code = self.run_report([result(models=[crash], cpu="INTEL(R) XEON(R) PLATINUM 8573C")],
                                   baseline=[result()])
        self.assertEqual(code, 0)
        self.assertIn("**On a CPU the baseline never drew**, so not counted as broken: "
                      "Linux x64 · py3.12 on Intel Xeon Platinum 8573C", md)

    def test_an_install_that_breaks_fails_even_on_a_new_cpu(self):
        md, code = self.run_report([result(ok=False, error="ERROR: bad dependency", cpu="Some new CPU")],
                                   baseline=[result()])
        self.assertEqual(code, 1)

    def test_something_that_starts_working_is_marked(self):
        md, code = self.run_report([result()], baseline=[result(ok=False, error="ERROR: no torch")])
        self.assertEqual(code, 0)
        self.assertIn("| Linux x64 | AMD EPYC 7763 | ✅ new |", md)
        self.assertIn("**Works now, did not before:** Linux x64 · py3.12", md)

    def test_missing_job_that_worked_on_main_fails(self):
        md, code = self.run_report([], baseline=[result()], planned=[spec()])
        self.assertEqual(code, 1)
        self.assertIn("| Linux x64 | ubuntu-24.04 | ❌ new |", md)

    def test_timeouts_show_a_timer_and_skip_the_rest(self):
        timed = {"key": "tts2", "label": "KittenTTS 2", "status": "timeout", "first_s": 30.0, "audio_s": 3.0,
                 "warm_s": [], "wav": "audio/tts2.wav", "error": "stream took longer than 10 min",
                 "checks": [{"name": "stream", "status": "timeout", "error": "took longer than 10 min"},
                            {"name": "clone", "status": "skipped", "error": "not run: an earlier step timed out"}]}
        md, code = self.run_report([result(spec(models=[TTS2]), models=[timed])])
        self.assertIn("| KittenTTS 2: speaks the sample text | ✅ |", md)
        self.assertIn("| KittenTTS 2: streaming (`generate_stream`) | ⏱️ |", md)
        self.assertIn("| KittenTTS 2: voice cloning with a transcript | ⏱️ |", md)

    def test_wer_failure_and_log_link(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.9, "transcript": "something else"}]
        jobs = [{"name": "Linux x64 · py3.12", "html_url": "https://example.test/1",
                 "started_at": "2026-10-05T10:00:00Z", "completed_at": "2026-10-05T10:45:00Z"}]
        md, code = self.run_report([result(asr_rows=rows)], baseline=[result()], jobs=jobs)
        self.assertEqual(code, 1)
        self.assertIn("Nano: speaks the sample text: WER 90.0%, Whisper heard “something else” · "
                      "[log](https://example.test/1)", md)
        self.assertIn("🐢 45 min", md)

    def test_logs_cannot_break_the_comment(self):
        crash = {"key": "nano", "label": "Nano", "status": "crash", "error": "exited", "log_tail": "```\nKilled"}
        md, code = self.run_report([result(models=[crash])], baseline=[result()])
        self.assertIn("````\n```\nKilled\n````", md)
        self.assertTrue(md.rstrip().endswith("</details>"))

    def test_comment_stays_under_github_limit(self):
        models = [{"key": f"m{j}", "label": f"M{j}", "repo": "KittenML/x", "checks": []} for j in range(5)]

        def job(i, **kw):
            return result(spec(id=f"p{i}", name=f"Platform {i}", models=models), **kw)
        now = [job(i, models=[model(label=f"M{j}", status="fail", first_s=None, error="x" * 300, trace="t" * 5000)
                              for j in range(5)]) for i in range(40)]
        base = [job(i, models=[model(label=f"M{j}") for j in range(5)]) for i in range(40)]
        md, code = self.run_report(now, baseline=base)
        self.assertLessEqual(len(md), report.COMMENT_LIMIT)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
