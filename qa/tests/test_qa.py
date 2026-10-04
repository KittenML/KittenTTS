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
         "models": [], "asr": ASR}
    s.update(kw)
    return s


def model(label="Nano", status="pass", **kw):
    m = {"key": label.lower(), "label": label, "status": status, "load_s": 1.0, "first_s": 1.2,
         "warm_s": [0.5, 0.4, 0.6], "audio_s": 4.0, "wav": f"audio/{label.lower()}.wav",
         "peak_rss_mb": 900, "checks": []}
    m.update(kw)
    return m


def result(s=None, ok=True, models=None, asr_rows=None, **install):
    inst = {"ok": ok, "refused": False, "secs": 90.0, "versions": {"kittenml": "0.9.3"}}
    inst.update(install)
    r = {"spec": s or spec(), "env": {"cpu": "AMD EPYC 7763", "cpu_count": 4, "ram_gb": 15.6},
         "install": inst, "package": {"checks": [{"name": "import", "status": "pass"}]},
         "models": models if models is not None else [model()]}
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

    def test_job_without_install_record_is_no_result(self):
        self.assertEqual(classify({"spec": spec()})[0], NO_RESULT)


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


class Report(unittest.TestCase):
    def run_report(self, results, planned=None):
        d = tempfile.mkdtemp()
        for i, r in enumerate(results):
            os.makedirs(os.path.join(d, "results", str(i)))
            with open(os.path.join(d, "results", str(i), "result.json"), "w") as f:
                json.dump(r, f)
        plan_path = os.path.join(d, "plan.json")
        with open(plan_path, "w") as f:
            json.dump({"jobs": planned or [r["spec"] for r in results]}, f)
        out = os.path.join(d, "out")
        subprocess.run([sys.executable, os.path.join(QA, "report.py"), os.path.join(d, "results"), out,
                        "--plan", plan_path], check=True, capture_output=True)
        gate = subprocess.run([sys.executable, os.path.join(QA, "report.py"), "--gate",
                               os.path.join(out, "summary.json")], capture_output=True, text=True)
        with open(os.path.join(out, "pr-comment.md")) as f:
            return f.read(), gate.returncode, gate.stdout

    def test_all_pass(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.0, "transcript": "Hello there."}]
        md, code, _ = self.run_report([result(asr_rows=rows)])
        self.assertEqual(code, 0)
        self.assertIn("✅ **All 1 supported platform jobs passed.**", md)
        self.assertIn("| Linux x64 · py3.12 |", md)
        self.assertIn("| 0.100 |", md)       # best warm run 0.4 s for 4 s of audio
        self.assertIn("AMD EPYC 7763 · 4 cores · 15.6 GB", md)

    def test_failure_and_missing_job_fail_the_gate(self):
        bad = result(spec(id="win-py3.12", name="Windows x64", runner="windows-2025"),
                     models=[model(status="fail", error="ValueError: bad audio", trace="Traceback...")])
        missing = spec(id="mac-py3.12", name="macOS", runner="macos-15")
        md, code, out = self.run_report([bad], planned=[bad["spec"], missing])
        self.assertEqual(code, 1)
        self.assertIn("❌ **2 platform jobs failed.**", md)
        self.assertIn("ValueError: bad audio", md)
        self.assertIn("macOS · py3.12: no result", md)
        self.assertIn("FAILED Windows x64 · py3.12", out)

    def test_wer_failure_marks_the_model_failed(self):
        rows = [{"wav": "audio/nano.wav", "label": "Nano", "wer": 0.9, "transcript": "something else"}]
        md, code, _ = self.run_report([result(asr_rows=rows)])
        self.assertEqual(code, 1)
        self.assertIn("❌ **1 platform job failed.**", md)
        self.assertIn("| Nano | Failed (WER) |", md)
        self.assertIn("| 0/1 |", md)

    def test_unsupported_platforms_are_listed_and_pass(self):
        s = spec(id="mac-intel", name="macOS Intel", runner="macos-15-intel", expect="install-fails",
                 reason="no torch wheels")
        md, code, _ = self.run_report([result(s, ok=False, error="ERROR: No matching distribution found for torch>=2.6")])
        self.assertEqual(code, 0)
        self.assertIn("## Not Supported (expected)", md)
        self.assertIn("no torch wheels", md)

    def test_comment_stays_under_github_limit(self):
        big = [result(spec(id=f"p{i}", name=f"Platform {i}"),
                      models=[model(label=f"M{j}", status="fail", error="x" * 300, trace="t" * 5000)
                              for j in range(5)]) for i in range(40)]
        md, code, _ = self.run_report(big)
        self.assertLessEqual(len(md), report.COMMENT_LIMIT + 200)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
