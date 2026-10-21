"""The voices and config files are UTF-8 and must be read as UTF-8 on every
platform.  On Windows the default text encoding is the ANSI code page (cp1252),
so a bare ``open()`` there fails on any non-ASCII byte (issue #158).  The tests
reproduce that by running the loader in a child interpreter whose default
encoding is ASCII (UTF-8 mode off, C locale, no locale coercion)."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NON_ASCII = "Zo\u00eb \u201cwarm\u201d \u2014 caf\u00e9"


def run_with_ascii_default_encoding(code):
    env = dict(os.environ, PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C",
               PYTHONPATH=str(REPO_ROOT))
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                          capture_output=True, text=True, encoding="utf-8")


class Utf8JsonLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_child_default_encoding_is_not_utf8(self):
        # Guard: if the child still reads UTF-8 by default, the tests below
        # would pass even with a bare open() and prove nothing.
        result = run_with_ascii_default_encoding(
            "import locale; print(locale.getpreferredencoding(False))")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("utf", result.stdout.lower().replace("-", ""))

    def test_load_voices_reads_non_ascii_voices_json(self):
        voices = self.root / "voices"
        voices.mkdir()
        entries = {"zoe": {"description": NON_ASCII, "reference": "zoe.wav",
                           "artifacts": "zoe.safetensors"}}
        (voices / "voices.json").write_text(json.dumps(entries, ensure_ascii=False),
                                            encoding="utf-8")
        result = run_with_ascii_default_encoding(f"""
            import pathlib, types
            from kittenml.kittentts2.model import KittenTTS2
            stub = types.SimpleNamespace(repo_dir=pathlib.Path({str(self.root)!r}), config={{}})
            voices = KittenTTS2._load_voices(stub)
            print(ascii(voices["zoe"]["description"]))
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), ascii(NON_ASCII))

    def test_resolve_repo_reads_non_ascii_local_config(self):
        (self.root / "config.json").write_text(json.dumps({"name": NON_ASCII}, ensure_ascii=False),
                                               encoding="utf-8")
        result = run_with_ascii_default_encoding(f"""
            from kittenml.repo import resolve_repo
            local_dir, repo_id, config = resolve_repo({str(self.root)!r})
            print(ascii(config["name"]))
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), ascii(NON_ASCII))


if __name__ == "__main__":
    unittest.main()
