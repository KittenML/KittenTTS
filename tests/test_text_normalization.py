import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kittentts import KittenTTS, NormalizedTextResult, load_from_local, normalize_text
from kittentts.preprocess import chunk_text


class TextNormalizationTests(unittest.TestCase):
    def test_issue_examples_normalize_for_tts(self):
        cases = {
            "Smith et al. 2024, pp. 31-35": "Smith et al twenty twenty-four pages thirty-one to thirty-five",
            "Fig. 2": "Figure two",
            "Dr. Rivera paid $12.50 at 3:05 p.m.": "Doctor Rivera paid twelve dollars and fifty cents at three oh five p m.",
            "Jan. 2026": "January twenty twenty-six",
            "version 2.4": "version two point four",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), expected)

    def test_normalizes_common_document_forms(self):
        cases = {
            "May 5, 2026": "May fifth, twenty twenty-six",
            "10:30 AM": "ten thirty a m",
            "$1,250.00": "one thousand two hundred fifty dollars",
            "9%": "nine percent",
            "v1.2.3": "v one point two point three",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), expected)

    def test_url_and_email_are_spoken_not_removed(self):
        normalized = normalize_text("Visit https://example.com or email hello@example.com.")

        self.assertIn("e x a m p l e dot c o m", normalized)
        self.assertIn("h e l l o at e x a m p l e dot c o m", normalized)

    def test_span_result_maps_replacements(self):
        result = normalize_text("Fig. 2", return_spans=True)

        self.assertIsInstance(result, NormalizedTextResult)
        self.assertEqual(result.text, "Figure two")
        self.assertEqual(
            [(span.originalStartChar, span.originalEndChar, span.reason) for span in result.spans],
            [(0, 4, "abbreviation"), (5, 6, "number")],
        )

    def test_chunking_does_not_split_common_abbreviations(self):
        self.assertEqual(
            chunk_text("Dr. Rivera paid $12.50 at 3:05 p.m."),
            ["Dr. Rivera paid $12.50 at 3:05 p.m."],
        )
        self.assertEqual(
            chunk_text("Smith et al. 2024, pp. 31-35"),
            ["Smith et al. 2024, pp. 31-35,"],
        )

    def test_unsupported_locale_fails_explicitly(self):
        with self.assertRaises(ValueError):
            normalize_text("Bonjour 2026", locale="fr-FR")


class LocalModelLoadingTests(unittest.TestCase):
    def _write_local_model(self, model_dir: Path):
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "type": "ONNX1",
                    "model_file": "model.onnx",
                    "voices": "voices.npz",
                    "speed_priors": {"Bella": 0.95},
                    "voice_aliases": {"Bella": "expr-voice-2-f"},
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "model.onnx").write_bytes(b"onnx")
        (model_dir / "voices.npz").write_bytes(b"voices")

    def test_load_from_local_uses_configured_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_local_model(model_dir)

            with patch("kittentts.get_model._create_onnx_model") as model_cls:
                model = load_from_local(model_dir, backend="cpu")

        self.assertIs(model, model_cls.return_value)
        model_cls.assert_called_once_with(
            model_path=str(model_dir / "model.onnx"),
            voices_path=str(model_dir / "voices.npz"),
            speed_priors={"Bella": 0.95},
            voice_aliases={"Bella": "expr-voice-2-f"},
            backend="cpu",
        )

    def test_kittentts_accepts_existing_local_model_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_local_model(model_dir)

            with patch("kittentts.get_model._create_onnx_model") as model_cls:
                model = KittenTTS(str(model_dir), backend="cpu")

        self.assertIs(model.model, model_cls.return_value)

    def test_load_from_local_requires_model_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"type": "ONNX1", "model_file": "missing.onnx", "voices": "voices.npz"}),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                load_from_local(model_dir)


if __name__ == "__main__":
    unittest.main()
