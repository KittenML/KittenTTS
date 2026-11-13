"""File generation honors the same text-cleaning choice as in-memory generation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from kittenml.kittentts_legacy.model import KittenTTSOnnx
from kittenml.kittentts_legacy.onnx_model import KittenTTS_1_Onnx
from kittenml.preprocess import TextPreprocessor


class OnnxFileOptionsTests(unittest.TestCase):
    def test_file_generation_can_disable_text_cleaning(self):
        for options, expected in (({}, "ten dollars,"),
                                  ({"clean_text": True}, "ten dollars,"),
                                  ({"clean_text": False}, "$10,")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                # Only inference is replaced; chunking, cleaning and WAV writing run normally.
                backend = KittenTTS_1_Onnx.__new__(KittenTTS_1_Onnx)
                backend.preprocessor = TextPreprocessor(remove_punctuation=False)
                backend.generate_single_chunk = Mock(return_value=np.zeros(240, dtype=np.float32))
                model = KittenTTSOnnx.__new__(KittenTTSOnnx)
                model.model = backend

                output = Path(directory) / "speech.wav"
                model.generate_to_file("$10", output, **options)

                backend.generate_single_chunk.assert_called_once_with(expected, "expr-voice-5-m", 1.0)
                self.assertEqual(sf.info(output).frames, 240)
                self.assertEqual(sf.info(output).samplerate, 24000)


if __name__ == "__main__":
    unittest.main()
