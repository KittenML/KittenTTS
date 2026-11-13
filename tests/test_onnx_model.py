"""Tests for the legacy ONNX (KittenTTS 0.8) backend.

These follow the project's mock-first convention (see tests/test_vllm.py): the
ONNX session is replaced with a recorder, so no 56 MB model download is needed
and the tests run on CPU without espeak at inference time. A *fake* phonemizer
feeds a known, over-limit phoneme string so the token budget is exercised
deterministically.
"""

import unittest

import numpy as np

from kittenml.kittentts_legacy.onnx_model import KittenTTS_1_Onnx, TextCleaner


MAX_SEQ_LEN = 512


class _FakePhonemizer:
    """Stand-in for EspeakBackend that returns a fixed phoneme string."""

    def __init__(self, output):
        self._output = output

    def phonemize(self, texts):
        return [self._output for _ in texts]


class _RecordingSession:
    """Captures the input_ids length of every forward pass."""

    def __init__(self):
        self.token_lengths = []

    def run(self, output_names, inputs):
        self.token_lengths.append(int(inputs["input_ids"].shape[1]))
        # Waveform with the 5000-sample tail the model normally trims.
        return [np.zeros((1, 30000), dtype=np.float32)]


def _make_model(phoneme_output, max_seq_len=MAX_SEQ_LEN):
    """Build a KittenTTS_1_Onnx without loading model or voice files."""
    model = KittenTTS_1_Onnx.__new__(KittenTTS_1_Onnx)
    model.text_cleaner = TextCleaner()
    model.phonemizer = _FakePhonemizer(phoneme_output)
    model.max_seq_len = max_seq_len
    model.voice_aliases = {}
    model.speed_priors = {}
    model.available_voices = ["expr-voice-2-f"]
    model.voices = {"expr-voice-2-f": np.zeros((400, 256), dtype=np.float32)}
    model.session = _RecordingSession()
    return model


# A phoneme string with far more tokens than the 512-position BERT can hold.
# "ab " repeated yields two mappable chars plus a space per word.
_OVER_LIMIT_PHONEMES = "ab " * 400


class OnnxMaxSeqLenTests(unittest.TestCase):
    def _count_tokens(self, text):
        model = _make_model(_OVER_LIMIT_PHONEMES)
        return model._prepare_inputs(text, "expr-voice-2-f")["input_ids"].shape[1]

    def test_over_limit_input_is_detected(self):
        # Guards the test: the fixture really does exceed the model limit, so the
        # assertions below are meaningful rather than vacuously true.
        self.assertGreater(self._count_tokens("anything"), MAX_SEQ_LEN)

    def test_long_input_is_split_under_max_seq_len(self):
        model = _make_model(_OVER_LIMIT_PHONEMES)
        text = "a long sentence of many short words without any commas or periods " * 10

        audio = model.generate_single_chunk(text, voice="expr-voice-2-f")

        self.assertTrue(model.session.token_lengths, "session.run was never called")
        too_long = [n for n in model.session.token_lengths if n > MAX_SEQ_LEN]
        self.assertEqual(too_long, [], f"forward pass exceeded {MAX_SEQ_LEN}: {model.session.token_lengths}")
        # An over-limit input must actually be split into several passes.
        self.assertGreater(len(model.session.token_lengths), 1)
        self.assertGreater(len(audio), 0)

    def test_short_input_is_a_single_run(self):
        model = _make_model("hello world")
        audio = model.generate_single_chunk("hi there", voice="expr-voice-2-f")

        self.assertEqual(len(model.session.token_lengths), 1)
        self.assertLessEqual(model.session.token_lengths[0], MAX_SEQ_LEN)
        self.assertGreater(len(audio), 0)

    def test_generate_does_not_feed_over_limit_chunks(self):
        model = _make_model(_OVER_LIMIT_PHONEMES)
        text = "cat dog hat run sit big red sun " * 40  # long run-on, clean_text off

        audio = model.generate(text, voice="expr-voice-2-f", clean_text=False)

        too_long = [n for n in model.session.token_lengths if n > MAX_SEQ_LEN]
        self.assertEqual(too_long, [], f"forward pass exceeded {MAX_SEQ_LEN}: {model.session.token_lengths}")
        self.assertTrue(model.session.token_lengths)
        self.assertGreater(len(audio), 0)

    def test_split_phonemes_respects_token_budget(self):
        model = _make_model(_OVER_LIMIT_PHONEMES)
        cleaner = model.text_cleaner

        pieces = model._split_phonemes(_OVER_LIMIT_PHONEMES)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            # +3 for the 0 / 10 / 0 special tokens the model prepends/appends.
            tokens = len(cleaner(piece)) + 3
            self.assertLessEqual(tokens, MAX_SEQ_LEN, f"piece too long: {tokens}")


if __name__ == "__main__":
    unittest.main()