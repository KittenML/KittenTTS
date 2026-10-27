import unittest

from kittenml.kittentts2.model import KittenTTS2
from kittenml.kittentts2.text import split_for_synthesis

SENTENCE = "भारत एक विशाल और विविधताओं से भरा देश है।"


class HindiChunkingTests(unittest.TestCase):
    def test_long_text_splits_at_danda(self):
        chunks = split_for_synthesis(" ".join([SENTENCE] * 25))
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.endswith("है।"), chunk[-20:])

    def test_danda_is_already_terminal(self):
        chunks = split_for_synthesis(" ".join([SENTENCE] * 25))
        self.assertFalse(any(c.endswith(("।,", "।.")) for c in chunks))

    def test_devanagari_chunk_budget_fits_its_speech(self):
        # Hindi is spoken at about 10 characters a second; a 344-character chunk
        # runs about 34 s, i.e. 850 tokens at 25 Hz.
        chunk = "क" * 344
        budget = KittenTTS2._chunk_budget(None, chunk, 2000, single_chunk=False)
        self.assertGreaterEqual(budget, 850)

    def test_english_chunk_budget_unchanged(self):
        chunk = "a" * 344
        budget = KittenTTS2._chunk_budget(None, chunk, 2000, single_chunk=False)
        self.assertEqual(budget, int(344 / 20.0 * 25 * 1.8))


if __name__ == "__main__":
    unittest.main()
