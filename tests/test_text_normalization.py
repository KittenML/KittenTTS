import unittest

from kittenml import NormalizedTextResult, normalize_text
from kittenml.preprocess import chunk_text


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

    def test_units_glued_to_numbers_are_expanded(self):
        cases = {
            "I ran 10km in 30min.": "I ran ten kilometers in thirty minutes",
            "Use 50% of 2.5GB for 300ms.": "Use fifty percent of two point five gigabytes for three hundred milliseconds",
            "The file is 500MB.": "The file is five hundred megabytes",
            "CPU at 2.8GHz and 25°C.": "CPU at two point eight gigahertz and twenty-five degrees Celsius",
            "It takes 5 sec.": "It takes five seconds",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw).rstrip("."), expected)

    def test_unit_expansion_leaves_unit_free_numbers_alone(self):
        cases = {
            "Python-3.10": "Python three point ten",
            "GPT-4": "GPT four",
            "It takes 5 minutes": "It takes five minutes",
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


if __name__ == "__main__":
    unittest.main()
