"""The languages this effort targets, and the identifiers each tool needs for them.

Every tool in `indic/` keys off this table instead of hardcoding codes, because the
same language is named four different ways across the stack: FLORES+ uses
`hin_Deva`, IndicConformer uses `hi`, Whisper uses `hindi`, and the Unicode block
is what the script-detection in `text/indic_text.py` needs. Getting one of them
wrong is silent -- the eval still runs, it just scores the wrong thing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str           # ISO 639-1, our canonical key
    name: str
    flores: str         # FLORES+ / NLLB code, e.g. "hin_Deva"
    fleurs: str         # FLEURS code, e.g. "hi_in"
    whisper: str        # Whisper language name
    script: str         # Unicode script name
    block: tuple        # (start, end) codepoints of the script's main block
    danda: bool         # uses U+0964/U+0965 as sentence terminators


LANGUAGES = {
    lang.code: lang for lang in [
        Language("hi", "Hindi",   "hin_Deva", "hi_in", "hindi",   "Devanagari", (0x0900, 0x097F), True),
        Language("bn", "Bengali", "ben_Beng", "bn_in", "bengali", "Bengali",    (0x0980, 0x09FF), True),
        Language("ta", "Tamil",   "tam_Taml", "ta_in", "tamil",   "Tamil",      (0x0B80, 0x0BFF), False),
        Language("te", "Telugu",  "tel_Telu", "te_in", "telugu",  "Telugu",     (0x0C00, 0x0C7F), False),
        Language("mr", "Marathi", "mar_Deva", "mr_in", "marathi", "Devanagari", (0x0900, 0x097F), True),
        Language("en", "English", "eng_Latn", "en_us", "english", "Latin",      (0x0000, 0x024F), False),
    ]
}

# The five Indic targets, in the order every report should print them.
INDIC = ["hi", "bn", "ta", "te", "mr"]
ALL = INDIC + ["en"]
