"""
text_preprocessing.py
A comprehensive text preprocessing library for NLP pipelines.
"""

import re
import unicodedata
from typing import Optional


# ─────────────────────────────────────────────
# Number → Words conversion
# ─────────────────────────────────────────────

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALE = ["", "thousand", "million", "billion", "trillion"]

_ORDINAL_EXCEPTIONS = {
    "one": "first", "two": "second", "three": "third", "four": "fourth",
    "five": "fifth", "six": "sixth", "seven": "seventh", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}

_CURRENCY_SYMBOLS = {
    "$": "dollar", "€": "euro", "£": "pound", "¥": "yen",
    "₹": "rupee", "₩": "won", "₿": "bitcoin",
}

_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"),  (90, "XC"),  (50, "L"),  (40, "XL"),
    (10, "X"),   (9, "IX"),   (5, "V"),   (4, "IV"), (1, "I"),
]
_RE_ROMAN = re.compile(
    r"\b(M{0,4})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b"
)


def _three_digits_to_words(n: int) -> str:
    """Convert a number 0–999 to English words."""
    if n == 0:
        return ""
    parts = []
    hundreds = n // 100
    remainder = n % 100
    if hundreds:
        parts.append(f"{_ONES[hundreds]} hundred")
    if remainder < 20:
        if remainder:
            parts.append(_ONES[remainder])
    else:
        tens_word = _TENS[remainder // 10]
        ones_word = _ONES[remainder % 10]
        parts.append(f"{tens_word}-{ones_word}" if ones_word else tens_word)
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """
    Convert an integer to its English word representation.

    Examples:
        1200      → "twelve hundred"
        1000      → "one thousand"
        1_000_000 → "one million"
        -42       → "negative forty-two"
        0         → "zero"
    """
    if not isinstance(n, int):
        n = int(n)
    if n == 0:
        return "zero"
    if n < 0:
        return f"negative {number_to_words(-n)}"

    # X00–X999 read as "X hundred" (e.g. 1200 → "twelve hundred")
    # Exclude exact multiples of 1000 (1000 → "one thousand", not "ten hundred")
    if 100 <= n <= 9999 and n % 100 == 0 and n % 1000 != 0:
        hundreds = n // 100
        if hundreds < 20:
            return f"{_ONES[hundreds]} hundred"

    parts = []
    for i, scale in enumerate(_SCALE):
        chunk = n % 1000
        if chunk:
            chunk_words = _three_digits_to_words(chunk)
            parts.append(f"{chunk_words} {scale}".strip() if scale else chunk_words)
        n //= 1000
        if n == 0:
            break

    return " ".join(reversed(parts))


def float_to_words(value, decimal_sep: str = "point") -> str:
    """
    Convert a float (or numeric string) to words, reading decimal digits individually.
    Accepts a string to preserve trailing zeros (e.g. "1.50" → "one point five zero").

    Examples:
        3.14   → "three point one four"
        -0.5   → "negative zero point five"
        "3.10" → "three point one zero"
        1.007  → "one point zero zero seven"
    """
    text = value if isinstance(value, str) else f"{value}"
    negative = text.startswith("-")
    if negative:
        text = text[1:]

    if "." in text:
        int_part, dec_part = text.split(".", 1)
        int_words = number_to_words(int(int_part)) if int_part else "zero"
        # Read each decimal digit individually; "0" → "zero"
        digit_map = ["zero"] + _ONES[1:]  # index 0 → "zero"
        dec_words = " ".join(digit_map[int(d)] for d in dec_part)
        result = f"{int_words} {decimal_sep} {dec_words}"
    else:
        result = number_to_words(int(text))

    return f"negative {result}" if negative else result


def roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to an integer."""
    val = {"I": 1, "V": 5, "X": 10, "L": 50,
           "C": 100, "D": 500, "M": 1000}
    result = 0
    prev = 0
    for ch in reversed(s.upper()):
        curr = val[ch]
        result += curr if curr >= prev else -curr
        prev = curr
    return result


# ─────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────

_RE_URL      = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL    = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
_RE_HASHTAG  = re.compile(r"#\w+")
_RE_MENTION  = re.compile(r"@\w+")
_RE_HTML     = re.compile(r"<[^>]+>")
_RE_PUNCT    = re.compile(r"[^\w\s]")
_RE_SPACES   = re.compile(r"\s+")

# Number: do NOT match a leading minus if it is immediately preceded by a letter
# (handles "gpt-3", "gpl-3", "v-2" etc.)
_RE_NUMBER   = re.compile(r"(?<![a-zA-Z])-?[\d,]+(?:\.\d+)?")

# Ordinals: 1st, 2nd, 3rd, 4th … 21st, 101st …
_RE_ORDINAL  = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# Percentages: 50%, 3.5%
_RE_PERCENT  = re.compile(r"(-?[\d,]+(?:\.\d+)?)\s*%")

# Currency: $100, €1,200.50, £50, $85K, $2.5M (optional scale suffix)
_RE_CURRENCY = re.compile(r"([$€£¥₹₩₿])\s*([\d,]+(?:\.\d+)?)\s*([KMBT])?(?![a-zA-Z\d])")

# Time: 3:30pm, 14:00, 3:30 AM — requires 2-digit minutes so "3:0" (score) doesn't match
_RE_TIME     = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)

# Ranges: 10-20, 100-200 (both sides numeric, hyphen between them)
_RE_RANGE    = re.compile(r"(?<!\w)(\d+)-(\d+)(?!\w)")

# Version/model names: gpt-3, gpt-3.5, v2.0, Python-3.10, GPL-3
# Letter(s) + hyphen + digit(s) [+ more version parts]
_RE_MODEL_VER = re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*)-(\d[\d.]*)(?=[^\d.]|$)")

# Measurement units glued to numbers: 100km, 50kg, 25°C, 5GB
_RE_UNIT     = re.compile(r"(\d+(?:\.\d+)?)\s*(km|kg|mg|ml|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|°[cCfF]|[cCfF]°|ms|ns|µs)\b",
                          re.IGNORECASE)

# Scale suffixes (uppercase only to avoid ambiguity): 7B, 340M, 1.5K, 2T
# Must NOT be preceded by a letter (so 'MB' is handled by unit regex first)
_RE_SCALE    = re.compile(r"(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*([KMBT])(?![a-zA-Z\d])")

# Scientific notation: 1e-4, 2.5e10, 6.022E23
_RE_SCI      = re.compile(r"(?<![a-zA-Z\d])(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![a-zA-Z\d])")

# Fractions: 1/2, 3/4, 2/3
_RE_FRACTION = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")

# Decades: 80s, 90s, 1980s, 2020s (number ending in 0 followed by 's')
_RE_DECADE   = re.compile(r"\b(\d{1,3})0s\b")

# Leading decimal (no digit before the dot): .5, .75
_RE_LEAD_DEC = re.compile(r"(?<!\d)\.([\d])")


# ─────────────────────────────────────────────
# Expansion helpers
# ─────────────────────────────────────────────

def _ordinal_suffix(n: int) -> str:
    """Return the ordinal word for n (e.g. 1 → 'first', 5 → 'fifth', 21 → 'twenty-first')."""
    word = number_to_words(n)
    # For hyphenated compounds like "twenty-one", convert only the last part
    if "-" in word:
        prefix, last = word.rsplit("-", 1)
        joiner = "-"
    else:
        parts = word.rsplit(" ", 1)
        prefix, last, joiner = (parts[0], parts[1], " ") if len(parts) == 2 else ("", parts[0], "")

    # Check exception table
    for base, ordinal in _ORDINAL_EXCEPTIONS.items():
        if last == base:
            last_ord = ordinal
            break
    else:
        # General rule
        if last.endswith("t"):
            last_ord = last + "h"
        elif last.endswith("e"):
            last_ord = last[:-1] + "th"
        else:
            last_ord = last + "th"

    return f"{prefix}{joiner}{last_ord}" if prefix else last_ord


def expand_ordinals(text: str) -> str:
    """
    Convert ordinal numbers to words.

    Examples:
        "1st place"  → "first place"
        "2nd floor"  → "second floor"
        "3rd base"   → "third base"
        "21st century" → "twenty-first century"
        "100th day"  → "one hundredth day"
    """
    def _replace(m: re.Match) -> str:
        return _ordinal_suffix(int(m.group(1)))
    return _RE_ORDINAL.sub(_replace, text)


def expand_percentages(text: str) -> str:
    """
    Expand percentage expressions.

    Examples:
        "50% off"    → "fifty percent off"
        "3.5% rate"  → "three point five percent rate"
        "-2% change" → "negative two percent change"
    """
    def _replace(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        if "." in raw:
            return float_to_words(float(raw)) + " percent"
        return number_to_words(int(raw)) + " percent"
    return _RE_PERCENT.sub(_replace, text)


def expand_currency(text: str) -> str:
    """
    Expand currency amounts, including optional scale suffixes.

    Examples:
        "$100"      → "one hundred dollars"
        "€1,200.50" → "twelve hundred euros and fifty cents"
        "£9.99"     → "nine pounds and ninety-nine cents"
        "$85K"      → "eighty five thousand dollars"
        "$2.5M"     → "two point five million dollars"
    """
    _scale_map = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}

    def _replace(m: re.Match) -> str:
        symbol = m.group(1)
        raw = m.group(2).replace(",", "")
        scale_suffix = m.group(3)          # e.g. "K", "M", or None
        unit = _CURRENCY_SYMBOLS.get(symbol, "")

        if scale_suffix:
            # e.g. $85K → "eighty five thousand dollars"
            scale_word = _scale_map[scale_suffix]
            num = float_to_words(raw) if "." in raw else number_to_words(int(raw))
            return f"{num} {scale_word} {unit}{'s' if unit else ''}".strip()

        if "." in raw:
            int_part, dec_part = raw.split(".", 1)
            dec_val = int(dec_part[:2].ljust(2, "0"))
            int_words = number_to_words(int(int_part))
            result = f"{int_words} {unit}s" if unit else int_words
            if dec_val:
                cents = number_to_words(dec_val)
                result += f" and {cents} cent{'s' if dec_val != 1 else ''}"
        else:
            val = int(raw)
            words = number_to_words(val)
            result = f"{words} {unit}{'s' if val != 1 and unit else ''}" if unit else words
        return result

    return _RE_CURRENCY.sub(_replace, text)


def expand_time(text: str) -> str:
    """
    Expand time expressions.

    Examples:
        "3:30pm"  → "three thirty pm"
        "14:00"   → "fourteen hundred"
        "9:05 AM" → "nine oh five am"
        "12:00pm" → "twelve pm"
    """
    def _replace(m: re.Match) -> str:
        h = int(m.group(1))
        mins = int(m.group(2))
        suffix = (" " + m.group(4).lower()) if m.group(4) else ""
        h_words = number_to_words(h)
        if mins == 0:
            return f"{h_words} hundred{suffix}" if not m.group(4) else f"{h_words}{suffix}"
        elif mins < 10:
            return f"{h_words} oh {number_to_words(mins)}{suffix}"
        else:
            return f"{h_words} {number_to_words(mins)}{suffix}"
    return _RE_TIME.sub(_replace, text)


def expand_ranges(text: str) -> str:
    """
    Expand numeric ranges.

    Examples:
        "10-20 items"   → "ten to twenty items"
        "pages 100-200" → "pages one hundred to two hundred"
        "2020-2024"     → "twenty twenty to twenty twenty-four"
    """
    def _replace(m: re.Match) -> str:
        lo = number_to_words(int(m.group(1)))
        hi = number_to_words(int(m.group(2)))
        return f"{lo} to {hi}"
    return _RE_RANGE.sub(_replace, text)


def expand_model_names(text: str) -> str:
    """
    Normalise version/model names that use letter-hyphen-number patterns,
    so the number is not misread as negative.

    Examples:
        "GPT-3"      → "GPT 3"
        "gpt-3.5"    → "gpt 3.5"
        "GPL-3"      → "GPL 3"
        "Python-3.10"→ "Python 3.10"
        "v2.0"       stays as "v2.0" (no hyphen — handled by number replacement)
        "IPv6"       stays as "IPv6"
    """
    return _RE_MODEL_VER.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)


def expand_units(text: str) -> str:
    """
    Expand common measurement units glued to numbers.

    Examples:
        "100km"  → "one hundred kilometers"
        "50kg"   → "fifty kilograms"
        "25°C"   → "twenty-five degrees Celsius"
        "5GB"    → "five gigabytes"
    """
    _unit_map = {
        "km": "kilometers", "kg": "kilograms", "mg": "milligrams",
        "ml": "milliliters", "gb": "gigabytes", "mb": "megabytes",
        "kb": "kilobytes", "tb": "terabytes",
        "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
        "mph": "miles per hour", "kph": "kilometers per hour",
        "ms": "milliseconds", "ns": "nanoseconds", "µs": "microseconds",
        "°c": "degrees Celsius", "c°": "degrees Celsius",
        "°f": "degrees Fahrenheit", "f°": "degrees Fahrenheit",
    }
    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        unit = m.group(2).lower()
        expanded = _unit_map.get(unit, m.group(2))
        num = float_to_words(float(raw)) if "." in raw else number_to_words(int(raw))
        return f"{num} {expanded}"
    return _RE_UNIT.sub(_replace, text)


def expand_roman_numerals(text: str, context_words: bool = True) -> str:
    """
    Expand Roman numerals that appear as standalone tokens (optionally
    only when preceded by a title-like word to avoid false positives).

    Examples:
        "World War II"     → "World War two"
        "Chapter IV"       → "Chapter four"
        "Louis XIV"        → "Louis fourteen"
        "mix I with V"     → left unchanged (ambiguous single letters)
    """
    _TITLE_WORDS = re.compile(
        r"\b(war|chapter|part|volume|act|scene|book|section|article|"
        r"king|queen|pope|louis|henry|edward|george|william|james|"
        r"phase|round|level|stage|class|type|version|episode|season)\b",
        re.IGNORECASE,
    )

    def _replace(m: re.Match) -> str:
        roman = m.group(0)
        if not roman.strip():
            return roman
        # Skip single ambiguous letters (I, V, X) unless context present
        if len(roman) == 1 and roman in "IVX":
            # Only expand if preceded by a title word
            start = m.start()
            preceding = text[max(0, start - 30): start]
            if not _TITLE_WORDS.search(preceding):
                return roman
        try:
            val = roman_to_int(roman)
            if val == 0:
                return roman
            return number_to_words(val)
        except Exception:
            return roman

    return _RE_ROMAN.sub(_replace, text)


def normalize_leading_decimals(text: str) -> str:
    """
    Normalise bare leading-decimal floats so the number pipeline handles them.

    Examples:
        ".5 teaspoons" → "0.5 teaspoons"
        "-.25 adjustment" → "-0.25 adjustment"
    """
    # Handle -.5 → -0.5 and .5 → 0.5
    text = re.sub(r"(?<!\d)(-)\.([\d])", r"\g<1>0.\2", text)
    return _RE_LEAD_DEC.sub(r"0.\1", text)


def expand_scientific_notation(text: str) -> str:
    """
    Expand scientific-notation numbers to spoken form.

    Examples:
        "1e-4"    → "one times ten to the negative four"
        "2.5e10"  → "two point five times ten to the ten"
        "6.022E23"→ "six point zero two two times ten to the twenty three"
    """
    def _replace(m: re.Match) -> str:
        coeff_raw = m.group(1)
        exp = int(m.group(2))
        coeff_words = float_to_words(coeff_raw) if "." in coeff_raw else number_to_words(int(coeff_raw))
        exp_words = number_to_words(abs(exp))
        sign = "negative " if exp < 0 else ""
        return f"{coeff_words} times ten to the {sign}{exp_words}"
    return _RE_SCI.sub(_replace, text)


def expand_scale_suffixes(text: str) -> str:
    """
    Expand standalone uppercase scale suffixes attached to numbers.

    Examples:
        "7B parameters" → "seven billion parameters"
        "340M model"    → "three hundred forty million model"
        "1.5K salary"   → "one point five thousand salary"
        "$100K budget"  → "$100K budget"  (currency handled upstream)
    """
    _map = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        suffix = m.group(2)
        scale_word = _map.get(suffix, suffix)
        num = float_to_words(raw) if "." in raw else number_to_words(int(raw))
        return f"{num} {scale_word}"

    return _RE_SCALE.sub(_replace, text)


def expand_fractions(text: str) -> str:
    """
    Expand simple numeric fractions.

    Examples:
        "1/2 cup"  → "one half cup"
        "3/4 mile" → "three quarters mile"
        "2/3 done" → "two thirds done"
        "5/8 inch" → "five eighths inch"
    """
    def _replace(m: re.Match) -> str:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return m.group()
        num_words = number_to_words(num)
        if den == 2:
            denom_word = "half" if num == 1 else "halves"
        elif den == 4:
            denom_word = "quarter" if num == 1 else "quarters"
        else:
            denom_word = _ordinal_suffix(den)
            if num != 1:
                denom_word += "s"
        return f"{num_words} {denom_word}"

    return _RE_FRACTION.sub(_replace, text)


def expand_decades(text: str) -> str:
    """
    Expand decade expressions to words.

    Examples:
        "the 80s"    → "the eighties"
        "the 1980s"  → "the nineteen eighties"
        "the 2020s"  → "the twenty twenties"
        "'90s music" → "nineties music"
    """
    _decade_map = {
        0: "hundreds", 1: "tens", 2: "twenties", 3: "thirties", 4: "forties",
        5: "fifties", 6: "sixties", 7: "seventies", 8: "eighties", 9: "nineties",
    }

    def _replace(m: re.Match) -> str:
        base = int(m.group(1))          # e.g. 8 for "80s", 198 for "1980s"
        decade_digit = base % 10
        decade_word = _decade_map.get(decade_digit, "")
        if base < 10:
            return decade_word
        century_part = base // 10       # e.g. 19 for 198
        return f"{number_to_words(century_part)} {decade_word}"

    return _RE_DECADE.sub(_replace, text)


def expand_ip_addresses(text: str) -> str:
    """
    Expand IPv4 addresses to spoken digits per octet.

    Examples:
        "192.168.1.1"  → "one nine two dot one six eight dot one dot one"
        "10.0.0.1"     → "one zero dot zero dot zero dot one"
    """
    _d = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
          "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

    def _octet(s: str) -> str:
        return " ".join(_d[c] for c in s)

    def _replace(m: re.Match) -> str:
        return " dot ".join(_octet(g) for g in m.groups())

    return re.sub(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b", _replace, text)


def expand_phone_numbers(text: str) -> str:
    """
    Expand US phone numbers to spoken digits before range expansion claims the hyphens.

    Examples:
        "555-1234"       → "five five five one two three four"
        "555-123-4567"   → "five five five one two three four five six seven"
        "1-800-555-0199" → "one eight zero zero five five five zero one nine nine"
    """
    _d = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
          "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

    def _digits(s: str) -> str:
        return " ".join(_d[c] for c in s)

    def _join(*groups) -> str:
        return " ".join(_digits(g) for g in groups)

    # Match longest pattern first to avoid partial matches
    # 11-digit: 1-800-555-0199
    text = re.sub(r"(?<!\d-)(?<!\d)\b(\d{1,2})-(\d{3})-(\d{3})-(\d{4})\b(?!-\d)",
                  lambda m: _join(*m.groups()), text)
    # 10-digit: 555-123-4567
    text = re.sub(r"(?<!\d-)(?<!\d)\b(\d{3})-(\d{3})-(\d{4})\b(?!-\d)",
                  lambda m: _join(*m.groups()), text)
    # 7-digit local: 555-1234 (not preceded or followed by digit-hyphen to avoid sub-matching)
    text = re.sub(r"(?<!\d-)\b(\d{3})-(\d{4})\b(?!-\d)",
                  lambda m: _join(*m.groups()), text)
    return text


# ─────────────────────────────────────────────
# Core preprocessing functions
# ─────────────────────────────────────────────

def replace_numbers(text: str, replace_floats: bool = True) -> str:
    """
    Replace all numeric tokens with their word equivalents.

    Examples:
        "There are 1200 students" → "There are twelve hundred students"
        "Pi is 3.14"              → "Pi is three point one four"
        "gpt-3 rocks"             → "gpt-3 rocks"  (hyphen not treated as minus)
    """
    def _replace(m: re.Match) -> str:
        raw = m.group().replace(",", "")
        try:
            if "." in raw and replace_floats:
                # Pass raw string so trailing zeros are preserved ("1.50" → "one point five zero")
                return float_to_words(raw)
            else:
                return number_to_words(int(float(raw)))
        except (ValueError, OverflowError):
            return m.group()
    return _RE_NUMBER.sub(_replace, text)


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_urls(text: str, replacement: str = "") -> str:
    """Remove URLs from text."""
    return _RE_URL.sub(replacement, text).strip()


def remove_emails(text: str, replacement: str = "") -> str:
    """Remove email addresses from text."""
    return _RE_EMAIL.sub(replacement, text).strip()


def remove_html_tags(text: str) -> str:
    """Strip HTML tags from text."""
    return _RE_HTML.sub(" ", text)


def remove_hashtags(text: str, replacement: str = "") -> str:
    """Remove hashtags (e.g. #NLP) from text."""
    return _RE_HASHTAG.sub(replacement, text)


def remove_mentions(text: str, replacement: str = "") -> str:
    """Remove @mentions from text."""
    return _RE_MENTION.sub(replacement, text)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return _RE_PUNCT.sub(" ", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip ends."""
    return _RE_SPACES.sub(" ", text).strip()


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize unicode characters (NFC, NFD, NFKC, or NFKD)."""
    return unicodedata.normalize(form, text)


def remove_accents(text: str) -> str:
    """Remove diacritical marks (accents) from characters."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions.

    Examples:
        "don't"   → "do not"
        "they're" → "they are"
        "I've"    → "I have"
    """
    contractions = {
        r"\bcan't\b":   "cannot",
        r"\bwon't\b":   "will not",
        r"\bshan't\b":  "shall not",
        r"\bain't\b":   "is not",
        r"\blet's\b":   "let us",
        r"\b(\w+)n't\b": r"\1 not",
        r"\b(\w+)'re\b": r"\1 are",
        r"\b(\w+)'ve\b": r"\1 have",
        r"\b(\w+)'ll\b": r"\1 will",
        r"\b(\w+)'d\b":  r"\1 would",
        r"\b(\w+)'m\b":  r"\1 am",
        r"\bit's\b":    "it is",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_stopwords(text: str, stopwords: Optional[set] = None) -> str:
    """
    Remove stopwords from text.

    Args:
        stopwords: Set of words to remove. Uses a built-in English set if None.
    """
    if stopwords is None:
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "this", "that",
            "these", "those", "it", "its", "i", "me", "my", "we", "our",
            "you", "your", "he", "she", "him", "her", "they", "them", "their",
        }
    tokens = text.split()
    return " ".join(t for t in tokens if t.lower() not in stopwords)


# ─────────────────────────────────────────────
# Pipeline helper
# ─────────────────────────────────────────────

class TextPreprocessor:
    """
    Configurable preprocessing pipeline.

    Usage:
        pp = TextPreprocessor(
            lowercase=True,
            replace_numbers=True,
            remove_urls=True,
            remove_html=True,
            remove_punctuation=True,
        )
        clean = pp("GPT-3 costs $0.002 per token — 50% cheaper than before!")
        # → "gpt three costs zero dollars and zero point two cents per token fifty percent cheaper than before"
    """

    def __init__(
        self,
        lowercase: bool = True,
        replace_numbers: bool = True,
        replace_floats: bool = True,
        expand_contractions: bool = True,
        expand_model_names: bool = True,
        expand_ordinals: bool = True,
        expand_percentages: bool = True,
        expand_currency: bool = True,
        expand_time: bool = True,
        expand_ranges: bool = True,
        expand_units: bool = True,
        expand_scale_suffixes: bool = True,
        expand_scientific_notation: bool = True,
        expand_fractions: bool = True,
        expand_decades: bool = True,
        expand_phone_numbers: bool = True,
        expand_ip_addresses: bool = True,
        normalize_leading_decimals: bool = True,
        expand_roman_numerals: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_html: bool = True,
        remove_hashtags: bool = False,
        remove_mentions: bool = False,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        stopwords: Optional[set] = None,
        normalize_unicode: bool = True,
        remove_accents: bool = False,
        remove_extra_whitespace: bool = True,
    ):
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self._stopwords = stopwords

    def __call__(self, text: str) -> str:
        return self.process(text)

    def process(self, text: str) -> str:
        cfg = self.config

        if cfg["normalize_unicode"]:
            text = normalize_unicode(text)
        if cfg["remove_html"]:
            text = remove_html_tags(text)
        if cfg["remove_urls"]:
            text = remove_urls(text)
        if cfg["remove_emails"]:
            text = remove_emails(text)
        if cfg["remove_hashtags"]:
            text = remove_hashtags(text)
        if cfg["remove_mentions"]:
            text = remove_mentions(text)
        if cfg["expand_contractions"]:
            text = expand_contractions(text)
        # IP addresses before normalize_leading_decimals (IPs contain dots before digits)
        if cfg["expand_ip_addresses"]:
            text = expand_ip_addresses(text)
        # Normalise bare leading decimals early so downstream regexes see "0.5" not ".5"
        if cfg["normalize_leading_decimals"]:
            text = normalize_leading_decimals(text)
        # Expand special forms before generic number replacement
        if cfg["expand_currency"]:
            text = expand_currency(text)
        if cfg["expand_percentages"]:
            text = expand_percentages(text)
        # Scientific notation before model-name expansion (e.g. "1e-4" contains "e-4")
        if cfg["expand_scientific_notation"]:
            text = expand_scientific_notation(text)
        if cfg["expand_time"]:
            text = expand_time(text)
        if cfg["expand_ordinals"]:
            text = expand_ordinals(text)
        if cfg["expand_units"]:
            text = expand_units(text)
        # Scale suffixes after units (units handles "MB"/"GB"; this handles bare "B"/"M")
        if cfg["expand_scale_suffixes"]:
            text = expand_scale_suffixes(text)
        if cfg["expand_fractions"]:
            text = expand_fractions(text)
        if cfg["expand_decades"]:
            text = expand_decades(text)
        # Phone numbers before ranges, otherwise NNN-NNNN is treated as a range
        if cfg["expand_phone_numbers"]:
            text = expand_phone_numbers(text)
        if cfg["expand_ranges"]:
            text = expand_ranges(text)
        if cfg["expand_model_names"]:
            text = expand_model_names(text)
        if cfg["expand_roman_numerals"]:
            text = expand_roman_numerals(text)
        if cfg["replace_numbers"]:
            text = replace_numbers(text, replace_floats=cfg["replace_floats"])
        if cfg["remove_accents"]:
            text = remove_accents(text)
        if cfg["remove_punctuation"]:
            text = remove_punctuation(text)
        if cfg["lowercase"]:
            text = to_lowercase(text)
        if cfg["remove_stopwords"]:
            text = remove_stopwords(text, self._stopwords)
        if cfg["remove_extra_whitespace"]:
            text = remove_extra_whitespace(text)

        return text


# ─────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pp = TextPreprocessor()

    cases = [
        # ── Numbers ────────────────────────────────────────────────────
        ("Plain integer",              "There are 1200 students and 42 teachers."),
        ("Large number",               "The project costs $1,000,000 and took 365 days."),
        ("Negative number",            "Temperature dropped to -5 degrees overnight."),
        ("Float",                      "Pi is approximately 3.14159."),
        ("Float trailing zero",        "The voltage is 1.50 volts."),
        ("Leading decimal",            "Add .5 teaspoons of salt and .25 cup of milk."),
        ("Negative leading decimal",   "A -.05 correction was applied."),
        ("Zero",                       "There were 0 errors and 0.0 warnings."),
        ("Comma thousands",            "The population is 7,900,000,000."),
        # ── Scientific notation ─────────────────────────────────────────
        ("Scientific e-notation",      "Learning rate is 1e-4, weight decay 1e-5."),
        ("Scientific capital E",       "Avogadro's number is 6.022E23."),
        ("Scientific large exp",       "The signal is 2.5e10 Hz."),
        # ── Scale suffixes ─────────────────────────────────────────────
        ("Model params B",             "We trained a 7B parameter model and a 13B variant."),
        ("Model params M",             "The 340M model beat the 7B on MMLU."),
        ("Scale suffix K",             "The salary was $85K per year."),
        # ── Currency ───────────────────────────────────────────────────
        ("Dollar amount",              "A coffee costs $4.99 here."),
        ("Euro amount",                "Rent is €1,200 per month."),
        ("Pound with cents",           "The book is £9.99."),
        # ── Percentages ────────────────────────────────────────────────
        ("Percentage",                 "Inflation rose by 3.5% last quarter."),
        ("Negative percentage",        "Stocks fell -2% today."),
        # ── Ordinals ───────────────────────────────────────────────────
        ("Ordinals 1st/2nd/3rd",       "She finished 1st, he came 2nd, I was 3rd."),
        ("Ordinal 21st",               "It's the 21st century and the 100th anniversary."),
        ("Ordinal 42nd",               "He ran his 42nd marathon."),
        ("Ordinal 33rd",               "On the 33rd floor."),
        # ── Fractions ──────────────────────────────────────────────────
        ("Half",                       "Cut the recipe in 1/2."),
        ("Quarters",                   "Add 3/4 cup of sugar and 1/4 teaspoon of salt."),
        ("Thirds",                     "The team completed 2/3 of the project."),
        ("Eighths",                    "The pipe is 5/8 inch in diameter."),
        # ── Time ───────────────────────────────────────────────────────
        ("12-hour time",               "The meeting starts at 3:30pm."),
        ("24-hour time",               "Departure at 14:00."),
        ("Time with oh",               "Alarm set for 9:05 AM."),
        ("Midnight",                   "The server restarts at 0:00."),
        # ── Decades ────────────────────────────────────────────────────
        ("Bare decade",                "The 80s music scene was iconic."),
        ("Full decade",                "She grew up listening to 1990s grunge."),
        ("2000s",                      "The 2000s brought social media."),
        ("2020s",                      "AI took off in the 2020s."),
        ("Apostrophe decade",          "Born in the '90s, raised on 2000s pop."),
        # ── Ranges ─────────────────────────────────────────────────────
        ("Numeric range",              "Read pages 10-20 for homework."),
        ("Year range",                 "The war lasted from 2020-2024."),
        ("Temperature range",          "Store between 5-10 degrees."),
        # ── Model / version names ───────────────────────────────────────
        ("GPT-3",                      "gpt-3 is pretty sick."),
        ("GPT-3.5",                    "They upgraded to GPT-3.5 last month."),
        ("GPL-3 license",              "This project is licensed under GPL-3."),
        ("Python version",             "Requires Python-3.10 or higher."),
        ("Multiple versions",          "Both CUDA-11 and CUDA-12 are supported."),
        # ── Units ──────────────────────────────────────────────────────
        ("Distance",                   "The trail is 42km long."),
        ("Weight",                     "Each package weighs 500kg."),
        ("Temperature °C",             "Water boils at 100°C."),
        ("Data size GB",               "Download the 2.5GB model file."),
        ("Frequency GHz",              "The CPU runs at 3.6GHz."),
        ("Latency ms",                 "Average latency is 12ms."),
        # ── HTML / URLs / emails ───────────────────────────────────────
        ("HTML tags",                  "<b>Hello</b> World! It's a great day."),
        ("URL and email",              "Visit https://example.com or email hello@example.com."),
        ("Hashtags and mentions",      "#NLP @user great post!"),
        # ── Contractions ───────────────────────────────────────────────
        ("Contractions",               "I don't know, won't you help? They've already left."),
        ("Ain't / let's",              "Ain't no mountain high enough. Let's go!"),
        # ── Edge / tricky cases ─────────────────────────────────────────
        ("Score / ratio",              "The final score was 3:0."),
        ("Aspect ratio",               "The display is 16:9."),
        ("IP address",                 "Connect to server at 192.168.1.1 on port 8080."),
        ("Phone number",               "Call us at 555-1234 or 1-800-555-0199."),
        ("Negative vs. hyphen",        "On a scale of -10 to 10, she rated it 8."),
        ("Ellipsis",                   "He paused... then spoke."),
        ("Em dash number",             "The result — 42 — surprised everyone."),
        # ── Mixed / real-world ──────────────────────────────────────────
        ("Research abstract",          "We trained a 7B parameter model for 100 epochs at 1e-4 learning rate."),
        ("GPT benchmark",              "GPT-4 scored 90% on the benchmark — 15% better than GPT-3.5."),
        ("News headline",              "Fed raises rates by 0.25%, S&P 500 drops 1.2%."),
        ("Startup pitch",              "We raised $2.5M in seed funding and are growing 20% month-over-month."),
        ("Tech spec",                  "The M3 chip runs at 4.05GHz with a 40M transistor GPU and 8GB RAM."),
    ]

    print("=" * 70)
    print("TextPreprocessor Demo")
    print("=" * 70)
    for label, text in cases:
        print(f"\n  [{label}]")
        print(f"  IN : {text}")
        print(f"  OUT: {pp(text)}")

    print("\n" + "=" * 70)
    print("number_to_words")
    print("=" * 70)
    for n in [0, 1, 12, 19, 20, 99, 100, 1000, 1200, 15_000, 1_000_000, -42, 999_999_999]:
        print(f"  {n:>15,} → {number_to_words(n)}")

    print("\n" + "=" * 70)
    print("float_to_words")
    print("=" * 70)
    for f in [3.14, -0.5, 1200.99, 3.10, 1.007, 0.001]:
        print(f"  {f} → {float_to_words(f)}")

    print("\n" + "=" * 70)
    print("expand_roman_numerals  (opt-in)")
    print("=" * 70)
    pp_roman = TextPreprocessor(expand_roman_numerals=True)
    for text in ["World War II ended in 1945.", "Chapter IV begins here.", "Louis XIV was king."]:
        print(f"  IN : {text}")
        print(f"  OUT: {pp_roman(text)}")
