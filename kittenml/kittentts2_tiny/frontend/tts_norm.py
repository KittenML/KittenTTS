#!/usr/bin/env python
# Copied from the Kitten training code (tts_norm.py). Vendored into kittenml: only import paths and asset paths changed.
"""TTS text normalisation: expand text to its SPOKEN form before G2P.

misaki already handles a lot -- currency, percent, large integers, `Dr.`, `etc.` -- so
this layer fixes only what it demonstrably gets wrong. Measured failures that motivated
each rule (misaki output on the left of the arrow):

    3:45pm       -> "θɹˈi:fˈɔɹTi..."   the COLON leaks into the phoneme string as a
                                       punctuation token, producing a spurious pause
    2026-10-13   -> "twenty twenty sixtenth thirteen"        mangled
    St. Paul St. -> "sent. pˈɔl sənt."  wrong word, period kept
    $24.5B       -> "...point five bi"  magnitude suffix not expanded
    555-0123     -> "...fˈIvwˌʌn..."    digits run together
    3/4          -> "three four"        not read as a fraction
    ext. 42      -> "ˈɛkst. ..."        not expanded

Sentence punctuation (. , ! ? ; and the dash) is PRESERVED -- the models use it for
prosody and the silence token. Only punctuation that would be mis-spoken is rewritten.

Order matters: dates and times must run before the generic number rule, or `2026-10-13`
becomes three separate cardinals.
"""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

UNITS = {
    "km": "kilometres", "mi": "miles", "cm": "centimetres", "mm": "millimetres",
    "kg": "kilograms", "lb": "pounds", "lbs": "pounds", "oz": "ounces",
    "ml": "millilitres", "l": "litres", "ft": "feet", "in": "inches",
    "hr": "hours", "hrs": "hours", "min": "minutes", "mins": "minutes",
    "sec": "seconds", "secs": "seconds", "ms": "milliseconds",
    "gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes", "tb": "terabytes",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "kw": "kilowatts", "mph": "miles per hour", "kph": "kilometres per hour",
}
MAG = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}
FRACTIONS = {(1, 2): "a half", (1, 3): "a third", (2, 3): "two thirds",
             (1, 4): "a quarter", (3, 4): "three quarters", (1, 8): "an eighth"}
ABBREV = {
    "mr": "Mister", "mrs": "Missus", "ms": "Miss", "dr": "Doctor",
    "prof": "Professor", "rev": "Reverend", "hon": "Honourable",
    "ext": "extension", "approx": "approximately", "vs": "versus",
    "est": "estimated", "dept": "department", "univ": "university",
    "fig": "figure", "eq": "equation", "ref": "reference", "vol": "volume",
    "incl": "including", "excl": "excluding", "min": "minimum",
    "max": "maximum", "avg": "average", "mt": "mount", "ft": "fort",
    "appt": "appointment", "pp": "pages", "jr": "Junior", "sr": "Senior",
    "inc": "Incorporated", "ltd": "Limited", "corp": "Corporation",
    "mins": "minutes", "hrs": "hours", "wks": "weeks", "yrs": "years",
}
DAYS = {"Mon": "Monday", "Tue": "Tuesday", "Tues": "Tuesday",
        "Wed": "Wednesday", "Weds": "Wednesday", "Thu": "Thursday",
        "Thur": "Thursday", "Thurs": "Thursday", "Fri": "Friday",
        "Sat": "Saturday", "Sun": "Sunday"}
MONTHS_ABBR = {"Jan": "January", "Feb": "February", "Mar": "March",
               "Apr": "April", "Jun": "June", "Jul": "July", "Aug": "August",
               "Sep": "September", "Sept": "September", "Oct": "October",
               "Nov": "November", "Dec": "December"}
SYMBOLS = {"%": " percent ", "&": " and ", "@": " at ", "+": " plus ",
           "=": " equals ", "×": " times ", "÷": " divided by ",
           "<": " less than ", ">": " greater than ", "~": " about ",
           "°": " degrees ", "€": " euros ", "£": " pounds ", "¥": " yen ",
           "©": " copyright ", "™": " trademark ", "§": " section "}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
         "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12}
ROMAN_CTX = r"(?:chapter|section|part|volume|book|act|phase|appendix|world war|type|class|figure|table|step|stage)"


def _card(n) -> str:
    try:
        return num2words(n)
    except Exception:
        return str(n)


def _year(y: int) -> str:
    if 1100 <= y <= 1999 or 2000 <= y <= 2099:
        try:
            return num2words(y, to="year")
        except Exception:
            pass
    return _card(y)


def _dates(s: str) -> str:
    def iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        return f"{MONTHS[mo-1]} {num2words(d, to='ordinal')}, {_year(y)}"
    s = re.sub(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", iso, s)

    def slash(m):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= a <= 12 and 1 <= b <= 31):
            return m.group(0)
        if y < 100:
            y += 2000 if y < 50 else 1900
        return f"{MONTHS[a-1]} {num2words(b, to='ordinal')}, {_year(y)}"
    s = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", slash, s)
    # bare 4-digit year after a month name
    s = re.sub(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b",
               lambda m: f"{m.group(1)} {num2words(int(m.group(2)), to='ordinal')}, "
                         f"{_year(int(m.group(3)))}", s, flags=re.I)
    return s


def _keep_final_dot(m, out: str) -> str:
    """The am/pm group ends in an optional "." and will swallow a SENTENCE-final period:
    "3:45pm." normalised to "...PM" with no punctuation left, costing the sentence its
    final intonation. Put the period back when the match really did end the sentence."""
    if not m.group(0).endswith("."):
        return out
    rest = m.string[m.end():]
    if rest.strip() == "" or (rest[:1].isspace() and rest.strip()[:1].isupper()):
        return out + "."
    return out


def _times(s: str) -> str:
    def t(m):
        h, mi, ampm = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
        if h > 23 or mi > 59:
            return m.group(0)
        suffix = ""
        if ampm:
            suffix = " AM" if ampm.startswith("a") else " PM"
        elif h == 0:
            h, suffix = 12, " AM"
        if mi == 0:
            spoken = f"{_card(h if h <= 12 or ampm else h)} o'clock" if not suffix \
                     else f"{_card(h if h <= 12 else h - 12)}"
        elif mi < 10:
            spoken = f"{_card(h if h <= 12 or not suffix else h - 12)} oh {_card(mi)}"
        else:
            spoken = f"{_card(h if h <= 12 or not suffix else h - 12)} {_card(mi)}"
        return _keep_final_dot(m, spoken + suffix)
    # 3:45pm / 07:05 / 14:30
    # \s* must sit INSIDE the optional am/pm group, or a bare H:MM eats the space that
    # follows it: "14:30 yesterday" became "fourteen thirtyyesterday".
    return re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*([ap]\.?m\.?))?", t, s, flags=re.I)


def _bare_hour(s: str) -> str:
    """9am / 9 PM / 11a.m. -- _times() requires H:MM, so these reached the G2P intact."""
    def t(m):
        h = int(m.group(1))
        if not 1 <= h <= 12:
            return m.group(0)
        return _keep_final_dot(
            m, f"{_card(h)} {'AM' if m.group(2).lower().startswith('a') else 'PM'}")
    return re.sub(r"\b(\d{1,2})\s*([ap])\.?m\.?(?![a-z])", t, s, flags=re.I)


def _phones(s: str) -> str:
    def ph(m):
        digits = re.sub(r"\D", "", m.group(0))
        if not 7 <= len(digits) <= 11:
            return m.group(0)
        groups, i = [], 0
        if len(digits) == 11 and digits[0] == "1":
            groups.append("one"); i = 1
        for a, b in ((i, i + 3), (i + 3, i + 6), (i + 6, len(digits))):
            if a < len(digits):
                groups.append(" ".join(_card(int(c)) for c in digits[a:b]))
        return ", ".join(g for g in groups if g)
    return re.sub(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]?\d{0,4}\b",
                  ph, s)


def _money(s: str) -> str:
    def m_(m):
        amt, suf = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        if suf:
            num = float(amt) if "." in amt else int(amt)
            whole = _card(int(num)) if float(num).is_integer() else \
                f"{_card(int(num))} point {' '.join(_card(int(d)) for d in amt.split('.')[1])}"
            return f"{whole} {MAG[suf]} dollars"
        if "." in amt:
            d, c = amt.split(".", 1)
            c = (c + "00")[:2]
            dollars = f"{_card(int(d))} dollar{'' if int(d) == 1 else 's'}"
            if int(c):
                return f"{dollars} and {_card(int(c))} cent{'' if int(c) == 1 else 's'}"
            return dollars
        return f"{_card(int(amt))} dollar{'' if int(amt) == 1 else 's'}"
    return re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?([KkMmBbTt])\b)?", m_, s)


def _magnitudes(s: str) -> str:
    return re.sub(r"\b(\d[\d,]*(?:\.\d+)?)\s*([KMBT])\b",
                  lambda m: f"{_decimal(m.group(1))} {MAG[m.group(2).lower()]}", s)


def _decimal(tok: str) -> str:
    tok = tok.replace(",", "")
    if not any(c.isdigit() for c in tok):
        return tok          # backstop: never raise out of the text front end

    if "." in tok:
        a, b = tok.split(".", 1)
        return f"{_card(int(a or 0))} point {' '.join(_card(int(d)) for d in b)}"
    return _card(int(tok))


def _fractions(s: str) -> str:
    def f_(m):
        a, b = int(m.group(1)), int(m.group(2))
        if (a, b) in FRACTIONS:
            return FRACTIONS[(a, b)]
        if b == 0:
            return m.group(0)
        return f"{_card(a)} {num2words(b, to='ordinal')}{'s' if a != 1 else ''}"
    return re.sub(r"\b(\d{1,3})/(\d{1,3})\b", f_, s)


def _versions(s: str) -> str:
    return re.sub(r"\b(\d+)\.(\d+)\.(\d+)\b",
                  lambda m: " point ".join(_card(int(g)) for g in m.groups()), s)


def _units(s: str) -> str:
    def u(m):
        n, unit = m.group(1), m.group(2).lower()
        val = _decimal(n)
        word = UNITS[unit]
        if n.replace(",", "") in ("1", "1.0"):
            word = word.rstrip("s") if word.endswith("s") else word
        return f"{val} {word}"
    # the number MUST start with a digit: "[\d,]+" also matches a bare comma, so
    # "Hello, in the morning" used to reach int("") and raise
    pat = r"\b(\d[\d,]*(?:\.\d+)?)\s?(" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b"
    s = re.sub(pat, u, s, flags=re.I)
    def deg(m):
        raw, unit = m.group(1), m.group(2).upper()
        neg = raw[0] in "-−"
        val = _decimal(raw[1:] if neg else raw)
        scale = "Celsius" if unit == "C" else "Fahrenheit"
        return f"{'minus ' if neg else ''}{val} degrees {scale}"
    s = re.sub(r"([-−]?[\d.]+)\s?°\s?([CF])\b", deg, s)
    return s


def _abbrev(s: str) -> str:
    # St. is ambiguous: "St. Paul" is Saint, "Paul St." is Street
    s = re.sub(r"\bSt\.\s+(?=[A-Z])", "Saint ", s)
    # keep the terminator when "St." ends the sentence, else it swallows the full stop
    s = re.sub(r"\bSt\.(?=\s*$)", "Street.", s)
    s = re.sub(r"\bSt\.(?=\s*[\"'\)])", "Street.", s)
    s = re.sub(r"\bSt\.(?!\s+[A-Z])", "Street", s)
    s = re.sub(r"\bAve\.", "Avenue", s)
    s = re.sub(r"\bRd\.", "Road", s)
    s = re.sub(r"\bBlvd\.", "Boulevard", s)
    s = re.sub(r"\bNo\.\s*(?=\d)", "number ", s)
    s = re.sub(r"\bNos\.\s*(?=\d)", "numbers ", s)
    for k, v in {**DAYS, **MONTHS_ABBR}.items():
        s = re.sub(rf"\b{k}\.", v, s)          # case-sensitive: see the note above
    for k, v in ABBREV.items():
        s = re.sub(rf"\b{k}\.", v, s, flags=re.I)
    return s


def _romans(s: str) -> str:
    pat = rf"\b({ROMAN_CTX})\s+([IVX]{{1,4}})\b"
    return re.sub(pat, lambda m: f"{m.group(1)} {_card(ROMAN[m.group(2).lower()])}"
                  if m.group(2).lower() in ROMAN else m.group(0), s, flags=re.I)


def _ordinals(s: str) -> str:
    return re.sub(r"\b(\d+)(?:st|nd|rd|th)\b",
                  lambda m: num2words(int(m.group(1)), to="ordinal"), s, flags=re.I)


def _ranges(s: str) -> str:
    return re.sub(r"\b(\d+)\s?[-–—]\s?(\d+)\b",
                  lambda m: f"{_card(int(m.group(1)))} to {_card(int(m.group(2)))}", s)


def _numbers(s: str) -> str:
    def n_(m):
        tok = m.group(0)
        # a 4-digit integer is usually a year in prose -- but ONLY if written bare.
        # "1,234" has thousands separators and is a cardinal, not a year; treating it
        # as one produced "twelve thirty-four books".
        if "," not in tok and re.fullmatch(r"\d{4}", tok) and 1100 <= int(tok) <= 2099:
            return _year(int(tok))
        return _decimal(tok)
    s = re.sub(r"(?<![\w.])[-−](\d)", r"minus \1", s)
    return re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", n_, s)


def normalize(text: str) -> str:
    """Expand `text` to spoken form. Keeps . , ! ? ; and dashes for prosody."""
    s = unicodedata.normalize("NFKC", str(text))
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"')
          .replace("—", " - ").replace("–", "-").replace("…", "..."))
    s = re.sub(r"https?://\S+|www\.\S+", " link ", s)
    s = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", " email address ", s)
    s = re.sub(r"\be\.\s?g\.", "for example,", s, flags=re.I)
    s = re.sub(r"\bi\.\s?e\.", "that is,", s, flags=re.I)
    s = re.sub(r"\betc\.", "etcetera", s, flags=re.I)
    s = re.sub(r"\bvs\.?\b", "versus", s, flags=re.I)
    s = _dates(s)
    s = _times(s)
    s = _bare_hour(s)
    s = _phones(s)
    s = _money(s)
    s = _versions(s)
    s = _fractions(s)
    s = _magnitudes(s)
    s = _units(s)
    s = _abbrev(s)
    s = _romans(s)
    s = _ordinals(s)
    s = _ranges(s)
    for k, v in SYMBOLS.items():
        s = s.replace(k, v)
    s = _numbers(s)
    # any colon that survived the time rule becomes a comma: same pause, no stray token
    s = s.replace(":", ",")
    s = re.sub(r"[_*`^|\\/\[\]{}<>]", " ", s)
    s = re.sub(r"\s+([.,!?;])", r"\1", s)
    # do not insert a space BEFORE another punctuation mark: "Mon.," would become
    # "Mon. ," and the \s+ cleanup on the line above has already run
    s = re.sub(r"([.,!?;])(?=[^\s\"',.!?;])", r"\1 ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


if __name__ == "__main__":
    import sys
    CASES = [
        "It cost $5.99 at the store.",
        "Dr. Smith lives on St. Paul St.",
        "The meeting is at 3:45pm on 2026-10-13.",
        "NASA's budget rose 12% to $24.5B.",
        "Chapter IV, section 3b.",
        "Call 555-0123 ext. 42.",
        "He ran 5km in 22min.",
        "The temperature is -3°C today.",
        "e.g. apples, i.e. fruit, etc.",
        "She owns 1,234 books & 7 cats.",
        "Flight AA1234 departs from JFK at 07:05.",
        "Version 2.0.1 fixes 3/4 of the bugs.",
        "Revenue grew 150% to $1.2B in Q3 2025.",
        "Mix 250ml water with 1.5kg flour at 180°C.",
        "Visit https://example.com or email a@b.com.",
        "The 3rd of May, 1999 — see pages 10-15.",
    ]
    for c in (sys.argv[1:] or CASES):
        print(f"  IN : {c}\n  OUT: {normalize(c)}\n")
