//! Text preprocessing pipeline for TTS input normalisation.
//!
//! Port of the Python `preprocess.py` — converts numbers, currencies, times,
//! ordinals, contractions, etc. into their spoken-word equivalents so the
//! phonemiser receives clean English text.

use fancy_regex::Regex;
use once_cell::sync::Lazy;

// ─────────────────────────────────────────────
// Number → Words conversion
// ─────────────────────────────────────────────

const ONES: &[&str] = &[
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

const SCALE: &[&str] = &["", "thousand", "million", "billion", "trillion"];

fn three_digits_to_words(n: u64) -> String {
    if n == 0 {
        return String::new();
    }
    let mut parts = Vec::new();
    let hundreds = n / 100;
    let remainder = n % 100;
    if hundreds > 0 {
        parts.push(format!("{} hundred", ONES[hundreds as usize]));
    }
    if remainder < 20 {
        if remainder > 0 {
            parts.push(ONES[remainder as usize].to_string());
        }
    } else {
        let tens_word = TENS[(remainder / 10) as usize];
        let ones_word = ONES[(remainder % 10) as usize];
        if ones_word.is_empty() {
            parts.push(tens_word.to_string());
        } else {
            parts.push(format!("{}-{}", tens_word, ones_word));
        }
    }
    parts.join(" ")
}

/// Convert an integer to its English word representation.
pub fn number_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    if n < 0 {
        return format!("negative {}", number_to_words(-n));
    }
    let n = n as u64;

    // X00–X999 read as "X hundred" (e.g. 1200 → "twelve hundred")
    if (100..=9999).contains(&n) && n % 100 == 0 && n % 1000 != 0 {
        let hundreds = n / 100;
        if hundreds < 20 {
            return format!("{} hundred", ONES[hundreds as usize]);
        }
    }

    let mut parts = Vec::new();
    let mut remaining = n;
    for scale in SCALE {
        let chunk = remaining % 1000;
        if chunk > 0 {
            let chunk_words = three_digits_to_words(chunk);
            if scale.is_empty() {
                parts.push(chunk_words);
            } else {
                parts.push(format!("{} {}", chunk_words, scale));
            }
        }
        remaining /= 1000;
        if remaining == 0 {
            break;
        }
    }
    parts.reverse();
    parts.join(" ")
}

/// Convert a float to words, reading decimal digits individually.
pub fn float_to_words(value: &str, decimal_sep: &str) -> String {
    let mut text = value.to_string();
    let negative = text.starts_with('-');
    if negative {
        text = text[1..].to_string();
    }

    let result = if text.contains('.') {
        let mut split = text.splitn(2, '.');
        let int_part = split.next().unwrap_or("0");
        let dec_part = split.next().unwrap_or("0");
        let int_words = if int_part.is_empty() {
            "zero".to_string()
        } else {
            number_to_words(int_part.parse::<i64>().unwrap_or(0))
        };
        let digit_names = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let dec_words: Vec<&str> = dec_part
            .chars()
            .map(|c| {
                let d = c.to_digit(10).unwrap_or(0) as usize;
                digit_names[d]
            })
            .collect();
        format!("{} {} {}", int_words, decimal_sep, dec_words.join(" "))
    } else {
        number_to_words(text.parse::<i64>().unwrap_or(0))
    };

    if negative {
        format!("negative {}", result)
    } else {
        result
    }
}

fn roman_to_int(s: &str) -> i64 {
    let val = |ch: char| -> i64 {
        match ch.to_ascii_uppercase() {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => 0,
        }
    };
    let mut result: i64 = 0;
    let mut prev: i64 = 0;
    for ch in s.chars().rev() {
        let curr = val(ch);
        if curr >= prev {
            result += curr;
        } else {
            result -= curr;
        }
        prev = curr;
    }
    result
}

// ─────────────────────────────────────────────
// Ordinal helpers
// ─────────────────────────────────────────────

fn ordinal_suffix(n: i64) -> String {
    let word = number_to_words(n);

    let (prefix, last, joiner) = if word.contains('-') {
        let idx = word.rfind('-').unwrap();
        (word[..idx].to_string(), word[idx + 1..].to_string(), "-")
    } else if word.contains(' ') {
        let idx = word.rfind(' ').unwrap();
        (word[..idx].to_string(), word[idx + 1..].to_string(), " ")
    } else {
        (String::new(), word.clone(), "")
    };

    let ordinal_exceptions: &[(&str, &str)] = &[
        ("one", "first"),
        ("two", "second"),
        ("three", "third"),
        ("four", "fourth"),
        ("five", "fifth"),
        ("six", "sixth"),
        ("seven", "seventh"),
        ("eight", "eighth"),
        ("nine", "ninth"),
        ("twelve", "twelfth"),
    ];

    let last_ord = ordinal_exceptions
        .iter()
        .find(|(base, _)| *base == last)
        .map(|(_, ord)| ord.to_string())
        .unwrap_or_else(|| {
            if last.ends_with('t') {
                format!("{}h", last)
            } else if last.ends_with('e') {
                format!("{}th", &last[..last.len() - 1])
            } else {
                format!("{}th", last)
            }
        });

    if prefix.is_empty() {
        last_ord
    } else {
        format!("{}{}{}", prefix, joiner, last_ord)
    }
}

// ─────────────────────────────────────────────
// Compiled regex patterns (lazy statics)
// ─────────────────────────────────────────────

static RE_URL: Lazy<Regex> = Lazy::new(|| Regex::new(r"https?://\S+|www\.\S+").unwrap());
static RE_EMAIL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b[\w.+\-]+@[\w\-]+\.[a-z]{2,}\b").unwrap());
static RE_HTML: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());
static RE_HASHTAG: Lazy<Regex> = Lazy::new(|| Regex::new(r"#\w+").unwrap());
static RE_MENTION: Lazy<Regex> = Lazy::new(|| Regex::new(r"@\w+").unwrap());
static RE_PUNCT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[^\w\s.,?!;:\-\u{2014}\u{2013}\u{2026}]").unwrap());
static RE_SPACES: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static RE_NUMBER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?<![a-zA-Z])-?[\d,]+(?:\.\d+)?").unwrap());
static RE_ORDINAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(\d+)(st|nd|rd|th)\b").unwrap());
static RE_PERCENT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(-?[\d,]+(?:\.\d+)?)\s*%").unwrap());
static RE_CURRENCY: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"([$€£¥₹₩₿])\s*([\d,]+(?:\.\d+)?)\s*([KMBT])?(?![a-zA-Z\d])").unwrap()
});
static RE_TIME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b").unwrap());
static RE_RANGE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?<!\w)(\d+)-(\d+)(?!\w)").unwrap());
static RE_MODEL_VER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b([a-zA-Z][a-zA-Z0-9]*)-(\d[\d.]*)(?=[^\d.]|$)").unwrap());
static RE_UNIT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(km|kg|mg|ml|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|°[cCfF]|[cCfF]°|ms|ns|µs)\b").unwrap()
});
static RE_SCALE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*([KMBT])(?![a-zA-Z\d])").unwrap());
static RE_SCI: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?<![a-zA-Z\d])(-?\d+(?:\.\d+)?)[eE]([+\-]?\d+)(?![a-zA-Z\d])").unwrap()
});
static RE_FRACTION: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d+)\s*/\s*(\d+)\b").unwrap());
static RE_DECADE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d{1,3})0s\b").unwrap());
static RE_LEAD_DEC: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?<!\d)\.(\d)").unwrap());
static RE_ROMAN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(M{0,4})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b").unwrap()
});
static RE_IP: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b").unwrap());
// Phone patterns
static RE_PHONE_11: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?<!\d-)(?<!\d)\b(\d{1,2})-(\d{3})-(\d{3})-(\d{4})\b(?!-\d)").unwrap()
});
static RE_PHONE_10: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?<!\d-)(?<!\d)\b(\d{3})-(\d{3})-(\d{4})\b(?!-\d)").unwrap());
static RE_PHONE_7: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?<!\d-)\b(\d{3})-(\d{4})\b(?!-\d)").unwrap());

// Contraction patterns
static RE_CANT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bcan't\b").unwrap());
static RE_WONT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bwon't\b").unwrap());
static RE_SHANT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bshan't\b").unwrap());
static RE_AINT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bain't\b").unwrap());
static RE_LETS: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\blet's\b").unwrap());
static RE_ITS: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bit's\b").unwrap());
static RE_NT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(\w+)n't\b").unwrap());
static RE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(\w+)'re\b").unwrap());
static RE_VE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(\w+)'ve\b").unwrap());
static RE_LL: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(\w+)'ll\b").unwrap());
static RE_D: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(\w+)'d\b").unwrap());
static RE_M: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(\w+)'m\b").unwrap());

static RE_TITLE_WORDS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(war|chapter|part|volume|act|scene|book|section|article|king|queen|pope|louis|henry|edward|george|william|james|phase|round|level|stage|class|type|version|episode|season)\b").unwrap()
});

// ─────────────────────────────────────────────
// Expansion functions
// ─────────────────────────────────────────────

fn expand_ordinals(text: &str) -> String {
    RE_ORDINAL
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let n: i64 = caps[1].parse().unwrap_or(0);
            ordinal_suffix(n)
        })
        .into_owned()
}

fn expand_percentages(text: &str) -> String {
    RE_PERCENT
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = caps[1].replace(',', "");
            if raw.contains('.') {
                format!("{} percent", float_to_words(&raw, "point"))
            } else {
                format!(
                    "{} percent",
                    number_to_words(raw.parse::<i64>().unwrap_or(0))
                )
            }
        })
        .to_string()
}

fn expand_currency(text: &str) -> String {
    let currency_name = |sym: &str| -> &str {
        match sym {
            "$" => "dollar",
            "€" => "euro",
            "£" => "pound",
            "¥" => "yen",
            "₹" => "rupee",
            "₩" => "won",
            "₿" => "bitcoin",
            _ => "",
        }
    };
    let scale_map = |s: &str| -> &str {
        match s {
            "K" => "thousand",
            "M" => "million",
            "B" => "billion",
            "T" => "trillion",
            _ => "",
        }
    };

    RE_CURRENCY
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let symbol = &caps[1];
            let raw = caps[2].replace(',', "");
            let scale_suffix = caps.get(3).map(|m| m.as_str());
            let unit = currency_name(symbol);

            if let Some(suffix) = scale_suffix {
                let scale_word = scale_map(suffix);
                let num = if raw.contains('.') {
                    float_to_words(&raw, "point")
                } else {
                    number_to_words(raw.parse::<i64>().unwrap_or(0))
                };
                let plural = if !unit.is_empty() {
                    format!("{}s", unit)
                } else {
                    String::new()
                };
                format!("{} {} {}", num, scale_word, plural)
                    .trim()
                    .to_string()
            } else if raw.contains('.') {
                let mut split = raw.splitn(2, '.');
                let int_part = split.next().unwrap_or("0");
                let dec_part = split.next().unwrap_or("0");
                let dec_str = format!("{:0<2}", &dec_part[..dec_part.len().min(2)]);
                let dec_val: i64 = dec_str.parse().unwrap_or(0);
                let int_val: i64 = int_part.parse().unwrap_or(0);
                let int_words = number_to_words(int_val);
                let mut result = if !unit.is_empty() {
                    format!("{} {}s", int_words, unit)
                } else {
                    int_words
                };
                if dec_val > 0 {
                    let cents = number_to_words(dec_val);
                    let cent_suffix = if dec_val != 1 { "cents" } else { "cent" };
                    result = format!("{} and {} {}", result, cents, cent_suffix);
                }
                result
            } else {
                let val: i64 = raw.parse().unwrap_or(0);
                let words = number_to_words(val);
                if !unit.is_empty() {
                    let plural = if val != 1 { "s" } else { "" };
                    format!("{} {}{}", words, unit, plural)
                } else {
                    words
                }
            }
        })
        .to_string()
}

fn expand_time(text: &str) -> String {
    RE_TIME
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let h: i64 = caps[1].parse().unwrap_or(0);
            let mins: i64 = caps[2].parse().unwrap_or(0);
            let suffix = caps
                .get(4)
                .map(|m| format!(" {}", m.as_str().to_lowercase()))
                .unwrap_or_default();
            let h_words = number_to_words(h);
            if mins == 0 {
                if caps.get(4).is_some() {
                    format!("{}{}", h_words, suffix)
                } else {
                    format!("{} hundred{}", h_words, suffix)
                }
            } else if mins < 10 {
                format!("{} oh {}{}", h_words, number_to_words(mins), suffix)
            } else {
                format!("{} {}{}", h_words, number_to_words(mins), suffix)
            }
        })
        .to_string()
}

fn expand_ranges(text: &str) -> String {
    RE_RANGE
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let lo = number_to_words(caps[1].parse::<i64>().unwrap_or(0));
            let hi = number_to_words(caps[2].parse::<i64>().unwrap_or(0));
            format!("{} to {}", lo, hi)
        })
        .to_string()
}

fn expand_model_names(text: &str) -> String {
    RE_MODEL_VER
        .replace_all(text, |caps: &fancy_regex::Captures| {
            format!("{} {}", &caps[1], &caps[2])
        })
        .to_string()
}

fn expand_units_map(u: &str) -> &str {
    match u {
        "km" => "kilometers",
        "kg" => "kilograms",
        "mg" => "milligrams",
        "ml" => "milliliters",
        "gb" => "gigabytes",
        "mb" => "megabytes",
        "kb" => "kilobytes",
        "tb" => "terabytes",
        "hz" => "hertz",
        "khz" => "kilohertz",
        "mhz" => "megahertz",
        "ghz" => "gigahertz",
        "mph" => "miles per hour",
        "kph" => "kilometers per hour",
        "ms" => "milliseconds",
        "ns" => "nanoseconds",
        "µs" => "microseconds",
        "°c" | "c°" => "degrees Celsius",
        "°f" | "f°" => "degrees Fahrenheit",
        _ => u,
    }
}

fn expand_units(text: &str) -> String {
    RE_UNIT
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = &caps[1];
            let unit_lower = caps[2].to_lowercase();
            let expanded = expand_units_map(&unit_lower);
            let num = if raw.contains('.') {
                float_to_words(raw, "point")
            } else {
                number_to_words(raw.parse::<i64>().unwrap_or(0))
            };
            format!("{} {}", num, expanded)
        })
        .to_string()
}

fn expand_scale_map(s: &str) -> &str {
    match s {
        "K" => "thousand",
        "M" => "million",
        "B" => "billion",
        "T" => "trillion",
        _ => s,
    }
}

fn expand_scale_suffixes(text: &str) -> String {
    RE_SCALE
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = &caps[1];
            let suffix = &caps[2];
            let scale_word = expand_scale_map(suffix);
            let num = if raw.contains('.') {
                float_to_words(raw, "point")
            } else {
                number_to_words(raw.parse::<i64>().unwrap_or(0))
            };
            format!("{} {}", num, scale_word)
        })
        .to_string()
}

fn expand_scientific_notation(text: &str) -> String {
    RE_SCI
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let coeff_raw = &caps[1];
            let exp: i64 = caps[2].parse().unwrap_or(0);
            let coeff_words = if coeff_raw.contains('.') {
                float_to_words(coeff_raw, "point")
            } else {
                number_to_words(coeff_raw.parse::<i64>().unwrap_or(0))
            };
            let exp_words = number_to_words(exp.abs());
            let sign = if exp < 0 { "negative " } else { "" };
            format!("{} times ten to the {}{}", coeff_words, sign, exp_words)
        })
        .to_string()
}

fn expand_fractions(text: &str) -> String {
    RE_FRACTION
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let num: i64 = caps[1].parse().unwrap_or(0);
            let den: i64 = caps[2].parse().unwrap_or(1);
            if den == 0 {
                return caps[0].to_string();
            }
            let num_words = number_to_words(num);
            let denom_word = if den == 2 {
                if num == 1 {
                    "half".to_string()
                } else {
                    "halves".to_string()
                }
            } else if den == 4 {
                if num == 1 {
                    "quarter".to_string()
                } else {
                    "quarters".to_string()
                }
            } else {
                let base = ordinal_suffix(den);
                if num != 1 {
                    format!("{}s", base)
                } else {
                    base
                }
            };
            format!("{} {}", num_words, denom_word)
        })
        .to_string()
}

fn expand_decades(text: &str) -> String {
    let decade_map = |d: u64| -> &'static str {
        match d {
            0 => "hundreds",
            1 => "tens",
            2 => "twenties",
            3 => "thirties",
            4 => "forties",
            5 => "fifties",
            6 => "sixties",
            7 => "seventies",
            8 => "eighties",
            9 => "nineties",
            _ => "",
        }
    };
    RE_DECADE
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let base: u64 = caps[1].parse().unwrap_or(0);
            let decade_digit = base % 10;
            let decade_word = decade_map(decade_digit);
            if base < 10 {
                decade_word.to_string()
            } else {
                let century_part = base / 10;
                format!("{} {}", number_to_words(century_part as i64), decade_word)
            }
        })
        .to_string()
}

fn expand_ip_addresses(text: &str) -> String {
    let digit_name = |c: char| -> &'static str {
        match c {
            '0' => "zero",
            '1' => "one",
            '2' => "two",
            '3' => "three",
            '4' => "four",
            '5' => "five",
            '6' => "six",
            '7' => "seven",
            '8' => "eight",
            '9' => "nine",
            _ => "",
        }
    };
    let octet_to_words =
        |s: &str| -> String { s.chars().map(digit_name).collect::<Vec<_>>().join(" ") };

    RE_IP
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let parts: Vec<String> = (1..=4).map(|i| octet_to_words(&caps[i])).collect();
            parts.join(" dot ")
        })
        .to_string()
}

fn expand_phone_numbers(text: &str) -> String {
    let digit_name = |c: char| -> &'static str {
        match c {
            '0' => "zero",
            '1' => "one",
            '2' => "two",
            '3' => "three",
            '4' => "four",
            '5' => "five",
            '6' => "six",
            '7' => "seven",
            '8' => "eight",
            '9' => "nine",
            _ => "",
        }
    };
    let digits_to_words =
        |s: &str| -> String { s.chars().map(digit_name).collect::<Vec<_>>().join(" ") };

    // 11-digit
    let text = RE_PHONE_11
        .replace_all(&text, |caps: &fancy_regex::Captures| {
            format!(
                "{} {} {} {}",
                digits_to_words(&caps[1]),
                digits_to_words(&caps[2]),
                digits_to_words(&caps[3]),
                digits_to_words(&caps[4])
            )
        })
        .to_string();
    // 10-digit
    let text = RE_PHONE_10
        .replace_all(&text, |caps: &fancy_regex::Captures| {
            format!(
                "{} {} {}",
                digits_to_words(&caps[1]),
                digits_to_words(&caps[2]),
                digits_to_words(&caps[3])
            )
        })
        .to_string();
    // 7-digit
    RE_PHONE_7
        .replace_all(&text, |caps: &fancy_regex::Captures| {
            format!(
                "{} {}",
                digits_to_words(&caps[1]),
                digits_to_words(&caps[2])
            )
        })
        .to_string()
}

fn expand_contractions(text: &str) -> String {
    let mut t = RE_CANT.replace_all(text, "cannot").to_string();
    t = RE_WONT.replace_all(&t, "will not").to_string();
    t = RE_SHANT.replace_all(&t, "shall not").to_string();
    t = RE_AINT.replace_all(&t, "is not").to_string();
    t = RE_LETS.replace_all(&t, "let us").to_string();
    t = RE_ITS.replace_all(&t, "it is").to_string();
    t = RE_NT.replace_all(&t, "$1 not").to_string();
    t = RE_RE.replace_all(&t, "$1 are").to_string();
    t = RE_VE.replace_all(&t, "$1 have").to_string();
    t = RE_LL.replace_all(&t, "$1 will").to_string();
    t = RE_D.replace_all(&t, "$1 would").to_string();
    t = RE_M.replace_all(&t, "$1 am").to_string();
    t
}

fn normalize_leading_decimals(text: &str) -> String {
    // -.5 → -0.5
    let re_neg = Regex::new(r"(?<!\d)(-)\.([\d])").unwrap();
    let text = re_neg.replace_all(text, "${1}0.$2").to_string();
    // .5 → 0.5
    RE_LEAD_DEC.replace_all(&text, "0.$1").to_string()
}

fn expand_roman_numerals(text: &str) -> String {
    RE_ROMAN
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let roman = caps[0].to_string();
            if roman.trim().is_empty() {
                return roman;
            }
            // Skip single ambiguous letters
            if roman.len() == 1 && "IVX".contains(&roman) {
                let start = caps.get(0).unwrap().start();
                let preceding = &text[start.saturating_sub(30)..start];
                if !RE_TITLE_WORDS.is_match(preceding).unwrap_or(false) {
                    return roman;
                }
            }
            let val = roman_to_int(&roman);
            if val == 0 {
                return roman;
            }
            number_to_words(val)
        })
        .to_string()
}

fn replace_numbers(text: &str, replace_floats: bool) -> String {
    RE_NUMBER
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = caps[0].replace(',', "");
            if raw.contains('.') && replace_floats {
                float_to_words(&raw, "point")
            } else {
                match raw.parse::<f64>() {
                    Ok(v) => number_to_words(v as i64),
                    Err(_) => caps[0].to_string(),
                }
            }
        })
        .to_string()
}

fn remove_urls(text: &str) -> String {
    RE_URL.replace_all(text, "").trim().to_string()
}

fn remove_emails(text: &str) -> String {
    RE_EMAIL.replace_all(text, "").trim().to_string()
}

fn remove_html_tags(text: &str) -> String {
    RE_HTML.replace_all(text, " ").to_string()
}

fn remove_hashtags(text: &str) -> String {
    RE_HASHTAG.replace_all(text, "").to_string()
}

fn remove_mentions(text: &str) -> String {
    RE_MENTION.replace_all(text, "").to_string()
}

fn remove_punctuation(text: &str) -> String {
    RE_PUNCT.replace_all(text, " ").to_string()
}

fn remove_extra_whitespace(text: &str) -> String {
    RE_SPACES.replace_all(text, " ").trim().to_string()
}

fn normalize_unicode(text: &str) -> String {
    // Rust strings are always valid UTF-8; we do NFC normalisation
    use std::iter::FromIterator;
    // Simple approach: we just return the string as-is since full NFC
    // requires the `unicode-normalization` crate. For TTS purposes this is acceptable.
    String::from_iter(text.chars())
}

fn remove_accents(text: &str) -> String {
    // Simplified: strip combining marks after NFD-ish decomposition.
    // For a full solution we'd use unicode-normalization crate.
    text.chars()
        .filter(|c| !('\u{0300}'..='\u{036f}').contains(c))
        .collect()
}

fn to_lowercase(text: &str) -> String {
    text.to_lowercase()
}

// ─────────────────────────────────────────────
// Configurable pipeline
// ─────────────────────────────────────────────

/// Configurable text preprocessing pipeline for TTS input normalisation.
pub struct TextPreprocessor {
    pub lowercase: bool,
    pub do_replace_numbers: bool,
    pub replace_floats: bool,
    pub do_expand_contractions: bool,
    pub do_expand_model_names: bool,
    pub do_expand_ordinals: bool,
    pub do_expand_percentages: bool,
    pub do_expand_currency: bool,
    pub do_expand_time: bool,
    pub do_expand_ranges: bool,
    pub do_expand_units: bool,
    pub do_expand_scale_suffixes: bool,
    pub do_expand_scientific_notation: bool,
    pub do_expand_fractions: bool,
    pub do_expand_decades: bool,
    pub do_expand_phone_numbers: bool,
    pub do_expand_ip_addresses: bool,
    pub do_normalize_leading_decimals: bool,
    pub do_expand_roman_numerals: bool,
    pub do_remove_urls: bool,
    pub do_remove_emails: bool,
    pub do_remove_html: bool,
    pub do_remove_hashtags: bool,
    pub do_remove_mentions: bool,
    pub do_remove_punctuation: bool,
    pub do_normalize_unicode: bool,
    pub do_remove_accents: bool,
    pub do_remove_extra_whitespace: bool,
}

impl Default for TextPreprocessor {
    /// Default matches the Python `TextPreprocessor(remove_punctuation=False)` used in the model.
    fn default() -> Self {
        TextPreprocessor {
            lowercase: true,
            do_replace_numbers: true,
            replace_floats: true,
            do_expand_contractions: true,
            do_expand_model_names: true,
            do_expand_ordinals: true,
            do_expand_percentages: true,
            do_expand_currency: true,
            do_expand_time: true,
            do_expand_ranges: true,
            do_expand_units: true,
            do_expand_scale_suffixes: true,
            do_expand_scientific_notation: true,
            do_expand_fractions: true,
            do_expand_decades: true,
            do_expand_phone_numbers: true,
            do_expand_ip_addresses: true,
            do_normalize_leading_decimals: true,
            do_expand_roman_numerals: false,
            do_remove_urls: true,
            do_remove_emails: true,
            do_remove_html: true,
            do_remove_hashtags: false,
            do_remove_mentions: false,
            do_remove_punctuation: false, // model default
            do_normalize_unicode: true,
            do_remove_accents: false,
            do_remove_extra_whitespace: true,
        }
    }
}

impl TextPreprocessor {
    /// Process text through the configured pipeline, exactly matching the Python
    /// `TextPreprocessor.process()` ordering.
    pub fn process(&self, text: &str) -> String {
        let mut text = text.to_string();

        if self.do_normalize_unicode {
            text = normalize_unicode(&text);
        }
        if self.do_remove_html {
            text = remove_html_tags(&text);
        }
        if self.do_remove_urls {
            text = remove_urls(&text);
        }
        if self.do_remove_emails {
            text = remove_emails(&text);
        }
        if self.do_remove_hashtags {
            text = remove_hashtags(&text);
        }
        if self.do_remove_mentions {
            text = remove_mentions(&text);
        }
        if self.do_expand_contractions {
            text = expand_contractions(&text);
        }
        if self.do_expand_ip_addresses {
            text = expand_ip_addresses(&text);
        }
        if self.do_normalize_leading_decimals {
            text = normalize_leading_decimals(&text);
        }
        if self.do_expand_currency {
            text = expand_currency(&text);
        }
        if self.do_expand_percentages {
            text = expand_percentages(&text);
        }
        if self.do_expand_scientific_notation {
            text = expand_scientific_notation(&text);
        }
        if self.do_expand_time {
            text = expand_time(&text);
        }
        if self.do_expand_ordinals {
            text = expand_ordinals(&text);
        }
        if self.do_expand_units {
            text = expand_units(&text);
        }
        if self.do_expand_scale_suffixes {
            text = expand_scale_suffixes(&text);
        }
        if self.do_expand_fractions {
            text = expand_fractions(&text);
        }
        if self.do_expand_decades {
            text = expand_decades(&text);
        }
        if self.do_expand_phone_numbers {
            text = expand_phone_numbers(&text);
        }
        if self.do_expand_ranges {
            text = expand_ranges(&text);
        }
        if self.do_expand_model_names {
            text = expand_model_names(&text);
        }
        if self.do_expand_roman_numerals {
            text = expand_roman_numerals(&text);
        }
        if self.do_replace_numbers {
            text = replace_numbers(&text, self.replace_floats);
        }
        if self.do_remove_accents {
            text = remove_accents(&text);
        }
        if self.do_remove_punctuation {
            text = remove_punctuation(&text);
        }
        if self.lowercase {
            text = to_lowercase(&text);
        }
        if self.do_remove_extra_whitespace {
            text = remove_extra_whitespace(&text);
        }

        text
    }
}
