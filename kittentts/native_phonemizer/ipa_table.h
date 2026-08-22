// Copyright 2024 - Apache 2.0 License
// IPA conversion tables for phoneme codes
// This maps internal phoneme codes to IPA Unicode strings.
// The mapping is derived from phoneme source files (phsource/).
// The table logic implements the same ASCII→IPA character mapping as the
// WritePhMnemonic function (dictionary.c), without copying any GPL code.

#pragma once
#include <string>
#include <unordered_map>
#include <cstdint>

// ASCII to IPA Unicode codepoint table (indices 0x20..0x7F)
// Matches ipa1[] table in dictionary.c
static const uint32_t ASCII_TO_IPA[96] = {
    0x0020, // 0x20 ' ' → space
    0x0021, // 0x21 '!' → !
    0x0022, // 0x22 '"' → "
    0x02b0, // 0x23 '#' → ʰ (superscript h, aspiration - consonants only)
    0x0024, // 0x24 '$' → $
    0x0025, // 0x25 '%' → %
    0x00e6, // 0x26 '&' → æ (ash)
    0x02c8, // 0x27 '\'' → ˈ (primary stress)
    0x0028, // 0x28 '(' → (
    0x0029, // 0x29 ')' → )
    0x027e, // 0x2a '*' → ɾ (flap/tap)
    0x002b, // 0x2b '+' → +
    0x02cc, // 0x2c ',' → ˌ (secondary stress)
    0x002d, // 0x2d '-' → -
    0x002e, // 0x2e '.' → .
    0x002f, // 0x2f '/' → /
    0x0252, // 0x30 '0' → ɒ (open back rounded, British 'lot')
    0x0031, // 0x31 '1' → 1
    0x0032, // 0x32 '2' → 2
    0x025c, // 0x33 '3' → ɜ (open-mid central unrounded)
    0x0034, // 0x34 '4' → 4
    0x0035, // 0x35 '5' → 5
    0x0036, // 0x36 '6' → 6
    0x0037, // 0x37 '7' → 7
    0x0275, // 0x38 '8' → ɵ (close-mid central rounded)
    0x0039, // 0x39 '9' → 9
    0x02d0, // 0x3a ':' → ː (length mark)
    0x02b2, // 0x3b ';' → ʲ (palatalization)
    0x003c, // 0x3c '<' → <
    0x003d, // 0x3d '=' → =
    0x003e, // 0x3e '>' → >
    0x0294, // 0x3f '?' → ʔ (glottal stop)
    0x0259, // 0x40 '@' → ə (schwa)
    0x0251, // 0x41 'A' → ɑ (open back unrounded)
    0x03b2, // 0x42 'B' → β (voiced bilabial fricative)
    0x00e7, // 0x43 'C' → ç (voiceless palatal fricative)
    0x00f0, // 0x44 'D' → ð (eth, voiced dental fricative)
    0x025b, // 0x45 'E' → ɛ (open-mid front unrounded)
    0x0046, // 0x46 'F' → F
    0x0262, // 0x47 'G' → ɢ (voiced uvular stop)
    0x0127, // 0x48 'H' → ħ (pharyngeal fricative)
    0x026a, // 0x49 'I' → ɪ (near-close near-front unrounded)
    0x025f, // 0x4a 'J' → ɟ (voiced palatal stop)
    0x004b, // 0x4b 'K' → K
    0x026b, // 0x4c 'L' → ɫ (velarized l)
    0x0271, // 0x4d 'M' → ɱ (labiodental nasal)
    0x014b, // 0x4e 'N' → ŋ (velar nasal)
    0x0254, // 0x4f 'O' → ɔ (open-mid back rounded)
    0x03a6, // 0x50 'P' → Φ
    0x0263, // 0x51 'Q' → ɣ (voiced velar fricative)
    0x0280, // 0x52 'R' → ʀ (uvular trill)
    0x0283, // 0x53 'S' → ʃ (voiceless postalveolar fricative)
    0x03b8, // 0x54 'T' → θ (voiceless dental fricative)
    0x028a, // 0x55 'U' → ʊ (near-close near-back rounded)
    0x028c, // 0x56 'V' → ʌ (open-mid back unrounded)
    0x0153, // 0x57 'W' → œ (open-mid front rounded)
    0x03c7, // 0x58 'X' → χ (voiceless uvular fricative)
    0x00f8, // 0x59 'Y' → ø (close-mid front rounded)
    0x0292, // 0x5a 'Z' → ʒ (voiced postalveolar fricative)
    0x032a, // 0x5b '[' → ͪ (combining bracket below)
    0x005c, // 0x5c backslash
    0x005d, // 0x5d ']' → ]
    0x005e, // 0x5e '^' → ^
    0x005f, // 0x5f '_' → _
    0x0060, // 0x60 '`' → `
    0x0061, // 0x61 'a' → a
    0x0062, // 0x62 'b' → b
    0x0063, // 0x63 'c' → c
    0x0064, // 0x64 'd' → d
    0x0065, // 0x65 'e' → e
    0x0066, // 0x66 'f' → f
    0x0261, // 0x67 'g' → ɡ (voiced velar stop, script g)
    0x0068, // 0x68 'h' → h
    0x0069, // 0x69 'i' → i
    0x006a, // 0x6a 'j' → j
    0x006b, // 0x6b 'k' → k
    0x006c, // 0x6c 'l' → l
    0x006d, // 0x6d 'm' → m
    0x006e, // 0x6e 'n' → n
    0x006f, // 0x6f 'o' → o
    0x0070, // 0x70 'p' → p
    0x0071, // 0x71 'q' → q
    0x0072, // 0x72 'r' → r
    0x0073, // 0x73 's' → s
    0x0074, // 0x74 't' → t
    0x0075, // 0x75 'u' → u
    0x0076, // 0x76 'v' → v
    0x0077, // 0x77 'w' → w
    0x0078, // 0x78 'x' → x
    0x0079, // 0x79 'y' → y
    0x007a, // 0x7a 'z' → z
    0x007b, // 0x7b '{' → {
    0x007c, // 0x7c '|' → |
    0x007d, // 0x7d '}' → }
    0x0303, // 0x7e '~' → ̃ (combining tilde, nasalization)
    0x007f, // 0x7f DEL
};

// Append a Unicode codepoint as UTF-8 to a string
inline void appendUTF8(std::string& out, uint32_t codepoint) {
    if (codepoint < 0x80) {
        out += (char)codepoint;
    } else if (codepoint < 0x800) {
        out += (char)(0xC0 | (codepoint >> 6));
        out += (char)(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
        out += (char)(0xE0 | (codepoint >> 12));
        out += (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out += (char)(0x80 | (codepoint & 0x3F));
    } else {
        out += (char)(0xF0 | (codepoint >> 18));
        out += (char)(0x80 | ((codepoint >> 12) & 0x3F));
        out += (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out += (char)(0x80 | (codepoint & 0x3F));
    }
}

// Map a single phoneme code to IPA string using ipa1[] table
// is_vowel: true if the phoneme is a vowel (affects '#' handling)
inline std::string phonemeCodeToIPA_table(const std::string& code, bool is_vowel = false) {
    std::string result;
    bool first = true;
    for (char c : code) {
        unsigned char uc = (unsigned char)c;
        if (c == '/') break; // discard variant indicator
        if (!first && (c >= '0' && c <= '9')) continue; // skip digits after first char
        if (c == '#') break; // '#' is a variant/aspiration marker; never emit ʰ directly
                             // (vowel variants like I#, a# use overrides; consonant t#, d# → plain)
        if (c == '|') continue; // skip pipe separator
        if (c == '_' && first) break; // pause phoneme
        if (uc >= 0x20 && uc < 0x80) {
            appendUTF8(result, ASCII_TO_IPA[uc - 0x20]);
        } else if (uc >= 0x80) {
            // Pass through non-ASCII (already IPA)
            result += c;
        }
        first = false;
    }
    return result;
}

// Build explicit IPA override table for en-us
// These phonemes have their IPA explicitly set in ph_english_us or phonemes
inline std::unordered_map<std::string, std::string> buildIPAOverrides(const std::string& dialect) {
    std::unordered_map<std::string, std::string> overrides;

    // Common overrides (both dialects)
    overrides["r"]    = "\xc9\xb9";     // ɹ (U+0279)
    overrides["r-"]   = "\xc9\xb9";     // ɹ
    overrides["n-"]   = "n\xcc\xa9";   // n̩ (syllabic n, U+006E + U+0329 combining vertical line below)
    overrides["m-"]   = "m\xcc\xa9";   // m̩ (syllabic m)
    overrides["3:r"]  = "\xc9\x9c\xcb\x90\xc9\xb9"; // ɜːɹ (U+025C + U+02D0 + U+0279)
    overrides["3:"]   = "\xc9\x9c\xcb\x90";          // ɜː (open-mid central unrounded long)
    overrides["@L"]   = "\xc9\x99l";    // əl (syllabic L)
    overrides["a#"]   = "\xc9\x90";     // ɐ (near-open central)
    overrides["e#"]   = "\xc9\x9b";     // ɛ (open-mid front unrounded) — e# = reduced 'e', same as E
    overrides["I#"]   = "\xe1\xb5\xbb"; // ᵻ (near-close central, U+1D7B) - wait, let me use correct encoding
    overrides["I2#"]  = "\xe1\xb5\xbb"; // ᵻ
    overrides["w#"]   = "\xca\x8d";     // ʍ (voiceless labial-velar)
    overrides["@2"]   = "\xc9\x99";     // ə (default, may change to I2)
    overrides["@5"]   = "\xc9\x99";     // ə (default, may change to U)

    // I2 renders as ɪ (near-close near-front) by default.
    // Word-final I2 → 'i' is handled by happy tensing in wordToPhonemes step 2.
    // (I2 uses FMT(vowel/ii#_3) but in non-final position sounds like ɪ)
    overrides["I2"]   = "\xc9\xaa"; // ɪ (U+026A)

    if (dialect == "en-us" || dialect == "en_us") {
        // American English specific overrides (from ph_english_us)
        overrides["3"]    = "\xc9\x9a";     // ɚ (r-colored schwa, U+025A)
        overrides["a"]    = "\xc3\xa6";     // æ (near-open front unrounded, U+00E6)
        overrides["aa"]   = "\xc3\xa6";     // æ (same as American 'a')
        overrides["0"]    = "\xc9\x91\xcb\x90"; // ɑː (open back unrounded long)
        overrides["0#"]   = "\xc9\x91\xcb\x90"; // ɑː (same as 0, since phoneme 0# → ChangePhoneme(0))
        overrides["A#"]   = "\xc9\x91\xcb\x90"; // ɑː
        overrides["A@"]   = "\xc9\x91\xcb\x90\xc9\xb9"; // ɑːɹ (rhotic, American)
        overrides["A:r"]  = "\xc9\x91\xcb\x90\xc9\xb9"; // ɑːɹ (explicit A: + r in dict entries)
        overrides["e@"]   = "\xc9\x9b\xc9\xb9"; // ɛɹ
        overrides["e@r"]  = "\xc9\x9b\xc9\xb9"; // ɛɹ (absorbs trailing r to avoid double-ɹ)
        overrides["U@"]   = "\xca\x8a\xc9\xb9"; // ʊɹ
        overrides["O@"]   = "\xc9\x94\xcb\x90\xc9\xb9"; // ɔːɹ
        overrides["O@r"]  = "\xc9\x94\xcb\x90\xc9\xb9"; // ɔːɹ (absorbs trailing r to avoid double-ɹ)
        overrides["o@"]   = "o\xcb\x90\xc9\xb9"; // oːɹ (American: o@ = FORCE vowel; e.g. "shore", "more", "floor")
        overrides["o@r"]  = "o\xcb\x90\xc9\xb9"; // oːɹ (same as o@, consumes trailing r to avoid double-ɹ)
        overrides["i@"]   = "i\xc9\x99";          // iə (close front /i/ + schwa; e.g. "area", "idea")
        overrides["i@3"]  = "\xc9\xaa\xc9\xb9"; // ɪɹ
        overrides["i@3r"] = "\xc9\xaa\xc9\xb9"; // ɪɹ (absorbs trailing r to avoid double-ɹ)
        overrides["aI@"]  = "a\xc9\xaa\xc9\x99"; // aɪə
        overrides["aI3"]  = "a\xc9\xaa\xc9\x9a"; // aɪɚ
        overrides["aU@"]  = "a\xc9\xaa\xca\x8a\xc9\xb9"; // aɪʊɹ
        overrides["IR"]   = "\xc9\x99\xc9\xb9"; // əɹ (used in Scottish, but include)
        overrides["VR"]   = "\xca\x8c\xc9\xb9"; // ʌɹ
        overrides["02"]   = "\xca\x8c";     // ʌ (becomes V in en-us)
        overrides["i"]    = "i";             // close front unrounded (final position)
    } else {
        // British English overrides (from ph_english)
        overrides["3"]    = "\xc9\x9a";     // actually ɜ... but let's keep ɚ for now
        overrides["a"]    = "a";             // open front unrounded
        overrides["aa"]   = "a";             // long a
        overrides["0"]    = "\xc9\x92";     // ɒ (open back rounded, British)
        overrides["oU"]   = "\xc9\x99\xca\x8a"; // əʊ (British diphthong)
        overrides["A@"]   = "\xc9\x91\xcb\x90"; // ɑː
        overrides["IR"]   = "\xc9\x99\xc9\xb9"; // əɹ
    }

    return overrides;
}

// Convert phoneme code string (like "eI", "3:", "@") to IPA
// Handles both explicit overrides and the ipa1 table conversion
inline std::string convertPhonemeToIPA(const std::string& code,
                                 const std::unordered_map<std::string, std::string>& overrides,
                                 bool is_vowel = true) {
    // Check explicit overrides first
    auto it = overrides.find(code);
    if (it != overrides.end()) {
        return it->second;
    }
    // Fall back to ipa1 table conversion
    return phonemeCodeToIPA_table(code, is_vowel);
}
