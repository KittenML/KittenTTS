// Copyright 2024 - Apache 2.0 License
// Parser for letter-to-phoneme rule files (en_rules format)
// This reads the GPL-licensed rule files but implements its own parsing
// and matching logic, independent of .

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <cstdint>

// ============================================================
// Character group definitions
// Built-in letter groups (A,B,C,F,G,H,Y in rule context)
// as defined by SetLetterBits() in tr_languages.c
// ============================================================
struct LetterGroups {
    std::set<char> groupA;  // A: vowels (aeiou)
    std::set<char> groupB;  // B: hard consonants (bcdfgjklmnpqstvxz)
    std::set<char> groupC;  // C: all consonants (bcdfghjklmnpqrstvwxz)
    std::set<char> groupF;  // F: voiceless consonants (cfhkpqstx)
    std::set<char> groupG;  // G: voiced consonants (bdgjlmnrvwyz)
    std::set<char> groupH;  // H: sonorants/soft consonants (hlmnr)
    std::set<char> groupY;  // Y: front vowels (eiy)
    std::set<char> groupK;  // K: non-vowels
    // User-defined L-groups (L01-L11 from .L## directives)
    // Each L-group can contain multi-char items (like "bE", "dE")
    std::vector<std::string> lgroups[100]; // lgroups[n] = items in L(n+1)

    void init() {
        // From tr_languages.c NewTranslator() SetLetterBits() calls
        // Group 0 (A): vowels
        for (char c : std::string("aeiou")) groupA.insert(c);
        // Group 1 (B): hard consonants (all consonants except h, r, w, y)
        for (char c : std::string("bcdfgjklmnpqstvxz")) groupB.insert(c);
        // Group 2 (C): all consonants
        for (char c : std::string("bcdfghjklmnpqrstvwxz")) groupC.insert(c);
        // Group 3 (H): sonorants/liquids
        for (char c : std::string("hlmnr")) groupH.insert(c);
        // Group 4 (F): voiceless consonants
        for (char c : std::string("cfhkpqstx")) groupF.insert(c);
        // Group 5 (G): voiced consonants
        for (char c : std::string("bdgjlmnrvwyz")) groupG.insert(c);
        // Group 6 (Y): for English, all vowels including y
        // (tr_languages.c L('e','n') case: SetLetterBits(tr, 6, "aeiouy"))
        for (char c : std::string("aeiouy")) groupY.insert(c);

        // K = non-vowels (complement of A)
        for (char c : std::string("bcdfghjklmnpqrstvwxyz")) groupK.insert(c);
    }

    bool isVowel(char c) const {
        return groupA.count(std::tolower(c)) > 0;
    }

    bool matchGroup(char groupChar, const std::string& word, int pos) const {
        if (pos < 0 || pos >= (int)word.size()) return false;
        char c = std::tolower(word[pos]);
        switch (groupChar) {
            case 'A': return groupA.count(c) > 0;
            case 'B': return groupB.count(c) > 0;
            case 'C': return groupC.count(c) > 0;
            case 'F': return groupF.count(c) > 0;
            case 'G': return groupG.count(c) > 0;
            case 'H': return groupH.count(c) > 0;
            case 'Y': return groupY.count(c) > 0;
            case 'K': return groupK.count(c) > 0;
            default: return false;
        }
    }
};

// ============================================================
// A single phoneme rule
// ============================================================
struct PhonemeRule {
    // Dialect condition: 0=always, 3=en-us, -3=not en-us, etc.
    int condition;
    bool condition_negated;

    // Left context (stored in natural order, scanned backward from match pos-1)
    std::string left_ctx;

    // The string to match (starting with the group key letter(s))
    std::string match;

    // Right context (scanned forward from match_end)
    std::string right_ctx;

    // Phoneme output
    std::string phonemes;

    // Characters to delete forward (from DEL_FWD '#' in right context)
    int del_fwd; // number of characters to delete after match

    // True if right context contains '@P' — a prefix rule that triggers re-translation
    // of the suffix as a new word (SUFX_P / RULE_ENDING mechanism).
    bool is_prefix;

    // True if right context contains '_S<N>' — a suffix-stripping rule.
    // When this rule fires at word end, strip suffix_strip_len chars from the word end,
    // re-phonemize the stem, and combine stem phonemes + this rule's phonemes.
    bool is_suffix;
    int suffix_strip_len;   // number of chars to strip from word end (from _SN)
    // SUFX_I (0x200): stem may have had 'y'→'i' change; restore 'i'→'y' before re-phonemizing
    int suffix_flags;

    PhonemeRule() : condition(0), condition_negated(false), del_fwd(0),
                    is_prefix(false), is_suffix(false), suffix_strip_len(0), suffix_flags(0) {}
};

// ============================================================
// Replacement rules (from .replace section)
// ============================================================
struct ReplaceRule {
    std::string from;
    std::string to;
};

// ============================================================
// Main rule set structure
// ============================================================
struct RuleSet {
    LetterGroups groups;

    // Replacement rules
    std::vector<ReplaceRule> replacements;

    // Rule groups indexed by group key (1 or 2 character string)
    // groups1[key] = list of rules for that key
    std::unordered_map<std::string, std::vector<PhonemeRule>> rule_groups;

    void init() {
        groups.init();
    }
};

// ============================================================
// Parse a context string (left or right context from rule file)
// Returns a list of context "tokens" in the order they appear
// ============================================================
struct CtxToken {
    enum Type {
        LITERAL,        // literal character(s)
        WORD_BOUNDARY,  // _ word boundary
        LETTER_GROUP,   // built-in group A,B,C,F,G,H,Y,K
        L_GROUP,        // user-defined L-group (Lxx)
        NO_VOWELS,      // X: no vowels to word end
        NOT_VOWEL,      // K: not a vowel
        DEL_FWD,        // # delete forward
        DIGIT,          // D digit
        NONALPHA,       // Z non-alphabetic
        STRESSED,       // & stressed
        ANY_VOWEL,      // A
        SYLLABLE,       // @ syllable marker
        CAPITAL,        // ! capital letter
        SUFFIX,         // S/P suffix rule (complex, mostly ignored)
        DOLLAR,         // $ dollar rule (mostly ignored for now)
        INC_SCORE,      // +
        DEC_SCORE,      // <
        SKIP_CHARS,     // J
    };
    Type type;
    char group_char;    // for LETTER_GROUP
    int l_group_id;     // for L_GROUP (1-based)
    std::string literal; // for LITERAL
    int count;          // for repeating matches (like +++ = 3 times)

    CtxToken() : type(LITERAL), group_char(0), l_group_id(0), count(1) {}
};

// Parse a context string from the rule file into tokens
std::vector<CtxToken> parseContext(const std::string& ctx, bool& has_del_fwd);

// ============================================================
// Rule file parser
// ============================================================
class RuleParser {
public:
    RuleParser();

    // Parse the en_rules file
    bool parseRuleFile(const std::string& filename, RuleSet& ruleset);

private:
    void parseGroupLine(const std::string& line, const std::string& current_group,
                        RuleSet& ruleset, int dialect_filter);

    // Parse a single rule line (may have leading condition like ?3)
    bool parseRuleLine(const std::string& line, PhonemeRule& rule, int& dialect_filter);

    // Tokenize context pattern
    std::vector<CtxToken> tokenizeContext(const std::string& ctx);
};
