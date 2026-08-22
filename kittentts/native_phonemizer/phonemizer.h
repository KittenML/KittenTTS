// Copyright 2024 - Apache 2.0 License
// Main IPA phonemizer interface
// Reads rule and dictionary files to phonemize English text to IPA.

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "rule_parser.h"
#include "ipa_table.h"

// ============================================================
// Phoneme token - represents one phoneme code from the rules
// ============================================================
struct PhToken {
    enum Type { STRESS_PRIMARY, STRESS_SECONDARY, PHONEME, PAUSE, SYLLABLE };
    Type type;
    std::string code;   // phoneme code (e.g., "eI", "@", "t")
    bool is_vowel;

    PhToken(Type t, const std::string& c, bool v = false)
        : type(t), code(c), is_vowel(v) {}
};

// ============================================================
// Main phonemizer class
// ============================================================
class IPAPhonemizer {
public:
    // dialect: "en-us" (American) or "en-gb" (British)
    // rules_path: path to en_rules file
    // list_path: path to en_list file
    explicit IPAPhonemizer(const std::string& rules_path,
                               const std::string& list_path,
                               const std::string& dialect = "en-us");

    // Phonemize a list of texts; returns IPA strings (one per input)
    // Matches the interface: phonemizer.phonemize([text])
    std::vector<std::string> phonemize(const std::vector<std::string>& texts);

    // Phonemize a single text string
    std::string phonemizeText(const std::string& text) const;

    bool isLoaded() const { return loaded_; }
    const std::string& getError() const { return error_; }

private:
    std::string dialect_;
    bool loaded_;
    std::string error_;

    // Dictionary: word → raw phoneme code string
    std::unordered_map<std::string, std::string> dict_;

    // Verb-form dictionary: word → phoneme string for the verb pronunciation ($verb entries).
    // Used for -ing/-ed suffix stripping where the verb form is needed (e.g. "live"→"lIv"
    // not "laIv", so "living"→"lˈɪvɪŋ" not "lˈaɪvɪŋ").
    std::unordered_map<std::string, std::string> verb_dict_;

    // Past-tense pronunciation: loaded from $past entries in en_list.
    // Only used when expect_past > 0 (i.e., after a $pastf word like "was", "were", "had").
    // e.g., "read $past" → rEd = ɹˈɛd (past) vs ri:d = ɹˈiːd (present/default).
    std::unordered_map<std::string, std::string> past_dict_;

    // Noun-context pronunciation: loaded from $noun entries in en_list (not stored in dict_).
    // Only used when expect_noun > 0 (i.e., after a $nounf word like "a", "my", "the", etc.).
    // e.g., "elaborate $noun" → I#lab3@t = ᵻlˈæbɚɹˌɪt; without noun context rules fire instead.
    std::unordered_map<std::string, std::string> noun_dict_;

    // Words with $pastf flag: trigger expect_past for the following word(s).
    // e.g., "was", "were", "is", "are", "been", "had", "have" (as auxiliaries).
    std::unordered_set<std::string> pastf_words_;

    // Words with $nounf flag: trigger expect_noun for the following word(s).
    // e.g., "a", "every", "my", "his", "her", "its", "our", "your", "their", "some", etc.
    std::unordered_set<std::string> nounf_words_;

    // Words with $verbf flag: trigger expect_verb for the following word(s).
    // e.g., "I", "we", "you", "they", "will", "would", "shall", "should", "to", etc.
    std::unordered_set<std::string> verbf_words_;

    // Rule set
    RuleSet ruleset_;

    // IPA override table
    std::unordered_map<std::string, std::string> ipa_overrides_;

    // Words that carry the $u (typically unstressed) flag in the dictionary
    std::unordered_set<std::string> unstressed_words_;

    // Words with $unstressend flag: stay at secondary stress even when utterance-final.
    // e.g. "ones w02nz $only $unstressend" — keeps ˌ at sentence end, not promoted to ˈ.
    std::unordered_set<std::string> unstressend_words_;

    // Words with $abbrev flag: always read as individual letter names when all-caps
    std::unordered_set<std::string> abbrev_words_;

    // Stressed syllable position from $N dict flags (1-based; 0 = not set)
    std::unordered_map<std::string, int> stress_pos_;

    // Word-level alt flags from $altN dict entries (bitmask: bit N-1 set when word has $altN)
    std::unordered_map<std::string, int> word_alt_flags_;

    // Dictionary for $atstart entries: used only when word is the first in the utterance.
    std::unordered_map<std::string, std::string> atstart_dict_;

    // Dictionary for $atend entries: used only when word is the last in the utterance.
    // e.g. "to tu: $u $atend" → "to" at sentence end gets full form tuː (not reduced tə).
    std::unordered_map<std::string, std::string> atend_dict_;

    // Dictionary for $capital entries: used only when word starts with a capital letter.
    // e.g. "Bologna b@loUn;@ $capital" → used for "Bologna" (city) not "bologna" (sausage).
    std::unordered_map<std::string, std::string> capital_dict_;

    // Words with $onlys flag: dict entry is only valid for the bare form or with 's' suffix.
    // When stripping non-s suffixes (e.g. -ed, -ing, -able), skip these dict entries.
    std::unordered_set<std::string> onlys_words_;

    // Bare-word override from $onlys entries that coexist with a plain (no-flag) entry.
    // e.g. "desert dEz3t $onlys" overrides "desert dI#z3:t" for bare-word lookup.
    // Suffix stripping still uses dict_ (the plain entry) for stems.
    std::unordered_map<std::string, std::string> onlys_bare_dict_;

    // Words with $only flag: dict entry is only valid for the isolated bare word form.
    // Should NOT be used as a stem for any suffix stripping (even -s).
    // E.g. "guid" has $only → should not suppress magic-e when processing "guiding"/"guided".
    std::unordered_set<std::string> only_words_;

    // Words where stress_pos_ came from a flag-only $N $onlys entry (noun-form-only stress).
    // These stress overrides should NOT be applied when phonemizing verb-derived stems
    // (e.g. "construct $1 $onlys" → noun has 1st-syll stress, but "constructing" uses verb rules).
    std::unordered_set<std::string> noun_form_stress_;

    // Words with a $verb flag-only en_list entry (no phoneme, just marks verb-context override).
    // These words have a separate verb pronunciation governed by rules, NOT by stress_pos_.
    // E.g. "conduct $verb" → verb form uses rules (2nd-syllable stress), not $1 noun-form stress.
    std::unordered_set<std::string> verb_flag_words_;

    // Compound prefix words: entries with $strend2 and bare (unstressed) phoneme.
    // Sorted by word length descending for longest-match-first.
    // e.g., "under" → "Vnd3", "over" → "oUv3", "through" → "Tru:"
    std::vector<std::pair<std::string, std::string>> compound_prefixes_;

    // Words with $strend2 and bare phoneme (no leading stress marker) that need
    // final-syllable (pick_last) stress placement in processPhonemeString step 5.
    // Mirrors compound_prefixes_ but as a set for O(1) lookup.
    // e.g., "become" (bIkVm), "within" (wIDIn), "without" (wIDaUt).
    std::unordered_set<std::string> strend_words_;

    // Words with BOTH $u2 AND $strend2 flags: function words that should carry
    // secondary stress (not primary) in sentence context (e.g. "together", "across").
    std::unordered_set<std::string> u2_strend2_words_;

    // Words with $strend2 whose dict phoneme starts with ',' (secondary-stressed).
    // These are NOT in strend_words_ (which requires bare/unstressed phoneme).
    // Like KEEP_SECONDARY, they stay secondary when followed by stressed content,
    // but get promoted to primary when phrase-final. E.g., "go" (,goU), "so" (,soU),
    // "up" (,Vp), "down" (,daUn), "doing" (,du:IN), "should" (,SUd), "might" (,maIt).
    std::unordered_set<std::string> comma_strend2_words_;

    // Words with $u+ flag whose dict phoneme has ',' (secondary) but no '\'' (primary).
    // These should keep secondary stress in sentence context (not promoted to primary).
    // e.g. "made" (m,eId $u+): keeps ˌ in sentence context.
    std::unordered_set<std::string> u_plus_secondary_words_;

    // Phrase dictionary: "word1 word2" → phoneme string for multi-word phrases.
    // Loaded from parenthesized entries in en_list (e.g. "(for the) f3D@2 $nounf").
    // Used for bigram cliticization in sentence context.
    std::unordered_map<std::string, std::string> phrase_dict_;

    // Split-phrase dictionary: "word1 word2" → (phoneme1, phoneme2) for phrases where
    // each word has its own phoneme separated by || in the en_list entry.
    // E.g., "(most of) moUst||@v" → phrase_split_dict_["most of"] = {"moUst", "@v"}.
    // (too much) t'u:||mVtS → "too much" → {"t'u:", "mVtS"}.
    std::unordered_map<std::string, std::pair<std::string,std::string>> phrase_split_dict_;

    // Phrase keys that should behave like KEEP_SECONDARY: secondary in sentence context
    // (when followed by stressed content), primary when phrase-final.
    // Loaded from phrase entries with $u2+ flag (e.g. "do not", "did not", "does not").
    std::unordered_set<std::string> keep_sec_phrase_keys_;

    // Load the word list (en_list)
    bool loadDictionary(const std::string& path);

    // Load the rules file (en_rules)
    bool loadRules(const std::string& path);

    // Normalize text for processing
    std::string normalizeText(const std::string& text) const;

    // Split text into words/tokens preserving punctuation
    struct Token {
        std::string text;
        bool is_word;
        bool needs_space_before;
    };
    std::vector<Token> tokenizeText(const std::string& text) const;

    // Get phoneme codes for a single word
    std::string wordToPhonemes(const std::string& word) const;

    // Apply rules to a word (for unknown words)
    // word_alt_flags: bitmask of $altN flags active for this word (-1 = look up from word_alt_flags_)
    // suffix_phoneme_only: when true, RULE_ENDING rules contribute their phoneme normally
    // (no stem re-phonemization). Mimics TranslateRules(word, NULL) behavior where
    // RULE_ENDING fires but doesn't trigger early return — the suffix phoneme is appended
    // to the accumulated first-pass phonemes rather than re-phonemizing the extracted stem.
    // Used when re-translating stems that contain further suffix rules (e.g. "ribosome"
    // contains "-some" RULE_ENDING; with suffix_phoneme_only=true, "ribosome" gives first-pass
    // phonemes rIb0soUm rather than re-phonemized ri:boUsoUm).
    std::string applyRules(const std::string& word, bool allow_suffix_strip = true,
                           int word_alt_flags = -1,
                           bool suffix_phoneme_only = false,
                           bool suffix_removed = false,
                           std::vector<bool>* out_replaced_e = nullptr,
                           std::vector<bool>* out_pos_visited = nullptr) const;

    // Apply replace rules (preprocessing)
    std::string applyReplacements(const std::string& word) const;

    // Match a single rule at position pos in word
    // group_length: 1 for single-char groups, 2 for two-char groups
    // word_alt_flags: bitmask of $altN flags for $w_altN right-context matching
    int matchRule(const PhonemeRule& rule, const std::string& word, int pos,
                   std::string& out_phonemes, int& advance, int& del_fwd_start, int& del_fwd_count,
                   int group_length = 1,
                   const std::string& phonemes_so_far = "",
                   int word_alt_flags = 0,
                   const std::vector<bool>* replaced_e_arr = nullptr,
                   bool suffix_removed = false) const;

    // Match left context (scan backward from pos-1)
    bool matchLeftContext(const std::string& ctx_str, const std::string& word, int pos) const;

    // Match right context (scan forward from pos+match_len)
    bool matchRightContext(const std::string& ctx_str, const std::string& word, int pos,
                           int& del_fwd_count) const;

    // Check a single context character against word position
    bool matchContextChar(char ctx_char, const std::string& word, int word_pos,
                          bool at_word_start, bool at_word_end) const;

    // Parse phoneme code string into IPA
    std::string phonemesToIPA(const std::string& phoneme_str) const;

    // Convert a single phoneme code to IPA
    std::string singleCodeToIPA(const std::string& code) const;

    // Check if a phoneme code is a vowel
    bool isVowelCode(const std::string& code) const;

    // Post-process phoneme string for dialect
    std::string postProcessPhonemes(const std::string& phonemes) const;

    // Apply stress and clean up the phoneme string.
    // force_final_stress: when true, place primary on LAST stressable vowel (for $strend2 words).
    std::string processPhonemeString(const std::string& raw, bool force_final_stress = false) const;

    // Apply $N stress position override: force primary on Nth vowel (1-based)
    std::string applyStressPosition(const std::string& raw, int n) const;
};
