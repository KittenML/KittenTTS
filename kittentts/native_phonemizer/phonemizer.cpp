// Copyright 2024 - Apache 2.0 License
// IPA Phonemizer implementation that reads the reference rule files.
// Original implementation - not derived from the reference GPL source code.

#include "phonemizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cassert>
#include <iostream>
#include <unordered_set>

// ============================================================
// UTF-8 / String helpers
// ============================================================
static std::string toLowerASCII(const std::string& s) {
    std::string result = s;
    for (char& c : result)
        c = (char)std::tolower((unsigned char)c);
    return result;
}

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> splitWS(const std::string& s) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token) result.push_back(token);
    return result;
}

static bool isVowelLetter(char c) {
    c = (char)std::tolower((unsigned char)c);
    return c=='a'||c=='e'||c=='i'||c=='o'||c=='u'||c=='y';
}

// ============================================================
// Dictionary (en_list) Parser
// ============================================================
bool IPAPhonemizer::loadDictionary(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        error_ = "Cannot open dictionary file: " + path;
        return false;
    }

    bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");

    std::string line;
    while (std::getline(f, line)) {
        // Remove comments
        size_t comment = line.find("//");
        if (comment != std::string::npos)
            line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        // Handle parenthesized multi-word phrase entries: "(word1 word2) phonemes [flags]"
        // These are phrase-level pronunciations used for bigram cliticization in sentences.
        // Only store simple 2-word phrases (no dots, no abbreviations) that have a phoneme.
        if (!line.empty() && line[0] == '(') {
            size_t close = line.find(')');
            if (close != std::string::npos && close > 1) {
                std::string phrase_words = trim(line.substr(1, close - 1));
                std::string rest = trim(line.substr(close + 1));
                // Parse phoneme from rest (first non-flag token)
                if (!rest.empty() && rest[0] != '$') {
                    std::vector<std::string> rp = splitWS(rest);
                    if (!rp.empty() && !rp[0].empty() && rp[0][0] != '$') {
                        // Only handle 2-word phrases (no dots in words, no pipes in phoneme)
                        // that have a normal phoneme (no || separators in phoneme string,
                        // those are complex pause-separated phrases we skip for now).
                        std::vector<std::string> words = splitWS(phrase_words);
                        bool has_atend = false, has_pause = false;
                        for (size_t ri = 1; ri < rp.size(); ri++) {
                            if (rp[ri] == "$atend") has_atend = true;
                            if (rp[ri] == "$pause") has_pause = true;
                        }
                        bool has_u2_plus = false;
                        for (size_t ri = 1; ri < rp.size(); ri++) {
                            if (rp[ri] == "$u2+") has_u2_plus = true;
                        }
                        if (words.size() == 2 && !has_atend && !has_pause &&
                            words[0].find('.') == std::string::npos &&
                            words[1].find('.') == std::string::npos) {
                            std::string key = toLowerASCII(words[0]) + " " + toLowerASCII(words[1]);
                            if (rp[0].find("||") != std::string::npos) {
                                // Split phrase: each word has its own phoneme separated by ||.
                                // Store as a pair for independent per-word phonemization.
                                // E.g. "(most of) moUst||@v" → {"moUst", "@v"}.
                                size_t pipe = rp[0].find("||");
                                phrase_split_dict_.emplace(key,
                                    std::make_pair(rp[0].substr(0, pipe),
                                                   rp[0].substr(pipe + 2)));
                            } else {
                                // If the phoneme has no explicit primary stress '\'' and no leading
                                // unstressed marker '%', prepend '%' to prevent last-resort stress
                                // insertion in processPhonemeString (these are function-word phrases
                                // that should remain unstressed in sentence context).
                                std::string phoneme = rp[0];
                                if (phoneme.find('\'') == std::string::npos && phoneme[0] != '%')
                                    phoneme = "%" + phoneme;
                                phrase_dict_.emplace(key, phoneme);  // first-entry-wins
                                // $u2+ phrases (e.g. "do not", "did not"): secondary in sentence
                                // context when followed by stressed content, primary phrase-final.
                                if (has_u2_plus)
                                    keep_sec_phrase_keys_.insert(key);
                            }
                        }
                    }
                }
            }
            continue;  // skip further processing for phrase entries
        }

        // Check for dialect condition
        int dialect_cond = 0;
        bool cond_negated = false;
        if (line[0] == '?') {
            size_t space = line.find_first_of(" \t");
            if (space != std::string::npos) {
                std::string cond_str = line.substr(1, space - 1);
                if (!cond_str.empty() && cond_str[0] == '!') {
                    cond_negated = true;
                    cond_str = cond_str.substr(1);
                }
                try { dialect_cond = std::stoi(cond_str); } catch(...) {}
                line = trim(line.substr(space));
            }
        }

        // Dialect filter
        // ?3 = General American (en-us); ?6 = American variant (one/of forms, also en-us)
        if (dialect_cond != 0) {
            bool match = false;
            if (dialect_cond == 3 || dialect_cond == 6) match = is_en_us;
            bool applies = cond_negated ? !match : match;
            if (!applies) continue;
        }

        // Parse word and phonemes
        std::vector<std::string> parts = splitWS(line);
        if (parts.size() < 2) continue;

        std::string word = parts[0];
        std::string phonemes_str = parts[1];

        // Normalize word to lowercase (needed before flag parsing)
        std::string norm_word = toLowerASCII(word);

        // Parse all flags.
        // POS-conditional pronunciation flags: these only apply in specific grammatical context.
        // Without POS context (isolation), skip such entries and fall back to rules.
        bool has_noun_flag = false;  // $noun: noun pronunciation only
        bool has_verb_flag = false;  // $verb: verb pronunciation only
        bool has_pastf_flag = false; // $pastf: this word sets expect_past for following words
        bool has_nounf_flag = false; // $nounf: this word sets expect_noun for following words
        bool has_verbf_flag = false; // $verbf: this word sets expect_verb for following words
        // $pastf: grammatical label (not a skip condition) — was/were/be/are use these in isolation
        bool has_past_flag = false;  // $past: tense-conditional (read past=rɛd vs present=riːd)
        bool has_atend_flag = false; // $atend/$allcaps: clause-final / all-caps context
        bool has_capital_flag = false; // $capital: only when word starts with capital letter
        bool has_atstart_flag = false; // $atstart: only at start of utterance
        bool has_onlys_flag = false;  // $onlys: only when followed by 's' — skip as default
        bool has_only_flag = false;   // $only: bare word only, not as suffix-stripping stem
        // Sentence-level modifier flags (don't prevent loading, but block $N stress storage):
        bool has_grammar_flag = false;
        int stress_n = 0;    // from $N flags (N=1..6)
        int stress_u_n = 0;  // from $uN flags: stressed syllable for weak words

        auto parseStressN = [](const std::string& flag) -> int {
            if (flag.size() == 2 && flag[0] == '$' &&
                flag[1] >= '1' && flag[1] <= '6')
                return flag[1] - '0';
            return 0;
        };
        auto parseStressUN = [](const std::string& flag) -> int {
            // $u1, $u2, $u3, $u1+, $u2+, $u3+ → returns N (stressed syllable)
            if (flag.size() >= 3 && flag[0] == '$' && flag[1] == 'u') {
                char nc = flag[2];
                if (nc >= '1' && nc <= '6') return nc - '0';
            }
            return 0;
        };

        // Check phonemes_str first (for flag-only entries like "lemonade $3")
        if (!phonemes_str.empty() && phonemes_str[0] == '$') {
            stress_n = parseStressN(phonemes_str);
        }

        // Check all flags
        bool has_strend2_flag = false;
        bool has_u2_flag = false;
        for (size_t fi = 2; fi < parts.size(); fi++) {
            const std::string& f = parts[fi];
            if (f == "$noun")    has_noun_flag = true;
            if (f == "$verb")    { has_verb_flag = true; has_grammar_flag = true; }
            // $past: tense-conditional (e.g. "read" past=rɛd vs present=riːd).
            if (f == "$past")    has_past_flag = true;
            // $pastf: marks auxiliaries that trigger expect_past for the following word(s).
            if (f == "$pastf")   has_pastf_flag = true;
            // $nounf: marks determiners/possessives that trigger expect_noun for next word(s).
            if (f == "$nounf")   { has_nounf_flag = true; has_grammar_flag = true; }
            // $verbf: marks subjects/modals that trigger expect_verb for next word(s).
            if (f == "$verbf")   { has_verbf_flag = true; has_grammar_flag = true; }
            if (f == "$atend" || f == "$allcaps" ||
                f == "$sentence") has_atend_flag = true;
            if (f == "$capital") has_capital_flag = true;
            if (f == "$atstart") has_atstart_flag = true;
            if (f == "$verbf" || f == "$strend2" ||
                f == "$alt2"   || f == "$alt3"  || f == "$only")
                has_grammar_flag = true;
            if (f == "$only") has_only_flag = true;
            if (f == "$onlys") has_onlys_flag = true;

            if (f == "$strend2") has_strend2_flag = true;
            if (f == "$u2") has_u2_flag = true;
            if (f == "$u+") {
                unstressed_words_.insert(norm_word);
                // Track $u+ words whose phoneme has ',' but no '\'' — secondary in sentences.
                if (phonemes_str.find(',') != std::string::npos &&
                    phonemes_str.find('\'') == std::string::npos)
                    u_plus_secondary_words_.insert(norm_word);
            }
            if (f == "$u")  unstressed_words_.insert(norm_word);
            if (f == "$unstressend") unstressend_words_.insert(norm_word);
            if (f == "$abbrev") abbrev_words_.insert(norm_word);
            // $altN flags (N=1..6): stored as bitmask for $w_altN rule context matching
            if (f.size() == 5 && f[0]=='$' && f[1]=='a' && f[2]=='l' && f[3]=='t' &&
                f[4] >= '1' && f[4] <= '6')
                word_alt_flags_[norm_word] |= (1 << (f[4] - '1'));
            // $u1/$u2/$u3 are pronunciation-variant selectors, NOT unstressed markers.
            // Do NOT insert into unstressed_words_ for these.
            (void)parseStressUN(f); // parsed but not used for unstressed_words_
            if (!stress_n) stress_n = parseStressN(f);
        }
        // Also check if phonemes_str itself is "$abbrev" (e.g. "gi\t$abbrev\t$allcaps")
        if (phonemes_str == "$abbrev") abbrev_words_.insert(norm_word);
        // $altN as phoneme_str (e.g. "graduate  $alt2")
        if (phonemes_str.size() == 5 && phonemes_str[0]=='$' && phonemes_str[1]=='a' &&
            phonemes_str[2]=='l' && phonemes_str[3]=='t' && phonemes_str[4] >= '1' && phonemes_str[4] <= '6')
            word_alt_flags_[norm_word] |= (1 << (phonemes_str[4] - '1'));
        // Also detect grammar flags and unstress flags in phonemes_str
        if (phonemes_str == "$verb" || phonemes_str == "$verbf" ||
            phonemes_str == "$nounf" || phonemes_str == "$pastf" ||
            phonemes_str == "$only")
            has_grammar_flag = true;
        if (phonemes_str == "$pastf") has_pastf_flag = true;
        if (phonemes_str == "$nounf") has_nounf_flag = true;
        if (phonemes_str == "$verbf") has_verbf_flag = true;

        // Track POS-following context: insert word into pastf/nounf/verbf sets if flagged.
        // These words set expect_past/expect_noun/expect_verb for subsequent words in sentence.
        if (has_pastf_flag) pastf_words_.insert(norm_word);
        if (has_nounf_flag) nounf_words_.insert(norm_word);
        if (has_verbf_flag) verbf_words_.insert(norm_word);
        // Handle $u+ and $u when they appear as the phoneme_str (no explicit phoneme)
        // e.g. "will $u+ $only $verbf $strend2" — $u+ is parts[1] = phonemes_str
        if (phonemes_str == "$u" || phonemes_str == "$u+")
            unstressed_words_.insert(norm_word);
        if (phonemes_str == "$u") has_grammar_flag = true; // skip phoneme entry
        if (phonemes_str == "$verb") has_verb_flag = true;
        // $pastf: not a skip condition — omit

        // Store $N stress position.
        // Normal entries: skip if grammar flag present (stress may be context-dependent).
        // Flag-only entries (phonemes_str starts with '$', e.g. "?3 cannot $2 $verbf"):
        //   the $N IS the pronunciation fact — store it even with grammar flags.
        bool is_flag_only = (!phonemes_str.empty() && phonemes_str[0] == '$');
        if (stress_n > 0 && !has_noun_flag && !has_verb_flag) {
            if (!has_grammar_flag || is_flag_only) {
                stress_pos_.emplace(norm_word, stress_n);  // first-entry-wins
                // Track $N $onlys flag-only entries: the stress override is noun-form-only.
                // Should NOT be applied when phonemizing verb-derived stems (-ing/-ed).
                if (is_flag_only && has_onlys_flag) {
                    noun_form_stress_.insert(norm_word);
                }
            }
        }
        // $uN: variant-selection flags — no stress_pos_ entry stored for these.

        // Skip if phonemes_str starts with $ (word flag only, no phoneme entry)
        if (is_flag_only) {
            // $altN flag-only entry (e.g. "sterile $alt2"): in the reference, this is
            // a last-entry-wins replacement that removes the prior phoneme entry,
            // so the word falls through to rules with the altN flag set.
            // E.g. "sterile stEraIl" (British) is overridden by "sterile $alt2",
            // causing rule "?3 &) ile (_$w_alt2 @L" to fire → American stˈɛɹəl.
            if (phonemes_str.size() == 5 && phonemes_str[0]=='$' && phonemes_str[1]=='a' &&
                phonemes_str[2]=='l' && phonemes_str[3]=='t' && phonemes_str[4] >= '1' && phonemes_str[4] <= '6') {
                dict_.erase(norm_word);  // Remove phoneme so rules fire with alt flag
            }
            // Flag-only $verb entry (no explicit phoneme) means "verb form uses rules".
            // Track these words so stemPh skips the noun-form stress_pos_ override.
            // NOTE: must handle here (before continue) since has_verb_flag check below is unreachable.
            if (has_verb_flag) verb_flag_words_.insert(norm_word);
            continue;
        }

        // Skip POS-conditional entries: store in separate dicts, not the default dict_.
        // $noun entries: only valid in noun-expecting context (after a $nounf word).
        // Without noun context, words fall through to rules (matching the reference behavior).
        if (has_noun_flag) {
            if (!is_flag_only) {
                // Store in noun_dict_; used only when expect_noun > 0 in sentence context.
                noun_dict_.emplace(norm_word, phonemes_str);
            }
            continue;
        }
        if (has_verb_flag) {
            if (!is_flag_only) {
                // Save verb form separately for -ing/-ed suffix stripping (e.g. live→lIv).
                verb_dict_.emplace(norm_word, phonemes_str);
            }
            continue;
        }

        // $atend: only at utterance end — store separately; do NOT overwrite dict_.
        // e.g. "to tu: $u $atend" → atend form tuː used when "to" is the last word.
        // Exception: entries with BOTH $atstart and $atend (only "me mi: $atstart $atend"
        // in en_list) are skipped entirely — the reference uses the $only entry (",mi:") in all
        // normal sentence contexts including first/last word position.
        if (has_atend_flag) {
            if (!has_atstart_flag && !phonemes_str.empty() && phonemes_str[0] != '$')
                atend_dict_[norm_word] = phonemes_str;
            continue;
        }
        // $capital: only when word starts with capital letter — store separately.
        if (has_capital_flag) {
            if (!phonemes_str.empty() && phonemes_str[0] != '$')
                capital_dict_[norm_word] = phonemes_str;
            continue;
        }
        // $atstart: only at utterance start — store separately; do NOT overwrite dict_.
        if (has_atstart_flag) {
            atstart_dict_[norm_word] = phonemes_str;
            continue;
        }
        // $past: tense-conditional (e.g. "read" past=rɛd). Store in past_dict_ for use
        // when expect_past > 0 (after a $pastf word like "was", "were", "had").
        if (has_past_flag) {
            if (!is_flag_only) {
                past_dict_.emplace(norm_word, phonemes_str);
            }
            continue;
        }
        // $onlys: pronunciation used when word is followed by 's' (e.g. plural/3rd-person form).
        // Store in dict_ only if no prior entry exists (don't overwrite a non-$onlys entry).
        // This prevents e.g. "chang tSaN $onlys" (for "changs") from overwriting "chang tSeIndZ"
        // (English "change" pronunciation used for "changed" suffix stripping).
        // But if $onlys is the only entry, it IS the default pronunciation (e.g. "present prEz@nt $onlys").
        if (has_onlys_flag) {
            // Dialect-specific entries (e.g. "?3 record rEk3d $onlys") override unconditional ones.
            // Use overwrite semantics when a dialect condition applies; otherwise first-entry-wins.
            bool newly_inserted;
            if (dialect_cond != 0) {
                dict_[norm_word] = phonemes_str;
                newly_inserted = false;  // may or may not be new, but always mark onlys below
                onlys_words_.insert(norm_word);
            } else {
                newly_inserted = dict_.emplace(norm_word, phonemes_str).second;
                // Only mark as $onlys when this entry IS the primary dict entry (first seen).
                // If a non-$onlys entry already exists (e.g. "chang tSeIndZ" before "chang tSaN $onlys"),
                // do NOT mark the word as onlys — the non-onlys entry covers suffix stripping too.
                if (newly_inserted) {
                    onlys_words_.insert(norm_word);
                } else if (!phonemes_str.empty() && phonemes_str[0] != '$') {
                    // A $onlys entry with phonemes coexists with a prior plain entry.
                    // Store in onlys_bare_dict_ so bare-word lookup uses this pronunciation.
                    // e.g. "desert dEz3t $onlys" overrides "desert dI#z3:t" for bare "desert".
                    // Suffix stripping still uses dict_ (the plain entry).
                    onlys_bare_dict_[norm_word] = phonemes_str;
                }
            }
            (void)newly_inserted;
            continue;
        }
        // Note: $only means "bare word form only" — isolation IS a bare word, so don't skip.
        // Note: $pastf is a grammatical label ("past form"), not a context requirement.
        //       the reference uses $pastf entries as default pronunciations (was, were, be, are, been...).

        // Both dialect-conditional and unconditional entries use last-entry-wins (overwrite).
        // Later entries in en_list are more specific/canonical (e.g. "not noUt // for compounds"
        // is overridden by "not ,n0t $verbextend $only $strend" which is the standalone form).
        dict_[norm_word] = phonemes_str;

        // $only: mark word so its dict entry is NOT used as a stem for suffix stripping.
        // The word's pronunciation IS correct for the isolated form (e.g. "guid" → "ɡuːɪd")
        // but should NOT suppress magic-e when processing derivatives ("guiding" → "guide+ing").
        if (has_only_flag) only_words_.insert(norm_word);

        // Register compound prefix: words with $strend2 whose phoneme is bare (not pre-stressed).
        // These form the first element of compound words with stress on the suffix.
        // e.g., "under" (Vnd3) + "stand" → ˌʌndɚstˈænd ("understand").
        // Exclude entries whose phoneme starts with ','/'\''/'%' (already have stress marker);
        // those don't shift stress to suffix (e.g., "down" = ",daUn", "up" = ",Vp").
        if (has_strend2_flag && norm_word.size() >= 2 &&
            !phonemes_str.empty() &&
            phonemes_str[0] != ',' && phonemes_str[0] != '\'' && phonemes_str[0] != '%') {
            compound_prefixes_.push_back({norm_word, phonemes_str});
            strend_words_.insert(norm_word);
        }
        // Track $strend2 words with leading ',' phoneme: they carry secondary stress in
        // sentences but get promoted to primary when phrase-final (no following stressed word).
        if (has_strend2_flag && !phonemes_str.empty() && phonemes_str[0] == ',')
            comma_strend2_words_.insert(norm_word);
        // Track $u2 $strend2 words: function words that get secondary stress in sentences.
        if (has_u2_flag && has_strend2_flag)
            u2_strend2_words_.insert(norm_word);
    }

    // Sort compound prefixes longest-first for greedy matching.
    std::sort(compound_prefixes_.begin(), compound_prefixes_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    // Post-load: remove content words from unstressed_words_ that should keep stress.
    // "made" has $u+ but the reference stresses it (primary or secondary) in sentence context.
    unstressed_words_.erase("made");

    return true;
}

// ============================================================
// Rule File (en_rules) Parser
// ============================================================
static void parseLGroupDef(const std::string& line, RuleSet& rs) {
    if (line.size() < 3 || line[1] != 'L') return;
    int id = 0;
    size_t i = 2;
    while (i < line.size() && std::isdigit((unsigned char)line[i])) {
        id = id * 10 + (line[i] - '0');
        i++;
    }
    if (id <= 0 || id >= 100) return;

    std::string rest = line.substr(i);
    auto items = splitWS(rest);
    for (auto& item : items) {
        if (item.substr(0,2) == "//") break;
        rs.groups.lgroups[id].push_back(item);
    }
}

bool IPAPhonemizer::loadRules(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        error_ = "Cannot open rules file: " + path;
        return false;
    }

    ruleset_.init();
    bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");

    std::string current_group;
    bool in_replace_section = false;
    std::string line;

    while (std::getline(f, line)) {
        size_t comment = line.find("//");
        if (comment != std::string::npos)
            line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        // Directives
        if (line[0] == '.') {
            if (line.size() >= 2 && line[1] == 'L') {
                parseLGroupDef(line, ruleset_);
                continue;
            } else if (line == ".replace") {
                in_replace_section = true;
                current_group = "";
                continue;
            } else if (line.substr(0, 6) == ".group") {
                in_replace_section = false;
                current_group = trim(line.substr(6));
                continue;
            }
        }

        if (in_replace_section) {
            auto parts = splitWS(line);
            if (parts.size() >= 2) {
                ReplaceRule rr;
                rr.from = parts[0];
                rr.to = parts[1];
                ruleset_.replacements.push_back(rr);
            }
            continue;
        }

        if (current_group.empty()) continue;

        // Parse rule line - may have leading ?condition
        int dialect_cond = 0;
        bool cond_negated = false;
        std::string rule_line = line;

        if (!line.empty() && line[0] == '?') {
            size_t space = line.find_first_of(" \t");
            if (space != std::string::npos) {
                std::string cond_str = line.substr(1, space - 1);
                if (!cond_str.empty() && cond_str[0] == '!') {
                    cond_negated = true;
                    cond_str = cond_str.substr(1);
                }
                try { dialect_cond = std::stoi(cond_str); } catch(...) {}
                rule_line = trim(line.substr(space));
            }
        }

        // Apply dialect filter
        if (dialect_cond != 0) {
            bool match = false;
            if (dialect_cond == 3) match = is_en_us;
            bool applies = cond_negated ? !match : match;
            if (!applies) continue;
        }

        // Tokenize the rule line preserving whitespace groups
        std::vector<std::string> tokens;
        {
            std::string tok;
            for (char c : rule_line) {
                if (std::isspace((unsigned char)c)) {
                    if (!tok.empty()) { tokens.push_back(tok); tok.clear(); }
                } else {
                    tok += c;
                }
            }
            if (!tok.empty()) tokens.push_back(tok);
        }
        if (tokens.empty()) continue;

        PhonemeRule rule;
        rule.condition = dialect_cond;
        rule.condition_negated = cond_negated;

        int ti = 0;

        // Left context: token ending with ')'
        if (ti < (int)tokens.size() && tokens[ti].back() == ')') {
            std::string lctx = tokens[ti];
            lctx.pop_back();
            rule.left_ctx = lctx;
            ti++;
        }

        // Match string: next token not starting with '('
        if (ti < (int)tokens.size() && tokens[ti][0] != '(') {
            rule.match = tokens[ti];
            ti++;
        } else {
            rule.match = current_group;
        }

        // Right context: token starting with '('
        if (ti < (int)tokens.size() && tokens[ti][0] == '(') {
            rule.right_ctx = tokens[ti].substr(1); // remove '('
            ti++;
        }

        // Phonemes: rest of tokens
        for (int j = ti; j < (int)tokens.size(); j++) {
            rule.phonemes += tokens[j];
        }

        // Skip rules with only $ flags (no phoneme output)
        if (!rule.phonemes.empty() && rule.phonemes[0] == '$') continue;
        // Skip empty match
        if (rule.match.empty()) continue;

        // Detect prefix rules: right context contains 'P' (the reference SUFX_P / RULE_ENDING).
        // In the reference rule notation, 'P' in right context marks a prefix boundary —
        // the matched string is a prefix; the suffix must be re-translated as a new word.
        // 'P' can appear after '@' (syllable marker) or after literal context chars:
        //   "@P2"  = syllable marker + prefix + score2  (e.g., "_) in (@P2")
        //   "deP2" = literal 'd','e' + prefix + score2  (e.g., "_) in (deP2")
        // We detect 'P' that is not part of a multi-char letter group (L-groups, etc.)
        // and is followed by a digit or is at end of context string or before '_'.
        {
            const std::string& rc = rule.right_ctx;
            for (size_t k = 0; k < rc.size(); k++) {
                if (rc[k] == 'P') {
                    // 'P' is the prefix marker when followed by a digit, end-of-string,
                    // or another context-special char. (Score modifiers are digits 1-9.)
                    bool is_prefix_marker = false;
                    if (k + 1 >= rc.size()) {
                        is_prefix_marker = true;  // P at end
                    } else {
                        char nc = rc[k + 1];
                        if (nc >= '1' && nc <= '9') is_prefix_marker = true;
                        else if (nc == '_' || nc == '+' || nc == '<') is_prefix_marker = true;
                    }
                    // But 'P' preceded by 'L' is an L-group reference (L01..L99), skip
                    if (k > 0 && rc[k - 1] == 'L') is_prefix_marker = false;
                    if (is_prefix_marker) { rule.is_prefix = true; break; }
                }
            }
        }

        // Detect suffix rules: right context contains 'S<N>[flags]' (the reference RULE_ENDING).
        // In the reference source notation, '_S<N>[flags]' means:
        //   _  = word boundary (handled by matchRightContextScore as usual)
        //   S  = RULE_ENDING suffix-stripping directive
        //   N  = number of chars to strip from word end when rule fires
        //   flags: i=SUFX_I (restore i→y in stem), m=SUFX_M (allow further suffixes),
        //          v=SUFX_V (use verb form), e=SUFX_E (add e to stem), d=SUFX_D (dedup consonant)
        // When such a rule fires at word-end, we discard partial phonemes, strip N chars
        // from the word end to get the stem, re-phonemize the stem (dict or rules),
        // and combine: stem_phonemes + this rule's phoneme output.
        {
            static const int SUFX_I_BIT = 0x200;
            static const int SUFX_M_BIT = 0x80000;
            static const int SUFX_V_BIT = 0x800;
            static const int SUFX_E_BIT = 0x100;
            static const int SUFX_D_BIT = 0x1000;
            static const int SUFX_Q_BIT = 0x4000;
            const std::string& rc = rule.right_ctx;
            for (size_t k = 0; k < rc.size(); k++) {
                if (rc[k] == 'S' && (k == 0 || rc[k-1] != 'L')) {
                    // 'S' not preceded by 'L' → RULE_ENDING suffix directive
                    size_t k2 = k + 1;
                    int n = 0;
                    int sflags = 0;
                    while (k2 < rc.size() && std::isdigit((unsigned char)rc[k2]))
                        n = n * 10 + (rc[k2++] - '0');
                    while (k2 < rc.size() && std::isalpha((unsigned char)rc[k2])) {
                        char fc = rc[k2++];
                        if (fc == 'i') sflags |= SUFX_I_BIT;
                        else if (fc == 'm') sflags |= SUFX_M_BIT;
                        else if (fc == 'v') sflags |= SUFX_V_BIT;
                        else if (fc == 'e') sflags |= SUFX_E_BIT;
                        else if (fc == 'd') sflags |= SUFX_D_BIT;
                        else if (fc == 'q') sflags |= SUFX_Q_BIT;
                        else if (fc == 'p') sflags |= 0x400; // SUFX_P (prefix)
                    }
                    if (n > 0) {
                        rule.is_suffix = true;
                        rule.suffix_strip_len = n;
                        rule.suffix_flags = sflags;
                        break;
                    }
                }
            }
        }

        // Store rule in group
        ruleset_.rule_groups[current_group].push_back(rule);
    }

    return true;
}

// ============================================================
// Constructor
// ============================================================
IPAPhonemizer::IPAPhonemizer(const std::string& rules_path,
                                    const std::string& list_path,
                                    const std::string& dialect)
    : dialect_(dialect), loaded_(false) {
    ipa_overrides_ = buildIPAOverrides(dialect);

    if (!loadDictionary(list_path)) return;
    if (!loadRules(rules_path)) return;
    loaded_ = true;
}

// ============================================================
// Apply replacement rules
// ============================================================
std::string IPAPhonemizer::applyReplacements(const std::string& word) const {
    std::string result = word;
    for (const auto& rr : ruleset_.replacements) {
        size_t pos = 0;
        while ((pos = result.find(rr.from, pos)) != std::string::npos) {
            result.replace(pos, rr.from.size(), rr.to);
            pos += rr.to.size();
        }
    }
    return result;
}

// ============================================================
// Context matching helpers
// ============================================================

// Check if word[pos..] starts with any item in the L-group
// Returns number of chars matched (0 if no match)
static int matchLGroupAt(const std::vector<std::string>& lgroup, const std::string& word, int pos) {
    if (pos < 0 || pos >= (int)word.size()) return 0;
    // Try longest items first
    int best = 0;
    for (const auto& item : lgroup) {
        if (item.empty()) continue;
        int ilen = (int)item.size();
        if (pos + ilen > (int)word.size()) continue;
        bool ok = true;
        for (int j = 0; j < ilen; j++) {
            char wc = (char)std::tolower((unsigned char)word[pos + j]);
            char ic = (char)std::tolower((unsigned char)item[j]);
            if (wc != ic) { ok = false; break; }
        }
        if (ok && ilen > best) best = ilen;
    }
    return best;
}

// Match left context: returns (score, true) if matched, (0, false) otherwise
// Left context is scanned RIGHT-TO-LEFT (from pos-1 backward)
// ctx_str is the left context in NATURAL order (as written in rules file)
// Scoring matches the reference MatchRule: distance_left starts at -2, increments by 2 per char
// Literal: 21-distance_left; Group non-C: 20-distance_left; Group C: 19-distance_left
// check_atstart ('_'): +4; STRESSED ('&'): +19; NOVOWELS ('X'): +3; DOUBLE ('%'): 21-distance_left
// INC_SCORE ('+'): +20; SYLLABLE ('@'): 18+count-distance_left
// phonemes_so_far: raw the reference phonemes assigned to letters 0..pos-1 (used for '&' stressed check)
std::pair<int,bool> matchLeftContextScore(const std::string& ctx_str, const std::string& word, int pos,
                                           const RuleSet& rs,
                                           const std::string& phonemes_so_far = "") {
    if (ctx_str.empty()) return {0, true};

    int word_pos = pos - 1;
    int ci = (int)ctx_str.size() - 1;
    int score = 0;
    int distance_left = -2; // increments by 2 each char consumed, capped at 19
    char prev_char = (pos > 0 && pos < (int)word.size()) ? word[pos] : 0; // first char of match

    while (ci >= 0) {
        char cc = ctx_str[ci];

        if (cc == '_') {
            // RULE_SPACE / check_atstart: word boundary, adds +4
            if (word_pos >= 0) return {0, false};
            ci--;
            score += 4;
            continue;
        }

        if (cc == '&') {
            // RULE_STRESSED: +19, no char consumed.
            // the reference word_stressed_count: count of vowels so far that are not explicitly
            // unstressed (no preceding '%' or '=' marker) and not inherently unstressed.
            // '#' after a vowel is NOT an unstress indicator (it's a phoneme modifier);
            // only explicit '%' or '=' preceding the vowel marks it as unstressed.
            // the reference evidence: 'I' from 'i' in "rigorous" triggers &, so 'I' is stressable;
            //                  '0' from 'o' in "nostra" triggers & for final 'a', so '0#' is stressable.
            bool found_stressed = false;
            // Scan phonemes_so_far for any stressable vowel not preceded by '%'/'='.
            // Must parse multi-char codes as units (e.g. 'aI', '3:', 'I2') so that '%' before
            // the FIRST char of a multi-char code correctly marks the WHOLE code as unstressed.
            // Bug: char-by-char scanning treats 'I' in '%aI' as a standalone stressable vowel.
            // Fix: multi-char-aware scan using the same S_MC table as processPhonemeString.
            // phUNSTRESSED codes (always unstressed regardless of '%' prefix):
            //   '@*' variants, 'i' alone (not 'i:' or 'i@'), 'a#', 'I#', 'I2'.
            // Stressable despite '#' suffix: '0#', 'E#', 'E2', 'I' alone.
            static const char* STRESSED_MC[] = {
                "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r","o@r","A@r","e@r",
                "eI","aI","aU","OI","oU","IR","VR",
                "e@","i@","U@","A@","O@","o@",
                "3:","A:","i:","u:","O:","e:","a:","aa",
                "@L","@2","@5",
                "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~","A#",
                nullptr
            };
            static const std::string VOWEL_PH_STRESSABLE = "aAeEiIoOuUV03";
            if (!phonemes_so_far.empty()) {
                size_t pi = 0;
                bool prev_unstressed = false;
                while (pi < phonemes_so_far.size()) {
                    char pc = phonemes_so_far[pi];
                    if (pc == '%' || pc == '=' || pc == ',') { prev_unstressed = true; pi++; continue; }
                    if (pc == '\'') { prev_unstressed = false; pi++; continue; }
                    // Try multi-char codes first
                    std::string code;
                    for (int mi = 0; STRESSED_MC[mi]; mi++) {
                        int mcl = (int)strlen(STRESSED_MC[mi]);
                        if (pi + (size_t)mcl <= phonemes_so_far.size() &&
                            phonemes_so_far.compare(pi, mcl, STRESSED_MC[mi]) == 0) {
                            code = std::string(STRESSED_MC[mi], mcl);
                            break;
                        }
                    }
                    if (code.empty()) code = std::string(1, pc);
                    bool is_vowel = !code.empty() && VOWEL_PH_STRESSABLE.find(code[0]) != std::string::npos;
                    if (is_vowel) {
                        // Check inherently unstressed
                        bool inherently_unstressed =
                            (code[0] == '@') ||               // schwa and variants
                            (code == "i") ||                   // 'i' alone (not 'i:', 'i@')
                            (code == "a#") ||                  // reduced 'a'
                            (code == "I#") ||                  // reduced ɪ
                            (code == "I2");                    // reduced ɪ variant
                        if (!inherently_unstressed && !prev_unstressed) {
                            found_stressed = true;
                            break;
                        }
                    }
                    prev_unstressed = false;
                    pi += code.size();
                }
            } else {
                for (int k = 0; k < pos; k++) {
                    if (isVowelLetter(word[k])) { found_stressed = true; break; }
                }
            }
            if (!found_stressed) return {0, false};
            ci--;
            score += 19;
            continue;
        }

        if (cc == '@') {
            // RULE_SYLLABLE in pre-context: count consecutive '@' chars (right-to-left)
            int syllable_count = 0;
            while (ci >= 0 && ctx_str[ci] == '@') { syllable_count++; ci--; }
            int vowel_groups = 0;
            // the reference counts syllable nuclei in the PHONEME string accumulated so far,
            // not vowel letters in the word. Consecutive vowel phoneme chars form one
            // group (e.g., diphthong 'eI' = 1 syllable, not 2).
            // This matters for e.g. `@@e) d`: "feIs" at 'd' → 'e','I' consecutive = 1
            // group → @@ fails (correct, "faced" stays voiced). "noUtI2s" at 'd' → 'oU'
            // and 'I' separated by 't' = 2 groups → @@ passes (correct, "noticed" devoiced).
            if (!phonemes_so_far.empty()) {
                static const std::string VOWEL_PH = "aAeEIiOUVu03@o";
                bool in_v2 = false;
                for (char c : phonemes_so_far) {
                    bool v = (VOWEL_PH.find(c) != std::string::npos);
                    if (v && !in_v2) { vowel_groups++; in_v2 = true; }
                    else if (!v) { in_v2 = false; }
                }
            } else {
                bool in_v2 = false;
                for (int wp = 0; wp < pos; wp++) {
                    bool v = isVowelLetter(word[wp]);
                    if (v && !in_v2) { vowel_groups++; in_v2 = true; }
                    else if (!v) { in_v2 = false; }
                }
            }
            if (syllable_count > vowel_groups) return {0, false};
            // SYLLABLE adds points like a char but doesn't consume
            int dist = distance_left + 2; if (dist > 19) dist = 19;
            score += 18 + syllable_count - dist;
            continue;
        }

        if (cc == '!') { ci--; continue; } // RULE_CAPITAL - skip (no strict check)

        if (cc == '%') {
            // RULE_DOUBLE: current left-context char must equal the char to its right
            if (word_pos < 0) return {0, false};
            char cur = (char)std::tolower((unsigned char)word[word_pos]);
            char nxt = (char)std::tolower((unsigned char)prev_char);
            if (cur != nxt) return {0, false};
            distance_left += 2; if (distance_left > 19) distance_left = 19;
            prev_char = word[word_pos];
            word_pos--;
            ci--;
            score += 21 - distance_left;
            continue;
        }

        if (cc == '+') { score += 20; ci--; continue; } // RULE_INC_SCORE
        if (cc == '<') { score -= 20; ci--; continue; } // RULE_DEC_SCORE

        if (cc == 'A' || cc == 'B' || cc == 'C' ||
            cc == 'F' || cc == 'G' || cc == 'H' || cc == 'Y') {
            if (!rs.groups.matchGroup(cc, word, word_pos)) return {0, false};
            distance_left += 2; if (distance_left > 19) distance_left = 19;
            int lg_pts = (cc == 'C') ? 19 : 20;
            prev_char = word[word_pos];
            word_pos--;
            ci--;
            score += lg_pts - distance_left;
            continue;
        }

        if (cc == 'K') {
            if (word_pos < 0) return {0, false};
            if (isVowelLetter(word[word_pos])) return {0, false};
            distance_left += 2; if (distance_left > 19) distance_left = 19;
            prev_char = word[word_pos];
            word_pos--;
            ci--;
            score += 20 - distance_left;
            continue;
        }

        if (cc == 'X') {
            // RULE_NOVOWELS: no vowels from word start to here, adds +3 (fixed, no distance)
            bool found_vowel = false;
            for (int k = 0; k <= word_pos; k++) {
                if (isVowelLetter(word[k])) { found_vowel = true; break; }
            }
            if (found_vowel) return {0, false};
            ci--;
            score += 3;
            continue;
        }

        if (cc == 'D') {
            if (word_pos < 0 || !std::isdigit((unsigned char)word[word_pos])) return {0, false};
            distance_left += 2; if (distance_left > 19) distance_left = 19;
            prev_char = word[word_pos];
            word_pos--;
            ci--;
            score += 21 - distance_left;
            continue;
        }

        if (cc == 'Z') {
            if (word_pos < 0 || std::isalpha((unsigned char)word[word_pos])) return {0, false};
            distance_left += 2; if (distance_left > 19) distance_left = 19;
            prev_char = word[word_pos];
            word_pos--;
            ci--;
            score += 21 - distance_left;
            continue;
        }

        // Check for L-group reference (L followed by digits)
        if (cc >= '0' && cc <= '9') {
            int gid = cc - '0';
            int ci2 = ci - 1;
            if (ci2 >= 0 && ctx_str[ci2] >= '0' && ctx_str[ci2] <= '9') {
                gid += (ctx_str[ci2] - '0') * 10;
                ci2--;
            }
            if (ci2 >= 0 && ctx_str[ci2] == 'L') {
                if (gid > 0 && gid < 100) {
                    int matched = matchLGroupAt(rs.groups.lgroups[gid], word, word_pos);
                    if (matched == 0) return {0, false};
                    distance_left += 2; if (distance_left > 19) distance_left = 19;
                    if (matched > 0) prev_char = word[word_pos - matched + 1];
                    word_pos -= matched;
                    score += 20 - distance_left;
                }
                ci = ci2 - 1;
                continue;
            }
            ci--;
            continue;
        }

        // 'E' in context = REPLACED_E (the reference marks silent 'e' after '#' rules as uppercase 'E').
        // We don't implement REPLACED_E marking, so 'E' context never matches.
        if (cc == 'E') return {0, false};

        // Literal character match
        if (word_pos < 0) return {0, false};
        char wc = (char)std::tolower((unsigned char)word[word_pos]);
        char mc = (char)std::tolower((unsigned char)cc);
        if (wc != mc) return {0, false};
        distance_left += 2; if (distance_left > 19) distance_left = 19;
        prev_char = word[word_pos];
        word_pos--;
        ci--;
        score += 21 - distance_left;
    }

    return {score, true};
}

// Match right context: returns (score, del_fwd_start, del_fwd_count, matched) if matched
// del_fwd_start: absolute word position where silent chars begin (-1 if none)
// del_fwd_count: number of chars to mark as silent/deleted
struct RightCtxResult { int score; int del_fwd_start; int del_fwd; bool matched; };

// initial_prev_char: last char of the match key (needed for RULE_DOUBLE '%')
// match_start: position in word where the rule's key starts (for Sn syllable-count condition)
RightCtxResult matchRightContextScore(const std::string& ctx_str, const std::string& word, int pos,
                                       const RuleSet& rs, char initial_prev_char = 0,
                                       int match_start = -1, int word_alt_flags = 0,
                                       const std::vector<bool>* replaced_e_arr = nullptr,
                                       bool suffix_removed = false) {
    if (ctx_str.empty()) return {0, -1, 0, true};

    int word_pos = pos;
    int ci = 0;
    int clen = (int)ctx_str.size();
    int score = 0;
    int distance_right = -6; // increments by 6 per char consumed, capped at 19
    char prev_char = initial_prev_char;
    int del_fwd_pos = -1; // position of REPLACED_E character to skip

    while (ci < clen) {
        char cc = ctx_str[ci];

        if (cc == '_') {
            // RULE_SPACE: word-end boundary; scores like a literal char
            if (word_pos < (int)word.size()) return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            ci++;
            score += 21 - distance_right;
            continue;
        }

        if (cc == '#') {
            // RULE_DEL_FWD: search for the first 'e' in range [pos, word_pos) and mark
            // it as REPLACED_E (magic-e deletion). Only fires when 'e' is actually found.
            // the reference: "for (p = *word + group_length; p < post_ptr; p++) if (*p=='e') ..."
            // Example: rule "iv (e#" → 'e' at range[0] is found and silenced (→ "ive" treated as /aɪv/).
            // Counter-example: "rhi (n#" → range [pos,word_pos) = just 'n', not 'e' → no deletion.
            if (del_fwd_pos < 0) {
                for (int sp = pos; sp < word_pos; sp++) {
                    if (word[sp] == 'e') { del_fwd_pos = sp; break; }
                }
            }
            ci++;
            continue;
        }

        if (cc == 'A' || cc == 'B' || cc == 'C' ||
            cc == 'F' || cc == 'G' || cc == 'H' || cc == 'Y') {
            if (!rs.groups.matchGroup(cc, word, word_pos)) return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            int lg_pts = (cc == 'C') ? 19 : 20;
            prev_char = word[word_pos];
            word_pos++;
            ci++;
            score += lg_pts - distance_right;
            continue;
        }

        if (cc == 'K') {
            // K = non-vowel. the reference words are null-terminated, so K matches '\0' at
            // word end (null is not a vowel). Treat word_pos >= word.size() as '\0' match.
            if (word_pos < (int)word.size() && isVowelLetter(word[word_pos])) return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            if (word_pos < (int)word.size()) prev_char = word[word_pos];
            word_pos++;
            ci++;
            score += 20 - distance_right;
            continue;
        }

        if (cc == 'X') {
            // RULE_NOVOWELS right-context: no vowels to word end; scores 19-distance
            bool found = false;
            for (int k = word_pos; k < (int)word.size(); k++) {
                if (isVowelLetter(word[k])) { found = true; break; }
            }
            if (found) return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            ci++;
            score += 19 - distance_right;
            continue;
        }

        if (cc == 'D') {
            if (word_pos >= (int)word.size() || !std::isdigit((unsigned char)word[word_pos]))
                return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            prev_char = word[word_pos];
            word_pos++;
            ci++;
            score += 21 - distance_right;
            continue;
        }

        if (cc == 'Z') {
            if (word_pos >= (int)word.size() || std::isalpha((unsigned char)word[word_pos]))
                return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            prev_char = word[word_pos];
            word_pos++;
            ci++;
            score += 21 - distance_right;
            continue;
        }

        if (cc == '%') {
            // RULE_DOUBLE: current char must equal previous char
            if (word_pos >= (int)word.size()) return {0, -1, 0, false};
            char cur = (char)std::tolower((unsigned char)word[word_pos]);
            char prv = (char)std::tolower((unsigned char)prev_char);
            if (cur != prv) return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            prev_char = word[word_pos];
            word_pos++;
            ci++;
            score += 21 - distance_right;
            continue;
        }

        if (cc == '+') { score += 20; ci++; continue; } // RULE_INC_SCORE
        if (cc == '<') { score -= 20; ci++; continue; } // RULE_DEC_SCORE

        if (cc == '@') {
            // RULE_SYLLABLE: count consecutive '@', require N vowel groups remaining
            int syllable_count = 0;
            while (ci < clen && ctx_str[ci] == '@') { syllable_count++; ci++; }
            int vowel_groups = 0;
            bool in_v = false;
            for (int wp = word_pos; wp < (int)word.size(); wp++) {
                bool v = isVowelLetter(word[wp]);
                if (v && !in_v) { vowel_groups++; in_v = true; }
                else if (!v) { in_v = false; }
            }
            if (syllable_count > vowel_groups) return {0, -1, 0, false};
            distance_right += 6; if (distance_right > 19) distance_right = 19;
            score += 18 + syllable_count - distance_right;
            continue;
        }

        if (cc == '&') { return {0, -1, 0, false}; } // RULE_STRESSED in right context - fail
        if (cc == '!') { ci++; continue; }

        if (cc == '$') {
            // Check for $w_altN word-level condition (e.g. "$w_alt2" in right context)
            // This fires only if the current word has the corresponding $altN flag in en_list.
            // Format: $w_alt followed by digit 1-6
            if (ci + 6 < clen && ctx_str[ci+1]=='w' && ctx_str[ci+2]=='_' &&
                ctx_str[ci+3]=='a' && ctx_str[ci+4]=='l' && ctx_str[ci+5]=='t' &&
                ctx_str[ci+6] >= '1' && ctx_str[ci+6] <= '6') {
                int alt_n = ctx_str[ci+6] - '0';
                int alt_bit = 1 << (alt_n - 1);
                if (!(word_alt_flags & alt_bit)) return {0, -1, 0, false};
                ci += 7; // skip "$w_altN"
                continue;
            }
            return {0, -1, 0, false}; // other $ flags - skip rule
        }

        if (cc == 'N') {
            if (ci+1 < clen && std::isdigit(ctx_str[ci+1])) {
                // Nn (N followed by digit): syllable/condition check — skip for now
                ci += 2; continue;
            } else {
                // N alone = RULE_NO_SUFFIX: fails when the word is being re-phonemized
                // as a stem after suffix removal (the reference FLAG_SUFFIX_REMOVED / translate.h:185).
                // When suffix was removed, this rule should not fire.
                if (suffix_removed) return {0, -1, 0, false};
                score += 1;
                ci++;
                continue;
            }
        }

        if (cc == 'P') {
            while (ci < clen && !std::isspace((unsigned char)ctx_str[ci])) ci++;
            continue;
        }

        if (cc == 'S') {
            // RULE_ENDING: the reference suffix-stripping directive (_S<N>[flags] in source).
            // This is a pure directive — it does NOT consume word chars and does NOT fail
            // the match. The suffix info (strip length N, flags i/m/v/e/d/q/t) is stored
            // on the PhonemeRule itself (set during loadRules). Here we just skip past
            // the number and flag characters in the context string.
            ci++; // skip 'S'
            while (ci < clen && std::isdigit((unsigned char)ctx_str[ci])) ci++;
            while (ci < clen && std::isalpha((unsigned char)ctx_str[ci])) ci++;
            // No score contribution, no position check — suffix action handled in applyRules
            continue;
        }

        // L-group reference
        if (cc == 'L' && ci+1 < clen && std::isdigit(ctx_str[ci+1])) {
            int gid = 0;
            ci++;
            while (ci < clen && std::isdigit(ctx_str[ci])) {
                gid = gid * 10 + (ctx_str[ci] - '0');
                ci++;
            }
            if (gid > 0 && gid < 100 && !rs.groups.lgroups[gid].empty()) {
                int matched = matchLGroupAt(rs.groups.lgroups[gid], word, word_pos);
                if (matched == 0) return {0, -1, 0, false};
                distance_right += 6; if (distance_right > 19) distance_right = 19;
                if (matched > 0) prev_char = word[word_pos + matched - 1];
                word_pos += matched;
                score += 20 - distance_right;
            }
            continue;
        }

        if (std::isdigit((unsigned char)cc)) { ci++; continue; }

        // 'E' in context = REPLACED_E (the reference marks silent 'e' after '#' rules as uppercase 'E').
        // Match if replaced_e_arr marks this position as a deleted 'e'.
        if (cc == 'E') {
            if (replaced_e_arr && word_pos < (int)word.size() &&
                word_pos < (int)replaced_e_arr->size() && (*replaced_e_arr)[word_pos]) {
                distance_right += 6; if (distance_right > 19) distance_right = 19;
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += 21 - distance_right;
                continue;
            }
            return {0, -1, 0, false};
        }

        // Literal character match
        if (word_pos >= (int)word.size()) return {0, -1, 0, false};
        char wc = (char)std::tolower((unsigned char)word[word_pos]);
        char mc = (char)std::tolower((unsigned char)cc);
        if (wc != mc) return {0, -1, 0, false};
        distance_right += 6; if (distance_right > 19) distance_right = 19;
        prev_char = word[word_pos];
        word_pos++;
        ci++;
        score += 21 - distance_right;
    }

    return {score, del_fwd_pos, (del_fwd_pos >= 0 ? 1 : 0), true};
}

// Match a rule at position pos in word, returns score (-1 if no match)
// group_length: length of the group key (1 for single-char groups, 2 for two-char groups)
// del_fwd_start: absolute word position of first silent char (-1 if none)
// del_fwd_count: number of chars to mark silent starting at del_fwd_start
int IPAPhonemizer::matchRule(const PhonemeRule& rule, const std::string& word, int pos,
                                  std::string& out_phonemes, int& advance,
                                  int& del_fwd_start, int& del_fwd_count,
                                  int group_length,
                                  const std::string& phonemes_so_far,
                                  int word_alt_flags,
                                  const std::vector<bool>* replaced_e_arr,
                                  bool suffix_removed) const {
    const std::string& match = rule.match;
    int mlen = (int)match.size();

    if (pos + mlen > (int)word.size()) return -1;

    // Compare match string case-insensitively
    for (int i = 0; i < mlen; i++) {
        if (std::tolower((unsigned char)word[pos+i]) != std::tolower((unsigned char)match[i]))
            return -1;
    }

    // Match left context
    auto [lscore, lmatch] = matchLeftContextScore(rule.left_ctx, word, pos, ruleset_, phonemes_so_far);
    if (!lmatch) return -1;

    // The last char of the match (needed for RULE_DOUBLE '%' in right context)
    char last_match_char = (mlen > 0) ? word[pos + mlen - 1] : 0;

    // Match right context (pass pos as match_start for Sn syllable-count condition)
    // word_alt_flags is passed in from applyRules (the caller controls which alt flags are active)
    auto rresult = matchRightContextScore(rule.right_ctx, word, pos + mlen, ruleset_, last_match_char, pos, word_alt_flags, replaced_e_arr, suffix_removed);
    if (!rresult.matched) return -1;

    // the reference scoring: base=1 + (additional match chars beyond group key) * 21 + context scores
    // additional_consumed = mlen - group_length (extra chars in the match beyond the group key)
    int additional_consumed = mlen - group_length;
    if (additional_consumed < 0) additional_consumed = 0;
    int dialect_bonus = (rule.condition != 0) ? 1 : 0;
    int total_score = 1 + additional_consumed * 21 + lscore + rresult.score + dialect_bonus;

    // Extract phonemes (remove $ flags)
    std::string ph = rule.phonemes;
    size_t dollar = ph.find('$');
    if (dollar != std::string::npos)
        ph = trim(ph.substr(0, dollar));

    out_phonemes = ph;
    advance = mlen;
    del_fwd_start = rresult.del_fwd_start;
    del_fwd_count = rresult.del_fwd;

    return total_score;
}

// ============================================================
// Apply rules to a word (letter-to-phoneme)
// ============================================================
std::string IPAPhonemizer::applyRules(const std::string& word_orig, bool allow_suffix_strip,
                                           int word_alt_flags_param,
                                           bool suffix_phoneme_only,
                                           bool suffix_removed,
                                           std::vector<bool>* out_replaced_e,
                                           std::vector<bool>* out_pos_visited) const {
    std::string word = applyReplacements(word_orig);
    std::string phonemes;
    int len = (int)word.size();

    // Determine word-level $altN flags for $w_altN rule context matching.
    // When word_alt_flags_param == -1, look up the word's own flags from en_list.
    // When called for stem re-phonemization via RULE_ENDING, caller passes 0
    // (stems don't inherit their dict flags in the reference TranslateRules path).
    int word_alt_flags;
    if (word_alt_flags_param >= 0) {
        word_alt_flags = word_alt_flags_param;
    } else {
        std::string wl;
        wl.reserve(word.size());
        for (char c : word) wl += (char)std::tolower((unsigned char)c);
        auto it = word_alt_flags_.find(wl);
        word_alt_flags = (it != word_alt_flags_.end()) ? it->second : 0;
    }
    // replaced_e[pos]: char at pos was marked by RULE_DEL_FWD (magic-e).
    // We still process rules for it, but silence it when only the default rule fires (score==0).
    std::vector<bool> replaced_e(len, false);
    // pos_visited[pos]: true if position pos was the start of a scan iteration.
    // Positions skipped by long-match advance are NOT visited.
    std::vector<bool> pos_visited(len, false);

    for (int pos = 0; pos < len; ) {
        if (pos < len) pos_visited[pos] = true;

        // Find best matching rule
        int best_score = -1;
        std::string best_phonemes;
        int best_advance = 1;
        int best_del_start = -1;
        int best_del_count = 0;

        // Helper to update best match; bonus is added to score (e.g. +35 for 2-char groups)
        // Helper: update best match if sc+bonus >= best_score.
        // Using >= matches the reference within-group tie-breaking (last rule wins for equal score),
        // which ensures longer-match rules (appearing later in file) beat shorter-match rules.
        // bonus: added to matchRule's score (e.g. +35 for 2-char groups)
        // group_length: 1 or 2, passed to matchRule for scoring
        std::string best_rule_match, best_rule_lctx, best_rule_rctx;
        bool best_is_prefix = false;
        bool best_is_suffix = false;
        int best_suffix_strip_len = 0;
        int best_suffix_flags = 0;
        auto try_rule_group = [&](const std::string& key, int bonus = 0, int group_length = 1) {
            auto it = ruleset_.rule_groups.find(key);
            if (it == ruleset_.rule_groups.end()) return;
            for (const auto& rule : it->second) {
                std::string ph;
                int adv, dfstart, dfcount;
                int sc = matchRule(rule, word, pos, ph, adv, dfstart, dfcount, group_length, phonemes, word_alt_flags, &replaced_e, suffix_removed);
                if (sc < 0) continue;
                // When suffix stripping is disabled (and NOT in suffix_phoneme_only mode),
                // skip suffix rules at word-end: the reference TranslateRules returns early
                // (without the phoneme) when RULE_ENDING fires with end_phonemes=NULL.
                // In suffix_phoneme_only mode, RULE_ENDING rules ARE selected but their
                // phoneme is just accumulated (no stem re-phonemization) — mimicking
                // the reference TranslateRules(word, NULL) first-pass behavior.
                if (!allow_suffix_strip && !suffix_phoneme_only && rule.is_suffix && pos + adv == len) continue;
                if (sc + bonus >= best_score) {
                    best_score = sc + bonus;
                    best_phonemes = ph;
                    best_advance = adv;
                    best_del_start = dfstart;
                    best_del_count = dfcount;
                    best_is_prefix = rule.is_prefix;
                    best_is_suffix = rule.is_suffix;
                    best_suffix_strip_len = rule.suffix_strip_len;
                    best_suffix_flags = rule.suffix_flags;
                    if (std::getenv("PHON_DEBUG2")) {
                        best_rule_match = rule.match;
                        best_rule_lctx = rule.left_ctx;
                        best_rule_rctx = rule.right_ctx;
                    }
                }
            }
        };

        // Determine effective first character for rule group lookup.
        // When a position has been marked as replaced_e (via RULE_DEL_FWD / '#' in context),
        // the reference has replaced the 'e' with 'E' (REPLACED_E = 'E') in its word buffer.
        // The lookup then uses groups1['E'], i.e., the .group E rules (not .group e).
        char pos_char = replaced_e[pos] ? 'E' : (char)std::tolower((unsigned char)word[pos]);

        // Try 2-char group with the reference +35 bonus (TranslateRules: match2.points += 35)
        if (pos + 1 < len) {
            std::string key2;
            key2 += pos_char;
            key2 += (char)std::tolower((unsigned char)word[pos+1]);
            try_rule_group(key2, 35, 2);
        }
        // Try 1-char group (no bonus, group_length=1)
        {
            std::string key1(1, pos_char);
            try_rule_group(key1, 0, 1);
        }

        if (best_score >= 0) {
            // For replaced_e positions, the .group E default rule emits "" (silent 'e'),
            // so no special workaround is needed here — best_phonemes is already "" for
            // the default case, and non-default rules emit the correct phoneme.
            std::string emit = best_phonemes;
            if (std::getenv("PHON_DEBUG")) {
                std::cerr << "  pos=" << pos << " char='" << word[pos]
                    << "' ph='" << emit << "' adv=" << best_advance
                    << " score=" << best_score
                    << (replaced_e[pos] ? " [repl-e]" : "")
                    << (best_is_prefix ? " [PREFIX]" : "")
                    << (best_is_suffix ? " [SUFFIX]" : "");
                if (std::getenv("PHON_DEBUG2"))
                    std::cerr << " rule=[" << best_rule_lctx << ")" << best_rule_match
                              << "(" << best_rule_rctx << "]";
                std::cerr << "\n";
            }
            // Handle phonSTRESS_PREV: '=' at the START of a rule output (not after a phoneme)
            // corresponds to the reference byte code phonSTRESS_PREV (code 8). It retroactively
            // promotes the last preceding stressable vowel to PRIMARY stress.
            // In the reference mnemonic text, phonSTRESS_PREV appears as '=' only at the very
            // start of a rule's output (e.g. rule "@) tu (lat → =tSU" in en_rules).
            // phonLENGTHEN (also '=') always appears AFTER a phoneme char (e.g. 'S=@n'),
            // so we distinguish by position: emit[0]=='=' is phonSTRESS_PREV.
            if (!emit.empty() && emit[0] == '=') {
                emit = emit.substr(1); // consume the phonSTRESS_PREV marker
                // Scan backward through accumulated phonemes to find last stressable vowel.
                // A stressable vowel is a vowel code char NOT immediately preceded by
                // '%' (unstressed) or '=' (diminished/unstressed) or '\'' (already primary).
                // the reference vowel first-chars in mnemonic form:
                static const std::string SP_VOWELS = "aAeEiIoOuUV03@";
                // Diphthong/multi-char code second chars: the current char is the SECOND
                // (or later) char of a multi-char phoneme code. We need to back up to find
                // the START of that code before inserting the stress marker.
                // Case 1: I or U preceded by a diphthong-start char (e, a, O, o, U, A, E)
                //   → 2nd char of eI, aI, OI, aU, oU, AU, etc.
                // Case 2: 'a' preceded by 'a' → 2nd char of 'aa' code (= æ)
                // Case 3: '@' preceded by a vowel char (A, e, i, O, o, U) → 2nd char of
                //   A@, e@, i@, O@, o@, U@ multi-char code (centring/rhotic diphthong)
                static const std::string DIPH_SECOND = "IU";
                static const std::string DIPH_START = "eaOoUAE";
                // Chars that, when followed by '@', indicate a 2-char rhotic code
                static const std::string AT_STARTERS = "AeioOU";

                // Helper: given position si in phonemes, return the prev non-boundary char and pos
                auto prevNonBnd = [&](int from) -> std::pair<char, int> {
                    for (int pi = from - 1; pi >= 0; pi--) {
                        if (phonemes[pi] != '\x01') return {phonemes[pi], pi};
                    }
                    return {0, -1};
                };

                int insert_at = -1;
                int slen = (int)phonemes.size();
                for (int si = slen - 1; si >= 0; si--) {
                    char sc = phonemes[si];
                    if (sc == '\x01') continue; // rule boundary — skip
                    if (SP_VOWELS.find(sc) == std::string::npos) continue; // not a vowel char
                    // Skip reduced/weak vowels: those followed by '2' or '#' modifier.
                    // e.g. 'I2' (weak ɪ), 'I#' (weak ɪ), 'a#' (ɐ), '0#' (unstressed ɑ).
                    // the reference phonSTRESS_PREV only promotes fully-stressable vowels.
                    {
                        int ni = si + 1;
                        while (ni < slen && phonemes[ni] == '\x01') ni++;
                        if (ni < slen && (phonemes[ni] == '2' || phonemes[ni] == '#')) continue;
                    }
                    // Found a vowel char. Check if it's the 2nd+ char of a multi-char code.
                    auto [prev_ch, prev_pos] = prevNonBnd(si);
                    // Case 1: I or U after a diphthong-start char → step back to diphthong start
                    if (DIPH_SECOND.find(sc) != std::string::npos &&
                        DIPH_START.find(prev_ch) != std::string::npos) {
                        si = prev_pos;
                        auto [pp2, pp2pos] = prevNonBnd(si);
                        prev_ch = pp2;
                    }
                    // Case 2: 'a' after 'a' → 2nd char of 'aa' code, step back to first 'a'
                    else if (sc == 'a' && prev_ch == 'a') {
                        si = prev_pos;
                        auto [pp2, pp2pos] = prevNonBnd(si);
                        prev_ch = pp2;
                    }
                    // Case 3: '@' after a rhotic-diphthong start char → step back
                    else if (sc == '@' && AT_STARTERS.find(prev_ch) != std::string::npos) {
                        si = prev_pos;
                        auto [pp2, pp2pos] = prevNonBnd(si);
                        prev_ch = pp2;
                    }
                    // Skip if already marked unstressed/diminished or already primary
                    if (prev_ch == '%' || prev_ch == '=') continue;
                    // This is the start of the stressable vowel — mark it
                    insert_at = si;
                    break;
                }
                if (insert_at >= 0) {
                    // the reference phonSTRESS_PREV only promotes if the found vowel is NOT already
                    // PRIMARY (i.e., vowel_stress[prev] < STRESS_IS_PRIMARY = 4).
                    // If the vowel at insert_at already has "'" immediately before it, the
                    // found vowel is already primary → phonSTRESS_PREV is a no-op.
                    char before_vowel = 0;
                    for (int pi = insert_at - 1; pi >= 0; pi--) {
                        if (phonemes[pi] != '\x01') { before_vowel = phonemes[pi]; break; }
                    }
                    if (before_vowel != '\'') {
                        // Vowel not already primary → promote and demote earlier primaries.
                        phonemes.insert(insert_at, "'");
                        // Demote any earlier "'" markers to '\x02' (protected secondary).
                        // '\x02' (not ',') is used so that ph_in.find(',') stays npos in
                        // processPhonemeString, allowing step 5a to run. The '\x02' markers
                        // are converted to ',' at the start of processPhonemeString, which
                        // then makes step 5a's backward cascade skip (ph.find(',') != npos),
                        // preventing spurious extra secondaries on top of the demoted ones.
                        for (int di = 0; di < insert_at; di++) {
                            if (phonemes[di] == '\'') {
                                phonemes[di] = '\x02'; // protected secondary
                            }
                        }
                    }
                    // else: vowel already primary → phonSTRESS_PREV is no-op
                }
            }
            phonemes += emit;
            phonemes += '\x01'; // rule boundary marker (stripped in processPhonemeString)

            // SUFFIX rule (_S<N>): strip N chars from word end, re-phonemize stem, combine.
            // This fires when: a rule with is_suffix=true matches AND its right context '_'
            // (word boundary) passes, i.e., pos + best_advance == word.size().
            // Following the reference TranslateRules: IMMEDIATELY return stem_ph + suffix_ph,
            // discarding any phonemes accumulated so far (they're re-derived from stem).
            //
            // When suffix_phoneme_only=true: mimic the reference TranslateRules(word, NULL) mode.
            // The RULE_ENDING's phoneme is already accumulated at line 1303 above.
            // Just advance past the matched text and continue — the accumulated first-pass
            // phonemes (stem + suffix phoneme) are returned at the end of the scan.
            // This avoids stem extraction/re-phonemization that would change vowel quality
            // (e.g. "ribosome" with NULL end_phonemes gives rIb0soUm not ri:boUsoUm).
            if ((allow_suffix_strip || suffix_phoneme_only) && best_is_suffix && pos + best_advance == len) {
                if (suffix_phoneme_only) {
                    // Accumulate phoneme and advance without stem re-phonemization.
                    // The suffix phoneme was already added at line 1303. Just advance.
                    pos += best_advance;
                    continue;
                }
                int strip = best_suffix_strip_len;
                if (strip <= 0 || strip > len) strip = best_advance; // fallback
                std::string stem = word.substr(0, (int)len - strip);

                // SUFX_I (0x200): stem may have had 'y'→'i' change; restore before phonemizing
                static const int SUFX_I_BIT = 0x200;
                static const int SUFX_M_BIT = 0x80000;
                static const int SUFX_E_BIT = 0x100;
                static const int SUFX_V_BIT = 0x800; // use verb form for stem
                if ((best_suffix_flags & SUFX_I_BIT) && !stem.empty() && stem.back() == 'i') {
                    stem.back() = 'y';
                }
                // SUFX_E (0x100): conditionally add 'e' back to stem (the reference RemoveEnding logic).
                // Add 'e' only when:
                //   (a) stem ends in vowel + hard consonant (IsVowel(prev) && IsHardCons(last)),
                //       with exception: stem ends in "ion"
                //   (b) OR stem matches add_e_additions patterns
                // Exception: if the stem (without 'e') is already in the dictionary, skip adding
                // 'e' so the dict lookup succeeds with the bare stem. E.g. "charit"→tSarIt is in
                // dict, so "charitable" should look up "charit" not "charite".
                // See the reference dictionary.c:3107-3138.
                if ((best_suffix_flags & SUFX_E_BIT) && !stem.empty() && stem.back() != 'e') {
                    // First: check if stem-without-e is in dict. If so, skip 'e' addition.
                    std::string stem_norm_bare = toLowerASCII(stem);
                    bool sfx_is_s_bare = (best_phonemes == "s" || best_phonemes == "z" ||
                                          best_phonemes == "I#z" || best_phonemes == "%I#z");
                    // When SUFX_V is set, check if stem+e is in verb_dict_ — if so, force add_e.
                    // This handles "used" → strip 'ed' → stem='us', but "use" is in verb_dict_
                    // (verb form ju:z). Without this, "us" in dict_ (pronoun) blocks 'e' addition.
                    if (!sfx_is_s_bare && (best_suffix_flags & SUFX_V_BIT)) {
                        std::string stem_with_e = stem + "e";
                        std::string stem_e_norm = toLowerASCII(stem_with_e);
                        if (verb_dict_.count(stem_e_norm) > 0) {
                            stem += 'e'; // force add 'e' to get the correct verb stem
                            goto done_sufx_e;
                        }
                    }
                    bool stem_bare_in_dict = false;
                    if (!sfx_is_s_bare) {
                        stem_bare_in_dict = verb_dict_.count(stem_norm_bare) > 0;
                    }
                    if (!stem_bare_in_dict && !onlys_words_.count(stem_norm_bare)
                        && !only_words_.count(stem_norm_bare))  // $only entries are bare-word only
                        stem_bare_in_dict = dict_.count(stem_norm_bare) > 0;
                    if (!stem_bare_in_dict) {
                        static const char* ADD_E_ADDITIONS[] = {
                            "c", "rs", "ir", "ur", "ath", "ns", "u",
                            "spong", "rang", "larg", nullptr
                        };
                        static const std::string VOWELS_INCL_Y = "aeiouy";
                        static const std::string HARD_CONS = "bcdfgjklmnpqstvxz"; // group B
                        bool add_e = false;
                        // Check add_e_additions
                        for (int ai = 0; ADD_E_ADDITIONS[ai]; ai++) {
                            size_t plen = strlen(ADD_E_ADDITIONS[ai]);
                            if (stem.size() >= plen &&
                                stem.compare(stem.size()-plen, plen, ADD_E_ADDITIONS[ai]) == 0) {
                                add_e = true; break;
                            }
                        }
                        // Check vowel + hard consonant at stem end (no "ion" exception)
                        if (!add_e && stem.size() >= 2) {
                            char last = std::tolower((unsigned char)stem.back());
                            char prev = std::tolower((unsigned char)stem[stem.size()-2]);
                            bool last_hard = HARD_CONS.find(last) != std::string::npos;
                            bool prev_vowel = VOWELS_INCL_Y.find(prev) != std::string::npos;
                            if (last_hard && prev_vowel) {
                                // Check "ion" exception: stem ends in "ion"
                                bool ion_exc = (stem.size() >= 3 &&
                                    stem.compare(stem.size()-3, 3, "ion") == 0);
                                if (!ion_exc) add_e = true;
                            }
                        }
                        if (add_e) stem += 'e';
                    }
                }
                done_sufx_e:;

                // the reference always calls TranslateWord2 (full word translation, including suffix
                // stripping) for the stem during SUFFIX-RULE re-phonemization. So we must always
                // allow suffix stripping in the stem. SUFX_M (0x80000) in the reference actually marks
                // that the SUFFIX itself can take further suffix additions, not the stem.
                // Setting this to true ensures e.g. "witnesses" → stem "witness" fires its own
                // "-ness" suffix rule: 'wItn@s' (schwa) not 'wItnEs' (ɛ).
                bool stem_allow_suffix = true;

                std::string stem_ph;
                if (!stem.empty()) {
                    // In the reference, the stem is first looked up in dict (LookupDictList),
                    // then re-translated by rules if not found.
                    // Also check verb_dict_: $verb entries (like "increase") have the correct
                    // pronunciation and should be used when deriving suffix forms (e.g. "increasingly").
                    // $onlys entries are only valid for bare form or 's' suffix; skip them for
                    // non-s suffixes (e.g. -ing, -ly, -ed, etc.). For -s/-es suffixes, onlys is valid.
                    std::string stem_norm = toLowerASCII(stem);
                    bool stem_is_onlys = onlys_words_.count(stem_norm) > 0;
                    // Detect -s/-es suffix phoneme: if suffix phoneme is a plain plural/3ps marker.
                    // For -s suffix: (1) $onlys entries ARE valid; (2) don't use verb_dict_ —
                    // the reference uses the noun/rule-based form for -s derivatives (e.g. "increases"
                    // gets rule-based primary on 1st syllable, not the $verb form's 2nd-syllable stress).
                    bool suffix_is_s = (best_phonemes == "s" || best_phonemes == "z" ||
                                        best_phonemes == "I#z" || best_phonemes == "%I#z");
                    if (suffix_is_s) stem_is_onlys = false; // onlys valid for -s suffix
                    // $only stems (e.g. "down" in "downtown"): their dict entry is for the bare
                    // word only, not as a compound suffix stem. Skip the dict so rules produce
                    // an unstressed phoneme without the leading ',' (e.g. "down"→d'aUn from rules,
                    // not ",daUn" from dict). Unlike $onlys, $only applies to all non-s suffixes.
                    // Note: for -s suffix, $only still valid (like $onlys).
                    if (only_words_.count(stem_norm) && !suffix_is_s) stem_is_onlys = true;
                    // For verb-derived suffixes (-ed, -ing, etc.), use verb_dict_ first:
                    // $verb entries give verb pronunciation (e.g. "use" verb → 'ju:z', not noun 'ju:s').
                    // Only use verb_dict_ when SUFX_V bit is set in the suffix rule — this marks
                    // morphological verbal suffixes (e.g. '-ed', '-ing', '-able'). Compound suffixes
                    // like '-hold', '-fold', '-man' don't have SUFX_V and should use noun/rule form.
                    // Exception: for -s suffix, don't use verb_dict_.
                    auto dt = dict_.end();
                    if (!suffix_is_s && (best_suffix_flags & SUFX_V_BIT)) {
                        auto vt = verb_dict_.find(stem_norm);
                        if (vt != verb_dict_.end()) dt = vt;
                    }
                    bool used_onlys_bare = false;
                    if (dt == dict_.end()) {
                        // For -s suffix: $onlys bare-word pronunciation is preferred (valid for bare/+s).
                        if (suffix_is_s) {
                            auto obit = onlys_bare_dict_.find(stem_norm);
                            if (obit != onlys_bare_dict_.end()) {
                                stem_ph = obit->second;
                                auto sp = stress_pos_.find(stem_norm);
                                if (sp != stress_pos_.end())
                                    stem_ph = applyStressPosition(stem_ph, sp->second);
                                used_onlys_bare = true;
                            }
                        }
                        if (!used_onlys_bare) {
                            dt = dict_.find(stem_norm);
                            if (stem_is_onlys) dt = dict_.end(); // skip $onlys entry
                        }
                    }
                    // When stem ends in 'e' and isn't in dict, try without the trailing 'e'.
                    // This handles "-oes" plurals: "tornadoes" → strip 's' → "tornadoe" (not in dict)
                    // → try "tornado" → in dict (tO@n'eIdoU). Same for "volcanoes", "tomatoes", etc.
                    std::string stem_no_e_norm;
                    if (dt == dict_.end() && stem_norm.size() > 1 && stem_norm.back() == 'e') {
                        stem_no_e_norm = stem_norm.substr(0, stem_norm.size() - 1);
                        auto dt_noe = dict_.find(stem_no_e_norm);
                        if (dt_noe != dict_.end() && !onlys_words_.count(stem_no_e_norm)
                            && !only_words_.count(stem_no_e_norm))
                            dt = dt_noe;
                    }
                    if (used_onlys_bare) {
                        // stem_ph already set from onlys_bare_dict_ above
                    } else if (dt != dict_.end() && !stem_is_onlys) {
                        stem_ph = dt->second;
                        auto sp = stress_pos_.find(stem_norm);
                        if (sp != stress_pos_.end())
                            stem_ph = applyStressPosition(stem_ph, sp->second);
                        // Note: stems with leading ',' (like "near"=",ni@3") are left as-is.
                        // phonemizeText promotes ',' → '\'' for isolated words; in sentence
                        // context the secondary stress is preserved (e.g. "nearly"→nˌɪɹli).
                    } else {
                        // Re-phonemize stem via rules (recursive).
                        // Combine the stem's own $altN flags (from its dict entry) with the
                        // original word's alt flags. This ensures:
                        // (a) "deliberately" → stem "deliberate" gets its own $alt2 flag, so
                        //     the rule ate(_$w_alt2+ → @t fires correctly.
                        // (b) "grammatical" ($alt3) → stem "grammatic" (no dict entry) inherits
                        //     alt3 from the parent word, so X) a ($w_alt3++++++ → a# fires for the
                        //     first vowel in the stem (matching the reference 'ɡɹɐm' not 'ɡɹæm').
                        {
                            auto salt = word_alt_flags_.find(stem_norm);
                            // Also try stem + 'e' (magic-e restored form) for alt flags,
                            // e.g. "fertil" → look up "fertile" which has $alt2.
                            if (salt == word_alt_flags_.end())
                                salt = word_alt_flags_.find(stem_norm + "e");
                            int stem_own_alt = (salt != word_alt_flags_.end()) ? salt->second : 0;
                            int combined_stem_alt = stem_own_alt | word_alt_flags;
                            stem_ph = applyRules(stem, stem_allow_suffix, combined_stem_alt, false, true);
                        }
                        // Apply $N stress position override for stems not in dict_.
                        // Skip noun-form-only overrides ($N $onlys flag-only entries like
                        // "content $1 $onlys") ONLY when the current suffix is verbal (SUFX_V).
                        // For non-verbal compound suffixes (e.g. "-ship", "-hood", "-dom"),
                        // the noun-form stress DOES apply — e.g. "intern $1 $onlys" + "-ship"
                        // → "internship" gets 1st-syllable stress from the noun form.
                        // Also skip for $verb flag words — separate verb-pronunciation handling.
                        auto sp2 = stress_pos_.find(stem_norm);
                        if (sp2 != stress_pos_.end() &&
                            (!noun_form_stress_.count(stem_norm) || !(best_suffix_flags & SUFX_V_BIT)) &&
                            !verb_flag_words_.count(stem_norm))
                            stem_ph = applyStressPosition(stem_ph, sp2->second);
                    }
                    stem_ph_done:;
                }

                // -ed devoicing: applies when any RULE_ENDING produces 'd#' (past tense).
                // After voiceless consonants, devoice to 't'. After t/d, insert epenthetic
                // vowel 'I#d'. The SUFX_V flag is NOT required — the reference applies this logic
                // for all RULE_ENDING 'd#' suffixes based on the re-translated stem's last
                // phoneme. E.g., `@@e) d (_S1 d#` (no SUFX_V) fires for "noticed" → stem
                // "notice" ends in 's' → devoice → 't'. For "faced"/"raced", the `@@e)d`
                // rule does NOT fire (phoneme-based @@ check: only 1 syllable in "feIs"/"reIs")
                // so simple `d→d` fires instead → no devoicing needed.
                if (best_phonemes == "d#" && !stem_ph.empty()) {
                    // Find last non-stress-marker char in stem_ph to identify last phoneme.
                    char last_ph = 0;
                    for (int sj = (int)stem_ph.size() - 1; sj >= 0; sj--) {
                        char c = stem_ph[sj];
                        // Skip stress/boundary markers and the \x01 rule-boundary char
                        if (c != '\'' && c != ',' && c != '%' && c != '=' && c != '\x01') {
                            last_ph = c;
                            break;
                        }
                    }
                    // Voiceless consonant last-chars: p, t, k, f, T(θ), S(ʃ), C(ç), x, X, h, s
                    static const std::string VOICELESS_LAST = "ptkfTSCxXhs";
                    if (last_ph == 't' || last_ph == 'd') {
                        best_phonemes = "I#d";
                    } else if (VOICELESS_LAST.find(last_ph) != std::string::npos) {
                        best_phonemes = "t";
                    }
                    // else: voiced consonant or vowel → keep d# (maps to /d/)
                }

                if (std::getenv("PHON_DEBUG"))
                    std::cerr << "[SUFFIX-RULE] word=" << word
                              << " strip=" << strip << " stem='" << stem
                              << "' stem_ph='" << stem_ph << "' suf_ph='" << best_phonemes << "'\n";

                // Return stem + suffix phonemes. Discard 'phonemes' accumulated so far.
                return stem_ph + best_phonemes;
            }

            // PREFIX rule: re-translate suffix as a new word so word-start context rules fire.
            // the reference SUFX_P mechanism: after consuming the prefix, TranslateRules restarts
            // for the suffix (e.g., "inter" + "national" → "national" gets _n) a (tional rule).
            if (best_is_prefix && pos + best_advance < len) {
                std::string suffix = word.substr(pos + best_advance);

                // When the @P prefix phoneme contains a full stressable vowel (like 'oU' in
                // "open" = "oUp@n") AND the suffix starts with a consonant, skip the @P
                // retranslation mechanism. Instead, continue processing the full word in-context.
                // This is necessary because suffix rules like `&) y (_ %i` check phonemes_so_far
                // for a stressable vowel ('&'). Standalone "ly" has no prior vowel in phonemes_so_far,
                // so 'y' gets 'aI' (word-boundary rule) instead of '%i' (happy tensed).
                // In-context processing of "openly": phonemes_so_far = "oUp@nl" has 'oU' → & fires.
                // E.g. "openly": "open" prefix → "oUp@n" (stressable 'oU'), suffix "ly" starts with 'l'
                //      → skip @P → process 'l','y' in context → 'y' → '%i' → ˈoʊpənli ✓.
                // Note: phonemes already contains the prefix phoneme (added at line 1472 above).
                // We just need to advance pos and continue the main loop.
                // Condition: prefix phoneme has a FULL stressable vowel (not reduced 'I2','I#','@','i').
                // This excludes prefixes like "dis" (dI2s = /dɪs/, only reduced 'I2') which SHOULD
                // use @P retranslation (e.g. "discontented" → correct stress via @P).
                {
                    bool prefix_has_stress =
                        (phonemes.find('\'') != std::string::npos ||
                         phonemes.find(',')  != std::string::npos ||
                         phonemes.find('%')  != std::string::npos);
                    bool cons_initial = !suffix.empty() && !isVowelLetter(suffix[0]);
                    // Full stressable vowel codes (non-reduced) in the prefix phoneme:
                    static const char* FULL_VOWELS[] = {
                        "eI","aI","aU","OI","oU","3:","A:","i:","u:","O:","e:","a:",
                        "aI@","aU@","oU#","i@","e@","A@","U@","O@",
                        "a","A","E","V","0","o",
                        nullptr
                    };
                    bool has_full_vowel = false;
                    {
                        const std::string& pfx = phonemes;
                        for (size_t pi = 0; pi < pfx.size() && !has_full_vowel; pi++) {
                            for (int fi = 0; FULL_VOWELS[fi]; fi++) {
                                const char* fv = FULL_VOWELS[fi];
                                size_t fvlen = strlen(fv);
                                if (pi + fvlen <= pfx.size() && pfx.compare(pi, fvlen, fv) == 0) {
                                    has_full_vowel = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (!prefix_has_stress && has_full_vowel) {
                        pos += best_advance;
                        continue;
                    }
                }
                // Use wordToPhonemes so dict entries are consulted for the suffix
                // (e.g. "unprecedented" → suffix "precedent" is in dict as prEsI#d@nt).
                // applyRules alone would miss the dict and produce a different phoneme.
                //
                // Exception: $onlys words (e.g. "suspect") should not use their dict entry
                // when they appear as the stem of a suffix-stripped compound. In the reference,
                // the full word "unsuspecting" triggers PREFIX "un" → retranslates "suspecting"
                // (with "-ing" still present), so SUFFIX-ING sees "suspecting" with $onlys
                // and applies rules. In our code, SUFFIX-ING runs first (stripping "-ing"),
                // leaving stem "unsuspect" → PREFIX → suffix "suspect". The $onlys flag
                // should cause rules to be applied (not dict) for these compound-stem contexts.
                std::string sfx_ph;
                if (onlys_words_.count(suffix)) {
                    sfx_ph = processPhonemeString(applyRules(suffix));
                } else if (noun_form_stress_.count(toLowerASCII(suffix)) ||
                           verb_flag_words_.count(toLowerASCII(suffix))) {
                    // Noun-form-only ($N $onlys flag-only) or $verb-flagged words:
                    // skip $N stress override in compound context (use verb/default form).
                    // e.g. "dis-" + "content": skip $1 noun-form stress → verb form kəntˈɛnt.
                    sfx_ph = processPhonemeString(applyRules(suffix));
                } else {
                    sfx_ph = wordToPhonemes(suffix);
                }
                std::string sfx_raw = sfx_ph; // for debug logging
                // The prefix phoneme already encodes stress: % (unstressed) or , (secondary).
                // The en_rules PREFIX entries directly specify: "out" → %aUt (unstressed),
                // "over"/"under" → ,oUv3/,Vnd3 (secondary). We preserve that as-is.
                //
                // Compound stress rule: when the prefix carries primary '\'' AND the suffix
                // also has primary '\'' AND the suffix is MULTI-SYLLABIC (2+ vowels), demote
                // the prefix's '\'' to ',' (secondary).
                // e.g., "micro" (m'aIkroU) + "plastics" (pl'aast=Iks, 2 syllables)
                // → m,aIkroUpl'aast=Iks → mˌaɪkɹoʊplˈæstɪks.
                // For mono-syllabic suffixes ("wave","chip"), keep prefix primary:
                // "micro"+'wave' → m'aIkroUw'eIv → mˈaɪkɹoʊwˌeɪv (secondary on suffix by 5a).
                // When prefix has ',' or '%' already, no demotion needed.
                if (phonemes.find('\'') != std::string::npos &&
                    sfx_ph.find('\'') != std::string::npos) {
                    // Compound stress rule for prefix+suffix combinations:
                    // Count vowel-letter groups in the suffix WORD to determine syllable count.
                    // - Multi-syllabic suffix (2+ vowel groups): demote PREFIX '\'' → ','.
                    //   e.g. "micro"+'plastics' (a,i = 2 groups) → secondary on micro.
                    // - Mono-syllabic suffix (1 vowel group): demote SUFFIX '\'' → ','.
                    //   e.g. "micro"+'wave' (a = 1 group) → secondary on wave.
                    //
                    // Exception: if the PREFIX phoneme is mono-syllabic (1 vowel phoneme code),
                    // ALWAYS demote the suffix's primary instead of the prefix's.
                    // e.g. "news" (n'u:z, 1 vowel) + "paper" (2 letter syll) → primary on news.
                    // e.g. "news" + "caster" (2 letter syll) → primary on news.
                    // Multi-syllabic prefixes (e.g. "micro" = m'aIkroU, 2 vowels) still use
                    // the suffix-syllable count to decide.
                    static const std::string VOW_LET = "aeiouAEIOU";
                    int sfx_syllables = 0;
                    bool in_vowel = false;
                    for (char ch : suffix) {
                        if (VOW_LET.find(ch) != std::string::npos) {
                            if (!in_vowel) { sfx_syllables++; in_vowel = true; }
                        } else {
                            in_vowel = false;
                        }
                    }
                    // Subtract silent trailing 'e' (magic-e): doesn't add a syllable
                    if (!suffix.empty() && (suffix.back() == 'e' || suffix.back() == 'E') &&
                        suffix.size() >= 2 && VOW_LET.find(suffix[suffix.size()-2]) == std::string::npos)
                        sfx_syllables = std::max(1, sfx_syllables - 1);

                    // Count vowel phoneme codes in the prefix.
                    int pfx_vowels = 0;
                    {
                        static const char* MC_VPH[] = {
                            "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r","o@r",
                            "A@r","e@r","eI","aI","aU","OI","oU","IR","VR","U@","A@","e@",
                            "i@","O@","o@","3:","A:","i:","u:","O:","e:","a:","aa",
                            "@L","@2","@5","I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A#",
                            nullptr
                        };
                        for (size_t pi2 = 0; pi2 < phonemes.size(); ) {
                            char c2 = phonemes[pi2];
                            if (c2 == '\'' || c2 == ',' || c2 == '%' || c2 == '=') { pi2++; continue; }
                            bool mch = false;
                            for (int mi = 0; MC_VPH[mi]; mi++) {
                                int ml = (int)strlen(MC_VPH[mi]);
                                if ((int)pi2 + ml <= (int)phonemes.size() &&
                                    phonemes.compare(pi2, ml, MC_VPH[mi]) == 0) {
                                    if (isVowelCode(std::string(MC_VPH[mi], ml))) pfx_vowels++;
                                    pi2 += ml; mch = true; break;
                                }
                            }
                            if (!mch) {
                                if (isVowelCode(std::string(1, c2))) pfx_vowels++;
                                pi2++;
                            }
                        }
                    }

                    // Guard: if the prefix's LAST phoneme vowel is a schwa-type (3, @, I2, I#)
                    // and sfx_syllables == 2, keep the prefix's primary stress (demote suffix).
                    // E.g. "super" (s'u:p3) + "nova"/"market" (2 syll) → primary on super.
                    // This reflects the reference compound behavior where short-schwa-ending prefixes
                    // keep primary for 2-syllable suffixes (the reference encodes this via @P vs @@@P rules).
                    // Prefixes ending in a FULL vowel (e.g. "micro" ends in oU) still demote for 2-syll.
                    bool pfx_ends_schwa = false;
                    if (sfx_syllables == 2 && pfx_vowels >= 2) {
                        // Find last vowel phoneme code in prefix
                        std::string last_v;
                        static const char* MC_VEND[] = {
                            "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r","o@r",
                            "A@r","e@r","eI","aI","aU","OI","oU","IR","VR","U@","A@","e@",
                            "i@","O@","o@","3:","A:","i:","u:","O:","e:","a:","aa",
                            "@L","@2","@5","I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A#",
                            nullptr
                        };
                        for (size_t pi2 = 0; pi2 < phonemes.size(); ) {
                            char c2 = phonemes[pi2];
                            if (c2 == '\'' || c2 == ',' || c2 == '%' || c2 == '=') { pi2++; continue; }
                            bool mch = false;
                            for (int mi = 0; MC_VEND[mi]; mi++) {
                                int ml = (int)strlen(MC_VEND[mi]);
                                if ((int)pi2 + ml <= (int)phonemes.size() &&
                                    phonemes.compare(pi2, ml, MC_VEND[mi]) == 0) {
                                    if (isVowelCode(std::string(MC_VEND[mi], ml))) last_v = std::string(MC_VEND[mi], ml);
                                    pi2 += ml; mch = true; break;
                                }
                            }
                            if (!mch) {
                                if (isVowelCode(std::string(1, c2))) last_v = std::string(1, c2);
                                pi2++;
                            }
                        }
                        // Schwa-type last vowel: 3 (ɚ), @ (ə), I2, I#, a#, @2, @5, @L
                        pfx_ends_schwa = (last_v == "3" || last_v == "3:" || last_v == "@" ||
                                          last_v == "@2" || last_v == "@5" || last_v == "@L" ||
                                          last_v == "I2" || last_v == "I#" || last_v == "a#");
                    }

                    if (sfx_syllables >= 2 && pfx_vowels >= 2 && !pfx_ends_schwa) {
                        // Multi-syllabic suffix AND multi-syllabic prefix (ending in full vowel):
                        // demote prefix primary to secondary.
                        // e.g. "micro" (m'aIkroU, ends in oU) + "plastics" → secondary on micro.
                        size_t pp = phonemes.find('\'');
                        if (pp != std::string::npos) phonemes[pp] = ',';
                    } else if (pfx_vowels == 1) {
                        // Mono-syllabic prefix (1 vowel, e.g. "news", "note", "out"):
                        // REMOVE the suffix's primary stress entirely (leave vowel unmarked/unstressed).
                        // the reference: "newspaper"→n'u:zpeIp3, "newsroom"→n'u:zru:m (no secondary on suffix).
                        size_t sp = sfx_ph.find('\'');
                        if (sp != std::string::npos) sfx_ph.erase(sp, 1);
                    } else {
                        // Mono-syllabic suffix OR prefix ends in schwa:
                        // demote suffix primary to secondary.
                        // e.g. "super" (s'u:p3, ends in 3) + "nova"/"market" → primary on super.
                        size_t sp = sfx_ph.find('\'');
                        if (sp != std::string::npos) sfx_ph[sp] = ',';
                    }
                    // Post-demotion: if the comma is immediately before a phUNSTRESSED vowel
                    // (@, I#, I2, a#), strip it entirely — schwa-type vowels never carry
                    // secondary stress. E.g. "battlement": m,@nt → m@nt (no stress on -ment).
                    {
                        size_t cp = sfx_ph.find(',');
                        if (cp != std::string::npos && cp + 1 < sfx_ph.size()) {
                            char nc = sfx_ph[cp+1];
                            bool phU = (nc == '@') ||
                                       (nc == 'I' && cp+2 < sfx_ph.size() &&
                                        (sfx_ph[cp+2]=='#'||sfx_ph[cp+2]=='2')) ||
                                       (nc == 'a' && cp+2 < sfx_ph.size() &&
                                        sfx_ph[cp+2]=='#');
                            if (phU) sfx_ph.erase(cp, 1);
                        }
                    }
                }
                // Strip \x01 rule-boundary markers from the prefix phoneme before combining.
                // The suffix (sfx_ph) was fully processed by wordToPhonemes, so its \x01
                // markers were already stripped. If the prefix still carries a \x01 (from
                // the rule-match at line 1501), the combined string would have a non-empty
                // rule_boundary_after in processPhonemeString — causing DIMINISHED steps
                // (5.5c, 5.5b, etc.) to re-fire on a string that was already processed.
                // the reference only runs SetWordStress on the suffix phoneme (not the combined
                // prefix+suffix), so the combined string must be treated as dict-like (no \x01).
                phonemes.erase(std::remove(phonemes.begin(), phonemes.end(), '\x01'), phonemes.end());
                if (std::getenv("PHON_DEBUG"))
                    std::cerr << "[PREFIX-RETRANS] suffix=" << suffix << " sfx_raw=" << sfx_raw
                              << " sfx_ph=" << sfx_ph
                              << " combined=" << (phonemes + sfx_ph) << "\n";
                phonemes += sfx_ph;
                return phonemes;
            }

            // Mark RULE_DEL_FWD characters (magic-e) as replaced_e (still processed but silenced if default)
            if (best_del_count > 0 && best_del_start >= 0) {
                for (int d = 0; d < best_del_count && best_del_start + d < len; d++) {
                    replaced_e[best_del_start + d] = true;
                }
            }
            pos += best_advance;
        } else {
            // No rule found - skip the character silently
            pos++;
        }
    }

    if (out_replaced_e) *out_replaced_e = replaced_e;
    if (out_pos_visited) *out_pos_visited = pos_visited;
    return phonemes;
}

// ============================================================
// Word to phoneme codes
// ============================================================
std::string IPAPhonemizer::wordToPhonemes(const std::string& word) const {
    std::string norm = toLowerASCII(word);

    // $capital: if word starts with a capital letter, check capital_dict_ first.
    if (!word.empty() && (unsigned char)word[0] >= 'A' && (unsigned char)word[0] <= 'Z') {
        auto cit = capital_dict_.find(norm);
        if (cit != capital_dict_.end()) {
            if (std::getenv("PHON_DEBUG")) std::cerr << "[DICT_CAP] " << norm << " -> " << cit->second << "\n";
            return processPhonemeString(cit->second);
        }
    }

    // Try full word in dictionary.
    // $onlys bare-word overrides take priority over the plain entry for bare-word lookup.
    // (Suffix stripping uses dict_ directly, bypassing onlys_bare_dict_.)
    {
        auto obit = onlys_bare_dict_.find(norm);
        auto it = (obit != onlys_bare_dict_.end()) ? dict_.end() : dict_.find(norm);
        const std::string* raw_ptr = nullptr;
        if (obit != onlys_bare_dict_.end()) {
            raw_ptr = &obit->second;
            if (std::getenv("PHON_DEBUG")) std::cerr << "[DICT_ONLYS] " << norm << " -> " << *raw_ptr << "\n";
        } else if (it != dict_.end()) {
            raw_ptr = &it->second;
            if (std::getenv("PHON_DEBUG")) std::cerr << "[DICT] " << norm << " -> " << *raw_ptr << "\n";
        }
        if (raw_ptr) {
            // Apply processPhonemeString to dict entries so they get:
            // - flap rule (e.g. "committee" → kəmˈɪɾi)
            // - secondary stress insertion (e.g. from our added dict overrides)
            // - @r → 3 conversion (American English)
            // Note: entries with % prefix (unstressed function words) are left alone
            //       except for minor normalizations (they won't get primary stress added).
            std::string raw = *raw_ptr;
            auto sit = stress_pos_.find(norm);
            if (sit != stress_pos_.end())
                raw = applyStressPosition(raw, sit->second);
            // $strend2 words with bare phonemes use final-syllable stress (the reference "end stress").
            bool is_strend = strend_words_.count(norm) > 0;
            return processPhonemeString(raw, is_strend);
        }
    }

    // Handle hyphenated compounds: "peer-reviewed" → phonemize "peer" + "reviewed" separately.
    // the reference treats each hyphen-separated segment as an independent word for phonemization,
    // so word-start rules (e.g., re-→rI#, be-→bI#) fire correctly on each segment.
    // Only split if not in the dictionary (handled above) and contains at least one hyphen.
    {
        size_t hyphen_pos = norm.find('-');
        if (hyphen_pos != std::string::npos && hyphen_pos > 0 && hyphen_pos + 1 < norm.size()) {
            std::string result;
            size_t seg_start = 0;
            bool all_ok = true;
            while (seg_start < norm.size()) {
                size_t next_hyphen = norm.find('-', seg_start);
                std::string seg = norm.substr(seg_start, next_hyphen == std::string::npos ? std::string::npos : next_hyphen - seg_start);
                if (seg.empty()) { all_ok = false; break; }
                // Each segment must have at least one letter
                bool has_letter = false;
                for (char c : seg) if (std::isalpha(c)) { has_letter = true; break; }
                if (!has_letter) { all_ok = false; break; }
                std::string seg_ipa = wordToPhonemes(seg);
                if (seg_ipa.empty()) { all_ok = false; break; }
                result += seg_ipa;
                if (next_hyphen == std::string::npos) break;
                seg_start = next_hyphen + 1;
            }
            if (all_ok && !result.empty()) {
                if (std::getenv("PHON_DEBUG"))
                    std::cerr << "[HYPHEN] " << norm << " -> " << result << "\n";
                return result;
            }
        }
    }

    // Handle possessive "'s": "people's" → phonemize "people" + suffix
    // Voice: sibilant-ending → ᵻz; unvoiced → s; else → z
    if (norm.size() >= 3 && norm[norm.size()-2] == '\'' && norm.back() == 's') {
        std::string base_poss = norm.substr(0, norm.size()-2);
        std::string base_ipa = wordToPhonemes(base_poss);
        if (!base_ipa.empty()) {
            // Get raw phoneme code of base
            std::string raw_code;
            auto poss_dict = dict_.find(base_poss);
            if (poss_dict != dict_.end()) {
                raw_code = processPhonemeString(poss_dict->second);
            } else {
                raw_code = processPhonemeString(applyRules(base_poss));
            }
            // Determine suffix phoneme using LETTER-based rules (the reference .group ' rules).
            // the reference "'s" suffix rules match on the ending LETTERS of the base word,
            // not the final phoneme. Rules (from en_rules .group '):
            //   sh → %I#z
            //   ch → z/2 (= check last phoneme: sibilant→I2z, unvoiced→s, else→z)
            //   se/s/ce/x → %I#z
            //   f/p/t/k → s
            //   och → s
            //   default → z
            // The z/2 phoneme: if prev=sibilant → InsertPhoneme(I2)+z, if prev=unvoiced → s, else z.
            std::string poss_suffix = "z"; // default
            std::string bl = base_poss; // base lowercase already normalized
            // Helper: last N letters
            auto ends_with_letters = [&](const std::string& sfx) {
                return bl.size() >= sfx.size() &&
                       bl.compare(bl.size() - sfx.size(), sfx.size(), sfx) == 0;
            };
            // Find last phoneme code in raw_code (skip stress markers, hyphens)
            std::string last_ph_code;
            {
                std::string rc = raw_code;
                // scan backwards to find the last 1-2 char phoneme code
                size_t ri = rc.size();
                while (ri > 0) {
                    ri--;
                    char c = rc[ri];
                    if (c == '\'' || c == ',' || c == '%' || c == '=' || c == '-') continue;
                    last_ph_code = std::string(1, c);
                    // check for 2-char code (e.g. "tS", "dZ")
                    if (ri >= 1) {
                        std::string two = rc.substr(ri-1, 2);
                        if (two == "tS" || two == "dZ" || two == "O:" || two == "A:" ||
                            two == "i:" || two == "u:" || two == "e:" || two == "I#" ||
                            two == "I2" || two == "@L" || two == "3:" || two == "eI" ||
                            two == "aI" || two == "aU" || two == "oU" || two == "OI")
                            last_ph_code = two;
                    }
                    break;
                }
            }
            // 3-char endings first
            if (ends_with_letters("och")) {
                poss_suffix = "s";
            // ch → z/2 logic (depends on last phoneme)
            } else if (ends_with_letters("ch")) {
                // sibilants: tS, dZ, s, z, S, Z
                static const std::vector<std::string> SIBILANTS_PH = {"tS","dZ","s","z","S","Z"};
                // voiced phoneme codes (consonants and vowels are voiced by default)
                static const std::string UNVOICED_PH_CHARS = "ptkfsTSx"; // p,t,k,f,s,T(θ),S(ʃ),x
                bool is_sib = false;
                for (auto& sp : SIBILANTS_PH) if (last_ph_code == sp) { is_sib = true; break; }
                if (is_sib) {
                    poss_suffix = "I2z"; // InsertPhoneme(I2) + ChangePhoneme(z)
                } else if (!last_ph_code.empty() &&
                           UNVOICED_PH_CHARS.find(last_ph_code[0]) != std::string::npos) {
                    poss_suffix = "s";
                } else {
                    poss_suffix = "z";
                }
            // 2-char endings
            } else if (ends_with_letters("se") || ends_with_letters("ce") ||
                       ends_with_letters("sh")) {
                poss_suffix = "I#z";
            // 1-char endings
            } else if (!bl.empty()) {
                char lc = bl.back();
                if (lc == 's' || lc == 'z' || lc == 'x') {
                    poss_suffix = "I#z";
                } else if (lc == 'f' || lc == 'p' || lc == 't' || lc == 'k') {
                    poss_suffix = "s";
                }
                // else: default z (covers e, h, vowels, voiced consonants, etc.)
            }
            if (std::getenv("PHON_DEBUG"))
                std::cerr << "[POSSESSIVE] " << norm << " base=" << base_poss
                          << " last_letter=" << (bl.empty() ? '?' : bl.back())
                          << " suf=" << poss_suffix << "\n";
            return processPhonemeString(raw_code + poss_suffix);
        }
    }

    // For single-letter words, try underscore prefix (letter name).
    // Exception: "a" as article — its pronunciation depends on sentence context
    // and is handled specially in phonemizeText (not here).
    if (norm.size() == 1 && norm != "a") {
        auto it = dict_.find("_" + norm);
        if (it != dict_.end()) return processPhonemeString(it->second);
    }

    // Try morphological suffix stripping: -ing and -ed
    // the reference handles these via suffix rules internally; we approximate by phonemizing the stem.
    if (norm.size() >= 5) {
        static const std::string VOWEL_CODES = "aAeEIiOUVu03@o";

        // Get phonemes for a candidate stem (dict lookup or rules+postprocess).
        // Returns "" if stem is invalid (no vowel letter or no vowel phoneme in result).
        // For verb-derived forms (-ing/-ed), the verb form dictionary takes priority
        // so that e.g. "live" uses verb form "lIv" not adj/noun form "laIv".
        auto stemPh = [&](const std::string& stem) -> std::string {
            if (stem.size() < 2) return "";
            // Reject stems with no vowel letter (e.g., "spr", "str")
            bool hv = false;
            for (char c : stem) if (isVowelLetter(c)) { hv = true; break; }
            if (!hv) return "";
            std::string ph;
            // Check verb_dict_ first (verb form needed for -ing/-ed derivatives)
            auto vt = verb_dict_.find(stem);
            if (vt != verb_dict_.end()) ph = vt->second;
            else {
                // $onlys dict entries are only valid for the bare form or with 's' suffix.
                // $only dict entries are only valid for the isolated bare word (not as any stem).
                // For non-s suffix stripping (e.g. -ed, -ing, -able), skip both and use rules.
                bool is_onlys = onlys_words_.count(stem) > 0 || only_words_.count(stem) > 0;
                auto jt = dict_.find(stem);
                if (jt == dict_.end() || is_onlys) {
                    // Magic-e restoration: try stem+"e" in dict before falling back to rules.
                    // E.g. "argued" → stem "argu" → not in dict → try "argue" → A@gju: (with j).
                    auto je = dict_.find(stem + "e");
                    if (je != dict_.end() && onlys_words_.count(stem + "e") == 0 && only_words_.count(stem + "e") == 0) jt = je;
                    else {
                        auto ve = verb_dict_.find(stem + "e");
                        if (ve != verb_dict_.end()) { ph = ve->second; jt = dict_.end(); }
                    }
                }
                if (!ph.empty()) {} // already set from verb_dict_ magic-e
                else if (jt != dict_.end() && !is_onlys) ph = jt->second;
                else {
                    // When stem has $verb flag-only entry (e.g. "deliberate $verb"), use alt1
                    // rules (verb form gives eIt for -ate, not alt2's @t noun/adj form).
                    int stem_alt_flags = verb_flag_words_.count(stem) ? 1 : -1;
                    std::string raw = applyRules(stem, true, stem_alt_flags);
                    // Apply $N stress position override if stem has one (e.g. "maintain $2").
                    // Skip for: (1) noun-form-only overrides ($N $onlys flag-only entries),
                    //            (2) words with $verb flag-only entries (their verb form uses rules).
                    auto sp = stress_pos_.find(stem);
                    if (sp != stress_pos_.end() &&
                        !noun_form_stress_.count(stem) &&
                        !verb_flag_words_.count(stem))
                        raw = applyStressPosition(raw, sp->second);
                    ph = processPhonemeString(raw);
                }
            }
            // Reject if result has no vowel phoneme code (e.g., silent-e-only stem)
            for (char c : ph) if (VOWEL_CODES.find(c) != std::string::npos) return ph;
            return "";
        };

        // -ing suffix
        if (norm.compare(norm.size()-3, 3, "ing") == 0) {
            std::string base = norm.substr(0, norm.size()-3);
            std::string sph;
            // For "-nging" endings where the base is not in the dictionary, bypass the
            // custom handler and let applyRules handle the full word.
            // the reference handles these via two-char group rules like "enging EndZIN"
            // (for "challenging", "exchanging") that fire on the full word but not on the stem.
            // Without this exclusion, "challeng" would get wrong rules vs "challenging".
            if (!(base.size() >= 2 && base.compare(base.size()-2, 2, "ng") == 0 &&
                  dict_.find(base) == dict_.end() && verb_dict_.find(base) == verb_dict_.end()))
            { // begin non-bypass block
            // For CVC-pattern stems (consonant after vowel), try magic-e stem first.
            // e.g. "writing" → base="writ" (CVC: i before t) → try "write" before "writ".
            // Special case: base ends in 'nc' (soft-c after nasal) → "danc"→"dance".
            // For CVCC stems (consonant cluster), try base directly first.
            // e.g. "singing" → base="sing" (ng is a cluster) → use "sing" not "singe".
            bool cvc_pattern = base.size() >= 2 &&
                !isVowelLetter(base.back()) &&
                isVowelLetter(base[base.size()-2]);
            // Also extend to CVRC pattern: vowel + 'r' + soft-consonant (e.g. "charg"→"charge",
            // "forc"→"force"). Magic-e is needed for soft-g/-c: "charge"→dʒ vs "charg"→ɡ.
            // Only applies when the final consonant has a magic-e form ('g'→dʒ, 'c'→s).
            // DO NOT extend to 'm', 'n', 't', 'p', etc. (e.g. "perform", "turn") — those words
            // correctly use rules directly without magic-e restoration.
            if (!cvc_pattern && base.size() >= 3 &&
                (base.back() == 'g' || base.back() == 'c') &&
                base[base.size()-2] == 'r' &&
                isVowelLetter(base[base.size()-3]))
                cvc_pattern = true;
            // For CVC bases ending in vowel+'n' (e.g. -en, -an, -in, -on, -un):
            // Heuristic: get the base's phonemes. If they end in '@n'/'@N' (schwa+n), the base
            // is a real word with a reduced vowel+n suffix (e.g. "harden"→h'A@d@n, "open"→oUp@n).
            // In that case, don't treat it as a magic-e stem.
            // If the base phonemes end in a stressed vowel+n (e.g. "confin"→k%@nf'In), it's not
            // a real word → proceed with magic-e to find "confine".
            if (cvc_pattern && base.size() >= 2 && base.back() == 'n') {
                char prev_vowel = base[base.size()-2];
                if (prev_vowel == 'e' || prev_vowel == 'a' || prev_vowel == 'i' ||
                    prev_vowel == 'o' || prev_vowel == 'u') {
                    std::string base_ph = stemPh(base);
                    if (!base_ph.empty() && base_ph.size() >= 2) {
                        // Check if base phoneme ends in '@n' or '@N' (schwa+nasal = real word)
                        char last2 = base_ph.back();
                        char last1 = base_ph[base_ph.size()-2];
                        bool ends_in_schwa_n = (last1 == '@' && (last2 == 'n' || last2 == 'N'));
                        // Also detect syllabic 'n-' ending: last char='-', second-to-last='n'
                        // e.g. "threaten" → phoneme ends in '?n-' (glottal stop + syllabic n)
                        //      "kitten" → phoneme ends in '*n-' (flap + syllabic n)
                        bool ends_in_syllabic_n = (last1 == 'n' && last2 == '-');
                        if (ends_in_schwa_n || ends_in_syllabic_n) cvc_pattern = false;  // Real word: harden, open, happen, threaten
                    }
                }
            }
            bool nc_pattern = base.size() >= 2 && base.back() == 'c' &&
                base[base.size()-2] == 'n';
            // Bases ending in 'el' (vowel+l): the 'e' is always a schwa, not a magic-e vowel.
            // E.g., "travel"→"traveled" should use stemPh("travel"), not stemPh("travele").
            // Words like "model", "cancel", "level" all have schwa+l endings.
            // (Contrast with "smil"→"smile" where 'i' before 'l' IS a magic-e pattern.)
            if (cvc_pattern && base.size() >= 2 && base.back() == 'l' &&
                base[base.size()-2] == 'e')
                cvc_pattern = false;
            // Bases ending in vowel+'w' (e.g. "renew", "stew"): 'w' is a digraph semi-vowel,
            // not a magic-e consonant. Adding 'e' would create "renewe" and trigger wrong rules.
            if (cvc_pattern && base.back() == 'w')
                cvc_pattern = false;
            // Bases ending in 'e'+'r' are rhotic schwa (ɚ) stems, not magic-e stems.
            // E.g. "ponder" → "pondering" should use "ponder" directly (not "pondere"
            // which triggers `d)ere(_` → 'i@3' ɪɹ instead of correct '%3' ɚ).
            // Other vowel+'r' endings ('ir', 'or', 'ur') may still need magic-e
            // (e.g. "retir"→"retire", "explor"→"explore").
            if (cvc_pattern && base.size() >= 2 && base.back() == 'r' &&
                std::tolower((unsigned char)base[base.size()-2]) == 'e')
                cvc_pattern = false;
            // If the base has an explicit $N stress override in the dictionary,
            // skip magic-e and use the base directly (e.g. "maintain $2" → "maintaining").
            // Note: $only words are excluded — their dict entry is restricted to bare/s-suffix
            // forms and should NOT suppress magic-e for -ing derivation.
            // E.g. "guid" has $only → "guiding" should use "guide" not "guid".
            bool base_has_stress_override = (stress_pos_.find(base) != stress_pos_.end() ||
                                             (dict_.find(base) != dict_.end() && !onlys_words_.count(base) && !only_words_.count(base)) ||
                                             verb_dict_.find(base) != verb_dict_.end());
            if ((cvc_pattern || nc_pattern) && !base_has_stress_override) {
                std::string magic_e_ing = base + "e";
                // For CVC-pattern bases, magic-e is valid only when the primary stress falls
                // on the last vowel of the base stem. If stress is on an earlier syllable
                // (e.g. "market"→m'A@kIt: stress on 'A@', not on final 'I'), magic-e would
                // incorrectly lengthen the unstressed vowel ("markete"→m'A@ki:t is wrong).
                // Strategy: get the base phoneme; if its primary stress is NOT on the last
                // vowel group, use the base phoneme directly (skip magic-e via rules).
                // Magic-e via dict-lookup is still attempted inside stemPh(base) itself.
                // nc_pattern (e.g., "danc"→"dance") always uses magic-e regardless.
                bool use_magic_e = true;  // default: use magic-e first
                std::string base_only_ph;
                if (cvc_pattern && !nc_pattern) {
                    base_only_ph = stemPh(base);
                    if (!base_only_ph.empty()) {
                        // Check if primary stress is on the last vowel group.
                        // Scan backward to find the last vowel group start; check if '\'
                        // immediately precedes it.
                        static const std::string VC = "aAeEiIoOuUV03@";
                        const std::string& bp = base_only_ph;
                        int last_v = -1;
                        for (int k = (int)bp.size() - 1; k >= 0; k--) {
                            if (VC.find(bp[k]) != std::string::npos) { last_v = k; break; }
                        }
                        if (last_v > 0) {
                            // Walk backward from last_v to find start of last vowel group
                            int vstart = last_v;
                            while (vstart > 0 &&
                                   (VC.find(bp[vstart-1]) != std::string::npos ||
                                    bp[vstart-1] == ':' || bp[vstart-1] == '#'))
                                vstart--;
                            // Magic-e if primary OR secondary stress is directly before this group.
                            // E.g. "finaliz"→f'aIna#l,Iz: ',' before 'I' → magic-e needed.
                            bool stressed_at_end = (vstart > 0 &&
                                                    (bp[vstart-1] == '\'' || bp[vstart-1] == ','));
                            // Also treat as stressed if there's NO primary stress at all
                            // (monosyllabic stem before processPhonemeString adds stress).
                            bool no_explicit_stress = (bp.find('\'') == std::string::npos);
                            // Also try magic-e when the last vowel is a FULL vowel (not weak).
                            // Weak vowels (I, I#, I2, @, @2, @5) at stem-end indicate
                            // the pronunciation is already stable (e.g. "market"→I,
                            // "basket"→I, "blanket"→I). Full vowels (a, E, A, U, 3, …)
                            // before the final consonant suggest a magic-e word (e.g.
                            // "breathtak"→a → "breathtake"→eI is the real word).
                            bool last_vowel_is_full = (last_v >= 0 &&
                                                       bp[last_v] != 'I' && bp[last_v] != '@');
                            use_magic_e = stressed_at_end || no_explicit_stress || last_vowel_is_full;
                        }
                        // else: last_v <= 0 means no clear vowel → keep use_magic_e = true
                    }
                }
                // For strend2 words (compounds like "become"): prefer rules over dict so that
                // the 'be' prefix rule fires (giving 'bI#' = ᵻ, not 'bI' = ɪ from the dict).
                if (use_magic_e) {
                    if (strend_words_.count(magic_e_ing)) {
                        std::string rules_ph = applyRules(magic_e_ing);
                        if (!rules_ph.empty()) sph = processPhonemeString(rules_ph);
                    }
                    if (sph.empty()) sph = stemPh(magic_e_ing);          // write→writing, danc→dance
                    if (sph.empty()) sph = base_only_ph.empty() ? stemPh(base) : base_only_ph;
                } else {
                    // No magic-e: the base phoneme is already correct (stress not on last vowel).
                    // However, since base is not in dict, the reference RULE_ENDING mechanism may apply
                    // SUFX_E to the full word (e.g. "handwriting" → RULE_ENDING strips 'ing' →
                    // "handwrit" → SUFX_E adds 'e' → "handwrite" → rules give 'aI').
                    // Instead of applying a partial magic-e heuristic here, leave sph empty so
                    // the code falls through to applyRules(norm), which correctly fires the
                    // RULE_ENDING and handles SUFX_E with suffix_removed=true.
                    // (For words NOT in this block — i.e., when base_has_stress_override=true
                    //  because base is in dict — the custom handler returns the dict phoneme;
                    //  those words never reach this else branch.)
                }
            } else {
                // For bases matching ADD_E_ADDITIONS patterns (e.g. "dispens" ends in "ns")
                // that are NOT in the dict, try magic-e first. Rationale: if base is a stem
                // like "dispens" (from "dispense"), phonemizing it in isolation fires the
                // word-final 's' SUFFIX rule giving 'z'. But "dispense" correctly gives 's'
                // (no SUFFIX fires). The ADD_E_ADDITIONS list identifies these cases.
                if (!base_has_stress_override && !base.empty()) {
                    static const char* ADD_E_STEM_PATS[] = {
                        "ns", "rs", nullptr
                    };
                    bool try_magic_e_first = false;
                    for (int ai = 0; ADD_E_STEM_PATS[ai]; ai++) {
                        size_t plen = strlen(ADD_E_STEM_PATS[ai]);
                        if (base.size() >= plen + 1 && // at least one char before the pattern
                            base.compare(base.size()-plen, plen, ADD_E_STEM_PATS[ai]) == 0) {
                            try_magic_e_first = true; break;
                        }
                    }
                    if (try_magic_e_first) {
                        sph = stemPh(base + "e");
                    }
                }
                if (sph.empty())
                    sph = stemPh(base);                                   // sing → singing, harden → hardening
                if (sph.empty() && !base.empty() && !isVowelLetter(base.back())) {
                    // Only try magic-e if the base already has a vowel (e.g. "hik"→"hike").
                    // Prevents "spr"→"spre" (spring, no vowel in "spr" → don't add magic-e).
                    bool base_has_vowel = false;
                    for (char bc : base) if (isVowelLetter(bc)) { base_has_vowel = true; break; }
                    if (base_has_vowel)
                        sph = stemPh(base + "e");
                }
            }
            if (sph.empty() && base.size() >= 2 && base.back() == base[base.size()-2])
                sph = stemPh(base.substr(0, base.size()-1));              // run → running
            if (sph.empty() && !base.empty() && base.back() == 'i')
                sph = stemPh(base.substr(0, base.size()-1) + "y");        // study → studying
            } // end non-bypass block
            if (!sph.empty()) {
                if (std::getenv("PHON_DEBUG"))
                    std::cerr << "[SUFFIX-ING] " << norm << " stem=" << base << " sph=" << sph << "\n";
                // If stem ends in '@L' (syllabic L from word-final 'l'), the syllabic context
                // goes away when '-ing' follows for most consonant clusters. Replace '@L' with
                // plain 'l' UNLESS the letter before 'l' is 't' (the reference: &t) ling → @LI2N,
                // e.g. "bottling") or the word ends in 'ngl' (the reference: ng) ling → @-lI2N,
                // e.g. "tingling").
                if (sph.size() >= 2 && sph.compare(sph.size()-2, 2, "@L") == 0 &&
                        base.size() >= 2 && base.back() == 'l' &&
                        base[base.size()-2] != 't' &&
                        !(base.size() >= 3 && base.compare(base.size()-3, 3, "ngl") == 0)) {
                    // Determine whether to keep the schwa based on the letter before 'l':
                    //   - vowel before 'l' (e.g. "travel"→el, "cancel"→el): the reference fires
                    //     the "&) eling → @lI2N" rule → keep schwa: @L → @l.
                    //   - consonant before 'l' (e.g. "struggle"→gl): no schwa-insertion rule
                    //     fires → drop schwa entirely: @L → l.
                    char penult = base[base.size()-2];
                    bool vowel_before_l = (penult=='a'||penult=='e'||penult=='i'||
                                           penult=='o'||penult=='u');
                    sph = sph.substr(0, sph.size()-2) + (vowel_before_l ? "@l" : "l");
                }
                // If the stem came from a $strend2 dict entry, use final-stress placement.
                bool stem_is_strend = strend_words_.count(base) > 0 ||
                                      strend_words_.count(base + "e") > 0;
                return processPhonemeString(sph + "%IN", stem_is_strend);
            }
        }

        // -ed suffix: pre-check for words ending in 'ced'/'ged' (force+d, source+d, cage+d...)
        // When a word ends in consonant+'e'+'d' where the consonant is 'c' or 'g',
        // stripping 'ed' loses the soft-consonant context (e.g. "forc" → hard-k instead of 's').
        // Fix: strip just 'd' to get "force"/"source" and phonemize with soft-c.
        // SUFFIX-CE-ED was previously used to handle words ending in 'ced'/'ged'.
        // It has been removed. These words are now handled by the SUFFIX-RULE in
        // applyRules (via `@C) ed (_S2dvei d#`) which uses SUFX_E to add 'e' back,
        // giving the correct soft-c/g pronunciation. This also correctly handles:
        // - "faced"/"raced": specific rules consume "face"/"ace" entirely, leaving 'd'
        //   to get the default rule → voiced 'd' (matches the reference).
        // - "forced"/"sourced": SUFFIX-RULE fires, SUFX_E restores 'e', soft-c 's' → 't'.
        // See below: SUFFIX-ED also skips "ced"/"ged" endings for the same reason.

        // -ed suffix
        if (norm.size() >= 4 && norm.compare(norm.size()-2, 2, "ed") == 0) {
            std::string base = norm.substr(0, norm.size()-2);
            // If base ends in 'e', it's likely not a simple past-tense -ed form.
            // Example: "indeed" → base="inde" ends in 'e' — this is a non-compound word
            // that should be handled by PREFIX rules (e.g., _) in (deP2 ,In → "in"+"deed").
            // Real -ed past tense stems end in consonants (walked→walk, started→start).
            if (!base.empty() && base.back() == 'e') goto skip_ed_suffix;
            // If base ends in 'u' (e.g., "issued"→base="issu"), the word is likely a
            // stem ending in 'ue' (issue→issued, glue→glued, cue→cued). In the reference, the
            // `@@e) d (_S1 d#` rule strips only 'd', getting stem "issue"/"glue" with the
            // 'e' intact. Skip SUFFIX-ED here so applyRules handles it correctly.
            if (!base.empty() && base.back() == 'u') goto skip_ed_suffix;
            // For "-ced" and "-ged" endings (soft consonant before 'e'+'d'), bypass the
            // custom handler and use the rule-based SUFFIX-RULE in applyRules.
            // The SUFFIX-RULE `@C) ed (_S2dvei d#` uses SUFX_E to restore 'e', giving the
            // correct soft-c/g pronunciation. the reference handles these directly via rules:
            // - "faced"/"raced": a specific rule consumes the whole "face"/"ace" prefix,
            //   leaving bare 'd' to get the default rule → voiced 'd'.
            // - "forced"/"sourced"/"laced"/etc.: SUFFIX-RULE fires, SUFX_E restores 'e',
            //   soft-c gives 's' → devoiced to 't'.
            // Exception: if base ('forc', 'sourc') is in dict (then dict has the right phoneme).
            if (norm.size() >= 4) {
                char penult = norm[norm.size()-3]; // char before 'e' in "-ced"/"-ged"
                bool is_soft_c = (penult == 'c' || penult == 'g');
                bool is_ng_ged = (penult == 'g' && norm.size() >= 6 && norm[norm.size()-4] == 'n');
                if (is_soft_c && !is_ng_ged &&
                    dict_.find(base) == dict_.end() && verb_dict_.find(base) == verb_dict_.end())
                    goto skip_ed_suffix;
            }
            // For "-nged" endings where the base is not in the dictionary, bypass the
            // custom handler and use the rule-based suffix mechanism (applyRules).
            // the reference handles these via rules in .group ng (e.g. "o) nged  Nd" for "longed")
            // and .group an (e.g. "r) anged (S1  d" for "arranged"/"ranged").
            // The custom handler incorrectly strips "ed" to get "arrang"/"rang" instead
            // of using the magic-e stem "arrange"/"range".
            if (norm.size() >= 5 && norm.compare(norm.size()-4, 4, "nged") == 0 &&
                dict_.find(base) == dict_.end() && verb_dict_.find(base) == verb_dict_.end())
                goto skip_ed_suffix;
            // For "-eted" endings where base is not in dict: the en_rules `&) eted (_S2 I#d` rule
            // correctly handles these by stripping 2 chars → base word, with NO magic-e.
            // Without this exclusion, CVC pattern would (wrongly) try stemPh(base+"e") giving
            // the magic-e form (e.g. "target"→"targete"→iː instead of correct ɪ).
            if (norm.size() >= 5 && norm.compare(norm.size()-4, 4, "eted") == 0 &&
                dict_.find(base) == dict_.end() && verb_dict_.find(base) == verb_dict_.end())
                goto skip_ed_suffix;
            // For "-mented" endings preceded by a consonant: the en_rules
            // `C) mented m'EntId` rule fires and gives 'Id' (full ɪ, not reduced ᵻ).
            // This covers "fragmented", "segmented", "pigmented", etc. where 'g' or
            // similar consonant precedes 'm'. "cemented" (vowel 'e' before 'm') is
            // not affected. Words where the stem has a $N stress override in the dict
            // (e.g. "tormented" → "torment $1") must keep using SUFFIX-ED so the
            // stress override is applied correctly.
            // Skip SUFFIX-ED for these so applyRules handles the full word.
            if (norm.size() >= 7 && norm.compare(norm.size()-6, 6, "mented") == 0) {
                char before_m = norm[norm.size()-7]; // char before 'm'
                std::string mented_stem = norm.substr(0, norm.size()-2); // e.g. "torment"
                bool stem_has_stress = (stress_pos_.count(mented_stem) > 0);
                if (!isVowelLetter(before_m) && !stem_has_stress)
                    goto skip_ed_suffix;
            }
            // If the base ends in 3+ consecutive consonants (e.g. "hundr" from "hundred"),
            // it's very unlikely to be a verb stem. Skip to let applyRules handle directly.
            // This prevents false -ed treatment of words like "hundred", "kindred", etc.
            {
                int trail_cons = 0;
                for (int bi = (int)base.size()-1; bi >= 0 && !isVowelLetter(base[bi]); bi--)
                    trail_cons++;
                if (trail_cons >= 3) goto skip_ed_suffix;
            }
            {
            std::string sph;
            // Same CVC/nc magic-e logic as for -ing, with dict-priority fix.
            {
                bool cvc_pattern = base.size() >= 2 &&
                    !isVowelLetter(base.back()) &&
                    isVowelLetter(base[base.size()-2]);
                // For CVC bases ending in vowel+'n': same schwa-n heuristic as for -ing.
                if (cvc_pattern && base.size() >= 2 && base.back() == 'n') {
                    char prev_vowel = base[base.size()-2];
                    if (prev_vowel == 'e' || prev_vowel == 'a' || prev_vowel == 'i' ||
                        prev_vowel == 'o' || prev_vowel == 'u') {
                        std::string base_ph = stemPh(base);
                        if (!base_ph.empty() && base_ph.size() >= 2) {
                            char last2 = base_ph.back();
                            char last1 = base_ph[base_ph.size()-2];
                            bool ends_in_schwa_n = (last1 == '@' && (last2 == 'n' || last2 == 'N'));
                            bool ends_in_syllabic_n = (last1 == 'n' && last2 == '-');
                            if (ends_in_schwa_n || ends_in_syllabic_n) cvc_pattern = false;
                        }
                    }
                }
                bool nc_pattern = base.size() >= 2 && base.back() == 'c' &&
                    base[base.size()-2] == 'n';
                // Bases ending in 'el': the 'e' is always schwa, not a magic-e vowel.
                // E.g., "travel"→"traveled" should use stemPh("travel"), not stemPh("travele").
                if (cvc_pattern && base.size() >= 2 && base.back() == 'l' &&
                    base[base.size()-2] == 'e')
                    cvc_pattern = false;
                // Bases ending in vowel+'w' (e.g. "renew", "stew"): 'w' is a digraph semi-vowel,
                // not a magic-e consonant. Adding 'e' creates wrong phonemization (e.g. "renewe"
                // triggers ren)ew(A rule giving 'ju:' instead of correct 'u:').
                if (cvc_pattern && base.back() == 'w')
                    cvc_pattern = false;
                // Bases ending in 'e'+'r' are rhotic schwa (ɚ) stems, not magic-e stems.
                // E.g. "ponder" → "pondered" should use "ponder" directly, not "pondere".
                if (cvc_pattern && base.size() >= 2 && base.back() == 'r' &&
                    std::tolower((unsigned char)base[base.size()-2]) == 'e')
                    cvc_pattern = false;
                // If the base has an explicit $N stress override, skip magic-e.
                // $only words are excluded: they can't be used as stems for non-s suffixes.
                bool base_has_stress_override2 = (stress_pos_.find(base) != stress_pos_.end() ||
                                                  (dict_.find(base) != dict_.end() && !onlys_words_.count(base) && !only_words_.count(base)) ||
                                                  verb_dict_.find(base) != verb_dict_.end());
                // SUFX_I: words like "identified" → base "identifi" ends in 'i' from 'y'→'i' change.
                // the reference `ied (_S2i [d]` rule restores 'y' before re-phonemizing.
                // When base ends in 'i' (likely a 'y'-ending verb stem), try base[:-1]+'y' first.
                if (!sph.empty()) {} // already set
                else if (!base.empty() && base.back() == 'i' && base.size() >= 2) {
                    std::string y_stem = base.substr(0, base.size()-1) + "y";
                    sph = stemPh(y_stem);  // "identifi"→"identify", "appli"→"apply", etc.
                }
                // Guard: if the CVC magic-e form (base+'e') starts with a PREFIX rule at
                // position 0, the word is a compound (e.g. "infrare" = "infra"+"re") rather
                // than a verb stem with silent-e. Skip magic-e to avoid retranslating the
                // PREFIX suffix as -ed morphology. Let the full word fall through to applyRules.
                auto hasPrefixAtStart = [&](const std::string& w) -> bool {
                    if (w.empty()) return false;
                    char c0 = std::tolower((unsigned char)w[0]);
                    auto checkGroup = [&](const std::string& key, int glen) -> bool {
                        auto it = ruleset_.rule_groups.find(key);
                        if (it == ruleset_.rule_groups.end()) return false;
                        for (const auto& rule : it->second) {
                            if (!rule.is_prefix) continue;
                            std::string ph; int adv, dfs, dfc;
                            int sc = matchRule(rule, w, 0, ph, adv, dfs, dfc, glen, "", 0, nullptr);
                            if (sc >= 0 && adv > 0 && adv < (int)w.size()) return true;
                        }
                        return false;
                    };
                    if (w.size() >= 2) {
                        std::string k2(1, c0); k2 += std::tolower((unsigned char)w[1]);
                        if (checkGroup(k2, 2)) return true;
                    }
                    return checkGroup(std::string(1, c0), 1);
                };
                if ((cvc_pattern || nc_pattern) && !base_has_stress_override2) {
                    std::string magic_e = base + "e";
                    if (hasPrefixAtStart(magic_e))
                        goto skip_ed_suffix;                              // compound word, not -ed verb
                    // Same use_magic_e heuristic as the -ing handler: for CVC bases not in dict,
                    // check if stress falls on the last vowel group. If not, leave sph empty so
                    // the code falls through to applyRules(norm), which fires RULE_ENDING+SUFX_E
                    // with suffix_removed=true. This correctly handles words like "spirited"
                    // (stem "spirit" has stress on 1st syllable, last vowel weak 'I' → no magic-e
                    // → applyRules fires suffix rule → re-translates "spirite" with suffix_removed
                    // → 'N'-guarded aIt rule fails → correct 'ɪ' instead of wrong 'aɪ').
                    bool use_magic_e_ed = true;
                    std::string base_only_ph_ed;
                    if (cvc_pattern && !nc_pattern) {
                        base_only_ph_ed = stemPh(base);
                        if (!base_only_ph_ed.empty()) {
                            static const std::string VC2 = "aAeEiIoOuUV03@";
                            const std::string& bp = base_only_ph_ed;
                            int last_v = -1;
                            for (int k = (int)bp.size() - 1; k >= 0; k--) {
                                if (VC2.find(bp[k]) != std::string::npos) { last_v = k; break; }
                            }
                            if (last_v > 0) {
                                int vstart = last_v;
                                while (vstart > 0 &&
                                       (VC2.find(bp[vstart-1]) != std::string::npos ||
                                        bp[vstart-1] == ':' || bp[vstart-1] == '#'))
                                    vstart--;
                                bool stressed_at_end = (vstart > 0 &&
                                                        (bp[vstart-1] == '\'' || bp[vstart-1] == ','));
                                bool no_explicit_stress = (bp.find('\'') == std::string::npos);
                                // Exclude '3' (ɚ rhotic schwa) from "full vowels": stems ending in
                                // rhotic schwa (e.g. "author"→'O:T%3, "ponder"→'p0nd3) are
                                // already complete rhotic vowels; adding magic-e (authore, pondere)
                                // would incorrectly change the 'or'/'er' sound.
                                bool last_vowel_is_full = (bp[last_v] != 'I' && bp[last_v] != '@' && bp[last_v] != '3');
                                use_magic_e_ed = stressed_at_end || no_explicit_stress || last_vowel_is_full;
                            }
                        }
                    }
                    if (use_magic_e_ed) {
                        if (sph.empty()) sph = stemPh(magic_e);          // smile→smiled, danc→danced
                        if (sph.empty()) sph = stemPh(base);
                    }
                    // else: leave sph empty → fall through to applyRules(norm)
                } else {
                    // For doubled-consonant stems from CVC inflection, the choice of whether to try
                    // the doubled or undoubled stem first depends on syllable count:
                    //
                    // Multi-syllable words (e.g. "rappelled", "compelled", "expelled"):
                    //   The en_rules `@@) lled (_S3v d` rule fires for the full word when there are
                    //   2+ syllable groups before 'lled'. To match this, we need the undoubled stem
                    //   ("rappel") so that the 'el' word-end rule gives correct stress.
                    //   Heuristic: count vowel groups in norm[0..len-4] (= before the "-lled" part).
                    //
                    // Single-syllable words (e.g. "pulled", "stalled", "called"):
                    //   The `@@) lled` rule does NOT fire (only 1 vowel group). The doubled stem
                    //   itself ("pull", "stall") contains the context needed for correct rules
                    //   (e.g. `p) ull Ul`). Try base first.
                    static const std::string CVC_DOUBLE_CONS = "lptmnrgdb";
                    bool base_has_double = base.size() >= 2 && base.back() == base[base.size()-2] &&
                                           CVC_DOUBLE_CONS.find(base.back()) != std::string::npos;
                    bool prefer_undoubled = false;
                    if (base_has_double) {
                        // Count vowel groups in the word before the doubled suffix
                        // norm = base + "ed", so norm before "lled" = norm[0..norm.size()-4]
                        // (for base ending in "ll"; for other doubles use norm[0..norm.size()-3])
                        int prefix_len = (int)norm.size() - 2 - (int)base.size() + (int)base.size() - 2;
                        // Simpler: count vowel groups in base[0..base.size()-2] (before the double)
                        std::string prefix = base.substr(0, base.size() - 2);
                        int vowel_groups = 0;
                        bool in_v = false;
                        for (char c : prefix) {
                            if (isVowelLetter(c)) { if (!in_v) { vowel_groups++; in_v = true; } }
                            else in_v = false;
                        }
                        prefer_undoubled = (vowel_groups >= 2);
                    }
                    if (sph.empty() && base_has_double && prefer_undoubled)
                        sph = stemPh(base.substr(0, base.size()-1));      // "rappel", "compel" (multi-syllable)
                    // For bases ending in "rs"/"ns" not in dict, try magic-e first to avoid
                    // word-final 's' SUFFIX rule giving wrong 'z'. E.g. "rehears" → try
                    // "rehearse" first so no SUFFIX fires. (Same logic as SUFFIX-ING handler.)
                    if (sph.empty() && !base_has_stress_override2 && !base.empty()) {
                        static const char* ADD_E_PATS_ED[] = { "ns", "rs", nullptr };
                        bool try_e_first = false;
                        for (int ai = 0; ADD_E_PATS_ED[ai]; ai++) {
                            size_t plen = strlen(ADD_E_PATS_ED[ai]);
                            if (base.size() >= plen + 1 &&
                                base.compare(base.size()-plen, plen, ADD_E_PATS_ED[ai]) == 0) {
                                try_e_first = true; break;
                            }
                        }
                        if (try_e_first) sph = stemPh(base + "e");
                    }
                    if (sph.empty()) sph = stemPh(base);                  // pull, stall, sign → signed
                    if (sph.empty() && base_has_double && !prefer_undoubled)
                        sph = stemPh(base.substr(0, base.size()-1));      // fallback for any missed cases
                    if (sph.empty() && !base.empty() && !isVowelLetter(base.back())) {
                        bool base_has_vowel = false;
                        for (char bc : base) if (isVowelLetter(bc)) { base_has_vowel = true; break; }
                        if (base_has_vowel)
                            sph = stemPh(base + "e");
                    }
                }
            }
            if (sph.empty() && base.size() >= 2 && base.back() == base[base.size()-2]) {
                sph = stemPh(base.substr(0, base.size()-1));              // stop → stopped (fallback)
                if (sph.empty())
                    sph = stemPh(base);
            }
            if (!sph.empty()) {
                // Voicing of -ed suffix: after t/d → 'I2#d' (ᵻd), after unvoiced → 't', else → 'd'
                static const std::string UNVOICED = "ptkfTSshx";
                char last = sph.back();
                std::string ed_ph;
                if (last == 't' || last == 'd') ed_ph = "I#d";
                else if (UNVOICED.find(last) != std::string::npos) ed_ph = "t";
                else ed_ph = "d";
                // Before devoicing, check if the 'e' in the full word is consumed by a
                // multi-char rule (not del_fwd). If so, the reference processes the 'd' as the
                // default 'd' rule (voiced), not via RULE_ENDING. This happens when:
                // - A literal rule matches the entire word minus the final 'd' (e.g. "type"
                //   rule fires for "typed" at pos=0, advance=4, leaving 'd' at pos=4).
                // - A 2-char rule embeds 'd' in its phoneme output (e.g. "c) oped (_  oUpd"
                //   fires for "coped", consuming "oped" including 'd').
                // In these cases, full-word applyRules (no suffix strip) correctly gives 'd'.
                // Detection: run applyRules on the full word; check if replaced_e is NOT set
                // at norm.size()-2 (the 'e' before the final 'd'). If not set, the 'e' was
                // consumed by a multi-char rule match (not del_fwd) → keep voiced 'd'.
                if (ed_ph == "t" && norm.size() >= 2) {
                    std::vector<bool> fw_replaced_e, fw_pos_visited;
                    std::string fw_ph = applyRules(norm, false, -1, false, false,
                                                   &fw_replaced_e, &fw_pos_visited);
                    int e_pos = (int)norm.size() - 2; // position of 'e' in norm
                    // Check whether the 'e' position was visited at all in the scan loop.
                    // If NOT visited, a long-match rule consumed it (e.g. "type" rule
                    // consumed "type" at pos=0 with advance=4, skipping 'e' at pos=3).
                    // In that case, the final 'd' was phonemized by default rule → voiced.
                    bool e_was_visited = (e_pos >= 0 && e_pos < (int)fw_pos_visited.size()
                                          && fw_pos_visited[e_pos]);
                    // Find last meaningful (non-\x01 boundary) char in fw_ph.
                    char fw_last = 0;
                    for (int ri = (int)fw_ph.size()-1; ri >= 0; ri--) {
                        if (fw_ph[ri] != '\x01') { fw_last = fw_ph[ri]; break; }
                    }
                    if (std::getenv("PHON_DEBUG"))
                        std::cerr << "[FW-CHECK] norm=" << norm << " e_pos=" << e_pos
                                  << " e_was_visited=" << e_was_visited
                                  << " fw_ph='" << fw_ph << "'"
                                  << " fw_last=" << fw_last << "\n";
                    if (!e_was_visited && fw_last == 'd') {
                        // The 'e' was consumed by a long-match rule (never visited as its
                        // own scan position). the reference processes 'd' by default rule → voiced.
                        if (std::getenv("PHON_DEBUG"))
                            std::cerr << "[SUFFIX-ED] " << norm << " stem=" << base
                                      << " sph=" << sph << " +d [full-word rule]\n";
                        return processPhonemeString(fw_ph);
                    }
                }
                if (std::getenv("PHON_DEBUG"))
                    std::cerr << "[SUFFIX-ED] " << norm << " stem=" << base << " sph=" << sph << " +" << ed_ph << "\n";
                return processPhonemeString(sph + ed_ph);
            }
            } // end inner block for -ed suffix body
            skip_ed_suffix:;
        }
    }

    // -ies suffix: plural/3rd-person of words ending in -y (butterflies, countries, studies)
    // Strip "ies", restore "y", phonemize the stem, add voiced 'z'.
    if (norm.size() >= 4 && norm.compare(norm.size()-3, 3, "ies") == 0) {
        std::string base = norm.substr(0, norm.size()-3) + "y";
        // stemPh uses a local lambda defined in the -ing/-ed block above; duplicate inline here.
        auto doStemPh = [&](const std::string& stem) -> std::string {
            if (stem.size() < 2) return "";
            bool hv = false;
            for (char c : stem) if (isVowelLetter(c)) { hv = true; break; }
            if (!hv) return "";
            std::string ph2;
            auto jt = dict_.find(stem);
            if (jt != dict_.end()) ph2 = jt->second;
            else ph2 = processPhonemeString(applyRules(stem));
            static const std::string VC = "aAeEIiOUVu03@o";
            for (char c : ph2) if (VC.find(c) != std::string::npos) return ph2;
            return "";
        };
        std::string sph = doStemPh(base);
        if (!sph.empty()) {
            // Check whether a specific en_rules rule fires for the full word that wouldn't
            // fire for the stem. E.g. `sp) e (cies i:` gives 'e'→'i:' in "species" but
            // NOT in "specy". We detect this by:
            // 1. If the stem's first vowel is a DIPHTHONG (e.g. 'eI', 'aI', 'aU', 'oU'),
            //    the stem is capturing morphological vowel lengthening — trust it.
            // 2. If the stem's first vowel is a SHORT MONOPHTHONG, compare with direct rules.
            //    A different first vowel in direct rules means a specific rule fired.
            static const std::string VOWELS_IES = "aAeEIiOUVu03@o";
            auto firstVowelPos = [&](const std::string& ph) -> int {
                for (int k = 0; k < (int)ph.size(); k++)
                    if (VOWELS_IES.find(ph[k]) != std::string::npos) return k;
                return -1;
            };
            int sv_pos = firstVowelPos(sph);
            bool stem_has_diphthong = (sv_pos >= 0 &&
                sv_pos + 1 < (int)sph.size() &&
                (sph[sv_pos+1] == 'I' || sph[sv_pos+1] == 'U'));  // eI, aI, aU, OI, oU
            if (!stem_has_diphthong) {
                // Stem gives short monophthong → check if direct rules give something different
                char sv = (sv_pos >= 0) ? sph[sv_pos] : 0;
                std::string full_raw = applyRules(norm, false, 0);
                int dv_pos = firstVowelPos(full_raw);
                char dv = (dv_pos >= 0) ? full_raw[dv_pos] : 0;
                // Only use direct rules when the first vowel is in a STRESSED context.
                // If the direct result's first vowel is preceded by '%' (unstressed marker),
                // the direct rules are confused by context (e.g. 'co' in "companies" fires
                // a reduced-vowel rule with lower score than in "company"). In that case,
                // trust the stem-based approach.
                bool direct_fv_unstressed = (dv_pos > 0 && full_raw[dv_pos-1] == '%');
                // Don't use direct rules when the direct result is a magic-e artifact from '-ies'.
                // E.g. "studies" (stem "study"→V/ʌ): direct rules see 'u'+'d'+'ie' as magic-e
                // and give 'u:'. This is wrong — the real vowel is ʌ from the stem.
                // Detectable by: stem first vowel is 'V' (ʌ, the short-u STRUT vowel) and the
                // direct first vowel starts with 'u' (long-u magic-e conversion V→u:).
                bool magic_e_strut = (sv == 'V' && (dv == 'u' || dv == 'U'));
                if (dv != 0 && sv != dv && !direct_fv_unstressed && !magic_e_strut) {
                    // Direct rules produce a different first vowel → specific rule fired
                    if (std::getenv("PHON_DEBUG"))
                        std::cerr << "[SUFFIX-IES-DIRECT] " << norm << " direct=" << full_raw << "\n";
                    return processPhonemeString(full_raw);
                }
            }
            if (std::getenv("PHON_DEBUG"))
                std::cerr << "[SUFFIX-IES] " << norm << " stem=" << base << " sph=" << sph << "\n";
            return processPhonemeString(sph + "z");
        }
    }

    // General -s suffix stripping against dictionary.
    // If the word ends in 's' and is not in dict_, try stripping -s or -es to find a stem.
    // This handles words like "systems", "organisms", "animals" where the base word IS in dict_.
    // Must run BEFORE magic-e and sibilant handlers so dict-based stems take priority.
    if (norm.size() >= 3 && norm.back() == 's' &&
        !(norm.size() >= 2 && norm[norm.size()-2] == 's')) {  // skip double-s roots (e.g. "business" ends in 'ss')
        static const std::string UNVOICED_S = "ptkfTSCxXhs";
        auto doStemPhS = [&](const std::string& stem) -> std::string {
            if (stem.size() < 2) return "";
            bool hv = false;
            for (char c : stem) if (isVowelLetter(c)) { hv = true; break; }
            if (!hv) return "";
            static const std::string VC3 = "aAeEIiOUVu03@o";
            std::string ph2;
            // For -s suffix, $onlys bare-word pronunciation takes priority (valid for bare/+s).
            auto obit_s = onlys_bare_dict_.find(stem);
            if (obit_s != onlys_bare_dict_.end()) {
                ph2 = processPhonemeString(obit_s->second);
            } else if (only_words_.count(stem)) {
                // Stem has $only entry: that entry is for bare word only (not suffix stems).
                // Prefer verb_dict_ if available (e.g. "close kloUs $only" + "close kloUz $verb"
                // → "closes" should use kloUz). Otherwise fall through to rules (ph2 stays empty).
                auto vt = verb_dict_.find(stem);
                if (vt != verb_dict_.end())
                    ph2 = processPhonemeString(vt->second);
            } else {
                auto jt = dict_.find(stem);
                if (jt != dict_.end()) {
                    ph2 = processPhonemeString(jt->second);
                } else {
                    // Also check stress_pos_: stems with flag-only dict entries (e.g. "insect $1")
                    // have no explicit phoneme in dict_ but DO have a stress position override.
                    // Phonemize via rules then apply stress override (e.g. "insects" → stem "insect"
                    // with $1 → primary on 1st syllable → ˈɪnsɛkts not ˌɪnsˈɛkts).
                    auto sp = stress_pos_.find(stem);
                    if (sp != stress_pos_.end()) {
                        std::string raw = applyRules(stem, true, 0);
                        if (!raw.empty())
                            ph2 = processPhonemeString(applyStressPosition(raw, sp->second));
                    }
                }
            }
            if (ph2.empty()) return ""; // only use dict/stress_pos-based stems
            for (char c : ph2) if (VC3.find(c) != std::string::npos) return ph2;
            return "";
        };
        // Try stem = word without trailing 's'
        std::string stem_s = norm.substr(0, norm.size()-1);
        // Skip -s stripping if stem ends in adjective/adverb suffixes that are not plurals.
        // Words like "various"→"variou", "glorious"→"gloriou", "continuous"→"continuou"
        // are not plurals; their "stem without s" is not a real word.
        // Common patterns: -ious, -eous, -uous, -ous (but allow rule-based for short stems).
        // Simple heuristic: if stem ends in 'u' preceded by a vowel-group suffix, skip.
        bool skip_s_strip = false;
        if (stem_s.size() >= 3 && stem_s.back() == 'u') {
            // Ends in 'u' → likely from -ious/-eous/-uous/-ous adjectives
            skip_s_strip = true;
        }
        std::string sph_s = skip_s_strip ? "" : doStemPhS(stem_s);
        if (!sph_s.empty()) {
            // Voicing: sibilant ending → ᵻz; after unvoiced → 's'; else → 'z'
            static const std::string SIBILANTS_PH = "SZsz"; // ʃ ʒ s z
            bool last_sibilant = SIBILANTS_PH.find(sph_s.back()) != std::string::npos;
            std::string s_ph = last_sibilant ? "I#z" :
                               (UNVOICED_S.find(sph_s.back()) != std::string::npos) ? "s" : "z";
            if (std::getenv("PHON_DEBUG"))
                std::cerr << "[SUFFIX-DICT-S] " << norm << " stem=" << stem_s << " sph=" << sph_s << " +" << s_ph << "\n";
            return processPhonemeString(sph_s + s_ph);
        }
        // Try stem = word without trailing 'es' (sibilant+es → ᵻz)
        // Only applies when stem ends in a sibilant phoneme (S/Z/s/z/C/dZ),
        // since that is the only context where '-es' gives /ɪz/.
        if (norm.size() >= 4 && norm[norm.size()-2] == 'e') {
            std::string stem_es = norm.substr(0, norm.size()-2);
            std::string sph_es = doStemPhS(stem_es);
            if (!sph_es.empty()) {
                // Guard: stem must end in a sibilant phoneme code for -es → ᵻz to apply.
                static const std::string SIBILANTS_ES = "SZszC"; // ʃ ʒ s z tʃ(C) dʒ(Z)
                bool stem_sibilant = (!sph_es.empty() &&
                    (SIBILANTS_ES.find(sph_es.back()) != std::string::npos));
                if (stem_sibilant) {
                    if (std::getenv("PHON_DEBUG"))
                        std::cerr << "[SUFFIX-DICT-ES] " << norm << " stem=" << stem_es << " sph=" << sph_es << " +I#z\n";
                    return processPhonemeString(sph_es + "I#z");
                }
            }
        }
    }

    // -[Ce]s suffix: magic-e stem + s (gives→give+z, moves→move+z, makes→make+s)
    // Only when: word ends in consonant+e+s (magic-e pattern) and the consonant before 'e'
    // is NOT a sibilant (s,z,x,c) — those cases are sibilant+es → /ɪz/ handled by rules.
    // Also excludes digraph sibilants 'ch' and 'sh' (teaches, washes) — handled separately below.
    if (norm.size() >= 4 && norm.back() == 's' &&
        !(norm.size() >= 2 && norm[norm.size()-2] == 's')) {  // skip double-s roots
        std::string base = norm.substr(0, norm.size()-1);     // strip trailing 's'
        if (base.size() >= 2 && base.back() == 'e') {
            char c_before_e = base[base.size()-2];
            // Only non-sibilant consonants before 'e' (sibilant+es handled by rules)
            bool is_consonant = !isVowelLetter(c_before_e);
            bool is_sibilant = (c_before_e == 's' || c_before_e == 'z' ||
                                c_before_e == 'x' || c_before_e == 'c');
            // Digraph sibilants: 'ch' (teaches→teach) and 'sh' (washes→wash)
            bool is_digraph_sibilant = (c_before_e == 'h' && base.size() >= 3 &&
                (base[base.size()-3] == 'c' || base[base.size()-3] == 's'));
            if (is_consonant && !is_sibilant && !is_digraph_sibilant) {
                auto doStemPh2 = [&](const std::string& stem) -> std::string {
                    if (stem.size() < 2) return "";
                    bool hv = false;
                    for (char c : stem) if (isVowelLetter(c)) { hv = true; break; }
                    if (!hv) return "";
                    std::string ph2;
                    auto jt = dict_.find(stem);
                    if (jt != dict_.end()) {
                        ph2 = jt->second;
                        auto sp = stress_pos_.find(stem);
                        if (sp != stress_pos_.end())
                            ph2 = applyStressPosition(ph2, sp->second);
                    } else {
                        // Determine whether to use first-pass (suffix_phoneme_only=true) mode.
                        // When stem ends in VOWEL + B_consonant + 'e' (magic-e pattern where
                        // a DEL_FWD rule fires, e.g. 'o (Be#'), the outer 's' RULE_ENDING is
                        // in .group E (REPLACED_E path: '&) Es (_S2e z') and has no SUFX_M.
                        // the reference calls TranslateRules(stem, NULL) = first-pass, accumulating
                        // inner RULE_ENDING phonemes without sub-stem extraction.
                        // E.g. "ribosome" (o+m+e): first-pass → rIb0soUm (correct).
                        // Without this, '-some' strips "ribo" → re-phonemizes in isolation
                        // with word-final 'i' rule XC)i(Co_ → i: (wrong).
                        // When stem does NOT end in this pattern, outer 's' rule has SUFX_M
                        // and uses normal mode (suffix_phoneme_only=false).
                        // E.g. "vegetable" (b+l+e): normal → '-able' strips → "veget" → I2t.
                        bool use_spo = false;
                        static const std::string GROUP_B_CHARS = "bcdfgjklmnpqstvxz";
                        static const std::string DELFWD_VOWELS = "aioy";
                        if (stem.size() >= 3 && stem.back() == 'e') {
                            char c_cons = (char)std::tolower((unsigned char)stem[stem.size()-2]);
                            char c_prev = (char)std::tolower((unsigned char)stem[stem.size()-3]);
                            bool c_is_B = GROUP_B_CHARS.find(c_cons) != std::string::npos;
                            bool v_is_delfwd = DELFWD_VOWELS.find(c_prev) != std::string::npos;
                            if (c_is_B && v_is_delfwd)
                                use_spo = true;
                        }
                        std::string raw = applyRules(stem, /*allow_suffix_strip=*/true, -1,
                                                     /*suffix_phoneme_only=*/use_spo);
                        auto sp = stress_pos_.find(stem);
                        if (sp != stress_pos_.end())
                            raw = applyStressPosition(raw, sp->second);
                        ph2 = processPhonemeString(raw);
                    }
                    static const std::string VC = "aAeEIiOUVu03@o";
                    for (char c : ph2) if (VC.find(c) != std::string::npos) return ph2;
                    return "";
                };
                std::string sph = doStemPh2(base);
                if (!sph.empty()) {
                    static const std::string UNVOICED = "ptkfTSCxXhs";
                    static const std::string SIBILANTS_PH = "SZsz"; // ʃ ʒ s z (also covers dʒ via Z)
                    bool last_sib = SIBILANTS_PH.find(sph.back()) != std::string::npos;
                    std::string s_ph = last_sib ? "I#z" :
                                       (UNVOICED.find(sph.back()) != std::string::npos) ? "s" : "z";
                    if (std::getenv("PHON_DEBUG"))
                        std::cerr << "[SUFFIX-S] " << norm << " stem=" << base << " sph=" << sph << " +" << s_ph << "\n";
                    return processPhonemeString(sph + s_ph);
                }
            }
        }
    }

    // -[ch/sh]es suffix: digraph sibilant + es → /ɪz/ (teaches, washes, reaches, touches)
    // 'ch' and 'sh' before 'e' are excluded from magic-e stripping above; handle here.
    if (norm.size() >= 5 && norm.back() == 's' && norm[norm.size()-2] == 'e' &&
        norm[norm.size()-3] == 'h' &&
        (norm[norm.size()-4] == 'c' || norm[norm.size()-4] == 's')) {
        std::string stem = norm.substr(0, norm.size()-2); // strip 'es' → "teach"/"wash"
        bool has_vowel = false;
        for (char c : stem) if (isVowelLetter(c)) { has_vowel = true; break; }
        if (has_vowel && stem.size() >= 2) {
            std::string sph;
            auto jt = dict_.find(stem);
            if (jt != dict_.end()) sph = jt->second;
            else sph = processPhonemeString(applyRules(stem));
            static const std::string VC_d = "aAeEIiOUVu03@o";
            bool has_ph_vowel = false;
            for (char c : sph) if (VC_d.find(c) != std::string::npos) { has_ph_vowel = true; break; }
            if (has_ph_vowel) {
                return processPhonemeString(sph + "I#z");
            }
        }
    }

    // -[x]es suffix: sibilant x+es plural (taxes→tax+ᵻz, boxes→box+ᵻz, fixes→fix+ᵻz)
    // The letter 'x' represents /ks/, a clear sibilant cluster; '-xes' always = /ksᵻz/
    if (norm.size() >= 4 &&
        norm.size() >= 3 && norm.compare(norm.size()-3, 3, "xes") == 0) {
        std::string stem = norm.substr(0, norm.size()-2); // strip 'es', keep 'x'
        bool has_vowel = false;
        for (char c : stem) if (isVowelLetter(c)) { has_vowel = true; break; }
        if (has_vowel && stem.size() >= 2) {
            std::string sph;
            auto jt = dict_.find(stem);
            if (jt != dict_.end()) sph = jt->second;
            else sph = processPhonemeString(applyRules(stem));
            static const std::string VC2 = "aAeEIiOUVu03@o";
            bool has_ph_vowel = false;
            for (char c : sph) if (VC2.find(c) != std::string::npos) { has_ph_vowel = true; break; }
            if (has_ph_vowel) {
                return processPhonemeString(sph + "I#z");
            }
        }
    }

    // -arily suffix: words ending in -arily (primarily, ordinarily, necessarily, temporarily)
    // the reference applies the rule: arily(_) → 'e@rI#l%i (stress on the -ari- syllable).
    // Strategy: strip -arily, phonemize the stem, demote stem's primary stress to secondary,
    // then append 'e@rI#l%i as the suffix phoneme codes.
    if (norm.size() >= 8 && norm.compare(norm.size()-5, 5, "arily") == 0) {
        // norm ends in "arily"; stem is everything before "arily"
        std::string stem_arily = norm.substr(0, norm.size()-5); // strip "arily"
        bool stem_has_vowel_a = false;
        for (char c : stem_arily) if (isVowelLetter(c)) { stem_has_vowel_a = true; break; }
        if (stem_has_vowel_a && stem_arily.size() >= 2) {
            // Phonemize "stem + ari" to get context-correct phonemes for the stem.
            // The trailing "-ari" triggers proper open-syllable vowel rules in the stem
            // (e.g. "prim" in open syllable "prima..." → aI, not I).
            std::string stem_with_ari = stem_arily + "ari";
            std::string sph_arily;
            auto jt_ar = dict_.find(stem_with_ari);
            if (jt_ar != dict_.end()) {
                sph_arily = jt_ar->second;
            } else {
                sph_arily = applyRules(stem_with_ari);
            }
            if (!sph_arily.empty()) {
                // Strip the trailing -ari phoneme suffix.
                // the reference rule: ar(_) at word end → 3 (e.g. "primari" → pr'aIm3ri, trailing is "3ri")
                //              "ari" in stressed context → 'A@ri (trailing is "'A@ri")
                // We strip one of these suffixes and replace with "'e@rI#l%i".
                std::string sph_stem;
                bool stripped = false;
                // Try stripping "'A@ri" first (stressed ari, 5 chars: ' A @ r i)
                if (sph_arily.size() >= 5 &&
                    sph_arily.compare(sph_arily.size()-5, 5, "'A@ri") == 0) {
                    // Stressed 'A@ri at end: just remove it; secondary stress already in stem
                    sph_stem = sph_arily.substr(0, sph_arily.size()-5);
                    stripped = true;
                } else if (sph_arily.size() >= 3 &&
                           (sph_arily.compare(sph_arily.size()-3, 3, "3ri") == 0 ||
                            sph_arily.compare(sph_arily.size()-3, 3, "@ri") == 0)) {
                    // Schwa+r "3ri"/"@ri" at end: remove it and demote primary stress → secondary
                    sph_stem = sph_arily.substr(0, sph_arily.size()-3);
                    // Demote primary stress ' → secondary ,
                    bool demoted = false;
                    std::string tmp;
                    for (char ch : sph_stem) {
                        if (ch == '\'' && !demoted) { tmp += ','; demoted = true; }
                        else tmp += ch;
                    }
                    sph_stem = tmp;
                    stripped = true;
                }
                if (stripped && !sph_stem.empty()) {
                    // Append the -arily suffix phoneme: 'e@rI#l%i
                    std::string combined = sph_stem + "'e@rI#l%i";
                    if (std::getenv("PHON_DEBUG"))
                        std::cerr << "[SUFFIX-ARILY] " << norm << " stem_ari=" << stem_with_ari
                                  << " sph=" << sph_arily << " stem_ph=" << sph_stem
                                  << " combined=" << combined << "\n";
                    return processPhonemeString(combined);
                }
            }
        }
    }



    // Try compound prefix splitting: if the word starts with a known $strend2 prefix
    // (a function word that shifts stress to the suffix, e.g. "under", "over", "through"),
    // return prefix-phonemes (secondary) + suffix-phonemes (full, with primary).
    // Matches the reference compound-word stress algorithm for $strend2-flagged prefixes.
    if (norm.size() >= 5 && !compound_prefixes_.empty()) {
        for (const auto& [pref, pref_ph] : compound_prefixes_) {
            // Require minimum prefix length of 4 to avoid false splits on short function words
            // (e.g., "his"=3 would wrongly split "history" as "his"+"tory").
            if (pref.size() < 4 || pref.size() >= norm.size()) continue;
            size_t sfx_len = norm.size() - pref.size();
            if (sfx_len < 2) continue;
            if (norm.compare(0, pref.size(), pref) != 0) continue;
            std::string suffix = norm.substr(pref.size());
            // Suffix must have a vowel letter
            bool sfx_vowel = false;
            for (char c : suffix) if (isVowelLetter(c)) { sfx_vowel = true; break; }
            if (!sfx_vowel) continue;
            // Suffix must be at least 4 chars OR be a recognized dict word.
            // This prevents false splits like "hers"+"elf"="herself" where "elf" (3 chars,
            // not in dict) is not a real compound-forming morpheme. Legitimate compounds
            // like "over"+"all" pass because "all" is in the dictionary.
            if (sfx_len < 4 && dict_.find(suffix) == dict_.end() &&
                verb_dict_.find(suffix) == verb_dict_.end()) continue;
            // Process prefix phonemes: apply full processing.
            // Multi-syllable prefix (≥2 vowel codes) → demote primary to secondary stress.
            // Mono-syllable prefix → remove stress marker (prefix is unstressed).
            std::string pfx_ph = processPhonemeString(pref_ph);
            {
                // Count syllables (vowel-phoneme tokens) in the processed prefix string.
                // Uses a multi-char-aware tokenizer so diphthongs (oU, eI, O@, ...) count as
                // one syllable rather than two.
                static const char* MC_VOWELS[] = {
                    "O@","o@","U@","A@","e@","i@","aI@3","aI3","aU@","aI@","i@3","3:r","A:r",
                    "o@r","A@r","e@r","eI","aI","aU","OI","oU","IR","VR",
                    "3:","A:","i:","u:","O:","e:","a:","aa",
                    "@L","@2","@5","I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A#",
                    nullptr
                };
                int nvowels = 0;
                for (size_t pi = 0; pi < pfx_ph.size(); ) {
                    char c = pfx_ph[pi];
                    if (c == '\'' || c == ',' || c == '%' || c == '=') { pi++; continue; }
                    // Try multi-char vowel codes first
                    bool matched = false;
                    for (int mi = 0; MC_VOWELS[mi]; mi++) {
                        int ml = (int)strlen(MC_VOWELS[mi]);
                        if ((int)pi + ml <= (int)pfx_ph.size() &&
                            pfx_ph.compare(pi, ml, MC_VOWELS[mi]) == 0) {
                            nvowels++;
                            pi += ml;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        if (isVowelCode(std::string(1, c))) nvowels++;
                        pi++;
                    }
                }
                if (nvowels >= 2) {
                    // Multi-syllable prefix: demote primary → secondary
                    for (char& c : pfx_ph)
                        if (c == '\'') { c = ','; break; }
                } else {
                    // Mono-syllable prefix: remove stress markers (prefix becomes unstressed)
                    pfx_ph.erase(std::remove(pfx_ph.begin(), pfx_ph.end(), '\''), pfx_ph.end());
                    pfx_ph.erase(std::remove(pfx_ph.begin(), pfx_ph.end(), ','), pfx_ph.end());
                }
            }
            // Get suffix phonemes through full wordToPhonemes (handles dict, -ing/-ed, rules).
            std::string sfx_ph = wordToPhonemes(suffix);
            std::string combined = pfx_ph + sfx_ph;
            // Apply $N stress position override for the full word (e.g. "overture" $1).
            // stress_pos_ is set for words with a flag-only en_list entry like "overture $1".
            {
                auto sit = stress_pos_.find(norm);
                if (sit != stress_pos_.end())
                    combined = processPhonemeString(applyStressPosition(combined, sit->second));
            }
            return combined;
        }
    }

    // Apply rules, post-process, and apply word-level voiced assimilation
    // Apply $N stress position override if this word has one in the dictionary
    std::string raw_ph = applyRules(norm);
    {
        auto sit = stress_pos_.find(norm);
        if (sit != stress_pos_.end())
            raw_ph = applyStressPosition(raw_ph, sit->second);
    }
    std::string ph = processPhonemeString(raw_ph);

    // 7. Final sibilant voicing normalization (applies when word ends in 's').
    //    Only applies when the word ends in the letter 's' (a morpheme suffix).
    //    Words like "science" end in 'ce' → 's' phoneme but are NOT a suffix.
    //    Skip when word ends in 'ss' (double-s root, like "pass", "lass") — always voiceless.
    //    Skip when word ends in '-us', '-ous', '-ius' etc. (Latin stems, adjective suffixes).
    //    These have 's' as part of the base word, not a morphological suffix.
    bool ends_in_us_suffix = (norm.size() >= 2 && norm.back() == 's' && norm[norm.size()-2] == 'u');
    if (!norm.empty() && norm.back() == 's' &&
        !(norm.size() >= 2 && norm[norm.size()-2] == 's') &&
        !ends_in_us_suffix) {
        static const std::string UNVOICED_FINALS = "ptkfTSCxXhs";

        // 7a. NOTE: voiced assimilation (final 's' → 'z' after voiced sounds) is intentionally
        // NOT applied here. The en_rules suffix rules already emit the correct voicing:
        // e.g. @)s(_NS1m → 'z', &t)s(_S1m → 's', etc. Adding a post-processing voicing step
        // would incorrectly voice root-final 's' in words like "gas" (→ 's' not 'z').

        // 7b. NOTE: devoicing of final 'z' after unvoiced consonants is NOT applied here.
        // the reference determines voicing purely from the letter context via suffix rules
        // (e.g. "graphs" → '@)s rule → 'z', not 's', because 'h' is letter before 's').
        // Our previous devoicing step was wrong for these cases.
    }

    if (std::getenv("PHON_DEBUG")) std::cerr << "[RULES] " << norm << " -> " << ph << "\n";
    return ph;
}

// ============================================================
// Apply $N stress position: remove all existing stress markers and
// insert primary stress (') before the Nth vowel (1-based).
// Used to implement en_list $N flags (e.g. "lemonade $3").
// ============================================================
std::string IPAPhonemizer::applyStressPosition(const std::string& raw, int n) const {
    // Same multi-char code table used in processPhonemeString
    static const char* S_MC2[] = {
        "aI@3","aU@r","aI@","aI3","aU@","i@3","3:r","A:r","o@r","A@r","e@r",
        "eI","aI","aU","OI","oU","tS","dZ","IR","VR",
        "e@","i@","U@","A@","O@","o@",
        "3:","A:","i:","u:","O:","e:","a:","aa",
        "@L","@2","@5",
        "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~","A#",
        "r-","w#","t#","d#","z#","t2","d2","n-","m-","l/","z/",
        nullptr
    };
    auto findC = [&](const std::string& s, size_t pos) -> std::string {
        for (int mi = 0; S_MC2[mi]; mi++) {
            int mclen = (int)strlen(S_MC2[mi]);
            if (pos + (size_t)mclen <= s.size() &&
                s.compare(pos, mclen, S_MC2[mi]) == 0)
                return std::string(S_MC2[mi], mclen);
        }
        return std::string(1, s[pos]);
    };

    // Remove ALL existing stress markers (' and ,) so step 5a can re-distribute
    std::string ph;
    ph.reserve(raw.size());
    for (char c : raw) if (c != '\'' && c != ',') ph += c;

    // Find and mark the Nth vowel
    int vowel_count = 0;
    size_t pi = 0;
    while (pi < ph.size()) {
        char c = ph[pi];
        if (c == '%' || c == '=' || c == '|') { pi++; continue; }
        std::string code = findC(ph, pi);
        if (isVowelCode(code)) {
            vowel_count++;
            if (vowel_count == n) {
                ph.insert(pi, 1, '\'');
                return ph;
            }
        }
        pi += code.size();
    }
    // If word has fewer syllables than N, return without inserting (step 5 will handle)
    return ph;
}

// ============================================================
// Post-process raw phoneme string (steps 1–6)
// Called by wordToPhonemes (for rule-derived and suffix-stripped words)
// ============================================================
std::string IPAPhonemizer::processPhonemeString(const std::string& ph_in, bool force_final_stress) const {
    std::string ph = ph_in;
    bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");
    bool trace_0 = std::getenv("PHON_TRACE0") != nullptr;
    auto chk0 = [&](const char* s) { if (trace_0) std::cerr << "[" << s << "] ph=" << ph << "\n"; };

    // Strip rule boundary markers (\x01) while recording positions.
    // rule_boundary_after[i] = true means a rule boundary exists after ph[i].
    // Used by step 5a to distinguish true centering diphthongs (aI@, aU@, etc. from one rule)
    // from separately-emitted codes that happen to look like them ("bio" = aI + @).
    std::vector<bool> rule_boundary_after;
    if (ph.find('\x01') != std::string::npos) {
        std::string stripped;
        stripped.reserve(ph.size());
        std::vector<bool> boundaries;
        boundaries.reserve(ph.size());
        for (size_t i = 0; i < ph.size(); i++) {
            if (ph[i] == '\x01') {
                if (!stripped.empty())
                    boundaries.back() = true;
            } else {
                stripped += ph[i];
                boundaries.push_back(false);
            }
        }
        ph = stripped;
        rule_boundary_after = std::move(boundaries);
    }
    // rule_boundary_after may be empty (no \x01 markers) → treated as all-false

    // Convert '\x02' (phonSTRESS_PREV-demoted secondary) to ',' (secondary stress).
    // '\x02' is used as a "protected secondary" marker by applyRules' phonSTRESS_PREV handler
    // so that ph_in.find(',') stays std::string::npos (allowing step 5a to run).
    // Step 5a runs, sees ',' markers (from this conversion) and knows not to add more secondary.
    // Since step5a_ran=true, 5a-cleanup is skipped — preserving these secondaries.
    for (char& c : ph) {
        if (c == '\x02') c = ',';
    }

    // 1. Velar nasal assimilation: n+k/g → N+k/g (ŋ before velar stops k, g)
    // the reference applies this in phoneme post-processing:
    // - "income" → ˈɪŋkʌm (n+k), "congress" → kˈɑːŋɡɹəs (n+g)
    // - "engage" → ɛŋɡˈeɪdʒ (n+g from 'en' group output + standalone 'g' rule)
    // Rule: 'n' immediately before 'k' or 'g' in phoneme string → 'N' (ŋ).
    for (size_t i = 0; i + 1 < ph.size(); i++) {
        if (ph[i] == 'n' && (ph[i+1] == 'k' || ph[i+1] == 'g')) {
            ph[i] = 'N';
        }
    }

    // 2. Happy tensing: word-final unstressed ɪ → i (American English)
    // In the reference en-us, a word-final unstressed /I/ (ɪ) is realized as /i/ (tense i).
    // This applies to: %I, I2, and bare I at word end (when not stressed = no ' or , before).
    if (is_en_us) {
        if (ph.size() >= 2 && ph[ph.size()-2] == '%' && ph[ph.size()-1] == 'I') {
            ph.resize(ph.size()-2);
            ph += 'i';
        } else if (ph.size() >= 2 && ph[ph.size()-2] == 'I' && ph[ph.size()-1] == '2') {
            ph.resize(ph.size()-2);
            ph += 'i';
        } else if (!ph.empty() && ph.back() == 'I') {
            // Bare I at word end: convert to i unless it's part of a diphthong (eI, aI, OI)
            // or is stress-marked, or is the only vowel (monosyllabic).
            char prev = (ph.size() >= 2) ? ph[ph.size()-2] : 0;
            // Don't convert if: stress-marked, or part of diphthong (prev is a vowel char)
            static const std::string DIPHTHONG_BEFORE = "eaOoU";
            bool part_of_diph = (prev != 0 && DIPHTHONG_BEFORE.find(prev) != std::string::npos);
            if (prev != '\'' && prev != ',' && !part_of_diph) {
                // Also don't convert if I is the only vowel in the word (monosyllabic)
                static const std::string ALL_VOWELS = "aAeEiIOUV03o";
                int vowel_count = 0;
                for (char c : ph) {
                    if (ALL_VOWELS.find(c) != std::string::npos) vowel_count++;
                }
                if (vowel_count > 1) {
                    ph.back() = 'i';
                }
            }
        }
    }

    // 3. Vowel reduction: unstressed back/front vowels → schwa in American English.
    if (is_en_us) {
        struct Repl { const char* from; const char* to; char not_followed_by; };
        static const Repl REDUCTIONS[] = {
            // Note: '%0#', '%0', '=0#', '=0' (unstressed ɑː/ɑ) are NOT reduced to schwa here.
            // the reference keeps ɑː in content word prefixes like "con-", "op-", "vol-".
            // Step 5.5d/e handle '0#' reduction between stress markers separately.
            // Note: %V/=V (unstressed ʌ) is NOT reduced to schwa — the reference keeps ʌ in
            // un-/up-/sub- prefixes (e.g. "unlike" → ʌnlˈaɪk, not ənlˈaɪk).
            {"%A:", "%@", 0}, {"=A:", "%@", 0},
            {"%A",  "%@", '@'}, {"=A",  "%@", '@'},  // don't reduce %A@ (rhotic diphthong)
            {nullptr, nullptr, 0}
        };
        for (int ri = 0; REDUCTIONS[ri].from; ri++) {
            std::string from = REDUCTIONS[ri].from;
            std::string to   = REDUCTIONS[ri].to;
            char not_follow  = REDUCTIONS[ri].not_followed_by;
            size_t rpos = 0;
            while ((rpos = ph.find(from, rpos)) != std::string::npos) {
                // Check 'not_followed_by' constraint
                if (not_follow != 0) {
                    size_t after = rpos + from.size();
                    if (after < ph.size() && ph[after] == not_follow) {
                        rpos += from.size();
                        continue;
                    }
                }
                ph.replace(rpos, from.size(), to);
                rpos += to.size();
            }
        }
    }

    chk0("after-3");
    // 3b. LOT+R → THOUGHT+R in American English (pre-rhotic vowel neutralization).
    // '0r' (ɑːɹ) → 'O:r' (ɔːɹ): in American English, the LOT vowel (0) before /r/
    // merges with THOUGHT (O:). This applies universally: "forest"→O:r, "moral"→O:r,
    // "origin"→O:r, "horrible"→O:r, "correspond"→O:r, etc.
    // (the reference en-rules 1.51 has rules giving '0r' in these contexts, but the reference
    // compiles different rules that give 'O:r' / 'O@' for all pre-rhotic LOT vowels.)
    if (is_en_us) {
        size_t rpos = 0;
        while ((rpos = ph.find("0r", rpos)) != std::string::npos) {
            ph.replace(rpos, 2, "O:r");
            rpos += 3;
        }
    }

    chk0("after-3b");
    // 3c. Strip morpheme-boundary schwa before r: @-r → r.
    // the reference rules produce @-r for sequences like "or" before "ative" (e.g., collaborative,
    // decorative) and dict entries like "average"=av@-rI2dZ, "separate"=sEp@-r@t.
    // In the reference phoneme output, the '-' is a morpheme boundary; the '@' before '-r' is
    // a weak schwa that gets elided, leaving just the r consonant.
    // (Verified: the reference gives kəlˈæbɹətˌɪv / ˈævɹɪdʒ / sˈɛpɹɪt — no schwa before ɹ.)
    {
        size_t i = 0;
        while (i + 2 < ph.size()) {
            if (ph[i] == '@' && ph[i+1] == '-' && ph[i+2] == 'r') {
                ph.erase(i, 2); // remove @- , keep r
            } else {
                i++;
            }
        }
    }

    // 4. Bare schwa '@' immediately before 'r' → r-colored schwa '3'.
    // The 'r' is absorbed into '3' (ɚ) when followed by a consonant or end-of-word,
    // since 'r' only stays as a separate onset when the next phoneme is a vowel.
    if (is_en_us) {
        // Pre-pass: a#r → @r (unstressed 'a' before 'r' acts like schwa before 'r';
        // e.g. "around" a#raUnd → @raUnd → step 4 converts @r → 3 → ɚɹˈaʊnd).
        for (size_t i = 0; i + 2 < ph.size(); i++) {
            if (ph[i] == 'a' && ph[i+1] == '#' && ph[i+2] == 'r') {
                ph[i] = '@';
                ph.erase(i+1, 1); // remove '#', leaving @r
            }
        }
        for (size_t rpos = 0; rpos + 1 < ph.size(); rpos++) {
            if (ph[rpos] != '@') continue;
            // Find 'r' after '@', possibly skipping stress/modifier marks (%=',)
            size_t r_pos = rpos + 1;
            while (r_pos < ph.size() &&
                   (ph[r_pos]=='\''||ph[r_pos]==','||ph[r_pos]=='%'||ph[r_pos]=='='))
                r_pos++;
            if (r_pos >= ph.size() || ph[r_pos] != 'r') continue;
            bool is_diphthong = (rpos > 0 && (
                ph[rpos-1] == 'o' || ph[rpos-1] == 'A' || ph[rpos-1] == 'U' ||
                ph[rpos-1] == 'O' || ph[rpos-1] == 'e' || ph[rpos-1] == 'i' ||
                ph[rpos-1] == 'I' || ph[rpos-1] == 'a'));
            if (!is_diphthong) {
                ph[rpos] = '3';
                // Absorb 'r' if followed by a consonant or end-of-word.
                // Skip stress/modifier markers (%=',) to find the actual next phoneme.
                size_t after_r = r_pos + 1;
                while (after_r < ph.size() &&
                       (ph[after_r]=='%'||ph[after_r]=='='||
                        ph[after_r]=='\''||ph[after_r]==','))
                    after_r++;
                bool next_is_vowel = (after_r < ph.size() &&
                    (ph[after_r]=='a'||ph[after_r]=='A'||ph[after_r]=='e'||
                     ph[after_r]=='E'||ph[after_r]=='i'||ph[after_r]=='I'||
                     ph[after_r]=='o'||ph[after_r]=='O'||ph[after_r]=='u'||
                     ph[after_r]=='U'||ph[after_r]=='@'||ph[after_r]=='3'));
                // Absorb 'r' if: (a) next is not a vowel, OR (b) '@' was unstressed-prefixed (=/%)
                // e.g. "factory" → '=@ri' → '=3ri' (ɚɹi), keeping r before vowel
                bool unstressed_pre = (rpos > 0 &&
                    (ph[rpos-1] == '=' || ph[rpos-1] == '%'));
                if (!next_is_vowel || unstressed_pre)
                    ph.erase(r_pos, 1); // absorb the 'r' into '3' (ɚ)
            }
        }
    }

    // 4b. Linking R: rhotacized vowels followed by another vowel get a linking 'r'.
    // This applies to: '3' (ɚ), '3:' (ɜːr), 'U@' (ʊɹ), and 'A@' (ɑːɹ).
    // Examples: "forever" (3 before E → insert r), "preferring" (3: before I → insert r),
    //           "during" (U@ before %I → insert r → dˈʊɹɹɪŋ),
    //           "RNA" (A@ before ,E2 → insert r → ˌɑːɹɹˌɛnˈeɪ).
    // the reference source: ph_english_us phoneme definitions have IfNextVowelAppend(r-) for these.
    if (is_en_us) {
        static const std::string VOWEL_STARTS = "aAeEiIoOuUV03@";
        for (size_t rpos = 0; rpos < ph.size(); rpos++) {
            int code_len = 0;
            if (ph[rpos] == '3') {
                // '3' or '3:'
                code_len = 1;
                if (rpos + 1 < ph.size() && ph[rpos+1] == ':') code_len = 2;
            } else if (ph[rpos] == 'U' && rpos + 1 < ph.size() && ph[rpos+1] == '@') {
                // 'U@' = ʊɹ (rhotic U, as in "during"→dˈʊɹɹɪŋ)
                code_len = 2;
            } else if (ph[rpos] == 'A' && rpos + 1 < ph.size() && ph[rpos+1] == '@') {
                // 'A@' = ɑːɹ (rhotic A, IfNextVowelAppend(r-) in ph_english_us)
                // e.g. "RNA" R→A@, followed by vowel E2 → insert r: A@r,E2 → ɑːɹɹ
                code_len = 2;
            }
            if (code_len == 0) continue;
            // Skip if already has linking r after the code
            size_t after_code = rpos + code_len;
            if (after_code < ph.size() && ph[after_code] == 'r') continue;
            // Find next phoneme (skip stress/modifier markers)
            size_t after = after_code;
            while (after < ph.size() &&
                   (ph[after] == '\'' || ph[after] == ',' ||
                    ph[after] == '%'  || ph[after] == '='))
                after++;
            if (after < ph.size() && VOWEL_STARTS.find(ph[after]) != std::string::npos) {
                ph.insert(after_code, "r");
                rpos += code_len; // skip past the inserted 'r'
            }
        }
    }

    // 4c. -tion stress fix: move primary stress to the vowel immediately before '-tion' (S@n).
    // the reference SetWordStress detects the -tion suffix and places primary stress on the
    // syllable immediately before the -tion. Rules often put primary earlier (e.g. on
    // the first vowel), so here we move it when the pattern is detected.
    // Examples: "institution" (u: before S), "extraction" (a before kS), "production" (V before kS).
    if (is_en_us) {
        // Find 'S' (ʃ) in the phoneme string that is part of -tion suffix.
        // We look for 'S' followed by '@n' (with optional modifier chars in between).
        // Also handles 'SN' (as in "nation"→ neIS@N).
        static const std::string VOWEL_CODES_4C = "aAeEIiOUVu03@o";
        size_t S_pos = std::string::npos;
        // Find the last 'S' that is followed by @n pattern (at end of word).
        for (size_t k = (ph.size() >= 1 ? ph.size() - 1 : 0); k > 0; k--) {
            if (ph[k] == 'S') {
                size_t after_S = k + 1;
                bool has_tion = false;
                // Scan up to 10 chars for '@' followed by 'n'/'N'
                for (size_t ki = after_S; ki < ph.size() && ki < after_S + 10 && !has_tion; ki++) {
                    char c = ph[ki];
                    if (c == '=' || c == '%' || c == '-' || c == '\'' || c == ',') continue;
                    if (c == '@') {
                        for (size_t ki2 = ki + 1; ki2 < ph.size() && ki2 < ki + 4; ki2++) {
                            char c2 = ph[ki2];
                            if (c2 == '-' || c2 == '=' || c2 == '%') continue;
                            if (c2 == 'n' || c2 == 'N') {
                                // Guard: @n must be terminal — not followed by more syllables.
                                // "-ciency" produces S@ns%i where @n is NOT a -tion suffix.
                                // "-tion" plurals produce S@nz (%i is never stressable anyway).
                                // Allow only: end-of-string, 'z' (plural e.g. "nations"→S@nz),
                                // modifier chars, or terminal 'i' (from %i/%y).
                                // Do NOT allow 's': "conscience"=kVnS@ns, "deficiency"=dfIS@ns%i.
                                bool truly_terminal = true;
                                static const std::string VOWEL_CHARS_4C = "aAeEiIoOuUV03@";
                                for (size_t ki3 = ki2 + 1; ki3 < ph.size(); ki3++) {
                                    char c3 = ph[ki3];
                                    if (c3 == '%' || c3 == '=' || c3 == '-' ||
                                        c3 == '\'' || c3 == ',') { continue; }
                                    if (c3 == 'i') continue; // terminal %i or =i
                                    if (VOWEL_CHARS_4C.find(c3) != std::string::npos) {
                                        truly_terminal = false; break; // more syllables follow
                                    }
                                    if (c3 == 'z') continue; // plural "-tions" → @nz
                                    // any other consonant (incl. 's'): not purely terminal -tion
                                    truly_terminal = false; break;
                                }
                                if (truly_terminal) has_tion = true;
                                break;
                            }
                            break;
                        }
                    } else {
                        break; // non-modifier char that's not '@' → not a -tion suffix
                    }
                }
                if (has_tion) { S_pos = k; break; }
            }
        }
        if (S_pos != std::string::npos) {
            // Find the vowel immediately before S_pos (skipping backward over consonants).
            // The vowel before -tion is where primary stress should go.
            size_t vowel_pos = std::string::npos;
            // Scan backward from S_pos-1 to find a vowel code (using multi-char table).
            // We need to find the START of the vowel code.
            static const char* MC4c[] = {
                "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r","o@r","A@r","e@r",
                "eI","aI","aU","OI","oU","IR","VR",
                "e@","i@","U@","A@","O@","o@",
                "3:","A:","i:","u:","O:","e:","a:","aa",
                "@L","@2","@5",
                "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~","A#",
                nullptr
            };
            // Do a forward scan from 0 up to S_pos to find the last vowel start
            {
                size_t pi = 0;
                while (pi < S_pos) {
                    char c = ph[pi];
                    if (c == '\'' || c == ',' || c == '%' || c == '=' || c == '-') { pi++; continue; }
                    // Try multi-char codes
                    std::string vcode;
                    for (int mi = 0; MC4c[mi]; mi++) {
                        int mcl = (int)strlen(MC4c[mi]);
                        if (pi + (size_t)mcl <= S_pos &&
                            ph.compare(pi, mcl, MC4c[mi]) == 0) {
                            vcode = std::string(MC4c[mi], mcl);
                            break;
                        }
                    }
                    if (vcode.empty()) vcode = std::string(1, c);
                    if (!vcode.empty() && VOWEL_CODES_4C.find(vcode[0]) != std::string::npos) {
                        vowel_pos = pi;  // remember the start of the last vowel before S
                    }
                    pi += vcode.size();
                }
            }
            if (vowel_pos != std::string::npos) {
                // Check: primary is NOT already on this vowel.
                // Must scan backward through consonant onsets (e.g. 'ju:' has '\'' before 'j',
                // not immediately before 'u:': "contribution" = k0#ntrIb'ju:S=@n where '\'' is
                // at position 8 before 'j', but the reference considers this primary on the 'u:' syllable).
                bool primary_on_vowel = false;
                for (int bi = (int)vowel_pos - 1; bi >= 0; bi--) {
                    char bc = ph[bi];
                    if (bc == '\'') { primary_on_vowel = true; break; }
                    // Hit another vowel character → stop (the '\'' belongs to a prior syllable)
                    if (VOWEL_CODES_4C.find(bc) != std::string::npos) break;
                    // consonant, stress marker, or other modifier: continue scanning backward
                }
                if (!primary_on_vowel) {
                    // Find existing primary stress position
                    size_t prime_pos = ph.find('\'');
                    // Guard: if the existing primary is already AFTER vowel_pos, the rules placed
                    // it correctly on a later syllable (e.g. "-ality" in "functionality" places
                    // primary on 'al', not on the pre-tion 'V'). Don't override.
                    bool primary_after_vowel = (prime_pos != std::string::npos && prime_pos >= vowel_pos);
                    if (!primary_after_vowel) {
                        if (prime_pos != std::string::npos && prime_pos < vowel_pos) {
                            ph[prime_pos] = ','; // demote to secondary
                        }
                        // Remove ',' immediately before vowel (if any — was secondary)
                        if (vowel_pos > 0 && ph[vowel_pos-1] == ',') {
                            ph.erase(vowel_pos-1, 1);
                            vowel_pos--;
                        }
                        // Promote this vowel to primary
                        ph.insert(vowel_pos, 1, '\'');
                    }
                }
            }
        }
    }

    // 4d. -ology stress fix: words ending in '-ology' pattern get primary stress on the
    // vowel before 'l@dZ' (the '-ol-' in technology, biology, etc.).
    // the reference SetWordStress places primary stress one syllable before '-ology'.
    // Rules typically place it earlier, so we move it here.
    if (is_en_us) {
        // Look for pattern: vowel + 'l' + (optional =/@) + '@dZ' + unstressed vowel at end
        // In phoneme codes: '0l=@dZ%i' or similar. Key: '0' = ɑː (the -ol- vowel).
        // We look for '0l' not immediately preceded by primary stress, followed by schwa+dZ.
        size_t ol_pos = std::string::npos;
        for (size_t k = 0; k + 2 < ph.size(); k++) {
            if (ph[k] == '0' && ph[k+1] == 'l') {
                // Check that after '0l' there's a pattern ending in dZ (or dZ2 etc.)
                // and the whole thing is at end of word (phoneme string)
                size_t after_l = k + 2;
                // Skip modifier chars
                while (after_l < ph.size() && (ph[after_l]=='=' || ph[after_l]=='%' || ph[after_l]==',')) after_l++;
                if (after_l < ph.size() && ph[after_l] == '@') {
                    size_t after_at = after_l + 1;
                    if (after_at < ph.size() && ph[after_at] == 'd' && after_at + 1 < ph.size() && ph[after_at+1] == 'Z') {
                        ol_pos = k;
                        break;
                    }
                }
            }
        }
        if (ol_pos != std::string::npos) {
            bool primary_on_ol = (ol_pos > 0 && ph[ol_pos-1] == '\'');
            if (!primary_on_ol) {
                size_t prime_pos = ph.find('\'');
                if (prime_pos != std::string::npos && prime_pos < ol_pos) {
                    ph[prime_pos] = ','; // demote to secondary
                }
                if (ol_pos > 0 && ph[ol_pos-1] == ',') {
                    ph.erase(ol_pos-1, 1);
                    ol_pos--;
                }
                ph.insert(ol_pos, 1, '\''); // promote '0' to primary
            }
        }
    }

    // 4e. -ic/-ical/-ics stress fix: words with these suffixes get primary stress on the
    // penultimate syllable before the suffix (the syllable ending in the vowel before 'Ik').
    // e.g. fantastic (f'antast=Ik → fanˈtæstɪk), technology (already handled by 4d).
    // the reference: stress before -ic suffix (penultimate syllable rule).
    // Pattern: find '=I' (unstressed vowel I) followed by 'k' near end of phoneme string.
    // Move primary to the last stressed vowel before '=Ik'.
    if (is_en_us) {
        // Check if phoneme string ends in '=Ik' or '=Ik@L' or '=Iks' (for -ic, -ical, -ics)
        // We look for the last '=I' followed by 'k' or 'k@L' or 'ks'
        size_t ik_pos = std::string::npos;
        // Search for '=Ik' near end of string
        for (size_t k = ph.size(); k > 0; k--) {
            if (ph[k-1] == 'k') {
                // Check preceding: ...=I or ...I
                if (k >= 2 && ph[k-2] == 'I') {
                    // Check if I is unstressed (preceded by = or at start or after consonant)
                    bool i_unstressed = (k >= 3 && (ph[k-3] == '=' || ph[k-3] == '%')) ||
                                        (k == 2);
                    if (i_unstressed) {
                        ik_pos = k - 3; // position of '='
                        break;
                    }
                    // Also handle case where I is not preceded by = (e.g. raw Ik at end)
                    break;
                }
            }
        }
        if (ik_pos != std::string::npos) {
            // ik_pos points to '=' before 'Ik'
            // Move primary stress to the last vowel before ik_pos
            // Check: is there exactly one vowel syllable before '=Ik'?
            // Count vowels before ik_pos using multi-char-aware scan (to handle 'aa', 'aI', etc.)
            // This prevents inserting stress INSIDE a multi-char code like 'aa'.
            int vowels_before = 0;
            size_t last_vowel_pos = std::string::npos;
            {
                // Known 2-char vowel-like codes that should be treated as a unit
                static const char* MC4e[] = {
                    "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r","o@r","A@r","e@r",
                    "eI","aI","aU","OI","oU","IR","VR",
                    "e@","i@","U@","A@","O@","o@",
                    "3:","A:","i:","u:","O:","e:","a:","aa",
                    "@L","@2","@5",
                    "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~","A#",
                    nullptr
                };
                static const std::string VOWEL_C = "aAeEiIoOuUV03@";
                size_t vi = 0;
                while (vi < ik_pos) {
                    char c = ph[vi];
                    if (c == '\'' || c == ',' || c == '%' || c == '=') { vi++; continue; }
                    // Try multi-char codes
                    std::string vcode;
                    for (int mi = 0; MC4e[mi]; mi++) {
                        int mcl = (int)strlen(MC4e[mi]);
                        if ((int)(ik_pos - vi) >= mcl && ph.compare(vi, mcl, MC4e[mi]) == 0) {
                            vcode = std::string(MC4e[mi], mcl);
                            break;
                        }
                    }
                    if (vcode.empty()) vcode = std::string(1, c);
                    // Check if it's a vowel code
                    if (!vcode.empty() && VOWEL_C.find(vcode[0]) != std::string::npos) {
                        vowels_before++;
                        last_vowel_pos = vi;  // position of the START of the vowel code
                    }
                    vi += vcode.size();
                }
            }
            // Only apply if more than 1 vowel before -ic (otherwise stress is on the only syllable)
            if (vowels_before >= 2 && last_vowel_pos != std::string::npos) {
                bool primary_on_last = (last_vowel_pos > 0 && ph[last_vowel_pos-1] == '\'');
                if (!primary_on_last) {
                    size_t prime_pos = ph.find('\'');
                    if (prime_pos != std::string::npos && prime_pos >= last_vowel_pos) {
                        // Primary is on or after last_vowel_pos - leave it
                        primary_on_last = true;
                    } else if (prime_pos != std::string::npos && prime_pos < last_vowel_pos) {
                        ph[prime_pos] = ','; // demote to secondary
                    }
                    if (!primary_on_last) {
                        // Remove any ',' before last_vowel_pos
                        if (last_vowel_pos > 0 && ph[last_vowel_pos-1] == ',') {
                            ph.erase(last_vowel_pos-1, 1);
                            last_vowel_pos--;
                        }
                        ph.insert(last_vowel_pos, 1, '\'');
                    }
                }
            }
        }
    }

    // 5. Insert primary stress '\'' if no primary stress marker present.
    // Also handles the case where only secondary stress ',' exists (e.g., compound words
    // via rules: ",Vnd3stand" → insert '\'' on the main stressed syllable).
    //
    // Pre-step 5: If ph has ',' but no '\'', and ph_in has ',' (from dict) but no '\'',
    // and the first ',' is NOT at position 0 (which would indicate the whole word is secondary-
    // stressed, like "on" = ",O2n"), convert the first ',' to '\'' so that step 5 finds a
    // primary and doesn't re-insert.
    // Pre-step 5: intentionally removed.
    // Step 5 correctly handles ',' (secondary) without '\'': the secondary_next flag
    // skips the secondary-marked vowel and places primary on the next stressable vowel.
    // Shared multi-char code table used by steps 5, 5a, and 5b.
    static const char* S_MC[] = {
        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r","o@r","A@r","e@r",
        "eI","aI","aU","OI","oU","tS","dZ","IR","VR",
        "e@","i@","U@","A@","O@","o@",
        "3:","A:","i:","u:","O:","e:","a:","aa",
        "@L","@2","@5",
        "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~","A#",
        "r-","w#","t#","d#","z#","t2","d2","n-","m-","l/","z/",
        nullptr
    };
    auto findCode = [&](size_t pos) -> std::string {
        for (int mi = 0; S_MC[mi]; mi++) {
            int mclen = (int)strlen(S_MC[mi]);
            if (pos + mclen <= ph.size() &&
                ph.compare(pos, mclen, S_MC[mi]) == 0)
                return std::string(S_MC[mi], mclen);
        }
        return std::string(1, ph[pos]);
    };

    // Step 5: skip if the phoneme string starts with ',' from a DICT entry (inherent secondary
    // stress, like "weren't"=,w3:nt, "being"=,bi:IN). These function words stay secondary and
    // the phonemizeText layer promotes them to primary in isolation.
    //
    // EXCEPTION: when the phoneme string comes from RULES (rule_boundary_after non-empty) and
    // starts with ',', the leading ',' is a rule-emitted secondary stress for a prefix syllable
    // (e.g., "entertain" → ,Ent3teIn: '-en-' is secondary, '-tain' needs primary).
    // In that case, run step 5 but treat the leading ',' as secondary so primary goes on the
    // NEXT stressable vowel. Example: ,Ent3teIn → ,Ent3t'eIn → ˌɛntɚtˈeɪn.
    bool starts_with_secondary = (!ph.empty() && ph[0] == ',');
    bool is_rule_leading_comma = starts_with_secondary && !rule_boundary_after.empty();
    if (ph.find('\'') == std::string::npos && (!starts_with_secondary || is_rule_leading_comma)) {
        auto hasStrongAfter = [&](size_t pi) -> bool {
            bool unst = false;
            while (pi < ph.size()) {
                char c2 = ph[pi];
                if (c2 == '%' || c2 == '=' || c2 == ',') { unst = true; pi++; continue; }
                if (c2 == '\'') break;
                std::string tc = findCode(pi);
                if (isVowelCode(tc)) {
                    if (!unst && tc != "@" && tc != "@2" && tc != "@5" && tc != "@L" && tc != "3")
                        return true;
                    unst = false;
                }
                pi += tc.size();
            }
            return false;
        };

        // Determine stress insertion strategy:
        // - When the phoneme string has explicit unstressed markers ('=') AND there are no
        //   stressable full vowels after the last '=', pick the LAST stressable vowel before
        //   the unstressed tail. This matches the reference behavior for words with -ia/-ium endings
        //   (e.g. "plutonium" plu:toUn=i@m → stress oU not u:, "california" kalIfo@n=i@ → o@).
        // - When no explicit unstress markers exist, or when '=' precedes a full vowel (like
        //   "portfolio" =I2oU has stressable oU after =), pick the FIRST stressable vowel.
        // force_final_stress: $strend2 words with bare phonemes use end-stress (pick_last).
        bool pick_last = force_final_stress;
        {
            size_t last_eq = ph.rfind('=');
            if (trace_0) std::cerr << "[step5] pick_last_init=" << pick_last << " last_eq=" << (last_eq==std::string::npos?-1:(int)last_eq) << "\n";
            if (last_eq != std::string::npos) {
                // Pre-scan: check for stressable vowels after last '='.
                // 'i@' is a centering diphthong that appears as the unstressed -ia/-ium suffix
                // (e.g. =i@m in plutonium) and counts as weak for this check.
                bool has_strong_after = false;
                for (size_t pi2 = last_eq + 1; pi2 < ph.size(); ) {
                    std::string code2 = findCode(pi2);
                    if (isVowelCode(code2)) {
                        bool is_weak = (code2 == "@2" || code2 == "@5" || code2 == "@L" ||
                                        code2 == "I2" || code2 == "I2#" || code2 == "I#" ||
                                        code2 == "a#" || code2 == "@" || code2 == "3" ||
                                        code2 == "i@" ||  // centering diphthong in unstressed suffix
                                        code2 == "i");    // happy-tensed final -y (=i): treat as weak
                        if (!is_weak) { has_strong_after = true; break; }
                    }
                    pi2 += code2.size();
                }
                pick_last = !has_strong_after;
                if (trace_0) std::cerr << "[step5] has_strong_after=" << has_strong_after << " pick_last=" << pick_last << "\n";
            }
        }

        // When the phoneme string comes from rules (rule_boundary_after non-empty), commas
        // emitted by rules (e.g. "multi" → "m,VltI") are rule artifacts that should not
        // cause step 5 to skip those vowels for primary stress placement. the reference
        // SetWordStress ignores rule-emitted secondary markers when placing primary.
        // Dict entries with ',' (e.g. "made"=m,eId for compound suffix) ARE respected.
        bool ignore_comma_for_primary = !rule_boundary_after.empty();

        // Exception: when the phoneme string has secondary stress on the FIRST vowel
        // (comma at pi>0, after initial consonants) AND a strong diphthong exists later,
        // treat the secondary as genuine and use pick_last to find the correct primary.
        // E.g. "locomotive": l,oUk0moUtIv → secondary on first oU, primary on second oU.
        // E.g. "multiply": m,VltIplaI → secondary on V, primary on aI.
        // E.g. "carbohydrate": k,A@boUhaIdr@It → secondary on A@, primary on aI.
        bool use_pick_last_for_secondary = false;
        if (ignore_comma_for_primary) {
            // Find first comma at pi>0 (not at word start)
            size_t comma_pi = std::string::npos;
            for (size_t spi = 0; spi < ph.size(); spi++) {
                if (ph[spi] == '\'') break;
                if (ph[spi] == ',') {
                    if (spi > 0) { comma_pi = spi; }
                    break;
                }
            }
            if (comma_pi != std::string::npos) {
                // Skip past comma, its secondary vowel, and look for a strong diphthong after
                size_t scan = comma_pi + 1;
                while (scan < ph.size() && (ph[scan] == '\'' || ph[scan] == ',' ||
                                            ph[scan] == '%' || ph[scan] == '=')) scan++;
                // Skip the secondary-marked vowel itself
                if (scan < ph.size()) {
                    std::string sv = findCode(scan);
                    if (isVowelCode(sv)) scan += sv.size();
                }
                // Scan remaining phonemes for a strong vowel (diphthong or long vowel)
                static const char* STRONG_DIPH[] = {
                    "oU","aI","eI","aU","OI","aI@","aI3","aU@","i:","u:","A:","E:","3:","o:","U:",nullptr
                };
                while (scan < ph.size()) {
                    if (ph[scan] == '\'') break; // already have primary
                    std::string code2 = findCode(scan);
                    for (int si = 0; STRONG_DIPH[si]; si++) {
                        if (code2 == STRONG_DIPH[si]) { use_pick_last_for_secondary = true; break; }
                    }
                    if (use_pick_last_for_secondary) break;
                    scan += code2.size();
                }
            }
        }
        if (use_pick_last_for_secondary) {
            pick_last = true;
            ignore_comma_for_primary = false; // treat this comma as genuine secondary
        }

        bool unstressed = false;
        bool secondary_next = false; // next vowel is secondary-stressed, skip for primary
        size_t last_strong_pos = std::string::npos;  // last non-schwa stressable vowel
        size_t last_schwa_pos  = std::string::npos;  // last schwa/3 (fallback)
        size_t secondary_vowel_pos = std::string::npos; // position of first secondary-marked vowel
        size_t insert_pos = std::string::npos;
        size_t pi = 0;
        while (pi < ph.size()) {
            char c = ph[pi];
            if (c == '\'') break; // primary already here (shouldn't happen)
            if (c == ',') {
                // Always treat the leading ',' as secondary (even for rule-derived phonemes),
                // so primary stress lands on the next stressable vowel, not on the '-en-' syllable.
                if (!ignore_comma_for_primary || pi == 0) secondary_next = true;
                pi++; continue;
            }
            if (c == '%' || c == '=') { unstressed = true; pi++; continue; }
            std::string code = findCode(pi);
            if (isVowelCode(code)) {
                if (secondary_next) {
                    secondary_next = false;
                    unstressed = false;
                    // Track first secondary-marked vowel as a primary fallback
                    // (used when no non-schwa primary candidate exists elsewhere).
                    if (secondary_vowel_pos == std::string::npos)
                        secondary_vowel_pos = pi;
                    pi += code.size();
                    continue;
                }
                if (unstressed) {
                    unstressed = false;
                    pi += code.size();
                    continue;
                }
                if (code == "@2" || code == "@5" || code == "@L" ||
                    code == "I2" || code == "I2#" || code == "I#" ||
                    code == "a#" || code == "i") {
                    // 'i' = happy-tensed final vowel (e.g. -y in "victory", "factory")
                    // Never a primary stress target; skip it so the preceding vowel wins.
                    pi += code.size();
                    continue;
                }
                if (code == "@" || code == "3") {
                    // Schwa/r-colored schwa: can only be stress position as last resort.
                    if (!hasStrongAfter(pi + code.size())) {
                        last_schwa_pos = pi;
                        if (!pick_last && insert_pos == std::string::npos)
                            insert_pos = pi;
                    }
                    pi += code.size();
                    continue;
                }
                // Non-schwa stressable vowel.
                if (pick_last) {
                    // In pick_last mode triggered by leading secondary: treat 'I' (ɪ) as
                    // a weak vowel (like in "-tive", "-ti-") so it doesn't steal primary
                    // from a preceding diphthong. E.g. "locomotive": 2nd oU wins over 'I'.
                    if (use_pick_last_for_secondary && code == "I") {
                        last_schwa_pos = pi; // fallback only
                    } else {
                        last_strong_pos = pi;  // keep updating: last wins
                    }
                } else {
                    insert_pos = pi;       // first wins: stop here
                    break;
                }
            }
            pi += code.size();
        }
        if (pick_last) {
            insert_pos = (last_strong_pos != std::string::npos) ? last_strong_pos : last_schwa_pos;
        } else {
            // When picking FIRST stressable, apply the centering/initial-diphthong skip:
            // If primary landed on a centering diphthong (o@, e@, i@, U@) or an initial
            // true diphthong (aI, oU, etc.), look forward for a better candidate vowel.
            if (insert_pos != std::string::npos) {
                std::string found_code = findCode(insert_pos);
                bool is_diphthong = false; // disabled: initial diphthong IS the primary stress target
                                           // (e.g. "apricot" eIprIk0t → ˈeɪpɹɪkˌɑːt, "acorn" eIkO@n → ˈeɪkɔːɹn)
                bool is_centering = false; // disabled: 'o@'/'e@' are primary stress targets in en-us
                if (is_diphthong || is_centering) {
                    static const char* CENTERING_DIPHS[] = {
                        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
                        "3:r","A:r","o@r","e@r","e@","i@","U@","o@","3:","A:","i:","u:","O:","e:","a:","aa",
                        nullptr
                    };
                    static const char* ALL_DIPHS[] = {
                        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","aI","aU","eI","OI","oU",
                        "3:r","A:r","o@r","e@r","e@","i@","U@","o@","3:","A:","i:","u:","O:","e:","a:","aa",
                        nullptr
                    };
                    auto isSkippable = [&](const std::string& c) {
                        const char** skip_list = is_centering ? CENTERING_DIPHS : ALL_DIPHS;
                        for (int di = 0; skip_list[di]; di++)
                            if (c == skip_list[di]) return true;
                        return false;
                    };
                    size_t pi2 = insert_pos + found_code.size();
                    bool better_found = false;
                    size_t better_pos = std::string::npos;
                    bool unst2 = false, sec2 = false;
                    while (pi2 < ph.size()) {
                        char c2 = ph[pi2];
                        if (c2 == ',') { sec2 = true; pi2++; continue; }
                        if (c2 == '%' || c2 == '=') { unst2 = true; pi2++; continue; }
                        if (c2 == '\'') { pi2++; continue; }
                        std::string code2 = findCode(pi2);
                        if (isVowelCode(code2)) {
                            if (sec2 || unst2 ||
                                code2 == "@" || code2 == "@2" || code2 == "@5" || code2 == "@L" ||
                                code2 == "I#" || code2 == "I2" || code2 == "a#" || code2 == "3" ||
                                code2 == "i" ||  // happy-tensed final vowel: never a better stress candidate
                                isSkippable(code2)) {
                                sec2 = false; unst2 = false;
                                pi2 += code2.size();
                                continue;
                            }
                            better_pos = pi2;
                            better_found = true;
                            break;
                        }
                        pi2 += code2.size();
                    }
                    if (better_found) insert_pos = better_pos;
                }
            }
        }
        if (trace_0) std::cerr << "[step5] insert_pos=" << (insert_pos==std::string::npos?-1:(int)insert_pos) << " ph_before_insert=" << ph << "\n";
        // When the phoneme starts with '%' (whole-phrase unstressed marker) AND the
        // only candidate for primary stress is a schwa ('@'/'3'), do NOT insert '\''.
        // Phrase dict entries like "%f3@" (for a), "%,Dan@" (than a) are function-word
        // clitics that should remain fully unstressed — '%' already marks that.
        // Without this guard, processPhonemeString inserts '\'' before '@' as last resort,
        // which then defeats the pct_unstressed check in sentence context.
        bool suppress_schwa_stress = false;
        if (insert_pos != std::string::npos && ph.size() > 0 && ph[0] == '%') {
            // Only suppress if the insert_pos points to a true schwa ('@' or lone '3' ≠ '3:'),
            // not a strong vowel like '3:' (ɜː). '3:' IS a primary stress target.
            bool at_schwa = (insert_pos < ph.size() && ph[insert_pos] == '@');
            bool at_r_schwa = (insert_pos < ph.size() && ph[insert_pos] == '3' &&
                               (insert_pos + 1 >= ph.size() || ph[insert_pos+1] != ':'));
            suppress_schwa_stress = (at_schwa || at_r_schwa);
        }
        // Also suppress primary-on-schwa when a secondary-marked vowel exists in the word.
        // Words like "gonna" (g,@n@) have `,` marking secondary on the first vowel;
        // adding primary on the final schwa fallback is wrong. Instead, leave the word
        // as-is (g,@n@) and let StepC handle promotion/retention based on sentence context.
        // In isolation, StepC promotes the `,` → `'` (ɡˈənə).
        // In sentence context, the `,` stays secondary (ɡˌənə) when followed by stressed words.
        if (!suppress_schwa_stress && secondary_vowel_pos != std::string::npos &&
            insert_pos != std::string::npos) {
            bool at_schwa_fb = (insert_pos < ph.size() && ph[insert_pos] == '@');
            bool at_r_schwa_fb = (insert_pos < ph.size() && ph[insert_pos] == '3' &&
                                  (insert_pos + 1 >= ph.size() || ph[insert_pos+1] != ':'));
            if (at_schwa_fb || at_r_schwa_fb)
                suppress_schwa_stress = true;
        }
        if (insert_pos != std::string::npos && !suppress_schwa_stress) {
            ph.insert(insert_pos, 1, '\'');
        } else {
            // Last resort: a# (phUNSTRESSED) as the only vowel in the word.
            // When forced to take primary stress, the reduced phoneme reverts to its
            // base form: stressed a# → 'a' (= æ in en-us).
            // e.g. "an" (a#2n) → no other stressable vowel → 'a#' stressed → 'a2n → ˈæn
            // Safe: words like "about" (a#baUt) find 'aU' first so insert_pos is set.
            size_t pi2 = 0;
            while (pi2 < ph.size()) {
                std::string code = findCode(pi2);
                if (code == "a#") {
                    // Insert stress before 'a#'.
                    // If a variant-marker digit follows (e.g. 'a#2' in "an"=a#2n), remove '#'
                    // so it maps to æ (= base 'a'). the reference: "an" → ˈæn.
                    // If NO digit follows (plain 'a#' as in "than"=Da#n), keep '#' → maps to ɐ.
                    // the reference: "than" → ðˈɐn.
                    ph.insert(pi2, 1, '\'');  // '\'' before 'a'
                    // Check if '#' is followed by a variant-marker digit
                    // (pi2+2 is '#' position after insert, pi2+3 is next char)
                    bool has_variant_digit = (pi2 + 3 < ph.size()) &&
                        (ph[pi2 + 3] >= '1' && ph[pi2 + 3] <= '9' &&
                         ph[pi2 + 3] != '3' && ph[pi2 + 3] != '8');
                    if (has_variant_digit) {
                        ph.erase(pi2 + 2, 1);  // remove '#' → 'a' maps to æ
                        // Strip variant-marker digits
                        while (pi2 + 2 < ph.size()) {
                            char dc = ph[pi2 + 2];
                            if (dc >= '1' && dc <= '9' && dc != '3' && dc != '8')
                                ph.erase(pi2 + 2, 1);
                            else break;
                        }
                    }
                    // else: keep 'a#' → ɐ
                    break;
                }
                if (isVowelCode(code)) break; // another vowel encountered first — leave unstressed
                pi2 += code.size();
            }
        }
    }

    // 5.0. Stress-shift for words with explicitly unstressed '=' suffix syllable:
    // If ph has primary '\'' AND has '=' (explicitly unstressed marker from rules, e.g. in
    // "-ity" suffix: =I#t%i), AND the primary '\'' is NOT on the last stressable vowel before '=':
    // move primary '\'' to the last stressable vowel before '=' (demoting the old '\'' to ',').
    // Also de-flap any '*' immediately before the new '\'' position (since flapping only occurs
    // before UNSTRESSED vowels, not before primary-stressed ones).
    // Example: "creativity" kri:'eI*Iv=I#*%i → kri:,eIt'Iv=I#*%i → kɹiːˌeɪtˈɪvᵻɾi.
    // Guard: only fire when '=' comes AFTER '\'' and there is a stressable vowel between them
    // that is different from the current '\'' vowel position.
    bool step50_fired = false;  // track if step 5.0 moved primary (suppress extra backward secondary)
    if (ph.find('\'') != std::string::npos) {
        size_t eq_pos = ph.rfind('=');  // last '=' in string
        size_t prim_pos = ph.find('\''); // first '\'' in string
        if (eq_pos != std::string::npos && prim_pos < eq_pos) {
            // Guard: if there's already another '\'' between prim_pos+1 and eq_pos,
            // the rules have correctly placed a second primary (e.g. "participation"
            // has pA@t'IsIp'eIS=@n with two primaries). Don't move/demote anything.
            bool has_second_primary = (ph.find('\'', prim_pos + 1) < eq_pos);
            if (has_second_primary) goto skip_step50;
            // Scan for the last stressable vowel between prim_pos+1 and eq_pos-1.
            size_t last_sv_pos = std::string::npos;
            size_t scan = prim_pos + 1;
            while (scan < eq_pos) {
                char c = ph[scan];
                if (c == '\'' || c == ',' || c == '%' || c == '=' || c == '*') { scan++; continue; }
                std::string code = findCode(scan);
                if (isVowelCode(code)) {
                    // Is it stressable? (not schwa/reduced)
                    bool is_weak = (code == "@" || code == "@2" || code == "@5" || code == "@L" ||
                                    code == "3" || code == "I#" || code == "I2" || code == "a#" ||
                                    code == "i");
                    bool is_stressed = (scan > 0 && (ph[scan-1] == '\'' || ph[scan-1] == ','));
                    if (!is_weak && !is_stressed) last_sv_pos = scan;
                }
                scan += code.size();
            }
            // If last_sv_pos is found and it's after prim_pos (not the '\'' vowel itself)
            if (trace_0) std::cerr << "[5.0] prim_pos=" << prim_pos << " eq_pos=" << eq_pos << " last_sv_pos=" << (last_sv_pos==std::string::npos?-1:(int)last_sv_pos) << " ph=" << ph << "\n";
            if (last_sv_pos != std::string::npos) {
                // Check the vowel at prim_pos+1 is NOT last_sv_pos (otherwise '\'' is already correct).
                // '\'' can appear before a consonant onset (e.g. 'ju:' has '\'' before 'j', not 'u:').
                // Advance through consonants to find the actual stressed vowel.
                size_t prim_vowel_pos = prim_pos + 1;
                while (prim_vowel_pos < ph.size()) {
                    std::string pvc = findCode(prim_vowel_pos);
                    if (isVowelCode(pvc)) break;
                    prim_vowel_pos += pvc.size();
                }
                if (trace_0) std::cerr << "[5.0] prim_vowel_pos=" << prim_vowel_pos << " last_sv_pos=" << last_sv_pos << "\n";
                if (prim_vowel_pos != last_sv_pos) {
                    // Move primary from prim_vowel_pos to last_sv_pos.
                    // First, demote '\'' → ','
                    ph[prim_pos] = ',';
                    // De-flap '*' immediately before last_sv_pos: '*' before primary is /t/ not /ɾ/
                    if (last_sv_pos > 0 && ph[last_sv_pos - 1] == '*') {
                        ph[last_sv_pos - 1] = 't';
                    }
                    // Insert '\'' before last_sv_pos
                    ph.insert(last_sv_pos, 1, '\'');
                    step50_fired = true;
                }
            }
        }
        skip_step50:;
    }
    if (trace_0) std::cerr << "[after-5.0] ph=" << ph << "\n";
    chk0("after-5.0");

    // 5a. Insert secondary stress ',' at even syllable distances from primary.
    // Rule: scan syllables before and after primary. Every 2nd syllable (distance 2, 4, 6...)
    // gets secondary stress if it is a stressable vowel (not schwa/@/3, not % prefix).
    // This approximates the reference SetWordStress() secondary placement.
    // SKIP if the input already contains secondary stress markers ',' (e.g., dict entries
    // with explicit secondary stress, or stems from suffix stripping).
    bool step5a_ran = false;
    if (ph.find('\'') != std::string::npos && ph_in.find(',') == std::string::npos) {
        step5a_ran = true;
        // Build a list of syllable (vowel) positions in ph, along with their properties.
        // A syllable entry: {pos_in_ph, code, is_stressable, has_unstressed_prefix, already_has_secondary}
        struct SylInfo {
            size_t pos;      // position in ph string
            std::string code;
            bool stressable;   // can receive secondary stress
            bool is_primary;   // has primary stress marker before it
            bool already_secondary; // already has ',' before it
        };
        std::vector<SylInfo> syls;

        {
            size_t pi = 0;
            bool unstressed_prefix = false;
            bool secondary_prefix = false;
            bool primary_prefix = false;
            while (pi < ph.size()) {
                char c = ph[pi];
                if (c == '\'') { primary_prefix = true; pi++; continue; }
                if (c == ',')  { secondary_prefix = true; pi++; continue; }
                if (c == '%' || c == '=') { unstressed_prefix = true; pi++; continue; }
                std::string code = findCode(pi);
                // Centering diphthong split: 'aI@', 'aU@', 'aI3', 'aU@r', 'i@3' etc. are 3+ char
                // codes that may represent a true centering diphthong OR two separate syllables.
                // E.g. "scientist" = s'aI@ntIst: rule "ie"→"aI@" (1 syllable nucleus).
                //      "biological" = baI@l'0...: rule "bi"→"baI" + rule "o"→"@" (2 nuclei).
                // Use rule boundary markers: if a \x01 boundary existed within the first
                // (code.size()-1) chars of the code, the trailing '@'/'3' is a separate syllable.
                // This is exact (derived from actual rule output boundaries), not a heuristic.
                if (code.size() >= 3 && !primary_prefix && !secondary_prefix) {
                    char last = code.back();
                    if (last == '@' || last == '3') {
                        bool has_boundary = false;
                        // rule_boundary_after is indexed by position in the ORIGINAL stripped
                        // phoneme string (before stress markers '\'',',',%,= were inserted).
                        // pi is a position in the CURRENT string (with markers inserted).
                        // Compute pi_orig by subtracting the count of stress markers before pi.
                        size_t pi_orig = pi;
                        for (size_t q = 0; q < pi; q++) {
                            char qc = ph[q];
                            if (qc == '\'' || qc == ',' || qc == '%' || qc == '=') pi_orig--;
                        }
                        for (size_t k = 0; k + 1 < code.size(); k++) {
                            size_t check_pos = pi_orig + k;
                            if (check_pos < rule_boundary_after.size() && rule_boundary_after[check_pos]) {
                                has_boundary = true;
                                break;
                            }
                        }
                        if (has_boundary) {
                            code = code.substr(0, code.size() - 1);  // drop trailing '@'/'3'
                        }
                    }
                }
                if (isVowelCode(code)) {
                    // Skip morpheme-boundary schwa '@-': the reference does not count it as a full
                    // syllable for secondary stress placement purposes.
                    // E.g. "realistic" = ri:@-l'Ist=Ik: the @-l schwa is not counted,
                    // so syllables_before_primary = 1 (only i:) → no secondary placed on ɹiː.
                    if (code == "@" && pi + 1 < ph.size() && ph[pi + 1] == '-') {
                        pi += code.size();
                        continue; // skip: morpheme-boundary schwa doesn't count for stress rhythm
                    }
                    SylInfo si;
                    si.pos = pi;
                    si.code = code;
                    si.is_primary = primary_prefix;
                    si.already_secondary = secondary_prefix;
                    // Stressable: not schwa/schwa-variants, not I#/I2, not bare 'i' (happy-tensed),
                    // not explicitly unstressed. Bare 'i' is typically a happy-tensed final vowel
                    // (e.g. "lazy" → lˈeɪzi, "companies" → kˈʌmpəniz) and should not receive
                    // secondary stress.
                    bool is_schwa = (code == "@" || code == "@2" || code == "@5" || code == "@L" || code == "3");
                    bool is_reduced = (code == "I#" || code == "I2" || code == "i" ||
                                       code == "a#");  // a# = ɐ, phUNSTRESSED in the reference
                    si.stressable = !is_schwa && !is_reduced && !unstressed_prefix;
                    syls.push_back(si);
                    primary_prefix = false;
                    secondary_prefix = false;
                    unstressed_prefix = false;
                }
                pi += code.size();
            }
        }

        // Find primary stress syllable index
        int primary_idx = -1;
        for (int si = 0; si < (int)syls.size(); si++) {
            if (syls[si].is_primary) { primary_idx = si; break; }
        }

        if (primary_idx >= 0 && syls.size() >= 3) {
            // Mark secondary stress at even distances (2, 4, ...) from primary
            // Both backwards and forwards. Only mark stressable syllables not already stressed.
            std::vector<int> to_mark_secondary; // syllable indices to get ','

            // Count total syllables before primary (for secondary stress qualification).
            int syllables_before_primary = primary_idx; // primary_idx = count of syllables before it

            // Backward scan: find leftmost stressable vowel at dist >= 2 from primary,
            // then cascade rightward every 2 syllables.
            // the reference uses leftmost placement (not strictly even-distance-from-primary):
            // e.g. "gastrointestinal" primary at idx=3 → secondary at idx=0 (dist=3, odd)
            // rather than idx=1 (dist=2, even). Secondary cascade then continues at +2.
            // Require at least 2 syllables (including schwa) before primary.
            // Words with only 1 syllable before primary don't need secondary (e.g. "apart").
            if (syllables_before_primary >= 2 && !step50_fired &&
                ph.find(',') == std::string::npos) {
                // Find leftmost stressable vowel at dist >= 2 from primary
                int first_sec = -1;
                for (int idx = 0; idx <= primary_idx - 2; idx++) {
                    if (syls[idx].stressable && !syls[idx].already_secondary &&
                        !syls[idx].is_primary) {
                        first_sec = idx;
                        break;
                    }
                }
                if (first_sec >= 0) {
                    to_mark_secondary.push_back(first_sec);
                    // Cascade: place additional secondaries every 2 syllables from first_sec
                    for (int idx = first_sec + 2; idx <= primary_idx - 2; idx += 2) {
                        if (syls[idx].stressable && !syls[idx].already_secondary &&
                            !syls[idx].is_primary) {
                            to_mark_secondary.push_back(idx);
                        }
                    }
                    // Additional pass: also mark even-distance positions from primary going backward.
                    // Handles words like "telecommunications" where leftmost (first_sec) is at odd
                    // distance from primary, causing the cascade to miss even-distance positions
                    // (e.g. first_sec=0 at dist=5 from primary=5; P-2=3 at even dist=2 is missed).
                    // Guard: only add if at least 2 syllables away from every already-added secondary.
                    // This prevents adjacent secondaries (e.g. "hydroelectric": oU at idx=1 is
                    // adjacent to aI at idx=0, so it's skipped; "telecommunications": u: at idx=3
                    // is 3 away from E at idx=0, so it's added).
                    for (int dist = 2; primary_idx - dist >= 0; dist += 2) {
                        int idx = primary_idx - dist;
                        if (!syls[idx].stressable || syls[idx].already_secondary || syls[idx].is_primary)
                            continue;
                        bool too_close = false;
                        for (int m : to_mark_secondary) {
                            if (std::abs(m - idx) < 2) { too_close = true; break; }
                        }
                        if (!too_close) to_mark_secondary.push_back(idx);
                    }
                }
            }

            // Forward scan: distance 2, 4, 6, ...
            // Add forward secondary when there is ≥1 stressable syllable after primary
            // AND it falls at even distance (2, 4, ...) from primary.
            // This handles compounds like "butterfly" (dist=2, schwa at dist=1) and
            // longer words like "legislative" (schwa at dist=2, stressable at dist=3).
            // Short 2-syllable words never reach here (syls.size() >= 3 gate above).
            // Adjacent stressable syllables (dist=1) are never marked secondary (not in loop).
            {
                int stressable_after = 0;
                for (int idx = primary_idx + 1; idx < (int)syls.size(); idx++)
                    if (syls[idx].stressable) stressable_after++;
                if (stressable_after >= 1) {
                    for (int dist = 2; primary_idx + dist < (int)syls.size(); dist += 2) {
                        int idx = primary_idx + dist;
                        // Don't add secondary between two primaries: if there is another
                        // primary stress anywhere after idx, the word has multiple primaries
                        // (e.g. "personification": p3s'0nIfIk'eIS@n has '0 and 'eI),
                        // and the reference does not insert secondary between them.
                        bool later_primary = false;
                        for (int k = idx + 1; k < (int)syls.size(); k++)
                            if (syls[k].is_primary) { later_primary = true; break; }
                        if (later_primary) break; // stop forward scan
                        if (syls[idx].stressable && !syls[idx].already_secondary &&
                            !syls[idx].is_primary) {
                            to_mark_secondary.push_back(idx);
                        } else if (!syls[idx].stressable && !syls[idx].is_primary) {
                            // Even-distance slot is non-stressable; try one further
                            int idx2 = primary_idx + dist + 1;
                            if (idx2 < (int)syls.size() &&
                                syls[idx2].stressable && !syls[idx2].already_secondary &&
                                !syls[idx2].is_primary) {
                                // Also check for later primary before idx2
                                bool later_p2 = false;
                                for (int k = idx2+1; k<(int)syls.size(); k++)
                                    if (syls[k].is_primary) { later_p2=true; break; }
                                if (!later_p2) to_mark_secondary.push_back(idx2);
                            }
                        }
                    }
                }
            }

            // Insert ',' before each marked syllable, processing from right to left
            // to avoid invalidating positions.
            std::sort(to_mark_secondary.begin(), to_mark_secondary.end(),
                      [](int a, int b){ return a > b; }); // largest pos first
            for (int idx : to_mark_secondary) {
                ph.insert(syls[idx].pos, 1, ',');
            }
        }
    }
    chk0("after-5a");

    // 5a-cleanup: Remove secondary stress markers that are at syllable distance 1 from primary.
    // Rules can emit ',' for prefix syllables (e.g. "sc" → "s," for "scientific"),
    // but adjacent-to-primary secondary stress is discarded by the reference SetWordStress.
    // A secondary stress at distance 1 is phonologically invalid in English.
    // Only run when step 5a did NOT run: if step 5a ran, it only inserts ',' at valid positions
    // (distance >= 2 from primary), so cleanup would only incorrectly remove them.
    // Also only run for rule-derived phonemes (!rule_boundary_after.empty()): dict entries
    // with explicit ',' (e.g. "rainforest" = r'eInf,0rI2st) have trusted stress marks that
    // should NOT be removed just because they happen to be at distance 1 from primary.
    if (!step5a_ran && !is_rule_leading_comma && !rule_boundary_after.empty() &&
        ph.find('\'') != std::string::npos && ph.find(',') != std::string::npos) {
        // Build simple syllable list with stress-prefix flags
        struct SylE { size_t pos; bool is_primary; bool is_secondary; };
        std::vector<SylE> syls2;
        {
            size_t pi2 = 0;
            bool prim2 = false, sec2 = false;
            while (pi2 < ph.size()) {
                char c = ph[pi2];
                if (c == '\'') { prim2 = true; pi2++; continue; }
                if (c == ',')  { sec2  = true; pi2++; continue; }
                if (c == '%' || c == '=') { pi2++; continue; }
                std::string code2 = findCode(pi2);
                if (isVowelCode(code2)) {
                    syls2.push_back({pi2, prim2, sec2});
                    prim2 = sec2 = false;
                }
                pi2 += code2.size();
            }
        }
        int prim_idx2 = -1;
        for (int si = 0; si < (int)syls2.size(); si++)
            if (syls2[si].is_primary) { prim_idx2 = si; break; }
        if (prim_idx2 >= 0) {
            // Find the primary '\'' position in ph (first occurrence)
            size_t prim_ph_pos = ph.find('\'');
            // Find ',' markers at syllable distance 1 from primary and remove them
            std::vector<size_t> commas_to_remove;
            for (int si = 0; si < (int)syls2.size(); si++) {
                if (syls2[si].is_secondary && std::abs(si - prim_idx2) == 1) {
                    // Scan backward from syllable pos to find the ',' marker
                    size_t syl_pos = syls2[si].pos;
                    size_t comma_pos = std::string::npos;
                    for (int bp = (int)syl_pos - 1; bp >= 0; bp--) {
                        if (ph[bp] == ',') { comma_pos = (size_t)bp; break; }
                        if (ph[bp] == '\'' || ph[bp] == '%') break;
                    }
                    if (comma_pos == std::string::npos) continue;
                    // Guard: if ',' and '\'' are within the same rule (no rule boundary
                    // between them in rule_boundary_after), the secondary stress is
                    // intentional (e.g. rule "oe(ve" → ",oU'E" in "whatsoever").
                    // In that case, keep the secondary stress — don't remove.
                    bool same_rule = false;
                    if (!rule_boundary_after.empty() && prim_ph_pos != std::string::npos) {
                        size_t lo = std::min(comma_pos, prim_ph_pos);
                        size_t hi = std::max(comma_pos, prim_ph_pos);
                        same_rule = true;
                        for (size_t rp = lo; rp < hi && rp < rule_boundary_after.size(); rp++) {
                            if (rule_boundary_after[rp]) { same_rule = false; break; }
                        }
                    }
                    if (!same_rule)
                        commas_to_remove.push_back(comma_pos);
                }
            }
            // Remove from right to left to preserve positions
            std::sort(commas_to_remove.begin(), commas_to_remove.end(),
                      [](size_t a, size_t b){ return a > b; });
            for (size_t pos : commas_to_remove)
                ph.erase(pos, 1);
        }
    }
    chk0("after-5a-cleanup");

    // 5a-trochaic: Secondary stress for compound prefix words.
    // When step 5a is skipped (ph_in has ',') and the leading char is NOT ',' (i.e., the ','
    // came from a rule-internal prefix, not from a dict entry with inherent leading secondary),
    // run the reference trochaic rule: find the first vowel V that has no stress marker AND both
    // its neighboring vowels also have no ',' or '\'' marker (only '%'/none ≤ STRESS_IS_UNSTRESSED).
    // This replicates SetWordStress trochaic algorithm for compound prefix words like
    // "electroencephalography": `%Il,EktroUEns...` → the 'E' in "encephalo" gets ','
    // because its neighbors (oU and following E) both have level <= 1 (no marker or %).
    if (!step5a_ran && !starts_with_secondary && ph.find('\'') != std::string::npos
        && ph_in.find(',') != std::string::npos) {
        // Build syllable list: {pos, stress_level} where level -1=no marker, 1=%,2=,,4='
        struct Syl5a3 { size_t pos; std::string code; int level; };
        std::vector<Syl5a3> syls_t;
        {
            size_t pi = 0;
            int cur_level = -1;
            while (pi < ph.size()) {
                char c = ph[pi];
                if (c == '\'') { cur_level = 4; pi++; continue; }
                if (c == ',')  { cur_level = 2; pi++; continue; }
                if (c == '%' || c == '=') { cur_level = 1; pi++; continue; }
                std::string code = findCode(pi);
                if (isVowelCode(code)) {
                    syls_t.push_back({pi, code, cur_level});
                    cur_level = -1;
                } else {
                    cur_level = -1;  // consonants reset (stress marker only applies to next vowel)
                }
                pi += code.size();
            }
        }
        // For trochaic neighbor check, find the "effective" neighbor level by
        // skipping schwa/schwa-variants (which the reference treats as phNONSYLLABIC /
        // phUNSTRESSED and does NOT count in SetWordStress's vowel_stress array).
        // E.g. in "photorealistic" = f,oUtoUri:@-l'IstIk, the next vowel after i:
        // is '@' (schwa, level=-1), but the EFFECTIVE next is I (level=4, primary).
        // Without skipping schwa, the trochaic would incorrectly fire for i:.
        auto effectiveLv = [&](int sv, int dir) -> int {
            for (int nv = sv; nv >= 0 && nv < (int)syls_t.size(); nv += dir) {
                const std::string& vc = syls_t[nv].code;
                if (vc=="@"||vc=="@2"||vc=="@5"||vc=="@L"||vc=="3") continue;
                return syls_t[nv].level;
            }
            return -1;  // boundary = not stressed
        };

        // When ph_in has no primary ('/' or '='), the raw rules produced max_stress<PRIMARY.
        // the reference SetWordStress then uses stress=PRIMARY for the first trochaic assignment,
        // which MOVES the primary to the trochaic position (instead of the pick_last fallback),
        // then uses stress=SECONDARY for the next eligible vowel.
        // When ph_in HAS primary, the reference uses stress=SECONDARY throughout (one insert + break).
        bool input_has_primary = (ph_in.find('\'') != std::string::npos ||
                                  ph_in.find('=') != std::string::npos);
        bool trochaic_primary_done = false;

        // Find leftmost vowel with no marker (level=-1) where both neighbors are <= 1
        for (int sv = 0; sv < (int)syls_t.size(); sv++) {
            if (syls_t[sv].level != -1) continue;  // already has a marker
            const std::string& vcode = syls_t[sv].code;
            // Skip phUNSTRESSED phonemes and schwa-variants (forced to STRESS_IS_UNSTRESSED=1
            // by GetVowelStress, so vowel_stress[v]=1 ≠ -1 → trochaic never fires for them).
            if (vcode=="@"||vcode=="@2"||vcode=="@5"||vcode=="@L"||vcode=="3"||
                vcode=="I#"||vcode=="I2"||vcode=="a#") continue;
            // Skip bare 'i' (happy-tensed final, not stressable)
            if (vcode == "i") continue;
            int prev_lv = effectiveLv(sv-1, -1);
            int next_lv = effectiveLv(sv+1, +1);
            // Trochaic condition: both neighbors <= STRESS_IS_UNSTRESSED (1 in the reference)
            if (prev_lv <= 1 && next_lv <= 1) {
                if (!input_has_primary && !trochaic_primary_done) {
                    // No primary in raw rules output: move the step-5 pick_last primary
                    // to this trochaic position (the reference stress=PRIMARY first assignment).
                    size_t prime_pos = ph.find('\'');
                    if (prime_pos != std::string::npos) {
                        ph.erase(prime_pos, 1);
                        // Adjust syls_t: positions after removed char decrease by 1,
                        // and clear any stale level=4 (the old pick_last primary).
                        for (auto& s : syls_t) {
                            if (s.pos > prime_pos) s.pos--;
                            if (s.level == 4) s.level = -1;  // clear old primary
                        }
                    }
                    // Insert primary before this vowel
                    ph.insert(syls_t[sv].pos, "'");
                    // Adjust subsequent syls_t positions
                    for (int nv = sv+1; nv < (int)syls_t.size(); nv++) syls_t[nv].pos++;
                    syls_t[sv].level = 4;  // mark as primary for effectiveLv
                    trochaic_primary_done = true;
                    // Continue loop to find secondary
                } else {
                    // Secondary: either input already had primary, or we already placed primary
                    ph.insert(syls_t[sv].pos, ",");
                    for (int nv = sv+1; nv < (int)syls_t.size(); nv++) syls_t[nv].pos++;
                    break;  // only one secondary
                }
            }
        }
        chk0("after-5a-trochaic");
    }

    // 5a-final: Add secondary stress to last stressable bare vowel when its direct predecessor
    // is a schwa/phUNSTRESSED phoneme. Matches the reference SetWordStress trochaic rule for
    // the last syllable: the right neighbor is always the sentinel (= STRESS_IS_UNSTRESSED = 1 ≤ 1),
    // so only the left condition matters. the reference left check uses the direct previous vowel's
    // stress (not skipping schwas), and treats schwa/phUNSTRESSED as level=1 (≤1 → trochaic fires).
    // This fires for words where step 5a was skipped (due to ',' in input) and the existing
    // 5a-trochaic step missed the last syllable (because effectiveLv skips schwas to find a
    // non-schwa neighbor, which may be primary rather than the immediately-preceding schwa).
    // Example: "metamorphosis" m,E*@m'o@f@sIs → last I preceded by @ → secondary → ˌɪs.
    if (is_en_us && ph.find('\'') != std::string::npos) {
        // Build syllable list: {pos, code, level} where level -1=bare, 1=%, 2=',, 4='
        struct SylFin { size_t pos; std::string code; int level; };
        std::vector<SylFin> syls_fin;
        {
            size_t pi = 0;
            int cur_level = -1;
            while (pi < ph.size()) {
                char c = ph[pi];
                if (c == '\'') { cur_level = 4; pi++; continue; }
                if (c == ',')  { cur_level = 2; pi++; continue; }
                if (c == '%' || c == '=') { cur_level = 1; pi++; continue; }
                std::string code = findCode(pi);
                if (isVowelCode(code)) {
                    // Skip morpheme-boundary schwa (@-): the reference treats it as non-syllabic
                    // (part of a centering diphthong), so it does not count as a preceding
                    // vowel for the trochaic rule. E.g. "realized" ri:@-laIz: the @- is
                    // phNONSYLLABIC → vowel_count=3 (2 vowels: i:, aI), not 3.
                    if (code == "@" && pi + 1 < ph.size() && ph[pi + 1] == '-') {
                        pi += code.size();
                        continue; // don't count as syllable
                    }
                    syls_fin.push_back({pi, code, cur_level});
                    cur_level = -1;
                } else if (ph[pi] == '-') {
                    // morpheme boundary: skip
                    pi++;
                    continue;
                } else {
                    // consonants do NOT reset cur_level: the stress marker applies to the
                    // next vowel encountered, regardless of intervening consonants.
                    // (e.g. "%bIn" → % sets level=1, b is consonant, I gets level=1)
                }
                pi += code.size();
            }
        }
        if (syls_fin.size() >= 2) {
            auto& last = syls_fin.back();
            const std::string& lcode = last.code;
            // Must be bare (level=-1) and stressable (not schwa/phUNSTRESSED type)
            static const std::vector<std::string> UNSTRESSED_CODES =
                {"@","@2","@5","@L","3","I#","I2","a#"};
            auto is_unstressed_type = [&](const std::string& c) {
                for (auto& uc : UNSTRESSED_CODES) if (c == uc) return true;
                return false;
            };
            bool last_is_stressable = !is_unstressed_type(lcode) && lcode != "i";
            if (last.level == -1 && last_is_stressable) {
                auto& prev = syls_fin[syls_fin.size() - 2];
                // Preceding vowel is "unstressed" if it's a phUNSTRESSED-type OR explicitly %.
                // These correspond to the reference vowel_stress[v-1] <= STRESS_IS_UNSTRESSED = 1.
                bool prev_unstressed = is_unstressed_type(prev.code) || prev.level == 1;
                if (prev_unstressed) {
                    if (std::getenv("PHON_TRACE0"))
                        std::cerr << "[5a-final] adding secondary to last " << lcode
                                  << " at pos " << last.pos << " (prev=" << prev.code << ")\n";
                    ph.insert(last.pos, ",");
                }
            }
        }
    }
    chk0("after-5a-final");

    // 5.5-dim0: Reduce phoneme `0` (ɑː) to `@` (schwa) when it receives DIMINISHED stress.
    // In the reference, `phoneme 0` has ChangeIfDiminished(@). English stress_flags=0x08 has S_MID_DIM
    // (0x10000) NOT set, so ALL "middle" vowels with effective stress ≤ UNSTRESSED get DIMINISHED.
    // A "middle" vowel is one that is:
    //   - not the first syllable (v != 1, 1-indexed)
    //   - not the last syllable (v != N)
    //   - not the penultimate syllable (v == N-1) when the last syllable's initial stress ≤ 1
    // Only reduce `0` with no stress marker (bare, level -1) or explicit UNSTRESSED (level 1);
    // `0` with secondary or primary stress is kept as-is.
    // Reference: the reference SetWordStress finalization loop, dictionary.c lines 1503-1519.
    if (is_en_us && ph.find('0') != std::string::npos) {
        // Build syllable list with correct levels (stress markers persist through consonants)
        struct Syl0D { size_t pos; std::string code; int level; };
        std::vector<Syl0D> syls0d;
        {
            size_t pi = 0;
            int cur_level = -1;
            while (pi < ph.size()) {
                char c = ph[pi];
                if (c == '\'') { cur_level = 4; pi++; continue; }
                if (c == ',')  { cur_level = 2; pi++; continue; }
                if (c == '%' || c == '=') { cur_level = 1; pi++; continue; }
                std::string code = findCode(pi);
                if (isVowelCode(code)) {
                    // skip @- (morpheme-boundary non-syllabic schwa)
                    if (code == "@" && pi + code.size() < ph.size() && ph[pi + code.size()] == '-') {
                        pi += code.size();
                        continue;
                    }
                    syls0d.push_back({pi, code, cur_level});
                    cur_level = -1;
                } else if (ph[pi] == '-') {
                    pi++;
                    continue;
                }
                // consonants don't reset cur_level
                pi += code.size();
            }
        }
        int N0 = (int)syls0d.size();
        // Process right-to-left to preserve insertion positions
        for (int v = N0 - 1; v >= 0; v--) {
            if (syls0d[v].code != "0") continue;
            int lv = syls0d[v].level;
            if (lv > 1) continue; // has explicit secondary/primary stress → not DIMINISHED
            int vnum = v + 1; // 1-indexed
            if (vnum == 1) continue; // first syllable → UNSTRESSED, not DIMINISHED
            if (vnum == N0) continue; // last syllable → UNSTRESSED, not DIMINISHED
            // Penultimate before an unstressed-initial final syllable?
            if (vnum == N0 - 1) {
                int last_lv = syls0d[N0 - 1].level;
                // last_lv ≤ 1 (unassigned -1 or explicit %/=) → UNSTRESSED final → skip
                if (last_lv <= 1) continue;
            }
            // This `0` is a middle vowel with stress ≤ UNSTRESSED → DIMINISHED → reduce to `@`
            if (std::getenv("PHON_TRACE0"))
                std::cerr << "[5.5-dim0] reducing 0→@ at pos " << syls0d[v].pos
                          << " (vnum=" << vnum << "/" << N0 << ")\n";
            ph.replace(syls0d[v].pos, 1, "@");
        }
    }
    chk0("after-5.5-dim0");

    // NOTE: In the reference en-us, phoneme I# has NO ChangeIfDiminished — it always stays as ᵻ.
    // Do not reduce I# → I here; the distinction is made by the rules (en_rules) directly.
    chk0("after-5.5-dimI#");

    // 5a-prime: Adjacent primary demotion (mimics the reference phonSTRESS_PREV GetVowelStress logic).
    // When phonSTRESS_PREV (leading '=' in a rule output) fires, it promotes a preceding vowel
    // to PRIMARY and demotes any earlier PRIMARY to SECONDARY. This happens in the byte stream
    // BEFORE SetWordStress runs. Words where phonSTRESS_PREV fires mid-word can end up with two
    // primaries at syllable distance 1 (e.g. "creativity": eI+I from =I#t%i rule).
    // the reference then outputs ˌeɪtˈɪ (secondary + primary) because phonSTRESS_PREV demoted the
    // earlier primary. Since our step 5a receives two primaries in the raw phoneme string
    // (inserted by applyRules' phonSTRESS_PREV handler), we do the demotion here, AFTER
    // step 5a has run (so the "," we insert does not confuse step 5a).
    //
    // Rule: if any two consecutive PRIMARY syllables in the syllable list are at distance 1
    // (adjacent, no unstressed syllable between them), demote the EARLIER one to secondary.
    // This matches the reference phonSTRESS_PREV: it scans backward → promotes the preceding vowel →
    // demotes earlier primaries. Result: adjacent primaries never coexist; earlier → secondary.
    // Guard: only fire when there are indeed multiple "'" markers and the word's raw phoneme
    // (ph_in) also has "'" (i.e., the double-primary came from rules, not dict+suffix).
    if (ph_in.find('\'') != std::string::npos) {
        // Build syllable list with primary/secondary flags
        struct SylP { size_t pos; bool is_primary; };
        std::vector<SylP> sylsp;
        {
            size_t pi = 0;
            bool prim = false;
            while (pi < ph.size()) {
                char c = ph[pi];
                if (c == '\'') { prim = true; pi++; continue; }
                if (c == ',' || c == '%' || c == '=') { prim = false; pi++; continue; }
                std::string code = findCode(pi);
                if (isVowelCode(code)) {
                    sylsp.push_back({pi, prim});
                    prim = false;
                }
                pi += code.size();
            }
        }
        // Find pairs of primaries at distance 1 and demote the earlier one
        // Process right-to-left so index positions stay valid
        std::vector<size_t> demote_positions; // positions of "'" to change to ","
        for (int si = (int)sylsp.size() - 1; si >= 1; si--) {
            if (sylsp[si].is_primary && sylsp[si-1].is_primary &&
                (si - (si-1)) == 1) {
                // Two consecutive primaries: demote the EARLIER one (si-1)
                // Find the "'" marker before sylsp[si-1].pos
                if (sylsp[si-1].pos > 0) {
                    size_t comma_pos = sylsp[si-1].pos - 1;
                    if (comma_pos < ph.size() && ph[comma_pos] == '\'') {
                        demote_positions.push_back(comma_pos);
                        sylsp[si-1].is_primary = false; // mark as demoted
                    }
                }
            }
        }
        // Apply demotions: change "'" to "," at these positions
        for (size_t pos : demote_positions) {
            ph[pos] = ',';
        }
        if (!demote_positions.empty()) {
            chk0("after-5a-prime");
        }
    }

    // 5.5. Reduce bare '0' (unstressed ɑː) → '@' (schwa) in American English.
    // This runs after step 5a so that secondary stress markers are already placed.
    // '0' preceded by '\'' (primary) or ',' (secondary) stays as ɑː; bare '0' → schwa.
    // Mirrors the reference SetWordStress post-placement vowel reduction.
    // Only active when step 4c's -ution pattern triggered (i.e. ph has 'u:S').
    if (is_en_us && ph.find("u:S") != std::string::npos) {
        for (size_t pi = 0; pi < ph.size(); pi++) {
            if (ph[pi] == '0') {
                bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                if (!stressed) ph[pi] = '@';
            }
        }
    }

    // 5.5b. Reduce bare 'E' (ɛ) → 'I2' (ɪ) in unstressed syllables that come AFTER
    // a secondary-stress marker ',' and BEFORE the primary stress '\''.
    // the reference reduces ɛ→ɪ in two contexts:
    // 1. After secondary stress ',' and before primary '\'' (e.g. "mathematics" m,aTEm'at →
    //    m,aTI2m'at → mˌæθɪmˈæ..., "allegations" ,alEg'eI → ,alI2g'eI...).
    // 2. After an explicitly unstressed '%' initial syllable (e.g. "expectation"
    //    %e#kspEkt'eIS=@n → %e#kspI2kt'eIS=@n → ɛkspɪktˈeɪʃən).
    //    In this case there's no secondary stress marker, but the '%' prefix signals
    //    the initial syllable is a reduced function prefix (like "ex-").
    //    Does NOT reduce the first vowel after '%' (that vowel is the '%'-marked one).
    //    Does NOT reduce E when no such context applies (e.g. "September" sEpt'E stays sɛptˈɛ).
    if (is_en_us) {
        size_t primary_pos = ph.find('\'');
        size_t secondary_pos_e = ph.find(',');
        // Context 1: secondary stress before primary
        // Guard: only apply to rule-derived phonemes (rule_boundary_after non-empty).
        // Dict entries use explicit vowel quality chosen by the en_list author; the reference
        // preserves them (e.g. "recommendation" → rEk@mEnd'eIS@n keeps ɛ in "-mend-").
        if (!rule_boundary_after.empty() &&
            primary_pos != std::string::npos &&
            secondary_pos_e != std::string::npos &&
            secondary_pos_e < primary_pos) {
            // Determine the secondary-stressed vowel (immediately after ',').
            // When secondary vowel is itself 'E', subsequent bare E reduces to '@' (schwa)
            // instead of 'I2'. E.g. "presentation" ,Ez+E → '@', not 'I2'.
            // But "mathematics" ,aT+E → 'I2'.
            std::string sec_vowel_code;
            {
                size_t sv = secondary_pos_e + 1;
                while (sv < primary_pos && (ph[sv] == '\'' || ph[sv] == ',' ||
                                            ph[sv] == '%' || ph[sv] == '='))
                    sv++;
                if (sv < primary_pos) sec_vowel_code = findCode(sv);
            }
            bool sec_is_E = (sec_vowel_code == "E");

            // Track how many vowels we've seen since the secondary stress marker.
            // sec_is_E reduces E→'@' only when there's NO other vowel between the
            // secondary-stressed vowel and the bare E (i.e., it's in the immediately
            // adjacent syllable). When another vowel intervenes (e.g. 'I' in "Mediterranean"),
            // the reference uses 'I2' (ɪ) not '@'. vowels_seen starts at 0; the secondary-stressed
            // vowel itself counts as 1 when encountered.
            int vowels_seen_since_sec = 0;
            for (size_t pi = secondary_pos_e + 1; pi < primary_pos; pi++) {
                if (ph[pi] == 'E') {
                    // Check it's not E# or E2
                    bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '#' || ph[pi+1] == '2'));
                    if (is_variant) { vowels_seen_since_sec++; pi++; continue; }
                    // Check if stressed (preceded by ' or ,)
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        // E before nasal 'n': reduce to '@' (schwa), not 'I2' (ɪ).
                        // e.g. "compensation" k,0mpEns → k,0mp@ns (ə not ɪ).
                        // e.g. "presentation" ,Ez+En → '@' (n follows bare E).
                        // Otherwise → 'I2' (ɪ). e.g. "vegetarian" ,EdZEt → 'I2' (t follows).
                        bool before_n = (pi + 1 < primary_pos && ph[pi+1] == 'n');
                        if (before_n) {
                            ph[pi] = '@';
                        } else {
                            ph.replace(pi, 1, "I2");
                            primary_pos++; // adjust for the extra character inserted
                            pi++; // skip over the '2' we just inserted
                        }
                    }
                    vowels_seen_since_sec++;
                } else {
                    // Track other vowel codes
                    std::string vc = findCode(pi);
                    if (isVowelCode(vc)) {
                        vowels_seen_since_sec++;
                        pi += vc.size() - 1; // -1 because the loop adds 1
                    }
                }
            }
        }
        // Context 3: primary stress before secondary stress, 'E' in between (unstressed).
        // the reference reduces unstressed 'E' in inter-stress syllables when primary comes first:
        //   - E before nasal 'n' → '@' (schwa), e.g. "challenging" tS'alEndZ,IN → tS'al@ndZ,IN
        //   - E elsewhere → 'I2' (ɪ), e.g. "basketball" b'aa#skEtb,O:l → b'aa#skI2tb,O:l
        // Guard: only apply to rule-derived phonemes (rule_boundary_after non-empty).
        if (!rule_boundary_after.empty() &&
            primary_pos != std::string::npos &&
            secondary_pos_e != std::string::npos &&
            secondary_pos_e > primary_pos) {
            for (size_t pi = primary_pos + 1; pi < secondary_pos_e; pi++) {
                if (ph[pi] == 'E') {
                    bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '#' || ph[pi+1] == '2'));
                    if (is_variant) { pi++; continue; }
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        bool before_n = (pi + 1 < secondary_pos_e && ph[pi+1] == 'n');
                        if (before_n) {
                            ph[pi] = '@';
                        } else {
                            ph.replace(pi, 1, "I2");
                            secondary_pos_e++; // adjust for extra char inserted
                            pi++; // skip the '2'
                        }
                    }
                }
            }
        }
        // Context 2: '%' initial unstressed syllable (like "ex-" prefix).
        // Scan for 'E' between the end of the '%'-marked initial syllable and primary stress.
        if (primary_pos != std::string::npos && !ph.empty() && ph[0] == '%') {
            // Find the end of the first vowel code after '%' (that's the '%'-marked vowel; skip it).
            size_t scan_start = 1;
            // Advance past the first vowel code
            while (scan_start < ph.size() && (ph[scan_start] == '\'' || ph[scan_start] == ',' ||
                                               ph[scan_start] == '%' || ph[scan_start] == '='))
                scan_start++;
            // Now scan_start points to the first vowel code; advance past it
            if (scan_start < ph.size()) {
                std::string fc = findCode(scan_start);
                if (isVowelCode(fc)) scan_start += fc.size();
            }
            // Now reduce any bare 'E' between scan_start and primary_pos
            for (size_t pi = scan_start; pi < primary_pos; pi++) {
                if (ph[pi] == 'E') {
                    bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '#' || ph[pi+1] == '2'));
                    if (is_variant) { pi++; continue; }
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        ph.replace(pi, 1, "I2");
                        primary_pos++;
                        pi++;
                    }
                }
            }
        }
        // Context 4: bare 'E' after ALL explicit stress markers ('\'' and ','), not the last vowel.
        // Middle unstressed vowels get DIMINISHED stress in the reference → ChangeIfDiminished(I2) → ɪ.
        // Exception (the reference penultimate check): if E is the PENULTIMATE vowel AND the final vowel
        // has phUNSTRESSED attribute (codes: '@','@2','@5','@L','3','I#'), E stays UNSTRESSED → ɛ.
        // e.g. "physiotherapy" f'IzI2,oUTEr@p%i → E followed by 2 vowels (@,i) → middle → I2. ✓
        // e.g. "processes" pr'0sEs%I#z → E followed by 1 vowel (I#, phUNSTRESSED) → penultimate → ɛ. ✓
        // Guard: only apply to rule-derived phonemes.
        if (!rule_boundary_after.empty() && primary_pos != std::string::npos) {
            // Find the first explicit stress marker position (scan from there onward).
            // Using first_stress (not last_stress) ensures we also catch 'E' that lies
            // between two secondary stress markers (e.g. "physiotherapist" f'IzI2,oUTEr,eIpIst:
            // 'E' is between positions 6 and 12, after ',', before second ',').
            size_t first_stress = std::string::npos;
            for (size_t pi = 0; pi < ph.size(); pi++) {
                if (ph[pi] == '\'' || ph[pi] == ',') { first_stress = pi; break; }
            }
            if (first_stress != std::string::npos) {
                // phUNSTRESSED codes: inherently unstressed vowels (the reference phUNSTRESSED flag)
                auto isUnstressedCode = [](const std::string& vc) {
                    return (vc == "@" || vc == "@2" || vc == "@5" || vc == "@L" ||
                            vc == "3" || vc == "I#");
                };
                for (size_t pi = first_stress + 1; pi < ph.size(); ) {
                    if (ph[pi] == 'E') {
                        bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '#' || ph[pi+1] == '2'));
                        if (!is_variant) {
                            bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                            if (!stressed) {
                                // Collect vowels after this E
                                std::vector<std::string> vowels_after;
                                for (size_t pj = pi + 1; pj < ph.size(); ) {
                                    std::string vc = findCode(pj);
                                    if (isVowelCode(vc)) vowels_after.push_back(vc);
                                    pj += vc.size();
                                }
                                // Fire if: 2+ vowels after E (middle), OR exactly 1 vowel that's NOT phUNSTRESSED
                                bool should_reduce = false;
                                if (vowels_after.size() >= 2) {
                                    should_reduce = true;
                                } else if (vowels_after.size() == 1 && !isUnstressedCode(vowels_after[0])) {
                                    should_reduce = true;
                                }
                                if (should_reduce) {
                                    bool before_n = (pi + 1 < ph.size() && ph[pi+1] == 'n');
                                    if (before_n) {
                                        ph[pi] = '@';
                                    } else {
                                        ph.replace(pi, 1, "I2");
                                        pi += 2; // skip '2' we just inserted
                                        continue;
                                    }
                                }
                            }
                        } else {
                            pi += 2; continue;
                        }
                    }
                    pi++;
                }
            }
        }
    }


    chk0("after-5.5b");
    // 5.5b-nasal. Reduce 'I2' → '@' (schwa) when 'I2' appears in an unstressed syllable
    // immediately before nasal 'n' in the region before primary stress '\''  .
    // the reference SetWordStress reduces ɪ → ə before nasal clusters in unstressed syllables.
    // E.g. "implementation" ,ImplImI2nt'eIS → ,ImplIm@nt'eIS → ˌɪmplɪməntˈeɪʃən.
    //      "supplementation" s,VplImI2nt'eIS → s,VplIm@nt'eIS → sˌʌplɪməntˈeɪʃən.
    if (is_en_us && false) { // DISABLED: was causing regression (phenomenon)
        size_t prim_b = ph.find('\'');
        if (prim_b != std::string::npos) {
            for (size_t pi = 0; pi + 1 < prim_b; pi++) {
                if (ph[pi] == 'I' && pi + 1 < ph.size() && ph[pi+1] == '2') {
                    // Check not stressed (no ' or , immediately before)
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    // Check followed by 'n'
                    bool before_n = (pi + 2 < ph.size() && ph[pi+2] == 'n');
                    if (!stressed && before_n) {
                        ph.erase(pi+1, 1); // remove '2': I2 → I (so it becomes bare 'I')
                        ph[pi] = '@';       // then @ replaces 'I'
                        prim_b--; // adjust for erased char
                    }
                }
            }
        }
    }

    // 5.5b2. Reduce bare 'V' (ʌ) → '@' (schwa) in unstressed syllables between
    // secondary ',' and primary '\'' in American English.
    // the reference reduces ʌ → schwa in unstressed inter-stress syllables.
    // E.g. "productivity" pr,0#dVkt'Iv → pr,0#d@kt'Iv → pɹˌɑːdəktˈɪvᵻɾi.
    if (is_en_us) {
        size_t prim_v = ph.find('\'');
        size_t sec_v  = ph.find(',');
        if (prim_v != std::string::npos && sec_v != std::string::npos && sec_v < prim_v) {
            for (size_t pi = sec_v + 1; pi < prim_v; pi++) {
                if (ph[pi] == 'V') {
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) ph[pi] = '@';
                }
            }
        }
    }

    chk0("after-5.5b2");
    // 5.5c. Reduce bare 'a' (æ) → 'a#' (ɐ) in unstressed syllables that come AFTER
    // a secondary-stress marker ',' or an explicit unstressed '%' prefix and BEFORE
    // the primary stress '\''.
    // This matches the reference behavior:
    // - With ',': "analytical" ,anal'I → ,ana#l'I (inter-stress 'a' reduced)
    //   but "fantastic" fant'ast stays as-is (no ',' before first 'a').
    // - With '%': "transatlantic" tr%ansatl'aan → tr%ansp'a#tl'aan (bare 'a' after '%' region)
    //   but the 'a' directly after '%' is protected (it IS the unstressed prefix vowel).
    // Only applies to bare 'a' that is NOT a diphthong start (aI, aU, a:, a@, a#).
    // Only applies to rule-derived phonemes: dict entries specify vowel quality explicitly
    // (e.g. "adaptation" adapt'eIS@n has bare 'a' = æ, not ɐ, and must not be reduced).
    if (is_en_us && !rule_boundary_after.empty()) {
        size_t primary_pos_a = ph.find('\'');
        if (primary_pos_a != std::string::npos) {
            size_t secondary_pos_a = ph.find(',');
            size_t pct_pos_a = ph.find('%');
            // Determine scan start: earliest of ',' or '%' that comes before primary.
            size_t scan_start_a = std::string::npos;
            if (secondary_pos_a != std::string::npos && secondary_pos_a < primary_pos_a)
                scan_start_a = secondary_pos_a + 1;
            // '%' prefix case: scan from AFTER the first vowel following '%'.
            // The first vowel after '%' is the prefix vowel (v=1, PROTECTED/UNSTRESSED).
            // Any bare 'a' after that first vowel and before primary is a middle vowel
            // → DIMINISHED → reduce to 'a#'. Handles:
            //   "explanation" (%e#ksplan'eIS=@n): first vowel=e#, then 'a' is middle → ɐ.
            //   "transatlantic" (tr%ansatl'aant): first vowel='a' after %, scan past it.
            // Note: combined prefix+suffix strings (e.g. "inflammation" %Inflam'eIS@n)
            // have no \x01 markers → rule_boundary_after is empty → this block is skipped.
            if (pct_pos_a != std::string::npos && pct_pos_a < primary_pos_a) {
                // Advance past '%' to find the first vowel phoneme code
                size_t pct_scan = pct_pos_a + 1;
                while (pct_scan < primary_pos_a && !isVowelCode(findCode(pct_scan)))
                    pct_scan++;
                if (pct_scan < primary_pos_a) {
                    // Skip past the first (protected) vowel code
                    pct_scan += findCode(pct_scan).size();
                }
                if (pct_scan < primary_pos_a) {
                    scan_start_a = (scan_start_a == std::string::npos) ? pct_scan :
                                   std::min(scan_start_a, pct_scan);
                }
            }
            // General case: no '%' or ',' triggered a scan, but 'a' might still be a
            // middle vowel. Example: "reclamation" (rI#klam'eIS=@n) — the 'a' is v=2
            // of 4 vowels → DIMINISHED. Scan from AFTER the first vowel before primary.
            // (If '%' or ',' already triggered, scan_start_a is already set correctly.)
            if (scan_start_a == std::string::npos) {
                // Find the first vowel in the string before primary
                size_t gen_scan = 0;
                while (gen_scan < primary_pos_a) {
                    if (ph[gen_scan] == '\'' || ph[gen_scan] == ',' || ph[gen_scan] == '%')
                        { gen_scan++; continue; }
                    std::string gc = findCode(gen_scan);
                    if (isVowelCode(gc)) {
                        // Skip past this first (protected) vowel
                        gen_scan += gc.size();
                        break;
                    }
                    gen_scan += gc.size();
                }
                if (gen_scan < primary_pos_a)
                    scan_start_a = gen_scan;
            }
            if (scan_start_a != std::string::npos) {
                // Reduce bare 'a' that is:
                //   - AFTER the scan_start position
                //   - BEFORE the primary stress marker
                //   - NOT immediately following ',', '\'', or '%' (protected vowel)
                //   - NOT a diphthong start (aI, aU, a:, a@, a#)
                for (size_t pi = scan_start_a; pi < primary_pos_a; pi++) {
                    if (ph[pi] == 'a') {
                        bool is_diphthong_start = (pi + 1 < ph.size() &&
                            (ph[pi+1] == 'I' || ph[pi+1] == 'U' || ph[pi+1] == ':' ||
                             ph[pi+1] == '@' || ph[pi+1] == '#'));
                        if (is_diphthong_start) continue;
                        // 'a' directly after a stress marker or '%' is the marked vowel itself
                        bool protected_vowel = (pi > 0 &&
                            (ph[pi-1] == '\'' || ph[pi-1] == ',' || ph[pi-1] == '%'));
                        if (!protected_vowel) {
                            ph.insert(pi + 1, 1, '#');
                            primary_pos_a++;
                            pi++;
                        }
                    }
                }
            }
        }
    }

    chk0("after-5.5c");
    // 5.5c2. Reduce bare 'a' (æ) → 'a#' (ɐ) in unstressed syllables AFTER primary '\''
    // and BEFORE secondary ',' (primary comes first). This is the mirror of 5.5c.
    // the reference reduces inter-stress 'a' in both orderings: e.g. "analyst" 'anal,I → 'ana#l,I
    // → ˈænɐlˌɪst. The 'a' in the second syllable is unstressed between primary and secondary.
    // Only applies when primary exists before secondary in the phoneme string.
    // Only for rule-derived phonemes (same rationale as 5.5c above).
    if (is_en_us && !rule_boundary_after.empty()) {
        size_t primary_pos_a2 = ph.find('\'');
        size_t secondary_pos_a2 = ph.find(',');
        if (primary_pos_a2 != std::string::npos &&
            secondary_pos_a2 != std::string::npos &&
            primary_pos_a2 < secondary_pos_a2) {
            for (size_t pi = primary_pos_a2 + 1; pi < secondary_pos_a2; pi++) {
                if (ph[pi] == 'a') {
                    bool is_diphthong_start = (pi + 1 < ph.size() &&
                        (ph[pi+1] == 'I' || ph[pi+1] == 'U' || ph[pi+1] == ':' ||
                         ph[pi+1] == '@' || ph[pi+1] == '#'));
                    if (is_diphthong_start) continue;
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        ph.insert(pi + 1, 1, '#');
                        secondary_pos_a2++;
                        pi++;
                    }
                }
            }
        }
    }

    chk0("after-5.5c2");
    // 5.5d. Reduce bare '0' (ɑː) → '@' (schwa) in unstressed syllables that come AFTER
    // a secondary-stress marker ',' and BEFORE the primary stress '\''.
    // the reference reduces unstressed ɑː→schwa in the inter-stress region.
    // E.g. "democratic" d,Em0kr'at → d,Em@kr'at → dˌɛməkɹˈæɾɪk.
    // Must NOT reduce '0' that is itself stressed (',0' or "'0") or part of '0#','02' variants.
    if (is_en_us) {
        size_t primary_pos_0 = ph.find('\'');
        size_t secondary_pos_0 = ph.find(',');
        if (primary_pos_0 != std::string::npos &&
            secondary_pos_0 != std::string::npos &&
            secondary_pos_0 < primary_pos_0) {
            for (size_t pi = secondary_pos_0 + 1; pi < primary_pos_0; pi++) {
                if (ph[pi] == '0') {
                    // Skip variant forms 0#, 02
                    bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '#' || ph[pi+1] == '2'));
                    if (is_variant) { pi++; continue; }
                    // Check if stressed
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        ph[pi] = '@';
                    }
                }
            }
        }
    }

    chk0("after-5.5d");
    // 5.5d2. Reduce bare '0' (ɑː) → '@' (schwa) in unstressed position AFTER primary '\''
    // and BEFORE the next secondary ',' that follows primary. Mirror of step 5.5d.
    // E.g. "demonstrate" d'Em0nstr,eIt → d'Em@nstr,eIt → dˈɛmənstɹˌeɪt.
    // E.g. "parallelogram" p,ar@l'El0gr,am: secondary at pos 1 is BEFORE primary at pos 6,
    //   so ph.find(',') would find pos 1 and condition primary<secondary would fail.
    //   Fix: find first ',' AFTER primary to get the secondary at pos 12 → reduces '0' → '@'.
    if (is_en_us) {
        size_t primary_d2 = ph.find('\'');
        // Find first secondary marker AFTER primary (not any secondary in the string).
        size_t secondary_d2 = (primary_d2 != std::string::npos) ? ph.find(',', primary_d2) : std::string::npos;
        if (primary_d2 != std::string::npos &&
            secondary_d2 != std::string::npos &&
            primary_d2 < secondary_d2) {
            for (size_t pi = primary_d2 + 1; pi < secondary_d2; pi++) {
                if (ph[pi] == '0') {
                    bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '#' || ph[pi+1] == '2'));
                    if (is_variant) { pi++; continue; }
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        ph[pi] = '@';
                    }
                }
            }
        }
    }

    chk0("after-5.5d2");
    // 5.5e. Reduce '0#' → '@' (schwa) when it precedes primary stress,
    // UNLESS there is an UNSTRESSED vowel code between '0#' and the primary stress.
    //
    // Examples:
    //   "contains"    k%0#nt'eInz    → no intermediate vowel    → reduce → kəntˈeɪnz
    //   "consolidate" k%0#ns,0lId'   → first inter vowel '0' is stressed (,) → reduce → kən...
    //   "condensation"k%0#ndE2ns'eI  → first inter vowel 'E' is UNSTRESSED   → keep → kɑːnd...
    //   "condemnation"k%0#ndE2mn'eI  → same → keep → kɑːnd...
    //   "contribute"  k0#ntr'Ibju:t  → no intermediate vowel (bare 0#, no %) → reduce → kən...
    //   "volcanic"    v0lk'an=Ik     → bare '0' (no hash), not affected here
    // Step 5.5d already covers the between-secondary-and-primary case.
    if (is_en_us) {
        static const std::string VOW_CODES_0 = "aAeEiIoOuUV03@";
        size_t primary_pos_e = ph.find('\'');
        if (trace_0) std::cerr << "[5.5e] ph=" << ph << " primary_pos_e=" << primary_pos_e << "\n";
        if (primary_pos_e != std::string::npos) {
            for (size_t pi = 0; pi < primary_pos_e; pi++) {
                if (ph[pi] == '0') {
                    bool stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (trace_0) std::cerr << "[5.5e] pi=" << pi << " ph[pi-1]='" << (pi>0?ph[pi-1]:'?') << "' stressed=" << stressed << "\n";
                    if (!stressed) {
                        bool has_hash = (pi + 1 < ph.size() && ph[pi+1] == '#');
                        bool is_02 = (pi + 1 < ph.size() && ph[pi+1] == '2');
                        if (is_02) { pi++; continue; } // skip '02' variant
                        // Only reduce '0#' (not bare '0' which is always ɑː).
                        // Reduce UNLESS the first vowel code between '0#' and primary
                        // stress is unstressed (no preceding '\''/','). An unstressed
                        // intermediate vowel means the rule included extra phonemes that
                        // make the ɑː syllable phonologically heavy (e.g. "conde" rule).
                        if (has_hash) {
                            bool has_unstressed_inter_vowel = false;
                            for (size_t si = pi + 2; si < primary_pos_e; si++) {
                                char c = ph[si];
                                if (VOW_CODES_0.find(c) != std::string::npos) {
                                    bool sv = (si > 0 && (ph[si-1] == '\'' || ph[si-1] == ','));
                                    if (!sv) has_unstressed_inter_vowel = true;
                                    break; // only check first intermediate vowel
                                }
                            }
                            if (!has_unstressed_inter_vowel) {
                                ph.replace(pi, 2, "@"); // '0#' → '@' (one char shorter)
                                primary_pos_e--;
                            }
                            // else: '0#' with unstressed intermediate vowel stays as ɑː
                        }
                        // bare '0' always stays as ɑː
                    }
                }
            }
        }
    }

    chk0("after-5.5e");
    // 5.5f. Reduce bare '0' (ɑː) → '@' (schwa) in unstressed syllables BEFORE primary '\''.
    // This complements step 5.5d (which requires a preceding secondary ',') by handling
    // the case where there is no secondary stress. the reference SetWordStress reduces all
    // unstressed '0' phonemes to schwa.
    // E.g. "astronomical" a#str0n'0m=Ik@L → a#str@n'0m=Ik@L → ɐstɹənˈɑːmɪkəl.
    // Guards: '0' must be bare (no '\'' or ',' immediately before it) and before primary.
    // Only apply when phoneme came from rules (has \x01 boundary markers), NOT from dict entries.
    // Dict entries (like "volcano" v0lk'eInoU) already have the correct vowel quality as
    // the reference intended; reducing them to schwa would give wrong output (vəl- instead of vɑːl-).
    if (is_en_us && !rule_boundary_after.empty()) {
        size_t prim_pos_5f = ph.find('\'');
        if (prim_pos_5f != std::string::npos) {
            for (size_t pi = 0; pi < prim_pos_5f; pi++) {
                if (ph[pi] == '0') {
                    bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '#' || ph[pi+1] == '2'));
                    if (is_variant) { pi++; continue; }
                    // Don't reduce if stressed (',', '\'') or explicitly marked ('%', '=')
                    // '%0' = rule-explicit unstressed ɑː (keep); bare '0' = reduce to '@'.
                    bool marked = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ',' ||
                                              ph[pi-1] == '%' || ph[pi-1] == '='));
                    // Only reduce '0' that is a standalone rule output (boundary after it).
                    // When '0' is bundled with a following phoneme (e.g. '0l' from rule
                    // `v)ol(C 0l`), there's no rule boundary after '0', so don't reduce it.
                    // E.g. "volcanic" v0lk'an=Ik: '0' at pos 1 has no boundary (part of 0l) → keep ɑː.
                    //      "astronomical" a#str0n'0m=: '0' at pos 5 has boundary → reduce to @.
                    bool is_standalone = (pi < rule_boundary_after.size() && rule_boundary_after[pi]);
                    // Don't reduce if this '0' is the first vowel in the pre-tonic region.
                    // the reference keeps the initial-syllable '0' as ɑː even when pre-tonic.
                    // E.g. "oncology" 0Nk'0l: first '0' at pos 0, no prior vowel → keep ɑː.
                    //      "astronomical" a#str0n'0m=: prior vowel 'a' → reduce to @.
                    static const std::string VOW_5F_SET = "aAeEiIoOuUV03@";
                    bool has_prior_vowel = false;
                    for (size_t j = 0; j < pi; j++) {
                        if (VOW_5F_SET.find(ph[j]) != std::string::npos) {
                            has_prior_vowel = true; break;
                        }
                    }
                    if (!marked && is_standalone && has_prior_vowel) ph[pi] = '@';
                }
            }
        }
    }
    chk0("after-5.5f");

    // 5.5g. Reduce bare 'E' → '@' in pre-tonic position when the 'E' follows a
    // '%'-marked syllable (with a vowel between '%' and 'E') and is immediately
    // followed by nasal 'n'. the reference reduces this 'E' to schwa in AmEng.
    // E.g. "fermentation" f%3:mEnt'eIS=@n → f%3:m@nt'eIS=@n (fɜːmənˈteɪʃən)
    // Guard: only when there's a vowel between '%' and 'E' so that 'E' is NOT
    // the nucleus of the '%'-syllable (e.g. %Env... keeps ɛ since E is right after %).
    if (is_en_us) {
        static const std::string VOW_5G = "aAeEiIoOuUV03@";
        size_t prime_pos_g = ph.find('\'');
        if (prime_pos_g != std::string::npos) {
            for (size_t pi = 1; pi < prime_pos_g; pi++) {
                if (ph[pi] == 'E') {
                    bool is_variant = (pi + 1 < ph.size() && (ph[pi+1] == '2' || ph[pi+1] == '#'));
                    if (is_variant) { pi++; continue; }
                    if (ph[pi-1] == '\'' || ph[pi-1] == ',') continue; // stressed
                    if (pi + 1 >= ph.size() || ph[pi+1] != 'n') continue; // must precede 'n'
                    // Find last '%' before this 'E'
                    size_t pct_pos = std::string::npos;
                    for (size_t j = 0; j < pi; j++) {
                        if (ph[j] == '%') pct_pos = j;
                    }
                    if (pct_pos == std::string::npos) continue;
                    // Check there's at least one vowel between '%' and 'E'
                    bool has_vowel_between = false;
                    for (size_t j = pct_pos + 1; j < pi; j++) {
                        if (VOW_5G.find(ph[j]) != std::string::npos) {
                            has_vowel_between = true; break;
                        }
                    }
                    if (has_vowel_between) ph[pi] = '@';
                }
            }
        }
    }

    // 5b. Post-stress bare 'I' before syllabic L (@L) → 'i' (American English)
    // Only converts 'I' that is immediately followed by 'k@L' or directly before '@L',
    // not arbitrary 'I' between stress and @L (to avoid changing "political" = p@l'ItIk@L).
    if (is_en_us) {
        size_t stress_pos = ph.find('\'');
        if (stress_pos != std::string::npos) {
            size_t al_pos = ph.find("@L");
            if (al_pos != std::string::npos && al_pos > stress_pos) {
                for (size_t pi = stress_pos + 1; pi < al_pos; pi++) {
                    if (ph[pi] == 'I') {
                        if (pi + 1 < ph.size() && (ph[pi+1] == '2' || ph[pi+1] == '#'))
                            continue;
                        bool seen_stressed_vowel = false;
                        for (size_t k = stress_pos + 1; k < pi; k++) {
                            char ch = ph[k];
                            if (ch=='a'||ch=='A'||ch=='e'||ch=='E'||ch=='i'||ch=='I'||
                                ch=='o'||ch=='O'||ch=='u'||ch=='U'||ch=='V'||ch=='0'||
                                ch=='3'||ch=='@') { seen_stressed_vowel = true; break; }
                        }
                        if (!seen_stressed_vowel) continue; // skip first I after stress
                        // Only convert if directly before '@L' (syllabic L) - e.g. "chemical" IkL → ikL.
                        // Don't convert before 'k@L' (e.g. "political" tIk@L → ɪ stays as ɪ).
                        bool directly_before_al = (pi + 1 < ph.size() &&
                            ph.compare(pi+1, 2, "@L") == 0);
                        // Don't convert if 'I' is part of a diphthong (OI, aI, eI, etc.)
                        // i.e., the preceding character is a vowel.
                        static const std::string DIPH_BEFORE = "aAeEoOuU";
                        bool part_of_diph = (pi > 0 && DIPH_BEFORE.find(ph[pi-1]) != std::string::npos);
                        if (directly_before_al && !part_of_diph) ph[pi] = 'i';
                    }
                }
            }
        }
    }

    // 5c. American English: reduce 3: (long ɜː) to 3 (short ɚ) in pre-tonic position.
    // the reference rules emit 3: for "er" by default, but post-processing reduces it to 3
    // when the syllable comes BEFORE the primary stress (pre-tonic position).
    // Post-tonic 3: (after primary stress) keeps its full ɜː quality, even without a marker.
    // E.g. "conversation" → kɑːnvɜːsˈeɪʃən: "ver" is pre-tonic → ɚ ✓
    //      "expert" → ˈɛkspɜːt: "pert" is post-tonic → keeps ɜː ✓
    //      "diversity" → daɪvˈɜːsᵻɾi: "ver" has primary → keeps ɜː ✓
    // Guard: only run when there's a primary stress marker (') in the string.
    if (is_en_us) {
        size_t prime_pos = ph.find('\'');
        if (prime_pos != std::string::npos) {
            size_t pi = 0;
            while (pi + 1 < ph.size()) {
                if (ph[pi] == '3' && ph[pi+1] == ':') {
                    bool is_stressed = (pi > 0 && (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    // When '%' explicitly precedes '3:', the rule assigned ɜː quality to this
                    // unstressed syllable intentionally (e.g. _f)erme(nt %3:mE → "ferment" keeps ɜː).
                    // the reference does NOT reduce '%3:' to '3' in post-processing; only plain '3:' reduces.
                    bool has_explicit_unstress = (pi > 0 && ph[pi-1] == '%');
                    // Pre-tonic: 3: appears before the primary stress position.
                    // Use strict < so that '3:' immediately adjacent to '\'' is NOT reduced.
                    // E.g. "h3:'sElf" (herself): '3:' at pi=1, prime at 3 → pi+2=3 < 3 = false
                    // → keeps ɜː (matches the reference). "p3:h'Aps" (perhaps): prime at 4 → 3 < 4 = true
                    // → reduces to ɚ (correct). When '\'' immediately follows '3:', the vowel
                    // in "3:" is in effect adjacent to the stressed syllable; the reference keeps ɜː there.
                    bool is_pretonic = (pi + 2 < prime_pos);
                    // Post-tonic with secondary stress after: '3:' between primary and secondary
                    // stress reduces to '3' (ɚ). E.g. "metallurgist" m%Et'al3:dZ,Ist → ɚ.
                    // But "metallurgy" m%Et'al3:dZ%i (no ',' after) → keeps ɜː.
                    bool has_secondary_after = (ph.find(',', pi + 2) != std::string::npos);
                    // When '%' immediately precedes the primary stress marker (e.g. %'Int3:nS,Ip for
                    // "internship"), the stress was forced by applyStressPosition and the secondary
                    // after it was added by step 5a — not an organic post-tonic position. In that
                    // case, has_secondary_after should NOT trigger ɜː→ɚ reduction.
                    bool pct_before_primary = (prime_pos > 0 && ph[prime_pos - 1] == '%');
                    // Only reduce '3:' when there is another vowel BEFORE it in the phoneme.
                    // If '3:' is the FIRST vowel (e.g. "personify" p3:s'0n..., "persona" p3:s'oUn@),
                    // the reference keeps ɜː. "conversation" (k%0#nv3:...) has '0' before '3:' → reduces.
                    static const std::string STEP5C_VOWELS = "aAeEiIoOuUV03@";
                    bool has_vowel_before = false;
                    for (size_t k = 0; k < pi && !has_vowel_before; k++)
                        if (STEP5C_VOWELS.find(ph[k]) != std::string::npos) has_vowel_before = true;
                    if (!is_stressed && !has_explicit_unstress && has_vowel_before &&
                        (is_pretonic || (has_secondary_after && !pct_before_primary))) {
                        ph.erase(pi+1, 1);  // remove ':' → '3:' → '3'
                        // Adjust prime_pos since we erased a char before it
                        if (prime_pos > 0) prime_pos--;
                        continue;
                    }
                    pi += 2;
                } else {
                    pi++;
                }
            }
        }
    }

    // 6. American English flap rule: /t/ → [ɾ] between a vowel and unstressed vowel.
    if (is_en_us) {
        // De-tap: *3n/m/N → t3n/m/N ("pattern", "western"-like words).
        // the reference rules produce '*' (tap) before '3' (ɚ) + nasal, but renders as plain 't'.
        for (size_t pi = 0; pi + 2 < ph.size(); pi++) {
            if (ph[pi] == '*' && ph[pi+1] == '3' &&
                (ph[pi+2] == 'n' || ph[pi+2] == 'm' || ph[pi+2] == 'N'))
                ph[pi] = 't';
        }

        static const std::string VOWEL_CHARS = "aAeEIiOUVu03@oY";
        for (size_t pi = 1; pi + 1 < ph.size(); pi++) {
            if (ph[pi] == 't') {
                char prev = ph[pi-1];
                // Check if preceding character is a vowel (or ':' length mark).
                // '#' suffix (as in I#, a#, E#) marks a vowel variant, so treat it as vowel-preceded.
                // '2' suffix in multi-char vowel codes (I2, E2, 02, O2) counts as vowel.
                // Check both prev directly AND prev='2' with a vowel two positions back.
                bool prev_vowel = (prev == ':' || prev == '#' || prev == 'r' ||
                                   VOWEL_CHARS.find(prev) != std::string::npos ||
                                   (prev == '2' && pi >= 2 &&
                                    VOWEL_CHARS.find(ph[pi-2]) != std::string::npos) ||
                                   // Syllabic-L (@L) acts as vowel for flapping: "loyalty" @L+t+i → ɾ
                                   (prev == 'L' && pi >= 2 && ph[pi-2] == '@'));
                if (!prev_vowel) continue;
                // 't#' is the the reference "flappable-t" 2-char phoneme code.
                // When the char after 't' is '#', skip past it to find the true next phoneme.
                size_t nxt_pos = (ph[pi+1] == '#') ? pi + 2 : pi + 1;
                char nxt = (nxt_pos < ph.size()) ? ph[nxt_pos] : 0;
                bool next_unstressed_vowel = false;
                if (nxt == '%' || nxt == '=') {
                    // Unstressed prefix ('%' or '=') followed by a vowel → flap target.
                    // Apply next2PhW(n) block: don't flap if the vowel is followed by 'n'.
                    if (nxt_pos + 1 < ph.size() && VOWEL_CHARS.find(ph[nxt_pos+1]) != std::string::npos) {
                        bool v2_long = (nxt_pos + 2 < ph.size() && ph[nxt_pos+2] == ':');
                        int n2p = v2_long ? (int)(nxt_pos + 3) : (int)(nxt_pos + 2);
                        bool n2_n = (n2p < (int)ph.size() && ph[n2p] == 'n');
                        bool v2_3c = (ph[nxt_pos+1] == '3' && v2_long);
                        next_unstressed_vowel = !(n2_n && !v2_3c);
                    }
                } else if (nxt != '\'' && nxt != ',' &&
                           VOWEL_CHARS.find(nxt) != std::string::npos) {
                    // the reference rule: ChangePhoneme(t#) when prevPhW(isVowel) AND nextPhW(isVowel)
                    // AND nextPh(isUnstressed) AND (NOT next2PhW(n) OR nextPhW(3:)).
                    // next2PhW(n): phoneme after the following vowel is alveolar 'n'.
                    // nextPhW(3:): following vowel is '3:' (ɜː) — this exception allows flap
                    //   e.g. "saturn" (s'at3:n) flaps, but "pattern" (p'at3n) does not.
                    bool is_3colon = (nxt == '3' && nxt_pos + 1 < ph.size() && ph[nxt_pos+1] == ':');
                    bool vowel_is_long = (nxt_pos + 1 < ph.size() && ph[nxt_pos+1] == ':');
                    int next2_pos = vowel_is_long ? (int)(nxt_pos + 2) : (int)(nxt_pos + 1);
                    // '@L' (syllabic L) is a single phoneme: skip the 'L' modifier to find
                    // the next real phoneme. E.g. "bottleneck" has t@Ln → skip L → see n →
                    // no flap. But "bottle" has t@L(end) → no n → flap fires.
                    if (nxt == '@' && next2_pos < (int)ph.size() && ph[next2_pos] == 'L')
                        next2_pos++;
                    bool next2_is_n = (next2_pos < (int)ph.size() && ph[next2_pos] == 'n');
                    // No flap when: next vowel followed by 'n' AND NOT the 3: exception.
                    next_unstressed_vowel = !(next2_is_n && !is_3colon);
                }
                if (next_unstressed_vowel) ph[pi] = '*';
            }
        }
    }

    // 6b. '-ness' suffix reduction: 'nEs' at word-end → 'n@s' (schwa, not ɛ).
    // the reference reduces the unstressed 'e' in "-ness" to schwa in American English.
    // Also handles 'n,Es' (spurious secondary stress on -ness → remove and use schwa).
    if (is_en_us) {
        // Pattern: optional-stress-marker + 'Es' at end, preceded by 'n'
        size_t plen = ph.size();
        if (plen >= 4 && ph[plen-1] == 's' && ph[plen-2] == 'E' &&
            ph[plen-3] == ',' && ph[plen-4] == 'n') {
            // n,Es → n@s
            ph.replace(plen-3, 3, "@s");
        } else if (plen >= 3 && ph[plen-1] == 's' && ph[plen-2] == 'E' &&
                   ph[plen-3] == 'n') {
            // nEs → n@s
            ph.replace(plen-2, 2, "@s");
        }
    }


    // 6c. Reduce 'oU#' (compound-prefix vowel) based on adjacent stress context.
    // In the reference, phoneme 'oU#' is defined as:
    //   IF thisPh(isStressed) → ChangePhoneme(0)     [if this vowel is primary-stressed → ɑː]
    //   IF nextVowel(isStressed) OR prevVowel(isStressed) → ChangePhoneme(@) [if adj primary-stressed vowel → @]
    //   Default → ChangePhoneme(oU)                   [keep as oʊ]
    // 'isStressed' in the reference = PRIMARY stress only (stress_level > STRESS_IS_SECONDARY, i.e., >= 4).
    // e.g. "cryptocurrency" kr'IptoU# → 'I is primary prev vowel → oU# → @
    // e.g. "Gastrointestinal" g,astroU#Int'Est → a is secondary → oU# stays oU
    if (is_en_us) {
        for (size_t i = 0; i + 2 < ph.size(); i++) {
            if (ph[i] == 'o' && ph[i+1] == 'U' && ph[i+2] == '#') {
                // Check if oU# itself is primary stressed (directly preceded by ')
                bool self_primary = (i > 0 && ph[i-1] == '\'');
                if (self_primary) {
                    ph.replace(i, 3, "0");
                    continue;
                }
                // Check prevVowel(isStressed): scan backward; if we encounter '\'' before finding
                // a non-primary stress separator (',', '%', '='), prev is primary-stressed.
                bool prev_primary = false;
                for (int j = (int)i - 1; j >= 0; j--) {
                    char c = ph[j];
                    if (c == '\'') { prev_primary = true; break; }
                    if (c == ',' || c == '%') break;
                }
                if (prev_primary) {
                    ph.replace(i, 3, "@");
                    continue;
                }
                // Check nextVowel(isStressed): scan forward for the nearest vowel.
                // If we find '\'' before the first vowel char, it's primarily stressed.
                // If we find a vowel char before '\'' (or find ','  first), it's not primary.
                bool next_primary = false;
                {
                    static const std::string VOWEL_FWD = "aAeEiIoOuUV03@";
                    for (size_t j = i + 3; j < ph.size(); j++) {
                        char c = ph[j];
                        if (c == '\'') { next_primary = true; break; }
                        if (c == ',') break;
                        if (VOWEL_FWD.find(c) != std::string::npos) break; // vowel without primary
                    }
                }
                if (next_primary) {
                    ph.replace(i, 3, "@");
                    continue;
                }
                // Default: keep as oU (remove #)
                ph.replace(i, 3, "oU");
            }
        }
    }

    // 6d. DISABLED: Unstressed rhotic FORCE reduction was incorrect.
    // Words like "encore"/"offshore" get o@ from rules and should keep oːɹ (FORCE vowel),
    // not reduce to ɚ. Words like "honor"/"actor"/"doctor" already emit %3 directly from
    // rules, so they don't need this step. Step 6d was firing incorrectly for "encore" etc.

    // 6e. '-ically' schwa elision: word-final 'k@li' → 'kli'.
    // The schwa in '-ical' is elided before '-ly': "typically"→tˈɪpɪkli, "basically"→bˈeɪsɪkli,
    // "historically"→hɪstˈɔːɹɪkli, etc.
    if (is_en_us && ph.size() >= 4 &&
        ph[ph.size()-4]=='k' && ph[ph.size()-3]=='@' &&
        ph[ph.size()-2]=='l' && ph[ph.size()-1]=='i') {
        ph.erase(ph.size()-3, 1); // remove '@': 'k@li' → 'kli'
    }

    // 6f. Secondary stress on syllabic-n in compound words.
    // E.g. "handwritten" → h'andrI?n- → add ˌ before n- (the reference: hˈændɹɪʔˌn̩).
    // Condition: phoneme ends in 'n-' (possibly preceded by '?' for glottal stop),
    // primary stress '\'' is already present, AND there are 2+ vowel groups before the
    // syllabic consonant. Simple 2-syllable words like "written"/"kitten" have only 1
    // vowel group (ri/kI) and don't get secondary stress. Compounds like "handwritten"
    // have 2+ vowel groups ('a','I') and do.
    {
        if (std::getenv("PHON_DEBUG")) std::cerr << "[6f] ph='" << ph << "' back=" << (ph.size()>=2 ? std::string(1,ph[ph.size()-2])+""+std::string(1,ph[ph.size()-1]) : "") << "\n";
        static const std::string VC_SN = "aAeEiIoOuUV03@";
        if (ph.size() >= 2 && ph[ph.size()-1] == '-' && ph[ph.size()-2] == 'n' &&
            ph.find('\'') != std::string::npos) {
            // Find start of the syllabic block ('?n-' or just 'n-')
            size_t sn_start = ph.size() - 2;
            if (sn_start > 0 && ph[sn_start-1] == '?') sn_start--;
            // Count vowel groups before sn_start
            int vgroups = 0;
            bool in_v = false;
            for (size_t vi = 0; vi < sn_start; vi++) {
                char vc = ph[vi];
                if (vc == '\'' || vc == ',' || vc == '%' || vc == '=') continue;
                if (VC_SN.find(vc) != std::string::npos) {
                    if (!in_v) { vgroups++; in_v = true; }
                } else {
                    in_v = false;
                }
            }
            // Only fire for true compound words (e.g. "handwritten", "typewritten").
            // A compound has no unstressed ('%'/'=') marker before the primary stress '\'' —
            // that would indicate an unstressed prefix (e.g. "un-", "en-") rather than a
            // stressed compound first-element. Check: scan before first '\'' for '%'/'='.
            bool has_unstressed_prefix = false;
            {
                size_t prime = ph.find('\'');
                if (prime != std::string::npos) {
                    for (size_t ui = 0; ui < prime; ui++) {
                        if (ph[ui] == '%' || ph[ui] == '=') { has_unstressed_prefix = true; break; }
                    }
                }
            }
            // Add secondary stress if 2+ vowel groups, no existing stress before n-,
            // and this is a true compound (no unstressed prefix).
            if (vgroups >= 2 && !has_unstressed_prefix &&
                (sn_start == 0 || ph[sn_start-1] != ',')) {
                ph.insert(sn_start, ",");
            }
        }
    }

    return ph;
}

// ============================================================
// Phoneme code string → IPA
// ============================================================
bool IPAPhonemizer::isVowelCode(const std::string& code) const {
    if (code.empty()) return false;
    char c = code[0];
    return c=='@'||c=='a'||c=='A'||c=='E'||c=='I'||c=='i'||c=='O'||c=='U'||c=='V'||
           c=='0'||c=='3'||c=='e'||c=='o'||c=='u';
}

std::string IPAPhonemizer::singleCodeToIPA(const std::string& code) const {
    if (code.empty()) return "";

    // Stress markers
    if (code == "'")  return "\xcb\x88"; // ˈ primary stress
    if (code == ",")  return "\xcb\x8c"; // ˌ secondary stress
    if (code == "%")  return "";          // unstressed - no marker
    if (code == "=")  return "";          // unstressed - no marker
    if (code == "==") return "";
    if (code == "|")  return "";          // syllable boundary
    if (code == "||") return "";          // word end

    // Check IPA overrides table
    auto it = ipa_overrides_.find(code);
    if (it != ipa_overrides_.end()) {
        return it->second;
    }

    // Use ipa1 table conversion
    bool is_vowel = isVowelCode(code);
    return phonemeCodeToIPA_table(code, is_vowel);
}

std::string IPAPhonemizer::phonemesToIPA(const std::string& phoneme_str) const {
    std::string result;
    if (phoneme_str.empty()) return result;

    // Remove trailing $ flags
    std::string pstr = phoneme_str;
    size_t dollar = pstr.find('$');
    if (dollar != std::string::npos)
        pstr = trim(pstr.substr(0, dollar));

    // Known multi-char phoneme codes (try in order - longer first)
    // We use a greedy matching approach
    static const char* MULTI_CODES[] = {
        // 4-char
        "aI@3", "aU@r", "i@3r",  // i@3r: absorbs trailing r after i@3 (ɪɹ) to avoid double-ɹ
        // 3-char (must come before 2-char prefixes)
        "aI@", "aI3", "aU@", "i@3", "3:r", "A:r",
        "I2#",  // ᵻ — must precede "I2" and "I#" to parse correctly (e.g. "l3:nI2#d")
        "o@r",  // rhotic o + explicit r (e.g. "o@ri" → oːɹi, avoids double ɹ)
        "O@r",  // ɔːɹ + explicit r (avoids double ɹ in words like "warring")
        "e@r",  // ɛɹ + explicit r (avoids double ɹ in words like "extraordinary")
        // Diphthongs (2-char)
        "eI", "aI", "aU", "OI", "oU", "tS", "dZ", "IR", "VR",
        "e@", "i@", "U@", "A@", "O@", "o@",
        // Long vowels
        "3:", "A:", "i:", "u:", "O:", "e:", "a:",
        // Double-letter vowels (the reference extended phoneme codes, e.g. aas/aan/aaf rules)
        "aa",
        // Schwa variants
        "@L", "@2", "@5",
        // Consonant variants
        "r-", "w#", "t#", "d#", "z#", "t2", "d2", "n-", "m-",
        "l/", "z/",
        // Vowel variants (2-char)
        "I2", "I#", "E2", "E#", "e#", "a#", "a2", "0#", "02", "O2",
        "A~", "O~", "A#",
        nullptr
    };

    int i = 0;
    int len = (int)pstr.size();
    std::string pending_stress;  // deferred stress marker (placed before vowel)
    bool last_was_unstress = false; // set after % or = to prevent diphthong matching
    bool last_code_was_vowel = false; // track whether last emitted code was a vowel (for ';' handling)

    while (i < len) {
        unsigned char c = (unsigned char)pstr[i];

        // Skip whitespace
        if (std::isspace(c)) { i++; continue; }

        // Separator chars
        if (pstr[i] == '|') { i++; continue; }
        if (pstr[i] == '-') { i++; continue; }  // syllable boundary marker, skip always
        if (pstr[i] == ';') {
            // Palatalization modifier: ';' after a consonant → output ʲ; after a vowel → skip.
            // e.g. "n;oU" (jalapeno) → nʲoʊ; "i;'0m" (geometry) → iˈɑːm (no ʲ).
            if (!last_code_was_vowel)
                result += "\xca\xb2"; // ʲ (U+02B2)
            i++;
            continue;
        }

        // Stress markers: defer until just before the next vowel
        if (pstr[i] == '\'' || pstr[i] == ',') {
            pending_stress = singleCodeToIPA(std::string(1, pstr[i]));
            last_was_unstress = false;
            i++;
            continue;
        }
        // Unstressed/syllable markers: clear pending stress, set flag
        if (pstr[i] == '%' || pstr[i] == '=') {
            pending_stress.clear();
            last_was_unstress = true; // next code is a single phoneme, not start of diphthong
            i++;
            continue;
        }

        // Determine what code we're about to emit
        std::string code;
        bool found = false;
        for (int mi = 0; MULTI_CODES[mi] != nullptr; mi++) {
            const char* mc = MULTI_CODES[mi];
            int mclen = (int)strlen(mc);
            // After an unstress marker (= or %), the next phoneme is a standalone phoneme,
            // not the start of a diphthong. Skip diphthong codes to prevent e.g. =i@n
            // being parsed as ɪə (diphthong i@) + n instead of i (close front) + @ + n.
            if (last_was_unstress && mclen == 2) {
                // After % or =, prevent splitting of vowel+schwa combinations that
                // should be read as separate vowels, e.g. =i@n → i + @n (not ɪə + n).
                // True diphthongs like aU, aI, oU, OI are NOT blocked — they are always
                // single phoneme units even when unstressed (e.g. %aUt = unstressed aʊt).
                // Consonant digraphs (tS, dZ) are also never blocked.
                static const char* DIPHTHONGS[] = {
                    "i@","U@", nullptr
                };
                bool is_diph = false;
                for (int di = 0; DIPHTHONGS[di]; di++) {
                    if (strcmp(mc, DIPHTHONGS[di]) == 0) { is_diph = true; break; }
                }
                if (is_diph) continue;
            }
            if (i + mclen <= len) {
                bool match = true;
                for (int j = 0; j < mclen; j++) {
                    if (pstr[i+j] != mc[j]) { match = false; break; }
                }
                if (match) {
                    code = std::string(mc, mclen);
                    i += mclen;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            code = std::string(1, (char)pstr[i]);
            i++;
        }
        last_was_unstress = false;

        // Skip variant-marker digits that follow a code (e.g. '2' in "a#2n").
        // A digit is a variant marker if its ipa1 mapping is itself (not a real IPA char).
        // Real phoneme digits: '0'→ɒ, '3'→ɜ, '8'→ɵ — do NOT skip those.
        while (i < len && pstr[i] >= '0' && pstr[i] <= '9') {
            unsigned char dc = (unsigned char)pstr[i];
            uint32_t mapped = ASCII_TO_IPA[dc - 0x20];
            if (mapped == dc) { i++; } // variant marker digit — skip
            else break;               // real IPA phoneme digit — leave for next iteration
        }

        // Emit pending stress before the next vowel or syllabic consonant (nucleus).
        // Syllabic consonants ('n-', 'm-', '@L', 'r-', 'l/') act as syllable nuclei
        // and therefore trigger stress emission, even though they are not vowels.
        // This handles e.g. "h'andrI,?n-" → hˈændɹɪʔˌn̩ (ˌ before n̩, after ʔ onset).
        bool is_syllabic = (code == "n-" || code == "m-" || code == "@L" ||
                            code == "r-" || code == "l/");
        if (!pending_stress.empty() && (isVowelCode(code) || is_syllabic)) {
            result += pending_stress;
            pending_stress.clear();
        }

        result += singleCodeToIPA(code);
        last_code_was_vowel = isVowelCode(code);
    }

    // Note: any remaining pending_stress with no following vowel is intentionally discarded.
    // This allows dict entries to use a trailing ',' as a "no-secondary-stress" marker
    // that prevents step 5a from adding automatic secondary stress (since 5a checks ph_in for ',').

    return result;
}

// ============================================================
// Add stress markers to IPA when none are present
// ============================================================
// Simple stress placement: add primary stress ˈ before the first vowel
// if no stress markers are present. This is a simplification.
static bool containsStressMarker(const std::string& ipa) {
    // Check for ˈ (0xCB 0x88) or ˌ (0xCB 0x8C)
    for (size_t i = 0; i + 1 < ipa.size(); i++) {
        if ((unsigned char)ipa[i] == 0xCB &&
            ((unsigned char)ipa[i+1] == 0x88 || (unsigned char)ipa[i+1] == 0x8C))
            return true;
    }
    return false;
}

// IPA vowel character detection (UTF-8)
// Only returns true for actual vowel characters (not consonants like ɹ, ɡ, etc.)
static bool isIPAVowelStart(const std::string& s, size_t i) {
    if (i >= s.size()) return false;
    unsigned char c = (unsigned char)s[i];
    // ASCII vowels: a, e, i, o, u
    if (c < 0x80) {
        char lc = (char)std::tolower(c);
        return lc=='a'||lc=='e'||lc=='i'||lc=='o'||lc=='u';
    }
    if (i + 1 >= s.size()) return false;
    unsigned char c2 = (unsigned char)s[i+1];

    if (c == 0xC3) {
        // æ=0xA6, ø=0xB8, œ=0x93
        return c2==0xA6||c2==0xB8||c2==0x93;
    }
    if (c == 0xC9) {
        // IPA vowels in U+024X-U+02BX range (second byte 0x90-0xBF):
        // ɐ=0x90, ɑ=0x91, ɒ=0x92, ɔ=0x94, ɕ=0x95(not vowel),
        // ə=0x99, ɚ=0x9A, ɛ=0x9B, ɜ=0x9C, ɞ=0x9E,
        // ɪ=0xAA, ɵ=0xB5
        // NON-vowels: ɡ=0xA1, ɟ=0x9F, ɹ=0xB9, ɸ=0xB8, etc.
        switch (c2) {
            case 0x90: // ɐ
            case 0x91: // ɑ
            case 0x92: // ɒ
            case 0x94: // ɔ
            case 0x99: // ə
            case 0x9A: // ɚ
            case 0x9B: // ɛ
            case 0x9C: // ɜ
            case 0x9E: // ɞ
            case 0xAA: // ɪ
            case 0xB5: // ɵ
                return true;
            default:
                return false;
        }
    }
    if (c == 0xCA) {
        // ʊ=0x8A, ʌ=0x8C
        return c2==0x8A||c2==0x8C;
    }
    if (c == 0xE1) {
        // ᵻ = U+1D7B = 0xE1 0xB5 0xBB
        if (i+2 < s.size()) {
            unsigned char c3 = (unsigned char)s[i+2];
            return c2==0xB5 && c3==0xBB;
        }
    }
    return false;
}

// Check if the IPA character at i is schwa (ə, U+0259 = 0xC9 0x99)
static bool isIPASchwa(const std::string& s, size_t i) {
    if (i + 1 >= s.size()) return false;
    return (unsigned char)s[i] == 0xC9 && (unsigned char)s[i+1] == 0x99;
}

static std::string addDefaultStress(const std::string& ipa) {
    if (ipa.empty() || containsStressMarker(ipa)) return ipa;

    // Find position of first vowel and insert ˈ before it
    // If first vowel is schwa (ə) AND there's a non-schwa vowel after it,
    // skip the initial schwa and stress the next vowel instead.
    // (Handles cases like "hello" h@loU → həlˈoʊ, "about" → əbˈaʊt)
    // ˈ = 0xCB 0x88
    std::string stress = "\xcb\x88";

    // Find first vowel position
    size_t first_vowel = std::string::npos;
    bool first_is_schwa = false;
    for (size_t i = 0; i < ipa.size(); ) {
        if (isIPAVowelStart(ipa, i)) {
            first_vowel = i;
            first_is_schwa = isIPASchwa(ipa, i);
            break;
        }
        unsigned char c = (unsigned char)ipa[i];
        if (c < 0x80) i++;
        else if (c < 0xE0) i += 2;
        else if (c < 0xF0) i += 3;
        else i += 4;
    }

    if (first_vowel == std::string::npos) return ipa; // no vowel found

    if (!first_is_schwa) {
        // Stress the first vowel directly
        return ipa.substr(0, first_vowel) + stress + ipa.substr(first_vowel);
    }

    // First vowel is schwa: look for a subsequent non-schwa vowel
    size_t schwa_end = first_vowel + 2; // ə is 2 bytes
    for (size_t i = schwa_end; i < ipa.size(); ) {
        if (isIPAVowelStart(ipa, i) && !isIPASchwa(ipa, i)) {
            // Found a non-schwa vowel after initial schwa - stress this one
            return ipa.substr(0, i) + stress + ipa.substr(i);
        }
        unsigned char c = (unsigned char)ipa[i];
        if (c < 0x80) i++;
        else if (c < 0xE0) i += 2;
        else if (c < 0xF0) i += 3;
        else i += 4;
    }

    // No non-schwa vowel found after initial schwa - stress the schwa itself
    for (size_t i = 0; i < ipa.size(); ) {
        if (isIPAVowelStart(ipa, i)) {
            return ipa.substr(0, i) + stress + ipa.substr(i);
        }
        unsigned char c = (unsigned char)ipa[i];
        if (c < 0x80) i++;
        else if (c < 0xE0) i += 2;
        else if (c < 0xF0) i += 3;
        else i += 4;
    }
    return ipa; // no vowel found, return as-is
}

// ============================================================
// Text tokenization
// ============================================================
std::vector<IPAPhonemizer::Token> IPAPhonemizer::tokenizeText(const std::string& text) const {
    std::vector<Token> tokens;
    std::string current_word;

    auto flush_word = [&]() {
        if (!current_word.empty()) {
            Token t;
            t.text = current_word;
            t.is_word = true;
            t.needs_space_before = !tokens.empty();
            tokens.push_back(t);
            current_word.clear();
        }
    };

    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];

        // Handle UTF-8 multi-byte chars
        if (c >= 0x80) {
            current_word += text[i++];
            while (i < text.size() && ((unsigned char)text[i] & 0xC0) == 0x80)
                current_word += text[i++];
            continue;
        }

        if (std::isalpha(c) || c == '\'') {
            current_word += (char)c;
            i++;
        } else if (c == '.' && current_word.size() == 1 && std::isupper((unsigned char)current_word[0])
                   && i+1 < text.size() && std::isupper((unsigned char)text[i+1])) {
            // Abbreviation detection: single uppercase letter followed by '.' and next char is uppercase.
            // E.g. "U.S." → treat as single abbreviation token by including the period in the word.
            // We include the period so the word is "U." and then we continue accumulating.
            current_word += '.';
            i++;
        } else if (c == '.' && !current_word.empty()) {
            // Check if current word looks like an abbreviation in progress (X. pattern):
            // If next char is uppercase or we already accumulated multiple letters with periods,
            // continue; otherwise flush.
            bool is_abbrev = false;
            // If current_word contains periods (like "U."), it's an abbreviation in progress.
            if (current_word.find('.') != std::string::npos) {
                is_abbrev = true; // already accumulating abbreviation like "U.S"
            }
            if (is_abbrev) {
                current_word += '.'; // include trailing period
                i++;
                // Flush the abbreviation as a word token
                flush_word();
            } else {
                flush_word();
                Token t;
                t.text = std::string(1, (char)c);
                t.is_word = false;
                t.needs_space_before = false;
                tokens.push_back(t);
                i++;
            }
        } else if (c == '-' && !current_word.empty() && i+1 < text.size() && std::isalpha(text[i+1])) {
            current_word += (char)c;
            i++;
        } else if (std::isdigit(c)) {
            current_word += (char)c;
            i++;
        } else if (std::isspace(c)) {
            flush_word();
            i++;
        } else {
            flush_word();
            Token t;
            t.text = std::string(1, (char)c);
            t.is_word = false;
            t.needs_space_before = false;
            tokens.push_back(t);
            i++;
        }
    }
    flush_word();
    return tokens;
}

// ============================================================
// Number-to-words conversion (cardinal, American English, the reference style)
// ============================================================
static std::string intToWords(int n) {
    static const char* ones[] = {
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    };
    static const char* tens_words[] = {
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"
    };
    if (n == 0) return "zero";
    std::string result;
    if (n >= 1000000) {
        result += intToWords(n / 1000000) + " million";
        n %= 1000000;
        if (n > 0) result += " ";
    }
    if (n >= 1000) {
        result += intToWords(n / 1000) + " thousand";
        n %= 1000;
        if (n > 0) result += " ";
    }
    if (n >= 100) {
        result += std::string(ones[n / 100]) + " hundred";
        n %= 100;
        if (n > 0) result += " ";
    }
    if (n >= 20) {
        result += tens_words[n / 10];
        n %= 10;
        if (n > 0) result += " " + std::string(ones[n]);
    } else if (n > 0) {
        result += ones[n];
    }
    return result;
}

// ============================================================
// Main phonemize functions
// ============================================================
std::string IPAPhonemizer::phonemizeText(const std::string& text) const {
    if (!loaded_) return "";

    bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");
    auto tokens = tokenizeText(text);
    std::string result;
    bool first_word = true;

    // Count word tokens to detect single-word isolation and find the last word index.
    int word_token_count = 0;
    size_t last_word_ti = std::string::npos;
    for (size_t tii = 0; tii < tokens.size(); tii++)
        if (tokens[tii].is_word) { word_token_count++; last_word_ti = tii; }
    bool is_isolated_word = (word_token_count == 1);

    // POS context counters (cf. the reference tr->expect_past, tr->expect_noun, tr->expect_verb).
    // Set by $pastf/$nounf/$verbf words, decremented each word; control alt pronunciations.
    int expect_past = 0;  // >0: next word(s) may use their $past pronunciation
    int expect_noun = 0;  // >0: next word(s) may use their $noun pronunciation
    int expect_verb = 0;  // >0: next word(s) may use their $verb pronunciation

    // Phrase (bigram) lookup table: "word1 word2" → phoneme code string
    // These correspond to the reference en_list phrase entries like "(has been) %ha#z%bIn $pastf"
    // The output IPA needs to look like a single word with primary stress.
    static const std::unordered_map<std::string, std::string> PHRASE_DICT = {
        {"has been", "h'azbi:n"},  // hˈæzbiːn - the reference contracts "has been" to single token
    };

    // Direct IPA cliticization table: "word1 word2" → merged IPA (direct output, no phoneme codes).
    // the reference merges these function-word pairs in normal speech rhythm.
    // Using UTF-8 strings directly for IPA characters.
    static const std::unordered_map<std::string, std::string> CLITIC_IPA = {
        // "of a" → əvə (schwa+v+schwa) - the reference always merges
        {"of a",    "\xc9\x99v\xc9\x99"},                      // əvə
        // "of the" → ʌvðə (ʌ+v+ð+schwa)
        {"of the",  "\xca\x8cv\xc3\xb0\xc9\x99"},              // ʌvðə
        // "in the" → ɪnðə
        {"in the",  "\xc9\xaan\xc3\xb0\xc9\x99"},              // ɪnðə
        // "on the" → ɔnðə
        {"on the",  "\xc9\x94n\xc3\xb0\xc9\x99"},              // ɔnðə
        // "from the" → fɹʌmðə
        {"from the","\x66\xc9\xb9\xca\x8cm\xc3\xb0\xc9\x99"}, // fɹʌmðə
        // "that a" → ðˌæɾə (secondary stress before æ, flap ɾ)
        {"that a",  "\xc3\xb0\xcb\x8c\xc3\xa6\xc9\xbe\xc9\x99"}, // ðˌæɾə
        // "I am" → aɪɐm
        {"i am",    "a\xc9\xaa\xc9\x90m"},                      // aɪɐm
        // "was a" → wʌzɐ (the reference merges in sentence context)
        {"was a",   "w\xca\x8cz\xc9\x90"},                      // wʌzɐ
        // "to be" → təbi (the reference merges infinitive marker+be)
        {"to be",   "t\xc9\x99" "bi"},                           // təbi
        // "out of" → ˌaʊɾəv (the reference merges with flap and secondary stress)
        {"out of",  "\xcb\x8c" "a\xca\x8a\xc9\xbe\xc9\x99v"},   // ˌaʊɾəv
    };

    for (size_t ti = 0; ti < tokens.size(); ti++) {
        const auto& token = tokens[ti];
        if (token.is_word) {

            // Number expansion: convert pure-digit tokens to spoken word form.
            // e.g., "16" → "sixteen", "20" → "twenty", "2011" → "two thousand eleven"
            {
                bool all_digits = !token.text.empty();
                for (char c : token.text)
                    if (!std::isdigit((unsigned char)c)) { all_digits = false; break; }
                if (all_digits) {
                    long long num_val = std::stoll(token.text);
                    if (num_val >= 0 && num_val <= 9999999LL) {
                        std::string num_words = intToWords((int)num_val);
                        std::istringstream iss(num_words);
                        std::string w;
                        bool is_first_sub = true;
                        while (iss >> w) {
                            std::string wph = wordToPhonemes(w);
                            std::string wipa = phonemesToIPA(wph);
                            wipa = addDefaultStress(wipa);
                            if (is_first_sub) {
                                if (token.needs_space_before && !first_word) result += ' ';
                                is_first_sub = false;
                            } else {
                                result += ' ';
                            }
                            result += wipa;
                            first_word = false;
                        }
                        continue;
                    }
                }
            }

            // All-caps acronym: read as individual letter names if marked $abbrev in dict.
            // the reference reads "GI" as G+I letter names when the lowercase "gi" has $abbrev flag.
            // This applies even when the lowercase word has a separate pronunciation entry.
            {
                std::string lower = toLowerASCII(token.text);
                bool all_upper = token.text.size() >= 2;
                for (char c : token.text)
                    if (!std::isupper((unsigned char)c)) { all_upper = false; break; }
                // Also spell out all-caps words whose lowercase form is not in the dictionary
                // (e.g. "DNA", "RNA", "FBI" → spelled as letters D-N-A, R-N-A, F-B-I).
                // the reference pronounces all-caps words as letter names unless the lowercase form
                // has a known pronunciation (e.g. "APPLE" → apple is in dict → normal).
                bool unknown_word = !dict_.count(lower);
                // Detect mixed-case abbreviations with no vowels (e.g. "PhD"):
                // the reference reads these as letter names (e.g. "pˌiːˈeɪtʃ dˈiː").
                // Detection: word has both upper and lower chars + no vowel letters + not in dict.
                // This covers "PhD", "McD", "BSc", etc. without catching normal capitalized words.
                bool mixed_case_no_vowel_abbrev = false;
                if (!all_upper && token.text.size() >= 2 && unknown_word) {
                    bool has_any_upper = false, has_any_lower = false, has_any_vowel = false;
                    for (char c : token.text) {
                        if (std::isupper((unsigned char)c)) has_any_upper = true;
                        else if (std::islower((unsigned char)c)) has_any_lower = true;
                        char lc = std::tolower((unsigned char)c);
                        if (lc=='a'||lc=='e'||lc=='i'||lc=='o'||lc=='u') has_any_vowel = true;
                    }
                    mixed_case_no_vowel_abbrev = has_any_upper && has_any_lower && !has_any_vowel;
                }
                if ((all_upper || mixed_case_no_vowel_abbrev) && (abbrev_words_.count(lower) > 0 || unknown_word)) {
                    // Read as individual letter names.
                    // For mixed-case abbreviations (e.g. "PhD"), split at lowercase→uppercase
                    // transitions first (camelCase split: "PhD"→["Ph","D"]).
                    // Each group becomes a separate output word (matching the reference behavior).
                    // All-uppercase words (e.g. "DNA") have no such transitions → single group.
                    std::vector<std::string> groups;
                    if (mixed_case_no_vowel_abbrev) {
                        std::string cur;
                        for (size_t ci = 0; ci < token.text.size(); ci++) {
                            if (ci > 0 && std::islower((unsigned char)token.text[ci-1])
                                       && std::isupper((unsigned char)token.text[ci])) {
                                if (!cur.empty()) groups.push_back(cur);
                                cur.clear();
                            }
                            cur += token.text[ci];
                        }
                        if (!cur.empty()) groups.push_back(cur);
                    } else {
                        groups.push_back(token.text);
                    }
                    // Helper: spell letters in a group as letter names, returning combined IPA.
                    auto spellGroup = [&](const std::string& grp) -> std::string {
                        std::vector<std::string> lipa;
                        for (char lc : grp) {
                            char lc_lower = std::tolower((unsigned char)lc);
                            std::string uk = std::string("_") + lc_lower;
                            auto uit = dict_.find(uk);
                            if (uit != dict_.end()) {
                                lipa.push_back(uit->second);
                            } else {
                                lipa.push_back(wordToPhonemes(std::string(1, lc)));
                            }
                        }
                        if (lipa.empty()) return "";
                        std::string codes;
                        for (size_t li = 0; li < lipa.size(); li++) {
                            std::string code = lipa[li];
                            if (li < lipa.size() - 1) {
                                // Make secondary stress
                                std::string mod; bool rep = false;
                                for (char c2 : code) {
                                    if ((c2 == '\'' || c2 == ',') && !rep) { mod += ','; rep = true; }
                                    else if (c2 != '\'' && c2 != ',') mod += c2;
                                }
                                if (!rep) mod = ',' + mod;
                                codes += mod;
                            } else {
                                // Primary stress on last letter
                                if (code.find('\'') == std::string::npos) codes += '\'' + code;
                                else codes += code;
                            }
                        }
                        // Apply processPhonemeString for linking-r etc.
                        return phonemesToIPA(processPhonemeString(codes));
                    };
                    bool first_grp = true;
                    for (const auto& grp : groups) {
                        std::string ipa = spellGroup(grp);
                        if (ipa.empty()) continue;
                        if (first_grp) {
                            if (token.needs_space_before && !first_word) result += ' ';
                        } else {
                            result += ' ';  // each camelCase group = separate word
                        }
                        result += ipa;
                        first_word = false;
                        first_grp = false;
                    }
                    continue;
                }
            }

            // Check for cliticization (bigram → direct IPA) FIRST.
            // These are function-word merges that the reference does consistently in natural speech.
            std::string ph_codes;
            bool phrase_matched = false;
            bool clitic_matched = false;
            bool phrase_pre_vowel_the = false; // phrase ends with ðə, needs ˈðɪ allophone
            std::string matched_phrase_key; // bigram key of the matched phrase dict entry
            {
                size_t tj = ti + 1;
                while (tj < tokens.size() && !tokens[tj].is_word) tj++;
                if (tj < tokens.size() && tokens[tj].is_word) {
                    std::string bigram = toLowerASCII(token.text) + " " + toLowerASCII(tokens[tj].text);
                    // Check direct IPA cliticization table first
                    auto cit = CLITIC_IPA.find(bigram);
                    if (cit != CLITIC_IPA.end()) {
                        if (token.needs_space_before && !first_word) result += ' ';
                        std::string clitic_ipa = cit->second;
                        // If the bigram ends with "the", check if the next word is
                        // vowel-initial: use ðɪ (U+026A = \xc9\xaa) instead of ðə (\xc9\x99).
                        if (toLowerASCII(tokens[tj].text) == "the") {
                            size_t tk = tj + 1;
                            while (tk < tokens.size() && !tokens[tk].is_word) tk++;
                            if (tk < tokens.size() && !tokens[tk].text.empty()) {
                                char fc = (char)std::tolower((unsigned char)tokens[tk].text[0]);
                                bool next_vowel = (fc=='a'||fc=='e'||fc=='i'||fc=='o'||fc=='u');
                                if (next_vowel) {
                                    // Check for /j/-onset (e.g. "university" → jˌuː): keep ðə
                                    bool j_onset = false;
                                    if (fc == 'u' || fc == 'e') {
                                        std::string nph = wordToPhonemes(toLowerASCII(tokens[tk].text));
                                        size_t npi = 0;
                                        while (npi < nph.size() && (nph[npi]=='\''||nph[npi]==','||nph[npi]=='%'||nph[npi]=='='))
                                            npi++;
                                        j_onset = (npi < nph.size() && nph[npi] == 'j');
                                    }
                                    if (!j_onset) {
                                        // Replace trailing ə (\xc9\x99) → ɪ (\xc9\xaa)
                                        if (clitic_ipa.size() >= 2 &&
                                            (unsigned char)clitic_ipa[clitic_ipa.size()-2] == 0xc9 &&
                                            (unsigned char)clitic_ipa[clitic_ipa.size()-1] == 0x99) {
                                            clitic_ipa[clitic_ipa.size()-1] = (char)0xaa;
                                        }
                                    }
                                }
                            }
                        }
                        result += clitic_ipa;
                        first_word = false;
                        ti = tj; // skip next word token
                        clitic_matched = true;
                    } else {
                        // Check phoneme code phrase dict (static hardcoded + loaded from en_list)
                        auto pit = PHRASE_DICT.find(bigram);
                        if (pit != PHRASE_DICT.end()) {
                            ph_codes = processPhonemeString(pit->second);
                            phrase_matched = true;
                            ti = tj; // skip next word token
                        } else {
                            auto pit2 = phrase_dict_.find(bigram);
                            if (pit2 != phrase_dict_.end()) {
                                std::string raw_phrase = pit2->second;
                                // Pre-vowel "the" allophone: if phrase ends with "D@2" (ðə)
                                // and next word token starts with a vowel, use "DI" (ðɪ) instead.
                                bool phrase_ends_the = (raw_phrase.size() >= 3 &&
                                    raw_phrase.compare(raw_phrase.size()-3, 3, "D@2") == 0);
                                if (phrase_ends_the) {
                                    // Check next word: if vowel-initial and not /j/-onset,
                                    // flag for IPA-level ðə → ˈðɪ substitution after phonemesToIPA.
                                    for (size_t tk = tj + 1; tk < tokens.size(); tk++) {
                                        if (tokens[tk].is_word && !tokens[tk].text.empty()) {
                                            char fc = (char)std::tolower((unsigned char)tokens[tk].text[0]);
                                            if (fc=='a'||fc=='e'||fc=='i'||fc=='o'||fc=='u') {
                                                bool j_onset = false;
                                                if (fc == 'u' || fc == 'e') {
                                                    std::string nph = wordToPhonemes(toLowerASCII(tokens[tk].text));
                                                    size_t npi = 0;
                                                    while (npi < nph.size() && (nph[npi]=='\''||nph[npi]==','||nph[npi]=='%'||nph[npi]=='='))
                                                        npi++;
                                                    j_onset = (npi < nph.size() && nph[npi] == 'j');
                                                }
                                                if (!j_onset)
                                                    phrase_pre_vowel_the = true;
                                            }
                                            break;
                                        }
                                    }
                                }
                                ph_codes = processPhonemeString(raw_phrase);
                                phrase_matched = true;
                                matched_phrase_key = bigram;
                                ti = tj; // skip next word token
                            }
                            // Check split-phrase dict: entries with || per-word phonemes
                            // (e.g. "(most of) moUst||@v", "(too much) t'u:||mVtS").
                            // Each word is emitted independently with appropriate stress.
                            if (!phrase_matched) {
                                auto psit = phrase_split_dict_.find(bigram);
                                if (psit != phrase_split_dict_.end()) {
                                    const std::string& ph1 = psit->second.first;
                                    const std::string& ph2 = psit->second.second;
                                    // Determine if the phrase has any explicit primary stress.
                                    bool phrase_has_primary =
                                        (ph1.find('\'') != std::string::npos ||
                                         ph2.find('\'') != std::string::npos);
                                    // Process each part:
                                    // - Parts with explicit stress ('/'): use processPhonemeString
                                    //   (preserves existing stress, applies vowel reductions).
                                    // - Parts WITHOUT explicit stress:
                                    //   * FIRST part with no phrase-primary: processPhonemeString
                                    //     adds default word-level stress (e.g. "most" → mˈoʊst).
                                    //   * All other unstressed parts: prepend '%' to prevent
                                    //     processPhonemeString from adding unwanted stress
                                    //     (e.g. "of" @v → %@v → əv, "much" mVtS → %mVtS → mʌtʃ).
                                    // Never call addDefaultStress on split-phrase parts.
                                    auto doSplitPart = [&](const std::string& ph, bool is_first) -> std::string {
                                        bool has_stress = (ph.find('\'') != std::string::npos ||
                                                           ph.find(',') != std::string::npos);
                                        std::string ph_proc = ph;
                                        if (!has_stress && !(is_first && !phrase_has_primary))
                                            ph_proc = "%" + ph; // prevent unwanted stress
                                        return phonemesToIPA(processPhonemeString(ph_proc));
                                    };
                                    // Emit word 1 (current token)
                                    std::string ipa1 = doSplitPart(ph1, true);
                                    if (token.needs_space_before && !first_word) result += ' ';
                                    result += ipa1;
                                    first_word = false;
                                    // Emit word 2 (next token tj)
                                    std::string ipa2 = doSplitPart(ph2, false);
                                    result += ' ';  // always space between split-phrase words
                                    result += ipa2;
                                    ti = tj; // skip next word token
                                    clitic_matched = true;
                                }
                            }
                        }
                    }
                }
            }
            if (clitic_matched) { continue; } else
            if (!phrase_matched) {
                // Handle abbreviations with periods (e.g. "U.S.", "U.K.", "N.Y.").
                // These are tokenized as single tokens but need letter-by-letter phonemization.
                // the reference renders "U.S." as one phoneme unit: jˌuːˈɛs (secondary+primary stress).
                if (token.text.find('.') != std::string::npos) {
                    // Collect the uppercase letters from the abbreviation token.
                    // "U.S." → letters {'U', 'S'}.
                    std::vector<std::string> letter_ipa;
                    for (size_t ci = 0; ci < token.text.size(); ci++) {
                        char lc = token.text[ci];
                        if (std::isalpha((unsigned char)lc)) {
                            // Try letter-name entry first (_X), else normal lookup.
                            char lc_lower = std::tolower((unsigned char)lc);
                            std::string underscore_key = std::string("_") + lc_lower;
                            auto uit = dict_.find(underscore_key);
                            if (uit != dict_.end()) {
                                letter_ipa.push_back(uit->second);
                            } else {
                                std::string lword(1, lc);
                                letter_ipa.push_back(wordToPhonemes(lword));
                            }
                        }
                    }
                    if (letter_ipa.size() >= 2) {
                        // Combine letters as a single phoneme unit with stress.
                        // letter_ipa contains raw phoneme CODES (e.g. "j'u:" for U, "'E2s" for S).
                        // Strategy: strip primary stress from all-but-last, make secondary;
                        //           keep primary on last letter. Then concatenate and convert to IPA.
                        // This preserves the reference convention of placing stress within the syllable
                        // (after onset consonants): j'u: → j,u: → IPA jˌuː.
                        std::string combined_codes;
                        for (size_t li = 0; li < letter_ipa.size(); li++) {
                            std::string code = letter_ipa[li];
                            if (li < letter_ipa.size() - 1) {
                                // Non-last: replace primary stress '\'' → secondary ','
                                // Also remove any existing ',' (avoid double secondary).
                                std::string modified;
                                bool replaced = false;
                                for (size_t ci = 0; ci < code.size(); ci++) {
                                    if (code[ci] == '\'') {
                                        if (!replaced) { modified += ','; replaced = true; }
                                    } else if (code[ci] == ',') {
                                        // skip existing secondary (will re-add once)
                                        if (!replaced) { modified += ','; replaced = true; }
                                    } else {
                                        modified += code[ci];
                                    }
                                }
                                if (!replaced) modified = ',' + modified; // no stress found: add secondary
                                combined_codes += modified;
                            } else {
                                // Last: keep as primary stress (ensure ' exists)
                                if (code.find('\'') == std::string::npos) {
                                    combined_codes += '\'' + code;
                                } else {
                                    combined_codes += code;
                                }
                            }
                        }
                        // Convert combined phoneme codes to IPA
                        std::string combined_ipa = phonemesToIPA(combined_codes);
                        // Output directly (already IPA, bypass further processing)
                        if (token.needs_space_before && !first_word) result += ' ';
                        result += combined_ipa;
                        first_word = false;
                        continue;
                    } else if (letter_ipa.size() == 1) {
                        ph_codes = letter_ipa[0];
                        // Fall through to normal output
                    } else {
                        ph_codes = wordToPhonemes(token.text);
                    }
                } else {
                    ph_codes = wordToPhonemes(token.text);
                }
            }

            // POS context override: apply $past/$noun/$verb pronunciations when context warrants.
            // the reference: after a $pastf word (was/were/had/...) expect_past > 0 → use $past entry.
            //         after a $nounf word (a/my/the/...) expect_noun > 0 → use $noun entry.
            //         after a $verbf word (I/we/will/to/...) expect_verb > 0 → use $verb entry.
            if (!is_isolated_word && !phrase_matched) {
                std::string lw = toLowerASCII(token.text);
                if (expect_past > 0) {
                    auto pit = past_dict_.find(lw);
                    if (pit != past_dict_.end())
                        ph_codes = processPhonemeString(pit->second);
                } else if (expect_noun > 0) {
                    auto nit = noun_dict_.find(lw);
                    if (nit != noun_dict_.end())
                        ph_codes = processPhonemeString(nit->second);
                } else if (expect_verb > 0) {
                    auto vit = verb_dict_.find(lw);
                    if (vit != verb_dict_.end())
                        ph_codes = processPhonemeString(vit->second);
                }
            }

            // $atstart override: if this is the first word token and it has an atstart entry,
            // use that pronunciation (e.g. "what" at utterance start → ",w02t" = ˌwʌt).
            if (first_word && !phrase_matched) {
                auto ait = atstart_dict_.find(toLowerASCII(token.text));
                if (ait != atstart_dict_.end())
                    ph_codes = ait->second;
            }

            // $atend override: if this is the last word token and it has an atend entry,
            // use that pronunciation (e.g. "to" at utterance end → "tu:" = tuː, not reduced tə).
            // Use raw phoneme string (no processPhonemeString): the $u flag already ensures
            // no stress marker is added, giving plain tuː (not tˈuː).
            if (ti == last_word_ti && !is_isolated_word && !phrase_matched) {
                auto aeit = atend_dict_.find(toLowerASCII(token.text));
                if (aeit != atend_dict_.end())
                    ph_codes = aeit->second;
            }

            // Special rule: "the" → ðɪ before vowel-starting words.
            // The dict has "the" = "D@2" (ðə). Before a vowel, the reference uses ðɪ.
            // Exception: before /j/-onset words (e.g. "university" → jˌuː...) → keep ðə.
            // Use '%DI' (whole-word unstressed prefix suppresses stress addition).
            if (toLowerASCII(token.text) == "the" && !ph_codes.empty()) {
                // Check if next word token starts with a vowel letter
                for (size_t tj = ti + 1; tj < tokens.size(); tj++) {
                    if (tokens[tj].is_word && !tokens[tj].text.empty()) {
                        char fc = (char)std::tolower((unsigned char)tokens[tj].text[0]);
                        if (fc=='a'||fc=='e'||fc=='i'||fc=='o'||fc=='u') {
                            // Check if the word phonemizes to a /j/-onset (e.g. "university" → jˌuː)
                            // the reference uses "the" (ðə) before /j/, not "the" (ðɪ)
                            bool j_onset = false;
                            if (fc == 'u' || fc == 'e') {
                                std::string nph = wordToPhonemes(toLowerASCII(tokens[tj].text));
                                // Strip leading stress markers to find first consonant/vowel
                                size_t npi = 0;
                                while (npi < nph.size() && (nph[npi]=='\''||nph[npi]==','||nph[npi]=='%'||nph[npi]=='='))
                                    npi++;
                                j_onset = (npi < nph.size() && nph[npi] == 'j');
                            }
                            if (!j_onset) {
                                // Pre-vocalic "the" = ðɪ, unstressed
                                ph_codes = "%DI";
                            }
                        }
                        break;
                    }
                    if (!tokens[tj].is_word) break; // punctuation - stop
                }
            }

            // Special case: "a" as isolated word → letter name "ˈeɪ" not article ɐ.
            // the reference gives ˈeɪ for isolated "a", ɐ for article "a" in sentences.
            // In sentence context, _a dict lookup gives letter name eI (wrong for article).
            // Force article phoneme a# (→ ɐ) for non-isolated "a".
            if (toLowerASCII(token.text) == "a") {
                if (is_isolated_word) {
                    ph_codes = "eI"; // letter name: ˈeɪ
                } else {
                    ph_codes = "a#"; // article: ɐ (unstressed)
                }
            }
            // Special case: "an" in sentence context.
            // the reference reduces "an" to ɐn before vowel-initial words (weak form),
            // but keeps it as æn before consonant-initial words (strong/stressed form).
            if (toLowerASCII(token.text) == "an" && !is_isolated_word) {
                // Find next word token to check if it starts with a vowel
                bool next_vowel_initial = false;
                {
                    size_t tj2 = ti + 1;
                    while (tj2 < tokens.size() && !tokens[tj2].is_word) tj2++;
                    if (tj2 < tokens.size() && !tokens[tj2].text.empty()) {
                        char fc = (char)std::tolower((unsigned char)tokens[tj2].text[0]);
                        next_vowel_initial = (fc=='a'||fc=='e'||fc=='i'||fc=='o'||fc=='u');
                    }
                }
                if (next_vowel_initial) {
                    ph_codes = "a#n"; // → ɐn (unstressed article before vowel)
                } else {
                    ph_codes = "an";  // → æn (stressed/unreduced before consonant)
                }
            }
            // Special case: "to" — strong form tʊ before vowels, weak form tə before consonants.
            // Isolated "to" → strong form tˈuː (letter/word in isolation).
            // Exception: before words that phonetically start with /j/ (e.g. "unify" → jˈuːnɪfˌaɪ,
            // "Europe" → jˈʊɹɹəp), "to" uses the weak form tə, not tʊ, because /j/ is a
            // glide/consonant in this context.
            if (toLowerASCII(token.text) == "to") {
                if (is_isolated_word || ti == last_word_ti) {
                    ph_codes = "tu:"; // strong form: tˈuː (isolation or utterance-final)
                } else {
                    // In sentences: tʊ before vowel-initial words, tə elsewhere
                    bool next_vowel_initial = false;
                    bool next_yod_initial = false;  // next word starts with /j/
                    {
                        size_t tj2 = ti + 1;
                        while (tj2 < tokens.size() && !tokens[tj2].is_word) tj2++;
                        if (tj2 < tokens.size() && !tokens[tj2].text.empty()) {
                            char fc = (char)std::tolower((unsigned char)tokens[tj2].text[0]);
                            next_vowel_initial = (fc=='a'||fc=='e'||fc=='i'||fc=='o'||fc=='u');
                            // Check if next word starts with phonetic /j/ (yod-initial).
                            // e.g. "unify"→jˈuː..., "Europe"→jˈʊɹ..., "unique"→jˈuː...
                            // Check first non-stress phoneme code of next word.
                            if (next_vowel_initial) {
                                std::string nw = toLowerASCII(tokens[tj2].text);
                                std::string nph = wordToPhonemes(nw);
                                // Strip stress/unstress markers from start
                                size_t si = 0;
                                while (si < nph.size() && (nph[si]=='\''||nph[si]==','||nph[si]=='%'||nph[si]=='=')) si++;
                                next_yod_initial = (si < nph.size() && nph[si] == 'j');
                            }
                        }
                    }
                    // Before /j/-initial words (like "unify", "Europe"), use weak form tə
                    // Use "t@5" (not "t@") so the @5 marker prevents cross-word @→3 rhotacization
                    // (the reference @5 phoneme does NOT have IF nextPh(isRhotic) → ChangePhoneme(3)).
                    ph_codes = (next_vowel_initial && !next_yod_initial) ? "tU" : "t@5";
                }
            }

            // Special case: "use" → verb form ju:z ($verb) when preceded by a pronoun.
            // the reference POS tagger detects verb context: "we use" → jˈuːz, "the use" → jˈuːs.
            // Heuristic: "use" is a verb if the previous word token is a personal pronoun.
            if (toLowerASCII(token.text) == "use" && !is_isolated_word) {
                static const std::unordered_set<std::string> PRONOUNS =
                    {"i", "we", "you", "they", "he", "she", "who"};
                std::string prev_word;
                // Find previous word token
                if (ti > 0) {
                    for (int tj = (int)ti - 1; tj >= 0; tj--) {
                        if (tokens[tj].is_word) { prev_word = toLowerASCII(tokens[tj].text); break; }
                    }
                }
                if (PRONOUNS.count(prev_word) > 0) {
                    ph_codes = "ju:z"; // verb form: jˈuːz
                }
            }

            // Post-process ph_codes for function words and stress normalization:

            // A. Whole-word unstressed (%prefix): the reference does NOT reduce vowels for % words.
            // The % prefix in en_list just marks these as function words (stress suppressed);
            // the phoneme string already has the correct vowel (e.g., %kan = kæn not kən).
            // No vowel reduction needed here — the % words' phonemes are already correct.
            // (Step A was previously reducing 'a'→'@' incorrectly for "can", "has", etc.)

            // B. $u-flagged words in sentence context: suppress primary stress.
            // Skip step B for phrase matches (bigram entries have their own phoneme coding).
            // the reference does NOT reduce vowels for $u words in sentence context;
            // the dict phoneme already has the correct vowel quality.
            // e.g., "have"=hav→hæv (not h@v), "our"=aU3→aʊɚ (not @ʊɚ).
            //
            // Words in KEEP_SECONDARY that also have $u: do NOT strip primary stress here;
            // instead let step C demote it to secondary stress (ˌ).
            // e.g., "within" $u2 $strend2: wID'In → step B skips → step C demotes → wID,In.
            static const std::unordered_set<std::string> STEP_B_KEEP_SECONDARY_WORDS =
                {"within", "without", "about", "across", "above", "among", "amongst",
                 "before", "upon", "below", "beside", "between", "beyond", "despite",
                 "except", "inside", "outside", "toward", "towards", "along",
                 "around", "behind", "beneath", "underneath",
                 // "over" and "under" get secondary stress as standalone prepositions:
                 "over", "under"};
            // For contractions like "there's", also check the base word (before apostrophe)
            std::string token_lower = toLowerASCII(token.text);
            std::string unstress_check = token_lower;
            auto apos_pos = token_lower.find('\'');
            if (apos_pos != std::string::npos)
                unstress_check = token_lower.substr(0, apos_pos);
            bool is_unstressed_word = !phrase_matched && (unstressed_words_.count(token_lower) > 0 ||
                                      (apos_pos != std::string::npos && unstressed_words_.count(unstress_check) > 0));
            // Words whose dict phoneme starts with '%' and whose only primary stress is a
            // "last resort" insertion before 'a#' should be unstressed in sentence context.
            // e.g. "has"=%h'a#z (step 5 last-resort inserts ' before a#): strip in sentences.
            // Content words with unstressed prefixes (e.g. "until"=%Vnt'Il, "effective"=%If'Ektiv)
            // have their primary ' before a strong vowel (not a#): do NOT strip.
            bool is_pct_word = false;
            if (!is_isolated_word && !ph_codes.empty() && ph_codes[0] == '%') {
                size_t prime_pos = ph_codes.find('\'');
                size_t hash_a_pos = ph_codes.find("'a#");
                // is_pct_word: primary exists, it's before 'a#', and it's the only primary
                if (prime_pos != std::string::npos && hash_a_pos != std::string::npos
                    && prime_pos == hash_a_pos) {
                    is_pct_word = true;
                }
            }
            bool is_step_b_keep_sec = STEP_B_KEEP_SECONDARY_WORDS.count(toLowerASCII(token.text)) > 0;
            if ((is_unstressed_word || is_pct_word) && !is_isolated_word && !is_step_b_keep_sec) {
                // Strip primary stress markers added by processPhonemeString step 5.
                ph_codes.erase(std::remove(ph_codes.begin(), ph_codes.end(), '\''),
                               ph_codes.end());
            }

            // C. Stress assignment for function words in sentence context.
            //
            // the reference behavior for function words:
            // - Words with ',' prefix in dict phoneme (e.g. ",bi:IN" = being, ",bVt" = but,
            //   "w,aIl" = while) → keep secondary stress ˌ in sentences (not promoted to primary)
            // - Words with `,` in their raw dict entry AND $strend2/$strend → keep secondary
            // - Words like "our", "our" with $u but strong vowel → get ˌ in sentence context
            //
            // Implementation:
            // KEEP_SECONDARY: words that should have ˌ (not ˈ) in sentence context.
            //   These include words with `,` in raw dict phoneme that shouldn't be promoted.
            // NEEDS_SECONDARY: words that after stress stripping (step B) should get ˌ added.
            {
                static const std::unordered_set<std::string> KEEP_SECONDARY =
                    {"on", "onto", "multiple", "multiples", "going",
                     "into", "any", "how", "where", "why",
                     // Words with ',' in dict phoneme AND $strend2/$strend in sentence context:
                     "being", "while", "but",
                     // Words that carry secondary stress as function words (verified vs the reference binary):
                     "across", "above", "among", "amongst", "before",
                     "within", "without", "upon", "below", "beside", "between",
                     "beyond", "underneath", "behind", "beneath",
                     // "over" and "under" carry secondary stress in sentence context (not primary):
                     "over", "under",
                     // "about" carries secondary stress in sentence context (ɐbˌaʊt), primary in isolation:
                     "about",
                     // "make" and "makes" carry secondary stress in sentence context:
                     "make", "makes"};
                // NEEDS_SECONDARY: words that should get ˌ added even if no ',' exists
                // (these are $u words that the reference assigns secondary stress to in sentence context)
                static const std::unordered_set<std::string> NEEDS_SECONDARY =
                    {"our"};

                // $strend2 words whose dict phoneme starts with ',' (secondary-stressed):
                // secondary in sentence context, promoted to primary when phrase-final.
                // E.g., "go" (,goU), "doing" (,du:IN), "so" (,soU), "should" (,SUd).
                bool is_strend_secondary = (comma_strend2_words_.count(toLowerASCII(token.text)) > 0 &&
                    !ph_codes.empty() && ph_codes[0] == ',' &&
                    ph_codes.find('\'') == std::string::npos);
                // -ing forms derived from $strend2 stems whose dict phoneme has secondary ',':
                // e.g., "making" from "make" ($strend2, "m,eIk" has ','). These should behave
                // like KEEP_SECONDARY: secondary before stressed content, primary phrase-final.
                // Only applies when stem dict phoneme has ',' — not for bare-phoneme strend2
                // words like "become" (bIkVm, no ',') whose -ing forms keep primary stress.
                bool is_ing_of_strend = false;
                {
                    std::string lw = toLowerASCII(token.text);
                    if (lw.size() > 3 && lw.compare(lw.size()-3, 3, "ing") == 0) {
                        std::string base3 = lw.substr(0, lw.size()-3);
                        for (const auto& sk : {base3, base3 + "e"}) {
                            if (strend_words_.count(sk) > 0) {
                                auto sit = dict_.find(sk);
                                if (sit != dict_.end() && sit->second.find(',') != std::string::npos) {
                                    is_ing_of_strend = true;
                                }
                                break;
                            }
                        }
                    }
                }
                // Phrase dict entries marked $u2+ (e.g. "did not", "do not", "does not"):
                // secondary in sentence context, primary phrase-final.
                bool is_keep_sec_phrase = (!matched_phrase_key.empty() &&
                    keep_sec_phrase_keys_.count(matched_phrase_key) > 0);
                bool keep_sec = !is_isolated_word &&
                                (KEEP_SECONDARY.count(toLowerASCII(token.text)) > 0 ||
                                 u2_strend2_words_.count(toLowerASCII(token.text)) > 0 ||
                                 u_plus_secondary_words_.count(toLowerASCII(token.text)) > 0 ||
                                 is_strend_secondary || is_ing_of_strend || is_keep_sec_phrase);
                bool needs_sec = !is_isolated_word &&
                                 NEEDS_SECONDARY.count(toLowerASCII(token.text)) > 0;

                // Promotion from secondary to primary: the reference promotes secondary-stress words
                // to primary when no following word in the utterance carries primary stress.
                // "following stressed word" = not in unstressed_words_ AND dict entry (if any)
                // doesn't start with ',' (secondary-only marker like "him", "me", "us", "them").
                // E.g., "about it" → about=primary; "about now" → about=secondary.
                // This applies to ALL KEEP_SECONDARY / u2_strend2 words (not u_plus_secondary).
                if (keep_sec && !u_plus_secondary_words_.count(toLowerASCII(token.text))) {
                    bool has_following_stressed = false;
                    for (size_t tj = ti + 1; tj < tokens.size(); tj++) {
                        if (tokens[tj].is_word) {
                            std::string fw = toLowerASCII(tokens[tj].text);
                            if (!unstressed_words_.count(fw)) {
                                // Words in KEEP_SECONDARY or u2_strend2_words_ will themselves
                                // get primary stress in this context (promoted), so they count
                                // as "stressed" here. Pure weak-form words (dict starts with ','
                                // but NOT in secondary-stress sets) are treated as unstressed.
                                // Also include comma_strend2 words: they get primary when
                                // phrase-final, so count as "stressed" for this check.
                                bool fw_is_strend_sec = comma_strend2_words_.count(fw) > 0;
                                // -ing forms of strend2 stems with secondary dict phoneme count as stressed
                                bool fw_is_ing_strend = false;
                                if (fw.size() > 3 && fw.compare(fw.size()-3, 3, "ing") == 0) {
                                    std::string fbase = fw.substr(0, fw.size()-3);
                                    for (const auto& sk2 : {fbase, fbase + std::string("e")}) {
                                        if (strend_words_.count(sk2) > 0) {
                                            auto fsit = dict_.find(sk2);
                                            if (fsit != dict_.end() && fsit->second.find(',') != std::string::npos)
                                                fw_is_ing_strend = true;
                                            break;
                                        }
                                    }
                                }
                                bool fw_in_secondary_set =
                                    KEEP_SECONDARY.count(fw) > 0 ||
                                    u2_strend2_words_.count(fw) > 0 ||
                                    fw_is_strend_sec || fw_is_ing_strend;
                                auto dit = dict_.find(fw);
                                bool fw_weak = !fw_in_secondary_set &&
                                               dit != dict_.end() &&
                                               !dit->second.empty() &&
                                               (dit->second[0] == ',' ||
                                                dit->second[0] == '%');  // % = fully unstressed
                                if (!fw_weak) {
                                    has_following_stressed = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (!has_following_stressed) {
                        keep_sec = false; // promote to primary
                        // If ph_codes has secondary ',' but no primary '\'', upgrade it.
                        if (ph_codes.find('\'') == std::string::npos &&
                            ph_codes.find(',') != std::string::npos) {
                            for (char& ch : ph_codes)
                                if (ch == ',') { ch = '\''; break; }
                        }
                    }
                }

                // Demote primary stress to secondary for KEEP_SECONDARY words in sentences
                if (keep_sec && ph_codes.find('\'') != std::string::npos) {
                    for (char& ch : ph_codes)
                        if (ch == '\'') { ch = ','; break; }
                }
                // Add secondary stress for NEEDS_SECONDARY words that have no stress marker
                if (needs_sec && ph_codes.find('\'') == std::string::npos &&
                    ph_codes.find(',') == std::string::npos) {
                    // Insert ',' before the first strong vowel
                    for (size_t pi = 0; pi < ph_codes.size(); pi++) {
                        char c = ph_codes[pi];
                        if (c=='a'||c=='A'||c=='e'||c=='E'||c=='i'||c=='I'||
                            c=='o'||c=='O'||c=='u'||c=='U'||c=='V'||c=='3') {
                            ph_codes.insert(pi, 1, ',');
                            break;
                        }
                    }
                }

                if (std::getenv("PHON_DEBUG")) {
                    std::cerr << "[StepC] word=" << token.text
                              << " ph_codes=" << ph_codes
                              << " is_isolated=" << is_isolated_word
                              << " keep_sec=" << keep_sec << "\n";
                }
                // Promote ',' to '\'' only in specific cases:
                // - In isolation (is_isolated_word): always promote (function words get
                //   primary stress when spoken alone).
                // - In sentences: only promote when ',' is NOT at position 0.
                //   A leading ',' means the word is inherently secondary-stressed (function
                //   word with ,phoneme in dict, like "weren't", "couldn't", "might").
                //   These stay secondary in sentences. A ',' elsewhere (e.g. compound prefix
                //   analysis result) gets promoted to primary in all contexts.
                {
                    size_t comma_pos = ph_codes.find(',');
                    // "Leading comma": ',' at position 0, or position 1 when '%' precedes it,
                    // OR the ',' appears before the word's first vowel (all preceding chars are
                    // consonants/markers). E.g. "gonna" (g,@n@): ',' at pos 1 after consonant 'g'
                    // → the secondary stress is word-initial in effect; stays secondary in sentences.
                    // In isolation, should_promote still fires (is_isolated_word=true).
                    bool leading_comma = (comma_pos == 0 ||
                        (comma_pos == 1 && !ph_codes.empty() && ph_codes[0] == '%'));
                    if (!leading_comma && comma_pos != std::string::npos) {
                        // Check if all chars before the comma are consonants (no vowel code).
                        // Vowel single-char codes: a A e E i I o O u U V 0 3 @
                        static const std::string VOWEL_CHARS = "aAeEiIoOuUV03@";
                        bool pre_comma_vowel = false;
                        for (size_t ci = 0; ci < comma_pos; ci++) {
                            char cc = ph_codes[ci];
                            if (cc == '\'' || cc == '%' || cc == '=') continue;
                            if (VOWEL_CHARS.find(cc) != std::string::npos) {
                                pre_comma_vowel = true; break;
                            }
                        }
                        if (!pre_comma_vowel) leading_comma = true;
                    }
                    // Phrase-matched entries starting with '%' that have secondary ',' but no
                    // primary '\'' carry genuine secondary stress — don't promote to primary.
                    // E.g. "%D%e@,A@" (there are) → ˌɑːɹ stays secondary, not promoted to ˈ.
                    bool is_pct_phrase = phrase_matched && !ph_codes.empty() &&
                                        ph_codes[0] == '%' && !is_isolated_word &&
                                        ph_codes.find('\'') == std::string::npos;
                    bool should_promote = (!keep_sec && !needs_sec &&
                        ph_codes.find('\'') == std::string::npos &&
                        comma_pos != std::string::npos &&
                        !is_pct_phrase &&
                        (is_isolated_word || !leading_comma));
                    if (should_promote) {
                        for (char& ch : ph_codes)
                            if (ch == ',') { ch = '\''; break; }
                    }
                }

                // $unstressend: word keeps secondary stress even when utterance-final.
                // Demote any primary stress to secondary for these words at sentence end.
                if (!is_isolated_word && ti == last_word_ti &&
                    unstressend_words_.count(toLowerASCII(token.text)) > 0) {
                    if (ph_codes.find('\'') != std::string::npos) {
                        for (char& ch : ph_codes)
                            if (ch == '\'') { ch = ','; break; }
                    }
                }
            }

            // D. Fix diphthong stress position: if pattern X'Y where XY is a diphthong,
            //    move '\'' to before X so the full diphthong gets the stress marker.
            {
                static const char* DIPHS[] = {"eI","aI","aU","OI","oU",nullptr};
                for (size_t si = 1; si + 1 < ph_codes.size(); si++) {
                    if (ph_codes[si] != '\'') continue;
                    char prev = ph_codes[si-1], next = ph_codes[si+1];
                    std::string pair = {prev, next};
                    for (int di = 0; DIPHS[di]; di++) {
                        if (pair == DIPHS[di]) {
                            ph_codes.erase(si, 1);
                            ph_codes.insert(si-1, 1, '\'');
                            break;
                        }
                    }
                }
            }

            // Inter-word T-flapping: if ph_codes ends with 't#' (the reference flappable-t marker)
            // and the next word token starts with a vowel phoneme, convert 't#' → '*' (flap)
            // before IPA conversion. the reference applies this in connected speech for function words
            // like "at" (at#) and "it" (It#) before vowel-initial words.
            if (ph_codes.size() >= 2 &&
                ph_codes[ph_codes.size()-2] == 't' &&
                ph_codes[ph_codes.size()-1] == '#') {
                // Find next word token
                size_t tnext = ti + 1;
                while (tnext < tokens.size() && !tokens[tnext].is_word) tnext++;
                if (tnext < tokens.size() && !tokens[tnext].text.empty()) {
                    // Get the phoneme of the next word and check its first phoneme code
                    std::string nph = wordToPhonemes(toLowerASCII(tokens[tnext].text));
                    // Strip leading stress/unstressed markers to find first real phoneme
                    size_t npi = 0;
                    while (npi < nph.size() && (nph[npi]=='\''||nph[npi]==','||nph[npi]=='%'||nph[npi]=='='))
                        npi++;
                    static const std::string VOWEL_STARTS = "aAeEIiOUVu03@oY";
                    bool next_vowel_onset = (npi < nph.size() && VOWEL_STARTS.find(nph[npi]) != std::string::npos);
                    if (next_vowel_onset) {
                        // Replace trailing 't#' → '*' (flap) for IPA conversion
                        ph_codes[ph_codes.size()-2] = '*';
                        ph_codes.erase(ph_codes.size()-1, 1);
                    }
                }
            }

            // Cross-word '@' → '3' rhotacization: standalone schwa before r-initial word.
            // the reference ph_english_us: phoneme @ { IF nextPh(isRhotic) THEN ChangePhoneme(3) }
            // e.g. "Siberia releases" → saɪbˈiəɹɪɚ (@ → 3 = ɚ before 'r' of "releases").
            // Only applies to standalone '@' (not diphthong-final '@' like 'i@' = ɪə).
            if (is_en_us && !is_isolated_word &&
                !ph_codes.empty() && ph_codes.back() == '@') {
                // Check it's standalone '@': preceding char must NOT be a bigram prefix.
                // Bigram prefixes for '@': A(A@), e(e@), E(E@?), O(O@), o(o@), U(U@), i(i@)
                bool standalone = true;
                if (ph_codes.size() >= 2) {
                    char prev = ph_codes[ph_codes.size()-2];
                    if (prev=='A'||prev=='e'||prev=='O'||prev=='o'||prev=='U'||prev=='i')
                        standalone = false;
                }
                if (standalone) {
                    // Find next word token and check if it starts with rhotic phoneme 'r'
                    for (size_t tj = ti + 1; tj < tokens.size(); tj++) {
                        if (!tokens[tj].is_word) break; // punctuation stops rule
                        if (tokens[tj].text.empty()) break;
                        std::string nph = wordToPhonemes(tokens[tj].text);
                        // Skip stress/modifier markers to find first real phoneme
                        size_t pi = 0;
                        while (pi < nph.size() &&
                               (nph[pi]=='\''||nph[pi]==','||nph[pi]=='%'||nph[pi]=='='))
                            pi++;
                        if (pi < nph.size() && nph[pi] == 'r') {
                            // Change trailing '@' to '3' (rhotic schwa ɚ)
                            ph_codes.back() = '3';
                        }
                        break;
                    }
                }
            }

            std::string ipa = phonemesToIPA(ph_codes);

            // Add default stress unless word is inherently unstressed:
            // - ph_codes starts with '%' (whole-word unstressed, e.g. "can" = "%kan")
            // - ph_codes contains @2/@5 (weak schwa variants, e.g. "the" = "D@2")
            // - word has $u flag AND no strong vowels remain after reduction
            //   (e.g. "of"→@v has only schwa; but "it"→It# has strong vowel I)
            // Isolated function words (single word in text) still get stressed
            // (the reference stresses function words when spoken in isolation)
            // '%' anywhere in phoneme string (without explicit stress markers) means the word
            // should be unstressed in sentence context. e.g. "when" → "w%En" (w + unstressed-E + n).
            // The '%' before a vowel code marks that vowel as inherently unstressed.
            bool pct_unstressed = (ph_codes.find('%') != std::string::npos &&
                                   ph_codes.find('\'') == std::string::npos &&
                                   (ph_codes.find(',') == std::string::npos || phrase_matched)) && !is_isolated_word;
            // All $u-flagged function words are unstressed in sentence context
            bool u_unstressed = is_unstressed_word && !is_isolated_word;
            // @2/@5 are "weak schwa" variants that are intrinsically unstressed phonemes.
            // the reference @2/@5 phoneme codes encode inherent lack of stress — the reference never
            // stresses these variants even in isolation (e.g., "the" → ðə, not ðˈə).
            bool weak_schwa = (ph_codes.find("@2") != std::string::npos ||
                               ph_codes.find("@5") != std::string::npos) &&
                              ph_codes.find('\'') == std::string::npos &&
                              ph_codes.find(',') == std::string::npos;
            // Article "a"/"an" in sentence context → ɐ (unstressed, no stress marker)
            bool article_a = (!is_isolated_word &&
                              (toLowerASCII(token.text) == "a" || toLowerASCII(token.text) == "an") &&
                              (ph_codes == "a#" || ph_codes == "a#n"));
            bool no_stress = pct_unstressed || weak_schwa || u_unstressed || article_a;
            if (!no_stress) ipa = addDefaultStress(ipa);

            // Pre-vowel "the" in phrase dict context: replace trailing ə → ɪ in IPA.
            // The phrase phoneme (e.g. "w%IDD@2" for "with the") keeps "@2" through
            // processPhonemeString; after phonemesToIPA we swap ə (0xC9 0x99) → ɪ (0xC9 0xAA).
            if (phrase_pre_vowel_the) {
                if (ipa.size() >= 2 &&
                    (unsigned char)ipa[ipa.size()-2] == 0xc9 &&
                    (unsigned char)ipa[ipa.size()-1] == 0x99) {
                    ipa[ipa.size()-1] = (char)0xaa;
                }
            }

            // R-linking (r-sandhi): words ending in ɚ (U+025A = 0xC9 0x9A) or ɜː (U+025C U+02D0
            // = 0xC9 0x9C 0xCB 0x90) get a linking ɹ (U+0279 = 0xC9 0xB9) before vowel-initial words.
            // e.g. "require unambiguous" → ɹᵻkwˈaɪɚɹ, "her apple" → hɜːɹ ˈæpəl
            // BUT NOT before words starting with [j] sound (e.g. "university"→jˌuːnɪvˈɜːsᵻɾi).
            // We check by phonemizing the next word and checking its first phoneme code.
            bool ends_in_rhotic_r = false;
            if (ipa.size() >= 2 &&
                (unsigned char)ipa[ipa.size()-2] == 0xC9 &&
                (unsigned char)ipa[ipa.size()-1] == 0x9A) {
                ends_in_rhotic_r = true;  // ends in ɚ
            } else if (ipa.size() >= 4 &&
                (unsigned char)ipa[ipa.size()-4] == 0xC9 &&
                (unsigned char)ipa[ipa.size()-3] == 0x9C &&
                (unsigned char)ipa[ipa.size()-2] == 0xCB &&
                (unsigned char)ipa[ipa.size()-1] == 0x90) {
                ends_in_rhotic_r = true;  // ends in ɜː
            }
            if (ends_in_rhotic_r) {
                // Find next word token
                for (size_t tj = ti + 1; tj < tokens.size(); tj++) {
                    if (tokens[tj].is_word && !tokens[tj].text.empty()) {
                        // the reference suppresses linking-r before conjunctions ("or", "and") when
                        // they are non-final (i.e., the conjunction has a following word).
                        // This reflects prosodic phrase-boundary suppression of r-sandhi.
                        std::string next_word_lc = tokens[tj].text;
                        for (char& cc : next_word_lc) cc = (char)std::tolower((unsigned char)cc);
                        bool is_conjunction = (next_word_lc == "or" || next_word_lc == "and");
                        if (is_conjunction && !first_word) {
                            // Check if there is another word after the conjunction
                            bool has_word_after = false;
                            for (size_t tk = tj + 1; tk < tokens.size(); tk++) {
                                if (tokens[tk].is_word && !tokens[tk].text.empty()) {
                                    has_word_after = true;
                                    break;
                                }
                                if (!tokens[tk].is_word) break;
                            }
                            if (has_word_after) break; // suppress linking-r
                        }

                        // For abbreviation tokens (containing '.') or all-caps acronyms
                        // (e.g. "RNA", "DNA"), use the first LETTER's phoneme for sandhi checking.
                        // All-caps acronyms are spelled as letter names; their first letter determines
                        // whether they are vowel-initial (e.g. R→A@→vowel, N→E2n→vowel, U→j→consonant).
                        std::string next_ph;
                        bool use_first_letter = (tokens[tj].text.find('.') != std::string::npos);
                        if (!use_first_letter && tokens[tj].text.size() >= 2) {
                            bool all_upper = true;
                            for (char c2 : tokens[tj].text)
                                if (!std::isupper((unsigned char)c2)) { all_upper = false; break; }
                            if (all_upper) {
                                std::string lower_nxt = toLowerASCII(tokens[tj].text);
                                if (!dict_.count(lower_nxt) || abbrev_words_.count(lower_nxt))
                                    use_first_letter = true;
                            }
                        }
                        if (use_first_letter) {
                            // Find first letter in the token
                            for (char lc : tokens[tj].text) {
                                if (std::isalpha((unsigned char)lc)) {
                                    next_ph = wordToPhonemes(std::string(1, lc));
                                    break;
                                }
                            }
                        } else {
                            next_ph = wordToPhonemes(tokens[tj].text);
                        }
                        // Skip stress/modifier markers to find first real phoneme
                        size_t pi = 0;
                        while (pi < next_ph.size() &&
                               (next_ph[pi] == '\'' || next_ph[pi] == ',' ||
                                next_ph[pi] == '%'  || next_ph[pi] == '='))
                            pi++;
                        bool next_vowel = (pi < next_ph.size() &&
                            isVowelCode(std::string(1, next_ph[pi])));
                        if (next_vowel) {
                            ipa += '\xC9';
                            ipa += '\xB9'; // append ɹ
                        }
                        break;
                    }
                    if (!tokens[tj].is_word) break; // punctuation stops r-linking
                }
            }

            // Cross-word /t/ flapping: function words ending in 't' before vowel-initial words.
            // In American English, unstressed word-final /t/ (after a vowel) flaps to [ɾ]
            // when the next word begins with a vowel. e.g. "it is" → ɪɾ ɪz, "it equally" → ɪɾ ˈiːkwəli.
            if (is_en_us && !is_isolated_word && !ipa.empty() && ipa.back() == 't') {
                std::string lc_word = toLowerASCII(token.text);
                // Apply specifically to "it" (pronoun), which the reference flaps before vowels.
                // Other function words ending in 't' (that, but, not) do NOT flap cross-word.
                if (lc_word == "it") {
                    // Find next word token and check if it is vowel-initial
                    for (size_t tj = ti + 1; tj < tokens.size(); tj++) {
                        if (!tokens[tj].is_word) continue;
                        if (tokens[tj].text.empty()) break;
                        std::string nph = wordToPhonemes(tokens[tj].text);
                        // Skip stress/modifier markers to find first real phoneme
                        size_t pi2 = 0;
                        while (pi2 < nph.size() && (nph[pi2]=='\''||nph[pi2]==','||
                                                    nph[pi2]=='%'||nph[pi2]=='=')) pi2++;
                        if (pi2 < nph.size()) {
                            char fc = nph[pi2];
                            bool nv = (fc=='a'||fc=='A'||fc=='e'||fc=='E'||fc=='i'||fc=='I'||
                                      fc=='o'||fc=='O'||fc=='u'||fc=='U'||fc=='V'||fc=='0'||
                                      fc=='3'||fc=='@');
                            if (nv) {
                                // Flap: replace trailing 't' with ɾ (U+027E = 0xC9 0xBE)
                                ipa.pop_back();
                                ipa += '\xC9';
                                ipa += '\xBE';
                            }
                        }
                        break;
                    }
                }
            }

            if (!ipa.empty()) {
                if (!first_word) result += " ";
                result += ipa;
                first_word = false;
            }

            // Update POS context counters for the next word.
            // the reference: $pastf sets expect_past=3 then decrements; $nounf sets expect_noun=2 then
            // decrements; $verbf sets expect_verb=2 then decrements.
            // We model this as set-then-decrement in one step (net: 2 or 1 for next word).
            {
                std::string lw = toLowerASCII(token.text);
                if (pastf_words_.count(lw)) {
                    expect_past = 3;  // will be decremented to 2 below
                    expect_noun = 0;
                    expect_verb = 0;
                } else if (nounf_words_.count(lw)) {
                    expect_noun = 2;  // will be decremented to 1 below
                    expect_past = 0;
                    expect_verb = 0;
                } else if (verbf_words_.count(lw)) {
                    expect_verb = 2;  // will be decremented to 1 below
                    expect_past = 0;
                    expect_noun = 0;
                }
                // Always decrement (the reference decrements in same step as setting).
                if (expect_past > 0) expect_past--;
                if (expect_noun > 0) expect_noun--;
                if (expect_verb > 0) expect_verb--;
            }

            next_token_label:;  // target for cliticization goto
        } else {
            // Punctuation: the reference does NOT output punctuation characters in IPA.
            // Commas and sentence boundaries become newlines (clause groups) in the reference
            // output, but we suppress them as they cause word count mismatches.
            (void)token; // suppress unused warning
        }
    }

    return result;
}

std::vector<std::string> IPAPhonemizer::phonemize(const std::vector<std::string>& texts) {
    std::vector<std::string> results;
    results.reserve(texts.size());
    for (const auto& text : texts)
        results.push_back(phonemizeText(text));
    return results;
}
