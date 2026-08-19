use std::collections::HashMap;

/// Maps phoneme/symbol characters to integer token IDs.
///
/// Replicates the Python `TextCleaner` class: builds a lookup from the same
/// ordered symbol table (pad + punctuation + ASCII letters + IPA letters).
pub struct TextCleaner {
    word_index: HashMap<char, i64>,
}

impl TextCleaner {
    pub fn new() -> Self {
        let pad = "$";
        let punctuation = ";:,.!?¡¿—…\"«»\"\" ";
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";
        let extra = "|‖";

        let mut symbols = Vec::new();
        symbols.push(pad.chars().next().unwrap());
        for c in punctuation.chars() {
            symbols.push(c);
        }
        for c in letters.chars() {
            symbols.push(c);
        }
        for c in letters_ipa.chars() {
            symbols.push(c);
        }
        for c in extra.chars() {
            symbols.push(c);
        }

        let mut word_index = HashMap::new();
        for (i, &ch) in symbols.iter().enumerate() {
            word_index.insert(ch, i as i64);
        }

        TextCleaner { word_index }
    }

    /// Convert a phoneme string to a sequence of token IDs.
    /// Characters not in the symbol table are silently skipped.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        text.chars()
            .filter_map(|c| self.word_index.get(&c).copied())
            .collect()
    }
}

impl Default for TextCleaner {
    fn default() -> Self {
        Self::new()
    }
}
