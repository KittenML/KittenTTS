//! ONNX model inference and voice embedding loading.
//!
//! Port of `onnx_model.py` — loads the ONNX session, reads voice embeddings
//! from an NPZ archive, phonemises input text via eSpeak, and runs inference.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{s, Array2};
use ndarray_npy::NpzReader;
use ort::session::Session;
use ort::value::Value;
use regex::Regex;

use crate::config::ModelConfig;
use crate::espeak::Espeak;
use crate::preprocess::TextPreprocessor;
use crate::text_cleaner::TextCleaner;

// ─────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────

/// Basic English tokenizer: split on word / non-word boundaries.
fn basic_english_tokenize(text: &str) -> Vec<String> {
    let re = Regex::new(r"\w+|[^\w\s]").unwrap();
    re.find_iter(text).map(|m| m.as_str().to_string()).collect()
}

/// Ensure text ends with punctuation; append a comma if needed.
fn ensure_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }
    let last = text.chars().last().unwrap();
    if ".!?,;:".contains(last) {
        text.to_string()
    } else {
        format!("{},", text)
    }
}

/// Split text into chunks for processing long texts.
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let re = Regex::new(r"[.!?]+").unwrap();
    let sentences: Vec<&str> = re.split(text).collect();
    let mut chunks = Vec::new();

    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        if sentence.len() <= max_len {
            chunks.push(ensure_punctuation(sentence));
        } else {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let mut temp_chunk = String::new();
            for word in words {
                if temp_chunk.len() + word.len() + 1 <= max_len {
                    if !temp_chunk.is_empty() {
                        temp_chunk.push(' ');
                    }
                    temp_chunk.push_str(word);
                } else {
                    if !temp_chunk.is_empty() {
                        chunks.push(ensure_punctuation(temp_chunk.trim()));
                    }
                    temp_chunk = word.to_string();
                }
            }
            if !temp_chunk.is_empty() {
                chunks.push(ensure_punctuation(temp_chunk.trim()));
            }
        }
    }
    chunks
}

// ─────────────────────────────────────────────
// KittenTTS Model
// ─────────────────────────────────────────────

/// Core KittenTTS ONNX model for text-to-speech synthesis.
pub struct KittenTTSModel {
    session: Session,
    voices: HashMap<String, Array2<f32>>,
    text_cleaner: TextCleaner,
    preprocessor: TextPreprocessor,
    espeak: Espeak,

    // Voice metadata
    pub available_voices: Vec<String>,
    pub all_voice_names: Vec<String>,
    pub voice_aliases: HashMap<String, String>,
    pub speed_priors: HashMap<String, f32>,
}

impl KittenTTSModel {
    /// Load a model from a directory containing config.json, the ONNX model, and voices.npz.
    pub fn from_dir(model_dir: &Path, espeak_data_path: Option<&str>) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config = ModelConfig::load(&config_path).context("Failed to load config.json")?;

        let model_path = model_dir.join(&config.model_file);
        let voices_path = model_dir.join(&config.voices);

        Self::new(
            &model_path,
            &voices_path,
            config.speed_priors,
            config.voice_aliases,
            espeak_data_path,
        )
    }

    /// Create a new model from explicit file paths.
    pub fn new(
        model_path: &Path,
        voices_path: &Path,
        speed_priors: HashMap<String, f32>,
        voice_aliases: HashMap<String, String>,
        espeak_data_path: Option<&str>,
    ) -> Result<Self> {
        // Load ONNX session
        let session = Session::builder()?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        // Load voice embeddings from NPZ
        let voices = Self::load_voices(voices_path)?;

        let available_voices = vec![
            "expr-voice-2-m",
            "expr-voice-2-f",
            "expr-voice-3-m",
            "expr-voice-3-f",
            "expr-voice-4-m",
            "expr-voice-4-f",
            "expr-voice-5-m",
            "expr-voice-5-f",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let all_voice_names = vec![
            "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let text_cleaner = TextCleaner::new();
        let preprocessor = TextPreprocessor::default();
        let espeak = Espeak::new(espeak_data_path).context("Failed to initialize eSpeak-NG")?;

        Ok(KittenTTSModel {
            session,
            voices,
            text_cleaner,
            preprocessor,
            espeak,
            available_voices,
            all_voice_names,
            voice_aliases,
            speed_priors,
        })
    }

    /// Load all voice embeddings from a .npz file.
    fn load_voices(path: &Path) -> Result<HashMap<String, Array2<f32>>> {
        let file = std::fs::File::open(path).context("Failed to open voices.npz")?;
        let mut npz = NpzReader::new(file).context("Failed to parse voices.npz")?;

        let mut voices = HashMap::new();
        let names = npz.names().context("Failed to read NPZ entry names")?;

        for name in &names {
            // Strip the .npy extension that ndarray-npy may include
            let clean_name = name.trim_end_matches(".npy").to_string();
            let arr: Array2<f32> = npz
                .by_name(&clean_name)
                .context(format!("Failed to read voice array '{}'", clean_name))?;
            voices.insert(clean_name, arr);
        }

        Ok(voices)
    }

    /// Resolve a voice name/alias to the internal voice key.
    fn resolve_voice<'a>(&'a self, voice: &'a str) -> Result<&'a str> {
        let resolved = self
            .voice_aliases
            .get(voice)
            .map(|s| s.as_str())
            .unwrap_or(voice);
        if !self.available_voices.contains(&resolved.to_string()) {
            anyhow::bail!(
                "Voice '{}' not available. Choose from: {:?}",
                voice,
                self.all_voice_names
            );
        }
        Ok(resolved)
    }

    /// Phonemise text using eSpeak-NG.
    fn phonemise(&self, text: &str) -> Result<String> {
        self.espeak
            .set_voice("en-us")
            .context("Failed to set eSpeak voice to en-us")?;

        self.espeak
            .text_to_phonemes(text)
            .context("eSpeak phonemisation failed")
    }

    /// Prepare ONNX model inputs from text and voice parameters.
    fn prepare_inputs(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
    ) -> Result<(Vec<i64>, Array2<f32>, f32)> {
        let voice = self.resolve_voice(voice)?;

        let mut speed = speed;
        if let Some(&prior) = self.speed_priors.get(voice) {
            speed *= prior;
        }

        // Phonemise
        let phonemes_raw = self.phonemise(text)?;
        let tokens_str = basic_english_tokenize(&phonemes_raw);
        let phonemes_joined = tokens_str.join(" ");

        #[cfg(debug_assertions)]
        println!("Phonemes: {}", phonemes_joined);

        let mut tokens = self.text_cleaner.encode(&phonemes_joined);

        #[cfg(debug_assertions)]
        println!("Tokens: {:?}", tokens);

        // Add start and end tokens
        tokens.insert(0, 0); // start token
        tokens.push(10); // end marker
        tokens.push(0); // pad

        // Get voice reference embedding
        let voice_arr = self
            .voices
            .get(voice)
            .ok_or_else(|| anyhow::anyhow!("Voice embedding '{}' not found in NPZ", voice))?;

        let ref_id = text.len().min(voice_arr.nrows() - 1);
        let ref_s = voice_arr.slice(s![ref_id..ref_id + 1, ..]).to_owned();

        Ok((tokens, ref_s, speed))
    }

    /// Generate speech for a single text chunk.
    fn generate_single_chunk(&mut self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let (tokens, ref_s, speed) = self.prepare_inputs(text, voice, speed)?;

        let input_ids_shape = vec![1, tokens.len()];
        let input_ids_vec = tokens;

        let speed_shape = vec![1];
        let speed_vec = vec![speed];

        let ref_s_shape = vec![ref_s.shape()[0], ref_s.shape()[1]];
        let ref_s_vec: Vec<f32> = ref_s.iter().copied().collect();

        let inputs = ort::inputs![
            "input_ids" => Value::from_array((input_ids_shape, input_ids_vec))?,
            "style" => Value::from_array((ref_s_shape, ref_s_vec))?,
            "speed" => Value::from_array((speed_shape, speed_vec))?,
        ];

        let outputs = self.session.run(inputs)?;

        // Extract audio output (first output tensor)
        let (_, audio_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract audio tensor")?;

        let total_len = audio_data.len();

        // Trim last 5000 samples (same as Python)
        let trim_len = 5000.min(total_len);
        let trimmed_len = total_len - trim_len;

        let audio: Vec<f32> = audio_data.iter().take(trimmed_len).copied().collect();

        Ok(audio)
    }

    /// Generate speech from text.
    ///
    /// Automatically chunks long texts, generates each chunk, and concatenates.
    pub fn generate(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        clean_text: bool,
    ) -> Result<Vec<f32>> {
        let text = if clean_text {
            self.preprocessor.process(text)
        } else {
            text.to_string()
        };

        let chunks = chunk_text(&text, 400);
        let mut all_audio = Vec::new();

        for chunk in &chunks {
            let audio = self.generate_single_chunk(chunk, voice, speed)?;
            all_audio.extend(audio);
        }

        Ok(all_audio)
    }

    /// Generate speech and save to a WAV file.
    pub fn generate_to_file(
        &mut self,
        text: &str,
        output_path: &Path,
        voice: &str,
        speed: f32,
        sample_rate: u32,
        clean_text: bool,
    ) -> Result<()> {
        let audio = self.generate(text, voice, speed, clean_text)?;

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer =
            hound::WavWriter::create(output_path, spec).context("Failed to create WAV file")?;

        for &sample in &audio {
            writer
                .write_sample(sample)
                .context("Failed to write audio sample")?;
        }

        writer.finalize().context("Failed to finalise WAV file")?;

        println!("Audio saved to {}", output_path.display());
        Ok(())
    }
}
