# Try to load espeak-ng library if available (needed on Windows)
# This sets up the espeak-ng library and data paths properly
try:
    import espeakng_loader
    espeakng_loader.load_library()
    # Set the data path environment variable required by espeak
    import os
    if 'ESPEAK_DATA_PATH' not in os.environ:
        os.environ['ESPEAK_DATA_PATH'] = str(espeakng_loader.get_data_path())
    # Tell phonemizer where to find the espeak library
    from phonemizer.backend.espeak.base import BaseEspeakBackend
    BaseEspeakBackend.set_library(str(espeakng_loader.get_library_path()))
except Exception:
    # If loader fails, phonemizer might still find system espeak
    pass

import numpy as np

import phonemizer
from phonemizer.backend import BACKENDS
import soundfile as sf
import onnxruntime as ort
from .preprocess import TextPreprocessor

def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

def ensure_punctuation(text):
    """Ensure text ends with punctuation. If not, add a comma."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in '.!?,;:':
        text = text + ','
    return text


def chunk_text(text, max_len=400):
    """Split text into chunks for processing long texts."""
    import re
    
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(sentence) <= max_len:
            chunks.append(ensure_punctuation(sentence))
        else:
            # Split long sentences by words
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_len:
                    temp_chunk += " " + word if temp_chunk else word
                else:
                    if temp_chunk:
                        chunks.append(ensure_punctuation(temp_chunk.strip()))
                    temp_chunk = word
            if temp_chunk:
                chunks.append(ensure_punctuation(temp_chunk.strip()))
    
    return chunks


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        
        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i

        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


class KittenTTS_1_Onnx:
    def __init__(self, model_path="kitten_tts_nano_preview.onnx", voices_path="voices.npz", speed_priors={}, voice_aliases={}):
        """Initialize KittenTTS with model and voice data.
        
        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the voices NPZ file
        """
        self.model_path = model_path
        self.voices = np.load(voices_path)
        
        # Configure ONNX Runtime for best audio quality and performance
        sess_options = ort.SessionOptions()
        
        # Use all available cores for parallel processing
        sess_options.intra_op_num_threads = 0  # 0 = use all cores
        sess_options.inter_op_num_threads = 0
        
        # Graph optimizations for better inference
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        
        # Get available providers (prefer CPUExecutionProvider for consistency)
        available_providers = ort.get_available_providers()
        providers = []
        
        # For TTS quality/consistency, CPU is often more deterministic than GPU
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')
        elif 'AzureExecutionProvider' in available_providers:
            providers.append('AzureExecutionProvider')
        
        # Create session with optimized settings
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        # Use the BACKENDS dict to get EspeakBackend (handles API differences across versions)
        EspeakBackend = BACKENDS.get('espeak') or BACKENDS.get('espeak-ng')
        if EspeakBackend is None:
            raise RuntimeError("No espeak backend available. Install espeak-ng and phonemizer.")
        
        self.phonemizer = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        self.speed_priors = speed_priors
        
        # Available voices
        self.available_voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
        ]
        self.voice_aliases = voice_aliases

        self.preprocessor = TextPreprocessor()
    
    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict:
        """Prepare ONNX model inputs from text and voice parameters."""
        if voice in self.voice_aliases:
            voice = self.voice_aliases[voice]

        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")
        
        if voice in self.speed_priors:
            speed = speed * self.speed_priors[voice]
        
        # Phonemize the input text
        phonemes_list = self.phonemizer.phonemize([text])
        
        # Process phonemes to get token IDs
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = ' '.join(phonemes)
        tokens = self.text_cleaner(phonemes)
        
        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(0)
        
        input_ids = np.array([tokens], dtype=np.int64)
        ref_id =  min(len(text), self.voices[voice].shape[0] - 1)
        ref_s = self.voices[voice][ref_id:ref_id+1]
        
        return {
            "input_ids": input_ids,
            "style": ref_s,
            "speed": np.array([speed], dtype=np.float32),
        }
    
    def generate(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0, clean_text: bool=True) -> np.ndarray:
        out_chunks = []
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        return np.concatenate(out_chunks, axis=-1)

    def generate_single_chunk(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Audio data as numpy array
        """
        onnx_inputs = self._prepare_inputs(text, voice, speed)
        
        outputs = self.session.run(None, onnx_inputs)
        audio = outputs[0]
        
        # Smart trimming: remove trailing silence while preserving actual audio content
        audio = self._smart_trim_trailing_silence(audio)
        
        # Normalize audio to prevent clipping and ensure consistent volume
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Soft normalization: don't over-compress, just prevent clipping
            if max_val > 0.95:
                audio = audio * (0.95 / max_val)

        return audio
    
    def _smart_trim_trailing_silence(self, audio: np.ndarray, threshold: float = 0.01, 
                                      padding_ms: float = 50.0, sample_rate: int = 24000) -> np.ndarray:
        """Trim trailing silence while preserving audio content.
        
        Args:
            audio: Audio data as numpy array
            threshold: Amplitude threshold for silence detection
            padding_ms: Milliseconds of padding to keep after audio ends
            sample_rate: Audio sample rate
            
        Returns:
            Trimmed audio data
        """
        if audio.shape[-1] < 8000:  # Don't trim very short audio
            return audio
        
        # Find the last sample above the threshold
        energy = np.abs(audio)
        above_threshold = np.where(energy > threshold)[0]
        
        if len(above_threshold) == 0:
            # All silence, return original
            return audio
        
        # Find the end of the last audio segment
        last_audio_sample = above_threshold[-1]
        
        # Add padding (default 50ms) to avoid cutting off decay
        padding_samples = int(padding_ms / 1000.0 * sample_rate)
        end_sample = min(last_audio_sample + padding_samples, audio.shape[-1])
        
        return audio[..., :end_sample]
    
    def generate_to_file(self, text: str, output_path: str, voice: str = "expr-voice-5-m", 
                          speed: float = 1.0, sample_rate: int = 24000, clean_text: bool=True,
                          subtype: str = 'PCM_16') -> None:
        """Synthesize speech and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
            clean_text: If true, it will cleanup the text. Eg. replace numbers with words.
            subtype: SoundFile subtype for quality (PCM_16, PCM_24, FLOAT)
        """
        audio = self.generate(text, voice, speed, clean_text=clean_text)
        
        # Ensure audio is float32 for best compatibility
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Write with specified subtype for quality
        sf.write(output_path, audio, sample_rate, subtype=subtype)
        print(f"Audio saved to {output_path} ({len(audio)/sample_rate:.2f}s at {sample_rate}Hz)")

