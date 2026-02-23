from misaki import en, espeak
import numpy as np
import phonemizer
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
        self.session = ort.InferenceSession(model_path)
        
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        self.speed_priors = speed_priors
        
        # Available voices dynamically loaded from the .npz file
        self.available_voices = list(self.voices.keys())
        
        # Default fallback aliases, can be extended by config
        default_aliases = {
            'Bella': 'expr-voice-2-f',
            'Jasper': 'expr-voice-2-m',
            'Luna': 'expr-voice-3-f',
            'Bruno': 'expr-voice-3-m',
            'Rosie': 'expr-voice-4-f',
            'Hugo': 'expr-voice-4-m',
            'Kiki': 'expr-voice-5-f',
            'Leo': 'expr-voice-5-m'
        }
        self.voice_aliases = default_aliases
        self.voice_aliases.update(voice_aliases)


        self.preprocessor = TextPreprocessor()
    
    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict:
        """Prepare ONNX model inputs from text and voice parameters."""
        # Try to resolve alias if necessary
        if voice in self.voice_aliases:
            voice = self.voice_aliases[voice]

        # Check if the requested voice exists in the dynamic voice dictionary
        if voice not in self.voices:
            fallback = list(self.aliases.keys())[0] if hasattr(self, 'aliases') and len(self.aliases) > 0 else (self.available_voices[0] if self.available_voices else None)
            error_msg = f"\n❌ Voice '{voice}' not found."
            error_msg += f"\n👉 Available native voices: {self.available_voices}"
            error_msg += f"\n👉 Available voice aliases: {list(self.voice_aliases.keys())}"
            if fallback:
                error_msg += f"\nPlease try using a valid voice like '{fallback}'."
            print(error_msg)
            raise ValueError(f"Voice '{voice}' not available.")
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
        
        # Trim audio
        audio = outputs[0][..., :-5000]

        return audio
    
    def generate_to_file(self, text: str, output_path: str, voice: str = "expr-voice-5-m", 
                          speed: float = 1.0, sample_rate: int = 24000, clean_text: bool=True) -> None:
        """Synthesize speech and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
            clean_text: If true, it will cleanup the text. Eg. replace numbers with words.
        """
        audio = self.generate(text, voice, speed, clean_text=clean_text)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")

