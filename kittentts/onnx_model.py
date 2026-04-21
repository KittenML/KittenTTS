import os
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ['ESPEAK_DATA_PATH'] = espeakng_loader.get_data_path()
import numpy as np
import phonemizer
import soundfile as sf
import onnxruntime as ort
from typing import Optional, Dict, Any, Tuple
from .preprocess import TextPreprocessor

try:
    from .chinese_processor import ChineseTextProcessor, has_chinese, extract_subtitle_info
    CHINESE_PROCESSOR_AVAILABLE = True
except ImportError:
    CHINESE_PROCESSOR_AVAILABLE = False


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
    def __init__(self, model_path="kitten_tts_nano_preview.onnx", voices_path="voices.npz", speed_priors={}, voice_aliases={}, backend=None):
        """Initialize KittenTTS with model and voice data.
        
        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the voices NPZ file
        """
        self.model_path = model_path
        self.voices = np.load(voices_path) 
        providers = []
        if backend == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif backend == "amd_gpu":
            providers = ["ROCMExecutionProvider"]
        elif backend == "cpu":
            providers = ["CPUExecutionProvider"]
        elif backend is None:
            providers = []
        else:
            raise ValueError("Unsupported backend")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        self.speed_priors = speed_priors
        
        self.available_voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
        ]
        self.all_voice_names = ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
        self.voice_aliases = voice_aliases

        self.preprocessor = TextPreprocessor(remove_punctuation=False)
        
        self.chinese_processor = None
        if CHINESE_PROCESSOR_AVAILABLE:
            self.chinese_processor = ChineseTextProcessor()
    
    def _detect_language(self, text: str) -> str:
        """Detect if text contains Chinese characters.
        
        Returns:
            'chinese' if contains Chinese, 'english' otherwise
        """
        if CHINESE_PROCESSOR_AVAILABLE and self.chinese_processor:
            if self.chinese_processor.has_chinese(text):
                return 'chinese'
        return 'english'
    
    def _process_chinese_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Process Chinese text for synthesis.
        
        Returns:
            Tuple of (processed_text, subtitle_info)
        """
        if not CHINESE_PROCESSOR_AVAILABLE or not self.chinese_processor:
            return text, {'original': text, 'pinyin': text, 'segments': []}
        
        processed_text, subtitle_info = self.chinese_processor.process(text, convert_pinyin=True)
        return processed_text, subtitle_info
    
    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict:
        """Prepare ONNX model inputs from text and voice parameters."""
        if voice in self.voice_aliases:
            voice = self.voice_aliases[voice]

        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")
        
        if voice in self.speed_priors:
            speed = speed * self.speed_priors[voice]
        
        phonemes_list = self.phonemizer.phonemize([text])
        
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = ' '.join(phonemes)
        tokens = self.text_cleaner(phonemes)
        
        tokens.insert(0, 0)
        tokens.append(10)
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
        """Generate audio from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            clean_text: If true, preprocess text
        
        Returns:
            Audio data as numpy array
        """
        original_text = text
        
        language = self._detect_language(text)
        
        subtitle_info = {'original': original_text, 'pinyin': '', 'segments': []}
        
        if language == 'chinese':
            text, subtitle_info = self._process_chinese_text(text)
            clean_text = False
        
        if clean_text:
            text = self.preprocessor(text)
        
        out_chunks = []
        for text_chunk in chunk_text(text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        
        return np.concatenate(out_chunks, axis=-1)
    
    def generate_with_subtitles(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0, clean_text: bool=True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate audio with subtitle information.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            clean_text: If true, preprocess text
        
        Returns:
            Tuple of (audio_data, subtitle_info)
            subtitle_info contains:
                - original: Original text
                - pinyin: Pinyin transcription (if Chinese)
                - processed: Processed text used for synthesis
                - language: Detected language ('chinese' or 'english')
                - segments: Detailed segment information
        """
        original_text = text
        
        language = self._detect_language(text)
        
        processed_text = text
        subtitle_info = {
            'original': original_text,
            'pinyin': '',
            'processed': '',
            'language': language,
            'segments': []
        }
        
        if language == 'chinese':
            processed_text, chinese_subtitle = self._process_chinese_text(text)
            subtitle_info['pinyin'] = chinese_subtitle.get('pinyin', '')
            subtitle_info['segments'] = chinese_subtitle.get('segments', [])
            clean_text = False
        else:
            if CHINESE_PROCESSOR_AVAILABLE:
                subtitle_info['segments'] = extract_subtitle_info(text).get('segments', [])
        
        if clean_text:
            processed_text = self.preprocessor(processed_text)
        
        subtitle_info['processed'] = processed_text
        
        out_chunks = []
        for text_chunk in chunk_text(processed_text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        
        audio = np.concatenate(out_chunks, axis=-1)
        
        return audio, subtitle_info

    def generate_stream(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0, clean_text: bool = True):
        """Generate audio chunk-by-chunk as a generator.

        Yields:
            numpy.ndarray: Audio data for each text chunk.
        """
        language = self._detect_language(text)
        
        if language == 'chinese':
            text, _ = self._process_chinese_text(text)
            clean_text = False
        
        if clean_text:
            text = self.preprocessor(text)
        
        for text_chunk in chunk_text(text):
            yield self.generate_single_chunk(text_chunk, voice, speed)

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
        
        audio = outputs[0][..., :-5000]

        return audio
    
    def generate_to_file(self, text: str, output_path: str, voice: str = "expr-voice-5-m", 
                          speed: float = 1.0, sample_rate: int = 24000, clean_text: bool=True) -> Dict[str, Any]:
        """Synthesize speech and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
            clean_text: If true, it will cleanup the text. Eg. replace numbers with words.
        
        Returns:
            Dictionary containing subtitle information
        """
        audio, subtitle_info = self.generate_with_subtitles(text, voice, speed, clean_text=clean_text)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")
        return subtitle_info
