import os
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ['ESPEAK_DATA_PATH'] = espeakng_loader.get_data_path()
import numpy as np
import phonemizer
import soundfile as sf
import onnxruntime as ort
from ..preprocess import TextPreprocessor, chunk_text, normalize_text

def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

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



# The ONNX legacy models bake a 512-position BERT encoder into the graph: both
# the position and the token-type embedding tables have 512 entries. Any input
# sequence longer than that makes the /bert/Expand node fail with
# "invalid expand shape", so 512 tokens is the hard per-pass limit for the 0.8
# models. It is exposed on each instance as `max_seq_len` so the text path can
# split long input to stay under it.
DEFAULT_MAX_SEQ_LEN = 512


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
        
        # Available voices
        self.available_voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
        ]
        self.all_voice_names = ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
        self.voice_aliases = voice_aliases

        self.preprocessor = TextPreprocessor(remove_punctuation=False)

        # Per-pass token budget enforced by the baked-in BERT embeddings; see
        # DEFAULT_MAX_SEQ_LEN. Kept as an instance attribute so a caller with a
        # different exported model can override it.
        self.max_seq_len = DEFAULT_MAX_SEQ_LEN
    
    def _resolve_voice(self, voice: str, speed: float):
        """Resolve voice aliases, validate the voice, and apply its speed prior."""
        if voice in self.voice_aliases:
            voice = self.voice_aliases[voice]

        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")

        if voice in self.speed_priors:
            speed = speed * self.speed_priors[voice]
        return voice, speed

    def _phonemize(self, text: str) -> str:
        """Phonemize text into the space-joined phoneme string the cleaner reads."""
        phonemes_list = self.phonemizer.phonemize([text])
        return ' '.join(basic_english_tokenize(phonemes_list[0]))

    def _inputs_from_phonemes(self, phonemes: str, voice: str, speed: float, ref_len: int) -> dict:
        """Build ONNX inputs from an already-phonemized string."""
        tokens = self.text_cleaner(phonemes)

        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(10)
        tokens.append(0)

        input_ids = np.array([tokens], dtype=np.int64)
        ref_id = min(ref_len, self.voices[voice].shape[0] - 1)
        ref_s = self.voices[voice][ref_id:ref_id+1]

        return {
            "input_ids": input_ids,
            "style": ref_s,
            "speed": np.array([speed], dtype=np.float32),
        }

    def _split_phonemes(self, phonemes: str) -> list:
        """Split a phoneme string into pieces that each fit under max_seq_len tokens.

        The BERT encoder baked into the ONNX graph only has `max_seq_len` (512)
        position and token-type embeddings, so a longer sequence makes
        /bert/Expand fail with "invalid expand shape". Cuts prefer word
        (space) boundaries and only fall back to a hard cut when a single word
        is itself longer than the budget.
        """
        budget = self.max_seq_len - 3  # room for the 0 / 10 / 0 special tokens
        word_index = self.text_cleaner.word_index_dictionary
        pieces = []
        start = 0
        count = 0
        i = 0
        n = len(phonemes)
        while i < n:
            c = phonemes[i]
            if c in word_index:
                count += 1
            if count > budget:
                if c == " ":
                    cut = i
                else:
                    sp = phonemes.rfind(" ", start, i)
                    cut = sp + 1 if sp >= start else i
                pieces.append(phonemes[start:cut])
                start = cut
                count = 0
                i = start
                continue
            i += 1
        if start < n:
            pieces.append(phonemes[start:])
        pieces = [p for p in pieces if p.strip()]
        return pieces or [""]

    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict:
        """Prepare ONNX model inputs from text and voice parameters."""
        voice, speed = self._resolve_voice(voice, speed)
        phonemes = self._phonemize(text)
        return self._inputs_from_phonemes(phonemes, voice, speed, len(text))
    
    def normalize_text(self, text: str, locale: str = "en-US", return_spans: bool = False):
        return normalize_text(text, locale=locale, return_spans=return_spans)

    def generate(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0, clean_text: bool=True) -> np.ndarray:
        out_chunks = []
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        return np.concatenate(out_chunks, axis=-1)

    def generate_stream(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0, clean_text: bool = True):
        """Generate audio chunk-by-chunk as a generator.

        Yields:
            numpy.ndarray: Audio data for each text chunk.
        """
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            yield self.generate_single_chunk(text_chunk, voice, speed)

    def generate_single_chunk(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0) -> np.ndarray:
        """Synthesize speech from text.

        The ONNX graph's BERT encoder only holds ``max_seq_len`` (512) position
        and token-type embeddings, so a single pass cannot take more than that
        many tokens. Longer text is phonemized once and split at word
        boundaries to stay under the limit, then the pieces are run and
        concatenated -- instead of failing with an opaque
        ``/bert/Expand`` "invalid expand shape" error.

        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)

        Returns:
            Audio data as numpy array
        """
        voice, speed = self._resolve_voice(voice, speed)
        phonemes = self._phonemize(text)
        pieces = self._split_phonemes(phonemes)
        ref_len = len(text)

        out = []
        for piece in pieces:
            onnx_inputs = self._inputs_from_phonemes(piece, voice, speed, ref_len)
            outputs = self.session.run(None, onnx_inputs)
            out.append(outputs[0][..., :-5000])

        return out[0] if len(out) == 1 else np.concatenate(out, axis=-1)
    
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
