import numpy as np
import phonemizer
import soundfile as sf
import onnxruntime as ort
import threading
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_english_tokenize(text: str) -> List[str]:
    """Optimized English tokenizer with caching."""
    import re
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


class TextCleaner:
    """Optimized text cleaner with pre-computed mappings."""
    
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        
        # Pre-compute mapping for O(1) lookup
        self.word_index_dictionary = {symbols[i]: i for i in range(len(symbols))}
        
        # Cache for processed strings
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._max_cache_size = 1000

    def __call__(self, text: str) -> List[int]:
        """Convert text to indices with caching."""
        # Check cache first
        with self._cache_lock:
            if text in self._cache:
                return self._cache[text].copy()
        
        # Process text
        indexes = []
        for char in text:
            if char in self.word_index_dictionary:
                indexes.append(self.word_index_dictionary[char])
        
        # Cache result if within size limit
        with self._cache_lock:
            if len(self._cache) < self._max_cache_size:
                self._cache[text] = indexes.copy()
        
        return indexes
    
    def clear_cache(self):
        """Clear the processing cache."""
        with self._cache_lock:
            self._cache.clear()


class KittenTTS_1_Onnx:
    """Enhanced KittenTTS ONNX model with real-time optimizations."""
    
    def __init__(self, 
                 model_path: str = "kitten_tts_nano_preview.onnx", 
                 voices_path: str = "voices.npz",
                 optimize_for_performance: bool = True,
                 enable_gpu: bool = True):
        """Initialize KittenTTS with performance optimizations.
        
        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the voices NPZ file
            optimize_for_performance: Enable performance optimizations
            enable_gpu: Try to use GPU acceleration if available
        """
        self.model_path = model_path
        self.optimize_for_performance = optimize_for_performance
        
        # Load voices with error handling
        try:
            self.voices = np.load(voices_path)
            logger.info(f"Loaded {len(self.voices.files)} voices from {voices_path}")
        except Exception as e:
            logger.error(f"Failed to load voices: {e}")
            raise
        
        # Initialize ONNX session with optimizations
        self.session = self._create_optimized_session(model_path, enable_gpu)
        
        # Initialize phonemizer with optimizations
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", 
            preserve_punctuation=True, 
            with_stress=True,
            njobs=1  # Single job for consistency in real-time
        )
        
        self.text_cleaner = TextCleaner()
        
        # Available voices
        self.available_voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
        ]
        
        # Performance monitoring
        self._generation_times = []
        self._lock = threading.RLock()
        
        # Input/output caching for identical requests
        self._io_cache = {}
        self._cache_lock = threading.RLock()
        self._max_io_cache_size = 100
        
        # Pre-compute common voice embeddings for faster access
        self._precomputed_voices = {}
        self._precompute_voices()
        
        logger.info("KittenTTS_1_Onnx initialized successfully")
    
    def _create_optimized_session(self, model_path: str, enable_gpu: bool) -> ort.InferenceSession:
        """Create an optimized ONNX runtime session."""
        providers = []
        
        if enable_gpu:
            # Try GPU providers first
            if ort.get_device() == 'GPU':
                providers.extend(['CUDAExecutionProvider', 'ROCMExecutionProvider'])
        
        # Always include CPU provider as fallback
        providers.append('CPUExecutionProvider')
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        if self.optimize_for_performance:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.inter_op_num_threads = 0  # Use all available cores
            sess_options.intra_op_num_threads = 0  # Use all available cores
        
        try:
            session = ort.InferenceSession(model_path, sess_options, providers=providers)
            logger.info(f"ONNX session created with providers: {session.get_providers()}")
            return session
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            # Fallback to basic session
            return ort.InferenceSession(model_path)
    
    def _precompute_voices(self):
        """Pre-compute voice embeddings for faster access."""
        for voice in self.available_voices:
            if voice in self.voices:
                self._precomputed_voices[voice] = self.voices[voice].copy()
        logger.info(f"Pre-computed {len(self._precomputed_voices)} voice embeddings")
    
    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generate a cache key for input parameters."""
        return f"{hash(text)}_{voice}_{speed}"
    
    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> Dict[str, np.ndarray]:
        """Prepare ONNX model inputs with optimizations and caching."""
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")
        
        # Check cache first
        cache_key = self._get_cache_key(text, voice, speed)
        with self._cache_lock:
            if cache_key in self._io_cache:
                return self._io_cache[cache_key].copy()
        
        start_time = time.time()
        
        # Phonemize the input text
        try:
            phonemes_list = self.phonemizer.phonemize([text])
        except Exception as e:
            logger.warning(f"Phonemization failed, using fallback: {e}")
            phonemes_list = [text]  # Fallback to raw text
        
        # Process phonemes to get token IDs
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = ' '.join(phonemes)
        tokens = self.text_cleaner(phonemes)
        
        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(0)
        
        input_ids = np.array([tokens], dtype=np.int64)
        
        # Use pre-computed voice embedding
        ref_s = self._precomputed_voices.get(voice, self.voices[voice])
        
        inputs = {
            "input_ids": input_ids,
            "style": ref_s,
            "speed": np.array([speed], dtype=np.float32),
        }
        
        # Cache the inputs if within size limit
        with self._cache_lock:
            if len(self._io_cache) < self._max_io_cache_size:
                self._io_cache[cache_key] = {k: v.copy() for k, v in inputs.items()}
        
        prep_time = time.time() - start_time
        logger.debug(f"Input preparation took {prep_time:.4f}s")
        
        return inputs
    
    def generate(self, 
                 text: str, 
                 voice: str = "expr-voice-5-m", 
                 speed: float = 1.0) -> np.ndarray:
        """Synthesize speech from text with performance optimizations.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Audio data as numpy array
        """
        start_time = time.time()
        
        try:
            # Prepare inputs
            onnx_inputs = self._prepare_inputs(text, voice, speed)
            
            # Run inference
            inference_start = time.time()
            outputs = self.session.run(None, onnx_inputs)
            inference_time = time.time() - inference_start
            
            # Process output
            if outputs and len(outputs) > 0:
                # Trim audio with bounds checking
                audio = outputs[0]
                if len(audio) > 15000:  # Ensure we have enough samples
                    audio = audio[5000:-10000]
                else:
                    audio = audio  # Use full audio if too short
            else:
                raise RuntimeError("No output from ONNX model")
            
            total_time = time.time() - start_time
            
            # Record performance metrics
            with self._lock:
                self._generation_times.append(total_time)
                if len(self._generation_times) > 100:  # Keep only recent times
                    self._generation_times.pop(0)
            
            logger.debug(f"Generation completed: {total_time:.4f}s (inference: {inference_time:.4f}s)")
            
            return audio
            
        except Exception as e:
            logger.error(f"Generation failed for text '{text[:50]}...': {e}")
            raise RuntimeError(f"Speech generation failed: {e}")
    
    def generate_batch(self, 
                      texts: List[str], 
                      voice: str = "expr-voice-5-m", 
                      speed: float = 1.0,
                      max_workers: int = 4) -> List[np.ndarray]:
        """Generate audio for multiple texts in parallel.
        
        Args:
            texts: List of input texts to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of audio arrays
        """
        if len(texts) == 1:
            return [self.generate(texts[0], voice, speed)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.generate, text, voice, speed) 
                for text in texts
            ]
            results = [future.result() for future in futures]
        
        return results
    
    def generate_to_file(self, 
                        text: str, 
                        output_path: str, 
                        voice: str = "expr-voice-5-m", 
                        speed: float = 1.0, 
                        sample_rate: int = 24000) -> None:
        """Synthesize speech and save to file with error handling.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
        """
        try:
            audio = self.generate(text, voice, speed)
            sf.write(output_path, audio, sample_rate)
            logger.info(f"Audio saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save audio to {output_path}: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        with self._lock:
            if not self._generation_times:
                return {"average_time": 0.0, "min_time": 0.0, "max_time": 0.0}
            
            times = self._generation_times.copy()
            
        return {
            "average_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "recent_samples": len(times)
        }
    
    def warm_up(self, voice: str = "expr-voice-5-m"):
        """Warm up the model for optimal performance."""
        logger.info("Warming up model...")
        warmup_texts = [
            "Hello",
            "This is a test.",
            "Real-time speech synthesis."
        ]
        
        for text in warmup_texts:
            self.generate(text, voice, speed=1.0)
        
        logger.info("Model warm-up completed")
    
    def clear_caches(self):
        """Clear all internal caches to free memory."""
        self.text_cleaner.clear_cache()
        with self._cache_lock:
            self._io_cache.clear()
        logger.info("Caches cleared")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'session') and self.session:
                del self.session
        except:
            pass


# Example usage with performance monitoring
if __name__ == "__main__":
    # Initialize with optimizations
    tts = KittenTTS_1_Onnx(
        model_path="kitten_tts_nano_preview.onnx",
        voices_path="voices.npz",
        optimize_for_performance=True,
        enable_gpu=True
    )
    
    # Warm up
    tts.warm_up()
    
    # Test text
    text = """
    It begins with an "Ugh!" Another mysterious stain appears on a favorite shirt. 
    Every trick has been tried, but the stain persists.
    """
    
    # Generate with timing
    start = time.time()
    audio = tts.generate(text, voice="expr-voice-5-m", speed=1.0)
    end = time.time()
    
    print(f"Generation took {end - start:.4f}s")
    print(f"Audio shape: {audio.shape}")
    print(f"Performance stats: {tts.get_performance_stats()}")
    
    # Save result
    tts.generate_to_file(text, "optimized_output.wav", voice="expr-voice-5-m")
