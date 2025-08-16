import numpy as np
import threading
import queue
import time
import logging
from typing import Generator, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
import soundfile as sf
from .utils import AudioBuffer, PerformanceProfiler

logger = logging.getLogger(__name__)


class RealtimeKittenTTS:
    """Real-time optimized TTS engine with streaming capabilities."""
    
    def __init__(self, 
                 base_model,
                 max_workers: int = 4,
                 buffer_size: int = 8192,
                 profiler: Optional[PerformanceProfiler] = None):
        """Initialize real-time TTS engine.
        
        Args:
            base_model: Base KittenTTS ONNX model
            max_workers: Number of worker threads for parallel processing
            buffer_size: Audio buffer size for streaming
            profiler: Performance profiler instance
        """
        self.base_model = base_model
        self.max_workers = max_workers
        self.profiler = profiler
        
        # Initialize audio buffer for streaming
        self.audio_buffer = AudioBuffer(buffer_size)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Caching system for repeated requests
        self._audio_cache = {}
        self._cache_lock = threading.RLock()
        self._max_cache_size = 50
        
        # Text chunking parameters
        self.sentence_endings = {'.', '!', '?', ';'}
        self.clause_endings = {',', ':', '-', '—'}
        
        # Performance tracking
        self._processing_queue = queue.Queue()
        self._is_processing = threading.Event()
        self._shutdown = threading.Event()
        
        # Background processing thread
        self._processor_thread = threading.Thread(target=self._background_processor, daemon=True)
        self._processor_thread.start()
        
        logger.info(f"RealtimeKittenTTS initialized with {max_workers} workers")
    
    def _background_processor(self):
        """Background thread for processing queued requests."""
        while not self._shutdown.is_set():
            try:
                # Get next processing job
                job = self._processing_queue.get(timeout=1.0)
                if job is None:  # Shutdown signal
                    break
                
                # Process the job
                self._process_job(job)
                self._processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Background processor error: {e}")
    
    def _process_job(self, job: Dict[str, Any]):
        """Process a single TTS job."""
        try:
            text = job['text']
            voice = job['voice']
            speed = job['speed']
            callback = job.get('callback')
            
            # Generate audio
            audio = self.base_model.generate(text, voice, speed)
            
            # Call callback if provided
            if callback:
                callback(audio)
                
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
    
    def generate_fast(self, 
                     text: str, 
                     voice: str = "expr-voice-5-m", 
                     speed: float = 1.0,
                     use_cache: bool = True) -> np.ndarray:
        """Generate audio with real-time optimizations.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            use_cache: Use caching for repeated requests
            
        Returns:
            Audio data as numpy array
        """
        if self.profiler:
            self.profiler.start_timing("realtime_generation")
        
        # Check cache first
        if use_cache:
            cache_key = f"{hash(text)}_{voice}_{speed}"
            with self._cache_lock:
                if cache_key in self._audio_cache:
                    if self.profiler:
                        self.profiler.end_timing("realtime_generation")
                    return self._audio_cache[cache_key].copy()
        
        # Generate audio using base model
        audio = self.base_model.generate(text, voice, speed)
        
        # Cache the result
        if use_cache:
            with self._cache_lock:
                if len(self._audio_cache) < self._max_cache_size:
                    self._audio_cache[cache_key] = audio.copy()
                elif len(self._audio_cache) >= self._max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._audio_cache))
                    del self._audio_cache[oldest_key]
                    self._audio_cache[cache_key] = audio.copy()
        
        if self.profiler:
            self.profiler.end_timing("realtime_generation")
        
        return audio
    
    def generate_streaming(self, 
                          text: str, 
                          voice: str = "expr-voice-5-m", 
                          speed: float = 1.0,
                          chunk_size: int = 100) -> Generator[np.ndarray, None, None]:
        """Generate audio in streaming chunks for real-time playback.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            chunk_size: Maximum characters per chunk
            
        Yields:
            Audio chunks as numpy arrays
        """
        if self.profiler:
            self.profiler.start_timing("streaming_generation")
        
        # Split text into processable chunks
        chunks = self._split_text_for_streaming(text, chunk_size)
        
        # Process chunks in parallel with sliding window
        futures = []
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(self.generate_fast, chunk, voice, speed, True)
            futures.append((i, future))
            
            # Yield results as they become available
            if len(futures) >= self.max_workers or i == len(chunks) - 1:
                # Sort by order and yield completed futures
                completed = []
                for idx, fut in futures:
                    if fut.done():
                        completed.append((idx, fut))
                
                # Remove completed futures and yield their results
                for idx, fut in sorted(completed):
                    try:
                        audio_chunk = fut.result()
                        yield audio_chunk
                        futures.remove((idx, fut))
                    except Exception as e:
                        logger.error(f"Chunk generation failed: {e}")
        
        # Yield any remaining futures
        for idx, future in sorted(futures):
            try:
                audio_chunk = future.result(timeout=10.0)
                yield audio_chunk
            except Exception as e:
                logger.error(f"Final chunk generation failed: {e}")
        
        if self.profiler:
            self.profiler.end_timing("streaming_generation")
    
    def _split_text_for_streaming(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into chunks optimized for streaming synthesis.
        
        Args:
            text: Input text to split
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def preprocess_for_streaming(self, text: str, voice: str, speed: float):
        """Preprocess text for streaming by queuing background jobs."""
        chunks = self._split_text_for_streaming(text, 50)
        
        for chunk in chunks:
            job = {
                'text': chunk,
                'voice': voice,
                'speed': speed,
                'callback': None
            }
            self._processing_queue.put(job)
    
    def clear_cache(self):
        """Clear the audio cache."""
        with self._cache_lock:
            self._audio_cache.clear()
        logger.info("Real-time engine cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._audio_cache),
                'max_cache_size': self._max_cache_size
            }
    
    def cleanup(self):
        """Cleanup resources and shutdown background threads."""
        logger.info("Shutting down real-time engine...")
        
        # Signal shutdown
        self._shutdown.set()
        self._processing_queue.put(None)  # Shutdown signal
        
        # Wait for processor thread to finish
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear resources
        self.clear_cache()
        
        logger.info("Real-time engine shutdown complete")


class StreamingKittenTTS:
    """Streaming TTS interface for real-time audio playback."""
    
    def __init__(self, realtime_engine: RealtimeKittenTTS, sample_rate: int = 24000):
        """Initialize streaming TTS interface.
        
        Args:
            realtime_engine: Real-time TTS engine instance
            sample_rate: Audio sample rate
        """
        self.engine = realtime_engine
        self.sample_rate = sample_rate
        self.is_streaming = threading.Event()
        
        # Audio playback queue
        self.playback_queue = queue.Queue(maxsize=10)
        self.playback_thread = None
        
    def start_streaming(self, 
                       text: str, 
                       voice: str = "expr-voice-5-m", 
                       speed: float = 1.0,
                       chunk_size: int = 100,
                       output_file: Optional[str] = None) -> threading.Thread:
        """Start streaming TTS synthesis and playback.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            chunk_size: Text chunk size for streaming
            output_file: Optional file to save complete audio
            
        Returns:
            Thread handle for the streaming process
        """
        def streaming_worker():
            try:
                self.is_streaming.set()
                all_audio = []
                
                for audio_chunk in self.engine.generate_streaming(text, voice, speed, chunk_size):
                    if not self.is_streaming.is_set():
                        break
                    
                    # Add to playback queue
                    try:
                        self.playback_queue.put(audio_chunk, timeout=1.0)
                        all_audio.append(audio_chunk)
                    except queue.Full:
                        logger.warning("Playback queue full, dropping audio chunk")
                
                # Save complete audio if requested
                if output_file and all_audio:
                    complete_audio = np.concatenate(all_audio)
                    sf.write(output_file, complete_audio, self.sample_rate)
                    logger.info(f"Complete audio saved to {output_file}")
                
            except Exception as e:
                logger.error(f"Streaming worker error: {e}")
            finally:
                self.is_streaming.clear()
                self.playback_queue.put(None)  # End signal
        
        # Start streaming thread
        stream_thread = threading.Thread(target=streaming_worker, daemon=True)
        stream_thread.start()
        
        return stream_thread
    
    def stop_streaming(self):
        """Stop the streaming process."""
        self.is_streaming.clear()
        logger.info("Streaming stopped")
    
    def get_audio_chunks(self) -> Generator[np.ndarray, None, None]:
        """Get audio chunks from the playback queue.
        
        Yields:
            Audio chunks for playback
        """
        while True:
            try:
                chunk = self.playback_queue.get(timeout=1.0)
                if chunk is None:  # End signal
                    break
                yield chunk
            except queue.Empty:
                if not self.is_streaming.is_set():
                    break
                continue
    
    def stream_to_file(self, 
                      text: str, 
                      output_file: str,
                      voice: str = "expr-voice-5-m", 
                      speed: float = 1.0,
                      chunk_size: int = 100):
        """Stream synthesis directly to file.
        
        Args:
            text: Input text to synthesize
            output_file: Output audio file path
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            chunk_size: Text chunk size for streaming
        """
        logger.info(f"Streaming synthesis to {output_file}")
        
        audio_chunks = []
        for chunk in self.engine.generate_streaming(text, voice, speed, chunk_size):
            audio_chunks.append(chunk)
        
        if audio_chunks:
            complete_audio = np.concatenate(audio_chunks)
            sf.write(output_file, complete_audio, self.sample_rate)
            logger.info(f"Streaming synthesis complete: {output_file}")
        else:
            logger.error("No audio chunks generated")


# Example usage
if __name__ == "__main__":
    from kittentts.onnx_model import KittenTTS_1_Onnx
    from kittentts.utils import PerformanceProfiler
    
    # Initialize components
    base_model = KittenTTS_1_Onnx(optimize_for_performance=True)
    profiler = PerformanceProfiler()
    realtime_engine = RealtimeKittenTTS(base_model, profiler=profiler)
    streaming_tts = StreamingKittenTTS(realtime_engine)
    
    # Test text
    test_text = """
    Real-time text-to-speech synthesis enables immediate audio feedback 
    for interactive applications. This streaming approach provides 
    low-latency audio generation suitable for live conversations, 
    virtual assistants, and interactive media.
    """
    
    # Test real-time generation
    print("Testing real-time generation...")
    start_time = time.time()
    audio = realtime_engine.generate_fast(test_text)
    generation_time = time.time() - start_time
    print(f"Real-time generation: {generation_time:.4f}s")
    print(f"Audio shape: {audio.shape}")
    
    # Test streaming to file
    print("Testing streaming synthesis...")
    streaming_tts.stream_to_file(test_text, "streaming_output.wav")
    
    # Print performance stats
    if profiler:
        stats = profiler.get_stats()
        print(f"Performance stats: {stats}")
    
    # Cleanup
    realtime_engine.cleanup()
