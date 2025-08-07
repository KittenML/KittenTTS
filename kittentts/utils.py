import numpy as np
import threading
import time
import queue
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Thread-safe circular audio buffer for streaming applications."""
    
    def __init__(self, size: int = 8192):
        """Initialize audio buffer.
        
        Args:
            size: Buffer size in samples
        """
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.available_samples = 0
        self.lock = threading.RLock()
        
    def write(self, audio_data: np.ndarray) -> int:
        """Write audio data to buffer.
        
        Args:
            audio_data: Audio samples to write
            
        Returns:
            Number of samples written
        """
        with self.lock:
            audio_data = audio_data.flatten().astype(np.float32)
            samples_to_write = min(len(audio_data), self.size - self.available_samples)
            
            if samples_to_write <= 0:
                return 0
            
            # Handle wrapping
            if self.write_pos + samples_to_write <= self.size:
                self.buffer[self.write_pos:self.write_pos + samples_to_write] = audio_data[:samples_to_write]
            else:
                # Split write across buffer boundary
                first_part = self.size - self.write_pos
                self.buffer[self.write_pos:] = audio_data[:first_part]
                self.buffer[:samples_to_write - first_part] = audio_data[first_part:samples_to_write]
            
            self.write_pos = (self.write_pos + samples_to_write) % self.size
            self.available_samples += samples_to_write
            
            return samples_to_write
    
    def read(self, num_samples: int) -> np.ndarray:
        """Read audio data from buffer.
        
        Args:
            num_samples: Number of samples to read
            
        Returns:
            Audio samples
        """
        with self.lock:
            samples_to_read = min(num_samples, self.available_samples)
            
            if samples_to_read <= 0:
                return np.array([])
            
            result = np.zeros(samples_to_read, dtype=np.float32)
            
            # Handle wrapping
            if self.read_pos + samples_to_read <= self.size:
                result = self.buffer[self.read_pos:self.read_pos + samples_to_read].copy()
            else:
                # Split read across buffer boundary
                first_part = self.size - self.read_pos
                result[:first_part] = self.buffer[self.read_pos:]
                result[first_part:] = self.buffer[:samples_to_read - first_part]
            
            self.read_pos = (self.read_pos + samples_to_read) % self.size
            self.available_samples -= samples_to_read
            
            return result
    
    def available(self) -> int:
        """Get number of available samples."""
        with self.lock:
            return self.available_samples
    
    def free_space(self) -> int:
        """Get number of free buffer slots."""
        with self.lock:
            return self.size - self.available_samples
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.read_pos = 0
            self.available_samples = 0


class VoiceManager:
    """Manager for voice embeddings with preloading and caching."""
    
    def __init__(self, model, preload: bool = True):
        """Initialize voice manager.
        
        Args:
            model: KittenTTS model instance
            preload: Whether to preload voice embeddings
        """
        self.model = model
        self.available_voices = model.available_voices
        self._preloaded_voices = {}
        self._voice_lock = threading.RLock()
        
        if preload:
            self._preload_all_voices()
    
    def _preload_all_voices(self):
        """Preload all available voice embeddings."""
        logger.info("Preloading voice embeddings...")
        start_time = time.time()
        
        with self._voice_lock:
            for voice in self.available_voices:
                if voice in self.model.voices:
                    self._preloaded_voices[voice] = self.model.voices[voice].copy()
        
        load_time = time.time() - start_time
        logger.info(f"Preloaded {len(self._preloaded_voices)} voices in {load_time:.3f}s")
    
    def preload_voice(self, voice: str):
        """Preload a specific voice embedding.
        
        Args:
            voice: Voice name to preload
        """
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available")
        
        with self._voice_lock:
            if voice in self.model.voices:
                self._preloaded_voices[voice] = self.model.voices[voice].copy()
                logger.info(f"Voice '{voice}' preloaded")
    
    def get_voice_embedding(self, voice: str) -> np.ndarray:
        """Get voice embedding, using preloaded version if available.
        
        Args:
            voice: Voice name
            
        Returns:
            Voice embedding array
        """
        with self._voice_lock:
            if voice in self._preloaded_voices:
                return self._preloaded_voices[voice].copy()
            elif voice in self.model.voices:
                # Load on demand
                embedding = self.model.voices[voice].copy()
                self._preloaded_voices[voice] = embedding
                return embedding
            else:
                raise ValueError(f"Voice '{voice}' not found")
    
    def get_voice_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available voices."""
        info = {}
        with self._voice_lock:
            for voice in self.available_voices:
                info[voice] = {
                    'preloaded': voice in self._preloaded_voices,
                    'available': voice in self.model.voices
                }
        return info
    
    def clear_preloaded(self):
        """Clear preloaded voice embeddings to free memory."""
        with self._voice_lock:
            self._preloaded_voices.clear()
        logger.info("Cleared preloaded voices")


class PerformanceProfiler:
    """Performance profiler for monitoring TTS operations."""
    
    def __init__(self, max_samples: int = 1000):
        """Initialize performance profiler.
        
        Args:
            max_samples: Maximum number of timing samples to keep
        """
        self.max_samples = max_samples
        self._timings = defaultdict(deque)
        self._active_timers = {}
        self._counters = defaultdict(int)
        self._lock = threading.RLock()
    
    def start_timing(self, operation: str):
        """Start timing an operation.
        
        Args:
            operation: Operation name
        """
        with self._lock:
            self._active_timers[operation] = time.time()
    
    def end_timing(self, operation: str):
        """End timing an operation.
        
        Args:
            operation: Operation name
        """
        end_time = time.time()
        with self._lock:
            if operation in self._active_timers:
                start_time = self._active_timers.pop(operation)
                duration = end_time - start_time
                
                # Add to timings deque
                if len(self._timings[operation]) >= self.max_samples:
                    self._timings[operation].popleft()
                self._timings[operation].append(duration)
    
    def increment_counter(self, counter: str, value: int = 1):
        """Increment a counter.
        
        Args:
            counter: Counter name
            value: Value to add
        """
        with self._lock:
            self._counters[counter] += value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with timing and counter statistics
        """
        with self._lock:
            stats = {
                'timings': {},
                'counters': dict(self._counters),
                'active_timers': list(self._active_timers.keys())
            }
            
            for operation, times in self._timings.items():
                if times:
                    times_list = list(times)
                    stats['timings'][operation] = {
                        'count': len(times_list),
                        'mean': np.mean(times_list),
                        'std': np.std(times_list),
                        'min': np.min(times_list),
                        'max': np.max(times_list),
                        'recent': times_list[-1] if times_list else 0
                    }
        
        return stats
    
    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self._timings.clear()
            self._active_timers.clear()
            self._counters.clear()
    
    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        
        print("\n=== Performance Statistics ===")
        
        if stats['timings']:
            print("\nTimings:")
            for operation, timing_stats in stats['timings'].items():
                print(f"  {operation}:")
                print(f"    Count: {timing_stats['count']}")
                print(f"    Mean:  {timing_stats['mean']:.4f}s")
                print(f"    Std:   {timing_stats['std']:.4f}s")
                print(f"    Min:   {timing_stats['min']:.4f}s")
                print(f"    Max:   {timing_stats['max']:.4f}s")
        
        if stats['counters']:
            print("\nCounters:")
            for counter, value in stats['counters'].items():
                print(f"  {counter}: {value}")
        
        if stats['active_timers']:
            print(f"\nActive timers: {', '.join(stats['active_timers'])}")


class TextChunker:
    """Intelligent text chunker for optimal TTS processing."""
    
    def __init__(self, 
                 max_chunk_size: int = 200,
                 sentence_endings: str = ".!?",
                 clause_endings: str = ",;:-"):
        """Initialize text chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            sentence_endings: Characters that end sentences
            clause_endings: Characters that end clauses
        """
        self.max_chunk_size = max_chunk_size
        self.sentence_endings = set(sentence_endings)
        self.clause_endings = set(clause_endings)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into optimal chunks for TTS processing.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text.strip()]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            # If sentence alone exceeds max size, split further
            if len(sentence) > self.max_chunk_size:
                # Split long sentence by clauses
                clause_chunks = self._split_long_sentence(sentence)
                for clause_chunk in clause_chunks:
                    if len(current_chunk) + len(clause_chunk) > self.max_chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = clause_chunk
                    else:
                        current_chunk += " " + clause_chunk if current_chunk else clause_chunk
            else:
                # Check if adding this sentence exceeds chunk size
                if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # More sophisticated sentence splitting
        pattern = r'[.!?]+\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence by clauses."""
        chunks = []
        current_chunk = ""
        
        words = sentence.split()
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > self.max_chunk_size and current_chunk:
                # Look for good break point
                if any(end in current_chunk[-5:] for end in self.clause_endings):
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    # Force break at word boundary
                    chunks.append(current_chunk.strip())
                    current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class ResourceMonitor:
    """Monitor system resources during TTS operations."""
    
    def __init__(self, check_interval: float = 1.0):
        """Initialize resource monitor.
        
        Args:
            check_interval: Interval between resource checks in seconds
        """
        self.check_interval = check_interval
        self._monitoring = False
        self._monitor_thread = None
        self._resource_history = deque(maxlen=100)
        
        try:
            import psutil
            self.psutil = psutil
            self.available = True
        except ImportError:
            self.psutil = None
            self.available = False
            logger.warning("psutil not available, resource monitoring disabled")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.available:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stoppe
