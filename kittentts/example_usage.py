#!/usr/bin/env python3
"""
Complete example demonstrating real-time KittenTTS capabilities.
This example showcases all the enhanced features including streaming,
real-time processing, performance monitoring, and adaptive quality.
"""

import time
import threading
import numpy as np
from pathlib import Path
from typing import List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import enhanced KittenTTS components
from kittentts import (
    KittenTTS, 
    RealtimeKittenTTS, 
    StreamingKittenTTS,
    PerformanceProfiler,
    ResourceMonitor,
    AdaptiveQualityManager,
    CacheManager
)


def demo_basic_usage():
    """Demonstrate basic enhanced TTS usage."""
    print("\n" + "="*50)
    print("DEMO 1: Basic Enhanced Usage")
    print("="*50)
    
    # Initialize with real-time optimizations
    tts = KittenTTS(
        model_name="KittenML/kitten-tts-nano-0.1",
        realtime_mode=True,
        max_workers=4,
        preload_voices=True,
        enable_profiling=True
    )
    
    # Warm up the model
    print("Warming up model...")
    tts.warm_up()
    
    # Single generation
    text = "Hello! This is a demonstration of real-time text-to-speech synthesis."
    
    print(f"Generating: '{text}'")
    start_time = time.time()
    audio = tts.generate(text, voice="expr-voice-5-m", speed=1.0)
    generation_time = time.time() - start_time
    
    print(f"Generated {len(audio)} samples in {generation_time:.4f}s")
    print(f"Real-time factor: {len(audio)/24000/generation_time:.2f}x")
    
    # Batch generation
    texts = [
        "First sentence for batch processing.",
        "Second sentence demonstrates parallel generation.",
        "Third sentence shows the power of concurrent synthesis."
    ]
    
    print("\nBatch generation...")
    start_time = time.time()
    batch_audio = tts.generate(texts, voice="expr-voice-5-f", speed=1.1)
    batch_time = time.time() - start_time
    
    total_samples = sum(len(audio) for audio in batch_audio)
    print(f"Generated {len(batch_audio)} clips ({total_samples} samples) in {batch_time:.4f}s")
    
    # Performance statistics
    stats = tts.get_performance_stats()
    if stats:
        print(f"\nPerformance stats: {stats}")


def demo_streaming_synthesis():
    """Demonstrate streaming TTS synthesis."""
    print("\n" + "="*50)
    print("DEMO 2: Streaming Synthesis")
    print("="*50)
    
    # Initialize with streaming capabilities
    tts = KittenTTS(
        realtime_mode=True,
        max_workers=6,
        enable_profiling=True
    )
    
    streaming_tts = StreamingKittenTTS(tts.realtime_engine)
    
    # Long text for streaming
    long_text = """
    Streaming text-to-speech synthesis represents a significant advancement in voice technology.
    By processing text in real-time chunks, we can provide immediate audio feedback for 
    interactive applications. This approach is particularly valuable for virtual assistants,
    live reading applications, and accessibility tools. The system intelligently breaks down
    text into optimal segments while maintaining natural speech flow and minimizing latency.
    Advanced caching mechanisms ensure that frequently used phrases are generated instantly,
    while adaptive quality management maintains optimal performance across different hardware
    configurations. This combination of technologies enables truly responsive voice interfaces
    that can keep pace with natural human conversation patterns.
    """
    
    print("Starting streaming synthesis...")
    print("Text length:", len(long_text), "characters")
    
    # Stream to file
    output_file = "streaming_demo.wav"
    start_time = time.time()
    
    streaming_tts.stream_to_file(
        long_text.strip(), 
        output_file,
        voice="expr-voice-4-m",
        speed=1.0,
        chunk_size=80
    )
    
    streaming_time = time.time() - start_time
    print(f"Streaming synthesis completed in {streaming_time:.4f}s")
    
    # Demonstrate real-time streaming with callback
    print("\nDemonstrating chunk-by-chunk streaming...")
    
    chunk_count = 0
    total_samples = 0
    
    def audio_callback(chunk: np.ndarray):
        nonlocal chunk_count, total_samples
        chunk_count += 1
        total_samples += len(chunk)
        print(f"  Received chunk {chunk_count}: {len(chunk)} samples")
    
    # Stream with callback
    for i, audio_chunk in enumerate(tts.generate_streaming(
        "This demonstrates real-time streaming synthesis with immediate chunk delivery.",
        voice="expr-voice-3-f",
        speed=1.0,
        chunk_size=60
    )):
        audio_callback(audio_chunk)
        # Simulate real-time playback delay
        time.sleep(0.1)
    
    print(f"Streamed {chunk_count} chunks with {total_samples} total samples")


def demo_performance_monitoring():
    """Demonstrate performance monitoring and adaptive quality."""
    print("\n" + "="*50)
    print("DEMO 3: Performance Monitoring & Adaptive Quality")
    print("="*50)
    
    # Initialize with full monitoring
    profiler = PerformanceProfiler()
    resource_monitor = ResourceMonitor()
    cache_manager = CacheManager()
    
    tts = KittenTTS(
        realtime_mode=True,
        enable_profiling=True
    )
    
    # Start monitoring
    resource_monitor.start_monitoring()
    
    # Initialize adaptive quality manager
    quality_manager = AdaptiveQualityManager(
        target_latency=0.3,
        resource_monitor=resource_monitor
    )
    
    print("Running performance test with adaptive quality...")
    
    test_texts = [
        "Short text for quick generation.",
        "This is a medium-length sentence that requires moderate processing time and resources.",
        "This is a much longer sentence that will take significantly more time to process and will help demonstrate the adaptive quality management system in action by providing a substantial computational load.",
        "Quick test again.",
        "Another medium sentence for consistent testing patterns.",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text[:50]}...")
        
        # Record performance
        profiler.start_timing("generation")
        start_time = time.time()
        
        # Get current quality settings
        settings = quality_manager.get_current_settings()
        print(f"Quality level: {quality_manager.current_quality}")
        print(f"Settings: {settings}")
        
        # Generate with current settings
        audio = tts.generate(text, voice="expr-voice-5-m", speed=1.0)
        
        latency = time.time() - start_time
        profiler.end_timing("generation")
        
        # Record performance for adaptive adjustment
        quality_manager.record_performance(latency)
        
        print(f"Generation time: {latency:.4f}s")
        print(f"Audio samples: {len(audio)}")
        
        # Check if quality should be adjusted
        if quality_manager.should_adjust_quality():
            new_quality = quality_manager.adjust_quality()
            print(f"Quality adjusted to: {new_quality}")
        
        # Show resource usage
        if resource_monitor.available:
            usage = resource_monitor.get_current_usage()
            if usage:
                print(f"CPU: {usage['cpu_percent']:.1f}%, Memory: {usage['memory_percent']:.1f}%")
    
    # Stop monitoring
    resource_monitor.stop_monitoring()
    
    # Show final statistics
    print("\n" + "-"*30)
    print("FINAL STATISTICS")
    print("-"*30)
    
    # Performance stats
    perf_summary = quality_manager.get_performance_summary()
    print(f"Performance Summary: {perf_summary}")
    
    # Profiler stats
    profiler.print_stats()
    
    # Resource usage history
    if resource_monitor.available:
        usage_stats = resource_monitor.get_usage_stats()
        print(f"Resource Usage Stats: {usage_stats}")


def demo_advanced_caching():
    """Demonstrate advanced caching capabilities."""
    print("\n" + "="*50)
    print("DEMO 4: Advanced Caching System")
    print("="*50)
    
    # Initialize with caching
    tts = KittenTTS(
        realtime_mode=True,
        enable_profiling=True
    )
    
    cache_manager = CacheManager(
        max_audio_cache_size=10,
        max_phoneme_cache_size=20,
        cache_ttl=60.0  # 1 minute for demo
    )
    
    # Test texts with some repetition
    test_texts = [
        "Hello, welcome to the caching demonstration.",
        "This text will be cached for faster subsequent access.",
        "Hello, welcome to the caching demonstration.",  # Repeat
        "Cache performance is crucial for real-time applications.",
        "This text will be cached for faster subsequent access.",  # Repeat
        "New unique text that hasn't been cached before.",
        "Cache performance is crucial for real-time applications.",  # Repeat
    ]
    
    print("Testing cache performance...")
    print(f"Total test texts: {len(test_texts)}")
    print(f"Unique texts: {len(set(test_texts))}")
    
    generation_times = []
    
    for i, text in enumerate(test_texts):
        print(f"\nGeneration {i+1}: {text[:30]}...")
        
        # Check if in cache first
        cache_key = f"{hash(text)}_expr-voice-5-m_1.0"
        cached_audio = cache_manager.get_audio(text, "expr-voice-5-m", 1.0)
        
        if cached_audio is not None:
            print("  ✓ Found in cache!")
            generation_time = 0.001  # Minimal cache lookup time
        else:
            print("  ✗ Not in cache, generating...")
            start_time = time.time()
            audio = tts.generate(text, voice="expr-voice-5-m", speed=1.0)
            generation_time = time.time() - start_time
            
            # Store in cache
            cache_manager.store_audio(text, "expr-voice-5-m", 1.0, audio)
        
        generation_times.append(generation_time)
        print(f"  Time: {generation_time:.4f}s")
    
    # Show cache statistics
    cache_stats = cache_manager.get_stats()
    print(f"\nCache Statistics: {cache_stats}")
    
    # Show timing comparison
    avg_cached_time = np.mean([t for t in generation_times if t < 0.01])
    avg_generated_time = np.mean([t for t in generation_times if t >= 0.01])
    
    print(f"\nTiming Analysis:")
    print(f"Average cached lookup: {avg_cached_time:.4f}s")
    print(f"Average generation: {avg_generated_time:.4f}s")
    print(f"Cache speedup: {avg_generated_time/avg_cached_time:.1f}x faster")


def demo_voice_management():
    """Demonstrate voice management and switching."""
    print("\n" + "="*50)
    print("DEMO 5: Voice Management & Fast Switching")
    print("="*50)
    
    tts = KittenTTS(
        realtime_mode=True,
        preload_voices=True,
        enable_profiling=True
    )
    
    print("Available voices:", tts.available_voices)
    print("Voice manager info:", tts.voice_manager.get_voice_info())
    
    # Test text for voice comparison
    test_text = "This voice comparison demonstrates the different vocal characteristics available."
    
    voice_times = {}
    
    for voice in tts.available_voices[:4]:  # Test first 4 voices
        print(f"\nTesting voice: {voice}")
        
        # Time the generation
        start_time = time.time()
        audio = tts.generate(test_text, voice=voice, speed=1.0)
        generation_time = time.time() - start_time
        
        voice_times[voice] = generation_time
        
        print(f"  Generation time: {generation_time:.4f}s")
        print(f"  Audio samples: {len(audio)}")
        
        # Save sample for comparison
        tts.generate_to_file(
            test_text, 
            f"voice_sample_{voice}.wav", 
            voice=voice, 
            speed=1.0
        )
    
    print(f"\nVoice switching performance:")
    for voice, time_taken in voice_times.items():
        print(f"  {voice}: {time_taken:.4f}s")
    
    # Test rapid voice switching
    print(f"\nRapid voice switching test...")
    voices_to_test = tts.available_voices[:3]
    switch_text = "Quick voice switch test."
    
    start_time = time.time()
    for i in range(6):  # 2 rounds of each voice
        voice = voices_to_test[i % len(voices_to_test)]
        audio = tts.generate(switch_text, voice=voice, speed=1.0)
        print(f"  Switch {i+1}: {voice} -> {len(audio)} samples")
    
    total_switch_time = time.time() - start_time
    print(f"Total rapid switching time: {total_switch_time:.4f}s")
    print(f"Average per switch: {total_switch_time/6:.4f}s")


def demo_concurrent_processing():
    """Demonstrate concurrent multi-threaded processing."""
    print("\n" + "="*50)
    print("DEMO 6: Concurrent Processing")
    print("="*50)
    
    tts = KittenTTS(
        realtime_mode=True,
        max_workers=8,
        enable_profiling=True
    )
    
    # Prepare multiple texts for concurrent processing
    concurrent_texts = [
        "First concurrent text processing task with moderate length content.",
        "Second parallel processing demonstration showing multi-threading capabilities.",
        "Third simultaneous generation test for performance evaluation purposes.",
        "Fourth concurrent synthesis task demonstrating scalable voice generation.",
        "Fifth parallel text-to-speech conversion showcasing efficient resource utilization.",
        "Sixth simultaneous audio synthesis demonstrating robust concurrent processing.",
        "Seventh parallel generation task showing consistent performance across threads.",
        "Eighth concurrent processing demonstration with reliable multi-threaded execution."
    ]
    
    print(f"Processing {len(concurrent_texts)} texts concurrently...")
    
    # Sequential processing for comparison
    print("\nSequential processing:")
    sequential_start = time.time()
    sequential_results = []
    for i, text in enumerate(concurrent_texts):
        audio = tts.generate(text, voice="expr-voice-5-m", speed=1.0)
        sequential_results.append(audio)
        print(f"  Completed {i+1}/{len(concurrent_texts)}")
    sequential_time = time.time() - sequential_start
    
    print(f"Sequential time: {sequential_time:.4f}s")
    
    # Concurrent processing
    print("\nConcurrent processing:")
    concurrent_start = time.time()
    concurrent_results = tts.generate(
        concurrent_texts, 
        voice="expr-voice-5-m", 
        speed=1.0
    )
    concurrent_time = time.time() - concurrent_start
    
    print(f"Concurrent time: {concurrent_time:.4f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.2f}x faster")
    print(f"Efficiency: {(sequential_time/concurrent_time)/tts.max_workers*100:.1f}%")
    
    # Verify results
    total_sequential_samples = sum(len(audio) for audio in sequential_results)
    total_concurrent_samples = sum(len(audio) for audio in concurrent_results)
    
    print(f"Sequential samples: {total_sequential_samples}")
    print(f"Concurrent samples: {total_concurrent_samples}")
    print(f"Results match: {total_sequential_samples == total_concurrent_samples}")


def demo_real_world_scenario():
    """Demonstrate a real-world interactive scenario."""
    print("\n" + "="*50)
    print("DEMO 7: Real-World Interactive Scenario")
    print("="*50)
    
    # Initialize full-featured TTS system
    profiler = PerformanceProfiler()
    resource_monitor = ResourceMonitor()
    
    tts = KittenTTS(
        realtime_mode=True,
        max_workers=4,
        enable_profiling=True
    )
    
    # Start monitoring
    resource_monitor.start_monitoring()
    
    # Simulate interactive dialogue system
    dialogue_exchanges = [
        {
            "user_input": "What's the weather like today?",
            "system_response": "I don't have access to current weather data, but I can help you with text-to-speech synthesis instead!",
            "voice": "expr-voice-5-f",
            "speed": 1.0
        },
        {
            "user_input": "Can you speak faster?",
            "system_response": "Absolutely! I can adjust my speaking speed to match your preferences. How's this pace?",
            "voice": "expr-voice-5-f",
            "speed": 1.3
        },
        {
            "user_input": "Try a different voice",
            "system_response": "Sure thing! I'll switch to a different voice. This voice has different characteristics and tone.",
            "voice": "expr-voice-4-m",
            "speed": 1.0
        },
        {
            "user_input": "Tell me about real-time TTS",
            "system_response": "Real-time text-to-speech synthesis enables immediate voice responses for interactive applications. It processes text chunks efficiently while maintaining natural speech flow and minimal latency.",
            "voice": "expr-voice-3-m",
            "speed": 0.9
        }
    ]
    
    print("Simulating interactive dialogue system...")
    
    total_response_time = 0
    total_audio_duration = 0
    
    for i, exchange in enumerate(dialogue_exchanges):
        print(f"\nExchange {i+1}:")
        print(f"User: {exchange['user_input']}")
        print(f"System: {exchange['system_response'][:50]}...")
        
        # Measure response generation time
        response_start = time.time()
        
        # Generate response audio
        audio = tts.generate(
            exchange['system_response'],
            voice=exchange['voice'],
            speed=exchange['speed']
        )
        
        response_time = time.time() - response_start
        audio_duration = len(audio) / 24000  # Assuming 24kHz sample rate
        
        total_response_time += response_time
        total_audio_duration += audio_duration
        
        print(f"  Voice: {exchange['voice']}, Speed: {exchange['speed']}")
        print(f"  Response time: {response_time:.4f}s")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Real-time factor: {audio_duration/response_time:.2f}x")
        
        # Show current resource usage
        if resource_monitor.available:
            usage = resource_monitor.get_current_usage()
            if usage:
                print(f"  CPU: {usage['cpu_percent']:.1f}%, Memory: {usage['memory_percent']:.1f}%")
        
        # Simulate brief pause between exchanges
        time.sleep(0.5)
    
    # Stop monitoring
    resource_monitor.stop_monitoring()
    
    # Final statistics
    print(f"\n" + "-"*40)
    print("DIALOGUE SESSION SUMMARY")
    print("-"*40)
    print(f"Total exchanges: {len(dialogue_exchanges)}")
    print(f"Total response time: {total_response_time:.4f}s")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Average response time: {total_response_time/len(dialogue_exchanges):.4f}s")
    print(f"Overall real-time factor: {total_audio_duration/total_response_time:.2f}x")
    
    # Performance statistics
    stats = tts.get_performance_stats()
    if stats:
        print(f"Performance stats: {stats}")


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark of all features."""
    print("\n" + "="*60)
    print("COMPREHENSIVE BENCHMARK")
    print("="*60)
    
    # Initialize systems
    profiler = PerformanceProfiler()
    resource_monitor = ResourceMonitor()
    
    # Test different configurations
    configurations = [
        {"realtime_mode": False, "max_workers": 1, "name": "Basic"},
        {"realtime_mode": True, "max_workers": 2, "name": "RT-2Workers"},
        {"realtime_mode": True, "max_workers": 4, "name": "RT-4Workers"},
        {"realtime_mode": True, "max_workers": 8, "name": "RT-8Workers"},
    ]
    
    benchmark_text = """
    This is a comprehensive benchmark text designed to test the performance 
    and capabilities of the enhanced KittenTTS system. It includes various 
    sentence structures, punctuation marks, and length variations to provide 
    a realistic evaluation of synthesis quality and speed. The system should 
    handle this text efficiently while maintaining high-quality output.
    """
    
    benchmark_results = []
    
    # Start resource monitoring
    resource_monitor.start_monitoring()
    
    for config in configurations:
        print(f"\nTesting configuration: {config['name']}")
        print(f"Settings: {config}")
        
        # Initialize TTS with current configuration
        tts = KittenTTS(
            realtime_mode=config['realtime_mode'],
            max_workers=config['max_workers'],
            enable_profiling=True
        )
        
        # Warm up
        tts.warm_up()
        
        # Run multiple iterations
        iteration_times = []
        for iteration in range(3):
            start_time = time.time()
            audio = tts.generate(benchmark_text.strip(), voice="expr-voice-5-m", speed=1.0)
            iteration_time = time.time() - start_time
            iteration_times.append(iteration_time)
            
            print(f"  Iteration {iteration+1}: {iteration_time:.4f}s ({len(audio)} samples)")
        
        # Calculate statistics
        avg_time = np.mean(iteration_times)
        std_time = np.std(iteration_times)
        min_time = np.min(iteration_times)
        max_time = np.max(iteration_times)
        
        # Get performance stats
        perf_stats = tts.get_performance_stats()
        
        result = {
            'config': config['name'],
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'perf_stats': perf_stats
        }
        
        benchmark_results.append(result)
        
        print(f"  Average: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Range: {min_time:.4f}s - {max_time:.4f}s")
    
    # Stop monitoring
    resource_monitor.stop_monitoring()
    
    # Display comparison
    print(f"\n" + "="*40)
    print("BENCHMARK COMPARISON")
    print("="*40)
    
    print(f"{'Configuration':<15} {'Avg Time':<10} {'Min Time':<10} {'Max Time':<10} {'Speedup':<8}")
    print("-" * 60)
    
    baseline_time = benchmark_results[0]['avg_time']
    
    for result in benchmark_results:
        speedup = baseline_time / result['avg_time']
        print(f"{result['config']:<15} {result['avg_time']:<10.4f} {result['min_time']:<10.4f} {result['max_time']:<10.4f} {speedup:<8.2f}x")
    
    # Resource usage summary
    if resource_monitor.available:
        usage_stats = resource_monitor.get_usage_stats()
        print(f"\nResource Usage During Benchmark:")
        print(f"CPU - Mean: {usage_stats.get('cpu', {}).get('mean', 0):.1f}%, Max: {usage_stats.get('cpu', {}).get('max', 0):.1f}%")
        print(f"Memory - Mean: {usage_stats.get('memory', {}).get('mean', 0):.1f}%, Max: {usage_stats.get('memory', {}).get('max', 0):.1f}%")


def main():
    """Run all demonstrations and benchmarks."""
    print("KittenTTS Enhanced Real-time Synthesis Demonstration")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demo_basic_usage()
        demo_streaming_synthesis() 
        demo_performance_monitoring()
        demo_advanced_caching()
        demo_voice_management()
        demo_concurrent_processing()
        demo_real_world_scenario()
        
        # Run comprehensive benchmark
        run_comprehensive_benchmark()
        
        print(f"\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nGenerated audio files:")
        audio_files = [
            "streaming_demo.wav",
            "voice_sample_expr-voice-2-m.wav", 
            "voice_sample_expr-voice-2-f.wav",
            "voice_sample_expr-voice-3-m.wav",
            "voice_sample_expr-voice-3-f.wav"
        ]
        
        for file in audio_files:
            if Path(file).exists():
                print(f"  ✓ {file}")
            else:
                print(f"  - {file} (not generated)")
    
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise
    
    finally:
        print(f"\nDemonstration complete. Check generated audio files for results.")


if __name__ == "__main__":
    main()
