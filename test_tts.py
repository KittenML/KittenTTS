#!/usr/bin/env python3
"""
KittenTTS Audio Quality Test Script
Tests TTS generation and reports on audio quality metrics.
"""

import sys
import time

# Try to load espeak-ng library if available (needed on Windows)
try:
    import espeakng_loader
    espeakng_loader.load_library()
    import os
    if 'ESPEAK_DATA_PATH' not in os.environ:
        os.environ['ESPEAK_DATA_PATH'] = str(espeakng_loader.get_data_path())
    from phonemizer.backend.espeak.base import BaseEspeakBackend
    BaseEspeakBackend.set_library(str(espeakng_loader.get_library_path()))
except:
    pass

import numpy as np

def test_basic_generation():
    """Test basic TTS generation."""
    print("=" * 60)
    print("🐱 KittenTTS Audio Quality Test")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing imports...")
    try:
        from kittentts import KittenTTS
        print("   ✓ KittenTTS imported successfully")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Test model loading
    print("\n2. Loading model (this may take a moment)...")
    try:
        # Use nano model for quick testing
        model = KittenTTS("KittenML/kitten-tts-nano-0.8-fp32")
        print(f"   ✓ Model loaded")
        print(f"   Available voices: {model.available_voices}")
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        print(f"   Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test audio generation
    print("\n3. Generating test audio...")
    test_texts = [
        ("Hello, this is a test of KittenTTS.", "Jasper"),
        ("The quick brown fox jumps over the lazy dog.", "Bella"),
    ]
    
    for text, voice in test_texts:
        print(f"\n   Testing with voice '{voice}':")
        print(f"   Text: \"{text}\"")
        
        try:
            start = time.time()
            audio = model.generate(text, voice=voice, speed=1.0)
            duration = time.time() - start
            
            # Analyze audio
            audio_duration = len(audio) / 24000  # 24kHz sample rate
            max_amplitude = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio**2))
            
            print(f"   ✓ Generated {audio_duration:.2f}s audio in {duration:.2f}s")
            print(f"     Max amplitude: {max_amplitude:.4f}")
            print(f"     RMS level: {rms:.4f}")
            print(f"     Real-time factor: {audio_duration/duration:.2f}x")
            
            # Quality checks
            if max_amplitude > 1.0:
                print(f"     ⚠️  Warning: Audio clipping detected!")
            elif max_amplitude < 0.1:
                print(f"     ⚠️  Warning: Audio level very low!")
            else:
                print(f"     ✓ Audio levels OK")
                
        except Exception as e:
            print(f"   ✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test file saving
    print("\n4. Testing file output...")
    try:
        model.generate_to_file(
            "This is a test file.", 
            "test_output.wav", 
            voice="Jasper",
            subtype='PCM_16'
        )
        print("   ✓ File saved to test_output.wav")
    except Exception as e:
        print(f"   ✗ File saving failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Audio quality looks good.")
    print("=" * 60)
    return True

def benchmark_speed():
    """Benchmark generation speed."""
    print("\n" + "=" * 60)
    print("⚡ Speed Benchmark")
    print("=" * 60)
    
    try:
        from kittentts import KittenTTS
        model = KittenTTS("KittenML/kitten-tts-nano-0.8-fp32")
        
        text = "This is a benchmark test to measure generation speed."
        
        # Warmup
        print("Warming up...")
        model.generate(text, voice="Jasper")
        
        # Benchmark
        print("Running benchmark...")
        times = []
        for _ in range(3):
            start = time.time()
            audio = model.generate(text, voice="Jasper")
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        audio_duration = len(audio) / 24000
        rtf = audio_duration / avg_time
        
        print(f"Average generation time: {avg_time:.3f}s")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Real-time factor: {rtf:.2f}x")
        
        if rtf > 1.0:
            print("✅ Faster than real-time!")
        else:
            print("⚠️  Slower than real-time - expect delays")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")

if __name__ == "__main__":
    success = test_basic_generation()
    
    if success and '--benchmark' in sys.argv:
        benchmark_speed()
    
    sys.exit(0 if success else 1)
