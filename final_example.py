#!/usr/bin/env python3
"""
KittenTTS final working example
"""

import sys
import os
import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def save_audio_wav(audio, filename, sample_rate=24000):
    """Save audio file in WAV format"""
    try:
        # Normalize audio data
        audio_normalized = np.int16(audio * 32767)
        
        # Create output folder if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Create WAV file
        with open(f'output/{filename}', 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write((36 + len(audio_normalized) * 2).to_bytes(4, 'little'))
            f.write(b'WAVE')
            
            # fmt chunk
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))
            f.write((1).to_bytes(2, 'little'))  # PCM format
            f.write((1).to_bytes(2, 'little'))  # 1 channel
            f.write(sample_rate.to_bytes(4, 'little'))
            f.write((sample_rate * 2).to_bytes(4, 'little'))  # byte rate
            f.write((2).to_bytes(2, 'little'))  # block align
            f.write((16).to_bytes(2, 'little'))  # bits per sample
            
            # data chunk
            f.write(b'data')
            f.write((len(audio_normalized) * 2).to_bytes(4, 'little'))
            f.write(audio_normalized.tobytes())
        
        print(f"💾 Audio file saved: output/{filename}")
        return True
    except Exception as e:
        print(f"❌ Audio save error: {e}")
        return False

def main():
    print("🎤 KittenTTS final example")
    print("=" * 40)
    
    try:
        # 1. Import KittenTTS
        print("📥 Loading KittenTTS...")
        from kittentts import KittenTTS
        print("✅ KittenTTS loaded!")
        
        # 2. Load model
        print("🤖 Loading model...")
        model = KittenTTS("KittenML/kitten-tts-nano-0.1")
        print("✅ Model loaded!")
        
        # 3. Show available voices
        print(f"\n🎵 Available voices:")
        for i, voice in enumerate(model.available_voices, 1):
            print(f"   {i}. {voice}")
        
        # 4. Test text
        text = "Welcome to the future of text-to-speech! KittenTTS is absolutely incredible - it's fast, lightweight, and produces crystal clear audio quality. This revolutionary AI model is changing the game with just 15 million parameters. Amazing technology!"
        print(f"\n📝 Test text: {text}")
        
        # 5. Generate audio - male voice
        print("\n🔊 Generating audio with male voice...")
        audio_male = model.generate(text, voice='expr-voice-5-m')
        print("✅ Male voice generated!")
        
        # 6. Generate audio - female voice
        print("🔊 Generating audio with female voice...")
        audio_female = model.generate(text, voice='expr-voice-5-f')
        print("✅ Female voice generated!")
        
        # 7. Save audio files
        print("\n💾 Saving audio files...")
        
        if save_audio_wav(audio_male, 'male_voice.wav'):
            print("✅ Male voice saved")
        
        if save_audio_wav(audio_female, 'female_voice.wav'):
            print("✅ Female voice saved")
        
        # 8. Audio information
        print(f"\n📊 Audio information:")
        print(f"   - Male voice length: {len(audio_male)} samples")
        print(f"   - Male voice duration: {len(audio_male)/24000:.2f} seconds")
        print(f"   - Female voice length: {len(audio_female)} samples")
        print(f"   - Female voice duration: {len(audio_female)/24000:.2f} seconds")
        print(f"   - Sample rate: 24000 Hz")
        print(f"   - Format: WAV (16-bit PCM)")
        
        # 9. Success message
        print("\n🎉 KittenTTS is working successfully!")
        print("📁 Created files in output folder:")
        print("   - output/male_voice.wav")
        print("   - output/female_voice.wav")
        print("\n💡 You can open these files in Windows Media Player or any other audio player!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Solution:")
        print("1. Install required libraries:")
        print("   pip install misaki[en]==0.7.4 espeakng_loader")
        print("2. Try again!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please provide information about the error.")

if __name__ == "__main__":
    main() 