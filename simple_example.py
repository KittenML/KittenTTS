from kittentts import KittenTTS
import numpy as np

# Load model
model = KittenTTS("KittenML/kitten-tts-nano-0.1")

# Convert text to speech
# text = "Hello! This is KittenTTS working perfectly."
text = "Welcome to the future of text-to-speech! KittenTTS is absolutely incredible - it's fast, lightweight, and produces crystal clear audio quality. This revolutionary AI model is changing the game with just 15 million parameters. Amazing technology!"
audio = model.generate(text, voice='expr-voice-5-m')

# Save audio file
audio_normalized = np.int16(audio * 32767)

# Create output folder if it doesn't exist
import os
os.makedirs('output', exist_ok=True)

# Create WAV file
with open('output/output.wav', 'wb') as f:
    # RIFF header
    f.write(b'RIFF')
    f.write((36 + len(audio_normalized) * 2).to_bytes(4, 'little'))
    f.write(b'WAVE')
    
    # fmt chunk
    f.write(b'fmt ')
    f.write((16).to_bytes(4, 'little'))
    f.write((1).to_bytes(2, 'little'))
    f.write((1).to_bytes(2, 'little'))
    f.write((24000).to_bytes(4, 'little'))
    f.write((48000).to_bytes(4, 'little'))
    f.write((2).to_bytes(2, 'little'))
    f.write((16).to_bytes(2, 'little'))
    
    # data chunk
    f.write(b'data')
    f.write((len(audio_normalized) * 2).to_bytes(4, 'little'))
    f.write(audio_normalized.tobytes())

print("✅ Audio file created: output/output.wav")