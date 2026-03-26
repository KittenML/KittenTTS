import numpy as np
import soundfile as sf
from kittentts import KittenTTS

SAMPLE_RATE = 24000

# Step 1: Load the model
m = KittenTTS("KittenML/kitten-tts-mini-0.8")  # 80M version (highest quality)
# m = KittenTTS("KittenML/kitten-tts-mini-0.8", backend="cuda")  # GPU version
# m = KittenTTS("KittenML/kitten-tts-micro-0.8")  # 40M version
# m = KittenTTS("KittenML/kitten-tts-nano-0.8")  # 15M version (tiny and faster)

# Step 2: Define text and voice
text = """One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to use the needle to sew a button on her shirt. She asked her mom for help."""

voice = "Bruno"

# Optional: try to import sounddevice for real-time playback
try:
    import sounddevice as sd
    has_audio = True
except (ImportError, OSError):
    has_audio = False

# Step 3: Stream audio chunk by chunk
print("Streaming audio...")
chunks = []
for i, chunk in enumerate(m.generate_stream(text=text, voice=voice)):
    audio = chunk.squeeze()
    chunks.append(audio)
    print(f"  Chunk {i + 1}: {len(audio)} samples ({len(audio) / SAMPLE_RATE:.2f}s)")
    if has_audio:
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()

# Save the full audio
full_audio = np.concatenate(chunks)
sf.write("output_streaming.wav", full_audio, SAMPLE_RATE)
print(f"Audio saved to output_streaming.wav ({len(full_audio) / SAMPLE_RATE:.2f}s total)")
