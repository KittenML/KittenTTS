"""KittenTTS 2 — get each chunk as soon as it is generated."""
import numpy as np
import soundfile as sf

from kittenml import KittenTTS

m = KittenTTS("KittenML/kitten-tts-2")

text = ("One day, a little girl named Lily found a needle in her room. "
        "She knew it was difficult to play with it because it was sharp. "
        "Lily wanted to use the needle to sew a button on her shirt. "
        "She asked her mom for help, and her mom smiled and said yes. "
        "They sat together by the window, where the afternoon light was best, "
        "and her mom showed her how to hold the fabric steady with one hand.")

# Each chunk is a 1-D float32 array. Text under ~380 characters yields once.
chunks = []
for chunk in m.generate_stream(text, voice="Luna"):
    print(f"{len(chunk) / m.sample_rate:.1f}s")
    chunks.append(chunk)

sf.write("output_streaming.wav", np.concatenate(chunks), m.sample_rate)
