"""KittenTTS 2 — the basics."""
import soundfile as sf

from kittenml import KittenTTS

m = KittenTTS("KittenML/kitten-tts-2")

print(m.available_voices)

text = "One day, a little girl named Lily found a needle in her room."
audio = m.generate(text, voice="Bruno")

sf.write("output.wav", audio, m.sample_rate)
