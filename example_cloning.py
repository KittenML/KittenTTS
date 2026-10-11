"""KittenTTS 2 — clone a voice from 5-30 seconds of a single speaker."""
import soundfile as sf

from kittenml import KittenTTS

m = KittenTTS("KittenML/kitten-tts-2")

# Set reference_language="en" or "ru" if the recording's language is known.
audio = m.generate("This is my own voice, cloned.", reference="my_voice.wav")
sf.write("cloned.wav", audio, m.sample_rate)

# Supplying the transcript skips Whisper.
audio = m.generate("The transcript is optional, but it helps.",
                   reference="my_voice.wav",
                   reference_text="What the recording actually says.")
sf.write("cloned2.wav", audio, m.sample_rate)
