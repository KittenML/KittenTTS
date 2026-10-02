"""KittenTTS 2 — smaller weights, smaller decoder, and decoding knobs."""
import soundfile as sf

from kittenml import KittenTTS

text = "The quick brown fox jumps over the lazy dog."

# Weights: "packed" (954 MB, default, lossless), "emb4" (469 MB, lossy
# embedding), "full" (3469 MB, plain bf16). Only the one you ask for downloads.
m = KittenTTS("KittenML/kitten-tts-2", weights="emb4")

# Decoder: "default" (459 MB) or "student_w4" (39 MB).
m = KittenTTS("KittenML/kitten-tts-2", decoder="student_w4")

# Device is auto-detected; force it if you want to.
m = KittenTTS("KittenML/kitten-tts-2", device="cpu")

# Sampling: a preset, or individual overrides.
audio = m.generate(text, voice="Bruno", preset="expressive")
audio = m.generate(text, voice="Bruno", temperature=0.7, top_p=0.9, top_k=50)

# Seed for a reproducible generation.
audio = m.generate(text, voice="Bruno", advanced={"seed": 1234})
sf.write("output.wav", audio, m.sample_rate)

# Everything `advanced=` accepts, with this model's values.
print(m.advanced_options)

# Skip text normalization, or just see what it would do.
m.generate("Read $12.50 as written.", voice="Bruno", normalize=False)
print(m.normalize_text("Dr. Chen arrives at 9:30 a.m. on the 3rd."))
