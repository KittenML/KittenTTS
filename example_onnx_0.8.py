"""KittenTTS 0.8 — the older ONNX models.

15M to 80M parameters, no cloning and no expression control, but tiny.
"""
import soundfile as sf

from kittenml import KittenTTS

m = KittenTTS("KittenML/kitten-tts-mini-0.8")     # 80M, highest quality
# m = KittenTTS("KittenML/kitten-tts-micro-0.8")  # 40M
# m = KittenTTS("KittenML/kitten-tts-nano-0.8")   # 15M, smallest

print(m.available_voices)

text = "One day, a little girl named Lily found a needle in her room."
audio = m.generate(text, voice="Bruno")
sf.write("output.wav", audio, 24000)

# Speed control, which KittenTTS 2 does not have.
sf.write("slow.wav", m.generate(text, voice="Luna", speed=0.8), 24000)

# ONNX Runtime can use an accelerator, with the matching onnxruntime build.
# m = KittenTTS("KittenML/kitten-tts-mini-0.8", backend="cuda")
