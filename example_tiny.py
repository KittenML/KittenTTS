"""KittenTTS 2 nano / pico / mini: small multilingual models, int8 ONNX (CPU) or float32 PyTorch.

kitten-tts-2-nano and kitten-tts-2-pico hold one model per language (en de es it pt ru hi fr);
kitten-tts-2-mini is one model for 16 languages. Only the files of the languages you use are downloaded.
"""
import soundfile as sf

from kittenml import KittenTTS

m = KittenTTS("KittenML/kitten-tts-2-nano")          # 17M parameters per language
# m = KittenTTS("KittenML/kitten-tts-2-pico")        # 7M, the smallest
# m = KittenTTS("KittenML/kitten-tts-2-mini")        # 82M, 16 languages in one model
# m = KittenTTS("path/to/kitten-tts-2-nano")         # a local copy of the repository
# m = KittenTTS("KittenML/kitten-tts-2-nano", backend="torch")   # the float32 weights in PyTorch

print(m.languages)
print(m.available_voices["en"])                    # ['Eleanor', 'Ruth', 'Maya', ...]
for v in m.voices("de"):
    print(v["name"], v["description"])            # Anna Female 1, ...

audio = m.generate("One day, a little girl named Lily found a needle in her room.", voice="Eleanor")
sf.write("nano_en.wav", audio, m.sample_rate)

# Another language: its first voice, or one of its voices by name (the voice sets the language)
sf.write("nano_de.wav", m.generate("Guten Morgen! Wie geht es dir heute?", language="de"), m.sample_rate)
sf.write("nano_de_paul.wav", m.generate("Guten Morgen! Wie geht es dir heute?", voice="Paul"), m.sample_rate)

# Speed (> 1 is faster), and streaming long text chunk by chunk
sf.write("nano_fast.wav", m.generate("This one is a little faster.", speed=1.2), m.sample_rate)
for i, chunk in enumerate(m.generate_stream("First sentence. Second sentence, a bit longer than the first.")):
    print(f"chunk {i}: {len(chunk) / m.sample_rate:.2f} s")
