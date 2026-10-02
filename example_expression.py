"""KittenTTS 2 — expression tags and other languages.

Expression control is in beta: it steers delivery rather than guaranteeing it.
"""
import soundfile as sf

from kittenml import KittenTTS

m = KittenTTS("KittenML/kitten-tts-2")

# A leading [emotion], inline <events>, and (((emphasis))) spans.
audio = m.generate(
    "[joyful] We won the grant <laugh> I can (((hardly))) believe it!",
    voice="Kiki",
    preset="expressive",   # "stable" is the default
)
sf.write("expressive.wav", audio, m.sample_rate)

# Emotions: angry contemplative excited joyful mundane
#           nervous sad stern surprised tender
# Events:   gasp giggle growl gulp laugh
#           pause scoff sigh sob um

# Other languages: the voice carries the accent. normalize=False because the
# text normalizer is English-tuned.
audio = m.generate("Guten Morgen. Ich wünsche dir einen wunderschönen Tag.",
                   voice="German", normalize=False)
sf.write("german.wav", audio, m.sample_rate)

# Also: Arabic, Spanish, French, Italian, Portuguese, Russian, Chinese, Hindi.
