
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=it">Italiano</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=id">Bahasa Indonesia</a>
        | <a href="https://openaitx.github.io/view.html?user=KittenML&project=KittenTTS&lang=as">অসমীয়া</
      </div>
    </div>
  </details>
</div>

# Kitten TTS 😻

Kitten TTS is an open-source realistic text-to-speech model with just 15 million parameters, designed for lightweight deployment and high-quality voice synthesis.

*Currently in developer preview*

[Join our discord](https://discord.com/invite/VJ86W4SURW)

[For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

Email the creators with any questions : info@stellonlabs.com


## ✨ Features

- **Ultra-lightweight**: Model size less than 25MB
- **CPU-optimized**: Runs without GPU on any device
- **High-quality voices**: Several premium voice options available
- **Fast inference**: Optimized for real-time speech synthesis



## 🚀 Quick Start

### Installation

```
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```



 ### Basic Usage 

```
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)

```





## 💻 System Requirements

Works literally everywhere



## Checklist 

- [x] Release a preview model
- [ ] Release the fully trained model weights
- [ ] Release mobile SDK 
- [ ] Release web version 

