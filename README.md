
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

<img width="607" height="255" alt="Screenshot 2026-02-18 at 8 33 04 PM" src="https://github.com/user-attachments/assets/f4646722-ba78-4b25-8a65-81bacee0d4f6" />



> **🎉 ANNOUNCEMENT:** New version of KittenTTS  is now available to download!

###  [Wanna hear samples? Try the model for free on Hugging Face Spaces by clicking this link](https://huggingface.co/spaces/KittenML/KittenTTS-Demo)



Kitten TTS is an open-source realistic text-to-speech model with just 15 million parameters, designed for lightweight deployment and high-quality voice synthesis.

*Currently in developer preview*

[Join our discord](https://discord.com/invite/VJ86W4SURW) 

[Our website](https://kittenml.com) 

[For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

Email the creators with any questions : info@stellonlabs.com


## Features

- **Ultra-lightweight**: Model size less than 25MB
- **CPU-optimized**: Runs without GPU on any device
- **High-quality voices**: Several premium voice options available
- **Fast inference**: Optimized for real-time speech synthesis


## Models

| Model | Params | Size | Link |
|-------|--------|------|------|
| kitten-tts-mini | 80M | 80MB | 🤗 [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) |
| kitten-tts-micro | 40M | 41MB | 🤗 [KittenML/kitten-tts-micro-0.8](https://huggingface.co/KittenML/kitten-tts-micro-0.8) |
| kitten-tts-nano | 15M | 56MB | 🤗 [KittenML/kitten-tts-nano-0.8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) |
| kitten-tts-nano-0.8-int8 | 15M | 25MB | 🤗 [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) |

> Some users are facing minor issues with the kitten-tts-nano-int8  model. We are looking into it. Please report to us if you face any issues. 

## Demo Video


https://github.com/user-attachments/assets/d80120f2-c751-407e-a166-068dd1dd9e8d



## Quick Start

### Installation

```
pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl
```



 ### Basic Usage 

```
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-mini-0.8")

audio = m.generate("This high quality TTS model works without a GPU", voice='Jasper' )

# available_voices : ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)

```





## System Requirements

Works literally everywhere. Needs python3.12. We recommend using conda. 



## Checklist 

- [x] Release a preview model
- [ ] Release the fully trained model weights
- [ ] Release mobile SDK 
- [ ] Release web version 

