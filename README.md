# Kitten TTS 😻

<img width="607" height="255" alt="Screenshot 2026-02-18 at 8 33 04 PM" src="https://github.com/user-attachments/assets/f4646722-ba78-4b25-8a65-81bacee0d4f6" />



> **🎉 ANNOUNCEMENT:** New version of KittenTTS  is now available to download!

###  [Wanna hear samples? Try the model for free on Hugging Face Spaces by clicking this link](https://huggingface.co/spaces/KittenML/KittenTTS-Demo)



Kitten TTS is an open-source realistic text-to-speech model with just 15 million parameters, designed for lightweight deployment and high-quality voice synthesis.

> **UI and enhancement made by Muhammad Umer**

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



```

### Graphical Interface (Windows)

We have provided a native, modern UI designed specifically for Windows! 
To use the UI:
1. Double-click **`Run_KittenTTS.bat`** from your File Explorer.
2. It will automatically build an environment if needed, and open the KittenTTS Desktop app.
3. Type your text, select your voice and model, and click **Generate & Play ▶**!

### Terminal CLI Usage

You can also use KittenTTS directly from your terminal/command line!
```bash
# Generate immediate audio
kittentts "Hello! I am speaking to you directly from the terminal" --voice Jasper --output out.wav

# View all dynamically loaded voices
kittentts --list-voices
```

 ### Basic Python Usage 

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

