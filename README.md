# Kitten TTS 😻

<img width="607" height="255" alt="Screenshot 2026-02-18 at 8 33 04 PM" src="https://github.com/user-attachments/assets/f4646722-ba78-4b25-8a65-81bacee0d4f6" />



> **🎉 ANNOUNCEMENT:** New version of KittenTTS  is now available to download!


Kitten TTS is an open-source realistic text-to-speech model with just 15 million parameters, designed for lightweight deployment and high-quality voice synthesis.

*Currently in developer preview*

[Join our discord](https://discord.com/invite/VJ86W4SURW) 

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
| kitten-tts-nano-int8 quantized | 15M | 19MB | 🤗 [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) |

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



## WebUI

KittenTTS includes a cute kitten-themed web interface for easy text-to-speech generation.

### Quick Start with Conda (Recommended)

```bash
# Create and activate a conda environment
conda create -n kittentts python=3.12 -y
conda activate kittentts

# Install KittenTTS
pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl

# Install additional WebUI dependencies
pip install fastapi uvicorn python-multipart

# Run the WebUI
python run_webui.py
```

### Quick Start with pip

```bash
# Install additional dependencies
pip install fastapi uvicorn python-multipart

# Run the WebUI
python run_webui.py
```

Open your browser and navigate to `http://localhost:7860`

### Features

- **4 Models**: Choose from Mini, Micro, Nano, and Nano INT8 variants
- **8 Voices**: Select from Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, and Leo
- **Speed Control**: Adjust speech speed from 0.5x to 2.0x
- **Dark/Light Mode**: Toggle between themes with automatic system detection
- **Audio Download**: Save generated audio as WAV files

### Command Line Options

```bash
python run_webui.py --host 0.0.0.0 --port 7860
```

## Docker Usage

Run KittenTTS WebUI in a containerized environment.

### Build the Image

```bash
docker build -t kittentts-webui .
```

### Run the Container

```bash
docker run -d -p 7860:7860 -v ~/.cache/huggingface:/root/.cache/huggingface kittentts-webui
```

The `-v` flag mounts the Hugging Face cache directory to persist downloaded models between container restarts.

### Access the WebUI

Open `http://localhost:7860` in your browser.

### Stop the Container

```bash
# List running containers to find the container ID
docker ps

# Stop the container
docker stop <container_id>
```

Or if you ran without `-d` (detached mode), press `Ctrl+C` in the terminal to stop.

## Checklist 

- [x] Release a preview model
- [ ] Release the fully trained model weights
- [ ] Release mobile SDK 
- [ ] Release web version

