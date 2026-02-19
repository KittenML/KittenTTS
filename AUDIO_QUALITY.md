# KittenTTS Audio Quality Optimization Guide

This guide explains how to get the best audio quality from KittenTTS.

## 🎯 Model Selection (Most Important!)

**The #1 factor for audio quality is model precision, not model size.**

### Model Precision Comparison

| Model | Params | Size | Precision | Quality | Use Case |
|-------|--------|------|-----------|---------|----------|
| **Nano (FP32)** | 15M | 56MB | 32-bit float | ⭐⭐⭐⭐⭐ **Best** | Recommended for best quality |
| Mini (INT8) | 80M | 80MB | 8-bit int | ⭐⭐⭐⭐ Good | Long-form content |
| Micro (INT8) | 40M | 41MB | 8-bit int | ⭐⭐⭐ Good | Balanced |
| Nano (INT8) | 15M | 19MB | 8-bit int | ⭐⭐ Basic | Resource-constrained |

### Why FP32 Sounds Better Than Larger INT8 Models

The neural network generates continuous audio waveforms. Precision matters:

- **FP32 (32-bit float)**: Smooth, continuous curves → natural speech
- **INT8 (8-bit integer)**: Stepped approximations → subtle artifacts

A smaller FP32 model (15M params, 56MB) produces smoother audio than a larger INT8 model (80M params, 80MB) because:
1. **No quantization artifacts** - Full precision preserves subtle prosody
2. **Smoother waveforms** - No stepped approximations in output
3. **Better pitch/rhythm** - Floating point preserves continuous variations

**Recommendation: Always use `kitten-tts-nano` (FP32) for best quality.**

## 🔧 Environment Setup

### Required Dependencies

For optimal audio quality, ensure all dependencies are properly installed:

```bash
# Install all dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Critical Components

| Component | Purpose | Quality Impact |
|-----------|---------|----------------|
| **ONNX Runtime** | Model inference engine | High - affects synthesis speed & stability |
| **Phonemizer** | Text → phoneme conversion | Critical - wrong phonemes = gibberish speech |
| **Espeak-ng** | Backend for phonemizer | Critical - version must match training environment |
| **NumPy < 2.0** | Array operations | Medium - version 2.x may have precision issues |

## 🎙️ Best Practices

### 1. Voice Selection

Different voices work better for different content:

| Voice | Gender | Best For | Notes |
|-------|--------|----------|-------|
| **Jasper** | Male | General use, clarity | Most consistent across environments |
| **Bella** | Female | Warm, friendly content | Good for conversational text |
| **Luna** | Female | Soft, melodic speech | Best for poetry/artistic content |
| **Bruno** | Male | Deep, authoritative | Good for announcements |

### 2. Speed Settings

- **1.0x (default)**: Most natural speech
- **0.8-0.9x**: More deliberate, clearer pronunciation
- **1.1-1.2x**: Faster but still natural
- **>1.5x**: May become distorted

### 3. Text Preparation

```python
# Good: Punctuation helps with prosody
text = "Hello! How are you today? I hope you're doing well."

# Avoid: Missing punctuation
text = "hello how are you today i hope youre doing well"

# Good: Numbers written out or let preprocessor handle them
text = "I have 3 cats and 2 dogs."  # Auto-converted to "three" and "two"

# Good: End sentences with punctuation
# This helps the model know when to pause
```

### 4. Audio Output Settings

When saving files, use appropriate bit depth:

```python
# Standard quality (recommended)
model.generate_to_file(text, "output.wav", subtype='PCM_16')

# Higher quality (larger file)
model.generate_to_file(text, "output.wav", subtype='PCM_24')
```

## 🔍 Troubleshooting

### Audio Sounds Robotic/Muffled

**Cause**: Wrong phonemization (espeak version mismatch)

**Fix**:
```bash
# Check espeak version
espeak-ng --version  # Should be 1.51 or later

# Reinstall phonemizer dependencies
pip install --force-reinstall phonemizer espeakng-loader
```

### Audio Has Static/Noise

**Cause**: Clipping or float precision issues

**Fix**:
- Lower the speed slightly (try 0.95x)
- Check NumPy version: `pip install "numpy<2.0"`

### Generation Is Slow

**Cause**: ONNX Runtime not using optimal settings

**Fix**: The model now auto-configures ONNX Runtime. If still slow:
```python
# Check available providers
import onnxruntime as ort
print(ort.get_available_providers())

# Should show ['CPUExecutionProvider'] at minimum
```

### Voice Sounds Wrong (Wrong Pitch/Gender)

**Cause**: Voice embeddings not loading correctly

**Fix**: 
1. Delete cached model: `~/.cache/huggingface/hub/KittenML_*`
2. Re-download: The model will re-download on next use

## 📊 Testing Your Setup

Run the diagnostic script:

```bash
# Check environment
python check_environment.py

# Test audio generation
python test_tts.py

# With speed benchmark
python test_tts.py --benchmark
```

## 🏗️ How It Works

```
Text Input
    ↓
Text Preprocessor (numbers → words, contractions, etc.)
    ↓
Phonemizer (espeak-ng) → IPA phonemes
    ↓
Tokenization → Integer IDs
    ↓
ONNX Model Inference
    ↓
Audio Trimming & Normalization
    ↓
24kHz Mono WAV Output
```

Each step can affect quality. The most critical is **phonemization** - if espeak produces different phonemes than expected, the neural network receives unfamiliar input.

## 🐛 Known Issues

1. **NumPy 2.0+**: May cause subtle audio differences. Stick to 1.24-1.26 for best results.

2. **Windows Espeak**: Sometimes requires manual PATH configuration. The `espeakng-loader` package helps but may need:
   ```python
   import espeakng_loader
   espeakng_loader.load_espeakng_library()
   ```

3. **Long Text**: Automatically chunked at 400 characters. Very long sentences may have slight discontinuities at chunk boundaries.

## 📈 Performance Metrics

On a modern CPU, you should expect:

| Model | Size | RTF (Real-Time Factor) | Quality |
|-------|------|----------------------|---------|
| Nano | 15M | 5-10x | Good |
| Micro | 40M | 3-5x | Better |
| Mini | 80M | 1-2x | Best |

RTF > 1 means faster than real-time (good for streaming).
