# Kitten TTS (小猫语音)

<p align="center">
  <img width="607" height="255" alt="Kitten TTS" src="https://github.com/user-attachments/assets/f4646722-ba78-4b25-8a65-81bacee0d4f6" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/KittenML/KittenTTS-Demo"><img src="https://img.shields.io/badge/演示-Hugging%20Face%20Spaces-orange" alt="Hugging Face Demo"></a>
  <a href="https://discord.com/invite/VJ86W4SURW"><img src="https://img.shields.io/badge/Discord-加入社区-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://kittenml.com"><img src="https://img.shields.io/badge/官网-kittenml.com-blue" alt="Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/许可证-Apache_2.0-green.svg" alt="License"></a>
</p>

> **最新动态：** Kitten TTS v0.8 已发布 —— 现已提供 15M、40M 和 80M 参数规模的模型。

Kitten TTS 是一个开源、轻量级的语音合成（TTS）库，基于 ONNX 构建。模型参数量从 15M 到 80M 不等（磁盘占用仅 25-80 MB），无需 GPU 即可在 CPU 上实现高质量的语音合成。

> **当前状态：** 开发者预览版 —— API 可能在版本更迭中发生变化。

**提供商业支持：** 如需集成协助、定制音色或企业授权，请 [联系我们](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)。

## 目录

- [特性](#特性)
- [可用模型](#可用模型)
- [演示](#演示)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [系统要求](#系统要求)
- [路线图](#路线图)
- [商业支持](#商业支持)
- [社区与支持](#社区与支持)
- [许可证](#许可证)

## 特性

- **超轻量级** —— 模型大小从 25 MB (int8) 到 80 MB，非常适合边缘设备部署。
- **CPU 优化** —— 基于 ONNX 的推理，无需 GPU 即可高效运行。
- **内置 8 种音色** —— Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, 和 Leo。
- **语速可调** —— 通过 `speed` 参数控制播放速率。
- **文本预处理** —— 内置流水线，自动处理数字、货币、单位等。
- **24 kHz 输出** —— 标准采样率的高质量音频。

## 可用模型

| 模型 | 参数量 | 体积 | 下载地址 |
|---|---|---|---|
| kitten-tts-mini | 80M | 80 MB | [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) |
| kitten-tts-micro | 40M | 41 MB | [KittenML/kitten-tts-micro-0.8](https://huggingface.co/KittenML/kitten-tts-micro-0.8) |
| kitten-tts-nano | 15M | 56 MB | [KittenML/kitten-tts-nano-0.8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) |
| kitten-tts-nano (int8) | 15M | 25 MB | [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) |

> **注意：** 部分用户反馈 `kitten-tts-nano-0.8-int8` 模型存在问题。如果您遇到故障，请 [提交 Issue](https://github.com/KittenML/KittenTTS/issues)。

## 演示

https://github.com/user-attachments/assets/d80120f2-c751-407e-a166-068dd1dd9e8d

### 在线体验

在浏览器中直接通过 [Hugging Face Spaces](https://huggingface.co/spaces/KittenML/KittenTTS-Demo) 体验 Kitten TTS。

## 快速开始

### 前置要求

- Python 3.8 或更高版本
- pip

### 安装

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl
```

### 基础用法

```python
from kittentts import KittenTTS

model = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = model.generate("This high-quality TTS model runs without a GPU.", voice="Jasper")

import soundfile as sf
sf.write("output.wav", audio, 24000)
```

### 进阶用法

```python
# 调整语速 (默认: 1.0)
audio = model.generate("Hello, world.", voice="Luna", speed=1.2)

# 直接保存到文件
model.generate_to_file("Hello, world.", "output.wav", voice="Bruno", speed=0.9)

# 列出可用音色
print(model.available_voices)
# ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
```

## API 参考

### `KittenTTS(model_name, cache_dir=None)`

从 Hugging Face Hub 加载模型。

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `model_name` | `str` | `"KittenML/kitten-tts-nano-0.8"` | Hugging Face 仓库 ID |
| `cache_dir` | `str` | `None` | 下载模型文件的本地缓存目录 |

### `model.generate(text, voice, speed, clean_text)`

将文本合成语音，返回 24 kHz 采样率的 NumPy 音频样本数组。

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `text` | `str` | -- | 待合成的输入文本 |
| `voice` | `str` | `"expr-voice-5-m"` | 音色名称（见可用音色列表） |
| `speed` | `float` | `1.0` | 语速倍率 |
| `clean_text` | `bool` | `False` | 是否预处理文本（展开数字、货币等） |

### `model.generate_to_file(text, output_path, voice, speed, sample_rate, clean_text)`

合成语音并直接写入音频文件。

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `text` | `str` | -- | 待合成的输入文本 |
| `output_path` | `str` | -- | 音频文件保存路径 |
| `voice` | `str` | `"expr-voice-5-m"` | 音色名称 |
| `speed` | `float` | `1.0` | 语速倍率 |
| `sample_rate` | `int` | `24000` | 音频采样率 (Hz) |
| `clean_text` | `bool` | `True` | 是否预处理文本 |

## 系统要求

- **操作系统：** Linux, macOS, 或 Windows
- **Python：** 3.8 或更高版本
- **硬件：** 在 CPU 上运行；无需 GPU
- **磁盘空间：** 取决于模型版本，约 25-80 MB

建议使用虚拟环境（如 conda, venv）以避免依赖冲突。

## 路线图

- [ ] 发布优化后的推理引擎
- [ ] 发布移动端 SDK
- [ ] 发布更高质量的 TTS 模型
- [ ] 发布多语言 TTS 支持
- [ ] 发布 KittenASR（自动语音识别）
- [ ] 还有其他需求？[请告诉我们](https://github.com/KittenML/KittenTTS/issues)

## 商业支持

我们为寻求将 Kitten TTS 集成到产品中的团队提供商业支持，包括集成协助、定制音色开发和企业级授权。

[联系我们](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview) 或发送邮件至 info@stellonlabs.com 讨论您的需求。

## 社区与支持

- **Discord:** [加入社区](https://discord.gg/VJ86W4SURW)
- **官网:** [kittenml.com](https://kittenml.com)
- **定制支持:** [申请表单单](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
- **GitHub Issues:** [问题反馈](https://github.com/KittenML/KittenTTS/issues)

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 协议开源。
