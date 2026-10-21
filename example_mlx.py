"""KittenTTS 2 on an Apple-silicon Mac — the same API, on the GPU.

    pip install "kittenml[mlx]"

The language model runs through MLX with its ternary weights left packed, and
audio decoding runs on MPS. Everything else — voices, cloning, expression tags,
streaming — is unchanged; see the other examples.
"""
import time

import soundfile as sf

from kittenml import KittenTTS

# backend="auto" is the portable spelling: MLX on an Apple-silicon Mac with mlx
# installed, and the default PyTorch backend anywhere else.
m = KittenTTS("KittenML/kitten-tts-2", backend="auto")
print(f"backend: {m.backend}, audio decoding on: {m.device}")

text = "One day, a little girl named Lily found a needle in her room."

# The first generation also compiles the Metal kernels both backends need; time
# the second to see the steady-state rate.
m.generate(text, voice="Bruno")

start = time.time()
audio = m.generate(text, voice="Bruno")
elapsed = time.time() - start
seconds = len(audio) / m.sample_rate
print(f"{seconds:.2f}s of audio in {elapsed:.2f}s -> {seconds / elapsed:.1f}x realtime")

sf.write("output.wav", audio, m.sample_rate)
