# Decoders

How KittenTTS 2 turns the model's codec tokens back into audio, and the
smaller decoders you can swap in.

Audio is decoded in two stages: a flow turns the model's codec tokens into a mel spectrogram,
then a vocoder turns that into audio. The flow can be swapped for a distilled, weight-quantised
student that is much smaller on disk:

```python
m = KittenTTS("KittenML/kitten-tts-2", decoder="student_w4")
```

| Decoder | Flow on disk | Notes |
|---|---|---|
| `default` | 459 MB | The stock flow, two sampling steps. Best quality |
| `student_w4` | 39 MB | Distilled single-step student, weights packed to 4 bits |

`model.available_decoders` lists them. Only the flow changes -- the tokenizer, speaker encoder
and vocoder are shared, so voices and prompts behave the same.

Two things worth knowing before choosing one. Quantisation shrinks storage and memory bandwidth,
not arithmetic, so these are **not meaningfully faster** without a kernel that multiplies in low
precision. And quality is below the default: measured mel L1 against the teacher is 0.250 for the
4-bit student versus 0.168 dense, and voices far from the student's 539-speaker training pool
degrade more than they do on the default decoder. Pick these for footprint.
