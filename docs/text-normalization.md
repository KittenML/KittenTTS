# Text normalization

Numbers, currencies, dates, units and abbreviations are expanded automatically before synthesis,
so `$12.50` is spoken rather than spelled. You can also call it directly:

```python
from kittenml import normalize_text

normalized = normalize_text("Dr. Rivera paid $12.50 at 3:05 p.m.")
# "Doctor Rivera paid twelve dollars and fifty cents at three oh five p m."

result = normalize_text("Fig. 2", return_spans=True)
print(result.text)
print(result.spans)
```

With `return_spans=True` the result also carries original-to-normalized character spans for each
changed segment.

KittenTTS 2 uses [`kitten-text-processing`](https://github.com/KittenML/kitten-text-processing)
for this, which reproduces the normalization the checkpoint was served with. It is pure Python
and pulled in automatically, so there is nothing to compile on any platform.

The `normalize_text` shown above is `kittenml/preprocess.py`, a separate in-repo implementation
that the ONNX models use and that is the one able to return spans.
