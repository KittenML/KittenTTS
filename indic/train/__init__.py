"""Training for the Indic adaptation, and the loader that keeps it honest.

Everything under `indic/train/` has to agree with the inference package on three
things: the token layout (`tokens.py`), how a prompt is assembled (`prompt.py`)
and how ternary weights are packed (`ternary.py`). A training pipeline that
reimplements any of them will drift from what the checkpoint was trained to see,
and the failure is silent -- the loss still goes down, it just goes down on the
wrong sequence.

So those three are imported, never copied. The ordinary import runs
`kittenml/kittentts2/__init__.py`, which pulls in the whole TTS stack -- the
vocoder, diffusers, librosa, the text normalizer -- to reach three files that
import nothing but `json` and `torch`. A training box holds precomputed shards
and no audio libraries, so that import fails there for no reason at all.
Loading by file path keeps the agreement without the dependency, the same way
`indic/bench/ternary_check.py` does.
"""

import importlib.util
import pathlib

_KITTEN2 = pathlib.Path(__file__).resolve().parents[2] / "kittenml" / "kittentts2"
_loaded = {}


def load_kittentts2(name):
    """Import one dependency-free module of the inference package by file path.

    Only for modules that stand alone: `tokens`, `prompt`, `ternary`. Anything
    that imports its siblings will fail here, which is the point -- it means the
    module was not safe to load this way.
    """
    if name not in _loaded:
        spec = importlib.util.spec_from_file_location(f"_kitten2_{name}",
                                                      _KITTEN2 / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded[name] = module
    return _loaded[name]
