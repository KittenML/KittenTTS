"""Read the selected checkpoint identically for Torch and vLLM."""

import json
from pathlib import Path


def available_weights(config):
    return ["full", *(name for name, key in (("packed", "lm_packed"), ("emb4", "lm_emb4"))
                      if config.get(key))]


def weights_path(repo_dir, config, name):
    if name == "full":
        return None
    key = {"packed": "lm_packed", "emb4": "lm_emb4"}.get(name)
    if key is None:
        raise ValueError(f"Unknown weights {name!r}. Choose from: {available_weights(config)}")
    declared = config.get(key)
    if not declared:
        raise ValueError(f"this repository declares no {name!r} weights")
    return Path(repo_dir) / declared


def plain_shards(repo_dir, config, lm_dir=None):
    """The bf16 ("full") safetensors files of the language model, in load order."""
    lm_dir = Path(lm_dir) if lm_dir is not None else Path(repo_dir) / config.get("lm_dir", "lm")
    return lm_dir, [s for s in sorted(lm_dir.glob("model*.safetensors")) if "-" not in s.stem]


def load_lm_weights(repo_dir, config, variant, lm_dir=None, include=None):
    """Unpack on CPU without re-quantizing or writing an expanded checkpoint.

    `include`, if given, is a predicate on tensor names: only those tensors are
    read (and, for packed files, unpacked), so callers that need a few tensors
    do not expand the whole model in host memory.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = weights_path(repo_dir, config, variant)
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"{variant!r} weights not found: {path}")
        if variant == "emb4":
            from .tl2_emb4 import load_state_dict
            return load_state_dict(path, include)
        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata() or {}
            if include is None:
                tensors = {k: handle.get_tensor(k) for k in handle.keys()}
            else:
                tensors = {k: handle.get_tensor(k) for k in handle.keys()
                           if include(k.split("::", 1)[0])}
        if include is not None and "packed" in metadata:
            packed = json.loads(metadata["packed"])
            metadata = {**metadata, "packed": json.dumps(
                {k: v for k, v in packed.items() if include(k)})}
        from .ternary import unpack_state_dict
        return unpack_state_dict(tensors, metadata)

    lm_dir, shards = plain_shards(repo_dir, config, lm_dir)
    state = {}
    for shard in shards:
        if include is None:
            state.update(load_file(str(shard)))
            continue
        with safe_open(str(shard), framework="pt") as handle:
            state.update({k: handle.get_tensor(k) for k in handle.keys() if include(k)})
    if not shards or (include is None and not state):
        raise FileNotFoundError(f"no plain model weights in {lm_dir}")
    return state


def iter_lm_weights(repo_dir, config, variant):
    """load_lm_weights as (name, tensor) pairs; the emb4 reader yields them one at a time."""
    if variant == "emb4":
        path = weights_path(repo_dir, config, variant)
        if not path.is_file():
            raise FileNotFoundError(f"{variant!r} weights not found: {path}")
        from .tl2_emb4 import iter_state_dict
        return iter_state_dict(path)
    return iter(load_lm_weights(repo_dir, config, variant).items())
