"""Read the selected checkpoint identically for Torch and vLLM."""

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


def load_lm_weights(repo_dir, config, variant, lm_dir=None):
    """Unpack on CPU without re-quantizing or writing an expanded checkpoint."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = weights_path(repo_dir, config, variant)
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"{variant!r} weights not found: {path}")
        if variant == "emb4":
            from .tl2_emb4 import load_state_dict
            return load_state_dict(path)
        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata() or {}
            tensors = {k: handle.get_tensor(k) for k in handle.keys()}
        from .ternary import unpack_state_dict
        return unpack_state_dict(tensors, metadata)

    lm_dir = Path(lm_dir) if lm_dir is not None else Path(repo_dir) / config.get("lm_dir", "lm")
    state = {}
    for shard in sorted(lm_dir.glob("model*.safetensors")):
        if "-" not in shard.stem:
            state.update(load_file(str(shard)))
    if not state:
        raise FileNotFoundError(f"no plain model weights in {lm_dir}")
    return state
