"""Locating a model repository and reading the config that says what it is.

Shared by both backends, and kept separate from `get_model` so the backends can
resolve a repository without importing the dispatcher that imports them.
"""

import json
import os

from huggingface_hub import hf_hub_download

# config.json "type" values, and the backend each selects.
ONNX_TYPES = ("ONNX1", "ONNX2")
KITTEN2_TYPE = "KITTEN2"


def resolve_repo(model_name, cache_dir=None):
    """Locate a model repository and read its config.

    Accepts a Hugging Face repo id ("KittenML/kitten-tts-mini-0.8"), a bare model
    name (assumed to be under KittenML), or a path to a local directory laid out
    like one of those repos — which is what you have before pushing a repo to the
    Hub.

    Returns (local_dir_or_none, repo_id_or_none, config).
    """
    if os.path.isdir(model_name):
        config_path = os.path.join(model_name, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"no config.json in {model_name}")
        with open(config_path) as f:
            return model_name, None, json.load(f)

    repo_id = model_name if "/" in model_name else f"KittenML/{model_name}"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir)
    with open(config_path) as f:
        return None, repo_id, json.load(f)
