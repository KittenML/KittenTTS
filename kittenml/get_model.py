"""The shared entry point: one constructor, two model families.

`KittenTTS(...)` reads the repository's config.json and builds whichever backend
it declares — `kittentts2` for the KittenTTS 2 speech language model, `kittentts_legacy`
for the ONNX models. Neither backend is imported until it is the one selected, so
a KittenTTS 2 install never needs espeak and an ONNX install never needs torch.
"""

from huggingface_hub import snapshot_download

from .kittentts_legacy import KittenTTSOnnx, download_from_huggingface  # noqa: F401
from .repo import KITTEN2_TYPE, ONNX_TYPES, resolve_repo


def _kitten2_files(config, decoder=None, weights=None):
    """The paths a KittenTTS 2 repository's config actually points at.

    A model repository may carry artifacts for other runtimes alongside these — a
    llama.cpp bundle, say — and `snapshot_download` takes everything by default.
    Narrowing it to what the config declares keeps an install from pulling gigabytes
    it cannot use, and needs no knowledge of what else happens to be in there.
    """
    import posixpath

    wanted = {"config.json"}
    lm_dir = config.get("lm_dir", "lm")
    # Only the selected weight variant. The packings are equivalent models at very
    # different sizes, so fetching all of them would defeat the point of having them.
    variant = weights or config.get("default_weights", "packed")
    declared = {"packed": config.get("lm_packed"), "emb4": config.get("lm_emb4")}
    if variant in declared and declared[variant]:
        wanted.add(declared[variant])
        wanted.add(posixpath.join(lm_dir, "*.json"))
        wanted.add(posixpath.join(lm_dir, "*.txt"))
    else:
        wanted.add(posixpath.join(lm_dir, "*"))
    for key, default in (("voices", "voices/voices.json"),
                         ("speaker_embedding", "speaker/model.safetensors")):
        path = config.get(key, default)
        wanted.add(posixpath.join(posixpath.dirname(path), "*"))
    # Only the decoder being used. The default decoder needs none of these files, and
    # fetching every alternative would add ~100 MB nobody asked for. Picking a
    # different one later constructs a new model, which downloads it then.
    decoders = config.get("decoders", {})
    name = decoder or config.get("default_decoder", "default")
    if name in decoders:
        wanted.add(decoders[name])
    return sorted(wanted)


class KittenTTS(KittenTTSOnnx):
    """Entry point for every KittenTTS model.

    The repository's config.json decides what you get back. An ONNX repository
    yields this class, which is the ONNX backend. A KittenTTS 2 repository yields
    a `kittentts2.KittenTTS2` instead — a language model with a larger API: voice
    cloning, expression tags, decoding presets.

        m = KittenTTS("KittenML/kitten-tts-mini-0.8")   # ONNX, runs on CPU
        m = KittenTTS("KittenML/kitten-tts-2")          # KittenTTS 2, GPU optional

    Both also accept a path to a local repository directory.
    """

    def __new__(cls, model_name="KittenML/kitten-tts-nano-0.8", cache_dir=None,
                backend=None, device=None, hf_token=None, decoder=None, weights=None,
                vllm_options=None):
        local_dir, repo_id, config = resolve_repo(model_name, cache_dir)
        model_type = config.get("type")

        if model_type == KITTEN2_TYPE:
            from .kittentts2 import KittenTTS2
            if backend == "vllm":
                from .kittentts2.vllm.backend import validate_environment
                validate_environment(device)
            elif vllm_options is not None:
                raise ValueError("vllm_options requires backend='vllm'")
            if local_dir is None:
                local_dir = snapshot_download(
                    repo_id=repo_id, cache_dir=cache_dir, token=hf_token,
                    allow_patterns=_kitten2_files(config, decoder, weights))
            # Returning a different class means __init__ below is never called.
            return KittenTTS2(local_dir, config, device=device, cache_dir=cache_dir,
                              hf_token=hf_token, decoder=decoder, weights=weights,
                              backend=backend, vllm_options=vllm_options)

        if model_type not in ONNX_TYPES:
            raise ValueError(
                f"Unsupported model type {model_type!r}; expected one of "
                f"{[*ONNX_TYPES, KITTEN2_TYPE]}")

        if backend == "vllm" or vllm_options is not None:
            raise ValueError("the vLLM backend requires a KittenTTS 2 model")

        instance = super().__new__(cls)
        # Hand the already-resolved repository to KittenTTSOnnx.__init__ rather
        # than making it hit the Hub a second time.
        instance._resolved = (local_dir, repo_id, config)
        return instance


def get_model(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None, backend=None):
    """Get a KittenTTS model (legacy function for backward compatibility)."""
    return KittenTTS(repo_id, cache_dir, backend=backend)
