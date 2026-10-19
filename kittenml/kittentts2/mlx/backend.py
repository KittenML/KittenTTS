"""Optional Apple-silicon language-model backend; waveform decoding stays in KittenTTS.

KittenTTS 2 spends nearly all of its time in the language model, one token at a
time, which makes it bound by how fast the weights can be read. On a Mac the
default backend reads them as bf16 on the CPU. This one keeps them in the 4-bit
form the published checkpoint already implies (see `loader.py`) and runs the
model on the GPU through MLX, so each step reads a quarter of the bytes and reads
them at GPU bandwidth. The weights are bit-for-bit the published ones and the
sampling pipeline is the same, so this is the same model, not a smaller one.

Audio decoding, voice cloning, text normalization and chunking are unchanged.
"""

import importlib.util
import json
import platform

import numpy as np

from .sample import build_logits_processors, sample


def validate_environment(device=None):
    """Fail early, and say what to install, rather than at the first generation."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("the MLX backend requires an Apple-silicon Mac")
    if importlib.util.find_spec("mlx") is None:
        raise ImportError('install the optional backend with pip install "kittenml[mlx]"')
    if device is not None and str(device) not in ("mps", "cpu"):
        raise ValueError(f"the MLX backend runs the language model on the Mac GPU; "
                         f"device={device!r} would only move audio decoding. Pass "
                         f"device='mps' or device='cpu' (or leave it unset).")


class MLXBackend:
    """The language model on Apple-silicon GPU, with its weights left packed."""

    def __init__(self, owner, options=None):
        validate_environment(owner.device)
        import mlx.core as mx
        from transformers import AutoTokenizer

        from .loader import load_lm_weights
        from .model import KittenLM

        if options:
            raise ValueError(f"unknown mlx_options: {sorted(options)}")
        self.mx = mx
        self.token_map = owner.token_map
        self.stop_ids = {owner.token_map.speech_end_id, owner.token_map.stop_id}

        lm_dir = owner.repo_dir / owner.config.get("lm_dir", "lm")
        if owner.weights != "packed":
            raise ValueError(
                f"the MLX backend reads the packed ternary weights, not "
                f"weights={owner.weights!r}: the packing is what lets it keep the "
                f"ternary weights at 4 bits exactly. Use weights='packed'.")
        path = owner._weights_path("packed")
        if not path.is_file():
            raise FileNotFoundError(f"packed weights not found: {path}")

        self.dtype = mx.bfloat16 if owner.config.get("dtype", "bf16") == "bf16" else mx.float32
        owner.tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
        dense, quantised, group_size = load_lm_weights(path, dtype=self.dtype)
        with open(lm_dir / "config.json") as handle:
            config = json.load(handle)
        self.model = KittenLM.from_config(
            config, quantised, group_size,
            spk_dim=owner.config.get("speaker_embedding_dim", 512)).load_dense(dense)
        self.model.eval()
        mx.eval(self.model)

    def seed(self, seed):
        """Seed MLX's global sampler state; KittenTTS 2 calls this per generation."""
        self.mx.random.seed(int(seed))

    def speaker_embedding(self, embedding):
        """Project a speaker embedding once, instead of once per chunk.

        Returns the hidden state that replaces the first position's token
        embedding at prefill — the MLX counterpart of the Torch backend's
        `spk_proj` forward hook.
        """
        projected = self.model.speaker_embedding(
            self.mx.array(np.asarray(embedding, np.float32)))
        self.mx.eval(projected)
        return projected.reshape(1, 1, -1)

    def generate_tokens(self, prompt, speaker_embedding, settings, max_new_tokens, advanced):
        """Run the LM for one chunk and return its audio codec ids (0-based)."""
        mx = self.mx
        processors = build_logits_processors(
            self.token_map,
            temperature=settings["temperature"], top_k=settings["top_k"],
            top_p=settings["top_p"], min_p=settings.get("min_p", 0.0),
            repetition_penalty=advanced["repetition_penalty"],
            repetition_window=advanced["repetition_window"],
            token_run_penalty=advanced["token_run_penalty"],
            token_run_grace=advanced["token_run_grace"],
            prompt_length=len(prompt))

        cache = self.model.new_cache()
        embeddings = self.model.embed(prompt)
        # Position 0 carries the speaker, exactly as in `speaker.attach_speaker_head`.
        embeddings = mx.concatenate(
            [speaker_embedding.astype(embeddings.dtype), embeddings[:, 1:]], axis=1)
        history = mx.array(prompt, mx.int32)[None]
        logits = self.model(embeddings, cache)

        generated = []
        for _ in range(int(max_new_tokens)):
            token = sample(history, logits, processors)
            value = int(token.item())
            if value in self.stop_ids:
                break
            generated.append(value)
            history = mx.concatenate([history, token[None].astype(mx.int32)], axis=1)
            logits = self.model(self.model.embed([value]), cache)

        low, high = self.token_map.audio_id_base, self.token_map.audio_id_end
        return [t - low for t in generated if low <= t < high]
