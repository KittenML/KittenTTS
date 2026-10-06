"""Optional GPU language-model backend; waveform decoding stays in KittenTTS."""

from dataclasses import asdict
import contextlib
import importlib.util
import os
import sys

import torch


@contextlib.contextmanager
def _engine_environment():
    # KittenTTS already applies its warpers. Use vLLM's native sampler by default
    # so a pip installation needs no nvcc for FlashInfer's sampling JIT. The
    # worker inherits these settings, including quiet startup defaults. Restore
    # the caller's environment afterwards and respect explicit overrides.
    defaults = {"VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_LOGGING_LEVEL": "WARNING", "TQDM_DISABLE": "1"}
    previous = {key: os.environ.get(key) for key in defaults}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_environment(device=None):
    if sys.platform != "linux":
        raise RuntimeError("the vLLM backend requires Linux and an NVIDIA CUDA GPU")
    if importlib.util.find_spec("vllm") is None:
        raise ImportError('install the optional backend with pip install "kittenml[vllm]"')
    selected = torch.device(device or "cuda")
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the vLLM backend requires an NVIDIA CUDA GPU")
    if selected.index not in (None, 0):
        raise ValueError("select the GPU with CUDA_VISIBLE_DEVICES and use device='cuda'")


class VLLMBackend:
    def __init__(self, owner, options=None):
        validate_environment(owner.device)
        from transformers import AutoConfig, AutoTokenizer
        from ..speaker import load_speaker_head

        options = dict(options or {})
        defaults = dict(gpu_memory_utilization=0.5, max_model_len=4096,
                        max_num_seqs=8, enforce_eager=False)
        unknown = set(options) - defaults.keys()
        if unknown:
            raise ValueError(f"unknown vllm_options: {sorted(unknown)}")
        defaults.update(options)
        self.max_model_len = defaults["max_model_len"]
        self.token_map = owner.token_map
        lm_dir = owner.repo_dir / owner.config.get("lm_dir", "lm")
        config = AutoConfig.from_pretrained(str(lm_dir))
        if config.model_type != "qwen3":
            raise ValueError("the vLLM backend currently supports Qwen3 KittenTTS 2 checkpoints")
        owner.dtype = torch.bfloat16 if owner.config.get("dtype", "bf16") == "bf16" else torch.float32
        owner.tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
        state = owner._load_lm_weights(lm_dir)
        self.embedding = state["model.embed_tokens.weight"].to(owner.dtype)
        self.spk_proj = load_speaker_head(config.hidden_size, str(lm_dir),
                                          owner.device, owner.dtype,
                                          spk_dim=owner.config.get("speaker_embedding_dim", 512),
                                          state=state)
        del state
        with _engine_environment():
            from vllm import LLM
            from vllm.config import LoggingConfig

            self.engine = LLM(
                model=str(lm_dir), tokenizer=str(lm_dir), dtype=owner.dtype,
                load_format="kittenml",
                model_loader_extra_config={"repo_dir": str(owner.repo_dir),
                                           "config": owner.config, "weights": owner.weights},
                enable_prompt_embeds=True, enable_prefix_caching=False,
                logits_processors=["kittenml.kittentts2.vllm.logits:KittenLogitsProcessor"],
                logging_config=LoggingConfig(), disable_log_stats=True, **defaults)

    def prepare_request(self, prompt, speaker_embedding, settings, max_new_tokens, advanced):
        from vllm import SamplingParams

        if len(prompt) + int(max_new_tokens) > self.max_model_len:
            raise ValueError("prompt and token budget exceed vllm_options['max_model_len']; "
                             "increase it or shorten the reference/token budget")
        with torch.inference_mode():
            embeds = self.embedding[torch.tensor(prompt, dtype=torch.long)].clone()
            embeds[0] = self.spk_proj(speaker_embedding).squeeze(0).cpu()
        sampling = {k: settings[k] for k in ("temperature", "top_k", "top_p")}
        sampling["min_p"] = settings.get("min_p", 0.0)
        sampling.update({k: advanced[k] for k in ("repetition_penalty", "repetition_window",
                                                  "token_run_penalty", "token_run_grace")})
        seed = advanced.get("seed")
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        params = SamplingParams(
            temperature=1.0, top_k=-1, top_p=1.0, min_p=0.0,
            repetition_penalty=1.0, max_tokens=int(max_new_tokens),
            stop_token_ids=[self.token_map.speech_end_id, self.token_map.stop_id],
            ignore_eos=True, detokenize=False, seed=int(seed),
            extra_args={"kittenml": {"prompt": prompt, "token_map": asdict(self.token_map),
                                      "sampling": sampling}})
        return {"prompt_embeds": embeds}, params

    def generate_tokens(self, prompt, speaker_embedding, settings, max_new_tokens, advanced):
        request, params = self.prepare_request(prompt, speaker_embedding, settings,
                                               max_new_tokens, advanced)
        output = self.engine.generate([request], [params], use_tqdm=False)[0]
        low, high = self.token_map.audio_id_base, self.token_map.audio_id_end
        return [t - low for t in output.outputs[0].token_ids if low <= t < high]
