"""KittenTTS's complete sampling pipeline for vLLM Model Runner V2."""

import torch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

from ..logits import build_logits_processors
from ..tokens import TokenMap


class KittenLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.req_states = req_states
        self.requests = {}

    def add_request(self, req_idx, sampling_params):
        extra = (sampling_params.extra_args or {}).get("kittenml")
        self.requests.pop(req_idx, None)
        if extra is None:
            return False
        prompt = torch.tensor(extra["prompt"], device=self.req_states.device,
                              dtype=torch.long)
        processors = build_logits_processors(
            TokenMap(**extra["token_map"]), **extra["sampling"],
            prompt_length=len(extra["prompt"]))
        self.requests[req_idx] = (prompt, processors)
        return True

    def apply(self, logits, ctx):
        # No speculative decoding: each row corresponds to one active request,
        # and the host sequence lengths are exact (no per-token GPU readback).
        for row, req_idx in enumerate(ctx.idx_mapping_np):
            request = self.requests.get(int(req_idx))
            if request is None:
                continue
            prompt, processors = request
            length = int(ctx.seq_lens_upper_bound_np[row])
            generated = self.req_states.all_token_ids.gpu[int(req_idx), len(prompt):length]
            history = torch.cat((prompt, generated)).unsqueeze(0)
            logits[row:row + 1] = processors(history, logits[row:row + 1])
        return logits
