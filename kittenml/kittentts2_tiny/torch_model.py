"""The PyTorch runtime: the float32 networks of one model, computing what the ONNX graph computes:

    tokens [T], style [1, 256], speed -> waveform [N] float32, durations [T] int64

style[:, :128] conditions the decoder (timbre), style[:, 128:] the prosody predictor. Durations are
round(sum of the sigmoid duration bins / speed), at least 1 frame. The decoder adds random noise, as in ONNX.
"""
from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import load_file

from .networks.modules import MODULES, build_modules


def _length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


class TorchModel:
    def __init__(self, architecture, weights_path, device=None, threads=None):
        if threads:
            torch.set_num_threads(int(threads))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = torch.nn.ModuleDict(build_modules(architecture))
        self.net.load_state_dict(load_file(weights_path), strict=True)
        self.net.to(self.device).eval()
        self.n_params = sum(p.numel() for k in MODULES for p in self.net[k].parameters())

    @torch.no_grad()
    def __call__(self, tokens, style, speed=1.0):
        m, dev = self.net, self.device
        tok = torch.as_tensor([tokens], dtype=torch.long, device=dev)
        st = torch.as_tensor(np.asarray(style, np.float32).reshape(1, -1), device=dev)
        sa, sp = st[:, :128], st[:, 128:]
        n_tok = tok.shape[-1]
        il = torch.as_tensor([n_tok], dtype=torch.long, device=dev)
        tm = _length_to_mask(il)
        t_en = m.text_encoder(tok, il, tm)
        bd = m.bert(tok, attention_mask=(~tm).int())
        d_en = m.bert_encoder(bd).transpose(-1, -2)
        d = m.predictor.text_encoder(d_en, sp, il, tm)
        x, _ = m.predictor.lstm(d)
        dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1) / float(speed)
        pr = torch.round(dur.reshape(-1)).clamp(min=1).long()
        # token -> frame alignment
        aln = torch.repeat_interleave(torch.eye(n_tok, device=dev), pr, dim=1)
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        F0, N = m.predictor.F0Ntrain(en, sp)
        F0 = F0.unsqueeze(0) if F0.dim() == 1 else F0
        N = N.unsqueeze(0) if N.dim() == 1 else N
        wav = m.decoder(t_en @ aln.unsqueeze(0), F0, N, sa)
        return wav.reshape(-1).float().cpu().numpy(), pr.cpu().numpy()
