"""Speaker embedding: the x-vector model KittenTTS 2's `spk_proj` head was trained on.

A standalone implementation of pyannote's `XVectorSincNet` — a SincNet frontend
feeding a TDNN stack, statistics pooling and a linear projection to 512
dimensions. The weights ship inside the KittenTTS 2 repository, so nothing here
reaches for a gated model at run time.

It is reimplemented rather than imported because `pyannote.audio` pulls in
lightning, torchmetrics, torch-audiomentations and speechbrain to run one
96 MB forward pass. Everything below is only that forward pass, and it is
verified to reproduce pyannote's embeddings exactly.

The model, and this implementation of it, derive from:

  pyannote/embedding (MIT) — Hervé Bredin, pyannote.audio
  https://huggingface.co/pyannote/embedding
  SincNet — Mirco Ravanelli, Yoshua Bengio, "Speaker Recognition from raw
  waveform with SincNet", SLT 2018
  The parameterised sinc filterbank follows asteroid-filterbanks' ParamSincFB
  (MIT) — Manuel Pariente et al.

See the LICENSE shipped alongside the weights in the model repository.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16000
EMBEDDING_DIM = 512


class ParamSincFilterbank(nn.Module):
    """Band-pass sinc filters parameterised by their cutoff frequencies.

    Each filter is described by two learned numbers — a low cutoff and a
    bandwidth — rather than by its taps, so the 80 filters carry 80 parameters
    instead of 80x251. The taps are rebuilt from those numbers on every call.

    Filters come in cosine/sine pairs: `n_filters // 2` of each, concatenated.
    """

    def __init__(self, n_filters=80, kernel_size=251, stride=10,
                 sample_rate=SAMPLE_RATE, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.half_kernel = kernel_size // 2

        self.low_hz_ = nn.Parameter(torch.zeros(n_filters // 2, 1))
        self.band_hz_ = nn.Parameter(torch.zeros(n_filters // 2, 1))
        # Both are restored from the checkpoint, which stores them as buffers.
        self.register_buffer("window_", torch.from_numpy(
            np.hamming(kernel_size)[:self.half_kernel]).float())
        self.register_buffer("n_", 2 * np.pi * (
            torch.arange(-self.half_kernel, 0.0).view(1, -1) / sample_rate))

    def _band_pass(self, low, high, kind):
        band = (high - low)[:, 0]
        ft_low = torch.matmul(low, self.n_)
        ft_high = torch.matmul(high, self.n_)
        if kind == "cos":                       # even filters, as in the SincNet paper
            left = ((torch.sin(ft_high) - torch.sin(ft_low)) / (self.n_ / 2)) * self.window_
            center = 2 * band.view(-1, 1)
            right = torch.flip(left, dims=[1])
        else:                                   # odd filters, the ParamSincFB extension
            left = ((torch.cos(ft_low) - torch.cos(ft_high)) / (self.n_ / 2)) * self.window_
            center = torch.zeros_like(band.view(-1, 1))
            right = -torch.flip(left, dims=[1])
        band_pass = torch.cat([left, center, right], dim=1) / (2 * band[:, None])
        return band_pass.view(self.n_filters // 2, 1, self.kernel_size)

    def filters(self):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2)
        return torch.cat([self._band_pass(low, high, "cos"),
                          self._band_pass(low, high, "sin")], dim=0)

    def forward(self, waveform):
        return F.conv1d(waveform, self.filters(), stride=self.stride)


class SincNet(nn.Module):
    """Learned filterbank frontend: three conv/pool/norm stages over raw audio."""

    def __init__(self, sample_rate=SAMPLE_RATE, stride=10):
        super().__init__()
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"SincNet expects {SAMPLE_RATE} Hz audio, got {sample_rate}")
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
        self.conv1d = nn.ModuleList([
            _SincConvWrapper(ParamSincFilterbank(80, 251, stride=stride,
                                                 sample_rate=sample_rate)),
            nn.Conv1d(80, 60, 5, stride=1),
            nn.Conv1d(60, 60, 5, stride=1),
        ])
        self.pool1d = nn.ModuleList([nn.MaxPool1d(3, stride=3) for _ in range(3)])
        self.norm1d = nn.ModuleList([nn.InstanceNorm1d(c, affine=True) for c in (80, 60, 60)])

    def forward(self, waveforms):
        outputs = self.wav_norm1d(waveforms)
        for i, (conv, pool, norm) in enumerate(zip(self.conv1d, self.pool1d, self.norm1d)):
            outputs = conv(outputs)
            if i == 0:
                # The sinc filters are band-pass, so only magnitude carries
                # information here; the sign is phase.
                outputs = torch.abs(outputs)
            outputs = F.leaky_relu(norm(pool(outputs)))
        return outputs


class _SincConvWrapper(nn.Module):
    """Holds the filterbank under the attribute name the checkpoint expects."""

    def __init__(self, filterbank):
        super().__init__()
        self.filterbank = filterbank

    def forward(self, waveform):
        return self.filterbank(waveform)


class XVectorSincNet(nn.Module):
    """SincNet frontend -> TDNN stack -> statistics pooling -> 512-dim embedding."""

    KERNEL_SIZES = (5, 3, 3, 1, 1)
    DILATIONS = (1, 2, 3, 1, 1)
    CHANNELS = (512, 512, 512, 512, 1500)

    def __init__(self, sample_rate=SAMPLE_RATE, sincnet_stride=10, dimension=EMBEDDING_DIM):
        super().__init__()
        self.sincnet = SincNet(sample_rate=sample_rate, stride=sincnet_stride)
        self.tdnns = nn.ModuleList()
        in_channels = 60
        for out_channels, kernel_size, dilation in zip(
                self.CHANNELS, self.KERNEL_SIZES, self.DILATIONS):
            self.tdnns.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation),
                nn.LeakyReLU(),
                nn.BatchNorm1d(out_channels),
            ])
            in_channels = out_channels
        self.embedding = nn.Linear(in_channels * 2, dimension)

    def forward(self, waveforms):
        """(batch, channel, samples) at 16 kHz -> (batch, 512)."""
        outputs = self.sincnet(waveforms).squeeze(dim=1)
        for tdnn in self.tdnns:
            outputs = tdnn(outputs)
        # Statistics pooling: mean and unbiased standard deviation over time,
        # concatenated, so a clip of any length becomes one fixed-size vector.
        pooled = torch.cat([outputs.mean(dim=-1),
                            outputs.std(dim=-1, correction=1)], dim=-1)
        return self.embedding(pooled)


def load_speaker_embedding_model(weights_path, device="cpu"):
    """Build the model and load the weights shipped with a KittenTTS 2 repository.

    Accepts either the safetensors file the repository ships or an original
    pyannote `pytorch_model.bin` checkpoint.
    """
    if str(weights_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(str(weights_path))
    else:
        checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)

    model = XVectorSincNet()
    # `loss_func.*` is the training-time classification head, and the sinc window
    # and time vectors are constants rebuilt in __init__. Neither belongs to
    # inference, so they are ignored whether or not the file happens to carry
    # them — the repository's own weights are saved without them.
    state = {k: v for k, v in state.items() if not k.startswith("loss_func.")}
    buffers = ("window_", "n_")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(buffers)]
    unexpected = [k for k in unexpected if not k.endswith(buffers)]
    if missing or unexpected:
        raise ValueError(f"speaker embedding weights do not match the model "
                         f"(missing={missing}, unexpected={unexpected})")
    return model.to(device).eval()
