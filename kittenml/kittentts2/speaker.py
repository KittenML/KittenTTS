"""Speaker conditioning for KittenTTS 2.

The model takes a speaker identity through a dedicated `spk_proj` head, not
through the prompt: a 512-dim pyannote embedding is projected to hidden size and
written over the first position's input embedding at prefill. That head is trained
as part of the checkpoint, so it lives alongside the weights and must be restored
explicitly — transformers knows nothing about it.

Built-in voices ship their embeddings precomputed, so the embedding model is only
loaded when cloning from a new reference clip. Whisper is likewise only loaded
when a caller clones without supplying the reference transcript.
"""

import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from .speaker_embedding import EMBEDDING_DIM as SPEAKER_EMBEDDING_DIM
from .speaker_embedding import SAMPLE_RATE

WHISPER_MODEL = "openai/whisper-large-v3"
# Whisper large-v3 in float16 on the GPU: 2.9 GiB of weights plus working memory.
# With less GPU memory free than this, it transcribes on the CPU instead.
WHISPER_GPU_BYTES = 3584 << 20
# The model for that CPU fallback: large-v3 turbo (large-v3's encoder, 4 decoder
# layers instead of 32). Measured in float32 on 10 LibriSpeech reference clips: the
# same words as large-v3 (WER 2.4% for both), 2.7x faster (26 s instead of 72 s per
# 10-second clip on a 6-core CPU), and 3.1 GB of host memory instead of 5.9 GB. In
# a cloning run on an 8-core laptop: 12 s instead of 43 s, and a host-memory peak
# 4.1 GB above the process instead of 8.5 GB.
WHISPER_CPU_FALLBACK_MODEL = "openai/whisper-large-v3-turbo"

logger = logging.getLogger(__name__)


def load_speaker_head(hidden, checkpoint_dir, device, dtype,
                      spk_dim=SPEAKER_EMBEDDING_DIM, state=None):
    """Restore the trained projection without constructing a language model."""
    head = nn.Sequential(nn.Linear(spk_dim, int(hidden), bias=True),
                         nn.LayerNorm(int(hidden))).to(dtype=dtype)

    if state is None:
        state = {}
        for shard in sorted(pathlib.Path(checkpoint_dir).glob("model*.safetensors")):
            state.update(load_file(str(shard)))
    head_state = {k[len("spk_proj."):]: v.to(dtype)
                  for k, v in state.items() if k.startswith("spk_proj.")}
    if not head_state:
        raise ValueError(
            f"no spk_proj.* weights in {checkpoint_dir} — this checkpoint cannot be "
            "speaker-conditioned, and generating from it would ignore the voice")
    head.load_state_dict(head_state, strict=True)
    return head.to(device=device, dtype=dtype).eval()


def attach_speaker_head(model, checkpoint_dir, device, dtype,
                        spk_dim=SPEAKER_EMBEDDING_DIM, state=None):
    """Restore `model.spk_proj` and inject it at prefill.

    A missing head is an error: speech would otherwise ignore the chosen voice.
    Pass an already-loaded `state` to avoid reading the checkpoint twice.
    """
    model.spk_proj = load_speaker_head(model.config.hidden_size, checkpoint_dir,
                                      device, dtype, spk_dim=spk_dim, state=state)

    def _inject_speaker(module, args, kwargs):
        embedding = kwargs.pop("speaker_embedding", None)
        if embedding is None:
            embedding = getattr(module, "_pending_speaker_embedding", None)
        input_ids = kwargs.get("input_ids")
        if embedding is None or input_ids is None:
            return args, kwargs
        embeds = module.get_input_embeddings()(input_ids)
        cache_position = kwargs.get("cache_position")
        past = kwargs.get("past_key_values")
        is_prefill = (int(cache_position[0]) == 0) if cache_position is not None else (
            past is None or (hasattr(past, "get_seq_length") and past.get_seq_length() == 0))
        # Only at prefill: position 0 carries the speaker, every later step is
        # ordinary token embedding.
        if is_prefill and embeds.shape[1] >= 1:
            projected = module.spk_proj(embedding.to(embeds.dtype)).unsqueeze(1)
            embeds = torch.cat([projected, embeds[:, 1:, :]], dim=1)
        kwargs.pop("input_ids", None)
        kwargs["inputs_embeds"] = embeds
        return args, kwargs

    model.register_forward_pre_hook(_inject_speaker, with_kwargs=True)
    return model


class SpeakerEncoder:
    """Turns a reference clip into the 512-dim embedding `spk_proj` expects.

    The weights ship inside the model repository, so this needs no network
    access and no gated model — see `speaker_embedding.py` for the architecture
    and for what it is derived from.
    """

    def __init__(self, weights_path, device="cuda"):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from .speaker_embedding import load_speaker_embedding_model
            self._model = load_speaker_embedding_model(self.weights_path, device=self.device)
        return self._model

    def __call__(self, wav_path):
        """Reference clip -> unit-norm 512-dim speaker embedding."""
        waveform = _load_waveform(wav_path, self.device)
        with torch.inference_mode():
            raw = self._load()(waveform).squeeze(0).float().cpu().numpy()
        norm = float(np.linalg.norm(raw))
        return (raw / norm).astype(np.float32) if norm > 0 else raw


def _load_waveform(wav_path, device, sample_rate=SAMPLE_RATE):
    """Read a clip as mono at the model's sample rate, shaped (1, 1, samples).

    Channels are averaged and resampling goes through torchaudio, matching how
    the embeddings this model was trained against were computed — a different
    resampler shifts the embedding slightly.
    """
    import torchaudio

    try:
        import soundfile as sf
        audio, file_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        audio = audio.T                                 # (time, channels) -> (channels, time)
    except Exception:                                   # formats libsndfile cannot open
        import librosa
        audio, file_rate = librosa.load(str(wav_path), sr=None, mono=False)
        audio = np.atleast_2d(audio)

    waveform = torch.from_numpy(np.ascontiguousarray(audio)).float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(file_rate) != sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(file_rate), sample_rate)
    return waveform.unsqueeze(0).to(device)


class ReferenceTranscriber:
    """Lazily-loaded Whisper, used only when a clone call omits the transcript.

    The reference transcript is part of the prompt, so cloning needs one either
    way; a caller should not have to type it out. Nothing is downloaded unless a
    clone actually arrives without one — the built-in voices ship their
    transcripts, so they never reach this.
    """

    def __init__(self, device="cuda", model_name=WHISPER_MODEL, cache_dir=None):
        self.device = device
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._pipeline = None
        # Where Whisper runs once loaded: "cuda", or "cpu" (see _choose_device),
        # and which model that is.
        self.resolved_device = None
        self.resolved_model = None

    def _choose_device(self):
        """The GPU, unless it has too little free memory for Whisper; then the CPU.

        Memory this process has cached but not used counts as free, since Whisper
        can reuse it.
        """
        if self.device != "cuda":
            return "cpu"
        free, _ = torch.cuda.mem_get_info()
        free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        self._free_bytes = free
        return "cuda" if free >= WHISPER_GPU_BYTES else "cpu"

    def _label_cpu_fallback(self, missing=None):
        """Warn about the CPU fallback once the model is known: its cost depends on it."""
        free = getattr(self, "_free_bytes", None)
        why = (f"{free >> 20:,} MiB of GPU memory free, too little for Whisper "
               f"(about {WHISPER_GPU_BYTES >> 20:,} MiB)" if free is not None
               else "too little GPU memory free for Whisper")
        if self.resolved_model == WHISPER_CPU_FALLBACK_MODEL:
            how = ("with Whisper large-v3 turbo. This is supported but not the optimal path: it "
                   "is slower (measured 12 s instead of 1.4 s on a GPU for a 10-second clip, on "
                   "an 8-core laptop CPU), uses about 4 GB of host memory, and its transcript, "
                   "and so the cloned voice, can differ from a GPU run, which uses large-v3")
        elif missing and self.resolved_model == WHISPER_MODEL:
            how = (f"with Whisper large-v3, because {missing} is not available (offline or not "
                   "downloaded). This is supported but not the optimal path: measured 43 s per "
                   "10-second clip on an 8-core laptop CPU and about 8.5 GB of host memory; "
                   f"downloading {missing} makes it about 3.5x faster")
        else:
            how = (f"with {self.resolved_model}. This is supported but not the optimal path: "
                   "it is slower than on a GPU")
        logger.warning("KittenTTS: %s, so the reference clip is transcribed on the CPU %s. "
                       "Pass reference_text to skip transcription.", why, how)

    def _model_for(self, device):
        """The checkpoint to load: the smaller fallback model when the GPU is too full."""
        if device == "cpu" and self.device == "cuda" and self.model_name == WHISPER_MODEL:
            return WHISPER_CPU_FALLBACK_MODEL
        return self.model_name

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self.resolved_device = self._choose_device()
            self.resolved_model = self._model_for(self.resolved_device)
            on_gpu = self.resolved_device == "cuda"

            def load(model):
                return pipeline(
                    "automatic-speech-recognition", model=model,
                    chunk_length_s=30, device=0 if on_gpu else -1,
                    dtype=torch.float16 if on_gpu else torch.float32,
                    model_kwargs={"cache_dir": self.cache_dir} if self.cache_dir else {})

            missing = None
            try:
                self._pipeline = load(self.resolved_model)
            except OSError:
                # The fallback model is a separate download; offline, use the one
                # that is already there.
                if self.resolved_model == self.model_name:
                    raise
                missing, self.resolved_model = self.resolved_model, self.model_name
                self._pipeline = load(self.model_name)
            if self.resolved_device == "cpu" and self.device == "cuda":
                self._label_cpu_fallback(missing)
        return self._pipeline

    def __call__(self, wav_path):
        import librosa
        wav, sample_rate = librosa.load(str(wav_path), sr=16000, mono=True)
        del sample_rate
        return self._load()(wav, generate_kwargs={"language": "en"})["text"].strip()
