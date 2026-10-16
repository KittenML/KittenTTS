"""The training loop: one path that runs on a laptop CPU and on eight H100s.

Nothing here is novel, which is the point -- a speech LM fine-tune fails for
boring reasons (a mask off by one, a frozen parameter that was supposed to
train, a resume that silently restarts the schedule), and the only defence is a
loop simple enough that `selfcheck.py` can assert every one of those properties
end to end. Every piece the self-check needs is a function it can call rather
than a step buried in `main`.

Three modes, in increasing order of what they can change and what they cost:

    embed   body frozen, train `embed_tokens` (tied to `lm_head`) and
            `spk_proj`. The cheapest useful thing for Indic text, because the
            BPE rows for Devanagari are the part of this checkpoint most likely
            to be undertrained. Nothing re-ternarizes, since nothing ternary
            moved.
    lora    adapters over the frozen ternary body, plus the embedding. Fits a
            24 GB card. Exporting needs a merge and a QAT pass to get back to
            ternary -- option (a) in docs/indic/training.md.
    full    QAT over all 196 body linears (`qat.py`) plus everything dense.
            Exports straight back to packed ternary.

    python -m indic.train.loop --toy --steps 20 --device cpu
    torchrun --nproc_per_node=8 -m indic.train.loop --checkpoint ... --mode full

`--toy` builds a 2-layer model with the released token layout shrunk to fit, so
the loop is runnable with no checkpoint and no data. It proves the plumbing, not
the model.
"""

import argparse
import contextlib
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.train import load_kittentts2
from indic.train.data import Collator, SyntheticDataset, tiny_token_map
from indic.train.qat import load_released_state, wrap_ternary

_tokens = load_kittentts2("tokens")

# Matched by name, not by `isinstance`, so the groups survive a LoRA wrap that
# renames the module path around the base weight.
EMBEDDING_PARAMS = ("embed_tokens", "lm_head")
SPEAKER_PARAMS = ("spk_proj",)


# ── reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed, rank=0):
    """Seed every generator the loop touches.

    Offset by rank so that data augmentation and dropout differ across
    processes -- identical streams in every rank would make a DDP batch N copies
    of the same noise -- while the run as a whole stays reproducible from
    `--seed`.
    """
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def rng_state():
    """Everything a resume has to restore to continue the same run."""
    return {"python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


# ── distributed ───────────────────────────────────────────────────────────────
def setup_distributed():
    """Join the process group when launched under torchrun, else stay single.

    Reading WORLD_SIZE rather than taking a `--distributed` flag is what lets the
    same command line work both ways: torchrun sets it, a bare `python -m` does
    not, and the loop below never branches on it again beyond the sampler and
    the DDP wrap.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        import torch.distributed as dist
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# ── model ─────────────────────────────────────────────────────────────────────
def toy_model(token_map, hidden=128, layers=2, dtype=torch.float32, seed=0):
    """A 2-layer Qwen3 over a shrunk copy of the released token layout.

    `hidden` must stay a multiple of `qat.GROUP_SIZE`: every ternary group is a
    contiguous run of 128 input features, so a narrower model has no groups at
    all and `wrap_ternary` would refuse it. 128 is the smallest width that still
    exercises the real grouping.
    """
    from transformers import AutoModelForCausalLM, Qwen3Config

    if hidden % 128:
        raise ValueError(f"toy hidden size {hidden} is not a multiple of the "
                         "ternary group width; QAT could not wrap it")
    vocab = token_map.reference_text_end_id + 1
    config = Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=hidden // 4, max_position_embeddings=2048,
        tie_word_embeddings=True)
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(config, dtype=dtype,
                                            attn_implementation="sdpa")


def released_model(checkpoint_dir, dtype=torch.bfloat16, weights=None):
    """Load the released 28-layer checkpoint, weights and speaker head together.

    Mirrors `KittenTTS2._build_lm`: built from the config rather than through
    `from_pretrained`, because the packed weights are not a file
    `from_pretrained` can find, and `lm_head` is legitimately absent from the
    state dict because it is tied. Also skips the random initialisation of 1.7B
    parameters that are overwritten three lines later -- about a minute of pure
    cost, and several on a loaded machine.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # transformers moved this between 4.x and 5.x and it is only an
    # optimisation, so a version with neither just pays the init cost.
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            no_init_weights = contextlib.nullcontext

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if weights is None:
        packed = checkpoint_dir / "model-ternary.safetensors"
        weights = packed if packed.is_file() else checkpoint_dir / "model.safetensors"
    weights = pathlib.Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"no language model weights at {weights}")

    if weights.name.endswith("-ternary.safetensors"):
        state = load_released_state(weights)
    else:
        from safetensors.torch import load_file
        state = load_file(str(weights))

    config = AutoConfig.from_pretrained(str(checkpoint_dir))
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    speaker_state = {k[len("spk_proj."):]: v for k, v in state.items()
                     if k.startswith("spk_proj.")}
    missing, _ = model.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("spk_proj.")},
        strict=False, assign=True)
    model.tie_weights()
    unfilled = [n for n in missing if n != "lm_head.weight"]
    if unfilled:
        raise ValueError(f"weights missing from the checkpoint: {unfilled[:5]}")
    attach_speaker_head(model, speaker_state, dtype=dtype)
    return model


def attach_speaker_head(model, state=None, speaker_dim=512, dtype=torch.float32):
    """Install `spk_proj` and the prefill hook, then make a miss impossible to ignore.

    The head's architecture is fixed by the checkpoint's own key names --
    `spk_proj.0.*` is the Linear and `spk_proj.1.*` the LayerNorm -- and
    `load_state_dict(strict=True)` is what holds this to it. It is rebuilt here
    rather than imported from `kittenml.kittentts2.speaker` only because that
    module cannot be loaded without the whole TTS stack; see `__init__.py`.

    The injection follows `speaker.py` line for line, because training and
    inference have to put the speaker in the same place. Two things are added:

    1. `input_ids` is also read from the positional arguments. The inference
       hook looks only at keyword arguments, which is fine for `generate()` but
       means `model(input_ids, ...)` in a training loop silently skips
       injection. Measured against the real hook on a toy forward: loss 5.4162
       and a `spk_proj` gradient of 75.4 when called by keyword, loss 5.4086 and
       a gradient of exactly 0 when called positionally -- no error either way.
    2. A second hook asserts that injection actually *happened*, by a flag the
       first one sets -- not by the presence of `inputs_embeds`, which is the
       same trap one level up. A trainer that builds embeddings itself and calls
       `model(inputs_embeds=..., labels=...)` finds no `input_ids` for the hook
       to read, so the speaker is never injected and `inputs_embeds` is there
       all the same. Refusing that call is the point: position 0's `[START]` id
       is a placeholder whose embedding is *meant* to be overwritten, so a
       skipped injection trains on a different input sequence than the model
       will generate from, and the loss curve looks entirely normal.
    """
    hidden = int(model.config.hidden_size)
    head = nn.Sequential(nn.Linear(speaker_dim, hidden, bias=True),
                         nn.LayerNorm(hidden)).to(dtype=dtype)
    if state is not None:
        head.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
    model.spk_proj = head.to(device=next(model.parameters()).device, dtype=dtype)

    def _inject_speaker(module, args, kwargs):
        module._speaker_injected = False
        embedding = kwargs.pop("speaker_embedding", None)
        if embedding is None:
            embedding = getattr(module, "_pending_speaker_embedding", None)
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids, args = args[0], args[1:]
        if embedding is None or input_ids is None:
            return args, kwargs
        embeds = module.get_input_embeddings()(input_ids)
        cache_position = kwargs.get("cache_position")
        past = kwargs.get("past_key_values")
        is_prefill = (int(cache_position[0]) == 0) if cache_position is not None else (
            past is None or (hasattr(past, "get_seq_length") and past.get_seq_length() == 0))
        if is_prefill and embeds.shape[1] >= 1:
            projected = module.spk_proj(embedding.to(embeds.dtype)).unsqueeze(1)
            embeds = torch.cat([projected, embeds[:, 1:, :]], dim=1)
            module._speaker_injected = True
        kwargs.pop("input_ids", None)
        kwargs["inputs_embeds"] = embeds
        return args, kwargs

    def _require_injection(module, args, kwargs):
        if not module.training:
            return args, kwargs
        if not getattr(module, "_speaker_injected", False):
            raise RuntimeError(
                "speaker conditioning did not reach the forward pass: no "
                "speaker_embedding was supplied, or the call passed inputs_embeds "
                "instead of input_ids so there was nothing for the hook to "
                "overwrite. Training on this batch would silently use the [START] "
                "embedding at position 0. Go through loop.forward_batch.")
        return args, kwargs

    model.register_forward_pre_hook(_inject_speaker, with_kwargs=True)
    model.register_forward_pre_hook(_require_injection, with_kwargs=True)
    return model


def forward_batch(model, batch, device=None):
    """The one call the whole loop makes.

    `input_ids=` by keyword and never `inputs_embeds`, and that is about the
    hook rather than about style: `speaker.py` reads `kwargs.get("input_ids")`
    and returns the call untouched when it is absent, so `model(ids, ...)` or a
    trainer that passes embeddings it built itself runs with no speaker
    conditioning, no error and no warning. `attach_speaker_head` above closes
    both -- the first by reading the positional argument too, the second by
    refusing the call -- and this is the single path that stays on the safe side
    of them.

    `use_cache=False` because gradient checkpointing and a KV cache are
    incompatible, and because a teacher-forced pass has nothing to cache.
    """
    if device is not None:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    return model(input_ids=batch["input_ids"],
                 attention_mask=batch["attention_mask"],
                 labels=batch["labels"],
                 speaker_embedding=batch["speaker_embedding"],
                 use_cache=False)


# ── what trains ───────────────────────────────────────────────────────────────
def set_trainable(model, mode, train_norms=True, train_speaker=True):
    """Freeze everything the mode does not train, and say what is left.

    Returns {name: numel} for the trainable parameters. The self-check asserts
    against this and against the gradients that actually arrive, which are not
    the same claim: `requires_grad=True` on a parameter the graph never reaches
    produces no gradient and no complaint.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def unfreeze(predicate):
        for name, parameter in model.named_parameters():
            if predicate(name):
                parameter.requires_grad_(True)

    if mode == "full":
        unfreeze(lambda n: True)
    elif mode == "embed":
        unfreeze(lambda n: any(k in n for k in EMBEDDING_PARAMS))
    elif mode == "lora":
        # LoRA's own parameters are the only new ones; peft marks them itself,
        # so here we only add back what peft freezes and we want trained.
        unfreeze(lambda n: "lora_" in n or any(k in n for k in EMBEDDING_PARAMS))
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from embed, lora, full")

    if train_speaker:
        unfreeze(lambda n: any(k in n for k in SPEAKER_PARAMS))
    if train_norms and mode != "embed":
        # The RMSNorms are full precision and tiny, and they are how a quantized
        # body absorbs a shift in input statistics -- freezing them while the
        # embedding moves is the fastest way to a plateau.
        unfreeze(lambda n: "norm" in n)

    return {name: parameter.numel() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def apply_lora(model, rank=64, alpha=128, dropout=0.05):
    """Wrap the 196 body linears in LoRA adapters, leaving them frozen underneath.

    Targeted by the same seven projection names QAT wraps, so the two modes act
    on exactly the same set and a pilot can compare them without the difference
    being which layers moved.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, config)


def cast_trainable_to_float32(model):
    """Hold trainable parameters in float32 while the frozen body stays bf16.

    AdamW on bf16 master weights loses most small updates outright -- bf16 has
    8 mantissa bits, so an update below about 0.4% of a weight's magnitude
    rounds away to nothing and the parameter never moves. The frozen body does
    not have that problem because it never gets an update, and leaving it bf16
    is most of the memory saving. Compute still runs in bf16 under autocast.
    """
    cast = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            cast.append(name)
    return cast


# ── optimizer and schedule ────────────────────────────────────────────────────
def parameter_groups(model, lr, lr_embed, weight_decay):
    """Three groups: embeddings at their own LR, then decay and no-decay.

    The embedding gets a higher LR because it is what Indic adaptation mostly
    has to move -- the Devanagari and Tamil BPE rows -- while the body is a
    working model that a large step would damage. One-dimensional parameters
    (norms, biases) are excluded from weight decay as usual: decaying a
    LayerNorm gain towards zero is a slow way to erase a layer.
    """
    groups = {"embed": [], "decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(key in name for key in EMBEDDING_PARAMS):
            groups["embed"].append(parameter)
        elif parameter.dim() <= 1:
            groups["no_decay"].append(parameter)
        else:
            groups["decay"].append(parameter)
    return [
        {"params": groups["embed"], "lr": lr_embed, "weight_decay": 0.0, "name": "embed"},
        {"params": groups["decay"], "lr": lr, "weight_decay": weight_decay, "name": "decay"},
        {"params": groups["no_decay"], "lr": lr, "weight_decay": 0.0, "name": "no_decay"},
    ]


def build_optimizer(model, args):
    groups = [g for g in parameter_groups(model, args.lr, args.lr_embed,
                                          args.weight_decay) if g["params"]]
    if not groups:
        raise ValueError("no trainable parameters; check --mode")
    if args.optim == "adamw8bit":
        # Paged 8-bit states are the difference between a full fine-tune fitting
        # on one 80 GB card and not: AdamW's two fp32 moments are 8 bytes per
        # parameter, which is 13.9 GB for this model on its own.
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)


def lr_lambda(step, warmup, total, min_ratio):
    """Linear warmup then cosine decay, as a multiplier on each group's own LR.

    A multiplier rather than an absolute schedule so that the embedding group
    keeps its higher LR through the whole run instead of being flattened onto
    the body's.
    """
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, args.warmup, args.steps, args.min_lr_ratio))


# ── checkpoints ───────────────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, step, args):
    """Everything needed to continue the run, including the RNG streams.

    Without the RNG state a resume reproduces the weights but not the sampling,
    so two runs that should be identical diverge from the resume point onwards
    and the only evidence is a loss curve with a seam in it.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    torch.save({"step": step,
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng_state(),
                "args": vars(args) if not isinstance(args, dict) else args}, path)
    return str(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    set_rng_state(state["rng"])
    return state["step"]


class JsonlLogger:
    """One JSON object per line, flushed every write.

    A training run that dies at hour nine should leave nine hours of readable
    metrics, which a buffered writer does not guarantee. Rank-aware: only rank 0
    writes, so the file is not eight interleaved copies.
    """

    def __init__(self, path, rank=0):
        self.handle = None
        if path and rank == 0:
            path = pathlib.Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(path, "a", encoding="utf-8")

    def write(self, record):
        if self.handle is not None:
            self.handle.write(json.dumps(record) + "\n")
            self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()


# ── the loop ──────────────────────────────────────────────────────────────────
def autocast_context(device, amp):
    if amp == "off":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                          dtype=torch.bfloat16)


def run_steps(model, batches, optimizer, scheduler, args, device, start_step=0,
              logger=None, on_step=None):
    """Run `args.steps - start_step` optimizer steps over an iterator of batches.

    Factored out of `main` so the self-check can drive the identical code with a
    hand-made batch iterator: an overfit test or a resume test that ran a
    *different* loop would prove nothing about this one.

    `batches` yields micro-batches; `args.accum` of them make one optimizer
    step. The loss is divided by `accum` before backward so that the gradient is
    the mean over the whole effective batch rather than its sum -- otherwise the
    effective LR scales with `--accum` and nobody notices until a run with a
    different accumulation diverges.
    """
    import torch.distributed as dist

    history = []
    model.train()
    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(args.accum):
            batch = next(batches)
            # Every micro-batch but the last skips DDP's all-reduce. Without
            # this, accumulating N micro-batches costs N gradient syncs across
            # the whole model to produce one update, which on eight GPUs is most
            # of the step time for no benefit.
            last = micro == args.accum - 1
            sync = contextlib.nullcontext() if last or not hasattr(model, "no_sync") \
                else model.no_sync()
            with sync:
                with autocast_context(device, args.amp):
                    output = forward_batch(model, batch, device)
                (output.loss / args.accum).backward()
            total += float(output.loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.clip))
        optimizer.step()
        scheduler.step()

        loss_value = total / args.accum
        record = {"step": step, "loss": loss_value, "grad_norm": grad_norm,
                  "lr": scheduler.get_last_lr()[0], "time": time.time()}
        history.append(loss_value)
        if logger is not None and (step % max(args.log_every, 1) == 0
                                   or step == args.steps - 1):
            logger.write(record)
        if on_step is not None:
            on_step(step, record)
        if (args.save_every and args.out and (step + 1) % args.save_every == 0
                and (not dist.is_initialized() or dist.get_rank() == 0)):
            save_checkpoint(pathlib.Path(args.out) / f"step{step + 1}.pt",
                            model, optimizer, scheduler, step + 1, args)
    return history


def cycle(loader):
    """An endless stream of batches, reshuffled each pass."""
    while True:
        for batch in loader:
            yield batch


def main(argv=None):
    args = parse_args(argv)
    rank, world_size, local_rank = setup_distributed()
    seed_everything(args.seed, rank)

    device = torch.device(args.device if args.device != "cuda" else f"cuda:{local_rank}")
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16

    if args.toy:
        token_map = tiny_token_map()
        model = toy_model(token_map, dtype=dtype, seed=args.seed)
        attach_speaker_head(model, speaker_dim=args.speaker_dim, dtype=dtype)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --toy")
        checkpoint_dir = pathlib.Path(args.checkpoint)
        config_path = checkpoint_dir.parent / "config.json"
        token_map = _tokens.TokenMap.from_config(json.loads(config_path.read_text()))
        model = released_model(checkpoint_dir, dtype=dtype)

    if args.mode == "lora":
        model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    elif args.mode == "full":
        wrapped = wrap_ternary(model, rule=args.qat_rule, resume=not args.toy)
        if not args.toy and len(wrapped) != 196:
            raise SystemExit(f"QAT wrapped {len(wrapped)} linears, expected 196")

    model = model.to(device)
    trainable = set_trainable(model, args.mode)
    cast_trainable_to_float32(model)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False})
        # Without this the checkpointed segments receive an input that does not
        # require grad whenever the embedding is frozen, and every body gradient
        # silently comes back zero.
        model.enable_input_require_grads()

    dataset = SyntheticDataset(token_map, size=args.dataset_size,
                               speaker_dim=args.speaker_dim, seed=args.seed)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        collate_fn=Collator(), drop_last=True)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=args.ddp_find_unused)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    # A resume restores the model, the optimizer moments, the schedule position
    # and the RNG streams, which is what `selfcheck.check_resume` verifies. It
    # does *not* fast-forward the shuffled data stream: the restored RNG puts the
    # next epoch's permutation back, but the examples already consumed inside the
    # current epoch are seen again. Harmless over a multi-epoch run, worth knowing
    # before reading a seam in the loss curve as a bug.
    start_step = load_checkpoint(args.resume, model, optimizer, scheduler,
                                 map_location=device) if args.resume else 0

    logger = JsonlLogger(pathlib.Path(args.out) / "train.jsonl" if args.out else None, rank)
    logger.write({"event": "start", "mode": args.mode, "world_size": world_size,
                  "trainable_params": sum(trainable.values()),
                  "trainable_tensors": len(trainable), "device": str(device),
                  "qat_rule": args.qat_rule if args.mode == "full" else None})
    if rank == 0:
        print(f"mode={args.mode}  trainable {sum(trainable.values()) / 1e6:.1f}M "
              f"in {len(trainable)} tensors  device={device}  world_size={world_size}")

    history = run_steps(model, cycle(loader), optimizer, scheduler, args, device,
                        start_step=start_step, logger=logger)
    if rank == 0:
        print(f"first loss {history[0]:.4f}  last loss {history[-1]:.4f}")
        if args.out:
            save_checkpoint(pathlib.Path(args.out) / "final.pt", model, optimizer,
                            scheduler, args.steps, args)
    logger.close()
    return history


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", action="store_true",
                    help="2-layer model over a shrunk token layout; no checkpoint needed")
    ap.add_argument("--checkpoint", help="path to the repository's lm/ directory")
    ap.add_argument("--mode", default="embed", choices=["embed", "lora", "full"])
    ap.add_argument("--qat-rule", default="absmax", choices=["absmax", "absmean"],
                    help="ternary forward rule for --mode full; see qat.py")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "off"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-5, help="body learning rate")
    ap.add_argument("--lr-embed", type=float, default=1e-4,
                    help="embedding learning rate; higher than the body on purpose")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--dataset-size", type=int, default=256)
    ap.add_argument("--speaker-dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--ddp-find-unused", action="store_true",
                    help="only needed if a mode leaves a trainable parameter off the graph")
    ap.add_argument("--resume", help="checkpoint to continue from")
    ap.add_argument("--out", help="directory for checkpoints and train.jsonl")
    args = ap.parse_args(argv)
    if args.amp == "auto":
        # bf16 autocast on CPU is both slow and not bitwise reproducible, and the
        # self-check depends on CPU runs being reproducible.
        args.amp = "off" if args.device == "cpu" else "bf16"
    return args


if __name__ == "__main__":
    main()
