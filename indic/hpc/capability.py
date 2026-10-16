"""What the GPU in front of you can actually run, and which dtype is safe on it.

A heterogeneous cluster hands a job whatever card the scheduler had free. The same
sbatch file lands on an A100 one night and a V100 the next, and nothing in the
stack says so out loud: `torch.cuda.is_bf16_supported()` returns True on a V100
(it counts software emulation by default), a cubin-less wheel fails only when the
first kernel launches, and an fp16 run produces audio that is simply wrong rather
than an error. This module turns all of that into something a job can assert on
before it spends an hour.

The dtype verdict for this checkpoint, measured rather than assumed
----------------------------------------------------------------------
`ternary.py` stores one bf16 scale per 128-wide weight group, and the obvious
worry is that fp16's much narrower exponent range (normals start at 2^-14 =
6.1e-5, versus bf16's 1.2e-38) would flush small scales to zero and silently
delete 128 weights at a time. Measured on `lm/model-ternary.safetensors`
(11,010,048 scales across 196 tensors):

    smallest nonzero scale   2.689e-04      4.4x above fp16's smallest normal
    largest scale            7.852e-01
    scales that change under bf16 -> fp16 -> fp32   0

So the worry does not apply here. bf16's significand is 8 bits and fp16's is 11,
so every bf16 value whose magnitude lies in [2^-24, 65504] converts to fp16
exactly -- verified exhaustively over all 65,280 finite bf16 values -- and this
checkpoint's scales sit comfortably inside that band. Converting the scales is
lossless; a future checkpoint quantised to much smaller scales would not be, which
is why `ternary_scales_fit_fp16()` below re-checks a file rather than trusting
this paragraph.

**fp16 is still refused, for a different and much harder reason.** A single
prefill through the 28 layers (bf16 on CPU, 288-token reference-supervised prompt)
peaks at these magnitudes:

    model.layers.27 output            3,194,880      49x the fp16 ceiling
    model.layers.27.mlp.down_proj     3,162,112
    model.layers.26 output              643,072
    ...  33 of 424 module outputs exceed 65504

The residual stream carries massive activations -- a known Qwen-family trait, and
here amplified by every layer -- so an fp16 forward pass saturates to `inf` inside
the model and emits NaN logits. No loss scaling helps: this is inference, and the
overflow is in the activations themselves, not in gradients.

What that means per generation
------------------------------
  sm_80+ (A100, A30, L4, L40S, H100, H200, B200, RTX 30/40/50)
      bf16 natively. The supported configuration; this is what the checkpoint
      shipped as.
  sm_70 / sm_75 (V100, T4, RTX 20)
      No bf16 tensor cores, and fp16 is numerically impossible as shown above.
      PyTorch will still *run* bf16 on these cards by emulating it in fp32, which
      is correct but gives up the tensor cores entirely, and `is_bf16_supported()`
      will not warn you. The only honest options are fp32 (correct, 6.9 GB of
      weights instead of 3.5, no tensor cores) or a different partition.
      `select_dtype` therefore refuses pre-Ampere unless the caller names which
      of those it wants.

    python -m indic.hpc.capability            # readable report, nonzero exit on a blocker
    python -m indic.hpc.capability --json     # the same thing for a log directory
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Ampere. The first generation with bf16 tensor cores, and the cut below which
# this checkpoint has no usable half-precision dtype at all.
BF16_MIN_MAJOR = 8
# Flash-Attention 2 ships kernels for sm_80 and up only; so does PyTorch's own
# flash backend inside SDPA. Below that, SDPA still works through its
# memory-efficient and math backends.
FLASH_MIN_MAJOR = 8

# fp16 limits, as the numbers the report quotes.
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14

# 3.5 GB of bf16 weights, plus the s3gen vocoder, plus a KV cache for a prompt
# that reaches a few thousand tokens once a reference clip is in it. Below this a
# task OOMs partway through, which across an array sweep looks like tasks failing
# at random rather than like a card being too small. Double it for the fp32 path.
MIN_VRAM_GB = 8.0

# Minimum Linux driver for a CUDA runtime major, under NVIDIA's minor-version
# compatibility guarantee: any 12.x runtime runs on a 525.60.13+ driver, so the
# build's exact minor does not have to match the node's. Clusters pin drivers per
# node image and a straggler node with an old driver is a classic heterogeneous
# failure, visible here instead of as "CUDA driver version is insufficient".
MIN_DRIVER_FOR_CUDA_MAJOR = {11: "450.80.02", 12: "525.60.13", 13: "580.65.06"}

# Env vars that change what the probe sees. Printed with the report because a
# task that disagrees with its neighbours usually disagrees here first.
RELEVANT_ENV = ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST", "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK",
                "PYTORCH_CUDA_ALLOC_CONF", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
                "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
                "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE", "TMPDIR"]


class UnsupportedHardware(RuntimeError):
    """The selected device cannot run this checkpoint as asked."""


@dataclasses.dataclass(frozen=True)
class Device:
    """One visible GPU, as the process sees it after CUDA_VISIBLE_DEVICES."""
    index: int
    name: str
    major: int
    minor: int
    total_memory: int
    multi_processor_count: int
    uuid: str

    @property
    def sm(self):
        return f"sm_{self.major}{self.minor}"

    @property
    def capability(self):
        return (self.major, self.minor)

    @property
    def vram_gb(self):
        return self.total_memory / 1024 ** 3

    @property
    def bf16_native(self):
        """True when bf16 runs on tensor cores rather than fp32 emulation."""
        return self.major >= BF16_MIN_MAJOR

    @property
    def fp16_is_only_tensor_core_dtype(self):
        """Volta and Turing: fp16 tensor cores, no bf16 ones.

        True here is a warning, not a recommendation -- see the module docstring
        for why fp16 is unusable for *this* checkpoint specifically.
        """
        return self.major == 7

    @property
    def flash_attention_capable(self):
        return self.major >= FLASH_MIN_MAJOR

    def covered_by(self, arch_list):
        """True when the installed torch wheel carries code this card can run.

        Two separate mechanisms, and getting either wrong costs a job: a cubin for
        sm_M.n runs on sm_M.m for any m >= n (binary compatibility holds across
        minor versions within a major, never across majors), and embedded PTX from
        compute_A.b JIT-compiles forward onto anything newer. A card older than
        everything in the wheel -- a V100 under a cu128 wheel, which ships
        sm_75 upward -- is covered by neither, and fails at the first kernel
        launch with "no kernel image is available for execution on the device".
        """
        for entry in arch_list:
            kind, _, version = entry.partition("_")
            # Hopper and Blackwell add architecture-specific suffixes ("sm_90a",
            # "sm_100f"). The suffix narrows which kernels are allowed, not which
            # card the cubin loads on, so the digits are what matters here.
            version = version.rstrip("abcdef")
            if not version.isdigit() or len(version) < 2:
                continue
            other = (int(version[:-1]), int(version[-1]))
            if kind == "sm" and other[0] == self.major and other[1] <= self.minor:
                return True
            if kind == "compute" and other <= self.capability:
                return True
        return False


@dataclasses.dataclass
class Capability:
    """Everything the probe could determine, without allocating on any device."""
    torch_version: str
    torch_cuda_build: str
    cuda_available: bool
    arch_list: list
    driver_version: str
    cudnn_version: object
    nccl_version: str
    nccl_available: bool
    flash_attn_installed: bool
    flash_attn_version: str
    devices: list
    env: dict
    host: str

    def blockers(self, require_gpu=False):
        """Reasons this node cannot run the checkpoint, as printable strings."""
        problems = []
        if require_gpu and not self.devices:
            # A job step that asked for a GPU and got none would otherwise fall
            # back to the CPU and run for its whole wall clock at 1/100th speed.
            problems.append("no CUDA device is visible, and --require-gpu was given. "
                            "Check the GPU request in the sbatch file and "
                            "CUDA_VISIBLE_DEVICES.")
        for device in self.devices:
            if not device.covered_by(self.arch_list):
                problems.append(
                    f"cuda:{device.index} {device.name} is {device.sm}, and this "
                    f"torch wheel was built for {', '.join(self.arch_list)} — no "
                    f"kernel image will load. Install a torch build that targets "
                    f"{device.sm}.")
            elif not device.bf16_native:
                problems.append(
                    f"cuda:{device.index} {device.name} is {device.sm} (pre-Ampere): "
                    f"no bf16 tensor cores, and fp16 overflows this checkpoint's "
                    f"activations. Run it in fp32 (pre_ampere='fp32') or steer the "
                    f"job to sm_80+ with --constraint.")
        driver_problem = self.driver_problem()
        if driver_problem:
            problems.append(driver_problem)
        return problems

    def driver_problem(self):
        """A driver too old for the torch build's CUDA runtime, or None."""
        if not (self.driver_version and self.torch_cuda_build):
            return None
        major = int(self.torch_cuda_build.split(".")[0])
        minimum = MIN_DRIVER_FOR_CUDA_MAJOR.get(major)
        if minimum is None:
            return None
        if _version_tuple(self.driver_version) < _version_tuple(minimum):
            return (f"driver {self.driver_version} is below {minimum}, the minimum "
                    f"for a CUDA {major}.x runtime; this torch is built against "
                    f"CUDA {self.torch_cuda_build}")
        return None

    def warnings(self):
        """Things that will not stop the job but will skew or slow it."""
        notes = []
        if not self.cuda_available:
            notes.append("no CUDA device visible — CPU only. Fine for staging and "
                         "text-side checks; inference will be very slow.")
        generations = {d.capability for d in self.devices}
        if len(generations) > 1:
            notes.append(
                "mixed GPU generations visible in one process: "
                + ", ".join(f"cuda:{d.index}={d.sm}" for d in self.devices)
                + ". Collectives run at the weakest card's dtype support and "
                  "timings are not comparable across ranks.")
        memories = {round(d.vram_gb) for d in self.devices}
        if len(memories) > 1:
            notes.append(f"unequal VRAM across visible devices ({sorted(memories)} GB): "
                         "a batch size that fits rank 0 can OOM another rank.")
        for device in self.devices:
            if device.bf16_native and not self.flash_attn_installed:
                notes.append(
                    f"cuda:{device.index} {device.sm} could use flash-attention-2 "
                    "but the package is not installed; SDPA will be used instead.")
                break
        for device in self.devices:
            needed = MIN_VRAM_GB * (1 if device.bf16_native else 2)
            if device.vram_gb < needed:
                notes.append(
                    f"cuda:{device.index} {device.name} has {device.vram_gb:.1f} GB, "
                    f"below the ~{needed:.0f} GB this checkpoint needs in "
                    f"{'bf16' if device.bf16_native else 'fp32'} with the vocoder and "
                    f"a reference prompt resident. Expect OOM partway through a task.")
        for device in self.devices:
            if device.fp16_is_only_tensor_core_dtype:
                notes.append(
                    f"cuda:{device.index} {device.name}: fp16 is the only tensor-core "
                    "dtype this card has, and it is unusable for this checkpoint "
                    "(activations reach ~3.2e6 against an fp16 ceiling of 65504).")
        if self.devices and not self.nccl_available:
            notes.append("torch has no NCCL — multi-GPU training will not start.")
        return notes

    def to_dict(self, require_gpu=False):
        out = dataclasses.asdict(self)
        out["devices"] = [{**dataclasses.asdict(d), "sm": d.sm,
                           "vram_gb": round(d.vram_gb, 2),
                           "bf16_native": d.bf16_native,
                           "covered_by_build": d.covered_by(self.arch_list)}
                          for d in self.devices]
        out["blockers"] = self.blockers(require_gpu)
        out["warnings"] = self.warnings()
        return out


def _version_tuple(text):
    return tuple(int(p) for p in str(text).split(".") if p.isdigit())


def _driver_version():
    """Driver version from nvidia-smi, or "" when it cannot be asked.

    nvidia-smi rather than pynvml because it is present wherever the driver is,
    and pynvml is an extra dependency that a minimal compute-node env often lacks.
    """
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True)
    except (subprocess.SubprocessError, OSError):
        return ""
    return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""


def probe():
    """Read every property this module reports. Allocates nothing on any device.

    Safe to call on a login node and inside a job step that has not yet claimed
    its GPU: `get_device_properties` queries the driver, it does not create a
    context large enough to matter, and nothing here runs a kernel.
    """
    import torch

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(Device(
                index=index, name=properties.name,
                major=properties.major, minor=properties.minor,
                total_memory=properties.total_memory,
                multi_processor_count=properties.multi_processor_count,
                uuid=str(getattr(properties, "uuid", ""))))

    nccl_version = ""
    try:
        import torch.cuda.nccl
        nccl_version = ".".join(str(p) for p in torch.cuda.nccl.version())
    except Exception:
        # No NCCL in this build, or no CUDA at all. Reported as empty rather than
        # raising, because a CPU-only login node is a legitimate place to run this.
        pass

    flash_spec = importlib.util.find_spec("flash_attn")
    flash_version = ""
    if flash_spec is not None:
        try:
            from importlib.metadata import version
            flash_version = version("flash_attn")
        except Exception:
            flash_version = "unknown"

    return Capability(
        torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda or "",
        cuda_available=torch.cuda.is_available(),
        arch_list=list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
        driver_version=_driver_version(),
        cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        nccl_version=nccl_version,
        nccl_available=bool(torch.distributed.is_nccl_available()),
        flash_attn_installed=flash_spec is not None,
        flash_attn_version=flash_version,
        devices=devices,
        env={k: os.environ[k] for k in RELEVANT_ENV if k in os.environ},
        host=platform.node())


def device_of(device=None):
    """Resolve `device` to a `Device`, or None when the run is on CPU.

    Accepts an index, a "cuda:N" string, a torch.device, or None for the current
    device. None on CPU rather than an exception, because the callers below have
    a defined answer for CPU.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return None
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return Device(index=index, name=properties.name, major=properties.major,
                  minor=properties.minor, total_memory=properties.total_memory,
                  multi_processor_count=properties.multi_processor_count,
                  uuid=str(getattr(properties, "uuid", "")))


def select_dtype(device=None, pre_ampere="refuse"):
    """The dtype to load this checkpoint in on `device`.

    Returns bf16 on sm_80 and newer, which is what the checkpoint was released as
    and the only configuration `kittenml/kittentts2/model.py` treats as fast.

    On CPU, returns fp32: bf16 CPU kernels fall back to fp32 emulation without
    AVX512-BF16 or AMX, so fp32 is the same speed and fewer surprises. CPU is for
    staging and text-side checks, not for a sweep.

    **Pre-Ampere (sm_70 V100, sm_75 T4) is refused by default.** Those cards have
    fp16 tensor cores and no bf16 ones, and fp16 cannot run this checkpoint: a
    measured prefill reaches 3.19e6 in the residual stream against an fp16 ceiling
    of 65504, so the forward pass saturates to inf and the logits come out NaN.
    The per-group ternary scales themselves are *not* the problem -- all
    11,010,048 of them convert bf16 -> fp16 exactly (see the module docstring) --
    but nothing else about the model survives. bf16 does run on those cards, by
    emulating it in fp32 with the tensor cores idle, and `torch.cuda.
    is_bf16_supported()` reports True for exactly that reason, which is how a
    heterogeneous sweep quietly ends up with one arm five times slower than the
    rest.

    So the caller has to say what it wants:

        pre_ampere="refuse"  (default)  raise UnsupportedHardware
        pre_ampere="fp32"               torch.float32 -- correct, 6.9 GB of
                                        weights instead of 3.5, no tensor cores
        pre_ampere="fp16"               torch.float16 -- refused anyway, with the
                                        measurement, because it produces NaN

    Steering the job away is usually the right answer: see `--constraint` in
    `indic/hpc/slurm/eval_array.sbatch`.
    """
    import torch

    resolved = device_of(device)
    if resolved is None:
        return torch.float32
    if resolved.bf16_native:
        return torch.bfloat16

    where = f"cuda:{resolved.index} {resolved.name} ({resolved.sm})"
    if pre_ampere == "fp32":
        return torch.float32
    if pre_ampere == "fp16":
        raise UnsupportedHardware(
            f"{where}: fp16 is refused for this checkpoint regardless of the card. "
            f"A measured prefill peaks at 3.19e6 in the residual stream (layer 27) "
            f"against fp16's maximum of {FP16_MAX:g}; 33 of 424 module outputs "
            f"exceed it, so the forward pass saturates to inf and the logits are "
            f"NaN. The ternary scales would convert losslessly, but that is not "
            f"what breaks. Use pre_ampere='fp32', or an sm_80+ card.")
    if pre_ampere != "refuse":
        raise ValueError(f"pre_ampere must be 'refuse', 'fp32' or 'fp16', "
                         f"not {pre_ampere!r}")
    raise UnsupportedHardware(
        f"{where} is pre-Ampere: no bf16 tensor cores, and fp16 overflows this "
        f"checkpoint's activations. PyTorch would run bf16 here by emulating it in "
        f"fp32 — correct but with the tensor cores idle, and "
        f"torch.cuda.is_bf16_supported() will not tell you. Choose explicitly: "
        f"select_dtype(pre_ampere='fp32') to accept the slower correct path, or "
        f"steer the job to sm_{BF16_MIN_MAJOR}0+ with Slurm's --constraint.")


def select_attn_implementation(device=None, dtype=None):
    """The `attn_implementation` string to hand transformers for `device`.

    "flash_attention_2" only when the card is sm_80+, the package is importable
    and the dtype is half precision — flash-attn has no fp32 kernels and no
    pre-Ampere ones, and asking for it anyway raises inside `from_config`, which
    `model.py` catches and retries as sdpa. Deciding here instead means the choice
    is logged rather than discovered by exception.

    "sdpa" otherwise, which is available on every supported torch and picks its own
    backend per device: its flash backend needs sm_80+ too, but its memory-efficient
    and math backends cover everything older, including fp32.
    """
    import torch

    resolved = device_of(device)
    if resolved is None:
        return "sdpa"
    if dtype is None:
        dtype = select_dtype(device, pre_ampere="fp32")
    if (resolved.flash_attention_capable
            and dtype in (torch.float16, torch.bfloat16)
            and importlib.util.find_spec("flash_attn") is not None):
        return "flash_attention_2"
    return "sdpa"


def ternary_scales_fit_fp16(packed_path):
    """Re-check the fp16 claim against an actual packed checkpoint.

    The module docstring's numbers are from `KittenML/kitten-tts-2`. A checkpoint
    produced by the QAT path, or by `indic/vocab/build_sarvam_vocab.py` followed by
    a re-pack, has its own scale distribution, and a group whose scale falls below
    fp16's smallest normal (2^-14) would lose mantissa bits — below 2^-24 it would
    flush to zero and delete 128 weights outright. This answers that question for
    one file instead of assuming the released checkpoint's answer carries over.

    Returns a dict; `fits` is False when any scale would change.
    """
    import torch
    from safetensors import safe_open

    smallest = float("inf")
    largest = 0.0
    total = changed = subnormal = flushed = 0
    with safe_open(str(packed_path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "kittenml-ternary-1.58":
            raise ValueError(f"not a packed ternary checkpoint: {packed_path}")
        for key in handle.keys():
            if not key.endswith("::scale"):
                continue
            scale = handle.get_tensor(key)
            flat = scale.float().abs().reshape(-1)
            nonzero = flat[flat > 0]
            total += flat.numel()
            changed += int((scale.to(torch.float16).float() != scale.float()).sum())
            if nonzero.numel():
                smallest = min(smallest, float(nonzero.min()))
                largest = max(largest, float(nonzero.max()))
                subnormal += int((nonzero < FP16_MIN_NORMAL).sum())
                flushed += int((nonzero < 2.0 ** -24).sum())
    return {"path": str(packed_path), "n_scales": total,
            "min_nonzero": smallest if total else 0.0, "max": largest,
            "below_fp16_min_normal": subnormal, "would_flush_to_zero": flushed,
            "changed_by_fp16_roundtrip": changed, "fits": changed == 0}


def format_report(capability, require_gpu=False):
    lines = [f"host           {capability.host}",
             f"python         {platform.python_version()}",
             f"torch          {capability.torch_version}"
             f"   (built against CUDA {capability.torch_cuda_build or 'n/a'})",
             f"driver         {capability.driver_version or 'unknown'}",
             f"cudnn          {capability.cudnn_version or 'n/a'}",
             f"nccl           {capability.nccl_version or 'n/a'}"
             f"   (available: {capability.nccl_available})",
             f"flash-attn     {capability.flash_attn_version if capability.flash_attn_installed else 'not installed'}",
             f"arch list      {', '.join(capability.arch_list) or 'n/a'}",
             ""]

    if not capability.devices:
        lines.append("devices        none visible (CPU only)")
    else:
        lines.append(f"{'dev':<9}{'name':<34}{'sm':<9}{'VRAM':>8}  "
                     f"{'bf16':<9}{'flash':<7}{'in build':<9}")
        for device in capability.devices:
            lines.append(
                f"{'cuda:' + str(device.index):<9}{device.name[:33]:<34}{device.sm:<9}"
                f"{device.vram_gb:>7.1f}G  "
                f"{('native' if device.bf16_native else 'EMULATED'):<9}"
                f"{('yes' if device.flash_attention_capable else 'no'):<7}"
                f"{('yes' if device.covered_by(capability.arch_list) else 'NO'):<9}")

    lines += ["", "policy"]
    for device in capability.devices or [None]:
        label = f"cuda:{device.index}" if device else "cpu"
        try:
            dtype = select_dtype(device.index if device else None)
            attention = select_attn_implementation(device.index if device else None,
                                                   dtype=dtype)
            lines.append(f"  {label:<8} dtype {str(dtype).removeprefix('torch.'):<10} "
                         f"attn {attention}")
        except UnsupportedHardware as error:
            lines.append(f"  {label:<8} REFUSED — {error}")

    if capability.env:
        lines += ["", "environment"]
        lines += [f"  {k:<24} {v}" for k, v in capability.env.items()]

    warnings = capability.warnings()
    if warnings:
        lines += ["", "warnings"]
        lines += [f"  ! {w}" for w in warnings]

    blockers = capability.blockers(require_gpu)
    lines += ["", "blockers"]
    lines += [f"  X {b}" for b in blockers] if blockers else ["  none"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--out", type=pathlib.Path, help="also write the JSON here")
    ap.add_argument("--require-gpu", action="store_true",
                    help="treat 'no CUDA device' as a blocker (use inside a job step)")
    ap.add_argument("--check-scales", type=pathlib.Path, metavar="PACKED_SAFETENSORS",
                    help="re-measure one packed checkpoint's ternary scales against "
                         "fp16's range")
    args = ap.parse_args()

    capability = probe()
    payload = capability.to_dict(args.require_gpu)
    if args.check_scales:
        payload["ternary_scales"] = ternary_scales_fit_fp16(args.check_scales)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(capability, args.require_gpu))
        if args.check_scales:
            scales = payload["ternary_scales"]
            print(f"\nternary scales in {scales['path']}")
            print(f"  {scales['n_scales']:,} scales, "
                  f"min nonzero {scales['min_nonzero']:.3e}, max {scales['max']:.3e}")
            print(f"  changed by an fp16 round trip: "
                  f"{scales['changed_by_fp16_roundtrip']:,} "
                  f"({'lossless' if scales['fits'] else 'LOSSY'})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    return 1 if capability.blockers(args.require_gpu) else 0


if __name__ == "__main__":
    sys.exit(main())
