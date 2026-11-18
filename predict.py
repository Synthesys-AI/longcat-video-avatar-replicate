"""Replicate (Cog) runner for LongCat-Video-Avatar 1.6.

A port of Meituan's `run_demo_avatar_single_audio_to_video.py` (LongCat-Video @ 58eb336)
to a long-lived single-GPU Cog runner:

  * one process, context-parallel size 1 (the demo's torchrun with nproc_per_node=1);
  * the 1.6 recipe from the model card: distilled 8-step sampling with the DMD (loraA)
    and OPSD (loraB) LoRAs mixed per step (`mix_steps=4, mix_alpha=1.0` by default),
    bf16 DiT, 25 fps, no loudness normalisation;
  * stage 1 is speech+image->video (ai2v) when an image is given, speech+text->video
    (at2v) otherwise; longer speech is covered by audio-driven video continuation
    (93 frames per segment, 13 of them carried over), exactly as the demo does.

Weights (~63 GB) are downloaded from pinned Hugging Face revisions in setup().
"""

import collections
import contextlib
import functools
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path as FsPath
from typing import Optional

# Both weight repos live on Hugging Face's Xet storage, which huggingface_hub fetches with
# hf_xet (HF_HUB_ENABLE_HF_TRANSFER has no effect on Xet files). High-performance mode starts
# with more parallel streams and sizes its buffers from RAM.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Lets PyTorch grow its memory segments instead of leaving reserved-but-unusable gaps (the
# first 720p run failed with 4.5 GB reserved and unallocated). Must be set before CUDA starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# LongCat wraps the small sparse-attention helpers (block pooling, top-k) in torch.compile.
# Eager they cost milliseconds; compiled, every new frame size would recompile mid-prediction.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from cog import BaseRunner, Input, Path

LONGCAT_SRC = "/opt/LongCat-Video"
if LONGCAT_SRC not in sys.path:
    sys.path.insert(0, LONGCAT_SRC)

AVATAR_REPO = "meituan-longcat/LongCat-Video-Avatar-1.6"
AVATAR_REVISION = "acfc628244201fc53fb6fe8a8903c5bac9bdf68f"
BASE_REPO = "meituan-longcat/LongCat-Video"
BASE_REVISION = "03b55529b1d1d4045f5fbe14d65c8c6e8116b278"

WEIGHTS_ROOT = FsPath(os.environ.get("LONGCAT_WEIGHTS", "/src/weights"))
AVATAR_DIR = WEIGHTS_ROOT / "LongCat-Video-Avatar-1.6"
BASE_DIR = WEIGHTS_ROOT / "LongCat-Video"

# Only what the single-person 1.6 path loads. Skips the INT8 DiT, whisper's fp32 /
# flax / .bin duplicates and the demo videos (~30 GB we'd never read).
AVATAR_FILES = [
    "config.json",
    "model_index.json",
    "base_model/*",
    "loraA/*",
    "loraB/*",
    "scheduler/*",
    "vocal_separator/*",
    "whisper-large-v3/config.json",
    "whisper-large-v3/generation_config.json",
    "whisper-large-v3/preprocessor_config.json",
    "whisper-large-v3/model.safetensors",
    "whisper-large-v3/added_tokens.json",
    "whisper-large-v3/merges.txt",
    "whisper-large-v3/normalizer.json",
    "whisper-large-v3/special_tokens_map.json",
    "whisper-large-v3/tokenizer.json",
    "whisper-large-v3/tokenizer_config.json",
    "whisper-large-v3/vocab.json",
]
BASE_FILES = ["tokenizer/*", "text_encoder/*", "vae/*"]

# The 1.6 recipe (the demo maps --model_type avatar-v1.6 onto the v1.5 code path).
MODEL_TYPE = "avatar-v1.5"
FPS = 25
AUDIO_STRIDE = 1
NUM_FRAMES = 93
NUM_COND_FRAMES = 13
NUM_STEPS = 8
GUIDANCE = 1.0
REF_IMG_INDEX = 10
MASK_FRAME_RANGE = 0

# A full first segment is 93 frames (3.72 s); each continuation adds 80 new frames (3.2 s).

# Replicate stops a prediction at 30 min. Stop starting new segments in time to
# finish the one in flight and encode, and return the video made so far.
TIME_BUDGET_SECONDS = 26 * 60
# Measured on one H100 with FlashAttention 3: ~50 s per 480p segment (3.2 s of new video;
# 58.6 s with FA2), plus ~50 s of audio prep and encoding. 90 s of video is 28 segments,
# about 25 min. 720p is NOT measured yet (estimated ~4x slower per segment).
MAX_DURATION = {"480p": 90.0, "720p": 15.0}

NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
    "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
    "many people in the background, walking backwards"
)
DEFAULT_PROMPT = (
    "A person is speaking naturally to the camera with expressive facial movements and "
    "subtle head and hand gestures."
)

# Frame size for speech+text mode (with an image, the image's aspect picks the size).
TEXT_MODE_SIZES = {
    "480p": {"16:9": (480, 832), "9:16": (832, 480), "1:1": (640, 640)},
    "720p": {"16:9": (768, 1280), "9:16": (1280, 768), "1:1": (960, 960)},
}

# Experimental block-sparse attention (LongCat's own Triton kernel, `sparse_attention` input).
# A block is one latent frame by 8x8 latent tokens (128x128 px); each block of queries
# attends only to the top (1 - sparsity) share of key blocks. Frame sizes must divide into
# 128-px blocks, so while it is on, sizes come from LongCat's 128-px bucket table.
SPARSE_CHUNK = (1, 8, 8)
SPARSE_MIN_QUERY_FRAMES = 8  # the reference/carried-frame attentions (1-5 frames) stay dense
TEXT_MODE_SIZES_SPARSE = {
    "480p": {"16:9": (512, 896), "9:16": (896, 512), "1:1": (640, 640)},
    "720p": {"16:9": (768, 1280), "9:16": (1280, 768), "1:1": (1024, 1024)},
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torch_gc() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# SageAttention2 (8-bit attention: INT8 QK, FP8 PV) from Hugging Face's kernel hub, which
# only has builds for torch 2.8+. Copied into the image by cog.yaml (keep in sync).
SAGE_KERNEL_REPO = "kernels-community/sage-attention"
SAGE_KERNEL_REVISION = "f81ef5e170c530160930bbb3394df612800e98e7"
SAGE_BAKED_DIR = FsPath("/opt/sage-attention")


def _load_sage():
    """Import the hub SageAttention package and return its H100 kernel function. (The
    package's own `sageattn()` prints a debug line on every call, so it is bypassed.)"""
    variant = _fa3_variant(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI, torch.version.cuda or "")
    rel = f"build/{variant}"
    pkg, where = _find_hub_package(SAGE_BAKED_DIR, rel, "_sage_attention_"), "image"
    if pkg is None:
        from huggingface_hub import snapshot_download

        root, where = WEIGHTS_ROOT / "sage-attention", "downloaded"
        snapshot_download(repo_id=SAGE_KERNEL_REPO, revision=SAGE_KERNEL_REVISION, local_dir=str(root),
                          allow_patterns=[f"{rel}/*"])
        pkg = _find_hub_package(root, rel, "_sage_attention_")
        if pkg is None:
            raise RuntimeError(f"no SageAttention kernel build for {variant}")
    _import_hub_package(pkg, "sage_attention_hub")
    core = sys.modules["sage_attention_hub.core"]
    return core.sageattn_qk_int8_pv_fp8_cuda_sm90, f"{SAGE_KERNEL_REPO}@{SAGE_KERNEL_REVISION[:7]} {variant} ({where})"


class _SageAttention:
    """Dense attention through SageAttention2's H100 kernel, on LongCat's [B, S, H, D]
    tensors. Used for a request only when it sets `sage_attention`."""

    def __init__(self, kernel=None):
        self.kernel = kernel
        self.available = False
        self.calls = 0

    def attend(self, q, k, v, softmax_scale=None):
        self.calls += 1
        return self.kernel(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=softmax_scale,
                           pv_accum_dtype="fp32+fp32")

    def self_test(self) -> str:
        """Against PyTorch SDPA, with more keys than queries (a continuation chunk) and a
        key length that is not a multiple of 128 (the kernel pads it)."""
        if torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError(f"needs an H100 (sm90), got sm{''.join(map(str, torch.cuda.get_device_capability()))}")
        g = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, 1500, 8, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 1500, 8, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        q[:, :, :, :16] *= 4  # peakier attention than plain noise
        scale = 128 ** -0.5
        out = self.attend(q, k, v, scale).float()
        ref = _sdpa(q, k, v, softmax_scale=scale).float()
        rel = ((out - ref).norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        if out.shape != q.shape or not (rel < 0.08 and cos > 0.995):
            raise RuntimeError(f"self-test mismatch: rel err {rel:.4f}, cosine {cos:.5f}")
        self.calls = 0
        self.available = True
        return f"self-test rel err {rel:.4f}, cosine {cos:.5f}"


# Experimental FP8 matmuls (`fp8_matmuls` input) for the DiT's big linear layers: each
# input row and each weight row is scaled into float8 e4m3 on the fly (no stored copy,
# so no extra GPU memory), multiplied with the H100's FP8 tensor cores via
# torch._scaled_mm, and accumulated in bf16. LoRA branches and small layers stay bf16.
_FP8_ON = [False]
FP8_MIN_ROWS = 256  # smaller matmuls (timestep / text projections) aren't worth it


def _fp8_targets(dit) -> list:
    """The DiT's big linear layers: inside the transformer blocks, not the adaLN
    modulation (it maps a 512-d timestep embedding), both sides >= 1024 and multiples of
    16 (FP8 tensor-core shapes)."""
    out = []
    for name, m in dit.named_modules():
        if not isinstance(m, torch.nn.Linear) or not name.startswith("blocks."):
            continue
        if "adaLN" in name or "modulation" in name:
            continue
        if min(m.in_features, m.out_features) < 1024 or m.in_features % 16 or m.out_features % 16:
            continue
        out.append((name, m))
    return out


def _install_fp8(dit, fp8_linear, counter: list) -> int:
    """Route the target layers through `fp8_linear` while `_FP8_ON[0]` is set.

    LongCat's LoRA wrapper replaces a layer's `forward` and calls `layer.org_forward`
    at call time (and moves `forward` back to `org_forward` when LoRAs are disabled), so
    the FP8-capable forward goes into `org_forward` when the layer is wrapped, else into
    `forward`. Either way it survives every enable/disable cycle, and the LoRA branches
    keep running in bf16 on top."""
    targets = _fp8_targets(dit)
    for _, lin in targets:
        lin._fp8 = (fp8_linear, counter)

        def forward(x, lin=lin):
            return _linear_maybe_fp8(lin, x, lin.weight)

        if hasattr(lin, "org_forward"):
            lin.org_forward = forward
        else:
            lin.forward = forward
    return len(targets)


def _linear_maybe_fp8(lin, x, weight):
    """`lin`'s linear with `weight`: the FP8 linear while `_FP8_ON[0]` is set on a layer
    `_install_fp8` picked (and the call is big enough), else plain bf16 F.linear."""
    fp8 = getattr(lin, "_fp8", None)
    if fp8 is not None and _FP8_ON[0] and x.numel() >= FP8_MIN_ROWS * x.shape[-1]:
        fp8_linear, counter = fp8
        counter[0] += 1
        return fp8_linear(x, weight, lin.bias)
    return torch.nn.functional.linear(x, weight, lin.bias)


def _make_fp8_linear():
    """Build the FP8 linear: a Triton kernel that quantizes each row in one pass
    (amax -> scale -> float8 e4m3), then torch._scaled_mm with per-row scales."""
    import triton
    import triton.language as tl

    @triton.jit
    def _rowwise_fp8(x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-6) / 448.0
        tl.store(q_ptr + row * stride_q + offs, (x / scale).to(tl.float8e4nv), mask=mask)
        tl.store(s_ptr + row, scale)

    def quantize_rows(x2d):
        rows, k = x2d.shape
        q = torch.empty((rows, k), device=x2d.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty((rows,), device=x2d.device, dtype=torch.float32)
        block = triton.next_power_of_2(k)
        _rowwise_fp8[(rows,)](x2d, q, scales, k, x2d.stride(0), q.stride(0), BLOCK=block,
                              num_warps=16 if block >= 8192 else 8)
        return q, scales

    def fp8_linear(x, weight, bias):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        xq, xs = quantize_rows(x2)
        wq, ws = quantize_rows(weight.contiguous())
        out = torch._scaled_mm(xq, wq.t(), scale_a=xs.view(-1, 1), scale_b=ws.view(1, -1),
                               out_dtype=x.dtype, use_fast_accum=True)
        if bias is not None:
            out = out + bias
        return out.view(*shape[:-1], weight.shape[0])

    return fp8_linear


def _fp8_self_test(fp8_linear) -> str:
    """Against bf16 F.linear on a DiT-sized layer (needs an H100: FP8 rowwise scaling)."""
    if torch.cuda.get_device_capability() < (9, 0):
        raise RuntimeError("needs an H100 (sm90) for FP8 matmuls")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    w = (torch.randn(12288, 4096, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(12288, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x, w, b).float()
    out = fp8_linear(x, w, b).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    if out.shape != ref.shape or not (rel < 0.06 and cos > 0.998):
        raise RuntimeError(f"self-test mismatch: rel err {rel:.4f}, cosine {cos:.5f}")
    return f"self-test rel err {rel:.4f}, cosine {cos:.5f}"


# Merged add-ons (`merge_lora` input, on by default). LongCat runs each LoRA add-on as a
# side branch of its layer: x·Aᵀ, then ·Bᵀ, two scalings and two additions, each a full
# pass over the activations (~37k tokens × up to 16k features per call at 480p). The
# profile put that at ~5.5 s of a 43 s segment. Here, while an add-on is active and the
# call is big enough, the layer's effective weight W + s·B·A is built in a temporary
# buffer (one addmm per add-on) and the activations go through ONE matmul. W itself is
# never written to, so nothing drifts from video to video, and the add-on state (the
# DMD/OPSD switch at `mix_steps`) is still read at call time.
_MERGE_ON = [False]
# Building W' costs ~2 passes over W; the branch costs ~9 passes over the output. So merge
# when rows >= in_features / 4 (video tokens); short calls (text tokens) keep the branch.
MERGE_ROWS_PER_FEATURE = 0.25


def _merged_weight(weight, active):
    """weight + sum(scale * up @ down) over the `active` (lora, scale) pairs, in a new
    buffer; `weight` is untouched. `up` is a Linear or LongCat's LoRAUPParallel, whose
    block i maps the i-th rank slice of the down projection to the i-th slice of rows."""
    out = torch.empty_like(weight)
    for k, (lora, scale) in enumerate(active):
        down = lora.lora_down.weight
        up = lora.lora_up
        ups = [b.weight for b in up.blocks] if hasattr(up, "blocks") else [up.weight]
        r, rows = down.shape[0] // len(ups), out.shape[0] // len(ups)
        for i, b in enumerate(ups):
            rs, ds = slice(i * rows, (i + 1) * rows), down[i * r:(i + 1) * r]
            if k == 0:
                torch.addmm(weight[rs], b, ds, alpha=scale, out=out[rs])
            else:
                out[rs].addmm_(b, ds, alpha=scale)
    return out


def _make_merging_factory(branch_factory, counter: list):
    """Wrap LongCat's `dit._create_multi_lora_forward(module, loras)` (called by
    `enable_loras`: at setup, and on any request `_set_lora_mix` sends through LongCat's
    `enable_lora_mix`): the forward it returns merges
    while `_MERGE_ON[0]` is set and the call is big enough, else runs LongCat's branch."""

    def create(module, loras):
        branch = branch_factory(module, loras)
        if type(module) is not torch.nn.Linear:  # e.g. a QuantizedLinear: keep the branch
            return branch
        scales = [float(lora.alpha_scale) for lora in loras]  # constant buffer: read once
        min_rows = module.in_features * MERGE_ROWS_PER_FEATURE

        def forward(x, *args, **kwargs):
            if (not _MERGE_ON[0] or args or kwargs or torch.is_grad_enabled()
                    or x.numel() < min_rows * x.shape[-1]):
                return branch(x, *args, **kwargs)  # (with grad on, W' would cut the add-ons' grads)
            active = [(lora, lora.multiplier * sc) for lora, sc in zip(loras, scales) if lora.use_lora]
            if not active:
                return branch(x)
            counter[0] += 1
            return _linear_maybe_fp8(module, x, _merged_weight(module.weight, active))

        forward.merges_lora, forward.loras = True, loras
        return forward

    return create


def _merge_self_test(dit, rows: int = 2048) -> str:
    """Merged vs LongCat's branch path on the biggest wrapped layer (and the biggest one
    with a split up-projection, if any), in both add-on states of the mix."""
    wrapped = [m for _, m in dit.named_modules()
               if getattr(m.forward, "merges_lora", False) and type(m) is torch.nn.Linear]
    if not wrapped:
        raise RuntimeError("no add-on layers went through the merging forward")
    split = [m for m in wrapped if any(hasattr(lora.lora_up, "blocks") for lora in m.forward.loras)]
    layers = [max(group, key=lambda m: m.weight.numel()) for group in (wrapped, split) if group]
    worst_rel, worst_cos, share = 0.0, 1.0, 1.0
    try:
        with torch.no_grad():
            for step in (0, 10**6):  # before and after the mix's switch
                dit.set_lora_mix_step(step)
                for layer in layers:
                    g = torch.Generator(device=layer.weight.device).manual_seed(0)
                    x = torch.randn(rows, layer.in_features, device=layer.weight.device,
                                    dtype=layer.weight.dtype, generator=g)
                    _MERGE_ON[0] = False
                    ref = layer(x).float()
                    base = layer.org_forward(x).float()
                    _MERGE_ON[0] = True
                    out = layer(x).float()
                    rel = ((out - ref).norm() / ref.norm()).item()
                    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
                    worst_rel, worst_cos = max(worst_rel, rel), min(worst_cos, cos)
                    share = min(share, ((ref - base).norm() / ref.norm()).item())
    finally:
        _MERGE_ON[0] = False
        dit.set_lora_mix_step(0)
    if not (worst_rel < 0.02 and worst_cos > 0.9998):
        raise RuntimeError(f"self-test mismatch: rel err {worst_rel:.4f}, cosine {worst_cos:.6f}")
    return (f"{len(wrapped)} layers, self-test rel err {worst_rel:.4f}, cosine {worst_cos:.6f}, "
            f"add-on share of the output >= {share:.3f}")


# Fused small steps (`fused_ops` input, on by default). The profile put about a third of
# each segment in small element-by-element steps that LongCat runs as separate fp32
# kernels, each a full pass over the activations: the LayerNorm + shift/scale
# ("modulate_fp32"), the affine LayerNorms before cross-attention, RMSNorm on q/k followed
# by 3D RoPE, and the FFN's SiLU * gate. Each becomes one Triton kernel: read bf16 once,
# do the same fp32 maths in registers (rounding to bf16 where LongCat does), write bf16
# once. The self-attention q/k come out already in FlashAttention's [B, S, H, D] layout,
# which also saves the copy LongCat makes before every attention call. Each kernel is
# checked against LongCat's own code at boot; one that fails keeps LongCat's path.
_FUSE_ON = [False]
_FUSED = {}  # op name -> kernel wrapper; only ops whose boot self-test passed
_FUSE_DEVICE = ["cuda"]  # tests route CPU tensors through stand-in kernels


def _fusable(x, dims=None) -> bool:
    return (x.device.type == _FUSE_DEVICE[0] and x.dtype == torch.bfloat16
            and (dims is None or x.dim() == dims) and x.stride(-1) == 1)


def _make_fused_kernels() -> dict:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["C", "S", "T", "a_sb", "a_st", "c_sb", "c_st"])
    def _ln_kernel(x_ptr, out_ptr, a_ptr, c_ptr, C, S, T, a_sb, a_st, c_sb, c_st, eps,
                   ADD_ONE: tl.constexpr, BLOCK: tl.constexpr):
        # One row of C: LayerNorm in fp32, then * a + c, where a/c are per (batch, frame)
        # (modulate: a = 1 + scale, c = shift) or per channel (affine: weight, bias).
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        d = tl.where(mask, x - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(d * d, axis=0) / C + eps)
        bt = row // S
        b = bt // T
        t = bt % T
        a = tl.load(a_ptr + b * a_sb + t * a_st + offs, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + b * c_sb + t * c_st + offs, mask=mask, other=0.0).to(tl.float32)
        if ADD_ONE:
            a = a + 1.0
        y = d * rstd * a + c
        tl.store(out_ptr + row * C + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit(do_not_specialize=["N", "sb", "sh", "sn", "ob", "oh", "on"])
    def _qk_rope_kernel(x_ptr, out_ptr, w_ptr, cos_ptr, sin_ptr, N, sb, sh, sn, ob, oh, on, eps,
                        H: tl.constexpr, D: tl.constexpr):
        # One token, all H heads of a [B, H, N, D] view: RMSNorm as LongCat does it
        # ((x * rsqrt(mean(x^2) + eps)) -> bf16, * weight -> bf16), then RoPE with this
        # token's cos/sin row, loaded once for all heads. rotate_half maps each pair
        # (2j, 2j+1) to (-x[2j+1], x[2j]), so every element reads its partner d ^ 1.
        pid = tl.program_id(0).to(tl.int64)
        b = pid // N
        n = pid % N
        h = tl.arange(0, H)
        d = tl.arange(0, D)
        base = b * sb + n * sn + h[:, None] * sh
        x = tl.load(x_ptr + base + d[None, :]).to(tl.float32)
        xp = tl.load(x_ptr + base + (d ^ 1)[None, :]).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        w = tl.load(w_ptr + d).to(tl.float32)
        wp = tl.load(w_ptr + (d ^ 1)).to(tl.float32)
        nx = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        nx = (nx * w[None, :]).to(tl.bfloat16).to(tl.float32)
        npart = (xp * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        npart = (npart * wp[None, :]).to(tl.bfloat16).to(tl.float32)
        cs = tl.load(cos_ptr + n * D + d)
        sn_ = tl.load(sin_ptr + n * D + d)
        sign = tl.where(d % 2 == 0, -1.0, 1.0)
        y = nx * cs[None, :] + (sign[None, :] * npart) * sn_[None, :]
        tl.store(out_ptr + b * ob + n * on + h[:, None] * oh + d[None, :], y.to(out_ptr.dtype.element_ty))

    @triton.jit(do_not_specialize=["R", "d1", "d2", "s0", "s1", "s2", "o0", "o1", "o2"])
    def _rms_kernel(x_ptr, out_ptr, w_ptr, R, d1, d2, s0, s1, s2, o0, o1, o2, eps,
                    D: tl.constexpr, BLOCK_R: tl.constexpr):
        # BLOCK_R rows of D from a strided [d0, d1, d2, D] tensor, RMSNorm as above.
        r = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = (r < R)[:, None]
        i2 = r % d2
        i1 = (r // d2) % d1
        i0 = r // (d2 * d1)
        d = tl.arange(0, D)
        x = tl.load(x_ptr + (i0 * s0 + i1 * s1 + i2 * s2)[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        nx = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        nx = nx * tl.load(w_ptr + d).to(tl.float32)[None, :]
        tl.store(out_ptr + (i0 * o0 + i1 * o1 + i2 * o2)[:, None] + d[None, :],
                 nx.to(out_ptr.dtype.element_ty), mask=m)

    @triton.jit(do_not_specialize=["n"])
    def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
        # LongCat: F.silu(a) (bf16) * b (bf16)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = (a / (1.0 + tl.exp(-a))).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offs, (s * b).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit(do_not_specialize=["C", "S", "T", "g_sb", "g_st"])
    def _gated_add_kernel(x_ptr, s_ptr, g_ptr, out_ptr, C, S, T, g_sb, g_st, BLOCK: tl.constexpr):
        # LongCat's residual update: (x + gate * s) in fp32 -> bf16, gate per (batch, frame).
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < C
        bt = row // S
        b = bt // T
        t = bt % T
        x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + b * g_sb + t * g_st + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * C + offs, (x + g * sv).to(out_ptr.dtype.element_ty), mask=mask)

    def ln_launch(x, a, c, a_strides, c_strides, S, T, eps, add_one):
        C = x.shape[-1]
        out = torch.empty_like(x)
        block = triton.next_power_of_2(C)
        _ln_kernel[(x.numel() // C,)](x, out, a, c, C, S, T, *a_strides, *c_strides, float(eps),
                                      ADD_ONE=add_one, BLOCK=block, num_warps=8 if block >= 2048 else 4)
        return out

    def modulate(x, eps, scale, shift):  # x [B, T, S, C] contiguous; scale/shift [B, T, 1, C]
        B, T, S, C = x.shape
        return ln_launch(x, scale, shift, (scale.stride(0), scale.stride(1)),
                         (shift.stride(0), shift.stride(1)), S, T, eps, True)

    def layer_norm(x, eps, weight, bias):  # x [..., C] contiguous
        rows = x.numel() // x.shape[-1]
        return ln_launch(x, weight, bias, (0, 0), (0, 0), max(rows, 1), 1, eps, False)

    def rms(x, weight, eps):  # any [d0, d1, d2, D] view; dense inputs keep their layout
        d0, d1, d2, D = x.shape
        out = torch.empty_like(x)
        R = d0 * d1 * d2
        _rms_kernel[(triton.cdiv(R, 32),)](
            x, out, weight, R, d1, d2, x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2), float(eps), D=D, BLOCK_R=32, num_warps=4)
        return out

    def qk_rope(x, weight, eps, cos, sin):  # x [B, H, N, D] view -> same shape, stored [B, N, H, D]
        B, H, N, D = x.shape
        out = torch.empty((B, N, H, D), device=x.device, dtype=x.dtype).permute(0, 2, 1, 3)
        _qk_rope_kernel[(B * N,)](x, out, weight, cos, sin, N, x.stride(0), x.stride(1), x.stride(2),
                                  out.stride(0), out.stride(1), out.stride(2), float(eps),
                                  H=H, D=D, num_warps=8)
        return out

    def silu_mul(a, b):
        out = torch.empty_like(a)
        n = a.numel()
        _silu_mul_kernel[(triton.cdiv(n, 4096),)](a, b, out, n, BLOCK=4096, num_warps=8)
        return out

    def gated_add(x, gate, xs, T):  # x, xs [B, N, C] contiguous; gate [B, T, 1, C] fp32
        B, N, C = x.shape
        out = torch.empty_like(x)
        block = triton.next_power_of_2(C)
        _gated_add_kernel[(B * N,)](x, xs, gate, out, C, N // T, T, gate.stride(0), gate.stride(1),
                                    BLOCK=block, num_warps=8 if block >= 2048 else 4)
        return out

    return {"modulate": modulate, "layer_norm": layer_norm, "qk_rope": qk_rope, "rms": rms,
            "silu_mul": silu_mul, "gated_add": gated_add}


def _rope_cos_sin(rope, q, grid_size, frame_index=None, num_ref_latents=None, token_offset=0):
    """The cos/sin rows LongCat's RotaryPositionalEmbedding.forward (@58eb336) uses for
    this call, as [tokens, D] fp32, through its own caches (same keys, same values)."""
    key_name = '.'.join([str(i) for i in grid_size]) + f"-{str(frame_index)}-{str(num_ref_latents)}"
    if key_name not in rope.freqs_dict:
        rope.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)
    cache_key = (rope.head_dim, tuple(rope.cp_split_hw or ()), key_name, q.device.type, q.device.index)
    cached = rope._gpu_cos_sin_cache.get(cache_key)
    if cached is None:
        freqs_cis = rope.freqs_dict[key_name].to(q.device).float()
        cached = (freqs_cis.cos()[None, None], freqs_cis.sin()[None, None])  # 'n d -> 1 1 n d'
        rope._gpu_cos_sin_cache[cache_key] = cached
    cos, sin = cached
    n = q.shape[-2]
    cos = cos[0, 0, token_offset:token_offset + n]
    sin = sin[0, 0, token_offset:token_offset + n]
    if cos.shape[0] != n:
        raise ValueError(f"RoPE table has {cos.shape[0]} rows for {n} tokens")
    return cos, sin


def _gated_residual(x, gate, xs, T, counter):
    """LongCat's residual update `(x + gate * xs).to(x.dtype)`, computed in fp32, with
    gate [B, T, 1, C] per frame: one fused kernel when the tensors allow it."""
    B, N, C = x.shape
    fn = _FUSED.get("gated_add")
    if (fn is not None and _fusable(x, 3) and x.is_contiguous() and _fusable(xs) and xs.is_contiguous()
            and xs.numel() == x.numel() and gate.dtype == torch.float32 and gate.stride(-1) == 1
            and tuple(gate.shape) == (B, T, 1, C) and N % T == 0):
        counter["gated_add"] += 1
        return fn(x, gate, xs, T)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        out = x + (gate * xs.view(B, -1, N // T, C)).view(B, -1, C)
    return out.to(x.dtype)


def _fused_block_forward(self, mod, counter, x, y, t, y_seqlen, latent_shape, num_cond_latents=None,
                         return_kv=False, kv_cache=None, skip_crs_attn=False, num_ref_latents=None,
                         audio_hidden_states=None, ref_img_index=None, mask_frame_range=None,
                         token_ref_target_masks=None, human_num=None):
    """LongCatAvatarSingleStreamBlock.forward (@58eb336), line for line, except that the
    three gated residual updates go through `_gated_residual` (one kernel each instead of
    a multiply, an add and a cast over the whole residual stream)."""
    amp32 = functools.partial(torch.amp.autocast, device_type="cuda", dtype=torch.float32)
    x_dtype = x.dtype
    B, N, C = x.shape
    T, _, _ = latent_shape
    with amp32():
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
    x_m = mod.modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)
    if kv_cache is not None:
        kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
        attn_outputs = self.attn.forward_with_kv_cache(
            x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache,
            num_ref_latents=num_ref_latents, ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
    else:
        attn_outputs = self.attn(
            x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv,
            num_ref_latents=num_ref_latents, ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
    if return_kv:
        x_s, kv_cache, x_ref_attn_map = attn_outputs
    else:
        x_s, x_ref_attn_map = attn_outputs
    x = _gated_residual(x, gate_msa, x_s, T, counter).to(x_dtype)

    if not skip_crs_attn:
        if kv_cache is not None:
            num_cond_latents = None
        x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents,
                                shape=latent_shape)
    if not skip_crs_attn:
        if kv_cache is not None:
            num_cond_latents = 0
        with amp32():
            audio_shift_mca, audio_scale_mca, audio_gate_mca = \
                self.audio_adaLN_modulation(t[:, num_cond_latents:]).unsqueeze(2).chunk(3, dim=-1)
        audio_output_cond, audio_output_noise = self.audio_cross_attn(
            self.pre_video_crs_attn_norm(x), self.pre_audio_crs_attn_norm(audio_hidden_states),
            shape=latent_shape, num_cond_latents=num_cond_latents, x_ref_attn_map=x_ref_attn_map,
            human_num=human_num)
        t_noise = T - num_cond_latents
        with amp32():
            audio_output_noise = mod.modulate_fp32(self.mod_norm_attn, audio_output_noise.view(B, t_noise, -1, C),
                                                   audio_shift_mca, audio_scale_mca).view(B, -1, C)
        if audio_output_cond is None:
            x = _gated_residual(x, audio_gate_mca, audio_output_noise, t_noise, counter).to(x_dtype)
        else:
            with amp32():
                audio_add_x = (audio_gate_mca * audio_output_noise.view(B, t_noise, -1, C)).view(B, -1, C)
                audio_add_x = torch.cat([audio_output_cond, audio_add_x], dim=1).contiguous()
            x = (x + audio_add_x).to(x_dtype)

    x_m = mod.modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N // T, C), shift_mlp, scale_mlp).view(B, -1, C)
    x_s = self.ffn(x_m)
    x = _gated_residual(x, gate_mlp, x_s, T, counter).to(x_dtype)
    if return_kv:
        return x, kv_cache
    return x


def _lean_forward_with_kv_cache(self, x, shape, num_cond_latents, kv_cache, num_ref_latents, ref_img_index):
    """LongCat's Attention.forward_with_kv_cache (@58eb336) for the case this runner uses
    (mask_frame_range 0, one person, no block-sparse attention), with the cached and new
    keys/values joined straight into FlashAttention's [B, S, H, D] layout: one copy each
    instead of a join plus the copy FlashAttention's wrapper makes. Same values."""
    B, N, C = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute((2, 0, 3, 1, 4)).unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    N_t, N_h, N_w = shape
    k_cache, v_cache = kv_cache
    assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, B]
    if k_cache.shape[0] == 1:
        k_cache = k_cache.repeat(B, 1, 1, 1)
        v_cache = v_cache.repeat(B, 1, 1, 1)
    q, k = self.rope_3d(q, k, (N_t + num_cond_latents, N_h, N_w), ref_img_index, num_ref_latents,
                        token_offset=k_cache.shape[2])
    k_full = torch.cat([k_cache.transpose(1, 2), k.transpose(1, 2)], dim=1).transpose(1, 2)
    v_full = torch.cat([v_cache.transpose(1, 2), v.transpose(1, 2)], dim=1).transpose(1, 2)
    x = self._process_attn(q, k_full, v_full, shape)
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj(x), None


def _install_fused(dit, dit_module, counter) -> int:
    """Route LongCat's small steps through `_FUSED` while `_FUSE_ON[0]` is set. Everything
    falls back to LongCat's own code per call when the op isn't available or the tensors
    aren't the bf16 layouts the kernels expect. Returns the number of patched modules."""
    F = torch.nn.functional
    patched = 0
    emb = getattr(dit, "x_embedder", None)
    if emb is not None:
        # LongCat's PatchEmbed3D returns a transposed view ([B, N, C] stored as [B, C, N]) and
        # every block's residual add keeps that layout, so the norms on the residual stream
        # can't read whole rows. Store it row by row once per forward; values are unchanged
        # (no matmul reads the residual directly, only through the norms).
        def embed_forward(*args, orig=emb.forward, **kwargs):
            out = orig(*args, **kwargs)
            return out.contiguous() if _FUSE_ON[0] and torch.is_tensor(out) else out

        emb.forward = embed_forward
        patched += 1
    orig_modulate = dit_module.modulate_fp32

    def modulate(norm_func, x, shift, scale):
        fn = _FUSED.get("modulate")
        if (fn is not None and _FUSE_ON[0] and _fusable(x, 4) and x.is_contiguous()
                and getattr(norm_func, "weight", 0) is None and getattr(norm_func, "bias", 0) is None
                and tuple(getattr(norm_func, "normalized_shape", ())) == (x.shape[-1],)
                and shift.dtype == torch.float32 and scale.dtype == torch.float32
                and tuple(shift.shape) == tuple(scale.shape) == (x.shape[0], x.shape[1], 1, x.shape[3])
                and shift.stride(-1) == 1 and scale.stride(-1) == 1):
            counter["modulate"] += 1
            return fn(x, norm_func.eps, scale, shift)
        return orig_modulate(norm_func, x, shift, scale)

    dit_module.modulate_fp32 = modulate

    def qk_fused_active():
        return _FUSE_ON[0] and "qk_rope" in _FUSED

    self_attn_norms = set()
    for name, m in dit.named_modules():
        if not name.startswith("blocks.") or not all(hasattr(m, a) for a in ("rope_3d", "q_norm", "k_norm")):
            continue
        if type(m.q_norm).__name__ != "RMSNorm_FP32" or type(m.k_norm).__name__ != "RMSNorm_FP32":
            continue
        qn, kn, rope = m.q_norm, m.k_norm, m.rope_3d
        self_attn_norms |= {id(qn), id(kn)}

        def q_norm(x, orig=qn.forward):  # folded into the RoPE kernel while fused
            return x if qk_fused_active() else orig(x)

        def k_norm(x, orig=kn.forward):
            return x if qk_fused_active() else orig(x)

        def rope_forward(q, k, grid_size, frame_index=None, num_ref_latents=None, token_offset=0,
                         qn=qn, kn=kn, rope=rope, orig_q=qn.forward, orig_k=kn.forward, orig=rope.forward):
            if not qk_fused_active():
                return orig(q, k, grid_size, frame_index, num_ref_latents, token_offset)
            H, D = q.shape[1], q.shape[-1]
            if not (_fusable(q, 4) and _fusable(k, 4) and k.shape == q.shape and D == rope.head_dim
                    and D >= 2 and D & (D - 1) == 0 and H >= 1 and H & (H - 1) == 0
                    and qn.weight.dtype == kn.weight.dtype == torch.bfloat16):
                return orig(orig_q(q), orig_k(k), grid_size, frame_index, num_ref_latents, token_offset)
            cos, sin = _rope_cos_sin(rope, q, grid_size, frame_index, num_ref_latents, token_offset)
            counter["qk_rope"] += 1
            fn = _FUSED["qk_rope"]
            return fn(q, qn.weight, qn.eps, cos, sin), fn(k, kn.weight, kn.eps, cos, sin)

        qn.forward, kn.forward, rope.forward = q_norm, k_norm, rope_forward
        if hasattr(m, "forward_with_kv_cache"):
            def forward_with_kv_cache(x, shape=None, num_cond_latents=None, kv_cache=None, num_ref_latents=None,
                                      ref_img_index=None, mask_frame_range=None, ref_target_masks=None,
                                      attn=m, orig=m.forward_with_kv_cache):
                if (_FUSE_ON[0] and "kv_layout" in _FUSED and not mask_frame_range and ref_target_masks is None
                        and num_cond_latents and not getattr(attn, "enable_bsa", False)):
                    counter["kv_layout"] += 1
                    return _lean_forward_with_kv_cache(attn, x, shape, num_cond_latents, kv_cache,
                                                       num_ref_latents, ref_img_index)
                return orig(x, shape=shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache,
                            num_ref_latents=num_ref_latents, ref_img_index=ref_img_index,
                            mask_frame_range=mask_frame_range, ref_target_masks=ref_target_masks)

            m.forward_with_kv_cache = forward_with_kv_cache
        patched += 1

    for name, m in dit.named_modules():
        if name.startswith("blocks.") and name.count(".") == 1 and all(
                hasattr(m, a) for a in ("adaLN_modulation", "audio_adaLN_modulation", "attn", "ffn")):
            def block_forward(*args, blk=m, orig=m.forward, **kwargs):
                if _FUSE_ON[0] and "gated_add" in _FUSED:
                    return _fused_block_forward(blk, dit_module, counter, *args, **kwargs)
                return orig(*args, **kwargs)

            m.forward = block_forward
            patched += 1

    for name, m in dit.named_modules():
        if not name.startswith("blocks."):
            continue
        cls = type(m).__name__
        if cls == "RMSNorm_FP32" and id(m) not in self_attn_norms:
            def rms_forward(x, m=m, orig=m.forward):
                fn = _FUSED.get("rms")
                D = x.shape[-1]
                if (fn is not None and _FUSE_ON[0] and _fusable(x, 4) and D == m.weight.shape[0]
                        and D >= 2 and D & (D - 1) == 0 and m.weight.dtype == torch.bfloat16):
                    counter["rms"] += 1
                    return fn(x, m.weight, m.eps)
                return orig(x)

            m.forward = rms_forward
            patched += 1
        elif cls == "LayerNorm_FP32" and m.weight is not None and m.bias is not None:
            def ln_forward(x, m=m, orig=m.forward):
                fn = _FUSED.get("layer_norm")
                if (fn is not None and _FUSE_ON[0] and _fusable(x) and x.is_contiguous()
                        and tuple(m.normalized_shape) == (x.shape[-1],)):
                    counter["layer_norm"] += 1
                    return fn(x, m.eps, m.weight, m.bias)
                return orig(x)

            m.forward = ln_forward
            patched += 1
        elif cls == "FeedForwardSwiGLU":
            def ffn_forward(x, m=m, orig=m.forward):
                fn = _FUSED.get("silu_mul")
                if fn is None or not _FUSE_ON[0] or not _fusable(x):
                    return orig(x)
                a, b = m.w1(x), m.w3(x)
                if _fusable(a) and a.is_contiguous() and b.is_contiguous() and a.shape == b.shape and b.dtype == a.dtype:
                    counter["silu_mul"] += 1
                    return m.w2(fn(a, b))
                return m.w2(F.silu(a) * b)

            m.forward = ffn_forward
            patched += 1
    return patched


def _fused_self_test(dit, orig_modulate, kernels: dict, unavailable: dict) -> str:
    """Each kernel against LongCat's own code on block 0's real layers, in bf16. Passing
    ops go into `_FUSED`; failing ones are recorded in `unavailable` and keep LongCat's path."""
    F = torch.nn.functional
    block = dit.blocks[0]
    attn = block.attn
    dev = next(dit.parameters()).device
    g = torch.Generator(device=dev).manual_seed(0)
    H, D = attn.num_heads, attn.head_dim
    C = H * D

    def rnd(*shape, dtype=torch.bfloat16, std=1.0):
        return (torch.randn(*shape, device=dev, generator=g) * std).to(dtype)

    def modulate_case():
        x = rnd(1, 3, 200, C, std=4.0)
        ada = rnd(1, 3, 1, 6 * C, dtype=torch.float32, std=0.2)
        shift, scale = ada.chunk(6, dim=-1)[:2]  # strided views, as in the block
        norm = block.mod_norm_attn
        yield kernels["modulate"](x, norm.eps, scale, shift), orig_modulate(norm, x, shift, scale)

    def layer_norm_case():
        seen = set()
        for _, m in block.named_modules():
            if type(m).__name__ == "LayerNorm_FP32" and m.weight is not None and m.bias is not None:
                c = m.normalized_shape[0]
                if c in seen:
                    continue
                seen.add(c)
                x = rnd(1, 300, c, std=3.0)
                yield kernels["layer_norm"](x, m.eps, m.weight, m.bias), type(m).forward(m, x)

    def qk_rope_case():
        qn, kn, rope = attn.q_norm, attn.k_norm, attn.rope_3d
        for grid, fi, nr, off in (((4, 8, 16), None, None, 0), ((6, 8, 16), 10, 1, 256)):
            qkv = rnd(1, 512, 3, H, D, std=2.0)
            q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # strided views, as in Attention
            rq, rk = type(rope).forward(rope, type(qn).forward(qn, q), type(kn).forward(kn, k),
                                        grid, fi, nr, token_offset=off)
            cos, sin = _rope_cos_sin(rope, q, grid, fi, nr, off)
            yield kernels["qk_rope"](q, qn.weight, qn.eps, cos, sin), rq
            yield kernels["qk_rope"](k, kn.weight, kn.eps, cos, sin), rk

    def rms_case():
        m = block.cross_attn.q_norm
        x = rnd(1, 512, H, D, std=2.0)
        yield kernels["rms"](x, m.weight, m.eps), type(m).forward(m, x)
        xp = x.permute(0, 2, 1, 3)  # dense permuted view (audio cross-attention's q)
        yield kernels["rms"](xp, m.weight, m.eps), type(m).forward(m, xp)
        kv = rnd(1, 77, 2, H, D).permute(2, 0, 3, 1, 4)[0]  # non-dense view
        yield kernels["rms"](kv, m.weight, m.eps), type(m).forward(m, kv)

    def silu_mul_case():
        a, b = rnd(1, 300, block.ffn.hidden_dim, std=2.0), rnd(1, 300, block.ffn.hidden_dim)
        yield kernels["silu_mul"](a, b), F.silu(a) * b

    def gated_add_case():
        x, xs = rnd(1, 3 * 200, C, std=8.0), rnd(1, 3 * 200, C)
        gate = rnd(1, 3, 1, 6 * C, dtype=torch.float32, std=0.5).chunk(6, dim=-1)[2]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            ref = x + (gate * xs.view(1, -1, 200, C)).view(1, -1, C)
        yield kernels["gated_add"](x, gate, xs, 3), ref.to(x.dtype)

    def kv_layout_case():  # pure layout change: must match LongCat's cached attention
        n_h, n_w, n_t, n_c = 4, 8, 3, 2
        x = rnd(1, n_t * n_h * n_w, C)
        kc, vc = rnd(1, H, n_c * n_h * n_w, D), rnd(1, H, n_c * n_h * n_w, D)
        ref = type(attn).forward_with_kv_cache(attn, x, shape=(n_t, n_h, n_w), num_cond_latents=n_c,
                                               kv_cache=(kc, vc), num_ref_latents=1, ref_img_index=10,
                                               mask_frame_range=0, ref_target_masks=None)[0]
        yield _lean_forward_with_kv_cache(attn, x, (n_t, n_h, n_w), n_c, (kc, vc), 1, 10)[0], ref

    cases = {"modulate": modulate_case, "layer_norm": layer_norm_case, "qk_rope": qk_rope_case,
             "rms": rms_case, "silu_mul": silu_mul_case, "gated_add": gated_add_case,
             "kv_layout": kv_layout_case}
    notes = []
    with torch.no_grad():
        for name, case in cases.items():
            try:
                same, total, worst_rel, worst_cos = 0, 0, 0.0, 1.0
                for out, ref in case():
                    if out.shape != ref.shape or out.dtype != ref.dtype:
                        raise RuntimeError(f"got {tuple(out.shape)} {out.dtype}, want {tuple(ref.shape)} {ref.dtype}")
                    o, r = out.float(), ref.float()
                    worst_rel = max(worst_rel, ((o - r).norm() / r.norm()).item())
                    worst_cos = min(worst_cos, F.cosine_similarity(o.flatten(), r.flatten(), dim=0).item())
                    same += (o == r).sum().item()
                    total += r.numel()
                if not total or not (worst_rel < 0.01 and worst_cos > 0.9999):
                    raise RuntimeError(f"mismatch: rel err {worst_rel:.4f}, cosine {worst_cos:.6f}")
                _FUSED[name] = kernels.get(name, True)  # kv_layout is plain torch, no kernel
                notes.append(f"{name} {same / total:.1%} identical")
            except Exception as exc:  # noqa: BLE001 - this op keeps LongCat's path
                unavailable[f"fused_ops.{name}"] = repr(exc)
                notes.append(f"{name} UNAVAILABLE ({exc!r})")
    return ", ".join(notes)


# Developer profiling (`profile` input): one continuation segment under torch.profiler,
# kernels grouped by name, plus labelled ranges for the code the remaining speed-up ideas
# target (LoRA branches, the fp32 norm/modulation helpers, RoPE).
_KERNEL_CATEGORIES = (  # first match wins; matched against the lower-cased kernel name
    ("attention", ("flash", "fmha", "attn", "sage", "qk_int8")),
    ("matmul", ("gemm", "xmma", "cutlass", "cublas", "nvjet", "matmul", "scaled_mm")),
    ("conv (VAE)", ("conv", "implicit", "winograd", "cudnn")),
    ("copy / concat", ("memcpy", "memset", "copy", "concat", "cat_")),
    ("norm / reduce", ("norm", "reduce", "softmax", "welford")),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "triton")),
)


def _kernel_category(name: str) -> str:
    low = name.lower()
    for category, keys in _KERNEL_CATEGORIES:
        if any(k in low for k in keys):
            return category
    return "other"


@contextmanager
def _profile_labels(dit):
    """Wrap the code worth measuring in record_function ranges; undone on exit."""
    from torch.profiler import record_function

    patched = []

    def label(obj, attr, name):
        fn = getattr(obj, attr)
        had = attr in vars(obj)

        def labelled(*args, **kwargs):
            with record_function("label: " + name):
                return fn(*args, **kwargs)

        setattr(obj, attr, labelled)
        patched.append((obj, attr, fn, had))

    try:
        import longcat_video.modules.avatar.longcat_video_dit_avatar as dit_module

        label(dit_module, "modulate_fp32", "modulate_fp32 (norm + shift/scale)")
    except Exception:  # noqa: BLE001 - labels are best effort
        pass
    for _, module in dit.named_modules():
        cls = type(module).__name__
        if cls in ("RMSNorm_FP32", "LayerNorm_FP32"):
            label(module, "forward", cls)
        elif cls.startswith("RotaryPositionalEmbedding"):
            label(module, "forward", "rope")
        elif hasattr(module, "org_forward"):  # a layer under LongCat's LoRA wrapper
            label(module, "forward", "LoRA-wrapped linears (base + LoRA)")
            label(module, "org_forward", "LoRA-wrapped linears (base only)")
    try:
        yield
    finally:
        for obj, attr, fn, had in reversed(patched):
            if had:
                setattr(obj, attr, fn)
            else:
                delattr(obj, attr)


@contextmanager
def _profiled_segment(dit):
    from torch.profiler import ProfilerActivity, profile

    with _profile_labels(dit):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            try:
                yield prof
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()


def _profile_report(prof, wall: float, top: int = 18) -> list:
    """Lines for the log: GPU time by kernel category, labelled ranges, top kernels."""
    from torch.autograd import DeviceType

    def dev(e, field):
        return float(getattr(e, field, None) or getattr(e, field.replace("device", "cuda"), 0.0) or 0.0)

    kernels, labels = [], {}
    for e in prof.key_averages():
        if e.key.startswith("label: "):
            name = e.key[len("label: "):]
            labels[name] = max(labels.get(name, 0.0), dev(e, "device_time_total"), dev(e, "self_device_time_total"))
        elif e.device_type == DeviceType.CUDA:
            kernels.append((e.key, dev(e, "self_device_time_total"), e.count))
    gpu = sum(t for _, t, _ in kernels) / 1e6
    lines = [f"[profile] segment: {wall:.1f}s wall, {gpu:.1f}s of GPU kernels ({gpu / max(wall, 1e-9):.0%} busy)"]
    by_category = collections.defaultdict(float)
    for name, t, _ in kernels:
        by_category[_kernel_category(name)] += t / 1e6
    for category, t in sorted(by_category.items(), key=lambda kv: -kv[1]):
        lines.append(f"[profile]   {category:14s} {t:6.2f}s  {t / max(gpu, 1e-9):5.1%}")
    if labels:
        lines.append("[profile] labelled code (GPU time inside the range):")
        for name, t in sorted(labels.items(), key=lambda kv: -kv[1]):
            lines.append(f"[profile]   {name:40s} {t / 1e6:6.2f}s")
    lines.append(f"[profile] top {top} kernels (self GPU time, calls, category):")
    for name, t, n in sorted(kernels, key=lambda r: -r[1])[:top]:
        lines.append(f"[profile]   {t / 1e6:6.2f}s {n:6d}x {_kernel_category(name):14s} {name[:80]}")
    return lines


def _gpu_free_note() -> str:
    free, total = torch.cuda.mem_get_info()
    held = (total - free) / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return (f"{free / 1e9:.1f} of {total / 1e9:.1f} GB free; PyTorch reserves {reserved:.1f} GB, "
            f"other CUDA users {max(held - reserved, 0):.1f} GB")


def _offloadable_text_encoder(pipe, device) -> dict:
    """Let the ~11 GB text encoder leave the GPU when a request needs the room (720p). It is
    used once per prompt (later segments reuse LongCat's prompt cache), so: before it actually
    encodes it is moved to `device` if needed; after each `encode_prompt` call, cache hit or
    not, it is moved to the CPU while `state["offload"]` is set. Same weights, same maths.
    Returns the state dict; `state["moves"]` lists (target, seconds) per move."""
    state = {"offload": False, "moves": []}
    inner, outer = pipe._get_t5_prompt_embeds, pipe.encode_prompt
    target_gpu = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)

    def move(target) -> None:
        te = pipe.text_encoder
        if next(te.parameters()).device != target:
            t0 = time.time()
            te.to(target)
            state["moves"].append((str(target), time.time() - t0))

    def get_t5_prompt_embeds(*args, **kwargs):
        move(target_gpu)  # encoding needs it on the GPU
        return inner(*args, **kwargs)

    def encode_prompt(*args, **kwargs):
        try:
            return outer(*args, **kwargs)
        finally:
            if state["offload"]:
                move(torch.device("cpu"))

    pipe._get_t5_prompt_embeds, pipe.encode_prompt = get_t5_prompt_embeds, encode_prompt
    return state


def _set_lora_mix(dit, mix_alpha: float, mix_steps: int) -> bool:
    """Set the per-request DMD/OPSD LoRA mix, leaving the same state as LongCat's
    `enable_lora_mix`. That call goes through `enable_loras`, whose `disable_all_loras` moves
    both LoRA networks to the CPU (`lora.to("cpu")`) before moving them back to the GPU:
    ~2.5 GB each way on every request, inside billed time. When setup already enabled this
    mix (same networks, layers still wrapped, weights still on the GPU), only the multipliers
    change per request. Returns True when that fast path was used."""
    if not 0.0 <= mix_alpha <= 1.0:
        raise ValueError(f"mix_alpha must be in [0, 1], got {mix_alpha}")
    cfg = getattr(dit, "lora_mix_config", None)
    networks = [dit.lora_dict.get(key) for key in ("dmd", "opsd")]
    loras = [lora for net in networks if net is not None for lora in net.loras]
    device = next(dit.parameters()).device
    ready = (
        cfg is not None and cfg.get("dmd_key") == "dmd" and cfg.get("opsd_key") == "opsd"
        and list(getattr(dit, "active_loras", [])) == ["dmd", "opsd"]
        and all(net is not None for net in networks) and bool(loras)
        and all(lora.lora_down.weight.device == device for lora in loras)
        and any(hasattr(m, "org_forward") for m in dit.modules())
    )
    if not ready:
        dit.enable_lora_mix("dmd", "opsd", mix_alpha, mix_steps)
        return False
    dit.lora_mix_config = {"dmd_key": "dmd", "opsd_key": "opsd", "mix_alpha": mix_alpha, "mix_steps": mix_steps}
    dit._lora_mix_active = None
    dit.set_lora_mix_step(0)  # the instance attribute: keeps _MixAwareKVCache's hook
    return True


def _gpu_postprocess_video(orig):
    """Wrap diffusers' VideoProcessor.postprocess_video (np output): run its own denormalize
    on the GPU ("pt" output), then the `* 255` and truncating uint8 cast predict.py applied
    on the CPU, and copy only uint8 frames back. diffusers' np path copies bf16 frames to the
    host, converts them to fp32 there and stacks them; predict.py then scaled and cast on one
    core. Same ops in the same precision, so the frames are bit-identical."""
    def postprocess_video(video, output_type="np"):
        if output_type != "np" or not torch.is_tensor(video):
            return orig(video, output_type=output_type)
        frames = orig(video, output_type="pt")  # [B, F, C, H, W], denormalized, same dtype
        return frames.permute(0, 1, 3, 4, 2).float().mul(255).to(torch.uint8).cpu().numpy()

    return postprocess_video


def _frames_uint8(frames) -> np.ndarray:
    """Decoded frames [F, H, W, 3] as uint8: already uint8 from `_gpu_postprocess_video`, or
    diffusers' float frames in [0, 1] (scaled and truncated exactly as before)."""
    return frames if frames.dtype == np.uint8 else (frames * 255).astype(np.uint8)


def _speed_up_vae(vae) -> str:
    """Exact VAE savings. LongCat's own cache-view switches (its demo turns them on): keep
    views of the 2-frame causal caches instead of cloning them, which is safe because nothing
    in the VAE writes in place. Nearest-exact upsampling straight in bf16: a pure copy, so the
    fp32 round trip around it changes nothing. And no `+ 0.0` pass in RMS norms without a
    bias. (Meituan's fused-padding switch is left off: it may change cuDNN's rounding.)"""
    vae.enable_decoder_cache_views()
    vae.enable_encoder_cache_views()
    up = rms = 0
    for m in vae.modules():
        cls = type(m).__name__
        if cls == "WanUpsample" and getattr(m, "mode", None) == "nearest-exact":
            m.forward = types.MethodType(torch.nn.Upsample.forward, m)
            up += 1
        elif cls == "WanRMS_norm" and isinstance(m.bias, float) and m.bias == 0.0:
            def rms_forward(x, m=m):
                return torch.nn.functional.normalize(x, dim=(1 if m.channel_first else -1)) * m.scale * m.gamma

            m.forward = rms_forward
            rms += 1
    return f"cache views on, {up} upsamples in bf16, {rms} bias-free RMS norms"


class _PhaseTimer:
    """Adds up GPU-synchronised wall time per named phase, so the logs show where a
    prediction's time goes (DiT steps vs. VAE vs. CPU work between them)."""

    def __init__(self, sync=None):
        self.sync = sync or (lambda: None)
        self.totals = collections.defaultdict(float)

    def reset(self) -> None:
        self.totals.clear()

    def snapshot(self) -> dict:
        return dict(self.totals)

    @contextmanager
    def phase(self, name: str):
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.totals[name] += time.perf_counter() - t0

    def wrap(self, obj, attr: str, name: str) -> None:
        fn = getattr(obj, attr)

        @functools.wraps(fn)  # keeps the signature visible to anything that inspects it
        def timed(*args, **kwargs):
            with self.phase(name):
                return fn(*args, **kwargs)

        setattr(obj, attr, timed)

    @staticmethod
    def breakdown(before: dict, after: dict, total: float) -> str:
        diff = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
        parts = [f"{k} {v:.1f}s" for k, v in sorted(diff.items(), key=lambda kv: -kv[1]) if v >= 0.05]
        parts.append(f"other {max(0.0, total - sum(diff.values())):.1f}s")
        return ", ".join(parts)


class _MixAwareKVCache:
    """Reuse the carried-over frames' keys/values while the DMD/OPSD LoRA mix is on.

    In a continuation segment the carried-over frames (and the reference frame) only attend
    to each other, get no text or audio cross-attention and sit at timestep 0, so their
    attention keys/values are the same at every denoising step, as long as the LoRA weights
    are. LongCat can compute them once per segment (`_cache_clean_latents`, its KV cache),
    but the mix swaps LoRA weights at step `mix_steps`, so Meituan's demo turns the cache
    off whenever the mix is on and recomputes those frames on every step. Here the cache is
    rebuilt when the LoRA state changes instead: twice per segment with the default mix.
    """

    def __init__(self, pipe):
        self.pipe, self.dit = pipe, pipe.dit
        self._orig_cache = pipe._cache_clean_latents
        self._orig_set_step = pipe.dit.set_lora_mix_step
        self._args = None
        self._built_for = None
        self.live = {}  # the dict object the denoising loop holds; refilled in place
        self.builds = 0
        pipe._cache_clean_latents = self._cache_clean_latents
        pipe.dit.set_lora_mix_step = self._set_lora_mix_step

    def _state(self):
        return getattr(self.dit, "_lora_mix_active", None)

    def _build(self) -> None:
        self.live.clear()  # drop the old keys/values (~6 GB at 480p) before computing new ones
        args, kwargs = self._args
        self._orig_cache(*args, **kwargs)  # one DiT pass over the carried frames, current LoRAs
        self.live.update(self.pipe.kv_cache_dict)
        self.pipe.kv_cache_dict = self.live
        self._built_for = self._state()
        self.builds += 1

    def _cache_clean_latents(self, *args, **kwargs) -> None:
        # Called once per segment before the loop: build for the LoRA state step 0 uses.
        if getattr(self.dit, "lora_mix_config", None) is not None:
            self._orig_set_step(0)
        self._args = (args, kwargs)
        self._build()

    def _set_lora_mix_step(self, step_index) -> None:
        self._orig_set_step(step_index)
        if self._args is not None and self._state() != self._built_for:
            self._build()

    def release(self) -> None:
        self.live.clear()
        self._args = None
        self._built_for = None
        if getattr(self.pipe, "kv_cache_dict", None) is self.live:
            self.pipe.kv_cache_dict = None


def _dense_bhsd(q, k, v):
    """Dense attention on [B, H, S, D] tensors: FA3 (or its SDPA stand-in) when registered."""
    fa = sys.modules.get("flash_attn_interface")
    if fa is None:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = fa.flash_attn_func(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
        softmax_scale=q.shape[-1] ** -0.5,
    )[0]
    return out.transpose(1, 2)


class _SparseAttention:
    """Routes LongCat's block-sparse attention calls: sparse for the big video attention,
    dense for small or non-tiling shapes. Off unless a request sets a sparsity."""

    def __init__(self, dit, bsa_3d, dense_fn):
        self.dit, self.bsa_3d, self.dense_fn = dit, bsa_3d, dense_fn
        self.available = False
        self.sparsity = 0.0
        self.calls = {"sparse": 0, "dense": 0}

    @staticmethod
    def fits(latent_shape) -> bool:
        return all(n % c == 0 for n, c in zip(latent_shape, SPARSE_CHUNK))

    def attend(self, q, k, v, latent_shape_q, latent_shape_k, sparsity=0.0, **_):
        if (latent_shape_q[0] >= SPARSE_MIN_QUERY_FRAMES
                and self.fits(latent_shape_q) and self.fits(latent_shape_k)):
            self.calls["sparse"] += 1
            chunk = list(SPARSE_CHUNK)
            return self.bsa_3d(q, k, v, latent_shape_q, latent_shape_k, sparsity=float(sparsity),
                               chunk_3d_shape_q=chunk, chunk_3d_shape_k=chunk)
        self.calls["dense"] += 1
        return self.dense_fn(q, k, v)

    def set(self, sparsity: float) -> bool:
        on = self.available and sparsity > 0
        self.sparsity = float(sparsity) if on else 0.0
        for block in self.dit.blocks:
            block.attn.enable_bsa = on
            block.attn.bsa_params = {"sparsity": self.sparsity} if on else None
        self.calls = {"sparse": 0, "dense": 0}
        return on

    def self_test(self) -> str:
        """Sparsity 0 keeps every block, so the kernel must match dense attention, including
        more keys than queries (a continuation chunk). Also checks a sparse call runs."""
        g = torch.Generator(device="cuda").manual_seed(0)
        shape_q, shape_k = (8, 16, 16), (12, 16, 16)
        def rnd(shape):
            return torch.randn(1, 4, shape[0] * shape[1] * shape[2], 128, device="cuda",
                               dtype=torch.bfloat16, generator=g)
        q, k, v = rnd(shape_q), rnd(shape_k), rnd(shape_k)
        chunk = list(SPARSE_CHUNK)
        out = self.bsa_3d(q, k, v, shape_q, shape_k, sparsity=0.0, chunk_3d_shape_q=chunk, chunk_3d_shape_k=chunk)
        err = (out.float() - self.dense_fn(q, k, v).float()).abs().max().item()
        if out.shape != q.shape or not err < 2e-2:
            raise RuntimeError(f"self-test mismatch: shape {tuple(out.shape)}, max err {err:.4f}")
        sparse = self.bsa_3d(q, k, v, shape_q, shape_k, sparsity=0.875, chunk_3d_shape_q=chunk, chunk_3d_shape_k=chunk)
        if not torch.isfinite(sparse).all():
            raise RuntimeError("sparse output is not finite")
        self.available = True
        return f"self-test max err {err:.4f}"


def _plan_chunks(target_frames: int) -> list:
    """Frames to generate per segment so the video just covers `target_frames`.

    Every segment but the last is a full NUM_FRAMES (Meituan's length). The last one is cut
    to what the audio still needs (rounded up to 4k+1 frames, the VAE's step), instead of
    rendering a full 3.2 s and trimming most of it away. The first segment yields all its
    frames; each continuation yields its frames minus the NUM_COND_FRAMES carried over.
    """
    def frames_for(n: int) -> int:
        return min(NUM_FRAMES, (max(n, 1) + 2) // 4 * 4 + 1)

    plan = [frames_for(target_frames)]
    have = plan[0]
    while have < target_frames:
        plan.append(frames_for(NUM_COND_FRAMES + target_frames - have))
        have += plan[-1] - NUM_COND_FRAMES
    return plan


class _BootLog:
    """Setup's stage times and notes. Setup's own output never reaches a prediction's logs,
    so the first prediction on each machine prints them once (`report`)."""

    def __init__(self):
        self.t_mark = time.time()
        self.stages, self.lines = [], []
        self.pending = True

    def log(self, text: str) -> None:
        print(text)
        self.lines.append(text)

    def mark(self, stage: str) -> None:
        """Close the stage that ran since the previous mark."""
        now = time.time()
        self.stages.append((stage, now - self.t_mark))
        self.t_mark = now

    def report(self) -> list:
        self.pending = False
        total = sum(t for _, t in self.stages)
        head = (f"[setup] this machine's boot (printed once): setup {total:.0f}s = "
                + ", ".join(f"{name} {t:.0f}s" for name, t in self.stages))
        return [head] + self.lines


def _start_weight_downloads(log) -> dict:
    """Start both pinned repos downloading at once in background threads; returns
    {"avatar": future, "base": future}, each resolving to the seconds it took (0 when the
    disk already had it). Setup loads the text encoder and VAE as soon as "base" lands."""
    from concurrent.futures import ThreadPoolExecutor
    from importlib import metadata

    from huggingface_hub import snapshot_download

    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(WEIGHTS_ROOT).free / 1e9
    try:
        xet = metadata.version("hf-xet")
    except metadata.PackageNotFoundError:
        xet = "not installed"
    log(f"[setup] {free_gb:.0f} GB free at {WEIGHTS_ROOT}; hf_xet {xet}, "
        f"HF_XET_HIGH_PERFORMANCE={os.environ.get('HF_XET_HIGH_PERFORMANCE')}")

    def fetch(repo, revision, local_dir, patterns):
        marker = local_dir / f".complete-{revision}"
        if marker.exists():
            return 0.0
        t0 = time.time()
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(local_dir),
                          allow_patterns=patterns, max_workers=16)
        marker.touch()
        secs = time.time() - t0
        gb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e9
        log(f"[setup] downloaded {repo}@{revision[:8]}: {gb:.1f} GB in {secs:.0f}s "
            f"({gb * 1000 / max(secs, 1e-6):.0f} MB/s, both repos at once)")
        return secs

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="weights")
    futures = {
        "avatar": pool.submit(fetch, AVATAR_REPO, AVATAR_REVISION, AVATAR_DIR, AVATAR_FILES),
        "base": pool.submit(fetch, BASE_REPO, BASE_REVISION, BASE_DIR, BASE_FILES),
    }
    pool.shutdown(wait=False)
    return futures


def _sdpa(q, k, v, softmax_scale=None, causal=False):
    """PyTorch SDPA on FlashAttention's [B, S, H, D] layout."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale, is_causal=causal
    )
    return out.transpose(1, 2)


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                 softmax_scale=None, causal=False, **_):
    """Variable-length attention on packed [total, H, D] tensors, one SDPA call per sequence."""
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    outs = [
        _sdpa(q[None, cq[i]:cq[i + 1]], k[None, ck[i]:ck[i + 1]], v[None, ck[i]:ck[i + 1]],
              softmax_scale=softmax_scale, causal=causal)[0]
        for i in range(len(cq) - 1)
    ]
    return torch.cat(outs, dim=0)


# FlashAttention 3 for torch 2.6 comes from Hugging Face's kernel hub (prebuilt per torch /
# C++ ABI / CUDA combo). PyTorch's own FA3 wheel needs torch>=2.9, which Replicate has no
# prebuilt base image for, so the image would ship all of torch and take hours to upload.
FA3_KERNEL_REPO = "kernels-community/flash-attn3"
FA3_KERNEL_REVISION = "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"  # keep in sync with cog.yaml
# cog.yaml copies this build into the image, so boots don't depend on the Hugging Face repo
# (its card says the old "model"-type kernel repos are being removed). Startup still
# downloads it if the copy is missing.
FA3_BAKED_DIR = FsPath("/opt/flash-attn3")


def _fa3_variant(torch_version: str, cxx11_abi: bool, cuda_version: str) -> str:
    """The kernel-hub build folder for this torch, e.g. torch26-cxx98-cu124-x86_64-linux."""
    major, minor = torch_version.split("+")[0].split(".")[:2]
    return f"torch{major}{minor}-{'cxx11' if cxx11_abi else 'cxx98'}-cu{cuda_version.replace('.', '')}-x86_64-linux"


def _find_hub_package(root: FsPath, rel: str, lib_prefix: str):
    """The folder with a kernel-hub build's `_ops.py` and the extension it loads, or None.

    Older builds (FA3 for torch 2.6) keep the package in build/<variant>/<name>/; newer
    ones (torch 2.8: FA3, SageAttention) put it straight in build/<variant>/, with <name>/
    only a redirect. Both are accepted.
    """
    base = root / rel
    for ops in sorted(base.glob("_ops.py")) + sorted(base.glob("*/_ops.py")):
        m = re.search(rf"from \. import ({lib_prefix}\w+)", ops.read_text())
        if m and any(ops.parent.glob(f"{m.group(1)}*.so")):
            return ops.parent
    return None


def _import_hub_package(pkg_dir: FsPath, name: str):
    """Import a kernel-hub package folder (its name may contain dashes) as `name`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_fa3():
    """Import FA3: an installed `flash_attn_interface` wheel if present, else the pinned
    kernel-hub build for this torch: the copy in the image, or downloaded (~0.8 GB)."""
    try:
        import flash_attn_interface as fa3

        return fa3, "installed wheel"
    except ImportError:
        pass

    variant = _fa3_variant(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI, torch.version.cuda or "")
    rel = f"build/{variant}"
    pkg, where = _find_hub_package(FA3_BAKED_DIR, rel, "_flash_attn3_"), "image"
    if pkg is None:
        from huggingface_hub import snapshot_download

        root, where = WEIGHTS_ROOT / "flash-attn3", "downloaded"
        pull = dict(repo_id=FA3_KERNEL_REPO, revision=FA3_KERNEL_REVISION, local_dir=str(root))
        snapshot_download(allow_patterns=[f"{rel}/*.py", f"{rel}/*.json"], **pull)
        ops = sorted((root / rel).glob("_ops.py")) + sorted((root / rel).glob("*/_ops.py"))
        if not ops:
            raise RuntimeError(f"no FA3 kernel build for {variant}")
        # Some builds ship several copies of the extension; fetch only the one _ops.py loads.
        lib = re.search(r"from \. import (_flash_attn3_\w+)", ops[0].read_text()).group(1)
        snapshot_download(allow_patterns=[f"{ops[0].parent.relative_to(root).as_posix()}/{lib}*.so"], **pull)
        pkg = _find_hub_package(root, rel, "_flash_attn3_")
        if pkg is None:
            raise RuntimeError(f"FA3 kernel download for {variant} is incomplete")
    module = _import_hub_package(pkg, "flash_attn3_hub")
    return module, f"{FA3_KERNEL_REPO}@{FA3_KERNEL_REVISION[:7]} {variant} ({where})"


# Per-request replacement for LongCat's dense attention calls (SageAttention), or None.
_DENSE_OVERRIDE = [None]


def _make_fa_shim(raw_func, raw_varlen) -> types.ModuleType:
    """A `flash_attn_interface` module for LongCat's FA3 code path. It always returns
    (out, lse) tuples, which LongCat unpacks, and lets a request route the dense
    (non-causal, no extra options) calls to `_DENSE_OVERRIDE[0]`. Varlen calls (the short
    text cross-attention) always use the raw backend."""
    shim = types.ModuleType("flash_attn_interface")

    def flash_attn_func(q, k, v, softmax_scale=None, causal=False, **kwargs):
        alt = _DENSE_OVERRIDE[0]
        if alt is not None and not causal and not kwargs:
            return alt(q, k, v, softmax_scale), None
        res = raw_func(q, k, v, softmax_scale=softmax_scale, causal=causal, **kwargs)
        return tuple(res) if isinstance(res, (tuple, list)) else (res, None)

    def flash_attn_varlen_func(*args, **kwargs):
        res = raw_varlen(*args, **kwargs)
        return tuple(res) if isinstance(res, (tuple, list)) else (res, None)

    shim.flash_attn_func = flash_attn_func
    shim.flash_attn_varlen_func = flash_attn_varlen_func
    return shim


def _install_attention_backend() -> str:
    """Pick the fastest attention that passes a self-test: "fa3", "fa2" or "sdpa".

    FA3 and SDPA both run through LongCat's FA3 code path, which imports
    `flash_attn_interface` at call time, so we register a module under that name.
    LongCat calls `x, *_ = flash_attn_func(...)` and `flash_attn_varlen_func(...)[0]`, i.e.
    it expects (out, lse) tuples. Some FA3 builds return just `out`, where `[0]` would
    silently take the first row, so both functions are wrapped to always return a tuple.
    FA2 is LongCat's own FA2 path (the dit's default). LONGCAT_ATTENTION=fa3|fa2|sdpa forces one.
    """
    want = os.environ.get("LONGCAT_ATTENTION", "auto").lower()
    if want in ("auto", "fa3"):
        try:
            t0 = time.time()
            fa3, source = _load_fa3()

            g = torch.Generator(device="cuda").manual_seed(0)
            q, k, v = (
                torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.bfloat16, generator=g)
                for _ in range(3)
            )
            scale = 128 ** -0.5
            res = fa3.flash_attn_func(q, k, v, softmax_scale=scale)
            out = res[0] if isinstance(res, (tuple, list)) else res
            err = (out.float() - _sdpa(q, k, v, softmax_scale=scale).float()).abs().max().item()
            if out.shape != q.shape or not err < 2e-2:
                raise RuntimeError(f"self-test mismatch: shape {tuple(out.shape)}, max err {err:.4f}")

            # Varlen, packed like LongCat's text cross-attention: 2 queries of 256, keys 100 + 200.
            qp = q[0, :512].contiguous()
            kp, vp = k[0, :300].contiguous(), v[0, :300].contiguous()
            cu_q = torch.tensor([0, 256, 512], device="cuda", dtype=torch.int32)
            cu_k = torch.tensor([0, 100, 300], device="cuda", dtype=torch.int32)
            res = fa3.flash_attn_varlen_func(
                q=qp, k=kp, v=vp, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=256, max_seqlen_k=200, softmax_scale=scale,
            )
            out_v = res[0] if isinstance(res, (tuple, list)) else res
            ref_v = _sdpa_varlen(qp, kp, vp, cu_q, cu_k, softmax_scale=scale)
            err_v = (out_v.float() - ref_v.float()).abs().max().item()
            if out_v.shape != qp.shape or not err_v < 2e-2:
                raise RuntimeError(f"varlen self-test mismatch: shape {tuple(out_v.shape)}, max err {err_v:.4f}")
            err = max(err, err_v)

            sys.modules["flash_attn_interface"] = _make_fa_shim(fa3.flash_attn_func, fa3.flash_attn_varlen_func)
            print(f"[setup] attention: FlashAttention 3 from {source} "
                  f"(self-test max err {err:.4f}, {time.time() - t0:.0f}s)")
            return "fa3"
        except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
            if want == "fa3":
                raise
            print(f"[setup] attention: FlashAttention 3 unavailable ({exc!r})")

    if want in ("auto", "fa2"):
        try:
            import flash_attn  # noqa: F401 - LongCat's FA2 path imports it at call time

            print(f"[setup] attention: FlashAttention 2 ({flash_attn.__version__})")
            return "fa2"
        except Exception as exc:  # noqa: BLE001
            if want == "fa2":
                raise
            print(f"[setup] attention: FlashAttention 2 unavailable ({exc!r})")

    sys.modules["flash_attn_interface"] = _make_fa_shim(
        lambda q, k, v, softmax_scale=None, causal=False, **_: _sdpa(q, k, v, softmax_scale=softmax_scale, causal=causal),
        _sdpa_varlen,
    )
    print("[setup] attention: PyTorch SDPA" + (" (forced)" if want == "sdpa" else ""))
    return "sdpa"


class Runner(BaseRunner):
    def setup(self) -> None:
        t_setup = time.time()
        self.boot = _BootLog()
        downloads = _start_weight_downloads(self.boot.log)  # runs while CUDA and the kernels set up

        from transformers import AutoTokenizer, UMT5EncoderModel
        from audio_separator.separator import Separator

        from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        from longcat_video.context_parallel import context_parallel_util
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.avatar.longcat_video_dit_avatar import (
            LongCatVideoAvatarTransformer3DModel,
        )
        from longcat_video.modules.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

        torch.backends.cuda.matmul.allow_tf32 = True
        # Let cuDNN time its algorithms for the VAE's 3D convolutions once per frame size.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

        # Single-process "torchrun": the pipeline reads its context-parallel state from
        # longcat_video.context_parallel, which needs a process group to exist.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
        context_parallel_util.init_context_parallel(
            context_parallel_size=1, global_rank=0, world_size=1
        )
        cp_split_hw = context_parallel_util.get_optimal_split(1)
        self.device = 0
        # FA3 and the SDPA fallback both go through LongCat's FA3 code path; FA2 is its own.
        self.attention = _install_attention_backend()
        self.boot.mark(f"CUDA init + {self.attention} attention, while downloading")

        downloads["base"].result()
        self.boot.mark("waiting for the text-encoder/VAE repo")
        tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR), subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(BASE_DIR), subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLWan.from_pretrained(
            str(BASE_DIR), subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.boot.mark("load text encoder + VAE")
        downloads["avatar"].result()
        self.boot.mark("waiting for the avatar repo")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(AVATAR_DIR), subfolder="scheduler", torch_dtype=torch.bfloat16
        )

        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            str(AVATAR_DIR),
            subfolder="base_model",
            cp_split_hw=cp_split_hw,
            enable_flashattn3=self.attention in ("fa3", "sdpa"),
            enable_flashattn2=self.attention == "fa2",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        dit.load_lora(
            str(AVATAR_DIR / "loraA" / "lora.safetensors"),
            "dmd",
            multiplier=1.0,
            lora_network_dim=128,
            lora_network_alpha=64,
        )
        dit.load_lora(
            str(AVATAR_DIR / "loraB" / "lora.safetensors"),
            "opsd",
            multiplier=1.0,
            lora_network_dim=128,
            lora_network_alpha=64,
        )
        dit.enable_lora_mix("dmd", "opsd", 1.0, 4)

        whisper_dir = str(AVATAR_DIR / "whisper-large-v3")
        audio_encoder = get_audio_encoder(whisper_dir, MODEL_TYPE)
        if getattr(audio_encoder, "decoder", None) is not None:
            # LongCat only calls audio_encoder.encoder (pipeline get_audio_embedding); the fp32
            # decoder would take ~3.6 GB of GPU memory for nothing.
            audio_encoder.decoder = None
        audio_encoder = audio_encoder.to(self.device)
        audio_feature_extractor = get_audio_feature_extractor(whisper_dir, MODEL_TYPE)

        self.tmp_root = FsPath(tempfile.mkdtemp(prefix="longcat-"))
        separator = Separator(
            output_dir=str(self.tmp_root / "vocals"),
            output_single_stem="vocals",
            model_file_dir=str(AVATAR_DIR / "vocal_separator"),
        )
        if separator.onnx_execution_provider == ["CUDAExecutionProvider"]:
            separator.onnx_execution_provider = [
                (
                    "CUDAExecutionProvider",
                    {"device_id": self.device, "cudnn_conv_algo_search": "HEURISTIC",
                     "arena_extend_strategy": "kSameAsRequested"},
                )
            ]
        separator.load_model("Kim_Vocal_2.onnx")
        self.separator = separator

        self.pipe = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
            audio_encoder=audio_encoder,
            audio_feature_extractor=audio_feature_extractor,
            model_type=MODEL_TYPE,
        )
        self.pipe.to(self.device)
        self.boot.mark("load DiT, LoRAs, audio models; move to GPU")
        vp = self.pipe.video_processor
        vp.postprocess_video = _gpu_postprocess_video(vp.postprocess_video)
        self.boot.log(f"[setup] VAE: {_speed_up_vae(self.pipe.vae)}; frames to uint8 on the GPU")
        self.kv_cache = _MixAwareKVCache(self.pipe)

        import longcat_video.modules.avatar.attention as avatar_attention

        self.unavailable = {}  # option -> why its boot self-test failed (shown when requested)
        self.sparse = _SparseAttention(dit, avatar_attention.flash_attn_bsa_3d, _dense_bhsd)
        avatar_attention.flash_attn_bsa_3d = self.sparse.attend
        try:
            self.boot.log(f"[setup] sparse attention available ({self.sparse.self_test()})")
        except Exception as exc:  # noqa: BLE001 - the input then runs dense
            self.unavailable["sparse_attention"] = repr(exc)
            self.boot.log(f"[setup] sparse attention unavailable ({exc!r}); requests run dense")
        self.fp8_available = False
        self.fp8_calls = [0]
        try:
            fp8_linear = _make_fp8_linear()
            note = _fp8_self_test(fp8_linear)
            layers = _install_fp8(dit, fp8_linear, self.fp8_calls)
            self.fp8_available = True
            self.boot.log(f"[setup] FP8 matmuls available on {layers} layers ({note})")
        except Exception as exc:  # noqa: BLE001 - the input then runs in bf16
            self.unavailable["fp8_matmuls"] = repr(exc)
            self.boot.log(f"[setup] FP8 matmuls unavailable ({exc!r}); requests run in bf16")
        self.merge_available = False
        self.merge_calls = [0]
        branch_factory = dit._create_multi_lora_forward
        try:
            dit._create_multi_lora_forward = _make_merging_factory(branch_factory, self.merge_calls)
            dit.enable_lora_mix("dmd", "opsd", 1.0, 4)  # rebuild the wrapped forwards through it
            note = _merge_self_test(dit)
            self.merge_available = True
            self.boot.log(f"[setup] merged add-ons available ({note})")
        except Exception as exc:  # noqa: BLE001 - requests then use LongCat's branches
            dit._create_multi_lora_forward = branch_factory
            dit.enable_lora_mix("dmd", "opsd", 1.0, 4)
            self.unavailable["merge_lora"] = repr(exc)
            self.boot.log(f"[setup] merged add-ons unavailable ({exc!r}); requests use separate branches")
        self.fused_calls = collections.Counter()
        try:
            import longcat_video.modules.avatar.longcat_video_dit_avatar as dit_module

            note = _fused_self_test(dit, dit_module.modulate_fp32, _make_fused_kernels(), self.unavailable)
            patched = _install_fused(dit, dit_module, self.fused_calls)
            self.boot.log(f"[setup] fused small steps on {patched} modules: {note}")
        except Exception as exc:  # noqa: BLE001 - requests then use LongCat's own steps
            _FUSED.clear()
            self.unavailable["fused_ops"] = repr(exc)
            self.boot.log(f"[setup] fused small steps unavailable ({exc!r}); requests use LongCat's steps")
        self.sage = _SageAttention()
        if self.attention in ("fa3", "sdpa"):  # Sage plugs into LongCat's FA3 code path
            try:
                self.sage.kernel, source = _load_sage()
                self.boot.log(f"[setup] SageAttention available from {source} ({self.sage.self_test()})")
            except Exception as exc:  # noqa: BLE001 - the input then runs on FA3
                self.unavailable["sage_attention"] = repr(exc)
                self.boot.log(f"[setup] SageAttention unavailable ({exc!r}); requests use {self.attention}")
        else:
            self.unavailable["sage_attention"] = f"needs the FA3 code path, attention is {self.attention}"
        # While sparse attention is on, image mode picks sizes from the 128-px bucket table.
        pipe_get_shape = type(self.pipe).get_condition_shape

        def get_condition_shape(condition, resolution, scale_factor_spatial=32):
            if self.sparse.sparsity > 0:
                scale_factor_spatial = max(scale_factor_spatial, 128)
            return pipe_get_shape(self.pipe, condition, resolution, scale_factor_spatial=scale_factor_spatial)

        self.pipe.get_condition_shape = get_condition_shape

        self.boot.mark("kernel self-tests and patches")
        self.timer = _PhaseTimer(torch.cuda.synchronize)
        self.timer.wrap(self.pipe.dit, "forward", "dit")
        self.timer.wrap(self.pipe.vae, "encode", "vae encode")
        self.timer.wrap(self.pipe, "_decode_video_latents", "vae decode")
        self.text_encoder_state = _offloadable_text_encoder(self.pipe, self.device)
        self.timer.wrap(self.pipe, "encode_prompt", "text")
        self._warm_up_audio()
        self.boot.mark("audio warm-up")
        self.boot.log(
            f"[setup] ready in {time.time() - t_setup:.0f}s, "
            f"{torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU "
            f"({_gpu_free_note()})"
        )

    def _warm_up_audio(self) -> None:
        """Run the audio prep once at boot. A machine's first prediction spent ~50 s more
        outside the video segments than later ones (192 s vs 126 s for the same 8 s clip,
        with only ~15 s explained by the segments), so that one-off cost lands here, in
        setup, instead of in a user's paid prediction. 15 s of audio gives the separator
        both full and partial batches."""
        import soundfile as sf
        from longcat_video.audio_process.vocal_separator import separate_vocals

        t0 = time.time()
        try:
            warm = self.tmp_root / "warmup"
            warm.mkdir(parents=True, exist_ok=True)
            sr, secs = 44100, 15.0
            t = np.arange(int(sr * secs), dtype=np.float32) / sr
            tone = 0.3 * np.sin(2 * np.pi * 220.0 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3.0 * t))
            noisy = tone + 0.01 * np.random.default_rng(0).standard_normal(t.shape).astype(np.float32)
            src, wav = warm / "in.wav", warm / "speech.wav"
            sf.write(str(src), np.stack([noisy, noisy], axis=1), sr)
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(wav)],
                check=True,
            )
            speech, rate = separate_vocals(self.separator, str(wav), duration=secs)
            pad = int(secs * rate) - len(speech)
            if pad > 0:
                speech = np.append(speech, np.zeros(pad, dtype=np.float32))
            self.pipe.get_audio_embedding(
                speech, fps=FPS * AUDIO_STRIDE, device=self.device, sample_rate=rate,
                model_type=MODEL_TYPE, use_loudness_norm=False,
            )
            self.boot.log(f"[setup] audio warm-up done in {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001 - warm-up is only an optimisation
            self.boot.log(f"[setup] audio warm-up skipped after {time.time() - t0:.1f}s ({exc!r})")
        finally:
            shutil.rmtree(self.tmp_root / "warmup", ignore_errors=True)
            _torch_gc()

    def run(
        self,
        audio: Path = Input(
            description="Speech for the avatar to say (mp3, wav, m4a, or a video file). The video lasts as long as the audio, up to max_duration.",
        ),
        image: Optional[Path] = Input(
            description="Reference image of the person. Leave empty to make the person from the prompt alone. The image's aspect ratio sets the video's.",
            default=None,
        ),
        prompt: str = Input(
            description="Describe the person, the setting and how they speak.",
            default=DEFAULT_PROMPT,
        ),
        resolution: str = Input(
            description="Output resolution. 720p is much slower, so it is capped at 15 s.",
            choices=["480p", "720p"],
            default="480p",
        ),
        aspect_ratio: str = Input(
            description="Frame shape when no image is given (ignored when there is one).",
            choices=["16:9", "9:16", "1:1"],
            default="16:9",
        ),
        max_duration: float = Input(
            description="Longest video to make, in seconds. Longer audio is cut. Limits: 90 s at 480p, 15 s at 720p (Replicate stops runs at 30 min).",
            default=30.0,
            ge=1.0,
            le=90.0,
        ),
        isolate_voice: bool = Input(
            description="Strip music and background noise from the audio before driving the lips (the output keeps your original audio).",
            default=True,
        ),
        mix_steps: int = Input(
            description="Advanced: leading denoising steps (of 8) that use the DMD LoRA mix. Higher can steady structure; 0 = OPSD only.",
            default=4,
            ge=0,
            le=8,
        ),
        mix_alpha: float = Input(
            description="Advanced: DMD weight during the mixed steps.",
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        seed: Optional[int] = Input(description="Random seed. Leave empty for a random one.", default=None),
        sparse_attention: float = Input(
            description=(
                "Experimental speed-up: the share of attention to skip, e.g. 0.75 or 0.875. 0 = off. "
                "While on, frame sizes snap to 128 px blocks (16:9 at 480p renders at 896x512). "
                "May soften faces or lip-sync."
            ),
            default=0.0,
            ge=0.0,
            le=0.9375,
        ),
        merge_lora: bool = Input(
            description=(
                "Fold the LoRA add-ons into each layer's weights just before it multiplies (one "
                "matrix multiply instead of a separate add-on branch). Faster; the video matches "
                "the separate-branch path up to rounding. Turn off for LongCat's original path."
            ),
            default=True,
        ),
        fused_ops: bool = Input(
            description=(
                "Run the model's small per-layer steps (normalisation, scale/shift, position "
                "encoding, the feed-forward gate) as single fused GPU kernels. Faster; same maths "
                "and precision, so the video differs only in small details. Turn off for "
                "LongCat's original steps."
            ),
            default=True,
        ),
        fp8_matmuls: bool = Input(
            description=(
                "Experimental speed-up: run the video model's big matrix multiplies in FP8 (8-bit) on "
                "the H100. The picture may differ slightly from the default."
            ),
            default=False,
        ),
        profile: bool = Input(
            description=(
                "For developers: profile the second segment and print where its GPU time goes "
                "(makes that segment a little slower)."
            ),
            default=False,
        ),
        sage_attention: bool = Input(
            description=(
                "Experimental speed-up: 8-bit attention (SageAttention2) for the video's attention. "
                "Same frame sizes; the picture may differ slightly from the default."
            ),
            default=False,
        ),
    ) -> Path:
        from diffusers.utils import load_image

        from longcat_video.audio_process.torch_utils import save_video_ffmpeg
        from longcat_video.audio_process.vocal_separator import separate_vocals

        t_start = time.time()
        if self.boot.pending:
            for line in self.boot.report():
                print(line)
        if seed is None:
            seed = secrets.randbelow(2**31)
        sparse_on = self.sparse.set(float(sparse_attention))
        sage_on = bool(sage_attention) and self.sage.available
        _DENSE_OVERRIDE[0] = self.sage.attend if sage_on else None
        fp8_on = bool(fp8_matmuls) and self.fp8_available
        _FP8_ON[0] = fp8_on
        self.fp8_calls[0] = 0
        merge_on = bool(merge_lora) and self.merge_available
        # 720p needs the text encoder's ~11 GB: it moves to CPU memory after encoding.
        self.text_encoder_state["offload"] = resolution == "720p"
        self.text_encoder_state["moves"].clear()
        _MERGE_ON[0] = merge_on
        self.merge_calls[0] = 0
        fuse_on = bool(fused_ops) and bool(_FUSED)
        _FUSE_ON[0] = fuse_on
        self.fused_calls.clear()
        torch.cuda.reset_peak_memory_stats()
        if fp8_matmuls and not fp8_on:
            print(f"[predict] FP8 matmuls are unavailable on this machine "
                  f"({self.unavailable.get('fp8_matmuls', '?')}); running in bf16")
        self.sage.calls = 0
        print(f"[predict] seed={seed} attention={self.attention}"
              + (f" sparse={sparse_attention}" if sparse_on else "")
              + (" sage=on" if sage_on else "")
              + (" fp8=on" if fp8_on else "")
              + (" merge=on" if merge_on else " merge=off")
              + (" fused=on" if fuse_on else " fused=off"))
        missing = {k: v for k, v in self.unavailable.items() if k.startswith("fused_ops")}
        if fused_ops and missing:
            print(f"[predict] fused steps unavailable on this machine, using LongCat's: {missing}")
        if merge_lora and not merge_on:
            print(f"[predict] merged add-ons are unavailable on this machine "
                  f"({self.unavailable.get('merge_lora', '?')}); using separate branches")
        if sage_attention and not sage_on:
            print(f"[predict] SageAttention is unavailable on this machine "
                  f"({self.unavailable.get('sage_attention', '?')}); running without it")
        if sparse_attention > 0 and not sparse_on:
            print(f"[predict] sparse attention is unavailable on this machine "
                  f"({self.unavailable.get('sparse_attention', '?')}); running dense")

        self.timer.reset()
        work = self.tmp_root / f"run-{int(t_start * 1000)}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            # 1. Audio: normalise to 44.1 kHz stereo wav, cut to the allowed length.
            cap = min(float(max_duration), MAX_DURATION[resolution])
            speech_wav = work / "speech.wav"
            with self.timer.phase("audio convert"):
                subprocess.run(
                    [
                        "ffmpeg", "-loglevel", "error", "-y", "-i", str(audio), "-vn",
                        "-ac", "2", "-ar", "44100", "-t", f"{cap:.3f}", str(speech_wav),
                    ],
                    check=True,
                )
            import soundfile as sf

            duration = sf.info(str(speech_wav)).duration
            if duration < 0.5:
                raise ValueError("The audio is shorter than half a second.")
            target_frames = max(1, round(duration * FPS))
            plan = _plan_chunks(target_frames)
            num_segments = len(plan)
            new_frames_total = plan[0] + sum(n - NUM_COND_FRAMES for n in plan[1:])
            generate_seconds = new_frames_total / FPS
            print(
                f"[predict] audio {duration:.2f}s -> {num_segments} segment(s) of "
                f"{'/'.join(map(str, plan))} frames, {resolution}, {'image' if image else 'text'} mode"
            )

            audio_seconds = generate_seconds + 1.0
            if isolate_voice:
                with self.timer.phase("voice isolation"):
                    speech, sr = separate_vocals(self.separator, str(speech_wav), duration=audio_seconds)
                if speech.size == 0 or float(np.abs(speech).max()) < 1e-4:
                    raise ValueError("No voice found in the audio. Try isolate_voice=false.")
            else:
                import librosa

                speech, sr = librosa.load(str(speech_wav), sr=16000, mono=True, duration=audio_seconds)
            pad = math.ceil((audio_seconds - len(speech) / sr) * sr)
            if pad > 0:
                speech = np.append(speech, np.zeros(pad, dtype=np.float32))

            with self.timer.phase("audio embedding"):
                full_audio_emb = self.pipe.get_audio_embedding(
                    speech,
                    fps=FPS * AUDIO_STRIDE,
                    device=self.device,
                    sample_rate=sr,
                    model_type=MODEL_TYPE,
                    use_loudness_norm=False,
                )
            if torch.isnan(full_audio_emb).any():
                raise ValueError("The audio could not be encoded (NaN embedding).")

            window = torch.arange(2 * 2 + 1) - 2

            def audio_window(start_idx: int, num_frames: int) -> torch.Tensor:
                centers = torch.arange(
                    start_idx, start_idx + AUDIO_STRIDE * num_frames, AUDIO_STRIDE
                ).unsqueeze(1) + window.unsqueeze(0)
                centers = torch.clamp(centers, min=0, max=full_audio_emb.shape[0] - 1)
                return full_audio_emb[centers][None, ...].to(self.device)

            # 2. Per-request LoRA mix: only the multipliers (see _set_lora_mix).
            with self.timer.phase("lora mix"):
                _set_lora_mix(self.pipe.dit, float(mix_alpha), int(mix_steps))
            generator = torch.Generator(device=self.device).manual_seed(seed)
            common = dict(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=NUM_STEPS,
                text_guidance_scale=GUIDANCE,
                audio_guidance_scale=GUIDANCE,
                generator=generator,
                output_type="both",
                use_distill=True,
            )

            prep = self.timer.snapshot()
            print(f"[predict] prep: {', '.join(f'{k} {v:.1f}s' for k, v in prep.items())} "
                  f"({time.time() - t_start:.1f}s since start)")

            # 3. First segment.
            audio_start = 0
            t_seg = time.time()
            if image is not None:
                cond_image = load_image(str(image)).convert("RGB")
                output, latent = self.pipe.generate_ai2v(
                    image=cond_image,
                    resolution=resolution,
                    num_frames=plan[0],
                    audio_emb=audio_window(audio_start, plan[0]),
                    **common,
                )
            else:
                sizes = TEXT_MODE_SIZES_SPARSE if sparse_on else TEXT_MODE_SIZES
                height, width = sizes[resolution][aspect_ratio]
                output, latent = self.pipe.generate_at2v(
                    height=height,
                    width=width,
                    num_frames=plan[0],
                    audio_emb=audio_window(audio_start, plan[0]),
                    **common,
                )
            video_array = _frames_uint8(output[0])
            del output
            _torch_gc()
            last_segment_seconds = time.time() - t_seg
            print(f"[predict] segment 1/{num_segments} ({plan[0]} frames) in {last_segment_seconds:.1f}s "
                  f"({self.timer.breakdown(prep, self.timer.snapshot(), last_segment_seconds)})")

            frame_h, frame_w = video_array.shape[1:3]
            total_frames = len(video_array) + sum(n - NUM_COND_FRAMES for n in plan[1:])
            frames = np.empty((total_frames, frame_h, frame_w, 3), dtype=np.uint8)
            frames[: len(video_array)] = video_array
            n_frames = len(video_array)
            # The continuation only reads the last NUM_COND_FRAMES frames (each preprocessed on
            # its own), so hand over just those instead of all 93.
            current_video = [PIL.Image.fromarray(f) for f in video_array[-NUM_COND_FRAMES:]]
            ref_latent = latent[:, :, :1].clone()

            # 4. Audio-driven continuation for the rest of the speech.
            for segment_idx in range(1, num_segments):
                elapsed = time.time() - t_start
                if elapsed + 1.25 * last_segment_seconds > TIME_BUDGET_SECONDS:
                    print(
                        f"[predict] WARNING: stopping after {segment_idx}/{num_segments} segments "
                        f"({n_frames / FPS:.1f}s of video) to stay inside Replicate's 30-minute limit"
                    )
                    break
                t_seg = time.time()
                phases_before = self.timer.snapshot()
                builds_before = self.kv_cache.builds
                audio_start += AUDIO_STRIDE * (plan[segment_idx - 1] - NUM_COND_FRAMES)
                chunk_frames = plan[segment_idx]
                profiling = bool(profile) and segment_idx == 1
                prof_ctx = _profiled_segment(self.pipe.dit) if profiling else contextlib.nullcontext()
                t_prof = time.time()
                with prof_ctx as prof:
                    output, latent = self.pipe.generate_avc(
                        video=current_video,
                        video_latent=latent,
                        height=frame_h,
                        width=frame_w,
                        num_cond_frames=NUM_COND_FRAMES,
                        use_kv_cache=True,  # carried frames computed once per LoRA state
                        offload_kv_cache=False,
                        enhance_hf=False,
                        num_frames=chunk_frames,
                        audio_emb=audio_window(audio_start, chunk_frames),
                        ref_latent=ref_latent,
                        ref_img_index=REF_IMG_INDEX,
                        mask_frame_range=MASK_FRAME_RANGE,
                        **common,
                    )
                if profiling:
                    for line in _profile_report(prof, time.time() - t_prof):
                        print(line)
                cache_builds = self.kv_cache.builds - builds_before
                self.kv_cache.release()
                new_frames = _frames_uint8(output[0])
                del output
                appended = new_frames[NUM_COND_FRAMES:]
                frames[n_frames : n_frames + len(appended)] = appended
                n_frames += len(appended)
                current_video = [PIL.Image.fromarray(f) for f in new_frames[-NUM_COND_FRAMES:]]
                last_segment_seconds = time.time() - t_seg
                print(
                    f"[predict] segment {segment_idx + 1}/{num_segments} ({chunk_frames} frames) "
                    f"in {last_segment_seconds:.1f}s "
                    f"(carried frames computed {cache_builds}x; "
                    f"{self.timer.breakdown(phases_before, self.timer.snapshot(), last_segment_seconds)})"
                )

            # 5. Encode with the (trimmed) original audio. Segments overshoot the speech by up
            # to ~3 s and ffmpeg -shortest does not reliably cut the video, so trim frames here.
            n_out = min(n_frames, max(1, round(duration * FPS)))
            out_dir = FsPath("/tmp/outputs")
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True)
            out_base = out_dir / "longcat-avatar"
            with self.timer.phase("write video"):
                save_video_ffmpeg(
                    torch.from_numpy(frames[:n_out]), str(out_base), str(speech_wav), fps=FPS, quality=10
                )
            print(f"[predict] write video {self.timer.totals['write video']:.1f}s")
            if sage_on:
                print(f"[predict] SageAttention: {self.sage.calls} attention calls")
            if fp8_on:
                print(f"[predict] FP8 matmuls: {self.fp8_calls[0]} calls")
            if merge_on:
                print(f"[predict] merged add-ons: {self.merge_calls[0]} layer calls")
            if fuse_on:
                print("[predict] fused steps: "
                      + ", ".join(f"{k} {v}" for k, v in sorted(self.fused_calls.items())))
            for target, secs in self.text_encoder_state["moves"]:
                print(f"[predict] text encoder moved to {target} in {secs:.1f}s")
            print(f"[predict] peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            if sparse_on:
                print(f"[predict] sparse attention: {self.sparse.calls['sparse']} sparse / "
                      f"{self.sparse.calls['dense']} dense calls, {frame_w}x{frame_h}")
            print(f"[predict] done in {time.time() - t_start:.0f}s")
            return Path(f"{out_base}.mp4")
        finally:
            _DENSE_OVERRIDE[0] = None
            _FP8_ON[0] = False
            _MERGE_ON[0] = False
            _FUSE_ON[0] = False
            self.sparse.set(0.0)
            self.kv_cache.release()
            if hasattr(self.pipe, "_clear_cache"):
                self.pipe._clear_cache()
            _torch_gc()
            shutil.rmtree(work, ignore_errors=True)
