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

import math
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path as FsPath
from typing import Optional

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

FIRST_SEGMENT_SECONDS = NUM_FRAMES / FPS  # 3.72 s
NEXT_SEGMENT_SECONDS = (NUM_FRAMES - NUM_COND_FRAMES) / FPS  # 3.2 s

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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torch_gc() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


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


def _segments_for(duration: float) -> int:
    if duration <= FIRST_SEGMENT_SECONDS:
        return 1
    return 1 + math.ceil((duration - FIRST_SEGMENT_SECONDS) / NEXT_SEGMENT_SECONDS)


def _download_weights() -> None:
    from huggingface_hub import snapshot_download

    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(WEIGHTS_ROOT).free / 1e9
    print(f"[setup] {free_gb:.0f} GB free at {WEIGHTS_ROOT}")
    for repo, revision, local_dir, patterns in (
        (AVATAR_REPO, AVATAR_REVISION, AVATAR_DIR, AVATAR_FILES),
        (BASE_REPO, BASE_REVISION, BASE_DIR, BASE_FILES),
    ):
        marker = local_dir / f".complete-{revision}"
        if marker.exists():
            continue
        t0 = time.time()
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=patterns,
            max_workers=16,
        )
        marker.touch()
        print(f"[setup] downloaded {repo}@{revision[:8]} in {time.time() - t0:.0f}s")


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
FA3_KERNEL_REVISION = "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"


def _fa3_variant(torch_version: str, cxx11_abi: bool, cuda_version: str) -> str:
    """The kernel-hub build folder for this torch, e.g. torch26-cxx98-cu124-x86_64-linux."""
    major, minor = torch_version.split("+")[0].split(".")[:2]
    return f"torch{major}{minor}-{'cxx11' if cxx11_abi else 'cxx98'}-cu{cuda_version.replace('.', '')}-x86_64-linux"


def _load_fa3():
    """Import FA3: an installed `flash_attn_interface` wheel if present, else the pinned
    kernel-hub build for this torch (downloaded next to the weights, ~0.8 GB)."""
    import importlib
    import re

    try:
        import flash_attn_interface as fa3

        return fa3, "installed wheel"
    except ImportError:
        pass

    from huggingface_hub import snapshot_download

    variant = _fa3_variant(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI, torch.version.cuda or "")
    root = WEIGHTS_ROOT / "flash-attn3"
    pkg = f"build/{variant}/flash_attn3"
    pull = dict(repo_id=FA3_KERNEL_REPO, revision=FA3_KERNEL_REVISION, local_dir=str(root))
    snapshot_download(allow_patterns=[f"{pkg}/*.py"], **pull)
    ops = root / pkg / "_ops.py"
    if not ops.exists():
        raise RuntimeError(f"no FA3 kernel build for {variant}")
    # The folder holds several builds of the extension; fetch only the one _ops.py loads.
    lib = re.search(r"from \. import (_flash_attn3_\w+)", ops.read_text()).group(1)
    snapshot_download(allow_patterns=[f"{pkg}/{lib}*.so"], **pull)
    sys.path.insert(0, str(root / "build" / variant))
    return importlib.import_module("flash_attn3"), f"{FA3_KERNEL_REPO}@{FA3_KERNEL_REVISION[:7]} {variant}"


def _install_attention_backend() -> str:
    """Pick the fastest attention that passes a self-test: "fa3", "fa2" or "sdpa".

    FA3 and SDPA both run through LongCat's FA3 code path, which imports
    `flash_attn_interface` at call time, so we register a module under that name.
    LongCat calls `x, *_ = flash_attn_func(...)` and `flash_attn_varlen_func(...)[0]`, i.e.
    it expects (out, lse) tuples. Some FA3 builds return just `out`, where `[0]` would
    silently take the first row, so both functions are wrapped to always return a tuple.
    FA2 is LongCat's own FA2 path (the dit's default). LONGCAT_ATTENTION=fa3|fa2|sdpa forces one.
    """
    import types

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

            raw_func, raw_varlen = fa3.flash_attn_func, fa3.flash_attn_varlen_func
            shim = types.ModuleType("flash_attn_interface")

            def flash_attn_func(*args, **kwargs):
                res = raw_func(*args, **kwargs)
                return tuple(res) if isinstance(res, (tuple, list)) else (res, None)

            def flash_attn_varlen_func(*args, **kwargs):
                res = raw_varlen(*args, **kwargs)
                return tuple(res) if isinstance(res, (tuple, list)) else (res, None)

            shim.flash_attn_func = flash_attn_func
            shim.flash_attn_varlen_func = flash_attn_varlen_func
            sys.modules["flash_attn_interface"] = shim
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

    shim = types.ModuleType("flash_attn_interface")
    shim.flash_attn_func = lambda q, k, v, softmax_scale=None, causal=False, **_: (
        _sdpa(q, k, v, softmax_scale=softmax_scale, causal=causal), None)
    shim.flash_attn_varlen_func = lambda *a, **kw: (_sdpa_varlen(*a, **kw), None)
    sys.modules["flash_attn_interface"] = shim
    print("[setup] attention: PyTorch SDPA" + (" (forced)" if want == "sdpa" else ""))
    return "sdpa"


class Runner(BaseRunner):
    def setup(self) -> None:
        t_setup = time.time()
        _download_weights()

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

        tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR), subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(BASE_DIR), subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLWan.from_pretrained(
            str(BASE_DIR), subfolder="vae", torch_dtype=torch.bfloat16
        )
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
        audio_encoder = get_audio_encoder(whisper_dir, MODEL_TYPE).to(self.device)
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
                    {"device_id": self.device, "cudnn_conv_algo_search": "HEURISTIC"},
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
        self.kv_cache = _MixAwareKVCache(self.pipe)
        print(
            f"[setup] ready in {time.time() - t_setup:.0f}s, "
            f"{torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU"
        )

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
    ) -> Path:
        from diffusers.utils import load_image

        from longcat_video.audio_process.torch_utils import save_video_ffmpeg
        from longcat_video.audio_process.vocal_separator import separate_vocals

        t_start = time.time()
        if seed is None:
            seed = secrets.randbelow(2**31)
        print(f"[predict] seed={seed} attention={self.attention}")

        work = self.tmp_root / f"run-{int(t_start * 1000)}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            # 1. Audio: normalise to 44.1 kHz stereo wav, cut to the allowed length.
            cap = min(float(max_duration), MAX_DURATION[resolution])
            speech_wav = work / "speech.wav"
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
            num_segments = _segments_for(duration)
            generate_seconds = FIRST_SEGMENT_SECONDS + (num_segments - 1) * NEXT_SEGMENT_SECONDS
            print(
                f"[predict] audio {duration:.2f}s -> {num_segments} segment(s), "
                f"{resolution}, {'image' if image else 'text'} mode"
            )

            audio_seconds = generate_seconds + 1.0
            if isolate_voice:
                speech, sr = separate_vocals(self.separator, str(speech_wav), duration=audio_seconds)
                if speech.size == 0 or float(np.abs(speech).max()) < 1e-4:
                    raise ValueError("No voice found in the audio. Try isolate_voice=false.")
            else:
                import librosa

                speech, sr = librosa.load(str(speech_wav), sr=16000, mono=True, duration=audio_seconds)
            pad = math.ceil((audio_seconds - len(speech) / sr) * sr)
            if pad > 0:
                speech = np.append(speech, np.zeros(pad, dtype=np.float32))

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

            def audio_window(start_idx: int) -> torch.Tensor:
                centers = torch.arange(
                    start_idx, start_idx + AUDIO_STRIDE * NUM_FRAMES, AUDIO_STRIDE
                ).unsqueeze(1) + window.unsqueeze(0)
                centers = torch.clamp(centers, min=0, max=full_audio_emb.shape[0] - 1)
                return full_audio_emb[centers][None, ...].to(self.device)

            # 2. Per-request LoRA mix (the networks stay loaded; this only sets multipliers).
            self.pipe.dit.enable_lora_mix("dmd", "opsd", float(mix_alpha), int(mix_steps))
            generator = torch.Generator(device=self.device).manual_seed(seed)
            common = dict(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                text_guidance_scale=GUIDANCE,
                audio_guidance_scale=GUIDANCE,
                generator=generator,
                output_type="both",
                use_distill=True,
            )

            # 3. First segment.
            audio_start = 0
            t_seg = time.time()
            if image is not None:
                cond_image = load_image(str(image)).convert("RGB")
                output, latent = self.pipe.generate_ai2v(
                    image=cond_image,
                    resolution=resolution,
                    audio_emb=audio_window(audio_start),
                    **common,
                )
            else:
                height, width = TEXT_MODE_SIZES[resolution][aspect_ratio]
                output, latent = self.pipe.generate_at2v(
                    height=height,
                    width=width,
                    audio_emb=audio_window(audio_start),
                    **common,
                )
            video_array = (output[0] * 255).astype(np.uint8)
            del output
            _torch_gc()
            last_segment_seconds = time.time() - t_seg
            print(f"[predict] segment 1/{num_segments} in {last_segment_seconds:.1f}s")

            frame_h, frame_w = video_array.shape[1:3]
            total_frames = len(video_array) + (num_segments - 1) * (NUM_FRAMES - NUM_COND_FRAMES)
            frames = np.empty((total_frames, frame_h, frame_w, 3), dtype=np.uint8)
            frames[: len(video_array)] = video_array
            n_frames = len(video_array)
            current_video = [PIL.Image.fromarray(f) for f in video_array]
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
                builds_before = self.kv_cache.builds
                audio_start += AUDIO_STRIDE * (NUM_FRAMES - NUM_COND_FRAMES)
                output, latent = self.pipe.generate_avc(
                    video=current_video,
                    video_latent=latent,
                    height=frame_h,
                    width=frame_w,
                    num_cond_frames=NUM_COND_FRAMES,
                    use_kv_cache=True,  # carried frames computed once per LoRA state
                    offload_kv_cache=False,
                    enhance_hf=False,
                    audio_emb=audio_window(audio_start),
                    ref_latent=ref_latent,
                    ref_img_index=REF_IMG_INDEX,
                    mask_frame_range=MASK_FRAME_RANGE,
                    **common,
                )
                cache_builds = self.kv_cache.builds - builds_before
                self.kv_cache.release()
                new_frames = (output[0] * 255).astype(np.uint8)
                del output
                appended = new_frames[NUM_COND_FRAMES:]
                frames[n_frames : n_frames + len(appended)] = appended
                n_frames += len(appended)
                current_video = [PIL.Image.fromarray(f) for f in new_frames]
                last_segment_seconds = time.time() - t_seg
                print(
                    f"[predict] segment {segment_idx + 1}/{num_segments} in {last_segment_seconds:.1f}s "
                    f"(carried frames computed {cache_builds}x)"
                )

            # 5. Encode with the (trimmed) original audio. Segments overshoot the speech by up
            # to ~3 s and ffmpeg -shortest does not reliably cut the video, so trim frames here.
            n_out = min(n_frames, max(1, round(duration * FPS)))
            out_dir = FsPath("/tmp/outputs")
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True)
            out_base = out_dir / "longcat-avatar"
            save_video_ffmpeg(
                torch.from_numpy(frames[:n_out]), str(out_base), str(speech_wav), fps=FPS, quality=10
            )
            print(f"[predict] done in {time.time() - t_start:.0f}s")
            return Path(f"{out_base}.mp4")
        finally:
            self.kv_cache.release()
            if hasattr(self.pipe, "_clear_cache"):
                self.pipe._clear_cache()
            _torch_gc()
            shutil.rmtree(work, ignore_errors=True)
