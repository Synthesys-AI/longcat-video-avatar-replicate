"""Replicate (Cog) predictor for LongCat-Video-Avatar 1.6.

A port of Meituan's `run_demo_avatar_single_audio_to_video.py` (LongCat-Video @ 58eb336)
to a long-lived single-GPU predictor:

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
from cog import BasePredictor, Input, Path

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
MAX_DURATION = {"480p": 120.0, "720p": 45.0}

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


class Predictor(BasePredictor):
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
        print(
            f"[setup] ready in {time.time() - t_setup:.0f}s, "
            f"{torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU"
        )

    def predict(
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
            description="Output resolution. 720p is slower and capped at a shorter length.",
            choices=["480p", "720p"],
            default="480p",
        ),
        aspect_ratio: str = Input(
            description="Frame shape when no image is given (ignored when there is one).",
            choices=["16:9", "9:16", "1:1"],
            default="16:9",
        ),
        max_duration: float = Input(
            description="Longest video to make, in seconds. Longer audio is cut. Hard limits: 120 s at 480p, 45 s at 720p.",
            default=60.0,
            ge=1.0,
            le=120.0,
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
        print(f"[predict] seed={seed}")

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
                audio_start += AUDIO_STRIDE * (NUM_FRAMES - NUM_COND_FRAMES)
                output, latent = self.pipe.generate_avc(
                    video=current_video,
                    video_latent=latent,
                    height=frame_h,
                    width=frame_w,
                    num_cond_frames=NUM_COND_FRAMES,
                    use_kv_cache=int(mix_steps) == 0,
                    offload_kv_cache=False,
                    enhance_hf=False,
                    audio_emb=audio_window(audio_start),
                    ref_latent=ref_latent,
                    ref_img_index=REF_IMG_INDEX,
                    mask_frame_range=MASK_FRAME_RANGE,
                    **common,
                )
                new_frames = (output[0] * 255).astype(np.uint8)
                del output
                appended = new_frames[NUM_COND_FRAMES:]
                frames[n_frames : n_frames + len(appended)] = appended
                n_frames += len(appended)
                current_video = [PIL.Image.fromarray(f) for f in new_frames]
                last_segment_seconds = time.time() - t_seg
                print(
                    f"[predict] segment {segment_idx + 1}/{num_segments} in {last_segment_seconds:.1f}s"
                )

            # 5. Encode with the (trimmed) original audio; ffmpeg -shortest trims the tail.
            out_dir = FsPath("/tmp/outputs")
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True)
            out_base = out_dir / "longcat-avatar"
            save_video_ffmpeg(
                torch.from_numpy(frames[:n_frames]), str(out_base), str(speech_wav), fps=FPS, quality=10
            )
            print(f"[predict] done in {time.time() - t_start:.0f}s")
            return Path(f"{out_base}.mp4")
        finally:
            if hasattr(self.pipe, "_clear_cache"):
                self.pipe._clear_cache()
            _torch_gc()
            shutil.rmtree(work, ignore_errors=True)
