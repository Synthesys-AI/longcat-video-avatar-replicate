# LongCat-Video-Avatar 1.6 on Replicate

A [Cog](https://github.com/replicate/cog) package that serves Meituan's
[LongCat-Video-Avatar-1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6)
(MIT) as a public Replicate model. It makes a talking-avatar video from speech, plus either
a reference image or a text prompt. Speech longer than about 3.7 s is covered by the model's
audio-driven video continuation.

## What runs

This follows the model card's 1.6 recipe on one GPU:
- bf16 DiT.
- Distilled 8-step sampling.
- DMD/OPSD LoRA mix, `mix_steps=4` and `mix_alpha=1.0` by default.
- Whisper-large-v3 audio features and Kim_Vocal_2 voice isolation.
- 25 fps output, muxed with your original audio.
- The 13 frames each segment carries over are computed once per LoRA setting (twice per segment), not on every step. That is LongCat's own KV cache, which Meituan's demo only uses with the LoRA mix off, because the mix changes the LoRA weights partway through. Here the cache is rebuilt when they change. Those frames never see the new frames, so the video is the same.
- FlashAttention 3 on the H100, from a pinned prebuilt build on Hugging Face's kernel hub (`kernels-community/flash-attn3`). The build copies it into the image, so boots don't depend on that repo, which Hugging Face says is being retired. Startup downloads it only if that copy is missing. torch is 2.8 (Meituan pins 2.6) because the hub's SageAttention builds start at 2.8, and Replicate has a prebuilt Cog base image for it; without one, the image carries all of torch and takes hours to upload. FA3 is self-tested against PyTorch SDPA at startup. If it fails, FlashAttention 2 takes over, then SDPA. Set `LONGCAT_ATTENTION=fa3|fa2|sdpa` to force one.

Meituan's inference code is cloned at a pinned commit, `58eb336`, when the image is built.
The weights, about 63 GB from pinned Hugging Face revisions, download in `setup()` when a
machine first boots. That keeps the image small enough to build on a GitHub runner.

| Input | Notes |
| --- | --- |
| `audio` | The speech (mp3, wav or m4a, or a video file). The output is as long as the audio. |
| `image` | Optional. Without it, the person is made from `prompt`, at `aspect_ratio`. |
| `resolution` | `480p` (up to 90 s) or `720p` (up to 15 s). |
| `max_duration` | Longer audio is cut at this point. |
| `isolate_voice` | Strips music or noise before lip-sync. The output keeps the original audio. |
| `mix_steps`, `mix_alpha` | The LoRA mix. Leave these at the defaults unless faces distort. |
| `seed` | Use it to reproduce a run. |
| `merge_lora` | On by default. Folds the LoRA add-ons into each layer's weights just before it multiplies (one matrix multiply instead of a separate add-on branch, with the merged weight built in a temporary buffer, so the stored weights never change). Measured on an 8 s clip: about 10–13% faster (99.7 s, vs 114.9 s with `false` on the same version and 110.7 s on the previous version; DiT time 11–12% lower in every chunk). Same scene, pose and lip timing, but small details such as blinks, hair and smoke differ (SSIM 0.90 vs `false`): the per-layer rounding differs by about 0.3%, and the 8 denoising steps amplify that into a slightly different take. Set it to `false` for LongCat's original path. |
| `fused_ops` | On by default. Runs the model's small per-layer steps (the fp32 LayerNorm + scale/shift, the LayerNorms before cross-attention, q/k RMSNorm + 3D RoPE, the feed-forward SiLU gate and the gated residual adds) as single Triton kernels, and joins cached keys/values straight into FlashAttention's layout that read and write each tensor once, with the same fp32 maths and bf16 rounding points as LongCat. Each kernel is checked against LongCat's code at boot and falls back to it if it disagrees. Measured on an 8 s clip: 75.0 s vs 98.2 s with `false` (DiT time 26–27% lower per chunk); video SSIM 0.987 / 41.5 dB vs `false`, same mouth shapes. Set it to `false` for LongCat's original steps. |
| `sage_attention` | Experimental, off by default. 8-bit attention (SageAttention2, from the same kernel hub) for the video's attention. Frame sizes stay the same; the picture may differ slightly from the default. Measured on an 8 s clip: about 2.5% faster, video nearly identical (SSIM 0.97). |
| `fp8_matmuls` | Experimental, off by default. FP8 for the DiT's big matrix multiplies on the H100. Measured on an 8 s clip: about 5% faster (6.6% with `sage_attention`), small visual differences (SSIM 0.90), same lip timing. |
| `sparse_attention` | Experimental, off by default. The share of attention to skip (e.g. `0.75`), using LongCat's block-sparse kernel. While it is on, frame sizes snap to 128 px blocks (16:9 at 480p renders at 896×512, not 832×480). Faces or lip-sync may soften. |

On one H100 with FlashAttention 3, merged add-ons and fused steps, 480p takes about 33 s for the first 3.7 s of video and 29 s per 3.2 s after that (39 s with merged add-ons only, 45 s before the merge, 50 s before carried-frame reuse, 58.6 s with FlashAttention 2). Almost all of that is the DiT's denoising steps (about 41 s of a 44 s segment); VAE decode is about 2 s. An 8 s clip takes about 71-73 s, even on a freshly booted machine, because setup warms up the audio prep. 82 s should take about 12.5 min. Two runs of the same version with the same seed are not bit-identical (SSIM 0.987 / 41.6 dB between two runs on the 8 s clip, already from the first frame), most likely because cuDNN picks convolution algorithms by timing them on each machine (`cudnn.benchmark`). A cold start adds about 4 min while the weights download. Replicate stops any prediction after 30 minutes. The predictor stops starting new segments
before it hits that limit and returns the video made so far, and the logs say so.

The last segment only generates the frames the audio still needs (rounded up to 4k+1), rather than a full 3.2 s that would mostly be trimmed; a clip under 3.7 s is a single short segment. The logs time each phase: audio prep, then per segment the DiT steps, VAE encode/decode, text encoding and everything else. Three exact savings outside the model: each request only sets the LoRA mix's multipliers (LongCat's `enable_lora_mix` moves both LoRA networks to the CPU and back every time), decoded frames become uint8 on the GPU, and the VAE uses LongCat's own cache views, bf16 nearest upsampling and no `+ 0.0` bias pass.

## Trying your own files

`scripts/run_local.sh <image> <audio> [prompt] [extra input JSON]` uploads both files to your Replicate account (Files API, private), runs the latest version on the whole recording and saves `<audio>-longcat.mp4` next to the audio (then `-2`, `-3`, … so earlier runs are never overwritten). It reads `REPLICATE_API_TOKEN`, or asks for it without echoing it. `DRY_RUN=1` shows the request instead.

## Publishing

Pushes go through `.github/workflows/push.yml` (it installs Cog v0.23.1 itself), started from the Actions tab with "Run workflow":
1. It creates the model on Replicate as **public** if it doesn't exist yet, on `gpu-h100`.
2. It builds the image and pushes it.
3. It can run an 8-second test prediction.

It needs two repository secrets. `REPLICATE_API_TOKEN` comes from replicate.com/account/api-tokens and is used for API calls. `REPLICATE_CLI_AUTH_TOKEN` comes from replicate.com/auth/token and is used for `cog login`, which rejects API tokens.

## Licence

The model weights and LongCat-Video code are MIT, © Meituan. This wrapper is MIT too.
