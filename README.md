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
- FlashAttention 3 on the H100, from a pinned prebuilt torch 2.6 build on Hugging Face's kernel hub (`kernels-community/flash-attn3`). The build copies it into the image, so boots don't depend on that repo, which Hugging Face says is being retired. Startup downloads it only if that copy is missing. PyTorch's own FA3 wheel needs torch 2.9, and Replicate has no prebuilt base image for that, so the image would carry all of torch and take hours to upload. FA3 is self-tested against PyTorch SDPA at startup. If it fails, FlashAttention 2 takes over, then SDPA. Set `LONGCAT_ATTENTION=fa3|fa2|sdpa` to force one.

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
| `sparse_attention` | Experimental, off by default. The share of attention to skip (e.g. `0.75`), using LongCat's block-sparse kernel. While it is on, frame sizes snap to 128 px blocks (16:9 at 480p renders at 896×512, not 832×480). Faces or lip-sync may soften. |

On one H100 with FlashAttention 3, 480p takes about 49 s for the first 3.7 s of video and 45 s per 3.2 s after that (50 s before carried-frame reuse, 58.6 s with FlashAttention 2). Almost all of that is the DiT's denoising steps (about 41 s of a 44 s segment); VAE decode is about 2 s. An 8 s clip takes about 2.3 min, even on a freshly booted machine, because setup warms up the audio prep. 82 s should take about 19 min. A cold start adds about 4 min while the weights download. Replicate stops any prediction after 30 minutes. The predictor stops starting new segments
before it hits that limit and returns the video made so far, and the logs say so.

The logs time each phase: audio prep, then per segment the DiT steps, VAE encode/decode, text encoding and everything else.

## Publishing

Pushes go through `.github/workflows/push.yml` (it installs Cog v0.23.1 itself), started from the Actions tab with "Run workflow":
1. It creates the model on Replicate as **public** if it doesn't exist yet, on `gpu-h100`.
2. It builds the image and pushes it.
3. It can run an 8-second test prediction.

It needs two repository secrets. `REPLICATE_API_TOKEN` comes from replicate.com/account/api-tokens and is used for API calls. `REPLICATE_CLI_AUTH_TOKEN` comes from replicate.com/auth/token and is used for `cog login`, which rejects API tokens.

## Licence

The model weights and LongCat-Video code are MIT, © Meituan. This wrapper is MIT too.
