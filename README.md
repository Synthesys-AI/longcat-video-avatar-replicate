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

Meituan's inference code is cloned at a pinned commit, `58eb336`, when the image is built.
The weights, about 63 GB from pinned Hugging Face revisions, download in `setup()` when a
machine first boots. That keeps the image small enough to build on a GitHub runner.

| Input | Notes |
| --- | --- |
| `audio` | The speech (mp3, wav or m4a, or a video file). The output is as long as the audio. |
| `image` | Optional. Without it, the person is made from `prompt`, at `aspect_ratio`. |
| `resolution` | `480p` (up to 120 s) or `720p` (up to 45 s). |
| `max_duration` | Longer audio is cut at this point. |
| `isolate_voice` | Strips music or noise before lip-sync. The output keeps the original audio. |
| `mix_steps`, `mix_alpha` | The LoRA mix. Leave these at the defaults unless faces distort. |
| `seed` | Use it to reproduce a run. |

Replicate stops any prediction after 30 minutes. The predictor stops starting new segments
before it hits that limit and returns the video made so far, and the logs say so.

## Publishing

Pushes go through `.github/workflows/push.yml` (it installs Cog v0.23.1 itself), started from the Actions tab with "Run workflow":
1. It creates the model on Replicate as **public** if it doesn't exist yet, on `gpu-h100`.
2. It builds the image and pushes it.
3. It can run an 8-second test prediction.

It needs two repository secrets. `REPLICATE_API_TOKEN` comes from replicate.com/account/api-tokens and is used for API calls. `REPLICATE_CLI_AUTH_TOKEN` comes from replicate.com/auth/token and is used for `cog login`, which rejects API tokens.

## Licence

The model weights and LongCat-Video code are MIT, © Meituan. This wrapper is MIT too.
