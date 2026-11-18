"""CPU tests for three exact speed-ups:
  * `_set_lora_mix`: the per-request LoRA mix without LongCat's enable_lora_mix round trip
    (its disable_all_loras moves both networks to the CPU and back every request);
  * `_gpu_postprocess_video` + `_frames_uint8`: decoded frames to uint8 with diffusers' own
    denormalize, bit-identical to diffusers' np output followed by `(x * 255).astype(uint8)`;
  * `_speed_up_vae`: LongCat's VAE cache views, bf16 nearest-exact upsampling, no `+ 0.0`.
`Dit` copies LongCat @58eb336's enable_lora_mix / set_lora_mix_step / enable_loras /
disable_all_loras; `WanUpsample` / `WanRMS_norm` copy autoencoder_kl_wan.py.
    python -m pytest tests/  or  python tests/test_quick_fixes.py
"""
import ast
import pathlib
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch, "np": np, "types": types}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in {"_set_lora_mix", "_gpu_postprocess_video", "_frames_uint8", "_speed_up_vae"}:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
set_lora_mix, gpu_postprocess_video = _ns["_set_lora_mix"], _ns["_gpu_postprocess_video"]
frames_uint8, speed_up_vae = _ns["_frames_uint8"], _ns["_speed_up_vae"]


# --- LoRA mix ---------------------------------------------------------------------------
class LoRA(nn.Module):
    def __init__(self, name, d=16, r=4):
        super().__init__()
        self.lora_name = name
        self.lora_down, self.lora_up = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
        nn.init.normal_(self.lora_up.weight, std=0.1)
        self.register_buffer("alpha_scale", torch.tensor(0.5))
        self.multiplier, self.use_lora = 1.0, True


class Net:
    def __init__(self, names):
        self.loras = [LoRA(n) for n in names]

    def set_multiplier(self, m):
        for lora in self.loras:
            lora.multiplier = m

    def set_use_lora(self, u):
        for lora in self.loras:
            lora.use_lora = u


class Dit(nn.Module):
    """Two layers with two LoRA networks, wrapped as LongCat wraps them."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a, self.b = nn.Linear(16, 16), nn.Linear(16, 16)
        self.lora_dict = {"dmd": Net(["a", "b"]), "opsd": Net(["a", "b"])}
        self.active_loras, self.lora_mix_config, self._lora_mix_active = [], None, None
        self.to_cpu_moves = 0

    # LongCat @58eb336
    def enable_lora_mix(self, dmd_key, opsd_key, mix_alpha, mix_steps):
        if not 0.0 <= mix_alpha <= 1.0:
            raise ValueError(f"mix_alpha must be in [0, 1], got {mix_alpha}")
        self.enable_loras([dmd_key, opsd_key])
        self.lora_mix_config = {"dmd_key": dmd_key, "opsd_key": opsd_key, "mix_alpha": mix_alpha, "mix_steps": mix_steps}
        self._lora_mix_active = None
        self.set_lora_mix_step(0)

    def set_lora_mix_step(self, step_index):
        if self.lora_mix_config is None:
            return
        is_mixed = step_index < self.lora_mix_config["mix_steps"]
        if is_mixed == self._lora_mix_active:
            return
        dmd, opsd = self.lora_dict[self.lora_mix_config["dmd_key"]], self.lora_dict[self.lora_mix_config["opsd_key"]]
        alpha = self.lora_mix_config["mix_alpha"]
        dm, om = (alpha if is_mixed else 0.0), (1.0 - alpha if is_mixed else 1.0)
        dmd.set_multiplier(dm)
        dmd.set_use_lora(dm != 0.0)
        opsd.set_multiplier(om)
        opsd.set_use_lora(om != 0.0)
        self._lora_mix_active = is_mixed

    def enable_loras(self, keys):
        self.disable_all_loras()
        per_module = {}
        for key in keys:
            for lora in self.lora_dict[key].loras:
                lora.to(next(self.parameters()).device)
                per_module.setdefault(lora.lora_name, []).append(lora)
            self.active_loras.append(key)
        for name, loras in per_module.items():
            module = getattr(self, name)
            if not hasattr(module, "org_forward"):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)

    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x):
            out = module.org_forward(x)
            for lora in loras:
                if lora.use_lora:
                    out = out + lora.lora_up(lora.lora_down(x)) * lora.multiplier * lora.alpha_scale
            return out
        return multi_lora_forward

    def disable_all_loras(self):
        for _, module in self.named_modules():
            if hasattr(module, "org_forward"):
                module.forward = module.org_forward
                delattr(module, "org_forward")
        for net in self.lora_dict.values():
            for lora in net.loras:
                lora.to("cpu")
                self.to_cpu_moves += 1
        self.active_loras.clear()


def mix_state(dit):
    return ([(lora.multiplier, lora.use_lora) for net in dit.lora_dict.values() for lora in net.loras],
            dict(dit.lora_mix_config), dit._lora_mix_active, list(dit.active_loras))


@torch.no_grad()
def test_fast_path_leaves_the_same_state_without_moving_weights():
    fast, full = Dit(), Dit()
    for d in (fast, full):
        d.enable_lora_mix("dmd", "opsd", 1.0, 4)  # what setup does
    moves = fast.to_cpu_moves
    x = torch.randn(3, 16)
    for alpha, steps in [(1.0, 4), (0.3, 2), (0.0, 0), (1.0, 4)]:  # successive requests
        assert set_lora_mix(fast, alpha, steps) is True
        full.enable_lora_mix("dmd", "opsd", alpha, steps)
        assert mix_state(fast) == mix_state(full), (alpha, steps)
        for step in range(8):
            fast.set_lora_mix_step(step)
            full.set_lora_mix_step(step)
            assert torch.equal(fast.a(x), full.a(x)) and torch.equal(fast.b(x), full.b(x))
    assert fast.to_cpu_moves == moves, "the fast path must not move LoRA weights"
    assert full.to_cpu_moves > moves


def test_falls_back_when_the_mix_is_not_set_up():
    d = Dit()
    assert set_lora_mix(d, 1.0, 4) is False  # first call: LongCat's full path
    assert d.lora_mix_config["mix_steps"] == 4 and hasattr(d.a, "org_forward")
    assert set_lora_mix(d, 0.5, 3) is True
    d.disable_all_loras()  # something unwrapped the layers: full path again
    assert set_lora_mix(d, 0.5, 3) is False and hasattr(d.a, "org_forward")


def test_fast_path_goes_through_the_instance_hook_and_checks_alpha():
    d = Dit()
    d.enable_lora_mix("dmd", "opsd", 1.0, 4)
    calls = []
    orig = d.set_lora_mix_step
    d.set_lora_mix_step = lambda step: (calls.append(step), orig(step))  # like _MixAwareKVCache
    set_lora_mix(d, 0.7, 4)
    assert calls == [0]
    try:
        set_lora_mix(d, 1.5, 4)
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass


# --- frames to uint8 ----------------------------------------------------------------------
def test_gpu_postprocess_is_bit_identical_to_diffusers_np_path():
    try:
        from diffusers.video_processor import VideoProcessor
    except ImportError:
        print("skip: diffusers not installed")
        return
    vp = VideoProcessor(vae_scale_factor=8)
    g = torch.Generator().manual_seed(0)
    video = (torch.rand(1, 3, 9, 16, 24, generator=g) * 2.6 - 1.3).clamp(-1, 1).to(torch.bfloat16)
    video[0, :, 0, 0, :5] = torch.tensor([-1.0, 1.0, 0.0, 0.5, -0.5], dtype=torch.bfloat16)[None]
    ref = (vp.postprocess_video(video)[0] * 255).astype(np.uint8)  # diffusers np + predict.py's old cast
    fast = gpu_postprocess_video(vp.postprocess_video)
    out = frames_uint8(fast(video)[0])
    assert out.dtype == np.uint8 and out.shape == ref.shape == (9, 16, 24, 3)
    assert np.array_equal(out, ref)
    assert fast(video, output_type="pt").dtype == torch.bfloat16  # other outputs untouched


def test_frames_uint8_keeps_the_old_float_path():
    x = np.random.default_rng(0).random((2, 4, 4, 3), dtype=np.float32)
    assert np.array_equal(frames_uint8(x), (x * 255).astype(np.uint8))
    u = (x * 255).astype(np.uint8)
    assert frames_uint8(u) is u


# --- VAE -------------------------------------------------------------------------------------
class WanUpsample(nn.Upsample):  # autoencoder_kl_wan.py
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanRMS_norm(nn.Module):  # autoencoder_kl_wan.py
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        shape = (dim, *((1, 1, 1) if not images else (1, 1))) if channel_first else (dim,)
        self.channel_first, self.scale = channel_first, dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class FakeVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
        self.norm = WanRMS_norm(8, images=False)
        self.norm_b = WanRMS_norm(8, images=False, bias=True)  # has a real bias: left alone
        self.switches = []

    def enable_decoder_cache_views(self):
        self.switches.append("decoder")

    def enable_encoder_cache_views(self):
        self.switches.append("encoder")


@torch.no_grad()
def test_vae_tweaks_are_bit_identical():
    vae = FakeVae().to(torch.bfloat16)
    nn.init.normal_(vae.norm.gamma, 1.0, 0.3)
    g = torch.Generator().manual_seed(1)
    img = (torch.randn(2, 8, 5, 7, generator=g) * 3).to(torch.bfloat16)
    vid = (torch.randn(1, 8, 3, 5, 7, generator=g) * 3).to(torch.bfloat16)
    ref = (vae.up(img), vae.norm(vid), vae.norm_b(vid))
    note = speed_up_vae(vae)
    out = (vae.up(img), vae.norm(vid), vae.norm_b(vid))
    assert vae.switches == ["decoder", "encoder"]
    assert "forward" in vars(vae.up) and "forward" in vars(vae.norm) and "forward" not in vars(vae.norm_b)
    for o, r in zip(out, ref):
        assert o.dtype == r.dtype and torch.equal(o, r)
    assert note.startswith("cache views on, 1 upsamples in bf16, 1 bias-free RMS norms")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
