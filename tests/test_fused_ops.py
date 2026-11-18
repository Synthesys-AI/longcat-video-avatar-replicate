"""CPU tests for the fused small steps (`fused_ops` input).

The Triton kernels need a GPU and are checked against LongCat's own code at boot
(`_fused_self_test`). Here `MIRRORS` re-state each kernel's maths line by line in torch
(same fp32 steps, same bf16 rounding points, same (2j, 2j+1) RoPE pairs, cos/sin row =
token index) and stand in for the kernels, so these tests check:
  * the kernel maths matches LongCat's (through the same boot self-test);
  * the routing: which modules are patched, the q/k RMSNorm folded into the RoPE call,
    cos/sin rows for token offsets and reference frames, fallbacks, the off switch;
  * a broken kernel is caught by the self-test and left out.
The LongCat classes below copy @58eb336 (blocks.py, avatar/rope_3d.py), with einops'
rotate_half written as a view.
    python -m pytest tests/  or  python tests/test_fused_ops.py
"""
import ast
import collections
import pathlib
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch, "collections": collections}
_names = {"_FUSE_ON", "_FUSED", "_FUSE_DEVICE"}
_funcs = {"_fusable", "_rope_cos_sin", "_install_fused", "_fused_self_test"}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _names for t in node.targets):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
    if isinstance(node, ast.FunctionDef) and node.name in _funcs:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
FUSE_ON, FUSED, FUSE_DEVICE = _ns["_FUSE_ON"], _ns["_FUSED"], _ns["_FUSE_DEVICE"]
rope_cos_sin, install_fused, fused_self_test = _ns["_rope_cos_sin"], _ns["_install_fused"], _ns["_fused_self_test"]
FUSE_DEVICE[0] = "cpu"
BF = torch.bfloat16


# --- LongCat @58eb336 -------------------------------------------------------------------
class RMSNorm_FP32(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm_FP32(nn.LayerNorm):
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inputs):
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape,
                            None if self.weight is None else self.weight.float(),
                            None if self.bias is None else self.bias.float(), self.eps).to(origin_dtype)


def modulate_fp32(norm_func, x, shift, scale):
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    return x.to(dtype)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotate_half(x):  # einops "... (d r) -> ... d r", r=2; stack((-x2, x1)); back
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryPositionalEmbedding(nn.Module):
    _gpu_cos_sin_cache = {}

    def __init__(self, head_dim):
        super().__init__()
        self.head_dim, self.cp_split_hw, self.freqs_dict = head_dim, [1, 1], {}

    def register_grid_size(self, grid_size, key_name, frame_index=None, num_ref_latents=None):
        # (a simpler table than LongCat's 3D one; only the cache plumbing matters here)
        n = grid_size[0] * grid_size[1] * grid_size[2]
        pos = torch.arange(n, dtype=torch.float32) + (0.0 if frame_index is None else 100.0 * frame_index)
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.freqs_dict[key_name] = (pos[:, None] * inv[None]).repeat_interleave(2, dim=-1)

    def forward(self, q, k, grid_size, frame_index=None, num_ref_latents=None, token_offset=0):
        key_name = '.'.join([str(i) for i in grid_size]) + f"-{str(frame_index)}-{str(num_ref_latents)}"
        if key_name not in self.freqs_dict:
            self.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)
        q_, k_ = q.float(), k.float()
        cache_key = (self.head_dim, tuple(self.cp_split_hw or ()), key_name, q.device.type, q.device.index)
        cached = self._gpu_cos_sin_cache.get(cache_key)
        if cached is None:
            freqs_cis = self.freqs_dict[key_name].to(q.device).float().to(q.device)
            cos, sin = freqs_cis.cos(), freqs_cis.sin()
            cached = (cos[None, None], sin[None, None])
            self._gpu_cos_sin_cache[cache_key] = cached
        cos, sin = cached
        token_count = q.shape[-2]
        cos = cos[:, :, token_offset:token_offset + token_count]
        sin = sin[:, :, token_offset:token_offset + token_count]
        assert cos.shape[-2] == token_count
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)
        return q_.type_as(q), k_.type_as(k)


# --- a one-block DiT with LongCat's module layout ----------------------------------------
H, D = 4, 16
C = H * D


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim = H, D
        self.qkv = nn.Linear(C, 3 * C)
        self.q_norm, self.k_norm = RMSNorm_FP32(D, 1e-6), RMSNorm_FP32(D, 1e-6)
        self.rope_3d = RotaryPositionalEmbedding(D)

    def forward(self, x, shape, frame_index=None, num_ref=None, token_offset=0):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        return self.rope_3d(q, k, shape, frame_index, num_ref, token_offset=token_offset)


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = RMSNorm_FP32(D, 1e-6)

    def forward(self, x):
        return self.q_norm(x.view(1, -1, H, D))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.cross_attn = Attention(), CrossAttention()
        self.mod_norm_attn = LayerNorm_FP32(C, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(C, eps=1e-6, elementwise_affine=True)
        self.pre_audio_crs_attn_norm = LayerNorm_FP32(24, eps=1e-6, elementwise_affine=True)
        self.ffn = FeedForwardSwiGLU(C, 96)


class Dit(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def build():
    torch.manual_seed(0)
    dit = Dit()
    for p in dit.parameters():  # distinct, non-trivial weights (q_norm != k_norm, affine LN)
        nn.init.normal_(p, mean=1.0 if p.dim() == 1 else 0.0, std=0.3 if p.dim() == 1 else 0.1)
    dit.to(BF)
    RotaryPositionalEmbedding._gpu_cos_sin_cache.clear()
    return dit, types.SimpleNamespace(modulate_fp32=modulate_fp32)


# --- the kernels' maths, line by line ------------------------------------------------------
def _ln(x, a, c, eps):
    x32 = x.float()
    mean = x32.sum(-1, keepdim=True) / x.shape[-1]
    d = x32 - mean
    rstd = torch.rsqrt((d * d).sum(-1, keepdim=True) / x.shape[-1] + eps)
    return (d * rstd * a + c).to(x.dtype)


def _rms(x, w, eps, cos, sin, out):
    x32 = x.float()
    rstd = torch.rsqrt((x32 * x32).sum(-1, keepdim=True) / x.shape[-1] + eps)
    n = (x32 * rstd).to(BF).float()
    n = (n * w.float()).to(BF).float()
    if cos is not None:
        ne, no = n[..., 0::2], n[..., 1::2]
        ce, co, se, so = cos[:, 0::2], cos[:, 1::2], sin[:, 0::2], sin[:, 1::2]
        n = torch.stack((ne * ce - no * se, no * co + ne * so), dim=-1).flatten(-2)
    out.copy_(n.to(out.dtype))
    return out


def _qk_rope(x, w, eps, cos, sin):
    B, Hh, N, Dd = x.shape
    return _rms(x, w, eps, cos, sin, torch.empty((B, N, Hh, Dd), dtype=x.dtype).permute(0, 2, 1, 3))


MIRRORS = {
    "modulate": lambda x, eps, scale, shift: _ln(x, scale.float() + 1.0, shift.float(), eps),
    "layer_norm": lambda x, eps, w, b: _ln(x, w.float(), b.float(), eps),
    "qk_rope": _qk_rope,
    "rms": lambda x, w, eps: _rms(x, w, eps, None, None, torch.empty_like(x)),
    "silu_mul": lambda a, b: ((a.float() / (1.0 + torch.exp(-a.float()))).to(BF).float() * b.float()).to(a.dtype),
}


def run_all(dit, mod, x, cases):
    blk = dit.blocks[0]
    outs = []
    for shape, fi, nr, off in cases:
        outs += list(blk.attn(x, shape, fi, nr, token_offset=off))
    x4 = x.view(1, 2, -1, C)
    ada = torch.randn(1, 2, 1, 6 * C, generator=torch.Generator().manual_seed(1))
    shift, scale = ada.chunk(6, dim=-1)[:2]
    outs += [mod.modulate_fp32(blk.mod_norm_attn, x4, shift, scale), blk.pre_crs_attn_norm(x),
             blk.cross_attn(x), blk.ffn(x)]
    return outs


CASES = [((2, 4, 8), None, None, 0), ((3, 4, 8), 10, 1, 32)]  # plain, then a cached-frames offset


def fresh(with_mirrors=True):
    FUSED.clear()
    FUSE_ON[0] = False
    dit, mod = build()
    counter = collections.Counter()
    install_fused(dit, mod, counter)
    if with_mirrors:
        FUSED.update(MIRRORS)
    return dit, mod, counter


@torch.no_grad()
def test_mirrors_pass_the_boot_self_test():
    FUSED.clear()
    dit, mod = build()
    unavailable = {}
    note = fused_self_test(dit, mod.modulate_fp32, MIRRORS, unavailable)
    assert set(FUSED) == set(MIRRORS) and not unavailable, (note, unavailable)
    assert "UNAVAILABLE" not in note


@torch.no_grad()
def test_a_broken_kernel_is_left_out():
    FUSED.clear()
    dit, mod = build()
    broken = dict(MIRRORS, qk_rope=lambda x, w, eps, cos, sin: _qk_rope(x, w, eps, cos, -sin))  # wrong rotation
    unavailable = {}
    note = fused_self_test(dit, mod.modulate_fp32, broken, unavailable)
    assert "qk_rope" not in FUSED and set(FUSED) == set(MIRRORS) - {"qk_rope"}
    assert "fused_ops.qk_rope" in unavailable and "qk_rope UNAVAILABLE" in note


@torch.no_grad()
def test_fused_route_matches_longcat_end_to_end():
    dit, mod, counter = fresh()
    x = torch.randn(1, 64, C).to(BF)
    ref = run_all(dit, mod, x, CASES)
    FUSE_ON[0] = True
    try:
        out = run_all(dit, mod, x, CASES)
    finally:
        FUSE_ON[0] = False
    assert counter == {"qk_rope": 2, "modulate": 1, "layer_norm": 1, "rms": 1, "silu_mul": 1}, counter
    for i, (o, r) in enumerate(zip(out, ref)):
        assert o.shape == r.shape and o.dtype == r.dtype, i
        torch.testing.assert_close(o.float(), r.float(), rtol=0.02, atol=0.02, msg=f"output {i}")
        assert (o == r).float().mean() > 0.9, f"output {i}: {(o == r).float().mean():.3f} identical"
    q = out[0]
    assert q.permute(0, 2, 1, 3).is_contiguous(), "q should come out in FlashAttention's [B, S, H, D] layout"


@torch.no_grad()
def test_rope_rows_follow_offset_and_reference_frame():
    dit, _, _ = fresh(with_mirrors=False)
    rope = dit.blocks[0].attn.rope_3d
    q = torch.zeros(1, H, 64, D)
    cos_a, _ = rope_cos_sin(rope, q, (3, 4, 8), 10, 1, 32)
    cos_b, _ = rope_cos_sin(rope, q, (3, 4, 8), None, None, 0)
    full = rope.freqs_dict["3.4.8-10-1"].cos()
    assert torch.equal(cos_a, full[32:96]) and not torch.equal(cos_a, cos_b)
    try:
        rope_cos_sin(rope, q, (3, 4, 8), None, None, 64)  # only 32 rows left
        raise AssertionError("expected a row-count error")
    except ValueError:
        pass


@torch.no_grad()
def test_off_switch_and_missing_ops_use_longcat():
    dit, mod, counter = fresh()
    x = torch.randn(1, 64, C).to(BF)
    ref = run_all(dit, mod, x, CASES)
    assert sum(counter.values()) == 0
    FUSED.pop("qk_rope")  # q/k norms must then run normally, not be skipped
    FUSE_ON[0] = True
    try:
        out = run_all(dit, mod, x, CASES)
    finally:
        FUSE_ON[0] = False
    assert "qk_rope" not in counter
    for o, r in zip(out[:4], ref[:4]):
        assert torch.equal(o, r)


@torch.no_grad()
def test_unsupported_tensors_fall_back_exactly():
    dit, mod, counter = fresh()
    dit.float()  # fp32 model: no kernel takes these
    x = torch.randn(1, 64, C)
    ref = run_all(dit, mod, x, CASES)
    FUSE_ON[0] = True
    try:
        out = run_all(dit, mod, x, CASES)
    finally:
        FUSE_ON[0] = False
    assert sum(counter.values()) == 0, counter
    for o, r in zip(out, ref):
        assert torch.equal(o, r)


def test_only_the_right_modules_are_patched():
    FUSED.clear()
    dit, mod = build()
    assert install_fused(dit, mod, collections.Counter()) == 2 * (1 + 1 + 2 + 1)  # attn, cross q_norm, 2 affine LNs, ffn
    blk = dit.blocks[0]
    assert "forward" not in vars(blk.mod_norm_attn), "no-affine LN is reached through modulate"
    assert mod.modulate_fp32 is not modulate_fp32
    for m in (blk.attn.q_norm, blk.attn.k_norm, blk.attn.rope_3d, blk.cross_attn.q_norm, blk.pre_crs_attn_norm, blk.ffn):
        assert "forward" in vars(m)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
