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
import functools
import pathlib
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch, "collections": collections, "functools": functools}
_names = {"_FUSE_ON", "_FUSED", "_FUSE_DEVICE"}
_funcs = {"_fusable", "_rope_cos_sin", "_install_fused", "_fused_self_test", "_gated_residual",
          "_fused_block_forward", "_lean_forward_with_kv_cache"}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _names for t in node.targets):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
    if isinstance(node, ast.FunctionDef) and node.name in _funcs:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
FUSE_ON, FUSED, FUSE_DEVICE = _ns["_FUSE_ON"], _ns["_FUSED"], _ns["_FUSE_DEVICE"]
rope_cos_sin, install_fused, fused_self_test = _ns["_rope_cos_sin"], _ns["_install_fused"], _ns["_fused_self_test"]
lean_forward_with_kv_cache = _ns["_lean_forward_with_kv_cache"]
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
        self.proj = nn.Linear(C, C)
        self.enable_bsa = False

    def _process_attn(self, q, k, v, shape):  # LongCat's FA3 branch; plain attention for the kernel
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # "B H S D -> B S H D", contiguous
        att = torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * D ** -0.5
        return torch.einsum("bhst,bthd->bshd", att.softmax(-1), v.float()).to(q.dtype).transpose(1, 2)

    def forward_with_kv_cache(self, x, shape=None, num_cond_latents=None, kv_cache=None, num_ref_latents=None,
                              ref_img_index=None, mask_frame_range=None, ref_target_masks=None):
        B, N, C_ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view((B, N, 3, self.num_heads, self.head_dim)).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        N_t, N_h, N_w = shape
        k_cache, v_cache = kv_cache
        assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, B]
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)
        if num_cond_latents is not None and num_cond_latents > 0:
            q, k = self.rope_3d(q, k, (N_t + num_cond_latents, N_h, N_w), ref_img_index, num_ref_latents,
                                token_offset=k_cache.shape[2])
            k_full = torch.cat([k_cache, k], dim=2).contiguous()
            v_full = torch.cat([v_cache, v], dim=2).contiguous()
        assert not mask_frame_range  # (LongCat's masked-reference branch is not exercised here)
        x = self._process_attn(q, k_full, v_full, shape)
        x = x.transpose(1, 2).reshape((B, N, C_))
        return self.proj(x), None

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


class PatchEmbed3D(nn.Module):
    def forward(self, x):  # LongCat's flatten=True path: [B, C, T, H, W] -> transposed [B, N, C] view
        return x.flatten(2).transpose(1, 2)


class Dit(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = PatchEmbed3D()
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


def _rms(x, w, eps):
    x32 = x.float()
    rstd = torch.rsqrt((x32 * x32).sum(-1, keepdim=True) / x.shape[-1] + eps)
    out = torch.empty_like(x)
    out.copy_(((x32 * rstd).to(BF).float() * w.float()).to(out.dtype))
    return out


def _qk_rope(x, w, eps, cos, sin):
    B, Hh, N, Dd = x.shape
    d = torch.arange(Dd)
    xp = x[..., d ^ 1].float()  # each element's RoPE partner
    x32 = x.float()
    rstd = torch.rsqrt((x32 * x32).sum(-1, keepdim=True) / Dd + eps)
    nx = ((x32 * rstd).to(BF).float() * w.float()).to(BF).float()
    npart = ((xp * rstd).to(BF).float() * w[d ^ 1].float()).to(BF).float()
    sign = torch.where(d % 2 == 0, -1.0, 1.0)
    out = torch.empty((B, N, Hh, Dd), dtype=x.dtype).permute(0, 2, 1, 3)
    out.copy_((nx * cos + (sign * npart) * sin).to(out.dtype))
    return out


def _gated_add(x, gate, xs, T):
    B, N, Cc = x.shape
    return (x.float() + (gate * xs.float().view(B, T, N // T, Cc)).view(B, N, Cc)).to(x.dtype)


MIRRORS = {
    "modulate": lambda x, eps, scale, shift: _ln(x, scale.float() + 1.0, shift.float(), eps),
    "layer_norm": lambda x, eps, w, b: _ln(x, w.float(), b.float(), eps),
    "qk_rope": _qk_rope,
    "rms": _rms,
    "silu_mul": lambda a, b: ((a.float() / (1.0 + torch.exp(-a.float()))).to(BF).float() * b.float()).to(a.dtype),
    "gated_add": _gated_add,
}
PLAIN = {"kv_layout"}  # ops with no kernel: a layout change, checked against LongCat directly


def run_all(dit, mod, latents, cases):
    blk = dit.blocks[0]
    x = dit.x_embedder(latents)  # the residual stream, as the blocks receive it
    # As in LongCat's block, layers read the residual only through a norm.
    ada = torch.randn(1, 2, 1, 6 * C, generator=torch.Generator().manual_seed(1))
    shift, scale = ada.chunk(6, dim=-1)[:2]
    x_m = mod.modulate_fp32(blk.mod_norm_attn, x.view(1, 2, -1, C), shift, scale).view(1, -1, C)
    outs = [x_m]
    for shape, fi, nr, off in cases:
        outs += list(blk.attn(x_m, shape, fi, nr, token_offset=off))
    xn = blk.pre_crs_attn_norm(x)
    return outs + [xn, blk.cross_attn(xn), blk.ffn(x_m)]


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
    assert set(FUSED) == set(MIRRORS) | PLAIN and not unavailable, (note, unavailable)
    assert "UNAVAILABLE" not in note and "kv_layout 100.0% identical" in note, note


@torch.no_grad()
def test_a_broken_kernel_is_left_out():
    FUSED.clear()
    dit, mod = build()
    broken = dict(MIRRORS, qk_rope=lambda x, w, eps, cos, sin: _qk_rope(x, w, eps, cos, -sin))  # wrong rotation
    unavailable = {}
    note = fused_self_test(dit, mod.modulate_fp32, broken, unavailable)
    assert "qk_rope" not in FUSED and set(FUSED) == (set(MIRRORS) | PLAIN) - {"qk_rope"}
    assert "fused_ops.qk_rope" in unavailable and "qk_rope UNAVAILABLE" in note


def latents(dtype=BF):
    return torch.randn(1, C, 2, 4, 8).to(dtype)  # 64 tokens


@torch.no_grad()
def test_residual_stream_is_row_major_only_while_fused():
    dit, _, _ = fresh()
    lat = latents()
    off = dit.x_embedder(lat)
    assert not off.is_contiguous(), "LongCat's own layout when off"
    FUSE_ON[0] = True
    try:
        on = dit.x_embedder(lat)
    finally:
        FUSE_ON[0] = False
    assert on.is_contiguous() and torch.equal(on, off)


@torch.no_grad()
def test_fused_route_matches_longcat_end_to_end():
    dit, mod, counter = fresh()
    x = latents()
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
    q = out[1]
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
    x = latents()
    ref = run_all(dit, mod, x, CASES)
    assert sum(counter.values()) == 0
    FUSED.pop("qk_rope")  # q/k norms must then run normally, not be skipped
    FUSED.pop("modulate")  # (so the attention input is LongCat's exactly)
    FUSE_ON[0] = True
    try:
        out = run_all(dit, mod, x, CASES)
    finally:
        FUSE_ON[0] = False
    assert "qk_rope" not in counter and "modulate" not in counter
    for o, r in zip(out[:5], ref[:5]):  # modulate output, then q/k for both cases
        assert torch.equal(o, r)


@torch.no_grad()
def test_unsupported_tensors_fall_back_exactly():
    dit, mod, counter = fresh()
    dit.float()  # fp32 model: no kernel takes these
    x = latents(torch.float32)
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
    assert install_fused(dit, mod, collections.Counter()) == 1 + 2 * (1 + 1 + 2 + 1)  # embedder; per block attn, cross q_norm, 2 affine LNs, ffn
    blk = dit.blocks[0]
    assert "forward" not in vars(blk.mod_norm_attn), "no-affine LN is reached through modulate"
    assert mod.modulate_fp32 is not modulate_fp32
    for m in (dit.x_embedder, blk.attn.q_norm, blk.attn.k_norm, blk.attn.rope_3d, blk.cross_attn.q_norm, blk.pre_crs_attn_norm, blk.ffn):
        assert "forward" in vars(m)


# --- the block forward: LongCat's, verbatim, on small stand-in sublayers -------------------
amp = torch.amp
Ct = 32  # timestep-embedding width


class SimpleAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(C, C)

    def forward(self, x, shape=None, num_cond_latents=None, return_kv=False, num_ref_latents=None,
                ref_img_index=None, mask_frame_range=None, ref_target_masks=None):
        out = self.lin(x)
        return (out, (x[:, :8].clone(), x[:, 8:16].clone()), None) if return_kv else (out, None)

    def forward_with_kv_cache(self, x, shape=None, num_cond_latents=None, kv_cache=None, num_ref_latents=None,
                              ref_img_index=None, mask_frame_range=None, ref_target_masks=None):
        return self.lin(x) + kv_cache[0].mean(), None


class SimpleCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(C, C)

    def forward(self, x, y, y_seqlen, num_cond_latents=None, shape=None):
        return self.lin(x)


class SimpleAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(C, C)

    def forward(self, x, cond, shape=None, num_cond_latents=None, x_ref_attn_map=None, human_num=None):
        n_c = (num_cond_latents or 0) * (x.shape[1] // shape[0])
        out = self.lin(x) + cond.mean()
        return (out[:, :n_c] if n_c else None), out[:, n_c:]


class AvatarBlock(nn.Module):
    """LongCatAvatarSingleStreamBlock's layout; forward copied verbatim from @58eb336."""

    def __init__(self):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(Ct, 6 * C, bias=True))
        self.audio_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(Ct, 3 * C, bias=True))
        self.mod_norm_attn = LayerNorm_FP32(C, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(C, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(C, eps=1e-6, elementwise_affine=True)
        self.pre_video_crs_attn_norm = LayerNorm_FP32(C, eps=1e-6, elementwise_affine=True)
        self.pre_audio_crs_attn_norm = LayerNorm_FP32(24, eps=1e-6, elementwise_affine=True)
        self.attn, self.cross_attn, self.audio_cross_attn = SimpleAttn(), SimpleCross(), SimpleAudio()
        self.ffn = FeedForwardSwiGLU(C, 96)

    def forward(self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None, return_kv=False, kv_cache=None,
                skip_crs_attn=False, num_ref_latents=None, audio_hidden_states=None, ref_img_index=None,
                mask_frame_range=None, token_ref_target_masks=None, human_num=None):
        x_dtype = x.dtype
        B, N, C = x.shape
        T, _, _ = latent_shape
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)
        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache, num_ref_latents=num_ref_latents, \
                                                            ref_img_index=ref_img_index, mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv, num_ref_latents=num_ref_latents, \
                                                            ref_img_index=ref_img_index, mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
        if return_kv:
            x_s, kv_cache, x_ref_attn_map = attn_outputs
        else:
            x_s, x_ref_attn_map = attn_outputs
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_msa * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = 0
            with amp.autocast(device_type='cuda', dtype=torch.float32):
                audio_shift_mca, audio_scale_mca, audio_gate_mca = \
                        self.audio_adaLN_modulation(t[:, num_cond_latents:]).unsqueeze(2).chunk(3, dim=-1) # [B, T, 1, C]
            audio_output_cond, audio_output_noise = self.audio_cross_attn(self.pre_video_crs_attn_norm(x), self.pre_audio_crs_attn_norm(audio_hidden_states), \
                                                                            shape=latent_shape, num_cond_latents=num_cond_latents, x_ref_attn_map=x_ref_attn_map, human_num=human_num)
            with amp.autocast(device_type='cuda', dtype=torch.float32):
                audio_output_noise = modulate_fp32(self.mod_norm_attn, audio_output_noise.view(B, T-num_cond_latents, -1, C), audio_shift_mca, audio_scale_mca).view(B, -1, C)
                audio_add_x = (audio_gate_mca * audio_output_noise.view(B, T-num_cond_latents, -1, C)).view(B, -1, C) # [B, N, C]
                if audio_output_cond is not None:
                    audio_add_x = torch.cat([audio_output_cond, audio_add_x], dim=1).contiguous()
            x = x + audio_add_x
            x = x.to(x_dtype)
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self.ffn(x_m)
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_mlp * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)
        if return_kv:
            return x, kv_cache
        else:
            return x


def block_dit():
    torch.manual_seed(3)
    dit = nn.Module()
    dit.blocks = nn.ModuleList([AvatarBlock()])
    for p in dit.parameters():
        nn.init.normal_(p, mean=1.0 if p.dim() == 1 else 0.0, std=0.3 if p.dim() == 1 else 0.1)
    dit.to(BF)
    blk = dit.blocks[0]
    blk.adaLN_modulation.float()  # fp32 here; on the GPU LongCat's fp32 autocast does this
    blk.audio_adaLN_modulation.float()
    FUSED.clear()
    FUSE_ON[0] = False
    counter = collections.Counter()
    install_fused(dit, types.SimpleNamespace(modulate_fp32=modulate_fp32), counter)
    return dit, counter


@torch.no_grad()
def test_block_forward_matches_longcat_exactly():
    dit, counter = block_dit()
    FUSED["gated_add"] = _gated_add  # only the residual add, so the rest is LongCat's own code
    blk = dit.blocks[0]
    T, h, w = 3, 2, 4
    N = T * h * w
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1, N, C, generator=g) * 4).to(BF)
    t = torch.randn(1, T, Ct, generator=g)
    audio = torch.randn(T, 5, 24, generator=g).to(BF)
    y = torch.zeros(1, 4, C, dtype=BF)
    kv = (torch.randn(1, 8, C, generator=g).to(BF), torch.randn(1, 8, C, generator=g).to(BF))
    modes = {  # name: (kwargs, residual adds the kernel should take)
        "first chunk (reference frame, audio cond)": (dict(num_cond_latents=1), 2),
        "continuation step (cached frames)": (dict(num_cond_latents=1, kv_cache=kv), 3),
        "carried-frame cache build": (dict(num_cond_latents=T, return_kv=True, skip_crs_attn=True), 2),
    }
    for name, (kw, calls) in modes.items():
        args = (x, y, t, [4], (T, h, w))
        ref = blk(*args, audio_hidden_states=audio, **kw)
        before = counter["gated_add"]
        FUSE_ON[0] = True
        try:
            out = blk(*args, audio_hidden_states=audio, **kw)
        finally:
            FUSE_ON[0] = False
        if kw.get("return_kv"):
            assert all(torch.equal(a, b) for a, b in zip(out[1], ref[1])), name
            out, ref = out[0], ref[0]
        assert torch.equal(out, ref), name
        assert counter["gated_add"] - before == calls, (name, counter)


@torch.no_grad()
def test_cached_attention_layout_is_identical():
    dit, mod, counter = fresh(with_mirrors=False)
    FUSED["kv_layout"] = True  # alone, so q/k still go through LongCat's norms and RoPE
    attn = dit.blocks[0].attn
    g = torch.Generator().manual_seed(2)
    n_t, n_h, n_w, n_c = 3, 4, 8, 2
    x = torch.randn(1, n_t * n_h * n_w, C, generator=g).to(BF)
    kc, vc = (torch.randn(1, H, n_c * n_h * n_w, D, generator=g).to(BF) for _ in range(2))
    kw = dict(shape=(n_t, n_h, n_w), num_cond_latents=n_c, kv_cache=(kc, vc), num_ref_latents=1,
              ref_img_index=10, mask_frame_range=0, ref_target_masks=None)
    ref = attn.forward_with_kv_cache(x, **kw)[0]
    FUSE_ON[0] = True
    try:
        out = attn.forward_with_kv_cache(x, **kw)[0]
        attn.enable_bsa = True  # block-sparse attention keeps LongCat's path
        attn.forward_with_kv_cache(x, **kw)
    finally:
        FUSE_ON[0] = False
        attn.enable_bsa = False
    assert torch.equal(out, ref) and counter["kv_layout"] == 1, counter

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
