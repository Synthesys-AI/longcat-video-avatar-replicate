"""CPU tests for the SDPA stand-ins that back LongCat's FlashAttention-3 code path.

Runs anywhere with torch (no GPU, no cog): pulls `_sdpa` / `_sdpa_varlen` out of
predict.py by AST so the module's cog/CUDA imports are never executed.
    python -m pytest tests/  or  python tests/test_attention_shims.py
"""
import ast
import pathlib
import types

import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_tree = ast.parse(SRC.read_text())
# `_DENSE_OVERRIDE` is the module-level switch `_make_fa_shim` reads at call time.
_ns = {"torch": torch, "types": types, "_DENSE_OVERRIDE": [None]}
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {"_sdpa", "_sdpa_varlen", "_fa3_variant", "_make_fa_shim"}:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
_sdpa, _sdpa_varlen, _fa3_variant = _ns["_sdpa"], _ns["_sdpa_varlen"], _ns["_fa3_variant"]
_make_fa_shim, _DENSE_OVERRIDE = _ns["_make_fa_shim"], _ns["_DENSE_OVERRIDE"]


def naive(q, k, v, scale):  # q: [S, H, D], k/v: [T, H, D]
    w = torch.einsum("shd,thd->hst", q, k) * scale
    return torch.einsum("hst,thd->shd", w.softmax(-1), v)


def test_sdpa_matches_naive_bshd():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 37, 4, 16), torch.randn(1, 53, 4, 16), torch.randn(1, 53, 4, 16)
    out = _sdpa(q, k, v, softmax_scale=0.25)
    assert out.shape == q.shape
    assert torch.allclose(out[0], naive(q[0], k[0], v[0], 0.25), atol=1e-5)


def test_varlen_matches_per_sequence():
    torch.manual_seed(1)
    q, k, v = torch.randn(30, 4, 16), torch.randn(25, 4, 16), torch.randn(25, 4, 16)
    cu_q, cu_k = torch.tensor([0, 10, 30], dtype=torch.int32), torch.tensor([0, 7, 25], dtype=torch.int32)
    out = _sdpa_varlen(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                       max_seqlen_q=20, max_seqlen_k=18, softmax_scale=0.25)
    assert out.shape == q.shape
    assert torch.allclose(out[:10], naive(q[:10], k[:7], v[:7], 0.25), atol=1e-5)
    assert torch.allclose(out[10:], naive(q[10:], k[7:], v[7:], 0.25), atol=1e-5)


def test_longcat_call_patterns_on_tuple_wrapped_results():
    """`x, *_ = f(...)` and `f_varlen(...)[0]` must yield the full output (batch size 1)."""
    q = torch.randn(1, 12, 2, 8)
    x, *_ = (_sdpa(q, q, q, softmax_scale=0.3), None)
    assert x.shape == (1, 12, 2, 8)
    cu = torch.tensor([0, 12], dtype=torch.int32)
    y = (_sdpa_varlen(q[0], q[0], q[0], cu, cu, softmax_scale=0.3), None)[0]
    assert y.shape == (12, 2, 8)
    # The bug the wrapper prevents: indexing a bare tensor drops a dimension.
    assert _sdpa_varlen(q[0], q[0], q[0], cu, cu, softmax_scale=0.3)[0].shape == (2, 8)


def test_fa3_variant_names_match_kernel_hub_folders():
    # The image's torch 2.6 PyPI wheel (CUDA 12.4, old C++ ABI) and a couple of others.
    assert _fa3_variant("2.6.0+cu124", False, "12.4") == "torch26-cxx98-cu124-x86_64-linux"
    assert _fa3_variant("2.8.0", True, "12.8") == "torch28-cxx11-cu128-x86_64-linux"
    assert _fa3_variant("2.10.1+cu130", True, "13.0") == "torch210-cxx11-cu130-x86_64-linux"



def test_shim_returns_tuples_and_routes_dense_calls_to_the_override():
    calls = []

    def raw(q, k, v, softmax_scale=None, causal=False, **kw):
        calls.append(("raw", softmax_scale, causal, kw))
        return "fa3-out"

    def raw_varlen(*a, **kw):
        calls.append(("varlen",))
        return "varlen-out"

    shim = _make_fa_shim(raw, raw_varlen)
    # No override: FA3, wrapped as a tuple the way LongCat unpacks it.
    x, *_ = shim.flash_attn_func("q", "k", "v", softmax_scale=0.5)
    assert x == "fa3-out" and calls[-1] == ("raw", 0.5, False, {})
    assert shim.flash_attn_varlen_func("q", "k", "v", cu_seqlens_q=None)[0] == "varlen-out"

    # A request turns SageAttention on: dense calls go there, varlen stays on FA3.
    def sage(q, k, v, scale):
        calls.append(("sage", scale))
        return "sage-out"

    _DENSE_OVERRIDE[0] = sage
    try:
        x, *_ = shim.flash_attn_func("q", "k", "v", softmax_scale=0.25)
        assert x == "sage-out" and calls[-1] == ("sage", 0.25)
        assert shim.flash_attn_varlen_func("q", "k", "v")[0] == "varlen-out"
        # Anything unusual (causal, extra options) still goes to FA3.
        assert shim.flash_attn_func("q", "k", "v", causal=True)[0] == "fa3-out"
        assert shim.flash_attn_func("q", "k", "v", window_size=(1, 1))[0] == "fa3-out"
    finally:
        _DENSE_OVERRIDE[0] = None
    assert shim.flash_attn_func("q", "k", "v")[0] == "fa3-out"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
