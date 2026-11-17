"""CPU tests for the SDPA stand-ins that back LongCat's FlashAttention-3 code path.

Runs anywhere with torch (no GPU, no cog): pulls `_sdpa` / `_sdpa_varlen` out of
predict.py by AST so the module's cog/CUDA imports are never executed.
    python -m pytest tests/  or  python tests/test_attention_shims.py
"""
import ast
import pathlib

import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_tree = ast.parse(SRC.read_text())
_ns = {"torch": torch}
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {"_sdpa", "_sdpa_varlen", "_fa3_variant"}:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
_sdpa, _sdpa_varlen, _fa3_variant = _ns["_sdpa"], _ns["_sdpa_varlen"], _ns["_fa3_variant"]


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
