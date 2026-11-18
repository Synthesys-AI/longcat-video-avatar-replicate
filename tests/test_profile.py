"""CPU tests for the developer `profile` input: kernel categories, labels that are fully
undone after the profiled segment, and a report built from a real (CPU-only) profile.
    python -m pytest tests/  or  python tests/test_profile.py
"""
import ast
import collections
import pathlib
from contextlib import contextmanager

import torch
import torch.nn as nn

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch, "collections": collections, "contextmanager": contextmanager}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_KERNEL_CATEGORIES" for t in node.targets):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in {"_kernel_category", "_profile_labels", "_profile_report"}:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
category, profile_labels, profile_report = _ns["_kernel_category"], _ns["_profile_labels"], _ns["_profile_report"]


def test_kernel_categories():
    assert category("void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128>>") == "attention"
    assert category("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x256x64") == "matmul"
    assert category("nvjet_tst_128x256_64x4_2x1_v_bz_coopA_NNT") == "matmul"
    assert category("cudnn_infer_ampere_scudnn_128x64_relu_interior_nn_v1") == "conv (VAE)"
    assert category("void at::native::(anonymous namespace)::CatArrayBatchedCopy<float>") == "copy / concat"
    assert category("void at::native::vectorized_elementwise_kernel<4, mul_functor>") == "elementwise"
    assert category("void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float>") == "norm / reduce"
    assert category("_rowwise_fp8") == "other"


class LayerNorm_FP32(nn.LayerNorm):
    pass


class RotaryPositionalEmbeddingX(nn.Module):
    def forward(self, q, k):
        return q, k


class FakeDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm_FP32(8)
        self.rope = RotaryPositionalEmbeddingX()
        self.wrapped = nn.Linear(8, 8)
        self.plain = nn.Linear(8, 8)
        base = self.wrapped.forward
        self.wrapped.org_forward = base                                  # as LongCat's LoRA wrapper leaves it
        self.wrapped.forward = lambda x: self.wrapped.org_forward(x) * 1.0


def test_labels_are_applied_then_fully_undone():
    dit = FakeDit()
    before = {"norm": "forward" in vars(dit.norm), "wrapped_fwd": dit.wrapped.forward,
              "wrapped_org": dit.wrapped.org_forward, "plain": "forward" in vars(dit.plain)}
    x = torch.randn(2, 8)
    ref = dit.wrapped(x)
    with profile_labels(dit):
        assert "forward" in vars(dit.norm)                               # labelled
        assert dit.wrapped.forward is not before["wrapped_fwd"]
        assert torch.equal(dit.wrapped(x), ref)                          # same maths
        dit.rope(x, x)
    assert ("forward" in vars(dit.norm)) == before["norm"]               # class method again
    assert dit.wrapped.forward is before["wrapped_fwd"] and dit.wrapped.org_forward is before["wrapped_org"]
    assert ("forward" in vars(dit.plain)) == before["plain"]


def test_report_from_a_real_profile():
    from torch.profiler import ProfilerActivity, profile

    dit = FakeDit()
    with profile_labels(dit):
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            dit.norm(torch.randn(4, 8))
            dit.wrapped(torch.randn(4, 8))
    lines = profile_report(prof, wall=1.0)
    assert lines[0].startswith("[profile] segment: 1.0s wall, 0.0s of GPU kernels")
    assert all(line.startswith("[profile]") for line in lines)
    assert any("LayerNorm_FP32" in line for line in lines)   # CPU-side label rows are reported


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
