"""CPU tests for the experimental `fp8_matmuls` input's routing: which DiT layers get the
FP8 path, and that it survives LongCat's LoRA wrapping (enable/disable every request).

The FP8 kernel itself needs an H100 and is self-tested at boot; here a recording stub
stands in for it. `LoRAHost` copies LongCat @58eb336's `enable_loras` /
`disable_all_loras` / `_create_multi_lora_forward` logic.
    python -m pytest tests/  or  python tests/test_fp8_routing.py
"""
import ast
import pathlib

import torch
import torch.nn as nn

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"_FP8_ON", "FP8_MIN_ROWS"} for t in node.targets):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
    if isinstance(node, ast.FunctionDef) and node.name in {"_fp8_targets", "_install_fp8", "_linear_maybe_fp8"}:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
fp8_targets, install_fp8, FP8_ON = _ns["_fp8_targets"], _ns["_install_fp8"], _ns["_FP8_ON"]


class Block(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(512, 6 * d))
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.small = nn.Linear(d, 64)  # too small for FP8
        self.w2 = nn.Linear(4 * d, d, bias=False)


class LoRAHost(nn.Module):
    """A DiT-like module tree plus LongCat's LoRA wrapping logic."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final = nn.Linear(1024, 1024)  # outside the blocks: left alone
        self.lora_calls = 0

    def enable_loras(self, names):
        self.disable_all_loras()
        for name, module in self.named_modules():
            if name in names:
                if not hasattr(module, "org_forward"):
                    module.org_forward = module.forward
                module.forward = self._create_multi_lora_forward(module)

    def _create_multi_lora_forward(self, module):
        def multi_lora_forward(x, *args, **kwargs):
            org_output = module.org_forward(x, *args, **kwargs)
            self.lora_calls += 1
            return org_output  # (the real one adds lora_up(lora_down(x)) * multiplier)
        return multi_lora_forward

    def disable_all_loras(self):
        for _, module in self.named_modules():
            if hasattr(module, "org_forward"):
                module.forward = module.org_forward
                delattr(module, "org_forward")


def recording_fp8(calls):
    def fp8_linear(x, weight, bias):
        calls.append(tuple(weight.shape))
        return torch.nn.functional.linear(x, weight, bias)
    return fp8_linear


def test_targets_are_the_big_block_layers_only():
    names = [n for n, _ in fp8_targets(LoRAHost())]
    assert names == ["blocks.0.qkv", "blocks.0.proj", "blocks.0.w2", "blocks.1.qkv", "blocks.1.proj", "blocks.1.w2"]


def test_routing_survives_lora_enable_disable_cycles():
    dit = LoRAHost()
    lora_layers = {"blocks.0.qkv", "blocks.1.proj", "blocks.0.small"}
    dit.enable_loras(lora_layers)  # setup leaves the LoRA mix enabled
    calls, counter = [], [0]
    assert install_fp8(dit, recording_fp8(calls), counter) == 6
    big = torch.randn(1, 300, 1024)   # 300 rows: FP8
    tiny = torch.randn(1, 10, 1024)   # 10 rows: stays bf16
    for request in range(3):          # every request re-enables the LoRA mix
        dit.enable_loras(lora_layers)
        lora_before, calls_before = dit.lora_calls, len(calls)
        FP8_ON[0] = True
        try:
            out = dit.blocks[0].qkv(big)        # LoRA-wrapped layer
            dit.blocks[0].proj(big)             # plain layer
            dit.blocks[1].proj(tiny)            # LoRA-wrapped, too few rows
            dit.final(big)                      # not a target
        finally:
            FP8_ON[0] = False
        assert out.shape == (1, 300, 3072)
        assert len(calls) - calls_before == 2, (request, calls)
        assert dit.lora_calls - lora_before == 2, "LoRA wrapper must still run on top"
        before = len(calls)
        dit.blocks[0].qkv(big)  # FP8 off: plain bf16
        assert len(calls) == before
    assert counter[0] == 6
    dit.disable_all_loras()  # with LoRAs off the FP8 path is still in place
    FP8_ON[0] = True
    try:
        dit.blocks[0].qkv(big)
    finally:
        FP8_ON[0] = False
    assert counter[0] == 7


def test_off_path_matches_nn_linear_exactly():
    dit = LoRAHost()
    lin = dit.blocks[0].proj
    x = torch.randn(2, 300, 1024)
    ref = nn.Linear.forward(lin, x)
    install_fp8(dit, recording_fp8([]), [0])
    assert torch.equal(lin(x), ref)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
