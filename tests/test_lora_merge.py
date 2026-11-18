"""CPU tests for the merged add-ons (`merge_lora` input): the merged layer matches
LongCat's separate LoRA branch, never changes the stored weight, follows the DMD/OPSD
switch at call time, survives the per-request re-wrapping, and hands FP8 the merged weight.

`LoRAModule` / `LoRAUPParallel` / `Host` copy LongCat @58eb336's `lora_utils.py` and the
avatar DiT's `enable_loras` / `_create_multi_lora_forward` / `set_lora_mix_step` logic.
    python -m pytest tests/  or  python tests/test_lora_merge.py
"""
import ast
import pathlib

import torch
import torch.nn as nn

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch}
_names = {"_FP8_ON", "FP8_MIN_ROWS", "_MERGE_ON", "MERGE_ROWS_PER_FEATURE"}
_funcs = {"_linear_maybe_fp8", "_merged_weight", "_make_merging_factory", "_merge_self_test"}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _names for t in node.targets):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
    if isinstance(node, ast.FunctionDef) and node.name in _funcs:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
MERGE_ON, FP8_ON = _ns["_MERGE_ON"], _ns["_FP8_ON"]
make_factory, merge_self_test = _ns["_make_merging_factory"], _ns["_merge_self_test"]


class LoRAUPParallel(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        xs = torch.chunk(x, len(self.blocks), dim=-1)
        return torch.cat([self.blocks[i](xs[i]) for i in range(len(self.blocks))], dim=-1)


class LoRAModule(nn.Module):
    def __init__(self, name, org, rank=8, alpha=4, n_sep=1):
        super().__init__()
        self.lora_name = name
        if n_sep > 1:
            self.lora_down = nn.Linear(org.in_features, n_sep * rank, bias=False)
            self.lora_up = LoRAUPParallel([nn.Linear(rank, org.out_features // n_sep, bias=False) for _ in range(n_sep)])
        else:
            self.lora_down = nn.Linear(org.in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, org.out_features, bias=False)
        self.register_buffer("alpha_scale", torch.tensor(alpha / rank))
        for p in self.parameters():  # trained-looking, non-zero up weights
            nn.init.normal_(p, std=0.05)
        self.multiplier, self.use_lora = 1.0, True


class Host(nn.Module):
    """Two add-on networks (dmd, opsd) on a few layers, wrapped LongCat's way."""

    def __init__(self, d=64, dtype=torch.float32):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.ffn = nn.Linear(d, 4 * d, bias=False)
        self.plain = nn.Linear(d, d)
        self.lora_dict = {
            key: [LoRAModule("qkv", self.qkv, n_sep=3), LoRAModule("ffn", self.ffn)] for key in ("dmd", "opsd")
        }
        for loras in self.lora_dict.values():
            for lora in loras:
                lora.to(dtype)
        self.to(dtype)
        self.lora_mix_config, self._lora_mix_active = None, None

    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            total = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_up(lora.lora_down(x.to(lora.lora_down.weight.dtype)))
                    total += lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
            return org_output + total
        return multi_lora_forward

    def enable_loras(self, keys):
        for _, m in self.named_modules():
            if hasattr(m, "org_forward"):
                m.forward = m.org_forward
                delattr(m, "org_forward")
        per_module = {}
        for key in keys:
            for lora in self.lora_dict[key]:
                per_module.setdefault(lora.lora_name, []).append(lora)
        for name, loras in per_module.items():
            module = getattr(self, name)
            if not hasattr(module, "org_forward"):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)

    def enable_lora_mix(self, dmd, opsd, mix_alpha, mix_steps):
        self.enable_loras([dmd, opsd])
        self.lora_mix_config = {"mix_alpha": mix_alpha, "mix_steps": mix_steps}
        self._lora_mix_active = None
        self.set_lora_mix_step(0)

    def set_lora_mix_step(self, step):
        mixed = step < self.lora_mix_config["mix_steps"]
        if mixed == self._lora_mix_active:
            return
        alpha = self.lora_mix_config["mix_alpha"]
        for key, mult in (("dmd", alpha if mixed else 0.0), ("opsd", 1.0 - alpha if mixed else 1.0)):
            for lora in self.lora_dict[key]:
                lora.multiplier, lora.use_lora = mult, mult != 0.0
        self._lora_mix_active = mixed


def merging_host(dtype=torch.float32, mix_alpha=1.0):
    dit = Host(dtype=dtype)
    counter = [0]
    dit._create_multi_lora_forward = make_factory(dit._create_multi_lora_forward, counter)
    dit.enable_lora_mix("dmd", "opsd", mix_alpha, 4)
    return dit, counter


def both_paths(dit, layer, x):
    MERGE_ON[0] = False
    ref = layer(x)
    MERGE_ON[0] = True
    try:
        out = layer(x)
    finally:
        MERGE_ON[0] = False
    return ref, out


@torch.no_grad()
def test_merged_matches_the_branch_in_every_mix_state():
    torch.manual_seed(0)
    for mix_alpha in (1.0, 0.3):              # one add-on at a time, then both at once
        dit, counter = merging_host(mix_alpha=mix_alpha)
        x = torch.randn(2, 40, 64)            # 80 rows >= 64/4: merged
        for step in (0, 3, 4, 7):             # before and after the switch at mix_steps=4
            dit.set_lora_mix_step(step)
            for layer in (dit.qkv, dit.ffn):  # split (3-way) and plain up-projection
                before = counter[0]
                ref, out = both_paths(dit, layer, x)
                assert counter[0] == before + 1, "the merged path must have run"
                torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_stored_weights_never_change():
    torch.manual_seed(1)
    dit, _ = merging_host()
    saved = {n: p.detach().clone() for n, p in dit.named_parameters()}
    MERGE_ON[0] = True
    try:
        for step in range(8):
            dit.set_lora_mix_step(step)
            dit.qkv(torch.randn(1, 64, 64))
            dit.ffn(torch.randn(1, 64, 64))
    finally:
        MERGE_ON[0] = False
    for n, p in dit.named_parameters():
        assert torch.equal(p, saved[n]), n


@torch.no_grad()
def test_small_calls_and_off_switch_use_the_branch():
    torch.manual_seed(2)
    dit, counter = merging_host()
    MERGE_ON[0] = True
    try:
        dit.qkv(torch.randn(1, 15, 64))       # 15 rows < 64/4: text-sized, keeps the branch
    finally:
        MERGE_ON[0] = False
    dit.qkv(torch.randn(1, 64, 64))           # switch off
    assert counter[0] == 0
    for lora in dit.lora_dict["dmd"] + dit.lora_dict["opsd"]:
        lora.use_lora = False                 # no add-on active: plain layer, via the branch
    x = torch.randn(1, 64, 64)
    MERGE_ON[0] = True
    try:
        out = dit.ffn(x)
    finally:
        MERGE_ON[0] = False
    assert counter[0] == 0 and torch.equal(out, nn.Linear.forward(dit.ffn, x))


@torch.no_grad()
def test_bf16_merge_is_as_close_to_float64_as_the_branch():
    torch.manual_seed(3)
    dit, _ = merging_host(dtype=torch.bfloat16)
    x = torch.randn(256, 64, dtype=torch.bfloat16)
    for layer in (dit.qkv, dit.ffn):
        loras = [lora for lora in dit.lora_dict["dmd"] if getattr(dit, lora.lora_name) is layer]
        w = layer.weight.double()
        for lora in loras:
            up = lora.lora_up
            ups = [b.weight for b in up.blocks] if hasattr(up, "blocks") else [up.weight]
            down = lora.lora_down.weight.double().chunk(len(ups), dim=0)
            w = w + float(lora.alpha_scale) * torch.cat([u.double() @ d for u, d in zip(ups, down)], dim=0)
        truth = x.double() @ w.T + (layer.bias.double() if layer.bias is not None else 0)
        ref, out = both_paths(dit, layer, x)
        err_branch = ((ref.double() - truth).norm() / truth.norm()).item()
        err_merged = ((out.double() - truth).norm() / truth.norm()).item()
        assert err_merged < max(2 * err_branch, 4e-3), (err_merged, err_branch)


@torch.no_grad()
def test_rewrapping_every_request_keeps_the_merge():
    torch.manual_seed(4)
    dit, counter = merging_host()
    x = torch.randn(1, 64, 64)
    MERGE_ON[0] = True
    try:
        for request in range(3):
            dit.enable_lora_mix("dmd", "opsd", 1.0, 4)  # what run() does per request
            dit.ffn(x)
    finally:
        MERGE_ON[0] = False
    assert counter[0] == 3


@torch.no_grad()
def test_fp8_gets_the_merged_weight():
    torch.manual_seed(5)
    dit, _ = merging_host()
    seen = []

    def fp8_linear(x, weight, bias):
        seen.append(weight)
        return torch.nn.functional.linear(x, weight, bias)

    dit.ffn._fp8 = (fp8_linear, [0])  # as _install_fp8 tags its layers
    x = torch.randn(1, 300, 64)       # >= FP8_MIN_ROWS rows
    ref, _ = both_paths(dit, dit.ffn, x)
    FP8_ON[0] = MERGE_ON[0] = True
    try:
        out = dit.ffn(x)
    finally:
        FP8_ON[0] = MERGE_ON[0] = False
    assert len(seen) == 1 and not torch.equal(seen[0], dit.ffn.weight)  # W', not W
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_with_grad_on_the_branch_runs():
    torch.manual_seed(7)
    dit, counter = merging_host()
    MERGE_ON[0] = True
    try:
        dit.ffn(torch.randn(1, 64, 64)).sum().backward()
    finally:
        MERGE_ON[0] = False
    assert counter[0] == 0 and dit.lora_dict["dmd"][1].lora_up.weight.grad is not None


def test_self_test_runs_on_the_host():
    torch.manual_seed(6)
    dit, _ = merging_host()
    note = merge_self_test(dit, rows=64)
    assert note.startswith("2 layers, self-test rel err"), note
    assert MERGE_ON[0] is False and dit._lora_mix_active is True  # left as found


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
