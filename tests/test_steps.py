"""CPU tests for the opt-in step counts: `_subset_schedule` runs a subset of the 8 trained
noise levels (and leaves the default schedule untouched), `_step_indices` validates the
developer override. `FakePipe.get_timesteps_sigmas` copies LongCat @58eb336's avatar-v1.5
branch, which returns all 8 levels whatever step count is asked for.
    python -m pytest tests/  or  python tests/test_steps.py
"""
import ast
import pathlib

import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"NUM_STEPS", "STEP_SUBSETS"} for t in node.targets):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
    if isinstance(node, ast.FunctionDef) and node.name in {"_step_indices", "_subset_schedule"}:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
step_indices, subset_schedule, SUBSETS = _ns["_step_indices"], _ns["_subset_schedule"], _ns["STEP_SUBSETS"]


class FakePipe:
    num_timesteps, num_distill_sample_steps = 1000, 8

    def get_timesteps_sigmas(self, sampling_steps, use_distill=False):  # LongCat, avatar-v1.5
        if use_distill:
            distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32)
            distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()
            distill_indices = self.num_timesteps - distill_indices
            sigmas = torch.flip(torch.linspace(0, 1, self.num_timesteps), [0])
            sigmas = torch.flip(sigmas[distill_indices], [0]).float()
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps)
        return sigmas.to(torch.float32)


def test_default_schedule_is_untouched():
    pipe = FakePipe()
    full = pipe.get_timesteps_sigmas(8, use_distill=True)
    state = subset_schedule(pipe)
    assert state["indices"] is None
    assert torch.equal(pipe.get_timesteps_sigmas(8, use_distill=True), full)
    assert len(full) == 8 and full[0] == 1.0


def test_subsets_are_trained_levels_starting_from_noise():
    pipe = FakePipe()
    full = pipe.get_timesteps_sigmas(8, use_distill=True)
    state = subset_schedule(pipe)
    for steps in (6, 4):
        state["indices"] = SUBSETS[steps]
        sig = pipe.get_timesteps_sigmas(len(SUBSETS[steps]), use_distill=True)
        assert len(sig) == steps and sig[0] == 1.0
        assert torch.equal(sig, full[list(SUBSETS[steps])])
        assert all(sig[i] > sig[i + 1] for i in range(steps - 1))
        assert sig[-1] == full[-1], "a draft still ends on the last trained level"
    state["indices"] = None
    assert torch.equal(pipe.get_timesteps_sigmas(8, use_distill=True), full)


def test_dmd_switch_is_remapped_to_the_kept_levels():
    dmd = lambda idx, mix_steps=4: sum(1 for i in idx if i < mix_steps)
    assert dmd(SUBSETS[8]) == 4 and dmd(SUBSETS[6]) == 2 and dmd(SUBSETS[4]) == 2
    assert dmd(step_indices(8, "0,4,6,7")) == 1


def test_override_parsing_and_validation():
    assert step_indices(6) == SUBSETS[6] and step_indices(4, " ") == SUBSETS[4]
    assert step_indices(8, "0, 4, 6, 7") == (0, 4, 6, 7)
    for bad in ("1,2,3", "0,3,2", "0,0,4", "0,8", "a,b", "0,4,"):
        try:
            step_indices(8, bad)
            raise AssertionError(f"{bad!r} should be rejected")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
