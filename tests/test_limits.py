"""CPU tests for the per-step video length limits: each limit's estimated render time stays
inside the runner's time budget (Replicate stops predictions at 30 min), from per-segment
times measured on an H100 (720p with 6 steps is an estimate).
    python -m pytest tests/  or  python tests/test_limits.py
"""
import ast
import math
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
            "MAX_DURATION", "TIME_BUDGET_SECONDS", "FPS", "NUM_FRAMES", "NUM_COND_FRAMES"} for t in node.targets):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
    if isinstance(node, ast.FunctionDef) and node.name == "_max_duration":
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
max_duration = _ns["_max_duration"]

# seconds per 3.2 s continuation segment, measured (720p/6 estimated); first segment +20%
SEGMENT_SECONDS = {("480p", 8): 28.0, ("480p", 6): 22.1, ("480p", 4): 16.0,
                   ("720p", 8): 97.6, ("720p", 6): 75.0, ("720p", 4): 51.3}


def estimated_render(resolution, steps, seconds):
    frames = round(seconds * _ns["FPS"])
    new_per_segment = _ns["NUM_FRAMES"] - _ns["NUM_COND_FRAMES"]
    continuations = max(0, math.ceil((frames - _ns["NUM_FRAMES"]) / new_per_segment))
    seg = SEGMENT_SECONDS[(resolution, steps)]
    return 1.2 * seg + continuations * seg + 10.0  # + audio prep and encode


def test_presets_and_custom_subsets():
    assert max_duration("480p", 8) == max_duration("480p", 4) == 90.0
    assert (max_duration("720p", 8), max_duration("720p", 6), max_duration("720p", 4)) == (45.0, 60.0, 90.0)
    assert max_duration("720p", 7) == 45.0 and max_duration("720p", 5) == 60.0 and max_duration("720p", 3) == 90.0


def test_every_limit_fits_the_time_budget():
    budget = _ns["TIME_BUDGET_SECONDS"]
    for (res, steps) in SEGMENT_SECONDS:
        secs = estimated_render(res, steps, max_duration(res, steps))
        assert secs < 0.97 * budget, (res, steps, round(secs), budget)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
