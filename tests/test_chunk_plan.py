"""CPU tests for `_plan_chunks`: how many frames each segment generates, so the last one
stops at the end of the audio instead of rendering a full 3.2 s and trimming it.
    python -m pytest tests/  or  python tests/test_chunk_plan.py
"""
import ast
import math
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"NUM_FRAMES", "NUM_COND_FRAMES"} for t in node.targets
    ):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "_plan_chunks":
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
plan_chunks, FULL, COND = _ns["_plan_chunks"], _ns["NUM_FRAMES"], _ns["NUM_COND_FRAMES"]


def new_frames(plan):
    return plan[0] + sum(n - COND for n in plan[1:])


def test_known_clips():
    assert plan_chunks(200) == [93, 93, 41]  # 8 s: the last segment was 93, 53 of its 80 new frames thrown away
    assert plan_chunks(62) == [65]           # 2.5 s: one short segment instead of 93 frames
    assert plan_chunks(93) == [93]
    assert plan_chunks(94) == [93, 17]       # one frame past a full segment: the smallest continuation
    p = plan_chunks(2049)                    # the 82 s demo speech
    assert len(p) == 26 and p[:-1] == [93] * 25 and p[-1] == 49


def test_every_length_is_covered_with_at_most_3_spare_frames():
    for target in range(1, 2400):
        p = plan_chunks(target)
        assert all(n % 4 == 1 and 1 <= n <= FULL for n in p), (target, p)
        assert all(n == FULL for n in p[:-1]), (target, p)
        assert target <= new_frames(p) <= target + 3, (target, p)
        # Same number of segments as before the change: only the last one got shorter.
        old_count = 1 if target <= FULL else 1 + math.ceil((target - FULL) / (FULL - COND))
        assert len(p) == old_count, (target, p, old_count)


def test_saving_on_short_clips():
    # 8 s: 3 segments used to render 93+80+80 = 253 frames for 200 needed.
    assert new_frames(plan_chunks(200)) == 201


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
