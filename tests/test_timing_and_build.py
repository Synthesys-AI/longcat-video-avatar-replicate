"""CPU tests for the phase timer behind the per-segment timing logs, and for the build
step that copies FlashAttention 3 into the image.
    python -m pytest tests/  or  python tests/test_timing_and_build.py
"""
import ast
import collections
import functools
import inspect
import pathlib
import re
import time
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "predict.py"
_tree = ast.parse(SRC.read_text())
_ns = {"collections": collections, "functools": functools, "time": time, "contextmanager": contextmanager}
for node in _tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "_PhaseTimer":
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
PhaseTimer = _ns["_PhaseTimer"]


def test_wrapped_calls_add_up_per_phase_and_keep_signatures():
    syncs = []
    timer = PhaseTimer(sync=lambda: syncs.append(1))

    class Pipe:
        def decode(self, latents, scale=1.0):
            time.sleep(0.02)
            return latents * scale

    pipe = Pipe()
    timer.wrap(pipe, "decode", "vae decode")
    assert pipe.decode(2, scale=3.0) == 6.0
    assert pipe.decode(1) == 1
    assert 0.03 < timer.totals["vae decode"] < 0.5
    assert len(syncs) == 4  # before and after each call, so GPU work is counted
    assert list(inspect.signature(pipe.decode).parameters) == ["latents", "scale"]


def test_nested_phases_and_breakdown():
    timer = PhaseTimer()
    before = timer.snapshot()
    with timer.phase("dit"):
        time.sleep(0.08)
    with timer.phase("text"):
        pass  # under 0.05 s: left out of the breakdown
    line = timer.breakdown(before, timer.snapshot(), total=0.5)
    assert re.fullmatch(r"dit 0\.[01]s, other 0\.[34]s", line), line
    timer.reset()
    assert timer.snapshot() == {}


def test_phase_counts_time_even_when_the_call_raises():
    timer = PhaseTimer()
    try:
        with timer.phase("voice isolation"):
            raise ValueError("no voice")
    except ValueError:
        pass
    assert "voice isolation" in timer.totals


def test_image_fa3_copy_uses_the_same_revision_as_predict():
    rev = re.search(r'FA3_KERNEL_REVISION = "([0-9a-f]{40})"', SRC.read_text()).group(1)
    cog = (ROOT / "cog.yaml").read_text()
    assert f"revision='{rev}'" in cog
    assert "local_dir='/opt/flash-attn3'" in cog
    assert 'FA3_BAKED_DIR = FsPath("/opt/flash-attn3")' in SRC.read_text()
    sage_rev = re.search(r'SAGE_KERNEL_REVISION = "([0-9a-f]{40})"', SRC.read_text()).group(1)
    assert f"revision='{sage_rev}'" in cog and "local_dir='/opt/sage-attention'" in cog
    assert 'SAGE_BAKED_DIR = FsPath("/opt/sage-attention")' in SRC.read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
