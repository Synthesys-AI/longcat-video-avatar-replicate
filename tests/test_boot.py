"""CPU tests for the cold-boot changes: both weight repos download at once in background
threads (a stand-in replaces huggingface_hub.snapshot_download), a repo already on disk is
skipped, and the boot report is printed exactly once.
    python -m pytest tests/  or  python tests/test_boot.py
"""
import ast
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"


def load(root):
    ns = {"os": os, "shutil": shutil, "time": time, "contextmanager": contextmanager}
    for node in ast.parse(SRC.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
                "AVATAR_REPO", "AVATAR_REVISION", "BASE_REPO", "BASE_REVISION", "AVATAR_FILES", "BASE_FILES"}
                for t in node.targets):
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
        if (isinstance(node, ast.ClassDef) and node.name == "_BootLog") or (
                isinstance(node, ast.FunctionDef) and node.name == "_start_weight_downloads"):
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
    ns.update(WEIGHTS_ROOT=root, AVATAR_DIR=root / "avatar", BASE_DIR=root / "base")
    return ns


class FakeHub:
    """Records when each repo's download starts and ends; writes one file per repo."""

    def __init__(self, delay=0.3):
        self.delay, self.calls, self.lock = delay, [], threading.Lock()

    def snapshot_download(self, repo_id, revision, local_dir, allow_patterns, max_workers):
        start = time.time()
        time.sleep(self.delay)
        pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(local_dir) / "weights.bin").write_bytes(b"x" * 1000)
        with self.lock:
            self.calls.append((repo_id, revision, tuple(allow_patterns), start, time.time()))


@contextmanager
def fake_hub(hub):
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = types.SimpleNamespace(snapshot_download=hub.snapshot_download)
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub")
        else:
            sys.modules["huggingface_hub"] = saved


def test_both_repos_download_at_once_then_skip_when_on_disk():
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        ns = load(root)
        hub, lines = FakeHub(), []
        with fake_hub(hub):
            futures = ns["_start_weight_downloads"](lines.append)
            secs = {k: f.result() for k, f in futures.items()}
        assert set(futures) == {"avatar", "base"} and all(s > 0 for s in secs.values())
        (a_start, a_end), (b_start, b_end) = [(c[3], c[4]) for c in hub.calls]
        assert max(a_start, b_start) < min(a_end, b_end), "the two repos should overlap"
        assert {c[0] for c in hub.calls} == {ns["AVATAR_REPO"], ns["BASE_REPO"]}
        assert any(c[2] == tuple(ns["BASE_FILES"]) for c in hub.calls)  # only the files we load
        assert (ns["AVATAR_DIR"] / f".complete-{ns['AVATAR_REVISION']}").exists()
        assert any("both repos at once" in line for line in lines) and "GB free" in lines[0]

        hub2 = FakeHub()
        with fake_hub(hub2):
            again = {k: f.result() for k, f in ns["_start_weight_downloads"](lines.append).items()}
        assert hub2.calls == [] and again == {"avatar": 0.0, "base": 0.0}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_failed_download_surfaces_when_waited_on():
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        ns = load(root)

        def broken(**kwargs):
            raise OSError("network down")

        with fake_hub(types.SimpleNamespace(snapshot_download=broken)):
            futures = ns["_start_weight_downloads"](lambda line: None)
            try:
                futures["base"].result()
                raise AssertionError("expected the download error")
            except OSError as exc:
                assert "network down" in str(exc)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_boot_report_is_printed_once_with_stage_times():
    ns = load(pathlib.Path(tempfile.gettempdir()))
    boot = ns["_BootLog"]()
    boot.log("[setup] note one")
    time.sleep(0.05)
    boot.mark("stage A")
    boot.mark("stage B")
    assert boot.pending
    lines = boot.report()
    assert not boot.pending
    assert lines[0].startswith("[setup] this machine's boot (printed once): setup ")
    assert "stage A 0s" in lines[0] and "stage B 0s" in lines[0]
    assert lines[1:] == ["[setup] note one"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
