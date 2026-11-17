"""CPU tests for finding and importing Hugging Face kernel-hub builds (FA3, SageAttention).

The hub changed its layout between builds: FA3 for torch 2.6 keeps the package in
build/<variant>/flash_attn3/, while the torch 2.8 builds put it straight in
build/<variant>/ with flash_attn3/ only a redirect. On torch 2.8 the old loader looked in
the wrong place and silently fell back to FlashAttention 2.
    python -m pytest tests/  or  python tests/test_hub_kernels.py
"""
import ast
import pathlib
import re
import sys
import tempfile

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"re": re, "sys": sys, "FsPath": pathlib.Path}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in {"_find_hub_package", "_import_hub_package"}:
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
find_pkg, import_pkg = _ns["_find_hub_package"], _ns["_import_hub_package"]
REL = "build/torch28-cxx11-cu128-x86_64-linux"


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_package(pkg, lib):
    write(pkg / "__init__.py", "from .interface import flash_attn_func\n")
    write(pkg / "interface.py", "def flash_attn_func():\n    return 'ok'\n")
    write(pkg / "_ops.py", f"import torch\nfrom . import {lib}\nops = torch.ops.{lib}\n")
    write(pkg / f"{lib}.abi3.so")


def test_new_layout_package_at_the_variant_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        make_package(root / REL, "_flash_attn3_5630d32")
        write(root / REL / "flash_attn3" / "__init__.py", "# redirect to the parent package\n")
        assert find_pkg(root, REL, "_flash_attn3_") == root / REL


def test_old_layout_package_in_a_subfolder():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        make_package(root / REL / "flash_attn3", "_flash_attn3_557701f")
        assert find_pkg(root, REL, "_flash_attn3_") == root / REL / "flash_attn3"


def test_incomplete_or_missing_builds_are_none():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        assert find_pkg(root, REL, "_flash_attn3_") is None
        make_package(root / REL, "_flash_attn3_5630d32")
        (root / REL / "_flash_attn3_5630d32.abi3.so").unlink()  # .py files only (download cut short)
        assert find_pkg(root, REL, "_flash_attn3_") is None
        # another kernel's extension doesn't count
        assert find_pkg(root, REL, "_sage_attention_") is None


def test_import_a_folder_with_dashes_and_relative_imports():
    with tempfile.TemporaryDirectory() as tmp:
        pkg = pathlib.Path(tmp) / REL
        make_package(pkg, "_flash_attn3_5630d32")
        (pkg / "_ops.py").unlink()  # the stub has no real extension to load
        module = import_pkg(pkg, "fa3_hub_under_test")
        assert module.flash_attn_func() == "ok"
        assert "fa3_hub_under_test.interface" in sys.modules


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
