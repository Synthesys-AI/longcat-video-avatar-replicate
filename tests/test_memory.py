"""CPU tests for `_offloadable_text_encoder`: the text encoder leaves the GPU after encoding
while a request asks for it (720p), is back on the GPU whenever it actually encodes, and is
never moved on a prompt-cache hit unless it has to leave. A recording stand-in replaces the
real encoder (no GPU needed); `FakePipe.encode_prompt` copies LongCat's prompt cache.
    python -m pytest tests/  or  python tests/test_memory.py
"""
import ast
import pathlib
import time
import types

import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {"torch": torch, "time": time}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "_offloadable_text_encoder":
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
offloadable = _ns["_offloadable_text_encoder"]
GPU, CPU = torch.device("cuda", 0), torch.device("cpu")


class FakeEncoder:
    def __init__(self):
        self.device, self.moves = GPU, []

    def parameters(self):
        yield types.SimpleNamespace(device=self.device)

    def to(self, target):
        self.moves.append(str(target))
        self.device = torch.device(target)
        return self


class FakePipe:
    def __init__(self):
        self.text_encoder, self.encoded = FakeEncoder(), []

    def _get_t5_prompt_embeds(self, prompt):
        assert self.text_encoder.device == GPU, "must encode on the GPU"
        self.encoded.append(prompt)
        return f"emb({prompt})"

    def encode_prompt(self, prompt):  # LongCat: return the cached result for the same key
        if getattr(self, "_prompt_cache_key", None) == prompt:
            return self._prompt_cache
        self._prompt_cache_key, self._prompt_cache = prompt, self._get_t5_prompt_embeds(prompt)
        return self._prompt_cache


def test_offload_only_while_requested_and_back_when_encoding():
    pipe = FakePipe()
    state = offloadable(pipe, 0)
    te = pipe.text_encoder

    assert pipe.encode_prompt("a") == "emb(a)" and te.moves == []        # 480p: stays put
    state["offload"] = True                                             # a 720p request
    assert pipe.encode_prompt("a") == "emb(a)"                          # cache hit, still leaves
    assert te.device == CPU and te.moves == ["cpu"]
    pipe.encode_prompt("a")                                             # later segment: no move
    assert te.moves == ["cpu"]
    pipe.encode_prompt("b")                                             # new prompt at 720p
    assert pipe.encoded == ["a", "b"] and te.moves == ["cpu", "cuda:0", "cpu"]

    state["offload"] = False                                            # back to 480p
    pipe.encode_prompt("b")                                             # cached: no need to move
    assert te.device == CPU and len(te.moves) == 3
    pipe.encode_prompt("c")                                             # encodes: back on the GPU
    assert te.device == GPU and te.moves[-1] == "cuda:0" and len(te.moves) == 4
    assert [t for t, _ in state["moves"]] == ["cpu", "cuda:0", "cpu", "cuda:0"]


def test_offload_happens_even_if_encoding_raises():
    pipe = FakePipe()

    def boom(prompt):
        raise RuntimeError("tokenizer failed")

    pipe._get_t5_prompt_embeds = boom  # captured by the wrapper as the inner call
    state = offloadable(pipe, 0)
    state["offload"] = True
    try:
        pipe.encode_prompt("x")
        raise AssertionError("expected the error")
    except RuntimeError:
        pass
    assert pipe.text_encoder.device == CPU


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
