"""CPU tests for `_MixAwareKVCache`, the wrapper that lets LongCat's carried-frame KV cache
run with the DMD/OPSD LoRA mix on.

The fakes mirror the parts of LongCat (@58eb336) the wrapper relies on:
  * `generate_avc` calls `self._cache_clean_latents(...)` once, takes the dict from
    `self._get_kv_cache_dict()` before the loop, and calls `self.dit.set_lora_mix_step(i)`
    at the top of every step, reading `kv_cache_dict.get(block)` inside the DiT;
  * `set_lora_mix_step` sets `_lora_mix_active = step < mix_steps` (no-op without a mix);
  * the cache a build produces reflects the LoRA state the DiT is in at that moment.
    python -m pytest tests/  or  python tests/test_kv_cache_mix.py
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.ClassDef) and node.name == "_MixAwareKVCache":
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
MixAwareKVCache = _ns["_MixAwareKVCache"]


class FakeDit:
    def __init__(self, mix_steps):
        self.lora_mix_config = None if mix_steps is None else {"mix_steps": mix_steps}
        self._lora_mix_active = None
        self.cache_passes = 0

    def set_lora_mix_step(self, step_index):
        if self.lora_mix_config is None:
            return
        self._lora_mix_active = step_index < self.lora_mix_config["mix_steps"]


class FakePipe:
    def __init__(self, dit):
        self.dit = dit
        self.kv_cache_dict = None

    def _cache_clean_latents(self, cond_latents, *rest, **kwargs):
        self.dit.cache_passes += 1
        state = self.dit._lora_mix_active
        self.kv_cache_dict = {b: (cond_latents, state) for b in range(3)}

    def _get_kv_cache_dict(self):
        return self.kv_cache_dict

    def generate_avc(self, cond_latents, steps=8):
        self._cache_clean_latents(cond_latents, 512, offload_kv_cache=False)
        kv = self._get_kv_cache_dict()
        used = []
        for i in range(steps):
            if self.dit.lora_mix_config is not None:
                self.dit.set_lora_mix_step(i)
            used.append((self.dit._lora_mix_active, [kv.get(b)[1] for b in range(3)], kv.get(0)[0]))
        return used


def run_segments(mix_steps, segments=3):
    dit = FakeDit(mix_steps)
    pipe = FakePipe(dit)
    cache = MixAwareKVCache(pipe)
    dit.set_lora_mix_step(0)  # what enable_lora_mix does
    dit.set_lora_mix_step(7)  # the previous segment (or stage 1) ended on the last step
    out = []
    for seg in range(segments):
        before = cache.builds
        used = pipe.generate_avc(cond_latents=f"seg{seg}")
        builds = cache.builds - before
        cache.release()
        out.append((used, builds))
    return dit, pipe, cache, out


def test_cache_always_matches_the_active_lora_state():
    for mix_steps in (None, 0, 1, 4, 7, 8):
        _, _, _, out = run_segments(mix_steps)
        for seg, (used, _) in enumerate(out):
            for step, (active, cached_states, latents) in enumerate(used):
                assert all(c == active for c in cached_states), (mix_steps, seg, step, active, cached_states)
                assert latents == f"seg{seg}", "a segment must never see the previous segment's cache"


def test_builds_once_per_lora_state():
    expected = {None: 1, 0: 1, 1: 2, 4: 2, 7: 2, 8: 1}
    for mix_steps, n in expected.items():
        dit, _, _, out = run_segments(mix_steps)
        assert [b for _, b in out] == [n] * 3, (mix_steps, [b for _, b in out])
        assert dit.cache_passes == 3 * n


def test_release_frees_the_cache():
    _, pipe, cache, _ = run_segments(4, segments=1)
    assert cache.live == {} and pipe.kv_cache_dict is None
    # A LoRA switch after release (e.g. enable_lora_mix on the next request) must not rebuild.
    before = cache.builds
    pipe.dit.set_lora_mix_step(0)
    pipe.dit.set_lora_mix_step(7)
    assert cache.builds == before


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
