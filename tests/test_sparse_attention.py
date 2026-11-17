"""CPU tests for `_SparseAttention`, the router behind the experimental `sparse_attention`
input: which LongCat attention calls go to the block-sparse kernel and which stay dense.

LongCat (@58eb336) calls `flash_attn_bsa_3d(q, k, v, latent_shape_q, latent_shape_k,
**bsa_params)` from the avatar attention whenever `enable_bsa` is set: for the big video
attention (queries: 20-23 latent frames, keys: 24-25) and for the reference / carried
frames' own attention (1-5 frames). The kernel needs every (T, H, W) to tile into blocks.
    python -m pytest tests/  or  python tests/test_sparse_attention.py
"""
import ast
import pathlib
import types

SRC = pathlib.Path(__file__).resolve().parents[1] / "predict.py"
_ns = {}
for node in ast.parse(SRC.read_text()).body:
    if isinstance(node, ast.ClassDef) and node.name == "_SparseAttention":
        _ns_node = node
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"SPARSE_CHUNK", "SPARSE_MIN_QUERY_FRAMES", "TEXT_MODE_SIZES_SPARSE"}
        for t in node.targets
    ):
        exec(compile(ast.Module([node], []), str(SRC), "exec"), _ns)
exec(compile(ast.Module([_ns_node], []), str(SRC), "exec"), _ns)
SparseAttention = _ns["_SparseAttention"]
SPARSE_CHUNK, TEXT_MODE_SIZES_SPARSE = _ns["SPARSE_CHUNK"], _ns["TEXT_MODE_SIZES_SPARSE"]


def make(available=True):
    calls = []
    dit = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(enable_bsa=False, bsa_params=None)) for _ in range(3)])
    bsa = lambda q, k, v, sq, sk, **kw: calls.append(("sparse", sq, sk, kw)) or "sparse"
    dense = lambda q, k, v: calls.append(("dense",)) or "dense"
    router = SparseAttention(dit, bsa, dense)
    router.available = available
    return router, dit, calls


def latent(h_px, w_px, frames):
    return (frames, h_px // 16, w_px // 16)


def test_video_attention_goes_sparse_small_attentions_stay_dense():
    r, _, calls = make()
    r.set(0.875)
    # 512x768 (the demo picture): continuation noise->all, first chunk noise->all
    assert r.attend(None, None, None, latent(512, 768, 20), latent(512, 768, 25), sparsity=0.875) == "sparse"
    assert r.attend(None, None, None, latent(512, 768, 23), latent(512, 768, 24), sparsity=0.875) == "sparse"
    # reference frame and carried frames attend among themselves: dense
    assert r.attend(None, None, None, latent(512, 768, 1), latent(512, 768, 1), sparsity=0.875) == "dense"
    assert r.attend(None, None, None, latent(512, 768, 4), latent(512, 768, 4), sparsity=0.875) == "dense"
    assert r.calls == {"sparse": 2, "dense": 2}
    _, sq, sk, kw = calls[0]
    assert kw == {"sparsity": 0.875, "chunk_3d_shape_q": list(SPARSE_CHUNK), "chunk_3d_shape_k": list(SPARSE_CHUNK)}


def test_sizes_that_do_not_tile_fall_back_to_dense():
    r, _, _ = make()
    r.set(0.75)
    # 480x832 (LongCat's usual 16:9 at 480p): 30x52 latent tokens do not split into 8x8 blocks
    assert r.attend(None, None, None, latent(480, 832, 20), latent(480, 832, 25), sparsity=0.75) == "dense"
    # 960x960 (720p square): 60x60
    assert r.attend(None, None, None, latent(960, 960, 20), latent(960, 960, 25), sparsity=0.75) == "dense"


def test_text_mode_sizes_tile_into_blocks():
    for res, sizes in TEXT_MODE_SIZES_SPARSE.items():
        for aspect, (h, w) in sizes.items():
            assert SparseAttention.fits(latent(h, w, 20)), (res, aspect, h, w)


def test_set_toggles_every_block_and_respects_availability():
    r, dit, _ = make()
    assert r.set(0.875) is True
    assert all(b.attn.enable_bsa and b.attn.bsa_params == {"sparsity": 0.875} for b in dit.blocks)
    assert r.set(0.0) is False
    assert all(not b.attn.enable_bsa and b.attn.bsa_params is None for b in dit.blocks)
    r2, dit2, _ = make(available=False)
    assert r2.set(0.875) is False and r2.sparsity == 0.0
    assert all(not b.attn.enable_bsa for b in dit2.blocks)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
