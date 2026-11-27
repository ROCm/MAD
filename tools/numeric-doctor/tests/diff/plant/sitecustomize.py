# Test only (tests/diff T3/T4): a planted SILENT bug in a real serving stack -- the default attention backend
# swaps the outputs of two heads in one layer (finite, plausible values; no NaN). Switching the attention backend
# removes it, so differential mode must find it (DIVERGE), the bisection must name `attention`, and per-module
# localisation must name the attention module of that layer.
#
#   PYTHONPATH=<REPO_ROOT>/tests/diff/plant:...  ND_PLANT=vllm_attn|sglang_attn [ND_PLANT_LAYER=3]
#
# vllm_attn    vllm.v1.attention.backends.rocm_attn.RocmAttentionImpl.forward (vLLM 0.33's default on gfx950): the
#              impl of model layer ND_PLANT_LAYER (from the layer's name) swaps heads 0 and 1 of `output`
# sglang_attn  sglang.srt.layers.attention.aiter_backend.AiterAttnBackend.forward_extend / forward_decode:
#              layer.layer_id == ND_PLANT_LAYER swaps heads 0 and 1 of the returned output
# Coverage tests (tests/coverage, #4) -- NaN plants, same patch points:
# vllm_attn_nan      the ND_PLANT_LAYER-th RocmAttentionImpl writes NaN into row 0 of its output, every call (a
#                    pipeline stage's own NaN: the layer decides the stage)
# vllm_layer_sqrt    vLLM's Qwen2DecoderLayer of model layer ND_PLANT_LAYER: row 0 of its hidden output := sqrt(x - 1e6)
#                    -- an aten.sqrt that MAKES the NaN from finite inputs, replayable without vLLM (eager runs:
#                    under torch.compile the layer is traced into Inductor code). vllm_layer_sqrt_: the same in
#                    place (sub_ / sqrt_ on the layer's output), so the op that makes the NaN is the surviving write
# sglang_graph_only  AiterAttnBackend.forward_decode, layer ND_PLANT_LAYER: row 0 := NaN ONLY while a graph is being
#                    recorded -- every replay of that graph is wrong, the eager decode is not (GRAPH_ONLY)
# sglang_decode_nan  the same, every decode call, graph or not (NOT_GRAPH_ONLY; a graphs-off capture sees it)
# Tensor ops only (no host read): the corruption is captured into HIP graphs like the kernel it stands for.
# Prints `[plant] ...`. Chains to the next sitecustomize on sys.path.
import importlib.abc
import importlib.util
import os
import re
import sys

_KIND = os.environ.get("ND_PLANT", "")
_LAYER = int(os.environ.get("ND_PLANT_LAYER", "3"))


def _say(msg):
    print("[plant] " + msg, flush=True)


_FIRED = set()


def _fired(where):
    if where not in _FIRED:
        _FIRED.add(where)
        _say("fired: %s (pid %d)" % (where, os.getpid()))


def _swap(o, heads):
    import torch
    # split views and copy_ only: under vLLM 0.33's graph capture any Python index (v[:, [0, 1]], even v[:, 0])
    # made a host tensor and failed the capture (2026-11-24)
    h0, h1, _ = torch.split(o.view(o.shape[0], heads, -1), [1, 1, heads - 2], dim=1)
    t = h0.clone()
    h0.copy_(h1)
    h1.copy_(t)


def _nan_row0(o):
    o[:1].fill_(float("nan"))


def _patch_vllm(mod):
    impl = mod.RocmAttentionImpl
    orig = impl.forward
    seen = {}

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **kw):
        out = orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **kw)
        # the model's layer index from the layer's name (model.layers.<n>.self_attn.attn); first-seen order counts per
        # process, which under pipeline parallelism is per stage (stage 1's first layer is its "0") -- the fallback
        m = re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "")))
        n = int(m.group(1)) if m else seen.setdefault(id(self), len(seen))
        if n == _LAYER and attn_metadata is not None:
            if _KIND == "vllm_attn_nan":
                _nan_row0(output[: query.shape[0]])
            else:
                _swap(output[: query.shape[0]], self.num_heads)
            _fired("vllm RocmAttentionImpl layer %d (%s)" % (n, _KIND))
        return out

    impl.forward = forward
    _say("vllm_attn planted: RocmAttentionImpl instance %d swaps heads 0/1 (pid %d)" % (_LAYER, os.getpid()))


def _patch_vllm_layer(mod):
    import torch
    cls = mod.Qwen2DecoderLayer
    orig = cls.forward

    def forward(self, *a, **kw):
        out = orig(self, *a, **kw)
        m = re.search(r"layers\.(\d+)\.", str(getattr(getattr(self.self_attn, "attn", None), "layer_name", "")))
        if m and int(m.group(1)) == _LAYER:
            h = out[0] if isinstance(out, tuple) else out
            if _KIND == "vllm_layer_sqrt_":          # in place: the NaN-making op writes the layer's own output
                h[:1].sub_(1e6).sqrt_()
            else:
                bad = torch.sqrt(h[:1].float() - 1e6).to(h.dtype)  # finite in, NaN out: aten.sqrt makes it
                h = torch.cat([bad, h[1:]], 0)
                out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            _fired("vllm Qwen2DecoderLayer layer %d (%s)" % (_LAYER, _KIND))
        return out
    cls.forward = forward
    _say("vllm_layer_sqrt planted: Qwen2DecoderLayer layer %d (pid %d)" % (_LAYER, os.getpid()))


def _patch_sglang(mod):
    cls = mod.AiterAttnBackend
    for name in ("forward_extend", "forward_decode"):
        orig = getattr(cls, name)

        def wrap(self, q, k, v, layer, *a, _orig=orig, **kw):
            o = _orig(self, q, k, v, layer, *a, **kw)
            if getattr(layer, "layer_id", -1) == _LAYER:
                _swap(o, layer.tp_q_head_num)
                _fired("sglang %s layer %d" % (_orig.__name__, _LAYER))
            return o
        setattr(cls, name, wrap)
    _say("sglang_attn planted: AiterAttnBackend layer %d swaps heads 0/1 (pid %d)" % (_LAYER, os.getpid()))


def _patch_sglang_nan(mod):
    import torch
    cls = mod.AiterAttnBackend
    orig = cls.forward_decode

    def forward_decode(self, q, k, v, layer, *a, **kw):
        o = orig(self, q, k, v, layer, *a, **kw)
        if getattr(layer, "layer_id", -1) == _LAYER:
            if _KIND == "sglang_decode_nan" or torch.cuda.is_current_stream_capturing():
                _nan_row0(o)
                _fired("%s layer %d (capturing=%s)" % (_KIND, _LAYER, torch.cuda.is_current_stream_capturing()))
        return o
    cls.forward_decode = forward_decode
    _say("%s planted: AiterAttnBackend.forward_decode layer %d (pid %d)" % (_KIND, _LAYER, os.getpid()))


_TARGETS = {}
if _KIND in ("vllm_attn", "vllm_attn_nan"):
    _TARGETS["vllm.v1.attention.backends.rocm_attn"] = _patch_vllm
elif _KIND in ("vllm_layer_sqrt", "vllm_layer_sqrt_"):
    _TARGETS["vllm.model_executor.models.qwen2"] = _patch_vllm_layer
elif _KIND == "sglang_attn":
    _TARGETS["sglang.srt.layers.attention.aiter_backend"] = _patch_sglang
elif _KIND in ("sglang_graph_only", "sglang_decode_nan"):
    _TARGETS["sglang.srt.layers.attention.aiter_backend"] = _patch_sglang_nan


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module, patch = spec.loader.exec_module, _TARGETS[name]

        def exec_and_patch(module):
            exec_module(module)
            patch(module)
        spec.loader.exec_module = exec_and_patch
        return spec


def _chain():
    here = os.path.dirname(os.path.abspath(__file__))
    seen = False
    for d in sys.path:
        if os.path.abspath(d or ".") == here:
            seen = True
            continue
        f = os.path.join(d or ".", "sitecustomize.py")
        if seen and os.path.isfile(f):
            name = "_sitecustomize_next_%x" % (hash(os.path.abspath(f)) & 0xffffffff)
            spec = importlib.util.spec_from_file_location(name, f)
            m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m
            spec.loader.exec_module(m)
            return


if _TARGETS:
    sys.meta_path.insert(0, _Finder())
_chain()
