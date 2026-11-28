# debug-numeric: vLLM's CUDA / HIP graphs, from the environment (no model edit).
#
#   PYTHONPATH=<REPO_ROOT>/scripts/vllm_graphs[:<REPO_ROOT>/scripts/vllm_capture]  python3 serve.py ...
#
# ND_VLLM_CUDAGRAPH_MODE=<NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE>
#     Overrides compilation_config.cudagraph_mode in every EngineArgs (vLLM has no env switch for it). NONE keeps
#     torch.compile and turns graph replay off: the per-op capture then sees the compiled pieces' kernels
#     (op_nan_capture "Inductor launches"), which a replayed graph launches without Python.  The capture arms of a
#     compiled-NaN investigation run this way (step4-run.md *Compiled NaN predicted*).
#
# ND_GRAPH_CHECK=1  (+ ND_GRAPH_CHECK_DIR=<dir>, ND_GRAPH_CHECK_MAX=<n captures, default 1>)
#     A NaN that only appears with graphs on.  Every graph REPLAY of a CUDAGraphWrapper (piecewise pieces, full
#     graphs) has its outputs checked (one host read per replay: a diagnostic arm, not a timing one).  At the first
#     replay whose output is non-finite while its inputs are finite, the same piece is run again WITHOUT the graph
#     (the wrapper's runnable) on the same inputs, inside the same forward context:
#       eager finite      -> GRAPH_ONLY: the graph computes what the piece does not -- host-side logic evaluated
#                            once at capture (a Python branch, a size, a pointer to a host or freed buffer),
#                            padding to the capture size, the shared graph pool
#       eager non-finite  -> NOT_GRAPH_ONLY: the piece itself makes the NaN; the per-op capture with graphs off
#                            (ND_VLLM_CUDAGRAPH_MODE=NONE) or C1 finds the kernel
#     and saves <dir>/graph_piece_<pid>_<n>.pt: the piece's FX code (when vLLM keeps the module: piecewise), its
#     inputs, both outputs, the batch descriptor (the padded capture size), which output rows are non-finite.
#     Prints `[vllm-graphs] GRAPH_CHECK verdict=<GRAPH_ONLY|NOT_GRAPH_ONLY> piece=<i> mode=<m> batch=<descriptor>
#     rows=<non-finite rows> of <n> -> <file>`.  Replay without the model: scripts/replay_graph_piece.py <file>.
#     A replay whose inputs are already non-finite is counted (`propagated`), not captured: the first piece that
#     turns finite inputs into a NaN is the one named.  Re-running a full graph (attention included) writes the
#     same KV-cache slots again with the same values.
#
#     vLLM's V2 GPU runner (vllm/v1/worker/gpu: newer models) replays FULL graphs itself, not
#     through CUDAGraphWrapper: there the check is on ModelCudaGraphManager.run_fullgraph -- the whole step's hidden
#     states (`runner=v2`); the re-run without the graph uses the step's own capture closure on the same static
#     inputs; the capsule is graph_replay_<pid>_<n>.pt (no piece: the model-level A/B is the reproducer).
#
# Validated with vLLM 0.33.0 (vllm/vllm-openai-rocm:v0.33.0-rocm100, 2026-11-21/22); the patch points are
# vllm.engine.arg_utils.EngineArgs.__post_init__, vllm.compilation.cuda_graph.CUDAGraphWrapper.__call__ and
# vllm.v1.worker.gpu.cudagraph_utils.{CudaGraphManager.capture, ModelCudaGraphManager.run_fullgraph}.
# Prints `[vllm-graphs] ...` lines.  A sitecustomize later on sys.path is still run.
import importlib.abc
import importlib.util
import os
import re
import sys

_MODE = os.environ.get("ND_VLLM_CUDAGRAPH_MODE", "").strip().upper()
_CHECK = os.environ.get("ND_GRAPH_CHECK", "0") == "1"
_STATE = {"saved": 0, "replays": 0, "propagated": 0, "pieces": {}}


def _say(msg):
    print("[vllm-graphs] " + msg, flush=True)


def _patch_args(mod):
    ea = mod.EngineArgs
    orig = ea.__post_init__

    def __post_init__(self, *a, **k):
        orig(self, *a, **k)
        from vllm.config.compilation import CUDAGraphMode
        cc = self.compilation_config
        if isinstance(cc, dict):
            cc["cudagraph_mode"] = _MODE
        else:
            cc.cudagraph_mode = CUDAGraphMode[_MODE]
        _say("cudagraph_mode=%s (ND_VLLM_CUDAGRAPH_MODE)" % _MODE)

    ea.__post_init__ = __post_init__


def _tensors(obj):
    import torch
    out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(_tensors(x))
    elif isinstance(obj, dict):
        for x in obj.values():
            out.extend(_tensors(x))
    return out


def _finite(ts):
    import torch
    fl = [t for t in ts if t.is_floating_point() and t.numel()]
    if not fl:
        return True
    return bool(torch.stack([torch.isfinite(t.float()).all() for t in fl]).all().item())


def _cpu(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return dict((k, _cpu(v)) for k, v in obj.items())
    return obj


def _bad_rows(ts):
    import torch
    for t in ts:
        if t.is_floating_point() and t.dim() >= 1 and not bool(torch.isfinite(t.float()).all()):
            r = (~torch.isfinite(t.float())).reshape(t.shape[0], -1).any(-1).nonzero().flatten().tolist()
            return r, t.shape[0]
    return [], 0


def _graph_module(wrapper):
    """The piece's FX GraphModule, when vLLM keeps it (piecewise backend), else None."""
    import torch
    r = wrapper.runnable
    for _ in range(4):
        for attr in ("graph", "gm", "graph_module", "split_gm", "module"):
            g = getattr(r, attr, None)
            if isinstance(g, torch.fx.GraphModule):
                return g
        r = getattr(r, "runnable", None) or getattr(r, "compiled_graph_for_general_shape", None)
        if r is None:
            break
    return None


def _patch_graphs(mod):
    import torch
    W = mod.CUDAGraphWrapper
    orig = W.__call__
    out_dir = os.environ.get("ND_GRAPH_CHECK_DIR") or os.getcwd()
    limit = int(os.environ.get("ND_GRAPH_CHECK_MAX", "1"))
    _say("graph check on: every graph replay's outputs checked; captures -> %s (max %d)" % (out_dir, limit))

    def __call__(self, *args, **kwargs):
        replay, bd, rmode = False, None, None
        try:
            from vllm.forward_context import get_forward_context, is_forward_context_available
            if is_forward_context_available() and not torch.cuda.is_current_stream_capturing():
                fc = get_forward_context()
                bd, rmode = fc.batch_descriptor, fc.cudagraph_runtime_mode
                e = self.concrete_cudagraph_entries.get(bd) if bd is not None else None
                replay = (rmode == self.runtime_mode and e is not None and e.cudagraph is not None)
        except Exception:                               # noqa: BLE001 -- never break the model
            replay = False
        out = orig(self, *args, **kwargs)
        if not replay or _STATE["saved"] >= limit:
            return out
        _STATE["replays"] += 1
        piece = _STATE["pieces"].setdefault(id(self), len(_STATE["pieces"]))
        try:
            outs = _tensors(out)
            if _finite(outs):
                return out
            if not _finite(_tensors((args, kwargs))):
                _STATE["propagated"] += 1
                return out
            graph_out = _cpu(out)
            rows, n = _bad_rows(outs)
            ref = self.runnable(*args, **kwargs)        # the same piece, no graph, same inputs
            eager_out = _cpu(ref)
            eager_finite = _finite(_tensors(ref))
            verdict = "GRAPH_ONLY" if eager_finite else "NOT_GRAPH_ONLY"
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "graph_piece_%d_%d.pt" % (os.getpid(), _STATE["saved"]))
            gm = _graph_module(self)
            cap = {"kind": "graph_piece", "verdict": verdict, "piece": piece,
                   "runtime_mode": str(rmode), "batch_descriptor": repr(bd),
                   "num_tokens": getattr(bd, "num_tokens", None),
                   "bad_rows": rows[:256], "rows": n, "args": _cpu(args), "kwargs": _cpu(kwargs),
                   "graph_out": graph_out, "eager_out": eager_out,
                   "runnable": type(self.runnable).__name__,
                   "fx_code": gm.code if gm is not None else None,
                   "torch": torch.__version__, "device": torch.cuda.get_device_name()}
            try:
                import vllm
                cap["vllm"] = vllm.__version__
            except Exception:                           # noqa: BLE001
                pass
            if gm is not None:
                # The code and the names it reads, not the GraphModule: unpickling one re-traces its code
                # symbolically, which Dynamo's generated pieces do not survive (2026-11-22, _get_data_attr).
                # replay_graph_piece.py executes the code as a plain function.
                cap["fx_globals"] = dict((k, repr(v)[:120]) for k, v in gm.forward.__globals__.items()
                                         if not k.startswith("__"))
                cap["fx_self_attrs"] = sorted(set(re.findall(r"\bself\.(\w+)", gm.code)))
            torch.save(cap, path)
            _STATE["saved"] += 1
            _say("GRAPH_CHECK verdict=%s piece=%d mode=%s batch=%s rows=%s of %d -> %s"
                 % (verdict, piece, rmode, bd, rows[:16], n, path))
        except Exception as e:                          # noqa: BLE001
            _say("graph check failed on piece %d: %s: %s" % (piece, type(e).__name__, str(e)[:200]))
        return out

    W.__call__ = __call__


def _patch_v2(mod):
    """The V2 GPU model runner (vllm/v1/worker/gpu): FULL graphs replayed by
    ModelCudaGraphManager.run_fullgraph(desc), which returns the hidden states (padding rows
    included).  Its capture builds one forward closure per descriptor (create_forward_fn); kept
    here, it re-runs the same step without the graph on the same static input buffers."""
    import torch
    M = getattr(mod, "ModelCudaGraphManager", None)
    C = getattr(mod, "CudaGraphManager", None)
    if M is None or C is None or not hasattr(M, "run_fullgraph"):
        _say("V2 runner: no ModelCudaGraphManager.run_fullgraph here -- not checked")
        return
    out_dir = os.environ.get("ND_GRAPH_CHECK_DIR") or os.getcwd()
    limit = int(os.environ.get("ND_GRAPH_CHECK_MAX", "1"))
    ocap, orun = C.capture, M.run_fullgraph

    def capture(self, create_forward_fn, *a, **k):
        self._nd_create_forward_fn = create_forward_fn
        return ocap(self, create_forward_fn, *a, **k)

    def run_fullgraph(self, desc):
        out = orun(self, desc)
        if _STATE["saved"] >= limit:
            return out
        _STATE["replays"] += 1
        try:
            hs = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)
            if hs is None or _finite([hs]):
                return out
            rows, n = _bad_rows([hs])
            graph_out = hs.detach().clone()
            verdict, eager_out = "GRAPH_REPLAY_NONFINITE", None
            cff = getattr(self, "_nd_create_forward_fn", None)
            if cff is not None:
                from vllm.config.compilation import CUDAGraphMode
                cff(desc, False)(CUDAGraphMode.NONE)        # same step, no graph: writes self.hidden_states
                eager = self.hidden_states[: desc.num_tokens]
                eager_out = eager.detach().cpu().clone()
                verdict = "GRAPH_ONLY" if _finite([eager]) else "NOT_GRAPH_ONLY"
                eager.copy_(graph_out)                      # the model goes on with the graph's output
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "graph_replay_%d_%d.pt" % (os.getpid(), _STATE["saved"]))
            torch.save({"kind": "graph_replay", "runner": "v2", "verdict": verdict, "descriptor": repr(desc),
                        "num_tokens": getattr(desc, "num_tokens", None), "num_reqs": getattr(desc, "num_reqs", None),
                        "bad_rows": rows[:256], "rows": n, "graph_out": graph_out.cpu(), "eager_out": eager_out,
                        "torch": torch.__version__}, path)
            _STATE["saved"] += 1
            _say("GRAPH_CHECK verdict=%s runner=v2 mode=FULL desc=%s rows=%s of %d -> %s"
                 % (verdict, desc, rows[:16], n, path))
        except Exception as e:                          # noqa: BLE001 -- never break the model
            _say("graph check (v2) failed: %s: %s" % (type(e).__name__, str(e)[:200]))
        return out

    C.capture, M.run_fullgraph = capture, run_fullgraph
    _say("graph check on (V2 runner): every FULL graph replay's hidden states checked; captures -> %s" % out_dir)


_TARGETS = {}
if _MODE:
    _TARGETS["vllm.engine.arg_utils"] = _patch_args
if _CHECK:
    _TARGETS["vllm.compilation.cuda_graph"] = _patch_graphs
    _TARGETS["vllm.v1.worker.gpu.cudagraph_utils"] = _patch_v2


class _Finder(importlib.abc.MetaPathFinder):
    """Patches each target module right after it executes, wherever it is first imported."""

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
    """Run the next sitecustomize on sys.path -- one AFTER this directory: this one shadows it, and an
    earlier one (another of ours on the same PYTHONPATH) already ran and chained here."""
    here = os.path.dirname(os.path.abspath(__file__))
    seen = False
    for d in sys.path:
        if os.path.abspath(d or ".") == here:
            seen = True
            continue
        f = os.path.join(d or ".", "sitecustomize.py")
        if seen and os.path.isfile(f):
            # registered in sys.modules: torch.compile re-imports the module of any function it traces
            # (a patch defined there), and an unregistered one is ModuleNotFoundError (2026-11-22)
            name = "_sitecustomize_next_%x" % (hash(os.path.abspath(f)) & 0xffffffff)
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return


if _TARGETS:
    sys.meta_path.insert(0, _Finder())
_chain()
