# debug-numeric: SGLang's CUDA / HIP graphs, from the environment (no model edit) -- the SGLang counterpart of
# scripts/vllm_graphs.
#
#   PYTHONPATH=<REPO_ROOT>/scripts/sglang_graphs[:<REPO_ROOT>/scripts/sglang_capture]  python3 serve.py ...
#   (or `python3 -m sglang.launch_server ...`: the switch and the check act in the scheduler process either way)
#
# ND_SGLANG_DISABLE_CUDA_GRAPH=1
#     Graphs off, through SGLang's own argument parsing -- disable_cuda_graph=True into sglang.Engine's keyword
#     arguments, --disable-cuda-graph into launch_server's argv (prepare_server_args) -- so decode steps run eagerly
#     and the per-op capture sees them. Not later: resolved server_args are read-only, and 0.5.23 runs graphs from
#     get_exec().graph.cuda_graph_config, resolved at startup (a late override of the legacy flag changed nothing;
#     ServerArgs.__post_init__ is msgspec-cached and cannot be patched -- 2026-11-24). The scheduler logs the
#     resolved decode backend. The capture lines of a decode-only NaN run this way
#     (`step4-run.md`); the controls stay as the user runs the model.
#
# ND_GRAPH_CHECK=1  (+ ND_GRAPH_CHECK_DIR=<dir>, ND_GRAPH_CHECK_MAX=<n checks, default 1>)
#     A NaN that only appears with graphs on. Every forward that SGLang served from a graph
#     (ModelRunner._forward_raw returned can_run_graph=True: a decode-graph replay or a piecewise prefill graph) has
#     its next-token logits checked (one host read per forward: a diagnostic arm, not a timing one). At the first
#     one that is non-finite, the same batch runs again WITHOUT graphs (both graph runners set aside for the call:
#     SGLang's own eager path) on the same forward batch:
#       eager finite      -> GRAPH_ONLY: the graph computes what the eager model does not -- host-side logic
#                            evaluated once at capture, padding to the capture size, the shared graph pool
#       eager non-finite  -> NOT_GRAPH_ONLY: the model itself makes the NaN; the per-op capture with graphs off
#                            (ND_SGLANG_DISABLE_CUDA_GRAPH=1) finds the op
#     and saves <dir>/sglang_graph_<pid>_<n>.pt (forward mode, batch size, both logits, the non-finite rows).
#     Prints `[sglang-graphs] GRAPH_CHECK verdict=<GRAPH_ONLY|NOT_GRAPH_ONLY> mode=<m> bs=<n> rows=<r> of <n> ->
#     <file>`. Re-running the batch writes its KV-cache slots again (the eager values): the run after a check is
#     not the user's run any more -- one check per process by default.
#
# Validated with SGLang 0.5.23 (2026-11-24): the patch points are
# sglang.srt.model_executor.model_runner.ModelRunner.__init__ / _forward_raw. Prints `[sglang-graphs] ...` lines.
# A sitecustomize later on sys.path is still run.
import importlib.abc
import importlib.util
import os
import sys

_OFF = os.environ.get("ND_SGLANG_DISABLE_CUDA_GRAPH", "") == "1"
_CHECK = os.environ.get("ND_GRAPH_CHECK", "0") == "1"
_TARGET = "sglang.srt.model_executor.model_runner"
_ENGINE = "sglang.srt.entrypoints.engine"
_SARGS = "sglang.srt.server_args"
_STATE = {"checks": 0, "forwards": 0}


def _say(msg):
    print("[sglang-graphs] " + msg, flush=True)


def _patch(mod):
    MR = mod.ModelRunner
    if _OFF:
        oinit = MR.__init__

        def __init__(self, *a, **kw):
            r = oinit(self, *a, **kw)
            try:                                        # evidence: the resolved decode-graph backend
                be = mod.get_exec().graph.cuda_graph_config.decode.backend
            except Exception:                           # noqa: BLE001 - another SGLang version
                be = "?"
            _say("graphs: decode backend %s, decode graph runner %s (pid %d)"
                 % (be, type(getattr(self, "decode_cuda_graph_runner", None)).__name__, os.getpid()))
            return r
        MR.__init__ = __init__
    if _CHECK:
        oraw = MR._forward_raw
        out_dir = os.environ.get("ND_GRAPH_CHECK_DIR") or os.getcwd()
        limit = int(os.environ.get("ND_GRAPH_CHECK_MAX", "1"))

        def _forward_raw(self, forward_batch, *a, **kw):
            import torch
            out = oraw(self, forward_batch, *a, **kw)
            _STATE["forwards"] += 1
            if not getattr(out, "can_run_graph", False) or _STATE["checks"] >= limit:
                return out
            lo = getattr(out, "logits_output", None)
            t = getattr(lo, "next_token_logits", None)
            if t is None or bool(torch.isfinite(t).all()):
                return out
            _STATE["checks"] += 1
            saved = (self.decode_cuda_graph_runner, getattr(self, "prefill_cuda_graph_runner", None))
            self.decode_cuda_graph_runner = None
            if hasattr(self, "prefill_cuda_graph_runner"):
                self.prefill_cuda_graph_runner = None
            try:
                out2 = oraw(self, forward_batch, *a, **kw)
            finally:
                self.decode_cuda_graph_runner = saved[0]
                if hasattr(self, "prefill_cuda_graph_runner"):
                    self.prefill_cuda_graph_runner = saved[1]
            t2 = getattr(getattr(out2, "logits_output", None), "next_token_logits", None)
            eager_finite = t2 is not None and bool(torch.isfinite(t2).all())
            verdict = "GRAPH_ONLY" if eager_finite else "NOT_GRAPH_ONLY"
            bad = (~torch.isfinite(t.float())).any(-1).nonzero().flatten().tolist()
            mode = str(getattr(forward_batch, "forward_mode", "?")).split(".")[-1]
            os.makedirs(out_dir, exist_ok=True)
            f = os.path.join(out_dir, "sglang_graph_%d_%d.pt" % (os.getpid(), _STATE["checks"]))
            torch.save({"verdict": verdict, "mode": mode, "batch_size": int(t.shape[0]), "rows": bad,
                        "graph_logits": t.detach().float().cpu(),
                        "eager_logits": None if t2 is None else t2.detach().float().cpu(),
                        "forward": _STATE["forwards"], "pid": os.getpid()}, f)
            _say("GRAPH_CHECK verdict=%s mode=%s bs=%d rows=%s of %d -> %s"
                 % (verdict, mode, t.shape[0], bad[:16], t.shape[0], f))
            return out
        MR._forward_raw = _forward_raw
    _say("patched %s.ModelRunner (%s) in pid %d" % (
        _TARGET, ", ".join(x for x, on in (("graphs off", _OFF), ("graph check", _CHECK)) if on), os.getpid()))


def _patch_engine(mod):
    Engine = mod.Engine
    oinit = Engine.__init__

    def __init__(self, *a, **kw):
        if _OFF and not kw.get("disable_cuda_graph"):
            kw["disable_cuda_graph"] = True
            _say("switch: disable_cuda_graph=True into sglang.Engine (ND_SGLANG_DISABLE_CUDA_GRAPH) in pid %d"
                 % os.getpid())
        return oinit(self, *a, **kw)
    Engine.__init__ = __init__


def _patch_server_args(mod):
    orig = mod.prepare_server_args

    def prepare_server_args(argv, *a, **kw):
        argv = list(argv)
        if _OFF and "--disable-cuda-graph" not in argv:
            argv.append("--disable-cuda-graph")
            _say("switch: --disable-cuda-graph into launch_server's arguments in pid %d" % os.getpid())
        return orig(argv, *a, **kw)
    mod.prepare_server_args = prepare_server_args


class _Finder(importlib.abc.MetaPathFinder):
    """Patches the model-runner (and, for graphs off, the engine / server-args) module right after it executes."""

    def find_spec(self, name, path, target=None):
        if name not in (_TARGET, _ENGINE, _SARGS) or (name == _TARGET and not (_OFF or _CHECK)) or \
                (name in (_ENGINE, _SARGS) and not _OFF):
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            {_TARGET: _patch, _ENGINE: _patch_engine, _SARGS: _patch_server_args}[name](module)
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
            name = "_sitecustomize_next_%x" % (hash(os.path.abspath(f)) & 0xffffffff)
            spec = importlib.util.spec_from_file_location(name, f)
            m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m
            spec.loader.exec_module(m)
            return


if _OFF or _CHECK:
    sys.meta_path.insert(0, _Finder())
_chain()
