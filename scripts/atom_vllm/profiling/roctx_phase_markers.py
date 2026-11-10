"""Emit roctx phase markers around each vLLM forward pass so rocm-trace-lite (RTL) records,
directly in the rtl trace, when every pass begins and whether it is prefill / decode / mixed.

Why: vLLM's host `execute_context...` annotation is only emitted when the torch profiler is active
(gpu_worker.py::annotate_profile is gated on self.profiler) and is Kineto-only, so it never reaches
the rtl (rocpd) trace. RTL is rocprofiler-sdk based and captures roctx ranges as rocpd_op rows with
gpuId = -1, so a roctx range pushed around each `Worker.execute_model` step lands in the same SQLite
DB as the GPU kernels and brackets them.

The label is built with vLLM's OWN `vllm.v1.utils.compute_iteration_details`, so it is byte-identical
to vLLM's native annotation:
    execute_context_<n_ctx_reqs>(<ctx_tokens>)_generation_<n_gen_reqs>(<gen_tokens>)
Downstream classification: prefill = gen_tokens==0, decode = ctx_tokens==0, mixed = both>0.

Activated only when VLLM_ROCTX_PHASE_MARKERS=1 (set by run_vllm.py for the profiling pass). The
patch installs via a sys.meta_path hook so it applies in every interpreter that imports the vLLM
worker -- including the spawned EngineCore/Worker processes (sitecustomize re-runs there).
"""
import functools
import importlib.abc
import importlib.util
import sys

_TARGET = "vllm.v1.worker.gpu_worker"


def _load_roctx():
    """Return (push, pop) roctx functions from RTL's globally-visible shim, or (None, None).

    RTL (`rtl trace`) LD_PRELOADs its own roctx shim, so the roctx symbols resolve from the global
    symbol table -- ctypes.CDLL(None). ONLY ranges pushed through these global symbols are captured
    by RTL; pushing through an explicit librocprofiler-sdk-roctx.so or libroctx64.so is NOT
    intercepted (verified empirically). Outside `rtl trace` the symbol is absent -> markers disabled.
    """
    import ctypes
    lib = ctypes.CDLL(None)
    if not hasattr(lib, "roctxRangePushA"):
        return None, None
    push = lib.roctxRangePushA
    push.argtypes = [ctypes.c_char_p]
    push.restype = ctypes.c_int
    pop = lib.roctxRangePop
    pop.restype = ctypes.c_int
    return push, pop


def _phase_label(scheduler_output):
    """Build the SAME label vLLM's annotate_profile emits, via vLLM's own compute_iteration_details
    (a context request = one that has NOT yet generated a token), so the rtl marker is byte-identical
    to vLLM's native annotation. Falls back to a scheduled-token heuristic only if that helper is
    unavailable (approximate near chunked-prefill boundaries; not the canonical label)."""
    try:
        from vllm.v1.utils import compute_iteration_details
        d = compute_iteration_details(scheduler_output)
        return (f"execute_context_{d.num_ctx_requests}({d.num_ctx_tokens})"
                f"_generation_{d.num_generation_requests}({d.num_generation_tokens})")
    except Exception:
        counts = scheduler_output.num_scheduled_tokens.values()
        ctx = [t for t in counts if t > 1]
        gen = [t for t in counts if t == 1]
        return f"execute_context_{len(ctx)}({sum(ctx)})_generation_{len(gen)}({sum(gen)})"


def _wrap_execute_model(Worker):
    orig = Worker.execute_model
    if getattr(orig, "_roctx_wrapped", False):
        return
    push, pop = _load_roctx()
    if push is None:
        print("[roctx_phase_markers] roctx shim not visible; markers disabled", file=sys.stderr)
        return
    import torch

    @functools.wraps(orig)
    def execute_model(self, scheduler_output, *args, **kwargs):
        active = bool(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
        if active:
            try:
                push(_phase_label(scheduler_output).encode())
            except Exception:
                active = False
        try:
            return orig(self, scheduler_output, *args, **kwargs)
        finally:
            if active:
                try:                       # sync so this pass's GPU kernels fall inside the window
                    torch.cuda.synchronize()
                except Exception:
                    pass
                pop()

    execute_model._roctx_wrapped = True
    Worker.execute_model = execute_model
    print("[roctx_phase_markers] installed on vllm Worker.execute_model", file=sys.stderr)


class _LoaderWrapper(importlib.abc.Loader):
    """Delegates to the real loader, then patches Worker.execute_model after the module executes."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _wrap_execute_model(module.Worker)
        except Exception as exc:           # never break the import
            print(f"[roctx_phase_markers] patch failed: {exc}", file=sys.stderr)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Patches the vLLM worker module right after it is first imported (in any process)."""

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)         # let the real finders resolve it now
        spec = importlib.util.find_spec(name)
        if spec is not None:
            spec.loader = _LoaderWrapper(spec.loader)
        return spec


def install():
    """Idempotently arrange for the worker patch to apply on import (or now, if already imported)."""
    if _TARGET in sys.modules:
        try:
            _wrap_execute_model(sys.modules[_TARGET].Worker)
        except Exception as exc:
            print(f"[roctx_phase_markers] patch failed: {exc}", file=sys.stderr)
    elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
