"""Emit roctx phase markers around each ATOM (native) forward pass so rocm-trace-lite (RTL) records,
directly in the rtl trace, when every pass begins and whether it is prefill / decode / mixed.

ATOM-native analog of the vLLM/SGLang marker: wrap
``atom.model_engine.model_runner.ModelRunner.forward(self, batch)`` (one call per forward pass) and
build the SAME canonical label the parser + dashboard expect:
    execute_context_<n_ctx_reqs>(<ctx_tokens>)_generation_<n_gen_reqs>(<gen_tokens>)
Downstream classification: prefill = gen_tokens==0, decode = ctx_tokens==0, mixed = both>0.

The numbers come straight from ATOM's ScheduledBatch (verified fields): total_seqs_num_prefill /
total_tokens_num_prefill / total_seqs_num_decode / total_tokens_num_decode / total_tokens_num.
Activated only when ATOM_ROCTX_PHASE_MARKERS=1 (set by run_atom.py for the profiling pass). The patch
installs via a sys.meta_path hook so it applies in every interpreter that imports the ATOM model
runner -- including ATOM's spawned engine-core/worker processes (sitecustomize re-runs there).
"""
import functools
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "atom.model_engine.model_runner"
# Diagnostic only: when ATOM_ROCTX_DEBUG=1, record each forward() call to a file to confirm the hook.
_DEBUG = os.environ.get("ATOM_ROCTX_DEBUG") == "1"
_DEBUG_LOG = os.environ.get("ATOM_ROCTX_DEBUG_LOG", "/tmp/prof/fwd_calls.log")


def _load_roctx():
    """Return (push, pop) roctx functions from RTL's globally-visible shim, or (None, None).

    RTL (`rtl trace`) LD_PRELOADs its own roctx shim; only ranges pushed through the global symbols
    (ctypes.CDLL(None)) are captured. Outside `rtl trace` the symbol is absent -> markers disabled.
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


def _phase_label(batch):
    """Build the canonical execute_context/generation label from an ATOM ScheduledBatch."""
    ctx_reqs = int(getattr(batch, "total_seqs_num_prefill", 0) or 0)
    ctx_tok = int(getattr(batch, "total_tokens_num_prefill", 0) or 0)
    gen_reqs = int(getattr(batch, "total_seqs_num_decode", 0) or 0)
    gen_tok = int(getattr(batch, "total_tokens_num_decode", 0) or 0)
    # Request counts are NOT reported for ATOM: the per-request split is not this engine's
    # native annotation (only vLLM emits execute_context natively). Emit 0 = "not reported";
    # the stage is derived purely from the authentic token counts, and the dashboard hides
    # 0-req counts. See kernel_summary_payload.stage_of (token-only classification).
    return f"execute_context_0({ctx_tok})_generation_0({gen_tok})"


def _is_active(batch):
    """A pass worth bracketing: has scheduled tokens."""
    total = getattr(batch, "total_tokens_num", None)
    if total is None:
        total = (int(getattr(batch, "total_tokens_num_prefill", 0) or 0)
                 + int(getattr(batch, "total_tokens_num_decode", 0) or 0))
    return int(total) > 0


def _wrap_forward(ModelRunner):
    orig = ModelRunner.forward
    if getattr(orig, "_roctx_wrapped", False):
        return
    push, pop = _load_roctx()
    if push is None:
        print("[roctx_phase_markers] roctx shim not visible; markers disabled", file=sys.stderr)
        return
    import torch

    @functools.wraps(orig)
    def forward(self, batch, *args, **kwargs):
        # Never push roctx / synchronize during CUDA-graph capture (illegal; corrupts capture).
        try:
            if torch.cuda.is_current_stream_capturing():
                return orig(self, batch, *args, **kwargs)
        except Exception:
            pass
        active = False
        try:
            active = _is_active(batch)
            if _DEBUG:
                try:
                    with open(_DEBUG_LOG, "a") as fh:
                        fh.write(f"forward {_phase_label(batch)} active={active}\n")
                except Exception:
                    pass
            if active:
                push(_phase_label(batch).encode())
        except Exception:
            active = False
        try:
            return orig(self, batch, *args, **kwargs)
        finally:
            if active:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                pop()

    forward._roctx_wrapped = True
    ModelRunner.forward = forward
    print("[roctx_phase_markers] installed on atom ModelRunner.forward", file=sys.stderr)


class _LoaderWrapper(importlib.abc.Loader):
    """Delegates to the real loader, then patches ModelRunner.forward after the module executes."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _wrap_forward(module.ModelRunner)
        except Exception as exc:           # never break the import
            print(f"[roctx_phase_markers] patch failed: {exc}", file=sys.stderr)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Patches the ATOM model_runner module right after it is first imported (in any process)."""

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)         # let the real finders resolve it now
        spec = importlib.util.find_spec(name)
        if spec is not None:
            spec.loader = _LoaderWrapper(spec.loader)
        return spec


def install():
    """Idempotently arrange for the patch to apply on import (or now, if already imported)."""
    if _TARGET in sys.modules:
        try:
            _wrap_forward(sys.modules[_TARGET].ModelRunner)
        except Exception as exc:
            print(f"[roctx_phase_markers] patch failed: {exc}", file=sys.stderr)
    elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
