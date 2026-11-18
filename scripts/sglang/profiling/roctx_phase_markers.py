"""Emit roctx phase markers around each SGLang forward pass so rocm-trace-lite (RTL) records,
directly in the rtl trace, when every pass begins and whether it is prefill / decode / mixed.

SGLang analog of the vLLM marker: instead of vLLM's ``Worker.execute_model`` we wrap
``sglang.srt.model_executor.model_runner.ModelRunner.forward`` (one call per forward pass), and we
build the SAME canonical label the parser + dashboard expect:
    execute_context_<n_ctx_reqs>(<ctx_tokens>)_generation_<n_gen_reqs>(<gen_tokens>)
Downstream classification: prefill = gen_tokens==0, decode = ctx_tokens==0, mixed = both>0.

The numbers come from the ForwardBatch (verified fields): forward_mode, batch_size, input_ids,
extend_num_tokens, extend_seq_lens. Activated only when SGLANG_ROCTX_PHASE_MARKERS=1 (set by
run_sglang.py for the profiling pass). The patch installs via a sys.meta_path hook so it applies in
every interpreter that imports the SGLang model runner -- including the spawned scheduler/TP-worker
processes (sitecustomize re-runs there).
"""
import functools
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "sglang.srt.model_executor.model_runner"
# Diagnostic only: when SGLANG_ROCTX_DEBUG=1, record each forward() call (mode, active) to a file so
# we can confirm ModelRunner.forward is the per-pass runtime hook. Inert in production.
_DEBUG = os.environ.get("SGLANG_ROCTX_DEBUG") == "1"
_DEBUG_LOG = os.environ.get("SGLANG_ROCTX_DEBUG_LOG", "/tmp/prof/fwd_calls.log")


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


def _phase_label(forward_batch):
    """Build the canonical execute_context/generation label from a SGLang ForwardBatch.

    decode  -> ctx=0/0,           gen=bs/ntok
    mixed   -> ctx=nExt/ext,      gen=(bs-nExt)/(ntok-ext)
    extend  -> ctx=bs/(ext|ntok), gen=0/0
    """
    mode = forward_batch.forward_mode
    ids = getattr(forward_batch, "input_ids", None)
    ntok = int(ids.shape[0]) if ids is not None else 0
    ext_raw = getattr(forward_batch, "extend_num_tokens", None)
    ext = int(ext_raw) if ext_raw else 0

    if mode.is_decode():
        ctx_tok, gen_tok = 0, ntok
    elif mode.is_mixed():
        ctx_tok, gen_tok = ext, ntok - ext
    else:  # EXTEND / prefill (and draft-extend variants)
        ctx_tok, gen_tok = (ext or ntok), 0
    # Request counts are NOT reported for SGLang: the per-request split is not this engine's
    # native annotation (only vLLM emits execute_context natively). Emit 0 = "not reported";
    # the stage is derived purely from the authentic token counts, and the dashboard hides
    # 0-req counts. See kernel_summary_payload.stage_of (token-only classification).
    return f"execute_context_0({ctx_tok})_generation_0({gen_tok})"


def _is_active(forward_batch):
    """A pass worth bracketing: has tokens and is not IDLE."""
    mode = forward_batch.forward_mode
    if hasattr(mode, "is_idle") and mode.is_idle():
        return False
    ids = getattr(forward_batch, "input_ids", None)
    return bool(ids is not None and int(ids.shape[0]) > 0)


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
    def forward(self, forward_batch, *args, **kwargs):
        # Never push roctx / synchronize during CUDA-graph capture -- doing so is illegal and
        # corrupts the capture. Pass straight through when the stream is capturing.
        try:
            if torch.cuda.is_current_stream_capturing():
                return orig(self, forward_batch, *args, **kwargs)
        except Exception:
            pass
        active = False
        try:
            active = _is_active(forward_batch)
            if _DEBUG:
                try:
                    mode = getattr(getattr(forward_batch, "forward_mode", None), "name", "?")
                    with open(_DEBUG_LOG, "a") as fh:
                        fh.write(f"forward mode={mode} active={active}\n")
                except Exception:
                    pass
            if active:
                push(_phase_label(forward_batch).encode())
        except Exception:
            active = False
        try:
            return orig(self, forward_batch, *args, **kwargs)
        finally:
            if active:
                try:                       # sync so this pass's GPU kernels fall inside the window
                    torch.cuda.synchronize()
                except Exception:
                    pass
                pop()

    forward._roctx_wrapped = True
    ModelRunner.forward = forward
    print("[roctx_phase_markers] installed on sglang ModelRunner.forward", file=sys.stderr)


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
    """Patches the SGLang model_runner module right after it is first imported (in any process)."""

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
