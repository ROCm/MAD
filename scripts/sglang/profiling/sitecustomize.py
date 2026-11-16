"""Auto-imported by Python at interpreter startup when this directory is on PYTHONPATH.

Installs the roctx prefill/decode/mixed phase markers into the SGLang model runner, but ONLY when
SGLANG_ROCTX_PHASE_MARKERS=1 (set by run_sglang.py for the rtl profiling pass). Inert otherwise, so
it is safe to leave this directory on PYTHONPATH for normal runs.

Living in sitecustomize is what makes the patch reach SGLang's spawned scheduler/TP-worker processes:
every fresh interpreter imports sitecustomize, re-installs the import hook, and the hook patches
ModelRunner.forward the moment that module loads in that process.
"""
import os

if os.environ.get("SGLANG_ROCTX_PHASE_MARKERS") == "1":
    try:
        import roctx_phase_markers
        roctx_phase_markers.install()
    except Exception as exc:               # never break interpreter startup
        import sys
        print(f"[sitecustomize] roctx_phase_markers not installed: {exc}", file=sys.stderr)
