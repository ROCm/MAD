#!/usr/bin/env python3
"""Scope VLLM_STARTUP_DP_UNIFORM to the warmup+cudagraph-capture block only, so the
GLM-5.2 EP16+MTP startup DP all_reduce is replaced by a local uniform fill DURING
capture (no deadlock) while runtime inference keeps the real cross-rank all_reduce.

Companion to apply_glm_vllm_startup_dp_uniform_fix.py (which teaches dp_utils to honor
the flag). Here we set the flag right before the warmup loop + capture_model() and clear
it immediately after, in a try/finally, so nothing at runtime is affected.

Only takes effect when VLLM_STARTUP_DP_UNIFORM_ENABLE=1 (so the wrap is a no-op unless
you opt in for EP16+MTP with cudagraphs). When enabled, decode can run
DECODE_CUDAGRAPH_MODE=FULL_AND_PIECEWISE instead of the eager NONE fallback.

Idempotent + anchor-based. Missing anchor warns-and-skips; changed anchor is a hard error.

Usage: apply_glm_vllm_startup_dp_uniform_worker_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/worker/gpu_worker.py"

OLD = """        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()"""

NEW = """        # We skip EPLB here since we don't want to record dummy metrics
        # GLM-5.2 EP16+MTP cudagraph fix: during warmup+capture, replace the DP
        # coordination all_reduce with a local uniform fill (dp_utils honors
        # VLLM_STARTUP_DP_UNIFORM) so capture does not deadlock on asymmetric rank
        # arrival. Scoped to this block only; runtime keeps the real all_reduce.
        _dpuni_enable = int(os.environ.get("VLLM_STARTUP_DP_UNIFORM_ENABLE", "0"))
        _dpuni_prev = os.environ.get("VLLM_STARTUP_DP_UNIFORM")
        if _dpuni_enable:
            os.environ["VLLM_STARTUP_DP_UNIFORM"] = "1"
        try:
            for size in sorted(warmup_sizes, reverse=True):
                logger.info("Compile and warming up model for size %d", size)
                self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
            self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

            # Warmup and tune the kernels used during model execution before
            # cuda graph capture.
            kernel_warmup(self)

            cuda_graph_memory_bytes = 0
            if not self.model_config.enforce_eager:
                cuda_graph_memory_bytes = self.model_runner.capture_model()
        finally:
            if _dpuni_enable:
                if _dpuni_prev is None:
                    os.environ.pop("VLLM_STARTUP_DP_UNIFORM", None)
                else:
                    os.environ["VLLM_STARTUP_DP_UNIFORM"] = _dpuni_prev"""


def find(root):
    p = os.path.join(root, REL)
    if os.path.isfile(p):
        return p
    try:
        import vllm  # noqa
        return os.path.join(os.path.dirname(vllm.__file__), REL)
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = find(sys.argv[1])
    if not path or not os.path.isfile(path):
        print(f"[startup-dp-uniform-worker] {REL} not found -- skipping.")
        return 0
    src = open(path).read()
    if "VLLM_STARTUP_DP_UNIFORM_ENABLE" in src:
        print("[startup-dp-uniform-worker] already applied -- skipping.")
        return 0
    if OLD not in src:
        print(
            "[startup-dp-uniform-worker] ERROR: warmup+capture block anchor not found in "
            "gpu_worker.py; vLLM changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1
    if "\nimport os" not in src and not src.startswith("import os"):
        src = src.replace("import torch\n", "import os\nimport torch\n", 1)
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[startup-dp-uniform-worker] wrapped warmup+capture with VLLM_STARTUP_DP_UNIFORM in {REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
