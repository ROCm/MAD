#!/usr/bin/env python3
"""Let GLM-5.2 EP16 (DP16 cross-node) + MTP capture decode cudagraphs WITHOUT the
startup DP all_reduce deadlocking -- so decode can run FULL_AND_PIECEWISE graphs
(perf-critical) instead of falling back to eager (CUDAGRAPH_MODE=NONE).

ROOT CAUSE (py-spy proven):
  compile_or_warm_up_model() runs dummy forwards at each warmup/capture size. Each goes
  through dispatch_cg_and_sync_dp(need_eager=False) -> sync_cudagraph_and_dp_padding(),
  which does a DP-group dist.all_reduce requiring ALL dp ranks at once. During cudagraph
  CAPTURE the ranks proceed at their own pace (one JIT-building an AITER kernel for a size
  while another already reached the all_reduce) -> asymmetric arrival -> deadlock. This is
  why EP16-MTP previously had to run with CUDAGRAPH_MODE=NONE (eager, slower).

  It is NOT a correctness collective at startup: every rank passes the SAME dummy shape at
  a given warmup/capture size, so the all_reduce result is uniform anyway.

FIX (surgical, dp_utils.py sync function):
  When VLLM_STARTUP_DP_UNIFORM=1, skip the dist.all_reduce and fill the coordination tensor
  LOCALLY as if every dp rank reported this rank's (identical) dummy values. All downstream
  logic (cg_mode selection, uniform-token agreement, padding, cudagraph_manager.dispatch)
  runs unchanged, so graphs still capture correctly -- just without the deadlocking
  collective. gpu_worker sets this env ONLY around warmup+capture and clears it after, so
  runtime inference keeps the true cross-rank all_reduce (see
  apply_glm_vllm_startup_dp_uniform_worker_fix.py).

  Gated VLLM_STARTUP_DP_UNIFORM (default "0" = stock all_reduce). Only meaningful while the
  worker sets it during startup; runtime is unaffected.

Idempotent + anchor-based. Missing anchor warns-and-skips; changed anchor is a hard error.

Usage: apply_glm_vllm_startup_dp_uniform_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/worker/gpu/dp_utils.py"

OLD = """    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)
    dist.all_reduce(tensor, group=group)"""

NEW = """    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)
    # GLM-5.2 EP16+MTP cudagraph fix: during startup warmup/capture every dp rank
    # passes the SAME dummy shape, so the all_reduce result is uniform. Capturing
    # graphs, ranks arrive asymmetrically and the collective deadlocks. When the
    # worker sets VLLM_STARTUP_DP_UNIFORM (startup only), fill the tensor locally as
    # if all ranks reported this rank's values -- no collective, graphs still capture.
    if int(os.environ.get("VLLM_STARTUP_DP_UNIFORM", "0")):
        for _r in range(dp_size):
            tensor[0][_r] = num_tokens
            tensor[1][_r] = desired_batch_desc.cg_mode.value
            tensor[2][_r] = uniform_token_count or 0
            tensor[3][_r] = max_query_len or -1
    else:
        dist.all_reduce(tensor, group=group)"""


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
        print(f"[startup-dp-uniform] {REL} not found -- skipping.")
        return 0
    src = open(path).read()
    if "VLLM_STARTUP_DP_UNIFORM" in src:
        print("[startup-dp-uniform] already applied -- skipping.")
        return 0
    if OLD not in src:
        print(
            "[startup-dp-uniform] ERROR: all_reduce anchor not found in dp_utils.py; "
            "vLLM changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1
    if "\nimport os" not in src and not src.startswith("import os"):
        src = src.replace("import torch\n", "import os\nimport torch\n", 1)
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[startup-dp-uniform] gated startup DP all_reduce behind VLLM_STARTUP_DP_UNIFORM in {REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
