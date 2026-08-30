#!/usr/bin/env python3
"""Skip the DP cudagraph-padding all_reduce during the eager profile run, which
deadlocks EP16 (DP>1 cross-node) + MTP speculative decode.

ROOT CAUSE (proven by py-spy stack dump on the wedged EP16-MTP workers):
  During startup, gpu_worker.determine_available_memory() calls model_runner.profile_run()
  even when kv_cache_memory_bytes is set (it is kept to compile kernels). With MTP the
  speculator's propose() runs a dummy forward that calls dispatch_cg_and_sync_dp() ->
  sync_cudagraph_and_dp_padding(), which does a DP-group all_reduce requiring ALL dp ranks.
  On EP16 (DP16 across 2 nodes) the ranks reach this barrier ASYMMETRICALLY (one node lags
  building/loading the fmoe 1tg kernel), so the all_reduce never completes -> deadlock.
  py-spy: half the ranks block in all_reduce (dp_utils.py:39, futex_wait); the other node's
  ranks are still RUNNING upstream. The 1tg kernel itself is FINE (verified: it loads and
  runs in ~2s single-process); the failure is purely the DP barrier during the eager profile
  forward. EP8 (single node, DP8 intra-node) doesn't stall because all ranks arrive together.

FIX (surgical, profile-run only):
  In dispatch_cg_and_sync_dp(), when need_eager=True (the profile run / no-cudagraph path),
  the subsequent sync_cudagraph_and_dp_padding() does an all_reduce whose result is DISCARDED
  (it early-returns the all-eager NONE descriptor). So skip the DP sync entirely on that path
  and return the eager descriptor directly. Real inference (need_eager=False) is untouched --
  the DP padding sync still runs where it is actually needed. Gated by
  VLLM_SKIP_DP_SYNC_ON_PROFILE (default "1" = apply); set to 0 to restore stock behavior.

Idempotent + anchor-based. Missing anchor warns-and-skips; changed anchor is a hard error.

Usage: apply_glm_vllm_dp_profile_sync_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/worker/gpu/dp_utils.py"

OLD = """    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:"""

NEW = """    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
        # GLM-5.2 EP16+MTP fix: on the eager/profile path the DP all_reduce below
        # early-returns the all-eager NONE descriptor (its result is discarded), but
        # the collective itself deadlocks when dp ranks reach it asymmetrically across
        # nodes (one node lags building the fmoe 1tg kernel). Skip it here; real
        # inference (need_eager=False) still runs the DP padding sync.
        if int(os.environ.get("VLLM_SKIP_DP_SYNC_ON_PROFILE", "1")):
            return batch_desc, None
    else:"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        # try to locate installed vllm
        try:
            import vllm  # noqa
            path = os.path.join(os.path.dirname(vllm.__file__), REL)
        except Exception:
            pass
    if not os.path.isfile(path):
        print(f"[dp-sync] {REL} not found -- skipping.")
        return 0
    src = open(path).read()
    if "VLLM_SKIP_DP_SYNC_ON_PROFILE" in src:
        print("[dp-sync] already applied -- skipping.")
        return 0
    if OLD not in src:
        print(
            "[dp-sync] ERROR: need_eager anchor not found in dp_utils.py; "
            "vLLM changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1
    if "import os" not in src.splitlines()[0:15].__str__():
        # ensure os is imported (dp_utils.py may not import it)
        if "\nimport os\n" not in src and not src.startswith("import os"):
            src = src.replace("import torch\n", "import os\n\nimport torch\n", 1)
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[dp-sync] skip DP all_reduce on eager profile run applied to {REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
