#!/usr/bin/env python3
"""Skip the DP-coordination all_reduce in set_forward_context that deadlocks
EP16 (DP>1 cross-node) + MTP speculative decode during startup profiling.

ROOT CAUSE (py-spy proven, 2nd/deeper site):
  The first fix (apply_glm_vllm_dp_profile_sync_fix.py) patched
  vllm/v1/worker/gpu/dp_utils.py (sync_cudagraph_and_dp_padding). A fresh py-spy dump showed
  the ACTUAL deadlock is a DIFFERENT all_reduce, one call deeper:

    all_reduce (distributed_c10d.py:3181)
    _run_ar (vllm/v1/worker/dp_utils.py:56)                  # <-- worker/dp_utils, NOT worker/gpu/
    _synchronize_dp_ranks (worker/dp_utils.py:134)
    coordinate_batch_across_dp (worker/dp_utils.py:225)
    set_forward_context (vllm/forward_context.py:299)
    _run_model -> _prefill -> propose (spec_decode/.../speculator.py)  # MTP speculator

  During the startup profile forward, the MTP speculator enters set_forward_context with
  num_tokens_across_dp=None. For an MoE model with data_parallel_size>1, forward_context then
  calls coordinate_batch_across_dp(), which does a DP-group all_reduce needing all 16 ranks.
  At EP16 (DP16 across 2 nodes) the ranks reach it asymmetrically (one node still running the MoE
  forward at 100% GPU) -> the all_reduce hangs -> futex_wait deadlock. Non-MTP decode (M=1) and
  EP8 (single node, ranks arrive together) do not hit this.

FIX (surgical, forward_context.py):
  forward_context already has an `elif num_tokens_across_dp is None:` branch that builds a LOCAL
  tensor (torch.tensor([num_tokens])) with NO collective. Route the profile/eager case down that
  local-tensor branch instead of the all_reduce branch, by adding an env gate to the collective
  branch's condition. DP padding + microbatching are already "disabled" here (the code comment
  says so), so the local tensor is a correct no-op stand-in during profiling. Real inference is
  unaffected: it passes a non-None num_tokens_across_dp (from the model runner's own sync), so
  neither branch fires.

  Gated VLLM_SKIP_FWDCTX_DP_AR (default "1" = apply). Set to 0 to restore stock behavior.

Idempotent + anchor-based. Missing anchor warns-and-skips; changed anchor is a hard error.

Usage: apply_glm_vllm_fwdctx_dp_sync_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "forward_context.py"

OLD = """        if (
            num_tokens_across_dp is None
            and vllm_config.parallel_config.data_parallel_size > 1
        ):"""

NEW = """        if (
            num_tokens_across_dp is None
            and vllm_config.parallel_config.data_parallel_size > 1
            and not int(os.environ.get("VLLM_SKIP_FWDCTX_DP_AR", "1"))
        ):"""


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
        print(f"[fwdctx-dp] {REL} not found -- skipping.")
        return 0
    src = open(path).read()
    if "VLLM_SKIP_FWDCTX_DP_AR" in src:
        print("[fwdctx-dp] already applied -- skipping.")
        return 0
    if OLD not in src:
        print(
            "[fwdctx-dp] ERROR: num_tokens_across_dp all_reduce anchor not found in "
            "forward_context.py; vLLM changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1
    # ensure os imported
    if "\nimport os" not in src and not src.startswith("import os"):
        src = src.replace("import torch\n", "import os\nimport torch\n", 1)
    # also fix the elif local-tensor branch to be dp_size-length (DPMetadata.make indexes [dp_rank])
    old2 = "            num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int32)"
    new2 = ("            _dp = vllm_config.parallel_config.data_parallel_size\n"
            "            num_tokens_across_dp = torch.full((_dp,), num_tokens, dtype=torch.int32) if _dp > 1 else torch.tensor([num_tokens], dtype=torch.int32)")
    if old2 in src and "torch.full" not in src:
        src = src.replace(old2, new2, 1)
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[fwdctx-dp] skip DP all_reduce in set_forward_context applied to {REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
