#!/usr/bin/env python3
"""Skip the compile_or_warm_up_model() elif dummy-forward for GLM-5.2 EP16 (DP16
cross-node) + MTP, which deadlocks in the MTP speculator's DP all_reduce.

ROOT CAUSE (py-spy proven on the wedged EP16-MTP decode workers):
  gpu_worker.compile_or_warm_up_model() has an if/elif:
    if use_v2_model_runner and not VLLM_SKIP_KERNEL_WARMUP: warmup_kernels(...)
    elif get_pp_group().is_last_rank: self.model_runner._dummy_run(...)   # sampler prealloc
  With VLLM_SKIP_KERNEL_WARMUP=1 the code takes the ELIF, whose _dummy_run drives the MTP
  speculator's propose() -> dispatch_cg_and_sync_dp() -> sync_cudagraph_and_dp_padding()
  -> a DP-group all_reduce. At EP16 (DP16 across 2 nodes) the ranks reach it asymmetrically
  and it never completes -> deadlock. py-spy:
    all_reduce (torch/distributed/distributed_c10d.py:3181)
    sync_cudagraph_and_dp_padding (vllm/v1/worker/gpu/dp_utils.py:41)
    propose (vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:282)
    _dummy_run (vllm/v1/worker/gpu/model_runner.py:771)
    compile_or_warm_up_model (vllm/v1/worker/gpu_worker.py:839)
  So VLLM_SKIP_KERNEL_WARMUP alone is NOT enough -- it just routes into this elif, which
  still runs the deadlocking dummy forward.

FIX (surgical, gpu_worker.py, warmup elif only):
  The elif _dummy_run only pre-allocates sampler/logits buffers (a fragmentation
  optimization, not a correctness requirement). Gate the elif behind VLLM_SKIP_WARMUP_DUMMY
  so neither warmup branch runs the dummy forward; the first real request pays a small
  one-time buffer alloc. Real inference is unaffected.

  Gated VLLM_SKIP_WARMUP_DUMMY (default "0" = stock). Set to 1 for EP16(+MTP).

Idempotent + anchor-based. Missing anchor warns-and-skips; changed anchor is a hard error.

Usage: apply_glm_vllm_skip_warmup_dummy_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/worker/gpu_worker.py"

OLD = "        elif get_pp_group().is_last_rank:"

NEW = ("        elif get_pp_group().is_last_rank and not int("
       "os.environ.get('VLLM_SKIP_WARMUP_DUMMY', '0')):")


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
        print(f"[skip-warmup-dummy] {REL} not found -- skipping.")
        return 0
    src = open(path).read()
    if "VLLM_SKIP_WARMUP_DUMMY" in src:
        print("[skip-warmup-dummy] already applied -- skipping.")
        return 0
    # Anchor must be unique: the warmup elif. There is exactly one
    # `elif get_pp_group().is_last_rank:` in compile_or_warm_up_model.
    if src.count(OLD) != 1:
        print(
            "[skip-warmup-dummy] ERROR: expected exactly one "
            "'elif get_pp_group().is_last_rank:' anchor in gpu_worker.py, found "
            f"{src.count(OLD)}; vLLM changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1
    if "\nimport os" not in src and not src.startswith("import os"):
        src = src.replace("import torch\n", "import os\nimport torch\n", 1)
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[skip-warmup-dummy] gated warmup elif _dummy_run behind VLLM_SKIP_WARMUP_DUMMY in {REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
