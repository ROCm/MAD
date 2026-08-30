#!/usr/bin/env python3
"""Skip the startup profile_run() forward for GLM-5.2 EP16 (DP16 cross-node) + MTP,
which deadlocks in the MoE all2all during determine_available_memory().

ROOT CAUSE (py-spy proven on the wedged EP16-MTP decode workers):
  gpu_worker.determine_available_memory() runs model_runner.profile_run() even when
  kv_cache_memory_bytes is preset -- the vLLM comment says it is kept "to compile the
  model for max_num_batched_tokens". That profile FORWARD executes the MoE all2all at
  max_num_batched_tokens. At EP16 (DP16 across 2 nodes over the MoRI-EP host proxy) the
  ranks reach the all2all asymmetrically and it never completes; torch.cuda.synchronize()
  (model_runner.py:847) then blocks forever (the GPU queue never drains). py-spy:
    synchronize (torch/accelerator/__init__.py:270)
    profile_run (vllm/v1/worker/gpu/model_runner.py:847)
    determine_available_memory (vllm/v1/worker/gpu_worker.py:498)
  EP8 (single-node DP8) and TP8 (dp_size=1) do not hit this. Non-MTP EP16 also stalls here;
  it is the profile forward's collective, not MTP-specific -- but MTP is the config we need.

FIX (surgical, gpu_worker.py, profile branch only):
  When kv_cache_memory_bytes is preset the profile_run is ONLY a JIT-compile warmup (memory
  is already decided), so its result is discarded. Gate it behind VLLM_SKIP_PROFILE_RUN so
  the deadlocking forward never runs; kernels JIT lazily on the first real request instead
  (pair with VLLM_SKIP_KERNEL_WARMUP=1, already used). The else-branch profile_run (real
  memory profiling, used when kv bytes are NOT preset) is untouched.

  Gated VLLM_SKIP_PROFILE_RUN (default "0" = stock). Set to 1 for EP16(+MTP).

Idempotent + anchor-based. Missing anchor warns-and-skips; changed anchor is a hard error.

Usage: apply_glm_vllm_skip_profile_run_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/worker/gpu_worker.py"

OLD = """        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()"""

NEW = """        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            # GLM-5.2 EP16+MTP fix: this profile forward runs the MoE all2all at
            # max_num_batched_tokens and deadlocks cross-node (DP16). Since kv bytes
            # are preset, profiling is only JIT warmup; gate it so kernels compile
            # lazily on the first real request instead.
            if not int(os.environ.get('VLLM_SKIP_PROFILE_RUN', '0')):
                self.model_runner.profile_run()"""


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
        print(f"[skip-profile-run] {REL} not found -- skipping.")
        return 0
    src = open(path).read()
    if "VLLM_SKIP_PROFILE_RUN" in src:
        print("[skip-profile-run] already applied -- skipping.")
        return 0
    if OLD not in src:
        print(
            "[skip-profile-run] ERROR: kv_cache_memory_bytes profile_run anchor not found "
            "in gpu_worker.py; vLLM changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1
    if "\nimport os" not in src and not src.startswith("import os"):
        src = src.replace("import torch\n", "import os\nimport torch\n", 1)
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[skip-profile-run] gated startup profile_run behind VLLM_SKIP_PROFILE_RUN in {REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
