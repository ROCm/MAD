#!/usr/bin/env python3
"""Route the top-k/top-p sampler to the PyTorch-native path instead of the slow
Triton path on this ROCm build (fixes multi-minute stall in profile_run).

PROBLEM
  On gfx942/ROCm, vLLM's sampler `forward = forward_native` -> apply_top_k_top_p.
  apply_top_k_top_p() dispatches to the Triton implementation whenever
  `HAS_TRITON and logits.shape[0] >= 8` (topk_topp_sampler.py:364). The Triton
  top-k/top-p kernel (topk_topp_triton.py:922, iterative CDF/histogram approx over
  the 163k vocab) is pathologically slow on this build -- the memory-profiling
  dummy run (_dummy_sampler_run, batch = max_num_seqs) hangs for 20-30 min at
  100% GPU and the ApiServers hit VLLM_ENGINE_READY_TIMEOUT. This is generic to
  the image (independent of MoRIIO/KDA/disagg).

FIX (surgical)
  Change the dispatch threshold so the profile/serve batch takes the PyTorch-native
  sort path (apply_top_k_top_p_pytorch), which is correct and fast here. We raise
  the `>= 8` gate to an effectively-never threshold rather than deleting the Triton
  import (keeps the module importable / other code paths intact).

  Correctness: apply_top_k_top_p_pytorch is the reference sort-based top-k/top-p
  (used already for small batches); routing all batches to it changes performance
  characteristics only, not sampling semantics.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_sampler_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/sample/ops/topk_topp_sampler.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-sampler] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src

    if "k3-sampler: force pytorch-native" in src:
        print("[k3-sampler] already applied.")
        return 0

    old = (
        "    if HAS_TRITON and logits.shape[0] >= 8:\n"
        "        return apply_top_k_top_p_triton(logits, k, p)\n"
    )
    new = (
        "    # k3-sampler: force pytorch-native top-k/top-p on ROCm/gfx942 (the\n"
        "    # Triton path stalls for minutes in profile_run on this build).\n"
        "    if False and HAS_TRITON and logits.shape[0] >= 8:\n"
        "        return apply_top_k_top_p_triton(logits, k, p)\n"
    )
    if old in src:
        src = src.replace(old, new, 1)
    else:
        # tolerate minor formatting drift: match the condition line alone
        cond = "    if HAS_TRITON and logits.shape[0] >= 8:"
        if cond in src:
            src = src.replace(
                cond,
                "    # k3-sampler: force pytorch-native (Triton stalls on ROCm)\n"
                "    if False and HAS_TRITON and logits.shape[0] >= 8:",
                1,
            )
        else:
            print("[k3-sampler] WARN: dispatch anchor not found -- sampler may "
                  "still use the slow Triton path. Review topk_topp_sampler.py.")
            return 0

    if src != orig:
        open(path, "w").write(src)
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[k3-sampler] ERROR: compile failed: {e}", file=sys.stderr)
            return 1
        print(f"[k3-sampler] patched {REL} -- forced pytorch-native top-k/top-p")

    # --- H2: neutralize the dummy-sampler warmup in profile_run ---------------
    # gpu_model_runner._dummy_sampler_run does a FULL-VOCAB (163k) vocab-parallel
    # all_gather via compute_logits at max_num_tokens. Under tp8xdp2 this collective
    # deadlocks (the two DP engines' profile forwards don't match on the inner TP
    # all-gather) -> 30+ min hang at 100% GPU, then ApiServer timeout. The dummy
    # sampler run is only a memory-profiling warmup; _dummy_run (the real forward,
    # called just before it) already sizes activation peak. Early-return a tiny
    # tensor so profile_run completes without the pathological gather.
    # --- H2: skip the profile-time sampler run on ROCm --------------------------
    # Mirrors ROCm/MAD custom-vLLM commit 77c2cf269 (Shiksha Patel): "The first
    # AITER top_k_top_p launch on MI300X leaves an HSA signal that never retires,
    # so the second sampler warm-up's implicit device sync hangs forever." This is
    # the profile_run:_sync_device hang under DP-EP-16 bring-up. Fix = skip the
    # profile-time _dummy_sampler_run call entirely (it's non-essential; the real
    # _dummy_run forward already sizes activation peak, and forward_native JIT-caches
    # at first real use). The amdsiloai PREBUILT image lacks this source fix.
    vllm_root = path.split("/v1/sample/ops/")[0]
    gmr = os.path.join(vllm_root, "v1", "worker", "gpu_model_runner.py")
    if os.path.isfile(gmr):
        g = open(gmr).read()
        gorig = g
        if "k3-sampler: skip profile-time sampler" in g:
            print("[k3-sampler] profile-time sampler run already skipped.")
        else:
            # Call site in profile_run: after _dummy_run(...is_profile=True), the
            # `if get_pp_group().is_last_rank:` guards the _dummy_sampler_run call.
            anchor = (
                "        hidden_states, last_hidden_states = self._dummy_run(\n"
                "            self.max_num_tokens, is_profile=True\n"
                "        )\n"
                "        if get_pp_group().is_last_rank:\n"
            )
            repl = (
                "        hidden_states, last_hidden_states = self._dummy_run(\n"
                "            self.max_num_tokens, is_profile=True\n"
                "        )\n"
                "        # k3-sampler: skip profile-time sampler run on ROCm (the\n"
                "        # AITER top_k_top_p kernel leaves an unretired HSA signal ->\n"
                "        # _sync_device hangs under DP-EP-16). Mirrors MAD 77c2cf269.\n"
                "        if current_platform.is_rocm():\n"
                "            output = None\n"
                "        elif get_pp_group().is_last_rank:\n"
            )
            if anchor in g:
                g = g.replace(anchor, repl, 1)
                # ensure current_platform is imported (it is, in this module)
                if "current_platform" not in gorig:
                    g = ("from vllm.platforms import current_platform  # k3-sampler\n"
                         + g)
                open(gmr, "w").write(g)
                try:
                    import py_compile
                    py_compile.compile(gmr, doraise=True)
                except Exception as e:
                    print(f"[k3-sampler] ERROR: gpu_model_runner compile: {e}",
                          file=sys.stderr)
                    return 1
                print("[k3-sampler] patched gpu_model_runner.py -- skip profile-time "
                      "sampler run on ROCm (mirrors MAD 77c2cf269)")
            else:
                print("[k3-sampler] WARN: profile_run _dummy_sampler_run call-site "
                      "anchor not found -- profile may still hang. Review "
                      "gpu_model_runner.profile_run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
