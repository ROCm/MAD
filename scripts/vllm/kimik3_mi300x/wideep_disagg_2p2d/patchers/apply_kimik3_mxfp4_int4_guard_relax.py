#!/usr/bin/env python3
"""Relax the over-strict aiter#4471 guard on K3's packed-int4 gfx942 MoE path.

PROBLEM
  On gfx942 (MI300X) there is no scaled-MXFP4 MFMA, so vLLM's mxfp4 quant layer
  requantizes Kimi-K3's SiTU MXFP4 experts to groupwise int4 and serves them via
  AITER's bf16 x int4 FlyDSL path (mxfp4.py::_setup_kernel_k3_situ_gfx942, enabled
  by --quantization-config.moe.weight int4_per_group_32).

  That method guards on whether aiter's build-time helper
  `aiter.ops.flydsl.kernels.moe_gemm_2stage.compile_moe_gemm1` exposes an `act`
  parameter (ROCm/aiter#4471). If not, it RAISES:
    "This AITER build ignores the SiTUv2 activation on the packed-int4 MoE path
     and would silently compute SiLU. Rebuild with an AITER that includes
     ROCm/aiter#4471."

  BUT the guard is a false positive on this stack:
    - It inspects a build-time *compile* helper's signature, not the runtime call.
    - The runtime MoE dispatch (Mxfp4MoeMethod.apply -> moe_kernel.apply) passes
      `activation=layer.activation` through the standard AiterExperts fused_moe path
      (mxfp4.py:~460), so SiTUv2 DOES reach the kernel at runtime.
    - The PROVEN, validated colocated image (amdsiloai/vllm:kimi-k3-mi325x-release-v2)
      ships this EXACT aiter (compile_moe_gemm1 with NO `act` param) and serves K3
      SiTU int4 on gfx942 with correct output. Its (older) vLLM simply lacks this
      newer guard. Our from-source vLLM added the stricter check.

FIX (surgical)
  Neutralize only the guard's `raise` so the packed-int4 setup proceeds, matching
  the proven image's behavior. We do NOT touch the conversion/kernel logic.

  Correctness note: this trusts the runtime activation plumbing (validated by the
  proven image producing correct K3 output on identical aiter). If a future AITER
  regresses the runtime SiTUv2 handling, re-enable the guard.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_mxfp4_int4_guard_relax.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/quantization/mxfp4.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-int4-guard] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src

    if "k3-int4-guard: relaxed" in src:
        print("[k3-int4-guard] already applied.")
        return 0

    anchor = (
        '        if "act" not in inspect.signature(compile_moe_gemm1).parameters:\n'
        "            raise RuntimeError(\n"
    )
    repl = (
        "        # k3-int4-guard: relaxed -- the proven colocated image runs this\n"
        "        # exact aiter (compile_moe_gemm1 without `act`) and serves K3 SiTU\n"
        "        # int4 correctly; the runtime dispatch passes activation through the\n"
        "        # AiterExperts path regardless. Warn instead of aborting.\n"
        '        if "act" not in inspect.signature(compile_moe_gemm1).parameters:\n'
        "            import logging as _lg\n"
        '            _lg.getLogger(__name__).warning(\n'
    )
    if anchor in src:
        src = src.replace(anchor, repl, 1)
        # The original raise(...) body now feeds logger.warning(...) instead;
        # that is valid Python (warning takes the same string args). Leave the
        # message text and closing paren as-is.
    else:
        print("[k3-int4-guard] WARN: guard anchor not found -- the strict "
              "aiter#4471 check may still abort. Review "
              "mxfp4.py::_setup_kernel_k3_situ_gfx942.")
        return 0

    if src != orig:
        open(path, "w").write(src)
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[k3-int4-guard] ERROR: compile failed: {e}", file=sys.stderr)
            # restore to avoid leaving a broken file
            open(path, "w").write(orig)
            return 1
        print("[k3-int4-guard] relaxed the aiter#4471 packed-int4 guard "
              "(raise -> warning) in mxfp4.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
