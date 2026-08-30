#!/usr/bin/env python3
"""Force the CK 2-stage MoE kernel (not the 1-stage ASM 'fmoe_1tg') for FP8
blockscale decode on gfx950, so MTP/speculative decode doesn't wedge.

ROOT CAUSE (traced through AITER source + confirmed by ROCm/aiter RFCs on the
gfx950 1-stage FP8-blockscale MoE fast path):
  aiter/fused_moe.py chooses between a 1-stage ASM MoE kernel and the CK 2-stage
  kernel. For QuantType.per_1x128 (GLM-5.2 FP8 blockscale) the default heuristic is

      run_1stage = token > 32 and (inter_dim % 128 == 0)

  - Non-MTP decode: token(M)=1  -> run_1stage=False -> CK 2-stage kernel (prebuilt,
    boots fine). This is why EP16 WITHOUT MTP works.
  - MTP decode: the speculative-verify step raises token(M) above 32 -> run_1stage=True
    -> the 1-stage ASM kernel fmoe_bf16_blockscaleFp8_..._1tg_ps_32x256, whose runtime
    LoadKernel wedges at EP16 (DP16, 256-expert shard) on gfx950. MTP is the ONLY path
    that dispatches this kernel; hence MTP hangs and non-MTP does not.

  AITER's own comment on that line says "for fp8 blockscale, ck has better performance
  so disable assembly kernel" -- i.e. CK is both correct AND faster here; the 1-stage
  ASM path is a mis-tuned fast-path on gfx950 (see ROCm/aiter gfx950 1-stage FP8
  blockscale MoE RFC: threshold + kernel-name resolution not tuned for this arch).

FIX (surgical, gfx950 fp8-blockscale only):
  Force run_1stage=False for the per_1x128 branch. Routes MTP decode to the working CK
  kernel. Gated by AITER_FORCE_CK_FMOE (default "1" = apply); set AITER_FORCE_CK_FMOE=0
  to restore stock behavior. No rebuild -- Python-level dispatch change.

Idempotent + anchor-based. Missing anchor warns-and-skips; a changed anchor is a hard
error (would silently keep MTP broken).

Usage: apply_glm_aiter_fmoe_1stage_gfx950_fix.py <aiter_install_dir_or_vllm_env>
"""
import os
import sys

REL_CANDIDATES = [
    "fused_moe.py",
    "aiter/fused_moe.py",
    "lib/python3.12/dist-packages/aiter/fused_moe.py",
]

OLD = """            if q_type == QuantType.per_1x128:
                # for fp8 blockscale, ck has better performance so disable assembly kernel
                run_1stage = token > 32 and (inter_dim % 128 == 0)"""

NEW = """            if q_type == QuantType.per_1x128:
                # for fp8 blockscale, ck has better performance so disable assembly kernel.
                # GLM-5.2 MTP fix: the 1-stage ASM 'fmoe_1tg' path wedges at EP16/DP16 on
                # gfx950 (mis-tuned fast path); MTP raises token>32 into it. Force CK.
                run_1stage = token > 32 and (inter_dim % 128 == 0)
                if get_gfx() == "gfx950" and int(os.environ.get("AITER_FORCE_CK_FMOE", "1")):
                    run_1stage = False"""


def find_path(root):
    for rel in REL_CANDIDATES:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    # last resort: locate installed aiter
    try:
        import aiter  # noqa
        return os.path.join(os.path.dirname(aiter.__file__), "fused_moe.py")
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <aiter_or_env_dir>", file=sys.stderr)
        return 2
    path = find_path(sys.argv[1])
    if not path or not os.path.isfile(path):
        print("[aiter-fmoe] fused_moe.py not found -- skipping.")
        return 0
    src = open(path).read()
    if "AITER_FORCE_CK_FMOE" in src:
        print("[aiter-fmoe] already applied -- skipping.")
        return 0
    if OLD not in src:
        print(
            "[aiter-fmoe] ERROR: run_1stage per_1x128 anchor not found in fused_moe.py; "
            "AITER changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[aiter-fmoe] forced CK 2-stage fmoe for gfx950 fp8-blockscale in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
