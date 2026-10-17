#!/usr/bin/env python3
"""Make f_a contiguous before f_b_proj in the KDA forward (fixes disagg GPU fault).

ROOT CAUSE (localized tool-free via synced breadcrumbs, top-down):
  The 2P/2D disagg producer GPU-faults in KDA layer 0's forward, at
  `g1 = self.f_b_proj(f_a)` (kimi_gdn_linear_attn.py). f_a is a NON-CONTIGUOUS
  slice from `projected_qkvgfab.split(...)` of the padded in_proj output. Feeding
  that strided view straight into f_b_proj's (bf16) GEMM faults on the small
  disagg N-1 prefill shape. The model author already flagged an "Inductor
  correctness issue with the row-strided G view" and padded in_proj to dodge it;
  the tiny disagg batch defeats that workaround.

  Not a quant issue: mxfp4 falls all LinearBase layers back to
  UnquantizedLinearMethod (only MoE experts are MXFP4); the KDA projections are
  plain bf16 in the checkpoint. The fault is the strided-view GEMM input.

FIX: `self.f_b_proj(f_a.contiguous())` -- materialize a dense f_a. Negligible cost
(f_a is [tokens, head_dim=128]).

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_kda_fa_contiguous.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-fa] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src
    if "f_a.contiguous()" in src:
        print("[k3-fa] already applied.")
        return 0

    anchor = "        g1 = self.f_b_proj(f_a)[0]\n"
    repl = (
        "        # k3-kda: f_a is a non-contiguous slice from the padded in_proj\n"
        "        # split; the strided view faults f_b_proj's GEMM on the small disagg\n"
        "        # prefill shape. Materialize a dense f_a first.\n"
        "        g1 = self.f_b_proj(f_a.contiguous())[0]\n"
    )
    if anchor not in src:
        print("[k3-fa] WARN anchor not found -- not applied.")
        return 0
    src = src.replace(anchor, repl, 1)
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-fa] ERROR compile: {e}", file=sys.stderr)
        return 1
    print("[k3-fa] f_a.contiguous() before f_b_proj applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
