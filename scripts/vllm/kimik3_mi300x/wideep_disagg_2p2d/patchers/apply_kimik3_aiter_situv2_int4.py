#!/usr/bin/env python3
"""Backport AITER PR #4471 (SiTUv2 in packed-int4 MoE stage1) SURGICALLY into the
grafted aiter's own files (keeps its imports; HEAD file-replace pulled missing
modules buffer_ops/vector).

WHY: gfx942 requantizes K3 MoE to packed-int4; the grafted aiter's FlyDSL stage1
kernel (compile_moe_gemm1, _abi3) hardcodes SiLU and ignores the requested SiTUv2
-> K3 MoE silently computes SiLU -> gibberish. The upper plumbing already threads
act/situ_beta/situ_linear_beta (compile_flydsl_moe_stage1 has the params, fused_moe
passes them); only (a) the int4_bf16 branch's compile_moe_gemm1 call doesn't forward
them, and (b) the kernel codegen lacks the SiTUv2 epilogue. This applies exactly
those two things as in-place edits.

5 anchor-based edits, idempotent, py_compile-checked. Also wipes stale _abi3 JIT.
Usage: apply_kimik3_aiter_situv2_int4.py <vllm_install_dir>  (aiter resolved from import)
"""
import os
import shutil
import sys


def _edit(path, subs, tag):
    src = open(path).read()
    orig = src
    for old, new in subs:
        if new.split("\n", 1)[0] in src and old not in src:
            # already applied (new present, old gone)
            continue
        if old not in src:
            print(f"[k3-situv2] {tag}: anchor NOT found:\n  {old[:70]!r}", file=sys.stderr)
            return False
        src = src.replace(old, new, 1)
    if src == orig:
        print(f"[k3-situv2] {tag}: no change (already applied).")
        return True
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-situv2] {tag}: compile failed, rolled back: {e}", file=sys.stderr)
        return False
    print(f"[k3-situv2] {tag}: applied.")
    return True


def main():
    try:
        import aiter
        A = os.path.dirname(aiter.__file__)
    except Exception as e:
        print(f"[k3-situv2] cannot import aiter: {e} -- skip.")
        return 0

    mg = os.path.join(A, "ops/flydsl/kernels/moe_gemm_2stage.py")
    mk = os.path.join(A, "ops/flydsl/moe_kernels.py")
    if not (os.path.isfile(mg) and os.path.isfile(mk)):
        print("[k3-situv2] aiter flydsl files not found -- skip.")
        return 0

    # idempotency: kernel already has _abi4 + situv2
    if "_abi4" in open(mg).read() and "def situv2(" in open(mg).read():
        print("[k3-situv2] already applied (_abi4 + situv2).")
        return 0

    # ---- moe_kernels.py: forward act/situ_beta/situ_linear_beta to the int4_bf16
    #      compile_moe_gemm1 call ----
    mk_subs = [(
        "            use_cshuffle_epilog=_use_cshuffle,\n"
        "            scale_is_bf16=True,\n"
        "            k_batch=k_batch,\n"
        "        )\n"
        "    else:\n"
        "        raise ValueError(\n"
        "            f\"Unsupported stage1 dtype combination: a_dtype={a_dtype}, b_dtype={b_dtype}\"\n",
        "            use_cshuffle_epilog=_use_cshuffle,\n"
        "            scale_is_bf16=True,\n"
        "            k_batch=k_batch,\n"
        "            act=act,\n"
        "            situ_beta=situ_beta,\n"
        "            situ_linear_beta=situ_linear_beta,\n"
        "        )\n"
        "    else:\n"
        "        raise ValueError(\n"
        "            f\"Unsupported stage1 dtype combination: a_dtype={a_dtype}, b_dtype={b_dtype}\"\n",
    )]
    if not _edit(mk, mk_subs, "moe_kernels.py"):
        return 1

    # ---- moe_gemm_2stage.py: 4 edits ----
    mg_subs = []

    # (1) helper fn before compile_moe_gemm1 + the 3 new params
    mg_subs.append((
        "@functools.lru_cache(maxsize=1024)\n"
        "def compile_moe_gemm1(\n",
        "def _stage1_activation_module_tag(\n"
        "    act: str, situ_beta: float, situ_linear_beta: float\n"
        ") -> str:\n"
        "    \"\"\"Filesystem-safe cache-key suffix for stage1 activation code.\"\"\"\n"
        "    if act == \"silu\":\n"
        "        return \"_silu\"\n"
        "    def float_tag(value: float) -> str:\n"
        "        return float(value).hex().replace(\"-\", \"m\").replace(\"+\", \"p\").replace(\".\", \"d\")\n"
        "    return f\"_situv2_sb{float_tag(situ_beta)}_slb{float_tag(situ_linear_beta)}\"\n"
        "\n"
        "\n"
        "@functools.lru_cache(maxsize=1024)\n"
        "def compile_moe_gemm1(\n",
    ))
    mg_subs.append((
        "    scale_is_bf16: bool = False,\n"
        "    k_batch: int = 1,\n"
        "):\n"
        "    \"\"\"Compile stage1 kernel",
        "    scale_is_bf16: bool = False,\n"
        "    k_batch: int = 1,\n"
        "    act: str = \"silu\",\n"
        "    situ_beta: float = 1.0,\n"
        "    situ_linear_beta: float = 1.0,\n"
        "):\n"
        "    if act not in (\"silu\", \"situv2\"):\n"
        "        raise ValueError(f\"act must be 'silu' or 'situv2', got {act!r}\")\n"
        "    if act == \"situv2\":\n"
        "        if situ_beta <= 0.0:\n"
        "            raise ValueError(f\"situ_beta must be > 0, got {situ_beta!r}\")\n"
        "        if situ_linear_beta <= 0.0:\n"
        "            raise ValueError(f\"situ_linear_beta must be > 0, got {situ_linear_beta!r}\")\n"
        "    \"\"\"Compile stage1 kernel",
    ))

    # (2) module_name tag: _abi3 -> _abi4 + _act_tag, and bind the name
    mg_subs.append((
        "    _split_k_tag = f\"_splitk{k_batch}\" if _is_splitk else \"\"\n"
        "    (\n"
        "        f\"mfma_moe1_{in_dtype}_{out_dtype}_{epilog_tag}\"\n"
        "        f\"_t{tile_m}x{tile_n}x{tile_k}\"\n"
        "        f\"{_gs_tag}{scale_tag}{_split_k_tag}\"\n"
        "        f\"_abi3\"  # also mask sentinel token ids on loads (X/scale_x) to avoid illegal address faults\n"
        "    ).replace(\"-\", \"_\")\n",
        "    _split_k_tag = f\"_splitk{k_batch}\" if _is_splitk else \"\"\n"
        "    _act_tag = _stage1_activation_module_tag(act, situ_beta, situ_linear_beta)\n"
        "    module_name = (\n"
        "        f\"mfma_moe1_{in_dtype}_{out_dtype}_{epilog_tag}\"\n"
        "        f\"_t{tile_m}x{tile_n}x{tile_k}\"\n"
        "        f\"{_gs_tag}{scale_tag}{_split_k_tag}{_act_tag}\"\n"
        "        f\"_abi4\"  # also mask sentinel token ids on loads (X/scale_x) to avoid illegal address faults\n"
        "    ).replace(\"-\", \"_\")\n",
    ))
    # (2b) name the kernel (so distinct activations don't collide in the JIT cache)
    mg_subs.append((
        "        @flyc.kernel\n"
        "        def moe_gemm1(\n",
        "        @flyc.kernel(name=module_name)\n"
        "        def moe_gemm1(\n",
    ))

    # (3) rewrite the silu() def into sigmoid/silu/situv2/apply_activation
    mg_subs.append((
        "            def silu(x):\n"
        "                # device fast path:\n"
        "                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32\n"
        "                #   sig = rcp(1 + emu)                   -> v_rcp_f32\n"
        "                #   y = x * sig\n"
        "                #\n"
        "                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup\n"
        "                # sequences that introduce extra compares/cndmasks.\n"
        "                t = x * (-1.4426950408889634)  # -log2(e)\n"
        "                emu = rocdl.exp2(T.f32, t)\n"
        "                den = 1.0 + emu\n"
        "                sig = rocdl.rcp(T.f32, den)\n"
        "                return x * sig\n",
        "            def sigmoid(x):\n"
        "                t = x * (-1.4426950408889634)  # -log2(e)\n"
        "                emu = rocdl.exp2(T.f32, t)\n"
        "                den = 1.0 + emu\n"
        "                return rocdl.rcp(T.f32, den)\n"
        "\n"
        "            def silu(x):\n"
        "                return x * sigmoid(x)\n"
        "\n"
        "            def situv2(gate, up):\n"
        "                gate_tanh = 2.0 * sigmoid(2.0 * (gate / situ_beta)) - 1.0\n"
        "                up_tanh = 2.0 * sigmoid(2.0 * (up / situ_linear_beta)) - 1.0\n"
        "                situ_gate = situ_beta * gate_tanh * sigmoid(gate)\n"
        "                situ_up = situ_linear_beta * up_tanh\n"
        "                return situ_gate * situ_up\n"
        "\n"
        "            def apply_activation(gate, up):\n"
        "                if const_expr(act == \"silu\"):\n"
        "                    return silu(gate) * up\n"
        "                return situv2(gate, up)\n",
    ))

    # (4) both apply sites: y = silu(vg) * vu -> y = apply_activation(vg, vu)
    # replace_all-style: do it manually since there are two identical occurrences
    src = open(mg).read()
    orig = src
    src = src.replace("y = silu(vg) * vu", "y = apply_activation(vg, vu)")
    if src != orig:
        open(mg, "w").write(src)

    if not _edit(mg, mg_subs, "moe_gemm_2stage.py"):
        return 1

    # final compile check both
    try:
        import py_compile
        py_compile.compile(mg, doraise=True)
        py_compile.compile(mk, doraise=True)
    except Exception as e:
        print(f"[k3-situv2] final compile failed: {e}", file=sys.stderr)
        return 1

    # wipe stale _abi3 JIT
    for cache in ("/opt/vllm_cache/aiter", "/opt/vllm_cache/aiter_jit", "/root/.aiter"):
        shutil.rmtree(cache, ignore_errors=True)

    print("[k3-situv2] backported #4471 SiTUv2 packed-int4 stage1 in-place (_abi4); "
          "wiped stale JIT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
