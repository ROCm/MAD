#!/usr/bin/env python3
"""Synced breadcrumbs INSIDE KimiGatedDeltaNetAttention.forward to localize the
fault within KDA layer 0 (the last surviving breadcrumb was 'attnres layer 0 START
type=KimiGatedDeltaNetAttention', and the KDA-entry debug at line ~412 never fired,
so the fault is in the projections (334-377) or _forward metadata setup).

Gated by K3_FWD_BREADCRUMB=1. Prints hidden_states shape/contiguity + in_proj
weight shape before the first GEMM, and after it. Edits
vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_kda_internal_bc.py <vllm_install_dir>
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
        print(f"[k3-kdabc] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src
    if "[k3-bc] KDA fwd START" in src:
        print("[k3-kdabc] already applied.")
        return 0

    anchor = (
        "        num_tokens = hidden_states.size(0)\n"
        "        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]\n"
    )
    repl = (
        "        num_tokens = hidden_states.size(0)\n"
        "        import os as _os_k3k\n"
        "        _k3k = _os_k3k.environ.get(\"K3_FWD_BREADCRUMB\", \"0\") == \"1\"\n"
        "        if _k3k:\n"
        "            import torch as _t_k3k, logging as _lg_k3k\n"
        "            _t_k3k.cuda.synchronize()\n"
        "            _lg_k3k.getLogger(__name__).warning(\n"
        "                \"[k3-bc] KDA fwd START prefix=%s hs=%s contig=%s in_proj_w=%s\",\n"
        "                getattr(self, \"prefix\", \"?\"), tuple(hidden_states.shape),\n"
        "                hidden_states.is_contiguous(),\n"
        "                tuple(self.in_proj_qkvgfab.weight.shape)\n"
        "                if hasattr(self.in_proj_qkvgfab, \"weight\") else \"?\")\n"
        "        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]\n"
        "        if _k3k:\n"
        "            import torch as _t_k3k, logging as _lg_k3k\n"
        "            _t_k3k.cuda.synchronize()\n"
        "            _lg_k3k.getLogger(__name__).warning(\"[k3-bc] KDA in_proj DONE+SYNCED\")\n"
    )
    if anchor not in src:
        print("[k3-kdabc] WARN anchor not found -- skip.")
        return 0
    src = src.replace(anchor, repl, 1)

    # around self._forward(...) + o_proj
    fwd_anchor = (
        "        self._forward(\n"
        "            mixed_qkv=mixed_qkv,\n"
        "            g1=g1,\n"
        "            g2=g2,\n"
        "            beta=beta,\n"
        "            core_attn_out=core_attn_out,\n"
        "        )\n"
        "        core_attn_out = rearrange(core_attn_out, \"1 n h d -> n (h d)\")\n"
        "        output[:] = self.o_proj(core_attn_out)[0]\n"
    )
    fwd_repl = (
        "        if _k3k:\n"
        "            import torch as _t_k3k, logging as _lg_k3k\n"
        "            _t_k3k.cuda.synchronize()\n"
        "            _lg_k3k.getLogger(__name__).warning(\n"
        "                \"[k3-bc] KDA projections DONE -> _forward; mixed_qkv=%s g1=%s g2=%s beta=%s\",\n"
        "                tuple(mixed_qkv.shape), tuple(g1.shape), tuple(g2.shape), tuple(beta.shape))\n"
        "        self._forward(\n"
        "            mixed_qkv=mixed_qkv,\n"
        "            g1=g1,\n"
        "            g2=g2,\n"
        "            beta=beta,\n"
        "            core_attn_out=core_attn_out,\n"
        "        )\n"
        "        if _k3k:\n"
        "            import torch as _t_k3k, logging as _lg_k3k\n"
        "            _t_k3k.cuda.synchronize()\n"
        "            _lg_k3k.getLogger(__name__).warning(\"[k3-bc] KDA _forward DONE+SYNCED\")\n"
        "        core_attn_out = rearrange(core_attn_out, \"1 n h d -> n (h d)\")\n"
        "        output[:] = self.o_proj(core_attn_out)[0]\n"
        "        if _k3k:\n"
        "            import torch as _t_k3k, logging as _lg_k3k\n"
        "            _t_k3k.cuda.synchronize()\n"
        "            _lg_k3k.getLogger(__name__).warning(\"[k3-bc] KDA o_proj DONE+SYNCED\")\n"
    )
    if fwd_anchor in src:
        src = src.replace(fwd_anchor, fwd_repl, 1)
    else:
        print("[k3-kdabc] WARN _forward anchor not found.")

    # sub-projection breadcrumbs (split / g_a-g_b / f_b)
    gp_anchor = (
        "            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]\n"
        "\n"
        "        g1 = self.f_b_proj(f_a)[0]\n"
        "        beta = beta.unsqueeze(0)\n"
    )
    gp_repl = (
        "            if _k3k:\n"
        "                import torch as _t_k3k, logging as _lg_k3k\n"
        "                _t_k3k.cuda.synchronize()\n"
        "                _lg_k3k.getLogger(__name__).warning(\n"
        "                    \"[k3-bc] KDA split DONE mixed_qkv=%s f_a=%s beta=%s\",\n"
        "                    tuple(mixed_qkv.shape), tuple(f_a.shape), tuple(beta.shape))\n"
        "            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]\n"
        "            if _k3k:\n"
        "                import torch as _t_k3k, logging as _lg_k3k\n"
        "                _t_k3k.cuda.synchronize()\n"
        "                _lg_k3k.getLogger(__name__).warning(\"[k3-bc] KDA g_a/g_b_proj DONE\")\n"
        "\n"
        "        g1 = self.f_b_proj(f_a)[0]\n"
        "        if _k3k:\n"
        "            import torch as _t_k3k, logging as _lg_k3k\n"
        "            _t_k3k.cuda.synchronize()\n"
        "            _lg_k3k.getLogger(__name__).warning(\"[k3-bc] KDA f_b_proj DONE g1=%s\", tuple(g1.shape))\n"
        "        beta = beta.unsqueeze(0)\n"
    )
    if gp_anchor in src:
        src = src.replace(gp_anchor, gp_repl, 1)
    else:
        print("[k3-kdabc] WARN sub-proj anchor not found.")

    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-kdabc] ERROR compile: {e}", file=sys.stderr)
        return 1
    print("[k3-kdabc] added KDA-internal synced breadcrumbs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
