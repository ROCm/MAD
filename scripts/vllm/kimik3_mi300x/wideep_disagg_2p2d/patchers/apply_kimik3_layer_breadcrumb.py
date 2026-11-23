#!/usr/bin/env python3
"""Per-layer synced breadcrumbs inside the K3 (AMD) model forward.

Gated by K3_FWD_BREADCRUMB=1. Pinpoints whether the disagg-producer GPU fault is
in embed or a specific decoder layer (and its attn type: KDA vs MLA). Each point
is followed by torch.cuda.synchronize() so the async fault surfaces at the exact
layer. Edits vllm/models/kimi_k3/amd/linear.py KimiLinearModel.forward.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_layer_breadcrumb.py <vllm_install_dir>
"""
import os
import sys

RELS = [
    "models/kimi_k3/amd/linear.py",
    "../vllm/models/kimi_k3/amd/linear.py",
]


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    root = sys.argv[1]
    path = None
    for rel in RELS:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        # search
        for base, _, files in os.walk(os.path.join(root, "models")):
            if "linear.py" in files and "kimi_k3" in base and base.endswith("amd"):
                path = os.path.join(base, "linear.py")
                break
    if path is None or not os.path.isfile(path):
        print("[k3-layerbc] amd/linear.py not found -- skip.")
        return 0
    src = open(path).read()
    orig = src
    if "[k3-bc] model.forward START" in src:
        print("[k3-layerbc] already applied.")
        return 0

    embed_anchor = (
        "        if get_pp_group().is_first_rank:\n"
        "            if inputs_embeds is not None:\n"
        "                hidden_states = inputs_embeds\n"
        "            else:\n"
        "                hidden_states = self.embed_input_ids(input_ids)\n"
        "            residual = None\n"
        "        else:\n"
        "            assert intermediate_tensors is not None\n"
        "            hidden_states = intermediate_tensors[\"hidden_states\"]\n"
        "            residual = intermediate_tensors[\"residual\"]\n"
    )
    embed_repl = (
        "        import os as _os_k3ml\n"
        "        _k3ml = _os_k3ml.environ.get(\"K3_FWD_BREADCRUMB\", \"0\") == \"1\"\n"
        "        if _k3ml:\n"
        "            import torch as _t_k3ml, logging as _lg_k3ml\n"
        "            _t_k3ml.cuda.synchronize()\n"
        "            _lg_k3ml.getLogger(__name__).warning(\n"
        "                \"[k3-bc] model.forward START input_ids=%s inputs_embeds=%s\",\n"
        "                (None if input_ids is None else tuple(input_ids.shape)),\n"
        "                (None if inputs_embeds is None else tuple(inputs_embeds.shape)))\n"
        + embed_anchor +
        "        if _k3ml:\n"
        "            import torch as _t_k3ml, logging as _lg_k3ml\n"
        "            _t_k3ml.cuda.synchronize()\n"
        "            _lg_k3ml.getLogger(__name__).warning(\"[k3-bc] embed DONE+SYNCED\")\n"
    )

    layer_anchor = (
        "                hidden_states, residual = layer(\n"
        "                    positions=positions,\n"
        "                    hidden_states=hidden_states,\n"
        "                    residual=residual,\n"
        "                )\n"
    )
    layer_repl = (
        "                if _k3ml:\n"
        "                    import torch as _t_k3ml, logging as _lg_k3ml\n"
        "                    _t_k3ml.cuda.synchronize()\n"
        "                    _lg_k3ml.getLogger(__name__).warning(\n"
        "                        \"[k3-bc] layer %d START (type=%s)\", layer_idx,\n"
        "                        type(getattr(layer, \"self_attn\", layer)).__name__)\n"
        + layer_anchor
    )

    if embed_anchor not in src:
        print("[k3-layerbc] WARN embed anchor not found -- skip.")
        return 0
    src = src.replace(embed_anchor, embed_repl, 1)
    if layer_anchor in src:
        src = src.replace(layer_anchor, layer_repl, 1)
    else:
        print("[k3-layerbc] WARN layer anchor not found (embed bc still added).")

    # attn_res branch loop (K3 uses this when attn_res_block_size is set)
    ar_anchor = (
        "            hidden_states, residual = layer(\n"
        "                positions=positions,\n"
        "                hidden_states=hidden_states,\n"
        "                residual=residual,\n"
        "            )\n"
        "            if (layer_idx + 1) in self.aux_hidden_state_layers:\n"
    )
    ar_repl = (
        "            if _k3ml:\n"
        "                import torch as _t_k3ml, logging as _lg_k3ml\n"
        "                _t_k3ml.cuda.synchronize()\n"
        "                _lg_k3ml.getLogger(__name__).warning(\n"
        "                    \"[k3-bc] attnres layer %d START (type=%s)\", layer_idx,\n"
        "                    type(getattr(layer, \"self_attn\", layer)).__name__)\n"
        + ar_anchor
    )
    if ar_anchor in src:
        src = src.replace(ar_anchor, ar_repl, 1)
    else:
        print("[k3-layerbc] WARN attn_res loop anchor not found.")

    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-layerbc] ERROR compile: {e}", file=sys.stderr)
        return 1
    print("[k3-layerbc] added embed + per-layer synced breadcrumbs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
