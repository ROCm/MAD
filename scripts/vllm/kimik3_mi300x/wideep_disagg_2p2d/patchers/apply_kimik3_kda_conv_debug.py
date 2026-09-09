#!/usr/bin/env python3
"""Log KDA-forward entry metadata (debug) for the 2P/2D disagg producer fault.

Gated by K3_KDA_CONV_DEBUG=1. Injected right after `m = attn_metadata_narrowed`
in kimi_gdn_linear_attn (fires for EVERY KDA layer, all sub-paths, before any
indexing). Prints conv_state/recurrent_state shapes, num_prefills/decodes,
num_actual_tokens, non_spec_state_indices min/max/count, has_initial_state
any/sum, and whether spec masks are set. The Python log flushes before the async
GPU kernel faults, so it captures the offending values on the N-1 partial prefill.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_kda_conv_debug.py <vllm_install_dir>
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
        print(f"[k3-kda-convdbg] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src
    if "k3-kda entry" in src:
        print("[k3-kda-convdbg] already applied.")
        return 0

    anchor = (
        "        m = attn_metadata_narrowed\n"
        "        has_initial_state = m.has_initial_state\n"
        "        non_spec_query_start_loc = m.non_spec_query_start_loc\n"
        "        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor\n"
    )
    inject = (
        "        m = attn_metadata_narrowed\n"
        "        import os as _os_k3e\n"
        "        if _os_k3e.environ.get(\"K3_KDA_CONV_DEBUG\", \"0\") == \"1\":\n"
        "            try:\n"
        "                import logging as _lg_k3e\n"
        "                _cs = self.kv_cache[0] if self.kv_cache is not None else None\n"
        "                _rs = self.kv_cache[1] if self.kv_cache is not None else None\n"
        "                _nsi = m.non_spec_state_indices_tensor\n"
        "                _his = m.has_initial_state\n"
        "                _lg_k3e.getLogger(__name__).warning(\n"
        "                    \"[k3-kda entry] layer=%s conv_state.shape=%s recur.shape=%s \"\n"
        "                    \"n_prefills=%s n_decodes=%s num_actual=%s \"\n"
        "                    \"nsi(min/max/n)=%s/%s/%s has_init(any/sum)=%s/%s \"\n"
        "                    \"spec_masks=%s\",\n"
        "                    getattr(self, \"prefix\", \"?\"),\n"
        "                    (tuple(_cs.shape) if _cs is not None else None),\n"
        "                    (tuple(_rs.shape) if _rs is not None else None),\n"
        "                    getattr(m, \"num_prefills\", None), getattr(m, \"num_decodes\", None),\n"
        "                    getattr(m, \"num_actual_tokens\", None),\n"
        "                    (int(_nsi.min()) if _nsi is not None and _nsi.numel() else None),\n"
        "                    (int(_nsi.max()) if _nsi is not None and _nsi.numel() else None),\n"
        "                    (int(_nsi.numel()) if _nsi is not None else None),\n"
        "                    (bool(_his.any()) if _his is not None else None),\n"
        "                    (int(_his.sum()) if _his is not None else None),\n"
        "                    (None if m.spec_sequence_masks is None else True),\n"
        "                )\n"
        "            except Exception as _e_k3e:\n"
        "                import logging as _lg_k3e\n"
        "                _lg_k3e.getLogger(__name__).warning(\"[k3-kda entry] err %s\", _e_k3e)\n"
        "        has_initial_state = m.has_initial_state\n"
        "        non_spec_query_start_loc = m.non_spec_query_start_loc\n"
        "        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor\n"
    )
    if anchor not in src:
        print("[k3-kda-convdbg] WARN: entry anchor not found -- not applied.")
        return 0
    src = src.replace(anchor, inject, 1)
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[k3-kda-convdbg] ERROR: compile failed: {e}", file=sys.stderr)
        open(path, "w").write(orig)
        return 1
    print("[k3-kda-convdbg] added KDA-entry debug logging (K3_KDA_CONV_DEBUG=1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
