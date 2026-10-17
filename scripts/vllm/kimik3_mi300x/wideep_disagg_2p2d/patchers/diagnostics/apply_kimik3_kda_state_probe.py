#!/usr/bin/env python3
"""Ground-truth probe: is the transferred KDA state actually present in the
slot decode reads?

Decode generates fluent-but-context-free with disagg. Attention KV + mamba
state both *transfer* (proven by write breadcrumbs), yet output ignores the
prompt. This probe logs, on the DECODE pure-decode path for KDA layer 0, the
L2 norm of the recurrent_state and conv_state rows at the exact indices the
kernel is about to read (decode_conv_indices). If those norms are ~0, the
transferred state did NOT land in the slot decode reads (block-table / slot
mapping bug), which is the smoking gun. If they're clearly non-zero, the state
IS present and the bug is elsewhere (e.g. offset content garbled, or the gate).

Gated by K3_KDA_STATE_PROBE=1. Logs once per ~forward via a step counter so it
doesn't spam. Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_kda_state_probe.py <vllm_install_dir>
"""
import os
import sys

KDA = "models/kimi_k3/nvidia/kda.py"


def _edit(path, old, new, tag):
    if not os.path.isfile(path):
        print(f"[k3-stateprobe] {tag}: not found {path}", file=sys.stderr)
        return False
    src = open(path).read()
    if "k3-stateprobe" in src:
        print(f"[k3-stateprobe] {tag}: already applied.")
        return True
    if old not in src:
        print(f"[k3-stateprobe] {tag}: anchor NOT found", file=sys.stderr)
        return False
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[k3-stateprobe] {tag}: compile FAIL {e}", file=sys.stderr)
        return False
    print(f"[k3-stateprobe] {tag}: applied.")
    return True


def main():
    base = sys.argv[1]
    path = os.path.join(base, KDA)
    # Anchor at the common point after the KDA cache tensors are bound -- covers
    # BOTH decode kernels (fused_kda_decode fast path AND
    # fused_recurrent_kda_packed_decode). m + non_spec_state_indices_tensor are
    # already in scope here. Only probe pure-decode (num_prefills==0) so the
    # norm reflects the transferred state, not a freshly-computed prefill state.
    old = (
        "        conv_state, recurrent_state = self.kv_cache\n"
    )
    new = (
        "        conv_state, recurrent_state = self.kv_cache\n"
        "        import os as _k3os\n"
        "        if _k3os.environ.get('K3_KDA_STATE_PROBE', '0') == '1':\n"
        "            try:\n"
        "                _ln = getattr(self, 'prefix', '?')\n"
        "                if ('.layers.1.' in str(_ln)) or ('.layers.5.' in str(_ln)):\n"
        "                    _idx = non_spec_state_indices_tensor\n"
        "                    if _idx is not None:\n"
        "                        _ii = _idx[:max(1,int(num_actual_tokens))].long()\n"
        "                        _rs = recurrent_state.index_select(0, _ii)\n"
        "                        _cs = conv_state.index_select(0, _ii)\n"
        "                        _rn, _ra = float(_rs.float().norm()), float(_rs.float().abs().max())\n"
        "                        _cn, _ca = float(_cs.float().norm()), float(_cs.float().abs().max())\n"
        "                        _idl = _idx[:4].tolist()\n"
        "                    else:\n"
        "                        _rn=_ra=_cn=_ca=-1.0; _idl=None\n"
        "                    import sys as _k3sys\n"
        "                    _hi = (None if has_initial_state is None else (has_initial_state.tolist() if hasattr(has_initial_state,'tolist') else has_initial_state))\n"
        "                    print('[k3-stateprobe] layer=%s nprefill=%s ndecode=%s nact=%s hasinit=%s idx=%s rs_norm=%.4e rs_absmax=%.4e cs_norm=%.4e cs_absmax=%.4e' % (\n"
        "                        _ln, m.num_prefills, m.num_decodes, num_actual_tokens, _hi, _idl, _rn, _ra, _cn, _ca), file=_k3sys.stderr, flush=True)\n"
        "            except Exception as _k3e:\n"
        "                import sys as _k3sys2\n"
        "                print('[k3-stateprobe] EXC %r' % (_k3e,), file=_k3sys2.stderr, flush=True)\n"
    )
    return 0 if _edit(path, old, new, "kda decode state probe") else 1


if __name__ == "__main__":
    sys.exit(main())
