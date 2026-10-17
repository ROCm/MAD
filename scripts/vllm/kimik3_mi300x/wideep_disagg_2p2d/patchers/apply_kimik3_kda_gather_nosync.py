#!/usr/bin/env python3
"""PERF/HANG FIX: make the KDA gather_initial_states OOB guard sync-free.

The k3-kda guard in
`vllm/model_executor/layers/mamba/ops/gather_initial_states.py` clamps state
indices into range (sync-free, on-device — correct) but then runs a diagnostic
`if bool((indices >= n).any()) or bool((indices < 0).any()):` block purely to log
a warning. Each `bool(...any())` forces a device->CPU sync (`_local_scalar_dense`
-> `memcpy_and_sync`), i.e. a full-stream drain — **per KDA layer, per prefill
chunk**. With 69 KDA layers and chunked prefill (~365 chunks at 750K /
batched=2048) that is ~25k forced syncs, which stalls so hard it presents as a
hang for contexts > ~500K (py-spy'd: DP rank stuck in this sync while all other
DP ranks wait at the batch-coordination all_reduce).

The clamp already guarantees a valid GPU address, so the diagnostic sync is pure
overhead. Gate it behind `K3_KDA_GATHER_LOG=1` (default OFF) so the hot path never
syncs. Correctness is unchanged (indices are still clamped).

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_kda_gather_nosync.py <vllm_install_dir>
"""
import os, sys

MARK = "k3-kda-nosync"
REL = "model_executor/layers/mamba/ops/gather_initial_states.py"

OLD = (
"    if bool((indices >= _n_state_blocks).any()) or bool((indices < 0).any()):\n"
"        import logging as _lg\n"
"        _bad = indices[(indices >= _n_state_blocks) | (indices < 0)]\n"
"        _lg.getLogger(__name__).warning(\n"
"            \"[k3-kda gather] clamped %d out-of-range state idx (n_blocks=%d, \"\n"
"            \"sample=%s); disagg producer prefill likely mis-flagged initial state.\",\n"
"            int(_bad.numel()), _n_state_blocks, _bad[:8].tolist(),\n"
"        )\n"
)
NEW = (
"    import os as _k3os  # " + MARK + ": clamp above is sync-free + safe; the\n"
"    # bool(...any()) diagnostic below forces a device->CPU sync EVERY call (per\n"
"    # KDA layer, per chunk) -> O(layers*chunks) stalls that hang ctx > ~500K.\n"
"    # Gate behind K3_KDA_GATHER_LOG=1 (default OFF); hot path never syncs.\n"
"    if _k3os.environ.get('K3_KDA_GATHER_LOG', '0') == '1':\n"
"        if bool((indices >= _n_state_blocks).any()) or bool((indices < 0).any()):\n"
"            import logging as _lg\n"
"            _bad = indices[(indices >= _n_state_blocks) | (indices < 0)]\n"
"            _lg.getLogger(__name__).warning(\n"
"                \"[k3-kda gather] clamped %d out-of-range state idx (n_blocks=%d, \"\n"
"                \"sample=%s); disagg producer prefill likely mis-flagged initial state.\",\n"
"                int(_bad.numel()), _n_state_blocks, _bad[:8].tolist(),\n"
"            )\n"
)


def main():
    if len(sys.argv) < 2:
        print(f"[{MARK}] usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 1
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr); return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied."); return 0
    if OLD not in src:
        print(f"[{MARK}] ANCHOR MISSING", file=sys.stderr); return 1
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    try:
        import py_compile; py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[{MARK}] compile FAIL {e}", file=sys.stderr); return 1
    print(f"[{MARK}] applied (KDA gather sync-free).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
