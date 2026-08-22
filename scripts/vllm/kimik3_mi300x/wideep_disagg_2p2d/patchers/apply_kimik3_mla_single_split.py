#!/usr/bin/env python3
"""TEST/FIX: force num_kv_splits=1 in TRITON_MLA forward_mqa (decode MLA attention).

Hypothesis (UPDATE 25): the disagg recall failure is NON-DETERMINISTIC at greedy
temp=0 because the TRITON_MLA decode kernel splits the KV sequence into
num_kv_splits>1 partials (attn_logits in a shared workspace) and merges them with a
NON-DETERMINISTIC reduction. On a remote-prefill request the paged KV block's tail
(beyond the live tokens) is UNINITIALIZED (RDMA writes only live bytes; a local
prefill inits the whole block), so some splits read garbage and the non-det merge
yields wildly varying wrong tokens. vLLM's own code sets num_kv_splits=1 under
VLLM_BATCH_INVARIANT "to ensure deterministic reduction" — but that global env is
"not supported" for our MoE/EP/MXFP4 config, so we force it locally here.

This edits vllm/v1/attention/backends/mla/triton_mla.py forward_mqa: replace the
num_kv_splits computation with a forced 1 (gated by env K3_MLA_SINGLE_SPLIT, default
ON when the patch is applied so a simple relaunch tests it). Single split = one
contiguous KV scan per query, deterministic, and reads only [0, seq_len) so no
cross-split garbage merge. Perf: one split is slower for very long seqs (less
parallelism) but correct; if it fixes recall we then do the perf-preserving fix
(zero-init decode KV blocks before RDMA write, keep multi-split).

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_mla_single_split.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/attention/backends/mla/triton_mla.py"
MARK = "k3-single-split"


def main():
    base = sys.argv[1]
    path = os.path.join(base, REL)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr)
        return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied.")
        return 0
    old = (
        "        # For batch invariance, use only 1 split to ensure deterministic reduction\n"
        "        if envs.VLLM_BATCH_INVARIANT:\n"
        "            num_kv_splits = 1\n"
        "        else:\n"
        "            num_kv_splits = _compute_num_kv_splits(\n"
        "                attn_metadata.max_seq_len, self._sm_count\n"
        "            )\n"
    )
    new = (
        "        # " + MARK + ": force single split on the disagg decode path so the\n"
        "        # split-KV reduction is deterministic AND never merges an\n"
        "        # uninitialized-page-tail partial (remote-prefill KV blocks have\n"
        "        # uninitialized tails). K3_MLA_SINGLE_SPLIT=0 restores multi-split.\n"
        "        import os as _k3ssos\n"
        "        if _k3ssos.environ.get('K3_MLA_SINGLE_SPLIT', '1') != '0':\n"
        "            num_kv_splits = 1\n"
        "        elif envs.VLLM_BATCH_INVARIANT:\n"
        "            num_kv_splits = 1\n"
        "        else:\n"
        "            num_kv_splits = _compute_num_kv_splits(\n"
        "                attn_metadata.max_seq_len, self._sm_count\n"
        "            )\n"
    )
    if old not in src:
        print(f"[{MARK}] anchor NOT found", file=sys.stderr)
        return 1
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[{MARK}] compile FAIL {e}", file=sys.stderr)
        return 1
    print(f"[{MARK}] applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
