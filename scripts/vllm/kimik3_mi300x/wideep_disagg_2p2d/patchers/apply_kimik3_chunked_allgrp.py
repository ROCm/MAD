#!/usr/bin/env python3
"""FIX: accumulate ALL kv-cache groups' block ids across CHUNKED PREFILL.

Bug: MoRIIO connector accumulates only the ATTENTION-group block list across
prefill chunks (updated_blocks = existing + new_block_ids[0]); the per-group
all_group_block_ids (used by k3-group-routing) and mamba block ids are captured
ONCE at first chunk (update_state_after_alloc) and never grown. So for prompts
> max_num_batched_tokens (multi-chunk prefill), only the FIRST chunk's KV blocks
are advertised/transferred -> decode gets only ~first-chunk tokens of KV ->
needle beyond ~chunk1 lost -> long-context NIAH fails.

Fix: in build_connector_meta's scheduled_cached_reqs loop, accumulate EVERY
group's new blocks (new_block_ids is a per-group tuple) into
self._reqs_save_allgrp[req_id] so the final-chunk add_new_req carries the FULL
per-group block lists. Depends on k3-group-routing (which adds _reqs_save_allgrp
+ all_group_block_ids threading). Gated by K3_GROUP_ROUTING via that patcher.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_chunked_allgrp.py <vllm_install_dir>
"""
import os, sys
MARK = "k3-chunked-allgrp"
REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"

OLD = (
"                if new_block_ids is not None:\n"
"                    block_ids = new_block_ids[0]\n"
)
NEW = (
"                if new_block_ids is not None:\n"
"                    block_ids = new_block_ids[0]\n"
"                    # " + MARK + ": accumulate ALL groups' new blocks across\n"
"                    # chunked prefill so all_group_block_ids grows with the\n"
"                    # request (not frozen at chunk 1). new_block_ids is a\n"
"                    # per-group tuple.\n"
"                    try:\n"
"                        if req_id in getattr(self, '_reqs_save_allgrp', {}):\n"
"                            _k3ca_cur = self._reqs_save_allgrp[req_id]\n"
"                            _k3ca_new = []\n"
"                            for _k3ca_gi in range(len(_k3ca_cur)):\n"
"                                _k3ca_add = (\n"
"                                    list(new_block_ids[_k3ca_gi])\n"
"                                    if _k3ca_gi < len(new_block_ids) else []\n"
"                                )\n"
"                                _k3ca_new.append(\n"
"                                    list(_k3ca_cur[_k3ca_gi]) + _k3ca_add\n"
"                                )\n"
"                            self._reqs_save_allgrp[req_id] = _k3ca_new\n"
"                            if req_id in getattr(self, '_reqs_save_mamba', {}) and len(_k3ca_new) > 1:\n"
"                                # keep mamba list (group 0..n-2 are mamba; group[-1] is MLA)\n"
"                                pass\n"
"                    except Exception:\n"
"                        pass\n"
)


def main():
    if len(sys.argv) < 2:
        print(f"[{MARK}] usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 1
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr)
        return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied.")
        return 0
    if OLD not in src:
        print(f"[{MARK}] ANCHOR MISSING", file=sys.stderr)
        return 1
    src = src.replace(OLD, NEW, 1)
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
