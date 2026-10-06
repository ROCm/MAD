#!/usr/bin/env python3
"""FIX chunked-prefill final-chunk detection using FRESH per-step COMPUTE progress.

ROOT CAUSE (proven by debug): the connector's chunked-prefill gates detect the
"final chunk" by BLOCK COUNT:
    num_prompt_tokens > len(block_ids) * self.block_size   (defer if True)
But the effective scheduler block_size is ~5760 (mamba-page-padded attention
page), so a <=5760-token prompt occupies ONE block. Chunked prefill still
computes it in multiple passes (max_num_batched_tokens=2048 at a time), yet the
block-count gate is already satisfied after chunk 1 (1*5760 >= 2611) -> the KV
transfer fires with only ~2048 tokens computed -> tokens past chunk 1 (the
needle at the end) are never transferred -> long-context recall dies exactly at
max_num_batched_tokens. When the prompt fits in <=1 block, block-count can NEVER
detect chunk completion.

The ONLY robust signal is COMPUTE progress. req.num_computed_tokens on the
producer Request is STALE (scheduler-process copy, =0), so we must use the FRESH
per-step data the scheduler passes into build_connector_meta:
  * scheduler_output.num_scheduled_tokens[req_id]              (scheduled THIS step)
  * scheduled_new_reqs[].num_computed_tokens                   (new req, before step)
  * scheduled_cached_reqs.num_computed_tokens[i]               (cached req, before step)
A request's prefill is COMPLETE this step when:
  computed_before + scheduled_this_step >= num_prompt_tokens - SLACK
SLACK (default 2, env K3_CHUNK_GATE_SLACK) absorbs the mamba N-1 prompt
truncation (P-side does request.num_prompt_tokens -= 1). Deferring until this is
true, then emitting the full accumulated block list, transfers ALL chunks' KV.

Implementation: a helper builds a {req_id: (computed_before, scheduled)} map from
scheduler_output at the top of build_connector_meta, stored on self so both gates
(entry defer + accumulation final-detect) use it. Falls back to the original
block-count gate if the data is missing.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_chunk_gate_fix.py <vllm_install_dir>
"""
import os, sys
MARK = "k3-chunk-gate"
REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"

# --- Part 1: build the fresh-progress map at the top of build_connector_meta,
# right after the WRITE/PRODUCER guard. Anchor on the comment block that opens
# the producer branch.
A_OLD = (
"        if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.PRODUCER:\n"
"            # This is the logic for checking against chunked prefill.\n"
"            # When the last chunk is identified,\n"
"            # It places the request metadata into the saving queue.\n"
)
A_NEW = (
"        if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.PRODUCER:\n"
"            # " + MARK + ": build FRESH per-step compute-progress map. Block-count\n"
"            # gates fail when the whole prompt fits in <=1 block (bs~5760);\n"
"            # compute progress is the only reliable final-chunk signal.\n"
"            self._k3_prog = {}\n"
"            try:\n"
"                _k3_nst = scheduler_output.num_scheduled_tokens\n"
"                for _k3_nr in getattr(scheduler_output, 'scheduled_new_reqs', []) or []:\n"
"                    _k3_rid = getattr(_k3_nr, 'req_id', None)\n"
"                    if _k3_rid is not None:\n"
"                        self._k3_prog[_k3_rid] = (\n"
"                            int(getattr(_k3_nr, 'num_computed_tokens', 0)),\n"
"                            int(_k3_nst.get(_k3_rid, 0)),\n"
"                        )\n"
"                _k3_cr = scheduler_output.scheduled_cached_reqs\n"
"                for _k3_ci, _k3_crid in enumerate(_k3_cr.req_ids):\n"
"                    self._k3_prog[_k3_crid] = (\n"
"                        int(_k3_cr.num_computed_tokens[_k3_ci]),\n"
"                        int(_k3_nst.get(_k3_crid, 0)),\n"
"                    )\n"
"            except Exception:\n"
"                self._k3_prog = {}\n"
"            # This is the logic for checking against chunked prefill.\n"
"            # When the last chunk is identified,\n"
"            # It places the request metadata into the saving queue.\n"
)

# Helper appended once to compute final-ness from the map.
HELPER_ANCHOR = "class MoRIIOConnector"
HELPER_CODE = (
"def _k3_prefill_done(self, req_id, req):  # " + MARK + "\n"
"    import os as _os\n"
"    prog = getattr(self, '_k3_prog', None)\n"
"    if not prog or req_id not in prog:\n"
"        return None\n"
"    cb, st = prog[req_id]\n"
"    slack = int(_os.environ.get('K3_CHUNK_GATE_SLACK', '2'))\n"
"    return (cb + st) >= (int(req.num_prompt_tokens) - slack)\n"
"\n\n"
)

# --- Part 2a: ENTRY defer gate (_reqs_need_save loop) ---
B_OLD = (
"        for req_id, (req, block_ids) in self._reqs_need_save.items():\n"
"            kv_params = self._req_kv_params.get(req_id, req.kv_transfer_params or {})\n"
"            if req.num_prompt_tokens > len(block_ids) * self.block_size:\n"
"                # not last chunk prefill\n"
"                self._reqs_need_pending_save[req_id] = (req, block_ids)\n"
"                continue\n"
)
B_NEW = (
"        for req_id, (req, block_ids) in self._reqs_need_save.items():\n"
"            kv_params = self._req_kv_params.get(req_id, req.kv_transfer_params or {})\n"
"            # " + MARK + ": defer by COMPUTE progress (fresh per-step), not block count.\n"
"            _k3done = _k3_prefill_done(self, req_id, req)\n"
"            if _k3done is None:\n"
"                _k3done = not (req.num_prompt_tokens > len(block_ids) * self.block_size)\n"
"            if os.environ.get('K3_CHUNK_GATE_DEBUG', '0') == '1':\n"
"                import logging as _k3be_lg\n"
"                _k3be_lg.getLogger(__name__).warning(\n"
"                    '[" + MARK + "-entry] req=%s nblk=%d npt=%d done=%s prog=%s',\n"
"                    req_id, len(block_ids), int(req.num_prompt_tokens), _k3done,\n"
"                    getattr(self, '_k3_prog', {}).get(req_id))\n"
"            if not _k3done:\n"
"                # not last chunk prefill\n"
"                self._reqs_need_pending_save[req_id] = (req, block_ids)\n"
"                continue\n"
)

# --- Part 2b: accumulation final-detect gate (scheduled_cached_reqs loop) ---
C_OLD = (
"                    if (\n"
"                        len(self._reqs_need_pending_save[req_id][1]) * self.block_size\n"
"                        >= req.num_prompt_tokens\n"
"                    ):\n"
)
C_NEW = (
"                    _k3done2 = _k3_prefill_done(self, req_id, req)  # " + MARK + "\n"
"                    if _k3done2 is None:\n"
"                        _k3done2 = (\n"
"                            len(self._reqs_need_pending_save[req_id][1]) * self.block_size\n"
"                            >= req.num_prompt_tokens\n"
"                        )\n"
"                    if os.environ.get('K3_CHUNK_GATE_DEBUG', '0') == '1':\n"
"                        import logging as _k3cg_lg\n"
"                        _k3cg_lg.getLogger(__name__).warning(\n"
"                            '[" + MARK + "-accum] req=%s nblk=%d npt=%d done=%s prog=%s',\n"
"                            req_id, len(self._reqs_need_pending_save[req_id][1]),\n"
"                            int(req.num_prompt_tokens), _k3done2,\n"
"                            getattr(self, '_k3_prog', {}).get(req_id))\n"
"                    if _k3done2:\n"
)

# --- Part 2c: POST-LOOP SWEEP (gate D). The accumulation loop above is nested
# under `if new_block_ids is not None:`. When the FINAL prefill chunk allocates NO
# new block (whole prompt <=1 block, bs~5760), new_block_ids is None on that
# chunk -> the deferred req is never re-examined -> stuck in
# _reqs_need_pending_save -> unmap MISS deadlock. Sweep after the loop: emit any
# pending req whose prefill completed THIS step (per the compute-progress map).
D_OLD = (
"                        del self._reqs_need_pending_save[req_id]\n"
"\n"
"        # Loop through scheduled reqs and convert to ReqMeta.\n"
"        for req_id, (req, block_ids) in self._reqs_need_recv.items():\n"
)
D_NEW = (
"                        del self._reqs_need_pending_save[req_id]\n"
"\n"
"            # " + MARK + "-sweep: emit deferred reqs whose FINAL chunk added no\n"
"            # new block (accum loop skipped them) but whose prefill is now done.\n"
"            try:\n"
"                for _k3sw_rid in list(self._reqs_need_pending_save.keys()):\n"
"                    _k3sw_req, _k3sw_bl = self._reqs_need_pending_save[_k3sw_rid]\n"
"                    _k3sw_done = _k3_prefill_done(self, _k3sw_rid, _k3sw_req)\n"
"                    if os.environ.get('K3_CHUNK_GATE_DEBUG', '0') == '1':\n"
"                        import logging as _k3sw_lg\n"
"                        _k3sw_lg.getLogger(__name__).warning(\n"
"                            '[" + MARK + "-sweep] req=%s nblk=%d npt=%d done=%s prog=%s',\n"
"                            _k3sw_rid, len(_k3sw_bl), int(_k3sw_req.num_prompt_tokens),\n"
"                            _k3sw_done, getattr(self, '_k3_prog', {}).get(_k3sw_rid))\n"
"                    if not _k3sw_done:\n"
"                        continue\n"
"                    _k3sw_kv = self._req_kv_params.pop(\n"
"                        _k3sw_rid, _k3sw_req.kv_transfer_params or {}\n"
"                    )\n"
"                    meta.add_new_req(\n"
"                        request_id=_k3sw_rid,\n"
"                        local_block_ids=self._reqs_need_pending_save[_k3sw_rid][1],\n"
"                        kv_transfer_params=_k3sw_kv,\n"
"                        write_mode=True,\n"
"                        mamba_local_block_ids=self._reqs_save_mamba.get(_k3sw_rid, []),\n"
"                        all_group_block_ids=self._reqs_save_allgrp.get(_k3sw_rid, None),\n"
"                    )\n"
"                    del self._reqs_need_pending_save[_k3sw_rid]\n"
"            except Exception as _k3sw_e:\n"
"                import logging as _k3sw_lg2\n"
"                _k3sw_lg2.getLogger(__name__).warning('[" + MARK + "-sweep] err %s', _k3sw_e)\n"
"\n"
"        # Loop through scheduled reqs and convert to ReqMeta.\n"
"        for req_id, (req, block_ids) in self._reqs_need_recv.items():\n"
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
    for old, tag in [(A_OLD, "A progress-map"), (B_OLD, "B entry-defer"),
                     (C_OLD, "C final-detect"), (D_OLD, "D post-loop-sweep"),
                     (HELPER_ANCHOR, "helper-anchor")]:
        if old not in src:
            print(f"[{MARK}] ANCHOR MISSING ({tag})", file=sys.stderr); return 1
    # insert module-level helper before the first class definition
    src = src.replace(HELPER_ANCHOR, HELPER_CODE + HELPER_ANCHOR, 1)
    src = (src.replace(A_OLD, A_NEW, 1).replace(B_OLD, B_NEW, 1)
              .replace(C_OLD, C_NEW, 1).replace(D_OLD, D_NEW, 1))
    open(path, "w").write(src)
    try:
        import py_compile; py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[{MARK}] compile FAIL {e}", file=sys.stderr); return 1
    print(f"[{MARK}] applied (compute-progress gate).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
