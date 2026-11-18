#!/usr/bin/env python3
"""Breadcrumbs for the MoRIIO WRITE-mode KV delivery (diagnose multi-node context-loss).

The multi-node transport now completes (handshake OK, ack OK, no timeout) but decode
generates context-free even with MORIIO_SKIP_MAMBA=1 (so it's the MLA-KV write itself,
not KDA). Suspicion: prefill RDMA-writes to the wrong decode rank's base addr / block
offsets, so decode reads zeros for the prompt. These logs expose the actual runtime
values per request so we can see the divergence in ONE run.

Logs (all one-line, INFO, gated so they don't spam every layer):
  PREFILL write:  decode_dp_rank, dst_engine_id (per-rank), first remote base addr,
                  #local_blocks, #remote_blocks, first few block ids.
  DECODE alloc:   in update_state_after_alloc WRITE branch -- remote_dp_rank, the
                  local block_ids decode allocated + count (what it tells prefill to
                  write into), num_external_tokens.

Idempotent, anchor-based, py_compile-checked. Remove after diagnosis.
Usage: apply_kimik3_moriio_write_bc.py <vllm_install_dir>
"""
import os
import sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
ENG = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"


def _edit(path, subs, tag):
    if not os.path.isfile(path):
        print(f"[k3-writebc] {tag}: not found -- skip.")
        return True
    src = open(path).read()
    orig = src
    for old, new, note in subs:
        if new.split("\n")[0].strip() and new.split("\n")[1].strip() in src and old not in src:
            continue
        if old not in src:
            print(f"[k3-writebc] {tag}: anchor NOT found ({note})", file=sys.stderr)
            return False
        src = src.replace(old, new, 1)
    if src == orig:
        print(f"[k3-writebc] {tag}: no change (already applied).")
        return True
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-writebc] {tag}: compile failed, rolled back: {e}", file=sys.stderr)
        return False
    print(f"[k3-writebc] {tag}: applied.")
    return True


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    base = sys.argv[1]

    # ENGINE: log the write target right after dst_engine_id gets the dp suffix +
    # the session/meta are fetched.
    eng_old = (
        "        # Get or create sessions\n"
        "        sessions, remote_moriio_meta = self.worker._get_built_session(\n"
        "            task.dst_engine_id\n"
        "        )\n"
    )
    eng_new = (
        "        # Get or create sessions\n"
        "        sessions, remote_moriio_meta = self.worker._get_built_session(\n"
        "            task.dst_engine_id\n"
        "        )\n"
        "        try:\n"
        "            _k3_ba = list(getattr(remote_moriio_meta, 'kv_caches_base_addr', []) or [])[:1]\n"
        "            import logging as _k3lg\n"
        "            _k3lg.getLogger('vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine').info(\n"
        "                '[k3-writebc] WRITE ptp=%s decode_dp_rank=%s dst=%s base0=%s remote_ip=%s',\n"
        "                getattr(self.worker, 'tp_rank', '?'),\n"
        "                request_info.decode_dp_rank, task.dst_engine_id, _k3_ba,\n"
        "                getattr(request_info, 'completion_remote_ip', None),\n"
        "            )\n"
        "        except Exception:\n"
        "            pass\n"
    )
    # ENGINE 2: log the actual transfer plan (block-id counts + first offsets +
    # remote_num_blocks) so an offset/num_blocks mismatch is visible.
    eng2_old = (
        "        local_off, remote_off, sizes = offsets\n"
    )
    eng2_new = (
        "        local_off, remote_off, sizes = offsets\n"
        "        try:\n"
        "            from vllm.v1.kv_cache_interface import MambaSpec as _K3MS_BC\n"
        "            _k3_spec = self.worker.layer_to_spec.get(task.layer_name)\n"
        "            _k3_ismamba = isinstance(_k3_spec, _K3MS_BC)\n"
        "            _k3_seen = getattr(self, '_k3_bc_seen', None)\n"
        "            if _k3_seen is None:\n"
        "                _k3_seen = set(); self._k3_bc_seen = _k3_seen\n"
        "            _k3_key = ('M' if _k3_ismamba else 'A')\n"
        "            if _k3_key not in _k3_seen:\n"
        "                _k3_seen.add(_k3_key)\n"
        "                import logging as _k3lg2\n"
        "                _k3lg2.getLogger('vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine').info(\n"
        "                    '[k3-writebc] PLAN layer=%s mamba=%s spec=%s n_local_blk=%s n_remote_blk=%s '\n"
        "                    'remote_num_blocks=%s loff0=%s roff0=%s sz0=%s nseg=%s',\n"
        "                    task.layer_name, _k3_ismamba, type(_k3_spec).__name__,\n"
        "                    len(task.local_block_ids), len(request_info.block_ids or []),\n"
        "                    getattr(remote_moriio_meta, 'num_blocks', '?'),\n"
        "                    (local_off[:1] if local_off else []), (remote_off[:1] if remote_off else []),\n"
        "                    (sizes[:1] if sizes else []), len(sizes or []),\n"
        "                )\n"
        "        except Exception as _e:\n"
        "            pass\n"
    )
    ok_eng = _edit(os.path.join(base, ENG),
                   [(eng_old, eng_new, "engine write target"),
                    (eng2_old, eng2_new, "engine transfer plan")],
                   "moriio_engine.py")

    # CONNECTOR: decode-side alloc in WRITE branch -- log what blocks decode allocated.
    conn_old = (
        "            else:\n"
        "                # WRITE mode, decode side: notify P that blocks are ready\n"
        "                assert request.kv_transfer_params is not None, (\n"
        "                    \"kv_transfer_params should not be None\"\n"
        "                )\n"
        "\n"
        "                remote_dp_rank = request.kv_transfer_params.get(\"remote_dp_rank\", 0)\n"
    )
    conn_new = (
        "            else:\n"
        "                # WRITE mode, decode side: notify P that blocks are ready\n"
        "                assert request.kv_transfer_params is not None, (\n"
        "                    \"kv_transfer_params should not be None\"\n"
        "                )\n"
        "\n"
        "                remote_dp_rank = request.kv_transfer_params.get(\"remote_dp_rank\", 0)\n"
        "                try:\n"
        "                    _k3_lb = blocks.get_block_ids()[0]\n"
        "                    logger.info(\n"
        "                        '[k3-writebc] DECODE-alloc self_dp=%s remote_dp_rank=%s "
        "n_local_blocks=%s first=%s num_external=%s',\n"
        "                        getattr(self, '_global_dp_rank', '?'), remote_dp_rank,\n"
        "                        len(_k3_lb), _k3_lb[:4], num_external_tokens,\n"
        "                    )\n"
        "                except Exception:\n"
        "                    pass\n"
    )
    ok_conn = _edit(os.path.join(base, CONN), [(conn_old, conn_new, "decode alloc")],
                    "moriio_connector.py")

    return 0 if (ok_eng and ok_conn) else 1


if __name__ == "__main__":
    sys.exit(main())
