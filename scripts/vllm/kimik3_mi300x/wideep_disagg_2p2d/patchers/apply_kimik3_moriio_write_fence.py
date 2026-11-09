#!/usr/bin/env python3
"""FIX/DIAGNOSTIC for the RDMA-write-then-notify ORDERING HAZARD (UPDATE 20).

Symptom: disagg recall is NON-DETERMINISTIC at greedy temp=0 (same prompt -> different
garbage each run), while colocated is deterministic+correct. Root cause: prefill
RDMA-writes KV into decode HBM, waits only for SENDER-side completion
(waiting_for_transfer_complete polls status.Succeeded()), then sends write_done over
a SEPARATE ZMQ/TCP path. Sender-local RDMA completion does not order against the
RECEIVER's HBM visibility, and TCP notify races the data landing -> decode reads
stale/partial HBM -> non-deterministic wrong recall.

This patch inserts a sender-side ORDERING FENCE in _finalize_if_complete, right
before send_notify(write_done). Modes via env K3_WRITE_FENCE:
  - 'delay' (default when enabled): sleep K3_WRITE_FENCE_MS milliseconds (default 20)
    before sending write_done. DIAGNOSTIC: if recall becomes correct+deterministic,
    the race is confirmed. Cheap, no MoRI API dependency.
  - 'off' / unset: no change (baseline).
(A read-back RDMA fence is the proper production fix but needs the per-transfer
session+offsets; staged separately once the delay confirms the race.)

Gated: only active when K3_WRITE_FENCE is set. Idempotent, anchor-based,
py_compile-checked.
Usage: apply_kimik3_moriio_write_fence.py <vllm_install_dir>
"""
import os
import sys

ENG = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
MARK = "k3-write-fence"


def main():
    base = sys.argv[1]
    path = os.path.join(base, ENG)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr)
        return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied.")
        return 0
    # H2: replace the per-event sync with a FULL device sync before the RDMA read,
    # so KV inserts on ANY stream (K3 MLA uses an aux_stream) are complete before
    # the transfer reads local cache. Gated by K3_WRITE_DEVSYNC=1. This targets the
    # intermittent per-position corruption (multi-stream write vs RDMA read race).
    h2_old = (
        "        # This event is used to synchronize the kv transfer and computation tasks.\n"
        "        task.event.synchronize()\n"
    )
    h2_new = (
        "        # This event is used to synchronize the kv transfer and computation tasks.\n"
        "        task.event.synchronize()\n"
        "        import os as _k3dsos\n"
        "        if _k3dsos.environ.get('K3_WRITE_DEVSYNC', '') in ('1', 'true', 'on'):\n"
        "            # " + MARK + ": full-device sync so aux-stream KV inserts finish\n"
        "            # before RDMA reads local cache (event only covers one stream).\n"
        "            try:\n"
        "                import torch as _k3t\n"
        "                _k3t.cuda.synchronize()\n"
        "            except Exception:\n"
        "                pass\n"
    )
    old = (
        "        # Send completion notification\n"
        "        self.worker.moriio_wrapper.send_notify(\n"
        "            transfer_id, remote_ip, remote_port, message_type=\"write_done\"\n"
        "        )\n"
    )
    new = (
        "        # " + MARK + ": ordering fence before write_done. The RDMA write and\n"
        "        # the ZMQ/TCP write_done travel different paths; sender-local RDMA\n"
        "        # completion does not guarantee the data is visible in the RECEIVER's\n"
        "        # HBM. Without a fence decode can read stale HBM (non-deterministic\n"
        "        # recall). 'delay' mode is the diagnostic; 'readback' the real fix.\n"
        "        import os as _k3wfos, time as _k3wftime\n"
        "        _k3wf = _k3wfos.environ.get('K3_WRITE_FENCE', '').lower()\n"
        "        if _k3wf in ('delay', '1', 'true', 'on'):\n"
        "            try:\n"
        "                _k3ms = float(_k3wfos.environ.get('K3_WRITE_FENCE_MS', '20'))\n"
        "            except Exception:\n"
        "                _k3ms = 20.0\n"
        "            _k3wftime.sleep(_k3ms / 1000.0)\n"
        "        # Send completion notification\n"
        "        self.worker.moriio_wrapper.send_notify(\n"
        "            transfer_id, remote_ip, remote_port, message_type=\"write_done\"\n"
        "        )\n"
    )
    if old not in src:
        print(f"[{MARK}] anchor NOT found", file=sys.stderr)
        return 1
    src = src.replace(old, new, 1)
    # H2: full-device-sync before RDMA read (targets multi-stream write race)
    if h2_old in src:
        src = src.replace(h2_old, h2_new, 1)
    else:
        print(f"[{MARK}] H2 devsync anchor NOT found (continuing with H1 only)", file=sys.stderr)
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
