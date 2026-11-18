#!/usr/bin/env python3
"""Decode-side ground truth: when a WRITE completes (write_done seen), read the
norm of decode's OWN KV slots to prove the RDMA bytes actually landed.

Write side is proven correct (non-zero source, right slots, no RDMA errors) yet
decode is context-free. This probe runs in the connector's get_finished thread
(NOT the @eager_break_during_capture forward, so logging works) and, on the first
write completion, logs the L2 norm of decode's attention slot and mamba slot. If
those are ~0, the RDMA write never reached decode's memory (transport delivered to
a wrong address / decode registered a different tensor). If non-zero, the bytes
ARE present and the bug is in how the KDA/attention kernels READ them.

Gated K3_DECODE_RECV_PROBE=1. Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_decode_recv_probe.py <vllm_install_dir>
"""
import os
import sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARK = "k3-recvprobe"


def main():
    base = sys.argv[1]
    path = os.path.join(base, CONN)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr)
        return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied.")
        return 0
    old = (
        "            if self.mode == MoRIIOMode.WRITE:\n"
        "                fresh = self.moriio_wrapper.pop_finished_write_req_ids()\n"
        "                # Accumulate with any completions that arrived before their\n"
        "                # transfer_id was registered in transfer_id_to_request_id.\n"
        "                self._unmatched_write_completions |= fresh\n"
        "                done_recving = self._unmatched_write_completions\n"
    )
    new = (
        "            if self.mode == MoRIIOMode.WRITE:\n"
        "                fresh = self.moriio_wrapper.pop_finished_write_req_ids()\n"
        "                # Accumulate with any completions that arrived before their\n"
        "                # transfer_id was registered in transfer_id_to_request_id.\n"
        "                self._unmatched_write_completions |= fresh\n"
        "                done_recving = self._unmatched_write_completions\n"
        "                import os as _k3rpos  # " + MARK + "\n"
        "                if fresh and _k3rpos.environ.get('K3_DECODE_RECV_PROBE','0')=='1' and not getattr(self,'_k3_rp_done',False):\n"
        "                    try:\n"
        "                        self._k3_rp_done = True\n"
        "                        from vllm.v1.kv_cache_interface import MambaSpec as _K3MS_RP\n"
        "                        _att=_mam=None\n"
        "                        for _ln,_t in self.kv_caches.items():\n"
        "                            _ism=isinstance(self.layer_to_spec.get(_ln),_K3MS_RP)\n"
        "                            if _ism and _mam is None: _mam=(_ln,_t)\n"
        "                            if (not _ism) and _att is None: _att=(_ln,_t)\n"
        "                        def _nz(t,slot):\n"
        "                            try:\n"
        "                                s=t[int(slot)].flatten(); \n"
        "                                _el=[round(float(x),3) for x in s[:6].float().tolist()]\n"
        "                                return (float(s.float().norm()), _el)\n"
        "                            except Exception as e: return (-2.0,[])\n"
        "                        _an=_nz(_att[1],1) if _att else (-1,[])\n"
        "                        _mn=_nz(_mam[1],2) if _mam else (-1,[])\n"
        "                        logger.info('[" + MARK + "] attn_slot1_norm=%.4e attn_el=%s mamba_slot2_norm=%.4e mamba_el=%s',\n"
        "                            _an[0], _an[1], _mn[0], _mn[1])\n"
        "                    except Exception as _e:\n"
        "                        logger.info('[" + MARK + "] EXC %r', _e)\n"
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
