#!/usr/bin/env python3
"""Log the ACTUAL handshake dial: self.tp_rank, dial_tp_rank, port, port_offset,
final path -- to find why all 8 prefill ranks fetch the SAME decode base (only
decode tp0's KV lands). Gated K3_HS_BC=1. Edits _moriio_handshake in
moriio_connector.py.
"""
import os, sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"


def main():
    base = sys.argv[1]
    path = os.path.join(base, CONN)
    src = open(path).read()
    if "k3-hsbc" in src:
        print("[k3-hsbc] already applied.")
        return 0
    old = (
        "        port_offset = get_port_offset(remote_dp_rank, dial_tp_rank, remote_tp_size)\n"
        "        path = make_zmq_path(\"tcp\", host, port + port_offset)\n"
    )
    new = (
        "        port_offset = get_port_offset(remote_dp_rank, dial_tp_rank, remote_tp_size)\n"
        "        path = make_zmq_path(\"tcp\", host, port + port_offset)\n"
        "        import os as _hsos  # k3-hsbc\n"
        "        if _hsos.environ.get('K3_HS_BC','0')=='1':\n"
        "            logger.info('[k3-hsbc] self_tp=%s dial_tp=%s remote_tp_size=%s remote_dp_rank=%s port=%s off=%s path=%s eid=%s',\n"
        "                getattr(self,'tp_rank','?'), dial_tp_rank, remote_tp_size, remote_dp_rank, port, port_offset, path, expected_engine_id)\n"
    )
    if old not in src:
        print("[k3-hsbc] anchor NOT found", file=sys.stderr)
        return 1
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[k3-hsbc] compile FAIL {e}", file=sys.stderr)
        return 1
    print("[k3-hsbc] applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
