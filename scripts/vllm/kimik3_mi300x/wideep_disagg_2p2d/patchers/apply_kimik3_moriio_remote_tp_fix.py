#!/usr/bin/env python3
"""ROOT-CAUSE FIX: remote_tp_size=1 collapses all prefill ranks to decode tp0.

Proven by K3_HS_BC=1: every prefill worker (self_tp=0..7) resolves dial_tp=0
because meta.tp_size (remote_tp_size) arrives as 1 (the router does not advertise
the decode pool's TP size for WRITE producer requests). get_moriio_remote_tp_rank(
k, local=8, remote=1) = k // (8//1) = 0 for all k -> all prefill ranks dial decode
tp0's handshake port (8405), fetch tp0's base, and RDMA-write ONLY into decode
tp0. Decode tp1..7 stay zero -> 7/8 of every attention (and KDA) shard is empty on
decode -> fluent-but-context-free output.

FIX: our P/D deployment is SYMMETRIC TP (prefill TP == decode TP == self.world_size).
When the advertised remote_tp_size is unknown/degenerate (<= 1) but we run multi-way
TP locally, treat the remote pool as the SAME TP size (self.world_size). Then
get_moriio_remote_tp_rank(k, 8, 8) = k, so prefill rank k dials decode rank k
(port 8405+k), writes to decode rank k's base, and all 8 decode shards get their KV.

Two hunks (both in moriio_connector.py):
  H1 _remote_tp_rank: extend the "0/unknown" normalization to also cover
     remote_tp_size == 1 when self.world_size > 1 (symmetric-TP assumption).
  H2 add_new_req is in moriio_common.py; instead of touching wire parsing we fix
     at the single choke point (_remote_tp_rank) AND at the port-offset call in
     _moriio_handshake, which independently uses the raw remote_tp_size for the
     port math. Normalize there too so port_offset = get_port_offset(dp, k, 8).

Idempotent, anchor-based, py_compile-checked. Symmetric-TP only (our target);
genuinely heterogeneous TP would need the router to advertise remote_tp_size.
Usage: apply_kimik3_moriio_remote_tp_fix.py <vllm_install_dir>
"""
import os
import sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARK = "k3-remote-tp-fix"


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

    # H1: _remote_tp_rank normalization (0 OR 1 -> world_size when local TP > 1).
    h1_old = (
        "    def _remote_tp_rank(self, remote_tp_size: int) -> int:\n"
        "        # 0/unknown remote TP == homogeneous (avoids collapsing all ranks to 0).\n"
        "        if remote_tp_size == 0:\n"
        "            remote_tp_size = self.world_size\n"
        "        return get_moriio_remote_tp_rank(self.tp_rank, self.world_size, remote_tp_size)\n"
    )
    h1_new = (
        "    def _remote_tp_rank(self, remote_tp_size: int) -> int:\n"
        "        # 0/unknown remote TP == homogeneous (avoids collapsing all ranks to 0).\n"
        "        # " + MARK + ": remote_tp_size==1 from an un-advertising router ALSO\n"
        "        # collapses every prefill rank to decode tp0 (k//local = 0). For our\n"
        "        # symmetric-TP P/D, normalize any degenerate (<=1) remote size to the\n"
        "        # local world_size so rank k -> decode rank k.\n"
        "        if remote_tp_size <= 1 and self.world_size > 1:\n"
        "            remote_tp_size = self.world_size\n"
        "        elif remote_tp_size == 0:\n"
        "            remote_tp_size = self.world_size\n"
        "        return get_moriio_remote_tp_rank(self.tp_rank, self.world_size, remote_tp_size)\n"
    )

    # H2: _moriio_handshake port-offset uses the raw remote_tp_size arg; normalize
    # it the same way so the dialed port = 8405 + rank (not 8405 + 0).
    h2_old = (
        "        dial_tp_rank = (\n"
        "            self._remote_tp_rank(remote_tp_size)\n"
        "            if remote_tp_rank is None\n"
        "            else int(remote_tp_rank)\n"
        "        )\n"
        "        port_offset = get_port_offset(remote_dp_rank, dial_tp_rank, remote_tp_size)\n"
    )
    h2_new = (
        "        dial_tp_rank = (\n"
        "            self._remote_tp_rank(remote_tp_size)\n"
        "            if remote_tp_rank is None\n"
        "            else int(remote_tp_rank)\n"
        "        )\n"
        "        # " + MARK + ": normalize degenerate remote_tp_size for the port math\n"
        "        # too, so prefill rank k dials decode port base+k (symmetric TP).\n"
        "        _k3_rts = remote_tp_size\n"
        "        if _k3_rts <= 1 and self.world_size > 1:\n"
        "            _k3_rts = self.world_size\n"
        "        port_offset = get_port_offset(remote_dp_rank, dial_tp_rank, _k3_rts)\n"
    )

    for old, new, tag in [(h1_old, h1_new, "H1 _remote_tp_rank"),
                          (h2_old, h2_new, "H2 handshake port_offset")]:
        if old not in src:
            print(f"[{MARK}] {tag}: ANCHOR MISSING", file=sys.stderr)
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
