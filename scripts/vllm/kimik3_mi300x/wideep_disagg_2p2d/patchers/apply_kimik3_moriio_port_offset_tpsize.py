#!/usr/bin/env python3
"""Fix MoRIIO per-rank port collision when DP-local > 1 AND TP > 1 (wide-EP).

ROOT CAUSE
  MoRIIO derives each rank's side-channel/notify/handshake port as
    base_port + get_port_offset(dp_rank, tp_rank[, tp_size])
  where get_port_offset(dp, tp, tp_size=1) == dp * tp_size + tp. The offset is
  UNIQUE per (dp, tp) only when tp_size is the REAL tp size. Most call sites omit
  it, so tp_size defaults to 1 and the offset collapses to (dp + tp). That is
  collision-free only when at most one of dp/tp varies per node -- i.e. the old
  TP8 x DP-local-1 shape. In the wide-EP TP2 x DP-local-4 shape the local ranks
  are (dp0..3, tp0..1) and (dp0,tp1) and (dp1,tp0) both map to offset 1:
    handshake_port 8405: dp0/tp1 -> 8406  AND  dp1/tp0 -> 8406  (COLLISION)
  => the second listener's `socket.bind('tcp://*:8406')` raises
     "zmq.error.ZMQError: Address already in use" and the moriio_handshake_listener
     thread dies; all engines then idle-spin on "No available shared memory
     broadcast block" and the pool never reaches startup.

FIX
  Pass the real tp_size to the 5 bare get_port_offset() calls so the offset is
  dp*tp_size + tp (a proper 2-D -> 1-D flattening, always unique). tp_size is
  already available at every site (a local `tp_size`, self.tp_size,
  self.moriio_config.tp_size, or self.worker.moriio_config.tp_size). The two
  call sites that ALREADY pass a tp_size (moriio_connector.py:1538 dial + :2621
  recv-callback) are left untouched. Producer and consumer both use the same
  formula so handshake/notify addressing stays matched.

  NOTE: base ports must be spaced >= dp_local*tp_size apart to avoid cross-family
  overlap. Defaults handshake=8405 / notify=61005 / local_ping=61555 are spaced
  by >=8, and TP2 x DP-local-4 needs only 8 slots -> no cross-family overlap.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_moriio_port_offset_tpsize.py <vllm_install_dir>
"""
import os
import sys

MORIIO = "distributed/kv_transfer/kv_connector/v1/moriio"


def _patch(path, subs, tag):
    if not os.path.isfile(path):
        print(f"[k3-portoff] {tag}: {path} not found -- skip.")
        return True
    src = open(path).read()
    orig = src
    for old, new, note in subs:
        if new in src and old not in src:
            continue  # already applied
        if old not in src:
            print(f"[k3-portoff] {tag}: anchor NOT found ({note}):\n  {old[:78]!r}",
                  file=sys.stderr)
            return False
        src = src.replace(old, new, 1)
    if src == orig:
        print(f"[k3-portoff] {tag}: no change (already applied).")
        return True
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-portoff] {tag}: compile failed, rolled back: {e}", file=sys.stderr)
        return False
    print(f"[k3-portoff] {tag}: applied.")
    return True


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    base = os.path.join(sys.argv[1], MORIIO)

    common_ok = _patch(
        os.path.join(base, "moriio_common.py"),
        [(
            "        port_offset = get_port_offset(dp_rank, tp_rank)\n",
            "        port_offset = get_port_offset(dp_rank, tp_rank, tp_size)  # k3-portoff\n",
            "common notify base offset",
        )],
        "moriio_common.py",
    )

    conn_ok = _patch(
        os.path.join(base, "moriio_connector.py"),
        [
            (
                "            target_port = remote_notify_port + get_port_offset(remote_dp_rank, tp_index)\n",
                "            target_port = remote_notify_port + get_port_offset(remote_dp_rank, tp_index, self.tp_size)  # k3-portoff\n",
                "release-notify target",
            ),
            (
                "                        target_port = remote_notify_port + get_port_offset(\n"
                "                            _remote_dp_rank_for_port, tp_index\n"
                "                        )\n",
                "                        target_port = remote_notify_port + get_port_offset(\n"
                "                            _remote_dp_rank_for_port, tp_index, self.tp_size  # k3-portoff\n"
                "                        )\n",
                "block-notify target",
            ),
            (
                "        self.side_channel_port: int = (\n"
                "            self.moriio_config.handshake_port\n"
                "            + get_port_offset(self.dp_rank, self.tp_rank)\n"
                "        )\n",
                "        self.side_channel_port: int = (\n"
                "            self.moriio_config.handshake_port\n"
                "            + get_port_offset(self.dp_rank, self.tp_rank, self.moriio_config.tp_size)  # k3-portoff\n"
                "        )\n",
                "worker side_channel_port bind",
            ),
        ],
        "moriio_connector.py",
    )

    eng_ok = _patch(
        os.path.join(base, "moriio_engine.py"),
        [(
            "        remote_port = remote_notify_port + get_port_offset(\n"
            "            _decode_dp_rank_for_port, self.worker.tp_rank\n"
            "        )\n",
            "        remote_port = remote_notify_port + get_port_offset(\n"
            "            _decode_dp_rank_for_port, self.worker.tp_rank, self.worker.moriio_config.tp_size  # k3-portoff\n"
            "        )\n",
            "engine notify target",
        )],
        "moriio_engine.py",
    )

    return 0 if (common_ok and conn_ok and eng_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
