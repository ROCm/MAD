#!/usr/bin/env python3
"""Skip the MoRIIO per-layer save hook for KDA/mamba layers under MORIIO_SKIP_MAMBA=1.

The attention forward calls connector.save_kv_layer(layer_name, kv_cache, ...) after
EACH attention-family layer (kv_transfer_utils.maybe_transfer_kv_layer). K3 layers 0-3
are KDA (mamba), so the very first save_kv_layer of the forward runs on a mamba cache:
it does the remote handshake and schedules an INLINE RDMA write (schedule_write_blocks
-> torch.cuda.Event().record on the layer tensor) on the compute stream, mid-forward.
Under 2P/2D that collides with the MoRI-EP all2all and faults all ranks before any KDA
compute (observed: fault precedes the KDA conv debug hook).

This makes save_kv_layer a no-op for mamba layers when MORIIO_SKIP_MAMBA=1, isolating
that path (the mamba-offset skip alone did NOT cover the save-hook entry/handshake).

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_moriio_save_skip_mamba.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-save-skip] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src
    if "k3-kda: MORIIO_SKIP_MAMBA" in src and "save hook" in src:
        print("[k3-save-skip] already applied.")
        return 0

    # Anchor on the worker save_kv_layer body prologue (the version that takes
    # metadata as first arg -> has the `if self.mode == MoRIIOMode.READR` guard
    # followed by `remote_engine_id = None`).
    anchor = (
        "        if not self.is_producer:\n"
        "            return\n"
        "        if self.mode == MoRIIOMode.READ:\n"
        "            return\n"
        "        remote_engine_id = None\n"
    )
    inject = (
        "        if not self.is_producer:\n"
        "            return\n"
        "        if self.mode == MoRIIOMode.READ:\n"
        "            return\n"
        "        # k3-kda: MORIIO_SKIP_MAMBA=1 -> skip the per-layer save hook for\n"
        "        # KDA/mamba layers (inline RDMA write on the compute stream at the\n"
        "        # first layers races the MoRI-EP all2all -> all-rank GPU fault).\n"
        "        import os as _os_k3s\n"
        "        if _os_k3s.environ.get(\"MORIIO_SKIP_MAMBA\", \"0\") == \"1\":\n"
        "            from vllm.v1.kv_cache_interface import MambaSpec as _K3MambaSpecSK\n"
        "            if isinstance(self.layer_to_spec.get(layer_name), _K3MambaSpecSK):\n"
        "                return\n"
        "        remote_engine_id = None\n"
    )
    n = src.count(anchor)
    if n == 0:
        print("[k3-save-skip] WARN: save_kv_layer anchor not found -- not applied.")
        return 0
    # Only the worker save_kv_layer has this exact prologue; replace the first.
    src = src.replace(anchor, inject, 1)

    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[k3-save-skip] ERROR: compile failed: {e}", file=sys.stderr)
        open(path, "w").write(orig)
        return 1
    print("[k3-save-skip] save_kv_layer now skips mamba layers under MORIIO_SKIP_MAMBA=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
