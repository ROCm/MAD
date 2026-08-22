#!/usr/bin/env python3
"""Diagnostics for the KDA/mamba MoRIIO transfer GPU memory fault.

Adds two things to compute_mamba_block_transfer_offsets in moriio_layout.py:

  1. MORIIO_SKIP_MAMBA=1 -> return ([],[],[]) so the KDA/mamba state transfer is
     a no-op (MLA KV still transfers). If the producer stops GPU-faulting under
     this flag, the fault is isolated to the KDA state write path.

  2. Bounds check: log [k3-kda OOB] when (block*stride + sub_off + size) exceeds
     the local mamba cache tensor extent (num_blocks*page_size_bytes) -- i.e. the
     exact offset that would form an out-of-bounds RDMA address.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_moriio_mamba_diag.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-mamba-diag] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src
    changed = 0

    # --- 1) MORIIO_SKIP_MAMBA early return ---
    if "MORIIO_SKIP_MAMBA" not in src:
        anchor = (
            "    if len(local_block_ids) > len(remote_block_ids):\n"
            "        raise ValueError(\n"
            '            "local_block_ids longer than remote_block_ids (mamba): "\n'
        )
        inject = (
            "    import os as _os\n"
            "    if _os.environ.get(\"MORIIO_SKIP_MAMBA\", \"0\") == \"1\":\n"
            "        return [], [], []\n"
            "\n"
            "    if len(local_block_ids) > len(remote_block_ids):\n"
            "        raise ValueError(\n"
            '            "local_block_ids longer than remote_block_ids (mamba): "\n'
        )
        if anchor in src:
            src = src.replace(anchor, inject, 1)
            changed += 1
        else:
            print("[k3-mamba-diag] WARN: skip-flag anchor not found.")

    # --- 2) bounds-check logging in the write loop ---
    if "[k3-kda OOB]" not in src:
        loop_old = (
            "    for lb, rb in zip(local_block_ids, remote_block_ids):\n"
            "        lbase = lb * stride\n"
            "        rbase = rb * stride\n"
            "        for off, sz in subregions:\n"
            "            offset_local[w] = lbase + off\n"
            "            offset_remote[w] = rbase + off  # tp_ratio==1 -> same sub-offset\n"
            "            sizes[w] = sz\n"
            "            w += 1\n"
        )
        loop_new = (
            "    try:\n"
            "        _tensor_bytes = int(kv_cache.numel()) * int(kv_cache.element_size())\n"
            "    except Exception:\n"
            "        _tensor_bytes = -1\n"
            "    for lb, rb in zip(local_block_ids, remote_block_ids):\n"
            "        lbase = lb * stride\n"
            "        rbase = rb * stride\n"
            "        for off, sz in subregions:\n"
            "            offset_local[w] = lbase + off\n"
            "            offset_remote[w] = rbase + off  # tp_ratio==1 -> same sub-offset\n"
            "            sizes[w] = sz\n"
            "            if _tensor_bytes >= 0 and (lbase + off + sz) > _tensor_bytes:\n"
            "                import logging as _lg\n"
            "                _lg.getLogger(__name__).error(\n"
            '                    "[k3-kda OOB] layer=%s lb=%d off=%d sz=%d end=%d > "\n'
            '                    "tensor_bytes=%d (stride=%d num_local=%d)",\n'
            "                    layer_name, lb, off, sz, lbase + off + sz,\n"
            "                    _tensor_bytes, stride, len(local_block_ids),\n"
            "                )\n"
            "            w += 1\n"
        )
        if loop_old in src:
            src = src.replace(loop_old, loop_new, 1)
            changed += 1
        else:
            print("[k3-mamba-diag] WARN: write-loop anchor not found (already "
                  "modified?); bounds check not added.")

    if changed == 0:
        print("[k3-mamba-diag] nothing to do (already applied).")
        return 0

    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[k3-mamba-diag] ERROR: compile failed: {e}", file=sys.stderr)
        open(path, "w").write(orig)
        return 1
    print(f"[k3-mamba-diag] applied {changed} change(s): MORIIO_SKIP_MAMBA "
          "flag + KDA OOB bounds logging.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
