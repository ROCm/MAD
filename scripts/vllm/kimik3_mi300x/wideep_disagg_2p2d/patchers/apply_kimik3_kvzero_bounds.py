#!/usr/bin/env python3
"""Bounds-guard the KV block-zeroing Triton kernel (the 2P/2D disagg producer fault).

ROOT CAUSE (found by pre-forward breadcrumbs, no GPU tools):
  execute_model -> _update_states zeros freshly-allocated attention blocks via
  KVBlockZeroer.zero_block_ids -> _zero_kv_blocks_kernel, BEFORE any layer forward.
  The kernel writes at seg_addr + block_id*page_size_el (v1/worker/utils.py). It
  never bounds-checks block_id against the segment tensor's block capacity. Under
  K3 hybrid (MLA + KDA) + 2P/2D disagg the scheduler emits an attention block id
  that exceeds a segment's capacity -> OOB write -> "Memory access fault" on all
  ranks, before the model forward. (Breadcrumb: "zero_block_ids DONE" prints, then
  the fault; the kernel launch is async so the Python line logs first.)

FIX:
  Record each segment's logical block capacity (seg_nblocks) in KVBlockZeroer.__init__,
  pass it to the kernel, and skip any (block, seg) whose block_id >= that capacity.
  Skipping is correct: only FullAttention/MLA managers record ids for zeroing, so an
  id past an MLA segment's capacity is spurious for that segment.

Edits v1/worker/utils.py in 4 spots (kernel sig, kernel guard, __init__ capacity,
zero_block_ids meta+call). Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_kvzero_bounds.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/worker/utils.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-kvzero] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src
    if "_K3MLASpec" in src:
        print("[k3-kvzero] already applied.")
        return 0

    subs = []

    # 0) THE FIX: skip MLAAttentionSpec in the zeroer (MLA subclasses
    #    FullAttentionSpec -> wrong stride math -> OOB write -> GPU fault).
    subs.append((
        "        for group in attn_groups_iter:\n"
        "            spec = group.kv_cache_spec\n"
        "            if not isinstance(spec, FullAttentionSpec):\n"
        "                continue\n"
        "            if group.kv_cache_group_id >= len(kernel_block_sizes):\n"
        "                continue\n",
        "        from vllm.v1.kv_cache_interface import MLAAttentionSpec as _K3MLASpec\n"
        "        _os_k3skip = __import__(\"os\").environ.get(\"K3_ZERO_SKIP_MLA\", \"1\") == \"1\"\n"
        "        for group in attn_groups_iter:\n"
        "            spec = group.kv_cache_spec\n"
        "            if not isinstance(spec, FullAttentionSpec):\n"
        "                continue\n"
        "            if _os_k3skip and isinstance(spec, _K3MLASpec):\n"
        "                continue\n"
        "            if group.kv_cache_group_id >= len(kernel_block_sizes):\n"
        "                continue\n",
    ))

    # 1) kernel signature: add seg_nblocks_ptr before N_SEGS
    subs.append((
        "    block_ids_ptr,\n"
        "    n_blocks,\n"
        "    N_SEGS: tl.constexpr,\n",
        "    block_ids_ptr,\n"
        "    n_blocks,\n"
        "    seg_nblocks_ptr,\n"
        "    N_SEGS: tl.constexpr,\n",
    ))

    # 2) kernel body: guard after loading block_id
    subs.append((
        "    block_id = tl.load(block_ids_ptr + block_index)\n"
        "    seg_addr = tl.load(seg_addrs_ptr + seg_index)\n",
        "    block_id = tl.load(block_ids_ptr + block_index)\n"
        "    # k3-kda: bounds-guard OOB block_id (would fault the GPU).\n"
        "    seg_nblk = tl.load(seg_nblocks_ptr + seg_index)\n"
        "    if block_id >= seg_nblk:\n"
        "        return\n"
        "    seg_addr = tl.load(seg_addrs_ptr + seg_index)\n",
    ))

    # 3a) __init__: seg_nblocks list init
    subs.append((
        "        seg_addrs: list[int] = []\n"
        "        seg_page_sizes: list[int] = []\n",
        "        seg_addrs: list[int] = []\n"
        "        seg_page_sizes: list[int] = []\n"
        "        seg_nblocks: list[int] = []  # k3-kda: per-segment block capacity\n",
    ))

    # 3b) __init__: compute + append capacity in the outer loop
    subs.append((
        "                outer_strides = [kv.stride(d) * el for d in outer_dims]\n"
        "                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):\n"
        "                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))\n"
        "                    seg_addrs.append(dp + off_bytes)\n"
        "                    seg_page_sizes.append(cur_page_el)\n",
        "                outer_strides = [kv.stride(d) * el for d in outer_dims]\n"
        "                seg_nblk = int(kv.shape[block_dim]) // max(1, ratio)\n"
        "                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):\n"
        "                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))\n"
        "                    seg_addrs.append(dp + off_bytes)\n"
        "                    seg_page_sizes.append(cur_page_el)\n"
        "                    seg_nblocks.append(seg_nblk)\n",
    ))

    # 3c) __init__: add seg_nblocks tensor to _meta
    subs.append((
        "            max_page_size_el // blk_size,\n"
        "            blk_size,\n"
        "            len(seg_addrs),\n"
        "        )\n",
        "            max_page_size_el // blk_size,\n"
        "            blk_size,\n"
        "            len(seg_addrs),\n"
        "            torch.tensor(seg_nblocks, dtype=torch.int64, device=self.device),\n"
        "        )\n",
    ))

    # 4) zero_block_ids: unpack + pass seg_nblocks
    subs.append((
        "        seg_addrs, seg_page_sizes, max_chunks, blk_size, n_segs = self._meta\n",
        "        seg_addrs, seg_page_sizes, max_chunks, blk_size, n_segs, seg_nblocks = (\n"
        "            self._meta\n"
        "        )\n",
    ))
    subs.append((
        "            idx,\n"
        "            n_blocks,\n"
        "            N_SEGS=n_segs,\n",
        "            idx,\n"
        "            n_blocks,\n"
        "            seg_nblocks,\n"
        "            N_SEGS=n_segs,\n",
    ))

    for old, new in subs:
        if old not in src:
            print(f"[k3-kvzero] WARN anchor not found:\n{old[:80]!r}\n-- ABORT (no partial edits).")
            return 0
        src = src.replace(old, new, 1)

    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-kvzero] ERROR compile failed: {e}", file=sys.stderr)
        return 1
    print("[k3-kvzero] bounds-guarded _zero_kv_blocks_kernel (skip OOB block_id).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
