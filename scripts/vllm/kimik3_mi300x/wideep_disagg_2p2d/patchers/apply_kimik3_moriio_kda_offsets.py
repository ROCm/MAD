#!/usr/bin/env python3
"""H2: mamba conv/ssm sub-projection OFFSET builder for MoRIIO (Kimi-K3 KDA).

Companion to apply_kimik3_moriio_kda_fix.py (which adds the MambaSpec geometry
branch to get_layer_transfer_geometry). This patcher makes the actual byte
offsets for a mamba layer's RDMA read cover the conv sub-projections [Q,K,V] and
the ssm/recurrent state, instead of a single uniform block_len region.

WHY a separate builder: compute_block_transfer_offsets() (moriio_layout.py) emits,
per (local_block, remote_block) pair, ONE transfer of `geometry.block_len` bytes
at `block_stride*block_id`. For an MLA/attention page that's correct. For a mamba
page the payload is a set of contiguous sub-regions within the page:
    [ Q | K | V | ssm ]   (DS layout; conv sub-projections then temporal state)
Each sub-region is (offset_within_page, size) from MambaConvSplitInfo:
    local_conv_offsets  -> [(0,Qb),(Qb,Kb),(Qb+Kb,Vb)]     conv sub-projections
    ssm follows conv:      (conv_bytes, ssm_bytes)
For homogeneous TP (2P/2D EP16: P_TP==D_TP==16 -> tp_ratio=1, local_rank_offset=0)
the remote offsets equal the local offsets, so per block we emit the SAME
(off, size) list on both sides, shifted by block_stride*block_id.

This mirrors nixl/base_worker.py _build_mamba_local/_build_mamba_remote, reduced
to the homogeneous-TP case (which is all 2P/2D EP16 needs). Hetero-TP is guarded
with a clear NotImplementedError (K3 disagg runs symmetric pools).

Approach: insert a dedicated `compute_mamba_block_transfer_offsets(...)` helper
into moriio_layout.py and route mamba layers to it from
`compute_block_transfer_offsets` (top-of-function dispatch on MambaSpec).

Idempotent + anchor-based, py_compile-checked. Usage:
    apply_kimik3_moriio_kda_offsets.py <vllm_install_dir>
"""
import os
import sys

LAYOUT = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], LAYOUT)
    if not os.path.isfile(path):
        print(f"[k3-kda-off] {LAYOUT} not found -- ABORT.")
        return 1
    src = open(path).read()
    orig = src

    if "compute_mamba_block_transfer_offsets" in src:
        print("[k3-kda-off] already applied.")
        return 0

    # The helper. `spec` carries the MambaSpec; we recompute the split locally
    # (cheap; called per layer per transfer batch, but small). block_stride is in
    # ELEMENTS (matches attention path which multiplies by element_size), so for
    # mamba we express stride in bytes via the page and pass element_size=1 by
    # using byte offsets directly.
    helper = '''

def compute_mamba_block_transfer_offsets(
    layer_name,
    kv_cache,
    spec,
    local_block_ids,
    remote_block_ids,
    remote_num_blocks,
    merge_fn,
):
    """k3-kda: byte offsets for a mamba (KDA/GDN) layer's conv+ssm sub-regions.

    Emits, per (local_block, remote_block) pair, one transfer per sub-region:
    conv sub-projections [Q,K,V] then the ssm/recurrent state. Homogeneous TP
    only (P_TP == D_TP); tp_ratio == 1 so remote offsets == local offsets.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        derive_mamba_conv_split,
    )

    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids (mamba): "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )

    split = derive_mamba_conv_split(spec, local_tp=1)
    conv_bytes, ssm_bytes = split.ssm_sizes
    page = int(conv_bytes + ssm_bytes)
    # Sub-regions within one page: conv sub-projections, then ssm.
    subregions = list(split.local_conv_offsets)  # [(off,size), ...] for Q,K,V
    subregions.append((int(conv_bytes), int(ssm_bytes)))  # ssm follows conv

    # Byte stride between blocks = full page (state blocks are indivisible; one
    # logical block == one physical page for mamba).
    stride = page

    n = len(local_block_ids) * len(subregions)
    offset_local = [0] * n
    offset_remote = [0] * n
    sizes = [0] * n
    w = 0
    for lb, rb in zip(local_block_ids, remote_block_ids):
        lbase = lb * stride
        rbase = rb * stride
        for off, sz in subregions:
            offset_local[w] = lbase + off
            offset_remote[w] = rbase + off  # tp_ratio==1 -> same sub-offset
            sizes[w] = sz
            w += 1
    return merge_fn(offset_local, offset_remote, sizes)
'''

    # Insert helper right before compute_block_transfer_offsets def.
    anchor_def = "def compute_block_transfer_offsets("
    if anchor_def not in src:
        print("[k3-kda-off] ERROR: compute_block_transfer_offsets def not found. ABORT.")
        return 1
    src = src.replace(anchor_def, helper.lstrip("\n") + "\n\n" + anchor_def, 1)

    # Route mamba layers to the helper at the top of compute_block_transfer_offsets.
    # Anchor: the body's first real statement (the length guard).
    route_anchor = (
        "    # A shorter (or empty) local list is the READ-mode"
    )
    route_code = (
        "    from vllm.v1.kv_cache_interface import MambaSpec as _K3MambaSpec  # k3-kda\n"
        "    _spec = layer_to_spec[layer_name]\n"
        "    if isinstance(_spec, _K3MambaSpec):  # k3-kda: conv/ssm sub-regions\n"
        "        return compute_mamba_block_transfer_offsets(\n"
        "            layer_name, kv_cache, _spec, local_block_ids,\n"
        "            remote_block_ids, remote_num_blocks, merge_fn,\n"
        "        )\n"
        "    # A shorter (or empty) local list is the READ-mode"
    )
    if route_anchor in src:
        src = src.replace(route_anchor, route_code, 1)
    else:
        print("[k3-kda-off] ERROR: compute_block_transfer_offsets body anchor "
              "not found. ABORT.")
        return 1

    if src != orig:
        open(path, "w").write(src)
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[k3-kda-off] ERROR: compile failed: {e}", file=sys.stderr)
            return 1
        print(f"[k3-kda-off] patched {LAYOUT} (mamba offset builder + routing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
