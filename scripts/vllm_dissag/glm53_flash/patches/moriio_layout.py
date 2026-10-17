# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)


class LayerTransferGeometry(NamedTuple):
    num_blocks: int
    block_size: int
    block_len: int
    slot_size_bytes: int
    block_stride: int
    local_kv_stride: int | None
    remote_kv_stride: int | None
    transfers_per_block: int
    regions_per_block: int
    split_kv_regions: bool


def build_layer_to_spec(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    layer_to_spec: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            layer_to_spec.update(
                {
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in group.layer_names
                }
            )
        else:
            layer_to_spec.update(
                {layer_name: group_spec for layer_name in group.layer_names}
            )
    return layer_to_spec


def is_mla_cache_layer(
    layer_to_spec: Mapping[str, KVCacheSpec], layer_name: str
) -> bool:
    try:
        spec = layer_to_spec[layer_name]
    except KeyError as e:
        raise ValueError(f"Missing KV cache spec for layer {layer_name}") from e
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))


def _content_packed_dim(spec: AttentionSpec) -> int:
    head_size_v = getattr(spec, "head_size_v", spec.head_size)
    return spec.head_size + head_size_v


def _spec_dim_matches(value: int, expected: int | None) -> bool:
    return expected is None or value == expected


def _kernel_layout_matches(
    spec: KVCacheSpec, kernel_block_size: int, num_kv_heads: int, head_dim: int
) -> bool:
    if kernel_block_size <= 0 or spec.block_size % kernel_block_size != 0:
        return False
    return _spec_dim_matches(
        num_kv_heads, getattr(spec, "num_kv_heads", None)
    ) and _spec_dim_matches(head_dim, getattr(spec, "head_size", None))


def _select_kernel_block_layout(
    layer_name: str, shape: torch.Size, spec: KVCacheSpec
) -> tuple[int, int, int]:
    axis2_matches = _kernel_layout_matches(spec, shape[2], shape[3], shape[4])
    axis3_matches = _kernel_layout_matches(spec, shape[3], shape[2], shape[4])

    if axis2_matches and axis3_matches and shape[2] != shape[3]:
        raise ValueError(
            f"Ambiguous MoRIIO kernel-block K/V cache shape for layer "
            f"{layer_name}: {tuple(shape)}"
        )
    if axis2_matches:
        return shape[2], shape[3], shape[4]
    if axis3_matches:
        return shape[3], shape[2], shape[4]

    raise ValueError(
        f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
        f"{tuple(shape)} does not contain block size {spec.block_size}"
    )


def get_layer_transfer_geometry(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    remote_num_blocks: int | None = None,
) -> LayerTransferGeometry:
    shape = kv_cache.shape
    stride = kv_cache.stride()
    element_size = kv_cache.element_size()
    spec = layer_to_spec[layer_name]
    is_mla_cache = is_mla_cache_layer(layer_to_spec, layer_name)

    if len(shape) == 3:
        num_blocks, block_size, latent_dim = shape
        slot_size_bytes = latent_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )

    if len(shape) == 4:
        # DSA indexer tail_cache: (num_blocks, 2, block_size, head_dim) where
        # dim1=2 packs K|score. Single contiguous per-block region; transfer
        # the whole block (both halves together). Correct for same-TP P/D.
        num_blocks = shape[0]
        block_size = shape[2]
        slot_size_bytes = shape[1] * shape[3] * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )

    if not is_mla_cache and len(shape) == 5 and shape[0] == 2:
        _, num_blocks = shape[:2]
        kernel_blocks_per_block = 1
        if shape[2] == spec.block_size:
            block_size, num_kv_heads, head_dim = shape[2:]
        elif shape[3] == spec.block_size:
            num_kv_heads, block_size, head_dim = shape[2:]
        else:
            kernel_num_blocks = num_blocks
            kernel_block_size, num_kv_heads, head_dim = _select_kernel_block_layout(
                layer_name, shape, spec
            )
            kernel_blocks_per_block = spec.block_size // kernel_block_size
            if kernel_num_blocks % kernel_blocks_per_block != 0:
                raise ValueError(
                    f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
                    f"{tuple(shape)} has {kernel_num_blocks} kernel blocks, "
                    f"not divisible by {kernel_blocks_per_block}"
                )
            num_blocks = kernel_num_blocks // kernel_blocks_per_block
            block_size = spec.block_size
        slot_size_bytes = num_kv_heads * head_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[1] * kernel_blocks_per_block,
            local_kv_stride=stride[0],
            remote_kv_stride=(
                stride[1] * kernel_blocks_per_block * (remote_num_blocks or num_blocks)
            ),
            transfers_per_block=2,
            regions_per_block=1,
            split_kv_regions=True,
        )

    if not is_mla_cache and len(shape) == 5 and shape[1] == 2:
        num_blocks = shape[0]
        if shape[2] == spec.block_size:
            block_size, num_kv_heads, head_dim = shape[2:]
            slot_size_bytes = num_kv_heads * head_dim * element_size
            block_len = block_size * slot_size_bytes
            return LayerTransferGeometry(
                num_blocks=num_blocks,
                block_size=block_size,
                block_len=block_len,
                slot_size_bytes=slot_size_bytes,
                block_stride=stride[0],
                local_kv_stride=stride[1],
                remote_kv_stride=stride[1],
                transfers_per_block=2,
                regions_per_block=2,
                split_kv_regions=False,
            )
        elif shape[3] == spec.block_size:
            num_kv_heads, block_size, head_dim = shape[2:]
        else:
            kernel_num_blocks = num_blocks
            kernel_block_size, _, _ = _select_kernel_block_layout(
                layer_name, shape, spec
            )
            kernel_blocks_per_block = spec.block_size // kernel_block_size
            if kernel_num_blocks % kernel_blocks_per_block != 0:
                raise ValueError(
                    f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
                    f"{tuple(shape)} has {kernel_num_blocks} kernel blocks, "
                    f"not divisible by {kernel_blocks_per_block}"
                )
            num_blocks = kernel_num_blocks // kernel_blocks_per_block
            block_size = spec.block_size
            block_stride = stride[0] * kernel_blocks_per_block
            block_len = block_stride * element_size
            slot_size_bytes = block_len // block_size
            return LayerTransferGeometry(
                num_blocks=num_blocks,
                block_size=block_size,
                block_len=block_len,
                slot_size_bytes=slot_size_bytes,
                block_stride=block_stride,
                local_kv_stride=None,
                remote_kv_stride=None,
                transfers_per_block=1,
                regions_per_block=1,
                split_kv_regions=False,
            )
        slot_size_bytes = num_kv_heads * head_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=stride[1],
            remote_kv_stride=stride[1],
            transfers_per_block=2,
            regions_per_block=2,
            split_kv_regions=False,
        )

    if (
        not is_mla_cache
        and isinstance(spec, AttentionSpec)
        and len(shape) == 4
        and shape[1] == spec.num_kv_heads
        and shape[2] == spec.block_size
        and shape[3] == _content_packed_dim(spec)
    ):
        num_blocks, num_kv_heads, block_size, packed_dim = shape
        slot_size_bytes = num_kv_heads * packed_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )

    cache_kind = "MLA" if is_mla_cache else "K/V"
    raise ValueError(
        f"Unsupported MoRIIO {cache_kind} cache shape for layer "
        f"{layer_name}: {tuple(shape)}"
    )


def iter_layer_registration_regions(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> list[tuple[torch.Tensor, int]]:
    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    region_len = geometry.num_blocks * geometry.regions_per_block * geometry.block_len
    if geometry.split_kv_regions:
        return [(cache, region_len) for cache in kv_cache]
    return [(kv_cache, region_len)]


def merge_contiguous_offsets(
    offsets_local: list[int],
    offsets_remote: list[int],
    sizes: list[int],
) -> tuple[list[int], list[int], list[int]]:
    if not offsets_local:
        return [], [], []
    if not (len(offsets_local) == len(offsets_remote) == len(sizes)):
        raise ValueError("Input list lengths mismatch")

    rows = sorted(zip(offsets_local, offsets_remote, sizes), key=lambda row: row[0])
    merged: list[list[int]] = []
    for local, remote, size in rows:
        if (
            merged
            and local == merged[-1][0] + merged[-1][2]
            and remote == merged[-1][1] + merged[-1][2]
        ):
            merged[-1][2] += size
        else:
            merged.append([local, remote, size])

    return (
        [row[0] for row in merged],
        [row[1] for row in merged],
        [row[2] for row in merged],
    )


def _mla_kernel_blocks_per_group_block(
    layer_name: str,
    kv_cache: "torch.Tensor",
    layer_to_spec,
    num_group_blocks=None,
) -> int:
    """Kernel-blocks-per-group-block (kbpb) for an MLA cache layer.

    vLLM's hybrid KV allocator assigns block-ids in the *group* block unit
    (all cache groups share one group-block count). An MLA cache tensor is
    paged at its own kernel block_size, so it holds num_group_blocks * kbpb
    kernel blocks (GLM-5.3-Flash: self_attn.attn 87678/4871=18,
    indexer.k_cache 43839/4871=9). Prefer the true group-block count; the
    spec.block_size // kernel_block_size fallback is correct only for .attn.
    Returns 1 (no expansion) for non-MLA layers or when counts don't divide.
    """
    try:
        if not is_mla_cache_layer(layer_to_spec, layer_name):
            return 1
        shape = kv_cache.shape
        if len(shape) != 3:
            return 1
        num_kernel_blocks = int(shape[0])
        if (
            num_group_blocks
            and num_group_blocks > 0
            and num_kernel_blocks % num_group_blocks == 0
        ):
            return max(1, num_kernel_blocks // num_group_blocks)
        kernel_bs = int(shape[1])
        spec = layer_to_spec[layer_name]
        grp_bs = int(getattr(spec, "block_size", kernel_bs) or kernel_bs)
        if (
            grp_bs > kernel_bs
            and grp_bs % kernel_bs == 0
            and num_kernel_blocks % (grp_bs // kernel_bs) == 0
        ):
            return grp_bs // kernel_bs
        return 1
    except Exception:
        return 1


def compute_block_transfer_offsets(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    local_block_ids: list[int],
    remote_block_ids: list[int],
    remote_num_blocks: int,
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
    num_group_blocks=None,
) -> tuple[list[int], list[int], list[int]]:
    # A shorter (or empty) local list is the READ-mode "drop the transfer, just
    # free the prefill blocks" case (full-prefix-hit / aborted-before-scheduled):
    # decode pulls fewer blocks than the prefill holds. The zip loop below pairs
    # local[i]<->remote[i] and sizes by len(local), so a short local transfers
    # only what decode allocated and an empty local is a no-op. A longer local
    # list is a genuine bug and still fails loudly.
    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids: "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )
    # MLA group->kernel block expansion: a group block-id N covers kbpb
    # contiguous kernel sub-blocks [N*kbpb .. N*kbpb+kbpb-1]. Expand both id
    # lists so each kernel sub-block transfers at its native kernel offset.
    # kbpb == 1 (the common case) leaves the lists unchanged.
    kbpb = _mla_kernel_blocks_per_group_block(
        layer_name, kv_cache, layer_to_spec, num_group_blocks
    )
    if kbpb > 1:
        local_block_ids = [
            lb * kbpb + j for lb in local_block_ids for j in range(kbpb)
        ]
        remote_block_ids = [
            rb * kbpb + j for rb in remote_block_ids for j in range(kbpb)
        ]
    geometry = get_layer_transfer_geometry(
        layer_name, kv_cache, layer_to_spec, remote_num_blocks
    )
    element_size = kv_cache.element_size()
    transfer_size_byte = geometry.block_len
    per_block = geometry.transfers_per_block
    total = len(local_block_ids) * per_block
    offset_local = [0] * total
    offset_remote = [0] * total
    sizes = [transfer_size_byte] * total

    import os as _os
    if _os.environ.get("MORIIO_OFFSET_DBG") == "1" and is_mla_cache_layer(
        layer_to_spec, layer_name
    ):
        _spec = layer_to_spec[layer_name]
        print(
            "[OFFDBG-OVL] layer={} shape={} stride={} spec_bs={} | "
            "local_ids={} remote_ids={} | geom(num_blocks={} block_size={} "
            "block_len={} block_stride={} kv_stride={}/{} per_block={} "
            "regions={} split={}) elem={} remote_num_blocks={} kbpb_applied={}".format(
                layer_name, tuple(kv_cache.shape), kv_cache.stride(),
                getattr(_spec, "block_size", "?"),
                list(local_block_ids)[:12], list(remote_block_ids)[:12],
                geometry.num_blocks, geometry.block_size, geometry.block_len,
                geometry.block_stride, geometry.local_kv_stride,
                geometry.remote_kv_stride, per_block, geometry.regions_per_block,
                geometry.split_kv_regions, element_size, remote_num_blocks,
                (len(local_block_ids)),
            ),
            flush=True,
        )

    w = 0
    for lb, rb in zip(local_block_ids, remote_block_ids):
        offset_local[w] = element_size * (lb * geometry.block_stride)
        offset_remote[w] = element_size * (rb * geometry.block_stride)
        w += 1
        if per_block == 2:
            assert geometry.local_kv_stride is not None
            assert geometry.remote_kv_stride is not None
            offset_local[w] = element_size * (
                geometry.local_kv_stride + lb * geometry.block_stride
            )
            offset_remote[w] = element_size * (
                geometry.remote_kv_stride + rb * geometry.block_stride
            )
            w += 1

    return merge_fn(offset_local, offset_remote, sizes)
