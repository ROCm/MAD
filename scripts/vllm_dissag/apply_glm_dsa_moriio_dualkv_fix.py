#!/usr/bin/env python3
"""Patch the MoRIIO KV connector to handle GLM-5.1 DSA's DUAL KV cache.

PROBLEM (root cause of the 2P2D "Reaped deferred sends / no finished_sending" stall):
  GLM-5.1 (GlmMoeDsaForCausalLM -> deepseek_v2.py) has TWO KV caches per layer:
    - main MLA latent KV  : MLAAttentionSpec, head_size = kv_lora_rank+rope (~576)
    - DSA indexer KV      : DeepseekV32IndexerCache, MLAAttentionSpec head_size = index_head_dim (128)
  Both are 3D ("use_mla"), but DIFFERENT latent dim -> DIFFERENT per-block byte size.
  The MoRIIO connector computes ONE global geometry from `first_kv_cache` and reuses it
  for every cache, so the indexer cache is transferred with the main-MLA block size ->
  wrong bytes/size -> the RDMA read for that region never reconciles -> completion notify
  is never produced -> decode reaps deferred sends after 60s -> request hangs.

FIX (surgical, per-layer geometry; no behavior change for single-cache MLA/DeepSeek):
  1. register_kv_caches: size each registered region by its OWN tensor (per-cache
     region_len), not the global self.block_len. Also fix the local_kv_cache_size
     append to use the current cache, not a stale loop var.
  2. _compute_block_transfer_offsets: derive shape from the PER-LAYER tensor
     (self.kv_caches[layer_name].shape) instead of the global self.kv_cache_shape,
     so transfer_size_byte / strides match that cache.
  3. _read_blocks: compute offsets PER LAYER inside the loop (was computed once from
     first_layer and reused for all layers).

Idempotent + anchor-based: each hunk checks if already applied / anchor present;
missing anchor -> warn-and-skip (so it is safe across connector revisions). A hunk
that finds its OLD anchor but fails to apply is a hard error (would silently keep the bug).

Usage: apply_glm_dsa_moriio_dualkv_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-dualkv] {REL} not found -- skipping (connector layout differs).")
        return 0

    src = open(path).read()
    orig = src
    applied = []

    # --- Hunk 1: per-cache region_len in register_kv_caches ---------------------
    h1_old = """        for cache_or_caches in kv_caches.values():
            cache_list = [cache_or_caches] if use_mla else cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                caches_data.append((base_addr, region_len, cache.device.index, ""))
                kv_caches_base_addr.append(base_addr)"""
    h1_new = """        for cache_or_caches in kv_caches.values():
            cache_list = [cache_or_caches] if use_mla else cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
                # DSA dual-KV fix: size each region by its OWN tensor, not the
                # global self.block_len (the DSA indexer cache has a different
                # latent dim than the main MLA cache).
                region_len = cache.nelement() * cache.element_size()
                caches_data.append((base_addr, region_len, cache.device.index, ""))
                kv_caches_base_addr.append(base_addr)"""
    if "region_len = cache.nelement() * cache.element_size()" in src:
        applied.append("h1 (already)")
    elif h1_old in src:
        src = src.replace(h1_old, h1_new, 1)
        applied.append("h1")
    else:
        print("[glm-dualkv] WARN: h1 anchor (region_len loop) not found -- skipping h1.")

    # --- Hunk 1b: local_kv_cache_size uses current kv_cache, not stale `cache` --
    h1b_old = "            self.local_kv_cache_size.append(cache.nelement() * cache.element_size())"
    h1b_new = "            self.local_kv_cache_size.append(kv_cache.nelement() * kv_cache.element_size())"
    if h1b_new in src:
        applied.append("h1b (already)")
    elif h1b_old in src:
        src = src.replace(h1b_old, h1b_new, 1)
        applied.append("h1b")
    else:
        print("[glm-dualkv] WARN: h1b anchor (local_kv_cache_size) not found -- skipping h1b.")

    # --- Hunk 2: per-layer shape in _compute_block_transfer_offsets -------------
    h2_old = """        assert self.kv_cache_shape is not None, "KV caches shape not initialized"
        is_mla = len(self.kv_cache_shape) == 3
        stride = self.kv_caches[layer_name].stride()
        sz = self.kv_caches[layer_name].element_size()
        if is_mla:
            blknum, blksize, hs = self.kv_cache_shape
            hn = 1
            block_stride = stride[0]
        else:
            _, blknum, blksize, hn, hs = self.kv_cache_shape"""
    h2_new = """        # DSA dual-KV fix: use the PER-LAYER tensor shape, not the global
        # self.kv_cache_shape (the DSA indexer cache differs from the main MLA).
        _layer_shape = tuple(self.kv_caches[layer_name].shape)
        assert len(_layer_shape) > 0, "KV caches shape not initialized"
        is_mla = len(_layer_shape) == 3
        stride = self.kv_caches[layer_name].stride()
        sz = self.kv_caches[layer_name].element_size()
        if is_mla:
            blknum, blksize, hs = _layer_shape
            hn = 1
            block_stride = stride[0]
        else:
            _, blknum, blksize, hn, hs = _layer_shape"""
    if "_layer_shape = tuple(self.kv_caches[layer_name].shape)" in src:
        applied.append("h2 (already)")
    elif h2_old in src:
        src = src.replace(h2_old, h2_new, 1)
        applied.append("h2")
    else:
        print("[glm-dualkv] WARN: h2 anchor (_compute_block_transfer_offsets head) not found -- skipping h2.")

    # --- Hunk 3: per-layer offsets in _read_blocks -----------------------------
    h3_old = """        first_layer = list(self.layer_name_to_local_kv_cache_metadata.keys())[0]
        offs = self._compute_block_transfer_offsets(
            first_layer, local_block_ids, remote_block_ids, remote_moriio_meta
        )

        for layer_name in self.layer_name_to_local_kv_cache_metadata:
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(
                layer_name
            )
            # TODO : apply multi-session batch-read when moriio support it
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )"""
    h3_new = """        # DSA dual-KV fix: compute offsets PER LAYER (the DSA indexer cache has a
        # different per-block size than the main MLA cache, so a single offs reused
        # across all layers mis-sizes the indexer transfer -> lost completion notify).
        for layer_name in self.layer_name_to_local_kv_cache_metadata:
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(
                layer_name
            )
            offs = self._compute_block_transfer_offsets(
                layer_name, local_block_ids, remote_block_ids, remote_moriio_meta
            )
            # TODO : apply multi-session batch-read when moriio support it
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )"""
    if "compute offsets PER LAYER" in src:
        applied.append("h3 (already)")
    elif h3_old in src:
        src = src.replace(h3_old, h3_new, 1)
        applied.append("h3")
    else:
        print("[glm-dualkv] WARN: h3 anchor (_read_blocks first_layer offsets) not found -- skipping h3.")

    if src != orig:
        try:
            open(path, "w").write(src)
        except OSError as e:
            print(f"[glm-dualkv] ERROR: write failed for {path}: {e}", file=sys.stderr)
            return 1
        print(f"[glm-dualkv] patched {path} -- hunks: {', '.join(applied)}")
    else:
        print(f"[glm-dualkv] no changes ({', '.join(applied) or 'nothing applied'}) for {path}")

    # py-compile sanity
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
        print("[glm-dualkv] py_compile OK")
    except Exception as e:  # noqa: BLE001
        print(f"[glm-dualkv] ERROR: patched file fails to compile: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
