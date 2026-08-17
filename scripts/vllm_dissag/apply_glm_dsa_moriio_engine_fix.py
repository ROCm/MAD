#!/usr/bin/env python3
"""Fix MoRIIO WRITE-path per-layer offset caching for GLM-5.1 DSA dual KV cache.

ROOT CAUSE (proven by instrumentation, job 37594):
  GLM-5.1 (GlmMoeDsaForCausalLM) registers num_layers=156 KV caches = 78 main MLA
  (per-block latent dim 576) + 78 DSA indexer caches (latent dim 132). TWO geometries.

  moriio_engine.py::MoRIIOEngine._prepare_transfer_plan computes the RDMA transfer
  offsets ONCE (for whatever layer arrives first) and caches them on
  request_info.transfer_offset, then REUSES that single offset/size tuple for ALL 156
  layers. The 78 indexer layers (dim 132) get written with the main-MLA geometry
  (dim 576) -> wrong byte size/offset -> those RDMA writes are malformed; the per-layer
  write accounting (writes_done) and/or the remote completion never reconciles ->
  the producer's send_notify (gated on writes_done >= num_layers) misbehaves and the
  decode side never receives a clean completion -> "Reaped deferred sends / no
  finished_sending after 60s" -> request hangs.

FIX (surgical, no dataclass change):
  Cache transfer offsets PER LAYER on the request_info via a dynamically-attached dict
  ``_transfer_offset_by_layer`` keyed by layer_name, instead of the single
  ``transfer_offset`` slot. Each of the 156 layers then transfers with its OWN geometry
  (the underlying _compute_block_transfer_offsets already takes layer_name and, with the
  companion dualkv patch h2, reads the per-layer tensor shape).

  Single-geometry models (DeepSeek-V3 / Hunyuan, 1 cache/layer) are unaffected:
  every layer has identical geometry, so per-layer caching yields the same offsets.

Idempotent + anchor-based. A missing anchor warns-and-skips; a found OLD anchor that
fails to apply is a hard error (would silently keep the stall).

Usage: apply_glm_dsa_moriio_engine_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-engine] {REL} not found -- skipping (engine layout differs).")
        return 0

    src = open(path).read()

    old = """        # Compute offsets if not cached
        if request_info.transfer_offset is None:
            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name,
                task.local_block_ids,
                request_info.block_ids,
                remote_moriio_meta,
            )
            request_info.transfer_offset = offsets

        # Get session index
        layer_names = list(self.worker.layer_name_to_local_kv_cache_metadata.keys())
        sess_idx = layer_names.index(task.layer_name)

        local_off, remote_off, sizes = request_info.transfer_offset"""

    new = """        # DSA dual-KV fix: cache offsets PER LAYER, not once per request. GLM-5.1 has
        # two cache geometries (main MLA dim 576 + DSA indexer dim 132); a single
        # cached offset reused across all 156 layers mis-sizes the indexer writes and
        # the completion never reconciles. Per-layer caching is identical for
        # single-geometry models (DeepSeek/Hunyuan).
        _off_by_layer = getattr(request_info, "_transfer_offset_by_layer", None)
        if _off_by_layer is None:
            _off_by_layer = {}
            request_info._transfer_offset_by_layer = _off_by_layer
        offsets = _off_by_layer.get(task.layer_name)
        if offsets is None:
            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name,
                task.local_block_ids,
                request_info.block_ids,
                remote_moriio_meta,
            )
            _off_by_layer[task.layer_name] = offsets
            # keep the legacy single-slot populated (first layer) for any external reader
            if request_info.transfer_offset is None:
                request_info.transfer_offset = offsets

        # Get session index
        layer_names = list(self.worker.layer_name_to_local_kv_cache_metadata.keys())
        sess_idx = layer_names.index(task.layer_name)

        local_off, remote_off, sizes = offsets"""

    if "_transfer_offset_by_layer" in src:
        print(f"[glm-engine] already patched (_transfer_offset_by_layer present) -- no-op.")
    elif old in src:
        src = src.replace(old, new, 1)
        open(path, "w").write(src)
        print(f"[glm-engine] patched per-layer offset caching in {path}")
    else:
        print(f"[glm-engine] WARN: anchor (_prepare_transfer_plan offset block) not found -- skipping (engine revision differs).")
        # Not fatal: without the anchor we can't safely patch; surface clearly.
        return 0

    try:
        import py_compile
        py_compile.compile(path, doraise=True)
        print("[glm-engine] py_compile OK")
    except Exception as e:  # noqa: BLE001
        print(f"[glm-engine] ERROR: compile failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
