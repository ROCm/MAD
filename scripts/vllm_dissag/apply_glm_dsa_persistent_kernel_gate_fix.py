#!/usr/bin/env python3
"""Gate OFF the AITER persistent sparse-MLA kernel for chunked-prefill batches.

ROOT CAUSE (ROCm/aiter #4076, vLLM #47042 / #47567):
  The AITER persistent MLA work-stealing kernel (mla_a8w8_qh16_qseqlen1_gqaratio16_ps,
  taken when work_meta_data from get_mla_metadata_v1 is non-None) is NUMERICALLY WRONG
  for multi-token (prefill-shaped) batches of qseqlen==1 entries. Pure decode (1 query
  token) and fresh single-chunk prefills are correct; the error only appears once a
  request becomes a CHUNKED-PREFILL CONTINUATION. The small per-token error COMPOUNDS
  through the KV cache across chunked-prefill passes until long-context decode collapses
  into repetition/garbage. Failure is gated on CHUNK COUNT, not raw context length
  (verified: 22k in 2 chunks = correct, 22k in 3 chunks = garbage).

  On this image (aiter 0.1.16.post3, before the aiter-side kernel fix #3921) GLM-5.1-FP8
  DSA collapses at ~16-18k prompt tokens. The aiter kernel fix is the long-term answer
  (AITERKER-132 / aiter #3921); this is the vLLM-side short-term gate (#47567), which
  costs ~no perf (decode + single-chunk prefill keep the persistent path).

FIX (port of vLLM PR #47567, adapted to this image's rocm_aiter_mla_sparse.py::build):
  In ROCMAiterMLASparseMetadataBuilder.build(), detect chunked-prefill continuations
  (a request with >1 query token this step whose total seq_len exceeds its query_len,
  i.e. part of its context was computed in an earlier chunk) and, when ANY request in
  the batch is such a continuation:
    * skip the get_mla_metadata_v1 persistent-metadata launch, and
    * pass work_meta_data=None to the metadata so mla_decode_fwd takes the CORRECT
      non-persistent split-KV path.
  Decode-only and single-chunk-prefill batches are unchanged (persistent path kept).

  Uses `seg_lengths` (per-request step query lengths, already computed at build() top)
  and `common_attn_metadata.seq_lens_cpu[:num_reqs].numpy()` (total seq lens). Both are
  present in this image's build().

Idempotent + anchor-based + self-skipping. Missing anchor -> warn+skip (safe across
image revisions / if a newer image already carries the aiter kernel fix). A found-old
anchor that fails to apply is a hard error (would silently keep the corruption).

Usage: apply_glm_dsa_persistent_kernel_gate_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"

# Anchor 1: the persistent-metadata guard. We insert the continuation detection
# just before it and AND it into the condition.
OLD1 = """        if metadata_key != self._prev_metadata_key:
            from aiter import get_mla_metadata_v1"""
NEW1 = """        # PERSISTENT-KERNEL GATE (aiter #4076 / vLLM #47567): the persistent
        # sparse-MLA work-stealing kernel is numerically wrong for chunked-prefill
        # continuation batches; the error compounds and breaks long-context decode.
        # Fall back to the correct non-persistent path whenever any request in the
        # batch is a chunked-prefill continuation (>1 query token this step AND
        # total seq_len > this step's query_len). Decode + single-chunk prefills
        # keep the fast persistent path -> no decode-throughput regression.
        # Slice to num_reqs and cast to int64 (vLLM #47567 hardening / Rohan138 PR#1)
        # so the masks cannot broadcast-mismatch under cudagraph padding.
        _step_query_lens = seg_lengths[:num_reqs].astype(np.int64)
        _total_seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs].numpy().astype(
            np.int64
        )
        _is_chunked_continuation = (_step_query_lens > 1) & (
            _total_seq_lens > _step_query_lens
        )
        _use_persistent = not bool(_is_chunked_continuation.any())
        if _use_persistent and metadata_key != self._prev_metadata_key:
            from aiter import get_mla_metadata_v1"""

# Anchor 2: the metadata construction passes the persistent buffer unconditionally.
# Gate it on _use_persistent.
OLD2 = "            work_meta_data=self._mla_work_meta_data,"
NEW2 = "            work_meta_data=(self._mla_work_meta_data if _use_persistent else None),"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-persist] {REL} not found -- skipping (backend layout differs).")
        return 0

    src = open(path).read()

    if "_is_chunked_continuation" in src or "_use_persistent" in src:
        print("[glm-persist] already patched (persistent-kernel gate present) -- no-op.")
        return 0

    # Both anchors must be present to apply safely.
    if OLD1 not in src:
        print("[glm-persist] WARN: persistent-metadata anchor (metadata_key guard) not "
              "found -- skipping (image may already carry the aiter kernel fix, or the "
              "backend was refactored).")
        return 0
    if OLD2 not in src:
        print("[glm-persist] ERROR: found the metadata_key guard but NOT the "
              "work_meta_data=self._mla_work_meta_data assignment -- refusing partial "
              "patch (would leave persistent kernel active). Aborting.", file=sys.stderr)
        return 1

    src = src.replace(OLD1, NEW1, 1)
    src = src.replace(OLD2, NEW2, 1)

    try:
        open(path, "w").write(src)
    except OSError as e:
        print(f"[glm-persist] ERROR: write failed for {path}: {e}", file=sys.stderr)
        return 1

    # Verify both edits landed.
    chk = open(path).read()
    if "_use_persistent = not bool(_is_chunked_continuation.any())" not in chk or \
       "if _use_persistent else None" not in chk:
        print("[glm-persist] ERROR: post-write verification failed.", file=sys.stderr)
        return 1

    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[glm-persist] ERROR: patched file fails to compile: {e}", file=sys.stderr)
        return 1

    print(f"[glm-persist] patched persistent-kernel gate (aiter #4076 / vLLM #47567) in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
