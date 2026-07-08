#!/usr/bin/env python3
"""Force-compile the GLM-5.1 DSA sparse-attention indexer Triton kernels at BOOT.

PROBLEM (root cause of the "first big prompt stalls the whole DP group" hang):
  The DSA indexer's Triton kernels are seq-length specialized:
    - v1/attention/ops/triton_fp8_mqa_logits.py flips `matrix_instr_nonkdim`
      at seq_len<=1024 and launches with grid=[(seq_len,)] (seq_len is a
      specialized kernel arg) -> a >1024-row prefill needs a *different* JIT
      specialization than a <=1024-row one.
  But the boot-time warmup never drives the indexer:
    - profile_run() calls _dummy_run(is_profile=True) with force_attention=False
      and cudagraph mode NONE -> attn_metadata stays None -> sparse_attn_indexer
      takes the `sparse_attn_indexer_fake` path (see the "careful! this will be
      None in dummy run" comment in layers/sparse_attn_indexer.py). The real
      kernels are never compiled.
    - _warmup_and_capture() only sets force_attention=True when the cudagraph
      runtime mode is FULL; the DSA indexer builder reports UNIFORM_BATCH, so on
      this ROCm/DP build the mixed prefill-decode graphs are PIECEWISE and
      force_attention stays False. Even when attention IS forced, capture uses
      uniform-decode / small mixed batches -- never a large prefill at
      max_num_batched_tokens -- so the >1024 specialization is still absent.
  Effect: the first >=8k prompt JIT-compiles the indexer kernel mid-inference on
  whichever DP rank happens to receive it. That rank falls out of the DP lockstep
  gloo all_reduce (coordinate_batch_across_dp) while it compiles -> the whole DP
  group collapses. It is also a general cold-cache robustness hole.

FIX (surgical, reuses vLLM's OWN metadata construction -- no hand-synthesized
tensors, so zero risk of a bad-input crash at boot):
  1. gpu_model_runner.py: add a `_maybe_warmup_dsa_indexer()` method. It is a
     strict NO-OP unless one of the runner's attention backends is (a subclass
     of) DeepseekV32IndexerBackend. When present, it runs
     `_dummy_run(..., force_attention=True, cudagraph_runtime_mode=NONE)` at TWO
     prefill-size regimes -- a small one (<=1024 rows) and a large one
     (max_num_batched_tokens, >1024) -- so BOTH Triton specializations compile.
     `force_attention=True` makes _dummy_run build a real
     DeepseekV32IndexerMetadata via the normal _build_attention_metadata path
     (num_prefills>0 because the default dummy batch is multi-token requests and
     the indexer decode_threshold is 1), which drives the real
     `sparse_attn_indexer` prefill kernels. The whole thing is wrapped in
     try/except that only WARNs -- a warmup failure must never crash boot.
  2. gpu_worker.py: call it from compile_or_warm_up_model, right after the
     existing warmup loop and before kernel_warmup(). At that point the KV cache
     is already allocated (initialize_from_config runs before
     compile_or_warm_up_model), which the forced-attention indexer path needs.

Idempotent + anchor-based (matches the other apply_glm_* patchers):
  * Each hunk self-detects if already applied (marker string present) -> no-op.
  * Missing anchor -> WARN and skip that hunk (safe across vllm revisions; the
    rebase may already warm the indexer natively or have refactored the site).
  * Anchor found but the file does not contain the applied marker and the
    replace produces no change -> hard error (would silently keep the bug).
  * py_compile at the end; hard error if the patched file won't compile.

Usage: apply_glm_dsa_indexer_warmup_fix.py <vllm_install_dir>
"""
import os
import sys

RUNNER_REL = "v1/worker/gpu_model_runner.py"
WORKER_REL = "v1/worker/gpu_worker.py"

MARKER = "glm-dsa-indexer-warmup"

# --- Hunk A: new method inserted immediately before `def capture_model` -------
# Anchor: the (unique) capture_model definition head in gpu_model_runner.py.
RUNNER_ANCHOR = "    def capture_model(self) -> int:\n"

RUNNER_METHOD = '''    def _maybe_warmup_dsa_indexer(self) -> None:
        """Force-compile the DSA sparse-attention indexer Triton kernels at boot.

        NO-OP unless this model actually has a DeepseekV32IndexerBackend (GLM-5.1
        DSA / DeepSeek V3.2). The indexer kernels are seq-length specialized
        (triton_fp8_mqa_logits flips matrix_instr_nonkdim at seq_len<=1024 and
        launches grid=[(seq_len,)]), and the normal profile/warmup passes never
        drive the indexer (attn_metadata is None -> the *_fake path). Without this
        the first large prompt JIT-compiles mid-inference and, under DP lockstep,
        stalls the whole group. We warm BOTH regimes: a small (<=1024) and a large
        (max_num_batched_tokens, >1024) prefill batch, using force_attention=True
        so _dummy_run builds a real indexer metadata via the standard path.
        """
        # {marker}
        try:
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerBackend,
            )
        except Exception:  # noqa: BLE001 -- backend module absent -> not a DSA build
            return

        has_indexer = False
        try:
            for attn_group in self._attn_group_iterator():
                backend = getattr(attn_group, "backend", None)
                if backend is not None and isinstance(backend, type) and issubclass(
                    backend, DeepseekV32IndexerBackend
                ):
                    has_indexer = True
                    break
        except Exception:  # noqa: BLE001 -- iterator shape changed -> stay a no-op
            return
        if not has_indexer:
            return

        # Two prefill-size regimes so both Triton specializations compile.
        # Small must be <=1024 rows; large must exceed 1024 (use the real max).
        max_tokens = int(self.max_num_tokens)
        small = min(512, max_tokens)
        sizes = []
        for s in (small, max_tokens):
            if s > 0 and s not in sizes:
                sizes.append(s)

        logger.info(
            "Warming up DSA indexer kernels at prefill sizes %s "
            "to avoid mid-inference JIT.",
            sizes,
        )
        for size in sizes:
            try:
                self._dummy_run(
                    size,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=True,
                    skip_eplb=True,
                    remove_lora=False,
                )
            except Exception as e:  # noqa: BLE001 -- warmup must NEVER crash boot
                logger.warning(
                    "DSA indexer warmup at size %d failed (%s); the kernel may "
                    "JIT-compile on first use instead.",
                    size,
                    e,
                )
        self._sync_device()

'''.replace("{marker}", MARKER)

# --- Hunk B: call site in gpu_worker.compile_or_warm_up_model -----------------
WORKER_ANCHOR = (
    "        self.model_runner.maybe_remove_all_loras("
    "self.model_runner.lora_config)\n"
    "\n"
    "        # Warmup and tune the kernels used during model execution before\n"
    "        # cuda graph capture.\n"
    "        kernel_warmup(self)\n"
)

WORKER_REPLACEMENT = (
    "        self.model_runner.maybe_remove_all_loras("
    "self.model_runner.lora_config)\n"
    "\n"
    "        # " + MARKER + ": force-compile the DSA sparse-attention indexer\n"
    "        # Triton kernels now (KV cache is allocated), so a large prompt\n"
    "        # never JIT-compiles them mid-inference and stalls DP lockstep.\n"
    "        # No-op unless this model has a DeepseekV32IndexerBackend.\n"
    "        if hasattr(self.model_runner, \"_maybe_warmup_dsa_indexer\"):\n"
    "            self.model_runner._maybe_warmup_dsa_indexer()\n"
    "\n"
    "        # Warmup and tune the kernels used during model execution before\n"
    "        # cuda graph capture.\n"
    "        kernel_warmup(self)\n"
)


def _patch_file(path, tag, anchor, apply_fn, already_marker):
    """Return 0 on success/no-op, 1 on hard error."""
    if not os.path.isfile(path):
        print(f"[{tag}] {path} not found -- skipping (layout differs).")
        return 0
    src = open(path).read()
    if already_marker in src:
        print(f"[{tag}] already applied ({already_marker} present) in {path} -- no-op.")
        return 0
    if anchor not in src:
        print(
            f"[{tag}] WARN: anchor not found in {path} -- skipping "
            "(assuming native warmup / refactor)."
        )
        return 0
    new_src = apply_fn(src)
    if new_src == src:
        print(
            f"[{tag}] ERROR: anchor found but patch produced no change in {path}.",
            file=sys.stderr,
        )
        return 1
    try:
        open(path, "w").write(new_src)
    except OSError as e:
        print(f"[{tag}] ERROR: failed to write patched {path}: {e}", file=sys.stderr)
        return 1
    if already_marker not in open(path).read():
        print(
            f"[{tag}] ERROR: post-write verification failed in {path}.",
            file=sys.stderr,
        )
        return 1
    print(f"[{tag}] patched {path} -- 1 hunk.")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    vllm_dir = sys.argv[1]

    runner_path = os.path.join(vllm_dir, RUNNER_REL)
    worker_path = os.path.join(vllm_dir, WORKER_REL)

    rc = 0

    # Hunk A: insert the method before capture_model.
    rc |= _patch_file(
        runner_path,
        "glm-dsa-warmup",
        RUNNER_ANCHOR,
        lambda s: s.replace(RUNNER_ANCHOR, RUNNER_METHOD + RUNNER_ANCHOR, 1),
        MARKER,
    )

    # Hunk B: call it from compile_or_warm_up_model.
    rc |= _patch_file(
        worker_path,
        "glm-dsa-warmup",
        WORKER_ANCHOR,
        lambda s: s.replace(WORKER_ANCHOR, WORKER_REPLACEMENT, 1),
        MARKER,
    )

    if rc:
        return 1

    # py-compile sanity for whichever files exist.
    try:
        import py_compile

        for p in (runner_path, worker_path):
            if os.path.isfile(p):
                py_compile.compile(p, doraise=True)
        print("[glm-dsa-warmup] py_compile OK")
    except Exception as e:  # noqa: BLE001
        print(
            f"[glm-dsa-warmup] ERROR: patched file fails to compile: {e}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
