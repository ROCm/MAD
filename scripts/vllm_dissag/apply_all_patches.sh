#!/usr/bin/env bash
# apply_all_patches.sh — apply every GLM-5.2 disagg (TP8 / EP8 / EP16 + MTP) source
# patch against an installed vLLM + AITER, idempotently and in a safe order.
#
# All apply_glm_*.py scripts are anchor-based and idempotent: re-running is a no-op,
# a missing anchor warns-and-skips, and a *changed* anchor is a hard error (so a vLLM
# bump can't silently drop a fix). Run this once inside the serving container/image
# after pip-installing vLLM + AITER, before launching prefill/decode.
#
# Usage:
#   apply_all_patches.sh [VLLM_DIR] [AITER_DIR]
# If dirs are omitted they are auto-detected via `python3 -c "import vllm/aiter"`.
#
# Exit non-zero if any patch hits a changed anchor (needs a human to re-anchor).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_DIR="${1:-$(python3 -c 'import os,vllm;print(os.path.dirname(vllm.__file__))' 2>/dev/null)}"
AITER_DIR="${2:-$(python3 -c 'import os,aiter;print(os.path.dirname(aiter.__file__))' 2>/dev/null)}"

echo "[apply-all] VLLM_DIR=${VLLM_DIR:-<not found>}"
echo "[apply-all] AITER_DIR=${AITER_DIR:-<not found>}"
[ -z "${VLLM_DIR}" ] && { echo "[apply-all] FATAL: vLLM not found; pass VLLM_DIR explicitly." >&2; exit 2; }

rc=0

# --- vLLM-targeted patches (order: connector/KV first, then DP/startup gates) ---
VLLM_PATCHES=(
  # MoRIIO connector / KV-transfer correctness (needed by all disagg modes)
  apply_glm_dsa_moriio_engine_fix.py
  apply_glm_dsa_moriio_gate_fix.py
  apply_glm_dsa_moriio_dualkv_fix.py
  apply_glm_moriio_abort_guard_fix.py
  # MTP KV-block correctness (enables TP8-MTP / EP8-MTP / EP16-MTP)
  apply_glm_dsa_moriio_mtp_blockfix.py
  # GLM sparse-MLA / DSA kernel + indexer + sampling correctness
  apply_glm_dsa_kernel_fix.py
  apply_glm_dsa_indexer_warmup_fix.py
  apply_glm_dsa_persistent_kernel_gate_fix.py
  apply_glm_aiter_sampling_oob_fix.py
  # EP16 (DP16 cross-node) + MTP startup-deadlock gates (env-gated, default off)
  apply_glm_vllm_dp_profile_sync_fix.py     # VLLM_SKIP_DP_SYNC_ON_PROFILE (default 1)
  apply_glm_vllm_fwdctx_dp_sync_fix.py      # VLLM_SKIP_FWDCTX_DP_AR       (default 1)
  apply_glm_vllm_skip_profile_run_fix.py    # VLLM_SKIP_PROFILE_RUN        (default 0)
  apply_glm_vllm_skip_warmup_dummy_fix.py   # VLLM_SKIP_WARMUP_DUMMY       (default 0)
)

for p in "${VLLM_PATCHES[@]}"; do
  if [ -f "${HERE}/${p}" ]; then
    echo "── ${p}"
    python3 "${HERE}/${p}" "${VLLM_DIR}" || rc=1
  else
    echo "── ${p}  (missing in tree, skipping)"
  fi
done

# --- AITER-targeted patch (EP16-MTP: force CK 2-stage MoE; env-gated, default off) ---
if [ -n "${AITER_DIR}" ] && [ -f "${HERE}/apply_glm_aiter_fmoe_1stage_gfx950_fix.py" ]; then
  echo "── apply_glm_aiter_fmoe_1stage_gfx950_fix.py   # AITER_FORCE_CK_FMOE (default 0)"
  python3 "${HERE}/apply_glm_aiter_fmoe_1stage_gfx950_fix.py" "${AITER_DIR}" || rc=1
else
  echo "── AITER fmoe patch skipped (AITER not found or script missing)"
fi

if [ "$rc" -ne 0 ]; then
  echo "[apply-all] DONE with ERRORS — a patch hit a changed anchor; re-anchor before serving." >&2
else
  echo "[apply-all] all patches applied/verified OK."
fi
exit "$rc"
