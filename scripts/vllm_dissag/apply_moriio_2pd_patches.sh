#!/bin/bash
# apply_moriio_2pd_patches.sh — Apply vLLM PR #39276 at container startup
# =============================================================================
# Downloads and applies the patch from vllm-project/vllm PR #39276 which adds:
#   1. engine_id collision fix (core.py, utils.py)
#   2. MoRIIOConnector multi-node DP fixes (moriio_connector.py, moriio_common.py)
#   3. MoRIIO robustness fixes (moriio_engine.py)
#
# Idempotent: already-applied patches are skipped via --forward flag.
# Once PR #39276 is merged upstream, this script becomes a no-op.
# =============================================================================

set -euo pipefail

PR_NUM=39276
PATCH_URL="https://github.com/vllm-project/vllm/pull/${PR_NUM}.patch"
PATCH_FILE="/tmp/vllm_pr_${PR_NUM}.patch"

# Locate the vLLM installation directory
VLLM_INSTALL_DIR=""
for _candidate in \
    /usr/local/lib/python3.12/dist-packages/vllm \
    /usr/local/lib/python3.*/dist-packages/vllm \
    $(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null); do
    if [ -d "$_candidate" ]; then
        VLLM_INSTALL_DIR="$_candidate"
        break
    fi
done

if [ -z "${VLLM_INSTALL_DIR}" ]; then
    echo "[PR#${PR_NUM}] ERROR: Cannot find vLLM installation directory"
    exit 1
fi

# The egg-info / dist-info root is one level up from the vllm package
VLLM_ROOT="$(dirname "${VLLM_INSTALL_DIR}")"
echo "[PR#${PR_NUM}] vLLM root: ${VLLM_ROOT}"

# Download the patch
echo "[PR#${PR_NUM}] Downloading patch from ${PATCH_URL}..."
if ! curl -sL "${PATCH_URL}" -o "${PATCH_FILE}" 2>/dev/null; then
    echo "[PR#${PR_NUM}] WARNING: Failed to download patch — check network connectivity"
    echo "[PR#${PR_NUM}] Trying to continue without patching..."
    exit 0
fi

# Verify we got a real patch file (not an HTML error page)
if ! head -1 "${PATCH_FILE}" | grep -q "^From "; then
    echo "[PR#${PR_NUM}] WARNING: Downloaded file is not a valid patch"
    echo "[PR#${PR_NUM}] First line: $(head -1 "${PATCH_FILE}")"
    echo "[PR#${PR_NUM}] Skipping patch application"
    rm -f "${PATCH_FILE}"
    exit 0
fi

PATCH_LINES=$(wc -l < "${PATCH_FILE}")
echo "[PR#${PR_NUM}] Downloaded patch: ${PATCH_LINES} lines"

# Apply the patch
# --forward: skip already-applied hunks (idempotent)
# --reject-file=-: don't create .rej files
# -p1 strips the first path component (a/vllm/... -> vllm/...)
echo "[PR#${PR_NUM}] Applying patch to ${VLLM_ROOT}..."
cd "${VLLM_ROOT}"

if patch -p1 --forward --reject-file=- < "${PATCH_FILE}" 2>&1; then
    echo "[PR#${PR_NUM}] Patch applied successfully"
elif [ $? -eq 1 ]; then
    echo "[PR#${PR_NUM}] Patch already applied or partially applied (some hunks skipped)"
else
    echo "[PR#${PR_NUM}] WARNING: Patch application had errors — some fixes may not be active"
fi

# Verify key files were patched by checking for known fix markers
echo "[PR#${PR_NUM}] Verifying patches..."
_ok=0
_total=0

_check_patch() {
    local file="$1"
    local marker="$2"
    local desc="$3"
    _total=$((_total + 1))
    if [ -f "${VLLM_INSTALL_DIR}/${file}" ] && grep -q "${marker}" "${VLLM_INSTALL_DIR}/${file}" 2>/dev/null; then
        echo "  ✓ ${desc}"
        _ok=$((_ok + 1))
    else
        echo "  ✗ ${desc} — marker '${marker}' not found in ${file}"
    fi
}

_check_patch "v1/engine/core.py" "dp_rank" "engine_id collision fix"
_check_patch "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py" "data_parallel_size_local" "multi-node DP sizing"
_check_patch "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py" "_req_kv_params" "kv_transfer_params caching"
_check_patch "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py" "_is_kv_master" "child node guard"
_check_patch "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py" "VLLM_MORIIO_TRANSFER_TIMEOUT_S" "transfer timeout"

echo "[PR#${PR_NUM}] Verification: ${_ok}/${_total} checks passed"

rm -f "${PATCH_FILE}"
echo "[PR#${PR_NUM}] Done"
