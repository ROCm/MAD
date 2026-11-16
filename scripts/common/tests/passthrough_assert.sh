#!/bin/bash
# Config passthrough: every value MAD derives (fabric, platform, recipe roles, the job's recipe file,
# probes) is a default -- a value passed through config (submit env, card / additional-context
# env_vars, mad-config, a site profile) always wins. Offline, on the launcher's own functions.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
_is() { [ "$1" = "$2" ] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s\n        got:  %s\n        want: %s\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }
cat > "$TMP/run.sh" <<'RUN'
. "$1/common/config/layers.sh"; . "$1/common/cluster.sh" >/dev/null 2>&1; config_begin
eval "$2" >/dev/null 2>&1
eval "echo \"$3\""
RUN
# run "VAR=given ..." "steps" "what to print"
run() { local pre="$1"; shift; env -i PATH="$PATH" HOME="$TMP" MAD_SCRIPTS_DIR="$DIR" CLUSTER_SYSFS_IB="$TMP/none" \
        $pre bash "$TMP/run.sh" "$DIR" "$@"; }

echo "=== fabric and platform ==="
for v in MAD_FABRIC_RAILS MAD_FABRIC_GID MAD_FABRIC_IFACE MAD_FABRIC_KV_NIC MAD_FABRIC_RDMA_TC MAD_FABRIC_RDMA_SL; do
    _is "$(run "$v=given" ":" "\${$v}")" "given" "$v from config wins over the fabric family's"
done
_is "$(run "CLUSTER_FABRIC_DERIVED=1" ":" '${CLUSTER_FABRIC_DERIVED}')" "1" "CLUSTER_FABRIC_DERIVED from config wins"
_is "$(run "GPUS_PER_NODE=2 MAD_PLATFORM=mi300x-oam-8" ":" '${GPUS_PER_NODE}')" "2" "GPUS_PER_NODE from config wins over the platform's"
_is "$(run "MAD_PLATFORM=mi300x-oam-8 MAD_PLATFORM_ACCELERATOR=gfx950" ":" '${MAD_PLATFORM_ACCELERATOR}')" "gfx950" \
    "MAD_PLATFORM_ACCELERATOR from config wins"
_is "$(run "MAD_GPU_ARCH=gfx942 MAD_GPU_COUNT=3" "_CLUSTER_ARCH_PROBED=0; cluster_detect_gpu_arch" '${MAD_GPU_COUNT}')" "3" \
    "MAD_GPU_COUNT from config is kept by the probe"
_is "$(run "MAD_GPU_MEMORY_BYTES=7" "cluster_gpu_memory_for_recipe /dev/null x" '${MAD_GPU_MEMORY_BYTES}')" "7" \
    "MAD_GPU_MEMORY_BYTES from config is kept"

echo ""
echo "=== recipe roles and the job's recipe ==="
Y="$DIR/vllm_dissag/models.yaml"
_is "$(run "MAD_GPU_ARCH=gfx942 MAD_ROLE_ENGINE_ARGS_PREFILL=--given" "cluster_check_recipe $Y Kimi-K3-MXFP4 vllm" '${MAD_ROLE_ENGINE_ARGS_PREFILL}')" \
    "--given" "MAD_ROLE_ENGINE_ARGS_<ROLE> from config wins over the resolved arguments"
_is "$(run "MAD_ROLE_ARTIFACT_DECODE=x MAD_ROLE_MODEL_PATH_DECODE=/given" "cluster_resolve_role_artifacts" '${MAD_ROLE_MODEL_PATH_DECODE}')" \
    "/given" "MAD_ROLE_MODEL_PATH_<ROLE> from config is not re-resolved"
mkdir -p "$TMP/logs"
_is "$(run "LOG_PATH=$TMP/logs SLURM_JOB_ID=9 MAD_RECIPE_FILE=/run_logs/mine.yaml" "config_recipe_file $Y Kimi-K3-MXFP4" '${MAD_RECIPE_FILE}')" \
    "/run_logs/mine.yaml" "MAD_RECIPE_FILE from config wins over the job's written recipe"
_is "$(run "LOG_PATH=$TMP/logs SLURM_JOB_ID=9 MAD_RECIPE_FILE=/run_logs/mine.yaml" "config_recipe_file $Y Kimi-K3-MXFP4" "\$(ls $TMP/logs/9 2>/dev/null)")" \
    "" "...and nothing is written"

echo ""
echo "======================================================"
echo "  passthrough_assert: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
