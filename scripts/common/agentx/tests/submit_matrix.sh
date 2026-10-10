#!/bin/bash
# AgentX campaign driver: parameterized sbatch fan-out over a matrix of cells.
# Intended to run FROM THE OCI LOGIN NODE (submit-only). Each cell submits one
# run_xPyD_models.slurm job (sglang_disagg or vllm_dissag entrypoint) with the
# AgentX env for that (backend x connector x mode x workload) combination and
# records JOBID -> cell in a job-map file for later harvesting.
#
# DRY_RUN=1 prints the sbatch commands WITHOUT submitting (validate on banff).
#
# Common env (shared by every cell):
#   MODEL_NAME       model to serve (default DeepSeek-V3)
#   PARTITION        sbatch -p partition (default amd-rccl)
#   DOCKER_IMAGE_NAME  vLLM image (used by backend=vllm cells)
#   SGLANG_IMAGE     sglang image (used by backend=sglang cells)
#   TIME             sbatch --time minutes (default 90)
#   JOB_MAP          job-map output file (default ./agentx_jobmap.<epoch>.tsv)
#
# Matrix: CELLS is a newline/semicolon list of cells, each a ':'-delimited tuple:
#   backend:connector:mode:workload:max_model_len:nodes
#     backend      sglang | vllm
#     connector    rixl | moriio
#     mode         0 (TP) | 1 (wideEP); maps to DP_MODE (sglang) / WIDE_EP (vllm)
#     workload     AGENTIC_WORKLOAD name (e.g. conformance_256k)
#     max_model_len MAX_MODEL_LEN (0 = auto-detect served window)
#     nodes        total nodes -> sbatch -N/-n
#
# Usage:
#   DRY_RUN=1 bash scripts/common/agentx/tests/submit_matrix.sh   # preview only
#   bash scripts/common/agentx/tests/submit_matrix.sh             # submit (login node)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-DeepSeek-V3}"
PARTITION="${PARTITION:-amd-rccl}"
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-<vllm-image>}"
SGLANG_IMAGE="${SGLANG_IMAGE:-<sglang-image>}"
TIME="${TIME:-90}"
JOB_MAP="${JOB_MAP:-$PWD/agentx_jobmap.$(date +%s).tsv}"

_is_dry=0
[ "${DRY_RUN:-0}" = "1" ] && _is_dry=1

# Default matrix (override by exporting CELLS). One cell per line:
#   backend:connector:mode:workload:max_model_len:nodes
CELLS="${CELLS:-$(cat <<'EOF'
sglang:moriio:1:conformance_256k:262144:3
sglang:moriio:1:conformance_512k:524288:3
vllm:rixl:0:conformance_256k:262144:2
vllm:moriio:1:conformance_256k:262144:2
EOF
)}"

echo "=== agentx submit matrix (DRY_RUN=${_is_dry}) ==="
echo "  model=${MODEL_NAME}  partition=${PARTITION}  time=${TIME}"
echo "  vllm_image=${DOCKER_IMAGE_NAME}  sglang_image=${SGLANG_IMAGE}"
echo "  job_map=${JOB_MAP}"
echo ""

[ "$_is_dry" = "1" ] || : > "$JOB_MAP"

_submit_cell() {
    local backend="$1" connector="$2" mode="$3" workload="$4" mml="$5" nodes="$6"

    local slurm_dir image jobname
    local -a envs
    envs=(BENCHMARK_SCRIPT=agentic "MODEL_NAME=${MODEL_NAME}" "AGENTIC_WORKLOAD=${workload}" "MAX_MODEL_LEN=${mml}")

    case "$backend" in
        sglang)
            slurm_dir="$REPO_ROOT/scripts/sglang_disagg"
            image="$SGLANG_IMAGE"
            envs+=("DOCKER_IMAGE_NAME=${image}" "DP_MODE=${mode}")
            [ "$connector" = "moriio" ] && [ "$mode" = "1" ] && envs+=("RUN_MORI=1")
            ;;
        vllm)
            slurm_dir="$REPO_ROOT/scripts/vllm_dissag"
            image="$DOCKER_IMAGE_NAME"
            envs+=("DOCKER_IMAGE_NAME=${image}" "CONNECTOR=${connector}" "WIDE_EP=${mode}")
            [ "$connector" = "moriio" ] && [ "$mode" = "1" ] && envs+=("RUN_MORI=1")
            ;;
        *)
            echo "  SKIP  unknown backend '$backend' in cell" >&2
            return 0
            ;;
    esac

    jobname="agx_${backend}_${connector}_ep${mode}_${workload}"

    local -a cmd
    cmd=(env "${envs[@]}" sbatch -N "$nodes" -n "$nodes" -p "$PARTITION" \
         --time="$TIME" -J "$jobname" "$slurm_dir/run_xPyD_models.slurm")

    echo "# cell: ${backend}:${connector}:mode${mode}:${workload}:mml${mml}:N${nodes}"
    printf '%q ' "${cmd[@]}"; echo

    if [ "$_is_dry" = "1" ]; then
        return 0
    fi
    # --parsable makes sbatch print just the JOBID; record JOBID -> cell.
    local jobid
    jobid="$(env "${envs[@]}" sbatch --parsable -N "$nodes" -n "$nodes" -p "$PARTITION" \
             --time="$TIME" -J "$jobname" "$slurm_dir/run_xPyD_models.slurm")"
    printf '%s\t%s:%s:mode%s:%s:mml%s:N%s\n' \
        "$jobid" "$backend" "$connector" "$mode" "$workload" "$mml" "$nodes" >> "$JOB_MAP"
    echo "  submitted JOBID=${jobid} -> ${JOB_MAP}"
    echo ""
}

while IFS= read -r line; do
    line="${line%%#*}"                       # strip trailing comments
    line="$(echo "$line" | tr ';' '\n')"     # allow ';'-separated cells too
    while IFS= read -r cell; do
        cell="$(echo "$cell" | xargs)"       # trim whitespace
        [ -n "$cell" ] || continue
        IFS=':' read -r backend connector mode workload mml nodes <<< "$cell"
        if [ -z "${nodes:-}" ]; then
            echo "  SKIP  malformed cell (need 6 ':'-fields): '$cell'" >&2
            continue
        fi
        _submit_cell "$backend" "$connector" "$mode" "$workload" "$mml" "$nodes"
    done <<< "$line"
done <<< "$CELLS"

if [ "$_is_dry" = "1" ]; then
    echo "=== DRY_RUN: nothing submitted ==="
else
    echo "=== submitted. job map: ${JOB_MAP} ==="
fi
