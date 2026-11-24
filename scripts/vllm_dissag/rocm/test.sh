#!/bin/bash
# P/D disaggregation on Crusoe, using the .buildkite/amd-disagg scripts from
# lcskrishna/vllm (branch rocm-crusoe) with cluster_crs.sh from this folder.
# Submits one Slurm job per cell: every model in MODELS, in every
# topology in TOPOLOGIES (<x>P<y>D), with every mode in WIDE_EP_MODES
# (0 = TP8, 1 = wide EP). A model's flags come from models.yaml; its folder is
# found under MODEL_ROOT.
#
# Each run gets its own clone, so a later run cannot change the scripts of jobs
# that are still running. The clone's commit and the submitted job IDs are
# written to $RUN_DIR/jobs.txt.
#
# Usage: DOCKER=<image> bash test.sh
#        DOCKER=<image> MODELS="GLM-5.2-FP8" TOPOLOGIES=1P1D bash test.sh
#        DRY_RUN=1 DOCKER=<image> bash test.sh   # print the sbatch lines only
#        SERIAL=1 DOCKER=<image> bash test.sh    # each job waits for the previous one
#        AFTER=<job-id> ...                      # the first job waits for <job-id>
set -euo pipefail

DOCKER=${DOCKER:?usage: DOCKER=<image> bash test.sh}
MODELS=${MODELS:-DeepSeek-V4-Pro MiniMax-M3-MXFP8 GLM-5.2-FP8}
TOPOLOGIES=${TOPOLOGIES:-1P1D 2P2D}
WIDE_EP_MODES=${WIDE_EP_MODES:-0 1}
MODEL_ROOT=${MODEL_ROOT:-/shared_nfs/models}
DRY_RUN=${DRY_RUN:-0}
SERIAL=${SERIAL:-0}
prev=${AFTER:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

VLLM_REPO=${VLLM_REPO:-https://github.com/lcskrishna/vllm.git}
VLLM_BRANCH=${VLLM_BRANCH:-rocm-crusoe}
RUN_DIR=${RUN_DIR:-/shared_nfs/$USER/mad-rocm-runs/$(date +%Y%m%d-%H%M%S)}
git clone --quiet --depth 1 -b "$VLLM_BRANCH" "$VLLM_REPO" "$RUN_DIR/vllm"
SCRIPTS=$RUN_DIR/vllm/.buildkite/amd-disagg
cp "$HERE/cluster_crs.sh" "$SCRIPTS/cluster_crs.sh"
cd "$SCRIPTS"

export DISAGG_SCRIPTS_DIR=$PWD
export IMAGE=$DOCKER
export SHARED_MOUNT=/shared_nfs
export LOG_ROOT=${LOG_ROOT:-/shared_nfs/$USER/disagg_logs}
export RUN_AFTER_HEALTH=accuracy
export ROUTER_TYPE=vllm-router
export CLUSTER_ENV_IN=/vllm-workspace/.buildkite/amd-disagg/cluster_crs.sh
export SKIP_IONIC_MOUNTS=1
export WORKSPACE_DIR=/shared_nfs/$USER/qa-workspace
mkdir -p "$LOG_ROOT"

EXCLUDE=${EXCLUDE:-crsuse2-m2m-v3-093,crsuse2-m2m-v3-108}
MANIFEST=$RUN_DIR/jobs.txt
echo "# $VLLM_REPO $VLLM_BRANCH $(git -C "$RUN_DIR/vllm" rev-parse HEAD) image=$DOCKER" > "$MANIFEST"

for model in $MODELS; do
    if ! grep -q "^ *- *model: *$model *$" models.yaml; then
        echo "SKIP $model: not in models.yaml" >&2
        continue
    fi
    model_path=$(find "$MODEL_ROOT" -mindepth 1 -maxdepth 2 -type d -name "$model" -print -quit)
    if [ -z "$model_path" ] || [ ! -f "$model_path/config.json" ]; then
        echo "SKIP $model: no folder named $model with a config.json under $MODEL_ROOT" >&2
        continue
    fi
    for topo in $TOPOLOGIES; do
        [[ $topo =~ ^([0-9]+)P([0-9]+)D$ ]] || { echo "bad topology: $topo" >&2; exit 1; }
        xp=${BASH_REMATCH[1]} yd=${BASH_REMATCH[2]}
        for wide_ep in $WIDE_EP_MODES; do
            name=${model,,}-${topo,,}-wideep${wide_ep}
            cmd=(sbatch --parsable --job-name="$name"
                 --account=amd-aifw-di --partition=amd-aig-sw-shared --qos=amd-aifw-di-qos
                 --nodes=$((xp + yd)) --ntasks-per-node=1 --gres=gpu:8 --exclusive --time=04:00:00
                 ${EXCLUDE:+--exclude=$EXCLUDE} ${prev:+--dependency=afterany:$prev}
                 --output="$LOG_ROOT/spur-%x-%j.out" --error="$LOG_ROOT/spur-%x-%j.err"
                 run_xPyD_disagg.slurm)
            if [ "$DRY_RUN" = 1 ]; then
                echo "MODEL_NAME=$model MODEL_DIR=$(dirname "$model_path") WIDE_EP_MODE=$wide_ep xP=$xp yD=$yd ${cmd[*]}"
                continue
            fi
            job=$(MODEL_NAME=$model MODEL_DIR=$(dirname "$model_path") \
                  WIDE_EP_MODE=$wide_ep xP=$xp yD=$yd "${cmd[@]}")
            echo "$job $name" | tee -a "$MANIFEST"
            [ "$SERIAL" = 1 ] && prev=$job
        done
    done
done
echo "jobs: $MANIFEST  logs: $LOG_ROOT/<job-name>-<job-id>.log"
