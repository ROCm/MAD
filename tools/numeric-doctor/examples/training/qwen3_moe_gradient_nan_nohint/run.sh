#!/bin/bash
#SBATCH --job-name=qwen3-moe-train
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
# Qwen3 MoE training (torchtitan qwen3_moe_debug, FSDP 4 x TP 2 x EP 4 on 8 GPUs). Run from this folder.
# On N nodes each node keeps that layout and the nodes are HSDP replicas (dp_replicate = N).
#
#   ./run.sh [N]       on a login node: fetch what is missing, then submit itself with sbatch on N nodes
#                      (default 1); on a GPU node (or under sbatch): train on the job's nodes and print
#                      loss / grad_norm per step
#   ./run.sh build     only prepare: torchtitan, the image, the seed checkpoint
#   ./run.sh status    show the newest job log
#
# RUNTIME=apptainer (default where installed): compute nodes may have no internet, so a missing image is
# pip-installed on the login node into a sandbox (sif/.sandbox.tar) and packed into sif/qwen3-moe.sif by the job.
# RUNTIME=docker: the job builds the image from ./Dockerfile on each node that does not have it yet.
#
# Environment (optional): RUNTIME=apptainer|docker (default: the first one installed)
#                         ACCOUNT= PARTITION= (cluster defaults)
#                         TIME=01:00:00 GPUS_FLAG=<sbatch GPU request; default --gpus-per-node=$NGPU where
#                           Slurm has GPU gres, none where nodes come whole>   (self-submission)
#                         STEPS=100 NGPU=8 OUT_DIR=./out/<date>-<job> EXTRA_ARGS=<more torchtitan args>
#                         SCRATCH_DIR=<apptainer build cache and container /tmp; default node-local>
set -u
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2      # physical path: apptainer binds
[ -f qwen3_moe.def ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

if [ "${1:-}" = status ]; then
  f=$(ls -t "$HERE"/logs/*.out 2>/dev/null | head -1); [ -n "$f" ] || { echo "no job logs in logs/"; exit 1; }
  squeue -u "$(id -un)" -n qwen3-moe-train 2>/dev/null; echo "== $f"; tail -40 "$f"; exit 0
fi
LOGIN=; [ -z "${SLURM_JOB_ID:-}" ] && [ ! -e /dev/kfd ] && LOGIN=1
NNODES=1
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then NNODES=$1; shift; [ "$NNODES" -ge 1 ] || { echo "number of nodes must be >= 1"; exit 2; }; fi
if [ -z "$LOGIN" ] && [ -n "${SLURM_JOB_ID:-}" ]; then   # in a job: train on all of its nodes
  [ "$NNODES" = 1 ] || [ "$NNODES" = "${SLURM_JOB_NUM_NODES:-1}" ] ||
    { echo "asked for $NNODES nodes but job $SLURM_JOB_ID has ${SLURM_JOB_NUM_NODES:-1}"; exit 2; }
  NNODES=${SLURM_JOB_NUM_NODES:-1}
fi

TITAN_REPO=https://github.com/pytorch/torchtitan.git
TITAN_COMMIT=ecae62f1591c3a73ed78a25d7c68f249bd61caa6
TITAN=$HERE/src/torchtitan
SEED_CKPT=$HERE/model/seed_checkpoint
SIF=$HERE/sif/qwen3-moe.sif
DIMAGE=qwen3-moe:$(sha256sum < "$HERE/Dockerfile" | cut -c1-12)    # docker: tagged by its recipe
RUNTIME=${RUNTIME:-$(for r in apptainer docker; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in apptainer|docker) ;; *) echo "no container runtime (apptainer, docker)"; exit 2 ;; esac
STEPS=${STEPS:-100} NGPU=${NGPU:-8} EXTRA_ARGS=${EXTRA_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)${SLURM_JOB_ID:+-$SLURM_JOB_ID}}

KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/qwen3-moe-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR
NAME=qwen3-moe-$$
cleanup() {
  [ "$RUNTIME" = docker ] && docker ps -q --filter "name=$NAME-" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1
  jobs -p | xargs -r kill 2>/dev/null; wait; [ -n "$KEEP_SCRATCH" ] || rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT; trap 'exit 130' TERM INT

titan_ready() {      # torchtitan at the pinned commit, as a plain source tree
  [ -f "$TITAN/torchtitan/train.py" ] && return 0
  local c=$SCRATCH_DIR/torchtitan
  git clone -q "$TITAN_REPO" "$c" && git -C "$c" checkout -q "$TITAN_COMMIT" &&
    mkdir -p "$TITAN" && git -C "$c" archive "$TITAN_COMMIT" | tar -x -C "$TITAN" ||
    { echo "cannot fetch torchtitan $TITAN_COMMIT"; return 1; }
  echo "TORCHTITAN $TITAN_COMMIT"
}
sandbox_ready() {    # login node: pip-install the image into a sandbox, kept as one tar (Lustre: few files)
  local t=$HERE/sif/.sandbox.tar b=$SCRATCH_DIR/sandbox log=$HERE/sif/.sandbox.log
  [ -s "$SIF" ] || [ -f "$t.done" ] && return 0
  mkdir -p "$HERE/sif"; echo "building the image sandbox; log: $log"
  apptainer build -F --fakeroot --sandbox "$b" "$HERE/qwen3_moe.def" > "$log" 2>&1 &&
    tar -C "$b" -cf "$t" . 2>> "$log" && touch "$t.done" ||
    { tail -30 "$log"; rm -f "$t"; echo "cannot build the sandbox (log: $log)"; return 1; }
  chmod -R u+rwX "$b" 2>/dev/null; rm -rf "$b"
}
image_ready() {      # job: pack the sandbox into the SIF if missing (docker: build the image on this node)
  local t=$HERE/sif/.sandbox.tar b=$SCRATCH_DIR/sandbox
  if [ "$RUNTIME" = docker ]; then
    docker image inspect "$DIMAGE" >/dev/null 2>&1 && return 0
    echo "building $DIMAGE on $(hostname -s); log: $SCRATCH_DIR/build.log"
    # The bases are public: build anonymously (an empty DOCKER_CONFIG), because a stored but expired Docker Hub
    # login makes the pull fail even for public images.
    mkdir -p "$SCRATCH_DIR/docker-anon" &&
      DOCKER_CONFIG=$SCRATCH_DIR/docker-anon docker build --network=host -t "$DIMAGE" - < "$HERE/Dockerfile" > "$SCRATCH_DIR/build.log" 2>&1 ||
      { tail -30 "$SCRATCH_DIR/build.log"; echo "cannot build $DIMAGE"; return 1; }
    return 0
  fi
  if [ ! -s "$SIF" ]; then
    [ -f "$t.done" ] || { echo "no $SIF and no sandbox: run ./run.sh on a login node first"; return 1; }
    mkdir -p "$b" && tar -C "$b" -xf "$t" && apptainer build -F "$SIF.part" "$b" > "$SCRATCH_DIR/pack.log" 2>&1 &&
      mv "$SIF.part" "$SIF" || { tail -30 "$SCRATCH_DIR/pack.log"; rm -f "$SIF.part"; echo "cannot pack $SIF"; return 1; }
    chmod -R u+rwX "$b" 2>/dev/null; rm -rf "$b" "$t" "$t.done" "$HERE/sif/.sandbox.log"
  fi
}
crun() {             # crun <command...>: in the image, torchtitan as the working directory
  local t rc
  if [ "$RUNTIME" = docker ]; then  # host network and IPC: torchrun rendezvous across nodes, RCCL
    docker run --rm --name "$NAME-$RANDOM" --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" \
      --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --network=host --ipc=host \
      -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$TITAN" -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" \
      -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 "$DIMAGE" "$@"
    return
  fi
  t=$(mktemp -d "$SCRATCH_DIR/home.XXXX")
  apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" -B "$HERE:$HERE:ro" -B "$OUT_DIR:$OUT_DIR" \
    --pwd "$TITAN" --env "USER=$(id -un)" --env "LOGNAME=$(id -un)" --env HF_HOME=/tmp/hf \
    --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 "$SIF" "$@"; rc=$?
  rm -rf "$t"; return $rc
}
seed_ready() {       # initial weights: torchtitan's seed checkpoint (qwen3_moe_debug, seed 42) on 1 GPU
  [ -f "$SEED_CKPT/step-0/.metadata" ] && return 0
  # Not under SCRATCH_DIR: that is node-local /tmp, which docker's daemon may not see (a private /tmp).
  local d=$HERE/model/.seed-work; rm -rf "$d"; mkdir -p "$d"
  echo "creating the seed checkpoint"
  ( OUT_DIR=$d; crun env MODULE=qwen3 CONFIG=qwen3_moe_debug NGPU=1 LOG_RANK=0 ./run_train.sh \
      --dump_folder="$d" --checkpoint.create_seed_checkpoint --checkpoint.enable \
      --debug.deterministic --debug.seed=42 ) > "$d/seed.log" 2>&1
  [ -f "$d/checkpoint/step-0/.metadata" ] || { tail -30 "$d/seed.log"; echo "seed checkpoint failed"; return 1; }
  mkdir -p "$SEED_CKPT" && cp -r "$d/checkpoint/step-0" "$SEED_CKPT/" && rm -rf "$d"
}

if [ -n "$LOGIN" ]; then  # login node: the network part, then submit (some sites need an explicit GPU request)
  titan_ready || exit 1
  [ "$RUNTIME" = apptainer ] && { sandbox_ready || exit 1; }
  command -v sbatch >/dev/null || { echo "no GPUs here and no sbatch"; exit 2; }
  mkdir -p "$HERE/logs"
  [ "${1:-}" = build ] && NNODES=1
  GPUS_FLAG=${GPUS_FLAG-$(sinfo -h -o %G 2>/dev/null | grep -q gpu && echo "--gpus-per-node=$NGPU")}
  # shellcheck disable=SC2086
  job=$(sbatch --parsable ${ACCOUNT:+-A "$ACCOUNT"} ${PARTITION:+-p "$PARTITION"} $GPUS_FLAG \
        -t "${TIME:-01:00:00}" -N "$NNODES" \
        --chdir="$HERE" -o "$HERE/logs/%x_%j.out" -e "$HERE/logs/%x_%j.out" "$HERE/run.sh" "$@") || exit 1
  echo "submitted job $job on $NNODES node(s); log: $HERE/logs/qwen3-moe-train_$job.out  (./run.sh status)"
  exit 0
fi

train_args() {       # the torchtitan flags; dp_replicate = number of nodes when there is more than one
  TRAIN_ARGS=(
    --dump_folder="$OUT_DIR/run"
    --parallelism.tensor_parallel_degree 2
    --parallelism.expert_parallel_degree 4
    --debug.deterministic
    --debug.seed=42
    --metrics.enable_tensorboard
    --metrics.log_freq=1
    --training.steps="$STEPS"
    --checkpoint.enable
    --checkpoint.export_dtype=bfloat16
    --checkpoint.load_only)
  [ "$NNODES" -gt 1 ] && TRAIN_ARGS+=(--parallelism.data_parallel_replicate_degree "$NNODES")
  # shellcheck disable=SC2206
  TRAIN_ARGS+=($EXTRA_ARGS)
}
if [ "${1:-}" = _node ]; then  # multi-node: one per node under srun; run_train.sh's torchrun, rendezvous at MASTER_ADDR
  image_ready || exit 1   # docker: an image per node
  train_args; r=${SLURM_NODEID:?}
  if [ "$r" = 0 ]; then out=$OUT_DIR/train.log; else out=$OUT_DIR/train.node$r.log; fi
  crun env MODULE=qwen3 CONFIG=qwen3_moe_debug PYTORCH_ALLOC_CONF=expandable_segments:True \
    TORCHFT_LIGHTHOUSE=http://localhost:29510 LOG_RANK=0 \
    torchrun --nnodes="$NNODES" --nproc_per_node="$NGPU" --node_rank="$r" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train --module qwen3 --config qwen3_moe_debug "${TRAIN_ARGS[@]}" > "$out" 2>&1
  exit $?
fi

titan_ready || exit 1
image_ready || exit 1
seed_ready || exit 1
[ "${1:-}" = build ] && exit 0

mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
d=$OUT_DIR/run log=$OUT_DIR/train.log
mkdir -p "$d/checkpoint" && ln -sfn "$SEED_CKPT/step-0" "$d/checkpoint/step-0"
if [ "$RUNTIME" = docker ]; then echo "runtime: docker $DIMAGE"; else echo "runtime: apptainer $SIF"; fi
crun python3 -c 'import torch; print("ENV torch", torch.__version__, "hip", torch.version.hip, "arch",
  torch.cuda.get_device_properties(0).gcnArchName, "gpus", torch.cuda.device_count())' 2>&1 | grep '^ENV'
echo "training: NNODES=$NNODES NGPU=$NGPU/node steps=$STEPS host=$(hostname -s) log=$log"
train_args
if [ "$NNODES" = 1 ]; then
  crun env MODULE=qwen3 CONFIG=qwen3_moe_debug NGPU="$NGPU" LOG_RANK=0 ./run_train.sh "${TRAIN_ARGS[@]}" > "$log" 2>&1
else
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) MASTER_PORT=${MASTER_PORT:-29500}
  echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')(other nodes log to train.node<N>.log)"
  export OUT_DIR STEPS NGPU NNODES EXTRA_ARGS MASTER_ADDR MASTER_PORT
  srun -N "$NNODES" --ntasks-per-node=1 ${SLURM_GPUS_PER_NODE:+--gpus-per-node="$NGPU"} -c "${SLURM_CPUS_ON_NODE:-1}" \
    --kill-on-bad-exit=1 "$HERE/run.sh" _node
fi
echo "exit code $?"
# "step:  1  loss:  8.03949  grad_norm:  2.6938 ..." (ANSI colours stripped) -> per-step table
sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -o 'step: *[0-9]* *loss: *[^ ]* *grad_norm: *[^ ]*' |
  awk '{ printf "  step %3d  loss %-10s grad_norm %s\n", $2, $4, $6 }' | tee "$OUT_DIR/steps.txt"
[ -s "$OUT_DIR/steps.txt" ] || { echo "no training steps logged; end of $log:"; tail -25 "$log"; }
