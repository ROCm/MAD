#!/bin/bash
#SBATCH --job-name=dit-fp8-train-nan
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# FP8 DiT training (torchao float8), one trainer per GPU on all GPUs of a node, and a verdict.
# Run from this folder; only the public rocm/primus:v26.3 image is downloaded.
#
#   ./run.sh                 train on every GPU, print a RESULT line per trial and a verdict
#   ./run.sh pull            only prepare the image (docker/podman: build; apptainer: SIF in SIF_DIR)
#   sbatch -p <partition> run.sh     same, on one exclusive SLURM node
#
# Environment (optional): GPUS=8 STEPS=30 SEED=<random> OUT_DIR=./out/<date>
#                         IMAGE=<image name> BASE=<base image>  TRAIN_ARGS=<extra dit_fp8_train.py arguments>
#                         RUNTIME=docker|podman|apptainer (default: the first one installed)
#                         SCRATCH_DIR=<podman storage, apptainer cache; default: temporary, node-local>
#                         SIF_DIR=./sif (apptainer images, kept between runs)
set -u
# sbatch runs a spooled copy of this file, so under SLURM the folder is the submit directory. Physical paths
# (pwd -P): Apptainer cannot bind to a path that runs through a symlink under $HOME.
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2
[ -f dit_fp8_train.py ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

# The image is built from ./Dockerfile on the public rocm/primus:v26.3, pinned by digest (Apptainer and Podman do
# not accept tag@digest): ROCm 7.2.1, PyTorch 2.10.0, torchao 0.15.0.
IMAGE=${IMAGE:-dit-fp8-train-nan:primus-v26.3}
BASE=${BASE:-docker.io/rocm/primus@sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482}
GPUS=${GPUS:-8} STEPS=${STEPS:-30} SEED=${SEED:-$RANDOM}
TRAIN_ARGS=${TRAIN_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)}

RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in docker|podman|apptainer) ;; *) echo "no container runtime (docker, podman, apptainer)"; exit 2 ;; esac
SIF_DIR=${SIF_DIR:-$HERE/sif}
KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/dit-fp8-train-nan-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
PODMAN=(podman --root "$SCRATCH_DIR/podman" --runroot "$SCRATCH_DIR/podman-run"
        --storage-opt overlay.ignore_chown_errors=true)
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR

NAME=dit-fp8-train-nan-$$
cleanup() {
  case $RUNTIME in
    docker) docker ps -q --filter "name=$NAME-" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 ;;
    podman) "${PODMAN[@]}" ps -q --filter "name=$NAME-" 2>/dev/null | xargs -r "${PODMAN[@]}" rm -f >/dev/null 2>&1 ;;
    apptainer) jobs -p | xargs -r kill 2>/dev/null; wait ;;
  esac
  [ -n "$KEEP_SCRATCH" ] && return
  [ "$RUNTIME" = podman ] && "${PODMAN[@]}" system reset -f >/dev/null 2>&1
  rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT; trap 'exit 130' TERM INT
sif() { echo "$SIF_DIR/$(tr '/:' '__' <<< "$1").sif"; }
PULL_HINT="if a stored registry login has expired, log out of docker.io (or log in again)"
image_ready() { # build $IMAGE from ./Dockerfile (apptainer: convert its base) if missing
  local s
  case $RUNTIME in
    # An empty DOCKER_CONFIG pulls the public base anonymously; --network=host: some daemons have no bridge.
    docker) docker image inspect "$IMAGE" >/dev/null 2>&1 ||
              { echo "building $IMAGE from $BASE (about 22 GB to pull)"; mkdir -p "$SCRATCH_DIR/docker-anon" &&
                DOCKER_CONFIG=$SCRATCH_DIR/docker-anon docker build -q --network=host -t "$IMAGE" \
                  --build-arg BASE="$BASE" "$HERE" >/dev/null; } ||
              { echo "cannot build $IMAGE; $PULL_HINT"; return 1; }
            echo "IMAGE $(docker image inspect "$IMAGE" --format '{{.Id}}') base $BASE" ;;
    podman) "${PODMAN[@]}" image exists "$IMAGE" ||
              { echo "building $IMAGE from $BASE"; "${PODMAN[@]}" build -q -t "$IMAGE" --build-arg BASE="$BASE" \
                  "$HERE" >/dev/null; } ||
              { echo "cannot build $IMAGE; $PULL_HINT"; return 1; }
            echo "IMAGE $("${PODMAN[@]}" image inspect "$IMAGE" --format '{{.Id}}') base $BASE" ;;
    # Apptainer runs the base itself: run.sh supplies what the Dockerfile adds (environment; script from $HERE).
    apptainer) s=$(sif "$IMAGE")
            [ -s "$s" ] || { echo "pulling $BASE to $s"; mkdir -p "$SIF_DIR" && apptainer pull -q "$s" "docker://$BASE"; } ||
              { echo "cannot pull $BASE; $PULL_HINT"; return 1; }
            echo "IMAGE $s $(stat -L -c '%s bytes, pulled %y' "$s")" ;;
  esac
}
drun() { # drun <name> <gpu> <command...>; extra K=V environment in the XENV array
  local name=$1 gpu=$2 kv flag=-e envs=() t; shift 2
  [ "$RUNTIME" = apptainer ] && flag=--env
  for kv in USER="$(id -un)" LOGNAME="$(id -un)" HOME=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
            TRITON_CACHE_DIR=/tmp/triton HIP_VISIBLE_DEVICES="$gpu" "${XENV[@]}"; do envs+=("$flag" "$kv"); done
  case $RUNTIME in
    docker) docker run --rm --name "$NAME-$name" \
              --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" --group-add video \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" "${envs[@]}" "$IMAGE" "$@" ;;
    podman) "${PODMAN[@]}" run --rm --name "$NAME-$name" \
              --userns=keep-id --group-add keep-groups --network=host --security-opt label=disable \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" "${envs[@]}" "$IMAGE" "$@" ;;
    # Not --rocm: it would bind the host's ROCm libraries over the image's.
    apptainer) t=$(mktemp -d "$SCRATCH_DIR/tmp-$name.XXXX")
              apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" \
                -B "$HERE:$HERE:ro" -B "$OUT_DIR:$OUT_DIR" --pwd "$HERE" "${envs[@]}" "$(sif "$IMAGE")" "$@"
              rm -rf "$t" ;;
  esac
}

case "${1:-}" in
  "") ;;
  pull) image_ready || exit 1; exit 0 ;;
  *) sed -n '7,18p' "$HERE/run.sh"; exit 2 ;;
esac
mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
echo "output: $OUT_DIR"
image_ready || exit 1
XENV=()
drun env 0 bash -c 'python3 -c "import torch, torchao; print(\"ENV torch\", torch.__version__, \"hip\",
  torch.version.hip, \"torchao\", torchao.__version__, torch.cuda.get_device_name(),
  torch.cuda.get_device_properties(0).gcnArchName)"; cat /opt/rocm/.info/version | sed "s/^/ENV rocm /"' 2>&1 |
  grep '^ENV'

echo "=== training: GPUs=$GPUS steps=$STEPS seed=$SEED host=$(hostname -s) runtime=$RUNTIME${TRAIN_ARGS:+ train_args=$TRAIN_ARGS}"
for g in $(seq 0 $((GPUS - 1))); do
  # The training environment: PyTorch TunableOp on, one results file per trainer.
  XENV=(PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME="$OUT_DIR/g${g}_tunableop_%d.csv")
  # shellcheck disable=SC2086
  drun "g$g" "$g" python3 "$HERE/dit_fp8_train.py" $TRAIN_ARGS --steps "$STEPS" --seed $((SEED + g * 1000)) \
    --log-every 5 > "$OUT_DIR/train_g$g.log" 2>&1 &
done
wait
results=$(grep -h '^RESULT' "$OUT_DIR"/train_g*.log)
tot=$(grep -c '^RESULT' <<< "$results")
bad=$(grep -c 'nan_step=[0-9]' <<< "$results")
[ -z "$results" ] && bad=0
echo "$results"
if [ "$tot" -ne "$GPUS" ]; then
  echo "WARNING: only $tot of $GPUS trials printed a RESULT line. First error:"
  grep -l -E 'Traceback|Error' "$OUT_DIR"/train_g*.log | head -1 | xargs -r tail -20
fi
echo
echo "=== summary: seed base $SEED, host $(hostname -s) ==="
echo "$bad/$tot trials with NaN"
if [ "$tot" -eq 0 ]; then echo "VERDICT INCOMPLETE (no results)"
elif [ "$bad" -gt 0 ]; then echo "VERDICT REPRODUCED"
else echo "VERDICT NOT_REPRODUCED"; fi
