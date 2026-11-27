#!/bin/bash
#SBATCH --job-name=qwen-hf-generate
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# Qwen2.5-0.5B inference (Hugging Face transformers, generate()) on every GPU of one node. Run from this folder;
# only the public base image and the transformers wheels are downloaded.
#
#   ./run.sh               build the image, run the workload on this node and print a verdict
#   ./run.sh pull          only prepare the image (docker/podman: build; apptainer: SIF + pydeps/)
#   sbatch -p <partition> run.sh              same, on one exclusive SLURM node
#
# Environment (optional): GPUS=8 PROCS=3 REQUESTS=1000 PROMPT=1280 NEW=8 BS=1 SEED=<random>
#                         OUT_DIR=./out/<date> TRANSFORMERS=5.20.2
#                         IMAGE=<image name> BASE=<base image>
#                         GEN_ARGS=<extra qwen_generate.py arguments, paths relative to this folder>
#                         RUNTIME=docker|podman|apptainer (default: the first one installed)
#                         SCRATCH_DIR=<podman storage, apptainer cache; default: temporary, node-local>
#                         SIF_DIR=./sif (apptainer images, kept between runs)
set -u
# sbatch runs a spooled copy of this file, so under SLURM the folder is the submit directory. Physical paths
# (pwd -P): Apptainer cannot bind to a path that runs through a symlink under $HOME, such as ~/proj.
if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname "${BASH_SOURCE[0]}"); fi
HERE=$(cd "$HERE" && pwd -P) && cd "$HERE" || exit 2
[ -f qwen_generate.py ] || { echo "run.sh must run from (or be submitted from) its own folder"; exit 2; }

# The image is built from ./Dockerfile on a public AMD base, pinned by digest (Apptainer and Podman do not accept
# tag@digest): rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0 (ROCm 7.2.0).
IMAGE=${IMAGE:-qwen-hf-generate:rocm7.2.0}
BASE=${BASE:-docker.io/rocm/pytorch@sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4}
TRANSFORMERS=${TRANSFORMERS:-5.20.2}
GPUS=${GPUS:-8} PROCS=${PROCS:-3} REQUESTS=${REQUESTS:-1000} PROMPT=${PROMPT:-1280} NEW=${NEW:-8} BS=${BS:-1}
SEED=${SEED:-$RANDOM} GEN_ARGS=${GEN_ARGS:-}
OUT_DIR=${OUT_DIR:-$HERE/out/$(date +%Y%m%d-%H%M%S)}

RUNTIME=${RUNTIME:-$(for r in docker podman apptainer; do command -v $r >/dev/null && { echo $r; break; }; done)}
case $RUNTIME in docker|podman|apptainer) ;; *) echo "no container runtime (docker, podman, apptainer)"; exit 2 ;; esac
SIF_DIR=${SIF_DIR:-$HERE/sif}
# Podman's storage and Apptainer's layer cache default to $HOME, which is too small for a ROCm image.
KEEP_SCRATCH=${SCRATCH_DIR:+1}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/qwen-hf-generate-$(id -u).XXXX")}
mkdir -p "$SCRATCH_DIR"
# Rootless podman without /etc/subuid ranges (HPC clusters): one uid is mapped, so image files cannot keep
# their owners, and host networking avoids the netns sysctls that need more uids.
PODMAN=(podman --root "$SCRATCH_DIR/podman" --runroot "$SCRATCH_DIR/podman-run"
        --storage-opt overlay.ignore_chown_errors=true)
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer-cache APPTAINER_TMPDIR=$SCRATCH_DIR

# The images have no passwd entry for the host uid, and getpass.getuser() fails without USER/LOGNAME. Running
# as the host user keeps the output writable on NFS, which squashes root.
NAME=qwen-hf-generate-$$
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
pydeps() { echo "$HERE/pydeps/$(tr '/:' '__' <<< "$1")-transformers$TRANSFORMERS"; }
# The bases are public; a stored but expired Docker Hub login still makes the pull fail.
PULL_HINT="if a stored registry login has expired, log out of docker.io (or log in again)"
image_ready() { # image_ready <image>: build it from ./Dockerfile (apptainer: convert its base, add pydeps/)
  local b s d
  b=$BASE
  case $RUNTIME in
    # An empty DOCKER_CONFIG pulls the public base anonymously: a stored but expired Docker Hub login makes the
    # pull fail even for public images. --network=host: some daemons have no default bridge network.
    docker) docker image inspect "$1" >/dev/null 2>&1 ||
              { echo "building $1 from $b"; mkdir -p "$SCRATCH_DIR/docker-anon" &&
                DOCKER_CONFIG=$SCRATCH_DIR/docker-anon docker build -q --network=host -t "$1" --build-arg BASE="$b" \
                  --build-arg TRANSFORMERS="$TRANSFORMERS" "$HERE" >/dev/null; } ||
              { echo "cannot build $1; $PULL_HINT"; return 1; }
            echo "IMAGE $(docker image inspect "$1" --format '{{.Id}}') base $b" ;;
    podman) "${PODMAN[@]}" image exists "$1" ||
              { echo "building $1 from $b"; "${PODMAN[@]}" build -q -t "$1" --build-arg BASE="$b" \
                  --build-arg TRANSFORMERS="$TRANSFORMERS" "$HERE" >/dev/null; } ||
              { echo "cannot build $1; $PULL_HINT"; return 1; }
            echo "IMAGE $("${PODMAN[@]}" image inspect "$1" --format '{{.Id}}') base $b" ;;
    # Apptainer runs the base itself and cannot build the Dockerfile: transformers goes to pydeps/<image>/ (on
    # PYTHONPATH at run time), without the packages the base already has, so torch and numpy stay the image's.
    apptainer) s=$(sif "$1") d=$(pydeps "$1")
            [ -s "$s" ] || { echo "pulling $b to $s"; mkdir -p "$SIF_DIR" && apptainer pull -q "$s" "docker://$b"; } ||
              { echo "cannot pull $b; $PULL_HINT"; return 1; }
            if [ ! -f "$d/.complete" ]; then
              echo "installing transformers==$TRANSFORMERS into $d"
              rm -rf "$d" && mkdir -p "$d" || return 1
              apptainer exec --cleanenv -B "$d:$d" "$s" bash -c '
                python3 -m pip install -q --no-cache-dir --target "$1" "transformers==$2" && python3 -I - "$1" <<"PY"
import importlib.metadata as md, pathlib, shutil, sys
target = pathlib.Path(sys.argv[1])
for info in sorted(target.glob("*.dist-info")):
    name = md.Distribution.at(info).metadata["Name"]
    try:
        md.distribution(name)            # -I: the image alone, without the target dir
    except md.PackageNotFoundError:
        continue
    for line in (info / "RECORD").read_text().splitlines():
        p = target / line.split(",")[0]
        if p.is_file() and target.resolve() in p.resolve().parents:
            p.unlink()
    shutil.rmtree(info, ignore_errors=True)
    print("pydeps: the image has", name, "- removed from", target)
for d in sorted((p for p in target.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
    if not any(d.iterdir()):
        d.rmdir()
PY' _ "$d" "$TRANSFORMERS" && touch "$d/.complete" || { echo "cannot install transformers into $d"; return 1; }
            fi
            echo "IMAGE $s $(stat -L -c '%s bytes, pulled %y' "$s") pydeps $d" ;;
  esac
}
drun() { # drun <name> <gpu> <image> <command...>; extra K=V environment in the XENV array
  local name=$1 gpu=$2 img=$3 kv flag=-e envs=() t; shift 3
  [ "$RUNTIME" = apptainer ] && flag=--env
  for kv in USER="$(id -un)" LOGNAME="$(id -un)" TORCHINDUCTOR_CACHE_DIR=/tmp/inductor HF_HUB_OFFLINE=1 \
            HIP_VISIBLE_DEVICES="$gpu" "${XENV[@]}"; do envs+=("$flag" "$kv"); done
  case $RUNTIME in
    docker) docker run --rm --name "$NAME-$name" \
              --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$img" "$@" ;;
    podman) "${PODMAN[@]}" run --rm --name "$NAME-$name" \
              --userns=keep-id --group-add keep-groups --network=host --security-opt label=disable \
              --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --shm-size=16g \
              -v "$HERE:$HERE:ro" -v "$OUT_DIR:$OUT_DIR" -w "$HERE" -e HOME=/tmp "${envs[@]}" "$img" "$@" ;;
    # Host /dev (with /dev/kfd) is mounted by default. Not --rocm: it binds the host's ROCm libraries into
    # the container instead of the image's own. Each container gets its own /tmp.
    apptainer) t=$(mktemp -d "$SCRATCH_DIR/tmp-$name.XXXX")
              apptainer exec --cleanenv --no-mount tmp,cwd --home "$t:/tmp" \
                -B "$HERE:$HERE:ro" -B "$OUT_DIR:$OUT_DIR" --pwd "$HERE" "${envs[@]}" \
                --env PYTHONPATH="$(pydeps "$img")" "$(sif "$img")" "$@"
              rm -rf "$t" ;;
  esac
}

case "${1:-run}" in
  run) ;;
  pull) image_ready "$IMAGE"; exit ;;
  *) sed -n '7,20p' "$HERE/run.sh"; exit 2 ;;
esac
mkdir -p "$OUT_DIR" && OUT_DIR=$(cd "$OUT_DIR" && pwd -P) || exit 2
echo "output: $OUT_DIR"
echo "=== image=$IMAGE processes/GPU=$PROCS GPUs=$GPUS requests=$REQUESTS prompt=$PROMPT new=$NEW bs=$BS" \
     "seed=$SEED host=$(hostname -s) runtime=$RUNTIME${GEN_ARGS:+ gen_args=$GEN_ARGS}"
image_ready "$IMAGE" || { echo "VERDICT INCOMPLETE (no image)"; exit 1; }
XENV=()
drun env 0 "$IMAGE" bash -c 'python3 -c "import torch, transformers; print(\"ENV torch\", torch.__version__, \"hip\",
  torch.version.hip, \"arch\", torch.cuda.get_device_properties(0).gcnArchName, \"transformers\",
  transformers.__version__)"; sed "s/^/ENV rocm /" /opt/rocm/.info/version 2>/dev/null' 2>&1 | grep '^ENV'
for g in $(seq 0 $((GPUS - 1))); do
  # shellcheck disable=SC2086
  drun "g$g" "$g" "$IMAGE" \
    python3 "$HERE/qwen_generate.py" $GEN_ARGS \
      --prompt-len "$PROMPT" --new-tokens "$NEW" --batch-size "$BS" --requests "$REQUESTS" \
      --trials "$PROCS" --seed $((SEED + g * 1000)) --log-every 100 \
    > "$OUT_DIR/g$g.log" 2>&1 &
done
wait
results=$(grep -h '^RESULT' "$OUT_DIR"/g*.log)
tot=$(grep -c '^RESULT' <<< "$results")
bad=$(grep -c 'nan_request=[0-9]' <<< "$results")
grep 'nan_request=[0-9]' <<< "$results"
if [ "$tot" -ne $((GPUS * PROCS)) ]; then
  echo "WARNING: only $tot of $((GPUS * PROCS)) processes printed a RESULT line. First error:"
  grep -l -E 'Traceback|Error' "$OUT_DIR"/g*.log | head -1 | xargs -r tail -20
fi
echo
echo "=== seed base $SEED, host $(hostname -s) ==="
echo "processes_with_nan=$bad/$tot"
if [ "$tot" -eq 0 ]; then echo "VERDICT INCOMPLETE (no results)"
elif [ "$bad" -gt 0 ]; then echo "VERDICT REPRODUCED"
else echo "VERDICT NOT_REPRODUCED (rerun, SEED is random)"; fi
