#!/bin/bash
#SBATCH --job-name=ep-benchmarking
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="ep_bench_slurm_job.out"
#SBATCH --error="ep_bench_slurm_job.err"

# This launcher's directory, mounted into the container. Under sbatch the scheduler
# runs a copy of this script from its spool directory, so its own path is not the
# repo: then the submit directory names it (`cd scripts/large-ep-benchmark && sbatch
# ...`, or sbatch from the MAD root). madengine runs it in place.
_resolve_launcher_dir() {
    local d
    for d in "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" \
             "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/scripts/large-ep-benchmark}" \
             "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n "$d" && -f "$d/run_ep_bench.sh" ]] && { (cd "$d" && pwd); return 0; }
    done
    echo "Error: cannot find scripts/large-ep-benchmark (run_ep_bench.sh) next to this script," \
         "in SLURM_SUBMIT_DIR or in $PWD" >&2
    return 1
}
LAUNCHER_DIR="$(_resolve_launcher_dir)" || exit 1

# Inside MAD, the site and the fabric come from common/cluster.sh, as for the other
# multinode launchers. The fabric values the caller passed, recorded before cluster.sh
# fills in its fallbacks, always reach the container; on ainic / thor2 so does what
# cluster.sh derived, since this benchmark's defaults (mlx5_0) name one CX7 cluster's
# devices. On cx7 / unknown the container keeps the defaults it always ran with.
MAD_SCRIPTS_DIR="$(cd "${LAUNCHER_DIR}/.." && pwd)"
_FABRIC_VARS="NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME MORI_RDMA_DEVICES MORI_IB_GID_INDEX MORI_SOCKET_IFNAME RCCL_AINIC_ROCE"
_FABRIC_PASSED=""
for _v in $_FABRIC_VARS; do [ -n "${!_v:-}" ] && _FABRIC_PASSED="$_FABRIC_PASSED $_v"; done
if [ -f "${MAD_SCRIPTS_DIR}/common/cluster.sh" ]; then
    . "${MAD_SCRIPTS_DIR}/common/cluster.sh"
    . "${MAD_SCRIPTS_DIR}/common/launcher/launcher_lib.sh"
fi

USER_NAME=$(whoami)
LOG_PATH=${LOG_PATH:-"${SHARED_MOUNT:-/shared_inference}/${USER_NAME}/model_blog_logs"}
DOCKER_IMAGE=${DOCKER_IMAGE:-ep-benchmarking:latest}
if [ -z "${IBDEVICES:-}" ]; then
    case "${CLUSTER_ARCHETYPE:-}" in
        ainic|thor2) IBDEVICES="${KV_IB_DEVICE:-mlx5_0}" ;;
        *)           IBDEVICES=mlx5_0 ;;
    esac
fi
SKIP_DEEPEP=${SKIP_DEEPEP:-0}

_FABRIC_FWD="$_FABRIC_PASSED"
case "${CLUSTER_ARCHETYPE:-}" in ainic|thor2) _FABRIC_FWD="$_FABRIC_VARS" ;; esac
# MoRI reads its own GID index; one fabric has one GID.
if [ -z "${MORI_IB_GID_INDEX:-}" ] && [ -n "${NCCL_IB_GID_INDEX:-}" ]; then
    case " $_FABRIC_FWD " in *" NCCL_IB_GID_INDEX "*) export MORI_IB_GID_INDEX="$NCCL_IB_GID_INDEX" ;; esac
fi
EP_FABRIC_ENV_ARGS=""
for _v in $_FABRIC_FWD; do [ -n "${!_v:-}" ] && EP_FABRIC_ENV_ARGS="$EP_FABRIC_ENV_ARGS -e $_v=${!_v}"; done
echo "[fabric] ${CLUSTER_ARCHETYPE:-unknown}; IBDEVICES ${IBDEVICES}; forwarded to the container:${EP_FABRIC_ENV_ARGS:- none (container defaults)}"

# The models share, mounted at /data where the site has it. Docker creates a missing
# bind-mount source as an empty directory, so it is not mounted where it is absent.
EP_DATA_MOUNT=""
[ -d "${SHARED_MOUNT:-/shared_inference}/models" ] && EP_DATA_MOUNT="-v ${SHARED_MOUNT:-/shared_inference}/models:/data"

# ------------------------
# SLURM Environment Variables
# ------------------------
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "ulimit: $(ulimit -a)"
echo ""

# Node information. Nodes are multi-homed, and the first `hostname -I` address can be
# one a peer cannot reach (a /31 RDMA rail): inside MAD the address on
# FABRIC_SUBNET_PREFIX (common/launcher/launcher_lib.sh) is preferred.
MASTER_PORT=2373  # Choose an open port
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
if declare -F launcher_node_ips >/dev/null; then
    SELECTED_NODES="$NODELIST"
    launcher_node_ips "${FABRIC_SUBNET_PREFIX:-}"
else
    MASTER_NODE=$(printf '%s\n' $NODELIST | head -n 1)
    IPS=()
    for NODE in $NODELIST; do
        IP=$(srun --nodes=1 -t 00:20:00 -w $NODE bash -c "hostname -I")
        IP=$(echo "$IP" | awk 'NR==1 {print $1}')
        IPS+=("$IP")
    done
    MASTER_ADDR="${IPS[0]}"
fi

echo "${IPS[*]}" | sed 's/ /,/g'

export MAD_REPO="$LAUNCHER_DIR"
export MAD_REPO_PATH="/root/mad-repo"

echo "----- Inside -----"
echo "${MAD_REPO}"
echo "${MAD_REPO_PATH}"
echo "${MAD_REPO_EP_SCRIPTS}"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

NNODES=$SLURM_NNODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "MASTER_ADDR is ${MASTER_ADDR}"
echo "MASTER_PORT is ${MASTER_PORT}"
echo "NNODES is ${NNODES}"

# The container writes the logs as root. On a root-squashed share that is an
# unprivileged user, so the job's log directory is made writable to it.
mkdir -p "$LOG_PATH/${SLURM_JOB_ID}" && chmod 0777 "$LOG_PATH/${SLURM_JOB_ID}" 2>/dev/null || true
echo "Log directory: $LOG_PATH/${SLURM_JOB_ID}"

export LOG_PATH=$LOG_PATH
export DOCKER_IMAGE=$DOCKER_IMAGE
export IBDEVICES=$IBDEVICES
export SKIP_DEEPEP=$SKIP_DEEPEP
export SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
export NNODES=$NNODES
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export EP_FABRIC_ENV_ARGS EP_DATA_MOUNT

srun bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
docker stop ep-bench;
sleep 10;
docker rm ep-bench;
sleep 10;
sleep 30;
docker pull $DOCKER_IMAGE;    
docker run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --privileged=true \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    -v $HOME:$HOME \
    $EP_DATA_MOUNT \
    -v $HOME/.ssh:/root/.ssh \
    -v /dev/infiniband:/dev/infiniband \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \
    -v /sys/class/net:/sys/class/net:ro \
    -v /sys/bus/pci:/sys/bus/pci:ro \
    --shm-size 64G \
    -v ${LOG_PATH}/${SLURM_JOB_ID}:/run_logs \
    -v $MAD_REPO:$MAD_REPO_PATH \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e NODE_RANK=$SLURM_PROCID \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e NNODES=$NNODES \
    -e IPADDRS=$IPADDRS \
    -e IBDEVICES=$IBDEVICES \
    -e SKIP_DEEPEP=$SKIP_DEEPEP \
    -e MAD_REPO_PATH=$MAD_REPO_PATH \
    -e MAD_REPO_EP_SCRIPTS=$MAD_REPO_EP_SCRIPTS \
    $EP_FABRIC_ENV_ARGS \
    --entrypoint /bin/bash \
    --name ep-bench \
    $DOCKER_IMAGE -c "
        echo Inside docker container ....
        echo ${MAD_REPO_PATH}
        $MAD_REPO_PATH/run_ep_bench.sh 2>&1 | tee /run_logs/ep_bench_results.log
"
'
EP_RC=$?

srun bash -c 'docker stop ep-bench 2>/dev/null || true; docker rm -f ep-bench 2>/dev/null || true'

# Inside MAD: publish the perf CSV run_ep_bench.sh wrote (parse_ep_to_csv.py) next to
# this launcher, the cards' multiple_results, and fail the job when a node failed or
# no benchmark produced a result (cluster.sh, cluster_job_verdict).
if declare -F cluster_job_verdict >/dev/null; then
    cluster_job_verdict "$EP_RC" "$LOG_PATH/${SLURM_JOB_ID}/perf.csv" "$(pwd)/perf_large_ep_bench.csv" 1
    exit $?
fi
exit "$EP_RC"
