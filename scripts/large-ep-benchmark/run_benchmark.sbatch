#!/bin/bash
#SBATCH --job-name=ep-benchmarking
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

USER_NAME=$(whoami)
PWD=$(pwd)
LOG_PATH=${LOG_PATH:-"${PWD}/slurm_output"}
DOCKER_IMAGE=${DOCKER_IMAGE_NAME:-${DOCKER_IMAGE:-ep-benchmarking:latest}}
SKIP_DEEPEP=${SKIP_DEEPEP:-1}

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

# Node information
USER_NAME=$(whoami)
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 -t 00:20:00 -w $MASTER_NODE bash -c "hostname -I")
MASTER_ADDR=$(echo "$MASTER_ADDR" | awk 'NR==1 {print $1}')
MASTER_PORT=2373  # Choose an open port

NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
IPS=()

for NODE in $NODELIST; do
    IP=$(srun --nodes=1 -t 00:20:00 -w $NODE bash -c "hostname -I")
    IP=$(echo "$IP" | awk 'NR==1 {print $1}')
    IPS+=("$IP")
done

echo "${IPS[*]}" | sed 's/ /,/g'

export MAD_REPO="$(pwd)"
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

mkdir -p "$LOG_PATH/${SLURM_JOB_ID}"
echo "Log directory: $LOG_PATH/${SLURM_JOB_ID}"

export LOG_PATH=$LOG_PATH
export DOCKER_IMAGE=$DOCKER_IMAGE
# Cluster network settings, forwarded into the containers only when set
export IBDEVICES GLOO_SOCKET_IFNAME MORI_RDMA_DEVICES MORI_DEVICE_NIC
export SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
export NNODES=$NNODES
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"

srun bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
docker stop ep-bench 2>/dev/null || true;
docker rm -f ep-bench 2>/dev/null || true;
sleep 2;
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
    -e IBDEVICES \
    -e GLOO_SOCKET_IFNAME \
    -e MORI_RDMA_DEVICES \
    -e MORI_DEVICE_NIC \
    -e SKIP_DEEPEP=$SKIP_DEEPEP \
    -e MAD_REPO_PATH=$MAD_REPO_PATH \
    -e MAD_REPO_EP_SCRIPTS=$MAD_REPO_EP_SCRIPTS \
    -e DOCKER_IMAGE=$DOCKER_IMAGE \
    --entrypoint /bin/bash \
    --name ep-bench \
    $DOCKER_IMAGE -c "
        set -o pipefail
        echo Inside docker container ....
        echo ${MAD_REPO_PATH}
        $MAD_REPO_PATH/run_ep_bench.sh 2>&1 | tee /run_logs/ep_bench_results.log
"
'
BENCH_RC=$?

srun bash -c 'docker stop ep-bench 2>/dev/null || true; docker rm -f ep-bench 2>/dev/null || true'

# madengine collects <submit dir>/slurm_output/perf_csv/*<job_id>*.csv
PERF_CSV="${LOG_PATH}/${SLURM_JOB_ID}/perf.csv"
if [ -f "$PERF_CSV" ]; then
    PERF_DIR="${SLURM_SUBMIT_DIR:-$PWD}/slurm_output/perf_csv"
    mkdir -p "$PERF_DIR" && cp "$PERF_CSV" "$PERF_DIR/results_${SLURM_JOB_ID}.csv" \
        && echo "perf.csv -> $PERF_DIR/results_${SLURM_JOB_ID}.csv"
fi

if [ "$BENCH_RC" -ne 0 ]; then
    echo "ERROR: benchmark failed on one or more nodes (srun exit $BENCH_RC)" >&2
fi
exit $BENCH_RC

