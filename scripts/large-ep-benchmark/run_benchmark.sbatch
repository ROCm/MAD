#!/bin/bash
#SBATCH --job-name=ep-benchmarking
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00         # Set a time limit for the job (HH:MM:SS)
#SBATCH --output="ep_bench_slurm_job.out"
#SBATCH --error="ep_bench_slurm_job.err"

USER_NAME=$(whoami)
PWD=$(pwd)
LOG_PATH=${LOG_PATH:-"/shared_inference/${USER_NAME}/model_blog_logs"}
DOCKER_IMAGE=${DOCKER_IMAGE:-ep-benchmarking:latest}
IBDEVICES=${IBDEVICES:-mlx5_0}
SKIP_DEEPEP=${SKIP_DEEPEP:-0}

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
export IBDEVICES=$IBDEVICES
export SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
export NNODES=$NNODES
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"

srun bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
docker stop ep-bench;
sleep 10;
docker rm ep-bench;
sleep 10;
docker stop $(docker ps -aq) && docker rm $(docker ps -aq);
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
    -v /shared_inference/models:/data \
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
    --entrypoint /bin/bash \
    --name ep-bench \
    $DOCKER_IMAGE -c "
        echo Inside docker container ....
        echo ${MAD_REPO_PATH}
        $MAD_REPO_PATH/run_ep_bench.sh 2>&1 | tee /run_logs/ep_bench_results.log
"
'

srun bash -c 'docker stop ep-bench 2>/dev/null || true; docker rm -f ep-bench 2>/dev/null || true'

