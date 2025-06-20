#!/bin/bash
#SBATCH --job-name=titan-multinode
#SBATCH --output=logs/slurm/titan-multinode.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=384                  # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation # modify based on your reservation settings

echo "get first node"
# Get the list of nodes and the first node (master node)
# node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
MASTER_ADDR=$(srun --ntasks=1 hostname | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29475}"

echo "Trying 'docker ps'..."
if docker ps; then
    echo "Docker is working."
    export container_command=docker
else
    echo "'docker ps' failed. Trying 'podman ps'..."
    if podman ps; then
        echo "Podman is working."
        export container_command=podman
    else
        echo "Both 'docker ps' and 'podman ps' failed."
        exit 1
    fi
fi

# Define the Docker image
export NCCL_IB_HCA=$(bash "get_nccl_ib_hca.sh")
echo $NCCL_IB_HCA

export DOCKER_IMAGE="${DOCKER_IMAGE:-"docker.io/rocm/pytorch-training:v25.6"}"
# Pull docker image
${container_command} pull $DOCKER_IMAGE
# Setup your keys for HF and WADNB
export HF_TOKEN="${HF_TOKEN:-your_huggingface_token}" # replace with your Hugging Face token
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}" # replace with your Weights & Biases API key

export TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "TIME_STAMP=$TIME_STAMP"

# Define the mount points
export CURRENT_DIR=${PWD}                             # change this path to Megatron-LM inside the docker
export NETWORK_INTERFACE=${NETWORK_INTERFACE:-"bond0"} # Can be get by run `ip a`
export TITAN_DIR=${TITAN_DIR:-"/workspace/torchtitan"} # change this path to Megatron-LM inside the docker
export CONTAINER_DIR=${HOME}
export HOST_MOUNT=${HOST_MOUNT:=${HOME}}               # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:=${HOME}}     # change this path to development workspace path inside the docker

# Run the Docker container with the script
srun bash -c '${container_command} stop $(${container_command} ps -q) ; \
  ${container_command} run --rm \
 --env MASTER_ADDR=$MASTER_ADDR \
 --env MASTER_PORT=$MASTER_PORT \
 --env PROCID=$SLURM_PROCID \
 --env NODEID=$SLURM_NODEID \
 --env NNODES=$SLURM_NNODES \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date) ; \
 cp $CURRENT_DIR/run_multinode_train.sh $TITAN_DIR ; \
 cd $TITAN_DIR; \
 pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --force-reinstall ; \
 pip3 install -r requirements.txt ; \
 pip3 install -e . ; \
 pip3 install typeguard; \
 pip3 install tyro ; \
 export NNODES=$SLURM_NNODES ; \
 export NODE_RANK=$SLURM_NODEID ;  \
 export NCCL_IB_HCA=$NCCL_IB_HCA ; \
 export NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE}; \
 export GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE}; \
 mkdir -p output/llama3_70b/NNODE_${SLURM_NNODES}/${TIME_STAMP} ; \ 
 CONFIG_FILE=./torchtitan/models/llama3/train_configs/llama3_70b.toml bash run_multinode_train.sh 2>&1 | tee output/llama3_70b/NNODE_${SLURM_NNODES}/${TIME_STAMP}/rank_${SLURM_NODEID}.log ; \ echo $(date) 
 " 
 '
