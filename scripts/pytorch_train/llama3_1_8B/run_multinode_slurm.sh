#!/bin/bash
#SBATCH --job-name=pytorch_multi_node
#SBATCH --nodes=2               # Number of nodes
#SBATCH --cpus-per-task=192     # CPU cores per task
#SBATCH --time=01:00:00         # Max time
#SBATCH --output=out_%j.log     # Output log
#SBATCH --error=err_%j.log      # Error log
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation 
#SBATCH --exclusive 
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1


MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 -w $MASTER_NODE hostname --ip-address | awk '{print $1}')
MASTER_PORT=29500  # Default port for pytorch

export NUM_PROCESSES=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export NUM_MACHINES=$SLURM_NNODES
export MACHINE_RANK=$SLURM_NODEID
export MAIN_PROCESS_IP=$(hostname -i)


# Stop any existing Docker containers to avoid unwanted interruptions
srun bash -c 'docker stop $(docker ps -a -q)'


# Build and launch the Docker container
srun bash -c '
    docker pull docker.io/rocm/pytorch-training:v25.5
    docker rm training_env
    docker images
    ibdev2netdev
    docker run -d --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband \
      --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
      -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 128G --name training_env docker.io/rocm/pytorch-training:v25.5 tail -f /dev/null
'

srun bash -c '
  docker exec \
    -e NNODES=$SLURM_NNODES \
    -e GPUS_ON_NODE=$SLURM_GPUS_ON_NODE \
    -e RANK=$SLURM_NODEID \
    -e HEAD_NODE_IP='"$MASTER_ADDR"' \
    training_env \
    bash -c "
      cd $PATH
      ibv_devices
      bash run_multinode.sh
    "
' 

srun bash -c 'docker stop $(docker ps -a -q)'
srun bash -c 'docker rm training_env





