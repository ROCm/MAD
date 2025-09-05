#!/bin/bash
#SBATCH --job-name=Torchtune-multinode
#SBATCH --output=logs/slurm/multinode-job.%j.out
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks-per-node=1         # One task per GPU -> total 8 tasks per node
#SBATCH --gres=gpu:8                # Request 8 GPUs per node (TW: gpu:amd:8)
#SBATCH --time=8:00:00              # Adjust as necessary
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation # modify based on your reservation settings

############################################
# Environment Setup
############################################
export MASTER_ADDR=$(srun --ntasks=1 hostname | head -n 1)
export MASTER_PORT=29500
export CONTAINER_NAME="torchtune-multinode"
export IMAGE="docker.io/rocm/pytorch-training:v25.7"

export NCCL_IB_DISABLE=1
export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your active interface (check with `ip a`)

#export NNODES=4

############################################
# Torchtune run parameters (edit here)
############################################
export TUNE_NNODES=2
export TUNE_NPROC_PER_NODE=8
export TUNE_RDZV_BACKEND="c10d"
export TUNE_CONFIG="llama3_3/70B_full_multinode"

export TUNE_CHECKPOINT_DIR="/workspace/models/Llama-3.3-70B-Instruct"
export TUNE_MAX_STEPS_PER_EPOCH=10
export TUNE_EPOCHS=1
export TUNE_COMPILE=True
export TUNE_BATCH_SIZE=4
export TUNE_GAS=1
export TUNE_FSDP_CPU_OFFLOAD=False
export TUNE_TOK_MAX_SEQ_LEN=8192
export TUNE_DATASET_PACKED=True
export TUNE_TOKENIZER_PATH="/workspace/models/Llama-3.3-70B-Instruct/original/tokenizer.model"
export TUNE_OUTPUT_DIR="/workspace/result_torchtune"

############################################
# Stop Existing Containers
############################################
srun bash -c 'docker stop $(docker ps -a -q)'

############################################
# Build & Launch Docker Container
############################################
srun bash -c '
    docker rm '"$CONTAINER_NAME"'
    docker pull '"$IMAGE"'

    docker run -d \
        --device /dev/dri \
        --device /dev/kfd \
        --device /dev/infiniband \
        --network host \
        --ipc host \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --privileged \
        -v $HOST_MOUNT:$CONTAINER_MOUNT \
        -v $HOST_MOUNT_FOR_RESULTS:$CONTAINER_MOUNT_FOR_RESULTS \
        -v $HOME:$HOME \
        -v /opt/bcm_232.1.132.8c/:/opt/bcm_232.1.132.8c/ \
        -v $HOME/.ssh:/root/.ssh \
        --shm-size 128G \
        --name $CONTAINER_NAME \
        $IMAGE tail -f /dev/null

    echo "ibv_devices"
    ibv_devices
'

############################################
# Execute Training in Docker Container
############################################
srun bash -c '
    echo "Trying to start container: '"$CONTAINER_NAME"'"
    docker start '"$CONTAINER_NAME"'
    docker ps

    docker exec \
        -e NODE_RANK=$SLURM_NODEID \
        -e MASTER_ADDR='"$MASTER_ADDR"' \
        -e MASTER_PORT='"$MASTER_PORT"' \
        -e BS='"$BS"' \
        -e MBS='"$MBS"' \
        -e TUNE_NNODES='"$TUNE_NNODES"' \
        -e TUNE_NPROC_PER_NODE='"$TUNE_NPROC_PER_NODE"' \
        -e TUNE_RDZV_BACKEND='"$TUNE_RDZV_BACKEND"' \
        -e TUNE_CONFIG='"$TUNE_CONFIG"' \
        -e TUNE_CHECKPOINT_DIR='"$TUNE_CHECKPOINT_DIR"' \
        -e TUNE_MAX_STEPS_PER_EPOCH='"$TUNE_MAX_STEPS_PER_EPOCH"' \
        -e TUNE_EPOCHS='"$TUNE_EPOCHS"' \
        -e TUNE_COMPILE='"$TUNE_COMPILE"' \
        -e TUNE_BATCH_SIZE='"$TUNE_BATCH_SIZE"' \
        -e TUNE_GAS='"$TUNE_GAS"' \
        -e TUNE_FSDP_CPU_OFFLOAD='"$TUNE_FSDP_CPU_OFFLOAD"' \
        -e TUNE_TOK_MAX_SEQ_LEN='"$TUNE_TOK_MAX_SEQ_LEN"' \
        -e TUNE_DATASET_PACKED='"$TUNE_DATASET_PACKED"' \
        -e TUNE_TOKENIZER_PATH='"$TUNE_TOKENIZER_PATH"' \
        -e TUNE_OUTPUT_DIR='"$TUNE_OUTPUT_DIR"' \
        '"$CONTAINER_NAME"' \
        bash -c "
            echo Inside container: NODE_RANK=\$NODE_RANK
            echo MASTER_ADDR=\$MASTER_ADDR
            echo MASTER_PORT=\$MASTER_PORT
            echo NNODES=\$TUNE_NNODES

            pwd
            cd /opt/bcm_232.1.132.8c/drivers_linux/bnxt_rocelib/libbnxt_re-232.0.155.5

            export DEBIAN_FRONTEND=noninteractive
            apt update
            apt install -y linux-headers-$(uname -r) libelf-dev
            apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils \
                           perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers \
                           libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev

            # Compile RoCE userlib
            cd libbnxt_re-232.0.155.5
            sh autogen.sh
            ./configure
            make
            find /usr/lib64/ /usr/lib -name \"libbnxt_re-rdmav*.so\" -exec mv {} {}.inbox \;
            make install all
            echo /usr/local/lib | tee -a /etc/ld.so.conf
            ldconfig
            cp -f bnxt_re.driver /etc/libibverbs.d/
            find . -name \"*.so\" -exec md5sum {} \;
            echo RoCE userlib compile complete

            cd /workspace/torchtune

            # === Finetune Run (parametrized) ===
            tune run \
                --nnodes \"\$TUNE_NNODES\" \
                --nproc_per_node \"\$TUNE_NPROC_PER_NODE\" \
                --rdzv_backend \"\$TUNE_RDZV_BACKEND\" \
                --rdzv_endpoint \"\$MASTER_ADDR:\$MASTER_PORT\" \
                full_finetune_distributed \
                --config \"\$TUNE_CONFIG\" \
                checkpointer.checkpoint_dir=\"\$TUNE_CHECKPOINT_DIR\" \
                max_steps_per_epoch=\"\$TUNE_MAX_STEPS_PER_EPOCH\" \
                epochs=\"\$TUNE_EPOCHS\" \
                compile=\"\$TUNE_COMPILE\" \
                batch_size=\"\$TUNE_BATCH_SIZE\" \
                gradient_accumulation_steps=\"\$TUNE_GAS\" \
                fsdp_cpu_offload=\"\$TUNE_FSDP_CPU_OFFLOAD\" \
                tokenizer.max_seq_len=\"\$TUNE_TOK_MAX_SEQ_LEN\" \
                dataset.packed=\"\$TUNE_DATASET_PACKED\" \
                tokenizer.path=\"\$TUNE_TOKENIZER_PATH\" \
                output_dir=\"\$TUNE_OUTPUT_DIR\"
        "
'

# Optional cleanup
# srun docker stop $CONTAINER_NAME
# srun docker rm $CONTAINER_NAME
