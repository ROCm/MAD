#!/bin/bash

# ======================================================== #
#                      SLURM HEADERS                       #
# ======================================================== #

#SBATCH --job-name=training_llama3_8B
#SBATCH --output=logs/multinode-job-llama3-8b.%j.out
#SBATCH --time=3:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 # setting this to 8 would launch 8 dockers on the single node with 8 GPU
#SBATCH --exclusive
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation

# SLURM_NNODES
#     Total number of nodes in the job's resource allocation. See SLURM_JOB_NUM_NODES. Included for backwards compatibility. 

# SLURM_NODEID
#     ID of the nodes allocated. 

# SLURM_NODELIST
#     List of nodes allocated to the job. See SLURM_JOB_NODELIST. Included for backwards compatibility. 

# srun echo $SLURM_NNODES
# echo $SLURM_LOCALID
# srun echo $SLURM_LOCALID
# srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES echo $SLURM_LOCALID
# echo 'echo $SLURM_NODEID' > script.sh
# srun bash ./script.sh
# srun echo $SLURM_NODEID
# srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES echo $SLURM_NODEID
# srun echo $SLURM_NODELIST
# srun echo $SLURM_JOB_NODELIST
# see https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904

set -x
OUTPUT_DIR="$HOME/output"

# Install required packages
echo '
apt install iproute2 -y
apt install -y linux-headers-"$(uname -r)" libelf-dev
apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev
' > $OUTPUT_DIR/install_packages.sh

# Environment variables
echo 'export XLA_FLAGS="--xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=0 --xla_gpu_enable_latency_hiding_scheduler=TRUE --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_memory_limit_slop_factor=95"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.967
export LD_LIBRARY_PATH=/usr/local/lib/:/opt/rocm/lib:$LD_LIBRARY_PATH' > $OUTPUT_DIR/maxtext_env_8b.sh

# Model Configuration
echo 'base_config: "base.yml"
run_name: "llama3_8b_training"
base_output_directory: "./"
hardware: "gpu"
steps: 50
model_name: "llama3-8b"
enable_checkpointing: False
attention: "cudnn_flash_te"
log_period: 100
  #inter-node parallelism strategy
dcn_data_parallelism: -1
dcn_fsdp_parallelism: 1
  #intra-node parallelism strategy
ici_fsdp_parallelism: 8
ici_data_parallelism: 1
remat_policy: "minimal_flash"
use_iota_embed: True
scan_layers: False
async_checkpointing: False
logits_dot_in_fp32: False
profiler: ""
dtype: "bfloat16"
quantization: ""
quantize_kvcache: False
kv_quant_axis: "heads_and_dkv"
kv_quant_dtype: "int8"
weight_dtype: bfloat16
checkpoint_is_quantized: False # Set to True if reading from a saved aqt quantized checkpoint
max_target_length: 8192
per_device_batch_size: 4
hf_path: "parquet" 
hf_train_files: "/home/amd/data/c4/en/partial-train/000*.parquet"
dataset_type: "hf"
tokenizer_path: "/home/amd/data/Llama-2-7b-hf/"' > $OUTPUT_DIR/llama3_8b_gpu.yml


srun hostname
# srun master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# srun export MASTER_ADDR=$master_addr
export MASTER_NAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(cat /etc/hosts |  grep $MASTER_NAME | awk '{print $1}' )
# MASTER_ADDR=$(cat /etc/hosts |  grep gpu-14 | awk '{print $1}' )
srun echo "MASTER_ADDR="$MASTER_ADDR

# srun ping $MASTER_ADDR

#If podman is available instead of docker, then you need this export otherwise
#comment the below line and uncomment the line after that
export docker=podman
#docker=docker

IMAGE="rocm/jax-training:maxtext-v25.5"

export NNODES=$SLURM_NNODES
export JAX_COORDINATOR_IP=$MASTER_ADDR
export JAX_COORDINATOR_PORT=1234

#Change this to one of the IP interfaces used for communication
export NCCL_SOCKET_IFNAME=ens8np0
echo $NCCL_SOCKET_IFNAME

# For Mellanox NIC
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
# For Broadcom Thor NIC, uncomment the line below and comment the line above
#export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
echo $NCCL_IB_HCA

# get the test data
pip install -U "huggingface_hub[cli]" > /dev/null
export PATH=~/.local/bin/:$PATH
huggingface-cli download meta-llama/Llama-2-7b-chat-hf  --local-dir $HOME/araina/data/Llama-2-7b-hf/ --cache-dir $HOME/araina/data > /dev/null
huggingface-cli download legacy-datasets/c4 --include "*.parquet" --repo-type dataset --local-dir $HOME/araina/data/c4 --revision refs/convert/parquet --cache-dir $HOME/araina/data > /dev/null



srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES \
    --export=ALL \
    bash -c '\
    NODE_RANK=$SLURM_PROCID; \
    NNODES=$SLURM_JOB_NUM_NODES; \
    $docker run --rm --privileged --network host \
    --device /dev/dri --device /dev/kfd \
    --cap-add=IPC_LOCK \
    --volume /dev/infiniband:/dev/infiniband \
    -v $HOME:$HOME \
    -v $HOME/araina/data:/home/amd/data \
    --tmpfs /dev/shm:size=50G \
    --mount type=bind,source='${OUTPUT_DIR}',target=/workspace/maxtext/output \
    -e NNODES=$NNODES \
    -e NODE_RANK=$NODE_RANK \
    -e JAX_COORDINATOR_IP=$JAX_COORDINATOR_IP \
    -e JAX_COORDINATOR_PORT=$JAX_COORDINATOR_PORT \
    -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    -e NCCL_IB_HCA=$NCCL_IB_HCA \
    -w /workspace/maxtext \
    '${IMAGE}' \
    /bin/bash -c "
        set -e
        echo \"Running Llama-3-8b\"
	echo '${IMAGE}'
        echo \"Coordinator IP: \$JAX_COORDINATOR_IP\"
	cp '${OUTPUT_DIR}'/install_packages.sh .
	cp '${OUTPUT_DIR}'/maxtext_env_8b.sh .
	cp '${OUTPUT_DIR}'/llama3_8b_gpu.yml MaxText/configs/llama3_8b_gpu.yml
	source install_packages.sh
        source maxtext_env_8b.sh	
        python MaxText/train.py MaxText/configs/llama3_8b_gpu.yml base_output_directory=output 2>&1 |& tee -a llama3_8b.real.log
    "'

