#!/bin/bash

# ======================================================== #
#                      SLURM HEADERS                       #
# ======================================================== #

#SBATCH --job-name=training_llama2_70B
#SBATCH --output=logs/multinode-job-llama2-70b.%j.out
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
apt install rdma-core -y
apt install -y linux-headers-"$(uname -r)" libelf-dev
apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev
' > $OUTPUT_DIR/install_packages.sh


# Environment variables
echo '
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export NVTE_USE_HIPBLASLT=1
export XLA_FLAGS="--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_graph_level=0 --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=0 --xla_gpu_enable_all_gather_combine_by_dim=FALSE"
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NVTE_FUSED_ATTN=1
export NCCL_DEBUG=VERSION
export NCCL_IB_TIMEOUT=20
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=0
' > $OUTPUT_DIR/maxtext_env_70b.sh


# Model Configuration
echo 'base_config: "base.yml"
run_name: "llama2_70b_training"
hardware: "gpu"
steps: 30
model_name: "llama2-70b"
enable_checkpointing: False
attention: "cudnn_flash_te"
dcn_data_parallelism: -1
dcn_fsdp_parallelism: 2
dcn_pipeline_parallelism: 1
dcn_tensor_parallelism: 1
dcn_sequence_parallelism: 1
ici_fsdp_parallelism: 8
ici_data_parallelism: 1
ici_sequence_parallelism: 1
ici_tensor_parallelism: 1
ici_pipeline_parallelism: 1

remat_policy: 'full'
optimizer_memory_host_offload: False
param_scan_axis: 1

use_iota_embed: True
scan_layers: True

profiler: ""

async_checkpointing: False
logits_dot_in_fp32: False
megablox: False
dtype: "bfloat16"
quantization: ""
quantize_kvcache: False
kv_quant_axis: "heads_and_dkv"
kv_quant_dtype: "int8"
weight_dtype: bfloat16
checkpoint_is_quantized: False # Set to True if reading from a saved aqt quantized checkpoint
per_device_batch_size: 15
max_target_length: 4096
dataset_type: "synthetic"
enable_goodput_recording: False
monitor_goodput: False
shardy: False
' > $OUTPUT_DIR/llama2_70b_gpu.yml


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
    -e JAX_COORDINATOR_IP='${JAX_COORDINATOR_IP}' \
    -e JAX_COORDINATOR_PORT='${JAX_COORDINATOR_PORT}' \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -w /workspace/maxtext \
    '${IMAGE}' \
    /bin/bash -c "
        set -e
        echo \"Running Llama-2-70b\"
        echo '${IMAGE}'
        echo \"Coordinator IP: \$JAX_COORDINATOR_IP\"
        cp '${OUTPUT_DIR}'/install_packages.sh .
        cp '${OUTPUT_DIR}'/maxtext_env_70b.sh .
        mkdir -p configs
        cp '${OUTPUT_DIR}'/llama2_70b_gpu.yml configs/llama2_70b_gpu.yml
        source install_packages.sh
        source maxtext_env_70b.sh
        python -m maxtext.trainers.pre_train.train configs/llama2_70b_gpu.yml 2>&1 |& tee -a llama2_70b.synthetic.log
        "'