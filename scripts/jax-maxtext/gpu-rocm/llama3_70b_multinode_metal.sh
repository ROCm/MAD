#!/bin/bash
#
# This is a version of training script for bare metal runs,
# without slurm or k8s, all you need is ssh access to the gpu servers
# 
# This script is to be invoked like this:
#
# for x in $(cat host_ip_file); \
#    do \
#      ssh root@$x "docker exec jax_train /workspace/maxtext/output/llama3_70b_multinode_metal.sh" & \
#    done
#
# on all the gpu servers, run the jax container and create
# jax_train like this:
# 
# 
# docker run -d -it --name jax_train --network host --ipc host \
#    --privileged --shm-size 64G --tmpfs /dev/shm:size=200G \
#		 --cap-add IPC_LOCK --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
#		 --group-add video --device /dev/kfd --device /dev/dri --device /dev/infiniband \
#		 --volume /dev/infiniband:/dev/infiniband -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
#		 -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro \
#		 -v /usr/local/lib:/usr/local/lib:ro -v /root/.ssh:/root/.ssh:ro \
#		 -v /mnt/testvfs/jax/output:/workspace/maxtext/output \
#		 -w /workspace/maxtext rocm/jax-training:maxtext-v25.5 bash
#
# this script llama3_70b_multinode_metal.sh needs to be placed in output directory of 
#  the docker, furthermore, output directory should be NFS mounted which is mounted
#  across all the containers, that way runjax script is the same across
#  all the containers. based on the host_info array and matching the $(hostname)
#  it will compute its own node rank
#


# creating the host_info array (create a plain text file with ip and name)
# and pass it to following one-line bash script to generate host_info array
#
#    rank=0; \
#    while read -r ip name; \
#    do \
#	printf "host_info[\"%s\"]=\"%s %d\"\n" $name $ip $rank \
#	rank=$((rank+1)) \
#    done <  /tmp/v16.names 
#
    
declare -A host_info

# host_info array is indexed by hostname, and has a tuple of (ip, rank) as value
host_info["node1"]="10.10.0.1 0"
host_info["node2"]="10.10.0.2 1"
host_info["node3"]="10.10.0.3 2"
host_info["node4"]="10.10.0.4 3"
host_info["node5"]="10.10.0.5 4"
host_info["node6"]="10.10.0.6 5"
host_info["node7"]="10.10.0.7 6"
host_info["node8"]="10.10.0.8 7"
host_info["node9"]="10.10.0.9 8"
host_info["node10"]="10.10.0.10 9"
host_info["node11"]="10.10.0.11 10"
host_info["node12"]="10.10.0.12 11"
host_info["node13"]="10.10.0.13 12"
host_info["node14"]="10.10.0.14 13"
host_info["node15"]="10.10.0.15 14"
host_info["node16"]="10.10.0.16 15"


export NNODES=2

# Get the current hostname
current_hostname=$(hostname)

if [[ -n "${host_info[$current_hostname]}" ]]; then
    read -r ip rank <<< "${host_info[$current_hostname]}"
    echo "Current Hostname: $current_hostname"
    echo "IP Address: $ip"
    echo "Rank: $rank"
    NODE_RANK=$rank
else
    echo "Hostname '$current_hostname' not found in the host dictionary."
    exit 1
fi

# change this to MASTER_ADDR
export JAX_COORDINATOR_IP='10.10.0.1'
export JAX_COORDINATOR_PORT=12345
export JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS=1800
export JAX_PROCESS_COUNT=${NNODES}
export JAX_PROCESS_INDEX=${rank}
export NODE_RANK=$rank

set -e
echo "Starting node $NODE_RANK of $NNODES"
echo "Coordinator IP: $JAX_COORDINATOR_IP"

apt update
apt install iproute2 -y
apt install rdma-core -y
apt install apt-utils -y
apt install -y linux-headers-"$(uname -r)" libelf-dev
apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils \
    infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev \
    rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers \
    libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev


mkdir -p /workspace/maxtext/output/configs
cat > /workspace/maxtext/output/configs/llama3_70b_gpu.yml <<EOF
base_config: "base.yml"
run_name: "llama3_70b_training"
hardware: "gpu"
steps: 30
model_name: "llama3-70b"
enable_checkpointing: False
attention: "cudnn_flash_te"
dcn_data_parallelism: -1
dcn_fsdp_parallelism: 1
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
per_device_batch_size: 7
max_target_length: 8192
dataset_type: "synthetic"
EOF


export LD_LIBRARY_PATH=/usr/local/lib/:/opt/rocm/lib:$LD_LIBRARY_PATH
export PROFILE_TB=1
export JAX_TRACEBACK_FILTERING=off
export NCCL_IB_TIMEOUT=20
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=0 --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_all_reduce_combine_threshold_bytes=8589934592  --xla_gpu_all_gather_combine_threshold_bytes=137438953472 --xla_gpu_enable_all_gather_combine_by_dim=FALSE"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export NVTE_FUSED_ATTN=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.975
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_IB_TC=41
export NCCL_IB_SL=0
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET # for NCCL debug. 
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
#export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_SOCKET_IFNAME=ens51f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
export UCX_IB_TRAFFIC_CLASS=41
export UCX_IB_SL=0
export UCX_TLS=tcp,self,sm
export NVTE_CK_BWD_V3=1
export NVTE_CK_V3_BF16_CVT=2


# Set environment variables for the run
export RUN_TIME=$(date "+%F-%H-%M")
export BASE_RUN_NAME="maxtext_aa"
export RUN_NAME="${BASE_RUN_NAME}-${RUN_TIME}"
export HOSTNAME=$(hostname)
export BASE_OUTPUT_DIRECTORY="/workspace/maxtext/output"
export OUT_FILE_NAME="${BASE_OUTPUT_DIRECTORY}/out-3.1_405b-NNodes${NNODES:-1}-${RUN_NAME}-${HOSTNAME}.file"

# Error handling
set -e
trap "echo \"Error on line $LINENO\"" ERR

echo "Starting training run with:"
echo "- Number of nodes: ${NNODES}"
echo "- Coordinator IP: ${JAX_COORDINATOR_IP}"
echo "- Output directory: ${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/"
echo "Running ${BASE_RUN_NAME} with output in ${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/"


set -e
echo "Running Llama-3-70b"
echo "Coordinator IP: \$JAX_COORDINATOR_IP"
#python MaxText/train.py MaxText/configs/llama3_70b_gpu.yml base_output_directory=output 2>&1 |& tee -a llama3_70b.real.log"

# Run the training
python /workspace/maxtext/MaxText/train.py /workspace/maxtext/output/configs/llama3_70b_gpu.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} 2>&1 | tee >(grep ".") > ${OUT_FILE_NAME}
