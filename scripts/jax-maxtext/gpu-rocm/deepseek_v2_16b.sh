#!/bin/bash

set -x
# Create output dir
OUTPUT_DIR="$HOME/output"
mkdir -p $OUTPUT_DIR

# Environment variables
echo ' export XLA_FLAGS=" --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_latency_hiding_scheduler=TRUE --xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0  --xla_gpu_autotune_level=5 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_memory_limit_slop_factor=95"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.967
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' > $OUTPUT_DIR/deepseek2_env_16b.sh

# Model Configuration
echo 'base_config: "base.yml"
run_name: "deepseek2_16b_1node"
base_output_directory: "./"
hardware: "gpu"
steps: 50
model_name: "deepseek2-16b"
enable_checkpointing: False
attention: "dot_product"
log_period: 100
#inter-node parallelism strategy
dcn_data_parallelism: -1
dcn_fsdp_parallelism: 1
#intra-node parallelism strategy
ici_fsdp_parallelism: 1
ici_data_parallelism: 1
ici_expert_parallelism: -1
remat_policy: "minimal_flash"
use_iota_embed: True
scan_layers: True
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
max_target_length: 4096
dataset_type: "synthetic"
per_device_batch_size: 8
megablox: False
capacity_factor: 1.25
sparse_matmul: False
sharding_tolerance: 0.05' > $OUTPUT_DIR/deepseek2_16b_gpu.yml



#If podman is available instead of docker, then you need this export otherwise
#comment the below line and uncomment the line after that
#export docker=podman
docker=docker

#docker run --rm --privileged --network host --device /dev/dri --device /dev/kfd --cap-add=IPC_LOCK --volume /dev/infiniband:/dev/infiniband -w /workspace/maxtext $IMAGE /bin/bash
#--cap-add=IPC_LOCK --volume /dev/infiniband:/dev/infiniband --tmpfs /dev/shm:size=50G -w /workspace/maxtext '${IMAGE}' /bin/bash

$docker run --rm --privileged --network host --device /dev/dri --device /dev/kfd \
  --cap-add=IPC_LOCK --volume /dev/infiniband:/dev/infiniband \
  -v $HOME:$HOME -v $HOME/data:/home/amd/data --tmpfs /dev/shm:size=50G \
  --mount type=bind,source=$OUTPUT_DIR,target=/workspace/maxtext/output \
  -w /workspace/maxtext $IMAGE /bin/bash -c "
    set -e
    echo \"Running Deepseek-v2-16b\"
    echo '${IMAGE}'
    cp $OUTPUT_DIR/deepseek2_env_16b.sh .
    cp $OUTPUT_DIR/deepseek2_16b_gpu.yml MaxText/configs/.
    source deepseek2_env_16b.sh
    python /workspace/maxtext/MaxText/train.py MaxText/configs/deepseek2_16b_gpu.yml base_output_directory=output 2>&1 |& tee -a  deepseek_v2_16b.log
   "
