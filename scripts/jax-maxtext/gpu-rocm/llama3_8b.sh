#!/bin/bash

set -x
# Create output dir
OUTPUT_DIR="$HOME/output"
mkdir -p $OUTPUT_DIR

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
hf_train_files: "/hf_cache/hub/datasets--legacy-datasets--c4/snapshots/5abe0d085aa23dd9db2a6c1e86cfce4e4db6f0c3/en/partial-train/000*.parquet"
dataset_type: "hf"
tokenizer_path: "meta-llama/Meta-Llama-3-8B"' > $OUTPUT_DIR/llama3_8b_gpu.yml

#If podman is available instead of docker, then you need this export otherwise
#comment the below line and uncomment the line after that
#export docker=podman
docker=docker



# get the test data
echo "For downloading data, we will mount \$HF_HOME to the docker and try to get llama tokenizer directly from there"
echo "Please set \$HF_HOME when calling this script, your HF_HOME is set as"
echo $HF_HOME
huggingface-cli download legacy-datasets/c4 --include "*.parquet" --repo-type dataset  --revision refs/convert/parquet


$docker run --rm --privileged --network host --device /dev/dri --device /dev/kfd \
  --cap-add=IPC_LOCK --volume /dev/infiniband:/dev/infiniband \
  -v $HOME:$HOME -v $HOME/data:/home/amd/data -v $HF_HOME:/hf_cache -e HF_HOME=/hf_cache --tmpfs /dev/shm:size=50G \
  --mount type=bind,source=$OUTPUT_DIR,target=/workspace/maxtext/output \
  -w /workspace/maxtext $IMAGE /bin/bash -c "
        set -e
        echo \"Running Llama-3-8b\"
	echo '${IMAGE}'
	cp $OUTPUT_DIR/maxtext_env_8b.sh .
	cp $OUTPUT_DIR/llama3_8b_gpu.yml MaxText/configs/llama3_8b_gpu.yml
        source maxtext_env_8b.sh	
        python MaxText/train.py MaxText/configs/llama3_8b_gpu.yml base_output_directory=output 2>&1 |& tee -a llama3_8b.real.log
    "

