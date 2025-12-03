# Training Performance Validation of Primus Docker with Megatron backend on the AMD Instinct Accelerators

## Overview

Primus framework with megatron backend is designed to enable efficient training of large-scale language models on AMD GPUs. By leveraging AMD Instinct™ MI300X/MI350X accelerators, Primus Megatron framwework delivers enhanced scalability, performance, and resource utilization for AI workloads. It is purpose-built to support models like Llama 2, Llama 3/3.1, DeepseekV2/V3, and Mixtral MOE, enabling developers to train next-generation AI models with greater efficiency. See the GitHub repository at [AMD-AIG-AIMA/Primus](https://github.com/AMD-AIG-AIMA/Primus).

>[!NOTE]
>`rocm/pytorch-training` docker hub registry will be depreciated, in the future, please go to `rocm/primus` for latest ROCm pytorch training dockers, which will cover all the pytorch training ecosystem frameworks (e.g. TorchTitan, TorchTune, Megatron-LM, etc.).
>

The ROCm PyTorch Training Docker `rocm/primus:v25.10` (`rocm/pytorch-training:v25.10`) container, available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html), provides a prebuilt, optimized environment for pre-training a model on the AMD Instinct™ MI300X, MI325X, MI350X and MI355X accelerator. This ROCm PyTorch Docker includes the following components:

| Software component  | Version            |
|---------------------|--------------------|
| ROCm               | 7.1.0              |
| Python            | 3.10          |
| PyTorch           | 2.10.0.dev20251112+rocm7.1   |
| Transformer Engine | 2.4.0.dev0+32e2d1d4      |
| Flash Attention   | 2.8.3               |
| hipBLASLt         | 09ab7153e2        |
| Triton            | 3.4.0                 |
| RCCL              | 2.27.7     |


## Supported features and models
Primus-Megatron-backend provides the following key features to train large language models efficiently:

* Primus Turbo with optimized attention and grouped gemm kernel
* Transformer Engine (TE)
* APEX
* GEMM tuning
* Torch.compile
* Flash Attention (FA) 3
* AITER Attention
* Fused kernels
* Pre-training
* FP8-GEMM
* Multi-node Support
* 3D parallelism: TP + SP + CP
* Distributed optimizer

The following models are pre-optimized for performance on the AMD Instinct MI300X accelerator.

* Llama 2 7B
* Llama 2 70B
* Llama 3/3.1 8B
* Llama 3/3.1/3.3 70B
* DeepSeek-V2-lite
* DeepSeek-V3
* Mixtral 8x7B
* Mixtral 8x22B
* Qwen 2.5 7/72B

## System validation steps
If you have already validated your system, skip this step; otherwise, please complete the following [system validation and optimization steps](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/prerequisite-system-validation.html) to set up your system before starting training.

### Disable NUMA auto-balancing
Generally, application performance can benefit from disabling NUMA auto-balancing. However, it might be detrimental to performance with certain types of workloads.

Run the command `cat /proc/sys/kernel/numa_balancing` to check your current NUMA (Non-Uniform Memory Access) settings. Output `0` indicates this setting is disabled. If there is no output or the output is `1`, run the following command to disable NUMA auto-balancing.

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```
See [Disable NUMA auto-balancing](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-disable-numa) for more information.


### Start training on AMD Instinct accelerators
The pre-built ROCm Primus-Megatron-backend environment allows users to quickly validate system performance, conduct training benchmarks, and achieve superior performance for models like Llama 2 and Llama 3.1. The docker is powered by Primus-turbo optimizations to achieve optimal performance.

This container should not be expected to provide generalized performance across all training workloads. Users should expect the container perform in the model configurations described below, but other configurations and run conditions are not validated by AMD. 
Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X accelerators with the AMD Megatron-LM Docker image.

---

# Training Procedure

## 1. Environment Setup

1. **Download Docker Image**
   Download the Docker image required for training:
   ```bash
   # MI300/MI325/MI35X
   docker pull rocm/primus:v25.10
   ```

3. **Launch Docker Container**
   Start the Docker container:
   ```bash
   docker run -it --device /dev/dri --device /dev/kfd --device /dev/infiniband --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME --shm-size 128G --name primus_training_env rocm/primus:v25.9_gfx942
   ```

5. **Execute the training_env container (optional if no already in the container)**
   ```bash
    docker start primus_training_env
    docker exec -it primus_training_env bash
   ```

The docker container hosts verified coomit `e16b27b` from [Primus repository](https://github.com/AMD-AGI/Primus/tree/e16b27bf6c1b2798f38848fc574fee60d9a9b902).
---

## 2. Configurations in Yaml Script (`‎examples/megatron/configs/`)

Primus defines training yaml for each model inside [‎examples/megatron/configs/](https://github.com/AMD-AGI/Primus/tree/e16b27bf6c1b2798f38848fc574fee60d9a9b902/examples/megatron/configs) repository. For example, use `examples/megatron/configs/llama3.1_8B-pretrain.yaml` for updating llama3.1_8B training parameters. Other yaml for the supported model can be found with `examples/megatron/configs/${MODEL_NAME}-pretrain.yaml` naming convention in this repository.

Users can toggle various training parameters such as `micro_batch_size`, `global_batch_size`, `train_iters` and other training paramaters inside the pretrain yamls. 

**Note**:
- Supported model definition can be found inside the [primus/configs/models/megatron/](https://github.com/AMD-AGI/Primus/tree/e16b27bf6c1b2798f38848fc574fee60d9a9b902/primus/configs/models/megatron) repository.
- To migrate existing workload from Rocm/Megatron-LM to primus or add new Workload, please follow the [Migration Guide](https://github.com/ROCm/MAD/blob/develop/benchmark/megatron_lm/Migration_Guide.md).

### 2.1 Dataset
You can use either mock data or real data for training. 

- **Mock Data:**
  The pretraining yaml scripts by default use `mock_data: true`. 
  
- **Real Data:**
  To use real data for training, set the variable `train_data_path: null` to your tokenized data path and set `mock_data: false`.
  
### 2.2 Tokenizer
In primus, each model uses tokenizer from huggingface. For example, llama3.1-8B model uses `tokenizer_model: meta-llama/Llama-3.1-8B` and `tokenizer_type: Llama3Tokenizer` defined in the [llama3.1-8B model](https://github.com/AMD-AGI/Primus/blob/e16b27bf6c1b2798f38848fc574fee60d9a9b902/examples/megatron/configs/llama3.1_8B-pretrain.yaml) definition. Please use HF_TOKEN with right permissions to access the tokenizer for each model.

```bash
# Export your HF_TOKEN in the workspace
export HF_TOKEN=<your_hftoken>
```
---

## 3. How to Run

### 3.1 Single Node Training
To run model training on a single node, go to `/workspace/Primus/` folder, and use the following command for setup. Once the setup is complete, use the individual model commands to start training:
```bash
pip install -r requirements.txt
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
```
#### MI300X Performance Configs 
```bash
#Set these variables for better performance only on MI300/MI325X
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
```

- **Llama3.1-8B FP8:**
```bash
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --fp8 hybrid
```

- **Llama3.1-8B BF16:**
```bash
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50
```

- **Llama2-7B FP8:**
```bash
EXP=examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --fp8 hybrid
```

- **Llama2-7B BF16:**
```bash
EXP=examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50
```

- **Llama3.1-70B BF16:**
```bash
EXP=examples/megatron/configs/MI300X/llama3.1_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50
```

- **Llama3.1-70B FP8 Proxy model on Single Node:**
```bash
EXP=examples/megatron/configs/MI300X/llama3.1_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --num_layers 40 \
    --fp8 hybrid \
    --no_fp8_weight_transpose_cache true
```
**Note:**
   - Please use >=2 nodes to run full llama 70B model with fp8 precision on MI300. MI35X can support full 70B model with fp8 precision in a single node. Please refer to MI35X config in next section.

- **Llama2-70B BF16:**
```bash
EXP=examples/megatron/configs/MI300X/llama2_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50
```

- **Llama3.3-70B BF16:**
```bash
EXP=examples/megatron/configs/MI300X/llama3.3_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --train_iters 50
```

Examples for MoE models with expert parallelism enabled, i.e, `expert_model_parallel_size > 1`

- **DeepSeekV2-Lite BF16:**
```bash
EXP=examples/megatron/configs/MI300X/deepseek_v2_lite-pretrain.yaml \
bash examples/run_pretrain.sh \
    --global_batch_size 256 \
    --train_iters 50
```

- **DeepSeekV3 BF16 3 layer proxy on Single Node:**
```bash
EXP=examples/megatron/configs/MI300X/deepseek_v3-pretrain.yaml \
bash examples/run_pretrain.sh \
    --num_layers 3 \
    --moe_layer_freq 1 \
    --micro_batch_size 3 \
    --global_batch_size 192 \
    --train_iters 50
```

- **Mixtral 8x7B:**
```bash
EXP=examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50
```

- **Mixtral 8x22B 4 layer proxy on Single Node:**
```bash
EXP=examples/megatron/configs/MI300X/mixtral_8x22B_v0.1-pretrain.yaml \
bash examples/run_pretrain.sh \
    --num_layers 4 \
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --train_iters 50
```

- **QWEN2.5 7B BF16:**
```bash
EXP=examples/megatron/configs/MI300X/qwen2.5_7B-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50
```

- **QWEN2.5 7B FP8:**
```bash
EXP=examples/megatron/configs/MI300X/qwen2.5_7B-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50 \
    --fp8 hybrid
```

- **QWEN2.5 72B BF16:**
```bash
EXP=examples/megatron/configs/MI300X/qwen2.5_72B-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50
```

#### MI35X Performance Configs
- **Llama3.1-8B FP8:**
```bash
EXP=examples/megatron/configs/MI355X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --fp8 hybrid \
    --micro_batch_size 4 \
    --global_batch_size 512
```

- **Llama3.1-8B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 4 \
    --global_batch_size 512
```

- **Llama2-7B FP8:**
```bash
EXP=examples/megatron/configs//MI355X/llama2_7B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --fp8 hybrid \
    --micro_batch_size 13 \
    --global_batch_size 416
```

- **Llama2-7B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/llama2_7B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 10 \
    --global_batch_size 640
```

- **Llama3.1-70B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/llama3.1_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 4 \
    --global_batch_size 32
```

- **Llama3.1-70B FP8:**
```bash
EXP=examples/megatron/configs//MI355X/llama3.1_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --fp8 hybrid \
    --no_fp8_weight_transpose_cache true \
    --micro_batch_size 3 \
    --global_batch_size 24
```

- **Llama2-70B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/llama2_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 17 \
    --global_batch_size 272
```

- **Llama3.3-70B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/llama3.3_70B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 6 \
    --global_batch_size 48
```

- **DeepSeekV2-Lite BF16:**
```bash
EXP=examples/megatron/configs//MI355X/deepseek_v2_lite-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 12 \
    --global_batch_size 768
```

- **DeepSeekV3 BF16 3 layer proxy on Single Node:**
```bash
EXP=examples/megatron/configs//MI355X/deepseek_v3-pretrain.yaml \
bash examples/run_pretrain.sh \
    --num_layers 3 \
    --moe_layer_freq 1 \
    --train_iters 50 \
    --micro_batch_size 8 \
    --global_batch_size 64
```

- **Mixtral 8x7B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/mixtral_8x7B_v0.1-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 4 \
    --global_batch_size 256
```

- **Mixtral 8x22B BF16 4 layer proxy on Single Node:**
```bash
EXP=examples/megatron/configs//MI355X/mixtral_8x22B_v0.1-pretrain.yaml \
bash examples/run_pretrain.sh \
    --num_layers 4 \
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --train_iters 50
```

- **QWEN2.5 7B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/qwen2.5_7B-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 16 \
    --global_batch_size 768
```

- **QWEN2.5 7B FP8:**
```bash
EXP=examples/megatron/configs//MI355X/qwen2.5_7B-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50 \
    --fp8 hybrid \
    --micro_batch_size 20 \
    --global_batch_size 800
```

- **QWEN2.5 72B BF16:**
```bash
EXP=examples/megatron/configs//MI355X/qwen2.5_72B-pretrain.yaml \
bash examples/run_pretrain.sh \
    --train_iters 50 \
    --micro_batch_size 16 \
    --global_batch_size 256
```
**Known Issues**:
- DeepSeekV3 proxy model and Mixtral 8x22B proxy model may exit with error due to memory free issue. However, this does not impacts training runs. All iterations, in this case 50, should have been completed before the exit and the results should also be available at the end.
  
### 3.2 Multi-node Training
To run training on multiple nodes, you can use the [run_slurm_pretrain.sh](https://github.com/AMD-AGI/Primus/blob/main/examples/run_slurm_pretrain.sh) script to launch multinode workloads. Below we list multinode setup and examples to run multinode tests.

MultiNode Setup:
```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus/
git checkout release/v25.10
git submodule update --init --recursive
export DOCKER_IMAGE=<DOCKER_IMAGE>
export HF_TOKEN=<your_HF_token>
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export NCCL_IB_HCA=<your_NCCL_IB_HCA> # specify which RDMA interfaces to use for communication
export NCCL_SOCKET_IFNAME=<your_NCCL_SOCKET_IFNAME> # your Network Interface
export GLOO_SOCKET_IFNAME=<your_GLOO_SOCKET_IFNAME> # your Network Interface
export NCCL_IB_GID_INDEX=3 # Set InfiniBand GID index for NCCL communication. Default is 3 for ROCE

# MI300/MI325 only extra settings
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1 
export NVTE_CK_IS_V3_ATOMIC_FP32=1
```

Notes:
* Make sure correct network drivers are installed on the nodes. If inside a docker, either install the drivers inside the docker container or pass the network drivers from the host while creating docker container.
* If `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` are not set, Primus will try to auto-detect. However, since NICs can vary accross different cluster, it is encouraged to explicitly export your NCCL parameters for the cluster.
* To find your network interface, you can use `ip a`.
* To find rdma interfaces, you can use `ibv_devices` to get the list of all the RDMA/IB  devices.

- **Llama3.1-8B FP8 8 Node:**
```bash
# Adjust the training parameters. For e.g., `global_batch_size: 8 * #single_node_bs` for 8 nodes in this case 
NNODES=8 EXP=examples/megatron/configs/llama3.1_8B-pretrain.yaml bash ./examples/run_slurm_pretrain.sh --global_batch_size 1024 --fp8 hybrid
```

- **Llama2-7B FP8 8 Node:**
```bash
# Adjust the training parameters. For e.g., `global_batch_size: 8 * #single_node_bs` for 8 nodes in this case 
NNODES=8 EXP=examples/megatron/configs/llama2_7B-pretrain.yaml bash ./examples/run_slurm_pretrain.sh --global_batch_size 2048 --fp8 hybrid
```

- **Llama3.1-70B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/llama3.1_70B-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 4 --global_batch_size 256 --recompute_num_layers 80 --no_fp8_weight_transpose_cache true --fp8 hybrid
```

- **Llama3.1-70B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/llama3.1_70B-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256 --recompute_num_layers 12
```

- **Llama2-70B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/llama2_70B-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 10 --global_batch_size 640 --recompute_num_layers 80 --no_fp8_weight_transpose_cache true --fp8 hybrid
```
 
- **Llama2-70B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/llama2_70B-pretrain.yaml bash ./examples/run_slurm_pretrain.sh --micro_batch_size 2  --global_batch_size 1536  --recompute_num_layers 12
```

- **Llama3.3-70B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/llama3.3_70B-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 4 --global_batch_size 256 --recompute_num_layers 80 --no_fp8_weight_transpose_cache true --fp8 hybrid
```

- **Llama3.3-70B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/llama3.3_70B-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256 --recompute_num_layers 12
```

- **Mixtral 8x7B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/mixtral_8x7B_v0.1-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 2 --global_batch_size 256
```

- **Qwen2.5-72B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/qwen2.5_72B-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 8 --global_batch_size 512 --recompute_num_layers 80 --no_fp8_weight_transpose_cache true --fp8 hybrid
```
---

## 4. Key Variables to Pay Attention To

- **fp8:**  
  `--fp8 hybrid`` enables fp8 GEMMS

- **use_torch_fsdp2:**  
  `use_torch_fsdp2: 1` enables torch fsdp-v2. 
  
  Note that if FSDP is enabled, then turn these variables to false `use_distributed_optimizer: false`, `overlap_param_gather: false`.

- **profile:**  
  To enable pytorch profiling, set all these parameter:
  ```bash
  profile: true
  use_pytorch_profiler: true
  profile_step_end: 7
  profile_step_start: 6
  ```
  
- **train_iters:**  
  Set the total number of iterations (default: 50).

- **mock_data:**
  By default set to true.

- **micro_batch_size:**
  Micro batch size

- **global_batch_size:**
  Global Batch size

- **recompute_granularity:**

  Activation Checkpointing (`null`, `sel` , `full`). Default: null.
  When set to `full`, also set `recompute_num_layers` and `recompute_method: (uniform or block)`

- **num_layers:**
  Using reduced number of layers as a proxy model

--- 
