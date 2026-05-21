# Training Performance Validation of Primus Docker with Megatron backend on the AMD Instinct Accelerators

## Overview

Primus framework with megatron backend is designed to enable efficient training of large-scale language models on AMD GPUs. By leveraging AMD Instinct™ MI300X/MI350X accelerators, Primus Megatron framwework delivers enhanced scalability, performance, and resource utilization for AI workloads. It is purpose-built to support models like Llama 2, Llama 3/3.1, DeepseekV2/V3, and Mixtral MOE, enabling developers to train next-generation AI models with greater efficiency. See the GitHub repository at [AMD-AIG-AIMA/Primus](https://github.com/AMD-AIG-AIMA/Primus).

>[!NOTE]
>`rocm/pytorch-training` docker hub registry will be depreciated, in the future, please go to `rocm/primus` for latest ROCm pytorch training dockers, which will cover all the pytorch training ecosystem frameworks (e.g. TorchTitan, TorchTune, Megatron-LM, etc.).
>

The ROCm PyTorch Training Docker `rocm/primus:v26.3` (`rocm/pytorch-training:v26.3`) container, available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html), provides a prebuilt, optimized environment for pre-training a model on the AMD Instinct™ MI300X, MI325X, MI350X and MI355X accelerator. This ROCm PyTorch Docker includes the following components:

| Software component | Version              |
|--------------------|----------------------|
| ROCm               | 7.2.1                |
| Python             | 3.12.3               |
| PyTorch            | 2.10.0+git94c6e04    |
| Transformer Engine | 2.12.0.dev0+40434cf6 |
| Flash Attention    | 2.8.3                |
| hipBLASLt          | 1.3.0-c4b2dc9869     |
| Triton             | 3.6.0                |
| RCCL               | 2.27.7               |

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
* Zebra-Llama 1B/3B/8B
* Qwen 3 30B (A3B)
* Qwen3-235B-A22B
* Qwen 3 32B (SFT/ LoRA)
* GPT-OSS-20B
* GPT-OSS-120B

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
   docker pull rocm/primus:v26.3
   ```

3. **Launch Docker Container**
   Start the Docker container:
   ```bash
   docker run -it --device /dev/dri --device /dev/kfd --device /dev/infiniband --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:/userHome --shm-size 128G --name primus_training_env rocm/primus:v26.3
   ```
   **Note**: It's not recommended to bind the `$HOME` directory to the container using `-v $HOME:$HOME`. A good practice is only bind the directory you need to the container.

5. **Execute the training_env container (optional if no already in the container)**
   ```bash
    docker start primus_training_env
    docker exec -it primus_training_env bash
   ```

The docker container hosts verified commit `e16b27b` from [Primus repository](https://github.com/AMD-AGI/Primus/tree/e16b27bf6c1b2798f38848fc574fee60d9a9b902).
---

## 2. Configurations in yaml files (`‎examples/megatron/configs/`)

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
```
#### MI300X Performance Configs
```bash
#Set these variables for better performance only on MI300/MI325X
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1 #for better performance
```

- **Llama3.1-8B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml
```

- **Llama3.1-8B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

- **Llama2-7B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama2_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-FP8-pretrain.yaml
```

- **Llama2-7B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama2_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

- **Llama3.1-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml
```

- **Llama3.1-70B FP8 Proxy model on Single Node:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B_fp8_proxy.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_70B-FP8-pretrain.yaml \
  --train_iters 50 \
  --num_layers 40 \
  --fp8 hybrid \
  --no_fp8_weight_transpose_cache true
```
**Note:**
   - Please use >=2 nodes to run full llama 70B model with fp8 precision on MI300. MI35X can support full 70B model with fp8 precision in a single node. Please refer to MI35X config in next section.

- **Llama2-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama2_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_70B-BF16-pretrain.yaml
```

- **Llama3.3-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.3_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.3_70B-BF16-pretrain.yaml
```

Examples for MoE models with expert parallelism enabled, i.e, `expert_model_parallel_size > 1`

- **DeepSeekV2-Lite BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_deepseek_v2_lite.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

- **DeepSeekV3 BF16 3 layer proxy on Single Node:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_deepseek_v3_proxy.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
  --num_layers 3 \
  --moe_layer_freq 1 \
  --micro_batch_size 3 \
  --global_batch_size 192 \
  --train_iters 50
```

- **Mixtral 8x7B:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_mixtral_8x7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-BF16-pretrain.yaml
```

- **Mixtral 8x22B 4 layer proxy on Single Node:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_mixtral_8x22B_proxy.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
  --num_layers 4 \
  --pipeline_model_parallel_size 1 \
  --micro_batch_size 1 \
  --global_batch_size 16 \
  --train_iters 50
```

- **QWEN2.5 7B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen2.5_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml
```

- **QWEN2.5 7B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen2.5_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen2.5_7B-FP8-pretrain.yaml
```

- **QWEN2.5 72B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen2.5_72B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen2.5_72B-BF16-pretrain.yaml
```

- **Zebra-Llama-1B BF16:**
```bash
PRIMUS_TRAIN_RUNTIME=legacy bash runner/primus-cli direct \
  --log_file /tmp/primus_zebra_llama_1B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/zebra_llama_1B-pretrain.yaml
```

- **Qwen3-32B BF16 LoRA:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_32b.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/qwen3_32b_lora_posttrain.yaml
```

- **Qwen3-32B BF16 SFT:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_32b_sft.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml
```

- **Qwen3-30B (A3B) BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_30B_A3B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen3_30B_A3B-BF16-pretrain.yaml
```

- **Qwen3-30B (A3B) FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_30B_A3B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen3_30B_A3B-FP8-pretrain.yaml
```

- **GPT-OSS-20B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_gpt_oss_20B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/gpt_oss_20B-BF16-pretrain.yaml
```

- **GPT-OSS-20B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_gpt_oss_20B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/gpt_oss_20B-FP8-pretrain.yaml
```

#### MI35X Performance Configs
- **Llama3.1-8B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-FP8-pretrain.yaml
```

- **Llama3.1-8B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

- **Llama2-7B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama2_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama2_7B-FP8-pretrain.yaml
```

- **Llama2-7B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama2_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml
```

- **Llama3.1-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml
```

- **Llama3.1-70B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml
```

- **Llama2-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama2_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama2_70B-BF16-pretrain.yaml
```

- **Llama3.3-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.3_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.3_70B-BF16-pretrain.yaml
```

- **DeepSeekV2-Lite BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_deepseek_v2_lite.log \
  -- train pretrain \
  --config examples/megatron/configs//MI355X/deepseek_v2_lite-BF16-pretrain.yaml
```

- **DeepSeekV3 BF16 3 layer proxy on Single Node:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_deepseek_v3_proxy.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml \
  --num_layers 3 \
  --moe_layer_freq 1 \
  --train_iters 50 \
  --micro_batch_size 8 \
  --global_batch_size 64
```

- **Mixtral 8x7B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_mixtral_8x7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/mixtral_8x7B_v0.1-BF16-pretrain.yaml
```

- **Mixtral 8x22B BF16 4 layer proxy on Single Node:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_mixtral_8x22B_proxy.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml
```

- **QWEN2.5 7B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen2.5_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen2.5_7B-BF16-pretrain.yaml
```

- **QWEN2.5 7B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen2.5_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen2.5_7B-FP8-pretrain.yaml
```

- **QWEN2.5 72B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen2.5_72B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen2.5_72B-BF16-pretrain.yaml
```

- **Zebra-Llama-1B BF16:**
```bash
PRIMUS_TRAIN_RUNTIME=legacy bash runner/primus-cli direct \
  --log_file /tmp/primus_zebra_llama_1B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/zebra_llama_1B-pretrain.yaml
```

- **Qwen3-32B BF16 LoRA:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_32b_lora.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

- **Qwen3-30B (A3B) BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_30B_A3B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml
```

- **Qwen3-30B (A3B) FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_30B_A3B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml
```

- **GPT-OSS-20B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_gpt_oss_20B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml
```

- **GPT-OSS-20B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_gpt_oss_20B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-FP8-pretrain.yaml
```

### 3.2 Multi-node Training
To run training on multiple nodes, you can use the [run_slurm_pretrain.sh](https://github.com/AMD-AGI/Primus/blob/main/examples/run_slurm_pretrain.sh) script to launch multinode workloads. Below we list multinode setup and examples to run multinode tests.

MultiNode Setup:
> **Verify NCCL / network env first.** The `pimus-cli` launcher script sets sensible `NCCL_*` defaults via `base_env.sh`, but auto-detection can pick the wrong device on multi-NIC nodes. Always confirm `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME` (set to the same value as `NCCL_SOCKET_IFNAME`) are correct for your fabric. If necessary, you can `export` these environment variables before running.

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus/
git checkout release/v26.3
git submodule update --init --recursive
export DOCKER_IMAGE=rocm/primus:v26.3
export HF_TOKEN=<your_HF_token>
export NCCL_IB_HCA=<your_NCCL_IB_HCA> # specify which RDMA interfaces to use for communication
export NCCL_SOCKET_IFNAME=<your_NCCL_SOCKET_IFNAME> # your Network Interface
export GLOO_SOCKET_IFNAME=<your_GLOO_SOCKET_IFNAME> # your Network Interface
export NCCL_IB_GID_INDEX=3 # Set InfiniBand GID index for NCCL communication. Default is 3 for ROCE

# MI300/MI325 only extra settings
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1 #for better performance
```

For clusters using AMD AINIC, the following environment variables should be set.
```bash
export USING_AINIC=1
export NCCL_PXN_DISABLE=0
export NCCL_IB_GID_INDEX=1
```

Notes:
* Make sure correct network drivers are installed on the nodes. If inside a docker, either install the drivers inside the docker container or pass the network drivers from the host while creating docker container.
* If `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` are not set, Primus will try to auto-detect. However, since NICs can vary accross different cluster, it is encouraged to explicitly export your NCCL parameters for the cluster.
* To find your network interface, you can use `ip a`.
* To find rdma interfaces, you can use `ibv_devices` to get the list of all the RDMA/IB  devices.

- **Llama3.1-8B FP8 8 Node:**
```bash
# Adjust the training parameters. For e.g., `global_batch_size: 8 * #single_node_bs` for 8 nodes in this case
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml bash ./examples/run_slurm_pretrain.sh --global_batch_size 1024
```

- **Llama2-7B FP8 8 Node:**
```bash
# Adjust the training parameters. For e.g., `global_batch_size: 8 * #single_node_bs` for 8 nodes in this case
NNODES=8 EXP=examples/megatron/configs/MI300X/llama2_7B-FP8-pretrain.yaml bash ./examples/run_slurm_pretrain.sh --global_batch_size 2048
```

- **Llama3.1-70B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.1_70B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 4 --global_batch_size 256 --recompute_num_layers 80
```

- **Llama3.1-70B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256 --recompute_num_layers 12
```

- **Llama2-70B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama2_70B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 10 --global_batch_size 640 --recompute_num_layers 80
```

- **Llama2-70B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama2_70B-BF16-pretrain.yaml bash ./examples/run_slurm_pretrain.sh --micro_batch_size 2  --global_batch_size 1536  --recompute_num_layers 12
```

- **Llama3.3-70B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.3_70B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 4 --global_batch_size 256 --recompute_num_layers 80
```

- **Llama3.3-70B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.3_70B-BF16-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256 --recompute_num_layers 12
```

- **Mixtral 8x7B BF16 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-BF16-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 2 --global_batch_size 256
```

- **Qwen2.5-72B FP8 8 Nodes:**
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/qwen2.5_72B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 8 --global_batch_size 512 --recompute_num_layers 80
```

- **Mixtral-8x22B BF16 8 Nodes**
Launch the training using the `primus-cli` (recommended)
```bash
# In the Primus directory
./primus-cli slurm srun -N 8 -- train pretrain --config examples/megatron/configs/MI300X/mixtral_8x22B_v0.1-BF16-pretrain.yaml --micro_batch_size 2 --global_batch_size 256
```

Launch the training using the legacy script
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/mixtral_8x22B_v0.1-BF16-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 2 --global_batch_size 256
```

- **Llama3.1-405B FP8 8 Nodes**
Launch the training using the `primus-cli` (recommended)
```bash
# In the Primus directory
./primus-cli slurm srun -N 8 -- train pretrain --config examples/megatron/configs/MI300X/llama3.1_405B-FP8-pretrain.yaml --micro_batch_size 1 --global_batch_size 256 --decoder_first_pipeline_num_layers 15 --decoder_last_pipeline_num_layers 15
```
We use TP=8 for Llama3.1-405B model on 8 nodes. Because it has 126 layers which is not divisable by 8, we need to set `decoder_first_pipeline_num_layers` and `decoder_last_pipeline_num_layers`.

Launch the training using the legacy script
```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.1_405B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256  --decoder_first_pipeline_num_layers 15 --decoder_last_pipeline_num_layers 15
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
