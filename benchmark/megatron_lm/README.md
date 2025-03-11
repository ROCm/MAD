# Training Performance Validation with ROCm Megatron-LM Training Docker on the AMD Instinct Accelerators

## Overview

ROCm Megatron-LM framework is a specialized fork of the robust Megatron-LM, designed to enable efficient training of large-scale language models on AMD GPUs. By leveraging AMD Instinct™ MI300X accelerators, AMD Megatron-LM delivers enhanced scalability, performance, and resource utilization for AI workloads. It is purpose-built to support models like Meta’s Llama 2, Llama 3, and Llama 3.1, enabling developers to train next-generation AI models with greater efficiency. See the GitHub repository at [ROCm/Megatron-LM](https://github.com/ROCm/Megatron-LM/).

For ease of use, AMD provides a ready-to-use Docker image for MI300X accelerators containing essential components, including PyTorch, PyTorch Lightning, ROCm libraries, and Megatron-LM utilities. It contains the following software to accelerate training workloads:

| Software component  | Version            |
|---------------------|--------------------|
| ROCm               | 6.3.0              |
| Python            | 3.10               |
| PyTorch           | 2.7.0a0+git637433   |
| Transformer Engine | 1.11               |
| Flash Attention   | 3.0.0               |
| hipBLASLt         | git258a2162         |
| Triton            | 3.1                 |


## Supported features and models
Megatron-LM provides the following key features to train large language models efficiently:

* Transformer Engine (TE)
* APEX
* GEMM tuning
* Torch.compile
* Flash Attention (FA) 3
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
* Llama 3/3.1 70B
* DeepSeek-V2-lite

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
The pre-built ROCm Megatron-LM environment allows users to quickly validate system performance, conduct training benchmarks, and achieve superior performance for models like Llama 2 and Llama 3.1.

This container should not be expected to provide generalized performance across all training workloads. Users should expect the container perform in the model configurations described below, but other configurations and run conditions are not validated by AMD. 
Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X accelerators with the AMD Megatron-LM Docker image.

---

# LLama Training Procedure

## 1. Environment Setup

1. **Download Docker Image**
   Download the Docker image required for training:
   ```bash
   docker pull rocm/megatron-lm:v25.3
   ```

2. **Launch Docker Container**
   Start the Docker container:
   ```bash
   docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name megatron_training_env rocm/megatron-lm:v25.3
   ```

3. **Execute the training_env container (optional if no already in the container)**
   ```bash
    docker start megatron_training_env
    docker exec -it megatron_training_env bash
   ```

The docker container hosts verified Megatron-LM repository, which is available in [megatron release branch](https://github.com/ROCm/Megatron-LM/tree/megatron_release_v25.3).

---

## 2. Configurations in Script (`Megatron-LM/examples/llama`)
Use `train_llama3.sh` for Llama3/3.1 models and `train_llama2.sh` for Llama2 models.

### 2.1 Network Interface
Update the network interface in the script to match your system’s network interface.
To find your network interface, run (out of the container):
```bash
ip a
```
Then, update the following variables in the script:
```bash
export NCCL_SOCKET_IFNAME=ens50f0np0
export GLOO_SOCKET_IFNAME=ens50f0np0
```

### 2.2 Dataset
You can use either mock data or real data for training.

- **Mock Data:**
  Use `MOCK_DATA` variable to toggle between mock and real data. The default value is 1.
  ```bash
  MOCK_DATA=1
  ```
- **Real Data:**
  Update the `DATA_PATH` to the location where your dataset is stored:
  ```bash
  MOCK_DATA=0
  DATA_PATH=${DATA_PATH:-"/data/bookcorpus_text_sentence"}  # Change to where your dataset is stored
  ```

### 2.3 Tokenizer
Tokenization is the process of converting raw text into tokens that can be processed by the model. For Llama models, this typically involves sub-word tokenization, where words are broken down into smaller units based on a fixed vocabulary. The tokenizer is trained along with the model on a large corpus of text, and it learns a fixed vocabulary that can represent a wide range of text from different domains. This allows Llama models to handle a variety of input sequences, including unseen words or domain-specific terms.

- **For Llama2 Training:**
  To train any of the Llama2 models that this Docker image supports, use the `Llama2Tokenizer`.

- **For Llama3 Training:**
  To train any of Llama 3 and Llama 3.1 models that this Docker image supports, use the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=meta-llama/Llama-3.1-8B  # For Llama3
  ```

### 2.4 Multi-node Training
If you're running multi-node training, update the following environment variables on each node. They can also be passed as command line arguments.

- **Master Address:**
  Change `localhost` to the master node's hostname:
  ```bash
  MASTER_ADDR="${MASTER_ADDR:-localhost}"
  ```

- **Number of Nodes:**
  Set the number of nodes you want to train on (e.g., 2, 4, 8):
  ```bash
  NNODES="${NNODES:-1}"
  ```

- **Node Rank:**
  Set the rank of each node (0 for master, 1 for the first worker node, etc.):
  ```bash
  NODE_RANK="${NODE_RANK:-0}"
  ```

- **DATA_CACHE_PATH:**
  Set `DATA_CACHE_PATH` to a common directory accessible by all the nodes (for eg, an NFS directory) for multi-node runs
  ```bash
  DATA_CACHE_PATH=/root/cache #Set to a common directory for multi-node runs
  ```

 - **Network Drivers Inside Docker:**
   For multi-node runs, make sure the correct network drivers are installed on the nodes. If inside a docker, either install the drivers inside the docker container or pass the network drivers from the host while creating the Docker container.


## 3. How to Run

### 3.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  bash examples/llama/train_llama3.sh
```

### 3.2 Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. For example, follow these steps for 2 Node run with Node0 as the master node :

- **On the Master Node0:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=0 bash examples/llama/train_llama3.sh
  ```

- **On the Worker Node1:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=1 bash examples/llama/train_llama3.sh
  ```
---

## 4. Key Variables to Pay Attention To

- **TE_FP8:**
  `0` for BP16 (default), `1` for FP8-GEMMS.

- **GEMM_TUNING:**
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **USE_FLASH_ATTN:**
  `1` to enable Flash Attention.

- **ENABLE_PROFILING:**
  `1` to enable PyTorch profiling for performance analysis.

- **transformer-impl:**
  `transformer_engine` to use the Transformer Engine (TE). Set to `local` if you want to disable TE.

- **MODEL_SIZE:**
  Set to `7B` or `70B` for Llama2, or `8B` or `70B` for Llama3/3.1.

- **TOTAL_ITERS:** 
  Set the total number of iterations (default: 10).

- **MOCK_DATA:**
  Use MOCK_DATA if set to 1, otherwise use the real data provided by user (DEFAULT: 1)

- **MBS:**
  Micro batch size

- **BS:**
  Global Batch size

- **TP:**
  Tensor parallel (1, 2, 4, 8)

- **SEQ_LENGTH**:
  Sequence Length

---

That's it! You've now set up the environment and configured the necessary settings for training Llama2 or Llama3 models.

# DeepSeek-V2-lite Training Procedure

## 1. Environment Setup

1. **Download Docker Image**
   Download the Docker image required for training:
   ```bash
   docker pull rocm/megatron-lm:v25.3
   ```

2. **Launch Docker Container**
   Start the Docker container:  
   ```bash
   docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name megatron_training_env rocm/megatron-lm:v25.3
   ```
   
   The docker container hosts verified Megatron-LM repository, which is available in [megatron release branch](https://github.com/ROCm/Megatron-LM/tree/megatron_release_v25.3).

---

## 2. Prepare Dataset
Skip this step, if you already have the dataset or you can download deepseek dataset using the command.

<pre>
mkdir deepseek-datasets
cd deepseek-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx
</pre>

## 3. Configurations in Script (`Megatron-LM/examples/deepseek_v2`)
Use `train_deepseekv2.sh` script

### 3.1 Dataset
You can use either mock data or real data for training.

- **Mock Data:**
  Use `MOCK_DATA` variable to toggle between mock and real data. Default value is 1. 
  ```bash
  MOCK_DATA=1
  ```
- **Real Data:**
  Update the `DATA_DIR` to the location where your dataset is stored:
  ```bash
  MOCK_DATA=0
  DATA_DIR="/root/data/deepseek-datasets"  # Change to where your dataset is stored
  ```

### 3.2 Tokenizer
DeepSeek-V2 uses `DeepSeekV2Tokenizer`

## 4. How to Run

### 4.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
cd /workspace/Megatron-LM
GEMM_TUNING=1 PR=bf16 MBS=4 AC=none bash examples/deepseek_v2/train_deepseekv2.sh
```

## 5. Key Variables to Pay Attention To

- **PR:**
  Stands for precision for training. `bf16` for Bf16 (default), `fp8` for FP8 GEMMS.

- **GEMM_TUNING:**
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **TRAIN_ITERS:**  
  Set the total number of iterations.

- **MOCK_DATA:**
  Use MOCK_DATA if set to 1, otherwise use the real data provided by user (DEFAULT: 1)

- **MBS:**
  Micro batch size

- **GBS:**
  Global Batch size

That's it! You'are now ready to train DeepSeek-V2-lite model.
