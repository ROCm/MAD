# Training Performance Validation with ROCm Megatron-LM Training Docker on the AMD Instinct Accelerators

**NOTE: ROCm Megatron-LM has limited support on the primus docker. Please follow the Primus with megatron-core framework guide to get started. We still maintain [backward compatibility with Rocm/Megatron-LM](https://github.com/ROCm/MAD/blob/develop/benchmark/megatron_lm/Migration_Guide.md#backward-compatibility-with-megatron-lm) framework for existing models.**

## Overview

ROCm Megatron-LM framework is a specialized fork of the robust Megatron-LM, designed to enable efficient training of large-scale language models on AMD GPUs. By leveraging AMD Instinct™ MI300X accelerators, AMD Megatron-LM delivers enhanced scalability, performance, and resource utilization for AI workloads. It is purpose-built to support models like Meta’s Llama 2, Llama 3, and Llama 3.1, enabling developers to train next-generation AI models with greater efficiency. See the GitHub repository at [ROCm/Megatron-LM](https://github.com/ROCm/Megatron-LM/).

For ease of use, AMD provides a ready-to-use Docker image for MI300X accelerators containing essential components, including PyTorch, PyTorch Lightning, ROCm libraries, and Megatron-LM utilities. It contains the following software to accelerate training workloads:

| Software component  | Version            |
|---------------------|--------------------|
| ROCm               | 6.4.3              |
| Python            | 3.10          |
| PyTorch           | 2.8.0a0+gitd06a406   |
| Transformer Engine | 2.2.0.dev0+54dd2bdc      |
| Flash Attention   | 3.0.0.post1               |
| hipBLASLt         | d1b517fc7a        |
| Triton            | 3.3.0                 |
| RCCL              | 2.22.3      |


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
The pre-built ROCm Megatron-LM environment allows users to quickly validate system performance, conduct training benchmarks, and achieve superior performance for models like Llama 2 and Llama 3.1.

This container should not be expected to provide generalized performance across all training workloads. Users should expect the container perform in the model configurations described below, but other configurations and run conditions are not validated by AMD. 
Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X accelerators with the AMD Megatron-LM Docker image.

---

# LLama Training Procedure

## 1. Environment Setup

1. **Download Docker Image**
   Download the Docker image required for training:
   ```bash
   docker pull rocm/megatron-lm:v25.8_py310
   ```

3. **Launch Docker Container**
   Start the Docker container:
   ```bash
   docker run -it --device /dev/dri --device /dev/kfd --device /dev/infiniband --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME --shm-size 128G --name megatron_training_env rocm/megatron-lm:v25.8_py310
   ```

5. **Execute the training_env container (optional if no already in the container)**
   ```bash
    docker start megatron_training_env
    docker exec -it megatron_training_env bash
   ```
   
6. **Megatron-LM Backward Compatibility setup:**
   Primus docker maintains Megatron-LM compatibility with limited support. To roll-back using Megatron-LM follow the steps outlined below. Once Megatron-LM is installed, follow the documentation to run workloads as usual.
```bash
cd /workspace/Megatron-LM/
pip uninstall megatron-core
pip install -e .
```
The docker container hosts verified commit `e8e9edc` from [Megatron-LM repository](https://github.com/ROCm/Megatron-LM/tree/rocm_dev).

---

## 2. Configurations in Script (`Megatron-LM/examples/llama`)
Use `train_llama3.sh` for Llama3/3.1 models and `train_llama2.sh` for Llama2 models.

### 2.1 Network Interface
Update the network interface in the script to match your system’s network interface.
To find your network interface, run (out of container):
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
  Use `MOCK_DATA` variable to toggle between mock and real data. Default value is 1.
  ```bash
  MOCK_DATA=1
  ```
- **Real Data:**
  Set `MOCK_DATA` to `0` and update the `DATA_PATH` or `DATA_DIR` variable as described for each model below.
  ```bash
  MOCK_DATA=0
  ```
- **Downloading the dataset for Llama:**
  Set argument `DATASET` to the dataset you would like to use. Currently, three datasets are supported `DATASET=wiki`, `DATASET=fineweb`, and `DATASET=bookcorpus`. Use the following command to download the dataset:
  ```bash
  DATASET=wiki TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf bash examples/llama/prepare_dataset.sh #for wiki-en dataset
  DATASET=bookcorpus TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf bash examples/llama/prepare_dataset.sh #for bookcorpus dataset
  ```
  where `TOKENIZER_MODEL` can be any accessible HuggingFace tokenizer. Remember to either pre-download the tokenizer or setup HuggingFace access otherwise when needed.
  
  Note: when training you need to set `DATA_PATH` to the specific file name prefix that is pointing to .bin or .idx file as shown below
  ```bash
  DATA_PATH="data/bookcorpus_text_sentence" # Change to where your dataset is stored.
  ```
- **Downloading the dataset for DeepSeekV2:**
  ```bash
  mkdir deepseek-datasets
  cd deepseek-datasets
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx
  ```
  Set `DATA_DIR` to `path-to/deepseek-datasets/` for training on real data.
  
- **Downloading the dataset for DeepSeekV3:**
  ```bash
  mkdir deepseek-datasets
  cd deepseek-datasets
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
  cd ..
  bash tools/run_make_pretraining_dataset_megatron.sh deepseek-datasets/SlimPajama.json DeepSeekV3Tokenizer text deepseek-datasets deepseek-ai/DeepSeek-V3
  ```
  Set `DATA_DIR` to `path-to/deepseek-datasets/` for training on real data.
  
- **Downloading the dataset for Mixtral 8x7B and 8X22B:**
  ```bash
  mkdir -p mixtral-dataset
  cd dataset
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx
  ```
  Set `DATA_DIR` to `/path/to/mixtral-dataset` for training on real data.
  
- **Downloading dataset for Qwen2.5 7/72B:**
  ```bash
  mkdir -p temp/qwen-datasets
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin
  wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx
  ```
  Set `DATA_DIR` to `/path/to/qwen-dataset` for training on real data.
  
### 2.3 Tokenizer
You can assign the path of existing tokenizer to the command line argument `TOKENIZER_MODEL`. If tokenizer is not found, it will be downloaded if publicly available. 

- **For Llama2 Training:**
  Uses either the `Llama2Tokenizer` or `HuggingFaceTokenizer`(default).

- **For Llama3.1 Training:**
  Use the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=meta-llama/Llama-3.1-8B  # For Llama3
  ```
- **For Llama3.3 Training:**
  If you do not have Llama3.3 tokenizer locally, you need to use your personal HuggingFace access token `HF_TOKEN` to download the tokenizer via this [link](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct). After you are authorized, can use your personal HuggingFace access token `HF_TOKEN` to download tokenizer and set the variable `TOKENIZER_MODEL` to the tokenizer path.
  ```bash
  TOKENIZER_MODEL="meta-llama/Llama-3.3-70B-Instruct"
  ```
- **For DeepSeekV2-Lite:**
  Use the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=deepseek-ai/DeepSeek-V2-Lite  # For DeepSeekV2-Lite
  ```
- **For DeepSeekV3:**
  Use the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=deepseek-ai/DeepSeek-V3  # For DeepSeekV3
  ```
- **For Mixtral MoE:**
  Download Mixtral Tokenizer
  ```bash
  mkdir tokenizer
  cd tokenizer
  export HF_TOKEN="hf_xxx" #set huggingface access token to be able to download tokenizer
  wget --header="Authorization: Bearer $HF_TOKEN" -O ./tokenizer.model https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/resolve/main/tokenizer.model
  cd ..
  ```
   Use the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=tokenizer/tokenizer.model
  ```
- **For Qwen2.5 Training:**
  Uses the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=Qwen/Qwen2.5-7B or Qwen/Qwen2.5-72B # For Qwen2.5
  ```
### 2.4 Multi-node Training
If you're running multi-node training, update the following environment variables on each node.They can also be passed as command line arguments.

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
   For multi-node runs, make sure correct network drivers are installed on the nodes. If inside a docker, either install the drivers inside the docker container or pass the network drivers from the host while creating docker container.

   ```bash
   # specify which RDMA interfaces to use for communication
   export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
   ```

---

## 3. How to Run

### 3.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
- **Llama3.1-8B FP8:**
```bash
TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh
```

- **Llama3.1-8B BF16:**
```bash
TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=0 SEQ_LENGTH=8192 MODEL_SIZE=8 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh
```

- **Llama2-7B FP8:**
```bash
TEE_OUTPUT=1 MBS=4 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=4096 MODEL_SIZE=7 TOTAL_ITERS=50 bash examples/llama/train_llama2.sh
```

- **Llama2-7B BF16:**
```bash
TEE_OUTPUT=1 MBS=4 BS=256 TP=1 TE_FP8=0 SEQ_LENGTH=4096 MODEL_SIZE=7 TOTAL_ITERS=50 bash examples/llama/train_llama2.sh
```

To run the training with `FSDP-v2` enabled, simply add `FSDP=1` argument, for example, use the following command:

- **Llama3-70B BF16:**
```bash
CKPT_FORMAT=torch_dist TEE_OUTPUT=1 MBS=3 BS=24 TP=1 TE_FP8=0 FSDP=1 RECOMPUTE=1 SEQ_LENGTH=8192 MODEL_SIZE=70 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh
```

- **Llama3-70B FP8 Proxy model on Single Node**
```bash
CKPT_FORMAT=torch_dist TEE_OUTPUT=1 RECOMPUTE=1 MBS=3 BS=24 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=70 FSDP=1 TOTAL_ITERS=10 NUM_LAYERS=40 bash examples/llama/train_llama3.sh
```
**Note:**
   - Please use >=2 nodes to run full llama 70B model with fp8 precision.

- **Llama2-70B BF16:**
```bash
CKPT_FORMAT=torch_dist TEE_OUTPUT=1 MBS=7 BS=56 TP=1 TE_FP8=0 FSDP=1 RECOMPUTE=1 SEQ_LENGTH=4096 MODEL_SIZE=70 TOTAL_ITERS=50 bash examples/llama/train_llama2.sh
```

- **Llama3.3-70B BF16:**
```bash
TOKENIZER_MODEL=meta-llama/Llama-3.3-70B-Instruct CKPT_FORMAT=torch_dist TEE_OUTPUT=1 RECOMPUTE=1 SEQ_LENGTH=8192 MBS=2 BS=16 TE_FP8=0 TP=1 PP=1 FSDP=1 MODEL_SIZE=70 TOTAL_ITERS=50 bash examples/llama/train_llama3.sh 
```
**Note:** 
   - It is suggested to use `TP=1` when FSDP is enabled, for higher throughput. And FSDP-v2 is not supported with pipeline parallelism, expert parallelism, MCore's distributed optimizer, gradient accumulation fusion and fp16.

Examples for MoE models with expert parallel:
- **DeepSeekV2-Lite**

**Note:** Please note DeepSeekV2-Lite is showing instability as GPU memory access fault for large iteration. Please use this workload through Primus framework for stability.

```bash
export NVTE_FUSED_ATTN_CK=0
GEMM_TUNING=1 PR=bf16 MBS=4 AC=none SEQ_LEN=4096 PAD_LEN=4096 TRAIN_ITERS=20 bash examples/deepseek_v2/train_deepseekv2.sh
```
     
- **DeepSeekV3 3 layer proxy on Single Node**
```bash
export NVTE_FUSED_ATTN_CK=0
FORCE_BANLANCE=true \
RUN_ENV=cluster \
MODEL_SIZE=671B \
TRAIN_ITERS=50 \
SEQ_LEN=4096 \
NUM_LAYERS=3 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32 \
PR=bf16 \
TP=1 PP=1 ETP=1 EP=8 \
GEMM_TUNING=1 \
NVTE_CK_USES_BWD_V3=1 \
USE_GROUPED_GEMM=true MOE_USE_LEGACY_GROUPED_GEMM=true \
GPT_LAYER_IN_TE=true \
bash examples/deepseek_v3/train_deepseekv3.sh
```
- **Mixtral 8x7B**
```bash
TOKENIZER_MODEL=<path/to/tokenizer.model> RECOMPUTE_NUM_LAYERS=0 TEE_OUTPUT=1 MBS=1 GBS=16 TP_SIZE=1 PP_SIZE=1 AC=none PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=4096 FORCE_BALANCE=true MOCK_DATA=1 RUN_ENV=cluster MODEL_SIZE=8x7B TRAIN_ITERS=50 bash examples/mixtral/train_mixtral_moe.sh
```
- **Mixtral 8x22B 4 layer proxy on Single Node**
```bash
TOKENIZER_MODEL=<path/to/tokenizer.model> RECOMPUTE_NUM_LAYERS=4 TEE_OUTPUT=1 MBS=1 GBS=16 TP_SIZE=1 PP_SIZE=1 AC=full NUM_LAYERS=4 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=8192 FORCE_BALANCE=true MOCK_DATA=1 RUN_ENV=cluster MODEL_SIZE=8x22B TRAIN_ITERS=50 bash examples/mixtral/train_mixtral_moe.sh
```

- **QWEN2.5 7B - BF16**
  ```bash
  bash examples/qwen/train_qwen2.sh TP=1 CP=1 PP=1 MBS=10 BS=640 TE_FP8=0 MODEL_SIZE=7 SEQ_LENGTH=2048 TOTAL_ITERS=50 MOCK_DATA=1 TOKENIZER_MODEL=Qwen/Qwen2.5-7B
  ```
- **QWEN2.5 7B - FP8**
  ```bash
  bash examples/qwen/train_qwen2.sh TP=1 CP=1 PP=1 MBS=10 BS=640 TE_FP8=1 MODEL_SIZE=7 SEQ_LENGTH=2048 TOTAL_ITERS=50 MOCK_DATA=1 TOKENIZER_MODEL=Qwen/Qwen2.5-7B
  ```
- **QWEN2.5 72B - BF16**
  ```bash
  bash examples/qwen/train_qwen2.sh FSDP=1 CP=1 PP=1 MBS=3 BS=24 TE_FP8=0 MODEL_SIZE=72 SEQ_LENGTH=2048 TOTAL_ITERS=50 MOCK_DATA=1 TOKENIZER_MODEL=Qwen/Qwen2.5-72B RECOMPUTE_ACTIVATIONS=full CKPT_FORMAT=torch_dist
  ```
  
### 3.2 Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. Example, follow these steps for 2 Node run with Node0 as master node :

- **On the Master Node0:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=0 bash examples/llama/train_llama3.sh
  ```

- **On the Worker Node1:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=1 bash examples/llama/train_llama3.sh
  ```
- **DeepSeekV3 Multi-node Reference**

  We provide an example scipt to enable training at scale under slurm environment. For example, to run the training on 16 nodes, one can use the following command
  ```bash
  sbatch examples/deepseek_v3/train_deepseek_v3_slurm.sh
  ```
---

## 4. Key Variables to Pay Attention To

- **TE_FP8:**  
  `0` for B16 (default), `1` for FP8.

- **GEMM_TUNING:**  
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **USE_FLASH_ATTN:**  
  `1` to enable Flash Attention.

- **FSDP:**  
  `1` to enable torch fsdp-v2. 
  
  Note that if FSDP is enabled, `--use-distributed-optimizer`, `--overlap-param-gather`, `--sequence-parallel` will be automatically set off. 

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

- **TP/TP_SIZE:**
  Tensor parallel (1, 2, 4, 8)

  Note `TP` is disabled with `FSDP`.

- **EP/EP_SIZE:**
  Expert parallel for MoE models
  
- **SEQ_LENGTH**:
  Sequence Length

- **PR:**
  Stands for precision for training. `bf16` for Bf16 (default), `fp8` for FP8 GEMMS.

- **AC:**
  Activation Checkpointing (`none`, `sel` , `full`). Default:`sel` (Selective).

- **NUM_LAYERS:**
  Using reduced number of layers as a proxy model

- **RECOMPUTE_NUM_LAYERS:**
  Number of layers used for checkpointing recompute

--- 
That's it! You've now set up the environment and configured the necessary settings for training Llama2, Llama3 or DeepSeek models.
