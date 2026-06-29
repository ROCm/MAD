# Training Performance Validation with AMD Pytorch Docker on the AMD Instinct Accelerators

## Overview

PyTorch is an open-source machine learning framework that is widely used for model training with GPU-optimized components for transformer-based models.

The ROCm PyTorch Training Docker `rocm/primus:v26.4` container, available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html), provides a prebuilt, optimized environment for fine-tuning, pre-training a model on the AMD Instinct™ MI300X and MI325X accelerator. This ROCm PyTorch Docker includes the following components:

| Software component | Version              |
|--------------------|----------------------|
| ROCm               | 7.14.0a20260608      |
| Python             | 3.12.3               |
| PyTorch            | 2.12.0+git7e98855    |
| Transformer Engine | 2.14.0.dev0+e6ede467 |
| Flash Attention    | 2.8.3                |
| hipBLASLt          | 1.4.0-c2fafc16       |
| Triton             | 3.7.0+gitb4e20bbe    |
| RCCL               | 2.28.9               |


## Models
Examples of the following models are pre-optimized for performance on the AMD Instinct MI300X and MI325X accelerator.
### Pre-training:
| Model          | Variants               |
|----------------|------------------------|
| **LLaMA 3.1**  | 8B, 70B, 405B          |
| **DeepSeek V3**| 16B                    |

Please note that some models, such as Llama 3, require an external license agreement through a third party (e.g. Meta).

## System validation steps
If you have already validated your system, skip this step; otherwise, please complete the following [system validation and optimization steps](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/prerequisite-system-validation.html) to set up your system before starting training.

### Disable NUMA auto-balancing
Generally, application performance can benefit from disabling NUMA auto-balancing. However, it might be detrimental to performance with certain types of workloads.

Run the command `cat /proc/sys/kernel/numa_balancing` to check your current NUMA (Non-Uniform Memory Access) settings. Output `0` indicates this setting is disabled. If there is no output or the output is `1`, run the following command to disable NUMA auto-balancing.

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```
See [Disable NUMA auto-balancing](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-disable-numa) for more information.


## Start training on AMD Instinct accelerators

>[!NOTE]
>The only models supported in this workflow are those listed in the above section.
>

This container should not be expected to provide generalized performance across all training workloads. Users should expect the container perform in the model configurations described below, but other configurations and run conditions are not validated by AMD.
Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X, MI325X, MI350X and MI355X accelerators with the Docker image.

Use the following instructions to reproduce the benchmark results on an
MI300X accelerator with a prebuilt Pytorch Docker image. For best performance on MI325X, MI350X and MI355X, user needs adjust configurations (e.g. batch sizes) accordingly.

Users have two choices to reproduce the benchmark results.

-   [MAD-integrated benchmarking](#mad-integrated-benchmarking)
-   [Standalone benchmarking](#standalone-benchmarking)

## MAD-integrated benchmarking

Clone the ROCm Model Automation and Dashboarding (MAD) repository to a local directory and install the required packages on the host machine.

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt
```

Use this command to run a performance benchmark test of the Llama 3.1 8B model through Primus on one GPU with float16 data type in the host machine.

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags primus_pyt_train_llama-3.1-8b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-primus_pyt_train_llama-3.1-8b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/perf.csv
```

#### Available models

| model_name                              |
| --------------------------------------- |
| primus_pyt_train_llama-3.1-8b                  |
| primus_pyt_train_llama-3.1-70b                 |
| primus_pyt_train_deepseek-v3-16b                 |


To start the pretraining benchmark, use the following command.
<pre lang="markdown"> ./pytorch_benchmark_report.sh -t $training_mode -m $model_repo -p $datatype  </pre>

## Standalone benchmarking

### Download the Docker image and required packages
Use the following command to pull the Docker image from the Docker hub

```
docker pull rocm/primus:v26.4
```

Run the Docker container
```
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:/userHome -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name training_env  rocm/primus:v26.4
```
**Note**: It's not recommended to bind the `$HOME` directory to the container using `-v $HOME:$HOME`. A good practice is only binding the directory you need to the container.

Execute the training_env container (optional if no already in the container)
```
docker start training_env
docker exec -it training_env bash
```

### Prepare training datasets and dependency
The following benchmarking examples may require downloading models and datasets from Hugging Face. To ensure successful access to gated repos, please set your `HF_TOKEN`
```
# pass your HF_TOKEN
export HF_TOKEN=$your_personal_hf_token
```

### Benchmarking Command
#### Pretraining
To start the pretraining benchmark, use the following command with the appropriate options. See the list of options and their descriptions below.

#### Primus with Torchtitan backend
Primus is available at `/workspace/Primus` directory

### Benchmarking examples
Go to Primus directory
```
cd /workspace/Primus
```

#### MI300X Performance Configs

```bash
#Set these variables only on MI300/MI325X
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
```

- **Llama3.1-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml
```
- **Llama3.1-8B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```
- **DeepSeek-V3-16b BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_deepseek_v3_16b.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/deepseek_v3_16b-pretrain.yaml
```
- **Llama3.1-70B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_70B-FP8-pretrain.yaml
```
- **Llama3.1-8B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml
```


#### MI35X Performance Configs
- **Llama3.1-70B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml
```
- **Llama3.1-8B BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```
- **DeepSeek-V3-16b BF16:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_deepseek_v3_16b.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/deepseek_v3_16b-pretrain.yaml
```
- **Llama3.1-70B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_70B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml
```
- **Llama3.1-8B FP8:**
```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_8B-FP8-pretrain.yaml
```

### Multi-node Training

Multi-node training using torchtitan is similar to megatron-LM. Refer [megatron multi-node training](../megatron_lm/README.md#32-multi-node-training) for how to set the environment variables.

Here are two examples for multinode training on MI355X.


- **Llama3.1 70B FP8 4 Nodes MI355**

Launch the training using the `primus-cli` (recommended)
```bash
# In the Primus directory
./primus-cli slurm srun -N 4 -- train pretrain --config examples/megatron/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml --training.local_batch_size 6 --training.global_batch_size 192 --training.mock_data True
```

Launch the training using the legacy script
```bash
NNODES=4 EXP=examples/megatron/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --training.local_batch_size 6 --training.global_batch_size 192 --training.mock_data True
```

- **Llama3.1-405B FP8 8 Nodes MI355**

Launch the training using the `primus-cli` (recommended)
```bash
# In the Primus directory
./primus-cli slurm srun -N 8 -- train pretrain --config examples/torchtitan/configs/MI355X/llama3.1_405B-FP8-pretrain.yaml --training.local_batch_size 3 --training.global_batch_size 192 --training.mock_data True
```

Launch the training using the legacy script
```bash
NNODES=8 EXP=examples/torchtitan/configs/MI355X/llama3.1_405B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --training.local_batch_size 3 --training.global_batch_size 192 --training.mock_data True
```

