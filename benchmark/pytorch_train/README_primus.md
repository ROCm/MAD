# Training Performance Validation with AMD Pytorch Docker on the AMD Instinct Accelerators

## Overview

PyTorch is an open-source machine learning framework that is widely used for model training with GPU-optimized components for transformer-based models.

The ROCm PyTorch Training Docker `rocm/pytorch-training:v25.8` container, available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html), provides a prebuilt, optimized environment for fine-tuning, pre-training a model on the AMD Instinct™ MI300X and MI325X accelerator. This ROCm PyTorch Docker includes the following components:

| Software component | Version              |
|--------------------|----------------------|
| ROCm               | 7.0.0                |
| Python             | 3.10.12              |
| PyTorch            | 2.9.0.dev20250821+rocm7.0.0.lw.git125803b7   |
| Transformer Engine | 2.2.0.dev0+c3bcaab1  |
| Flash Attention    | 2.8.3                |
| hipBLASLt          | 1.1.0-911283acd1     |


## Models
Examples of the following models are pre-optimized for performance on the AMD Instinct MI300X and MI325X accelerator.
### Pre-training:
| Model          | Variants              |
|----------------|------------------------|
| **LLaMA 3.1**   | 8B, 70B         |


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

Use this command to run a performance benchmark test of the Llama 3.1 8B model on one GPU with float16 data type in the host machine.

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags pyt_train_llama-3.1-8b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-pyt_train_llama-3.1-8b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/perf.csv
```

#### Available models

| model_name                              |
| --------------------------------------- |
| primus_pyt_train_llama-3.1-8b                  |
| primus_pyt_train_llama-3.1-70b                 |

> ⚠️ **Note on pretraining with Primus Torchtitan**
>
> Currently, Primus torchtitan models are run with Primus-Turbo enabled for enhanced performance. 
> To disable Primus-Turbo please modify respective config file `scripts/primus/pytorch_train/primus_torchtitan_scripts/llama3_[8B|70B]-[BF16|FP8].yaml`.

To start the pretraining benchmark, use the following command.
<pre lang="markdown"> ./pytorch_benchmark_report.sh -t $training_mode -m $model_repo -p $datatype  </pre>

## Standalone benchmarking

### Download the Docker image and required packages
Use the following command to pull the Docker image from the Docker hub

```
docker pull rocm/pytorch-training:v25.8
```

Run the Docker container
```
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name training_env  rocm/pytorch-training:v25.9
```

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
Example 1: Llama 3.1 70B with BF16 precision with [Primus](https://github.com/AMD-AGI/Primus) [Torchtitan](https://github.com/ROCm/torchtitan)
Use this command to run a benchmark of the Llama 3.1 70B model.
```
EXP=examples/torchtitan/configs/llama3.1_70B-BF16-pretrain.yaml examples/run_pretrain.sh
```

Example 2: Llama 3.1 8B with BF16 precision with [Primus](https://github.com/AMD-AGI/Primus) [Torchtitan](https://github.com/ROCm/torchtitan)
```
EXP=examples/torchtitan/configs/llama3.1_8B-BF16-pretrain.yaml examples/run_pretrain.sh
```

Example 3: Llama 3.1 70B with FP8 precision with [Primus](https://github.com/AMD-AGI/Primus) [Torchtitan](https://github.com/ROCm/torchtitan)
```
EXP=examples/torchtitan/configs/llama3.1_70B-FP8-pretrain.yaml examples/run_pretrain.sh
```

Example 4: Llama 3.1 8B with FP8 precision with [Primus](https://github.com/AMD-AGI/Primus) [Torchtitan](https://github.com/ROCm/torchtitan)
```
EXP=examples/torchtitan/configs/llama3.1_8B-FP8-pretrain.yaml examples/run_pretrain.sh
```

#### MI300X Performance Configs
- **Llama3.1-70B BF16:**
```bash
EXP=examples/torchtitan/configs/llama3.1_70B-BF16-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 4
```
- **Llama3.1-8B BF16:**
```bash
EXP=examples/torchtitan/configs/llama3.1_8B-BF16-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 4
```
- **Llama3.1-70B FP8:**
```bash
EXP=examples/torchtitan/configs/llama3.1_70B-FP8-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 3 
```
- **Llama3.1-8B FP8:**
```bash
EXP=examples/torchtitan/configs/llama3.1_8B-FP8-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 5 
```
#### MI325X Performance Configs
- **Llama3.1-70B BF16:**
```bash
EXP=examples/torchtitan/configs/llama3.1_70B-BF16-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 6
```
- **Llama3.1-8B BF16:**
```bash
EXP=examples/torchtitan/configs/llama3.1_8B-BF16-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 6
```
- **Llama3.1-70B FP8:**
```bash
EXP=examples/torchtitan/configs/llama3.1_70B-FP8-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 5 
```
- **Llama3.1-8B FP8:**
```bash
EXP=examples/torchtitan/configs/llama3.1_8B-FP8-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 7 
```
#### MI35X Performance Configs
- **Llama3.1-70B BF16:**
```bash
EXP=examples/torchtitan/configs/llama3.1_70B-BF16-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 8
```
- **Llama3.1-8B BF16:**
```bash
EXP=examples/torchtitan/configs/llama3.1_8B-BF16-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 5
```
- **Llama3.1-70B FP8:**
```bash
EXP=examples/torchtitan/configs/llama3.1_70B-FP8-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 6 
```
- **Llama3.1-8B FP8:**
```bash
EXP=examples/torchtitan/configs/llama3.1_8B-FP8-pretrain.yaml \
bash examples/run_pretrain.sh \
    --metrics.enable_tensorboard false --profiling.enable_profiling false \
    --training.batch_size 8 
```

### Standalone torchtitan examples
Standalone [Torchtitan](https://github.com/ROCm/torchtitan) is available at `/workspace/torchtitan` in the docker image
Go to TorchTitan direcotry and download tokenizer

```
cd /workspace/torchtitan
python3 scripts/download_tokenizer.py \
    --repo_id meta-llama/Meta-Llama-3-8B \
    --tokenizer_path "original" \
    --hf_token=${HF_TOKEN}
```

Example 1: Llama 3.1 70B with BF16 precision with [Torchtitan](https://github.com/ROCm/torchtitan)
Use this command to run a benchmark of the Llama 3.1 70B model.
```
CONFIG_FILE="./llama3_70b_fsdp_bf16.toml" ./run_train.sh
```

Example 2: Llama 3.1 8B with BF16 precision with [Torchtitan](https://github.com/ROCm/torchtitan)
```
CONFIG_FILE="./llama3_8b_fsdp_bf16.toml" ./run_train.sh
```

Example 3: Llama 3.1 70B with FP8 precision with [Torchtitan](https://github.com/ROCm/torchtitan)
```
CONFIG_FILE="./llama3_70b_fsdp_fp8.toml" ./run_train.sh
```

Example 4: Llama 3.1 8B with FP8 precision with [Torchtitan](https://github.com/ROCm/torchtitan)
```
CONFIG_FILE="./llama3_8b_fsdp_fp8.toml" ./run_train.sh
```

