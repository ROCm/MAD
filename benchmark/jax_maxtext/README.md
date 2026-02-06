# Training Performance Validation with ROCm Maxtext-jax Training Docker on the AMD Instinct Accelerators

## Overview

MaxText framework for ROCm is a specialized fork from upstream MaxText, designed to enable training of large language model (LLM) on AMD GPUs. By leveraging AMD Instinct™ MI300X and MI355X GPUs, MaxText delivers great scalability, performance, and resource utilization for AI workload. See the GitHub repository at [ROCm/maxtext](https://github.com/ROCm/maxtext/).

AMD provides a ready-to-use Docker image for AMD Instinct MI300X and MI355X GPUs containing essential components, including Jax, XLA, ROCm libraries, and MaxText utilities. It contains the following software components to accelerate training workloads:

>[!NOTE]
>Shardy is a new config in JAX 0.6.0. You might get related errors if it's not configured correctly. For now you can turn it off by setting `shardy=False` during the training run. You can also follow the [migration guide](https://docs.jax.dev/en/latest/shardy_jax_migration.html) to enable it.
>

>[!NOTE]
>There are known issues of rocm/jax-training:maxtext-v26.1.
> 1. It is known that you may see NaNs in the losses when setting `packing=True`. As a workaround, we suggest turning off input sequence packing (`packing=False`). We are working to fix this.
> 2. Docker `rocm/jax-training:maxtext-v26.1` will not have the [Primus](https://github.com/AMD-AGI/Primus/tree/main) repo. It is planned to be supported in the next release.

| Software component | Version        |
|--------------------|----------------|
| ROCm               | 7.1.1         |
| Jax                | 0.8.2          |
| Python             | 3.12.3        |
| Transformer Engine | 2.8.0.dev0+aec00a7f |
| hipBLASLt          | 1.2.x          |


## Supported features and models
MaxText supports the following key features to train large language models efficiently:

* Transformer Engine (TE)
* Flash Attention (FA) 3, with or without input sequence packing
* GEMM tuning
* Multi-node Support
* NANOO FP8 (for MI300X) or FP8 (for MI355X)

The following models are pre-optimized for performance on the AMD Instinct MI300X and MI355X accelerator.

* Llama 2 7B
* Llama 2 70B
* Llama 3/3.1 8B
* Llama 3/3.1 70B
* Llama 3.1 405B
* Llama 3.3 70B
* DeepSeek-V2-lite (16B)
* Mixtral-8x7B

Note: Some models, such as Llama 3, require an external license agreement through a third party (for example, Meta).


## System validation
If you have already validated your system, skip this step. Otherwise, please complete the following [system validation and optimization steps](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/prerequisite-system-validation.html#train-a-model-system-validation) to set up your system before starting training.


## Environment setup
This Docker image is optimized for specific model configurations outlined below. Performance can vary for other training workloads, as AMD doesn’t validate configurations and run conditions outside those described.

For multinode, we need to make sure we have all the packages installed based on the network device we use. You can check multi node examples on how to install these packages before running the workload. You need to only do the set up below if you are using multinode with RDMA, otherwise skip this part.

Install the packages below for building and installing the RDMA driver:
```bash
apt install iproute2 -y
apt install -y linux-headers-"$(uname -r)" libelf-dev
apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev
```
Please refer to your NIC manufacturer's webpage for further steps about compiling and install the RoCE driver. .e.g. for Broadcom, please refer to the section **Compiling Broadcom NIC Software from Source** in [Ethernet Networking Guide for AMD Instinct MI300X GPU Clusters](https://docs.broadcom.com/doc/957608-AN2XX)

Set the following env variables. You can again check the multinode examples on how to set these variables.
- **Master Address:**
  Change `localhost` to the master node's hostname:
  ```bash
  export MASTER_ADDR="${MASTER_ADDR:-localhost}"
  ```

- **Number of Nodes:**
  Set the number of nodes you want to train on (e.g., 2, 4, 8):
  ```bash
  export NNODES="${NNODES:-1}"
  ```

- **Node Rank:**
  Set the rank of each node (0 for master, 1 for the first worker node, etc.):
  ```bash
  export NODE_RANK="${NODE_RANK:-0}"
  ```
- **Network Interface**
  Update the network interface in the script to match your system’s network interface.
  To find your network interface, run (out of container):
  ```bash
  ip a
  ```
  Then, update the following variables in the script:
  ```bash
  export NCCL_SOCKET_IFNAME=ens50f0np0
  ```
 - **RDMA Interface**
   First make sure that packages above are installed on all the nodes. Then set the RDMA interfaces to use for communication.
   ```bash
   # If using Broadcom NIC
   export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
   # If using Mellanox NIC
   export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
   ```
>[!NOTE]
>The only models supported in this workflow are those listed in the above section.
>

This container should not be expected to provide generalized performance across all training workloads. Users should expect the container perform in the model configurations described below, but other configurations and run conditions are not validated by AMD.
Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X, MI325X, MI350X, MI355X accelerators with the Docker image.

Use the following instructions to reproduce the benchmark results on an
MI300X or MI355X accelerator with a prebuilt JAX Docker image.

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

Run models through MAD-integrated benchmarking with the following command

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags <mad_model> --keep-model-dir --live-output --timeout 28800
```

For example, use this command to run a performance benchmark test of the Llama 2 7B model on one GPU with bf16 data type in the host machine.

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags jax_maxtext_train_llama-2-7b --keep-model-dir --live-output --timeout 28800
```

>[!NOTE]
>The madengine package is now available allowing for the replacement of run_models.py.
>
```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 madengine run --tags jax_maxtext_train_llama-2-7b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-jax_maxtext_train_llama-2-7b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/perf.csv
```

#### Available models

| model_name                              |
| --------------------------------------- |
| jax_maxtext_train_llama-2-7b            |
| jax_maxtext_train_llama-2-70b           |
| jax_maxtext_train_llama-3.1-8b          |
| jax_maxtext_train_llama-3.1-70b         |
| jax_maxtext_train_llama-3.1-405b        |
| jax_maxtext_train_llama-3.3-70b         |
| jax_maxtext_train_deepseek-v2-lite-16b  |
| jax_maxtext_train_mixtral-8x7b          |

## Standalone benchmarking

Download and launch the Docker image

Use the following command to pull the Docker image from Docker Hub.

```
docker pull rocm/jax-training:maxtext-v26.1
```
### Single Node Training examples

#### Setup
>[!NOTE]
>Please adjust the following variables based on your environment.
>

Export variables
- MAD_SECRETS_HFTOKEN is your HuggingFace token to access models, tokenizers, data. See this [page](https://huggingface.co/docs/hub/en/security-tokens) for more info.
- HF_HOME is where huggingface_hub will store local data, please refer to [Huggingface cli Document](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#hf-download) on how to download the data. If you already have downloaded/cached huggingface artifacts, set this variable to that path. Downloaded files typically get cached to a place like this: `~/.cache/huggingface`.
```
export MAD_SECRETS_HFTOKEN=<Your HuggingFace token>
export HF_HOME=<Location of saved/cached HuggingFace models>
```

Launch the Docker container.

```
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v $HOME/.ssh:/root/.ssh -v $HF_HOME:/hf_cache -e HF_HOME=/hf_cache -e MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN --shm-size 64G --name training_env rocm/jax-training:maxtext-v26.1
```

Execute the training_env container (optional if not already in the container)
```
docker start maxtext_training
docker exec -it maxtext_training bash
```

Clone Model Automation and Dashboarding (MAD) repo
```
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/jax-maxtext
```

Run setup scripts to install libraries and datasets needed for benchmarking
```
./jax-maxtext_benchmark_setup.sh -m <model>
```

Run the benchmark in quantized or unquantized mode.

```
# For unquantized training
./jax-maxtext_benchmark_report.sh -m <model>

# Or for quantized training
./jax-maxtext_benchmark_report.sh -m <model> -q nanoo_fp8
```

The performance results should be written to a file in the parent folder.

### Benchmarking examples

#### Example commands
1.	**Single-node training with Llama 2 7B model**

Setup
```
./jax-maxtext_benchmark_setup.sh -m Llama-2-7B
```

For unquantized training
```
./jax-maxtext_benchmark_report.sh -m Llama-2-7B
```

Or for nanoo_fp8 quantized training on MI300X
```
./jax-maxtext_benchmark_report.sh -m Llama-2-7B -q nanoo_fp8
```

Or for fp8 quantized training on MI355X
```
./jax-maxtext_benchmark_report.sh -m Llama-2-7B -q fp8
```

2.	**Single-node training with Llama 2 70B model**

Setup
```
./jax-maxtext_benchmark_setup.sh -m Llama-2-70B
```

For unquantized training
```
./jax-maxtext_benchmark_report.sh -m Llama-2-70B
```

Or for nanoo_fp8 quantized training on MI300X
```
./jax-maxtext_benchmark_report.sh -m Llama-2-70B -q nanoo_fp8
```

Or for fp8 quantized training on MI355X
```
./jax-maxtext_benchmark_report.sh -m Llama-2-70B -q fp8
```

3.	**Single-node training with Llama 3.1 8B model**

Setup
```
./jax-maxtext_benchmark_setup.sh -m Llama-3.1-8B
```

For unquantized training
```
./jax-maxtext_benchmark_report.sh -m Llama-3.1-8B
```

Or for nanoo_fp8 quantized training on MI300X
```
./jax-maxtext_benchmark_report.sh -m Llama-3.1-8B -q nanoo_fp8
```

Or for fp8 quantized training on MI355X
```
./jax-maxtext_benchmark_report.sh -m Llama-3.1-8B -q fp8
```

4.	**Single-node training with Llama 3.1 70B model**

Setup
```
./jax-maxtext_benchmark_setup.sh -m Llama-3.1-70B
```

For unquantized training
```
./jax-maxtext_benchmark_report.sh -m Llama-3.1-70B
```

Or for fp8 quantized training on MI355X
```
./jax-maxtext_benchmark_report.sh -m Llama-3.1-70B -q fp8
```

5.	**Single-node training with Llama 3.3 70B model**

Setup
```
./jax-maxtext_benchmark_setup.sh -m Llama-3.3-70B
```

For unquantized training
```
./jax-maxtext_benchmark_report.sh -m Llama-3.3-70B
```

Or for fp8 quantized training on MI355X
```
./jax-maxtext_benchmark_report.sh -m Llama-3.3-70B -q fp8
```

6.	**Single-node training with DeepSeek2 16B model**

Setup
```
./jax-maxtext_benchmark_setup.sh -m DeepSeek-V2-lite
```

For unquantized training
```
./jax-maxtext_benchmark_report.sh -m DeepSeek-V2-lite
```

Or for nanoo_fp8 quantized training on MI300X
```
./jax-maxtext_benchmark_report.sh -m DeepSeek-V2-lite -q nanoo_fp8
```

Or for fp8 quantized training on MI355X
```
./jax-maxtext_benchmark_report.sh -m DeepSeek-V2-lite -q fp8
```

7.	**Single-node training with Mixtral-8x7B model**

Setup
```
./jax-maxtext_benchmark_setup.sh -m Mixtral-8x7B
```

For unquantized training
```
./jax-maxtext_benchmark_report.sh -m Mixtral-8x7B
```

Or for nanoo_fp8 quantized training on MI300X
```
./jax-maxtext_benchmark_report.sh -m Mixtral-8x7B -q nanoo_fp8
```

Or for fp8 quantized training on MI355X
```
./jax-maxtext_benchmark_report.sh -m Mixtral-8x7B -q fp8
```



### Multi-Node Training examples
Note: these scripts will launch the docker and execute the benchmark, so **please run them outside of any docker**.

The examples below use Slurm for running on multiple nodes. The unified multinode benchmark script accepts a configuration file that specifies the model and training parameters.

#### Running Multi-Node Training

To run multi-node training, use the following command:

```bash
sbatch -N <NUM_NODES> jax_maxtext_multinode_benchmark.sh <config_file.yml> [docker_image]
```

**Parameters:**
- `<NUM_NODES>`: Number of nodes to use for training (e.g., 2, 4, 8)
- `<config_file.yml>`: Path to the YAML configuration file containing model and training parameters
- `[docker_image]`: (Optional) Docker image to use. If not specified, defaults to `rocm/jax-training:maxtext-v26.1`

**Configuration files** are available in the `scripts/jax-maxtext/env_scripts/` directory for different models and GPU architectures:

For MI300X (gfx942):
- `llama2_7b.yml` - Llama 2 7B
- `llama2_70b.yml` - Llama 2 70B
- `llama3_8b.yml` - Llama 3 8B
- `llama3_70b.yml` - Llama 3 70B

For MI355X (gfx950):
- `gfx950_llama2_7b.yml` - Llama 2 7B
- `gfx950_llama2_70b.yml` - Llama 2 70B
- `gfx950_llama3_8b.yml` - Llama 3 8B
- `gfx950_llama3_70b.yml` - Llama 3 70B
- `gfx950_llama3.1_405b.yml` - Llama 3.1 405B

#### Example Commands

1. **Multi-node training with Llama 2 7B model on 2 nodes:**
```bash
sbatch -N 2 jax_maxtext_multinode_benchmark.sh env_scripts/llama2_7b.yml
```

2. **Multi-node training with Llama 2 70B model on 4 nodes with custom image:**
```bash
sbatch -N 4 jax_maxtext_multinode_benchmark.sh env_scripts/llama2_70b.yml rocm/jax-training:maxtext-v26.1
```

3. **Multi-node training with Llama 3 8B model on 2 nodes:**
```bash
sbatch -N 2 jax_maxtext_multinode_benchmark.sh env_scripts/llama3_8b.yml
```

4. **Multi-node training with Llama 3 70B model on 8 nodes:**
```bash
sbatch -N 8 jax_maxtext_multinode_benchmark.sh env_scripts/llama3_70b.yml
```

5. **Multi-node training with Llama 3.1 405B model on MI355X (gfx950) with 8 nodes:**
```bash
sbatch -N 8 jax_maxtext_multinode_benchmark.sh env_scripts/gfx950_llama3.1_405b.yml
```

## Profiling with rocprofv3

If you need to collect a trace and the JAX profiler isn't working then you can use rocprofv3 as a temporary workaround like this:
```
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --rccl-trace --output-format pftrace -d ./v3_traces -- python3 app.py
```
- Just replace `python3 app.py` with any command line command that you want to run such as `./jax-maxtext_benchmark_report.sh -m Llama-2-7B`.
- You can set the directory where you want the .json traces to be saved using `-d <TRACE_DIRECTORY>`
- The resulting traces can be opened in perfetto: https://ui.perfetto.dev/