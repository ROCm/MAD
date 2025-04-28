# Training Performance Validation with ROCm Maxtext-jax Training Docker on the AMD Instinct Accelerators

## Overview

MaxText framework for ROCm is a specialized fork from upstream MaxText, designed to enable training of large language model (LLM) on AMD GPUs. By leveraging AMD Instinct™ MI300X GPUs, MaxText delivers great scalability, performance, and resource utilization for AI workload. See the GitHub repository at [ROCm/maxtext](https://github.com/ROCm/maxtext/).

AMD provides a ready-to-use Docker image for AMD Instinct MI300X GPUs containing essential components, including Jax, XLA, ROCm libraries, and MaxText utilities. It contains the following software components to accelerate training workloads:

| Software component  | Version            |
|---------------------|--------------------|
| ROCm               | 6.3.4              |
| Jax            | 0.4.35               |
| Python            | 3.10.12               |
| Transformer Engine | 1.12.0.dev0+b8b92dc     |
| hipBLASLt         | 0.13.0-ae9c477a         |


## Supported features and models
MaxText supports the following key features to train large language models efficiently:

* Transformer Engine (TE)
* Flash Attention (FA) 3
* GEMM tuning
* Multi-node Support

The following models are pre-optimized for performance on the AMD Instinct MI300X accelerator.

* Llama 2 7B
* Llama 2 70B
* Llama 3/3.1 8B
* Llama 3/3.1 70B
* Llama 3.3 70B
* DeepSeek-V2-lite (16B) 

Note: Some models, such as Llama 3, require an external license agreement through a third party (for example, Meta).

### Not supported future:
By default, Maxtext would try to use packed input format for training, which is not supported yet. For the current docker, using packed input format would leads to attention computation for tokens across different inputs. We are planning to add support for packed input format in our next release.

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

### Download and launch the Docker image

1.	Use the following command to pull the Docker image from Docker Hub.
```
docker pull rocm/jax-training:maxtext-v25.5
```

2.	Launch the Docker container.
```
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged    -v  $HOME/.ssh:/root/.ssh  --shm-size 128G --name maxtext_training rocm/jax-training:maxtext-v25.5
```


### Single Node Training examples
Note: these scripts will launch the docker and execute the benchmark, so **please run it outside of any docker**. 

Please make sure $HF_HOME is set before running the test. Refer to this [Readme](https://github.com/ROCm/maxtext/blob/main/benchmarks/gpu-rocm/readme.md) for more details on downloading the llama models before running the benchmark.

1.	**Single-node training with Llama 2 7B model**\
Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama2_7b.sh
```

Run the benchmark for single node traininig
```
IMAGE="rocm/jax-training:maxtext-v25.5" bash ./llama2_7b.sh
```
2.	**Single-node training with Llama 2 70B model**\
Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama2_70b.sh
```

Run the benchmark for single node traininig
```
IMAGE="rocm/jax-training:maxtext-v25.5" bash ./llama2_70b.sh
```
3.	**Single-node training with Llama 3 8B model**\
Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama3_8b.sh
```

Run the benchmark for single node traininig
```
IMAGE="rocm/jax-training:maxtext-v25.5" bash ./llama3_8b.sh
```
4.	**Single-node training with Llama 3 70B model**\
Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama3_70b.sh
```

Run the benchmark for single node traininig
```
IMAGE="rocm/jax-training:maxtext-v25.5" bash ./llama3_70b.sh
```

5.	**Single-node training with Llama 3.3 70B model**\
Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama3.3_70b.sh
```

Run the benchmark for single node traininig
```
IMAGE="rocm/jax-training:maxtext-v25.5" bash ./llama3.3_70b.sh
```

6.	**Single-node training with DeepSeek2 16B model**\
Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/deepseek_v2_16b.sh
```
Run the benchmark for single node traininig
```
IMAGE="rocm/jax-training:maxtext-v25.5" bash ./deepseek_v2_16b.sh
```

Note: \
The reported TFLOP/s by Maxtext for deepseek is not accurate, please use the Tokens/s as performance indicator.

### Multi-Node Training examples

The examples below use slurm for running on multiple nodes.

1. **Multi-node training with Llama 2 7B model**\
   Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama2_7b_multinode.sh
```

Run the benchmark for multi-node node traininig
```
sbatch -N <num_nodes> llama2_7b_multinode.sh
```

2. **Multi-node training with Llama 2 70B model**\
   Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama2_70b_multinode.sh
```

Run the benchmark for multinode node traininig
```
sbatch -N <num_nodes> llama2_70b_multinode.sh
```

3. **Multi-node training with Llama 3 8B model**\
   Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama3_8b_multinode.sh
```

Run the benchmark for multinode node traininig
```
sbatch -N <num_nodes> llama3_8b_multinode.sh
```

4. **Multi-node training with Llama 3 70B model**\
   Download the benchmarking script:
```
wget https://raw.githubusercontent.com/ROCm/maxtext/refs/heads/main/benchmarks/gpu-rocm/llama3_70b_multinode.sh
```

Run the benchmark for multinode node traininig
```
sbatch -N <num_nodes> llama3_70b_multinode.sh
```
