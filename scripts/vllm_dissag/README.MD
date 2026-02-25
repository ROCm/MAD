## List of Models - focus VLLM Dissagerated P/D inference

Dense Models
- deepseek-ai/DeepSeek-V3 (https://huggingface.co/deepseek-ai/DeepSeek-V3)
- amd/Llama-3.3-70B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV)
- amd/Llama-3.1-405B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV)
- openai/gpt-oss-120b - (https://huggingface.co/openai/gpt-oss-120b)

This repository contains scripts and documentation to launch PD Disaggregation using the Nixl framework for above models. You will find setup instructions, node assignment details and benchmarking commands.

## 📝 Prerequisites

- A Slurm cluster with required Nodes -> xP + yD + 1 (minimum size 3: xP=1 and xD=1)
- Docker container with VLLM, Nixl, etcd and NIC drivers built-in. Refer to Building the Docker image section below.
- Access to a shared filesystem for log collection( cluster specific)


## Building the Docker image


```bash
docker build  -t vllm_dissag_pd_image -f vllm_dissag_inference.ubuntu.amd.Dockerfile .
```

## Scripts and Benchmarking
Run instructions - scripts/vllm_dissag/README.MD

Few files of significance:

scripts/vllm_dissag/run_xPyD_models.slurm - slurm script to launch docker containers on all nodes using sbatch or salloc  
scripts/vllm_dissag/vllm_disagg_server.sh - Script that runs inside each docker to start required proxy, prefill and decode services
scripts/vllm_dissag/benchmark_xPyD.sh - Benchmark script to run VLLM benchmarking tool for performance measurement

## Sbatch run command (one-liner)
```bash

# Clone the repo
git clone https://github.com/ROCm/MAD.git
cd scripts/vllm_dissag

# Sbatch run command [run from the above folder]
export DOCKER_IMAGE_NAME=<DOCKER IMAGE NAME>
export xP=<num_prefill_nodes>; export yD=<num_decode_nodes>; export MODEL_NAME=Llama-3.1-8B-Instruct; sbatch -N <num_nodes> -n <num_nodes> --nodelist=<Nodes> run_xPyD_models.slurm

# num_nodes = xP + xD + 1
```

## Benchmark parser ( for CONCURRENCY logs) to tabulate different data

```bash
python3 benchmark_parser.py <log_path/benchmark_XXX_CONCURRENCY.log
```
