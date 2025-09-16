## List of Models - focus SGlang Disaggerated P/D inference

Dense Models
- Qwen3-32B (https://huggingface.co/Qwen/Qwen3-32B)
- meta-llama/Llama-3.1-8B-Instruct (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- amd/Llama-3.3-70B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV)
- amd/Llama-3.1-405B-Instruct-FP8-KV (https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV)

Small Experts Models
- DeepSeek-V3 (https://huggingface.co/deepseek-ai/DeepSeek-V3)
- Mixtral-8x7B-v0.1 (https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

This repository contains scripts and documentation to launch PD Disaggregation using the Mooncake framework for above models. You will find setup instructions, node assignment details and benchmarking commands.

## 📝 Prerequisites

- A Slurm cluster with required Nodes -> xP + yD + 1 (minimum size 3: xP=1 and xD=1)
- Docker container with SGLang, Mooncake, etcd and NIC drivers built-in. Refer to Building the Docker image section below.
- Access to a shared filesystem for log collection( cluster specific)


## Building the Docker image
Access the Dockerfile located https://github.com/ROCm/MAD/docker/sglang_disagg_inference.ubuntu.amd.Dockerfile
It uses ‘lmsysorg/sglang:v0.5.2rc1-rocm700-mi30x’ as the base docker image.

```bash
docker build  -t sglang_disagg_pd_image -f sglang_disagg_inference.ubuntu.amd.Dockerfile .
```

## Scripts and Benchmarking
Run instructions - scripts/sglang_disagg/README.MD

Few files of significance:

scripts/sglang_disagg/run_xPyD_models.slurm - slurm script to launch docker containers on all nodes using sbatch or salloc <br>
scripts/sglang_disagg/sglang_disagg_server.sh - Script that runs inside each docker to start required proxy, prefill and decode services <br>
scripts/sglang_disagg/benchmark_xPyD.sh - Benchmark script to run GSM8K for accuracy and sglang benchmarking tool for performance measurement <br>
scripts/sglang_disagg/benchmark_parser.py - Log parser script to be run on CONCURRENY benchmark log file to generate tabulated data <br>

## Sbatch run command (one-liner)
```bash

# Clone the repo
git clone https://github.com/ROCm/MAD.git
cd scripts/sglang_disagg

# Sbatch run command [run from the above folder]
export DOCKER_IMAGE_NAME=<DOCKER IMAGE NAME>
export xP=<num_prefill_nodes>; export yD=<num_decode_nodes>; export MODEL_NAME=Llama-3.1-8B-Instruct; sbatch -N <num_nodes> -n <num_nodes> --nodelist=<Nodes> run_xPyD_models.slurm

# num_nodes = xP + xD + 1
```

## Post execution Log files:
A directory inside the LOG_PATH variable in the slurm script is created by the name of slurm_job_ID. 

Inside that folder:

pd_sglang_bench_serving.sh_NODE<>.log - Overall log per ser Node 
etcd_NODE<>.log  - for etcd services
decode_NODE<>.log - Decode services
prefill_NODE<>.log - prefill services


## Benchmark parser ( for CONCURRENCY logs) to tabulate different data

```bash
python3 benchmark_parser.py <log_path/benchmark_XXX_CONCURRENCY.log
```

## sample curl command to test launched server ( from docker on the proxy node)
```bash
curl -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" -d '{ "text": "Let me tell you a story ", "sampling_params": { "temperature": 0.3 } }'
```

## Known Issues
For larger models, such as DeepSeekV3 and Llama-3.1-405B-Instruct-FP8-KV and higher concurrency(512+), errors with below signature is observed:<br>
_'<TransferEncodingError: 400, message:\n  Not enough data to satisfy transfer length header.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n '_<br>
This leads to dropping requests and lower throughput.This issue is being discussed on the SGLang forums.
