#!/bin/bash

# Before running this code - request 5 nodes from salloc 
#salloc -N 5 --ntasks-per-node=1 --nodelist=<Nodes> --gres=gpu:8 -p <partition> -t 12:00:00
#Sample Commands
export xP=2; export yD=2; export MODEL_NAME=Qwen3-32B;                          bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
export xP=2; export yD=2; export MODEL_NAME=Qwen3-30B-A3B;                      bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
export xP=2; export yD=2; export MODEL_NAME=Mixtral-8x7B-v0.1;                   bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
export xP=2; export yD=2; export MODEL_NAME=Llama-3.1-8B-Instruct;               bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
export xP=2; export yD=2; export MODEL_NAME=Llama-3.1-405B-Instruct-FP8-KV;      bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
export xP=2; export yD=2; export MODEL_NAME=amd-Llama-3.3-70B-Instruct-FP8-KV;   bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log


#Or directly run with sbatch command
#export DOCKER_IMAGE_NAME=<DOCKER IMAGE NAME>
#export xP=<num_prefill_nodes>; export yD=<num_decode_nodes>; export MODEL_NAME=Llama-3.1-8B-Instruct; sbatch -N <num_nodes> -n <num_nodes> --nodelist=<Nodes> run_xPyD_models.slurm

# === Agentic replay benchmark (aiperf inferencex-agentx-mvp) ===
# Selected via BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh, or the AGENTIC=1 shorthand.
# AGENTIC=1 auto-enables server metrics (gpu_cache_hit_rate) + radix prefix cache.
#
# DeepSeek-V3 1P/1D (canonical, DP_MODE=1 wideEP):
#export DOCKER_IMAGE_NAME=<mori-sglang-image>
#export AGENTIC=1 RUN_MORI=1 DP_MODE=1 xP=1 yD=1 MODEL_NAME=DeepSeek-V3
#export DURATION=900 AGENTIC_CONC=16 AGENTIC_CACHE_WARMUP_DURATION=300 MAX_MODEL_LEN=160000
#sbatch -N 3 -n 3 -p amd-rccl --nodelist=<3-nodes> run_xPyD_models.slurm
#
# Dense model 1P/1D (TP-only, DP_MODE=0) quick smoke:
#export AGENTIC=1 RUN_MORI=1 DP_MODE=0 xP=1 yD=1 MODEL_NAME=Llama-3.1-8B-Instruct
#export DURATION=120 AGENTIC_CONC=8
#sbatch -N 3 -n 3 -p amd-rccl --nodelist=<3-nodes> run_xPyD_models.slurm

