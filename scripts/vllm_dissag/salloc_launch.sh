#!/bin/bash

# Before running this code - request 5 nodes from salloc 
#salloc -N 5 --ntasks-per-node=1 --nodelist=<Nodes> --gres=gpu:8 -p <partition> -t 12:00:00
#Sample Commands
export xP=2; export yD=2; export MODEL_NAME=Llama-3.1-405B-Instruct-FP8-KV;      bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
export xP=1; export yD=1; export MODEL_NAME=amd-Llama-3.3-70B-Instruct-FP8-KV;   bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
export xP=2; export yD=2; export MODEL_NAME=DeepSeek-V3;   bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log


#Or directly run with sbatch command
#export DOCKER_IMAGE_NAME=<DOCKER IMAGE NAME>
#export xP=<num_prefill_nodes>; export yD=<num_decode_nodes>; export MODEL_NAME=Llama-3.1-8B-Instruct; sbatch -N <num_nodes> -n <num_nodes> --nodelist=<Nodes> run_xPyD_models.slurm

