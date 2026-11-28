#!/bin/bash
# Test only (tests/coverage G1): one vLLM model over ND_NNODES nodes (nd's `all` spec lines set ND_NNODES /
# ND_NODE_RANK / ND_MASTER_ADDR / MASTER_PORT). vLLM's multi-node launch without Ray: the head node (rank 0) runs
# the server -- through scripts/serve_run.sh, which sends the requests -- and every other node runs headless
# workers (`--headless`); an offline vllm.LLM cannot be a follower ("collective_rpc should not be called on
# follower node", 2026-11-24).
#   bash vllm_multinode.sh <model> <requests.jsonl> <tp> <pp> [more vllm serve args]
set -u
model=$1 req=$2 tp=$3 pp=$4; shift 4
# Each node sees exactly its share of the world (tp x pp / nnodes GPUs). vLLM 0.33 judges "same node as the
# reader" as rank // device_count (shm_broadcast create_from_process_group_single_reader): with all 8 GPUs
# visible and fewer ranks per node, the other node's ranks count as local, no remote response queue is made,
# and the head's engine stops at `assert remote_message_queue is not None` (2026-11-25).
lws=$(( tp * pp / ${ND_NNODES:?} ))
if [ -z "${CUDA_VISIBLE_DEVICES:-}${HIP_VISIBLE_DEVICES:-}" ]; then
  vis=$(seq -s, 0 $(( lws - 1 ))); export CUDA_VISIBLE_DEVICES=$vis HIP_VISIBLE_DEVICES=$vis
fi
common="--tensor-parallel-size $tp --pipeline-parallel-size $pp --distributed-executor-backend mp \
  --nnodes ${ND_NNODES:?} --master-addr ${ND_MASTER_ADDR:?} --master-port ${MASTER_PORT:-29511} \
  --max-model-len 512 --gpu-memory-utilization 0.3 --seed 0 $*"
if [ "${ND_NODE_RANK:?}" = 0 ]; then
  exec bash "$(dirname "$0")/../../scripts/serve_run.sh" --ready-timeout 1200 --requests "$req" \
    --server "vllm serve $model --port {port} --node-rank 0 $common"
else
  exec vllm serve "$model" --headless --node-rank "$ND_NODE_RANK" $common
fi
