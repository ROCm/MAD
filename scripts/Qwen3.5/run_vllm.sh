#!/bin/bash
# run_vllm.sh — launch vllm replicas across the cluster. Thin wrapper over lib/run_engine.sh.
# Usage: MODEL=qwen3-next-fp8 TP=4 [DP=2] [ACTION=sanity|serve] ./run_vllm.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=vllm exec bash "$HERE/lib/run_engine.sh"
