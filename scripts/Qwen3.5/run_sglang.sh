#!/bin/bash
# run_sglang.sh — launch sglang replicas across the cluster. Thin wrapper over lib/run_engine.sh.
# Usage: MODEL=qwen3-next-fp8 TP=4 [DP=2] [ACTION=sanity|serve] ./run_sglang.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=sglang exec bash "$HERE/lib/run_engine.sh"
