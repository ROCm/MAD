#!/bin/bash
# run_atom.sh — launch atom replicas across the cluster. Thin wrapper over lib/run_engine.sh.
# Usage: MODEL=qwen3-next-fp8 TP=4 [DP=2] [ACTION=sanity|serve] ./run_atom.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=atom exec bash "$HERE/lib/run_engine.sh"
