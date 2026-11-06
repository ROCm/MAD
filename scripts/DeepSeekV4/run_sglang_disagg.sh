#!/bin/bash
# run_sglang_disagg.sh — DSv4 SGLang PD-disagg (sglang_router + MoRI). EXPERIMENTAL (R1 template).
# Usage: MODEL=dsv4-pro TOPO=1p1d|2p1d_dpa [ACTION=bench|serve] ./run_sglang_disagg.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=sglang-disagg exec bash "$HERE/lib/run_disagg.sh"
