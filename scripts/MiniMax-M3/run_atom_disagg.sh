#!/bin/bash
# run_atom_disagg.sh — MiniMax-M3 ATOM PD-disagg (atomesh + mooncake). Thin wrapper.
# Usage: MODEL=minimaxm3-fp4 TOPO=1p1d|2p1d_dpa [ACTION=bench|serve] ./run_atom_disagg.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=atom-disagg MODEL="${MODEL:-minimaxm3-fp4}" exec bash "$HERE/lib/run_disagg.sh"
