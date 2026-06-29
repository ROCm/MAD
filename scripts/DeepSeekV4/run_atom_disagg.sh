#!/bin/bash
# run_atom_disagg.sh — DSv4 ATOM PD-disagg (atomesh + mooncake). Thin wrapper.
# Usage: MODEL=dsv4-pro TOPO=1p1d|2p1d_dpa [ACTION=bench|serve] ./run_atom_disagg.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=atom-disagg exec bash "$HERE/lib/run_disagg.sh"
