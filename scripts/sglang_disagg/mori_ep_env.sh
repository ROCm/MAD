#!/bin/bash
# NCCL/MORI RDMA configuration for multi-NIC nodes.
# Toggle USE_CX7_NICS to switch between CX7 rail NICs and mlx5_1 management NIC.
#
# This file is now a thin aggregator: the config was extracted into config/
# (bucket-aligned). It sources the pieces in dependency order. The single
# `source mori_ep_env.sh` call in the entrypoint keeps working unchanged.
# See config/README.md and ../CONFIG.md.

_MORI_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${_MORI_ENV_DIR}/config/nic-selection.env.sh"
# shellcheck disable=SC1091
source "${_MORI_ENV_DIR}/config/framework.env.sh"
# shellcheck disable=SC1091
source "${_MORI_ENV_DIR}/config/connectors.env.sh"
