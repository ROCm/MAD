#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

export HF_TOKEN=$MAD_SECRETS_HFTOKEN

# Parse named arguments. Each flag is handled independently: the previous shape consumed
# --model_repo and then unconditionally ran a second case, so a card passing no
# --quantization - which is most of them - printed "Unknown parameter passed:" with an empty
# value and called the undefined `usage`, then shifted an empty argument list. It only
# carried on because set -e is off, but it makes a healthy run look like a broken one.
#
# The value is checked before `shift 2`: with the flag last and no value, shift fails
# without consuming anything, $# never decreases, and the loop spins forever. set -e is off
# here, so nothing stops it - a missing argument would hang the job instead of failing it.
while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --model_repo|--quantization)
        if [[ "$#" -lt 2 ]]; then
          echo "FATAL: $1 requires a value" >&2; exit 1
        fi
        if [[ "$1" == "--model_repo" ]]; then MODEL_REPO="$2"; else QUANTIZATION="$2"; fi
        shift 2 ;;
      *) echo "FATAL: unknown parameter passed to run.sh: '$1'" >&2; exit 1 ;;
    esac
done

echo "Model repo: $MODEL_REPO"

###############################################################################
# Multi-node topology: translate what the launcher forwarded into what MaxText's
# initialize_jax_for_gpu reads. The SLURM_* reads below are fallbacks for direct
# invocation, not the normal source.
# See .claude/skills/mad-slurm-multinode/references/gotchas.md, section jax_maxtext.
###############################################################################
export NNODES="${NNODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}}"
# Must be a plain integer before any arithmetic below: `[[ $NNODES -gt 1 ]]` expands a bare
# name, so NNODES=GPUS_PER_NODE would silently evaluate as 8, and a non-numeric value makes
# the test exit 2, which reads as "single node" for a job that is not one.
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [[ "$NNODES" -lt 1 ]]; then
  echo "FATAL: NNODES='$NNODES' is not a positive integer" >&2
  exit 1
fi
# Explicit default, never derived from NPROC_PER_NODE: that variable means different things
# per launcher, so deriving num_gpus from it is silently wrong rather than fatal.
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

if [[ "${NNODES}" -gt 1 ]]; then
  # Rank 0 is the first host in the allocation, which is also where the coordinator
  # lives.
  #
  # No default to 0: every node would claim rank 0 and hang in rendezvous, silently.
  if [[ -z "${NODE_RANK:-}" && -z "${SLURM_PROCID:-}" ]]; then
    echo "FATAL: multi-node run (NNODES=${NNODES}) with neither NODE_RANK nor SLURM_PROCID set; every node would claim rank 0 and the ranks would never rendezvous" >&2
    exit 1
  fi
  export NODE_RANK="${NODE_RANK:-${SLURM_PROCID}}"
  # scontrol is not in this image, so the fallback is attempted only if it exists.
  # JAX_COORDINATOR_IP first: it is MaxText's own input and what the standalone launcher
  # scripts/jax-maxtext/jax_maxtext_multinode_benchmark.sh passes, so a caller that already
  # resolved the endpoint must not be rejected for lacking madengine's MASTER_ADDR.
  COORD_HOST="${JAX_COORDINATOR_IP:-${MASTER_ADDR:-}}"
  if [[ -z "$COORD_HOST" && -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    COORD_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
  fi
  if [[ -z "$COORD_HOST" ]]; then
    echo "FATAL: cannot determine the JAX coordinator. Set JAX_COORDINATOR_IP or MASTER_ADDR" >&2
    echo "       to the rank-0 host; SLURM_JOB_NODELIST needs scontrol, absent from this image." >&2
    exit 1
  fi
  COORD_IP=$(getent ahostsv4 "$COORD_HOST" | awk '{print $1; exit}')
  # Loopback means every rank waits on its own coordinator: a hang, not an error.
  case "$COORD_IP" in
    ""|127.*)
      echo "FATAL: coordinator '$COORD_HOST' resolved to '${COORD_IP:-<empty>}' (loopback/empty); JAX ranks would never rendezvous" >&2
      exit 1
      ;;
  esac
  export JAX_COORDINATOR_IP="$COORD_IP"
  # Same precedence as the host above: a caller that stated the endpoint keeps its port,
  # otherwise the launcher's MASTER_PORT, otherwise the default.
  export JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT:-${MASTER_PORT:-29500}}"
  # NOTE: the rendezvous timeout is a MaxText CONFIG key, not an environment variable, so
  # it is applied in jax-maxtext_benchmark_report.sh as
  # jax_distributed_initialization_timeout=<n> (override with JAX_RENDEZVOUS_TIMEOUT_S).
  echo "JAX coordinator: ${JAX_COORDINATOR_IP}:${JAX_COORDINATOR_PORT} rank=${NODE_RANK}/${NNODES} gpus_per_node=${GPUS_PER_NODE}"
fi

model=""   # never inherit this from the environment
if [[ "$MODEL_REPO" == "jax_maxtext_train_llama-3.1-8b" ]]; then
  model="Llama-3.1-8B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-3.1-70b" ]]; then
  model="Llama-3.1-70B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-3.1-405b" ]]; then
  model="Llama-3.1-405B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-3.3-70b" ]]; then
  model="Llama-3.3-70B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-2-7b" ]]; then
  model="Llama-2-7B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_llama-2-70b" ]]; then
  model="Llama-2-70B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_deepseek-v2-lite-16b" ]]; then
  model="DeepSeek-V2-lite"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_mixtral-8x7b" ]]; then
  model="Mixtral-8x7B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_qwen3-14b" ]]; then
  model="Qwen3-14B"
elif [[ "$MODEL_REPO" == "jax_maxtext_train_qwen3-30b-a3b" ]]; then
  model="Qwen3-30B-A3B"
fi

if [ -z "$model" ]; then
  echo "FATAL: no model mapping for MODEL_REPO='$MODEL_REPO'" >&2
  exit 1
fi

# Invoked through `bash`, not `./`: the executable bit does not survive the checkout.
# Setup failure is loud but NOT fatal - these cards use synthetic data, and the gated-repo
# tokenizer fetch fails on every run on hosts without access.
# See .claude/skills/mad-slurm-multinode/references/gotchas.md, section jax_maxtext.
bash ./jax-maxtext_benchmark_setup.sh -m "$model" && _setup_rc=0 || _setup_rc=$?
if [ "$_setup_rc" -ne 0 ]; then
  echo "WARNING: benchmark setup failed (rc=${_setup_rc}). Continuing: these cards use" >&2
  echo "         synthetic data, so this only matters for a card needing a real dataset." >&2
fi
# Quoted: an empty QUANTIZATION unquoted collapses to nothing and leaves a bare `-q`,
# which getopts "m:q:" rejects with "option requires an argument". The bf16 multi-node card
# passes no --quantization at all, so this fires in practice.
if ! bash ./jax-maxtext_benchmark_report.sh -m "$model" -q "${QUANTIZATION:-}"; then
  echo "FATAL: benchmark report step failed; not reporting a pass" >&2
  exit 1
fi

# Single node keeps its historical contract: madengine scrapes the FIRST "performance:"
# match, so the reporter emits its median only for multi-node runs. The median still reaches
# the CSV either way.
if [[ "${NNODES}" -eq 1 ]]; then
  echo "performance: 1 pass"
fi
