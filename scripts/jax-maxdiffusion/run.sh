#!/usr/bin/env bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

# Wrapper for Primus JAX/MaxDiffusion pretrain when run via madengine (local, SLURM, K8s).
# Sets EXP from PRIMUS_CONFIG_PATH or --config_path, runs Primus examples/run_pretrain.sh
# with BACKEND=MaxDiffusion, then extracts fps/tflops into primus_perf_output.csv for
# madengine multiple_results. Same shape as scripts/jax-maxtext/run.sh.
set -e

# madengine invokes this as `cd run_directory && bash run.sh ...`.
RUN_DIR="$(pwd)"

# Primus root: repo checkout, then image COPY / K8s ConfigMap extract, then env, then legacy paths.
script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$script_dir/../Primus/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="$(cd "$script_dir/../Primus" && pwd)"
elif [[ -f "/workspace/Primus/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="/workspace/Primus"
elif [[ -n "${PRIMUS_ROOT:-}" ]]; then
  :
elif [[ -f "/opt/primus/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="/opt/primus"
elif [[ -f "/workspace/examples/run_pretrain.sh" ]]; then
  export PRIMUS_ROOT="/workspace"
else
  echo "ERROR: Could not find Primus run_pretrain.sh. Set PRIMUS_ROOT or use a repo with scripts/Primus submodule." >&2
  exit 1
fi

# EXP is required by run_pretrain.sh. --config_path must also be stripped from the
# forwarded args: run_pretrain.sh appends leftovers to the training command and it is
# not a valid MaxDiffusion flag.
forward_args=()
if [[ -n "${PRIMUS_CONFIG_PATH:-}" ]]; then
  export EXP="$PRIMUS_CONFIG_PATH"
  forward_args=("$@")
else
  export EXP=""
  args=("$@")
  i=0
  while [[ $i -lt ${#args[@]} ]]; do
    if [[ "${args[i]}" == "--config_path" && -n "${args[i+1]:-}" ]]; then
      export EXP="${args[i+1]}"
      i=$((i + 2))
      continue
    fi
    forward_args+=("${args[i]}")
    i=$((i + 1))
  done
fi

if [[ -z "$EXP" ]]; then
  echo "ERROR: --config_path or PRIMUS_CONFIG_PATH required." >&2
  exit 1
fi

# Makes run_pretrain.sh launch primus/cli train pretrain rather than torchrun.
export BACKEND="MaxDiffusion"

export MAXDIFFUSION_PATH="${MAXDIFFUSION_PATH:-/workspace/maxdiffusion}"
export BACKEND_PATH="${BACKEND_PATH:-$MAXDIFFUSION_PATH}"

# The image already satisfies requirements-maxdiffusion.txt and owns the pinned
# maxdiffusion stack (patched source, specific transformers/torch), so the per-run pip
# install can only clobber it. PRIMUS_SKIP_PIP=0 restores it.
export PRIMUS_SKIP_PIP="${PRIMUS_SKIP_PIP:-1}"

# HF_TOKEN for Primus prepare: explicit, then MAD convention, then madengine v2.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
elif [[ -n "${MAD_SECRETS_HFTOKEN:-}" ]]; then
  export HF_TOKEN="$MAD_SECRETS_HFTOKEN"
elif [[ -n "${MAD_SECRET_HFTOKEN:-}" ]]; then
  export HF_TOKEN="$MAD_SECRET_HFTOKEN"
fi

# Cache weights on the mounted checkout, not Primus's default /workspace/hf_cache in the
# container's writable layer: flux_dev pulls ~58GB and this host's root filesystem also
# holds /var/lib/docker. A re-run then reuses the download instead of refetching.
export HF_HOME="${HF_HOME:-/myworkspace/hf_cache}"

# This wrapper deliberately exports no perf/arch env. All XLA_FLAGS and NVTE/HIP/HSA
# tunables travel with each config's top-level env: block, and the arch-gated ones are
# applied in-process before JAX init by primus/backends/maxdiffusion/env_spec.py.

# I/O contract, not a knob: tells Primus where to write the log this wrapper parses.
mkdir -p "$RUN_DIR/output"
export TRAIN_LOG="$RUN_DIR/output/log_mp_pretrain_$(basename "$EXP" .yaml).txt"

# The trainer writes per-step JSON metrics here (configs bind metrics_file to it). This is
# the reliable perf source: the per-step stdout line does not survive the Primus launcher's
# stdout handling. Parent of run_directory, so it outlives madengine's cleanup.
export PERF_METRICS_FILE="$RUN_DIR/../perf_metrics_$(basename "$EXP" .yaml).jsonl"
rm -f "$PERF_METRICS_FILE"

# Without these, a hard exit during trainer teardown (a fatal HIP/JAX abort in
# cleanup on_error) discards block-buffered stdout and the traceback, leaving only
# "launcher exited with code 1". The fault handler covers SIGSEGV/SIGABRT/SIGFPE.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# EXP paths are relative to PRIMUS_ROOT. No exec: the perf extractor runs after training.
# The `||` is what keeps set -e from exiting here, so a failed run still gets parsed.
cd "$PRIMUS_ROOT"
exitcode=0
bash "$PRIMUS_ROOT/examples/run_pretrain.sh" "${forward_args[@]}" || exitcode=$?

# madengine resolves multiple_results against its own CWD (the parent of run_directory)
# and deletes run_directory before parsing perf, so the CSV must go to the parent.
PERF_OUT="$RUN_DIR/../primus_perf_output.csv"
if [[ -f "$TRAIN_LOG" ]]; then
  extract_script="${script_dir}/extract_maxdiffusion_perf.py"
  [[ -f "$RUN_DIR/extract_maxdiffusion_perf.py" ]] && extract_script="$RUN_DIR/extract_maxdiffusion_perf.py"
  python3 "$extract_script" "$TRAIN_LOG" "$PERF_OUT" --model-id "$(basename "$EXP" .yaml)" \
    --metrics-file "$PERF_METRICS_FILE" || true
fi
exit "$exitcode"
