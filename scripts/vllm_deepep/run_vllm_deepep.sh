#!/bin/bash
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
#
# vLLM expert-parallel serving over an all-to-all backend, single node, 8 GPUs.
#
# Selects the backend with ALL2ALL_BACKEND (deepep_v2 | mori_high_throughput)
# so the same script produces both arms of a backend comparison with nothing
# else changed -- which is the only way the comparison means anything.
#
# Emits perf_vllm_deepep.csv (models.json: multiple_results).
#
#################################################################################
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-V2-Lite}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-deepep_v2}"
PORT="${PORT:-18000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
ISL="${ISL:-256}"
OSL="${OSL:-128}"
CONCURRENCY="${CONCURRENCY:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"

# AITER tunes GEMM shapes at run time behind a global lock, so an unwarmed run
# measures the tuner, not the server: P99 TTFT lands around 20 s on the first
# pass against a few hundred ms once warm, and roughly 270 s on DeepSeek-R1.
# The discarded count is a reported field, not a hidden constant -- a
# comparison that does not state it is not interpretable.
WARMUP_RUNS="${WARMUP_RUNS:-2}"
MEASURED_RUNS="${MEASURED_RUNS:-3}"

# seq 1 0 (or 1 <negative>) silently produces no iterations, so a bad value
# here does not fail -- it skips straight to a header-only CSV and the script
# exits 0, which MAD records as a successful run with zero measurements.
_require_positive_int() {
  local name="$1" value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name}=${value} must be a positive integer" >&2
    exit 1
  fi
}
_require_positive_int WARMUP_RUNS "${WARMUP_RUNS}"
_require_positive_int MEASURED_RUNS "${MEASURED_RUNS}"

PYTHON="${PYTHON:-/opt/venv/bin/python}"
VLLM="${VLLM:-/opt/venv/bin/vllm}"
LOG_DIR="${LOG_DIR:-$(pwd)}"
SERVER_LOG="${LOG_DIR}/vllm_${ALL2ALL_BACKEND}_server.log"
# MAD ingests multiple_results in long form: it requires exactly the columns
# model / performance / metric and drops everything else (tools/utils.py:1231
# raises if one is missing; :1247-1250 reads only those three). So the
# diagnostic context that makes a number interpretable -- backend, load shape,
# discarded warm-ups -- cannot live in that file. It goes to a companion CSV
# that is not registered as multiple_results and is therefore not parsed.
RESULT_CSV="${LOG_DIR}/perf_vllm_deepep.csv"
DETAIL_CSV="${LOG_DIR}/perf_vllm_deepep_detail.csv"

echo "=== vllm_deepep: ${MODEL_REPO} over ${ALL2ALL_BACKEND} ==="

# --- backend-specific environment -------------------------------------------
# NCCL_CUMEM_ENABLE is required by BOTH backends: RCCL refuses symmetric memory
# (and ncclDevCommCreate) without VMM.
export NCCL_CUMEM_ENABLE=1
export VLLM_ROCM_USE_AITER=1
# vLLM refuses MoRI without AITER fused MoE, and running the two arms on
# different MoE implementations would make the comparison meaningless, so both
# arms use it.
export VLLM_ROCM_USE_AITER_MOE=1
export PYTHONPATH="${PYTHONPATH:-}:/opt/rocm/share/amd_smi"
export HSA_NO_SCRATCH_RECLAIM=1

case "${ALL2ALL_BACKEND}" in
  deepep_v2)
    # GIN stays on. It carries no payload inside a single XGMI domain, but it
    # is the path that matters for scale-out, and testing a mode nobody
    # deploys is not testing. Requires /dev/infiniband in the container.
    export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-2}"
    export EP_GIN_QUEUE_DEPTH="${EP_GIN_QUEUE_DEPTH:-0}"
    export EP_NIC_NAME="${EP_NIC_NAME:-bnxt_re0}"
    export VLLM_DEEPEP_TIMEOUT_SECS="${VLLM_DEEPEP_TIMEOUT_SECS:-1800}"
    ;;
  mori_high_throughput)
    # Works around a null dereference in MoRI's CollectAndSortCandidates. It
    # skips GPU/NIC affinity matching, which should not matter for a
    # single-node run where traffic stays on XGMI -- but that is an
    # expectation, so it is recorded in the CSV rather than left implicit.
    export MORI_DISABLE_TOPO="${MORI_DISABLE_TOPO:-1}"
    ;;
  *)
    echo "unknown ALL2ALL_BACKEND: ${ALL2ALL_BACKEND}" >&2
    exit 1
    ;;
esac

# --- model ------------------------------------------------------------------
# vLLM has native architectures for both configured models
# (vllm/model_executor/models/deepseek_v2.py covers DeepseekV2ForCausalLM and
# DeepseekV3ForCausalLM) and current transformers ships its own DeepseekV2Config
# / DeepseekV3Config, so AutoConfig resolves without the repository's
# auto_map -- trust_remote_code is not required for either DeepSeek-V2-Lite or
# DeepSeek-R1. It defaults off. Executing a mutable HF repo's Python as root in
# this container (host networking, elevated capabilities, device access) is
# not something to enable by default; TRUST_REMOTE_CODE=1 is an explicit
# opt-in for a model that genuinely needs it.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
TRUST_REMOTE_CODE_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_REMOTE_CODE_ARGS=(--trust-remote-code)
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL="${MODEL_PATH}"
  # Not "main": that would claim this is the Hub's moving revision, when it
  # is not on the Hub at all. Recorded as empty -- "not applicable" -- and
  # rendered explicitly in the CSV below rather than through a ":-main"
  # default, which cannot distinguish "unset" from "deliberately empty".
  MODEL_REVISION_LABEL=""
elif [[ -n "${MODEL_REVISION:-}" ]]; then
  # Resolve to one local snapshot and serve/benchmark THAT path. A --revision
  # flag on `vllm serve` alone is not enough: `vllm bench serve` loads its own
  # tokenizer independently, at the model's default (moving) Hub revision --
  # https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py,
  # `get_tokenizer(tokenizer_id, ...)` with no revision argument, and `vllm
  # bench serve` has no --revision flag to give it one. So a moved `main`
  # would tokenize the benchmark's prompts differently from what the pinned
  # server understands, and the mismatch is silent -- both sides consider
  # their own tokenizer authoritative. Downloading once and pointing both
  # commands at the same local directory removes the second revision
  # entirely, rather than trying to keep two of them in sync.
  echo "resolving ${MODEL_REPO}@${MODEL_REVISION} to a local snapshot..."
  # repo/revision passed as argv, not interpolated into the Python source: a
  # ref containing a quote would otherwise produce a SyntaxError at best, and
  # arbitrary Python at worst, from a value that is supposed to be inert data.
  MODEL="$("${PYTHON}" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
" "${MODEL_REPO}" "${MODEL_REVISION}")"
  echo "resolved to ${MODEL}"
  MODEL_REVISION_LABEL="${MODEL_REVISION}"
else
  echo "MODEL_REVISION is not set; serving ${MODEL_REPO} at its moving" >&2
  echo "'main' Hub revision. Results are not reproducible run to run." >&2
  MODEL="${MODEL_REPO}"
  MODEL_REVISION_LABEL="main"
fi

# --- measurement channels ----------------------------------------------------
# Off by default, and deliberately: RCCL's INFO logging and the torch profiler both cost
# throughput, so the numbers in perf_vllm_deepep.csv have to come from a run without
# PROFILE_ENABLE. Reading the adapters' sysfs counters perturbs nothing, so RDMA_COUNTERS=1 on
# its own measures a run whose throughput is still comparable with an unprofiled one.
#
# What each channel is for, and why more than one: a backend that carries its expert all-to-all
# outside RCCL -- which is what both deepep_v2 and mori do -- is invisible to the RCCL log, so the
# traces are the only place its kernels are named, and the adapter counters the only place its
# bytes are counted. Any single channel would miss the thing being compared.
PROFILE_ENABLE="${PROFILE_ENABLE:-0}"
RDMA_COUNTERS="${RDMA_COUNTERS:-${PROFILE_ENABLE}}"
# Default under the working directory, not /run_logs. This workload mounts no /run_logs -- unlike
# sglang_disagg, whose manifest binds one -- so artifacts written there would land in the
# container's writable layer and disappear with it, while the run still reported them as collected.
# The working directory is inside the bind-mounted workspace, so it outlives the container.
PROFILE_DIR="${PROFILE_DIR:-${PWD}/profile_${ALL2ALL_BACKEND}}"

_rdma_sampler_pid=""
_rdma_counters_out=""

if [[ "${PROFILE_ENABLE}" == "1" ]]; then
  mkdir -p "${PROFILE_DIR}/rccl" "${PROFILE_DIR}/traces"
  # One file per process. The server has eight engine subprocesses and they all log; without %p
  # they interleave into one file and every record is torn at a random point.
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,NET,ENV,COLL}"
  export NCCL_DEBUG_FILE="${PROFILE_DIR}/rccl/vllm.%h.%p.log"
  # vLLM writes a trace per rank here when the profiler is started; the benchmark below is what
  # drives it. Left unset the profiler is simply off, which is why this is inside the guard.
  export VLLM_TORCH_PROFILER_DIR="${PROFILE_DIR}/traces"
  echo "profiling on: RCCL logs and torch traces under ${PROFILE_DIR}"
  echo "NOTE: throughput from a PROFILE_ENABLE=1 run is not comparable with an unprofiled one" >&2
fi

# The sampler is shared with the sglang_disagg workload rather than copied: it reads sysfs and
# knows nothing about either engine, so a second copy would only be a second place for the counter
# schema to drift. In the tree it is a symlink in this directory; madengine copies a model's own
# script directory with `cp -L`, which turns the symlink into the real file next to this script.
#
# Resolution is next-to-the-script first, then the sibling directory for anyone running from a
# checkout. Neither step may abort: `$(cd ...)` on a missing directory returns non-zero, and under
# `set -e` that killed the whole workload here -- with profiling off, which is every normal run.
_resolve_sampler() {
  local _here _candidate
  _here="$(dirname "${BASH_SOURCE[0]}")"
  for _candidate in "${_here}/rdma_counters.sh" "${_here}/../sglang_disagg/rdma_counters.sh"; do
    [[ -f "${_candidate}" ]] && { printf '%s\n' "${_candidate}"; return 0; }
  done
  return 0   # empty: _start_rdma_counters says so and carries on
}
_RDMA_SAMPLER="${_RDMA_SAMPLER:-$(_resolve_sampler)}"

_start_rdma_counters() {
  [[ "${RDMA_COUNTERS}" == "1" ]] || return 0
  if [[ -z "${_RDMA_SAMPLER}" || ! -f "${_RDMA_SAMPLER}" ]]; then
    echo "RDMA_COUNTERS=1 but rdma_counters.sh was not found next to this script" >&2
    return 0
  fi
  mkdir -p "${PROFILE_DIR}/rdma"
  _rdma_counters_out="${PROFILE_DIR}/rdma/vllm_$(hostname -s).csv"
  # Through `bash` rather than executed directly: a file that arrives by `cp` or a fresh checkout
  # does not always keep its exec bit, and the sampler is a plain bash script either way.
  bash "${_RDMA_SAMPLER}" --out "${_rdma_counters_out}" \
    --interval "${RDMA_COUNTERS_INTERVAL:-30}" &
  _rdma_sampler_pid=$!
  # A baseline before the server starts, or the window begins mid-run and the difference between
  # the arms is an undercount that looks like a whole measurement.
  #
  # Liveness is checked before the file, not after. The sampler writes its header and only then
  # discovers it has nothing to sample, so a file-first test reported "RDMA counters: <path>" for
  # a sampler that had already exited -- a claim that counters were being collected when none
  # were. A header-only file is still kept: the report reads it as a node that was sampled and
  # found nothing, which is not the same as a node never sampled.
  local _deadline=$(( $(date +%s) + ${RDMA_COUNTERS_START_TIMEOUT_S:-60} )) _st=0
  while (( $(date +%s) < _deadline )); do
    if ! kill -0 "${_rdma_sampler_pid}" 2>/dev/null; then
      wait "${_rdma_sampler_pid}" 2>/dev/null || _st=$?
      _rdma_sampler_pid=""
      if (( _st == 0 )); then
        echo "RDMA sampler found nothing to sample; no counters for this node"
      else
        echo "WARNING: RDMA sampler exited ${_st} before its first sample" >&2
        echo "sampler exited ${_st}" > "${_rdma_counters_out}.failed" 2>/dev/null || true
      fi
      return 0
    fi
    [[ -s "${_rdma_counters_out}" ]] && { echo "RDMA counters: ${_rdma_counters_out}"; return 0; }
    sleep 0.2
  done
  # Marked, not merely warned about: left unmarked the sampler keeps running and the report sees
  # an ordinary multi-sample file whose window starts after the server.
  echo "WARNING: no RDMA baseline sample before the server started; the window is a floor" >&2
  echo "no baseline sample before the server started" \
    > "${_rdma_counters_out}.failed" 2>/dev/null || true
}

# SIGTERM, not SIGKILL: the sampler takes one last sample on it, so the window closes where the
# run does instead of at the second-to-last sample.
_stop_rdma_counters() {
  [[ -n "${_rdma_sampler_pid}" ]] || return 0
  local _status=0
  kill -TERM "${_rdma_sampler_pid}" 2>/dev/null || true
  wait "${_rdma_sampler_pid}" 2>/dev/null || _status=$?
  if (( _status != 0 )); then
    echo "WARNING: RDMA sampler exited ${_status}; its window is incomplete" >&2
    echo "sampler exited ${_status}" > "${_rdma_counters_out}.failed" 2>/dev/null || true
  fi
  _rdma_sampler_pid=""
}

# --- server -----------------------------------------------------------------
# Refuse to start if something already answers on PORT. Otherwise a stale
# server left by an earlier run satisfies the readiness probe below, the newly
# launched one dies with "address already in use", and the benchmark measures
# the old process while the CSV labels it with this run's backend and model --
# a wrong number that looks entirely well-formed.
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "something is already serving on port ${PORT}; refusing to start." >&2
  echo "stop it, or set PORT to a free port." >&2
  exit 1
fi

# _server_gone -- true only when neither the process group nor the pid still exists. The group
# matters more than the pid here: `--data-parallel-size 8` means the server has eight engine
# subprocesses, and those are what hold the GPUs after the parent is gone.
_proc_scan() {
  local _pgid="$1" _stat _raw _after _pid _comm _state _pgrp _wchan
  for _stat in /proc/[0-9]*/stat; do
    [[ -r "$_stat" ]] || continue
    read -r _raw < "$_stat" 2>/dev/null || continue
    # Fields: pid (comm) state ppid pgrp ... -- comm is arbitrary text in parentheses and can
    # itself contain spaces and ')', so everything is taken relative to its *last* ')'.
    _pid=${_raw%% *}
    _comm=${_raw#*(}; _comm=${_comm%)*}
    _after=${_raw##*) }
    read -r _state _ _pgrp _ <<< "$_after"
    [[ "$_pgrp" == "$_pgid" || "$_pid" == "$_pgid" ]] || continue
    _wchan=""
    [[ -r "/proc/${_pid}/wchan" ]] && read -r _wchan < "/proc/${_pid}/wchan" 2>/dev/null
    echo "${_pid} ${_state} ${_comm} ${_wchan:-?}"
  done
}

# True when nothing in the group is still running. A zombie does not count: it holds no memory,
# no GPU and no file descriptors, only a row in the process table kept until someone calls wait().
#
# `kill -0` succeeds on a zombie, which is what this asked before. In the sglang launcher that
# cost every shutdown the full SIGINT/SIGTERM/SIGKILL escalation against servers that had already
# exited, and a "survived SIGKILL" warning on every run -- job 472379 showed 36 such survivors,
# all state=Z, because the container's init never reaps the engine subprocesses. vLLM's
# data-parallel engines are orphaned the same way, so the same test is wrong here.
#
# Kept in step with the copy in ../sglang_disagg/sglang_disagg_mori_io_ep.sh. Duplicated rather
# than sourced: this script has to stay runnable on its own.
_server_gone() {
  # Signals stay the fallback where /proc cannot be read: a wrong "gone" would let the container
  # be torn down with workers still running, which is worse than the false alarm this replaces.
  if [[ ! -r /proc/self/stat ]]; then
    kill -0 -- "-${SERVER_PID}" 2>/dev/null && return 1
    kill -0 "${SERVER_PID}" 2>/dev/null && return 1
    return 0
  fi
  local _pid _state _rest
  while read -r _pid _state _rest; do
    [[ "$_state" == "Z" ]] && continue
    return 1
  done < <(_proc_scan "${SERVER_PID}")
  return 0
}

_wait_gone() {
  local _i
  for ((_i = 1; _i <= $1; _i++)); do
    _server_gone && return 0
    sleep 1
  done
  return 1
}

# Stop the server, and say so plainly if it did not stop.
#
# This replaces `trap 'kill ${SERVER_PID}'`, which was wrong in three ways: it signalled the pid
# and not the group, so the eight engine subprocesses were never told to stop; it did not wait,
# so this script exited and madengine tore the container down mid-teardown; and it never checked,
# so a server left holding 8 GPUs was indistinguishable from a clean exit. On this fleet that is
# not hypothetical -- every run of the sglang launcher on 2026-09-14 reported a server that
# outlived its shutdown.
#
# Deliberately does not touch the exit status. A shutdown that happened after the last measurement
# is not a reason to fail a run and discard its CSV; the warning is the deliverable.
_shutdown_server() {
  [[ -n "${SERVER_PID:-}" ]] || return 0
  local _sig _grace
  for _sig in INT TERM KILL; do
    kill "-${_sig}" -- "-${SERVER_PID}" 2>/dev/null \
      || kill "-${_sig}" "${SERVER_PID}" 2>/dev/null \
      || return 0
    # SIGINT gets the widest budget: it is the one vLLM unwinds its engine subprocesses on, and
    # that unwind is what lets a profiled run flush its output.
    [[ "${_sig}" == "INT" ]] && _grace="${SHUTDOWN_GRACE_S:-60}" || _grace=30
    _wait_gone "${_grace}" && return 0
    echo "server did not stop on SIG${_sig} after ${_grace}s" >&2
  done
  echo "WARNING: server ${SERVER_PID} survived SIGKILL and may still hold GPUs on $(hostname -s)" >&2
  _shutdown_failed=1
  return 0
}

_shutdown_failed=0

# Servers first and the sampler last, so the counter window covers them. A server that outlived
# its shutdown kept putting traffic on these adapters, so the window is no longer this run's --
# recorded through the same `.failed` marker the sampler's own failures use, which the report
# already reads as "a floor, not a total".
_cleanup_on_exit() {
  _shutdown_server
  if (( _shutdown_failed )) && [[ -n "${_rdma_counters_out}" ]]; then
    echo "a server survived shutdown; the counter window is not this run's" \
      > "${_rdma_counters_out}.failed" 2>/dev/null || true
  fi
  _stop_rdma_counters
}

_start_rdma_counters

# Monitor mode for the launch alone, so the server leads its own process group. Without it the job
# joins this script's group and `kill -- -PID` above would signal this script too.
set -m
"${VLLM}" serve "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --all2all-backend "${ALL2ALL_BACKEND}" \
  --enforce-eager \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  "${TRUST_REMOTE_CODE_ARGS[@]}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
set +m
trap _cleanup_on_exit EXIT

echo "server pid ${SERVER_PID}, log ${SERVER_LOG}"

# Startup is dominated by AITER JIT (~5-10 min cold, more on a large model),
# so wait on the endpoint rather than on a fixed sleep.
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "server died during startup; tail of ${SERVER_LOG}:" >&2
    tail -40 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 15
done
# The loop above breaks on success or on the server dying; a server that
# stays alive without ever answering healthy falls through to here. Checked
# explicitly rather than left to run bare under set -e: an unguarded failure
# at this point aborts with no message and no log tail, making a cold-JIT
# hang or an init deadlock look identical to every other exit.
if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "server never became healthy after $((240 * 15))s; tail of ${SERVER_LOG}:" >&2
  tail -40 "${SERVER_LOG}" >&2
  exit 1
fi

# Bounds a single benchmark invocation. Without it, a stall in the EP
# collective after a healthy startup -- not covered by the health check above,
# which only proves the server came up -- would hang here indefinitely,
# occupying all 8 GPUs for as long as the job is left running. 1800s is
# generous relative to a normal run (seconds to low minutes at the sizes this
# script uses) rather than tuned to any one workload; override for larger
# ISL/OSL/NUM_PROMPTS combinations.
BENCH_TIMEOUT_SECS="${BENCH_TIMEOUT_SECS:-1800}"

bench_once() {
  # --trust-remote-code forwarded: bench serve loads its own tokenizer
  # (get_tokenizer(...)) independently of the server, and a model that
  # genuinely needs the opt-in would otherwise serve successfully and then
  # fail here, after the server already came up.
  timeout --kill-after=30 "${BENCH_TIMEOUT_SECS}" \
    "${VLLM}" bench serve --backend vllm --model "${MODEL}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dataset-name random \
    --random-input-len "${ISL}" --random-output-len "${OSL}" \
    --max-concurrency "${CONCURRENCY}" --num-prompts "${NUM_PROMPTS}" \
    --ignore-eos "${TRUST_REMOTE_CODE_ARGS[@]}" 2>&1
}

# Shared between warm-up and measured runs: a nonzero "Failed requests" count
# does not fail the process, so checking the exit status alone is not enough
# in either loop. For a warm-up specifically, a partial failure leaves AITER's
# GEMM shapes still untuned -- exactly what warm-up exists to avoid -- while
# pushing that cold-tuning cost into the first MEASURED run instead, silently.
_failed_requests() {
  echo "$1" | awk '/Failed requests/ {print $NF; exit}'
}

# RFC4180-quoting via Python's csv module rather than string concatenation:
# MODEL is attacker-influenced-in-practice input (MODEL_PATH, or a resolved
# Hub snapshot path) and MODEL_REVISION is an arbitrary ref -- either could
# contain a comma, a quote, or a newline, any of which shifts columns or
# splits one row into several under plain `echo "a,b,c"`. Fields passed as
# argv, not interpolated into source, for the same reason the snapshot
# resolution above does it that way.
_csv_row() {
  "${PYTHON}" -c '
import csv, sys
csv.writer(sys.stdout, lineterminator=chr(10)).writerow(sys.argv[1:])
' "$@"
}

echo "--- ${WARMUP_RUNS} warm-up run(s), discarded ---"
for i in $(seq 1 "${WARMUP_RUNS}"); do
  # The *measurements* are discarded, not the diagnostics: redirecting to
  # /dev/null would leave a failed warm-up reporting only its run number, and
  # the vLLM error that explains it is the whole content of the failure.
  if ! out="$(bench_once)"; then
    echo "warm-up run ${i} failed:" >&2
    echo "${out}" | tail -40 >&2
    exit 1
  fi
  failed="$(_failed_requests "${out}")"
  if [[ "${failed}" != "0" ]]; then
    echo "warm-up run ${i}: ${failed:-an unparseable} failed-request count;" >&2
    echo "AITER's GEMM shapes are likely still untuned. Refusing to proceed" >&2
    echo "into measured runs on a warm-up that did not actually warm anything." >&2
    echo "${out}" | tail -40 >&2
    exit 1
  fi
done

echo "model,performance,metric" > "${RESULT_CSV}"
echo "backend,model,model_revision,concurrency,isl,osl,num_prompts,run,total_tok_s,median_tpot_ms,p99_ttft_ms,warmup_discarded,mori_disable_topo" \
  > "${DETAIL_CSV}"

for i in $(seq 1 "${MEASURED_RUNS}"); do
  # Assigning from a command substitution that fails would abort here under
  # `set -e` with the captured output still sitting unused in ${out} -- the run
  # would die without ever showing why. Handle the status explicitly.
  if ! out="$(bench_once)"; then
    echo "run ${i}: benchmark exited nonzero:" >&2
    echo "${out}" | tail -40 >&2
    rm -f "${RESULT_CSV}" "${DETAIL_CSV}"
    exit 1
  fi
  echo "${out}" | tail -25
  # `exit` after the first hit, deliberately. Newer `vllm bench serve` prints an
  # automatic steady-state block in addition to the overall one, with the same
  # labels; without this each awk returns two newline-separated values, the
  # non-empty check still passes, and every CSV row becomes two malformed
  # physical rows. The wheel is supplied externally, so its output format is
  # not ours to pin.
  tput_val=$(echo "${out}" | awk '/Total token throughput/ {print $NF; exit}')
  tpot_val=$(echo "${out}" | awk '/Median TPOT/ {print $NF; exit}')
  ttft_val=$(echo "${out}" | awk '/P99 TTFT/ {print $NF; exit}')

  # `vllm bench serve` exits 0 while reporting failed requests, and still
  # prints the three metric lines -- as zeros. Ingested, those are
  # indistinguishable from a successful measurement, which is worse than no
  # data. The most likely cause is ISL + OSL exceeding MAX_MODEL_LEN, where
  # every request fails and the run looks merely slow.
  # Require an explicit zero. A missing field is not a pass: if the summary
  # format changes and drops or renames this line, treating absence as success
  # would silently disable the safeguard on exactly the runs it exists for.
  failed="$(_failed_requests "${out}")"
  if [[ "${failed}" != "0" ]]; then
    if [[ -z "${failed}" ]]; then
      echo "run ${i}: no 'Failed requests' field in the summary; cannot confirm" >&2
      echo "the run succeeded, so refusing to record a result." >&2
    else
      echo "run ${i}: ${failed} failed request(s); refusing to record a result." >&2
      echo "check that ISL(${ISL}) + OSL(${OSL}) < MAX_MODEL_LEN(${MAX_MODEL_LEN})." >&2
    fi
    echo "${out}" | tail -40 >&2
    rm -f "${RESULT_CSV}" "${DETAIL_CSV}"
    exit 1
  fi
  # An empty field means the summary did not parse -- a changed bench output
  # format, or a run that died mid-way. Either way it is not a zero.
  if [[ -z "${tput_val}" || -z "${tpot_val}" || -z "${ttft_val}" ]]; then
    echo "run ${i}: could not parse the benchmark summary; refusing to record a result." >&2
    echo "${out}" | tail -40 >&2
    rm -f "${RESULT_CSV}" "${DETAIL_CSV}"
    exit 1
  fi

  # Each run is its own set of rows rather than a pre-averaged one, so a run
  # that has not converged stays visible instead of being hidden in a mean.
  # MAD prefixes these labels with the model-card name, which is what carries
  # the backend and the model.
  _csv_row "run${i}_throughput" "${tput_val}" "total_token_throughput_tok_s" \
    >> "${RESULT_CSV}"
  _csv_row "run${i}_tpot" "${tpot_val}" "median_tpot_ms" >> "${RESULT_CSV}"
  _csv_row "run${i}_ttft" "${ttft_val}" "p99_ttft_ms" >> "${RESULT_CSV}"

  # MODEL, not MODEL_REPO: with MODEL_PATH set the server and the benchmark
  # both ran against the local path, and recording the repo id would label the
  # row with a model that was never loaded.
  _csv_row "${ALL2ALL_BACKEND}" "${MODEL}" "${MODEL_REVISION_LABEL}" \
    "${CONCURRENCY}" "${ISL}" "${OSL}" "${NUM_PROMPTS}" "${i}" \
    "${tput_val}" "${tpot_val}" "${ttft_val}" "${WARMUP_RUNS}" \
    "${MORI_DISABLE_TOPO:-}" >> "${DETAIL_CSV}"
done

echo "=== results (${RESULT_CSV}) ==="
cat "${RESULT_CSV}"
echo "=== detail (${DETAIL_CSV}) ==="
cat "${DETAIL_CSV}"
