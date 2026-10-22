#!/bin/bash
# replica_entry.sh — runs INSIDE one replica's container.
# Starts the engine server on $PORT, waits healthy, then runs ACTION.
# Env (from launcher): ENGINE_KIND, MODEL_PATH, PORT, TP, SERVE_FLAGS, SCHED, ACTION
#   ACTION=sanity -> 5-prompt accuracy + perf smoke (conc32 128/128)
#   ACTION=serve  -> boot + stay up (for router-fronted serving)
set -uo pipefail
SC=/scripts
# IMPORTANT: do NOT source lib_inferencex.sh before launching the server.
# Sourcing it perturbs the env (PYTHON*/cache state) and makes ATOM's torch.distributed.nn
# JIT remote_module instantiation crash with `IndexError: string index out of range` in
# importlib cache_from_source. ATOM works perfectly when launched in a clean env (== the
# bare runbook command). So: launch server FIRST, source the bench lib AFTER.

KIND="${ENGINE_KIND:-vllm}"
MODEL_PATH="${MODEL_PATH:?}"; PORT="${PORT:?}"; TP="${TP:?}"
ACTION="${ACTION:-sanity}"
SERVER_LOG=/out/server.log
# Shared logical knobs (from model.yaml `defaults`, resolved+passed by run_engine.sh).
# Translated below to each engine's specific flag name so ALL engines get the SAME values.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

start_server() {
  local sched="--no-async-scheduling"; [[ "$TP" -gt 1 ]] && sched="--async-scheduling"
  case "$KIND" in
    vllm)
      vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs "$MAX_NUM_SEQS" $sched ${SERVE_FLAGS:-} > "$SERVER_LOG" 2>&1 & ;;
    sglang)
      # SGLang flag names differ: --context-length (ctx), --mem-fraction-static (mem-util), --max-running-requests (seqs)
      python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
        --tensor-parallel-size "$TP" --context-length "$MAX_MODEL_LEN" \
        --mem-fraction-static "$GPU_MEM_UTIL" --max-running-requests "$MAX_NUM_SEQS" \
        ${SERVE_FLAGS:-} > "$SERVER_LOG" 2>&1 & ;;
    atom)
      # bare runbook invocation, clean env. Same shared knobs as vLLM (matching flag names).
      python -m atom.entrypoints.openai_server --model "$MODEL_PATH" -tp "$TP" --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs "$MAX_NUM_SEQS" ${SERVE_FLAGS:-} > "$SERVER_LOG" 2>&1 & ;;
    *) echo "bad ENGINE_KIND=$KIND"; exit 2 ;;
  esac
  SERVER_PID=$!
}

# 1) launch server in CLEAN env
start_server

# 2) NOW source the InferenceX helpers (health-wait + bench) — after the server process exists
source "$SC/lib/lib_inferencex.sh"
export EVAL_ONLY=false RUN_EVAL=false PROFILE=0
export SGLANG_TORCH_PROFILER_DIR="" VLLM_TORCH_PROFILER_DIR="" GPU_MONITOR_PID=""

cleanup(){ [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
echo "[replica] server healthy on :$PORT"

if [[ "$ACTION" == "serve" ]]; then
  echo "[replica] ACTION=serve — staying up"; wait "$SERVER_PID"; exit 0
fi

# ACTION=sweep: accuracy gate, then loop SWEEP_SHAPES x SWEEP_CONC through the bench client.
# For BENCH_EXTERNAL engines (ATOM), signal ready + hold; run_engine.sh drives the external sweep.
if [[ "$ACTION" == "sweep" ]]; then
  PORT="$PORT" MODEL="$MODEL_PATH" OUT=/out/sanity_accuracy.json python3 "$SC/lib/check_accuracy.py" || true
  if [[ "${BENCH_EXTERNAL:-0}" == "1" ]]; then
    echo "[replica] sweep + BENCH_EXTERNAL=1 — ready; external sweep driven by host"; touch /out/SERVER_READY
    wait "$SERVER_PID"; exit 0
  fi
  SWEEP_SHAPES="${SWEEP_SHAPES:-1024,1024 8192,1024 16384,1024}"
  SWEEP_CONC="${SWEEP_CONC:-4 8 16 32 64 128 256}"
  for shape in $SWEEP_SHAPES; do
    ISL=${shape%,*}; OSL=${shape#*,}
    for c in $SWEEP_CONC; do
      echo "[sweep] isl=$ISL osl=$OSL conc=$c"
      run_benchmark_serving --model "$MODEL_PATH" --port "$PORT" --backend vllm \
        --input-len "$ISL" --output-len "$OSL" --random-range-ratio 1.0 \
        --num-prompts "$((c * 10))" --max-concurrency "$c" \
        --result-filename "sweep_isl${ISL}_osl${OSL}_c${c}" --result-dir /out \
        --bench-serving-dir "$SC" || echo "[sweep] FAILED isl=$ISL osl=$OSL conc=$c"
    done
  done
  echo "[replica] sweep done"; exit 0
fi

# ACTION=sanity: accuracy (always — pure urllib, safe in any image) then perf smoke.
SMOKE_CONC="${SMOKE_CONC:-8}"; SMOKE_ISL="${SMOKE_ISL:-128}"; SMOKE_OSL="${SMOKE_OSL:-128}"
PORT="$PORT" MODEL="$MODEL_PATH" OUT=/out/sanity_accuracy.json python3 "$SC/lib/check_accuracy.py" || true

# BENCH_EXTERNAL=1 (e.g. ATOM): the vendored bench client crashes in this image's Python env
# (importlib cache_from_source IndexError). Skip in-container bench; the host (run_engine.sh)
# runs it from a clean container against this server's port, then this replica stays up until killed.
if [[ "${BENCH_EXTERNAL:-0}" == "1" ]]; then
  echo "[replica] BENCH_EXTERNAL=1 — server ready for external bench; holding on :$PORT"
  touch /out/SERVER_READY
  wait "$SERVER_PID"; exit 0
fi

# NOTE: run_benchmark_serving (lib_inferencex.sh) already injects --num-warmups=2*conc internally;
# do NOT pass --num-warmups here (its arg parser rejects it -> "Unknown parameter").
run_benchmark_serving --model "$MODEL_PATH" --port "$PORT" --backend vllm \
  --input-len "$SMOKE_ISL" --output-len "$SMOKE_OSL" --random-range-ratio 1.0 \
  --num-prompts "$((SMOKE_CONC * 10))" --max-concurrency "$SMOKE_CONC" \
  --result-filename "perf_c${SMOKE_CONC}_${SMOKE_ISL}x${SMOKE_OSL}" --result-dir /out \
  --bench-serving-dir "$SC" || true
echo "[replica] sanity done"
