#!/bin/bash
# Agentic trace-replay helpers (aiperf driver) for the disaggregated P/D launchers.
#
# Ported from SemiAnalysisAI/InferenceX benchmarks/benchmark_lib.sh (agentic
# section). Drives aiperf's `inferencex-agentx-mvp` scenario against a live
# disagg endpoint (SGLang router :2322 or vLLM proxy) and writes aiperf
# artifacts + an aggregate JSON + plots. Does NOT touch parse_to_csv.py/perf.csv.
#
# The toolkit (customized aiperf + agentx scenario + WEKA trace loaders +
# result aggregator) is installed at RUN TIME into a throwaway uv venv that is
# kept OUT of the inference server's site-packages, from pinned commits.
#
# Consumed env (set by the caller / launcher):
#   MODEL_PATH            path the server was launched with (required)
#   MODEL                 served model name aiperf sends (auto-resolved if unset)
#   AGENTIC_PORT          endpoint port (default 2322 for sglang router)
#   AGENTIC_CONC          session-tree concurrency (default 16)
#   DURATION              measurement window seconds (default 120; scenario min 900)
#   RESULT_DIR            output dir (default /run_logs/$SLURM_JOB_ID)
#   MODEL_PREFIX          model family key for trace-loader default (optional)
#   WEKA_LOADER_OVERRIDE  pin a specific trace loader (optional)
#   DRY_RUN=1             print the assembled command + resolved values, then exit 0
#
# Pins (Phase 0 blocker): concrete commits, overridable by env. Bump by editing
# these defaults after re-validating against a fresh smoke run.
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_PIN="${AGENTIC_UTILS_PIN:-ef8a17ecf0c3679dc12020eddab3c1a36d285b58}"   # InferenceX main @ 2026-07-25
AIPERF_PIN="${AIPERF_PIN:-0d2aa0572ac685943d38c580675c4a61023581d3}"             # utils/aiperf submodule (cquil11/aiperf-agentx-v1.0)

set -o pipefail

agentic_log()  { echo "[agentic] $*"; }
agentic_err()  { echo "[agentic][ERROR] $*" >&2; }
agentic_die()  { agentic_err "$*"; exit 1; }

# --------------------------------------------------------------------------
# Runtime install (isolated uv venv, pinned sources)
# --------------------------------------------------------------------------
AGENTIC_RUNTIME_DIR="${AGENTIC_RUNTIME_DIR:-${TMPDIR:-/tmp}/mad-agentic-${SLURM_JOB_ID:-$$}}"
INFMAX_WS="${INFMAX_WS:-${AGENTIC_RUNTIME_DIR}/InferenceX}"
AIPERF_VENV="${AIPERF_VENV:-${AGENTIC_RUNTIME_DIR}/venv}"
AIPERF_UV_INSTALL_DIR="${AIPERF_UV_INSTALL_DIR:-${AGENTIC_RUNTIME_DIR}/uv/bin}"
AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-${AGENTIC_RUNTIME_DIR}/uv-cache}"
AIPERF_PYTHON="${AIPERF_VENV}/bin/python"
AIPERF_CLI="${AIPERF_VENV}/bin/aiperf"
AIPERF_HF_CLI="${AIPERF_VENV}/bin/hf"
AIPERF_DEPS_READY=0
AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"

ensure_agentic_uv() {
    if command -v uv >/dev/null 2>&1; then
        AIPERF_UV_BIN="$(command -v uv)"
        return
    fi
    AIPERF_UV_BIN="${AIPERF_UV_INSTALL_DIR}/uv"
    if [ ! -x "$AIPERF_UV_BIN" ]; then
        mkdir -p "$AIPERF_UV_INSTALL_DIR"
        curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$AIPERF_UV_INSTALL_DIR" sh
    fi
    [ -x "$AIPERF_UV_BIN" ] || agentic_die "uv installation did not create $AIPERF_UV_BIN"
}

_clone_inferencex_pinned() {
    if [ -d "$INFMAX_WS/.git" ]; then
        agentic_log "InferenceX checkout already present at $INFMAX_WS"
        return
    fi
    command -v git >/dev/null 2>&1 || { apt-get update && apt-get install -y git; }
    mkdir -p "$(dirname "$INFMAX_WS")"
    agentic_log "Cloning InferenceX @ $INFERENCEX_PIN (+ aiperf submodule @ $AIPERF_PIN)"
    git clone --filter=blob:none "$INFERENCEX_REPO" "$INFMAX_WS"
    git -C "$INFMAX_WS" checkout --quiet "$INFERENCEX_PIN"
    # Pull the aiperf submodule at the commit InferenceX pins (holds the
    # inferencex-agentx-mvp scenario + semianalysis_cc_traces_weka loaders).
    git -C "$INFMAX_WS" submodule update --init --recursive utils/aiperf
    local got
    got="$(git -C "$INFMAX_WS/utils/aiperf" rev-parse HEAD 2>/dev/null)"
    if [ "$got" != "$AIPERF_PIN" ]; then
        agentic_log "WARN: aiperf submodule at $got, expected $AIPERF_PIN (InferenceX pin drift)"
    fi
}

install_agentic_deps() {
    [ "$AIPERF_DEPS_READY" = "1" ] && return
    ensure_agentic_uv
    _clone_inferencex_pinned

    # aiperf must NOT share site-packages with the inference server: installing
    # it into SGLang/vLLM's Python can upgrade fastapi/starlette/transformers
    # under the live server. Build a throwaway venv instead.
    rm -rf "$AIPERF_VENV"
    mkdir -p "$AIPERF_UV_CACHE_DIR"
    UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" "$AIPERF_UV_BIN" venv --python "$(command -v python3)" "$AIPERF_VENV"
    UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" "$AIPERF_UV_BIN" pip install --python "$AIPERF_PYTHON" \
        -r "$INFMAX_WS/utils/agentic-benchmark/requirements.txt" \
        -e "$INFMAX_WS/utils/aiperf" \
        "datasets>=4.7.0" "huggingface_hub[cli]>=0.25.0" urllib3 requests

    [ -x "$AIPERF_CLI" ] && [ -x "$AIPERF_HF_CLI" ] || \
        agentic_die "isolated aiperf environment incomplete at $AIPERF_VENV"
    AIPERF_DEPS_READY=1
}

# --------------------------------------------------------------------------
# Trace source resolution (loader name is pure; download is retried)
# --------------------------------------------------------------------------
# Sets TRACE_LOADER + TRACE_DATASET; does no I/O so DRY_RUN can call it.
resolve_trace_loader() {
    local default_loader
    case "${MODEL_PREFIX:-}" in
        dsv4*|deepseek*|DeepSeek*|glm5*|minimaxm3*)
            default_loader="semianalysis_cc_traces_weka_062126" ;;      # 1M-ctx families: full corpus
        *)
            default_loader="semianalysis_cc_traces_weka_062126_256k" ;; # shorter-ctx: 256k-capped
    esac
    TRACE_LOADER="${WEKA_LOADER_OVERRIDE:-$default_loader}"
    case "$TRACE_LOADER" in
        semianalysis_cc_traces_weka_062126)      TRACE_DATASET="semianalysisai/cc-traces-weka-062126" ;;
        semianalysis_cc_traces_weka_062126_256k) TRACE_DATASET="semianalysisai/cc-traces-weka-062126-256k" ;;
        semianalysis_cc_traces_weka_061526)      TRACE_DATASET="semianalysisai/cc-traces-weka-061526" ;;
        semianalysis_cc_traces_weka_061526_256k) TRACE_DATASET="semianalysisai/cc-traces-weka-061526-256k" ;;
        *) agentic_die "unknown WEKA_LOADER_OVERRIDE='$TRACE_LOADER' (see resolve_trace_loader)";;
    esac
    TRACE_SOURCE_FLAG="--public-dataset $TRACE_LOADER"
}

# Download the dataset into the shared HF cache with retries (3 attempts,
# 900s each, backoff). Fails the run only after all attempts.
resolve_trace_source() {
    resolve_trace_loader
    agentic_log "Trace loader: $TRACE_LOADER ($TRACE_DATASET)"
    local attempts="${AGENTIC_TRACE_DL_ATTEMPTS:-3}"
    local per_timeout="${AGENTIC_TRACE_DL_TIMEOUT:-900}"
    local i backoff=30
    for ((i = 1; i <= attempts; i++)); do
        agentic_log "trace download attempt $i/$attempts (timeout ${per_timeout}s)"
        if timeout "$per_timeout" "$AIPERF_HF_CLI" download --repo-type dataset "$TRACE_DATASET"; then
            return 0
        fi
        agentic_log "attempt $i failed"
        [ "$i" -lt "$attempts" ] && { sleep "$backoff"; backoff=$((backoff * 2)); }
    done
    agentic_die "trace download failed after $attempts attempts ($TRACE_DATASET)"
}

# --------------------------------------------------------------------------
# Endpoint helpers: model-name alignment + router readiness (CRITICAL)
# --------------------------------------------------------------------------
# aiperf --model MUST equal the server's registered served-model name or every
# request 404s. Prefer the router's advertised id; fall back to basename.
resolve_served_model_name() {
    local base="http://127.0.0.1:${AGENTIC_PORT}"
    local name=""
    name="$(curl -sf "${base}/v1/models" 2>/dev/null \
            | "${AIPERF_PYTHON:-python3}" -c 'import sys,json;
d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' 2>/dev/null)"
    if [ -z "$name" ]; then
        name="$(basename "${MODEL_PATH:-}")"
        agentic_log "served-model name not advertised; falling back to basename: $name"
    fi
    [ -n "$name" ] || agentic_die "could not resolve a served model name (set MODEL explicitly)"
    MODEL="$name"
    agentic_log "aiperf --model resolved to: $MODEL"
}

wait_for_router_ready() {
    local base="http://127.0.0.1:${AGENTIC_PORT}"
    local max="${AGENTIC_ROUTER_READY_TIMEOUT:-600}"
    local waited=0 step=5
    agentic_log "waiting for endpoint readiness at $base (max ${max}s)"
    while (( waited < max )); do
        if curl -sf "${base}/v1/models" >/dev/null 2>&1; then
            agentic_log "endpoint ready after ${waited}s"; return 0
        fi
        sleep "$step"; waited=$((waited + step))
    done
    agentic_die "endpoint not ready after ${max}s at $base"
}

# --------------------------------------------------------------------------
# aiperf command assembly
# --------------------------------------------------------------------------
build_replay_cmd() {
    local result_dir="$1"
    local duration="${DURATION:-120}"
    local conc="${AGENTIC_CONC:-16}"
    # Model-size-aware cache warmup: DeepSeek/large families need a longer warm.
    local cache_warmup
    case "${MODEL_PREFIX:-}${MODEL:-}${MODEL_PATH:-}" in
        *[Dd]eep[Ss]eek*|*dsv4*|*[Kk]imi*|*[Gg][Ll][Mm]*) cache_warmup="${AGENTIC_CACHE_WARMUP_DURATION:-300}" ;;
        *) cache_warmup="${AGENTIC_CACHE_WARMUP_DURATION:-60}" ;;
    esac

    export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
    export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

    REPLAY_CMD="$AIPERF_CLI profile --scenario inferencex-agentx-mvp"
    REPLAY_CMD+=" --url http://localhost:${AGENTIC_PORT}"
    REPLAY_CMD+=" --endpoint /v1/chat/completions --endpoint-type chat --streaming"
    REPLAY_CMD+=" --model $MODEL"
    REPLAY_CMD+=" --concurrency $conc"
    REPLAY_CMD+=" --benchmark-duration $duration"
    REPLAY_CMD+=" --random-seed 42"
    REPLAY_CMD+=" --failed-request-threshold $AIPERF_FAILED_REQUEST_THRESHOLD"
    REPLAY_CMD+=" --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75"
    REPLAY_CMD+=" --agentic-cache-warmup-duration $cache_warmup"
    REPLAY_CMD+=" --warmup-grace-period ${AGENTIC_WARMUP_GRACE_PERIOD:-1800}"
    REPLAY_CMD+=" --use-server-token-count --tokenizer-trust-remote-code"
    REPLAY_CMD+=" --no-gpu-telemetry --slice-duration 1.0"
    if [ -n "${AGENTIC_SERVER_METRICS:-}" ]; then
        REPLAY_CMD+=" --server-metrics ${AGENTIC_SERVER_METRICS}"
    fi
    REPLAY_CMD+=" --num-dataset-entries ${AGENTIC_NUM_DATASET_ENTRIES:-393}"
    if [ -n "${MAX_MODEL_LEN:-}" ] && [ "${MAX_MODEL_LEN}" != "0" ]; then
        REPLAY_CMD+=" --max-context-length $MAX_MODEL_LEN"
    fi
    REPLAY_CMD+=" --output-artifact-dir $result_dir/aiperf_artifacts"
    # The scenario enforces a 900s minimum; smoke runs opt into --unsafe-override
    # (marks submission_valid=false, expected for non-canonical runs).
    if [ "$duration" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE:-false}" = "true" ]; then
        REPLAY_CMD+=" --unsafe-override"
    fi
    REPLAY_CMD+=" $TRACE_SOURCE_FLAG"
}

# --------------------------------------------------------------------------
# Run + aggregate + rollback
# --------------------------------------------------------------------------
write_agentic_result_json() {
    local result_dir="$1"
    # process_agentic_result reads aiperf artifacts from RESULT_DIR and writes
    # $AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json (RESULT_FILENAME is required).
    local result_filename="${AGENTIC_RESULT_FILENAME:-agentic_${SLURM_JOB_ID:-0}_xP${xP:-1}_yD${yD:-1}_${MODEL_NAME:-model}}"
    AGENTIC_RESULT_JSON="${AGENTIC_OUTPUT_DIR:-$result_dir}/${result_filename}.json"
    # process_agentic_result requires KV_OFFLOADING; "none" is the no-offload case
    # (and requires KV_OFFLOAD_BACKEND to be empty). All other metadata env vars
    # default cleanly, so the aggregate JSON's metrics are unaffected.
    ( cd "$INFMAX_WS" && \
      RESULT_DIR="$result_dir" AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$result_dir}" \
      RESULT_FILENAME="$result_filename" \
      KV_OFFLOADING="${KV_OFFLOADING:-none}" \
      "$AIPERF_PYTHON" -m utils.agentic.aggregation.process_agentic_result )
    "$AIPERF_PYTHON" "$INFMAX_WS/utils/generate_aiperf_plots.py" "$result_dir" 2>&1 || true
    agentic_log "aggregate JSON: $AGENTIC_RESULT_JSON"
}

run_agentic_replay_and_write_outputs() {
    local result_dir="$1"
    local replay_rc
    mkdir -p "$result_dir"
    echo "$REPLAY_CMD" > "$result_dir/benchmark_command.txt"

    $REPLAY_CMD 2>&1 | tee "$result_dir/benchmark.log"
    replay_rc=${PIPESTATUS[0]}

    write_agentic_result_json "$result_dir"

    # Best-effort post-benchmark health check (PASS/WARN on error + cache-hit).
    local _validator="$(dirname "${BASH_SOURCE[0]}")/validate_agentic_result.sh"
    if [ -f "$_validator" ] && [ -n "${AGENTIC_RESULT_JSON:-}" ]; then
        AIPERF_PYTHON="$AIPERF_PYTHON" bash "$_validator" "$AGENTIC_RESULT_JSON" || true
    fi

    if [ "$replay_rc" -ne 0 ]; then
        # Automated rollback: mark the run invalid, leave logs for triage.
        echo '{"submission_valid": false, "reason": "replay_rc='"$replay_rc"'"}' \
            > "$result_dir/RUN_INVALID.json"
        agentic_err "agentic replay exited $replay_rc (results written, run marked invalid)"
        return "$replay_rc"
    fi
    agentic_log "agentic replay complete -> $result_dir"
}

# --------------------------------------------------------------------------
# DRY_RUN: resolve everything possible without contacting a server, print, exit
# --------------------------------------------------------------------------
agentic_dry_run() {
    local result_dir="$1"
    resolve_trace_loader
    : "${AIPERF_CLI:=aiperf}"
    if [ -z "${MODEL:-}" ]; then MODEL="$(basename "${MODEL_PATH:-<unset>}")"; fi
    build_replay_cmd "$result_dir"
    cat <<EOF
[agentic][DRY_RUN] resolved configuration
  MODEL (aiperf --model) : $MODEL
  MODEL_PATH             : ${MODEL_PATH:-<unset>}
  AGENTIC_PORT           : ${AGENTIC_PORT}
  AGENTIC_CONC           : ${AGENTIC_CONC:-16}
  DURATION               : ${DURATION:-120}
  trace loader / dataset : ${TRACE_LOADER} / ${TRACE_DATASET}
  RESULT_DIR             : ${result_dir}
  InferenceX pin         : ${INFERENCEX_PIN}
  aiperf pin             : ${AIPERF_PIN}

[agentic][DRY_RUN] assembled command:
${REPLAY_CMD}
EOF
}
