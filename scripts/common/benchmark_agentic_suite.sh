#!/bin/bash
# AgentX suite driver: run a LIST of workloads in sequence against ONE served
# endpoint. Reads a config (agentic.yaml: serving + run + workloads[]) via the
# config loader, then per workload:
#   source=profile -> materialize_corpus() (generate + verify N/N pre-gate)
#   source=hf      -> resolve the --public-dataset loader (download at run time)
#   -> context_compat_check() -> build_replay_cmd() -> run into <RESULT_DIR>/<name>/
#      (optionally sweeping concurrency), then a combined suite summary.
#
# Serving is GLOBAL: one model/endpoint, N workloads. Env vars override the file;
# AGENTIC_WORKLOAD=<name> runs a single entry. DRY_RUN=1 prints the resolved
# N-workload plan + each per-workload command + context verdicts (no server).
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reuse an already-sourced lib (hook path) or source it ourselves (standalone).
if ! declare -F build_replay_cmd >/dev/null 2>&1; then
    _lib=""
    for _cand in "${AGENTIC_LIB:-}" "${_here}/agentic_lib.sh" "${_here}/common/agentic_lib.sh"; do
        if [ -n "$_cand" ] && [ -f "$_cand" ]; then _lib="$_cand"; break; fi
    done
    [ -n "$_lib" ] || { echo "[agentic][ERROR] agentic_lib.sh not found (set AGENTIC_LIB)" >&2; exit 1; }
    # shellcheck source=/dev/null
    source "$_lib"
fi

SUITE_PY="python3"
command -v "$SUITE_PY" >/dev/null 2>&1 || SUITE_PY="${AIPERF_PYTHON:-python3}"
CONFIG_TOOL="$AGENTX_DIR/agentx_config.py"

: "${AGENTIC_PORT:=2322}"
: "${DURATION:=900}"
RESULT_DIR="${RESULT_DIR:-/run_logs/${SLURM_JOB_ID:-0}}"

# Config arg: an explicit file, else rely on the AGENTIC_WORKLOAD synth.
_cfg_args=()
if [ -n "${AGENTIC_CONFIG:-}" ]; then
    [ -f "$AGENTIC_CONFIG" ] || agentic_die "AGENTIC_CONFIG not found: $AGENTIC_CONFIG"
    _cfg_args=(--config "$AGENTIC_CONFIG")
elif [ -z "${AGENTIC_WORKLOAD:-}" ]; then
    agentic_die "suite driver needs AGENTIC_CONFIG=<agentic.yaml> or AGENTIC_WORKLOAD=<name>"
fi

# Resolve global serving + run + the workload name list.
_cfg_shell="$("$SUITE_PY" "$CONFIG_TOOL" "${_cfg_args[@]}" --emit-config-shell)" \
    || agentic_die "config load failed"
eval "$_cfg_shell"

# Serving is global: apply non-'auto' serving values to the per-workload env.
if [ "${SUITE_SERVING_MODEL}" != "auto" ]; then MODEL="${MODEL:-$SUITE_SERVING_MODEL}"; fi
if [ "${SUITE_PORT}" != "auto" ]; then AGENTIC_PORT="$SUITE_PORT"; fi
if [ "${SUITE_SERVER_METRICS}" != "auto" ]; then AGENTIC_SERVER_METRICS="$SUITE_SERVER_METRICS"; fi
MAX_MODEL_LEN="${SUITE_MAX_MODEL_LEN}"

_is_dry=0
[ "${DRY_RUN:-0}" = "1" ] && _is_dry=1

mkdir -p "$SUITE_CORPUS_DIR"

if [ "$_is_dry" = "1" ]; then
    cat <<EOF
[agentic][DRY_RUN] resolved suite plan
  config                 : ${AGENTIC_CONFIG:-<synth AGENTIC_WORKLOAD=${AGENTIC_WORKLOAD:-}>}
  serving.model          : ${SUITE_SERVING_MODEL}
  serving.max_model_len  : ${SUITE_MAX_MODEL_LEN}
  serving.port           : ${SUITE_PORT} (AGENTIC_PORT=${AGENTIC_PORT})
  serving.server_metrics : ${SUITE_SERVER_METRICS}
  run.concurrency        : ${SUITE_CONCURRENCY}
  run.duration           : ${SUITE_DURATION}
  workloads (${SUITE_WORKLOAD_NAMES})
  RESULT_DIR             : ${RESULT_DIR}
  SUITE_CORPUS_DIR       : ${SUITE_CORPUS_DIR}
EOF
else
    install_agentic_deps
    wait_for_router_ready
    if [ -z "${MODEL:-}" ] || [ "${MODEL:-}" = "auto" ]; then resolve_served_model_name; fi
fi

# --------------------------------------------------------------------------
# Per-workload loop
# --------------------------------------------------------------------------
SUITE_SUMMARY_JSON="${RESULT_DIR}/suite_summary.json"
_summary_rows=()

for name in $SUITE_WORKLOAD_NAMES; do
    _profile_json="${SUITE_CORPUS_DIR}/${name}.profile.json"
    _wl_shell="$("$SUITE_PY" "$CONFIG_TOOL" "${_cfg_args[@]}" --workload "$name" \
                 --profile-out "$_profile_json" --emit-workload-shell)" \
        || agentic_die "workload resolve failed: $name"
    # Resets WL_* for this iteration.
    WL_LOADER=""; WL_PROFILE_FILE=""; WL_MODEL_TAG=""
    eval "$_wl_shell"

    # Per-workload trace source + env.
    CORPUS_DIR=""
    if [ "$WL_SOURCE" = "hf" ]; then
        WEKA_LOADER_OVERRIDE="$WL_LOADER"
    fi
    export WL_SOURCE CORPUS_DIR

    # Context compatibility vs the served window.
    context_compat_check "$name" "$WL_ISL_TAIL" "$MAX_MODEL_LEN"
    if [ "${CONTEXT_VERDICT}" = "SKIP" ]; then
        agentic_err "[$name] skipped (context)"
        _summary_rows+=("$name|SKIP(context)|-|-")
        continue
    fi
    export AGENTIC_MAX_CONTEXT_LENGTH

    # Materialize (generate + verify) for profile workloads.
    _verify_out=""
    if [ "$WL_SOURCE" = "profile" ]; then
        if [ "$_is_dry" = "1" ]; then
            CORPUS_DIR="${SUITE_CORPUS_DIR}/${name}"
        else
            # materialize_corpus runs in a subshell (command substitution), so set
            # CORPUS_DIR in THIS shell too (same deterministic path) — otherwise the
            # subshell's assignment is lost and --input-file is built empty.
            CORPUS_DIR="${SUITE_CORPUS_DIR}/${name}"
            _verify_out="$(materialize_corpus "$name" "$WL_PROFILE_FILE")"
            echo "$_verify_out"
        fi
    fi

    resolve_trace_loader

    # Concurrency sweep (single value => flat result dir; list => per-conc subdirs).
    _conc_list="$WL_CONCURRENCY"
    _n_conc=$(echo "$_conc_list" | wc -w)
    for conc in $_conc_list; do
        AGENTIC_CONC="$conc"
        DURATION="$WL_DURATION"
        if [ "$_n_conc" -gt 1 ]; then
            _rdir="${RESULT_DIR}/${name}/conc${conc}"
        else
            _rdir="${RESULT_DIR}/${name}"
        fi
        build_replay_cmd "$_rdir"
        if [ "$_is_dry" = "1" ]; then
            cat <<EOF

[agentic][DRY_RUN] workload='${name}' source='${WL_SOURCE}' conc=${conc} duration=${DURATION}
  context verdict        : ${CONTEXT_VERDICT} (--max-context-length ${AGENTIC_MAX_CONTEXT_LENGTH})
  trace source           : ${TRACE_SOURCE_FLAG}
  result dir             : ${_rdir}
  command:
${REPLAY_CMD}
EOF
        else
            run_agentic_replay_and_write_outputs "$_rdir" || true
        fi
    done

    if [ "$_is_dry" != "1" ]; then
        _summary_rows+=("$name|${WL_SOURCE}|${CONTEXT_VERDICT}|${_rdir}")
    fi
done

# --------------------------------------------------------------------------
# Combined suite summary (real runs)
# --------------------------------------------------------------------------
if [ "$_is_dry" != "1" ]; then
    agentic_log "==== suite summary ===="
    mkdir -p "$RESULT_DIR"
    RESULT_DIR="$RESULT_DIR" SUITE_WORKLOAD_NAMES="$SUITE_WORKLOAD_NAMES" \
    "$SUITE_PY" - "$SUITE_SUMMARY_JSON" <<'PY' || true
import json, os, sys, glob
out_path = sys.argv[1]
result_dir = os.environ["RESULT_DIR"]
names = os.environ.get("SUITE_WORKLOAD_NAMES", "").split()
rows = []
for name in names:
    wl_dir = os.path.join(result_dir, name)
    rec = {"workload": name, "result_dir": wl_dir}
    agg = sorted(glob.glob(os.path.join(wl_dir, "**", "*.json"), recursive=True))
    conformance = err = cache = None
    for f in agg:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rm = d.get("request_metrics") or {}
        sm = d.get("server_metrics") or {}
        if isinstance(rm.get("cache"), dict):
            conformance = rm["cache"].get("theoretical_cache_hit_rate", conformance)
        if isinstance(sm.get("cache"), dict):
            cache = sm["cache"].get("gpu_cache_hit_rate", cache)
        ra = d.get("request_accounting") or {}
        prof = ra.get("records_profiled")
        drop = ra.get("records_error_dropped")
        if isinstance(prof, (int, float)) and isinstance(drop, (int, float)) and (prof + drop) > 0:
            err = drop / (prof + drop)
    if os.path.exists(os.path.join(wl_dir, "RUN_INVALID.json")):
        rec["status"] = "INVALID"
    rec["theoretical_cache_hit_rate"] = conformance
    rec["gpu_cache_hit_rate"] = cache
    rec["error_rate"] = err
    rows.append(rec)
json.dump({"workloads": rows}, open(out_path, "w"), indent=2)
print(f"{'workload':<16}{'err_rate':>10}{'cache_hit':>12}{'theo_hit':>12}")
print("-" * 50)
for r in rows:
    er = "-" if r["error_rate"] is None else f"{r['error_rate']*100:.1f}%"
    ch = "-" if r["gpu_cache_hit_rate"] is None else f"{r['gpu_cache_hit_rate']*100:.1f}%"
    th = "-" if r["theoretical_cache_hit_rate"] is None else f"{r['theoretical_cache_hit_rate']*100:.1f}%"
    print(f"{r['workload']:<16}{er:>10}{ch:>12}{th:>12}")
print("-" * 50)
print(f"suite summary JSON -> {out_path}")
PY
    agentic_log "suite complete -> $RESULT_DIR"
fi
