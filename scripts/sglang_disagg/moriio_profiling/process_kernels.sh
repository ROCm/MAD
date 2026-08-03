#!/bin/bash
# Strict post-run orchestration for integrated SGLang MoRI I/O profiling.
set -euo pipefail

usage() {
    echo "usage: process_kernels.sh JOBID" >&2
    echo "       process_kernels.sh {run|verify|trace|analyze} JOBID" >&2
    exit 2
}

case "${1:-}" in
    run|verify|trace|analyze)
        OP="$1"
        J="${2:-}"
        [ "$#" -eq 2 ] || usage
        ;;
    "")
        usage
        ;;
    *)
        OP="run"
        J="$1"
        [ "$#" -eq 1 ] || usage
        ;;
esac

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HR="${HR:-/shared_inference/${USER:-aarai}/model_blog_logs/$J}"
XP="${XP:-1}"
YD="${YD:-1}"
E="${ROCPROF_EXPECT_PER_NODE:-8}"
OUT="${OUT:-$D/artifacts/pull_${J}}"
TOOLS="${TRACE_TOOLS:-$D/trace_tools.py}"
RUN_MORI="${RUN_MORI:-0}"

MULTI_SWEEP=0
SWEEP_KEYS=()
SWEEP_CLIENTS=()
SWEEP_MANIFESTS=()
SWEEP_PREFIXES=()
SWEEP_OUTPUTS=()

PREFILL_DIRS=()
PREFILL_LOGS=()
DECODE_DIRS=()
DECODE_LOGS=()
DIRS=()
for ((i=0; i<XP; i++)); do
    PREFILL_DIRS+=("$HR/rocprof_prefill_NODE$i")
    PREFILL_LOGS+=("$HR/prefill_NODE$i.log")
done
for ((i=0; i<YD; i++)); do
    node=$((XP+i))
    DECODE_DIRS+=("$HR/rocprof_decode_NODE$node")
    DECODE_LOGS+=("$HR/decode_NODE$node.log")
done
DIRS=("${PREFILL_DIRS[@]}" "${DECODE_DIRS[@]}")

pid_of() {
    local base
    base=$(basename "$1" "$2")
    echo "${base##*_}"
}

verify_capture() {
    local grand_expected=0 grand_ok=0 any_bad=0
    local dir name role pid missing count suffix
    declare -A role_expected role_ok

    echo "===== [verify] job=$J xP=$XP yD=$YD workers/node=$E ====="
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "[verify] ERROR: missing $dir" >&2
            return 2
        fi
        name=$(basename "$dir")
        role="prefill"
        [[ "$name" == *decode* ]] && role="decode"

        unset kernels markers results
        declare -A kernels=(["__none"]=1)
        declare -A markers=(["__none"]=1)
        declare -A results=(["__none"]=1)
        local kernel_count=0 marker_count=0 result_count=0
        shopt -s nullglob
        for file in "$dir"/*_kernel_trace.csv; do
            kernels["$(pid_of "$file" _kernel_trace.csv)"]=1
            kernel_count=$((kernel_count+1))
        done
        for file in "$dir"/*_marker_api_trace.csv; do
            markers["$(pid_of "$file" _marker_api_trace.csv)"]=1
            marker_count=$((marker_count+1))
        done
        for file in "$dir"/*_results.json; do
            results["$(pid_of "$file" _results.json)"]=1
            result_count=$((result_count+1))
        done
        shopt -u nullglob

        for suffix in kernel_trace.csv marker_api_trace.csv results.json; do
            case "$suffix" in
                kernel_trace.csv) count=$kernel_count ;;
                marker_api_trace.csv) count=$marker_count ;;
                results.json) count=$result_count ;;
            esac
            if (( count < E )); then
                echo "[verify] ERROR: $name has $count/$E $suffix files" >&2
                any_bad=1
            fi
        done

        declare -A all_pids=(["__none"]=1)
        for pid in "${!kernels[@]}" "${!markers[@]}" "${!results[@]}"; do
            [[ -n "$pid" && "$pid" != "__none" ]] && all_pids["$pid"]=1
        done
        local ok=0 details="" pid_count=$(( ${#all_pids[@]} - 1 ))
        for pid in "${!all_pids[@]}"; do
            [ "$pid" = "__none" ] && continue
            missing=""
            [ -n "${kernels[$pid]:-}" ] || missing="${missing}kernel,"
            [ -n "${markers[$pid]:-}" ] || missing="${missing}marker,"
            [ -n "${results[$pid]:-}" ] || missing="${missing}results,"
            if [ -z "$missing" ]; then
                ok=$((ok+1))
            else
                details="$details $pid[missing:${missing%,}]"
            fi
        done
        grand_expected=$((grand_expected+E))
        grand_ok=$((grand_ok+ok))
        role_expected["$role"]=$(( ${role_expected[$role]:-0} + E ))
        role_ok["$role"]=$(( ${role_ok[$role]:-0} + ok ))
        if (( ok == E && pid_count == E )); then
            echo "  [OK]  $name: $ok/$E workers complete"
        else
            echo "  [BAD] $name: $ok/$E complete; pids=$pid_count$details" >&2
            any_bad=1
        fi
    done
    for role in prefill decode; do
        [ -n "${role_expected[$role]:-}" ] || continue
        echo "[verify] ${role^^}: ${role_ok[$role]:-0}/${role_expected[$role]}"
    done
    if (( any_bad )); then
        echo "[verify] ERROR: capture incomplete ($grand_ok/$grand_expected)" >&2
        return 1
    fi
    echo "[verify] OK: all $grand_ok/$grand_expected workers complete"
}

sha256_file() {
    python3 - "$1" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for block in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(block)
print(h.hexdigest())
PY
}

manifest_prefix() {
    python3 - "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    manifest = json.load(fh)
prefix = manifest.get("request_id_prefix")
if not isinstance(prefix, str) or not prefix:
    raise SystemExit("missing or invalid request_id_prefix")
print(prefix)
PY
}

discover_sweeps() {
    local fixed_csv="$HR/rocprof_probe_client.csv"
    local fixed_manifest="$HR/rocprof_probe_manifest.json"
    local keyed_csvs=() keyed_manifests=() file key rid_prefix expected_prefix
    declare -A clients=() manifests=() prefixes=()
    SWEEP_KEYS=()
    SWEEP_CLIENTS=()
    SWEEP_MANIFESTS=()
    SWEEP_PREFIXES=()
    SWEEP_OUTPUTS=()
    MULTI_SWEEP=0
    shopt -s nullglob
    keyed_csvs=("$HR"/rocprof_probe_client_i*_isl*_osl*_c*.csv)
    keyed_manifests=("$HR"/rocprof_probe_manifest_i*_isl*_osl*_c*.json)
    shopt -u nullglob
    if [[ -e "$fixed_csv" && -e "$fixed_manifest" && ( ${#keyed_csvs[@]} -gt 0 || ${#keyed_manifests[@]} -gt 0 ) ]]; then
        echo "[process_kernels.sh] ERROR: fixed and keyed request-correlation artifacts are ambiguous" >&2
        return 1
    fi
    if [[ -e "$fixed_csv" || -e "$fixed_manifest" ]]; then
        if [[ ! -e "$fixed_csv" || ! -e "$fixed_manifest" ]]; then
            echo "[process_kernels.sh] ERROR: orphan fixed request-correlation artifact" >&2
            return 1
        fi
        if [[ ! -s "$fixed_csv" || ! -s "$fixed_manifest" ]]; then
            echo "[process_kernels.sh] ERROR: empty fixed request-correlation artifact" >&2
            return 1
        fi
        if ! rid_prefix=$(manifest_prefix "$fixed_manifest"); then
            echo "[process_kernels.sh] ERROR: invalid request_id_prefix in $fixed_manifest" >&2
            return 1
        fi
        if [[ "$rid_prefix" != "profile-${J}" ]]; then
            echo "[process_kernels.sh] ERROR: fixed manifest request_id_prefix mismatch: $rid_prefix" >&2
            return 1
        fi
        SWEEP_KEYS=("fixed")
        SWEEP_CLIENTS=("$fixed_csv")
        SWEEP_MANIFESTS=("$fixed_manifest")
        SWEEP_PREFIXES=("$rid_prefix")
        SWEEP_OUTPUTS=("$OUT")
        return 0
    fi
    if (( ${#keyed_csvs[@]} == 0 && ${#keyed_manifests[@]} == 0 )); then
        echo "[process_kernels.sh] ERROR: normal benchmark request-correlation outputs are missing" >&2
        return 1
    fi
    for file in "${keyed_csvs[@]}"; do
        if [[ ! -s "$file" ]]; then
            echo "[process_kernels.sh] ERROR: empty keyed client CSV: $file" >&2
            return 1
        fi
        key=$(basename "$file")
        key=${key#rocprof_probe_client_}
        key=${key%.csv}
        if [[ ! "$key" =~ ^i[1-9][0-9]*_isl[0-9]+_osl[0-9]+_c[0-9]+$ ]]; then
            echo "[process_kernels.sh] ERROR: invalid client sweep key: $key" >&2
            return 1
        fi
        if [[ -n "${clients[$key]+x}" ]]; then
            echo "[process_kernels.sh] ERROR: duplicate client sweep key: $key" >&2
            return 1
        fi
        clients[$key]="$file"
    done
    for file in "${keyed_manifests[@]}"; do
        if [[ ! -s "$file" ]]; then
            echo "[process_kernels.sh] ERROR: empty keyed manifest: $file" >&2
            return 1
        fi
        key=$(basename "$file")
        key=${key#rocprof_probe_manifest_}
        key=${key%.json}
        if [[ ! "$key" =~ ^i[1-9][0-9]*_isl[0-9]+_osl[0-9]+_c[0-9]+$ ]]; then
            echo "[process_kernels.sh] ERROR: invalid manifest sweep key: $key" >&2
            return 1
        fi
        if [[ -n "${manifests[$key]+x}" ]]; then
            echo "[process_kernels.sh] ERROR: duplicate manifest sweep key: $key" >&2
            return 1
        fi
        manifests[$key]="$file"
    done
    for file in "${keyed_csvs[@]}"; do
        key=$(basename "$file")
        key=${key#rocprof_probe_client_}
        key=${key%.csv}
        if [[ -z "${manifests[$key]+x}" ]]; then
            echo "[process_kernels.sh] ERROR: orphan client CSV for sweep $key" >&2
            return 1
        fi
        if ! rid_prefix=$(manifest_prefix "${manifests[$key]}"); then
            echo "[process_kernels.sh] ERROR: invalid request_id_prefix for sweep $key" >&2
            return 1
        fi
        expected_prefix="profile-${J}-${key}"
        if [[ "$rid_prefix" != "$expected_prefix" ]]; then
            echo "[process_kernels.sh] ERROR: request_id_prefix mismatch for sweep $key: $rid_prefix" >&2
            return 1
        fi
        if [[ -n "${prefixes[$rid_prefix]+x}" ]]; then
            echo "[process_kernels.sh] ERROR: duplicate request_id_prefix: $rid_prefix" >&2
            return 1
        fi
        prefixes[$rid_prefix]=1
        SWEEP_KEYS+=("$key")
        SWEEP_CLIENTS+=("${clients[$key]}")
        SWEEP_MANIFESTS+=("${manifests[$key]}")
        SWEEP_PREFIXES+=("$rid_prefix")
        SWEEP_OUTPUTS+=("$OUT/sweeps/$key")
    done
    for key in "${!manifests[@]}"; do
        if [[ -z "${clients[$key]+x}" ]]; then
            echo "[process_kernels.sh] ERROR: orphan manifest for sweep $key" >&2
            return 1
        fi
    done
    MULTI_SWEEP=1
}

trace_sweep() {
    local sweep_label="$1" rid_prefix="$2" client_csv="$3" client_manifest="$4" sweep_out="$5"
    local rid_filter="${rid_prefix}-"
    trap 'rc=$?; echo "[process_kernels.sh] ERROR: sweep $sweep_label failed (status=$rc)" >&2; exit "$rc"' ERR
    mkdir -p "$OUT/_staging"
    local stage="$OUT/_staging/run.$$.$sweep_label"
    mkdir -p "$stage"
    trap 'rm -rf "$stage"' RETURN

    local trace="$stage/trace.json"
    local expected=$(( (XP + YD) * E ))
    python3 "$TOOLS" build-trace \
        --prefill-dir "${PREFILL_DIRS[@]}" \
        --decode-dir "${DECODE_DIRS[@]}" \
        --request-logs "${PREFILL_LOGS[@]}" "${DECODE_LOGS[@]}" \
        --rid-prefix "$rid_filter" \
        --out "$trace" --expect-workers "$expected"

    local combined="$stage/roctx_mori_clean_prefill_decode_${J}.json"
    local probe="$stage/roctx_mori_clean_probe_only_${J}.json"
    mv "$trace" "$combined"
    cp "$combined" "$probe"
    local combined_hash probe_hash
    combined_hash=$(sha256_file "$combined")
    probe_hash=$(sha256_file "$probe")
    if [ "$combined_hash" != "$probe_hash" ]; then
        echo "[process_kernels.sh] ERROR: compatibility trace hashes differ" >&2
        return 1
    fi

    if [[ "$RUN_MORI" == "1" ]]; then
        python3 "$TOOLS" correlate \
            --prefill-dir "${PREFILL_DIRS[@]}" \
            --prefill-logs "${PREFILL_LOGS[@]}" \
            --out-csv "$stage/request_mori_map_${J}.csv" \
            --out-summary "$stage/request_mori_map_${J}.md" \
            --rid-prefix "$rid_filter" --require-complete
    else
        echo "[process_kernels.sh] RUN_MORI=0: skipping MoRI request/KV correlation"
    fi

    cp "$client_csv" "$client_manifest" "$stage/"
    local reqstats_mode=()
    [[ "$RUN_MORI" == "1" ]] || reqstats_mode+=(--no-mori)
    python3 "$TOOLS" reqstats \
        --job "$J" --xp "$XP" --yd "$YD" --out-dir "$stage" --splits \
        --prefill-dirs "${PREFILL_DIRS[@]}" \
        --decode-dirs "${DECODE_DIRS[@]}" \
        --prefill-logs "${PREFILL_LOGS[@]}" \
        --decode-logs "${DECODE_LOGS[@]}" \
        --client-csv "$client_csv" --client-manifest "$client_manifest" \
        --require-data --require-client \
        --rid-prefix "$rid_filter" \
        "${reqstats_mode[@]}"

    local required=(
        "$combined" "$probe"
        "$stage/reqstats_per_request_${J}.csv"
        "$stage/reqstats_per_request_${J}_prefill.csv"
        "$stage/reqstats_per_request_${J}_decode.csv"
        "$stage/$(basename "$client_csv")"
        "$stage/$(basename "$client_manifest")"
    )
    if [[ "$RUN_MORI" == "1" ]]; then
        required+=(
            "$stage/request_mori_map_${J}.csv"
            "$stage/request_mori_map_${J}.md"
        )
    fi
    local file
    for file in "${required[@]}"; do
        [ -s "$file" ] || {
            echo "[process_kernels.sh] ERROR: required output missing: $file" >&2
            return 1
        }
    done
    mkdir -p "$sweep_out"
    for file in "${required[@]}"; do
        mv -f "$file" "$sweep_out/$(basename "$file")"
    done
    rmdir "$stage"
    trap - RETURN
    trap - ERR
    echo "[process_kernels.sh] trace SHA-256=$combined_hash"
    if [[ "$RUN_MORI" == "1" ]]; then
        echo "[process_kernels.sh] strict trace/request/MoRI outputs finalized in $sweep_out"
    else
        echo "[process_kernels.sh] SGLang trace/request outputs finalized without MoRI correlation in $sweep_out"
    fi
}

trace_phase() {
    verify_capture
    discover_sweeps
    local index
    for ((index=0; index<${#SWEEP_KEYS[@]}; index++)); do
        echo "[process_kernels.sh] processing sweep ${SWEEP_KEYS[$index]}"
        trace_sweep "${SWEEP_KEYS[$index]}" "${SWEEP_PREFIXES[$index]}" \
            "${SWEEP_CLIENTS[$index]}" "${SWEEP_MANIFESTS[$index]}" "${SWEEP_OUTPUTS[$index]}"
    done
}

analyze_phase() {
    mkdir -p "$OUT/analyze_phase" "$OUT/_staging"
    local failures=0 statuses=() dir label temporary final
    for dir in "${DIRS[@]}"; do
        label=$(basename "$dir")
        temporary="$OUT/_staging/analyze_${label}.$$"
        final="$OUT/analyze_phase/$label"
        rm -rf "$temporary"
        if python3 "$TOOLS" analyze "$dir" "$temporary" "$label"; then
            rm -rf "$final"
            mv "$temporary" "$final"
            statuses+=("$label=ok")
        else
            rm -rf "$temporary"
            echo "[process_kernels.sh] WARN: analysis failed for $label" >&2
            statuses+=("$label=failed")
            failures=$((failures+1))
        fi
    done
    echo "[process_kernels.sh] analysis statuses=${statuses[*]} (best-effort failures=$failures)"
    return 0
}

case "$OP" in
    verify)
        verify_capture
        ;;
    trace)
        trace_phase
        ;;
    analyze)
        analyze_phase
        ;;
    run)
        trace_phase
        analyze_phase
        echo "===== [process_kernels.sh] capture complete ====="
        echo "workers=$(( (XP + YD) * E ))/$(( (XP + YD) * E ))"
        if (( MULTI_SWEEP )); then
            echo "sweeps=$OUT/sweeps (${#SWEEP_KEYS[@]})"
        else
            echo "combined=$OUT/roctx_mori_clean_prefill_decode_${J}.json"
            echo "probe_only=$OUT/roctx_mori_clean_probe_only_${J}.json"
            [[ "$RUN_MORI" == "1" ]] && echo "request_mori_map=$OUT/request_mori_map_${J}.csv"
            echo "request_stats=$OUT/reqstats_per_request_${J}.csv"
        fi
        ;;
esac
