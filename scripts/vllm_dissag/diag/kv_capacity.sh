#!/usr/bin/env bash
# kv_capacity.sh -- report the KV cache capacity a run actually got.
#
# WHY THIS EXISTS
# ---------------
# KV capacity is the metric that decides topology, and it is the one metric the
# benchmark output never prints. Throughput and TPOT are reported in a tidy table at
# the end of every sweep; KV capacity is emitted once per DP worker, at INFO, during
# boot, into a per-node log, and then scrolls away under several thousand lines of
# engine startup.
#
# That asymmetry has real consequences. On this stack EP16 is SLOWER than EP8 on both
# throughput and latency (measured: +20-23% TPOT, ~1.0x throughput for 2x the nodes),
# so if you only read the benchmark table there is no reason to ever choose it. The
# reason to choose it is capacity -- 1.44M KV tokens per decode node versus ~805k --
# and that number is invisible unless you go looking. A metric that only appears when
# you already suspect it matters is not a reported metric.
#
# It is also the number a customer needs to answer "how many concurrent long-context
# users fit", which is usually the actual procurement question.
#
# USAGE
#   diag/kv_capacity.sh <job_id> [node ...]
#   diag/kv_capacity.sh skyriver_20260817_101940 skyriver04 skyriver05 skyriver06 skyriver07
#
# Reads /models/common/logs/<job_id>/ on each node. Nodes default to the four skyRiver
# hosts. Read-only: greps logs, changes nothing, safe to run against a live job.
set -uo pipefail

JOB="${1:-}"
if [ -z "$JOB" ]; then
    echo "usage: $0 <job_id> [node ...]" >&2
    exit 2
fi
shift || true
NODES=("$@")
[ "${#NODES[@]}" -eq 0 ] && NODES=(skyriver04 skyriver05 skyriver06 skyriver07)

LOG_ROOT="${LOG_ROOT:-/models/common/logs}"

echo "KV cache capacity -- job ${JOB}"
echo "=============================================================================="
printf '%-14s %-8s %14s %16s %12s\n' NODE ROLE "KV MEMORY" "KV TOKENS" "MAX CONC"
echo "------------------------------------------------------------------------------"

total_tokens=0
found_any=0
# Dedupe key is (role, NODE-rank), NOT (node, role). Rank 0 holds RELAYED copies of every
# other node's log -- see the CROSS-NODE LOG RELAY block in launch_disagg_skyriver.sh,
# which copies decode_NODE<n>.log onto rank 0 because /models/common is a local disk per
# node and the upstream driver assumes a shared FS. So globbing decode_NODE*.log on
# skyriver04 returns skyriver06's and skyriver07's logs too. Summing per-node
# double-counted a 2P/2D job as 5.47M tokens when the true figure is 2.74M -- exactly the
# kind of number that would have looked plausible in a customer document.
#
# Deduping on the whole line does not work either: the same rank reports
# "60.82 GiB" and "60.9 GiB" on different DP workers (rounding), so `sort -u` keeps both.
# Rank identity is the only stable key.
declare -A seen_tok seen_mem seen_conc seen_role

for n in "${NODES[@]}"; do
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" "
        for f in ${LOG_ROOT}/${JOB}/prefill_NODE*.log ${LOG_ROOT}/${JOB}/decode_NODE*.log; do
            [ -f \"\$f\" ] || continue
            case \"\$f\" in *prefill*) role=prefill ;; *) role=decode ;; esac
            rank=\$(basename \"\$f\" .log | sed 's/.*NODE//')
            mem=\$(grep -ohE 'Available KV cache memory: [0-9.]+ GiB' \"\$f\" 2>/dev/null | head -1 | grep -oE '[0-9.]+ GiB')
            line=\$(grep -ohE 'GPU KV cache size: [0-9,]+ tokens.*' \"\$f\" 2>/dev/null | head -1)
            tok=\$(echo \"\$line\" | grep -oE '[0-9,]+ tokens' | head -1 | tr -d ', tokens')
            conc=\$(echo \"\$line\" | grep -oE '[0-9.]+x' | head -1)
            [ -n \"\$tok\" ] && echo \"\${role}|\${rank}|\${mem:-?}|\${tok}|\${conc:-?}\"
        done
    " 2>/dev/null)

    [ -z "$out" ] && continue

    while IFS='|' read -r role rank mem tok conc; do
        [ -z "$tok" ] && continue
        key="${role}${rank}"
        # First writer wins. A relayed copy and the original are the same rank reporting
        # the same pool; taking the first avoids averaging near-identical numbers, which
        # would only invent false precision.
        [ -n "${seen_tok[$key]:-}" ] && continue
        seen_tok[$key]="$tok"; seen_mem[$key]="$mem"
        seen_conc[$key]="$conc"; seen_role[$key]="$role"
    done <<< "$out"
done

commafy() { echo "$1" | sed -e ':a' -e 's/\([0-9]\)\([0-9]\{3\}\)\($\|,\)/\1,\2\3/;ta'; }

for key in $(printf '%s\n' "${!seen_tok[@]}" | sort); do
    found_any=1
    rank="${key#"${seen_role[$key]}"}"
    printf '%-14s %-8s %14s %16s %12s\n' "NODE${rank}" "${seen_role[$key]}" \
        "${seen_mem[$key]}" "$(commafy "${seen_tok[$key]}")" "${seen_conc[$key]}"
    # Only DECODE ranks hold the KV pool that serves generation. Summing prefill into the
    # total would overstate capacity -- prefill KV is transient and handed off.
    [ "${seen_role[$key]}" = "decode" ] && total_tokens=$((total_tokens + seen_tok[$key]))
done

echo "------------------------------------------------------------------------------"
if [ "$found_any" = "0" ]; then
    echo "No KV cache lines found. Either the job has not finished booting (the line is"
    echo "printed after profiling, ~several minutes in) or the job id / log root is wrong."
    exit 1
fi

printf 'Aggregate DECODE KV: %s tokens\n' "$(commafy "$total_tokens")"

# Translate into the units a capacity question is actually asked in. MAX_LEN default
# matches GLM_MAX_MODEL_LEN's shipped 65536; override to model your own context length.
MAX_LEN="${MAX_LEN:-65536}"
if [ "$total_tokens" -gt 0 ] && [ "$MAX_LEN" -gt 0 ]; then
    printf 'At %s tokens/request: ~%d concurrent requests across all decode nodes\n' \
        "$(commafy "$MAX_LEN")" "$((total_tokens / MAX_LEN))"
    echo
    echo "Note: that is a CAPACITY ceiling, not a throughput or latency claim. Whether the"
    echo "deployment is usable at that concurrency is a separate question -- at con=128,"
    echo "median TTFT is already ~18 s because requests queue. Capacity says what fits in"
    echo "memory; it does not say what fits in your SLA."
fi
