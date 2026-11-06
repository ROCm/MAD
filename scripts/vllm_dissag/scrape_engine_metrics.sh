#!/bin/bash
# Server-side metrics straight from each vLLM API server's /metrics (never the router: a GET
# <router>/metrics lands on one arbitrary backend).
#
#   scrape_engine_metrics.sh snapshot <dir> <tag>   counters/histograms once -> <dir>/prom-<tag>.prom
#   scrape_engine_metrics.sh sample <dir>           gauges every $EVERY s until killed -> <dir>/samples.tsv
#
# Endpoints: METRICS_ENDPOINTS="role=host:port ..." (role: prefill, decode or coloc). Each block
# in prom-<tag>.prom starts with "# EP role=<role> ep=<host:port>" so multiturn_metrics.py can
# attribute samples by role. With no endpoints this is a no-op.
set -uo pipefail
cmd=${1:?usage: $0 snapshot|sample <dir> [tag]} dir=${2:?dir}
mkdir -p "$dir"
EPS=${METRICS_ENDPOINTS:-}
EVERY=${EVERY:-5}
GAUGES='^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_requests_kv_fetch_by_stage)\{'
[[ -n "$EPS" ]] || exit 0

case $cmd in
  snapshot)
    out=$dir/prom-${3:?tag}.prom; : > "$out"
    for kv in $EPS; do
      echo "# EP role=${kv%%=*} ep=${kv#*=} ts=$(date +%s.%N)" >> "$out"
      curl -sf -m 30 "http://${kv#*=}/metrics" | grep -v '^#' >> "$out" ||
        echo "# EP-ERROR ${kv#*=}" >> "$out"
    done ;;
  sample)
    out=$dir/samples.tsv
    [[ -s $out ]] || printf 'ts\trole\tep\tmetric\tlabels\tvalue\n' > "$out"
    while :; do
      ts=$(date +%s)
      for kv in $EPS; do
        curl -sf -m 10 "http://${kv#*=}/metrics" | grep -E "$GAUGES" |
          awk -v ts="$ts" -v role="${kv%%=*}" -v ep="${kv#*=}" '{
            i = index($0, "{"); j = index($0, "}")
            printf "%s\t%s\t%s\t%s\t%s\t%s\n", ts, role, ep, substr($0, 1, i - 1), substr($0, i + 1, j - i - 1), $NF }' >> "$out"
      done
      sleep "$EVERY"
    done ;;
  *) echo "usage: $0 snapshot|sample <dir> [tag]" >&2; exit 2 ;;
esac
