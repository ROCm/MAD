#!/bin/bash
# Offline checks for the vLLM agentic path: aiperf's --url (build_replay_cmd) and the models shim
# (agentic_models_shim.py) against stdlib fake router/prefill servers. No cluster / no GPUs.
# SHIM=<path> checks another copy of the shim (e.g. an older revision). Exits 0 if all assertions hold.
#
# Covers:
#   - aiperf --url is AGENTIC_TRAFFIC_PORT when set, AGENTIC_PORT otherwise; vllm_disagg.sh sets it
#     to the router port when it starts the shim
#   - SSE relayed per event (not in 8 KB blocks)
#   - upstream 5xx relayed immediately (no replay to the prefill server)
#   - a stream that breaks off upstream reaches the client as incomplete, and is logged
#   - /v1/models readiness follows the router; prefill fallback only on router /health 404
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$DIR/../common/agentic_lib.sh"
pass=0; fail=0

_url() { # AGENTIC_PORT [AGENTIC_TRAFFIC_PORT]
  env -i PATH="$PATH" HOME="$HOME" AGENTIC_PORT="$1" ${2:+AGENTIC_TRAFFIC_PORT=$2} TRACE_SOURCE_FLAG= \
    bash -c 'source "$1" >/dev/null 2>&1; build_replay_cmd /tmp/x; grep -o -- "--url [^ ]*" <<<"$REPLAY_CMD"' _ "$LIB"
}
_eq() { [[ "$1" == "$2" ]] && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (got '%s' want '%s')\n" "$3" "$1" "$2"; fail=$((fail+1)); }; }

echo "=== aiperf --url (build_replay_cmd) ==="
_eq "$(_url 30001)" "--url http://localhost:30001" "AGENTIC_TRAFFIC_PORT unset -> AGENTIC_PORT"
_eq "$(_url 30001 30000)" "--url http://localhost:30000" "AGENTIC_TRAFFIC_PORT set -> traffic port"
_eq "$(grep -c 'AGENTIC_TRAFFIC_PORT="${BENCHMARK_PORT}"' "$DIR/vllm_disagg.sh")" "1" \
  "vllm_disagg.sh points aiperf at the router when it starts the shim"

python3 "$DIR/tests/shim_check.py" "${SHIM:-$DIR/agentic_models_shim.py}" && rc=0 || rc=1

echo ""
echo "url: $pass passed, $fail failed; shim: $([[ $rc == 0 ]] && echo passed || echo FAILED)"
[[ "$fail" == "0" && "$rc" == "0" ]]
