#!/bin/bash
# Regenerate golden argv fixtures for tests/parity_check.sh.
# Only run when a legacy-equivalent launcher change is intentional; review the
# resulting `git diff tests/golden/` before committing.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"   # .../scripts/vllm_dissag
TMP=$(mktemp -d); mkdir -p "$TMP/bin" /run_logs/PARITY
cat > "$TMP/bin/vllm" <<'V'
#!/bin/bash
{ for a in "$@"; do printf '%s\n' "$a"; done; } >> "${PARITY_OUT:-/dev/stdout}"
exit 0
V
chmod +x "$TMP/bin/vllm"
export MODEL_PATH=/models/DeepSeek-V3 MODEL_NAME=DeepSeek-V3 SLURM_JOB_ID=PARITY
export MASTER_ADDR=10.0.0.1   # mori kv-config proxy_ip source
norm(){ sed -E 's/"kv_ip": "[^"]*"/"kv_ip": "HOSTIP"/'; }
G="$DIR/tests/golden"

cap_mori(){ local role="$1" ds="$2" da="$3" kr="$4" pf="$5" sr="$6" out="$7"
 ( set +u; PATH="$TMP/bin:$PATH"
   sed -n '/^setup_mori_env() {/,/^}/p;/^build_kv_transfer_config() {/,/^}/p;/^launch_vllm_worker() {/,/^}/p' "$DIR/vllm_disagg_mori_ep.sh" > "$TMP/lm.sh"; source "$TMP/lm.sh"
   NODE_RANK=0 _GPUS_PER_NODE=8 DP_PARALLEL_SIZE_LOCAL=8 RPC_PORT=13345 SERVE_PORT=20005 KV_PORT=9711 PROXY_PORT=10001 PROXY_PING_PORT=36367 LOCAL_PING_PORT=61555 HANDSHAKE_PORT=8405 NOTIFY_PORT=61005
   export PARITY_OUT="$TMP/r"; : >"$TMP/r"; launch_vllm_worker "$ds" "$da" "$kr" "$pf" "$role" "$sr" 2>/dev/null; wait 2>/dev/null )
 norm < "$TMP/r" > "$out"; }
cap_dep(){ local orole="$1" be="$2" ds="$3" da="$4" kr="$5" eid="$6" pf="$7" sr="$8" out="$9"
 ( set +u; PATH="$TMP/bin:$PATH"
   sed -n '/^setup_deepep_env() {/,/^}/p;/^build_kv_transfer_config() {/,/^}/p;/^launch_vllm_worker() {/,/^}/p' "$DIR/vllm_disagg_server_deepep.sh" > "$TMP/ld.sh"; source "$TMP/ld.sh"
   NODE_RANK=0 host_ip=10.0.0.1 DP_SIZE_LOCAL=8 SERVER_PORT=2584 RPC_PORT=13345 KV_PORT=14600
   export PARITY_OUT="$TMP/r"; : >"$TMP/r"; launch_vllm_worker "$orole" "$be" "$ds" "$da" "$kr" "$eid" "$pf" "$sr" 2>/dev/null; wait 2>/dev/null )
 norm < "$TMP/r" > "$out"; }
# rixl: reproduce the REAL legacy eval "$CMD $CFG" path exactly.
cap_rixl(){ local eid="$1" kr="$2" out="$3"
 ( set +u; PATH="$TMP/bin:$PATH"; host_ip=10.0.0.1; SERVER_PORT=2584
   declare -A M=( ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1" )
   CFG="${M[DeepSeek-V3]}"
   export PARITY_OUT="$TMP/r"; : >"$TMP/r"
   CMD="vllm serve \${MODEL_PATH} --port \$SERVER_PORT --trust-remote-code --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"${eid}\", \"kv_role\": \"${kr}\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"\${host_ip}\"'\", \"kv_port\": 14600}'"
   CMD="$CMD $CFG"; eval "$CMD" )
 norm < "$TMP/r" > "$out"; }

cap_rixl pd-run kv_producer "$G/rixl_prefill.txt"
cap_rixl pd-decode kv_consumer "$G/rixl_decode.txt"
cap_mori master 16 10.0.0.1 kv_producer prefill ""  "$G/mori_prefill_master.txt"
cap_mori child  16 10.0.0.1 kv_producer prefill 8   "$G/mori_prefill_child.txt"
cap_mori master 16 10.0.0.3 kv_consumer decode ""   "$G/mori_decode_master.txt"
cap_mori child  16 10.0.0.3 kv_consumer decode 8    "$G/mori_decode_child.txt"
cap_dep prefill_master deepep_high_throughput 16 10.0.0.1 kv_producer pd-prefill prefill "" "$G/deepep_prefill_master.txt"
cap_dep prefill_child  deepep_high_throughput 16 10.0.0.1 kv_producer pd-prefill prefill 8 "$G/deepep_prefill_child.txt"
cap_dep decode_master  deepep_low_latency     16 10.0.0.3 kv_consumer pd-decode  decode  "" "$G/deepep_decode_master.txt"
cap_dep decode_child   deepep_low_latency     16 10.0.0.3 kv_consumer pd-decode  decode  8  "$G/deepep_decode_child.txt"
rm -rf "$TMP"; echo "regenerated"
