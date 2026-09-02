#!/bin/bash
# Offline argv + env assertions for the unified launcher: checks that each connector ×
# WIDE_EP × role cell emits the expected `vllm serve` flags/env (and omits the wrong ones).
# No cluster / no GPUs. Exits 0 if all assertions hold.
#
# Covers:
#   - moriio+TP (Llama): exactly ONE --compilation-config, has --disable-custom-all-reduce,
#     has --tensor-parallel-size, NO -tp 1 / --enable-expert-parallel / --all2all-backend
#   - moriio+wideEP (DSV3): -tp 1 + --data-parallel-size + --enable-expert-parallel +
#     --all2all-backend mori_high_throughput + --block-size 16, exactly ONE --compilation-config
#   - slurm docker -e forwards the RDMA-fix env (expandable_segments:False x2, IPC_MODE_LEGACY=0)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM="$DIR/run_xPyD_models.slurm"
pass=0; fail=0

# emit argv for a cell
_argv() { # connector wide_ep ep_backend model model_path
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=1 yD=1 CONNECTOR="$1" WIDE_EP="$2" EP_BACKEND="$3" \
    MODEL_NAME="$4" MODEL_PATH="$5" MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}

# emit argv for a cell with a KV_OFFLOAD tier set (+ optional OFFLOAD_BACKEND, OFFLOAD_DISK_PATH)
_argv_off() { # connector wide_ep ep_backend model model_path kv_offload [offload_backend] [disk_path]
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=1 yD=1 CONNECTOR="$1" WIDE_EP="$2" EP_BACKEND="$3" \
    MODEL_NAME="$4" MODEL_PATH="$5" KV_OFFLOAD="$6" ${7:+OFFLOAD_BACKEND="$7"} ${8:+OFFLOAD_DISK_PATH="$8"} \
    MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}

_has()   { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot(){ grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }
_count() { local n; n="$(grep -cF -- "$2" <<<"$1")"; [[ "$n" == "$3" ]] && { printf "  PASS  %s (=%s)\n" "$4" "$n"; pass=$((pass+1)); } || { printf "  FAIL  %s (got %s want %s)\n" "$4" "$n" "$3"; fail=$((fail+1)); }; }

echo "=== moriio + TP (Llama-70B) ==="
A="$(_argv moriio 0 '' amd-Llama-3.3-70B-Instruct-FP8-KV /m/Llama)"
_has    "$A" "--tensor-parallel-size" "has --tensor-parallel-size"
_has    "$A" "--disable-custom-all-reduce" "has --disable-custom-all-reduce"
_count  "$A" "--compilation-config" 1 "exactly one --compilation-config"
_hasnot "$A" "--enable-expert-parallel" "no --enable-expert-parallel"
_hasnot "$A" "--all2all-backend" "no --all2all-backend"
_hasnot "$A" "--data-parallel-size" "no --data-parallel-size"

echo ""
echo "=== moriio + wideEP (DeepSeek-V3, EP) ==="
B="$(_argv moriio 1 mori DeepSeek-V3 /m/DSV3)"
_has    "$B" "--enable-expert-parallel" "has --enable-expert-parallel"
_has    "$B" "--data-parallel-size" "has --data-parallel-size"
_has    "$B" "mori_high_throughput" "prefill all2all = mori_high_throughput"
_has    "$B" "--block-size" "has --block-size"
_has    "$B" "16" "block-size value 16 present"
_count  "$B" "--compilation-config" 1 "exactly one --compilation-config"
_hasnot "$B" "--tensor-parallel-size" "no --tensor-parallel-size (uses -tp 1)"

echo ""
echo "=== KV_OFFLOAD=cpu native (tiered prefix cache; rixl TP Llama-3.1-405B) ==="
O="$(_argv_off rixl 0 '' Llama-3.1-405B-Instruct-FP8-KV /m/L405 cpu)"
_has    "$O" "MultiConnector" "wraps in MultiConnector"
_has    "$O" "OffloadingConnector" "adds OffloadingConnector (native default)"
_has    "$O" "cpu_bytes_to_use" "OffloadingConnector has cpu_bytes_to_use"
_has    "$O" "NixlConnector" "base NixlConnector preserved inside MultiConnector"
_hasnot "$O" "LMCacheConnectorV1" "no LMCache on the native path"
_has    "$O" "--enable-prefix-caching" "prefix caching enabled under offload"
_hasnot "$O" "--no-enable-prefix-caching" "no --no-enable-prefix-caching under offload"

echo ""
echo "=== KV_OFFLOAD=cpu native + OFFLOAD_DISK_PATH (fs tier; rixl TP 405B) ==="
OF="$(_argv_off rixl 0 '' Llama-3.1-405B-Instruct-FP8-KV /m/L405 cpu native /mnt/kv)"
_has    "$OF" "OffloadingConnector" "native OffloadingConnector on the fs path"
_has    "$OF" "cpu_bytes_to_use" "CPU tier preserved with an fs tier below it"
_has    "$OF" "TieringOffloadingSpec" "fs tier uses TieringOffloadingSpec"
_has    "$OF" "secondary_tiers" "config has secondary_tiers"
_has    "$OF" '"type": "fs"' "secondary tier type is fs"
_hasnot "$OF" "LMCacheConnectorV1" "no LMCache on the native fs path"

echo ""
echo "=== KV_OFFLOAD=cpu native WITHOUT disk path emits no fs tier ==="
_hasnot "$O" "secondary_tiers" "plain native cpu has no secondary_tiers"
_hasnot "$O" "TieringOffloadingSpec" "plain native cpu has no TieringOffloadingSpec"

echo ""
echo "=== KV_OFFLOAD=cpu OFFLOAD_BACKEND=lmcache (LMCacheConnectorV1; rixl TP 405B) ==="
L="$(_argv_off rixl 0 '' Llama-3.1-405B-Instruct-FP8-KV /m/L405 cpu lmcache)"
_has    "$L" "MultiConnector" "wraps in MultiConnector"
_has    "$L" "LMCacheConnectorV1" "adds LMCacheConnectorV1"
_has    "$L" "NixlConnector" "base NixlConnector preserved inside MultiConnector"
_hasnot "$L" "OffloadingConnector" "no native OffloadingConnector on the lmcache path"
_hasnot "$L" "cpu_bytes_to_use" "no OffloadingConnector cpu_bytes_to_use on lmcache path"
_has    "$L" "--enable-prefix-caching" "prefix caching enabled under lmcache offload"

echo ""
echo "=== KV_OFFLOAD=none is unchanged (no MultiConnector; disagg recipe intact) ==="
N="$(_argv_off rixl 0 '' Llama-3.1-405B-Instruct-FP8-KV /m/L405 none)"
_hasnot "$N" "MultiConnector" "no MultiConnector when offload disabled"
_hasnot "$N" "OffloadingConnector" "no OffloadingConnector when offload disabled"
_has    "$N" "--no-enable-prefix-caching" "prefix caching stays off (base recipe)"

echo ""
echo "=== connector platform env files carry the RDMA-fix env ==="
# The ROCm-7.2.3 GPU-RDMA env now lives in per-connector .env files; the slurm
# sources connectors/<CONNECTOR>.env and forwards each var via docker -e.
S="$(cat "$SLURM")"
_has "$S" 'CONNECTOR_ENV_FILE="${SCRIPT_DIR}/connectors/${CONNECTOR}.env"' "slurm sources connector .env"
_has "$S" '${CONNECTOR_ENV_ARGS}' "slurm forwards CONNECTOR_ENV_ARGS in docker run"
for cf in moriio rixl; do
  F="$DIR/connectors/${cf}.env"
  if [[ -f "$F" ]]; then
    E="$(cat "$F")"
    _has "$E" "PYTORCH_ALLOC_CONF=expandable_segments:False" "${cf}.env: PYTORCH_ALLOC_CONF"
    _has "$E" "PYTORCH_HIP_ALLOC_CONF=expandable_segments:False" "${cf}.env: PYTORCH_HIP_ALLOC_CONF"
    _has "$E" "HSA_ENABLE_IPC_MODE_LEGACY=0" "${cf}.env: IPC_MODE_LEGACY=0"
    _has "$E" "MORI_GPU_ARCHS=gfx942" "${cf}.env: MORI_GPU_ARCHS"
  else
    printf "  FAIL  connectors/%s.env missing\n" "$cf"; fail=$((fail+1))
  fi
done
# parse check: the slurm's KEY=${KEY:-VAL} loop yields correct -e args (+ override wins)
_parse() { # $1=connector ; reads its .env with same logic as the slurm
  local A="" l k v
  while IFS= read -r l; do
    [[ "$l" =~ ^[[:space:]]*# || -z "${l// }" ]] && continue
    k="${l%%=*}"; v="${l#*=}"; A+=" -e ${k}=${!k:-$v}"
  done < "$DIR/connectors/$1.env"
  printf '%s' "$A"
}
_has "$(_parse moriio)" "-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False" "parse yields HIP_ALLOC -e arg"
_has "$(PYTORCH_HIP_ALLOC_CONF=expandable_segments:True _parse moriio)" "-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True" "submit-time override wins"

echo ""
echo "======================================================"
echo "  argv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
