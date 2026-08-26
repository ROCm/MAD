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

_has()   { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasre() { grep -qE -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
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
echo "=== moriio + wideEP (Kimi-K3-MXFP4, 2P/2D TP2×DP8) ==="
_argv_k3() {
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=2 yD=2 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
    MODEL_NAME=Kimi-K3-MXFP4 MODEL_PATH=/m/K3 \
    MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}
C="$(_argv_k3)"
_has    "$C" "--tensor-parallel-size" "K3 has --tensor-parallel-size"
_hasre   "$C" "^2$" "K3 TP=2"
_has    "$C" "--data-parallel-size" "K3 has --data-parallel-size"
_hasre   "$C" "^8$" "K3 dp_size=8 (line)"
_has    "$C" "--data-parallel-size-local" "K3 has dp_local flag"
_hasre   "$C" "^4$" "K3 dp_local=4 (line)"
_has    "$C" "--enable-expert-parallel" "K3 has EP"
_has    "$C" "moriio_pod_hosts" "K3 kv config has pod hosts"
_has    "$C" "--api-server-count=8" "K3 api-server-count=dp_size"
_has    "$C" "--reasoning-parser" "K3 reasoning parser flag"
_has    "$C" "kimi_k3" "K3 reasoning parser value"
_hasnot "$C" "-tp 1" "K3 not -tp 1"

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
