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
# assert a flag line ($2) is immediately followed by an exact value line ($3) in argv $1
_hasadj() { grep -A1 -xF -- "$2" <<<"$1" | grep -qxF -- "$3" && { printf "  PASS  %s\n" "$4"; pass=$((pass+1)); } || { printf "  FAIL  %s (want %s -> %s)\n" "$4" "$2" "$3"; fail=$((fail+1)); }; }

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
_hasadj  "$C" "--tensor-parallel-size" "2" "K3 TP=2 (value adjacent to flag)"
_has    "$C" "--data-parallel-size" "K3 has --data-parallel-size"
_hasadj  "$C" "--data-parallel-size" "8" "K3 dp_size=8 (adjacent)"
_has    "$C" "--data-parallel-size-local" "K3 has dp_local flag"
_hasadj  "$C" "--data-parallel-size-local" "4" "K3 dp_local=4 (adjacent)"
_has    "$C" "--enable-expert-parallel" "K3 has EP"
_has    "$C" "moriio_pod_hosts" "K3 kv config has pod hosts"
_has    "$C" "--api-server-count=8" "K3 api-server-count=dp_size"
_has    "$C" "--reasoning-parser" "K3 reasoning parser flag"
_has    "$C" "kimi_k3" "K3 reasoning parser value"
_hasnot "$C" "-tp 1" "K3 not -tp 1"

echo ""
echo "=== -e EP_TP_SIZE=1 override beats the recipe (K3 falls back to plain wideEP -tp 1) ==="
Cov="$(env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=2 yD=2 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
    MODEL_NAME=Kimi-K3-MXFP4 MODEL_PATH=/m/K3 EP_TP_SIZE=1 \
    MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f')"
_hasadj "$Cov" "-tp" "1" "EP_TP_SIZE=1 override -> -tp 1 (adjacent)"
_hasnot "$Cov" "--tensor-parallel-size" "EP_TP_SIZE=1 override -> no --tensor-parallel-size"

echo ""
echo "=== EP_TP_SIZE divisibility guard rejects an indivisible value ==="
if env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=2 yD=2 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
    MODEL_NAME=Kimi-K3-MXFP4 MODEL_PATH=/m/K3 EP_TP_SIZE=3 \
    MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" >/dev/null 2>&1; then
  printf "  FAIL  EP_TP_SIZE=3 should be rejected (indivisible by GPUS_PER_NODE=8)\n"; fail=$((fail+1))
else
  printf "  PASS  EP_TP_SIZE=3 rejected (indivisible by GPUS_PER_NODE=8)\n"; pass=$((pass+1))
fi

echo ""
echo ""
echo "=== non-K3 wideEP: EP_TP_SIZE dormant (plain -tp 1, no role split) ==="
# emit argv for an explicit rank/topology cell
_argv_rank() { # connector wide ep model rank xP yD
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK="$5" xP="$6" yD="$7" CONNECTOR="$1" WIDE_EP="$2" EP_BACKEND="$3" \
    MODEL_NAME="$4" MODEL_PATH=/m/M MASTER_ADDR=10.0.0.1 \
    IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}
Dm="$(_argv_rank moriio 1 mori DeepSeek-V3 0 2 2)"
_hasadj "$Dm" "-tp" "1" "DSV3 wideEP -> -tp 1 (adjacent)"
_hasnot "$Dm" "--tensor-parallel-size" "DSV3 wideEP -> no --tensor-parallel-size"
_hasnot "$Dm" "moriio_pod_hosts" "DSV3 wideEP -> no K3 pod-hosts in kv-config"
_has    "$Dm" "--api-server-count" "DSV3 wideEP -> has --api-server-count"
Dh="$(_argv_rank moriio 1 mori DeepSeek-V3 1 2 2)"
_has    "$Dh" "--headless" "DSV3 wideEP headless child -> --headless"
_hasnot "$Dh" "--kv-transfer-config" "DSV3 wideEP headless child -> no --kv-transfer-config"

echo ""
echo "=== EP_TP_SIZE knob is model-agnostic (DSV3 behaves like K3 per value) ==="
_argv_eptp() { # model ep_tp_size
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=2 yD=2 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
    MODEL_NAME="$1" MODEL_PATH=/m/M EP_TP_SIZE="$2" MASTER_ADDR=10.0.0.1 \
    IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}
_guard_rejects() { # model ep_tp_size  (indivisible -> vllm_disagg exits nonzero)
  if env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
      DRY_RUN=1 NODE_RANK=0 xP=2 yD=2 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
      MODEL_NAME="$1" MODEL_PATH=/m/M EP_TP_SIZE="$2" MASTER_ADDR=10.0.0.1 \
      IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
      GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
      bash "$DIR/vllm_disagg.sh" >/dev/null 2>&1; then
    printf "  FAIL  %s EP_TP_SIZE=%s should be rejected (indivisible)\n" "$1" "$2"; fail=$((fail+1))
  else
    printf "  PASS  %s EP_TP_SIZE=%s rejected by divisibility guard\n" "$1" "$2"; pass=$((pass+1))
  fi
}
for M in DeepSeek-V3 Kimi-K3-MXFP4; do
  E1="$(_argv_eptp "$M" 1)"
  _hasadj "$E1" "-tp" "1" "$M EP_TP_SIZE=1 -> -tp 1 (adjacent)"
  _has    "$E1" "--api-server-count=8" "$M EP_TP_SIZE=1 -> api-server-count=8"
  E4="$(_argv_eptp "$M" 4)"
  _hasadj "$E4" "--tensor-parallel-size" "4" "$M EP_TP_SIZE=4 -> TP=4 (adjacent)"
  _has    "$E4" "--api-server-count=4" "$M EP_TP_SIZE=4 -> api-server-count=4"
  _guard_rejects "$M" 3
done

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

# Per-shape warmup must stay opt-in: it is on the shared sweep path, so a default-on gate
# would change the measured TPOT of every already-validated recipe. GLM opts in via its
# models.yaml env:, and both GLM-only knobs need a docker -e line or they cannot be
# A/B-tested from the submit side (the recipe applies whenever the key is absent).
echo ""
echo "=== per-shape warmup is opt-in, not default-on ==="
B="$(cat "$DIR/benchmark_xPyD.sh")"
_has    "$B" '${SHAPE_WARMUP:-0}' "benchmark_xPyD.sh: warmup gate defaults OFF"
_hasnot "$B" '${SHAPE_WARMUP:-1}' "benchmark_xPyD.sh: gate is not default-on"
_OPTIN="$(python3 - "$DIR/models.yaml" <<'PY'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])) or {}
optin = [m for m, c in y.items()
         if isinstance(c, dict) and (c.get("env") or {}).get("SHAPE_WARMUP") == "1"]
print("[" + ",".join(sorted(optin)) + "]")
PY
)"
_has "$_OPTIN" "[GLM-5.1-FP8]" "models.yaml: GLM-5.1-FP8 is the ONLY warmup opt-in"
_has "$(cat "$SLURM")" '${SHAPE_WARMUP:+-e SHAPE_WARMUP=' "slurm forwards SHAPE_WARMUP override"
_has "$(cat "$SLURM")" '${USE_INDUCTOR_GRAPH_PARTITION:+-e USE_INDUCTOR_GRAPH_PARTITION=' "slurm forwards IGP override"

echo ""
echo "======================================================"
echo "  argv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
