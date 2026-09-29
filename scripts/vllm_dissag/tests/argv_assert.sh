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

# emit argv for a cell at a given NODE_RANK (xP=1 yD=1: rank 0 prefill, rank 1 decode)
_argv_at() { # rank connector wide_ep ep_backend model model_path
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK="$1" xP=1 yD=1 CONNECTOR="$2" WIDE_EP="$3" EP_BACKEND="$4" \
    MODEL_NAME="$5" MODEL_PATH="$6" MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}

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
echo "=== moriio + wideEP (Kimi-K3, gfx942 a8w4 SiTUv2) ==="
# Kimi-K3 on gfx942 must requantise the MoE to int4: without --quantization-config
# the a16w4 SiTUv2 heuristic kernel cannot be codegen'd and vLLM dies with
# "LLVM ERROR: Do not know how to expand this operator's operand!" inside
# determine_available_memory (benchmark/kimi_k3/mi300x/README.md:160).
K="$(_argv_at 0 moriio 1 mori Kimi-K3 /m/Kimi-K3)"
_has    "$K" "--quantization-config" "prefill: has --quantization-config"
_has    "$K" "int4_per_group_32" "prefill: MoE requantised to int4_per_group_32"
_has    "$K" "--enable-expert-parallel" "prefill: has --enable-expert-parallel (wideEP-only model)"
_has    "$K" "--reasoning-parser" "prefill: has --reasoning-parser"
_has    "$K" "kimi_k3" "prefill: reasoning parser is kimi_k3"
_has    "$K" '"cudagraph_mode":"NONE"' "prefill: cudagraph_mode NONE"
_has    "$K" "mori_high_throughput" "prefill: all2all = mori_high_throughput"
_has    "$K" '"kv_role":"kv_producer"' "prefill: kv_role = kv_producer"
_count  "$K" "--compilation-config" 1 "prefill: exactly one --compilation-config"
_hasnot "$K" "mori_low_latency" "prefill: not the decode all2all backend"

KD="$(_argv_at 1 moriio 1 mori Kimi-K3 /m/Kimi-K3)"
_has    "$KD" "--quantization-config" "decode: has --quantization-config"
_has    "$KD" '"cudagraph_mode":"PIECEWISE"' "decode: cudagraph_mode PIECEWISE"
_has    "$KD" "mori_low_latency" "decode: all2all = mori_low_latency"
_has    "$KD" '"kv_role":"kv_consumer"' "decode: kv_role = kv_consumer"
_count  "$KD" "--compilation-config" 1 "decode: exactly one --compilation-config"
_hasnot "$KD" "mori_high_throughput" "decode: not the prefill all2all backend"

# Kimi-K3 is wideEP-only; moriio+TP for it is untested and must stay gated.
# Scoped to the array's own line: "Kimi-K3" also appears in MORI_EP_VALID_MODELS,
# so grepping the whole file would pass even with Kimi-K3 removed from this gate.
W="$(grep -m1 '^WIDE_EP_ONLY_MODELS=' "$SLURM")"
_has "$W" 'WIDE_EP_ONLY_MODELS=' "slurm declares WIDE_EP_ONLY_MODELS"
_has "$W" '"Kimi-K3"' "slurm gates Kimi-K3 as wideEP-only"
# MORI_EP_VALID_MODELS is a line-continued array, so take the whole block.
M="$(sed -n '/^MORI_EP_VALID_MODELS=/,/^)/p' "$SLURM")"
_has "$M" '"Kimi-K3"' "slurm accepts Kimi-K3 as a MoRI-EP model"
# validate_model_name runs before any connector gate, so a model missing from
# VALID_MODELS exits 1 no matter which lists above accept it.
V="$(sed -n '/^VALID_MODELS=/,/^)/p' "$SLURM")"
_has "$V" '"Kimi-K3"' "slurm VALID_MODELS accepts Kimi-K3"
_has "$V" '"Kimi-K3-MXFP4"' "slurm VALID_MODELS accepts Kimi-K3-MXFP4"

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
# At EP_TP_SIZE=1 the router only targets master-node ranks; a pod-host list there
# would misroute them on the K3 connector (it maps rank -> host by remote_dp_size).
_hasnot "$Dm" "moriio_pod_hosts" "DSV3 wideEP 2P/2D -> no pod-hosts (EP_TP_SIZE=1)"
_has    "$Dm" "--api-server-count" "DSV3 wideEP -> has --api-server-count"
Dh="$(_argv_rank moriio 1 mori DeepSeek-V3 1 2 2)"
_has    "$Dh" "--headless" "DSV3 wideEP headless child -> --headless"
_hasnot "$Dh" "--kv-transfer-config" "DSV3 wideEP headless child -> no --kv-transfer-config"

echo ""
echo "=== EP_TP_SIZE>1 needs equal pools ==="
if env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=2 yD=1 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
    MODEL_NAME=Kimi-K3-MXFP4 MODEL_PATH=/m/K3 MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" >/dev/null 2>&1; then
  echo "  FAIL  EP_TP_SIZE=2 with xP=2 yD=1 should be rejected"; fail=$((fail+1))
else
  echo "  PASS  EP_TP_SIZE=2 with xP=2 yD=1 rejected"; pass=$((pass+1))
fi

echo ""
echo "=== TP_SIZE from cluster.sh does not leak into the wideEP layout ==="
# cluster.sh exports TP_SIZE=GPUS_PER_NODE for the colocated launcher; the disagg
# launcher reads only EP_TP_SIZE, so DSV3 stays TP1/DP16.
Dt="$(env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=2 yD=2 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
    MODEL_NAME=DeepSeek-V3 MODEL_PATH=/m/M TP_SIZE=8 MASTER_ADDR=10.0.0.1 \
    IPADDRS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f')"
_hasadj "$Dt" "-tp" "1" "TP_SIZE=8 in env -> DSV3 still -tp 1"
_hasadj "$Dt" "--data-parallel-size" "16" "TP_SIZE=8 in env -> DSV3 still dp_size=16"

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
echo "=== moriio + wideEP (Kimi-K3-MXFP4-MI355X, native MXFP4, 2P/2D TP1xDP16) ==="
M5="$(_argv_rank moriio 1 mori Kimi-K3-MXFP4-MI355X 0 2 2)"
_hasadj "$M5" "-tp" "1" "MI355X: -tp 1 (EP_TP_SIZE=1)"
_hasadj "$M5" "--data-parallel-size" "16" "MI355X: dp_size=16"
_hasadj "$M5" "--data-parallel-size-local" "8" "MI355X: dp_local=8"
_hasnot "$M5" "--quantization-config" "MI355X: no int4 requant (native MXFP4 on gfx950)"
_hasnot "$M5" "moriio_pod_hosts" "MI355X: no pod-hosts at EP_TP_SIZE=1"
_hasadj "$M5" "--max-num-batched-tokens" "4096" "MI355X: max-num-batched-tokens 4096"
_has    "$M5" "kimi_k3" "MI355X: kimi_k3 reasoning parser"
_count  "$M5" "--compilation-config" 1 "MI355X: exactly one --compilation-config"
M5D="$(_argv_rank moriio 1 mori Kimi-K3-MXFP4-MI355X 2 2 2)"
_has    "$M5D" '"cudagraph_mode":"PIECEWISE"' "MI355X decode: cudagraph PIECEWISE"
_has    "$M5D" "mori_low_latency" "MI355X decode: all2all = mori_low_latency"
_has "$(cat "$SLURM")" 'CONNECTOR_ENV_ARGS+=" -e MORI_GPU_ARCHS=${MAD_GPU_ARCH}"' "slurm: MoRI JIT arch follows the detected GPU"

echo ""
echo "=== GPU-arch gate (cluster_require_gpu_arch) ==="
# Runs on the batch node under both CI paths, so this is where MADENGINE and
# STANDALONE agree on which GPUs a recipe may use. MAD_GPU_ARCH stands in for the
# probe. Strict mode, because run_multinode.slurm sources cluster.sh under it.
_gate() { # detected allowed [extra env]
  env -i PATH="$PATH" HOME="$HOME" MAD_GPU_ARCH="$1" ${3:+$3} bash -c \
    'set -euo pipefail; . "$0" >/dev/null; cluster_require_gpu_arch M "$1"' \
    "$DIR/../common/cluster.sh" "$2" >/dev/null 2>&1
}
_gate gfx942 gfx942 && { echo "  PASS  gfx942 recipe on gfx942 runs"; pass=$((pass+1)); } || { echo "  FAIL  gfx942 recipe on gfx942 refused"; fail=$((fail+1)); }
_gate gfx950 gfx942 && { echo "  FAIL  gfx942 recipe on gfx950 was allowed"; fail=$((fail+1)); } || { echo "  PASS  gfx942 recipe on gfx950 refused"; pass=$((pass+1)); }
_gate gfx950 "gfx942,gfx950" && { echo "  PASS  multi-arch recipe on gfx950 runs"; pass=$((pass+1)); } || { echo "  FAIL  multi-arch recipe on gfx950 refused"; fail=$((fail+1)); }
_gate gfx950 "" && { echo "  PASS  unrestricted recipe runs anywhere"; pass=$((pass+1)); } || { echo "  FAIL  unrestricted recipe refused"; fail=$((fail+1)); }
_gate gfx950 gfx942 GPU_ARCH_CHECK=0 && { echo "  PASS  GPU_ARCH_CHECK=0 bypasses"; pass=$((pass+1)); } || { echo "  FAIL  GPU_ARCH_CHECK=0 did not bypass"; fail=$((fail+1)); }
_has "$(cat "$SLURM")" 'cluster_require_gpu_arch "${MODEL_NAME}" "${GPU_ARCHS}"' "slurm gates on the recipe's GPU_ARCHS"
_has "$(cat "$SLURM")" '${PERF_GPU_ARCH:+-e PERF_GPU_ARCH=' "slurm forwards the detected arch for the perf CSV"
if python3 "$DIR/../common/check_gpu_arch_declarations.py" "$DIR/../.." >/dev/null; then
  echo "  PASS  card skip_gpu_arch agrees with recipe GPU_ARCHS"; pass=$((pass+1))
else
  echo "  FAIL  card skip_gpu_arch disagrees with recipe GPU_ARCHS:"; fail=$((fail+1))
  python3 "$DIR/../common/check_gpu_arch_declarations.py" "$DIR/../.." | sed 's/^/        /'
fi

echo ""
echo "=== rixl passes the resolved GPU_MEMORY_UTILIZATION (as moriio does) ==="
# SLURM jobs 445932 / 446044: rixl/TP passed no --gpu-memory-utilization, so vLLM used its
# built-in 0.92 and RCCL had no room for its first all-reduce.
Rt="$(_argv rixl 0 '' amd-Llama-3.3-70B-Instruct-FP8-KV /m/Llama)"
_hasadj "$Rt" "--gpu-memory-utilization" "0.8" "rixl TP: launcher fallback reaches vllm serve"
Rd="$(_argv rixl 1 deepep DeepSeek-V3 /m/DSV3)"
_hasadj "$Rd" "--gpu-memory-utilization" "0.8" "rixl deepep: launcher fallback reaches vllm serve"
Ro="$(env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=1 yD=1 CONNECTOR=rixl WIDE_EP=0 EP_BACKEND='' \
    MODEL_NAME=amd-Llama-3.3-70B-Instruct-FP8-KV MODEL_PATH=/m/Llama MASTER_ADDR=10.0.0.1 \
    IPADDRS=10.0.0.1,10.0.0.2 GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router \
    ROUTER_PORT=30000 GPU_MEMORY_UTILIZATION=0.7 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f')"
_hasadj "$Ro" "--gpu-memory-utilization" "0.7" "rixl TP: a submit-time GPU_MEMORY_UTILIZATION wins"
_count "$Ro" "--gpu-memory-utilization" 1 "rixl TP: the flag appears once"

echo "=== llama-70B TP: RCCL's MSCCL path is off (its scratch buffers filled the GPU) ==="
# The same export vllm_disagg.sh runs before vllm serve, taken from the script itself.
_YENV="$(awk '/_yaml_env="\$\(python3 - <<.PY.$/{f=1;next} f&&/^PY$/{exit} f' "$DIR/vllm_disagg.sh")"
_yexp() { MODELS_YAML="$DIR/models.yaml" MODEL_NAME="$1" python3 -c "$_YENV"; }
_has    "$(_yexp amd-Llama-3.3-70B-Instruct-FP8-KV)" "export RCCL_MSCCL_ENABLE=0" "llama-70B exports RCCL_MSCCL_ENABLE=0"
_hasnot "$(_yexp Qwen3-32B)" "RCCL_MSCCL_ENABLE" "scoped to llama-70B (Qwen3-32B unchanged)"
_hasnot "$(RCCL_MSCCL_ENABLE=1 MODELS_YAML="$DIR/models.yaml" MODEL_NAME=amd-Llama-3.3-70B-Instruct-FP8-KV python3 -c "$_YENV")" "RCCL_MSCCL_ENABLE" "a submit-time RCCL_MSCCL_ENABLE still wins"

echo "=== rixl deepep honours the recipe's KV knobs (as moriio does) ==="
# SLURM job 445925: DeepSeek-V3 on rixl/deepep ran with a hardcoded --block-size 1 and GPU-faulted.
Kd="$(_argv rixl 1 deepep DeepSeek-V3 /m/DSV3)"
_hasadj "$Kd" "--block-size" "16" "rixl deepep DeepSeek-V3: KV_BLOCK_SIZE=16 from models.yaml"
_hasadj "$Kd" "--kv-cache-memory-bytes" "20000000000" "rixl deepep DeepSeek-V3: KV_CACHE_MEMORY_BYTES from models.yaml"
_hasadj "$Kd" "--kv-cache-dtype" "fp8" "rixl deepep DeepSeek-V3: KV_CACHE_DTYPE from models.yaml"
_count "$Kd" "--block-size" 1 "rixl deepep: --block-size appears once"
Km="$(_argv moriio 1 mori DeepSeek-V3 /m/DSV3)"
_hasadj "$Km" "--block-size" "16" "moriio DeepSeek-V3: same block size (reference)"

echo "======================================================"
echo "  argv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
