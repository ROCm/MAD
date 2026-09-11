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
