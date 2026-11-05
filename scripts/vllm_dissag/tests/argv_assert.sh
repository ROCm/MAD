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
echo "=== slurm forwards DSV4 MoRI-EP / NIAH knobs ==="
_has "$S" '${DSV4_EAGER:+-e DSV4_EAGER=' "slurm forwards DSV4_EAGER"
_has "$(cat "$DIR/connectors/moriio.sh")" "[dsv4-patch-roster]" "moriio dumps patch roster"
_has "$(cat "$DIR/connectors/moriio.sh")" "s/^DSV4_PATCHES_BAKED=//p" "patch roster reads the image's DSV4_PATCHES_BAKED stamp"
_has "$S" '-e SKIP_RUNTIME_PATCH=${SKIP_RUNTIME_PATCH:-1}' "slurm defaults SKIP_RUNTIME_PATCH=1 (patches baked)"
_has "$S" '${DSV4_TRANSFER_ATTN:+-e DSV4_TRANSFER_ATTN=' "slurm forwards DSV4_TRANSFER_ATTN"

# The MoRI dispatch trim is vllm#63412, baked unconditionally into the DSV4
# image. There is no MORI_TRIM_DISPATCH knob left to drop.
echo ""
echo "=== MoRI dispatch trim is baked, with no knob ==="
M="$(cat "$DIR/connectors/moriio.sh")"
W="$(cat "$DIR/run_wideep_bench.sh")"
_has    "$M" "patches=BAKED" "moriio reports patches baked"
_hasnot "$M" "apply_mori_trim_dispatch.py" "moriio does not runtime-apply trim"
_hasnot "$M" "MORI_TRIM_DISPATCH" "connector has no trim knob"
_hasnot "$S" "MORI_TRIM" "slurm forwards no trim knob"
_hasnot "$W" "MORI_TRIM" "wrapper sets no trim knob"
_hasnot "$W" 'SKIP_RUNTIME_PATCH="${SKIP_RUNTIME_PATCH:-0}"' \
        "wrapper does not force SKIP_RUNTIME_PATCH=0"
_TRIM="$(python3 - "$DIR/models.yaml" <<'PY'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])) or {}
hits = [m for m, c in sorted(y.items())
        if isinstance(c, dict) and "MORI_TRIM_DISPATCH" in (c.get("env") or {})]
print("models.yaml trim entries: " + (", ".join(hits) if hits else "NONE"))
PY
)"
_has "$_TRIM" "NONE" "models.yaml does not pin trim"
# The patches come from a vLLM fork branch cut at the base image's own vLLM
# commit; the image fetches them at build.
_DN="$(cat "$DIR/../../docker/vllm_disagg_inference.dsv4.ubuntu.amd.Dockerfile")"
_has "$_DN" "ARG BASE_IMAGE=vllm/vllm-openai-rocm:nightly-rocm72-83c078d955cd6e0c63d3b12abe669c8de936a370" "DSV4 Dockerfile is FROM the 83c078d rocm72 nightly"
_has "$_DN" "ARG PATCHES_REPO=https://github.com/basemam/vllm.git" "DSV4 Dockerfile fetches patches from the vLLM fork"
_has "$_DN" "ARG PATCHES_PATH=mad/scripts/vllm_disagg/patches" "DSV4 Dockerfile patch path in fork"
_has "$_DN" "bash /opt/dsv4_patches/bake.sh" "DSV4 Dockerfile runs bake.sh"
_has "$_DN" "ARG ROUTER_REF=37b37a43c577cd7d83bad7b10b8b4650325b4fc7" "DSV4 Dockerfile bakes vllm-router 37b37a43"
_hasnot "$_DN" "MORI_REF" "DSV4 Dockerfile keeps the base image's MoRI"
_hasnot "$_DN" "VLLM_REF" "DSV4 Dockerfile keeps the base image's vLLM"
_hasnot "$_DN" "COPY scripts/vllm_dissag/dsv4_patches/" "DSV4 Dockerfile does not copy local patches"
_Y4="$(python3 - "$DIR/models.yaml" <<'PY'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])) or {}
for m in ("DeepSeek-V4-Flash-FP8", "DeepSeek-V4-Pro-FP8"):
    c = y[m]
    f = c.get("dp_flags", "")
    e = c.get("env") or {}
    ok = ("--disable-hybrid-kv-cache-manager" in f and "--no-async-scheduling" in f
          and e.get("VLLM_USE_V2_MODEL_RUNNER") == "0")
    print(f"{m}={'OK' if ok else 'MISSING'}")
PY
)"
_has "$_Y4" "DeepSeek-V4-Flash-FP8=OK" "Flash yaml: HMA off, async off, V1 runner"
_has "$_Y4" "DeepSeek-V4-Pro-FP8=OK" "Pro yaml: HMA off, async off, V1 runner"
_hasnot "$(ls -d "$DIR/dsv4_patches" 2>/dev/null)" "dsv4_patches" "cookbook has no dsv4_patches dir"
APPLY_ROOT="$(ls "$DIR"/apply_*.py 2>/dev/null || true)"
_hasnot "$APPLY_ROOT" "apply_" "cookbook root has no apply_*.py"

# The container-side init section. models.yaml is loaded INSIDE the container by
# vllm_disagg.sh with a bare `import yaml`, twice, and it resolves both the
# per-model env block and the per-role serve flags -- so PyYAML is a critical
# dependency of the serve path, not a reporting nicety. Images are a moving
# target, so the init step must keep covering it. pandas is the reporting half
# (benchmark_parser.py, the only ITL/TTFT/TPOT reporter).
echo ""
echo "=== container creation barrier is bounded ==="
B="$(cat "$DIR/socket_barrier.py")"
V="$(cat "$DIR/vllm_disagg.sh")"
_has "$B" '"--timeout"' "barrier accepts a deadline"
_has "$B" 'never opened their port' "barrier names the peers that never showed"
_has "$B" 'default=0' "barrier default stays wait-forever for the other callers"
_has "$B" 'failed to bind local barrier port' "barrier bind failure is fatal"
_has "$V" 'CONTAINER_BARRIER_TIMEOUT_SECONDS:-1800' "launcher bounds the barrier"
_has "$V" 'container creation barrier failed' "launcher aborts when the barrier times out"
_has "$V" 'tcp_listen_kill' "launcher clears the barrier port without ss"

echo ""
echo "=== container init ensures the python deps the serve path imports ==="
_has "$V" "Initialization — container-side boot setup" "vllm_disagg has an init section"
_has "$V" "_ensure_py_deps" "init defines the dep check"
_has "$V" "_init_container_env" "init has an umbrella hook for future steps"
_has "$V" 'PY_BOOT_DEPS-yaml:PyYAML pandas' "init default covers BOTH PyYAML and pandas"
_has "$V" "import os, yaml, shlex" "models.yaml is still parsed with a bare import yaml"
_has "$V" 'PyYAML absent' "a failed PyYAML install says the serve path is affected"
_has "$V" 'pandas absent is harmless' "a failed pandas install says reporting only"
# Nothing in init may abort: a compute node can be offline from PyPI, and no
# reporting library is worth losing a multi-hour serving cell.
_INIT="$(python3 - "$DIR/vllm_disagg.sh" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'^_ensure_py_deps\(\) \{.*?^\}', s, re.S | re.M)
body = m.group(0) if m else ""
print("FOUND" if body else "MISSING",
      "ABORTS" if re.search(r'\b(exit 1|exit 2)\b', body) else "NONFATAL")
PY
)"
_has "$_INIT" "FOUND NONFATAL" "init dep step is non-fatal (no exit on failure)"

_has "$S" '${NIAH_HALT_ON_FAIL:+-e NIAH_HALT_ON_FAIL=' "slurm forwards NIAH_HALT_ON_FAIL"
_has "$S" '${NIAH_LIST_PRIME:+-e NIAH_LIST_PRIME=' "slurm forwards NIAH_LIST_PRIME"
_has "$S" 'validate)     BENCHMARK_SCRIPT_FILE="benchmark_validate.sh"' "slurm maps validate -> benchmark_validate.sh"
_has "$S" 'accuracy)     BENCHMARK_SCRIPT_FILE="benchmark_accuracy.sh"' "slurm maps accuracy -> benchmark_accuracy.sh"
_has "$S" '-e ACC_THINKING=$ACC_THINKING' "slurm forwards ACC_THINKING"
_hasnot "$S" '[gpu-free]' "slurm has no pre-docker VRAM wait (boot matches the PR branch)"
_hasnot "$M" '_dsv4_boot_reapply' "moriio does not re-run patch scripts at boot (baked at build)"
_hasnot "$M" 'apply_[a-z0-9_]+' "moriio runs no apply_*.py"
_has "$M" "grep -q '^DSV4_PATCHES_BAKED=' /app/versions.txt" "moriio checks the bake.sh stamp"
_hasnot "$M" "grep -q '_original_topk_ids'" "moriio does not re-grep mori.py markers (bake.sh did at build)"
_has "$S" '${NIAH_STOP:+-e NIAH_STOP=' "slurm forwards NIAH_STOP"
_has "$S" '${NIAH_LOGPROBS:+-e NIAH_LOGPROBS=' "slurm forwards NIAH_LOGPROBS"
_has "$S" '${NIAH_TERSE:+-e NIAH_TERSE=' "slurm forwards NIAH_TERSE"
_has "$S" '${NIAH_STOP_BLANK:+-e NIAH_STOP_BLANK=' "slurm forwards NIAH_STOP_BLANK"
_has "$S" '${NIAH_PAIR_SLEEP_S:+-e NIAH_PAIR_SLEEP_S=' "slurm forwards NIAH_PAIR_SLEEP_S"
_has "$S" '${NIAH_WARMUP_SCORE_SLEEP_S:+-e NIAH_WARMUP_SCORE_SLEEP_S=' "slurm forwards NIAH_WARMUP_SCORE_SLEEP_S"
_has "$S" '${DSV4_STORAGE_UPSTREAM_SPAN:+-e DSV4_STORAGE_UPSTREAM_SPAN=' "slurm forwards DSV4_STORAGE_UPSTREAM_SPAN"
_has "$S" '${CONTAINER_BARRIER_TIMEOUT_SECONDS:+-e CONTAINER_BARRIER_TIMEOUT_SECONDS=' "slurm forwards CONTAINER_BARRIER_TIMEOUT_SECONDS"
_has "$S" '${ROUTER_BOOT_INSTALL:+-e ROUTER_BOOT_INSTALL=' "slurm forwards ROUTER_BOOT_INSTALL"
_has "$S" '${ROUTER_REPO:+-e ROUTER_REPO=' "slurm forwards ROUTER_REPO"
_has "$S" '${ROUTER_REF:+-e ROUTER_REF=' "slurm forwards ROUTER_REF"
W="$(cat "$DIR/run_wideep_bench.sh")"
_has "$W" 'DOCKER_IMAGE_NAME="${IMAGE_ARG:-${DOCKER_IMAGE_NAME:-}}"' "wrapper takes the image from --image or DOCKER_IMAGE_NAME"
_hasnot "$W" 'pytorch-private' "wrapper names no private image"
_has "$W" 'EXTRA_ENV+=(DSV4_ENABLE_HMA=0)' "wrapper pins DSV4 HMA=0"
_hasnot "$W" 'DSV4_ENABLE_HMA="${DSV4_ENABLE_HMA' "wrapper does not take HMA from the submit env"
_has "$(cat "$DIR/connectors/moriio.sh")" 'export DSV4_ENABLE_HMA=0' "moriio forces HMA=0 at boot"
_has "$W" 'unset NIAH_LIST_PRIME' "wrapper accuracy unsets NIAH_LIST_PRIME"
_has "$W" 'NIAH_LIST_PRIME="1."' "wrapper Flash NIAH defaults list-prime 1."
_has "$W" 'Pro NIAH: never prime' "wrapper Pro NIAH forces the unprimed stem"
_has "$W" 'BENCHMARK_SCRIPT=validate' "wrapper validate sets BENCHMARK_SCRIPT"
_has "$W" 'BENCHMARK_SCRIPT=accuracy' "wrapper accuracy sets BENCHMARK_SCRIPT"
_has "$(cat "$DIR/benchmark_accuracy.sh")" '/ready' "accuracy waits on proxy /ready"
_hasnot "$W" 'ROUTER_BOOT_INSTALL="${ROUTER_BOOT_INSTALL:-git}"' "wrapper does not git-boot vllm-router (baked in the DSV4 image)"

echo ""
echo "=== moriio + wideEP (DeepSeek-V4-Flash-FP8) ==="
C="$(_argv moriio 1 mori DeepSeek-V4-Flash-FP8 /m/DSV4F)"
_has    "$C" "--enable-expert-parallel" "DSV4 has --enable-expert-parallel"
_has    "$C" "--data-parallel-size" "DSV4 has --data-parallel-size"
_has    "$C" "deepseek_v4" "DSV4 tokenizer-mode deepseek_v4"
_has    "$C" "--block-size" "DSV4 has --block-size"
_has    "$C" "256" "DSV4 block-size 256 present"
_count  "$C" "--compilation-config" 1 "DSV4 exactly one --compilation-config"
_hasnot "$C" "--tensor-parallel-size" "DSV4 no --tensor-parallel-size (uses -tp 1)"

echo ""
echo "=== shared NIAH path keeps develop's behaviour for non-DSV4 models ==="
# These four defaults are never read on a DSV4 run (the wrapper sets NIAH_METHOD
# and NIAH_MAXTOK explicitly in both model blocks), so they exist purely to leave
# GLM / DSV3 / Hy3 on the behaviour they had before this branch.
NSH="$(cat "${DIR}/benchmark_niah.sh")"
NPY="$(cat "${DIR}/benchmark_niah.py")"
_has "$NPY" 'os.environ.get("NIAH_METHOD", "product")' "niah.py defaults to the model-agnostic product stem"
_has "$NSH" 'NIAH_MAXTOK="${NIAH_MAXTOK:-2048}"' "niah.sh keeps develop's 2048 answer budget"
_has "$NSH" '/v1/models' "niah.sh falls back to the /v1/models readiness probe"
_has "$NSH" 'WARN: neither /ready' "niah.sh readiness failure is non-fatal"
_has "$NSH" 'parse_to_csv.py' "niah.sh still emits madengine perf.csv rows"
_has "$NPY" 'no scored answer' "niah.py exits 1 when every request is NO-RESULT"
_has "$NSH" 'every request is NO-RESULT' "niah.sh documents harness NO-RESULT as niah_rc"
_has "$V" 'exit "${_BENCH_RC:-0}"' "launcher returns the benchmark script status"
_has "$(cat "$DIR/benchmark_xPyD.sh")" 'sweep_rc=$rc' "xPyD tracks cell failures past tee"
_TCP_CHECK="$(python3 - <<'PY'
line = "  46: 010310AC:9C4C 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 57245 1 ffff8803c23ec000 100 0 0 10 -1"
f = line.split()
print("inode="+f[9], "uid="+f[7], "ptr="+f[11])
PY
)"
_has "$_TCP_CHECK" "inode=57245" "awk \$10 on /proc/net/tcp is inode (not uid)"
_has "$_TCP_CHECK" "uid=0" "awk \$8 on /proc/net/tcp is uid"
_has "$(cat "$DIR/tcp_listen.sh")" 'print $10' "tcp_listen_inodes prints awk inode field \$10"
_has "$W" 'NIAH_METHOD="${NIAH_METHOD:-hybrid}"' "wrapper sets DSV4 method explicitly"
_has "$W" 'NIAH_MAXTOK="${NIAH_MAXTOK:-512}"' "wrapper sets DSV4 maxtok explicitly"

_has "$(cat "$DIR/run_wideep_bench.sh")" 'MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-31367}"' "router discovery port defaults below the ephemeral range"
_has "$(cat "$DIR/connectors/moriio.sh")" "did not register with the router" "router registration timeout is fatal"
_has "$(cat "$DIR/connectors/moriio.sh")" "could not bind discovery port" "router discovery bind failure is fatal"
_hasnot "$(cat "$DIR/connectors/moriio.sh" "$DIR/run_wideep_bench.sh" "$DIR/vllm_disagg.sh")" ":-36367" "no default inside the ephemeral range"
_has "$(cat "$DIR/vllm_disagg.sh")" "worker died during boot" "boot wait fails fast on a dead worker"
_has "$(cat "$DIR/vllm_disagg.sh")" '_mark_boot_fatal "worker died during boot' "boot fail-fast writes the BOOT_FATAL marker"
_has "$(cat "$DIR/vllm_disagg.sh")" '_mark_boot_fatal "timeout waiting for' "boot timeout writes the BOOT_FATAL marker"
_has "$(cat "$DIR/connectors/moriio.sh")" '_mark_boot_fatal "router registration timeout"' "router registration timeout writes the BOOT_FATAL marker"
_has "$(cat "$DIR/run_xPyD_models.slurm")" 'srun --kill-on-bad-exit=1 --nodelist=' "main step ends when one node exits nonzero"
_has "$(cat "$DIR/run_xPyD_models.slurm")" 'BOOT_FATAL" ]; then' "host task exits nonzero on BOOT_FATAL"
_has "$(cat "$DIR/run_xPyD_models.slurm")" 'rm -f "$LOG_PATH/$SLURM_JOB_ID/BOOT_FATAL"' "stale marker cleared before launch (requeue)"

echo ""
echo "======================================================"
echo "  argv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
