#!/bin/bash
# ============================================================================================
# SUPERSEDED (2026-06-28). The proven per-node pattern (host pre-clean via clean_node.sh,
# detached -d launch to dodge the empty-log race, and server+bench as SIBLING containers in
# ONE --exclusive srun to avoid the --overlap deadlock) is now folded into the single source
# of truth: lib/run_engine.sh (launch_external_bench_node). Prefer:
#     ENGINE=atom MODEL=qwen3-next-fp8 TP=4 ACTION=sweep bash lib/run_engine.sh
# This standalone driver is kept only as a minimal one-node reproducer for the ATOM team.
# ============================================================================================
# sweep_config.sh — run ONE (engine, TP) FP8 config on ONE node, full-node via MANUAL REPLICAS.
# Engine-native DP is BROKEN for this FP8 model (see docs/native_DP_broken_fp8.md), so we launch
# N = 8/TP INDEPENDENT single-server replicas, each plain TP-k on its own GPU group + own port.
# Replicas are identical & independent -> per-replica metrics; full-node aggregate = per-replica
# throughput x N. For the sweep we benchmark ONE representative replica (replica 0) per point
# (all replicas identical); node aggregate is computed offline as (metric x N).
#
# Args (env): ENGINE=vllm|sglang|atom  TP=1|2|4  NODE=<node>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts_qwen_v2
RESV=<your_slurm_reservation>
MROOT=/path/to/models
MDIR=Qwen3-Next-80B-A3B-Instruct-FP8
ENGINE="${ENGINE:?}"; TP="${TP:?}"; NODE="${NODE:?}"
DP=$(( 8 / TP ))          # number of independent replicas to fill the node
MML=17408                 # covers 16384+1024
SHAPES="${SHAPES:-1024,1024 8192,1024 16384,1024}"
CONC="${CONC:-4 8 16 32 64 128 256}"
STAMP=$(python3 -c "import time;print(time.strftime('%Y%m%d_%H%M%S'))")
RES="$HERE/../results_v2/sweep_fp8_${ENGINE}_tp${TP}_x${DP}_${STAMP}"; mkdir -p "$RES"; chmod 777 "$RES"

VLLM_IMG=vllm/vllm-openai-rocm:nightly-9037498c22891e55b594f567fb91d9b4efbf3e99
SGL_IMG=lmsysorg/sglang:v0.5.14-rocm720-mi35x
ATOM_IMG=rocm/atom-dev:vllm-v0.22.0-nightly_20260617
case "$ENGINE" in vllm) IMG=$VLLM_IMG;; sglang) IMG=$SGL_IMG;; atom) IMG=$ATOM_IMG;; esac
BENCH_IMG=$VLLM_IMG       # clean bench client for ALL engines (uniform; sidesteps ATOM bench crash)

# GPUs for replica 0 = first TP GPUs; we benchmark replica 0 (all replicas identical).
GPUS0=$(seq -s, 0 $((TP-1)))
PORT=8000
sched="--no-async-scheduling"; [[ $TP -gt 1 ]] && sched="--async-scheduling"
# Knobs aligned with the run_engine.sh vLLM/SGLang sweep for apples-to-apples:
# gpu_mem_util=0.8, max_num_seqs=256, max_model_len=17408 (=$MML).
case "$ENGINE" in
  vllm)   SRVCMD="vllm serve $MROOT/$MDIR --host 0.0.0.0 --port $PORT --tensor-parallel-size $TP --max-model-len $MML --gpu-memory-utilization 0.8 --max-num-seqs 256 --no-enable-prefix-caching $sched --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype fp8" ;;
  sglang) SRVCMD="python3 -m sglang.launch_server --model-path $MROOT/$MDIR --host 0.0.0.0 --port $PORT --tp-size $TP --context-length $MML --mem-fraction-static 0.8 --max-running-requests 256 --trust-remote-code --disable-radix-cache --attention-backend aiter" ;;
  atom)   SRVCMD="python -m atom.entrypoints.openai_server --model $MROOT/$MDIR -tp $TP --port $PORT --max-model-len $MML --gpu-memory-utilization 0.8 --max-num-seqs 256 --no-enable_prefix_caching" ;;
esac

echo "[sweep] $ENGINE TP=$TP (rep0 of $DP) node=$NODE -> $RES"
echo "[sweep] NOTE: benchmarking replica 0 (1 TP-$TP server on GPUs $GPUS0); node aggregate = metric x $DP"

# Pre-flight clean on the host (frees VRAM zombies from prior --rm containers; the #1 cause of
# ATOM OOM-at-init). Weak `docker kill` alone misses detached zombies — use the full cleaner.
echo "[sweep] pre-flight clean $NODE"
srun --reservation="$RESV" --nodelist="$NODE" --nodes=1 --ntasks=1 --overlap --mem=0 \
  bash "$HERE/lib/clean_node.sh" > "$RES/clean.log" 2>&1 || echo "[sweep] WARN clean (see clean.log)"

srun --reservation="$RESV" --nodelist="$NODE" --nodes=1 --ntasks=1 --exclusive bash -c "
  set -uo pipefail
  docker rm -f sweep_${ENGINE}_tp${TP} >/dev/null 2>&1 || true   # remove any lingering named container (name conflict guard)
  sleep 4
  echo '[node] start $ENGINE TP=$TP server on GPUs $GPUS0'
  # Launch DETACHED (-d) so we have a real container id. Logs via 'docker logs' to server.log.
  # (Old approach raced: 'docker run ... &' + immediate 'docker ps --filter name' on a 75GB image
  #  saw no container on the first loop tick -> false 'exited' with an empty log.)
  cid=\$(docker run -d --name sweep_${ENGINE}_tp${TP} --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc host --shm-size 128G --network host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=$GPUS0 -e VLLM_ROCM_USE_AITER=1 -e SAFETENSORS_FAST_GPU=1 -e SGLANG_USE_AITER=1 \
    -v $MROOT:$MROOT \
    $IMG -c '$SRVCMD')
  echo \"[node] container \$cid\"
  up=0; for i in \$(seq 1 180); do
    docker logs sweep_${ENGINE}_tp${TP} > $RES/server.log 2>&1 || true
    curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { up=1; break; }
    # only declare exit if the container is actually gone (not just slow to register)
    st=\$(docker inspect -f '{{.State.Status}}' sweep_${ENGINE}_tp${TP} 2>/dev/null || echo missing)
    [ \"\$st\" = exited ] || [ \"\$st\" = missing ] && { echo \"[node] server container \$st\"; break; }
    sleep 5
  done
  docker logs sweep_${ENGINE}_tp${TP} > $RES/server.log 2>&1 || true
  echo \"[node] SERVER_UP=\$up\"
  if [ \$up -ne 1 ]; then echo '[node] FAILED tail:'; tail -25 $RES/server.log; docker kill sweep_${ENGINE}_tp${TP} >/dev/null 2>&1; exit 1; fi
  for shape in $SHAPES; do
    ISL=\${shape%,*}; OSL=\${shape#*,}
    for c in $CONC; do
      echo \"[bench] isl=\$ISL osl=\$OSL conc=\$c\"
      docker run --rm --entrypoint bash --network host -v $HERE:/scripts -v $MROOT:$MROOT -v $RES:/out \
        $BENCH_IMG -c \"python3 /scripts/utils/bench_serving/benchmark_serving.py \
          --model $MROOT/$MDIR --backend vllm --base-url http://0.0.0.0:$PORT --dataset-name random \
          --random-input-len \$ISL --random-output-len \$OSL --random-range-ratio 1.0 \
          --num-prompts \$(( c * 10 )) --num-warmups \$(( c * 2 )) --max-concurrency \$c \
          --request-rate inf --ignore-eos --save-result --percentile-metrics ttft,tpot,itl,e2el \
          --result-dir /out --result-filename sweep_isl\${ISL}_osl\${OSL}_c\${c}.json\" \
        > $RES/bench_isl\${ISL}_osl\${OSL}_c\${c}.log 2>&1 || echo \"  [bench] FAILED isl=\$ISL conc=\$c\"
    done
  done
  echo '[node] standdown'; docker kill sweep_${ENGINE}_tp${TP} >/dev/null 2>&1; sleep 3
" > "$RES/run.log" 2>&1
echo "[sweep] $ENGINE TP=$TP done -> $RES"
