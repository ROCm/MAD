#!/bin/bash
# run_engine.sh — generic launcher for ANY engine across the cluster, given (ENGINE, MODEL, TP, DP).
# Called by run_vllm.sh / run_sglang.sh / run_atom.sh (which just set ENGINE).
# Aggregated serving; each replica node-local. Scales 1->6 nodes via cluster.yaml `nodes`.
#
# Env in: ENGINE, MODEL, TP, [DP], [ACTION=sanity|serve], [MAX_MODEL_LEN]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # scripts_qwen_v2
CLUSTER="$HERE/cluster.yaml"; MODELS="$HERE/model.yaml"
ENGINE="${ENGINE:?}"; MODEL="${MODEL:?}"; TP="${TP:?}"; ACTION="${ACTION:-sanity}"
cfg(){ python3 "$HERE/lib/cfg.py" "$@"; }

RESV=$(cfg "$CLUSTER" cluster reservation)
ALLOC=$(cfg "$CLUSTER" cluster srun_alloc); ALLOC="${ALLOC:---reservation=$RESV}"
GPN=$(cfg "$CLUSTER" cluster gpus_per_node)
MROOT=$(cfg "$CLUSTER" cluster models_root)
RROOT=$(cfg "$CLUSTER" cluster results_root)
DP="${DP:-$((GPN / TP))}"
MDIR=$(cfg "$MODELS" model "$MODEL" dir)
VALID_TP=$(cfg "$MODELS" model "$MODEL" valid_tp)
IMG=$(cfg "$MODELS" engine "$ENGINE" image)
SERVE_FLAGS=$(cfg "$MODELS" engine "$ENGINE" serve_flags)
ENTRY=$(cfg "$MODELS" engine "$ENGINE" entrypoint_override); ENTRY="${ENTRY:-bash}"
PATCH=$(cfg "$MODELS" engine "$ENGINE" patch_mount)
BENCH_EXTERNAL=$(cfg "$MODELS" engine "$ENGINE" bench_external)   # "True"/""
BENCH_IMAGE=$(cfg "$MODELS" engine "$ENGINE" bench_image)
[[ "$BENCH_EXTERNAL" == "True" ]] && BX=1 || BX=0

# shared logical knobs: env override > per-model override > global defaults{}
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(cfg "$MODELS" default "$MODEL" max_model_len)}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-$(cfg "$MODELS" default "$MODEL" gpu_memory_util)}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(cfg "$MODELS" default "$MODEL" max_num_seqs)}"

case " $VALID_TP " in *" $TP "*) :;; *) echo "ERROR: TP=$TP not in valid_tp=[$VALID_TP] for $MODEL"; exit 1;; esac
[[ $((DP*TP)) -le $GPN ]] || { echo "ERROR: DP*TP=$((DP*TP)) > gpus_per_node=$GPN"; exit 1; }

# engine env vars (space-separated k=v from yaml) -> docker -e args
ENV_ARGS=""
for kv in $(cfg "$MODELS" engine "$ENGINE" env); do ENV_ARGS+=" -e $kv"; done
PATCH_ARG=""; [[ -n "$PATCH" ]] && PATCH_ARG="-v $HERE/$PATCH"

STAMP=$(python3 -c "import time;print(time.strftime('%Y%m%d_%H%M%S'))")
RUNROOT="$RROOT/${MODEL}_${ENGINE}_tp${TP}_dp${DP}_${STAMP}"; mkdir -p "$RUNROOT"; chmod 777 "$RUNROOT"
echo "[$ENGINE] MODEL=$MODEL TP=$TP DP=$DP action=$ACTION image=$IMG"
NODES_ARG=""; [[ -n "${NODES:-}" ]] && NODES_ARG="--nodes $NODES"
python3 "$HERE/lib/placement.py" --cluster "$CLUSTER" --tp "$TP" --dp "$DP" $NODES_ARG > "$RUNROOT/placement.tsv"
echo "[$ENGINE] placement ($(wc -l < "$RUNROOT/placement.tsv") replicas):"; sed 's/^/    /' "$RUNROOT/placement.tsv"
echo "[$ENGINE] config: max_model_len=$MAX_MODEL_LEN gpu_mem_util=$GPU_MEM_UTIL max_num_seqs=$MAX_NUM_SEQS"

# --- PRE-FLIGHT: ensure each target node's GPUs are CLEAN before deploying (kills zombie VRAM) ---
UNIQ_NODES=$(cut -f1 "$RUNROOT/placement.tsv" | sort -u)
echo "[$ENGINE] cleaning GPUs on: $(echo $UNIQ_NODES | tr '\n' ' ')"
for node in $UNIQ_NODES; do
  srun $ALLOC --nodelist="$node" --nodes=1 --ntasks=1 --overlap --mem=2G \
    bash "$HERE/lib/clean_node.sh" > "$RUNROOT/clean_${node}.log" 2>&1 \
    && echo "  [clean] $node OK" || echo "  [clean] $node WARN (see clean_${node}.log)"
done

# --- IN-CONTAINER BENCH (vLLM/SGLang, BX=0) -------------------------------------------------
# The bench client runs INSIDE the engine container via replica_entry.sh (ACTION drives it).
# One backgrounded --overlap srun per replica; replicas are independent and run concurrently.
launch_replica() {
  local node="$1" gpus="$2" port="$3" idx="$4"
  local rdir="$RUNROOT/replica_${idx}_${node}_p${port}"; mkdir -p "$rdir"; chmod 777 "$rdir"
  srun $ALLOC --nodelist="$node" --nodes=1 --ntasks=1 --overlap --mem=0 bash -c "
    docker run --rm --entrypoint $ENTRY \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc host --shm-size 64G --network host \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -e HIP_VISIBLE_DEVICES=$gpus -e CUDA_VISIBLE_DEVICES=$gpus \
      $ENV_ARGS \
      -e ENGINE_KIND=$ENGINE -e ACTION=$ACTION -e PORT=$port -e TP=$TP -e BENCH_EXTERNAL=$BX \
      -e MAX_MODEL_LEN=$MAX_MODEL_LEN -e GPU_MEM_UTIL=$GPU_MEM_UTIL -e MAX_NUM_SEQS=$MAX_NUM_SEQS \
      -e SMOKE_CONC=${SMOKE_CONC:-8} -e SMOKE_ISL=${SMOKE_ISL:-128} -e SMOKE_OSL=${SMOKE_OSL:-128} \
      -e SWEEP_SHAPES='${SWEEP_SHAPES:-1024,1024 8192,1024 16384,1024}' -e SWEEP_CONC='${SWEEP_CONC:-4 8 16 32 64 128 256}' \
      -e MODEL_PATH=$MROOT/$MDIR -e SERVE_FLAGS='$SERVE_FLAGS' \
      -v $MROOT:$MROOT -v $HERE:/scripts -v $rdir:/out $PATCH_ARG \
      $IMG /scripts/lib/replica_entry.sh
  " > "$rdir/replica.log" 2>&1 &
}

# --- EXTERNAL BENCH (ATOM, BX=1) ------------------------------------------------------------
# ATOM's vendored bench client crashes inside its own image, so the bench must run from a CLEAN
# (vLLM) container. CRITICAL: server + bench run as SIBLING containers inside ONE --exclusive
# srun on the node. The old design used a 2nd `--overlap` bench srun, which DEADLOCKED: the
# server's `--mem=0` srun claims all node memory, so the bench srun's `--mem=4G` request never
# schedules. One srun = one allocation = no contention. (Proven pattern; see git history.)
launch_external_bench_node() {
  local node="$1" gpus="$2" port="$3" idx="$4"
  local rdir="$RUNROOT/replica_${idx}_${node}_p${port}"; mkdir -p "$rdir"; chmod 777 "$rdir"
  local cname="extbench_${ENGINE}_tp${TP}_${idx}"
  local shapes="${SWEEP_SHAPES:-1024,1024 8192,1024 16384,1024}"
  local concs="${SWEEP_CONC:-4 8 16 32 64 128 256}"
  [[ "$ACTION" == "sanity" ]] && { shapes="${SMOKE_ISL:-128},${SMOKE_OSL:-128}"; concs="${SMOKE_CONC:-8}"; }
  srun $ALLOC --nodelist="$node" --nodes=1 --ntasks=1 --exclusive bash -c "
    set -uo pipefail
    docker rm -f $cname >/dev/null 2>&1 || true
    sleep 4
    echo '[node] start $ENGINE TP=$TP server on GPUs $gpus :$port'
    # Detached (-d): get a real container id and stream logs; avoids the empty-log launch race.
    cid=\$(docker run -d --name $cname --entrypoint $ENTRY \
      --device=/dev/kfd --device=/dev/dri --group-add video --ipc host --shm-size 64G --network host \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -e HIP_VISIBLE_DEVICES=$gpus -e CUDA_VISIBLE_DEVICES=$gpus $ENV_ARGS \
      -e ENGINE_KIND=$ENGINE -e ACTION=serve -e PORT=$port -e TP=$TP \
      -e MAX_MODEL_LEN=$MAX_MODEL_LEN -e GPU_MEM_UTIL=$GPU_MEM_UTIL -e MAX_NUM_SEQS=$MAX_NUM_SEQS \
      -e MODEL_PATH=$MROOT/$MDIR -e SERVE_FLAGS='$SERVE_FLAGS' \
      -v $MROOT:$MROOT -v $HERE:/scripts -v $rdir:/out $PATCH_ARG \
      $IMG /scripts/lib/replica_entry.sh)
    echo \"[node] container \$cid\"
    up=0; for i in \$(seq 1 180); do
      docker logs $cname > $rdir/server.log 2>&1 || true
      curl -sf http://localhost:$port/health >/dev/null 2>&1 && { up=1; break; }
      st=\$(docker inspect -f '{{.State.Status}}' $cname 2>/dev/null || echo missing)
      { [ \"\$st\" = exited ] || [ \"\$st\" = missing ]; } && { echo \"[node] server \$st\"; break; }
      sleep 5
    done
    docker logs $cname > $rdir/server.log 2>&1 || true
    echo \"[node] SERVER_UP=\$up\"
    [ \$up -ne 1 ] && { echo '[node] FAILED tail:'; tail -25 $rdir/server.log; docker rm -f $cname >/dev/null 2>&1; exit 1; }
    for shape in $shapes; do
      ISL=\${shape%,*}; OSL=\${shape#*,}
      for c in $concs; do
        echo \"[bench] isl=\$ISL osl=\$OSL conc=\$c\"
        docker run --rm --entrypoint bash --network host -v $HERE:/scripts -v $MROOT:$MROOT -v $rdir:/out \
          $BENCH_IMAGE -c \"python3 /scripts/utils/bench_serving/benchmark_serving.py \
            --model $MROOT/$MDIR --backend vllm --base-url http://0.0.0.0:$port --dataset-name random \
            --random-input-len \$ISL --random-output-len \$OSL --random-range-ratio 1.0 \
            --num-prompts \$(( c * 10 )) --num-warmups \$(( c * 2 )) --max-concurrency \$c \
            --request-rate inf --ignore-eos --save-result --percentile-metrics ttft,tpot,itl,e2el \
            --result-dir /out --result-filename sweep_isl\${ISL}_osl\${OSL}_c\${c}.json\" \
          > $rdir/bench_isl\${ISL}_osl\${OSL}_c\${c}.log 2>&1 || echo \"  [bench] FAILED isl=\$ISL conc=\$c\"
      done
    done
    echo '[node] standdown'; docker rm -f $cname >/dev/null 2>&1; sleep 3
  " > "$rdir/replica.log" 2>&1 &
}

n=0
while IFS=$'\t' read -r node gpus port idx; do
  if [[ "$BX" == "1" && ( "$ACTION" == "sanity" || "$ACTION" == "sweep" ) ]]; then
    launch_external_bench_node "$node" "$gpus" "$port" "$idx"   # ATOM: server+bench in one srun
  else
    launch_replica "$node" "$gpus" "$port" "$idx"                # vLLM/SGLang: in-container bench
  fi
  echo "[$ENGINE] launched replica $idx on $node gpus=$gpus port=$port"; n=$((n+1))
done < "$RUNROOT/placement.tsv"
echo "[$ENGINE] $n replicas launching -> $RUNROOT"

wait
echo "[$ENGINE] done -> $RUNROOT (check replica_*/replica.log, server.log, sweep_*.json)"
