#!/bin/bash
# run_disagg.sh — MiniMax-M3 MXFP4 PD-disaggregated serving + benchmark across multiple nodes.
# Faithful port of InferenceX server_atom.sh (atomesh + mooncake) for minimaxm3-fp4-mi355x-atom-disagg.
#
# Env in: ENGINE=atom-disagg  MODEL=minimaxm3-fp4  TOPO=1p1d|2p1d_dpa  [ACTION=bench|serve]
#
# Layout per topology (topo.py resolves nodes):
#   prefill worker(s) = kv_producer servers (port 8010)   [worker 0's node also hosts the router]
#   decode  worker(s) = kv_consumer servers (port 8020)
#   router (atomesh) on prefill-0, port 8000  <- bench drives THIS endpoint
#
# Each server runs in its own docker container via a per-node `srun --exclusive`. KV flows
# node->node over RDMA (mooncake). Bench runs from a clean container against the router
# (aggregate throughput; no x N extrapolation — this is a real disagg instance).
# All workers TP4, STP only (DECODE_MTP_SIZE=0); DPA topology adds --enable-dp-attention.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # scripts/MiniMax-M3
CLUSTER="$HERE/cluster.yaml"; MODELS="$HERE/model.yaml"
ENGINE="${ENGINE:-atom-disagg}"
MODEL="${MODEL:-minimaxm3-fp4}"; TOPO="${TOPO:?set TOPO=1p1d|2p1d_dpa}"; ACTION="${ACTION:-bench}"
cfg(){ python3 "$HERE/lib/cfg.py" "$@"; }
topo(){ python3 "$HERE/lib/topo.py" "$@"; }

RESV=$(cfg "$CLUSTER" cluster reservation)
MROOT=$(cfg "$CLUSTER" cluster models_root)
RROOT=$(cfg "$CLUSTER" cluster results_root)
RPORT=$(cfg "$CLUSTER" cluster router_port)
PPORT=$(cfg "$CLUSTER" cluster prefill_port)
DPORT=$(cfg "$CLUSTER" cluster decode_port)
HPORT=$(cfg "$CLUSTER" cluster handshake_port)
MDIR=$(cfg "$MODELS" model "$MODEL" dir)
IMG=$(cfg "$MODELS" engine "$ENGINE" image)
ROUTER=$(cfg "$MODELS" engine "$ENGINE" router)
KVCONN=$(cfg "$MODELS" engine "$ENGINE" kv_connector)
GMU=$(cfg "$MODELS" default "$MODEL" gpu_memory_util)
MNS=$(cfg "$MODELS" default "$MODEL" max_num_seqs)
KVDT=$(cfg "$MODELS" default "$MODEL" kv_cache_dtype)
BS=$(cfg "$MODELS" default "$MODEL" block_size)
MML=$(cfg "$MODELS" default "$MODEL" max_model_len)
MBT=$(cfg "$MODELS" default "$MODEL" max_num_batched_tokens)
TP=$(cfg "$MODELS" default "$MODEL" tp)
SHAPES=$(topo "$MODELS" "$TOPO" field shapes)
CONC=$(topo "$MODELS" "$TOPO" field conc)
DPATTN=$(python3 -c "import yaml;t=yaml.safe_load(open('$MODELS'))['topologies']['$TOPO'];print('1' if t['decode'].get('dp_attn') else '0')")

ENV_ARGS=""; for kv in $(cfg "$MODELS" engine "$ENGINE" env); do ENV_ARGS+=" -e $kv"; done

STAMP=$(python3 -c "import time;print(time.strftime('%Y%m%d_%H%M%S'))")
RUN="$RROOT/${MODEL}_${ENGINE}_${TOPO}_${STAMP}"; mkdir -p "$RUN"; chmod 777 "$RUN"
topo "$CLUSTER" "$MODELS" "$TOPO" roles > "$RUN/roles.tsv"
echo "[disagg] ENGINE=$ENGINE MODEL=$MODEL TOPO=$TOPO router=$ROUTER kv=$KVCONN image=$IMG"
echo "[disagg] roles:"; sed 's/^/    /' "$RUN/roles.tsv"

PREFILL_NODE0=$(awk -F'\t' '$1=="prefill" && $2=="0"{print $3}' "$RUN/roles.tsv")
PREFILL_IPS=""; DECODE_IPS=""

ip_of(){ srun --reservation="$RESV" --nodelist="$1" --nodes=1 --ntasks=1 --overlap --mem=1G \
  bash -c "ip route get 1.1.1.1 2>/dev/null | awk '/src/{print \$7}'" 2>/dev/null | grep -oE '([0-9]+\.){3}[0-9]+' | head -1; }

# --- PRE-FLIGHT: clean every node in this topology ------------------------------------------
NODES=$(cut -f3 "$RUN/roles.tsv" | sort -u)
echo "[disagg] cleaning nodes: $(echo $NODES | tr '\n' ' ')"
for n in $NODES; do
  srun --reservation="$RESV" --nodelist="$n" --nodes=1 --ntasks=1 --overlap --mem=0 \
    bash "$HERE/lib/clean_node.sh" > "$RUN/clean_${n}.log" 2>&1 && echo "  [clean] $n OK" || echo "  [clean] $n WARN"
done

# --- resolve router IP (prefill node 0) -----------------------------------------------------
ROUTER_IP=$(ip_of "$PREFILL_NODE0")
echo "[disagg] router will be on $PREFILL_NODE0 ($ROUTER_IP):$RPORT"

# DP-attention arg for M3 (non-DSv4: DPA only, no --enable-tbo which is DSv4-specific).
dpa_arg(){ [[ "$1" == "1" ]] && echo "--enable-dp-attention" || true; }
# Optional model-length caps (M3: max_model_len 32768, max_num_batched_tokens 32768).
LEN_ARGS=""; [[ -n "$MML" ]] && LEN_ARGS="$LEN_ARGS --max-model-len $MML"
[[ -n "$MBT" ]] && LEN_ARGS="$LEN_ARGS --max-num-batched-tokens $MBT"
# KV cache dtype: skip the flag when 'auto' (M3 uses the engine default).
KV_ARG=""; [[ -n "$KVDT" && "$KVDT" != "auto" ]] && KV_ARG="--kv-cache-dtype $KVDT"

# --- launch one server (prefill|decode) on a node in its own srun+docker --------------------
launch_server() {
  local role="$1" node="$2" port="$3" kvrole="$4" rdir="$RUN/${role}_${node}"
  mkdir -p "$rdir"; chmod 777 "$rdir"
  local extra=""
  [[ "$role" == "prefill" ]] && extra="$(cfg "$MODELS" engine "$ENGINE" prefill_flags)" || extra="$(cfg "$MODELS" engine "$ENGINE" decode_flags)"
  [[ "$DPATTN" == "1" ]] && extra="$extra $(dpa_arg 1)"
  local SRV="python3 -m atom.entrypoints.openai_server --model $MROOT/$MDIR \
      --host 0.0.0.0 --server-port $port -tp $TP \
      $KV_ARG --block-size $BS --gpu-memory-utilization $GMU --max-num-seqs $MNS \
      $LEN_ARGS $extra \
      --kv-transfer-config '{\"kv_role\":\"$kvrole\",\"kv_connector\":\"$KVCONN\",\"proxy_ip\":\"$ROUTER_IP\",\"handshake_port\":$HPORT}'"
  srun --reservation="$RESV" --nodelist="$node" --nodes=1 --ntasks=1 --exclusive bash -c "
    docker rm -f mm3_${role} >/dev/null 2>&1 || true; sleep 3
    cid=\$(docker run -d --name mm3_${role} --entrypoint bash \
      --device=/dev/kfd --device=/dev/dri --group-add video --ipc host --shm-size 128G --network host \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged \
      -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $ENV_ARGS \
      -v $MROOT:$MROOT -v $HERE:/scripts -v $rdir:/out \
      $IMG -c '$SRV')
    echo \"[$role] container \$cid on $node:$port\"
    for i in \$(seq 1 300); do
      docker logs mm3_${role} > $rdir/server.log 2>&1 || true
      curl -sf http://localhost:$port/health >/dev/null 2>&1 && { echo READY > $rdir/READY; break; }
      st=\$(docker inspect -f '{{.State.Status}}' mm3_${role} 2>/dev/null || echo missing)
      { [ \"\$st\" = exited ] || [ \"\$st\" = missing ]; } && { echo \"[$role] container \$st\"; break; }
      sleep 5
    done
    docker logs mm3_${role} > $rdir/server.log 2>&1 || true
    # hold the container alive for the duration (router + bench run elsewhere); wait on it
    [ -f $rdir/READY ] && docker wait mm3_${role} || { echo '[FAIL] tail:'; tail -25 $rdir/server.log; exit 1; }
  " > "$rdir/srun.log" 2>&1 &
  echo "$!"
}

# Build router prefill/decode arg lists (need IPs)
declare -a SRV_PIDS=()
while IFS=$'\t' read -r role widx node port; do
  ipn=$(ip_of "$node")
  if [[ "$role" == "prefill" ]]; then PREFILL_IPS+="$ipn "; else DECODE_IPS+="$ipn "; fi
done < "$RUN/roles.tsv"

echo "[disagg] launching servers..."
while IFS=$'\t' read -r role widx node port; do
  kvrole="kv_producer"; [[ "$role" == "decode" ]] && kvrole="kv_consumer"
  pid=$(launch_server "$role" "$node" "$port" "$kvrole"); SRV_PIDS+=("$pid")
  echo "  [$role $widx] $node:$port ($kvrole) srun-pid $pid"
done < "$RUN/roles.tsv"

# --- wait for all servers READY -------------------------------------------------------------
echo "[disagg] waiting for all servers healthy (timeout ~25min)..."
ALLREADY=0
for i in $(seq 1 300); do
  r=$(find "$RUN" -name READY 2>/dev/null | wc -l); n=$(wc -l < "$RUN/roles.tsv")
  [[ "$r" -ge "$n" ]] && { ALLREADY=1; echo "  all $n servers READY"; break; }
  sleep 5
done
if [[ "$ALLREADY" != "1" ]]; then echo "[disagg] NOT all servers came up — see $RUN/*/server.log"; fi

# --- start the atomesh router on prefill node 0 ---------------------------------------------
RDIR="$RUN/router"; mkdir -p "$RDIR"; chmod 777 "$RDIR"
PF_ARGS=""; for ip in $PREFILL_IPS; do PF_ARGS+=" --prefill http://$ip:$PPORT"; done
DC_ARGS=""; for ip in $DECODE_IPS; do DC_ARGS+=" --decode http://$ip:$DPORT"; done
RFLAGS=$(cfg "$MODELS" engine "$ENGINE" router_flags)
ROUTER_CMD="/usr/local/bin/atomesh launch --host 0.0.0.0 --port $RPORT --pd-disaggregation $PF_ARGS $DC_ARGS $RFLAGS"
echo "[disagg] router cmd: $ROUTER_CMD"
srun --reservation="$RESV" --nodelist="$PREFILL_NODE0" --nodes=1 --ntasks=1 --overlap --mem=8G bash -c "
  docker rm -f mm3_router >/dev/null 2>&1 || true
  cid=\$(docker run -d --name mm3_router --entrypoint bash --network host \
    -v $HERE:/scripts $IMG -c '$ROUTER_CMD')
  for i in \$(seq 1 60); do
    docker logs mm3_router > $RDIR/router.log 2>&1 || true
    curl -sf http://localhost:$RPORT/v1/models >/dev/null 2>&1 && { echo READY > $RDIR/READY; break; }
    sleep 5
  done
  docker logs mm3_router > $RDIR/router.log 2>&1 || true
" > "$RDIR/srun.log" 2>&1 &

for i in $(seq 1 60); do [[ -f "$RDIR/READY" ]] && { echo "[disagg] router READY on $ROUTER_IP:$RPORT"; break; }; sleep 5; done
[[ -f "$RDIR/READY" ]] || echo "[disagg] router did NOT become ready — see $RDIR/router.log"

if [[ "$ACTION" == "serve" ]]; then
  echo "[disagg] ACTION=serve — endpoint http://$ROUTER_IP:$RPORT (router). Servers held; Ctrl-C / scancel to stop."
  wait; exit 0
fi

# --- ACTION=bench: drive the router at each (shape,conc) -------------------------------------
BENCH_IMG=$(cfg "$MODELS" engine "$ENGINE" bench_image)
BENCH_IMG="${BENCH_IMG:-vllm/vllm-openai-rocm:nightly-9037498c22891e55b594f567fb91d9b4efbf3e99}"
echo "[disagg] benchmarking router http://$ROUTER_IP:$RPORT  shapes='$SHAPES' conc='$CONC'"
for shape in $SHAPES; do
  ISL=${shape%,*}; OSL=${shape#*,}
  for c in $CONC; do
    echo "  [bench] isl=$ISL osl=$OSL conc=$c"
    srun --reservation="$RESV" --nodelist="$PREFILL_NODE0" --nodes=1 --ntasks=1 --overlap --mem=8G bash -c "
      docker run --rm --entrypoint bash --network host -v $HERE:/scripts -v $RUN:/out \
        $BENCH_IMG -c 'python3 /scripts/utils/bench_serving/benchmark_serving.py \
          --model $MROOT/$MDIR --backend vllm --base-url http://$ROUTER_IP:$RPORT --dataset-name random \
          --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1.0 \
          --num-prompts \$(( $c * 10 )) --num-warmups \$(( $c * 2 )) --max-concurrency $c \
          --request-rate inf --ignore-eos --save-result --percentile-metrics ttft,tpot,itl,e2el \
          --result-dir /out --result-filename disagg_isl${ISL}_osl${OSL}_c${c}.json'
    " > "$RUN/bench_isl${ISL}_osl${OSL}_c${c}.log" 2>&1 || echo "    [bench] FAILED isl=$ISL conc=$c"
  done
done

# --- teardown -------------------------------------------------------------------------------
echo "[disagg] teardown: stopping servers + router"
for n in $NODES; do
  srun --reservation="$RESV" --nodelist="$n" --nodes=1 --ntasks=1 --overlap --mem=1G \
    bash -c "docker rm -f mm3_prefill mm3_decode mm3_router >/dev/null 2>&1" 2>/dev/null || true
done
echo "[disagg] done -> $RUN"
