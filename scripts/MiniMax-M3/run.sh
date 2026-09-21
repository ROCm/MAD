#!/bin/bash
# run.sh — MAD launcher for the VERBATIM InferenceX MiniMax-M3 MXFP4 ATOM disagg benchmark.
#
# The engine logic under benchmarks/multi_node/amd_utils/ is copied UNCHANGED from
# SemiAnalysisAI/InferenceX. InferenceX normally drives it via job.slurm/submit.sh; this wrapper
# reproduces the same per-node env contract + container mounts so it runs standalone from MAD.
#
# All model flags (TP/DPA, kv fp8, online_quant, block_size, env, MTP) are read by server_atom.sh
# from amd_utils/models_atom.yaml — this wrapper sets NONE of them, so behavior matches InferenceX.
#
# Usage:
#   TOPO=1p1d      ISL=8192 OSL=1024 ./run.sh          # 2 nodes, conc 1-256
#   TOPO=1p1d      ISL=1024 OSL=1024 ./run.sh          # 2 nodes, conc 1-256
#   TOPO=2p1d_dpa  ISL=8192 OSL=1024 ./run.sh          # 3 nodes, conc 256-1024
#   NODES=n1,n2,n3 TOPO=2p1d_dpa ./run.sh              # pin nodes
#   ACTION=dry ./run.sh                                # print server/bench cmds, don't launch
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # scripts/MiniMax-M3
AMD_UTILS="$HERE/benchmarks/multi_node/amd_utils"
CLUSTER="$HERE/cluster.yaml"
IMG="rocm/atom-dev:nightly_202607011530"                       # from models_atom.yaml / config
MODEL_NAME="MiniMax-M3-MXFP4"                                  # must match models_atom.yaml key + dir

y(){ python3 -c "import yaml,sys;print(yaml.safe_load(open('$CLUSTER')).get('$1',''))"; }
RESV=$(y reservation); GPN=$(y gpus_per_node); MROOT=$(y models_root); RROOT=$(y results_root)
RPORT=$(y router_port); PPORT=$(y prefill_port); DPORT=$(y decode_port); HPORT=$(y handshake_port)
mapfile -t CFG_NODES < <(python3 -c "import yaml;[print(n) for n in yaml.safe_load(open('$CLUSTER')).get('nodes',[])]")

TOPO="${TOPO:-1p1d}"; ACTION="${ACTION:-bench}"; ISL="${ISL:-8192}"; OSL="${OSL:-1024}"
# Topologies mirror configs/minimaxm3-fp4-mi355x-atom-disagg.yaml (all TP4, STP: DECODE_MTP_SIZE=0).
case "$TOPO" in
  1p1d)     xP=1; yD=1; DPA=false; DEF_CONC="1 2 4 8 16 32 64 128 256" ;;
  2p1d_dpa) xP=2; yD=1; DPA=true;  DEF_CONC="256 512 768 1024" ;;
  *) echo "ERROR: TOPO must be 1p1d|2p1d_dpa (got $TOPO)"; exit 1 ;;
esac
TP=4
CONC_LIST="${CONC_LIST:-$DEF_CONC}"
CONC_X=$(echo "$CONC_LIST" | tr ' ' 'x')      # server_atom.sh derives decode --max-num-seqs from this
NNODES=$(( xP + yD ))                          # 1 node/worker at TP4 on 8-GPU nodes

# --- node selection --------------------------------------------------------------------------
if [[ -n "${NODES:-}" ]]; then IFS=',' read -ra NODES <<< "$NODES"; else NODES=("${CFG_NODES[@]}"); fi
[[ ${#NODES[@]} -lt $NNODES ]] && { echo "ERROR: need $NNODES nodes, have ${#NODES[@]} (fill cluster.yaml nodes: or pass NODES=)"; exit 1; }
NODES=("${NODES[@]:0:$NNODES}")

STAMP=$(date +%Y%m%d_%H%M%S)
RUN="$RROOT/minimaxm3-fp4_atom_${TOPO}_isl${ISL}_${STAMP}"; mkdir -p "$RUN"; chmod 777 "$RUN"
echo "[run] TOPO=$TOPO xP=$xP yD=$yD DPA=$DPA nodes=${NODES[*]} shape=${ISL}/${OSL} conc='$CONC_LIST'"
echo "[run] image=$IMG model=$MODEL_NAME run=$RUN"
RESV_ARG=""; [[ -n "$RESV" && "$RESV" != "<your_slurm_reservation>" ]] && RESV_ARG="--reservation=$RESV"

# --- resolve node IPs (IPADDRS = prefill nodes first, then decode; server_atom.sh slices it) --
declare -a IPS=()
for n in "${NODES[@]}"; do
  ip=$(srun $RESV_ARG --nodelist="$n" --nodes=1 --ntasks=1 --overlap --mem=1G \
        bash -c "ip route get 1.1.1.1 2>/dev/null | awk '/src/{print \$7}'" 2>/dev/null \
        | grep -oE '([0-9]+\.){3}[0-9]+' | head -1)
  IPS+=("$ip"); echo "  $n -> $ip"
done
IPADDRS=$(IFS=,; echo "${IPS[*]}"); NODE0_IP="${IPS[0]}"

DRY=0; [[ "$ACTION" == "dry" ]] && DRY=1
RDMA="--device=/dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 --init"

# --- launch verbatim server_atom.sh on each node (NODE_RANK self-selects role) ----------------
# Mounts: recipe root -> /workspace (so bench.sh ../../benchmark_lib.sh + REPO_ROOT/utils resolve);
#         ATOM_WS_PATH -> /workspace/benchmarks/multi_node/amd_utils (server_atom.sh's own dir).
rank=0
for n in "${NODES[@]}"; do
  rdir="$RUN/node${rank}_${n}"; mkdir -p "$rdir"; chmod 777 "$rdir"
  srun $RESV_ARG --nodelist="$n" --nodes=1 --ntasks=1 --exclusive bash -c "
    docker rm -f mm3_atom_n${rank} >/dev/null 2>&1 || true
    docker run --rm --name mm3_atom_n${rank} --entrypoint bash \
      --device=/dev/kfd --device=/dev/dri $RDMA --group-add video --ipc host --shm-size 128G \
      --network host --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      -e ATOM_WS_PATH=/workspace/benchmarks/multi_node/amd_utils \
      -e ENGINE=atom-disagg \
      -e NODE_RANK=${rank} -e NODE0_ADDR=${NODE0_IP} -e IPADDRS='${IPADDRS}' \
      -e MODEL_DIR=${MROOT} -e MODEL_NAME=${MODEL_NAME} \
      -e xP=${xP} -e yD=${yD} -e GPUS_PER_NODE=${GPN} \
      -e PREFILL_TP_SIZE=${TP} -e DECODE_TP_SIZE=${TP} \
      -e PREFILL_ENABLE_EP=false -e DECODE_ENABLE_EP=false \
      -e PREFILL_ENABLE_DP=${DPA} -e DECODE_ENABLE_DP=${DPA} \
      -e DECODE_MTP_SIZE=0 \
      -e PREFILL_PORT=${PPORT} -e DECODE_PORT=${DPORT} -e ROUTER_PORT=${RPORT} -e HANDSHAKE_PORT=${HPORT} \
      -e BENCH_INPUT_LEN=${ISL} -e BENCH_OUTPUT_LEN=${OSL} -e BENCH_MAX_CONCURRENCY='${CONC_X}' \
      -e BENCH_REQUEST_RATE=inf -e BENCH_RANDOM_RANGE_RATIO=1 -e BENCH_NUM_PROMPTS_MULTIPLIER=10 \
      -e DRY_RUN=${DRY} -e SLURM_JOB_ID=${STAMP} \
      -v ${MROOT}:${MROOT} -v ${HERE}:/workspace -v ${rdir}:/run_logs/slurm_job-${STAMP} \
      ${IMG} /workspace/benchmarks/multi_node/amd_utils/server.sh
  " > "$rdir/srun.log" 2>&1 &
  echo "  [launch] node${rank} ($n) dispatched (role self-selected by NODE_RANK)"
  rank=$(( rank + 1 ))
done

echo "[run] all $NNODES nodes dispatched. node0=${NODES[0]} hosts atomesh router :$RPORT + drives bench."
echo "[run] tail node0: $RUN/node0_${NODES[0]}/srun.log"
wait
echo "[run] done -> $RUN"
