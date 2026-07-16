#!/bin/bash
# launch.sh — multi-node ATOM PD-disagg launcher for MiniMax-M3 MXFP4 (the `job.slurm`-equiv).
# Drives the VENDORED, verbatim InferenceX server_atom.sh across N nodes via docker+srun.
#
# Faithful to the InferenceX chain:
#   minimaxm3_..._atom-disagg.sh -> submit.sh -> job.slurm -> server_atom.sh
# We replace submit.sh+job.slurm (the slurm plumbing) with this; server_atom.sh runs unmodified.
#
# Env (with defaults):
#   TOPO=1p1d|2p1d_dpa    topology (sets xP/yD/dp-attn)          [default 1p1d]
#   ISL, OSL              bench shape                            [default 8192 1024]
#   CONC_LIST            space-separated concurrencies          [topology default]
#   ACTION=bench|serve|dry   bench (default) | hold endpoint | print cmds only
#
# Topologies (from minimaxm3-fp4-mi355x-atom-disagg). All workers TP4 (no EP); STP only.
#   1p1d     : xP=1 yD=1, 2 nodes, TP4, dp-attn OFF (conc 1-256;   shapes 8192/1024 & 1024/1024)
#   2p1d_dpa : xP=2 yD=1, 3 nodes, TP4, dp-attn ON  (conc 256-1024; shape 8192/1024)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # scripts/MiniMax-M3/atom_disagg
ROOT="$(cd "$HERE/.." && pwd)"                                # scripts/MiniMax-M3
cfg(){ python3 "$ROOT/lib/cfg.py" "$@"; }
CLUSTER="$ROOT/cluster.yaml"

RESV=$(cfg "$CLUSTER" cluster reservation)
MROOT=$(cfg "$CLUSTER" cluster models_root)
RROOT=$(cfg "$CLUSTER" cluster results_root)
RPORT=$(cfg "$CLUSTER" cluster router_port)
PPORT=$(cfg "$CLUSTER" cluster prefill_port)
DPORT=$(cfg "$CLUSTER" cluster decode_port)
HPORT=$(cfg "$CLUSTER" cluster handshake_port)
IMG=$(cfg "$ROOT/model.yaml" engine atom-disagg image)
MODEL_NAME="MiniMax-M3-MXFP4"

TOPO="${TOPO:-1p1d}"; ACTION="${ACTION:-bench}"
ISL="${ISL:-8192}"; OSL="${OSL:-1024}"
case "$TOPO" in
  1p1d)     xP=1; yD=1; DPATTN=0; DEF_CONC="1 2 4 8 16 32 64 128 256" ;;
  2p1d_dpa) xP=2; yD=1; DPATTN=1; DEF_CONC="256 512 768 1024" ;;
  *) echo "ERROR: TOPO must be 1p1d|2p1d_dpa (got $TOPO)"; exit 1 ;;
esac
CONC_LIST="${CONC_LIST:-$DEF_CONC}"
# server_atom.sh expects BENCH_MAX_CONCURRENCY 'x'-separated (it does `tr 'x' '\n'|sort -n|tail -1`
# to derive the decode server's --max-num-seqs). Our bench.sh wants space-separated. Keep both.
CONC_X=$(echo "$CONC_LIST" | tr ' ' 'x')
TP=4                                  # TP4 throughout (M3 on gfx950: plain TP4 beats TP8 / TP4-EP)
NNODES=$(( xP + yD ))                  # 1 node/worker at TP4 on 8-GPU nodes

# --- node selection: explicit NODES= override (REQUIRED for parallel launches to avoid
#     two launches racing to grab the same 'first free' nodes), else auto-pick free ones. ----
if [[ -n "${NODES:-}" ]]; then
  IFS=',' read -ra NODES <<< "$NODES"
  [[ ${#NODES[@]} -lt $NNODES ]] && { echo "ERROR: NODES= has ${#NODES[@]}, need $NNODES"; exit 1; }
  NODES=("${NODES[@]:0:$NNODES}")
else
  mapfile -t ALL < <(scontrol show hostnames "$(scontrol show res "$RESV" 2>/dev/null | grep -oE 'Nodes=[^ ]+' | head -1 | cut -d= -f2)")
  BUSY=$(squeue -u "$USER" -h -o "%N" 2>/dev/null | scontrol show hostnames 2>/dev/null | sort -u)
  NODES=(); for n in "${ALL[@]}"; do grep -qx "$n" <<<"$BUSY" || NODES+=("$n"); [[ ${#NODES[@]} -ge $NNODES ]] && break; done
  [[ ${#NODES[@]} -lt $NNODES ]] && { echo "ERROR: need $NNODES free nodes, have ${#NODES[@]}"; exit 1; }
fi

STAMP=$(python3 -c "import time;print(time.strftime('%Y%m%d_%H%M%S'))")
RUN="$RROOT/minimaxm3_atom_${TOPO}_isl${ISL}_${STAMP}"; mkdir -p "$RUN"; chmod 777 "$RUN"
echo "[launch] TOPO=$TOPO xP=$xP yD=$yD nodes=${NODES[*]}  shape=${ISL}/${OSL} conc='$CONC_LIST'"
echo "[launch] image=$IMG  run=$RUN"

# --- resolve node IPs (IPADDRS = prefill nodes first, then decode; server_atom.sh slices it) -
declare -a IPS=()
for n in "${NODES[@]}"; do
  ip=$(srun --reservation="$RESV" --nodelist="$n" --nodes=1 --ntasks=1 --overlap --mem=1G \
        bash -c "ip route get 1.1.1.1 2>/dev/null | awk '/src/{print \$7}'" 2>/dev/null \
        | grep -oE '([0-9]+\.){3}[0-9]+' | head -1)
  IPS+=("$ip"); echo "  $n -> $ip"
done
IPADDRS=$(IFS=,; echo "${IPS[*]}")
NODE0="${NODES[0]}"; NODE0_IP="${IPS[0]}"

# --- pre-flight clean every node ------------------------------------------------------------
for n in "${NODES[@]}"; do
  srun --reservation="$RESV" --nodelist="$n" --nodes=1 --ntasks=1 --overlap --mem=0 \
    bash "$ROOT/lib/clean_node.sh" > "$RUN/clean_${n}.log" 2>&1 && echo "  [clean] $n OK" || echo "  [clean] $n WARN"
done

DRY=0; [[ "$ACTION" == "dry" ]] && DRY=1

# --- RDMA device flags (THE fix for mooncake 30s timeouts) ----------------------------------
# The ATOM disagg image (rocm/atom-dev:nightly_202607011530) carries its own bnxt_re RDMA stack.
# The fix for mooncake handshake timeouts is exposing --device=/dev/infiniband + --ulimit
# memlock=-1, NOT bind-mounting host libs. DO NOT bind-mount host libibverbs over the
# container's: host and container can have incompatible IBVERBS_PRIVATE ABIs, so mounting host
# libs BREAKS the container's RDMA tools. Expose devices + memlock only.
RDMA_DEVICES="--device=/dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 --init"
RDMA_MOUNT_SNIPPET='RM=""'   # no host-lib mounts (container stack is self-sufficient)

# --- launch server_atom.sh on each node, NODE_RANK = its index ------------------------------
# server_atom.sh logic: rank 0 = prefill0 + atomesh router (+ bench); rank<NODE_OFFSET = prefill;
# rank>=NODE_OFFSET = decode. We pass the SAME env to every node; the script self-selects by rank.
rank=0
for n in "${NODES[@]}"; do
  rdir="$RUN/node${rank}_${n}"; mkdir -p "$rdir"; chmod 777 "$rdir"
  srun --reservation="$RESV" --nodelist="$n" --nodes=1 --ntasks=1 --exclusive bash -c "
    docker rm -f mm3_atom_n${rank} >/dev/null 2>&1 || true
    $RDMA_MOUNT_SNIPPET   # builds \$RM (host RDMA lib bind-mounts) on THIS node
    echo \"[rdma] mounts: \$RM\"
    docker run --rm --name mm3_atom_n${rank} --entrypoint bash \
      --device=/dev/kfd --device=/dev/dri $RDMA_DEVICES \$RM -v /sys:/sys \
      --group-add video --ipc host --shm-size 128G \
      --network host --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -e ATOM_WS_PATH=/ws -e NODE_RANK=${rank} -e NODE0_ADDR=${NODE0_IP} -e IPADDRS='${IPADDRS}' \
      -e MODEL_DIR=${MROOT} -e MODEL_NAME=${MODEL_NAME} \
      -e xP=${xP} -e yD=${yD} -e GPUS_PER_NODE=8 \
      -e PREFILL_TP_SIZE=${TP} -e DECODE_TP_SIZE=${TP} \
      -e PREFILL_ENABLE_DP=$([[ $DPATTN == 1 ]] && echo true || echo false) -e PREFILL_ENABLE_EP=1 \
      -e DECODE_ENABLE_DP=$([[ $DPATTN == 1 ]] && echo true || echo false)  -e DECODE_ENABLE_EP=1 \
      -e PREFILL_PORT=${PPORT} -e DECODE_PORT=${DPORT} -e ROUTER_PORT=${RPORT} -e HANDSHAKE_PORT=${HPORT} \
      -e MEM_FRAC_STATIC=0.8 -e KV_CACHE_DTYPE=auto -e BLOCK_SIZE=128 -e MAX_NUM_SEQS=256 \
      -e MAX_MODEL_LEN=32768 -e MAX_NUM_BATCHED_TOKENS=32768 \
      -e SPEC_DECODING= -e DECODE_MTP_SIZE=0 \
      -e AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
      -e BENCH_INPUT_LEN=${ISL} -e BENCH_OUTPUT_LEN=${OSL} -e BENCH_MAX_CONCURRENCY='${CONC_X}' \
      -e BENCH_REQUEST_RATE=inf -e BENCH_RANDOM_RANGE_RATIO=1 -e BENCH_NUM_PROMPTS_MULTIPLIER=10 \
      -e DRY_RUN=${DRY} -e SLURM_JOB_ID=${STAMP} \
      -v ${MROOT}:${MROOT} -v ${HERE}:/ws -v ${ROOT}:/scripts -v ${rdir}:/run_logs/slurm_job-${STAMP} \
      ${IMG} /ws/server_atom.sh
  " > "$rdir/srun.log" 2>&1 &
  echo "  [launch] node${rank} ($n) server_atom.sh dispatched"
  rank=$(( rank + 1 ))
done

echo "[launch] all $NNODES nodes dispatched. node0=$NODE0 hosts atomesh router :$RPORT + drives bench."
echo "[launch] tail node0: $RUN/node0_${NODE0}/srun.log"
wait
echo "[launch] done -> $RUN  (results: disagg_*.json under node0_*/)"
