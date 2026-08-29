#!/bin/bash
# Non-SLURM driver for sglang_disagg DSV4-Flash EP8/EP16 on skyRiver (bnxt/Thor2).
# Mirrors the docker-run + env block of run_xPyD_models.slurm, but launches per-node
# over SSH (no SLURM). Runs the framework's own sglang_disagg_mori_io_ep.sh inside.
#
# Usage:
#   EP8  1P1D:  xP=1 yD=1  PREFILL_NODES=nodeA  DECODE_NODES=nodeB  bash run_dsv4_nonslurm.sh
#   EP16 2P2D:  xP=2 yD=2  PREFILL_NODES="nodeA nodeB"  DECODE_NODES="nodeC nodeD"  bash run_dsv4_nonslurm.sh
set -u

IMG="${IMG:-localhost/mad_dsv4_disagg:pr}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash-FP8-E4M3}"
COOKBOOK_HOST="${COOKBOOK_HOST:-/root/sglang_disagg_cookbook}"
COOKBOOK_IN="/sgl-cookbook"
xP="${xP:?set xP}"; yD="${yD:?set yD}"
DP_MODE="${DP_MODE:-1}"
PREFILL_NODES="${PREFILL_NODES:?space-separated prefill mgmt hostnames}"
DECODE_NODES="${DECODE_NODES:?space-separated decode mgmt hostnames}"
JIT_CACHE="${JIT_CACHE:-/root/.mad_jit_cache/$(echo "$IMG"|tr '/:' '__')}"

ALL_NODES=($PREFILL_NODES $DECODE_NODES)
# Fabric .200-subnet IP per node (MoRI/disagg bootstrap rides the RDMA fabric).
fabip(){ ssh -n "$1" "ip -br -4 addr show | awk '\$3 ~ /^192\.168\.200\./{print \$3}' | cut -d/ -f1 | head -1"; }
IPADDRS=""
for n in "${ALL_NODES[@]}"; do IPADDRS+="${IPADDRS:+,}$(fabip "$n")"; done
MASTER_ADDR=$(echo "$IPADDRS" | cut -d, -f1)
echo "IPADDRS(fabric)=$IPADDRS  MASTER=$MASTER_ADDR  xP=$xP yD=$yD DP_MODE=$DP_MODE"

launch_node(){
  local node="$1" rank="$2"
  local mgmtif; mgmtif=$(ssh -n "$node" "ip route | awk '/^default/{for(i=1;i<=NF;i++)if(\$i==\"dev\")print \$(i+1)}' | head -1")
  local hostlib; hostlib=$(ssh -n "$node" "ls /usr/local/lib/libbnxt_re-rdmav34.so 2>/dev/null | head -1")
  local bnxtmnt=""; [[ -n "$hostlib" ]] && bnxtmnt="-v ${hostlib}:${hostlib}:ro"
  local DATA_MNTS=""; for d in /mnt/md0 /mnt/nvme1 /mnt/nvme2 /mnt/nvme3; do ssh -n "$node" "[ -d $d ]" 2>/dev/null && DATA_MNTS+=" -v $d:$d"; done
  # bnxt/Thor2 RDMA devices, sorted by fabric subnet octet (rank i -> same rail both ends)
  local RAILDEVS; RAILDEVS=$(ssh -n "$node" 'for dv in /sys/class/infiniband/bnxt_re_bond*; do d=$(basename $dv); nd=$(ls $dv/device/net 2>/dev/null|head -1); m=$(basename $(readlink /sys/class/net/$nd/master 2>/dev/null) 2>/dev/null); ip=$(ip -br -4 addr show $m 2>/dev/null|awk "{print \$3}"|cut -d/ -f1); echo "$(echo $ip|cut -d. -f3) $d"; done | sort -n | awk "{print \$2}" | paste -sd,')
  # persistent JIT cache + orphan-lock sweep (no live compiler)
  ssh -n "$node" "mkdir -p ${JIT_CACHE}/mori ${JIT_CACHE}/aiter; if ! pgrep -x cc1plus>/dev/null && ! pgrep -x hipcc>/dev/null; then find ${JIT_CACHE} \\( -name 'lock_module_*' -o -name '*.lock' -o -name 'lock' \\) -delete 2>/dev/null; fi"
  ssh -n "$node" "docker rm -f dsv4_${MODEL_NAME}_r${rank} >/dev/null 2>&1
    docker run -d --name dsv4_${MODEL_NAME}_r${rank} \
      --network host --ipc host --privileged \
      --device /dev/kfd --device /dev/dri --device /dev/infiniband \
      --group-add video --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
      --ulimit memlock=-1 --ulimit nofile=1048576 --ulimit nproc=-1 \
      -v /models:/models ${DATA_MNTS} \
      -v /sys/kernel/config:/sys/kernel/config -v /sys/kernel/debug:/sys/kernel/debug \
      -v /etc/libibverbs.d:/etc/libibverbs.d:ro ${bnxtmnt} \
      -v ${JIT_CACHE}:/jit_cache \
      -v ${COOKBOOK_HOST}:${COOKBOOK_IN} \
      -e MORI_JIT_CACHE_DIR=/jit_cache/mori -e AITER_JIT_DIR=/jit_cache/aiter \
      -e OPENBLAS_NUM_THREADS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e NUMEXPR_NUM_THREADS=4 -e GOTO_NUM_THREADS=4 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -e IB_DEVICES=${RAILDEVS} -e MORI_RDMA_DEVICES=${RAILDEVS} -e NCCL_IB_HCA=${RAILDEVS} -e MORI_IB_GID_INDEX=3 -e NCCL_IB_GID_INDEX=3 -e MORI_RDMA_TC=41 -e MORI_RDMA_SL=0 \
      -e MODEL_NAME=${MODEL_NAME} -e MODEL_PATH=${MODEL_PATH} \
      -e xP=${xP} -e yD=${yD} -e DP_MODE=${DP_MODE} -e RUN_MORI=1 -e USE_CX7_NICS=0 -e SKIP_BENCHMARK=${SKIP_BENCHMARK:-1} -e SKIP_CURL_TEST=${SKIP_CURL_TEST:-1} -e KEEP_ALIVE=${KEEP_ALIVE:-1} \
      -e MASTER_ADDR=${MASTER_ADDR} -e NODE_RANK=${rank} -e IPADDRS=${IPADDRS} \
      -e NCCL_SOCKET_IFNAME=${mgmtif} -e GLOO_SOCKET_IFNAME=${mgmtif} \
      -e MOONCAKE_COOKBOOK_PATH=${COOKBOOK_IN} \
      --entrypoint /bin/bash ${IMG} -lc '
        if [ -e /usr/local/lib/libbnxt_re-rdmav34.so ]; then echo /usr/local/lib>/etc/ld.so.conf.d/bnxt.conf; ldconfig; fi
        cd ${COOKBOOK_IN}
        bash ${COOKBOOK_IN}/sglang_disagg_mori_io_ep.sh
      ' > /dev/null && echo \"  ${node} rank ${rank} launched (mgmtif=${mgmtif})\""
}

rank=0
for n in $PREFILL_NODES; do launch_node "$n" "$rank"; rank=$((rank+1)); done
for n in $DECODE_NODES;  do launch_node "$n" "$rank"; rank=$((rank+1)); done
echo "All nodes launched. Router/API on first prefill node fabric IP ${MASTER_ADDR} (port per framework, default 3000/8000)."
