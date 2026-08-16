#!/bin/bash
# 2-node MoRI EP probe A/B. Usage: run_ep_probe.sh <image> <tag> [node0] [node1]
IMG=${1:-localhost/rocmshared/mori-wideep-glm:v027}
TAG=${2:-a}
N0=${3:?name node0, e.g. run_ep_probe.sh $IMG $TAG n01 n02}; N1=${4:?name node1}
PROBE_DIR=$(cd "$(dirname "$0")" && pwd)
CN=epprobe_$TAG
PORT=29577

railmap(){ ssh -n $1 "for d in /sys/class/infiniband/*; do dv=\$(basename \$d);
  nd=\$(ls \$d/device/net 2>/dev/null|head -1);
  m=\$(basename \$(readlink /sys/class/net/\$nd/master 2>/dev/null) 2>/dev/null);
  ip=\$(ip -br -4 addr show \$m 2>/dev/null|awk '{print \$3}'|cut -d/ -f1);
  echo \"\$(echo \$ip|cut -d. -f3) \$dv\"; done | sort -n"; }

IP0=$(ssh -n $N0 "ip -br -4 addr show | awk '\$3 ~ /^192\.168\.200\./ {print \$3}' | cut -d/ -f1 | head -1")
MGMT0=$(ssh -n $N0 "hostname -I | tr ' ' '\n' | grep -E '^10\.67\.' | head -1")
echo "== probe [$TAG] image=$IMG  master=$MGMT0  QP=${QP:-4} TC=${TC:-0} NDEV=${NDEV:-8} =="

for i in 0 1; do
  eval N=\$N$i
  R200=$(railmap $N | awk '$1==200{print $2;exit}')
  # NDEV: how many rail-ordered devices to expose to MoRI (default all 8).
  # NDEV=1 reproduces the historically-validated single-pinned-device config.
  DEVS=$(railmap $N | awk '{print $2}' | head -n "${NDEV:-8}" | paste -sd,)
  scp -q $PROBE_DIR/mori_ep_probe.py $N:/tmp/mori_ep_probe.py
  ssh -n $N "docker rm -f $CN >/dev/null 2>&1
  docker run -d --name $CN --network host --ipc host \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --group-add video --group-add render \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit nofile=524288 --ulimit nproc=100000 \
    -v /tmp/mori_ep_probe.py:/probe.py -v /sys:/sys \
    -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
    -v /sys/kernel/config:/sys/kernel/config -v /sys/kernel/debug:/sys/kernel/debug \
    -e MORI_RDMA_DEVICES=$DEVS -e MORI_IB_GID_INDEX=3 \
    -e MORI_NUM_QP_PER_PE=${QP:-4} -e MORI_RDMA_TC=${TC:-0} -e MORI_RDMA_SL=${SL:-0} \
    -e NCCL_IB_HCA=$R200 -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=bond0 \
    -e GLOO_SOCKET_IFNAME=bond0 -e MORI_SHMEM_HEAP_SIZE=8G \
    --entrypoint sleep $IMG infinity >/dev/null && echo \"  $N up (devs=$DEVS rail200=$R200)\""
done
sleep 3
for i in 0 1; do
  eval N=\$N$i
  ssh -n -f $N "docker exec $CN bash -lc 'cd / && torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$i --master_addr=$MGMT0 --master_port=$PORT /probe.py' > /tmp/epprobe_$TAG.log 2>&1"
done
echo "  launched; polling..."
for t in $(seq 1 30); do
  sleep 10
  D=$(ssh -n $N0 "grep -cE 'ALL OK|Traceback|error 110|Aborted' /tmp/epprobe_$TAG.log 2>/dev/null" 2>/dev/null)
  [ "${D:-0}" -gt 0 ] && break
done
for i in 0 1; do
  eval N=\$N$i
  echo "----- $N -----"
  ssh -n $N "grep -E 'probe\]|error 110|Traceback|Assertion|RuntimeError|Aborted' /tmp/epprobe_$TAG.log 2>/dev/null | tail -12; echo \"QP110_count=\$(grep -c 'error 110' /tmp/epprobe_$TAG.log 2>/dev/null)\""
done
for i in 0 1; do eval N=\$N$i; ssh -n $N "docker rm -f $CN >/dev/null 2>&1"; done
