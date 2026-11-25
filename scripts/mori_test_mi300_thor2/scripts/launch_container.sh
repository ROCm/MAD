#!/bin/bash
IMG="$1"
LIB=$(ls /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so 2>/dev/null | head -1)
sudo docker rm -f mori_host 2>/dev/null
sudo docker run -d --name mori_host --entrypoint sleep \
  --network host --ipc host --privileged \
  --ulimit memlock=-1:-1 --ulimit nproc=100000:100000 --shm-size 64g --cap-add SYS_PTRACE \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v /home:/home -v /lib/modules:/lib/modules \
  -v /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0:/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro \
  -v "$LIB":/usr/local/lib/libbnxt_re-rdmav34.so:ro \
  "$IMG" infinity
sleep 4
sudo docker exec mori_host bash -c '
rm -f /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav59.so 2>/dev/null
cp -f /usr/local/lib/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so 2>/dev/null
echo "driver bnxt_re" > /etc/libibverbs.d/bnxt_re.driver
ldconfig 2>/dev/null
echo -n "abi="; strings /lib/x86_64-linux-gnu/libibverbs.so.1|grep -m1 IBVERBS_PRIVATE
echo -n " lib="; strings /usr/local/lib/libbnxt_re-rdmav34.so 2>/dev/null | grep -m1 -E "^23[0-9]\."
ibv_devinfo -d rdma3 2>&1 | grep -m1 PORT_ACTIVE || echo NO-DEV'
