---
name: moriio-nics-thor2
description: Sets the RDMA NIC and socket-interface environment for vllm_dissag MoRIIO runs on nodes with Broadcom Thor2 NICs (bnxt_en / bnxt_re, e.g. MI325X). Use before submitting run_xPyD_models.slurm on a cluster whose /sys/class/infiniband lists rdma* devices rather than mlx5_*.
---

# MoRIIO on Broadcom Thor2 NICs

The harness defaults (`connectors/moriio.sh`) are the CX-7 layout (`mlx5_*`, `eth0`). On Thor2 they
name devices that do not exist and bring-up fails, so set the NICs explicitly at submit time.

## 1. Confirm the NIC type (on a compute node host, not in the container)

```bash
srun -N1 -w <node> bash -c 'for d in /sys/class/infiniband/*; do echo "$(basename $d) $(basename $(readlink $d/device/driver))"; done'
```

Thor2 shows `rdma0 bnxt_en` ... `rdma7 bnxt_en`. If it shows `mlx5_* mlx5_core`, use the
`moriio-nics-cx7` skill instead.

## 2. Get the device list and control interface

Run on the host (the image has no `ip` tool). It lists the active, addressed RDMA devices that are
not behind the default-route (control) interface:

```bash
srun -N1 -w <node> bash -c '
def=$(ip route show default | awk "{print \$5; exit}")
for d in $(ls /sys/class/infiniband); do
  nd=$(ls /sys/class/infiniband/$d/device/net 2>/dev/null | head -1)
  [ -n "$nd" ] && [ "$nd" != "$def" ] || continue
  ip -4 -o addr show "$nd" | grep -q inet || continue
  grep -q ACTIVE /sys/class/infiniband/$d/ports/1/state && echo "$d"
done | paste -sd, -; echo "$def"'
```

On the MI325X cluster this prints `rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7` and `eno0`.

## 3. Export before sbatch

```bash
export MORI_RDMA_DEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7   # the list from step 2
export NCCL_IB_HCA=$MORI_RDMA_DEVICES
export MORI_SOCKET_IFNAME=eno0                                             # the interface from step 2
```

- `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` follow `MORI_SOCKET_IFNAME`; do not set them separately.
- Keep the GID index default (3) for both MoRI and NCCL.
- The slurm mounts the host `libbnxt_re` provider into the container; nothing to install.
- The harness sets `MORI_RDMA_TC=41` for every cluster. The validated Thor2 runs used MoRI's default traffic
  class; if KV transfers stall, set `MORI_RDMA_TC` to the cluster's RoCE lossless class first.

## 4. Submit (GLM-5.3-Flash example)

```bash
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=rocmshared/vllm-glm53flash:glmv5.3-flash.gfx942
export MODEL_NAME=GLM-5.3-Flash-FP8-gfx942 MODEL_WEIGHTS_NAME=GLM-5.3-Flash-FP8 CONNECTOR=moriio WIDE_EP=1 xP=1 yD=1
sbatch -N 2 -n 2 --nodelist=<prefill_node,decode_node> run_xPyD_models.slurm
```

## 5. Verify

- Container env: `docker exec <c> env | grep -E "MORI_RDMA_DEVICES|NCCL_IB_HCA|MORI_SOCKET_IFNAME"` shows the Thor2 values.
- `docker exec <c> ibv_devices` includes `rdma0`..`rdma7`. (libibverbs "couldn't load driver" warnings for other vendors are harmless).
- A first request through the router returns; a KV-transfer timeout in the decode log means the device list,
  interface or traffic class is wrong.
