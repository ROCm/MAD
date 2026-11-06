---
name: moriio-nics-cx7
description: Checks and sets the RDMA NIC and socket-interface environment for vllm_dissag MoRIIO runs on nodes with NVIDIA/Mellanox ConnectX-7 NICs (mlx5_core, e.g. MI300X). Use before submitting run_xPyD_models.slurm on a cluster whose /sys/class/infiniband lists mlx5_* devices.
---

# MoRIIO on ConnectX-7 NICs

The harness defaults (`connectors/moriio.sh`) are the CX-7 layout of the MI300X cluster:
`MORI_RDMA_DEVICES` / `NCCL_IB_HCA` = `mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9`,
`MORI_SOCKET_IFNAME=eth0`, GID index 3. On that layout there is nothing to set. Confirm it matches.

## 1. Confirm the NIC type (on a compute node host, not in the container)

```bash
srun -N1 -w <node> bash -c 'for d in /sys/class/infiniband/*; do echo "$(basename $d) $(basename $(readlink $d/device/driver))"; done'
```

CX-7 shows `mlx5_* mlx5_core`. If it shows `rdma* bnxt_en`, use the `moriio-nics-thor2` skill instead.

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

On the MI300X cluster this prints `mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9` and `eth0`
(`mlx5_1` backs `eth0`, the control interface; `mlx5_6` has no address).

- Same as the defaults: submit without NIC settings.
- Different: export the printed values before sbatch:

```bash
export MORI_RDMA_DEVICES=<list from step 2>
export NCCL_IB_HCA=$MORI_RDMA_DEVICES
export MORI_SOCKET_IFNAME=<interface from step 2>
```

Use the RDMA device names (`mlx5_*`), never the network interface names. On the MI300X cluster the backend
interfaces are *named* `rdma0`..`rdma7`, which are not valid values here.

## 3. Submit (GLM-5.3-Flash example)

```bash
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=rocmshared/vllm-glm53flash:glmv5.3-flash.gfx942
export MODEL_NAME=GLM-5.3-Flash-FP8-gfx942 CONNECTOR=moriio WIDE_EP=1 xP=1 yD=1
sbatch -N 2 -n 2 --nodelist=<prefill_node,decode_node> run_xPyD_models.slurm
```

## 4. Verify

- `docker exec <c> ibv_devices` includes the `mlx5_*` devices from step 2. (libibverbs "couldn't load driver" warnings for other vendors are harmless).
- A first request through the router returns; a KV-transfer timeout in the decode log means the device list
  or interface is wrong.
