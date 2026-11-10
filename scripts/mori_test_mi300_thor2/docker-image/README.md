# Preserved MoRI-EP validated container

The working MoRI-EP test container (`mori_host`) was committed to a local image on
both serving nodes so the exact validated userspace/build is not lost when the
scratch container is removed.

## Image
- **Name:** `moriep-validated:235-async_ll`
- **Committed on:** node-a (192.0.2.10) and node-b (192.0.2.11), 2026-08-02
- **Size:** ~41 GB
- **Base image it was derived from:**
  `rocm/vllm-dev:vllm-wideep_06_29_2026_Shiksha_dp16_2p2d_mori_v1.2.1_aiter_v0.1.16.post3_nightlybase_mori121`

## What's baked in (beyond the base image)
- MoRI built from source at `/tmp/mori-src` — commit `12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4`
  (`0.1.1.dev1+g12d1bc32d`), `BUILD_UMBP=OFF`, `MORI_GPU_ARCHS=gfx942`.
- Host-consistent v34 RDMA stack (libibverbs 1.14.39 + `libbnxt_re-rdmav34.so`, image's v59
  provider removed) — set up by `../scripts/launch_container.sh`.
- The validated launchers in `/tmp`: `go.sh` (EP internode, async_ll), `io_cpu.sh`, `io_gpu.sh`.

## How the container is (re)created from scratch
This is not a Dockerfile-built image — it is a `docker commit` of a running container that
was set up imperatively. To rebuild the equivalent from the base image:
1. `../scripts/launch_container.sh <BASE_IMAGE>`  → starts `mori_host` with the v34 RDMA stack.
2. `docker exec mori_host bash -c "$(cat ../scripts/build_mori.sh)"`  → builds MoRI.
The commit simply snapshots the result of those two steps so the ~15 min MoRI build is cached.

## Save/export the image to NFS (optional, to move between nodes)
```bash
# on a node that has it:
sudo docker save moriep-validated:235-async_ll | gzip > /mnt/nfs/cookbook/moriep-validated-235-async_ll.tar.gz
# on another node:
gunzip -c /mnt/nfs/cookbook/moriep-validated-235-async_ll.tar.gz | sudo docker load
```
