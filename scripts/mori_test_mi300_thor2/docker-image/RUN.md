# rocm/vllm-dev:vllm_wideEp_Mori_tests_August2_2026 — self-contained MoRI-EP/IO test image

**Pushed:** `docker.io/rocm/vllm-dev:vllm_wideEp_Mori_tests_August2_2026`
**Digest:** `sha256:d260b8273d13fad99b71a1cd46f8d131d157749ff9664b112f8f848a7ca4a893`
**Built:** 2026-08-02 · from `moriep-validated:235-async_ll` via `Dockerfile.mori-tests`

## What's baked in (no runtime bind-mounts needed for RDMA)
| Ingredient | Version |
|---|---|
| MoRI (built from source at `/tmp/mori-src`) | commit `12d1bc32` (`0.1.1.dev1+g12d1bc32d`) |
| vLLM async_ll kernel-selection patch | applied |
| bnxt userspace provider `libbnxt_re-rdmav34.so` | 235.2.86.0 |
| `libibverbs.so.1` | v34 (IBVERBS_PRIVATE_34) — image's v59 removed |
| `librdmacm`, `libnl-3`, `libnl-route-3` | host 39.0 / 3.x |
| `/etc/libibverbs.d/bnxt_re.driver` | present |
| ROCm / PyTorch / py3.12 / gfx942 | 7.2.3 (from base) |
| test scripts | `/opt/mori-tests/ep_pair_test.sh`, `io_pair_test.sh` |

**Host requirements (matching stack the image expects):** bnxt_re kernel driver **235.2.86.0**,
firmware **238.1.138.0**, kernel **5.15.0-177-generic**. The image supplies the *userspace*; the
kernel driver + firmware live on the host.

## Self-containment — verified
Run with ONLY device flags (no `-v` lib mounts): `ibv_devinfo -d rdma3` → `PORT_ACTIVE`, 8 RDMA
devices, verbs ABI `IBVERBS_PRIVATE_34`, `import mori` → `0.1.1.dev1+g12d1bc32d`. A real 2-node
EP async_ll run from this image passed with 0 errors on both ranks.

## Run (the device/privilege flags CANNOT be baked into any image — pass them at run time)
```bash
docker run -d --name mori_host \
  --network host --ipc host --privileged \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v /lib/modules:/lib/modules:ro \
  --ulimit memlock=-1:-1 --ulimit nproc=100000:100000 --shm-size 64g --cap-add SYS_PTRACE \
  --entrypoint sleep \
  rocm/vllm-dev:vllm_wideEp_Mori_tests_August2_2026 infinity
```

### MoRI-EP internode (async_ll) — run rank1 first, then rank0
```bash
# on node B (rank1):
docker exec mori_host bash /opt/mori-tests/ep_pair_test.sh 1 <master_mgmt_ip> rdma3 29100
# on node A (rank0 / master):
docker exec mori_host bash /opt/mori-tests/ep_pair_test.sh 0 <master_mgmt_ip> rdma3 29100
# expected: "Node N Dispatch Pass" + "Node N Combine Pass" each round;
#           final "rank: N error times: 0 appear round: set()"
```

### MoRI-IO CPU write sweep
```bash
docker exec mori_host bash /opt/mori-tests/io_pair_test.sh 1 <master_mgmt_ip> <nodeB_mgmt_ip> rdma3 29500  # target
docker exec mori_host bash /opt/mori-tests/io_pair_test.sh 0 <master_mgmt_ip> <nodeA_mgmt_ip> rdma3 29500  # initiator
# expected: sweep table peaking ~48.4 GB/s @ 64 MiB
```

Baked env defaults (override at run time as needed): `MORI_GPU_ARCHS=gfx942`, `MORI_IB_GID_INDEX=3`,
`MORI_RDMA_SL=3`, `MORI_RDMA_TC=104`, `MORI_RDMA_DEVICES=rdma3`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`,
`*_SOCKET_IFNAME=eno8303`, `PYTHONPATH=/tmp/mori-src`.

## Rebuild from scratch
```bash
# context needs: Dockerfile.mori-tests, libs/ (5 host RDMA .so), tests/ (ep + io scripts)
docker build -f Dockerfile.mori-tests -t rocm/vllm-dev:vllm_wideEp_Mori_tests_August2_2026 .
```
