# =============================================================================
# Self-contained MoRI-EP + MoRI-IO test image for MI300X + Broadcom Thor2 (BCM57608)
# Portable build — starts FROM the public base image (no local committed image needed).
#
# What it produces: an image that runs the MoRI-EP (async_ll) + MoRI-IO benchmarks with
# NO runtime library bind-mounts, because it bakes in the host v34 RDMA userspace.
#
# BEFORE building you MUST:
#   1. Be on a host already running the bnxt 235.2.86.0 driver (see ../driver-235.2.86.0/).
#   2. Run  bash collect_host_libs.sh   -> populates ./libs/ with the 5 host RDMA .so files.
#   3. Have Docker Hub pull access to the base image + build-time internet (MoRI git clone).
#
# Build:
#   docker build -f Dockerfile -t <your-registry>/vllm_wideEp_Mori_tests .
#
# Run (device access CANNOT be baked — these are docker-run flags):
#   docker run -d --name mori_host --network host --ipc host --privileged \
#     --device /dev/kfd --device /dev/dri --device /dev/infiniband \
#     -v /lib/modules:/lib/modules:ro --ulimit memlock=-1:-1 --shm-size 64g \
#     --entrypoint sleep <your-tag> infinity
# =============================================================================
ARG BASE=docker.io/rocm/vllm-dev:vllm-wideep_06_29_2026_Shiksha_dp16_2p2d_mori_v1.2.1_aiter_v0.1.16.post3_nightlybase_mori121
FROM ${BASE}

# --- 1. Build MoRI from source at the pinned CI-green commit 12d1bc32 (needs build-time internet) ---
COPY build_mori.sh /opt/mori-tests/build_mori.sh
RUN bash /opt/mori-tests/build_mori.sh && \
    python3 -c "import mori; print('MoRI', mori.__version__)"

# --- 2. Apply the vLLM MoRI-EP async_ll kernel-selection patch (Thor2 has no PCIe atomic-completer;
#         the default InterNodeV1 atomic kernel hangs — AsyncLL uses WRITE+poll). Idempotent. ---
COPY apply_async_ll_patch.py /opt/mori-tests/apply_async_ll_patch.py
RUN python3 /opt/mori-tests/apply_async_ll_patch.py || echo "[thor2] async_ll patch skipped (kernel-select anchor absent)"

# --- 3. Bake the host v34 RDMA userspace (from ./libs/, gathered by collect_host_libs.sh) ---
#     The base image's own libibverbs is v59 (IBVERBS_PRIVATE_59) which REJECTS the bnxt kernel
#     ABI-8 provider -> 0 RDMA devices. We force the host v34 stack instead.
COPY libs/libibverbs.so.1.14.39.0        /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0
COPY libs/librdmacm.so.1.3.39.0          /usr/lib/x86_64-linux-gnu/librdmacm.so.1.3.39.0
COPY libs/libnl-3.so.200.26.0            /usr/lib/x86_64-linux-gnu/libnl-3.so.200.26.0
COPY libs/libnl-route-3.so.200.26.0      /usr/lib/x86_64-linux-gnu/libnl-route-3.so.200.26.0
COPY libs/libbnxt_re-rdmav34.so          /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so

RUN set -eux; \
    cd /usr/lib/x86_64-linux-gnu; \
    # force libibverbs.so.1 -> the v34 lib; remove the base image's v59 (1.16.62.0) + repoint aliases
    rm -f libibverbs.so.1 libibverbs.so libibverbs.so.1.16.62.0; \
    ln -sf libibverbs.so.1.14.39.0 libibverbs.so.1; \
    ln -sf libibverbs.so.1.14.39.0 libibverbs.so; \
    rm -f librdmacm.so.1 librdmacm.so; \
    ln -sf librdmacm.so.1.3.39.0   librdmacm.so.1; \
    ln -sf librdmacm.so.1.3.39.0   librdmacm.so; \
    ln -sf libnl-3.so.200.26.0     libnl-3.so.200; \
    ln -sf libnl-route-3.so.200.26.0 libnl-route-3.so.200; \
    # bnxt provider: expose under /usr/local/lib + into the libibverbs provider dir; remove the v59 provider
    ln -sf /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so; \
    mkdir -p /usr/lib/x86_64-linux-gnu/libibverbs; \
    cp -f /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
    rm -f /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav59.so; \
    # tell libibverbs to load the bnxt provider
    mkdir -p /etc/libibverbs.d; echo "driver bnxt_re" > /etc/libibverbs.d/bnxt_re.driver; \
    echo "/usr/local/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/zz-bnxt.conf; \
    ldconfig; \
    # build-time assert: libibverbs.so.1 now resolves to v34 (fails the build otherwise)
    test "$(strings /usr/lib/x86_64-linux-gnu/libibverbs.so.1 | grep -m1 IBVERBS_PRIVATE)" = "IBVERBS_PRIVATE_34"

# --- 4. Bake the test scripts ---
COPY ep_pair_test.sh /opt/mori-tests/ep_pair_test.sh
COPY io_pair_test.sh /opt/mori-tests/io_pair_test.sh
RUN chmod +x /opt/mori-tests/*.sh

# --- 5. MoRI / RoCE env defaults for this fabric (override at run time as needed) ---
# Fabric-specific values are build-args so you can bake YOUR cluster's defaults:
#   docker build --build-arg SOCKET_IFNAME=<your-mgmt-nic> --build-arg GID_INDEX=<n> ...
# (These are only defaults — the test scripts also honor SOCKET_IFNAME/MORI_* at run time.)
ARG SOCKET_IFNAME=eno8303
ARG RDMA_DEVICES=rdma3
ARG GID_INDEX=3
ARG RDMA_SL=3
ARG RDMA_TC=104
ENV MORI_GPU_ARCHS=gfx942 \
    MORI_IB_GID_INDEX=${GID_INDEX} \
    MORI_RDMA_SL=${RDMA_SL} \
    MORI_RDMA_TC=${RDMA_TC} \
    MORI_RDMA_DEVICES=${RDMA_DEVICES} \
    MORI_EP_FORCE_ASYNC_LL=1 \
    MORI_ENABLE_SDMA=1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    HSA_ENABLE_IPC_MODE_LEGACY=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:False \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
    GLOO_SOCKET_IFNAME=${SOCKET_IFNAME} \
    MORI_SOCKET_IFNAME=${SOCKET_IFNAME} \
    NCCL_SOCKET_IFNAME=${SOCKET_IFNAME} \
    PYTHONPATH=/tmp/mori-src

LABEL org.opencontainers.image.title="MoRI-EP/IO tests — MI300X + Broadcom Thor2" \
      mori.commit="12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4" \
      bnxt.driver="235.2.86.0" \
      bnxt.firmware="238.1.138.0" \
      rocm.version="7.2.3" \
      linux.kernel="5.15.0-177-generic" \
      verbs.abi="v34 (IBVERBS_PRIVATE_34)" \
      gpu.arch="gfx942"
