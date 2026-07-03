# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
# =============================================================================
# vllm_disagg_mori_ep_fullsource.ubuntu.amd.Dockerfile
#   MoRI-EP DeepSeek-V3 2P2D/4P4D (DP/EP=16,32, MI300X)
# Self-contained vLLM model-server image, reconstructed from a named public base
# image plus public Git component pins (no dependency on any opaque prebuilt tag).
#
# There is NO public prebuilt image — build your own and pass it to the launcher:
#   docker build -f docker/vllm_disagg_mori_ep_fullsource.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg-mori-ep:local .
#   export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg-mori-ep:local
# (BASE_IMAGE is a gated nightly; override --build-arg BASE_IMAGE=... as needed.)
# =============================================================================
# Reconstructs the validated v1.2.1 (mori121) runtime stack by applying the recipe's
# component pins ON TOP of the open ROCm vLLM ci_base, cloning each source from
# public Git (no local build-contexts). Mirrors dist-inf-cookbook
# Dockerfile.vllm.mori121_shareable:
#
#   - BASE: rocm/vllm-dev:ci_base-0fcd9b99... (open ROCm 7.2 / cp312 CI base).
#   - MoRI  -> built from ROCm/MoRI @ v1.2.1 (BUILD_UMBP=OFF).
#   - AITER -> 0.1.16.post3 prebuilt rocm7.2 wheel + flydsl 0.2.2; stale JIT wiped.
#   - vLLM  -> COMPILED from shikamd123/vllm @
#     vllm_2p2d_wide-ep_write_shikpate_test_06_29_customer (Wide-EP multi-pod PD, the
#     connector/router reference for the 2P2D DP=EP=16 topology). Full compile: it is
#     a different commit than the base's, so a .py-only overlay would be ABI-mismatched.
#   - RDMA fix (expandable_segments:False x2 + HSA_ENABLE_IPC_MODE_LEGACY=0) is NOT baked
#     here — it lives in scripts/vllm_dissag/connectors/<connector>.env and the launcher
#     forwards it via docker -e. ROCm 7.2.3 cannot dmabuf-export VMM memory, else MoRI
#     RegisterRdmaMemoryRegion EFAULTs (errno 14) on the first disagg WRITE.
#   - vllm-router (vllm-project/router PR#181 = DP-rank round-robin + 2P2D KV-notify
#     dpfix) built in -> no external router binary needed.
#   - validated recipe knobs baked as ENV. The MoRIIO disagg fixes (#39276 notify,
#     #41751 LL split, DP-rank hash-failsafe) are native in this vLLM (no runtime patcher).
#
# Build context = repo root:
#   docker build -f docker/vllm_disagg_mori_ep_fullsource.ubuntu.amd.Dockerfile -t <registry>/<tag> .
#
# BASE_IMAGE is the open rocm/vllm-dev ci_base pinned by the validated recipe
# (dist-inf-cookbook Dockerfile.vllm.mori121_shareable). Override --build-arg
# BASE_IMAGE=... to build on a different ROCm base. vLLM compile is long (~30-60 min).
# =============================================================================

ARG BASE_IMAGE=rocm/vllm-dev:ci_base-0fcd9b99cc9d63202da4c858d8ebc6582c9e2491
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32

# -----------------------------------------------------------------------------
# 1. MoRI: replace the base's bundled MoRI with the validated ROCm/MoRI @ v1.2.1
#    (the version for the 06_29 mori121 image, dist-inf-cookbook
#    Dockerfile.vllm.mori121_shareable). v1.2.1 carries the EP/RDMA correctness fixes
#    plus the ROCm-7.2.3 dmabuf registration path used by the connector .env
#    (expandable_segments:False). MoRI is JIT-built, so this swaps the JIT sources the
#    kernels compile from at runtime.
#    BUILD CONFIG: match the cookbook build — MORI_GPU_ARCHS=gfx942, BUILD_UMBP=OFF,
#    DEFAULT NIC backends. Do NOT pass USE_IONIC=OFF / USE_BNXT=OFF: disabling NIC
#    backends produced a MoRI that deadlocked at the cross-node EP all-to-all init.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=v1.2.1
ENV MORI_GPU_ARCHS=gfx942
# Newer MoRI added the UMBP subsystem which requires gRPC (grpcpp/grpcpp.h) not
# present in this base; UMBP is unrelated to the EP dispatch/combine kernels, so
# disable it to avoid pulling in a gRPC build dependency.
ENV BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
# Build/install matches dist-inf-cookbook Dockerfile.vllm.mori121_shareable for v1.2.1:
# `BUILD_UMBP=OFF pip install .` (default build isolation). apt/pip build tooling kept
# for bases that lack it; harmless where already present.
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true && \
    rm -rf /tmp/mori-src && \
    git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
    cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
    BUILD_UMBP=OFF pip install . && \
    python3 -c "import mori, mori.io, mori.ops; print('MoRI OK at', mori.__path__[0])" && \
    mkdir -p /app && echo "MORI_REF=${MORI_REF}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
    rm -rf /tmp/mori-src

# -----------------------------------------------------------------------------
# 2. AITER: install 0.1.16.post3 (prebuilt rocm7.2 wheel + flydsl 0.2.2), then
#    invalidate the stale prewarmed AITER JIT cache compiled against the old .so.
# -----------------------------------------------------------------------------
ARG AITER_VERSION=0.1.16.post3
ARG AITER_WHEEL_URL="https://github.com/ROCm/aiter/releases/download/v0.1.16.post3/amd_aiter-0.1.16.post3%2Brocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
RUN echo "Bumping AITER to ${AITER_VERSION} from ${AITER_WHEEL_URL}" && \
    _W="/tmp/$(basename "${AITER_WHEEL_URL}" | sed 's/%2B/+/g')" && \
    curl -fL --retry 3 --retry-delay 2 -o "${_W}" "${AITER_WHEEL_URL}" && \
    (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
    pip install --no-deps "${_W}" && \
    pip install "flydsl==0.2.2" && \
    rm -f "${_W}" && \
    python3 - <<'PYEOF'
from importlib.metadata import version as v, PackageNotFoundError
vm = None
for n in ("amd-aiter", "amd_aiter", "aiter"):
    try: vm = v(n); break
    except PackageNotFoundError: pass
assert vm and vm.split("+", 1)[0] == "0.1.16.post3", f"AITER not 0.1.16.post3: {vm!r}"
print("AITER OK:", vm)
PYEOF
RUN rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache" && \
    echo "AITER_VERSION=${AITER_VERSION}" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 3. vLLM: compile from source at the 06_29 validated Wide-EP WRITE-mode branch
#    (matches the published dist-inf-cookbook mori121 image). Full source compile
#    (the base ships a different commit). The MoRIIO disagg fixes (#39276 notify,
#    #41751 LL split, DP-rank hash-failsafe) are native in this branch, so no runtime
#    patcher is needed. Override VLLM_REF to rebuild a different commit; build only
#    committed commits (no working-tree edits).
# -----------------------------------------------------------------------------
# VLLM_REPO/REF are a PUBLIC GitHub repo + branch (the Wide-EP WRITE-mode vLLM the
# dist-inf-cookbook mori121 image builds from). Override to your own vLLM fork/branch.
ARG VLLM_REPO=https://github.com/shikamd123/vllm.git
ARG VLLM_REF=vllm_2p2d_wide-ep_write_shikpate_test_06_29_customer
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    MAX_JOBS=${MAX_JOBS}
RUN rm -rf /tmp/vllm-src && \
    git clone "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && git checkout "${VLLM_REF}" && \
    echo "VLLM_REF=${VLLM_REF}@$(git rev-parse HEAD)" >> /app/versions.txt && \
    pip uninstall -y vllm 2>/dev/null || true && \
    pip install --no-deps --no-build-isolation -v . && \
    python3 -c "import vllm; print('vLLM', vllm.__version__, 'from', vllm.__file__)" && \
    rm -rf /tmp/vllm-src

# Cross-check MoRI + AITER survived the vLLM install (no silent downgrade).
RUN python3 - <<'PYEOF'
from importlib.metadata import version as v, PackageNotFoundError
def get(names):
    for n in names:
        try: return v(n)
        except PackageNotFoundError: pass
    return None
av = get(("amd-aiter", "amd_aiter", "aiter"))
assert av and av.split("+", 1)[0] == "0.1.16.post3", f"AITER downgraded: {av!r}"
import mori, mori.io, mori.ops
print("Post-vLLM check OK: AITER", av, "+ MoRI importable")
PYEOF

# -----------------------------------------------------------------------------
# 4. vllm-router (DP-rank round-robin + MoRIIO connector) — built in, so NO
#    external vllm-router binary is needed (leave ROUTER_BINARY unset).
#    Source = vllm-project/router PR #181 branch, which now carries BOTH the
#    round-robin DP-rank fix (11841c0d) AND the 2P2D KV-notify fix (6409ac1:
#    remote_dp_rank_override + remote_dp_size). The KV-notify fix is REQUIRED:
#    without it the 2P2D EP=16 run reproducibly wedges with "remote blocks never
#    arrived" deferred-write expiries (decode notify targets the wrong DP rank).
#    This is the exact source of the validated vllm-router-2p2d-dpfix binary.
#    Pinned Rust toolchain (>=1.88: router deps time/home require rustc 1.88).
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=ravgupta/discovery-dp-rank-roundrobin
ARG RUST_TOOLCHAIN=1.88.0
RUN if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi && \
    export PATH="/root/.cargo/bin:${PATH}" && \
    rm -rf /tmp/vllm-router-src && \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
    cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
    cargo build --release && \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
    vllm-router --help 2>&1 | grep -q moriio && \
    echo "VLLM_ROUTER_REF=${ROUTER_REPO}@${ROUTER_REF}@$(git -C /tmp/vllm-router-src rev-parse HEAD)" >> /app/versions.txt && \
    rm -rf /tmp/vllm-router-src

# -----------------------------------------------------------------------------
# 5. Recipe config defaults baked as ENV (the validated v1.2.1 / mori121 knobs).
#    Overridable at run time; the launcher still reads these via ${VAR:-}.
# -----------------------------------------------------------------------------
# Decode/attention path (block=1 + AITER MLA GPU-faults -> block=16 + Triton MLA)
ENV KV_BLOCK_SIZE=16 \
    VLLM_ROCM_USE_AITER_MLA=0 \
    VLLM_CUDAGRAPH_MODE=PIECEWISE \
    KV_CACHE_DTYPE=fp8 \
    GPU_MEMORY_UTILIZATION=0.80 \
    KV_CACHE_MEMORY_BYTES=20000000000 \
    TORCHINDUCTOR_BENCHMARK_KERNEL=0
# NOTE: the ROCm-7.2.3 GPU-RDMA runtime env (expandable_segments:False x2,
# HSA_ENABLE_IPC_MODE_LEGACY=0, MORI_GPU_ARCHS, HSA_NO_SCRATCH_RECLAIM) is NOT baked
# here. It lives in scripts/vllm_dissag/connectors/<connector>.env and the slurm
# launcher forwards it via `docker -e` (must reach PID 1; PyTorch reads alloc-conf at
# import). Keeping it out of the image means one editable home per connector and no
# image rebuild when the platform requirement changes. If you run this image WITHOUT
# the launcher, set those vars yourself (see connectors/moriio.env).
# Per-role cudagraph: prefill NONE (PIECEWISE prefill deadlocks the multi-node DP
# capture barrier), decode PIECEWISE (keeps the ITL win).
ENV PREFILL_CUDAGRAPH_MODE=NONE \
    DECODE_CUDAGRAPH_MODE=PIECEWISE
# Per-role all-to-all (decode low-latency = ~13% lower ITL)
ENV VLLM_ALL2ALL_BACKEND=mori_high_throughput \
    PREFILL_MORI_BACKEND=mori_high_throughput \
    DECODE_MORI_BACKEND=mori_low_latency
# (MoRIIO disagg fixes are native in the compiled vLLM (06_29 wide-ep WRITE branch);
#  there is no runtime patcher, so no SKIP_RUNTIME_PATCH gate.)
# Caches: use the image's /opt/vllm_cache (rebaked clean after the AITER bump)
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr
# MoRI / RDMA fabric tuning (validated on OCI MI300X RoCEv2; override per-cluster)
ENV MORI_RDMA_TC=41 MORI_RDMA_SL=0 MORI_IO_SL=1 \
    MORI_IB_ENABLE_RELAXED_ORDERING=1 MORI_IB_GID_INDEX=1 \
    MORI_NUM_QP_PER_PE=8 VLLM_MORIIO_QP_PER_TRANSFER=2 VLLM_MORIIO_NUM_WORKERS=4 \
    HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=1

# -----------------------------------------------------------------------------
# 6. CRITICAL: scrub build-time MoRI JIT state. The `import mori` verification
#    steps above compile/lock MoRI EP kernels under /root/.mori/jit on THIS build
#    host, leaving stale .hsaco.lock files (ep_internode_v1, ep_internode_v1ll, ...).
#    At runtime on the cluster, MoriAll2AllManager finds those locks, waits on a
#    build-in-progress whose owner PID is long gone, and DEADLOCKS at ep:0 init.
#    A clean image ships /root/.mori empty -> runtime compiles fresh.
#    Clearing these makes the from-source image boot clean on 2P2D/4P4D.
# -----------------------------------------------------------------------------
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED: /root/.mori + /tmp/mori_jit_* cleared at build end" >> /app/versions.txt

RUN cat /app/versions.txt 2>/dev/null | tail -20 || true
