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
# vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile
#   GLM-5.1-FP8 (MLA + DeepSeek Sparse Attention) MoRI-EP WideEP disagg image.
#   PER-MODEL image, isolated from the base vllm_disagg_inference Dockerfile
#   (which stays pinned to the DeepSeek-V3 / R1 stack), so each model can pin its
#   own vLLM/AITER/MoRI without disturbing the others.
#
#   ALL connectors in one image: moriio (TP + MoRI-EP wideEP) + rixl (NIXL TP +
#   DeepEP wideEP). = the fullsource MoRI stack, plus a UCX/RIXL/rocSHMEM/DeepEP
#   transport layer gated by --build-arg WITH_NIXL (default 0 = MoRI-EP only).
#
#   docker build -f docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:glmv5.1 .
#   export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:glmv5.1
#
# STATUS (GLM-5.1-FP8 on this stack): validated on 1P/1D EP8 and 2P/2D EP16 — NIAH
# 2k-35k retrieves with no length collapse and no crash (long-context accuracy fixed
# via vLLM #47766). The 4P/4D EP32 token corruption is FIXED by the MoRI combine()
# original-topk change (vLLM 623fdc946b): measured 4P/4D EP32 NIAH 8k 0/10 -> 10/10,
# perf-neutral. NOTE: 4P/4D has not been re-validated on the bumped pins below; 2P/2D
# EP16 has (NIAH 2k 10/10, 8k 10/10; 8192/1024 con32 TPOT ~59 ms).
# (BASE_IMAGE is a gated nightly; override --build-arg BASE_IMAGE=...; vLLM compile ~30-60 min.)
# =============================================================================
# Builds the GLM-5.1 runtime stack by applying component pins ON TOP of a
# purpose-built ROCm/vLLM/MoRI base, cloning each overridden source from public Git
# (no local build-contexts):
#
#   - BASE: rocm/vllm-dev:ci_base-dedbf6be8b1afa17a6220473b9c8c98242ac1c03
#     (ROCm + torch nightly). The stages below OVERRIDE the base's vLLM/MoRI/AITER
#     with the pins we validate for GLM DSA.
#   - MoRI  -> built from ROCm/MoRI @ 624002c897a3 (BUILD_UMBP=OFF). Bumped from
#     42e895472b08 alongside the AITER bump; co-validated 2P/2D EP16.
#   - AITER -> raviguptaamd/aiter @ 624e43586b (ROCm/aiter 1d872fa + the gfx942 gqa64
#     decode fix filed upstream as ROCm/aiter#4957) from source + flydsl 0.3.1;
#     stale JIT wiped. (#47766 keeps persistent MLA ON -> aiter native gqa64 fold.)
#   - vLLM  -> COMPILED from raviguptaamd/vllm @ 094820b5d (branch
#     glm5.1-dsa-wideEP_on_d626108b = upstream d626108b + the 10 GLM DSA commits,
#     incl. the EP32 combine-topk fix). Full compile: a different commit than the
#     base's, so a .py-only overlay would be ABI-mismatched.
#   - RDMA fix (expandable_segments:False x2 + HSA_ENABLE_IPC_MODE_LEGACY=0) is NOT baked
#     here — it lives in scripts/vllm_dissag/connectors/<connector>.env and the launcher
#     forwards it via docker -e. ROCm 7.2.3 cannot dmabuf-export VMM memory, else MoRI
#     RegisterRdmaMemoryRegion EFAULTs (errno 14) on the first disagg WRITE.
#   - vllm-router (vllm-project/router PR#181 = DP-rank round-robin + 2P2D KV-notify
#     dpfix) built in -> no external router binary needed.
#   - MoRIIO disagg fixes (#39276 notify, #41751 LL split, DP-rank hash-failsafe) native.
#
# THIS IMAGE IS THE CONTRACT. The MAD side (scripts/vllm_dissag) is catalog/config only
# and ships NO runtime .py patchers, so EVERY GLM DSA source fix must be carried
# in-source HERE, by VLLM_REF below plus the MoRI/AITER pins. Serving GLM-5.1-FP8 from an
# image built off an older vLLM ref is unsupported and fails SILENTLY: it boots, then
# produces garbage output or stalls the disagg KV transfer, with nothing to fall back on
# (measured: rocm/pytorch-private:glm-dockerimage-built-09072026 scored NIAH 2k 0/10).
# If you need a fix, move the pin and rebuild — do not re-add runtime patchers to MAD.
# =============================================================================

ARG BASE_IMAGE=rocm/vllm-dev:ci_base-dedbf6be8b1afa17a6220473b9c8c98242ac1c03
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32
# NIXL/RIXL transport for the rixl connector. GLM-5.1 is served over MoRI-EP, so this
# stack is dead weight here. Default 0 => lean MoRI-EP-only image, which is how the
# validated image was built. Set --build-arg WITH_NIXL=1 to get the rixl connector.
ARG WITH_NIXL=0
ARG NIC_COMPILATION_ARCH="cx7"

# -----------------------------------------------------------------------------
# 1. MoRI: replace the base's bundled MoRI with the commit GLM-5.1 DSA wideEP was
#    validated on, ROCm/MoRI @ 624002c897a3. NOTE this is NOT tag v1.2.1 that the base
#    vllm_disagg_inference Dockerfile pins. It carries the EP/RDMA correctness fixes
#    this recipe needs plus the ROCm-7.2.3 dmabuf registration path used by the
#    connector .env (expandable_segments:False). MoRI is JIT-built, so this swaps the
#    JIT sources the kernels compile from at runtime.
#    BUILD CONFIG: match the cookbook build — MORI_GPU_ARCHS=gfx942, BUILD_UMBP=OFF,
#    DEFAULT NIC backends. Do NOT pass USE_IONIC=OFF / USE_BNXT=OFF: disabling NIC
#    backends produced a MoRI that deadlocked at the cross-node EP all-to-all init.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
# 624002c897a3: validated MoRI tip for GLM DSA WideEP disagg (bumped from 42e895472b08,
# which predates the recv-sizing fixes the VLLM_MORI_* knobs need). The base's bundled
# amd_mori regressed GLM DSA (GPU fault on the aiter DSA decode kernel), so we build
# from source at this pinned commit by DEFAULT; WITH_MORI_BUILD=0 falls back for debug.
ARG WITH_MORI_BUILD=1
ARG MORI_REF=624002c897a3
ENV MORI_GPU_ARCHS=gfx942
# Newer MoRI added the UMBP subsystem which requires gRPC (grpcpp/grpcpp.h) not
# present in this base; UMBP is unrelated to the EP dispatch/combine kernels, so
# disable it to avoid pulling in a gRPC build dependency.
ENV BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
# Build/install COMMAND (not the version) matches dist-inf-cookbook
# Dockerfile.vllm.mori121_shareable:
# `BUILD_UMBP=OFF pip install .` (default build isolation). apt/pip build tooling kept
# for bases that lack it; harmless where already present.
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    mkdir -p /app && \
    if [ "${WITH_MORI_BUILD}" != "1" ]; then \
        python3 -c "import mori, mori.io, mori.ops; print('MoRI (bundled) OK at', mori.__path__[0])" && \
        echo "MORI_REF=BUNDLED (base amd_mori, WITH_MORI_BUILD=0)" >> /app/versions.txt ; \
    else \
        pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true && \
        rm -rf /tmp/mori-src && \
        git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
        cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
        BUILD_UMBP=OFF pip install . && \
        python3 -c "import mori, mori.io, mori.ops; print('MoRI OK at', mori.__path__[0])" && \
        echo "MORI_REF=${MORI_REF}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
        rm -rf /tmp/mori-src ; \
    fi

# -----------------------------------------------------------------------------
# 2. AITER: built from source at raviguptaamd/aiter @ 624e43586b (WITH_AITER_BUILD=1).
#    That is ROCm/aiter 1d872fa plus a 7-line fix (filed upstream as ROCm/aiter#4957):
#    newer aiter claims native gfx942 support for gqa64 fp8 decode and routes it to
#    mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps, which GPU-faults; the fix lets gqa64 fall
#    through to aiter's capture-safe persistent view-fold, so cudagraph decode is kept.
#    The fork ref is TEMPORARY — revert AITER_REPO to ROCm/aiter once #4957 merges.
#    --build-arg WITH_AITER_BUILD=0 falls back to the bundled aiter for debugging.
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/raviguptaamd/aiter.git
ARG WITH_AITER_BUILD=1
ARG AITER_REF=624e43586b
RUN if [ "${WITH_AITER_BUILD}" != "1" ]; then \
        echo "AITER: using BUNDLED base aiter (WITH_AITER_BUILD=0)" && \
        python3 -c "import importlib.metadata as m; print('aiter (bundled)', m.version('amd-aiter'))" && \
        echo "AITER_REF=BUNDLED (base amd-aiter, WITH_AITER_BUILD=0)" >> /app/versions.txt ; \
    else \
        echo "Compiling STOCK AITER (no fork) from ${AITER_REPO}@${AITER_REF}" && \
        rm -rf /tmp/aiter-src && \
        git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
        cd /tmp/aiter-src && git checkout "${AITER_REF}" && \
        git submodule update --init --recursive && \
        (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
        pip install --no-build-isolation --no-deps -v . && \
        pip install --no-deps -U "flydsl==0.3.1" && \
        echo "AITER_REF=${AITER_REF}@$(git rev-parse HEAD) (aiter + ROCm/aiter#4957 gqa64 fix)" >> /app/versions.txt && \
        rm -rf /tmp/aiter-src && \
        rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache" ; \
    fi

# -----------------------------------------------------------------------------
# 3. vLLM: compile from source at the GLM-5.1 DSA wideEP branch. Full source compile
#    (the base ships a different commit). The MoRIIO disagg fixes (#39276 notify,
#    #41751 LL split, DP-rank hash-failsafe) AND the GLM DSA fixes are native in this
#    branch, so no runtime patcher is needed — and none exists in MAD, which is why
#    this ref is a hard requirement rather than a preference. Override VLLM_REF to
#    rebuild a different commit; build only committed commits (no working-tree edits).
# -----------------------------------------------------------------------------
# VLLM_REPO/REF are a PUBLIC GitHub repo + branch. Override to your own vLLM fork/branch.
ARG VLLM_REPO=https://github.com/raviguptaamd/vllm.git
# REPRODUCIBILITY: this default is a BRANCH NAME, so it is mutable — `docker build`
# resolves it to whatever the tip is on the day you build, and two builds can ship
# different engines. For an auditable rebuild pass the exact commit:
#   --build-arg VLLM_REF=094820b5deeb1b93733586ca8942589e385a25dc
# which is the tip every number in models.yaml was measured on. /app/versions.txt in the
# built image records whichever sha was resolved.
#
# What the ref carries: the 10 GLM DSA commits (per-req-ctx metadata key #47766, DSA
# indexer KV transfer, invalid-token sentinel, MoRI EP sizing knobs, and the EP32
# combine() original-topk fix) on top of upstream vLLM d626108b (2026-08-20).
ARG VLLM_REF=094820b5deeb1b93733586ca8942589e385a25dc
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
# Assert presence, not a version string: we pin aiter by commit and the reported version
# varies by build. Do NOT `import aiter` here — it pulls torch->amdsmi->libamd_smi.so,
# which is not loadable in the no-GPU build sandbox.
assert av, "AITER missing after vLLM install (expected bundled 0.1.19 or source-built ref)"
import mori, mori.io, mori.ops
print("Post-vLLM check OK: AITER", av, "present + MoRI importable")
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
ARG ROUTER_REF=ravgupta/dp-roundrobin-on-tip
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
# 4b. WITH_NIXL=1: UCX + RIXL(+nixlbench) + rocSHMEM + DeepEP from source,
#     so the rixl connector (NIXL TP + DeepEP wideEP) is present. Default is 0
#     (MoRI-EP only). Single guarded RUN so WITH_NIXL=0 skips it (no layers, no cost).
# -----------------------------------------------------------------------------
ENV _ROCM_DIR=/opt/rocm \
    _UCX_SOURCE=https://github.com/ROCm/ucx.git \
    _UCX_BRANCH=da3fac2a \
    _UCX_INSTALL_DIR=/usr/local/ucx/ \
    _RIXL_SOURCE=https://github.com/ROCm/RIXL.git \
    _RIXL_BRANCH=f33a5599 \
    _RIXL_INSTALL_DIR=/usr/local/RIXL/install \
    _NIXLBENCH_INSTALL_DIR=/usr/local/RIXL
RUN if [ "${WITH_NIXL}" != "1" ]; then \
      echo "WITH_NIXL=${WITH_NIXL}: skipping UCX/RIXL/rocSHMEM/DeepEP (MoRI-EP + base DeepEP only)"; \
    else set -e && \
      echo "WITH_NIXL=1: building UCX + RIXL + rocSHMEM + DeepEP" && \
      apt-get update && apt-get install -y \
        autoconf automake libtool autogen pkg-config m4 gcc make \
        librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
        libibverbs-dev rdma-core strace libgflags-dev \
        libaio-dev liburing-dev libcpprest-dev libgrpc-dev libgrpc++-dev \
        libprotobuf-dev protobuf-compiler-grpc wget && \
      pip install meson==0.64.0 "pybind11[global]" pyyaml && \
      # UCX
      cd /tmp && git clone "${_UCX_SOURCE}" && cd ucx && git checkout "${_UCX_BRANCH}" && \
        ./autogen.sh && mkdir -p build && cd build && \
        ../configure --prefix="${_UCX_INSTALL_DIR}" --with-rocm="${_ROCM_DIR}" \
          --disable-go --disable-java --disable-assertions --enable-mt && \
        make -j && make install && \
      # googletest (RIXL dep)
      cd /tmp && wget -q https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz && \
        tar -xzf v1.14.0.tar.gz && cd googletest-1.14.0 && mkdir -p build && cd build && \
        cmake -DBUILD_SHARED_LIBS=on .. && make -j && make install && \
      # RIXL + python bindings
      cd /tmp && git clone "${_RIXL_SOURCE}" && cd RIXL && git checkout "${_RIXL_BRANCH}" && \
        meson setup build/ --prefix="${_RIXL_INSTALL_DIR}" -Ducx_path="${_UCX_INSTALL_DIR}" \
          -Ddisable_gds_backend=true -Dcudapath_inc="${_ROCM_DIR}/include" -Dcudapath_lib="${_ROCM_DIR}/lib" && \
        cd build && ninja && ninja install && cd /tmp/RIXL && \
        pip install --config-settings=setup-args="-Dcudapath_inc=${_ROCM_DIR}/include" \
                    --config-settings=setup-args="-Dcudapath_lib=${_ROCM_DIR}/lib" \
                    --config-settings=setup-args="-Ducx_path=${_UCX_INSTALL_DIR}" \
                    --config-settings=setup-args="-Ddisable_gds_backend=true" . && \
      # rocSHMEM (DeepEP dep)
      cd /tmp && git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && \
        cd rocm-systems && git sparse-checkout set --cone projects/rocshmem && git checkout develop && \
        mkdir -p /tmp/rocshmem-build && cd /tmp/rocshmem-build && \
        /tmp/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends \
          -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS="${GFX_COMPILATION_ARCH}" && \
      # DeepEP (build develop against the installed vLLM/torch)
      cd /tmp && git clone https://github.com/ROCm/DeepEP.git && cd DeepEP && \
        PYTORCH_ROCM_ARCH="${GFX_COMPILATION_ARCH}" CFLAGS="-O3 -fPIC" \
          CXXFLAGS="-O3 -fPIC --offload-arch=${GFX_COMPILATION_ARCH}" HIP_CXX_FLAGS="-O3 -fPIC" \
          python3 setup.py --variant rocm --nic "${NIC_COMPILATION_ARCH}" build develop && \
      echo "WITH_NIXL build complete" >> /app/versions.txt && \
      rm -rf /tmp/ucx /tmp/googletest-1.14.0 /tmp/v1.14.0.tar.gz /tmp/rocm-systems /tmp/rocshmem-build; \
    fi
ENV LD_LIBRARY_PATH="/usr/local/ucx/lib:/usr/local/lib:/usr/local/RIXL/install/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/ucx/bin:${PATH}"

# -----------------------------------------------------------------------------
# 5. Cache locations (structural: WHERE the JIT/compile caches live in the image).
#    These are the mount target for the launcher's persistent host JIT cache.
# -----------------------------------------------------------------------------
# The image ships NO runtime recipe / tuning / platform ENV. By design, everything
# run-tunable is applied at launch, so this image stays a clean binary/library artifact
# and the same image serves any model/cluster without a rebuild:
#   - model-serving recipe (KV_BLOCK_SIZE, KV_CACHE_DTYPE, *_CUDAGRAPH_MODE, *_MORI_BACKEND,
#     GPU_MEMORY_UTILIZATION, KV_CACHE_MEMORY_BYTES, VLLM_ROCM_USE_AITER_MLA, ...)
#       -> scripts/vllm_dissag/models.yaml   (per-model env:, so dense vs MoE differ)
#   - ROCm-7.2.3 GPU-RDMA platform env (expandable_segments:False x2, MORI_GPU_ARCHS,
#     HSA_ENABLE_IPC_MODE_LEGACY=0, HSA_NO_SCRATCH_RECLAIM) and the MoRI/RDMA fabric
#     tuning (MORI_RDMA_TC/SL, MORI_IB_GID_INDEX, MORI_NUM_QP_PER_PE, VLLM_MORIIO_*, ...)
#       -> scripts/vllm_dissag/connectors/<connector>.env  (cluster-editable, no rebuild)
# The slurm launcher forwards both via `docker -e` (platform env must reach PID 1 -
# PyTorch reads alloc-conf at import). Running this image WITHOUT the launcher: set the
# vars you need yourself (see connectors/moriio.env + models.yaml for the values).
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

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
