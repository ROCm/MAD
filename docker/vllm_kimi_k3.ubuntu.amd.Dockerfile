# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
# vllm_kimi_k3.ubuntu.amd.Dockerfile
#
#   The one Kimi-K3 image for every Kimi-K3 vLLM card, on MI300X (gfx942) and
#   MI355X (gfx950), colocated multinode and disaggregated alike. It replaces
#   pyt_vllm_kimi_k3_mi300x, pyt_vllm_kimi_k3_mi355x and vllm_disagg_inference.kimik3,
#   which were the same stack apart from the GPU arch and a few pins
#   (vLLM, MoRI and AITER were identical in all three).
#
#   GPU ARCH. Built for exactly one arch, MAD_SYSTEM_GPU_ARCHITECTURE (gfx942 or
#   gfx950): madengine passes it, and the MAD CI pipeline sets it from the compute
#   nodes it probed. Everything arch-specific follows it: MoRI's JIT target, the
#   vLLM compile, and, with WITH_NIXL=1, rocSHMEM and DeepEP. Runtime differences
#   between the arches (AITER MLA gfx950-only, int4 MoE requant on gfx942) are
#   recipe knobs in scripts/*/models.yaml, not build steps.
#
#   AITER. Built from source at the commit the Kimi-K3 release images ship:
#   ROCm/aiter 68e42f5f (0.1.17.dev395, "k3-for-amd"). Both earlier donor images,
#   amdsiloai/vllm:kimi-k3-mi325x-release-v2 and vllm/vllm-openai-rocm:kimi-k3,
#   carry that same layer (digest e2b3951e36ca), with its Kimi-K3 tuned MoE
#   configs, aiter.ops.triton.conv, and flydsl 0.2.4. Building it here replaces
#   copying their site-packages over a separately installed AITER, so the image
#   has one base and no second image to pull.
#
#   Build (context = repo root):
#     docker build -f docker/vllm_kimi_k3.ubuntu.amd.Dockerfile \
#       --build-arg MAD_SYSTEM_GPU_ARCHITECTURE=gfx942 -t <registry>/vllm-kimi-k3:gfx942 .
#
#   WITH_NIXL=1 (default) adds UCX/RIXL/rocSHMEM/DeepEP for the rixl connector;
#   the Kimi recipes run moriio, so =0 is a faster build with the same serving path.
#
# STATUS
#   gfx942 (MI300X): the stack the Kimi-K3 cards run today; single-needle NIAH passed
#   10K-900K on 2 prefill + 2 decode nodes with these pins (PR #193), with AITER
#   supplied by grafting the same 68e42f5f build this file now compiles.
#   gfx950 (MI355X): BRING-UP, NOT VALIDATED. Nothing shows MoRIIO disagg with
#   Kimi-K3 has run on gfx950: the vLLM ref is the fork's gfx942 branch. The same
#   stack has run MoRIIO + MoRI-EP on gfx950 for GLM, which needed EP16
#   startup-deadlock gates and a MoRI combine fix that are NOT in this vLLM ref, and
#   ionic NICs needed a patched MoRI. Expect the first gfx950 runs to find those.
#
# NO RUNTIME PATCHERS
#   All Kimi-K3 MoRIIO connector fixes (4-KV-cache-group block routing, multi-chunk
#   compute-progress prefill gate, KDA gather sync-free) are committed in the vLLM
#   source this builds (VLLM_REF below). Nothing patches site-packages at start.
#
# PINNING
#   Every source is pinned to an immutable commit SHA, not a branch name: these are
#   personal forks whose branches can be force-pushed or deleted, and MAD needs the
#   image to be rebuildable to the same bits a year from now. The human-readable
#   branch each SHA came from is in the comment above it.
# =============================================================================

# Image ARGs consumed by FROM must be declared before the first FROM (buildkit
# global scope); declaring them later scopes them to a single stage and the second
# FROM resolves blank.
#
# Open ROCm vLLM CI base - same one the shared disagg image builds on.
ARG BASE_IMAGE=rocm/vllm-dev:ci_base-0fcd9b99cc9d63202da4c858d8ebc6582c9e2491

FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

# Pin the *build toolchain*, not just the sources.
#
# Every source below is pinned to an immutable commit SHA, which makes the build
# look reproducible -- but pip builds wheels in an isolated environment and
# resolves build dependencies (setuptools, wheel) fresh from PyPI at build time.
# So a newer setuptools published after this file was last exercised can break a
# build whose sources have not moved at all. That is exactly what happened:
# setuptools >= 80 added
#     assert isinstance(self.compiler, CCompiler)
# to distutils' build_ext.build_extension, which MoRI's legacy
# Cython.Distutils.build_ext path violates, failing the amd_mori wheel with
#     AssertionError: run() must precede build_extension()
# while every pinned SHA was still correct.
#
# PIP_CONSTRAINT reaches inside pip's isolated build environments, which a plain
# `pip install setuptools==X` in the image does not. Set globally so later stages
# (AITER, vLLM, router) cannot regress the same way.
ARG SETUPTOOLS_CONSTRAINT="setuptools<80"
RUN printf '%s\n' "${SETUPTOOLS_CONSTRAINT}" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# The GPU this image is built for. No default: an image built for the wrong arch
# fails at runtime, so the build refuses to guess. K3_GFX_ARCH carries it to each
# build step (named so it is not read as a fixed arch by madengine's Dockerfile
# arch check, which parses GFX_COMPILATION_ARCH / PYTORCH_ROCM_ARCH / GPU_ARCHS).
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV K3_GFX_ARCH=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN case "${K3_GFX_ARCH}" in \
      gfx942|gfx950) echo "Kimi-K3 image for ${K3_GFX_ARCH}" ;; \
      *) echo "MAD_SYSTEM_GPU_ARCHITECTURE must be gfx942 or gfx950, got '${K3_GFX_ARCH}'" >&2; exit 1 ;; \
    esac && mkdir -p /app && echo "GPU_ARCH=${K3_GFX_ARCH}" >> /app/versions.txt
ARG MAX_JOBS=32
ARG NVCC_THREADS=8
# UCX/RIXL/rocSHMEM/DeepEP for the rixl connector. On by default so the image serves
# every connector; the Kimi recipes use moriio, and =0 builds faster without it.
ARG WITH_NIXL=1
ARG NIC_COMPILATION_ARCH="cx7"

# -----------------------------------------------------------------------------
# 1. MoRI v1.2.2 (the K3 recipe's pin; the shared disagg image pins v1.2.1).
#    JIT-built, so this swaps the sources the EP kernels compile from at runtime.
#    Do NOT pass USE_IONIC=OFF / USE_BNXT=OFF - disabling NIC backends produced a
#    MoRI that deadlocked at the cross-node EP all-to-all init.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
# tag v1.2.2
ARG MORI_REF=fe12a11a7d6c6acd0771b772366ed9ed5e0d3d44
ENV MORI_GPU_ARCHS=${K3_GFX_ARCH}
# UMBP needs gRPC headers absent from this base and is unrelated to EP dispatch.
ENV BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
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
    mkdir -p /app && echo "MORI_REF=${MORI_REF}" >> /app/versions.txt && \
    rm -rf /tmp/mori-src

# -----------------------------------------------------------------------------
# 2. AITER from source at the Kimi-K3 release commit, and flydsl 0.2.4.
#    68e42f5f carries the Kimi-K3 tuned MoE configs (kimik3_{a8w4,fp4}_tuned_fmoe,
#    without which K3's MoE profiling shape falls back to a heuristic FlyDSL kernel
#    that aborts LLVM), aiter.ops.triton.conv (K3's vision tower), and the #3658
#    top_k_top_p fix DP-EP disagg needs. K3's int4 SiTUv2 path on gfx942
#    (_setup_kernel_k3_situ_gfx942 -> compile_moe_gemm1) needs flydsl >= 0.2.4.
#    Same build as the GLM-5.1 image's AITER step; kernels JIT for the GPU at runtime.
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
# branch k3-for-amd (carlushuang/aiter-k3) = ROCm/aiter 0.1.17.dev395
ARG AITER_REF=68e42f5f461556596ae294200f1a3f13378c8582
RUN rm -rf /tmp/aiter-src && \
    git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
    cd /tmp/aiter-src && git checkout "${AITER_REF}" && \
    git submodule update --init --recursive && \
    (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
    GPU_ARCHS="${K3_GFX_ARCH}" pip install --no-build-isolation --no-deps -v . && \
    pip install --no-deps --force-reinstall "flydsl==0.2.4" && \
    python3 -c "import importlib.metadata as m; v=m.version('amd-aiter'); \
assert '68e42f5f' in v, v; assert m.version('flydsl')=='0.2.4'; \
f=[str(x) for x in m.files('amd-aiter') if str(x).startswith('aiter/ops/triton/conv/')]; \
assert f, 'aiter.ops.triton.conv missing'; print('AITER OK', v, '+ flydsl 0.2.4')" && \
    echo "AITER_REF=${AITER_REF}" >> /app/versions.txt && \
    rm -rf /tmp/aiter-src /opt/vllm_cache/aiter_jit /root/.aiter

# -----------------------------------------------------------------------------
# 3. vLLM: full source compile of the K3 + MoRIIO branch. All K3 connector fixes
#    are committed in this tree - there is no runtime patcher:
#      - 4-KV-cache-group block routing (K3 allocates 3 KDA/mamba groups + 1 MLA;
#        the stock connector hardcoded 2-group indices and sent MLA KV to mamba
#        block ids, so decode read empty blocks and generated without context)
#      - multi-chunk prefill transfer gated on compute progress, not block count
#        (the block-count gate fired after chunk 1 for prompts fitting one padded
#        block, so only max_num_batched_tokens of KV ever crossed)
#      - KDA gather made sync-free (a per-layer per-chunk device->CPU sync that
#        turned >500K-token prefills into an apparent hang)
#    Repo is public; no credentials are needed or accepted here (the upstream
#    recipe took a GH_TOKEN build-arg, which would bake the token into image
#    metadata - removed).
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/raviguptaamd/vllm.git
# branch kimi-k3-wideep-disagg-fullsource-v3 @ 2026-08-11
ARG VLLM_REF=862bfd8ca4db78b9cbcbcf9ec6013638e3ae6543
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${K3_GFX_ARCH} \
    MAX_JOBS=${MAX_JOBS}
# MAX_JOBS/NVCC_THREADS are set INLINE on the pip line: under the legacy builder
# the ENV above does not reach the pip build subprocess, and vLLM's setup.py
# compute_num_jobs then dies on `int("")`.
RUN rm -rf /tmp/vllm-src && \
    git clone "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && git checkout "${VLLM_REF}" && \
    echo "VLLM_REF=${VLLM_REF}" >> /app/versions.txt && \
    pip uninstall -y vllm 2>/dev/null || true && \
    MAX_JOBS="${MAX_JOBS:-32}" NVCC_THREADS="${NVCC_THREADS:-8}" \
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
assert av and "68e42f5f" in av, f"AITER replaced by the vLLM install: {av!r}"
import mori, mori.io, mori.ops
print("Post-vLLM check OK: AITER", av, "+ MoRI importable")
PYEOF

# -----------------------------------------------------------------------------
# 4. vllm-router: DP-rank round-robin + the 2P2D KV-notify fix
#    (remote_dp_rank_override + remote_dp_size). Without the notify fix a 2P2D
#    EP16 run reproducibly wedges on "remote blocks never arrived" deferred-write
#    expiries, because decode's notify targets the wrong DP rank.
#    Same source the shared disagg image uses; pinned to its SHA here.
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
# branch ravgupta/discovery-dp-rank-roundrobin @ 2026-08-24 (the 2P2D KV-notify fix)
ARG ROUTER_REF=82dc9811af17412e6e24b5942a5486bc502df23a
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
    echo "VLLM_ROUTER_REF=${ROUTER_REF}" >> /app/versions.txt && \
    rm -rf /tmp/vllm-router-src

# -----------------------------------------------------------------------------
# 4b. NIXL/RIXL transport (WITH_NIXL=1, the default). Identical to the shared disagg
#     image, with rocSHMEM pinned as it pins it.
# -----------------------------------------------------------------------------
ENV _ROCM_DIR=/opt/rocm \
    _UCX_SOURCE=https://github.com/ROCm/ucx.git \
    _UCX_BRANCH=da3fac2a \
    _UCX_INSTALL_DIR=/usr/local/ucx/ \
    _RIXL_SOURCE=https://github.com/ROCm/RIXL.git \
    _RIXL_BRANCH=f33a5599 \
    _RIXL_INSTALL_DIR=/usr/local/RIXL/install \
    _NIXLBENCH_INSTALL_DIR=/usr/local/RIXL
ARG ROCSHMEM_REF=9c43b23229cf2f835b838f72c0d1af3d4876167c
RUN if [ "${WITH_NIXL}" != "1" ]; then \
      echo "WITH_NIXL=${WITH_NIXL}: skipping UCX/RIXL/rocSHMEM/DeepEP (MoRI-EP only)"; \
    else set -e && \
      echo "WITH_NIXL=1: building UCX + RIXL + rocSHMEM + DeepEP" && \
      apt-get update && apt-get install -y \
        autoconf automake libtool autogen pkg-config m4 gcc make \
        librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
        libibverbs-dev rdma-core strace libgflags-dev \
        libaio-dev liburing-dev libcpprest-dev libgrpc-dev libgrpc++-dev \
        libprotobuf-dev protobuf-compiler-grpc wget && \
      pip install meson==0.64.0 "pybind11[global]" pyyaml && \
      cd /tmp && git clone "${_UCX_SOURCE}" && cd ucx && git checkout "${_UCX_BRANCH}" && \
        ./autogen.sh && mkdir -p build && cd build && \
        ../configure --prefix="${_UCX_INSTALL_DIR}" --with-rocm="${_ROCM_DIR}" \
          --disable-go --disable-java --disable-assertions --enable-mt && \
        make -j && make install && \
      cd /tmp && wget -q https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz && \
        tar -xzf v1.14.0.tar.gz && cd googletest-1.14.0 && mkdir -p build && cd build && \
        cmake -DBUILD_SHARED_LIBS=on .. && make -j && make install && \
      cd /tmp && git clone "${_RIXL_SOURCE}" && cd RIXL && git checkout "${_RIXL_BRANCH}" && \
        meson setup build/ --prefix="${_RIXL_INSTALL_DIR}" -Ducx_path="${_UCX_INSTALL_DIR}" \
          -Ddisable_gds_backend=true -Dcudapath_inc="${_ROCM_DIR}/include" -Dcudapath_lib="${_ROCM_DIR}/lib" && \
        cd build && ninja && ninja install && cd /tmp/RIXL && \
        pip install --config-settings=setup-args="-Dcudapath_inc=${_ROCM_DIR}/include" \
                    --config-settings=setup-args="-Dcudapath_lib=${_ROCM_DIR}/lib" \
                    --config-settings=setup-args="-Ducx_path=${_UCX_INSTALL_DIR}" \
                    --config-settings=setup-args="-Ddisable_gds_backend=true" . && \
      cd /tmp && git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && \
        cd rocm-systems && git sparse-checkout set --cone projects/rocshmem && git checkout "${ROCSHMEM_REF}" && \
        mkdir -p /tmp/rocshmem-build && cd /tmp/rocshmem-build && \
        /tmp/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends \
          -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS="${K3_GFX_ARCH}" && \
      cd /tmp && git clone https://github.com/ROCm/DeepEP.git && cd DeepEP && \
        PYTORCH_ROCM_ARCH="${K3_GFX_ARCH}" CFLAGS="-O3 -fPIC" \
          CXXFLAGS="-O3 -fPIC --offload-arch=${K3_GFX_ARCH}" HIP_CXX_FLAGS="-O3 -fPIC" \
          python3 setup.py --variant rocm --nic "${NIC_COMPILATION_ARCH}" build develop && \
      echo "WITH_NIXL build complete" >> /app/versions.txt && \
      rm -rf /tmp/ucx /tmp/googletest-1.14.0 /tmp/v1.14.0.tar.gz /tmp/rocm-systems /tmp/rocshmem-build; \
    fi
ENV LD_LIBRARY_PATH="/usr/local/ucx/lib:/usr/local/lib:/usr/local/RIXL/install/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/ucx/bin:${PATH}"

# -----------------------------------------------------------------------------
# 5. Cache locations (structural: WHERE the JIT/compile caches live in the image).
#    Mount target for the launcher's persistent host JIT cache.
#
#    This image ships NO runtime recipe / tuning / platform ENV, matching the
#    shared disagg image. Everything run-tunable is applied at launch so the same
#    image serves any cluster without a rebuild:
#      - K3 serving recipe (VLLM_ROCM_USE_AITER_MLA=0, AITER_SITUV2_A8W4,
#        KV_CACHE_MEMORY_BYTES, *_CUDAGRAPH_MODE, *_MORI_BACKEND, ...)
#          -> scripts/vllm_dissag/models.yaml, entry "Kimi-K3"
#      - ROCm-7.2.3 GPU-RDMA platform env + MoRI fabric tuning
#          -> scripts/vllm_dissag/connectors/moriio.env
#    The slurm launcher forwards both via `docker -e` (platform env must reach
#    PID 1 - PyTorch reads alloc-conf at import).
# -----------------------------------------------------------------------------
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

# -----------------------------------------------------------------------------
# 6. CRITICAL: scrub build-time MoRI JIT state. The `import mori` verifications
#    above compile/lock MoRI EP kernels under /root/.mori/jit on the BUILD host,
#    leaving stale .hsaco.lock files. At runtime MoriAll2AllManager finds those
#    locks, waits on a build-in-progress whose owner PID is long gone, and
#    DEADLOCKS at ep:0 init. A clean image ships /root/.mori empty.
# -----------------------------------------------------------------------------
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED: /root/.mori + /tmp/mori_jit_* cleared at build end" >> /app/versions.txt

RUN cat /app/versions.txt 2>/dev/null | tail -20 || true
