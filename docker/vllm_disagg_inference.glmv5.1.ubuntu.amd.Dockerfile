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
#   (which stays pinned to the DeepSeek-V3 / R1 stack). This split lets each model
#   pin its own vLLM/AITER/MoRI without disturbing the others -- add a new
#   vllm_disagg_inference.<model>.ubuntu.amd.Dockerfile per future model
#   (e.g. Kimi-2.6) rather than repinning the shared DSV3 image.
#
#   ALL connectors in one image: moriio (TP + MoRI-EP wideEP) + rixl (NIXL TP +
#   DeepEP wideEP). = the fullsource MoRI stack, plus a UCX/RIXL/rocSHMEM/DeepEP
#   transport layer gated by --build-arg WITH_NIXL (default 0 = MoRI-EP only).
#
#   docker build -f docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:glmv5.1 .
#   export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:glmv5.1
#
#   WITH_NIXL=0 (default) => MoRI-EP only (moriio TP/wideEP + deepep-from-base); lean.
#   WITH_NIXL=1 => builds UCX + RIXL(+nixlbench) + rocSHMEM + DeepEP from source,
#     so all four connector combos (moriio TP/wideEP, rixl NIXL TP, DeepEP
#     wideEP) are present (~+30-45 min build vs WITH_NIXL=0).
#
# STATUS (GLM-5.1-FP8 on this stack): NIAH 2k-35k retrieves with no length collapse
# and no crash on 1P/1D EP8 (51/60) and 2P/2D EP16 (55/60) — see the GLM-5.1-FP8
# recipe in scripts/vllm_dissag/models.yaml for the full measurements;
# long-context accuracy fixed via vLLM #47766 (persistent sparse-MLA kept
# ON). 4P/4D EP32 is a KNOWN OPEN DEFECT: token corruption at ALL context lengths
# (garbage output even at 2k), distinct from the long-context bug; prime suspect is
# the moriep all-to-all combine at EP32 scale -> deferred to future work. Use 1P/1D
# and 2P/2D only. (BASE_IMAGE is a gated nightly; override --build-arg BASE_IMAGE=...)
# =============================================================================
# Builds the GLM-5.1 runtime stack by applying component pins ON TOP of a
# purpose-built ROCm/vLLM/MoRI base, cloning each overridden source from public Git
# (no local build-contexts):
#
#   - BASE: rocm/vllm-dev:ci_base-dedbf6be8b1afa17a6220473b9c8c98242ac1c03
#     (ROCm + torch nightly). The stages below OVERRIDE the base's vLLM/MoRI/AITER
#     with the pins we validate for GLM DSA.
#   - MoRI  -> built from ROCm/MoRI @ 42e895472b08 (validated for GLM DSA, BUILD_UMBP=OFF).
#     = ROCm/mori#366 (2026-06-05), 32 commits BEHIND tag v1.2.1 (2026-06-25); see the
#     note at MORI_REF for what that pin does and does not contain. (main LATEST
#     120d2de broke the connector KV-notify handshake, hence a pin, not main.)
#   - AITER -> STOCK ROCm/aiter @ e03fa6040 compiled from source + flydsl 0.1.7-0.1.9;
#     stale JIT wiped. (#47766 keeps persistent MLA ON -> aiter native gqa64 fold.)
#   - vLLM  -> COMPILED from raviguptaamd/vllm @ d723eb305e (tip of
#     glm5.1-dsa-wideEP_on_vllm-v0.27 when validated; GLM DSA + #47766 metadata-key).
#     Full compile: a different commit than the base's, so a .py-only overlay would be
#     ABI-mismatched.
#   - RDMA fix (expandable_segments:False x2 + HSA_ENABLE_IPC_MODE_LEGACY=0) is NOT baked
#     here — it lives in scripts/vllm_dissag/connectors/<connector>.env and the launcher
#     forwards it via docker -e. ROCm 7.2.3 cannot dmabuf-export VMM memory, else MoRI
#     RegisterRdmaMemoryRegion EFAULTs (errno 14) on the first disagg WRITE.
#   - vllm-router (vllm-project/router PR#181 = DP-rank round-robin + 2P2D KV-notify
#     dpfix) built in -> no external router binary needed.
#   - validated recipe knobs baked as ENV. The MoRIIO disagg fixes (#39276 notify,
#     #41751 LL split, DP-rank hash-failsafe) are native in this vLLM (no runtime patcher).
#
# THIS IMAGE IS THE CONTRACT. The MAD side (scripts/vllm_dissag) is catalog/config only
# and ships NO runtime .py patchers: connector_runtime_patch in connectors/moriio.sh is a
# no-op for every model. So EVERY GLM DSA source fix must be carried in-source HERE, by
# VLLM_REF below (d723eb305e) plus the MoRI/AITER pins. Serving
# GLM-5.1-FP8 from an image built off an older vLLM ref is unsupported: it boots, then
# produces garbage output or stalls the disagg KV transfer, with nothing to fall back on.
# That failure is silent, and it has been measured: the pre-v0.27 lab image
# rocm/pytorch-private:glm-dockerimage-built-09072026 scored NIAH 2k 0/10 on the 1P/1D
# EP8 smoke (slurm job 216847). The supported build of this Dockerfile is published as
# rocmshared/pytorch-private:glm5.1-vllm027-b8; scripts/vllm_dissag/models.yaml names the
# same tag and lists the unsupported older images.
# If you need a fix, move the pin and rebuild — do not re-add runtime patchers to MAD.
#
# Build context = repo root:
#   docker build -f docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile -t <registry>/<tag> .
#
# BASE_IMAGE is the purpose-built ROCm/vLLM/MoRI base above. Override --build-arg
# BASE_IMAGE=... to build on a different ROCm base. vLLM compile is long (~30-60 min).
# =============================================================================

ARG BASE_IMAGE=rocm/vllm-dev:ci_base-dedbf6be8b1afa17a6220473b9c8c98242ac1c03
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32
# NIXL/RIXL transport for the rixl connector. GLM-5.1 is served over MoRI-EP + MoRI-IO,
# so the UCX/RIXL/rocSHMEM/DeepEP stack is dead weight here: it lengthens the build and
# ships transports this recipe never selects. Default 0 => lean MoRI-EP-only image, which
# is also exactly how the validated image (glm5.1-vllm027-b8) was built. Set
# --build-arg WITH_NIXL=1 only if you need the rixl connector from this same Dockerfile.
ARG WITH_NIXL=0
ARG NIC_COMPILATION_ARCH="cx7"

# -----------------------------------------------------------------------------
# 1. MoRI: replace the base's bundled MoRI with the commit GLM-5.1 DSA wideEP was
#    validated on, ROCm/MoRI @ 42e895472b08 (= ROCm/mori#366, 2026-06-05). NOTE this
#    is NOT tag v1.2.1 (e31d426a, 2026-06-25) that the base vllm_disagg_inference
#    Dockerfile pins, and it is not the MoRI in the older mori121 lab image — it is
#    32 commits older than v1.2.1. It carries the EP/RDMA correctness fixes this
#    recipe needs plus the ROCm-7.2.3 dmabuf registration path used by the connector
#    .env (expandable_segments:False). MoRI is JIT-built, so this swaps the JIT
#    sources the kernels compile from at runtime.
#    BUILD CONFIG: match the cookbook build — MORI_GPU_ARCHS=gfx942, BUILD_UMBP=OFF,
#    DEFAULT NIC backends. Do NOT pass USE_IONIC=OFF / USE_BNXT=OFF: disabling NIC
#    backends produced a MoRI that deadlocked at the cross-node EP all-to-all init.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
# 42e895472b08: validated MoRI tip for GLM DSA WideEP disagg. The v0.27 base bundles
# amd_mori 1.0.0, but the bundled build regressed GLM DSA (b1: GPU fault on the aiter
# DSA decode kernel), so we build MoRI from source at this pinned commit by DEFAULT
# (WITH_MORI_BUILD=1). Set --build-arg WITH_MORI_BUILD=0 only to fall back to the
# base's bundled mori for debugging.
ARG WITH_MORI_BUILD=1
ARG MORI_REF=42e895472b08
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
# 2. AITER: the v0.27 base bundles amd-aiter 0.1.19 (+ flydsl 0.2.4), but bundled 0.1.19
#    GPU-faults on the GLM DSA decode kernel mla_a8w8_qh64_gqaratio64_v3 (confirmed b1 on
#    this v0.27 base, same regression as the old stack). So we build aiter from source at
#    the validated commit e03fa6040 by DEFAULT (WITH_AITER_BUILD=1). aiter > e03fa6040
#    reintroduces the fault; do not bump without re-running long-ctx NIAH. Set
#    --build-arg WITH_AITER_BUILD=0 only to fall back to the bundled aiter for debugging.
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG WITH_AITER_BUILD=1
ARG AITER_REF=e03fa6040
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
        pip install --no-deps -U "flydsl>=0.1.7,<0.1.9" && \
        echo "AITER_REF=${AITER_REF}@$(git rev-parse HEAD) (stock ROCm/aiter, no fork)" >> /app/versions.txt && \
        rm -rf /tmp/aiter-src && \
        rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache" ; \
    fi

# -----------------------------------------------------------------------------
# 3. vLLM: compile from source at the GLM-5.1 DSA wideEP branch. This is NOT the
#    06_29 Wide-EP WRITE-mode vLLM that the base vllm_disagg_inference Dockerfile and
#    the dist-inf-cookbook mori121 image build from — that vLLM predates the in-source
#    DSA fixes and cannot serve GLM-5.1-FP8 (see models.yaml). Full source compile (the
#    base ships a different commit). The MoRIIO disagg fixes (#39276 notify, #41751 LL
#    split, DP-rank hash-failsafe) AND the GLM DSA fixes are native in this branch, so
#    no runtime patcher is needed — and none exists in MAD, which is why this ref is a
#    hard requirement rather than a preference. Override VLLM_REF to rebuild a
#    different commit; build only committed commits (no working-tree edits).
# -----------------------------------------------------------------------------
# VLLM_REPO/REF are a PUBLIC GitHub repo + commit. Override to your own vLLM fork/ref.
ARG VLLM_REPO=https://github.com/raviguptaamd/vllm.git
# REPRODUCIBILITY: pinned to a SHA, not the branch name, because the branch is mutable
# and has already advanced once since GLM-5.1 was validated (cda3648602 on 2026-08-10 ->
# the MoRI EP sizing commits e8c186f71/d723eb305 on 2026-08-15). A branch name would make
# `docker build` resolve to whatever the tip is on the day you build, which is not what
# "the image is the contract" can mean. The sha below is the tip of
# glm5.1-dsa-wideEP_on_vllm-v0.27 that every number in models.yaml was measured on.
# To move the pin, pass --build-arg VLLM_REF=<sha> and re-validate; /app/versions.txt in
# the built image records the resolved sha either way.
#
# glm5.1-dsa-wideEP_on_vllm-v0.27 = upstream v0.27 tip dedbf6be8b + 9 ROCm commits
# (tip d723eb305 as of 2026-08-18). Core DSA 3: per-req-ctx metadata key (#47766), DSA
# indexer KV transfer (reworked onto upstream's native MoRIIO connector), invalid-token
# sentinel. Plus 4 v0.27 fixes: concat_and_cache_mla positional (stable-ABI),
# splitting_ops out of the compiled graph (MLA "unknown parameter type"),
# sparse-indexer bounds-guard, and the decisive sentinel -1->0 (cda3648602 — aiter
# mla_decode_fwd derefs -1 -> GPU fault at disagg long-ctx). Plus 2 MoRI EP sizing
# commits. NIAH-validated 1P/1D EP8 + 2P/2D EP16, 2k-35k; see the models.yaml recipe
# for the per-role cudagraph modes actually used (decode FULL_AND_PIECEWISE).
ARG VLLM_REF=d723eb305eb78d1bda0ed357b2b54cc29487221f
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
# Verify the aiter install survived the vLLM install (present, not silently downgraded
# to a base-bundled wheel). We pin aiter by commit (e03fa6040), whose reported version
# string varies by build, so assert presence rather than a hardcoded commit substring. Do NOT
# `import aiter` here: it pulls torch->amdsmi->libamd_smi.so, not loadable in the no-GPU
# build sandbox (same reason the Stage-2 verify reads mla.py from disk instead).
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
