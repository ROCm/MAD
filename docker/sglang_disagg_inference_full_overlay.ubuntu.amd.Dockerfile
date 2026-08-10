# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
# SGLang Disaggregated P/D — full overlay (rdma-core + RCCL + MoRI + KV-transfer)
#
# Single-Dockerfile equivalent of the chained overlays:
#   base sglang -> RCCL overlay (+smifix) -> MoRI overlay -> KV-transfer (RIXL)
#   -> Mooncake -> rdma-core
# Built in one `docker build` step so madengine's single-dockerfile build path
# produces the whole stack at once (no base_docker chaining between overlays).
#
# The default base is a rocm/sgl-dev tag that ALREADY carries sglang, aiter,
# sgl-kernel, MoRI, RIXL and Mooncake built for the right GPU target, so every
# component stage below defaults to OFF and the image is the base plus rdma-core.
# Each stage is a substitution knob: turn one on to A/B that component against
# what the base ships. Pick the base tag whose suffix matches the GPUs
# (`-mi35x` = gfx950/MI35x, `-mi30x` = gfx942/MI30x) and set BUILD_GPU_TARGETS
# and MORI_GPU_ARCHS to match.
#
# Stages, all gated by build args (override via --build-arg):
#   rdma-core : RDMA_CORE_VERSION (default 63.0, the only stage ON by default)
#               — see the stage comment; this is what makes queue-pair creation
#               work on Broadcom Thor2 (bnxt_re).
#   RCCL      : ENABLE_RCCL_OVERLAY=1 → ROCm/rocm-systems develop @ RCCL_COMMIT
#               (default 78e8ba0) + smifix
#   MoRI      : ENABLE_MORI_OVERLAY=1 → ROCm/mori @ MORI_COMMIT
#               (default a14e6992, includes #366)
#   NIXL      : ENABLE_NIXL_OVERLAY=1 → ROCm/RIXL @ RIXL_COMMIT (default
#               f33a5599) + ROCm/ucx @ UCX_COMMIT (the AMD NIXL implementation;
#               exposed to SGLang as `nixl` via alias).
#               NOTE: this is NOT ai-dynamo/nixl. The rocm KV transport for
#               KV_TRANSFER_BACKEND=nixl is ROCm/RIXL; recipe mirrors the proven
#               scripts/kvcache_transfer_bench/Dockerfile.
#   Mooncake  : ENABLE_MOONCAKE_OVERLAY=1 → kvcache-ai/Mooncake @ MOONCAKE_REF
#               (default v0.3.12.post1, the first release with the cross-host
#               guard)
#
# To reproduce the pre-sgl-dev behaviour (build everything on a plain sglang
# base), pass BASE_DOCKER=lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x
# BUILD_GPU_TARGETS=gfx942 MORI_GPU_ARCHS=gfx942 and set the four ENABLE_* args
# to 1.
###############################################################################
ARG BASE_DOCKER=rocm/sgl-dev:v0.5.16-rocm720-mi35x-20260807
FROM $BASE_DOCKER

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

###############################################################################
# 1) RCCL overlay — rebuild RCCL from source so the RCCL under test wins over
#    the base image's librccl. (mirrors sglang_disagg_inference_rccl_overlay)
#    OFF by default: the sgl-dev base already carries an librccl with the right
#    code objects, and keeping it makes the base the control arm of the A/B.
###############################################################################
ARG ENABLE_RCCL_OVERLAY=0
ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_BRANCH=develop
ARG RCCL_COMMIT=78e8ba0
ARG RCCL_INSTALL_DIR=/opt/rccl
ARG BUILD_GPU_TARGETS=gfx950

ENV RCCL_HOME=${RCCL_INSTALL_DIR}
# Prepend the overlay RCCL so it wins over the base image's librccl. Harmless
# when ENABLE_RCCL_OVERLAY=0: the directory is then empty and the base librccl
# resolves as usual.
ENV LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

RUN mkdir -p "${RCCL_INSTALL_DIR}"
WORKDIR /opt

RUN if [[ "${ENABLE_RCCL_OVERLAY}" == "1" ]]; then \
      set -e; \
      sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true; \
      apt-get -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake ninja-build pkg-config make patch; \
      rm -rf /var/lib/apt/lists/*; \
    else \
      echo "[full-overlay] RCCL overlay disabled — skipping build deps"; \
    fi

RUN if [[ "${ENABLE_RCCL_OVERLAY}" == "1" ]]; then \
      set -e; \
      if [[ -n "${RCCL_COMMIT}" ]]; then \
        git clone "${RCCL_REPO}" /tmp/rccl; \
      else \
        git clone --depth 1 --branch "${RCCL_BRANCH}" "${RCCL_REPO}" /tmp/rccl; \
      fi; \
      cd /tmp/rccl; \
      if [[ -n "${RCCL_BRANCH}" ]]; then git checkout "${RCCL_BRANCH}" || true; fi; \
      if [[ -n "${RCCL_COMMIT}" ]]; then git checkout "${RCCL_COMMIT}"; fi; \
      if [[ -d projects/rccl ]]; then \
        cd projects/rccl && git submodule update --init --recursive; \
        echo "/tmp/rccl/projects/rccl" > /tmp/BLD_RCCL_HOME.txt; \
      else \
        git submodule update --init --recursive; \
        echo "/tmp/rccl" > /tmp/BLD_RCCL_HOME.txt; \
      fi; \
      BLD_RCCL_HOME=$(cat /tmp/BLD_RCCL_HOME.txt); \
      cd "${BLD_RCCL_HOME}"; \
      ./install.sh --amdgpu_targets="${BUILD_GPU_TARGETS}" --prefix="${RCCL_INSTALL_DIR}"; \
      rm -rf /tmp/rccl /tmp/BLD_RCCL_HOME.txt; \
    else \
      echo "[full-overlay] RCCL overlay disabled — keeping the base image librccl"; \
    fi

# Re-add the rocm_smi dependency the rocm-systems RCCL build drops (smifix).
# Newer rocm-systems RCCL builds librccl WITHOUT a DT_NEEDED on librocm_smi64.so;
# torch's libtorch_hip.so relies on librccl to transitively pull in rocm_smi, so
# without this `import torch` dies with "undefined symbol: rsmi_init".
RUN if [[ "${ENABLE_RCCL_OVERLAY}" == "1" ]]; then \
      set -e; \
      command -v patchelf >/dev/null 2>&1 || pip install --no-cache-dir patchelf >/dev/null 2>&1 \
        || { apt-get -o Acquire::Retries=5 update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends patchelf && rm -rf /var/lib/apt/lists/*; }; \
      librccl="$(readlink -f ${RCCL_INSTALL_DIR}/lib/librccl.so)"; \
      smi_soname="$(basename "$(ls /opt/rocm*/lib/librocm_smi64.so.[0-9]* 2>/dev/null | grep -E 'librocm_smi64\.so\.[0-9]+$' | awk 'NR==1')")"; \
      test -n "${smi_soname}" || { echo "ROCM_SMI_SONAME_NOT_FOUND"; exit 1; }; \
      if readelf -d "${librccl}" 2>/dev/null | grep -q "NEEDED.*${smi_soname}"; then \
        echo "RCCL_SMI_NEEDED_ALREADY_PRESENT ${smi_soname}"; \
      else \
        patchelf --add-needed "${smi_soname}" "${librccl}" && echo "RCCL_SMI_NEEDED_ADDED ${smi_soname} -> ${librccl}"; \
      fi; \
    fi

# Sanity: report which librccl the loader resolves and which GPU targets it
# carries, then confirm torch still imports with that librccl first (the exact
# case the smifix regressed). The reporting runs in a subshell with pipefail off
# and can never fail the build: a pipeline whose reader exits early (`awk ...
# exit`, `head -1`) sends SIGPIPE to the writer, which pipefail+`set -e` would
# turn into a lost build.
RUN set -e; \
    if [[ "${ENABLE_RCCL_OVERLAY}" == "1" ]]; then \
      test -e "${RCCL_INSTALL_DIR}/lib/librccl.so" || { echo "RCCL_OVERLAY_MISSING"; exit 1; }; \
      echo "RCCL_OVERLAY_OK $(ls -l ${RCCL_INSTALL_DIR}/lib/librccl.so*)"; \
    fi; \
    ( set +e +o pipefail; \
      resolved="$(ldconfig -p | awk '/librccl\.so\.1/ && !seen++ {print $NF}')"; \
      echo "RCCL_RESOLVED ${resolved:-<none in ldconfig; LD_LIBRARY_PATH decides>}"; \
      for so in "${RCCL_INSTALL_DIR}/lib/librccl.so" /opt/rocm*/lib/librccl.so; do \
        [[ -e "$so" ]] || continue; \
        echo "RCCL_TARGETS $so : $(strings -a "$(readlink -f "$so")" | grep -oE 'gfx9[0-9]+' | sort -u | paste -sd, -)"; \
      done ); \
    python3 -c "import torch; print('RCCL_OVERLAY_TORCH_OK', torch.__version__)"

###############################################################################
# 2) MoRI overlay — substitutable MoE EP all-to-all (dispatch/combine, IBGDA).
#    Default ROCm/mori @ a14e6992 includes #366 (internode decode hang fix).
#    OFF by default: the sgl-dev base already ships MoRI for the right arch.
###############################################################################
ARG ENABLE_MORI_OVERLAY=0
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_BRANCH=main
ARG MORI_COMMIT=a14e6992ffa95478e83127fe2672afff2840856f
ARG MORI_WHEEL_URL=
ARG MORI_GPU_ARCHS=gfx950
ARG MORI_VERSION=1.2.0
ARG MORI_SRC_DIR=/sgl-workspace/mori

ENV MORI_GPU_ARCHS=${MORI_GPU_ARCHS} \
    MORI_SKIP_PRECOMPILE=1 \
    CMAKE_BUILD_TYPE=Release \
    SETUPTOOLS_SCM_PRETEND_VERSION=${MORI_VERSION}

WORKDIR /sgl-workspace

# The build deps are installed here rather than inherited from the RCCL stage,
# because that stage is now optional and this one has to stand on its own.
#
# The source build pins cmake<4.0 and disables pip's build isolation: an
# isolated build env resolves the newest cmake from PyPI, and cmake >= 4.0
# changed the JSON emitted by gtest_discover_tests. MoRI's test binaries cannot
# run on a GPU-less build host, so they produce empty JSON and the build fails
# across all test targets. Keeping the pin also makes the build reproducible
# regardless of what PyPI currently resolves cmake to.
RUN set -e; \
    if [[ "${ENABLE_MORI_OVERLAY}" != "1" ]]; then \
      echo "[full-overlay] MoRI overlay disabled — keeping the image's MoRI"; \
    elif [[ -n "${MORI_WHEEL_URL}" ]]; then \
      echo "[mori-overlay] installing prebuilt wheel: ${MORI_WHEEL_URL}"; \
      pip install --no-cache-dir --force-reinstall "${MORI_WHEEL_URL}"; \
    else \
      echo "[mori-overlay] source build ${MORI_REPO}@${MORI_BRANCH}${MORI_COMMIT:+ (${MORI_COMMIT})} archs=${MORI_GPU_ARCHS}"; \
      apt-get -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake ninja-build pkg-config make patch; \
      rm -rf /var/lib/apt/lists/*; \
      rm -rf "${MORI_SRC_DIR}"; \
      if [[ -n "${MORI_COMMIT}" ]]; then \
        git clone "${MORI_REPO}" "${MORI_SRC_DIR}"; \
        cd "${MORI_SRC_DIR}" && git checkout "${MORI_COMMIT}"; \
      else \
        git clone --depth 1 --branch "${MORI_BRANCH}" "${MORI_REPO}" "${MORI_SRC_DIR}"; \
      fi; \
      cd "${MORI_SRC_DIR}" && git submodule update --init --recursive || true; \
      pip install --no-cache-dir "cmake<4.0" setuptools wheel pybind11 ninja; \
      pip install --no-build-isolation --no-cache-dir --force-reinstall .; \
    fi

# Sanity: MoRI must import and report the version in place, whether it came from
# the overlay or from the base image.
RUN python3 -c "import mori, importlib.metadata as m; \
print('MORI_OK', getattr(mori,'__version__', m.version('amd_mori')))" \
    || { echo 'MORI_IMPORT_FAILED'; exit 1; }

###############################################################################
# 3) KV-transfer overlay — ROCm UCX + ROCm/RIXL (the AMD "nixl" KV transport).
#    Mirrors the proven recipe in scripts/kvcache_transfer_bench/Dockerfile.
#    SGLang's KV_TRANSFER_BACKEND=nixl imports `nixl`; we alias rixl -> nixl.
#    OFF by default: the sgl-dev base already ships a working nixl.
###############################################################################
ARG ENABLE_NIXL_OVERLAY=0
ARG ROCM_PATH=/opt/rocm
ARG KV_WORKSPACE=/sgl-workspace
ARG UCX_REPO=https://github.com/ROCm/ucx.git
ARG UCX_COMMIT=da3fac2a
ARG RIXL_REPO=https://github.com/ROCm/RIXL.git
ARG RIXL_COMMIT=f33a5599

ENV UCX_HOME=${KV_WORKSPACE}/ucx
ENV RIXL_HOME=${KV_WORKSPACE}/rixl
ENV PATH=${UCX_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${RIXL_HOME}/lib/x86_64-linux-gnu:${UCX_HOME}/lib:${LD_LIBRARY_PATH}

WORKDIR ${KV_WORKSPACE}

RUN if [[ "${ENABLE_NIXL_OVERLAY}" == "1" ]]; then \
      set -e; \
      sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true; \
      apt-get -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake ninja-build pkg-config make patch autoconf automake libtool \
        cython3 libaio-dev libibverbs-dev librdmacm-dev libpci-dev \
        libgflags-dev libgoogle-glog-dev; \
      rm -rf /var/lib/apt/lists/*; \
      pip install --no-cache-dir -U meson ninja pybind11 pyyaml build wheel; \
    else \
      echo "[full-overlay] NIXL overlay disabled — skipping build deps"; \
    fi

# --- ROCm UCX ---------------------------------------------------------------
RUN if [[ "${ENABLE_NIXL_OVERLAY}" == "1" ]]; then \
      set -e; \
      git clone "${UCX_REPO}" "${UCX_HOME}.src" && cd "${UCX_HOME}.src"; \
      git checkout "${UCX_COMMIT}"; \
      ./autogen.sh && mkdir -p build && cd build; \
      ../configure --prefix="${UCX_HOME}" --enable-shared --disable-static \
          --disable-doxygen-doc --enable-optimizations --enable-devel-headers \
          --with-rocm="${ROCM_PATH}" --with-verbs --with-dm --enable-mt; \
      make -j"$(nproc)" && make install; \
      cd "${KV_WORKSPACE}" && rm -rf "${UCX_HOME}.src"; \
    else \
      echo "[full-overlay] NIXL overlay disabled — keeping the image's UCX"; \
    fi

# --- ROCm/RIXL (the AMD NIXL implementation) --------------------------------
RUN if [[ "${ENABLE_NIXL_OVERLAY}" == "1" ]]; then \
      set -e; \
      git clone "${RIXL_REPO}" "${RIXL_HOME}" && cd "${RIXL_HOME}"; \
      git checkout "${RIXL_COMMIT}" && (git submodule update --init --recursive || true); \
      meson setup build --prefix="${RIXL_HOME}" -Ducx_path="${UCX_HOME}" -Drocm_path="${ROCM_PATH}"; \
      cd build && ninja && ninja install; \
    else \
      echo "[full-overlay] NIXL overlay disabled — keeping the image's nixl"; \
    fi

# Install the meson-built RIXL python package into site-packages directly.
# The upstream contrib/build-wheel.sh requires uv + py3.12 + auditwheel, which
# the py3.10 sglang base lacks; `ninja install` already produced the
# cpython-310 bindings under the RIXL prefix, so copy them into site-packages
# and alias `nixl` -> `rixl` for SGLang's KV_TRANSFER_BACKEND=nixl import path.
RUN if [[ "${ENABLE_NIXL_OVERLAY}" == "1" ]]; then \
      set -e; \
      echo "=== RIXL python install tree ==="; \
      find "${RIXL_HOME}" -type d \( -name site-packages -o -name dist-packages \) | sed 's/^/PYDIR: /'; \
      pysp="$(find "${RIXL_HOME}" -type d \( -name site-packages -o -name dist-packages \) | head -1)"; \
      test -n "${pysp}" || { echo "RIXL_PY_INSTALL_NOT_FOUND"; find "${RIXL_HOME}" -name '_bindings*.so' -o -name '*.py' | head; exit 1; }; \
      echo "RIXL_PY_SRC=${pysp}"; ls -la "${pysp}"; \
      SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
      copied=0; \
      for d in "${pysp}"/*/; do \
        name="$(basename "$d")"; \
        case "$name" in *dist-info|__pycache__) continue;; esac; \
        rm -rf "${SP}/${name}"; cp -a "$d" "${SP}/${name}"; echo "INSTALLED_PY_PKG ${name} -> ${SP}/${name}"; copied=1; \
      done; \
      test "$copied" = 1 || { echo "NO_RIXL_PKG_COPIED"; exit 1; }; \
      echo "import rixl, sys; sys.modules['nixl'] = rixl" > "${SP}/nixl_alias.pth"; \
      echo "NIXL_ALIAS_WRITTEN ${SP}/nixl_alias.pth"; \
    fi

# Sanity: nixl must import, whether it came from the overlay or the base image,
# because KV_TRANSFER_BACKEND=nixl imports exactly this name. Strict when the
# overlay built it; a report when the base is expected to supply it.
RUN set -e; \
    if [[ "${ENABLE_NIXL_OVERLAY}" == "1" ]]; then \
      python3 -c "import nixl; print('NIXL_IMPORT_OK', getattr(nixl,'__file__','?'))"; \
      python3 -c "import rixl, importlib.metadata as m; print('RIXL_VERSION', m.version('rixl'))" || true; \
    else \
      python3 -c "import nixl; print('NIXL_IMPORT_OK', getattr(nixl,'__file__','?'))" \
        || echo "NIXL_NOT_IN_BASE_IMAGE (set ENABLE_NIXL_OVERLAY=1 to build it)"; \
    fi; \
    python3 -c "from nixl._api import nixl_agent, nixl_agent_config; print('NIXL_AGENT_OK')" \
      || echo "NIXL_AGENT_API_DIFFERS (verify sglang nixl import path at runtime)"

###############################################################################
# 4) Runtime python deps + Mooncake KV-transfer backend.
#    These were formerly built/pip-installed by the launcher at job start
#    (scripts/sglang_disagg/sglang_disagg_mori_io_ep.sh); baked into the image so
#    the launcher no longer mutates the runtime environment.
###############################################################################
# Runtime python deps formerly pip-installed by the launcher.
RUN pip install --no-cache-dir py-spy pyyaml pandas \
 && pip install --no-cache-dir --ignore-installed --force-reinstall flask

# Mooncake KV-transfer backend (KV_TRANSFER_BACKEND=mooncake).
#
# The pin matters. Mooncake before upstream 45b84d3 ("[TE] Fix cross-node RDMA
# KV transfer under rdma+hip multi-protocol on AMD", #2725, 2026-07-03) has no
# MC_DISABLE_HIP and no isHipReachableTarget: a GPU address registered under
# both rdma and hip keeps hip's hardcoded priority for *any* target, so a
# cross-node KV chunk is pushed through the intra-node-only HIP transport and
# every transfer dies in hipIpcOpenMemHandle ("invalid device pointer").
# MC_USE_HIP_IPC=0 does not help — it only swaps IPC handles for fabric memory,
# which is just as intra-node. v0.3.12.post1 is the first release carrying the
# fix; #2725 was itself validated on a bnxt_re fabric.
#
# The guard lives inside `#ifdef ENABLE_MULTI_PROTOCOL`, whose CMake option
# defaults to OFF, so a build without it silently produces a mooncake with the
# same broken routing — which is why this builds from source rather than taking
# the pip wheel, and why it greps the installed package for MC_DISABLE_HIP (a
# getenv() literal inside that block) to prove the guard really landed.
ARG ENABLE_MOONCAKE_OVERLAY=0
ARG MOONCAKE_REPO=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_REF=v0.3.12.post1
# Upstream default is ON, and it decides how GPU buffers become RDMA memory
# regions: ON exports a dmabuf via hsa and calls ibv_reg_dmabuf_mr, OFF calls
# plain ibv_reg_mr on the device pointer. Which one works is a property of the
# verbs provider, so the choice belongs to the deployment.
ARG MOONCAKE_HIP_DMABUF=ON

RUN if [[ "${ENABLE_MOONCAKE_OVERLAY}" == "1" ]]; then \
      set -e; \
      apt-get -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config patchelf; \
      rm -rf /var/lib/apt/lists/*; \
      git clone "${MOONCAKE_REPO}" /tmp/mooncake; \
      cd /tmp/mooncake; \
      git checkout "${MOONCAKE_REF}"; \
      echo "MOONCAKE_BUILD_REF ${MOONCAKE_REF} $(git rev-parse --short HEAD)"; \
      grep -q isHipReachableTarget mooncake-transfer-engine/src/multi_transport.cpp \
        || { echo "MOONCAKE_REF_MISSING_CROSS_HOST_HIP_GUARD"; exit 1; }; \
      bash dependencies.sh -y; \
      cmake -S . -B build -G Ninja -DUSE_HIP=ON -DUSE_CUDA=OFF \
            -DENABLE_MULTI_PROTOCOL=ON -DCMAKE_BUILD_TYPE=Release \
            -DUSE_HIP_DMABUF="${MOONCAKE_HIP_DMABUF}" \
            -DBUILD_UNIT_TESTS=OFF; \
      cmake --build build -j"$(nproc)"; \
      cmake --install build; \
      bash scripts/build_wheel.sh; \
      whl="$(ls mooncake-wheel/dist/mooncake_transfer_engine*.whl | awk 'NR==1')"; \
      echo "MOONCAKE_WHEEL ${whl}"; \
      pip install --no-cache-dir --force-reinstall "${whl}"; \
      pkg="$(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))')"; \
      hit="$(grep -rls MC_DISABLE_HIP "${pkg}" || true)"; \
      echo "MOONCAKE_GUARD_IN ${hit:-<nothing>}"; \
      [[ -n "${hit}" ]] || { echo "MOONCAKE_GUARD_NOT_COMPILED_IN ${pkg}"; exit 1; }; \
      cd /; rm -rf /tmp/mooncake; \
    else \
      echo "[full-overlay] Mooncake overlay disabled — keeping the image's mooncake"; \
    fi

# Report which mooncake is in place and whether it can route a cross-node
# transfer around the HIP transport at all. Informational only: never fail the
# build here (see the RCCL report above for why).
RUN ( set +e +o pipefail; \
      pkg="$(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))' 2>/dev/null)"; \
      echo "MOONCAKE_PACKAGE ${pkg:-<not found>}"; \
      echo "MOONCAKE_CROSS_HOST_GUARD ${pkg:+$(grep -rls MC_DISABLE_HIP "$pkg" | tr '\n' ' ')}" )

###############################################################################
# 5) rdma-core from source — the Broadcom Thor2 (bnxt_re) queue-pair fix.
#
# The bnxt_re provider shipped with rdma-core 39.0/50.0 corrupts the verbs
# command buffer for work issued off worker threads, so the kernel rejects
# ibv_create_qp with EFAULT (errno 14) and the KV-transfer backends lose their
# queue pairs. Every backend hits it, each in its own way: mooncake logged 2604
# creation failures over a 2P2D run, NIXL could not start at all because UCX
# treats one failed interface QP as fatal, and MoRI built queue pairs that
# reached RTS and then moved zero bytes.
#
# Measured on one node, one container, same kernel, 8 threads x 64 queue pairs,
# only the userspace changing: 39.0 creates 33-66 of 512, and 63.0 creates 512
# of 512. With 63.0 all three backends run a full 2P2D DeepSeek-R1 sweep with
# zero queue-pair failures.
#
# ON by default (empty string keeps the base image's rdma-core), because the
# defect is silent: nothing in the logs points at the provider, and each backend
# fails differently enough to look like three unrelated bugs.
#
# Two things happen beyond the plain build. The distro packages are removed, so
# exactly one libibverbs is left on disk instead of two of different vintages.
# And the vendor bnxt_re provider that /etc/ld.so.conf.d puts on the loader path
# is dropped: it advertises kernel uABI 7-8, and a host running the upstream
# driver (uABI 1) has every device rejected with "Driver bnxt_re does not
# support the kernel ABI of 1", after which RCCL finds no IB plugin at all.
#
# ORDERING IS LOAD-BEARING: the source rdma-core is installed by REPLACING the
# distro libibverbs/librdmacm packages via `dpkg -r --force-all`, which leaves
# still-installed dependents (libucx0, openmpi, ...) with dangling deps, so any
# later apt command aborts with "Unmet dependencies". Every apt operation must
# run before the dpkg removal; only dpkg and `ninja install` may follow it. This
# is also why the stage is last.
###############################################################################
ARG RDMA_CORE_VERSION=63.0

RUN if [[ -n "${RDMA_CORE_VERSION}" ]]; then \
      set -e; \
      apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential pkg-config make \
        libnl-3-dev libnl-route-3-dev libudev-dev libssl-dev libsystemd-dev \
        python3-docutils pandoc; \
      git clone --depth 1 --branch "v${RDMA_CORE_VERSION}" \
        https://github.com/linux-rdma/rdma-core.git /tmp/rdma-core; \
      mkdir -p /tmp/rdma-core/build && cd /tmp/rdma-core/build; \
      cmake -GNinja \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
        -DCMAKE_INSTALL_SYSCONFDIR=/etc \
        -DCMAKE_INSTALL_RUNDIR=/run \
        -DCMAKE_BUILD_TYPE=Release \
        -DNO_PYVERBS=1 \
        ..; \
      ninja -j"$(nproc)"; \
      DEBIAN_FRONTEND=noninteractive apt-get purge -y python3-docutils pandoc; \
      DEBIAN_FRONTEND=noninteractive apt-get autoremove -y; \
      rm -rf /var/lib/apt/lists/*; \
      to_remove=""; \
      for p in ibverbs-providers libibverbs1 libibverbs-dev ibverbs-utils \
               librdmacm1 librdmacm-dev rdma-core libibumad3 infiniband-diags; do \
        dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" \
          && to_remove="$to_remove $p"; \
      done; \
      [ -n "$to_remove" ] && dpkg -r --force-all $to_remove; \
      ninja install; \
      rm -f /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so \
            /usr/local/lib/libbnxt_re-rdmav34.so; \
      ldconfig; \
      cd / && rm -rf /tmp/rdma-core; \
      echo "RDMA_CORE_FROM_SOURCE ${RDMA_CORE_VERSION}"; \
    else \
      echo "RDMA_CORE_FROM_SOURCE <disabled, keeping base rdma-core>"; \
    fi

# Sanity: report the providers the loader can actually see, so a failed
# replacement is visible in the build log rather than at the first
# ibv_create_qp, and confirm python still imports (linker not corrupted).
RUN set -e; \
    ( set +e +o pipefail; \
      echo "IBVERBS_PROVIDERS $(ls /usr/lib/x86_64-linux-gnu/libibverbs/ 2>/dev/null | tr '\n' ' ')"; \
      echo "LIBIBVERBS_SO $(readlink -f /usr/lib/x86_64-linux-gnu/libibverbs.so.1 2>/dev/null)" ); \
    python3 -c "import torch; print('RDMA_CORE_TORCH_OK', torch.__version__)"

LABEL rdma_core_version="${RDMA_CORE_VERSION}"
LABEL rccl_overlay="${ENABLE_RCCL_OVERLAY}"
LABEL rccl_commit="${RCCL_COMMIT}"
LABEL mori_overlay="${ENABLE_MORI_OVERLAY}"
LABEL nixl_overlay="${ENABLE_NIXL_OVERLAY}"
LABEL mooncake_overlay="${ENABLE_MOONCAKE_OVERLAY}"
LABEL mooncake_ref="${MOONCAKE_REF}"
LABEL mooncake_hip_dmabuf="${MOONCAKE_HIP_DMABUF}"
LABEL build_gpu_targets="${BUILD_GPU_TARGETS}"
