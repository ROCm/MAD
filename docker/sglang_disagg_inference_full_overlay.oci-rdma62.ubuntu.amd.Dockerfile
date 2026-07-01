# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
# SGLang Disaggregated P/D — MERGED full overlay (RCCL + MoRI + KV-transfer)
#
# Single-Dockerfile equivalent of the three chained overlays:
#   base sglang -> RCCL overlay (+smifix) -> MoRI overlay -> KV-transfer (RIXL)
# Built in one `docker build` step so madengine's single-dockerfile build path
# produces the whole stack at once (no base_docker chaining between overlays).
#
# Pins (override via --build-arg):
#   RCCL  : ROCm/rocm-systems develop @ RCCL_COMMIT (default 78e8ba0) + smifix
#   MoRI  : ROCm/mori @ MORI_COMMIT (default a14e6992, includes #366)
#   NIXL  : ROCm/RIXL @ RIXL_COMMIT (default f33a5599) + ROCm/ucx @ UCX_COMMIT
#           (the AMD NIXL implementation; exposed to SGLang as `nixl` via alias).
#           NOTE: this is NOT ai-dynamo/nixl. The rocm KV transport for
#           KV_TRANSFER_BACKEND=nixl is ROCm/RIXL; recipe mirrors the proven
#           scripts/kvcache_transfer_bench/Dockerfile.
###############################################################################
ARG BASE_DOCKER=lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x
FROM $BASE_DOCKER

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

###############################################################################
# 1) RCCL overlay — rebuild RCCL from source so the RCCL under test wins over
#    the base image's librccl. (mirrors sglang_disagg_inference_rccl_overlay)
###############################################################################
ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_BRANCH=develop
ARG RCCL_COMMIT=78e8ba0
ARG RCCL_INSTALL_DIR=/opt/rccl
ARG BUILD_GPU_TARGETS=gfx942

ENV RCCL_HOME=${RCCL_INSTALL_DIR}
# Prepend the overlay RCCL so it wins over the base image's librccl.
ENV LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

RUN mkdir -p "${RCCL_INSTALL_DIR}"
WORKDIR /opt

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    apt-get -o Acquire::Retries=5 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake ninja-build pkg-config make patch && \
    rm -rf /var/lib/apt/lists/*

RUN if [[ -n "${RCCL_COMMIT}" ]]; then \
      git clone "${RCCL_REPO}" /tmp/rccl; \
    else \
      git clone --depth 1 --branch "${RCCL_BRANCH}" "${RCCL_REPO}" /tmp/rccl; \
    fi && \
    cd /tmp/rccl && \
    if [[ -n "${RCCL_BRANCH}" ]]; then git checkout "${RCCL_BRANCH}" || true; fi && \
    if [[ -n "${RCCL_COMMIT}" ]]; then git checkout "${RCCL_COMMIT}"; fi && \
    if [[ -d projects/rccl ]]; then \
      cd projects/rccl && git submodule update --init --recursive; \
      echo "/tmp/rccl/projects/rccl" > /tmp/BLD_RCCL_HOME.txt; \
    else \
      git submodule update --init --recursive; \
      echo "/tmp/rccl" > /tmp/BLD_RCCL_HOME.txt; \
    fi

RUN set -e && \
    BLD_RCCL_HOME=$(cat /tmp/BLD_RCCL_HOME.txt) && \
    cd "${BLD_RCCL_HOME}" && \
    ./install.sh --amdgpu_targets="${BUILD_GPU_TARGETS}" --prefix="${RCCL_INSTALL_DIR}"

RUN rm -rf /tmp/rccl /tmp/BLD_RCCL_HOME.txt

# Re-add the rocm_smi dependency the rocm-systems RCCL build drops (smifix).
# Newer rocm-systems RCCL builds librccl WITHOUT a DT_NEEDED on librocm_smi64.so;
# torch's libtorch_hip.so relies on librccl to transitively pull in rocm_smi, so
# without this `import torch` dies with "undefined symbol: rsmi_init".
RUN set -e; \
    command -v patchelf >/dev/null 2>&1 || pip install --no-cache-dir patchelf >/dev/null 2>&1 \
      || { apt-get -o Acquire::Retries=5 update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends patchelf && rm -rf /var/lib/apt/lists/*; }; \
    librccl="$(readlink -f ${RCCL_INSTALL_DIR}/lib/librccl.so)"; \
    smi_soname="$(basename "$(ls /opt/rocm*/lib/librocm_smi64.so.[0-9]* 2>/dev/null | grep -E 'librocm_smi64\.so\.[0-9]+$' | head -1)")"; \
    test -n "${smi_soname}" || { echo "ROCM_SMI_SONAME_NOT_FOUND"; exit 1; }; \
    if readelf -d "${librccl}" 2>/dev/null | grep -q "NEEDED.*${smi_soname}"; then \
      echo "RCCL_SMI_NEEDED_ALREADY_PRESENT ${smi_soname}"; \
    else \
      patchelf --add-needed "${smi_soname}" "${librccl}" && echo "RCCL_SMI_NEEDED_ADDED ${smi_soname} -> ${librccl}"; \
    fi

# Sanity: overlay librccl present AND torch imports with overlay librccl first.
RUN set -e; \
    test -e "${RCCL_INSTALL_DIR}/lib/librccl.so" || { echo "RCCL_OVERLAY_MISSING"; exit 1; }; \
    echo "RCCL_OVERLAY_OK $(ls -l ${RCCL_INSTALL_DIR}/lib/librccl.so*)"; \
    python3 -c "import torch; print('RCCL_OVERLAY_TORCH_OK', torch.__version__)"

RUN pip list 2>/dev/null | grep -iE "sglang|torch" || true

###############################################################################
# 2) MoRI overlay — substitutable MoE EP all-to-all (dispatch/combine, IBGDA).
#    Default ROCm/mori @ a14e6992 includes #366 (internode decode hang fix).
###############################################################################
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_BRANCH=main
ARG MORI_COMMIT=a14e6992ffa95478e83127fe2672afff2840856f
ARG MORI_WHEEL_URL=
ARG MORI_GPU_ARCHS=gfx942
ARG MORI_VERSION=1.2.0
ARG MORI_SRC_DIR=/sgl-workspace/mori

ENV MORI_GPU_ARCHS=${MORI_GPU_ARCHS} \
    MORI_SKIP_PRECOMPILE=1 \
    CMAKE_BUILD_TYPE=Release \
    SETUPTOOLS_SCM_PRETEND_VERSION=${MORI_VERSION}

WORKDIR /sgl-workspace

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    apt-get -o Acquire::Retries=5 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake ninja-build pkg-config make patch && \
    rm -rf /var/lib/apt/lists/*

RUN set -e; \
    if [[ -n "${MORI_WHEEL_URL}" ]]; then \
      echo "[mori-overlay] installing prebuilt wheel: ${MORI_WHEEL_URL}"; \
      pip install --no-cache-dir --force-reinstall "${MORI_WHEEL_URL}"; \
    else \
      echo "[mori-overlay] source build ${MORI_REPO}@${MORI_BRANCH}${MORI_COMMIT:+ (${MORI_COMMIT})} archs=${MORI_GPU_ARCHS}"; \
      rm -rf "${MORI_SRC_DIR}"; \
      if [[ -n "${MORI_COMMIT}" ]]; then \
        git clone "${MORI_REPO}" "${MORI_SRC_DIR}"; \
        cd "${MORI_SRC_DIR}" && git checkout "${MORI_COMMIT}"; \
      else \
        git clone --depth 1 --branch "${MORI_BRANCH}" "${MORI_REPO}" "${MORI_SRC_DIR}"; \
      fi; \
      cd "${MORI_SRC_DIR}" && git submodule update --init --recursive || true; \
      pip install --no-cache-dir --force-reinstall .; \
    fi

# Sanity: MoRI must import and report the overlay version.
RUN python3 -c "import mori, importlib.metadata as m; \
print('MORI_OVERLAY_OK', getattr(mori,'__version__', m.version('amd_mori')))" \
    || { echo 'MORI_OVERLAY_IMPORT_FAILED'; exit 1; }

###############################################################################
# 3) KV-transfer overlay — ROCm UCX + ROCm/RIXL (the AMD "nixl" KV transport).
#    Mirrors the proven recipe in scripts/kvcache_transfer_bench/Dockerfile.
#    SGLang's KV_TRANSFER_BACKEND=nixl imports `nixl`; we alias rixl -> nixl.
###############################################################################
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

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    apt-get -o Acquire::Retries=5 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake ninja-build pkg-config make patch autoconf automake libtool \
      cython3 libaio-dev libibverbs-dev librdmacm-dev libpci-dev \
      libgflags-dev libgoogle-glog-dev && \
    rm -rf /var/lib/apt/lists/*

# meson/ninja/pybind11 build frontends for the RIXL meson build + wheel.
RUN pip install --no-cache-dir -U meson ninja pybind11 pyyaml build wheel

# --- ROCm UCX ---------------------------------------------------------------
RUN set -e; \
    git clone "${UCX_REPO}" "${UCX_HOME}.src" && cd "${UCX_HOME}.src" && \
    git checkout "${UCX_COMMIT}" && \
    ./autogen.sh && mkdir -p build && cd build && \
    ../configure --prefix="${UCX_HOME}" --enable-shared --disable-static \
        --disable-doxygen-doc --enable-optimizations --enable-devel-headers \
        --with-rocm="${ROCM_PATH}" --with-verbs --with-dm --enable-mt && \
    make -j"$(nproc)" && make install && \
    cd "${KV_WORKSPACE}" && rm -rf "${UCX_HOME}.src"

# --- ROCm/RIXL (the AMD NIXL implementation) --------------------------------
RUN set -e; \
    git clone "${RIXL_REPO}" "${RIXL_HOME}" && cd "${RIXL_HOME}" && \
    git checkout "${RIXL_COMMIT}" && (git submodule update --init --recursive || true); \
    meson setup build --prefix="${RIXL_HOME}" -Ducx_path="${UCX_HOME}" -Drocm_path="${ROCM_PATH}" && \
    cd build && ninja && ninja install

# Install the meson-built RIXL python package into site-packages directly.
# The upstream contrib/build-wheel.sh requires uv + py3.12 + auditwheel, which
# the py3.10 sglang base lacks; `ninja install` already produced the
# cpython-310 bindings under the RIXL prefix, so copy them into site-packages
# and alias `nixl` -> `rixl` for SGLang's KV_TRANSFER_BACKEND=nixl import path.
RUN set -e; \
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
    echo "NIXL_ALIAS_WRITTEN ${SP}/nixl_alias.pth"

# Sanity: nixl (== rixl) import must succeed; report version + agent symbol.
RUN set -e; \
    python3 -c "import nixl; print('NIXL_IMPORT_OK', getattr(nixl,'__file__','?'))"; \
    python3 -c "import rixl, importlib.metadata as m; print('RIXL_VERSION', m.version('rixl'))" || true; \
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

# Mooncake KV-transfer backend (KV_TRANSFER_BACKEND=mooncake). Mirrors
# docker/sglang_disagg_inference_kvtransfer_overlay.ubuntu.amd.Dockerfile:62-74.
# Default: pip wheel pin; set MOONCAKE_COMMIT for a source build.
ARG MOONCAKE_VERSION=0.3.6.post1
ARG MOONCAKE_REPO=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_COMMIT=
RUN set -e; \
    if [[ -n "${MOONCAKE_COMMIT}" ]]; then \
      echo "[full-overlay] Mooncake source ${MOONCAKE_REPO}@${MOONCAKE_COMMIT}"; \
      git clone "${MOONCAKE_REPO}" /tmp/mooncake && cd /tmp/mooncake && \
      git checkout "${MOONCAKE_COMMIT}" && (git submodule update --init --recursive || true); \
      pip install --no-cache-dir --force-reinstall . && rm -rf /tmp/mooncake; \
    elif [[ -n "${MOONCAKE_VERSION}" ]]; then \
      echo "[full-overlay] Mooncake pip mooncake-transfer-engine==${MOONCAKE_VERSION}"; \
      pip install --no-cache-dir --force-reinstall "mooncake-transfer-engine==${MOONCAKE_VERSION}"; \
    else \
      echo "[full-overlay] Mooncake: keeping base image version"; \
    fi

# Sanity: mooncake must import alongside nixl/rixl (non-fatal — module path may vary).
RUN python3 -c "import mooncake; print('MOONCAKE_OVERLAY_OK')" \
    || echo "MOONCAKE_IMPORT_DIFFERS (verify sglang mooncake import path at runtime)"

###############################################################################
# 5) OCI cluster workaround: build + install rdma-core v62 from source.
#    The OCI-CX7 host stack needs a newer libibverbs/librdmacm/libmlx5 than the
#    base image ships. Formerly built at job start by the launcher; baked here so
#    the runtime env is not mutated. This variant is OCI-only; the base overlay
#    keeps the image-default rdma-core.
###############################################################################
ARG RDMA_VER=v62.0
RUN set -e; \
    git clone --branch "${RDMA_VER}" --depth 1 https://github.com/linux-rdma/rdma-core.git /tmp/rdma-core && \
    cd /tmp/rdma-core && mkdir -p build && cd build && \
    cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr -DNO_MAN_PAGES=1 .. && \
    ninja && ninja install && ldconfig && \
    rm -rf /tmp/rdma-core

# Sanity: rebuilt libibverbs present and python still imports (linker not corrupted).
RUN set -e; \
    ls -l /usr/lib/libibverbs.so* 2>/dev/null || ls -l /usr/lib/*/libibverbs.so* 2>/dev/null || true; \
    python3 -c "import torch; print(\"OCI_RDMA62_TORCH_OK\", torch.__version__)"
