# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
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
# JAX/MaxText training image with a candidate RCCL built from source.
#
#   base : rocm/jax-training:maxtext-v26.5
#   +    : RCCL built from ${RCCL_REPO}@${RCCL_BRANCH}[/${RCCL_COMMIT}] for
#          ${BUILD_GPU_TARGETS} (gfx942=MI300X/MI325X, gfx950=MI350X/MI355X)
#   +    : candidate librccl installed over EVERY librccl on disk
#   +    : optional rdma-core ${RDMA_CORE_VERSION} from source (Broadcom Thor2
#          bnxt_re EFAULT fix on some fabrics; empty = keep base rdma-core)
#
# This is the JAX/MaxText counterpart of
# docker/primus_megatron_train_rccl_overlay.ubuntu.amd.Dockerfile and follows its
# structure and reasoning; what differs is the base image and how the candidate
# librccl is spliced in. A fix that applies to both belongs in both.
#
# WHY overwrite every librccl on disk rather than LD_LIBRARY_PATH: /opt/rocm/lib
# is NOT in this base's ldconfig cache, so a path override is unreliable, and a
# partially-overridden image runs and reports a number for the wrong library.
# There is no torch/lib to special-case here, so it is one find-based sweep.
#
# The clone keeps .git, so RCCL bakes the real commit into its version banner.
#
# IMAGE LAYOUT: RCCL is compiled in a throwaway `rccl-builder` stage and only the
# installed ${RCCL_INSTALL_DIR} tree is COPYed into the final image, keeping the
# source/build tree and the build toolchain out of the shipped image.
#
ARG BASE_DOCKER=rocm/jax-training:maxtext-v26.5

# ---- shared base: apt mirror hygiene (so both stages resolve the same) ------
FROM ${BASE_DOCKER} AS apt-base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
RUN set -e && \
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true && \
    if [ -d /etc/apt/sources.list.d ]; then \
      sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
      sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
      sed -i -E 's/^Components:.*/Components: main restricted universe multiverse/g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    fi

# ---- Stage 1 (throwaway): build + verify candidate RCCL ---------------------
# Everything heavy (full RCCL clone, submodules, build tree, gcc/cmake/ninja)
# lives here and is discarded; only ${RCCL_INSTALL_DIR} is carried forward.
FROM apt-base AS rccl-builder

ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_BRANCH=develop
ARG RCCL_COMMIT=
ARG RCCL_INSTALL_DIR=/opt/rccl
# gfx942 = MI300X / MI325X ; gfx950 = MI350X / MI355X
ARG BUILD_GPU_TARGETS=gfx950
# Extra cmake options forwarded to RCCL's install.sh, e.g.
# "-DFAULT_INJECTION=OFF". Empty = RCCL's own defaults.
ARG RCCL_CMAKE_OPTIONS=

RUN mkdir -p "${RCCL_INSTALL_DIR}"
WORKDIR /opt

RUN apt-get -o Acquire::ForceIPv4=true \
            -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 \
            update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      cmake \
      ninja-build \
      pkg-config \
      make \
      patch \
      build-essential \
      libnuma-dev \
      libatomic1 \
      patchelf && \
    rm -rf /var/lib/apt/lists/*

# clone the candidate RCCL and record the exact built commit. A pinned SHA is not
# reachable from a shallow clone, so clone fully when RCCL_COMMIT is set.
RUN if [[ -n "${RCCL_COMMIT}" ]]; then \
      git clone "${RCCL_REPO}" /tmp/rccl; \
    else \
      git clone --depth 1 --branch "${RCCL_BRANCH}" "${RCCL_REPO}" /tmp/rccl; \
    fi && \
    cd /tmp/rccl && \
    if [[ -n "${RCCL_COMMIT}" ]]; then git checkout "${RCCL_COMMIT}"; \
    elif [[ -n "${RCCL_BRANCH}" ]]; then git checkout "${RCCL_BRANCH}"; fi && \
    if [[ -d projects/rccl ]]; then \
      cd projects/rccl && git submodule update --init --recursive; \
      echo "/tmp/rccl/projects/rccl" > /tmp/BLD_RCCL_HOME.txt; \
    else \
      git submodule update --init --recursive; \
      echo "/tmp/rccl" > /tmp/BLD_RCCL_HOME.txt; \
    fi && \
    git -C "$(cat /tmp/BLD_RCCL_HOME.txt)" rev-parse HEAD > "${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA" && \
    echo "RCCL_BUILT_SHA=$(cat ${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA)"

RUN set -e && \
    BLD_RCCL_HOME=$(cat /tmp/BLD_RCCL_HOME.txt) && \
    cd "${BLD_RCCL_HOME}" && \
    ./install.sh --amdgpu_targets="${BUILD_GPU_TARGETS}" --prefix="${RCCL_INSTALL_DIR}" \
                 ${RCCL_CMAKE_OPTIONS:+--cmake-options "${RCCL_CMAKE_OPTIONS}"}

# ---- build-verification gate: artifact exists + 0 undefined GDAKI/atomic ----
# Fail first if the built librccl is missing (otherwise `nm` would error, the
# counts would default to 0, and the gate would wrongly pass). The `|| true` on
# the grep counts is only to tolerate "no match" (grep exit 1) under -o pipefail.
#
# The gate also checks that RCCL_CMAKE_OPTIONS was actually honoured when it names
# FAULT_INJECTION, because cmake merely warns about an unrecognised -D cache
# variable: on a commit that does not define it, ON and OFF build the same library
# while the provenance file records different options. A positive-control string is
# required first - a marker count of 0 means "not built in" only if `strings` found
# something else in the same file.
#
# The option is located as a whole token (-DFAULT_INJECTION=v, -DFAULT_INJECTION:BOOL=v,
# or "-D FAULT_INJECTION=v") and its value read with cmake's own truth rules, which are
# case-insensitive and accept far more than ON/1. A substring test would call
# -DFAULT_INJECTION=TRUE a request for OFF and then fail a build that is in fact correct,
# and would also fire on an unrelated option whose name merely contains FAULT_INJECTION.
RUN set -e; \
    LIB="${RCCL_INSTALL_DIR}/lib/librccl.so"; \
    echo "=== librccl.so ==="; ls -laL "${RCCL_INSTALL_DIR}/lib/" | grep -i rccl || true; \
    [ -e "$LIB" ] || { echo "BUILD GATE FAIL: $LIB is missing"; exit 1; }; \
    nd=$(nm -D --undefined-only "$LIB" | grep -cE "[Gg]daki|GDAKI" || true); \
    na=$(nm -D --undefined-only "$LIB" | grep -E "__atomic_" | grep -vc "@LIBATOMIC" || true); \
    echo "gdaki_undefined=${nd:-0} unversioned_atomic_undefined=${na:-0}"; \
    if [ "${nd:-0}" -ne 0 ] || [ "${na:-0}" -ne 0 ]; then echo "BUILD GATE FAIL: undefined symbols present"; exit 1; fi; \
    ctl=$(strings -a "$LIB" | grep -c "ncclCommInitRank" || true); \
    [ "${ctl:-0}" -gt 0 ] || { echo "BUILD GATE FAIL: positive control absent - strings found nothing in $LIB, so any marker count below would be meaningless"; exit 1; }; \
    set -f; fi_seen=0; fi_val=""; prev=""; \
    for tok in ${RCCL_CMAKE_OPTIONS}; do \
      case "${prev}${tok}" in \
        -DFAULT_INJECTION=*|-DFAULT_INJECTION:*=*) fi_seen=1; fi_val="${tok#*=}" ;; \
      esac; \
      if [ "$tok" = "-D" ]; then prev="-D"; else prev=""; fi; \
    done; set +f; \
    if [ "$fi_seen" = 0 ]; then \
      case "${RCCL_CMAKE_OPTIONS}" in *FAULT_INJECTION*) \
        echo "WARNING: RCCL_CMAKE_OPTIONS mentions FAULT_INJECTION but carries no"; \
        echo "         -DFAULT_INJECTION[:TYPE]=<value> token, so the effect check is skipped." ;; \
      esac; \
    fi; \
    if [ "$fi_seen" = 1 ]; then \
      case "${fi_val^^}" in ""|0|OFF|NO|FALSE|N|IGNORE|NOTFOUND|*-NOTFOUND) want=0 ;; *) want=1 ;; esac; \
      got=$(strings -a "$LIB" | grep -c "NET/IB ops-fault" || true); \
      echo "fault_injection value='${fi_val}' requested=${want} marker_count=${got} positive_control=${ctl}"; \
      if { [ "$want" = 1 ] && [ "${got:-0}" -eq 0 ]; } || { [ "$want" = 0 ] && [ "${got:-0}" -ne 0 ]; }; then \
        echo "BUILD GATE FAIL: RCCL_CMAKE_OPTIONS='${RCCL_CMAKE_OPTIONS}' did not take effect."; \
        echo "  cmake only warns about an unknown -D cache variable, so a commit that does not"; \
        echo "  define this option builds fine and yields a library identical to the other arm,"; \
        echo "  while the provenance file claims they differ."; exit 1; fi; \
    fi; \
    echo "GATE PASS (built RCCL @ $(cat ${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA))"

# ---- final image ------------------------------------------------------------
FROM apt-base

ARG BASE_DOCKER
ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_BRANCH=develop
# Re-declared in this stage only so the provenance file can record it: without it a
# pinned-commit build reports RCCL_BRANCH=develop and nothing else, which reads as a
# branch-tip build. RCCL_BUILT_SHA is the authoritative value either way.
ARG RCCL_COMMIT=
ARG RCCL_INSTALL_DIR=/opt/rccl
# gfx942 = MI300X / MI325X ; gfx950 = MI350X / MI355X
ARG BUILD_GPU_TARGETS=gfx950
ARG RCCL_CMAKE_OPTIONS=
# Optional rdma-core from source (e.g. 63.0 for the Broadcom Thor2 bnxt_re
# EFAULT fix). Empty string keeps the base image's rdma-core untouched.
ARG RDMA_CORE_VERSION=

ENV WORKSPACE_DIR=/workspace
ENV RCCL_HOME=${RCCL_INSTALL_DIR}
# Prepend the candidate RCCL dir but KEEP the base image's paths (ROCm libs).
# The candidate librccl is also installed over every on-disk copy + ldconfig, so
# this is belt-and-suspenders rather than the sole resolution mechanism.
ENV LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

RUN mkdir -p "${WORKSPACE_DIR}"
WORKDIR ${WORKSPACE_DIR}

# Bring in ONLY the installed candidate RCCL tree (no source/build artifacts).
COPY --from=rccl-builder ${RCCL_INSTALL_DIR} ${RCCL_INSTALL_DIR}

# binutils for the final verification (nm / strings / readelf); libatomic1 is a
# runtime dep of the candidate librccl.
RUN apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=5 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      binutils libatomic1 && \
    rm -rf /var/lib/apt/lists/*

# ---- Stage 2 (optional): rdma-core from source ------------------------------
# Builds only when RDMA_CORE_VERSION is non-empty (e.g. 63.0). Replaces the
# distro rdma-core to pick up the Broadcom Thor2 bnxt_re EFAULT fix; the build
# toolchain it needs is installed inside the script, so the default image
# (RDMA_CORE_VERSION empty) never pays for it.
#
# ORDERING IS LOAD-BEARING: `dpkg -r --force-all` leaves dependents with dangling
# deps, so every apt operation MUST run BEFORE it; only dpkg and `ninja install`
# may follow. Installing over the distro tree instead of replacing it leaves both
# provider sets in place and libibverbs can still resolve the old one.
#
# Inlined, not COPYed: MAD builds model images with ./docker as the context.
RUN if [[ -n "${RDMA_CORE_VERSION}" ]]; then \
      set -e; \
      apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential pkg-config make \
        libnl-3-dev libnl-route-3-dev libudev-dev libssl-dev libsystemd-dev \
        python3-docutils pandoc; \
      rm -rf /var/lib/apt/lists/*; \
      git clone --depth 1 --branch "v${RDMA_CORE_VERSION}" https://github.com/linux-rdma/rdma-core.git /tmp/rdma-core; \
      cd /tmp/rdma-core && git log -1 --format='%H %s'; \
      mkdir build && cd build; \
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
               librdmacm1 librdmacm-dev rdma-core libibumad3 libibmad5 \
               infiniband-diags; do \
        if dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed"; then \
          to_remove="$to_remove $p"; \
        fi; \
      done; \
      if [[ -n "$to_remove" ]]; then \
        echo "removing distro rdma packages:$to_remove"; \
        dpkg -r --force-all $to_remove; \
      fi; \
      ninja install && ldconfig; \
      rm -f /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav*.so /usr/local/lib/libbnxt_re-rdmav*.so; \
      cd / && rm -rf /tmp/rdma-core; \
      shopt -s nullglob; \
      prov=(/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav*.so); \
      shopt -u nullglob; \
      echo "bnxt_re providers: ${#prov[@]}"; \
      [[ ${#prov[@]} -gt 0 ]] && ls -l "${prov[@]}"; \
      if [[ ${#prov[@]} -eq 0 ]]; then \
        echo "GATE FAIL: no bnxt_re provider after rdma-core install"; exit 1; \
      fi; \
    else \
      echo "RDMA_CORE_VERSION empty -- keeping the base image's rdma-core"; \
    fi

# ---- Stage 3: install candidate librccl over EVERY librccl on disk ----------
# -type f, so the .so/.so.1 symlinks keep pointing at what is replaced. An empty
# target list fails the build: such an image would run against the base library.
RUN set -e; \
    SRC="$(ls -L ${RCCL_INSTALL_DIR}/lib/librccl.so.1.0 2>/dev/null || ls -L ${RCCL_INSTALL_DIR}/lib/librccl.so)"; \
    [ -n "$SRC" ] || { echo "GATE FAIL: no librccl in ${RCCL_INSTALL_DIR}"; exit 1; }; \
    echo "candidate librccl: $SRC"; \
    canon_src="$(readlink -f "$SRC")"; \
    find / -xdev -type f -name 'librccl.so*' -not -path "${RCCL_INSTALL_DIR}/*" \
      2>/dev/null > /opt/RCCL_TARGETS.txt || true; \
    [ -s /opt/RCCL_TARGETS.txt ] || { echo "GATE FAIL: no librccl found in base image"; exit 1; }; \
    while read -r t; do \
      [ "$(readlink -f "$t")" = "$canon_src" ] && continue; \
      echo "  overwrite: $t"; cp -fL --remove-destination "$SRC" "$t"; \
    done < /opt/RCCL_TARGETS.txt; \
    ldconfig || true

# ---- provenance: recorded INSIDE the image ---------------------------------
# The build log is not carried by the image. RCCL_CMAKE_OPTIONS is recorded too:
# two images can share a commit and a version banner and still differ.
RUN cp "${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA" /opt/RCCL_BUILT_SHA && \
    pip3 list > /opt/RCCL_IMAGE_PIP_LIST.txt && \
    { echo "RCCL_BUILT_SHA=$(cat /opt/RCCL_BUILT_SHA)"; \
      echo "RCCL_REPO=${RCCL_REPO}"; \
      echo "RCCL_BRANCH=${RCCL_BRANCH}"; \
      echo "RCCL_COMMIT=${RCCL_COMMIT}"; \
      echo "RCCL_CMAKE_OPTIONS=${RCCL_CMAKE_OPTIONS}"; \
      echo "BUILD_GPU_TARGETS=${BUILD_GPU_TARGETS}"; \
      echo "BASE_DOCKER=${BASE_DOCKER}"; \
      echo "RDMA_CORE_VERSION=${RDMA_CORE_VERSION}"; \
      echo "LIBRCCL_TARGETS:"; cat /opt/RCCL_TARGETS.txt; \
    } > /opt/RCCL_IMAGE_PROVENANCE.txt && \
    cat /opt/RCCL_IMAGE_PROVENANCE.txt
