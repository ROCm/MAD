# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
#
# Primus megatron-lm training image with a candidate RCCL built from source.
#
#   base : rocm/primus:v26.4  (ROCm 7.2.x, torch 2.10, python3.12)
#   +    : RCCL built from ${RCCL_REPO}@${RCCL_BRANCH}[/${RCCL_COMMIT}] for
#          ${BUILD_GPU_TARGETS} (gfx942=MI300X/MI325X, gfx950=MI350X/MI355X)
#   +    : candidate librccl installed over EVERY location torch resolves
#   +    : optional rdma-core ${RDMA_CORE_VERSION} from source (Broadcom Thor2
#          bnxt_re EFAULT fix on some fabrics; empty = keep base rdma-core)
#
# WHY install the candidate librccl into BOTH the system rocm lib dir AND
# torch/lib (not just LD_LIBRARY_PATH=/opt/rccl/lib):
#   the primus v26.x base ships NO bundled librccl under torch/lib and
#   libtorch_hip.so links the SYSTEM /opt/rocm/lib/librccl.so.1 (verified via ldd
#   on v26.3). LD_LIBRARY_PATH alone is not enough, so we overwrite the system
#   librccl and drop a copy into torch/lib (RPATH=$ORIGIN) as belt-and-suspenders.
#   This is also safe on earlier v26.2/v26.3 bases.
#
# The git-clone source keeps .git, so RCCL's git_version.cmake bakes the real
# commit into the binary as a separate rcclGitHash string "<ref>:<shorthash>"
# (e.g. "HEAD:c67fbe4" for a detached checkout) — no tarball/hash-override hack
# required. The final gate confirms one of those baked short hashes is a prefix
# of the recorded RCCL_BUILT_SHA.
#
# IMAGE LAYOUT: RCCL is compiled in a throwaway `rccl-builder` stage and only
# the installed ${RCCL_INSTALL_DIR} tree is COPYed into the final image. This
# keeps the full RCCL source/build tree AND the heavy build toolchain
# (build-essential, cmake, ninja) out of the shipped image.
#
ARG BASE_DOCKER=rocm/primus:v26.4

# ---- shared base: apt mirror hygiene (so both stages resolve the same) ------
FROM ${BASE_DOCKER} AS apt-base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
# Force https mirrors + ensure the universe/multiverse components are enabled so
# build deps resolve consistently in both stages.
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
ARG BUILD_GPU_TARGETS=gfx942

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
      libatomic1 \
      patchelf && \
    rm -rf /var/lib/apt/lists/*

# clone the candidate RCCL and record the exact built commit
RUN if [[ -n "${RCCL_COMMIT}" ]]; then \
      git clone "${RCCL_REPO}" /tmp/rccl; \
    else \
      git clone --depth 1 --branch "${RCCL_BRANCH}" "${RCCL_REPO}" /tmp/rccl; \
    fi && \
    cd /tmp/rccl && \
    if [[ -n "${RCCL_BRANCH}" ]]; then git checkout "${RCCL_BRANCH}"; fi && \
    if [[ -n "${RCCL_COMMIT}" ]]; then git checkout "${RCCL_COMMIT}"; fi && \
    if [[ -d projects/rccl ]]; then \
      cd projects/rccl && git submodule update --init --recursive; \
      echo "/tmp/rccl/projects/rccl" > /tmp/BLD_RCCL_HOME.txt; \
    else \
      git submodule update --init --recursive; \
      echo "/tmp/rccl" > /tmp/BLD_RCCL_HOME.txt; \
    fi && \
    git -C "$(cat /tmp/BLD_RCCL_HOME.txt)" rev-parse HEAD > "${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA" && \
    echo "RCCL_BUILT_SHA=$(cat ${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA)"

# v26.4 compat: ROCm ships as pip wheels and the amdclang++ wrapper in
# _rocm_sdk_devel/bin/ resolves its clang++/clang-23 helpers relative to its own
# directory -- but those binaries live ONLY in _rocm_sdk_devel/lib/llvm/bin/.
# RCCL's device-code compile invokes bin/amdclang++ directly and dies with
# "amdclang++: binary '.../_rocm_sdk_devel/bin/clang++' does not exist". Symlink
# the helpers into bin/ so the wrapper resolves. No-op on v26.3 / non-wheel bases.
# (Fixing CC/CXX is NOT enough: --offload-device-only still calls bin/amdclang++.)
RUN SDK="$(ls -d /opt/venv/lib/python*/site-packages/_rocm_sdk_devel 2>/dev/null || true)" && \
    if [ -n "$SDK" ] && [ ! -e "$SDK/bin/clang++" ] && [ -d "$SDK/lib/llvm/bin" ]; then \
      for b in clang clang++ clang-23; do ln -sf ../lib/llvm/bin/$b "$SDK/bin/$b"; done; \
      echo "[v26.4-fix] symlinked clang/clang++/clang-23 into $SDK/bin"; \
    fi

RUN set -e && \
    BLD_RCCL_HOME=$(cat /tmp/BLD_RCCL_HOME.txt) && \
    cd "${BLD_RCCL_HOME}" && \
    ./install.sh --amdgpu_targets="${BUILD_GPU_TARGETS}" --prefix="${RCCL_INSTALL_DIR}"

# ---- build-verification gate: artifact exists + 0 undefined GDAKI/atomic ----
# Fail first if the built librccl is missing (otherwise `nm` would error, the
# counts would default to 0, and the gate would wrongly pass). The `|| true` on
# the grep counts is only to tolerate "no match" (grep exit 1) under -o pipefail.
RUN set -e; \
    LIB="${RCCL_INSTALL_DIR}/lib/librccl.so"; \
    echo "=== librccl.so ==="; ls -laL "${RCCL_INSTALL_DIR}/lib/" | grep -i rccl || true; \
    [ -e "$LIB" ] || { echo "BUILD GATE FAIL: $LIB is missing"; exit 1; }; \
    nd=$(nm -D --undefined-only "$LIB" | grep -cE "[Gg]daki|GDAKI" || true); \
    na=$(nm -D --undefined-only "$LIB" | grep -E "__atomic_" | grep -vc "@LIBATOMIC" || true); \
    echo "gdaki_undefined=${nd:-0} unversioned_atomic_undefined=${na:-0}"; \
    if [ "${nd:-0}" -ne 0 ] || [ "${na:-0}" -ne 0 ]; then echo "BUILD GATE FAIL: undefined symbols present"; exit 1; fi; \
    echo "GATE PASS (built RCCL @ $(cat ${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA))"

# ---- final image ------------------------------------------------------------
FROM apt-base

ARG BASE_DOCKER
ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_BRANCH=develop
ARG RCCL_INSTALL_DIR=/opt/rccl
# gfx942 = MI300X / MI325X ; gfx950 = MI350X / MI355X
ARG BUILD_GPU_TARGETS=gfx942
# Optional rdma-core from source (e.g. 63.0 for the Broadcom Thor2 bnxt_re
# EFAULT fix). Empty string keeps the base image's rdma-core untouched.
ARG RDMA_CORE_VERSION=

ENV WORKSPACE_DIR=/workspace
ENV RCCL_HOME=${RCCL_INSTALL_DIR}
# Prepend the candidate RCCL dir but KEEP the base image's paths (ROCm libs).
# The candidate librccl is also installed over the system path + ldconfig, so
# this is belt-and-suspenders rather than the sole resolution mechanism.
ENV LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
ENV NCCL_DEBUG=WARN

RUN mkdir -p "${WORKSPACE_DIR}"
WORKDIR /opt

# Bring in ONLY the installed candidate RCCL tree (no source/build artifacts).
COPY --from=rccl-builder ${RCCL_INSTALL_DIR} ${RCCL_INSTALL_DIR}

# Minimal tools needed to splice the candidate librccl into the image and to run
# the final ELF verification: patchelf (--add-needed), binutils (objdump / nm /
# strings / readelf) and libatomic1 (runtime dep of the candidate librccl).
RUN apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=5 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      patchelf binutils libatomic1 && \
    rm -rf /var/lib/apt/lists/*

# ---- Stage 2: install candidate librccl over EVERY location torch resolves --
# Overwrite the system /opt/rocm librccl AND drop a copy into torch/lib so the
# candidate always wins regardless of how libtorch_hip is linked. Also
# add-needed librocm_smi64 (folded-in rsmifix) so `import torch` resolves
# rsmi_init on bases where the overlay librccl lacks it in NEEDED.
# add_needed() adds librocm_smi64 to NEEDED idempotently (skips if already
# present, so re-running never appends duplicates) and FAILS the build if
# patchelf cannot apply it -- `import torch` depends on this resolving rsmi_init.
RUN set -e && \
    add_needed() { \
      local lib="$1"; \
      if objdump -p "$lib" | grep -qE "NEEDED[[:space:]]+librocm_smi64"; then \
        echo "  add_needed: already present in $lib"; return 0; \
      fi; \
      patchelf --add-needed librocm_smi64.so.1 "$lib"; \
      objdump -p "$lib" | grep -qE "NEEDED[[:space:]]+librocm_smi64" || { echo "PATCHELF FAILED on $lib"; exit 1; }; \
      echo "  add_needed: applied to $lib"; \
    }; \
    src=$(ls -L "${RCCL_INSTALL_DIR}/lib/librccl.so.1.0" 2>/dev/null || ls -L "${RCCL_INSTALL_DIR}/lib/librccl.so") && \
    echo "candidate librccl: $src" && \
    add_needed "$src" && \
    if [ -e "/opt/rocm/lib/librccl.so.1" ]; then \
      ROCM_LIB=$(dirname "$(readlink -f /opt/rocm/lib/librccl.so.1)"); \
      echo "system rocm lib dir: $ROCM_LIB"; \
      ls -la "$ROCM_LIB"/librccl.so* 2>/dev/null || true; \
      for f in "$ROCM_LIB"/librccl.so*; do \
        [ -f "$f" ] && [ ! -L "$f" ] && { echo "overwriting real file: $f"; cp -fL "$src" "$f"; add_needed "$f"; }; \
      done; \
      ldconfig; \
      echo "after:"; ls -laL "$ROCM_LIB"/librccl.so.1 2>/dev/null || true; \
    else \
      echo "[rccl-overlay] /opt/rocm/lib/librccl.so.1 not present -- targeted /opt/rocm/lib overwrite skipped (global sweep covers it)"; \
    fi && \
    TORCH_LIB=$(ls -d /opt/venv/lib/python*/site-packages/torch/lib 2>/dev/null | head -1) && \
    [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ] && echo "torch lib dir: $TORCH_LIB" && \
    for f in "$TORCH_LIB"/librccl.so*; do \
      if [ -f "$f" ] && [ ! -L "$f" ]; then cp -fL "$src" "$f"; add_needed "$f"; fi; \
    done && \
    cp -fL "$src" "$TORCH_LIB/librccl.so.1.0" && \
    ln -sf librccl.so.1.0 "$TORCH_LIB/librccl.so.1" && \
    ln -sf librccl.so.1   "$TORCH_LIB/librccl.so" && \
    add_needed "$TORCH_LIB/librccl.so.1.0" && \
    if [ ! -e "$TORCH_LIB/librocm_smi64.so" ]; then \
      smi_src=$(ls /opt/rocm/lib/librocm_smi64.so.1 2>/dev/null || \
                ls /opt/venv/lib/python*/site-packages/_rocm_sdk_libraries/lib/librocm_smi64.so.1 2>/dev/null || \
                find /opt -name 'librocm_smi64.so.1' -not -type l 2>/dev/null | head -1 || true); \
      if [ -n "$smi_src" ]; then \
        cp -v -L "$smi_src" "$TORCH_LIB/librocm_smi64.so"; \
        ln -sfv librocm_smi64.so "$TORCH_LIB/librocm_smi64.so.1"; \
        echo "librocm_smi64 copied from: $smi_src"; \
      else \
        echo "[rccl-overlay] librocm_smi64.so.1 not found -- skipping copy (add-needed already applied to candidate)"; \
      fi; \
    fi && \
    echo "=== global sweep: overwrite EVERY real librccl on disk with candidate ===" && \
    canon_src=$(readlink -f "$src") && \
    for f in $(find / -name 'librccl.so*' -not -type l 2>/dev/null); do \
      [ "$(readlink -f "$f")" = "$canon_src" ] && continue; \
      echo "  overwrite: $f"; cp -fL "$src" "$f"; add_needed "$f"; \
    done && \
    ldconfig
# WHY the global sweep: v26.4 splits ROCm libs into _rocm_sdk_libraries/lib
# (runtime .so) and _rocm_sdk_devel/lib (dev). torch maps librccl.so.1 from
# _rocm_sdk_libraries at RUNTIME (soname wins once loaded), which the targeted
# /opt/rocm + torch/lib overwrites above do NOT cover -- so the candidate is
# built and shipped but the STOCK base librccl is what actually runs. Overwriting
# every non-symlink librccl on disk makes whichever copy the loader maps the
# candidate, independent of ROCm's per-version layout churn.

# ---- Stage 3 (optional): rdma-core from source ------------------------------
# Builds only when RDMA_CORE_VERSION is non-empty (e.g. 63.0). Replaces the
# distro rdma-core to pick up the Broadcom Thor2 bnxt_re EFAULT fix. The build
# toolchain it needs is installed here (not in the base image) so the default
# image (RDMA_CORE_VERSION empty) never pays for it.
#
# ORDERING IS LOAD-BEARING: the source rdma-core is installed by REPLACING the
# distro libibverbs/librdmacm packages via `dpkg -r --force-all`. That leaves
# still-installed dependents (libopenmpi3t64, libucx0, rdmacm-utils, ...) with
# dangling deps, so ANY apt command after that point aborts with "Unmet
# dependencies". Therefore every apt operation (install + the docs-tooling
# purge/autoremove) MUST run BEFORE the dpkg removal, while the dpkg state is
# still consistent. Only dpkg + `ninja install` may run afterwards.
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
      cd / && rm -rf /tmp/rdma-core; \
    else \
      echo "RDMA_CORE_VERSION empty — keeping base image rdma-core"; \
    fi

# ---- final verification: import torch + every librccl == candidate ----------
RUN set -e && \
    if [[ -n "${RDMA_CORE_VERSION}" ]]; then \
      echo "=== rdma-core ===" && readelf -d /usr/lib/x86_64-linux-gnu/libibverbs.so.1 | grep SONAME && \
      { ls /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav*.so 2>/dev/null | head || true; }; \
    fi && \
    echo "=== import torch (must succeed) ===" && \
    if [ -x /opt/venv/bin/python3 ]; then PY=/opt/venv/bin/python3; else PY=$(command -v python3); fi && \
    echo "using python: $PY" && \
    "$PY" -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)" && \
    echo "=== ldd libtorch_hip rccl resolution ===" && \
    TL=$(ls -d /opt/venv/lib/python*/site-packages/torch/lib | head -1) && \
    ldd "$TL/libtorch_hip.so" | grep -iE "rccl|smi" && \
    BUILT_SHA=$(cat "${RCCL_INSTALL_DIR}/RCCL_BUILT_SHA") && \
    echo "=== ASSERT: EVERY librccl on disk is the candidate built @ ${BUILT_SHA} ===" && \
    for p in $(find / -name 'librccl.so*' -not -type l 2>/dev/null); do \
      v=$(strings "$p" 2>/dev/null | grep -m1 -oE "RCCL version 2\.[0-9]+\.[0-9]+"); \
      echo "$v" | grep -q "RCCL version 2." || { echo "ASSERT FAIL: $p is not RCCL 2.x ($p -> ${v:-NONE})"; exit 1; }; \
      h=""; \
      for cand in $(strings "$p" 2>/dev/null | grep -oE "[A-Za-z0-9_./-]+:[0-9a-f]{7,40}" | sed 's/.*://' | sort -u); do \
        case "$BUILT_SHA" in "$cand"*) h="$cand"; break;; esac; \
      done; \
      echo "  $p -> ${v:-NONE} / baked=${h:-NONE}"; \
      [ -n "$h" ] || { echo "ASSERT FAIL: $p has no baked git commit matching built SHA $BUILT_SHA (upstream 'Unknown' fallback?)"; exit 1; }; \
    done && \
    echo "=== candidate RCCL commit: ${BUILT_SHA} ==="

# primus_base is derived from BASE_DOCKER so the label cannot drift from the
# actual base when BASE_DOCKER is overridden.
LABEL primus_base="${BASE_DOCKER}"
LABEL rccl_repo="${RCCL_REPO}"
LABEL rccl_branch="${RCCL_BRANCH}"
LABEL build_gpu_targets="${BUILD_GPU_TARGETS}"
LABEL rdma_core_version="${RDMA_CORE_VERSION}"

WORKDIR ${WORKSPACE_DIR}

# Record final Python environment for posterity.
RUN pip3 list
