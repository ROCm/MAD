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
# vllm_disagg_inference.glmv5.3_quark_mxfp4_attnfp8.ubuntu.amd.Dockerfile
#   GLM-5.3 MXFP4 (amd/GLM-5.3-Quark-MXFP4-AttnFP8, GlmMoeDsaForCausalLM) P/D image
#   for MI355X (gfx950): MoRI-IO READ KV transfer + MoRI-EP intra-node expert
#   parallel, MTP-3. PER-MODEL image: it carries its own ROCm 10 / vLLM / MoRI / AITER
#   pins, so the shared vllm_disagg_inference Dockerfiles stay untouched.
#
#   docker build -f docker/vllm_disagg_inference.glmv5.3_quark_mxfp4_attnfp8.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:glmv5.3_quark_mxfp4_attnfp8 .
#   (repo root is the build context: the vLLM patches are COPYed from docker/patches/.)
#
# What it reproduces: the engine image used for the llm-d P/D runs of
# amd/GLM-5.3-Quark-MXFP4-AttnFP8 rev bcd1ec31 on MI355X (on the deployment that passed:
# GSM8K 5-shot 91.28%, NIAH 18/18 at 64k/256k/512k, multi-turn needle 7/8 at ~246k and 8/8
# at ~490k; two earlier deployments scored NIAH 17/18 and 16/18). That image
# was built with vLLM's own docker/docker-bake-rocm.hcl (target vllm-openai) at vLLM
# 8673602b with these overrides, then the patches below were overlaid:
#   MORI_BRANCH=a5ac2bc9c7a5  AITER_BRANCH=v0.1.27  AINIC_VERSION=1.117.5-a-196
#   PYTORCH_ROCM_ARCH=AITER_ROCM_ARCH=gfx950
# This file rebuilds the same pins from public sources, as stages on top of the public
# vLLM ROCm runtime image built from byte-identical docker/Dockerfile.rocm{,_base}:
#   - BASE: vllm/vllm-openai-rocm:base-nightly-40941f2c (ROCm SDK 10.0.0 / TheRock,
#     torch 2.13.0+rocm10.0.0, triton 3.8.0+git669b31ac, AITER v0.1.27, MoRI
#     v1.2.3.post1, AINIC 1.117.5-a-77, all-arch). Its Dockerfiles and ROCm
#     requirements equal those at vLLM cd1effa2e7, the parent of the tested commit.
#   - AINIC -> 1.117.5-a-196 userspace (libionic 39.0.26.08.28), matching host FW
#     1.117.5-a-196. Set AINIC_VERSION to the version that matches YOUR host NIC FW.
#   - MoRI  -> ROCm/mori @ a5ac2bc9 (main; carries ROCm/mori#736, the atomics-free
#     InterNode path behind MORI_EP_DISABLE_RDMA_ATOMICS). No MoRI release has it yet.
#   - AITER -> v0.1.27 from the base (built there for gfx942;gfx950). AITER_REBUILD=1
#     rebuilds it for GFX_COMPILATION_ARCH only, as the tested image was.
#   - vLLM  -> public cd1effa2e7 + patches 0000..0007, compiled for gfx950 with the
#     same steps as Dockerfile.rocm (rust frontend, csrc, wheel). Patch 0000 is the
#     tested commit 8673602b itself (not on a public ref), so the patched source tree
#     equals the tested one (8673602b + 0001-0004, 0006, 0007; git tree d1528b9b).
#
# STATUS: built through the vllm-src stage (patch apply) against public cd1effa2e7, which
# yields the tested source tree. The later stages (MoRI, vLLM csrc/rust/wheel, final
# checks) and a GPU run of the result have NOT been done yet; until then the results above
# belong to the bake-built image, not to an image built from here. The compile stages are
# hours of CPU on a cold cache (estimate; AITER_REBUILD=1 adds more).
#
# THE PATCHES ARE THE CONTRACT. Every vLLM fix this recipe needs is in
# docker/patches/vllm_disagg_inference.glmv5.3_quark_mxfp4_attnfp8/ (one file per fix, header says
# what it does, its upstream PR, and when to drop it). They are applied to the source
# before compiling; the build fails if any patch does not apply cleanly or if its body
# differs from SHA256SUMS. The diff hunks are the ones that ran in the tested image
# (0003, 0004 and 0007 had only their commit-message text reworded). The overlay is
# temporary: drop each patch once its upstream fix is in VLLM_REF.
#
# No serving recipe is baked here: block size, KV bytes, MTP, MoRI env (including
# VLLM_MORI_RECV_ROW_ALIGN=4096, required with patch 0004) are run-time settings, see
# scripts/vllm_dissag/models.yaml (GLM-5.3-Quark-MXFP4-AttnFP8) and README.MD.
# =============================================================================

ARG BASE_DOCKER=vllm/vllm-openai-rocm:base-nightly-40941f2c4ba96fba0a7e282a5355e89cd74d9ac5@sha256:3f24c91d967855a6a4f57b9c817d027931e29437d39f6b6de6fd2b374d14b807
# One arch for vLLM csrc, MoRI and (if rebuilt) AITER, as in the tested image. The base
# image's own ENV PYTORCH_ROCM_ARCH lists every arch, so stages pass this explicitly.
ARG GFX_COMPILATION_ARCH="gfx950"
ARG MAX_JOBS=""
ARG AINIC_VERSION=1.117.5-a-196
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=a5ac2bc9c7a5d933c720c83060c4e454adbfce5b
ARG AITER_REBUILD=0
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_BRANCH=v0.1.27
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=cd1effa2e7d204b22527f14739734751f38ed737
# The tested image reported this version. The patched tree built here is identical to
# that commit's tree + the same patches, so the version string is pinned to it.
ARG VLLM_VERSION=0.32.0rc1.dev1492+g8673602b8

FROM ${BASE_DOCKER} AS base
USER root
ENTRYPOINT []
WORKDIR /app

# -----------------------------------------------------------------------------
# 1. MoRI, built like Dockerfile.rocm_base's build_mori stage (UMBP ON). Full clone:
#    the wheel version (1.2.5.dev86+ga5ac2bc9c for the default ref) comes from git.
# -----------------------------------------------------------------------------
FROM base AS mori-build
ARG MORI_REPO
ARG MORI_REF
ARG GFX_COMPILATION_ARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgrpc++-dev libprotobuf-dev \
       protobuf-compiler protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*
RUN git clone "${MORI_REPO}" /app/mori \
    && cd /app/mori \
    && git checkout "${MORI_REF}" \
    && test "$(git rev-parse HEAD)" = "${MORI_REF}" \
    && git submodule update --init --recursive \
    && ln -sfn "${ROCM_PATH}" /opt/rocm \
    && pip install --no-cache-dir "setuptools_scm>=6.2" \
    && PYTORCH_ROCM_ARCH="${GFX_COMPILATION_ARCH}" \
       CMAKE_PREFIX_PATH="${ROCM_PATH}:${ROCM_PATH}/lib/rocm_sysdeps" \
       python3 setup.py bdist_wheel --dist-dir=dist \
    && grep -qx 'BUILD_UMBP:BOOL=ON' build/CMakeCache.txt \
    && mkdir -p /app/install && cp dist/*.whl /app/install/

# -----------------------------------------------------------------------------
# 2. AITER, optional. Default AITER_REBUILD=0 keeps the base's v0.1.27 build.
# -----------------------------------------------------------------------------
FROM base AS aiter-build
ARG AITER_REBUILD
ARG AITER_REPO
ARG AITER_BRANCH
ARG GFX_COMPILATION_ARCH
RUN mkdir -p /app/install \
    && if [ "${AITER_REBUILD}" = "1" ]; then \
         git clone --recursive --branch "${AITER_BRANCH}" "${AITER_REPO}" /app/aiter \
         && cd /app/aiter && git submodule update --init --recursive \
         && pip install --no-cache-dir -r requirements.txt pyyaml \
         && PREBUILD_KERNELS=1 AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS="${GFX_COMPILATION_ARCH}" \
            python3 setup.py bdist_wheel --dist-dir=dist \
         && cp dist/*.whl /app/install/ ; \
       fi

# -----------------------------------------------------------------------------
# 3. Runtime: base + AINIC userspace matching the host NIC FW + MoRI (+ AITER).
#    a-196's libionic is versioned below the base's a-77 one, hence pinned versions
#    and --allow-downgrades.
# -----------------------------------------------------------------------------
FROM base AS runtime
ARG AINIC_VERSION
ARG MORI_REF
ARG GFX_COMPILATION_ARCH
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} jammy main" \
       > /etc/apt/sources.list.d/amdainic.list \
    && apt-get update \
    && v_lib="$(apt-cache madison libionic1 | awk -v r="/${AINIC_VERSION} " 'index($0, r) {print $3; exit}')" \
    && v_common="$(apt-cache madison ionic-common | awk -v r="/${AINIC_VERSION} " 'index($0, r) {print $3; exit}')" \
    && test -n "${v_lib}" && test -n "${v_common}" \
    && apt-get install -y --no-install-recommends --allow-downgrades \
       "libionic1=${v_lib}" "libionic-dev=${v_lib}" "ionic-common=${v_common}" \
    && test "$(dpkg-query -W -f='${Version}' libionic1)" = "${v_lib}" \
    && rm -rf /var/lib/apt/lists/* \
    && echo "MAD AINIC_VERSION=${AINIC_VERSION} libionic1=${v_lib} ionic-common=${v_common}" >> /app/versions.txt
RUN --mount=type=bind,from=mori-build,src=/app/install,target=/i/mori \
    --mount=type=bind,from=aiter-build,src=/app/install,target=/i/aiter \
    set -e; \
    pip uninstall -y amd_mori 2>/dev/null || true; \
    pip install --no-cache-dir -c /etc/rocm-constraints.txt /i/mori/*.whl; \
    if ls /i/aiter/*.whl >/dev/null 2>&1; then \
      pip install --no-cache-dir --no-deps --force-reinstall /i/aiter/*.whl; fi; \
    if pip check 2>&1 | grep -iE '^(torch|triton) '; then echo "ERROR: torch/triton broken"; exit 1; fi; \
    mori_v="$(python3 -c "import importlib.metadata as m; print(m.version('amd_mori'))")"; \
    aiter_v="$(python3 -c "import importlib.metadata as m; print(m.version('amd-aiter'))")"; \
    case "${MORI_REF}" in *[!0-9a-f]*) ;; \
      *) case "${mori_v}" in *"g$(echo "${MORI_REF}" | cut -c1-9)"*) ;; \
           *) echo "ERROR: amd_mori ${mori_v} was not built from ${MORI_REF}"; exit 1 ;; esac ;; esac; \
    python3 -c "from mori.cpp import UMBPClient"; \
    echo "MAD MORI_REF=${MORI_REF} amd_mori=${mori_v}" >> /app/versions.txt; \
    echo "MAD AITER amd-aiter=${aiter_v}" >> /app/versions.txt
ENV PYTORCH_ROCM_ARCH=${GFX_COMPILATION_ARCH} \
    AITER_ROCM_ARCH=${GFX_COMPILATION_ARCH}

# -----------------------------------------------------------------------------
# 4. vLLM source: public commit + patches. Each patch body is checked against
#    SHA256SUMS, then `git apply --check` + apply in filename order; any failure
#    stops the build. Built from base, so the vLLM compile does not wait for MoRI.
# -----------------------------------------------------------------------------
FROM base AS vllm-src
ARG VLLM_REPO
ARG VLLM_REF
COPY docker/patches/vllm_disagg_inference.glmv5.3_quark_mxfp4_attnfp8/ /app/patches/
RUN set -e; \
    git clone --filter=blob:none --no-checkout "${VLLM_REPO}" /app/vllm; \
    git -C /app/vllm checkout -q "${VLLM_REF}"; \
    test "$(git -C /app/vllm rev-parse HEAD)" = "$(git -C /app/vllm rev-parse "${VLLM_REF}^{commit}")"; \
    cd /app/patches; \
    test "$(ls *.patch | wc -l)" = "$(wc -l < SHA256SUMS)" \
      || { echo "ERROR: the .patch files and SHA256SUMS list different patches"; exit 1; }; \
    for p in $(ls *.patch | sort); do \
      sed '1,/^# ---- end of MAD header/d' "$p" > "/tmp/$p"; \
      echo "$(grep -F "  $p" SHA256SUMS | cut -d' ' -f1)  /tmp/$p" | sha256sum -c - \
        || { echo "ERROR: $p body differs from SHA256SUMS"; exit 1; }; \
      git -C /app/vllm apply --check -v "/tmp/$p" \
        || { echo "ERROR: $p does not apply cleanly on ${VLLM_REF} + the patches before it"; exit 1; }; \
      git -C /app/vllm apply "/tmp/$p"; \
      echo "[$p] applied"; \
    done; \
    { echo "MAD VLLM_REF=${VLLM_REF}"; \
      echo "MAD VLLM_PATCHES=$(awk '{printf "%s%s:%s", sep, $2, $1; sep=","}' SHA256SUMS)"; } > /app/vllm-versions.txt

# -----------------------------------------------------------------------------
# 5. vLLM rust frontend (Dockerfile.rocm rust-toolchain + rust-build stages).
# -----------------------------------------------------------------------------
FROM vllm-src AS rust-build
ARG VLLM_VERSION
ENV CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup PATH=/root/.cargo/bin:${PATH}
WORKDIR /app/vllm
RUN TOOLCHAIN="$(grep '^channel' rust-toolchain.toml | sed 's/.*= *"\(.*\)"/\1/')" \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none \
    && rustup toolchain install "${TOOLCHAIN}" \
    && rustup default "${TOOLCHAIN}" \
    && uv pip install --system -r requirements/build/rust.txt \
    && CARGO_BUILD_JOBS="$(nproc)" SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" \
       bash tools/build_rust.sh \
    && test -x vllm/vllm-rs

# -----------------------------------------------------------------------------
# 6. vLLM native extensions (Dockerfile.rocm triton-kernels + csrc stages).
#    requirements/rocm.txt is emptied in this stage only: this wheel carries just .so files.
# -----------------------------------------------------------------------------
FROM vllm-src AS csrc
ARG GFX_COMPILATION_ARCH
ARG MAX_JOBS
ARG ROCM_TRITON_KERNELS_COMMIT=669b31acc1dd1b3fd93286afd5db67f65d9f7557
ADD https://codeload.github.com/ROCm/triton/tar.gz/${ROCM_TRITON_KERNELS_COMMIT} /tmp/rocm-triton.tar.gz
RUN mkdir -p /opt/rocm-triton-kernels \
    && tar --extract --gzip --file /tmp/rocm-triton.tar.gz \
        --directory /opt/rocm-triton-kernels --strip-components=4 --no-same-owner \
        "triton-${ROCM_TRITON_KERNELS_COMMIT}/python/triton_kernels/triton_kernels" \
    && test -f /opt/rocm-triton-kernels/__init__.py \
    && rm /tmp/rocm-triton.tar.gz
ENV TRITON_KERNELS_SRC_DIR=/opt/rocm-triton-kernels ROCM_TRITON_KERNELS_COMMIT=${ROCM_TRITON_KERNELS_COMMIT}
WORKDIR /app/vllm
RUN : > requirements/rocm.txt \
    && PYTORCH_ROCM_ARCH="${GFX_COMPILATION_ARCH}" \
       SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+rocm.csrc.build PYTHONDONTWRITEBYTECODE=1 \
       LDFLAGS="-fuse-ld=mold" MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
       python3 setup.py bdist_wheel --dist-dir=/dist \
    && rm -rf build

# -----------------------------------------------------------------------------
# 7. Full vLLM wheel around the csrc + rust artifacts (Dockerfile.rocm wheel stage).
# -----------------------------------------------------------------------------
FROM vllm-src AS wheel
ARG VLLM_VERSION
COPY --from=csrc /dist /precompiled
COPY --from=rust-build /app/vllm/vllm/vllm-rs /app/vllm/vllm/_rust_*.so /rust/
WORKDIR /app/vllm
RUN cp /rust/* vllm/ \
    && ! grep -n 'git+' requirements/common.txt requirements/rocm.txt \
    && pip list --format=freeze \
        | grep -iE '^(torch|triton|torchvision|torchaudio|amdsmi|flash[-_]attn|amd[-_]aiter)==' >> requirements/rocm.txt \
    && SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" VLLM_USE_PRECOMPILED=1 \
       VLLM_PRECOMPILED_WHEEL_LOCATION="$(ls /precompiled/*.whl)" VLLM_DOCKER_BUILD_CONTEXT=1 \
       PYTHONDONTWRITEBYTECODE=1 python3 setup.py bdist_wheel --dist-dir=/dist

# -----------------------------------------------------------------------------
# 8. Final image.
# -----------------------------------------------------------------------------
FROM runtime AS final
ARG VLLM_REF
ARG VLLM_VERSION
ARG MORI_REF
ARG GFX_COMPILATION_ARCH
RUN --mount=type=bind,from=wheel,src=/dist,target=/dist \
    pip install --no-cache-dir --no-deps /dist/*.whl
COPY --from=vllm-src /app/vllm/benchmarks /app/vllm/benchmarks
COPY --from=vllm-src /app/vllm/examples /app/vllm/examples
COPY --from=vllm-src /app/patches /app/patches

# Offline checks: vLLM and MoRI import, _C carries the arch, and every vllm/ file the
# patches touch is installed byte-identical to the patched source tree.
RUN --network=none --mount=type=bind,from=vllm-src,src=/app/vllm,target=/src \
    --mount=type=bind,from=vllm-src,src=/app/vllm-versions.txt,target=/tmp/vllm-versions.txt \
    set -e; \
    sp="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    python3 -c "import vllm, vllm._C; print('vLLM', vllm.__version__)"; \
    llvm-objdump --offloading "$sp/vllm/_C.abi3.so" | grep -qw "${GFX_COMPILATION_ARCH}"; \
    n=0; \
    for f in $(sed -n 's|^diff --git a/\(vllm/[^ ]*\) b/.*|\1|p' /app/patches/*.patch | sort -u); do \
      cmp "$sp/$f" "/src/$f" || { echo "ERROR: installed $f differs from the patched source"; exit 1; }; \
      n=$((n + 1)); \
    done; \
    echo "patched files verified: $n"; \
    python3 -c "import mori, mori.io, mori.ops; print('MoRI OK')"; \
    cat /tmp/vllm-versions.txt >> /app/versions.txt; \
    echo "MAD VLLM_VERSION=$(python3 -c 'import vllm; print(vllm.__version__)')" >> /app/versions.txt

# The imports above can JIT-compile MoRI EP kernels under /root/.mori and leave .lock
# files owned by a build-time PID. At run time MoRI waits on those locks forever and EP
# init deadlocks, so the image ships /root/.mori empty and kernels compile on first use.
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori

LABEL org.opencontainers.image.version="${VLLM_VERSION}" \
      mad.vllm.ref="${VLLM_REF}" \
      mad.mori.ref="${MORI_REF}" \
      mad.gpu.arch="${GFX_COMPILATION_ARCH}"

RUN pip list 2>/dev/null | grep -iE '^(vllm|torch|triton|amd-aiter|amd[-_]mori) ' ; cat /app/versions.txt
