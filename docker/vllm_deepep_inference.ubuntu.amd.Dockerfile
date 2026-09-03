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
# vllm_deepep_inference.ubuntu.amd.Dockerfile
#   vLLM expert-parallel serving over DeepEP on MI350X (gfx950), validated
#   with DeepSeek-R1 FP8 at TP=1 / DP=8 / EP=8 on a single node.
#
#   docker build -f docker/vllm_deepep_inference.ubuntu.amd.Dockerfile \
#     --build-arg DEEPEP_REPO=<url> --build-arg DEEPEP_COMMIT=<sha> \
#     --build-arg VLLM_WHEEL=<path/to/vllm-*.whl relative to the build context> \
#     -t <your-registry>/vllm-deepep:local .
#
#   VLLM_WHEEL defaults to vllm.whl in the context root. The wheel must be
#   inside the build context; docker cannot check a COPY source in advance, so
#   a missing file fails at that COPY.
#
#   BASE_IMAGE defaults to the same ROCm vLLM base the sibling disagg image
#   uses. This recipe was validated against a THERock ROCm 7.14 / Torch 2.11
#   base, so override it if your base predates the ROCm version DeepEP needs.
#
#   Stages, each gated so a rebuild only pays for what changed:
#     DEEPEP_REPO / DEEPEP_COMMIT  DeepEP build (EP_TARGET_HIP=1,
#                                  EP_DISABLE_LEGACY=1). Required; the source is
#                                  not vendored here.
#     AITER_COMMIT                 AITER, with the zero-token get_ksplit guard.
#     RDMA_CORE_VERSION            rdma-core from source (default 63.0),
#                                  replacing the distro packages. Empty keeps
#                                  the base image's.
#     ENABLE_TORCHVISION_STUB      Text-only torchvision surface, for ROCm bases
#                                  whose torchvision ABI breaks `import vllm`.
#
#   Runtime notes that are easy to lose:
#     * NCCL_CUMEM_ENABLE=1 is mandatory. Without VMM, RCCL answers
#       "Communicator does not support symmetric memory" and the ElasticBuffer
#       is never created.
#     * GIN is a deliberate choice on a single node. DeepEP moves no payload
#       through it inside one XGMI domain -- is_scaleup_nvlink is true and the
#       barrier and dispatch/combine run over symmetric memory -- but the buffer
#       constructor still asserts on ginType and reserves queue pairs unless GIN
#       is disabled. Keeping it on (NCCL_GIN_TYPE=2, EP_GIN_QUEUE_DEPTH=0)
#       exercises the path that matters for scale-out and requires
#       --device=/dev/infiniband and --ulimit memlock=-1 on the container;
#       EP_DISABLE_GIN=1 avoids both but tests a mode nobody deploys.
#     * Mount the JIT caches. AITER tunes GEMM shapes at run time behind a
#       global build lock, and with data parallelism one rank tunes while the
#       others wait in the collective:
#         -v <host>/aiter_build:/opt/aiter/aiter/jit/build
#         -v <host>/aiter_configs:/tmp/aiter_configs
#         -v <host>/deep_ep:/root/.deep_ep
# =============================================================================

ARG BASE_IMAGE=rocm/vllm-dev:ci_base-0fcd9b99cc9d63202da4c858d8ebc6582c9e2491
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Build args are declared immediately before the stage that consumes them, not
# in one block up top: an ARG in scope becomes part of the cache key of every
# following layer, so hoisting them all would make a change to AITER_COMMIT (or
# the wheel path, or the rdma-core version) rebuild DeepEP and everything after
# it -- defeating the staged-rebuild property this file is organised around.
ARG PYTHON=/opt/venv/bin/python
ARG GPU_TARGETS=gfx950

ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${GPU_TARGETS} \
    PYTHONPATH=${PYTHONPATH}:/opt/rocm/share/amd_smi \
    EP_TARGET_HIP=1 \
    EP_DISABLE_LEGACY=1

###############################################################################
# 1) DeepEP and vLLM.
###############################################################################
ARG DEEPEP_REPO
ARG DEEPEP_COMMIT
RUN test -n "${DEEPEP_REPO}" || { echo "DEEPEP_REPO is required"; exit 1; }; \
    test -n "${DEEPEP_COMMIT}" || { echo "DEEPEP_COMMIT is required"; exit 1; }; \
    set -e; \
    git clone "${DEEPEP_REPO}" /opt/deepep; \
    git -C /opt/deepep checkout -f "${DEEPEP_COMMIT}"; \
    cd /opt/deepep; \
    ${PYTHON} setup.py build; \
    so="$(find build -name '_C.cpython-*.so' | sort | tail -1)"; \
    test -n "${so}"; \
    cp "${so}" deep_ep/; \
    ${PYTHON} -m pip install --no-deps --no-build-isolation .; \
    site_dir="$(${PYTHON} -c "import site; print(site.getsitepackages()[0] + '/deep_ep')")"; \
    mkdir -p "${site_dir}/include"; \
    cp -a deep_ep/include/. "${site_dir}/include/"

ARG VLLM_WHEEL=vllm.whl
COPY ${VLLM_WHEEL} /tmp/vllm.whl
RUN ${PYTHON} -m pip install --no-deps /tmp/vllm.whl && rm -f /tmp/vllm.whl

###############################################################################
# 2) Text-only torchvision surface.
#
# Some ROCm bases ship a torchvision whose ABI does not match the installed
# torch. vLLM imports it transitively through a multimodal config even for
# text-only models, so `import vllm` fails before serving starts.
###############################################################################
ARG ENABLE_TORCHVISION_STUB=1
RUN if [ "${ENABLE_TORCHVISION_STUB}" = "1" ]; then \
      set -e; \
      site_dir="$(${PYTHON} -c "import site; print(site.getsitepackages()[0])")"; \
      ${PYTHON} -m pip uninstall -y torchvision || true; \
      mkdir -p "${site_dir}/torchvision/transforms"; \
      printf '%s\n' \
        '"""Text-only torchvision surface: import-compatible, no operators."""' \
        '__version__ = "0.0.0+text-only"' \
        'def _unsupported(*args, **kwargs):' \
        '    raise NotImplementedError("text-only torchvision stub")' \
        > "${site_dir}/torchvision/__init__.py"; \
      printf '%s\n' \
        'from enum import Enum' \
        'class InterpolationMode(Enum):' \
        '    NEAREST = "nearest"' \
        '    BILINEAR = "bilinear"' \
        '    BICUBIC = "bicubic"' \
        > "${site_dir}/torchvision/transforms/__init__.py"; \
      ${PYTHON} -c "from torchvision.transforms import InterpolationMode; print('TORCHVISION_STUB_OK')"; \
    else \
      echo "TORCHVISION_STUB <disabled>"; \
    fi

###############################################################################
# 3) AITER.
#
# The get_ksplit guard returns 0 when a split has no work instead of dividing
# by zero, which a data-parallel profile microbatch can trigger.
###############################################################################
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=d9e5ef7ce08ee7045d583aed768cff41aa9210fe
RUN set -e; \
    git clone "${AITER_REPO}" /opt/aiter; \
    git -C /opt/aiter checkout -f "${AITER_COMMIT}"; \
    ${PYTHON} - <<'PY'
path = "/opt/aiter/aiter/fused_moe.py"
src = open(path).read()
needle = "    tg_num = tgN * tgM\n"
guard = (
    "    tg_num = tgN * tgM\n"
    "    # A data-parallel profile microbatch can contain no tokens on this\n"
    "    # rank. There is no work to split in that case.\n"
    "    if tg_num == 0:\n"
    "        return 0\n"
)
if "if tg_num == 0:" not in src:
    assert needle in src, "get_ksplit prologue not found"
    open(path, "w").write(src.replace(needle, guard, 1))
print("AITER_KSPLIT_GUARD_OK")
PY
RUN set -e; \
    git -C /opt/aiter submodule update --init --recursive; \
    cd /opt/aiter; \
    ${PYTHON} -m pip install --no-cache-dir -r requirements.txt; \
    AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=${GPU_TARGETS} \
      ${PYTHON} -m pip install --no-cache-dir --no-build-isolation -e .

###############################################################################
# 4) rdma-core from source, replacing the distro packages and dropping the
#    vendor bnxt_re provider.
#
#    Keep this stage last. It installs over the distro rdma-core rather than
#    removing it: removing libibverbs1 and friends would leave every dependent
#    with an unmet dependency and break apt here and in any downstream image.
###############################################################################
ARG RDMA_CORE_VERSION=63.0
RUN if [ -n "${RDMA_CORE_VERSION}" ]; then \
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
      : "Install over the distro packages rather than removing them. dpkg -r"; \
      : "--force-all on libibverbs1 and friends leaves every dependent with an"; \
      : "unmet dependency, so any later apt operation -- including one in a"; \
      : "downstream image built FROM this one -- aborts. The from-source"; \
      : "install targets the same prefix and shadows those files, and the"; \
      : "provider cleanup below removes the part that actually matters, so the"; \
      : "removal bought nothing that is not handled here."; \
      ninja install; \
      : "Drop every bnxt_re provider except the one this build just installed."; \
      : "The base image can ship an out-of-tree vendor provider whose name is"; \
      : "neither predictable nor tied to the rdma-core ABI (observed:"; \
      : "libbnxt_re-235.2.86.0.so, alongside -rdmavNN.so builds). If one"; \
      : "survives, libibverbs loads it against the new rdma-core and every run"; \
      : "warns 'Driver bnxt_re does not support the kernel ABI'. Deleting the"; \
      : "symlink alone is not enough -- ldconfig regenerates SONAME links from"; \
      : "the real file, so the .so itself must go."; \
      : "The keeper comes from cmake's install manifest, i.e. from what THIS"; \
      : "build produced. Globbing the destination cannot distinguish a fresh"; \
      : "provider from a stale one: a base carrying -rdmav60 would win a"; \
      : "lexical sort against the -rdmav59 that v63 installs, and the wrong"; \
      : "file would survive."; \
      keep="$(grep -m1 -E '/libbnxt_re[^/]*\.so$' install_manifest.txt || true)"; \
      test -n "${keep}" || { echo "rdma-core ${RDMA_CORE_VERSION} installed no bnxt_re provider"; exit 1; }; \
      keep_real="$(readlink -f "${keep}")"; \
      roots=""; \
      for d in /usr/lib/x86_64-linux-gnu/libibverbs /usr/local/lib \
               /usr/local/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu; do \
        [ -d "$d" ] && roots="$roots $d"; \
      done; \
      : "roots is filtered first: find exits 1 on a missing root, and this"; \
      : "block runs under set -e with a global pipefail."; \
      if [ -n "$roots" ]; then \
        find $roots -maxdepth 1 -name 'libbnxt_re*.so*' \
          | while read -r f; do \
              [ "$(readlink -f "$f")" = "${keep_real}" ] || rm -f "$f"; \
            done; \
      fi; \
      ldconfig; \
      echo "IBVERBS_BNXT_KEPT ${keep}"; \
      cd / && rm -rf /tmp/rdma-core; \
      echo "IBVERBS_PROVIDERS $(ls /usr/lib/x86_64-linux-gnu/libibverbs/ 2>/dev/null | tr '\n' ' ')"; \
    else \
      echo "RDMA_CORE_FROM_SOURCE <disabled, keeping base rdma-core>"; \
    fi

RUN ${PYTHON} -c "import deep_ep, torch, vllm; assert torch.version.hip; print('VLLM_DEEPEP_BUILD_OK')"

# Clear the base image's entrypoint, as the sibling vLLM images do. madengine
# keeps a container alive with `docker run ... <image> cat` and then drives it
# with `docker exec`; any ENTRYPOINT here turns that keepalive into
# `<entrypoint> cat`, which exits immediately and takes the run with it.
ENTRYPOINT []
