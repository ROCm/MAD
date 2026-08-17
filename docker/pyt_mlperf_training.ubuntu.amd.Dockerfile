# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
FROM ${BASE_DOCKER}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Re-declared so the LABEL at the end of the file can still see it.
ARG BASE_DOCKER

ARG MLPERF_WORKSPACE=/workspace
ARG BUILD_GPU_TARGETS=gfx950
ARG MEGATRON_REPO=https://github.com/ROCm/Megatron-LM.git
# NeMo 2.7.3 pins megatron-core 0.15.0rc8 (requirements/manifest.json ->
# NVIDIA/Megatron-LM@bf1a5035). Anything newer breaks the recipe at runtime:
# 0.16 dropped `no_weight_decay_cond` from get_megatron_optimizer() and the
# dist_checkpointing tensorstore strategy NeMo imports.
ARG MEGATRON_COMMIT=core_r0.15.0_rocm
ARG NEMO_REPO=https://github.com/NVIDIA/NeMo.git
ARG NEMO_COMMIT=v2.7.3
ARG NEMO_RUN_REF=v0.10.0
ARG TE_REPO=https://github.com/ROCm/TransformerEngine.git
ARG TE_COMMIT=release_v2.17_rocm
ARG MLPERF_BUILD_PROFILE=full
ARG TE_WHEEL_URL=
ARG TE_ROCM_WHEEL_URL=
# Optional rdma-core from source (63.0 carries the Broadcom Thor2 bnxt_re
# ibv_create_qp EFAULT fix). Empty string keeps the distro rdma-core.
ARG RDMA_CORE_VERSION=63.0

ENV WORKSPACE_DIR=${MLPERF_WORKSPACE}
ENV DEPS_DIR=${MLPERF_WORKSPACE}/deps
ENV RCCL_DEBUG=WARN
ENV PIP_NO_CACHE_DIR=1
ENV ROCM_PATH=/opt/rocm
ENV ROCM_HOME=/opt/rocm
ENV PATH=/opt/rocm/bin:${PATH}

RUN mkdir -p "${WORKSPACE_DIR}" "${DEPS_DIR}"
WORKDIR ${WORKSPACE_DIR}

RUN set -e && \
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true && \
    if [ -d /etc/apt/sources.list.d ]; then \
      sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
      sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
      sed -i -E 's/^Components:.*/Components: main restricted universe multiverse/g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    fi && \
    apt-get -o Acquire::ForceIPv4=true \
            -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 \
            update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git \
      curl \
      wget \
      ca-certificates \
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
      make \
      patch \
      openssh-client \
      jq \
      numactl \
      libnuma-dev \
      rdma-core \
      infiniband-diags \
      ibverbs-utils \
      pciutils \
      iproute2 \
      net-tools \
      && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
      pybind11 \
      ninja \
      packaging \
      pyyaml \
      wandb \
      mlperf-logging

WORKDIR ${DEPS_DIR}

RUN git clone --recursive "${MEGATRON_REPO}" megatron_lm && \
    cd megatron_lm && \
    git checkout "${MEGATRON_COMMIT}" && \
    pip uninstall -y megatron-core || true && \
    pip install -e . && \
    cd megatron/core/datasets && \
    make

ENV PYTHONPATH="${DEPS_DIR}/megatron_lm"

# The ROCm base ships its own torch/torchvision/torchaudio/triton builds. Pin
# each to the version already installed rather than vetoing it: the ROCm torch
# wheel declares an exact triton dependency, so `triton<0` makes the whole
# resolution impossible. The CUDA wheels NeMo would otherwise pull stay vetoed.
RUN python3 - > /tmp/nemo_rocm_constraints.txt <<'PY'
import importlib.metadata as md

for pkg in ("torch", "triton", "torchvision", "torchaudio"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}<0")

print("""lightning<=2.4.0
pytorch-lightning<=2.4.0
torchmetrics>=0.11.0
nvidia-cublas-cu12<0
nvidia-cuda-cupti-cu12<0
nvidia-cuda-nvrtc-cu12<0
nvidia-cuda-runtime-cu12<0
nvidia-cudnn-cu12<0
nvidia-cufft-cu12<0
nvidia-cufile-cu12<0
nvidia-curand-cu12<0
nvidia-cusolver-cu12<0
nvidia-cusparse-cu12<0
nvidia-cusparselt-cu12<0
nvidia-nccl-cu12<0
nvidia-nvjitlink-cu12<0
nvidia-nvshmem-cu12<0
nvidia-nvtx-cu12<0
nvidia-cudnn-cu13<0
nvidia-cusparselt-cu13<0
nvidia-nccl-cu13<0
cuda-toolkit<0
cuda-bindings<0
cuda-python<0
cupy-cuda12x<0
bitsandbytes<0
xformers<0""")
PY

RUN git clone "${NEMO_REPO}" nemo && \
    cd nemo && \
    git checkout "${NEMO_COMMIT}"

RUN cd "${DEPS_DIR}/nemo" && python3 - <<'PY'
import pathlib
import re

nemo_root = pathlib.Path(".")
output_path = pathlib.Path("/tmp/nemo_filtered_requirements.txt")

files = [
    nemo_root / "requirements" / "requirements_lightning.txt",
    nemo_root / "requirements" / "requirements_common.txt",
    nemo_root / "requirements" / "requirements_nlp.txt",
]

extras = [
    "pytorch-lightning<=2.4.0,>2.2.1",
    "peft<=0.18.0",
    "ruamel.yaml",
    "text-unidecode",
    "wget",
]

blocked_exact = {
    "torch",
    "torchaudio",
    "torchvision",
    "triton",
    "transformers",
    "huggingface-hub",
    "numpy",
    "mamba-ssm",
    "accelerated-scan",
    "megatron-core",
    "bitsandbytes",
    "xformers",
    # NeMo's tiktoken==0.7.0 pin is for its own reference stack; a newer one is
    # installed from our own requirements.txt instead.
    "tiktoken",
}

blocked_prefixes = (
    "nvidia-",
    "cuda-",
    "cupy",
)

# nvidia-modelopt is pure Python on top of torch, not a CUDA wheel, and
# nemo.collections.llm.api imports it unconditionally, so the "nvidia-" veto
# above must not swallow it.
allowed_exact = {
    "nvidia-modelopt",
}

requirements = []
seen = set()

def normalized_name(requirement: str) -> str:
    head = requirement.split(";", 1)[0].strip()
    head = re.split(r"[<>=!~\[]", head, 1)[0].strip()
    return head.lower().replace("_", "-")

def add_requirement(requirement: str) -> None:
    requirement = requirement.strip()
    if not requirement or requirement.startswith("#") or requirement.startswith("-r"):
        return
    name = normalized_name(requirement)
    if name in blocked_exact:
        return
    if name not in allowed_exact and any(
        name.startswith(prefix) for prefix in blocked_prefixes
    ):
        return
    if requirement not in seen:
        seen.add(requirement)
        requirements.append(requirement)

for file_path in files:
    for raw_line in file_path.read_text().splitlines():
        line = raw_line.split(" #", 1)[0].strip()
        add_requirement(line)

for requirement in extras:
    add_requirement(requirement)

output_path.write_text("\n".join(requirements) + "\n")
print(output_path.read_text(), end="")
PY

RUN cd "${DEPS_DIR}/nemo" && \
    PIP_CONSTRAINT=/tmp/nemo_rocm_constraints.txt \
    pip install --no-build-isolation -r /tmp/nemo_filtered_requirements.txt && \
    PIP_CONSTRAINT=/tmp/nemo_rocm_constraints.txt \
    pip install --no-build-isolation --no-deps -e ".[nlp]"

RUN pip install "git+https://github.com/NVIDIA/NeMo-Run.git@${NEMO_RUN_REF}"

RUN cat > /workspace/requirements.txt <<'EOF'
git+https://github.com/mlcommons/logging.git@5.0.0-rc2
git+https://github.com/NVIDIA/mlperf-common.git@68cf1d0d5e3de3351e66abb696d0e2d011aabf47
huggingface_hub
transformers~=4.57.0
tiktoken
plotly==6.0.0
nbformat==5.10.4
kaleido==0.2.1
redis==5.2.1
EOF

# Important: keep this after NeMo so these pins are not overwritten.
RUN pip install -r /workspace/requirements.txt

RUN cd "${WORKSPACE_DIR}" && python3 - <<'PY'
import torch
import lightning
import nemo
import nemo_run
import transformers
import sentencepiece
import h5py
import ijson
import wget
import mlperf_logging
from torch.utils.cpp_extension import IS_HIP_EXTENSION

print(f"torch.__version__={torch.__version__}")
print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
print(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}")
print(f"IS_HIP_EXTENSION={IS_HIP_EXTENSION}")
print(f"lightning.__version__={lightning.__version__}")
print(f"nemo.__version__={getattr(nemo, '__version__', 'unknown')}")
print(f"nemo_run.__version__={getattr(nemo_run, '__version__', 'unknown')}")
print(f"transformers.__version__={transformers.__version__}")
print(f"sentencepiece.__version__={sentencepiece.__version__}")
print(f"h5py.__version__={h5py.__version__}")
print(f"ijson.__version__={getattr(ijson, '__version__', 'unknown')}")
print(f"wget.__file__={wget.__file__}")
print(f"mlperf_logging.__file__={mlperf_logging.__file__}")
if not IS_HIP_EXTENSION:
    raise RuntimeError("ROCm PyTorch extension support is unavailable before TransformerEngine build")
PY

# NVTE_CK_JIT=0: the CK-JIT path drives the CK fused-attention build through
# ck_build_interceptor.py, and aiter probes that interceptor with `<compiler> -v`,
# demanding exit 0. ROCm's clang++.cfg includes rocm.cfg, which carries
# `-Wl,--enable-new-dtags`; a linker flag makes a bare `clang++ -v` attempt a link
# and fail with "undefined symbol: main", so the probe aborts the build. With
# CK_JIT off the same kernels are built ahead of time via QoLA.
RUN if [[ -n "${TE_WHEEL_URL}" && -n "${TE_ROCM_WHEEL_URL}" ]]; then \
      pip install "${TE_WHEEL_URL}" "${TE_ROCM_WHEEL_URL}"; \
    elif [[ "${MLPERF_BUILD_PROFILE}" == "full" ]]; then \
      git clone --recursive "${TE_REPO}" TransformerEngine && \
      cd TransformerEngine && \
      git checkout "${TE_COMMIT}" && \
      git submodule update --init --recursive && \
      export PYTHONPATH="${DEPS_DIR}/TransformerEngine/3rdparty/hipify_torch:${PYTHONPATH}" && \
      NVTE_CK_JIT=0 \
      NVTE_FUSED_ATTN_AOTRITON=0 \
      NVTE_USE_ROCM=1 \
      NVTE_ROCM_ARCH="${BUILD_GPU_TARGETS}" \
      NVTE_FRAMEWORK="pytorch" \
      NVTE_USE_HIPBLASLT=1 \
      MAX_JOBS=128 \
      PYTORCH_ROCM_ARCH="${BUILD_GPU_TARGETS}" \
      GPU_ARCHS="${BUILD_GPU_TARGETS}" \
      pip install --no-build-isolation -e .; \
    else \
      echo "Skipping TransformerEngine install for MLPERF_BUILD_PROFILE=${MLPERF_BUILD_PROFILE}"; \
    fi

RUN cd "${WORKSPACE_DIR}" && python3 - <<'PY'
import os
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

print(f"final.torch.__version__={torch.__version__}")
print(f"final.torch.version.hip={getattr(torch.version, 'hip', None)}")
print(f"final.IS_HIP_EXTENSION={IS_HIP_EXTENSION}")

if os.getenv("MLPERF_BUILD_PROFILE", "full") == "full":
    import transformer_engine
    print(f"transformer_engine.__file__={transformer_engine.__file__}")
else:
    print("TransformerEngine validation skipped for smoke profile")
PY

# rdma-core from source, replacing the distro packages, so RCCL can create queue
# pairs on Broadcom Thor2 (bnxt_re) fabrics.
#
# ORDERING IS LOAD-BEARING: the removal via `dpkg -r --force-all` leaves
# still-installed dependents with dangling deps, so every apt operation MUST run
# before it. Keep this stage last and only dpkg + `ninja install` after.
RUN if [[ -n "${RDMA_CORE_VERSION}" ]]; then \
      set -e; \
      apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=5 update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
      echo "RDMA_CORE_VERSION empty — keeping distro rdma-core"; \
    fi

RUN if [[ -n "${RDMA_CORE_VERSION}" ]]; then \
      readelf -d /usr/lib/x86_64-linux-gnu/libibverbs.so.1 | grep SONAME && \
      { ls /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav*.so 2>/dev/null | head || true; }; \
    fi && \
    python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)"

# NeMo's logger dereferences nvidia_resiliency_ext.ptl_resiliency unconditionally
# (nemo/lightning/nemo_logger.py:253), so the package has to be importable even
# though its CUDA-only parts are never used here. --no-deps keeps it from pulling
# the nvidia-* CUDA wheels; the pure-Python callbacks import fine on ROCm.
RUN pip install --no-deps nvidia-resiliency-ext==0.6.0 && \
    python3 -c "import nvidia_resiliency_ext.ptl_resiliency.local_checkpoint_callback as m; print('resiliency-ext ok:', m.LocalCheckpointCallback)"

LABEL mlperf_base="${BASE_DOCKER}"
LABEL build_gpu_targets="${BUILD_GPU_TARGETS}"
LABEL rdma_core_version="${RDMA_CORE_VERSION}"

WORKDIR ${WORKSPACE_DIR}
