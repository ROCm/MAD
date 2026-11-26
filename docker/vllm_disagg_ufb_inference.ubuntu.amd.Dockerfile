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
# vllm_disagg_ufb_inference.ubuntu.amd.Dockerfile
#   vLLM disagg (P/D) image on the UFB ROCm 10.2 base. vLLM is rebuilt from
#   source at upstream 37ea94f71 (2026-11-14), with AITER at the version
#   that commit pins. Kernels are built for gfx950 (MI350X/MI355X).
#
#   PINS:
#   - BASE  -> rocm/ufb-private:vllm-0.32.0-ubuntu24.04-py3.14-prereleases-device-all-rocm10.2.0rc1-5df63a7203
#              Ubuntu 24.04, Python 3.14.10, ROCm 10.2.0rc1, RCCL 2.32.3.
#              Held at the base version by /keep.txt and checked in section 5:
#              torch 2.14.0+rocm10.2.0rc1, torchvision 0.29.0+rocm10.2.0rc1,
#              torchaudio 2.11.0.3+rocm10.2.0rc1, triton 3.8.0+git669b31ac,
#              numpy 2.4.6, MoRI (amd-mori) 1.2.3.post2.dev48+g2958fce6b,
#              flash-attn 2.8.3, tilelang 0.1.10.
#   - vLLM  -> 0.32.1.dev0+g5df63a720 -> 0.34.0rc1.dev479+g37ea94f71.rocm102
#              vllm-project/vllm 37ea94f715acc34f84bc3728ec3e365e40eb53ac,
#              built from source. torchvision is dropped from setup.py's
#              per-GPU requirements (section 3).
#   - AITER -> 0.1.24.post1 -> 0.1.29
#              ROCm/aiter v0.1.29 (AITER_BRANCH in upstream
#              docker/Dockerfile.rocm_base at 37ea94f71), built from source.
#
#   Other base packages may be added or upgraded by vLLM's requirements.
#
#   STATUS: tested on MI355X with .buildkite/amd-disagg (cluster_crs.sh,
#   vllm-router), DeepSeek-V4-Pro over MoRIIO, gsm8k 5-shot
#   flexible-extract / strict-match, all pass: TP8 1P1D (0.9469 / 0.9462),
#   DP8+EP8 1P1D (0.9515 / 0.9515), DP16+EP16 2P2D (0.9484 / 0.9469).
#   The 2P2D run set NCCL_NET=IB, as .buildkite/amd-disagg/cluster_crs.sh
#   now does; the 1P1D runs did not.
#
#   /app/versions.txt must have BASE_IMAGE, one BASE_PKG line per package
#   held by /keep.txt, and DELTA_PKG lines for vllm and amd-aiter at the
#   versions above. Read it with:
#     docker run --rm --entrypoint cat <image> /app/versions.txt
#
# Build from the repo root after a docker login that can pull
# rocm/ufb-private. Requires BuildKit (Docker >= 23) for the RUN heredocs.
# The build takes about 40 minutes, most of it AITER.
# Some pip installs print "ERROR: pip's dependency resolver does not
# currently take into account ..."; this does not fail the build.
#   DOCKER_BUILDKIT=1 docker build \
#     -f docker/vllm_disagg_ufb_inference.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:ufb-rocm10.2 .
#   docker push <your-registry>/vllm-disagg:ufb-rocm10.2
#
# To run it, set IMAGE to this image and SKIP_IONIC_MOUNTS (tested: 0 for
# 1P1D, 1 for 2P2D) in .buildkite/amd-disagg/test.sh, then from that
# directory run `bash test.sh` (TP8 1P1D), `WIDE_EP_MODE=1 bash test.sh`
# (DP8+EP8 1P1D) or `WIDE_EP_MODE=1 xP=2 yD=2 bash test.sh` (DP16+EP16 2P2D).
# =============================================================================

ARG BASE_IMAGE=rocm/ufb-private:vllm-0.32.0-ubuntu24.04-py3.14-prereleases-device-all-rocm10.2.0rc1-5df63a7203

# -----------------------------------------------------------------------------
# 1. Build base: record the held package versions, add build dependencies.
#    /keep.txt holds the installed versions of torch, triton, MoRI and the
#    other held packages. Every pip install below runs with -c /keep.txt,
#    so none of them is replaced. The apt packages are build tools only and
#    do not reach the final image.
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base
ARG PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 - <<'EOF'
import importlib.metadata as m
names = ["torch", "torchvision", "torchaudio", "triton", "numpy", "amdsmi",
         "amd-mori", "flash-attn", "tilelang"]
with open("/keep.txt", "w") as f:
    for n in names:
        try:
            f.write(f"{n}=={m.version(n)}\n")
        except m.PackageNotFoundError:
            pass
print(open("/keep.txt").read())
EOF
RUN apt-get update && apt-get install -y --no-install-recommends git pkg-config \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. AITER: build a wheel from source at v0.1.29.
#    Kernels are prebuilt for GPU_ARCH against the base's triton
#    (AITER_USE_SYSTEM_TRITON=1).
# -----------------------------------------------------------------------------
FROM base AS build_aiter
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_BRANCH=v0.1.29
ARG GPU_ARCH=gfx950
ARG PIP_BREAK_SYSTEM_PACKAGES=1
RUN git clone --recursive --branch ${AITER_BRANCH} ${AITER_REPO} /aiter \
 && cd /aiter \
 && python3 -m pip install -c /keep.txt -r requirements.txt pyyaml \
 && PREBUILD_KERNELS=1 AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS="${GPU_ARCH}" \
      python3 setup.py bdist_wheel --dist-dir=/dist

# -----------------------------------------------------------------------------
# 3. vLLM: compile from source at upstream main commit 37ea94f71
#    (HIP kernels plus the Rust frontend). vLLM's setup.py adds per-GPU
#    requirements for torch, torchvision and rocm. The gfx950 torchvision
#    packages are neither in the base nor on public PyPI, so
#    torchvision is removed from that list (sed below) and the base's
#    torchvision is used. The torch and rocm entries are unchanged.
# -----------------------------------------------------------------------------
# VLLM_VERSION is passed as SETUPTOOLS_SCM_PRETEND_VERSION. Change it together
# with VLLM_COMMIT and the expected versions in section 5.
FROM base AS build_vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT=37ea94f715acc34f84bc3728ec3e365e40eb53ac
ARG VLLM_VERSION=0.34.0rc1.dev479+g37ea94f71
ARG GPU_ARCH=gfx950
ARG PIP_BREAK_SYSTEM_PACKAGES=1
ENV PATH=/root/.cargo/bin:${PATH}
RUN ulimit -n "$(ulimit -Hn)" && git clone ${VLLM_REPO} /vllm && cd /vllm && git checkout ${VLLM_COMMIT} \
 && sed -i "s/\"torchvision\", //" setup.py \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none \
 && rustup toolchain install "$(grep '^channel' rust-toolchain.toml | sed 's/.*= *"\(.*\)"/\1/')" \
 && python3 -m pip install -c /keep.txt -r requirements/build/rust.txt \
      cmake ninja "setuptools>=78.1.1,<80" setuptools-scm setuptools-rust wheel jinja2 packaging \
 && SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" VLLM_TARGET_DEVICE=rocm \
    VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_BUILD_JOBS=32 PYTORCH_ROCM_ARCH="${GPU_ARCH}" MAX_JOBS="$(nproc)" \
      python3 setup.py bdist_wheel --dist-dir=/dist

# -----------------------------------------------------------------------------
# 4. Final image: install the AITER and vLLM wheels on the base.
#    The first install (--force-reinstall --no-deps) replaces the base's
#    vLLM and AITER. The second adds or upgrades their other
#    dependencies from PyPI as needed; packages in /keep.txt cannot change.
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG PIP_BREAK_SYSTEM_PACKAGES=1
COPY --from=base /keep.txt /keep.txt
COPY --from=build_aiter /dist/ /wheels/
COPY --from=build_vllm /dist/ /wheels/
RUN ls /wheels \
 && python3 -m pip install -c /keep.txt --force-reinstall --no-deps /wheels/*.whl \
 && python3 -m pip install -c /keep.txt /wheels/*.whl \
 && rm -rf /wheels

# -----------------------------------------------------------------------------
# 5. Verify: held base packages unchanged, vLLM and AITER at the pinned
#    versions, vLLM's compiled extensions import (AITER: version check
#    only). Results go to /app/versions.txt.
# -----------------------------------------------------------------------------
RUN mkdir -p /app && echo "BASE_IMAGE=${BASE_IMAGE}" >> /app/versions.txt && python3 - <<'EOF'
import importlib.metadata as m
kept = dict(l.split("==") for l in open("/keep.txt").read().split())
changed = {n: (v, m.version(n)) for n, v in kept.items() if m.version(n) != v}
assert not changed, f"UFB packages changed: {changed}"
want = {"vllm": "0.34.0rc1.dev479", "amd-aiter": "0.1.29"}
got = {n: m.version(n) for n in want}
bad = {n: got[n] for n in want if not got[n].startswith(want[n])}
assert not bad, f"wrong versions: {bad}"
with open("/app/versions.txt", "a") as f:
    f.writelines(f"BASE_PKG {n}=={v}\n" for n, v in kept.items())
    f.writelines(f"DELTA_PKG {n}=={v}\n" for n, v in got.items())
print("UFB kept:", kept)
print("delta:", got)
EOF
RUN python3 -c "import vllm, vllm.entrypoints.openai.api_server, vllm._C, vllm._rocm_C; print('imports ok', vllm.__version__)"

RUN echo "=== Build versions ===" && cat /app/versions.txt
