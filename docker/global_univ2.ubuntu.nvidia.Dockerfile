# CONTEXT {'gpu_vendor': 'NVIDIA', 'guest_os': 'UBUNTU'}
#
# NVIDIA counterpart of global_univ2.ubuntu.amd.Dockerfile, built by madengine
# image_spec cards when the build context selects gpu_vendor NVIDIA.
#
# madengine resolves BASE_DOCKER and TORCH_* from compat/cuda.json (synced from
# replicate/cog): BASE_DOCKER is an nvidia/cuda devel image for the CUDA that
# the selected torch was built for. All ARGs can be overridden through
# docker_build_arg like any other Dockerfile.
#
# Steps: python -> pip prerequisites -> torch -> component wheels (in order)
#        -> pinned source clones -> /opt/mad/build_info.json
ARG BASE_DOCKER=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM $BASE_DOCKER
ARG BASE_DOCKER
# Empty = the distribution python3. Otherwise installed with uv.
ARG PYTHON_VERSION=""
# e.g. "torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"
ARG TORCH_PACKAGES=""
ARG TORCH_INDEX_URL=""
# ',' separates component wheels, installed in order.
ARG COMPONENT_WHEEL_URLS=""
# Optional sha256 per component, ',' separated in the same order (empty = unchecked).
ARG COMPONENT_SHA256=""
ARG PIP_PREREQS_EXTRA=""
# ',' separates members; each is <repo-url>/commit/<commit-sha>.
ARG COMPONENT_REPOS=""
# Resolved image_spec plan (JSON) from madengine; recorded as a label only.
ARG MAD_BUILD_PLAN=""
LABEL org.amd.mad.build_plan="${MAD_BUILD_PLAN}"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace
# =============================================================================
# Step 1: Python in a venv at /opt/mad/venv (same path as the AMD backend)
# pipefail: a failed uv download piped into sh must fail here, not as "uv: not found".
# =============================================================================
RUN set -eo pipefail; \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git jq python3 python3-venv wget; \
    if [ -n "$PYTHON_VERSION" ]; then \
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh; \
        UV_NATIVE_TLS=1 uv venv --seed --python "$PYTHON_VERSION" /opt/mad/venv; \
    else \
        python3 -m venv /opt/mad/venv; \
    fi
ENV PATH="/opt/mad/venv/bin:${PATH}"
# =============================================================================
# Step 2: pip prerequisites, torch, component wheels
# =============================================================================
RUN set -e; \
    pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir numpy ${PIP_PREREQS_EXTRA}; \
    if [ -n "$TORCH_PACKAGES" ]; then \
        pip install --no-cache-dir \
            ${TORCH_INDEX_URL:+--index-url "$TORCH_INDEX_URL"} ${TORCH_PACKAGES}; \
    fi; \
    IFS=',' read -r -a URLS <<< "$COMPONENT_WHEEL_URLS"; \
    IFS=',' read -r -a SHAS <<< "$COMPONENT_SHA256"; \
    unset IFS; \
    mkdir -p /tmp/wheels; \
    for idx in "${!URLS[@]}"; do \
        url="${URLS[$idx]}"; \
        [ -n "$url" ] || continue; \
        file="/tmp/wheels/$(basename "$url")"; \
        wget -q -O "$file" "$url"; \
        if [ -n "${SHAS[$idx]:-}" ]; then \
            echo "${SHAS[$idx]}  $file" | sha256sum -c -; \
        fi; \
        pip install --no-cache-dir "$file"; \
    done; \
    rm -rf /tmp/wheels
# =============================================================================
# Step 3: Clone component sources at pinned commits
# =============================================================================
RUN set -e; \
    IFS=',' read -r -a REPOS <<< "$COMPONENT_REPOS"; \
    unset IFS; \
    for member in "${REPOS[@]}"; do \
        member=$(echo "$member" | xargs); \
        [ -n "$member" ] || continue; \
        case "$member" in \
            */commit/*) ;; \
            *) echo "ERROR: COMPONENT_REPOS must be <url>/commit/<sha>: $member"; exit 1 ;; \
        esac; \
        repo="${member%%/commit/*}"; \
        sha="${member##*/commit/}"; \
        name=$(basename "$repo" .git); \
        git clone "$repo" "/workspace/$name"; \
        git -C "/workspace/$name" fetch origin "$sha"; \
        git -C "/workspace/$name" checkout --detach FETCH_HEAD; \
        git -C "/workspace/$name" submodule update --init --recursive; \
    done
# =============================================================================
# Step 4: Record the installed stack for madengine (build_manifest image_info)
# =============================================================================
COPY global_univ2_build_info.py /opt/mad/build_info.py
RUN python3 /opt/mad/build_info.py > /opt/mad/build_info.json
ENTRYPOINT [""]
