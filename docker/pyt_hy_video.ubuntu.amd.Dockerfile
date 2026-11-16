# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

# Additional step for base images that ship setup_pytorch_env.
# Guard on the setup command itself rather than on the workspace directory, so this
# same Dockerfile still builds against a plain upstream base, which does not ship it.
# The --mad flag installs into the image's existing environment instead of creating a
# venv. There is no /opt/venv-py<X.Y> to link, so this block has no symlink step and --
# importantly -- no "ENV PATH=/opt/venv/bin:${PATH}" line after it. Adding one would not
# fail the build (ENV never touches the filesystem, and the shell silently skips PATH
# entries that do not exist), it would just be dead PATH that hides which interpreter is
# actually in use. Leave it out.
RUN if command -v setup_pytorch_env >/dev/null 2>&1; then \
        echo "setup_pytorch_env found. Running env setup..."; \
        setup_pytorch_env --mad || { \
            echo "Setup failed."; \
            exit 1; \
        }; \
        echo "Setup successful."; \
        for pair in python:python3 pip:pip3; do \
            name="${pair%%:*}"; target="${pair#*:}"; \
            if ! command -v "$name" >/dev/null 2>&1; then \
                ln -s "$(command -v "$target")" "/usr/local/bin/$name" || { \
                    echo "Failed to link $name -> $target."; \
                    exit 1; \
                }; \
                echo "Linked $name -> $(command -v "$target")"; \
            fi; \
        done; \
    else \
        echo "setup_pytorch_env is not on PATH."; \
        echo "Nothing to set up."; \
    fi

# The ROCm pip SDK install location varies across base images (venv site-packages
# vs system dist-packages, and by Python version), so it cannot be hardcoded.
# Resolve it once here and expose stable symlinks that the ENV block below can
# reference -- ENV performs ${VAR} substitution only and never runs $(...).
#
# Guard on the rocm-sdk command, the same way Block 1 guards on setup_pytorch_env,
# so this Dockerfile still builds against a plain upstream base, which ships the
# classic /opt/rocm tree and no pip SDK. On that base the same three symlinks are
# pointed at the legacy layout, which keeps every ENV below valid on both bases
# and means the ENV wall itself needs no guard.
RUN set -eu; \
    if command -v rocm-sdk >/dev/null 2>&1; then \
        echo "ROCm pip SDK detected."; \
        ROCM_SDK_ROOT="$(rocm-sdk path --root)"; \
        [ -d "${ROCM_SDK_ROOT}" ] || { echo "ROCm SDK root not found: ${ROCM_SDK_ROOT}"; exit 1; }; \
        ln -sfn "${ROCM_SDK_ROOT}" /opt/rocm_sdk; \
        ROCM_SDK_CORE="$(dirname "${ROCM_SDK_ROOT}")/_rocm_sdk_core"; \
        [ -d "${ROCM_SDK_CORE}" ] || ROCM_SDK_CORE="${ROCM_SDK_ROOT}"; \
        ln -sfn "${ROCM_SDK_CORE}" /opt/rocm_sdk_core; \
        [ -d "${ROCM_SDK_ROOT}/share/amd_smi" ] || { echo "amd_smi bindings not found under ${ROCM_SDK_ROOT}"; exit 1; }; \
        ln -sfn "${ROCM_SDK_ROOT}/share/amd_smi" /opt/amd_smi; \
        echo "ROCm SDK root: ${ROCM_SDK_ROOT}"; \
        echo "ROCm SDK core: ${ROCM_SDK_CORE}"; \
    else \
        echo "rocm-sdk is not on PATH, so this is not a ROCm pip SDK image."; \
        [ -d /opt/rocm ] || { echo "No ROCm pip SDK and no /opt/rocm; cannot resolve a ROCm root."; exit 1; }; \
        ln -sfn /opt/rocm /opt/rocm_sdk; \
        ln -sfn /opt/rocm /opt/rocm_sdk_core; \
        if [ -d /opt/rocm/share/amd_smi ]; then \
            ln -sfn /opt/rocm/share/amd_smi /opt/amd_smi; \
        else \
            mkdir -p /opt/amd_smi; \
        fi; \
        echo "Falling back to the legacy /opt/rocm layout."; \
        echo "ROCm SDK root: /opt/rocm"; \
        echo "ROCm SDK core: /opt/rocm"; \
    fi


ENV ROCM_HOME="/opt/rocm_sdk" \
    ROCM_PATH="/opt/rocm_sdk" \
    ROCM_SOURCE_DIR="/opt/rocm_sdk" \
    ROCM_BIN="/opt/rocm_sdk/bin" \
    ROCM_CMAKE="/opt/rocm_sdk/lib/cmake" \
    SITE_PACKAGES="/opt/rocm_sdk/lib/python3.10/site-packages" \
    ROCM_CMAKE_PREFIX="/opt/rocm_sdk/lib/cmake"

# Explicitly set the library paths for ROCm and the ROCm SDK core to guarantee that llama.cpp always can find the necessary libraries
ENV CMAKE_PREFIX_PATH="${ROCM_CMAKE}" \
    HIP_DEVICE_LIB_PATH="${ROCM_HOME}/amdgcn/bitcode" \
    ROCM_DEVICE_LIB_PATH="${ROCM_HOME}/amdgcn/bitcode" \
    ROCM_SYSDEPS_INCLUDE="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    CPLUS_INCLUDE_PATH="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    C_INCLUDE_PATH="${ROCM_HOME}/lib/rocm_sysdeps/include" \
    PKG_CONFIG_PATH="${ROCM_HOME}/lib/rocm_sysdeps/lib/pkgconfig" \
    LD_LIBRARY_PATH="/opt/rocm_sdk/lib:/opt/rocm_sdk/lib/host-math/lib:/opt/rocm_sdk/lib/rocm_sysdeps/lib:/opt/rocm_sdk_core/lib:/opt/rocm_sdk_core/lib/rocm_sysdeps/lib" \
    LIBRARY_PATH="/opt/rocm_sdk/lib:/opt/rocm_sdk/lib/host-math/lib:/opt/rocm_sdk/lib/rocm_sysdeps/lib" \
    PATH="/opt/rocm_sdk/bin:${PATH}"

# AMD-SMI Python bindings: ROCm 10.1+ pip SDK dropped setup.py from
# share/amd_smi (rocm-systems #9340); expose the module via PYTHONPATH instead.
ENV PYTHONPATH="/opt/amd_smi"

ARG work_dir=/hunyuanvideo

WORKDIR $work_dir

# git for the source clones below; opencv-python runtime libraries
# (none of these are present in slim base images)
RUN apt update && \
    apt install -y git libgl1 libglib2.0-0

RUN pip install diffusers==0.32.2 distvae yunchang==0.6.0 opencv-python accelerate
# diffusers 0.32.2 imports symbols that transformers 5.x removed
RUN pip install transformers==4.57.6
RUN pip install imageio imageio-ffmpeg
RUN pip install beautifulsoup4==4.12.3
RUN pip install sentencepiece==0.2.1

# xDiT repository
#RUN git clone git@github.com:mqhc2020/xDiT.git -b rocm_opt && \
RUN git clone https://github.com/xdit-project/xDiT && \
    cd xDiT && git checkout 775a5263d95518a733e4f239ad21228b755598bb && \
    pip install --no-deps -e .

# flash attn: built from source only when the base image doesn't already provide it.
# Pinned 2026-10-15 to upstream main commit 1fc02211 (flash_attn 2.8.4, committed 2026-10-12).
# Why: PyTorch 2.14+ headers require C++20, and upstream builds the ROCm path with -std=c++20
# since edb5c76 (#2899). No release tag includes that fix yet (v2.8.3.post2 still uses
# -std=c++17), so pin a tested commit instead of tracking main. Move to a release tag once
# one ships with -std=c++20
ARG FA_BRANCH="1fc0221183e799363001394a12dd9e78e15f5dda"
ARG FA_REPO="https://github.com/Dao-AILab/flash-attention.git"
# Build for the GPU arch madengine detects (passed as a build arg)
ARG MAD_SYSTEM_GPU_ARCHITECTURE
ENV HIP_ARCHITECTURES=${MAD_SYSTEM_GPU_ARCHITECTURE}
RUN echo HIP_ARCHITECTURES = ${HIP_ARCHITECTURES}
RUN if python3 -c "import flash_attn" 2>/dev/null; then \
        echo "flash_attn $(python3 -c 'import flash_attn; print(flash_attn.__version__)') found in base image, skipping source build"; \
    else \
        git clone ${FA_REPO} && \
        cd flash-attention && \
        git checkout ${FA_BRANCH} && \
        git submodule update --init && \
        GPU_ARCHS=${HIP_ARCHITECTURES} python3 setup.py bdist_wheel --dist-dir=dist && \
        pip install dist/*.whl; \
    fi

# RPD profiling
RUN apt update && \
    apt install -y sqlite3 libsqlite3-dev libfmt-dev cmake xxd
RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData && \
    cd rocmProfileData && \
    git submodule update --init rlog && \
    make && make install && \
    cd rocpd_python && python3 setup.py install && cd .. && \
    cd rpd_tracer && python3 setup.py install && cd ..
