FROM localhost/mad-mori-ep:gfx950-v1

ARG MORI_COMMIT=241461c0aaf8be2a502397668d4b3e1aab90a188

WORKDIR /app

# Remove old mori completely (including stale C++ extensions that confuse profiler detection)
RUN pip uninstall -y mori amd-mori amd_mori 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3.12/dist-packages/mori* && \
    rm -rf /app/mori

RUN git clone --recursive https://github.com/ROCm/mori.git /app/mori && \
    cd /app/mori && \
    git checkout ${MORI_COMMIT} && \
    PYTORCH_ROCM_ARCH=gfx950 pip install -e . && \
    echo "MORI updated to $(git rev-parse --short HEAD) on $(date -u +%Y-%m-%d)"

RUN sed -i "s|^MORI_BRANCH:.*|MORI_BRANCH: $(cd /app/mori && git rev-parse --short HEAD)|" /app/versions.txt && \
    cat /app/versions.txt

WORKDIR /app
