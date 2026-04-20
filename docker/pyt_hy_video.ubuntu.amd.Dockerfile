# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

ARG work_dir=/hunyuanvideo

WORKDIR $work_dir

# Pin transformers 4.x for diffusers 0.32.2 (FLAX_WEIGHTS_NAME removed in v5).
RUN pip install diffusers==0.32.2 distvae yunchang==0.6.0 opencv-python accelerate "transformers>=4.44.0,<5.0"
RUN pip install imageio imageio-ffmpeg
RUN pip install beautifulsoup4==4.12.3
RUN pip install sentencepiece==0.1.99
RUN pip install numpy==1.26.4

# xDiT repository
#RUN git clone git@github.com:mqhc2020/xDiT.git -b rocm_opt && \
RUN git clone https://github.com/xdit-project/xDiT && \
    cd xDiT && git checkout 775a5263d95518a733e4f239ad21228b755598bb && \
    pip install --no-deps -e .

# flash attn (avoid ARG name PYTORCH_ROCM_ARCH: base image ENV can shadow it and expand to "")
# ROCm flash-attention: FA_GPU_ARCH=native needs a visible GPU at compile time and fails in CI/docker.
# Coerce native/empty to gfx942 for headless CI; for MI350 pass --build-arg FA_GPU_ARCH=gfx950 (needs FA_SHA with gfx950 in setup.py).
#ARG FA_SHA="b3ae4966b2567811880db10d9e040a775b99c7d7"
#ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
#ARG FA_GPU_ARCH=gfx942
#RUN git clone ${FA_REPO} && \
#    cd flash-attention && \
#    git checkout ${FA_SHA} && \
#    git submodule update --init && \
#    F='${FA_GPU_ARCH}' && \
#    if [ -z "$F" ]; then F=gfx942; fi && \
#    if [ "$F" = "native" ]; then F=gfx942; fi && \
#    GPU_ARCHS="$F" python3 setup.py bdist_wheel --dist-dir=dist && \
#    pip install dist/*.whl;
RUN git clone https://github.com/ROCm/flash-attention.git \
    && cd flash-attention \
    && FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .



# RPD profiling
RUN apt update && \
    apt install -y sqlite3 libsqlite3-dev libfmt-dev
# Upstream uses pip --user; venv disallows that.
RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData && \
    cd rocmProfileData && \
    sed -i 's/pip install --user/pip install/g' rocpd_python/Makefile && \
    make && make install && \
    cd rocpd_python && python setup.py install && cd .. && \
    cd rpd_tracer && python setup.py install && cd ..
