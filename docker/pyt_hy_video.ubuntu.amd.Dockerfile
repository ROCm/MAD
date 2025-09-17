# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=docker.io/rocm/pytorch:latest
FROM $BASE_DOCKER

ARG work_dir=/hunyuanvideo

WORKDIR $work_dir

RUN pip install diffusers==0.32.2 distvae yunchang==0.6.0 opencv-python accelerate
RUN pip install imageio imageio-ffmpeg
RUN pip install beautifulsoup4==4.12.3
RUN pip install sentencepiece==0.1.99
RUN pip install numpy==1.26.4

# xDiT repository
#RUN git clone git@github.com:mqhc2020/xDiT.git -b rocm_opt && \
RUN git clone https://github.com/xdit-project/xDiT && \
    cd xDiT && git checkout 775a5263d95518a733e4f239ad21228b755598bb && \
    pip install --no-deps -e .

# flash attn
ARG FA_SHA="22c0358"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
ARG PYTORCH_ROCM_ARCH="gfx942"
RUN git clone ${FA_REPO} && \
    cd flash-attention && \
    git checkout ${FA_SHA} && \
    git submodule update --init && \
    GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl;

# RPD profiling
RUN apt update && \
    apt install -y sqlite3 libsqlite3-dev libfmt-dev
RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData && \
    cd rocmProfileData && \
    make && make install && \
    cd rocpd_python && python setup.py install && cd .. && \
    cd rpd_tracer && python setup.py install && cd ..
