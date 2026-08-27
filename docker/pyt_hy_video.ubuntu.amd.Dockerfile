# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

ARG work_dir=/hunyuanvideo

WORKDIR $work_dir

#RUN pip install diffusers==0.32.2 distvae yunchang==0.6.0 opencv-python accelerate
RUN pip install transformers==4.56.2 diffusers==0.32.2 distvae yunchang==0.6.0 opencv-python-headless accelerate
RUN pip install imageio imageio-ffmpeg
RUN pip install beautifulsoup4==4.12.3
RUN pip install sentencepiece>=0.2.0
RUN pip install numpy==1.26.4

# xDiT repository
#RUN git clone git@github.com:mqhc2020/xDiT.git -b rocm_opt && \
RUN git clone https://github.com/xdit-project/xDiT && \
    cd xDiT && git checkout 775a5263d95518a733e4f239ad21228b755598bb && \
    pip install --no-deps -e .

# flash attn
ARG FA_SHA="83f9e450cd10e20701fb109db9c7703d376f282b"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
ARG PYTORCH_ROCM_ARCH="gfx942"
RUN git clone ${FA_REPO} && \
    cd flash-attention && \
    git checkout ${FA_SHA} && \
    git submodule update --init && \
    GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl;
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    xxd \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# RPD profiling
#RUN apt update && \
#    apt install -y sqlite3 libsqlite3-dev libfmt-dev
#RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData && \
#    cd rocmProfileData && \
#    make && make install && \
#    cd rocpd_python && python setup.py install && cd .. && \
#    cd rpd_tracer && python setup.py install && cd ..


# RPD profiling
RUN apt update && \
    apt install -y sqlite3 libsqlite3-dev libfmt-dev
RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData && \
    cd rocmProfileData && \
    make && \
    make install -C rocpd_python && \
    make install -C rpd_tracer && \
    ldconfig
                                           
