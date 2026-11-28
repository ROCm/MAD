# Image for the Qwen2.5-0.5B SGLang serving workload: SGLang's public ROCm release image for MI350X/MI355X and the
# serving script. Nothing is installed: torch, SGLang and ROCm are the image's own.
#   docker build -t qwen-sglang-serve:v0.5.23 .
ARG BASE=lmsysorg/sglang:v0.5.23-rocm10-mi35x@sha256:cff45eca86da10fbe140f85df196ef241e932bbae8036afa45942e633c3ed6c5
FROM ${BASE}

ENV HF_HUB_OFFLINE=1 SGLANG_CACHE_DIR=/opt/sglang-cache
# The base does not ship SGLang's Rust radix tree (the default prefix cache): every new container would fetch its
# crates from crates.io and compile it (about a minute) on the first request. Build it once, here.
RUN python3 -c "import sglang.srt.mem_cache.rust_tree_core.extension" && \
    rm -rf /opt/sglang-cache/rust_extensions/targets && chmod -R a+rwX /opt/sglang-cache

WORKDIR /opt/qwen_sglang_serve
COPY qwen_sglang_serve.py .
