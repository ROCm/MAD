# Kimi-K3 inference on AMD Instinct MI350X / MI355X

## Overview

[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) is Moonshot AI's first open-source model in the trillion-plus class (2.8 T parameters). It is a Mixture-of-Experts model with native MXFP4 quantization (QAT) and always-on reasoning, and ships as a single checkpoint — there is no separate AMD-quantized variant.

MAD supports Kimi-K3 day-0 inference across **three** serving frameworks on AMD Instinct MI350X / MI355X (gfx950):

| Framework | MAD tag | Docker image | Recipe |
|-----------|---------|-------------|--------|
| **vLLM** | `pyt_vllm_kimi-k3` | `vllm/vllm-openai-rocm:kimi-k3` | [recipes.vllm.ai](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x) |
| **SGLang** | `pyt_sglang_kimi-k3` | `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` | [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) |
| **ATom** | `pyt_atom_kimi-k3` | `rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3` | ROCm ATom |

## Hardware requirements

- **8x MI350X or MI355X** (gfx950 architecture, TP8)
- ~1680 GB minimum GPU memory footprint (fits 8x MI355X @ 2304 GB total)
- Does **not** fit on 8x MI300X (gfx942) — all tags set `skip_gpu_arch: gfx942`
- Checkpoint is ~1.56 TB — ensure the model cache volume has enough space
- Disable NUMA auto-balancing on the host for optimal performance:
  ```sh
  sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
  ```

## Quick start (MAD-integrated)

Install [madengine](https://github.com/ROCm/madengine) and clone this repository, then run any of the three frameworks with a single command:

```sh
# vLLM
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output

# SGLang
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output

# ATom
madengine run --tags pyt_atom_kimi-k3 --keep-model-dir --live-output
```

To use pre-downloaded weights instead of downloading from HuggingFace:

```sh
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output \
  --additional-context '{"docker_mounts": {"/model_weights": "/path/to/Kimi-K3"}, "docker_env_vars": {"MAD_DATAHOME": "/model_weights"}}'
```

## Framework details

### vLLM

- **Tag**: `pyt_vllm_kimi-k3`
- **Image**: [`vllm/vllm-openai-rocm:kimi-k3`](https://hub.docker.com/r/vllm/vllm-openai-rocm)
- **Dockerfile**: [docker/pyt_vllm_kimi_k3.ubuntu.amd.Dockerfile](../../docker/pyt_vllm_kimi_k3.ubuntu.amd.Dockerfile)
- **Config**: [scripts/vllm/configs/default.yaml](../../scripts/vllm/configs/default.yaml) (Kimi-K3 block)
- **Benchmark**: Online serving at concurrency 1 / 8 / 32 / 128, input 1024, output 1024

### SGLang

- **Tag**: `pyt_sglang_kimi-k3`
- **Image**: [`lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727`](https://hub.docker.com/r/lmsysorg/sglang-rocm)
- **Dockerfile**: [docker/pyt_sglang_kimi_k3.ubuntu.amd.Dockerfile](../../docker/pyt_sglang_kimi_k3.ubuntu.amd.Dockerfile)
- **Config**: [scripts/sglang/configs/kimi_k3.yaml](../../scripts/sglang/configs/kimi_k3.yaml)
- **Benchmark**: Online serving at concurrency 2 / 4 / 8 / 16 / 32, input 8192, output 1024

### ATom

- **Tag**: `pyt_atom_kimi-k3`
- **Image**: [`rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3`](https://hub.docker.com/r/rocm/atom-dev)
- **Dockerfile**: [docker/pyt_atom_kimi_k3.ubuntu.amd.Dockerfile](../../docker/pyt_atom_kimi_k3.ubuntu.amd.Dockerfile)
- **Config**: [scripts/atom/configs/default.yaml](../../scripts/atom/configs/default.yaml)
- **Benchmark**: Online serving at concurrency 64 / 128 / 256, input 1024 and 4096, output 1024

## Standalone serving (without madengine)

### vLLM

```sh
docker pull vllm/vllm-openai-rocm:kimi-k3

docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size 16G --network host \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --cap-add=SYS_PTRACE \
  -v /path/to/Kimi-K3:/model_weights \
  --env VLLM_ROCM_USE_AITER=1 --env SAFETENSORS_FAST_GPU=1 \
  --env AITER_SITUV2_A8W4=1 --env AITER_BF16_FP8_MOE_BOUND=0 \
  --env VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  --entrypoint bash vllm/vllm-openai-rocm:kimi-k3 \
  -c "vllm serve /model_weights --dtype auto -tp 8 --trust-remote-code \
      --enable-prefix-caching --load-format auto --gpu-memory-utilization 0.95 \
      --moe-backend auto --mm-encoder-tp-mode data --max-num-seqs 128 \
      --max-num-batched-tokens 4096 --reasoning-parser kimi_k3 \
      --language-model-only --disable-uvicorn-access-log"
```

### SGLang

```sh
docker pull lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727

docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size 16G --network host \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --cap-add=SYS_PTRACE \
  -v /path/to/Kimi-K3:/model_weights \
  --env SGLANG_USE_AITER=1 --env SGLANG_AITER_K3_OPT=1 \
  --env AITER_FLYDSL_FORCE=1 --env AITER_SITUV2_A8W4=1 \
  --entrypoint bash lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727 \
  -c "sglang serve --model-path /model_weights --trust-remote-code \
      --tp-size 8 --attention-backend triton --dtype bfloat16 \
      --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 \
      --host 0.0.0.0 --port 8000 --disable-radix-cache \
      --reasoning-parser kimi_k3 --tool-call-parser kimi_k3"
```

### ATom

```sh
docker pull rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3

docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size 16G --network host \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --cap-add=SYS_PTRACE \
  -v /path/to/Kimi-K3:/model_weights \
  --env ATOM_LOADER_USE_THREADPOOL=1 --env ATOM_LOADER_THREADPOOL_WORKERS=16 \
  --env ATOM_SYNC_AFTER_LOAD=1 --env ATOM_DIST_TIMEOUT_SECONDS=3600 \
  --env ATOM_USE_TRITON_GEMM=1 --env AITER_USE_GROUPED_GEMM=0 \
  --env ATOM_USE_TRITON_MOE=0 --env AITER_FLYDSL_FORCE=1 \
  --env AITER_FORCE_GFX1250=0 --env ATOM_USE_UNIFIED_ATTN=1 \
  --env ATOM_FORCE_ATTN_TRITON=1 \
  --entrypoint bash rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3 \
  -c "python -m atom.entrypoints.openai_server \
      --model /model_weights --kv_cache_dtype fp8 -tp 8 --trust-remote-code \
      --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 10240 \
      --gpu-memory-utilization 0.93 --block-size 128 --no-enable_prefix_caching"
```

## References

- [moonshotai/Kimi-K3 on HuggingFace](https://huggingface.co/moonshotai/Kimi-K3)
- [vLLM Kimi-K3 recipe (MI355X)](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x)
- [vLLM day-0 blog post](https://vllm.ai/blog/2026-07-27-k3)
- [SGLang Kimi-K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- [SGLang day-0 tracking issue](https://github.com/sgl-project/sglang/issues/32548)

## Licensing

Your use of this application is subject to the terms of the applicable component-level license. See the individual framework benchmark pages ([vLLM](../vllm/README.md#licensing-information-%EF%B8%8F), [SGLang](../sglang/README.md)) for full licensing details.
