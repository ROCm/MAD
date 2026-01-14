# LLM inference performance validation with vLLM on the AMD Instinct MI300X accelerator

## Overview 🎉
--------

vLLM is a toolkit and library for large language model (LLM) inference and serving. It
deploys the PagedAttention algorithm, which reduces memory consumption
and increases throughput by leveraging dynamic key and value allocation
in GPU memory. vLLM also incorporates many recent LLM acceleration and
quantization algorithms. In addition, AMD implements high-performance custom
kernels and modules in vLLM to enhance performance further.

This Docker image packages vLLM with PyTorch for AMD Instinct™ MI300X, MI325X, MI350X and MI355X
accelerators. It includes:

-   ✅ ROCm™ 7.0.0
-   ✅ vLLM 0.11.2 (0.11.2.dev673+g839868462.rocm700)
-   ✅ PyTorch 2.9.0 (2.9.0a0+git1c57644)
-   ✅ hipBLASLt 1.0

With this Docker image, users can quickly validate the expected inference performance numbers on the Instinct accelerators listed above. 
This guide also provides tips and techniques so that users can get optimal performance with popular AI models.


## Reproducing benchmark results 🚀
-----------------------------

Use the following instructions to reproduce the benchmark results on an
MI300X/MI325X/MI350X/MI355X accelerator with a prebuilt vLLM Docker image.

Users have two choices to reproduce the benchmark results.

-   [MAD-integrated benchmarking](#mad-integrated-benchmarking)
-   [Standalone benchmarking](#standalone-benchmarking)

### NUMA balancing setting

To optimize performance, disable automatic NUMA balancing. Otherwise, the GPU
might hang until the periodic balancing is finalized. For further
details, refer to the [AMD Instinct MI300X system optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing) guide.

```sh
# disable automatic NUMA balancing
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
# check if NUMA balancing is disabled (returns 0 if disabled)
cat /proc/sys/kernel/numa_balancing
0
```

### Advanced features and known issues 🚨

For the experimental features and known issues concerning ROCm optimization efforts on vLLM, see the developer's guide at [ROCm/vLLM](https://github.com/ROCm/vllm/blob/main/docs/dev-docker/README.md).

To override the benchmark configs, specify a certain benchmark to use, or add your own configs, please see the [vllm benchmark script](../../scripts/vllm/run.sh) and the [CSV configs](../../scripts/vllm/configs/)

>[!NOTE]
>If you're using this docker image on other AMD GPUs e.g. MI2xx or Radeon GPUs, please add
>`export VLLM_ROCM_USE_AITER=0` to your commands, since AITER is only supported on the gfx942 and gfx950 architectures.

>[!NOTE]
>There is a known regression with AITER for MoE models such as Mixtral and DeepSeek-R1. Consider using the previous release `rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103` for better performance.

### Download the Docker image 🐳

The following command pulls the Docker image from Docker Hub.

```sh
docker pull rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210
```

### MAD-integrated benchmarking

Clone the ROCm Model Automation and Dashboarding (MAD) repository to a local directory and install the required packages on the host machine.

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt
```

Use this command to run a performance benchmark test of the Llama 3.1 8B model on one GPU with float16 data type in the host machine. 

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
madengine run --tags pyt_vllm_llama-3.1-8b_fp8 --keep-model-dir --live-output
```

ROCm MAD launches a Docker container with the name `container_ci-pyt_vllm_llama-3.1-8b_fp8`. The benchmark results of the model are collected at `perf_Llama-3.1-8B-Instruct.csv`.

Although the following models are pre-configured to collect online serving performance data,
users can also directly run the vLLm benchmark scripts and change the benchmarking parameters. Refer to the [Standalone benchmarking](#standalone-benchmarking) section.

#### Available models

>[!NOTE]
>The MXFP4 models are only supported on the gfx950 architecture i.e. MI350X/MI355X accelerators.	
>```

| MAD model name                         | Model repo                             |
| -------------------------------------- | -------------------------------------- |
| pyt_vllm_deepseek-r1                   | [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| pyt_vllm_gpt-oss-20b                   | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| pyt_vllm_gpt-oss-120b                  | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| pyt_vllm_llama-2-70b                   | [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |
| pyt_vllm_llama-3.1-8b                  | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| pyt_vllm_llama-3.1-8b_fp8              | [amd/Llama-3.1-8B-Instruct-FP8-KV](https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV) |
| pyt_vllm_llama-3.1-405b                | [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) |
| pyt_vllm_llama-3.1-405b_fp8            | [amd/Llama-3.1-405B-Instruct-FP8-KV](https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV) |
| pyt_vllm_llama-3.1-405b_fp4            | [amd/Llama-3.1-405B-Instruct-MXFP4-Preview](https://huggingface.co/amd/Llama-3.1-405B-Instruct-MXFP4-Preview) |
| pyt_vllm_llama-3.3-70b                 | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| pyt_vllm_llama-3.3-70b_fp8             | [amd/Llama-3.3-70B-Instruct-FP8-KV](https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV) |
| pyt_vllm_llama-3.3-70b_fp4             | [amd/Llama-3.3-70B-Instruct-MXFP4-Preview](https://huggingface.co/amd/Llama-3.3-70B-Instruct-MXFP4-Preview) |
| pyt_vllm_llama-4-scout-17b-16e         | [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) |
| pyt_vllm_llama-4-maverick-17b-128e     | [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) |
| pyt_vllm_llama-4-maverick-17b-128e_fp8 | [meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) |
| pyt_vllm_mixtral-8x7b                  | [mistralai/Mixtral-8x7B-Instruct-v0.1](https://hugggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) |
| pyt_vllm_mixtral-8x7b_fp8              | [amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV](https://hugggingface.co/amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV) |
| pyt_vllm_mixtral-8x22b                 | [mistralai/Mixtral-8x22B-Instruct-v0.1](https://hugggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) |
| pyt_vllm_mixtral-8x22b_fp8             | [amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV](https://hugggingface.co/amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV) |
| pyt_vllm_phi-4                         | [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) |
| pyt_vllm_qwen3-8b                      | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| pyt_vllm_qwen3-32b                     | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| pyt_vllm_qwen3-30b-a3b                 | [Qwen/Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) |
| pyt_vllm_qwen3-30b-a3b_fp8             | [Qwen/Qwen3-30B-A3B-Thinking-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8) |
| pyt_vllm_qwen3-235b-a22b               | [Qwen/Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| pyt_vllm_qwen3-235b-a22b_fp8           | [Qwen/Qwen3-235B-A22B-Thinking-2507-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8) |


### Standalone benchmarking              
-----------------------------

Users also can run the benchmark tool after they launch a Docker container. For more information, please see the configs in [scripts/vllm/configs/](../../scripts/vllm/configs/) as well as the documentation for the [vLLM engine](https://docs.vllm.ai/en/latest/configuration/engine_args.html#engineargs) and the [vLLM benchmarks](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md)

#### Docker launch
```sh
docker pull rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210

docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HUGGINGFACE_HUB_CACHE=/workspace --name test rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210
```

#### Latency Command
```
model=amd/Llama-3.1-8B-Instruct-FP8-KV
tp=1
batch_size=16
in=1024
out=1024
dtype=auto
kv_cache_dtype=fp8
max_num_seqs=1024
max_num_batched_tokens=8192
max_model_len=131072

vllm bench latency --model $model \
    -tp $tp \
    --batch-size $batch_size \
    --input-len $in \
    --output-len $out \
    --dtype $dtype \
    --kv-cache-dtype $kv_cache_dtype \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --max-model-len $max_model_len \
    --output-json ${model}_latency.json \
```

#### Throughput Command
```
model=amd/Llama-3.1-8B-Instruct-FP8-KV
tp=1
num_prompts=1024
in=1024
out=1024
dtype=auto
kv_cache_dtype=fp8
max_num_seqs=1024
max_num_batched_tokens=8192
max_model_len=131072

vllm bench throughput --model $model \
    -tp $tp \
    --num-prompts $num_prompts \
    --input-len $in \
    --output-len $out \
    --dtype $dtype \
    --kv-cache-dtype $kv_cache_dtype \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --max-model-len $max_model_len \
    --output-json ${model}_throughput.json \
```

#### Serving Command

1. Start the server
```
model=amd/Llama-3.1-8B-Instruct-FP8-KV
tp=1
dtype=auto
kv_cache_dtype=fp8
max_num_seqs=1024
max_num_batched_tokens=8192
max_model_len=131072

vllm serve $model \
    -tp $tp \
    --dtype $dtype \
    --kv-cache-dtype $kv_cache_dtype \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --max-model-len $max_model_len \
    --no-enable-prefix-caching \
    --swap-space 16 \
    --disable-log-requests

    # Wait for model to load and server is ready to accept requests
```

2. On another terminal on the same machine, run the benchmark:
```
# Connect to the container
docker exec -it test bash

# Wait for the server to start
until curl -s http://localhost:8000/v1/models; do sleep 30; done

# Run the benchmark
model=amd/Llama-3.1-8B-Instruct-FP8-KV
max_concurrency=1
num_prompts=10
in=1024
out=1024
vllm bench serve --model $model \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --dataset-name random \
    --ignore-eos \
    --max-concurrency $max_concurrency \
    --num-prompts $num_prompts \
    --random-input-len $in \
    --random-output-len $out \
    --save-result \
    --result-filename ${model}_serving.json
```

#### Accuracy Command

1. Start the server
```
model=amd/Llama-3.1-8B-Instruct-FP8-KV
tp=1
dtype=auto
kv_cache_dtype=fp8
max_num_seqs=1024
max_num_batched_tokens=8192
max_model_len=131072

vllm serve $model \
    -tp $tp \
    --dtype $dtype \
    --kv-cache-dtype $kv_cache_dtype \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --max-model-len $max_model_len \
    --no-enable-prefix-caching \
    --swap-space 16 \
    --disable-log-requests

    # Wait for model to load and server is ready to accept requests
```

2. On another terminal on the same machine, run the benchmark:
```
# Connect to the container
docker exec -it test bash

# Wait for the server to start
until curl -s http://localhost:8000/v1/models; do sleep 30; done

# Install lm-eval
pip install lm-eval[api]

# Run the benchmark
model=amd/Llama-3.1-8B-Instruct-FP8-KV
lm_eval --model local-completions \
    --model_args model=$model,max_gen_toks=2048,num_concurrent=256,max_retries=10,base_url=http://localhost:8000/v1/completions \
    --tasks gsm8k --limit 250 --output_path ./tmp
```

>[!NOTE]
>If you encounter this error, pass your access-authorized Hugging Face token to the gated models.
>```sh
>OSError: You are trying to access a gated repo.
>
># pass your HF_TOKEN
>export HF_TOKEN=$your_personal_hf_token
>```

## References 🔎
----------

For an overview of the optional performance features of vLLM with
ROCm software, see [vLLM inference performance testing](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html).

To learn more about the options for the vllm benchmark scripts, see
<https://github.com/ROCm/vllm/tree/main/benchmarks>.

To learn how to run LLM models from Hugging Face or your own model, see the
[Using ROCm for AI](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/index.html) section of the ROCm documentation.

To learn how to optimize inference on LLMs, see the
[Fine-tuning LLMs and inference optimization](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/index.html) section of the ROCm documentation.

For a list of other ready-made Docker images for ROCm, see the 
[ROCm Docker image support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html).

## Licensing information ⚠️
---------------------

Your use of this application is subject to the terms of the applicable
component-level license identified below. To the extent any subcomponent
in this container requires an offer for corresponding source code, AMD
hereby makes such an offer for corresponding source code form, which
will be made available upon request. By accessing and using this
application, you are agreeing to fully comply with the terms of this
license. If you do not agree to the terms of this license, do not access
or use this application.

The application is provided in a container image format that includes
the following separate and independent components:

| Package | License                                          | URL                  |
| ------- | ------------------------------------------------ | -------------------- |
| Ubuntu  | Creative Commons CC-BY-SA Version 3.0 UK License | [Ubuntu Legal](https://ubuntu.com/legal) |
| ROCm    | Custom/MIT/Apache V2.0/UIUC OSL                  | [ROCm Licensing Terms](https://rocm.docs.amd.com/en/latest/about/license.html) |
| PyTorch | Modified BSD                                     | [PyTorch License](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| vLLM    | Apache License 2.0                               | [vLLM License](https://github.com/vllm-project/vllm/blob/main/LICENSE)  |

### Disclaimer

The information contained herein is for informational purposes only and
is subject to change without notice. In addition, any stated support is
planned and is also subject to change. While every precaution has been
taken in the preparation of this document, it may contain technical
inaccuracies, omissions and typographical errors, and AMD is under no
obligation to update or otherwise correct this information. Advanced
Micro Devices, Inc. makes no representations or warranties with respect
to the accuracy or completeness of the contents of this document, and
assumes no liability of any kind, including the implied warranties of
noninfringement, merchantability or fitness for purposes, with respect
to the operation or use of AMD hardware, software or other products
described herein. No license, including implied or arising by estoppel,
to any intellectual property rights is granted by this document. Terms
and limitations applicable to the purchase or use of AMD's products are
as set forth in a signed agreement between the parties or in AMD\'s
Standard Terms and Conditions of Sale.

### Notices and attribution

© 2025 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD
Arrow logo, Instinct, Radeon Instinct, ROCm, and combinations thereof
are trademarks of Advanced Micro Devices, Inc.

Docker and the Docker logo are trademarks or registered trademarks of
Docker, Inc. in the United States and/or other countries. Docker, Inc.
and other parties may also have trademark rights in other terms used
herein. Linux® is the registered trademark of Linus Torvalds in the U.S.
and other countries.    

All other trademarks and copyrights are property of their respective
owners and are only mentioned for informative purposes.   


## Changelog
----------
This release note summarizes notable changes since the previous docker release.

12/09 release:
- Improved performance on Llama 3 MXFP4 due to AITER updates + kernel fusions

11/05 release:
- Turned AITER on by default
- Fixed Qwen 3 235B rms_norm segfault issue
- Known performance drop on Llama 4 models due to upstream vLLM bug https://github.com/vllm-project/vllm/issues/26320

10/06 release:
- ROCm 7.0 Docker image with MI350X/MI355X support
- Added support and benchmark instructions for:
    * Llama 4 Scout and Maverick
    * Deepseek-R1-0528
    * Llama 3.3 70B MXFP4 amd Llama 3.1 405B MXFP4 (MI35x only)
    * gpt-oss 20b and 120b
    * Qwen 3 32B, 30B-A3B, and 235B-A22B
- Dropped deprecated `--max-seq-len-to-capture` flag
- Bump --gpu-memory-utilization to 0.9 and make it configurable in config csv

## Support 
----------
You can report bugs through our GitHub [issue tracker](https://github.com/ROCm/MAD/issues).
