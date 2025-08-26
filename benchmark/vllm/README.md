# LLM inference performance validation with vLLM on the AMD Instinct MI300X accelerator

## Overview 🎉
--------

vLLM is a toolkit and library for large language model (LLM) inference and serving. It
deploys the PagedAttention algorithm, which reduces memory consumption
and increases throughput by leveraging dynamic key and value allocation
in GPU memory. vLLM also incorporates many recent LLM acceleration and
quantization algorithms. In addition, AMD implements high-performance custom
kernels and modules in vLLM to enhance performance further.

This Docker image packages vLLM with PyTorch for an AMD Instinct™ MI300X
accelerator. It includes:

-   ✅ ROCm™ 6.4.1
-   ✅ vLLM 0.10.0 (0.10.1.dev395+g340ea86df.rocm641)
-   ✅ PyTorch 2.7.0 (2.7.0+gitf717b2a)
-   ✅ hipBLASLt 0.15

With this Docker image, users can quickly validate the expected inference performance numbers on the MI300X accelerator. 
This guide also provides tips and techniques so that users can get optimal performance with popular AI models.


## Reproducing benchmark results 🚀
-----------------------------

Use the following instructions to reproduce the benchmark results on an
MI300X accelerator with a prebuilt vLLM Docker image.

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

### Download the Docker image 🐳

The following command pulls the Docker image from Docker Hub.

```sh
docker pull rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812
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
madengine run --tags pyt_vllm_llama-3.1-8b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-pyt_vllm_llama-3.1-8b`. The throughput and serving reports of the model are collected in the following files: 
`pyt_vllm_llama_3.1-8b_throughput.csv`
`pyt_vllm_llama_3.1-8b_serving.csv`

Although the following models are pre-configured to collect offline throughput and online serving performance data,
users can also change the benchmarking parameters. Refer to the [Standalone benchmarking](#standalone-benchmarking) section.

#### Available models

| model_name                             |
| -------------------------------------- |
| pyt_vllm_llama-3.1-8b                  |
| pyt_vllm_llama-3.1-70b                 |
| pyt_vllm_llama-3.1-405b                |
| pyt_vllm_llama-2-70b                   |
| pyt_vllm_mixtral-8x7b                  |
| pyt_vllm_mixtral-8x22b                 |
| pyt_vllm_qwq-32b                       |
| pyt_vllm_llama-3.1-8b_fp8              |
| pyt_vllm_llama-3.1-70b_fp8             |
| pyt_vllm_llama-3.1-405b_fp8            |
| pyt_vllm_mixtral-8x7b_fp8              |
| pyt_vllm_mixtral-8x22b_fp8             |
| pyt_vllm_phi-4                         |


### Standalone benchmarking              
-----------------------------

Users also can run the benchmark tool after they launch a Docker container.

```sh
docker pull rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812

docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HUGGINGFACE_HUB_CACHE=/workspace --name test rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812
```

Now clone the ROCm MAD repository inside the Docker image and move to the benchmark scripts directory at *~/MAD/scripts/vllm*. 

```sh
git clone https://github.com/ROCm/MAD
cd MAD/scripts/vllm
```

#### Command

```sh
./run.sh --config $CONFIG_CSV --model_repo $MODEL_REPO ... {overrides}
```

>[!NOTE]
>If you encounter this error, pass your access-authorized Hugging Face token to the gated models.
>```sh
>OSError: You are trying to access a gated repo.
>
># pass your HF_TOKEN
>export HF_TOKEN=$your_personal_hf_token
>```

>[!NOTE]
>We currently recommend running with `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1` for best performance.

#### Variables

| Name         | Options                                 | Description                                      |
| ------------ | --------------------------------------- | ------------------------------------------------ |
| $config      | configs/default.csv                     | Run configs from the CSV matching the model repo and benchmark |
|              | configs/extended.csv                    |                                 |
|              | configs/performance.csv                 |                                 |
| $benchmark   | throughput                              | Measure offline end-to-end throughput              |
|              | serving                                 | Measure online serving performance             |
|              | all                                     | Measure both offline throughput and online serving |
| $model_repo  | meta-llama/Llama-3.1-8B-Instruct   | [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| (float16)    | meta-llama/Llama-3.1-70B-Instruct  | [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)                            |
|              | meta-llama/Llama-3.1-405B-Instruct | [Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)                           |                 |
|              | meta-llama/Llama-2-70b-chat-hf          | [Llama 2 70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)                               |
|              | mistralai/Mixtral-8x7B-Instruct-v0.1    | [Mixtral MoE 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)                         |
|              | mistralai/Mixtral-8x22B-Instruct-v0.1   | [Mixtral MoE 8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)                        |
|              | Qwen/QwQ-32B                            | [QwQ 32B](https://huggingface.co/Qwen/QwQ-32B)                                                      |
|              | microsoft/phi-4                         | [Phi-4](https://huggingface.co/microsoft/phi-4)                                                    |
| $model_repo  | amd/Llama-3.1-8B-Instruct-FP8-KV   | [Llama 3.1 8B](https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV)                            |
| (float8)     | amd/Llama-3.1-70B-Instruct-FP8-KV  | [Llama 3.1 70B](https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV)                            |
|              | amd/Llama-3.1-405B-Instruct-FP8-KV | [Llama 3.1 405B](https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV)                           |
|              | amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV   | [Mixtral MoE 8x7B](https://huggingface.co/amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV)                        |
|              | amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV  | [Mixtral MoE 8x22B](https://huggingface.co/amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV)                       |
|              | amd/Mistral-7B-v0.1-FP8-KV              | [Mistral 7B](https://huggingface.co/amd/Mistral-7B-v0.1-FP8-KV)                                   |
| overrides    | See [run.sh](../../scripts/vllm/run.sh)  | Additional overrides to the config CSV |

#### Run the benchmark tests on the MI300X accelerator 🏃

Here are some examples and the test results:

- Benchmark example - throughput

  Use this command to benchmark the throughput of the Llama 3.1 70B model on 8 GPUs with the float16 and float8 data type.

  ```sh
  export MAD_MODEL_NAME=pyt_vllm_llama-3.1-70b
  ./run.sh --config configs/default.csv --model_repo meta-llama/Llama-3.1-70B-Instruct --benchmark throughput
  export MAD_MODEL_NAME=pyt_vllm_llama-3.1-70b_fp8
  ./run.sh -s --config configs/default.csv --model_repo amd/Llama-3.1-70B-Instruct-FP8-KV --benchmark throughput
  ```

  The throughput reports are available at:

  - `./pyt_vllm_llama-3.1-70b_throughput.csv`
  - `./pyt_vllm_llama-3.1-70b_fp8_throughput.csv`

- Benchmark example - serving

  Use this command to benchmark the serving of the Llama 3.1 70B model on 8 GPUs with the float16 and float8 data type.

  ```sh
  export MAD_MODEL_NAME=pyt_vllm_llama-3.1-70b
  ./run.sh --config configs/default.csv --model_repo meta-llama/Llama-3.1-70B-Instruct --benchmark serving
  export MAD_MODEL_NAME=pyt_vllm_llama-3.1-70b_fp8
  ./run.sh -s --config configs/default.csv --model_repo amd/Llama-3.1-70B-Instruct-FP8-KV --benchmark serving
  ```

  The serving reports are available at:

  - `./pyt_vllm_llama-3.1-70b_serving.csv`
  - `./pyt_vllm_llama-3.1-70b_fp8_serving.csv`

>[!NOTE]
>Throughput is calculated as:
>-   `throughput_tot = requests * (input lengths + output lengths) / elapsed_time`
>-   `throughput_gen = requests * output lengths / elapsed_time`

## References 🔎
----------

For an overview of the optional performance features of vLLM with
ROCm software, see [ROCm performance](https://github.com/ROCm/vllm/blob/main/ROCm_performance.md).

To learn more about the options for the offline throughput and online serving
benchmark scripts, see
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

- Add additional environment and benchmark overrides; the full list can be seen in [the run script](../../scripts/vllm/run.sh)
- Removed deprecated models (Llama 2 7B, Mistral 7B, Qwen 2 7B, Qwen 2 72B, Gemma 2 27B, DeepSeek 16B MoE, DBRX Instruct, Falcon 180B)
- Updated run script to use config CSVs and added [default.csv](../../scripts/vllm/configs/default.csv), [extended.csv](../../scripts/vllm/configs/extended.csv), and [performance.csv](../../scripts/vllm/configs/performance.csv) to support various models
- Soft-deprecated offline latency benchmark in favor of online serving
- AITER now supports FP8 KV cache

## Support 
----------
You can report bugs through our GitHub [issue tracker](https://github.com/ROCm/MAD/issues).
