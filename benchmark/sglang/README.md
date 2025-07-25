# LLM inference performance validation with SGLang on the AMD Instinct MI300X accelerator

## Overview 🎉
--------

SGLang is a large language model (LLM) inference and serving engine. 

This Docker image packages SGLang with PyTorch for an AMD Instinct™ MI300X
accelerator. It includes:

-   ✅ ROCm™ 6.3.0
-   ✅ SGLang 0.4.5 (0.4.5-rocm)
-   ✅ PyTorch 2.6.0 (2.6.0a0+git8d4926e)

With this Docker image, users can quickly validate the expected inference performance numbers on the MI300X accelerator. 
This guide also provides tips and techniques so that users can get optimal performance with popular AI models.


## Reproducing benchmark results 🚀
-----------------------------

Use the following instructions to reproduce the benchmark results on an
MI300X accelerator with a prebuilt SGLang Docker image.

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

### Download the Docker image 🐳

The following command pulls the Docker image from Docker Hub.

```sh
docker pull lmsysorg/sglang:v0.4.5-rocm630
```

### MAD-integrated benchmarking

Clone the ROCm Model Automation and Dashboarding (MAD) repository to a local directory and install the required packages on the host machine.

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt
```

Use this command to run a performance benchmark test of the DeepSeek-R1-Distill-Qwen 32B model on 8 GPUs with bfloat16 data type in the host machine. 

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
madengine run --tags pyt_sglang_deepseek-r1-distill-qwen-32b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-pyt_sglang_deepseek-r1-distill-qwen-32b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/perf_DeepSeek-R1-Distill-Qwen-32B.csv
```

Although the following models are pre-configured to collect latency and throughput performance data,
users can also change the benchmarking parameters. Refer to the [Standalone benchmarking](#standalone-benchmarking) section.

#### Available models

| model_name                              |
| --------------------------------------- |
| pyt_sglang_deepseek-r1-distill-qwen-32b  |

### Standalone benchmarking              
-----------------------------

Users also can run the benchmark tool after they launch a Docker container.

```sh
docker pull lmsysorg/sglang:v0.4.5-rocm630
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HUGGINGFACE_HUB_CACHE=/workspace --name test lmsysorg/sglang:v0.4.5-rocm630
```

Now clone the ROCm MAD repository inside the Docker image and move to the benchmark scripts directory at *~/MAD/scripts/sglang*. 

```sh
git clone https://github.com/ROCm/MAD
cd MAD/scripts/sglang
```

#### Command

```sh
./sglang_benchmark_report.sh -s $test_option -m $model_repo -g $num_gpu -d $datatype [-a $dataset]
```

>[!NOTE]
>The input sequence length, output sequence length, and tensor parallel (TP) are already configured. You don't need to specify them with this script.

>[!NOTE]
>If you encounter this error, pass your access-authorized Hugging Face token to the gated models.
>```sh
>OSError: You are trying to access a gated repo.
>
># pass your HF_TOKEN
>export HF_TOKEN=$your_personal_hf_token
>```

#### Variables

| Name         | Options                                 | Description                                      |
| ------------ | --------------------------------------- | ------------------------------------------------ |
| $test_option | latency                                 | Measure decoding token latency                   |
|              | throughput                              | Measure token generation throughput              |
|              | all                                     | Measure both throughput and latency              |
| $model_repo  |                                         |                                                  |
| (bfloat16)   | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|[DeepSeek-R1-Distill-Qwen 32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| $num_gpu     |      8                                  | Number of GPUs                                   |
| $datatype    | bfloat16                                | Data type                                        |
| $dataset     | random                                  | Dataset                                          |

#### Run the benchmark tests on the MI300X accelerator 🏃

Here are some examples and the test results:

- Benchmark example - latency

  Use this command to benchmark the latency of the DeepSeek-R1-Distill-Qwen 32B model on 8 GPUs with the bfloat16 data type.

  ```sh
  ./sglang_benchmark_report.sh -s latency -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -g 8 -d bfloat16
  ```

  The latency reports are available at:

  - `./reports_bfloat16/summary/DeepSeek-R1-Distill-Qwen-32B_latency_report.csv`

- Benchmark example - throughput

  Use this command to benchmark the throughput of the DeepSeek-R1-Distill-Qwen 32B model on 8 GPUs with the bfloat16 data type.

  ```sh
  ./sglang_benchmark_report.sh -s throughput -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -g 8 -d bfloat16 -a random
  ```

  The throughput reports are available at:

  - `./reports_bfloat16/summary/DeepSeek-R1-Distill-Qwen-32B_throughput_report.csv`

>[!NOTE]
>Throughput is calculated as:
>-   `throughput_tot = requests * (input lengths + output lengths) / elapsed_time`
>-   `throughput_gen = requests * output lengths / elapsed_time`

## References 🔎
----------

To learn more about the options for latency and throughput
benchmark scripts, see
<https://github.com/sgl-project/sglang/tree/main/benchmark/blog_v0_2>.

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
| SGLang    | Apache License 2.0                             | [SGLang License](https://github.com/sgl-project/sglang/blob/main/LICENSE)  |

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
This release note summarizes notable changes since the previous docker release (April 28, 2025).
-   The SGLang version number was incremented to 0.4.5.
-   The bfloat16 data type benchmark test was added to include the following models: DeepSeek-R1-Distill-Qwen 32B.



## Support 
----------
You can report bugs through our GitHub [issue tracker](https://github.com/ROCm/MAD/issues).
