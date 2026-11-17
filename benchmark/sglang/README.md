# LLM inference performance validation with SGLang on AMD Instinct accelerators

## Overview 🎉
--------

SGLang is a large language model (LLM) inference and serving engine. 

MAD builds its SGLang image from [docker/pyt_sglang.ubuntu.amd.Dockerfile](../../docker/pyt_sglang.ubuntu.amd.Dockerfile)
on top of `lmsysorg/sglang:v0.5.18-rocm720-mi35x`, which includes:

-   ✅ ROCm™ 7.2.0
-   ✅ SGLang 0.5.18

Three model families build from their own Dockerfiles instead, because their support landed in a
different SGLang ROCm build:

| Models | Dockerfile | Base image |
| ------ | ---------- | ---------- |
| DeepSeek-V4 (`pyt_sglang_deepseek-v4-*`) | [docker/pyt_sglang_dsv4.ubuntu.amd.Dockerfile](../../docker/pyt_sglang_dsv4.ubuntu.amd.Dockerfile) | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` |
| FLUX (`pyt_sglang_flux.*`) | [docker/pyt_sglang_flux.ubuntu.amd.Dockerfile](../../docker/pyt_sglang_flux.ubuntu.amd.Dockerfile) | `lmsysorg/sglang-rocm:v0.5.15-rocm720-mi35x-20260712` |
| Kimi-K3 (`pyt_sglang_kimi-k3*`) | [docker/pyt_sglang_kimi_k3.ubuntu.amd.Dockerfile](../../docker/pyt_sglang_kimi_k3.ubuntu.amd.Dockerfile) | `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` |

>[!IMPORTANT]
>The default base is the MI350X/MI355X (gfx950) build. Its `sgl_kernel` ops are compiled for gfx950
>only and segfault on an MI300X (gfx942). For MI300X, build with
>`--build-arg BASE_DOCKER=<matching lmsysorg/sglang-rocm mi30x image>`, as described in the Dockerfile.

With this Docker image, users can quickly validate the expected inference performance numbers on AMD Instinct accelerators. 
This guide also provides tips and techniques so that users can get optimal performance with popular AI models.


## Reproducing benchmark results 🚀
-----------------------------

Use the following instructions to reproduce the benchmark results on an
AMD Instinct accelerator with a prebuilt SGLang Docker image.

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

The following command pulls the default Docker image from Docker Hub. For the DeepSeek-V4, FLUX and
Kimi-K3 models, pull the base image listed in the [Overview](#overview-) instead.

```sh
docker pull lmsysorg/sglang:v0.5.18-rocm720-mi35x
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

ROCm MAD launches a Docker container with the name `container_ci-pyt_sglang_deepseek-r1-distill-qwen-32b`. The benchmark configs come from
[scripts/sglang/configs/default.yaml](../../scripts/sglang/configs/default.yaml); entries for the same model in
[scripts/sglang/configs/perf.yaml](../../scripts/sglang/configs/perf.yaml) fill in any tuned `env` / `extra_args`
that the base config does not set. The reports of the model are collected in the following path:

```sh
~/MAD/perf_DeepSeek-R1-Distill-Qwen-32B.csv
```

Although the following models are pre-configured to collect latency and throughput performance data,
users can also change the benchmarking parameters. Refer to the [Standalone benchmarking](#standalone-benchmarking) section.

#### Available models

| model_name                              | Model                                    | Notes                                  |
| --------------------------------------- | ---------------------------------------- | -------------------------------------- |
| pyt_sglang_deepseek-r1-distill-qwen-32b | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |                                        |
| pyt_sglang_deepseek-r1                  | deepseek-ai/DeepSeek-R1                  |                                        |
| pyt_sglang_deepseek-v4-flash            | deepseek-ai/DeepSeek-V4-Flash            | gfx950 only                            |
| pyt_sglang_deepseek-v4-pro              | deepseek-ai/DeepSeek-V4-Pro              | gfx950 only                            |
| pyt_sglang_llama-3.3-70b                | meta-llama/Llama-3.3-70B-Instruct        |                                        |
| pyt_sglang_llama-3.3-70b_fp8            | amd/Llama-3.3-70B-Instruct-FP8-KV        |                                        |
| pyt_sglang_flux.1-dev                   | black-forest-labs/FLUX.1-dev             | gfx950 only; `-throughput` and `-latency` variants run those benchmarks instead of serving |
| pyt_sglang_flux.2-dev                   | black-forest-labs/FLUX.2-dev             | gfx950 only                            |
| pyt_sglang_wan2.2-t2v-a14b              | Wan-AI/Wan2.2-T2V-A14B-Diffusers         | `-throughput` and `-latency` variants run those benchmarks instead of serving |
| pyt_sglang_kimi-k3                      | moonshotai/Kimi-K3                       | gfx950 only; see the note below        |
| pyt_sglang_kimi-k3_dspark               | moonshotai/Kimi-K3                       | gfx950 only; see the note below        |

All entries except the Kimi-K3 ones read [scripts/sglang/configs/default.yaml](../../scripts/sglang/configs/default.yaml).
"gfx950 only" means the entry sets `skip_gpu_arch: gfx942`. The DeepSeek-V4 and FLUX entries build from
their own images; see the [Overview](#overview-).

>[!NOTE]
>`models.json` also defines two generic entries, `sglang-perf` and `sglang-accuracy`. They have empty `args` and
>write `perf_auto.csv`, because they are meant to be driven by a dispatcher that supplies `--model_repo`,
>`--config` and `MAD_OUTPUT_CSV`. `run.sh` infers the benchmark from the tag name: a name containing `-perf`
>runs `serving`, `-accuracy` runs `accuracy`, and anything else runs `all` (serving, then accuracy).
>Run on their own, these two entries fail because no model is given.

>[!NOTE]
>The two `pyt_sglang_kimi-k3*` entries are the exception to most of what is described above. They track the
>AMD day-0 recipes in [sgl-project/sglang#32548](https://github.com/sgl-project/sglang/issues/32548)
>(day-0 support: [#32541](https://github.com/sgl-project/sglang/pull/32541), see also the
>[SGLang K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)) and differ in
>four ways:
>
>- **Image.** They build from `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` via
>  [docker/pyt_sglang_kimi_k3.ubuntu.amd.Dockerfile](../../docker/pyt_sglang_kimi_k3.ubuntu.amd.Dockerfile),
>  not the shared `lmsysorg/sglang:v0.5.18-rocm720-mi35x` above. K3 support is not in a versioned
>  SGLang ROCm release yet.
>- **Benchmark.** They measure *online serving* only (`sglang.launch_server` + `sglang.bench_serving`)
>  through [scripts/sglang/run_kimi_k3.sh](../../scripts/sglang/run_kimi_k3.sh) and
>  [scripts/sglang/configs/kimi_k3.yaml](../../scripts/sglang/configs/kimi_k3.yaml). Each entry
>  selects one variant of that config (`nospec` or `dspark`) with `--variant`.
>- **Hardware.** 8x MI350X/MI355X (gfx950) TP8 only, hence `skip_gpu_arch: gfx942`. The checkpoint is
>  large, so make sure `HF_HUB_CACHE` has room; `pyt_sglang_kimi-k3_dspark` additionally pulls the
>  [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark) draft checkpoint.
>- **Invocation.** They carry the same `pyt` / `sglang` / `inference` tags as every other entry, so a
>  broad `--tags sglang` run includes them and will pull the checkpoint and hold 8 GPUs. To run them
>  on their own, select them by name:
>
>```sh
>madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output
>madengine run --tags pyt_sglang_kimi-k3_dspark --keep-model-dir --live-output
>```
>
>The sweep is 8192-token input / 1024-token output at concurrency 1/4/8/16/32/64/128/256, with
>`--random-range-ratio 1.0` so every request has exactly those lengths (other SGLang configs use the
>runner default of 0.8). The sweep covers the issue's rows except concurrency 2. The issue does not state
>its input and output lengths; they were recovered from its own tables, where
>`(E2EL - TTFT) / TPOT + 1` lands on ~1024 output tokens on every row and
>`concurrency x (inp + out) / E2EL` reproduces the reported total throughput only at 8192 input tokens.
>Keeping that shape is what makes MAD's numbers comparable to the issue's.
>
>The runner emits `--tensor-parallel-size` where the issue writes `--tp`. There is no `--tp` server
>argument; `tp_size` is `--tp-size` with the single alias `--tensor-parallel-size`, and `--tp` resolves
>only through argparse prefix matching, which a future `--tp*` option would silently break.

>[!WARNING]
>The published performance tables were measured on **MI355X**. A commenter on the tracking issue reports
>much weaker results on **MI350X** with untuned AITER kernels. Both report as gfx950, so `skip_gpu_arch`
>cannot distinguish them — treat MI350X numbers from this recipe as unvalidated.

### Standalone benchmarking              
-----------------------------

Users also can run the benchmark tool after they launch a Docker container.

```sh
docker pull lmsysorg/sglang:v0.5.18-rocm720-mi35x
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HF_HUB_CACHE=/workspace --name test lmsysorg/sglang:v0.5.18-rocm720-mi35x
```

Now clone the ROCm MAD repository inside the Docker image and move to the benchmark scripts directory at *~/MAD/scripts/sglang*. 

```sh
git clone https://github.com/ROCm/MAD
cd MAD/scripts/sglang
```

#### Command

```sh
python3 run_sglang.py --config $config --model $model_repo --benchmark $benchmark [filters]
```

`run_sglang.py` reads every entry in `$config` for `$model_repo`, expands the space-separated lists
(`tp`, `inp`, `out`, `bs`, `max_concurrency`, ...) into one run per combination, and writes the results
to `perf_<model name>.csv` in the current directory. If a `perf.yaml` sits next to `$config`, the runner
merges it in first (pass `--no-perf-merge` to skip this). `./run.sh --model_repo $model_repo --config $config`
is the wrapper that MAD calls; it also installs `lm-eval` and moves the CSV up one directory.

>[!NOTE]
>The input sequence length, output sequence length, tensor parallel (TP), and batch sizes or concurrencies
>come from the config file. Edit the config, or pass a filter such as `--inp 1024 --bs 8` to run a subset.

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
| $config      | configs/default.yaml                    | Benchmark configs (see [scripts/sglang/configs](../../scripts/sglang/configs/)) |
| $model_repo  | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|[DeepSeek-R1-Distill-Qwen 32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| $benchmark   | latency                                 | Measure decoding token latency (`sglang.bench_one_batch`) |
|              | throughput                              | Measure offline token generation throughput (`sglang.bench_offline_throughput`) |
|              | serving                                 | Measure online serving (`sglang.launch_server` + `sglang.bench_serving`) |
|              | accuracy                                | Run the accuracy evaluation (see `configs/acc.yaml`) |
|              | all                                     | Run every benchmark the config defines for the model |

#### Run the benchmark tests 🏃

Here are some examples:

- Benchmark example - latency

  Use this command to benchmark the latency of the DeepSeek-R1-Distill-Qwen 32B model on 8 GPUs with the bfloat16 data type.

  ```sh
  python3 run_sglang.py --config configs/default.yaml --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --benchmark latency
  ```

- Benchmark example - a single latency point

  Use the filters to run one combination from the sweep, for example input 1024, output 1024 and batch size 8.

  ```sh
  python3 run_sglang.py --config configs/default.yaml --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --benchmark latency --inp 1024 --out 1024 --bs 8
  ```

  The results of both examples are written to:

  - `./perf_DeepSeek-R1-Distill-Qwen-32B.csv`

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
