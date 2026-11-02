# LLM inference performance validation with ATOM on AMD Instinct accelerators

## Overview 🎉
--------

[ATOM](https://github.com/ROCm/ATOM) is AMD's lightweight LLM inference and serving
engine for AMD Instinct accelerators. It is built on [AITER](https://github.com/ROCm/aiter) kernels and exposes an
OpenAI-compatible server (`atom.entrypoints.openai_server`).

MAD builds its ATOM image from [docker/pyt_atom.ubuntu.amd.Dockerfile](../../docker/pyt_atom.ubuntu.amd.Dockerfile)
on top of `rocm/atom-dev:latest`, which includes:

-   ✅ ROCm™
-   ✅ ATOM
-   ✅ AITER
-   ✅ PyTorch

>[!NOTE]
>`rocm/atom-dev:latest` is a rolling development image, so the component versions change from build to build.
>Run `pip3 list` inside the container (the Dockerfile also prints it at build time) to see what you got.

With this Docker image, users can quickly validate the expected inference performance numbers on AMD Instinct accelerators.
This guide also provides tips and techniques so that users can get optimal performance with popular AI models.

>[!NOTE]
>This README covers the standalone ATOM engine in [scripts/atom](../../scripts/atom/). Running vLLM with the
>ATOM plugin backend is a separate path; see [scripts/atom_vllm/README.md](../../scripts/atom_vllm/README.md).


## Reproducing benchmark results 🚀
-----------------------------

Use the following instructions to reproduce the benchmark results on an
AMD Instinct accelerator with a prebuilt ATOM Docker image.

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
docker pull rocm/atom-dev:latest
```

### MAD-integrated benchmarking

Clone the ROCm Model Automation and Dashboarding (MAD) repository to a local directory and install the required packages on the host machine.

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt
```

Use this command to run a performance benchmark test of the Llama 3.3 70B FP8 model on 8 GPUs in the host machine.

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
madengine run --tags pyt_atom_llama-3.3-70b_fp8 --keep-model-dir --live-output
```

ROCm MAD launches a Docker container with the name `container_ci-pyt_atom_llama-3.3-70b_fp8`. The benchmark configs come from
[scripts/atom/configs/default.yaml](../../scripts/atom/configs/default.yaml); entries for the same model in
[scripts/atom/configs/perf.yaml](../../scripts/atom/configs/perf.yaml) fill in any tuned `env` / `extra_args`
that the base config does not set. The reports of the model are collected in the following path:

```sh
~/MAD/perf_Llama-3.3-70B-Instruct-FP8-KV.csv
```

Although the following models are pre-configured to collect serving performance data,
users can also change the benchmarking parameters. Refer to the [Standalone benchmarking](#standalone-benchmarking) section.

#### Available models

| model_name                 | Model                                    | Notes                                  |
| -------------------------- | ---------------------------------------- | -------------------------------------- |
| pyt_atom_llama-3.3-70b_fp8 | amd/Llama-3.3-70B-Instruct-FP8-KV        |                                        |
| pyt_atom_llama-3.3-70b_fp4 | amd/Llama-3.3-70B-Instruct-MXFP4-Preview | gfx950 only                            |
| pyt_atom_kimi-k3           | moonshotai/Kimi-K3                       | gfx950 only; see the note below        |

All three entries read [scripts/atom/configs/default.yaml](../../scripts/atom/configs/default.yaml) and run online
serving (`atom.entrypoints.openai_server` + `atom.benchmarks.benchmark_serving`) at TP8 with an FP8 KV cache.
"gfx950 only" means the entry sets `skip_gpu_arch: gfx942`.

| model_name                 | Input / output length | Max concurrency            |
| -------------------------- | --------------------- | -------------------------- |
| pyt_atom_llama-3.3-70b_fp8 | 1000, 4000, 10000 / 1000 | 128, 256                |
| pyt_atom_llama-3.3-70b_fp4 | 1000, 4000, 10000 / 1000 | 64, 128, 256            |
| pyt_atom_kimi-k3           | 8192 / 1024           | 1, 4, 8, 16, 32, 64, 128, 256 |

The two Llama entries also carry the `atom_default` tag, so `madengine run --tags atom_default` runs both of them.

>[!NOTE]
>`pyt_atom_kimi-k3` tracks the AMD day-0 recipe for Kimi-K3 (ROCm/ATOM `recipes/Kimi-K3.md`); its tuned server
>arguments and `ATOM_*` / `AITER_*` environment come from the Kimi-K3 entry in
>[scripts/atom/configs/perf.yaml](../../scripts/atom/configs/perf.yaml). It needs 8x MI350X/MI355X (gfx950) and
>a large checkpoint, so make sure `HF_HUB_CACHE` has room. It does not carry the `atom_default` tag,
>so run it explicitly by name:
>
>```sh
>madengine run --tags pyt_atom_kimi-k3 --keep-model-dir --live-output
>```
>
>See [benchmark/kimi_k3/README.md](../kimi_k3/README.md) for a side-by-side of the vLLM, SGLang and ATOM Kimi-K3 runs.

>[!NOTE]
>`models.json` also defines two generic entries, `atom-perf` and `atom-accuracy`. They have empty `args` and
>write `perf_auto.csv`, because they are meant to be driven by a dispatcher that supplies `--model_repo`,
>`--config` and `MAD_OUTPUT_CSV`. `run.sh` infers the benchmark from the tag name: a name containing `-perf`
>runs `serving`, `-accuracy` runs `accuracy`, and anything else runs `all` (serving, then accuracy).
>Run on their own, these two entries fail because no model is given.

#### Config files

| File | Purpose |
| ---- | ------- |
| [configs/default.yaml](../../scripts/atom/configs/default.yaml)   | The serving sweeps that the `pyt_atom_*` entries run |
| [configs/extended.yaml](../../scripts/atom/configs/extended.yaml) | Llama 3.1 8B (BF16 and FP8-KV) at TP1, 1024/1024, concurrency 1 |
| [configs/unittest.yaml](../../scripts/atom/configs/unittest.yaml) | Short serving and accuracy runs on Llama 3.1 8B with `--load_dummy` weights |
| [configs/perf.yaml](../../scripts/atom/configs/perf.yaml)         | Per-model tuned `extra_args` / `env`, taken from the [ATOM recipes](https://github.com/ROCm/ATOM/tree/main/recipes) |
| [configs/acc.yaml](../../scripts/atom/configs/acc.yaml)           | lm_eval accuracy settings: task-level defaults keyed by `hf_pipeline_tag`, plus per-model overrides |

A run config sets the shape of each run (`benchmark`, `hf_pipeline_tag`, `model`, `tp`, `inp`, `out`,
`kv_cache_dtype`, `max_concurrency`). The runner then fills in the rest:

- **perf.yaml.** For every run config whose model matches a `perf.yaml` entry, the runner adds the entry's
  `extra_args` and `env` keys that the run config does not already set. The run config always wins. The merged
  result is written to `perf_config.yaml` next to the config; the original file is never modified. Pass
  `--no-perf-merge` to skip this step, or `--perf <yaml>` to use a different file.
- **acc.yaml.** For accuracy runs, the runner first applies the task-level `evaluations` for the config's
  `hf_pipeline_tag` (for `text-generation`, 0-shot gsm8k through `local-completions`), then deep-merges the
  model's own overrides on top. The merged result is written to `acc_config.yaml`.
- **arch_overrides.** A run config can carry an `arch_overrides` map keyed by GPU architecture (for example
  `gfx950`). The block that matches `MAD_SYSTEM_GPU_ARCHITECTURE` is applied to that config.
- **Number of prompts.** Unless a config sets `num_prompts`, each serving run sends
  `num_prompts_multiplier x max_concurrency` prompts. The multiplier defaults to 10; a `perf.yaml` entry can
  change it with `bench_serving: {num_prompts_multiplier: N}` (DeepSeek-R1 uses 2).

### Standalone benchmarking
-----------------------------

Users also can run the benchmark tool after they launch a Docker container.

```sh
docker pull rocm/atom-dev:latest
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HF_HUB_CACHE=/workspace --name test rocm/atom-dev:latest
```

Now clone the ROCm MAD repository inside the Docker image and move to the benchmark scripts directory at *~/MAD/scripts/atom*.

```sh
git clone https://github.com/ROCm/MAD
cd MAD/scripts/atom
```

#### Command

```sh
python3 run_atom.py --config $config --model $model_repo --benchmark $benchmark [filters]
```

`run_atom.py` reads every entry in `$config` for `$model_repo`, expands the space-separated lists
(`tp`, `inp`, `out`, `num_prompts`, `max_concurrency`) into one run per combination, and writes the results
to `perf_<model name>.csv` in the current directory. If a `perf.yaml` sits next to `$config`, the runner
merges it in first (pass `--no-perf-merge` to skip this). `./run.sh --model_repo $model_repo --config $config`
is the wrapper that MAD calls; it also installs `lm-eval` and moves the CSV up one directory.

>[!NOTE]
>The input sequence length, output sequence length, tensor parallel (TP) size and concurrencies
>come from the config file. Edit the config, or pass a filter such as `--inp 1000 --max_concurrency 128` to run a subset.

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
| $config      | configs/default.yaml                    | Benchmark configs (see [scripts/atom/configs](../../scripts/atom/configs/)) |
| $model_repo  | amd/Llama-3.3-70B-Instruct-FP8-KV       | [Llama 3.3 70B FP8-KV](https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV) |
| $benchmark   | serving                                 | Measure online serving (`atom.entrypoints.openai_server` + `atom.benchmarks.benchmark_serving`) |
|              | accuracy                                | Run the lm_eval accuracy evaluation (see `configs/acc.yaml`) |
|              | all                                     | Run serving, then accuracy, for each config |
| filters      | `--tp`, `--inp`, `--out`, `--num_prompts`, `--max_concurrency` | Keep only the combinations that match |
| --profile    |                                         | Kernel-profiling run; see [Kernel profiling](#kernel-profiling) |

#### Run the benchmark tests 🏃

Here are some examples:

- Benchmark example - serving

  Use this command to run the full Llama 3.3 70B FP8 serving sweep on 8 GPUs.

  ```sh
  python3 run_atom.py --config configs/default.yaml --model amd/Llama-3.3-70B-Instruct-FP8-KV --benchmark serving
  ```

- Benchmark example - a single serving point

  Use the filters to run one combination from the sweep, for example input 1000, output 1000 and concurrency 128.

  ```sh
  python3 run_atom.py --config configs/default.yaml --model amd/Llama-3.3-70B-Instruct-FP8-KV --benchmark serving --inp 1000 --out 1000 --max_concurrency 128
  ```

- Benchmark example - accuracy

  Use this command to run the gsm8k accuracy evaluation defined in `configs/acc.yaml`.

  ```sh
  python3 run_atom.py --config configs/default.yaml --model amd/Llama-3.3-70B-Instruct-FP8-KV --benchmark accuracy --inp 1000 --max_concurrency 128
  ```

  The results of these examples are written to:

  - `./perf_Llama-3.3-70B-Instruct-FP8-KV.csv`

Each serving run adds six rows to the CSV: `throughput_tot` and `throughput_gen` in tok/sec, and the median
`ttft`, `tpot`, `itl` and `e2el` in ms. The `cmd` column records the exact server and client commands that ran.

>[!NOTE]
>Throughput is calculated as:
>-   `throughput_tot = requests * (input lengths + output lengths) / elapsed_time`
>-   `throughput_gen = requests * output lengths / elapsed_time`

#### Serving Command

`run_atom.py` runs the following commands for each combination. To reproduce one by hand:

1. Start the server
```
model=amd/Llama-3.3-70B-Instruct-FP8-KV
tp=8

python -m atom.entrypoints.openai_server \
    --model $model \
    -tp $tp \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 131072 \
    --max-model-len 8192

    # Wait for model to load and server is ready to accept requests
```

The arguments after `-tp` are the Llama 3.3 70B entry in [configs/perf.yaml](../../scripts/atom/configs/perf.yaml);
use the entry for your model.

2. On another terminal on the same machine, run the benchmark:
```
# Connect to the container
docker exec -it test bash

# Wait for the server to start
until curl -s http://localhost:8000/v1/models; do sleep 30; done

# Run the benchmark
model=amd/Llama-3.3-70B-Instruct-FP8-KV
in=1000
out=1000
max_concurrency=128
num_prompts=1280

python -m atom.benchmarks.benchmark_serving \
    --model $model \
    --backend vllm \
    --base-url http://localhost:8000 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --dataset-name random \
    --ignore-eos \
    --request-rate inf \
    --max-concurrency $max_concurrency \
    --num-prompts $num_prompts \
    --random-input-len $in \
    --random-output-len $out \
    --save-result \
    --result-dir ./ \
    --result-filename ${model##*/}_serving.json
```

#### Accuracy Command

The MAD accuracy runs (the `atom-accuracy` tag) are driven by [configs/acc.yaml](../../scripts/atom/configs/acc.yaml), which defines the lm_eval task, endpoint and generation settings per model. The steps below reproduce the default `text-generation` run by hand.

1. Start the server as in step 1 of the [Serving Command](#serving-command).

2. On another terminal on the same machine, run the evaluation:
```
# Connect to the container
docker exec -it test bash

# Wait for the server to start
until curl -s http://localhost:8000/v1/models; do sleep 30; done

# Install lm-eval
pip install lm-eval[api]

# Run the evaluation
model=amd/Llama-3.3-70B-Instruct-FP8-KV
lm_eval --model local-completions \
    --model_args model=$model,max_gen_toks=2048,num_concurrent=64,max_retries=3,base_url=http://localhost:8000/v1/completions,tokenized_requests=False \
    --tasks gsm8k --num_fewshot 0 --output_path ./tmp
```

Some models override these settings in `acc.yaml`; for example, gpt-oss-120b uses 3-shot and Kimi-K3 uses 5-shot.

#### Kernel profiling

Add `--profile` to `run.sh` or `run_atom.py` to turn each serving run into a kernel-profiling run. The server runs
under [rocm-trace-lite](https://github.com/amathews-amd/rocm-trace-lite) (`rtl trace --mode full`) with roctx
markers around each prefill, decode and mixed pass (see [scripts/atom/profiling](../../scripts/atom/profiling/)).
Each run writes `rtl_trace.db` under `profile_<model>_<tp>/` and aggregates it into
`./profile/kernel_summary_payload.json`. A profiling run sends at most 5 prompts unless the config sets `num_prompts`.

```sh
./run.sh --model_repo amd/Llama-3.3-70B-Instruct-FP8-KV --config configs/default.yaml --benchmark serving --profile
```

When `--profile` is passed, `run.sh` builds and installs rocm-trace-lite inside the container before it starts.

## References 🔎
----------

To learn more about ATOM, its supported models and its tuned recipes, see
<https://github.com/ROCm/ATOM> and <https://github.com/ROCm/ATOM/tree/main/recipes>.

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
| ATOM    | MIT License                                      | [ATOM License](https://github.com/ROCm/ATOM/blob/main/LICENSE) |

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

© 2026 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD
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
-   Initial ATOM benchmark guide: Llama 3.3 70B FP8 and MXFP4 serving, Kimi-K3 serving, lm_eval accuracy, and kernel profiling.


## Support
----------
You can report bugs through our GitHub [issue tracker](https://github.com/ROCm/MAD/issues).
