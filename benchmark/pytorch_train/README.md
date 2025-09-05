# Training Performance Validation with AMD Pytorch Docker on the AMD Instinct Accelerators

## Overview

PyTorch is an open-source machine learning framework that is widely used for model training with GPU-optimized components for transformer-based models.

The ROCm PyTorch Training Docker `rocm/pytorch-training:v25.7` container, available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html), provides a prebuilt, optimized environment for fine-tuning, pre-training a model on the AMD Instinct™ MI300X and MI325X accelerator. This ROCm PyTorch Docker includes the following components:

| Software component | Version              |
|--------------------|----------------------|
| ROCm               | 6.4.2 |
| Python             | 3.10.18              |
| PyTorch            | 2.8.0a0+gitd06a406   |
| Transformer Engine | 2.2.0.dev0+94e53dd8      |
| Flash Attention    | 3.0.0.post1          |
| hipBLASLt          | 1.1.0-4b9a52edfc     |
| Triton             | 3.3.0                |


## Models
Examples of the following models are pre-optimized for performance on the AMD Instinct MI300X and MI325X accelerator.
### Pre-training:
| Model          | Variants              |
|----------------|------------------------|
| **LLaMA 3.1**   | 8B, 70B         |
| **FLUX.1-dev**  | –                    |
### Finetuning:
| Model          | Variants              |
|----------------|------------------------|
| **GPT OSS**     | 20B, 120B           |
| **LLaMA 4**     | 17B_16E                    |
| **LLaMA 3.2 Vision** | 11B, 90B           |
| **LLaMA 3.2**   | 1B, 3B                 |
| **LLaMA 3.3**   | 70B                    |
| **LLaMA 3.1**   | 8B, 70B, 405B          |
| **LLaMA 3**     | 8B, 70B                |
| **LLaMA 2**     | 7B, 13B, 70B           |
| **Qwen 2**     | 1.5B, 7B           |
| **Qwen 2.5**     | 32B, 72B           |
| **Qwen 3**     | 8B, 32B           |


Please note that some models, such as Llama 3, require an external license agreement through a third party (e.g. Meta).

## System validation steps
If you have already validated your system, skip this step; otherwise, please complete the following [system validation and optimization steps](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/prerequisite-system-validation.html) to set up your system before starting training.

### Disable NUMA auto-balancing
Generally, application performance can benefit from disabling NUMA auto-balancing. However, it might be detrimental to performance with certain types of workloads.

Run the command `cat /proc/sys/kernel/numa_balancing` to check your current NUMA (Non-Uniform Memory Access) settings. Output `0` indicates this setting is disabled. If there is no output or the output is `1`, run the following command to disable NUMA auto-balancing.

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```
See [Disable NUMA auto-balancing](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-disable-numa) for more information.


## Start training on AMD Instinct accelerators

>[!NOTE]
>The only models supported in this workflow are those listed in the above section.
>

This container should not be expected to provide generalized performance across all training workloads. Users should expect the container perform in the model configurations described below, but other configurations and run conditions are not validated by AMD.
Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X and MI325X accelerators with the Docker image.

Use the following instructions to reproduce the benchmark results on an
MI300X accelerator with a prebuilt Pytorch Docker image.

Users have two choices to reproduce the benchmark results.

-   [MAD-integrated benchmarking](#mad-integrated-benchmarking)
-   [Standalone benchmarking](#standalone-benchmarking)

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
python3 tools/run_models.py --tags pyt_train_llama-3.1-8b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-pyt_train_llama-3.1-8b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/perf.csv
```

#### Available models

| model_name                              |
| --------------------------------------- |
| pyt_train_llama-2-7b                    |
| pyt_train_llama-2-13b                   |
| pyt_train_llama-2-70b                   |
| pyt_train_llama-3-8b                    |
| pyt_train_llama-3-70b                   |
| pyt_train_llama-3.1-8b                  |
| pyt_train_llama-3.1-70b                 |
| pyt_train_llama-3.1-405b                |
| pyt_train_llama-3.2-1b                  |
| pyt_train_llama-3.2-3b                  |
| pyt_train_llama-3.2-vision-11b          |
| pyt_train_llama-3.2-vision-90b          |
| pyt_train_llama-3.3-70b                 |
| pyt_train_llama-4-scout-17b-16e         |
| pyt_train_flux                          |
| pyt_train_gpt_oss_20b                   |
| pyt_train_gpt_oss_120b                  |
| pyt_train_qwen2-1.5b                    |
| pyt_train_qwen2-7b                      |
| pyt_train_qwen2.5-32b                   |
| pyt_train_qwen2.5-72b                   |
| pyt_train_qwen3-8b                      |
| pyt_train_qwen3-32b                     |

### Standalone benchmarking

### Download the Docker image and required packages
Use the following command to pull the Docker image from the Docker hub

```
docker pull rocm/pytorch-training:v25.7
```

Run the Docker container
```
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name training_env  rocm/pytorch-training:v25.7
```

Execute the training_env container (optional if no already in the container)
```
docker start training_env
docker exec -it training_env bash
```

### Clone the ROCm MAD repository
In the Docker container, clone the ROCm MAD repository and navigate to the benchmark scripts directory at /workspace/MAD/scripts/pytorch-training
```
git clone https://github.com/ROCm/MAD
cd MAD/scripts/pytorch-train
```

### Prepare training datasets and dependency
The following benchmarking examples may require downloading models and datasets from Hugging Face. To ensure successful access to gated repos, please set your `HF_TOKEN`
```
# pass your HF_TOKEN
export HF_TOKEN=$your_personal_hf_token
```
Run setup scripts to install libraries and datasets needed for benchmarking
```
./pytorch_benchmark_setup.sh
```
Following libraries will be installed with the script above:
|Benchmark Model     | Library       | Reference                                      |
|--------------------|---------------|--------------------------------------------------|
| Llama-3.1-8B, Flux | accelerate    | [Huggingface Accelerator](https://huggingface.co/docs/accelerate/en/index) |
| Llama-3.1-70B      | torchdata     | [TorchData](https://pytorch.org/data/beta/index.html) |
| Llama-3.1-8B,70B, Flux| datasets   | 3.2.0 [Datasets](https://huggingface.co/docs/datasets/en/index)|
| Llama-3.1-70B      | tomli         | [Tomli](https://pypi.org/project/tomli/) |
| Llama-3.1-70B, Flux| tensorboard   | 2.18.0 [TensorBoard](https://www.tensorflow.org/tensorboard) |
| Llama-3.1-70B      | tiktoken      | [tiktoken](https://github.com/openai/tiktoken) |
| Llama-3.1-70B      | blobfile      | [blobfile](https://pypi.org/project/blobfile/) |
| Llama-3.1-70B      | tabulate      | [tabulate](https://pypi.org/project/tabulate/) |
| Llama-3.1-70B      | wandb         | [W&B](https://github.com/wandb/wandb) |
| Llama-3.1-70B, Flux| sentencepiece | 0.2.0 [SentencePiece](https://github.com/google/sentencepiece) |
| Flux               | csvkit        | 2.0.1 [CSVKit](https://csvkit.readthedocs.io/en/latest/) |
| Flux               | deepspeed     | 0.16.2 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) |
| Flux               | diffusers     | 0.31.0 [Diffusers](https://huggingface.co/docs/diffusers/en/index) |
| Flux               | GitPython     | 3.1.44 [GitPython](https://github.com/gitpython-developers/GitPython) |
| Flux               | opencv-python-headless |4.10.0.84 [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) |
| Flux               | peft          | 0.14.0 [PEFT](https://huggingface.co/docs/peft/en/index) |
| Flux               | protobuf      | 5.29.2 [protobuf](https://github.com/protocolbuffers/protobuf) |
| Flux               | pytest        | 8.3.4 [PyTest](https://docs.pytest.org/en/stable/) |
| Flux               | python-dotenv | 1.0.1 [python-dotenv](https://pypi.org/project/python-dotenv/) |
| Flux               | seaborn       | 0.13.2 [seaborn](https://seaborn.pydata.org/) |
| Flux               | transformers  | 4.47.0 [Transformers](https://huggingface.co/docs/transformers/en/index) |

Following Models will be downloaded from Huggingface
* [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
* [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)

Following Datasets will be downloaded
* [bghira/pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k)

### Benchmarking Command
#### Pretraining
To start the pretraining benchmark, use the following command with the appropriate options. See the list of options and their descriptions below.

<pre lang="markdown"> ./pytorch_benchmark_report.sh -t $training_mode -m $model_repo -p $datatype -s $sequence_length </pre>

> ⚠️ **Note on Flux 2 Model Support**
>
> Currently, Flux models are **not supported out-of-the-box** on `rocm/pytorch-training:v25.7`.
>
> ✅ **Solution:** To use Flux, please refer to the image: `rocm/pytorch-training:v25.6`.
>
> 📄 **Documentation Guide:**  
> [ROCm PyTorch Training Docker Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html)


##### 🧩 Pretraining Configuration Optionss
|Name               | Options        | Description                                      |
|--------------------|---------------|--------------------------------------------------|
| $training_mode    | pretrain       | Benchmark pretraining                  |
| $datatype        | FP8 or BF16    | Currently, only Llama 3.1 8B example supports FP8 precision |
| $model_repo       | Llama-2-70B   | [Llama 2 70B](https://github.com/meta-llama/llama-models/tree/main/models/llama2)            |
|                  | Llama-3.1-8B  | [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B)            |
|                  | Llama-3.1-70B  | [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B)            |
|                  | Llama-3.3-70B | [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)      |
|                  | Flux           | [Flux.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| $sequence_length  | Sequence length for language model | Between 2048 and 8192 (default 8192) |

>[!NOTE]
>Occasionally, downloading the flux dataset may fail. In the event of this error, manually download the flux dataset from Hugging Face [black-forest-labs/FLUX.1-dev · Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev). Once downloaded, save it to '/workspace/FluxBenchmark' to ensure that the test script can access and utilize the dataset appropriately.

```
raise ReadTimeoutError(urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)
```


#### Finetuning
To start the finetuning benchmark, use the following command.
<pre lang="markdown"> ./pytorch_benchmark_report.sh -t $training_mode -m $model_repo -p $datatype  </pre>

##### 🧩 Finetuning Configuration Options

| Name               | Options           | Description                                                                 |
|--------------------|-------------------|-----------------------------------------------------------------------------|
| `$training_mode`   | finetune_fw       | Full-weight finetuning (BF16 supported)                                     |
|                    | finetune_lora     | LoRA finetuning (BF16 supported)                                            |
|                    | finetune_qlora    | qLoRA finetuning (BF16 supported)                                           |
|                    | HF_finetune_lora  | LoRA finetuning using Huggingface PEFT                                      |
| `$datatype`        | FP8 or BF16       |     All models support BF16; FP8 is only available for full-weight fine-tuning         |
| `$model_repo`      | Llama-4-17B_16E    | [Llama 4 Scout 17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E )      |
|                    | Llama-3.3-70B      | [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)                        |
|                    | Llama-3.2-Vision-90B| [Llama 3.2 90B Vision](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)   |
|                    | Llama-3.2-Vision-11B|  [Llama 3.2 11B Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)  |
|                    | Llama-3.2-3B       | [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B)                                                   |
|                    | Llama-3.2-1B       | [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B)                                                               |
|                    | Llama-3.1-405B     | [Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B)                                        |
|                    | Llama-3.1-70B      | [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B)                        |
|                    | Llama-3.1-8B       | [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B)                         |
|                    | Llama-3-70B        | [Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)                                                 |
|                    | Llama-3-8B         |[Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)                                                   |
|                    | Llama-2-70B        | [Llama 2 70B](https://github.com/meta-llama/llama-models/tree/main/models/llama2) |
|                    | Llama-2-13B        | [Llama 2 13B](https://github.com/meta-llama/llama-models/tree/main/models/llama2) |
|                    | Llama-2-7B         | [Llama 2 7B](https://github.com/meta-llama/llama-models/tree/main/models/llama2) |
|                    | GPT-OSS-20B        | [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) |
|                    | GPT-OSS-120B        | [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-120b) |
|                    | Qwen2-1.5B        | [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) |
|                    | Qwen2-7B        | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) |
|                    | Qwen2.5-32B        | [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) |
|                    | Qwen2.5-72B        | [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
|                    | Qwen3-32B        | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
|                    | Qwen3-8B        | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| `$sequence_length` | 2048 – 16384       | Sequence length for the language model                    |


##### Finetuning Support Matrix

| Model Name           | finetune_fw | finetune_lora | finetune_qlora | 
|----------------------|-------------|----------------|----------------|
| Llama 3.1 70B        | ✅           | ✅              | ✅              |
| Llama 3.1 8B         | ✅           | ✅              | ❌              |
| Llama 3.1 405B       | ❌           | ❌              | ✅              |
| Llama 3.3 70B        | ✅           | ✅              | ✅              |
| Llama 3 70B          | ✅           | ✅              | ❌              |
| Llama 3 8B           | ✅           | ✅              | ❌              |
| Llama 3.2 3B         | ✅           | ✅              | ❌              |
| Llama 3.2 1B         | ✅           | ✅              | ❌              |
| Llama 3.2 Vision 11B | ✅           | ❌             | ❌             |
| Llama 3.2 Vision 90B | ✅           | ❌              | ❌             |
| Llama 2 70B          | ❌           | ✅              | ✅              |
| Llama 2 13B          | ✅           | ✅              | ❌              |
| Llama 2 7B           | ✅           | ✅              | ✅              |
| Llama 4 17B_16E (scout)  | ✅           | ✅              | ❌              |
| GPT-OSS-20B   | ❌           | ✅              | ❌              |
| GPT-OSS-120B   | ❌           | ✅              | ❌              |
| Qwen2-1.5B   | ✅             | ✅              | ❌              |
| Qwen2-7B   | ✅            | ✅              | ❌              |
| Qwen2.5-72B   | ❌           | ✅              | ❌              |
| Qwen2.5-32B   | ❌           | ✅              | ❌              |
| Qwen3-8B   | ✅            | ✅              | ❌              |
| Qwen3-32B   | ❌           | ✅              | ❌              |


> ℹ️ **Note on Finetuning Support Matrix**
>
> In the table above, a **❌** indicates that the **upstream [`torchtune`](https://github.com/pytorch/torchtune)** repository does **not currently provide YAML configuration files** for that specific finetuning method and model combination.
>
> ✅ Users can still **easily configure** your own YAML files to enable support for these cases by following existing patterns under the **`/workspace/torchtune/recipes/configs/`** directory.
> 
> GPT-OSS models are supported using [HuggingFace PEFT](https://huggingface.co/docs/peft/en/index).
>
> Reference examples for Qwen models have been included, and other variants can be readily tested and configured through MAD.

##### Torchtune
> 📌 **Benchmark Setup Note**
>
> - All other LLaMA models are evaluated using **`alpaca_dataset`**
>
> - ✅ For vision models (11B and 90B) with LoRA and QLoRA support, use the following `torchtune` commit for compatibility:
  ```bash
  git checkout 48192e23188b1fc524dd6d127725ceb2348e7f0e
   ```
>
> ⚠️ **Note on LLaMA 2 Maximum Sequence Length**
> If you encounter an error like:
> `ValueError: seq_len (16384) of input tensor should be smaller than max_seq_len (4096)`
> It means your input sequence exceeds the allowed limit.
>
> ✅ **Solution:** Make sure your tokenized input is **≤ 4096 tokens**.
> You may need to truncate or split longer sequences before passing them to the model.
>
> 🧪 Results will be based on commit **`b4c98ac2a37f0397d64c22579aed415ce7264db6`** from the upstream [**torchtune**](https://github.com/pytorch/torchtune) repository for **reproducibility**.
> Users can also clone and use the **latest upstream version** to obtain **updated results**.


```
./pytorch_benchmark_report.sh -t {finetune_fw, finetune_lora} -p BF16 -m Llama-4-scout-17B-16E
```
```
./pytorch_benchmark_report.sh -t {finetune_fw, finetune_lora} -p BF16 -m Llama-3.2-3B
```
```
./pytorch_benchmark_report.sh -t {finetune_fw, finetune_lora, finetune_qlora} -p BF16 -m Llama-3.3-70B
```

##### Huggingface PEFT
Following example will run the benchmarking example of GPT-OSS models with [Ultra chat dataset](https://huggingface.co/datasets/smangrul/ultrachat-10k-chatml) using [HuggingFace PEFT](https://huggingface.co/docs/peft/en/index)

```
./pytorch_benchmark_report.sh -t HF_finetune_lora -p BF16 -m GPT-OSS-20B
```

### Benchmarking examples

Example 1: Llama 3.1 70B with BF16 precision with [Torchititan](https://github.com/ROCm/torchtitan)
Use this command to run a benchmark of the Llama 3.1 70B model.
```
./pytorch_benchmark_report.sh -t pretrain -p BF16 -m Llama-3.1-70B -s 8192
```

Example 2: Llama 3.1 8B with FP8 precision using transformer engine (TE) and [Huggingface Accelerator](https://huggingface.co/docs/accelerate/en/index)
```
./pytorch_benchmark_report.sh -t pretrain -p FP8 -m Llama-3.1-8B -s 8192
```

Example 3: Flux.1 Dev with BF16 precision with [FluxBenchmark](https://github.com/ROCm/FluxBenchmark)
```
./pytorch_benchmark_report.sh -t pretrain -p BF16 -m Flux
```

Example 4: Torchtune full weight finetuning with Llama 3.1 70B
```
./pytorch_benchmark_report.sh -t finetune_fw -p BF16 -m Llama-3.1-70B
```

Example 5: Torchtune LoRA finetuning with Llama 3.1 70B
```
./pytorch_benchmark_report.sh -t finetune_lora -p BF16 -m Llama-3.1-70B
```

Example 6: Torchtune full weight finetuning with Llama 3.3 70B
```
./pytorch_benchmark_report.sh -t finetune_fw -p BF16 -m Llama-3.3-70B
```

Example 7: Torchtune LoRA finetuning with Llama 3.3 70B
```
./pytorch_benchmark_report.sh -t finetune_lora -p BF16 -m Llama-3.3-70B
```

Example 8: Torchtune qLoRA finetuning with Llama 3.3 70B
```
./pytorch_benchmark_report.sh -t finetune_qlora -p BF16 -m Llama-3.3-70B
```

Example 9: Torchtune full weight finetuning with Llama 3.2 Vision 11B
```
./pytorch_benchmark_report.sh -t finetune_fw -p BF16 -m Llama-3.2-vision-11B
```

Example 10: Torchtune full weight finetuning with Llama 3.2 Vision 90B
```
./pytorch_benchmark_report.sh -t finetune_fw -p BF16 -m Llama-3.2-vision-90B
```

Example 11: Torchtune full weight finetuning with Llama 4 17B_16E
```
./pytorch_benchmark_report.sh -t finetune_fw -p BF16 -m Llama-4-scout-17B-16E
```

Example 12: Torchtune LoRA finetuning with Llama 4 17B_16E
```
./pytorch_benchmark_report.sh -t finetune_lora -p BF16 -m Llama-4-scout-17B-16E
```

Example 13: Huggingface PEFT LoRA finetuning with GPT-OSS-120B
```
./pytorch_benchmark_report.sh -t HF_finetune_lora -p BF16 -m GPT-OSS-120B
```

Example 14: Torchtune full weight finetuning with Llama 3.1 70B using FP8
```
./pytorch_benchmark_report.sh -t finetune_fw -p FP8 -m Llama-3.1-70B
```
### Multinode Training with Torchtitan

Our framework supports multinode training with Torchtitan. To launch training on a SLURM cluster for the Llama3-70B model (adjust the `*.toml` configuration inside the slurm script if you’re using a different model), run:

```bash
cd scripts/pytorch_train
sbatch run_slurm_train.sh
```

### Multinode Training with Torchtune
Our framework supports multinode training with Torchtune. To launch training on a SLURM cluster for the Llama3.3-70B model, run:
> 📌 **Benchmark Setup Note**
>
> - By default the Llama3.3-70B model is finetuned using **`alpaca_dataset`**
> - Adjust the `*.[**yaml**](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_3/70B_full_multinode.yaml)` configuration inside the upstream [**torchtune**](https://github.com/pytorch/torchtune) if you’re using a different model
> - Number of nodes, and all parameters can be tuned form the slurm script **`Torchtune_Multinode.sh`**
> - Set the `mounting paths` inside the slurm script.

```bash
huggingface-cli login # Get access to HF llama model space
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./models/Llama-3.3-70B-Instruct # Download the llama 3.3 model locally
cd scripts/pytorch_train
sbatch Torchtune_Multinode.sh
```
**Note:** After the run is finished, the log files will be there `result_torchtune` directory
