# Training Performance Validation with AMD Pytorch Docker on the AMD Instinct Accelerators

## Overview

PyTorch is an open-source machine learning framework that is widely used for model training with GPU-optimized components for transformer-based models.

The ROCm PyTorch Training Docker `rocm/pytorch-training:v25.3` container, available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html), provides a prebuilt, optimized environment for fine-tuning, pre-training a model on the AMD Instinct™ MI300X and MI325X accelerator. This ROCm PyTorch Docker includes the following components:

| Software component  | Version            |
|---------------------|--------------------|
| ROCm               | 6.3.0              |
| Python            | 3.10               |
| PyTorch           | 2.7.0a0+git637433   |
| Transformer Engine | 1.11               |
| Flash Attention   | 3.0.0               |
| hipBLASLt         | git258a2162         |
| Triton            | 3.1                 |

## Models
Examples of the following models are pre-optimized for performance on the AMD Instinct MI300X and MI325X accelerator.
* Llama3.1-8B
* Llama3.1-70B
* FLUX.1-dev

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

### Download the Docker image and required packages
Use the following command to pull the Docker image from the Docker hub
```
docker pull rocm/pytorch-training:v25.3
```

Run the Docker container
```
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name training_env  rocm/pytorch-training:v25.3
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
* [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
* [bghira/pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k)

### Benchmarking Command
#### Pretraining
To start the pretraining benchmark, use the following command with the appropriate options. See the list of options and their descriptions below.

./pytorch_benchmark_report.sh -t $training_mode -m $model_repo -p $datatype -s $sequence_length

Options and available models
|Name               | Options        | Description                                      |
|--------------------|---------------|--------------------------------------------------|
| $training_mode    | pretrain       | Benchmark pretraining                  |
|                   | finetune_fw    | Full weight finetuning, only support example of Llama 3.1 70B with BF16 |
|                  | finetune_lora  | LoRA finetuning, only support example of Llama 3.1 70B with BF16 |
| $datatype        | FP8 or BF16    | Currently, only Llama 3.1 8B example supports FP8 precision |
| $model_repo       | Llama-3.1-8B   | [Llama 3.1 8B](https://github.com/meta-llama/llama3)            |
|                  | Llama-3.1-70B  | [Llama 3.1 70B](https://github.com/meta-llama/llama3)            |
|                  | Flux           | [Flux.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| $sequence_length  | Sequence length for language model | Between 2048 and 8192 (default 8192) |

#### Finetuning


To start the finetuning benchmark, use the following command. It will run the benchmarking example of Llama 2 70B with wiki-text dataset using AMD branch of [torchtune](https://github.com/AMD-AIG-AIMA/torchtune)

```
./pytorch_benchmark_report.sh -t {finetune_fw, finetune_lora} -p BF16 -m Llama-3.1-70B
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
