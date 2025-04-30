# MPT-30B Training on MAD

## Overview 🎉
This repository provides a complete Docker-based training environment for the MPT-30B model using the llm-foundry framework. MPT-30B is a 30-billion parameter decoder-style transformer from the Mosaic Pretrained Transformer (MPT) family. 

This Docker image packages the MPT-30B training with PyTorch for an AMD Instinct™ MI300X
accelerator. It includes:

-   ✅ ROCm™ 6.3.4
-   ✅ PyTorch 2.7.0 (2.7.0a0+git6374332)
-   ✅ FlashAttention 3.0.0.post1

With this image, users can easily build, run, and validate the training process for MPT-30B while reviewing detailed logs and performance metrics.

---

## Training Environment Setup 🚀

### Streamlining the Workflow Using MAD's Tool
To simplify the workflow, we can directly utilize the run_model tool provided by MAD. The parameter configurations have been set under the pyt_mpt30b_training tag within the model.json file. With a single command, we can now seamlessly build the Docker container, download the model, and fine-tune it.

```sh
pip3 install -r requirements.txt
python3 tools/run_models.py --tags pyt_mpt30b_training --live-output --clean-docker-cache --keep-model-dir
```
If failed downloading data, try to `export MAD_SECRETS_HFTOKEN=` your huggingface token, which you can generate following [these steps](https://huggingface.co/docs/hub/security-tokens).

### Adjusting Total Training Duration ⏱️

1. **Open the configuration file**

   ```bash
   vi scripts/pyt_mpt30b_training/mpt-30b-instruct.yaml
   ```

2. **Edit `max_duration`**

   | Desired duration | Example change |
   |------------------|----------------|
   | 100 batches      | `max_duration: 10ba` ➜ `max_duration: 100ba` |
   | 5 epochs         | `max_duration: 10ba` ➜ `max_duration: 5ep` |

`max_duration` accepts either **`<number>ba`** for batches or **`<number>ep`** for epochs. Adjust as needed to control the total training run time.

### Enable Tunable Operator ⚙️

For improved performance (training throughput), consider enabling the tunable OP feature. Although this may increase the initial training time, it typically results in a performance gain of approximately 9.3%. Detailed steps are outlined below:
To collect performance data using PyTorch’s Tunable Operators feature, include the `--tunableop on` argument in your run.

By default, the `pyt_mpt30b_training` model already includes `--tunableop off` in its configuration. To customize the behavior, edit the `models.json`, find `pyt_mpt30b_training` config and modify the `args` field to `--tunableop on` accordingly.

 This triggers a two-pass run: a warm-up followed by a performance-collection run, generating a `gemm_result_<dataset>.csv` file for analysis.

### Dockerfile Contents
The Dockerfile is built with the following guidelines:
- **Public Base Docker Image:** Utilizes an open base image : rocm/vllm:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6
- **Public GitHub Repositories:** Integrates code from public sources (e.g., the flash-attetion and llm-foundry repository).
- **Publicly Accessible Packages/Utilities:** All dependencies are installed from public registries.

### Building the Docker Image
If you utilize Streamlining workflow you can ignore following steps.
Clone the repository and navigate to the project directory. Then build the Docker image with:

```sh
docker build --build-arg MAD_SYSTEM_GPU_ARCHITECTURE=gfx942 -f docker/pyt_mpt30b_training.ubuntu.amd.Dockerfile -t mosaic_mpt30b_image .
```
mosaic_mpt30b_image could be your name.

### Running the Docker Container
To start the container, run:

```sh
docker run -it --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --shm-size=8G --name=mosaic_mpt30b -v $PWD:/workspace/MAD mosaic_mpt30b_image
```
Now clone the ROCm MAD repository inside the Docker image and move to the benchmark scripts directory at ~/MAD/scripts/pyt_mpt30b_training.

```sh
cd /workspace/MAD/
cd scripts/pyt_mpt30b_training/
```

This command launches an interactive session in the container for training and debugging.
Single-node training is an ideal approach for developers seeking to use Composer on AMD GPUs with minimal setup. By leveraging the Docker image, you can launch Composer scripts on a single node with a single command. As outlined in the #requirements section, this container comes fully provisioned with all necessary dependencies, eliminating the need for additional installations.

### Executing the Training Script
Once inside the container, run the training script using the hyperparameters described in file mpt-30b-instruct.yaml:

```sh
source run.sh
```

This script will begin the training process using the configuration file provided. Training logs, checkpoints, and performance metrics will be output to the console.

If you want to open tunable Op flag, just run
```sh
source run.sh --tunableop on
```

### Interpreting the Output
After launching the training script, you can review:

 - Training Logs: Real-time display of loss metrics, accuracy, and training progress.
 - Model Checkpoints: Periodically saved model snapshots for potential resume or evaluation.
 - Performance Metrics: Detailed summaries of training speed and training loss metrics.

`Performance` (throughput/samples_per_sec)

    Overall throughput, measuring the total samples processed per second. Higher values indicate better hardware utilization.

`Performance per device` (throughput/samples_per_sec)

    Throughput on a per-device basis, showing how each GPU or CPU is performing.

`Language Cross Entropy` (metrics/train/LanguageCrossEntropy)

    Measures prediction accuracy. Lower cross entropy suggests the model’s output is closer to the expected distribution.

`Training Loss` (loss/train/total)

    Overall training loss. A decreasing trend indicates the model is learning effectively.

 For a complete understanding of the training progress, refer to the files and the log messages printed to the terminal.

### MPT-30B Model Introduction
MPT-30B is a large-scale, decoder-only transformer pretrained from scratch on 1 trillion tokens of English text and code. Developed by MosaicML, it belongs to the Mosaic Pretrained Transformer (MPT) family—models featuring optimized transformer architectures for efficient training and inference.
This [blog](https://www.databricks.com/blog/mpt-30b) provides additional information.

The model has been modified from a standard transformer in the following ways:
* It uses [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf)
* It uses [ALiBi (Attention with Linear Biases)](https://arxiv.org/abs/2108.12409) and does not use positional embeddings
* It does not use biases

| Hyperparameter | Value |
|----------------|-------|
|n_parameters | 29.95B |
|n_layers | 48 |
| n_heads | 64 |
| d_model | 7168 |
| vocab size | 50432 |
| sequence length | 8192 |

## Key Features
 - Extensive Pretraining: Trained on 1T tokens, offering robust language and code comprehension.
 - Extended Context Window: Supports an 8K-token context, further expandable via fine-tuning, leveraging ALiBi for long-sequence extrapolation.
 - Efficient Execution: Harnesses FlashAttention for accelerated training and inference.
 - Commercial-Friendly License: Permits commercial usage, contrasting with more restrictive alternatives.
 - Open-Source Codebase: Built upon llm-foundry for transparency and extensibility.

## Training Data
Data was tokenized using the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) BPE tokenizer. 

The vocabulary size of 50,432 tokens was chosen to be a multiple of 128, following recommendations from [MEGATRON-LM](https://arxiv.org/abs/1909.08053).

## Model Tile Submission 

 - Title: MPT-30B
 - Sub-title: Benchmark | MPT-30B | Training
 - Description: Benchmark container for MPT-30B—a 30-billion parameter model from the Mosaic Pretrained Transformer series—optimized for high accuracy in language tasks.

## Licensing Information ⚠️

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
| llm-foundry  | Apache License 2.0                          | [MosaicML License](https://github.com/mosaicml/llm-foundry)  |
| Flash-Attention  | BSD-3-Clause license                    | [FlashAttention License](https://github.com/Dao-AILab/flash-attention)  |

### Disclaimer
The information contained herein is for informational purposes only and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD's products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

### Notices and attribution
© 2024 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD Arrow logo, Instinct, Radeon Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc.

Docker and the Docker logo are trademarks or registered trademarks of Docker, Inc. in the United States and/or other countries. Docker, Inc. and other parties may also have trademark rights in other terms used herein. Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.    

All other trademarks and copyrights are property of their respective owners and are only mentioned for informative purposes.

## Changelog
Initial Release: 
The ROCm software version number is 6.3.4.

The PyTorch version number is 2.7.0. (2.7.0a0+git6374332)

## Support 

You can report bugs through our GitHub [issue tracker](https://github.com/ROCm/MAD/issues).