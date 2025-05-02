# MAD - Model Automation and Dashboarding

## Overview

MAD (Model Automation and Dashboarding) is a comprehensive platform for:
- Running and benchmarking AI/ML models across various GPU architectures
- Automating model execution through containerized environments
- Maintaining historical performance data
- Generating performance tracking dashboards

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Running Models](#running-models)
  - [Basic Usage](#basic-usage)
  - [Tag Functionality](#tag-functionality)
  - [Custom Timeouts](#custom-timeouts)
  - [Advanced Options](#advanced-options)
- [Adding New Models](#adding-new-models)
  - [Model Configuration](#model-configuration)
  - [Creating Dockerfiles](#creating-dockerfiles)
  - [Scripts Setup](#scripts-setup)
  - [Tagging Guidelines](#tagging-guidelines)
- [Performance Reporting](#performance-reporting)
  - [Single Result Reporting](#single-result-reporting)
  - [Multiple Results Reporting](#multiple-results-reporting)
- [Environment Variables](#environment-variables)
  - [System Environment Variables](#system-environment-variables)
  - [Model Environment Variables](#model-environment-variables)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Getting Started

### Prerequisites

- Docker
- Python 3.9+
- Access to GPU resources (preferably AMD)

### Installation

Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

### Quick Start

To run a model:

```bash
python3 tools/run_models.py --tags <model_name>
```

## Running Models

### Basic Usage

The `run_models.py` script is the main CLI for MAD. It runs models defined in `models.json` through Docker containers:

```bash
python3 tools/run_models.py [options]
```

For each model, the script:
1. Builds a Docker image named `ci-$(model_name)`
2. Starts a container named `container_$(model_name)` 
3. Clones the repository specified in the model config
4. Runs the model script
5. Compiles performance results into `perf.csv` and `perf.html`

### Tag Functionality

Tags allow you to run specific subsets of models:

```bash
# Run a specific model
python3 tools/run_models.py --tags pyt_huggingface_bert

# Run all PyTorch models
python3 tools/run_models.py --tags pyt

# Run multiple tags (comma-separated)
python3 tools/run_models.py --tags pyt,fp16
```

Tags can be defined in `models.json` or passed via command line. A model's name is automatically considered a tag.

### Custom Timeouts

Control execution timeouts in three ways:
1. Default timeout: 7200 seconds (2 hours)
2. Model-specific timeout in `models.json`: `"timeout": 3600`
3. Command-line override: `--timeout 3600`

Setting timeout to 0 disables the timeout entirely.

### Advanced Options

```
usage: tools/run_models.py [-h] [--tags TAGS] [--timeout TIMEOUT] [--live-output] [--clean-docker-cache] [--keep-alive] [--keep-model-dir] [-o OUTPUT] [--log-level LOG_LEVEL]

Run the application of MAD, Model Automation and Dashboarding v1.0.0.

options:
  -h, --help            show this help message and exit
  --tags TAGS           Tags to run model (can be multiple).
  --timeout TIMEOUT     Timeout for the application running model in seconds, default timeout of 7200 (2 hours).
  --live-output         Prints output in real-time directly on STDOUT.
  --clean-docker-cache  Rebuild docker image without using cache.
  --keep-alive          Keep the container alive after the application finishes running.
  --keep-model-dir      Keep the model directory after the application finishes running.
  -o OUTPUT, --output OUTPUT
                        Output file for the result.
  --log-level LOG_LEVEL Log level for the logger.
```

**Debugging Tips:**
- Use `--live-output` to see logs in real-time instead of only in log files
- Use `--keep-alive` to prevent container deletion after execution for debugging
- Note: With `--keep-alive`, you must manually stop and remove the container before running the same model again

## Adding New Models

### Model Configuration

1. **Create a workload name** following the format: `<framework>_<project>_<workload>`:
   ```
   tf2_huggingface_gpt2
   ```

2. **Add model configuration to `models.json`**:
   ```json
   {
       "name": "tf2_bert_large",
       "url": "https://github.com/ROCmSoftwarePlatform/bert",
       "dockerfile": "docker/tf2_bert_large",
       "scripts": "scripts/tf2_bert_large",
       "n_gpus": "4",
       "owner": "john.doe@amd.com",
       "training_precision": "fp32",
       "tags": [
           "per_commit",
           "tf2",
           "bert",
           "fp32"
       ],
       "args": ""
   }
   ```

   **Configuration Fields:**

   | Field               | Description                                                                |
   |---------------------| ---------------------------------------------------------------------------|
   | name                | A unique model name                                                        |
   | url                 | Model repository URL to clone                                              |
   | dockerfile          | Path to the Dockerfile                                                     |
   | scripts             | Model script directory path                                                |
   | data                | Optional field denoting data for script                                    |
   | n_gpus              | Number of GPUs exposed inside container. '-1' => all available GPUs        |
   | timeout             | Model-specific timeout, default is 2 hours                                 |
   | owner               | Email address for model owner                                              |
   | training\_precision | Precision, currently used only for reporting                               |
   | tags                | List of tags for selecting model                                           |
   | multiple\_results   | Optional parameter for multiple results, pointing to CSV with results      |
   | args                | Extra arguments passed to model scripts                                    |

### Creating Dockerfiles

Create or reuse a Dockerfile in the `docker` directory:

```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
FROM rocm/tensorflow

# Install dependencies
RUN apt update && apt install -y \
    unzip 
RUN pip3 install pandas

# Download data
RUN URL=https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip && \
    wget --directory-prefix=/data -c $URL && \
    ZIP_NAME=$(basename $URL) && \
    unzip /data/$ZIP_NAME -d /data
```

### Scripts Setup

1. Create a directory in `scripts/` with all files needed for execution
2. Include a `run.sh` script (or specify a different script name in the model config)
3. Here's a sample script:

```bash
# setup model
MODEL_CONFIG_DIR=/data/uncased_L-24_H-1024_A-16
BATCH=2
SEQ=512
TRAIN_DIR=bert_large_ba${BATCH}_seq${SEQ}
TRAIN_STEPS=100
TRAIN_WARM_STEPS=10
LEARNING_RATE=1e-4
DATA_SOURCE_FILE_PATH=sample_text.txt
DATA_TFRECORD=sample_text_seq${SEQ}.tfrecord
MASKED_LM_PROB=0.15
calc_max_pred() {
    echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
}
MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

# Prepare data
python3 create_pretraining_data.py \
    --input_file=$DATA_SOURCE_FILE_PATH \
    --output_file=$DATA_TFRECORD \
    --vocab_file=$MODEL_CONFIG_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=$SEQ \
    --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
    --masked_lm_prob=$MASKED_LM_PROB \
    --random_seed=12345 \
    --dupe_factor=5

# Train model
python3 run_pretraining.py \
    --input_file=$DATA_TFRECORD \
    --output_dir=$TRAIN_DIR \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=$MODEL_CONFIG_DIR/bert_config.json \
    --train_batch_size=$BATCH \
    --max_seq_length=$SEQ \
    --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
    --num_train_steps=$TRAIN_STEPS \
    --num_warmup_steps=$TRAIN_WARM_STEPS \
    --learning_rate=$LEARNING_RATE \
    2>&1 | tee log.txt

# Report performance metric
python3 get_bert_model_metrics.py $TRAIN_DIR
```

### Tagging Guidelines

- Add relevant tags to your model in `models.json`
- Common tag categories include:
  - Framework (tf2, pyt, jax, etc.)
  - Model architecture (bert, resnet50, llama, etc.)
  - Precision (fp16, fp32, etc.)
  - Task types (training, inference, etc.)
- A model's name is automatically included as a tag

## Performance Reporting

### Single Result Reporting

For models with a single performance metric:
1. Print the performance in this format:
   ```
   performance: PERFORMANCE_NUMBER PERFORMANCE_METRIC
   ```
2. Example: 
   ```
   performance: 3.0637370347976685 examples/sec
   ```

### Multiple Results Reporting

For models with multiple performance metrics:
1. Add `"multiple_results": "results.csv"` to the model config in `models.json`
2. Generate a CSV file with three columns: `models,performance,metric`
3. Each row should contain a different result

## Environment Variables

### System Environment Variables

MAD provides system information through environment variables with the `MAD_` prefix:

| Variable                    | Description                          |
|-----------------------------|--------------------------------------|
| MAD_SYSTEM_GPU_ARCHITECTURE | GPU Architecture for the host system |
| MAD_RUNTIME_NGPUS           | Number of GPUs available to the model|

### Model Environment Variables

Model-specific environment variables have the `MAD_MODEL_` prefix:

| Field                       | Description                                |
|-----------------------------|--------------------------------------------|
| MAD_MODEL_NAME              | Model's name in `models.json`              |
| MAD_MODEL_NUM_EPOCHS        | Number of epochs                           |
| MAD_MODEL_BATCH_SIZE        | Batch size                                 |

## Troubleshooting

- **Container already exists**: When using `--keep-alive`, manually remove the container:
  ```bash
  docker stop container_<model_name>
  docker rm container_<model_name>
  ```

- **Docker build fails**: Try rebuilding without cache:
  ```bash
  python3 tools/run_models.py --clean-docker-cache --tags <model_name>
  ```

- **Execution timeout**: Extend the timeout:
  ```bash
  python3 tools/run_models.py --timeout 10800 --tags <model_name>
  ```

## License

© 2025 Advanced Micro Devices, Inc. All Rights Reserved.

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.

AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.

THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. 

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
