## Megatron-LM Migration to Primus megatron-core

Primus supports megatron-core as backend optimization library, henceforth replacing ROCM Megatron-LM. This document outlines the steps to migrate workload from rocm/megatron-lm to Primus with megatron-core backend.

### Model Architecture:
- Rocm-Megatron-LM defines model architecture parameters in the training scripts. For example, llama3 8B model parameters are defined in [examples/llama/train_llama3.sh](https://github.com/ROCm/Megatron-LM/blob/rocm_dev/examples/llama/train_llama3.sh#L117) as shown below:
```bash
    HIDDEN_SIZE=4096 
    FFN_HIDDEN_SIZE=14336 
    NUM_LAYERS=32 
    NUM_HEADS=32 
    NUM_KV_HEADS=8
```

- Primus defines the model architecture through model yamls inside the `primus/configs/models/megatron/` repository. For example, llama3-8B model architecture parameters are defined in [primus/configs/models/megatron/llama3_8B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/primus/configs/models/megatron/llama3_8B.yaml) as shown below:
```bash
bases:
  - llama3_base.yaml

tokenizer_type: Llama3Tokenizer
tokenizer_model: meta-llama/Llama-3.1-8B


ffn_hidden_size: 14336
hidden_size: 4096
num_attention_heads: 32
num_layers: 32
num_query_groups: 8
```

Primus model config files follow hierarchical design, i.e., new model config yaml’s can inherit/use existing model config files by importing existing model yaml files as bases. For example, [llama3.1_8B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/primus/configs/models/megatron/llama3.1_8B.yaml) utilized `llama3_8B.yaml` as base and overrides few parameters, as shown below. In this scenario, llama3.1_8B overrides/redefines the `max_position_embeddings` value
```bash
bases:
  - llama3_8B.yaml

tokenizer_type: Llama3Tokenizer
tokenizer_model: meta-llama/Llama-3.1-8B

max_position_embeddings: 131072
```

**TIPS**:
- Primus provides `llama_base.yaml` as the base configuration, which can be used as bases for adding new model architecture. Example, [mixtral_base.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/primus/configs/models/megatron/mixtral_base.yaml) and [deepseek_v3_base.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/primus/configs/models/megatron/deepseek_v3_base.yaml) define `llama_base.yaml` as it bases.
```bash
# Example mixtral_base.yaml:

bases:
  - llama_base.yaml

init_method_std: 0.01
rotary_base: 1000000
qk_layernorm: false

group_query_attention: true
num_query_groups: 8

# moe parameters
num_experts: 8
moe_router_topk: 2
moe_router_load_balancing_type: aux_loss
moe_aux_loss_coeff: 1e-2
moe_grouped_gemm: true
moe_token_dispatcher_type: alltoall
```

- It is recommended to add a new `${MODEL_NAME}_base.yaml` to add a new category of model and define new models on top of it. Example, in order to add Qwen2.5 models in primus, we define [qwen2.5_base.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/primus/configs/models/megatron/qwen2.5_base.yaml) and build [qwen2.5_7B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/primus/configs/models/megatron/qwen2.5_7B.yaml) and [qwen2.5_72b](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/primus/configs/models/megatron/qwen2.5_72B.yaml) using `qwen2.5_base.yaml` as the base. 

### Training Parameters:
- Rocm Megatron-LM also defines the training parameters, like batch size, tensor-parallelism, precision, etc, in the training scripts. For example, llama3 8B model parameters are defined in [examples/llama/train_llama3.sh](https://github.com/ROCm/Megatron-LM/blob/rocm_dev/examples/llama/train_llama3.sh) as shown below:
```bash
TP="${TP:-8}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
BS="${BS:-8}"
```

Due to the hierarchical nature of model yaml files, Primus defines the training parameters in top level yaml files, which can be found under [examples/megatron/configs/](https://github.com/AMD-AIG-AIMA/Primus/tree/v0.1.0-rc1/examples/megatron/configs) repository. For example, [llama3.1_8B](https://github.com/AMD-AIG-AIMA/Primus/blob/v0.1.0-rc1/examples/megatron/configs/llama3.1_8B-pretrain.yaml) model training yaml is defined in `examples/megatron/configs/llama3.1_8B-pretrain.yaml` as shown below. The model architecture defined in previous step is imported in the training yaml. Users can then override the default training parameters in this file.
```bash
    # model to run
    model: llama3.1_8B.yaml  # Model architecture yaml
    overrides:
      # log
      # disable_wandb: false
      # disable_tensorboard: false
      stderr_sink_level: DEBUG

      log_avg_skip_iterations: 2
      log_avg_reset_interval: 50

      train_iters: 50
      micro_batch_size: 2
      global_batch_size: 128

      seq_length: 8192
      max_position_embeddings: 8192

      lr: 1.0e-5
      min_lr: 0.0
      lr_warmup_iters: 2
      lr_decay_iters: null
      lr_decay_style: cosine
      weight_decay: 0.1
      adam_beta1: 0.9
      adam_beta2: 0.95
      eod_mask_loss: true
      init_method_std: 0.008
      norm_epsilon: 1.0e-6
```

# Backward Compatibility with Megatron-LM

Primus docker will maintain Megatron-LM compatibility with limited support. To roll-back using Megatron-LM follow the steps outlined below. Once Megatron-LM is installed, follow the documentation to run workloads as usual.
```bash
cd /workspace/Megatron-LM/
pip uninstall megatron-core
pip install -e .
```
