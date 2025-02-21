# Training
This code is used for benchmarking Pytorch based pre-training on a synthesized dataset for a single node

Need to install `accelerate` library
```
pip install accelerate
```

Listed below are some example run commands for the model benchmarked in this repository using FSDP sharding strategy.

## Run commands for Llama3.1-8B training with 4k sequence length
### MI300
### FP8 Precision
```
accelerate launch --config_file fsdp_fp8.yaml ./train_llama.py --max_seq_len=4096 --batch_size=4
```

## Run commands for Llama3.1-8B training with 8k sequence length
### MI300
### FP8 Precision
```
accelerate launch --config_file fsdp_fp8.yaml ./train_llama.py --max_seq_len=8192 --batch_size=2
```
# Reference Performance
(with wrapping, no torch.compile)
| Models  | Batch size | Sequence length | Avg TFLOP/s | Avg tokens/s | Peak Memory (GB) |
| -------|  -------    | ----------      |    -------  | ----------   | ----------  |
| Llama 3.1 8B |  4 | 4096 | 670.24 | 13021.85 | 128.99 |
| Llama 3.1 8B |  5 | 4096 | 679.78 | 13207.11 | 153.03 |
| Llama 3.1 8B |  6 | 4096 | 697.18 | 13545.23 | 177.10 |
| Llama 3.1 8B |  2 | 8192 | 665.35 | 11488.88 | 129.12 |
| Llama 3.1 8B |  3 | 8192 | 711.90 | 12557.27 | 177.24 |
