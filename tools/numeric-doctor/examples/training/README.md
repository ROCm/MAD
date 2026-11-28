This directory contains the training examples of using Numeric Doctor.

| Example | What it is |
| --- | --- |
| [`dit_fp8_train_nan/`](dit_fp8_train_nan/) | a small DiT trained in FP8 (torchao float8) with PyTorch TunableOp on, one trainer per GPU on 8 gfx950 GPUs: `Gradient norm is not finite` at step 0 on every GPU; self-contained, one command, public `rocm/primus` image |
| [`qwen_train_nan/`](qwen_train_nan/) | Qwen2.5-0.5B fp32 training (TF32 enabled), 3 trainers per GPU on 8 gfx950 GPUs: some trials hit a NaN after a few dozen to a few hundred steps; self-contained, one command |
| [`qwen3_moe_gradient_nan_nohint/`](qwen3_moe_gradient_nan_nohint/) | Qwen3 MoE FSDP+TP+EP training (torchtitan) on 8 gfx950 GPUs: grad_norm jumps to the thousands and beyond on some steps, then `Loss is not finite`; one command on a Slurm cluster (Apptainer or Docker) |
| [`softmoe_train_nan/`](softmoe_train_nan/) | a 20-layer soft-MoE transformer trained from scratch, layers 3-18 `torch.compile`'d and the rest eager, stops with `Loss is not finite` after a few dozen steps; one GPU, public `rocm/pytorch` image |
