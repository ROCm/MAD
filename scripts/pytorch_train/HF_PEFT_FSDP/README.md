
## Overview

This README provides instructions for fine-tuning a model using the LoRA (Low-Rank Adaptation) approach with the Hugging Face library with the Ultrachat200 example. We will utilize FSDP (Fully Sharded Data Parallel) for single-node training (8 GPUs). 


### Environment Setup

```bash
docker pull rocm/pytorch-training:v25.5

docker run --rm -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME/.ssh:/root/.ssh -v /shared/amdgpu/home/kailash_gogineni_qle:/shared/amdgpu/home/kailash_gogineni_qle --name YOUR_NAME rocm/pytorch-training:v25.5

cd HF_PEFT_FSDP

# For formal testing we should use the correct model, not the unofficial mirror.
hf login
hf download meta-llama/Llama-2-70b-chat-hf --local-dir ./models/Llama-2-70b-chat-hf --exclude 'original/*.pth'
```

### Finetuning Command
```bash
# Make sure to set the model_name or _path
MODEL_DIR=./models/Llama-2-70b-chat-hf bash run_peft_fsdp.sh
```
