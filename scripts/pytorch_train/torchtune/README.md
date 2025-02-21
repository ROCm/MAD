# Finetuning
This section describes finetuning llama-3.1-70b using wikitext dataset on a single node using [Torchtune](https://github.com/AMD-AIG-AIMA/torchtune) utility.

### Environment setup

```bash
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged    -v  $HOME/.ssh:/root/.ssh  -v /home/amd:/home/amd --shm-size 128G --name YOUR_NAME_HERE rocm/pytorch-training:v25.3
pip3 install torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.3
```

Clone MAD
```
cd /workspace
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/pytorch_train/torchtune
```

For formal testing we should use the correct model, not the unofficial mirror.
```
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir ./models/Llama-3.1-70B-Instruct --exclude 'original/*.pth'
```

To download the wikitext dataset, go to "pytorch-training-benchmark" directory and do (train and test splits will be saved):
```
python dataset.py
```

If any error downloading the data, do:
```
pip install datasets
```

cd /workspace/torchtune

For full finetuning, go to "torchtune" directory and do:

Copy both the 'wikitext_finetune.sh' and 'llama_3_1_70b_full_finetune_recipe.yaml' into the torchtune directory
```
cp -r /workspace/MAD/scripts/pytorch_train/torchtune/wikitext_finetune.sh .
cp -r /workspace/MAD/scripts/pytorch_train/torchtune/llama_3_1_70b_full_finetune_recipe.yaml .
```
For LORA finetuning, go to "torchtune" directory and do:

Copy both the 'wikitext_lora_finetune.sh' and 'llama_3_1_70b_lora_finetune_recipe.yaml' into the torchtune directory
```
cp -r /workspace/MAD/scripts/pytorch_train/torchtune/wikitext_lora_finetune.sh .
cp -r /workspace/MAD/scripts/pytorch_train/torchtune/llama_3_1_70b_lora_finetune_recipe.yaml .
```

### Full Finetuning Testing Command
The script `wikitext_finetune.sh` runs the finetuning test on `llama-3.1-70b` model with a wikitext dataset on top of the docker. Remove `MAX_STEPS=30` if you want to run for 1 complete epoch.
```
MODEL_DIR=./models/Llama-3.1-70B-Instruct COMPILE=True CPU_OFFLOAD=False PACKED=False SEQ_LEN=null ACTIVATION_CHECKPOINTING=True TUNE_ENV=True MBS=64 GAS=1 EPOCHS=1 SEED=42 VALIDATE=True MAX_STEPS=30 bash wikitext_finetune.sh
```

### LORA Finetuning Testing Command
The script `wikitext_finetune.sh` runs the finetuning test on `llama-3.1-70b` model with a wikitext dataset on top of the docker. Remove `MAX_STEPS=30` if you want to run for 1 complete epoch.
```
MODEL_DIR=./models/Llama-3.1-70B-Instruct COMPILE=True CPU_OFFLOAD=False PACKED=False SEQ_LEN=null ACTIVATION_CHECKPOINTING=True TUNE_ENV=True MBS=64 GAS=1 EPOCHS=1 SEED=42 VALIDATE=True MAX_STEPS=30 bash wikitext_lora_finetune.sh
```

### Performance Result (Full Finetuning)
Result for `MAX_STEPS=30` on a single node (8 GPUs) - AMD Instinct MI300X:TW044
```
Max memory alloc: 137.2001576423645
Average tokens/s/gpu: 92.0694
Unmasked tokens/s/gpu:  145.143
```

### Performance Result (LORA Finetuning)
Result for `MAX_STEPS=30` on a single node (8 GPUs) - AMD Instinct MI300X:TW044
```
Max memory alloc: 117.79637384414673
Average tokens/s/gpu: 65.7681
Unmasked tokens/s/gpu:  169.299
