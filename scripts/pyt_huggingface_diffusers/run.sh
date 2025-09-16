#!/bin/bash

pip uninstall xformers
MAX_JOBS=128 pip install git+https://github.com/ROCm/xformers.git@02e7602d97f09d79952530f6041ea7607f3d0d9d

cd /myworkspace

if [ -d "/myworkspace/diffusers" ]; then
    echo "huggingface diffusers exists."
    rm -rf diffusers
    echo "Re clone huggingface diffusers."
    TARGET_DIR="diffusers"
    REPO_URL="https://github.com/huggingface/diffusers.git"
    PATCH_TARGET_DIR="/myworkspace/diffusers"
    PATCH_FILE="/myworkspace/scripts/pyt_huggingface_diffusers/diffusers_0.patch"

    git clone "$REPO_URL" "$TARGET_DIR"
    cd diffusers
    #git checkout 23a4ff84881ada4bca7af7b815d58f8c48ccc13d
    git checkout 66e50d4e248a32ef8f8698cf3e6f0e1040f74cfc
    patch -p1 -d "$PATCH_TARGET_DIR" < "$PATCH_FILE"
    rm /opt/conda/envs/py_3.10/lib/python3.10/site-packages/diffusers/models/attention_processor.py
    cp /myworkspace/diffusers/src/diffusers/models/attention_processor.py /opt/conda/envs/py_3.10/lib/python3.10/site-packages/diffusers/models
    popd

else
    echo "huggingface diffusers directory does not exist."
    TARGET_DIR="diffusers"
    REPO_URL="https://github.com/huggingface/diffusers.git"
    PATCH_TARGET_DIR="/myworkspace/diffusers"
    PATCH_FILE="/myworkspace/scripts/pyt_huggingface_diffusers/diffusers_0.patch"
    
    git clone "$REPO_URL" "$TARGET_DIR"
    cd diffusers
    #git checkout 23a4ff84881ada4bca7af7b815d58f8c48ccc13d
    git checkout 66e50d4e248a32ef8f8698cf3e6f0e1040f74cfc
    patch -p1 -d "$PATCH_TARGET_DIR" < "$PATCH_FILE"
    rm /opt/conda/envs/py_3.10/lib/python3.10/site-packages/diffusers/models/attention_processor.py
    cp /myworkspace/diffusers/src/diffusers/models/attention_processor.py /opt/conda/envs/py_3.10/lib/python3.10/site-packages/diffusers/models
    popd
fi

cd /myworkspace/scripts/pyt_huggingface_diffusers

bash sweep_sdxl_ds_finetune_tuned.sh -r 1k
bash sweep_sdxl_ds_finetune_tuned.sh -r 2k

python merge_csv.py
