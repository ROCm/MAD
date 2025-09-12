#!bin/bash
set +ex

export TORCH_BLAS_PREFER_HIPBLASLT=1

CUR_DIR=`pwd`

cd /
mkdir -p data/ml-20m
cd /data/ml-20m
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip

LOG_DIR=$CUR_DIR/logs

cd $CUR_DIR
pushd $CUR_DIR
cd PyTorch/Recommendation/NCF/
pip install -r requirements.txt
./prepare_dataset.sh

mkdir -p $LOG_DIR

# fp16
torchrun --nproc_per_node=1 ncf.py --data /data/cache/ml-20m --epochs 2 --batch_size 10000000 2>&1 | tee $LOG_DIR/ncf_fp16.log
# fp32
torchrun --nproc_per_node=1 ncf.py --data /data/cache/ml-20m --epochs 2 --batch_size 8000000 2>&1 | tee $LOG_DIR/ncf_fp32.log

popd
python3 get_ncf_model_metrics.py $LOG_DIR/ncf_fp16.log $LOG_DIR/ncf_fp32.log results_ncf.csv
cp results_ncf.csv ..
