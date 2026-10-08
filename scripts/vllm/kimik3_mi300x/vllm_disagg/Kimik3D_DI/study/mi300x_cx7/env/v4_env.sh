# MI300X + CX-7 study env — TP2/DP8/EP16, model on local NVMe
export TP_SIZE=2 DP_SIZE=8 DP_LOCAL=4
export MAX_MODEL_LEN=320000
export KV_CACHE_MEMORY_BYTES=20000000000
export K3_WRITE_READBACK=1
export MODEL_DIR=/mnt/m2m_nobackup/models_blog/Kimi-K3-MXFP4
export IMAGE=kimik3-wideep-disagg:v4-mi300x-cx7
# fabric: mlx5/RoCE (see fabric_env.sh) — enable the mlx5 mount block, disable bnxt
export FABRIC=mlx5
export THOR2_BNXT_FIX=0
source "$(dirname "${BASH_SOURCE[0]}")/fabric_env.sh"
