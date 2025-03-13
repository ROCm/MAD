
export LAUNCHER="accelerate launch \
    --config_file fsdp_fp8.yaml \
    --num_processes $((NNODES * GPUS_ON_NODE)) \
    --num_machines $NNODES \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE_IP \
    --main_process_port 29500 \
    "
export SCRIPT="train_llama.py"
export SCRIPT_ARGS=" \
    --config=./configs/Llama3.1-8B.json \
    --log_file=result.log \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 

eval $CMD
 