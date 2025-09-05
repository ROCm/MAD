
export LAUNCHER="accelerate launch \
    --num_processes $((NNODES * GPUS_ON_NODE)) \
    --num_machines $NNODES \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE_IP \
    --main_process_port 29500 \
    "
export SCRIPT="fsdp2_fp8.py"
export SCRIPT_ARGS=" \
    --sequence-length=8192 \
    --num-steps=100 \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 

eval $CMD
 
