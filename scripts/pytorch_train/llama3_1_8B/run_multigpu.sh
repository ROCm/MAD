export LAUNCHER="accelerate launch \
    --config_file fsdp_fp8.yaml \
    --num_processes 8 \
    --num_machines 1 \
    "
export SCRIPT="train_llama.py"
export SCRIPT_ARGS=" \
    --config=./configs/Llama3.1-8B.json \
    --log_file=llama_seq8192_bs3.log \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 

eval $CMD