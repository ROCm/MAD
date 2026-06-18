#!/bin/bash
# set -x

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-2373}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
IPADDRS="${IPADDRS:-localhost}"
IBDEVICES="${IBDEVICES:-mlx5_0}"
SKIP_DEEPEP="${SKIP_DEEPEP:-0}"

host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

#echo "Waiting at the container creation barrier on $host_name"
#python $MAD_REPO_PATH/utils/socket_barrier.py --local-ip ${host_ip} --local-port 2200 --enable-port --node-ips ${IPADDRS} --node-ports 2200

echo "$IPADDRS"
echo "$IBDEVICES"

SERVER_IP=$(echo "$IPADDRS" | awk -F',' '{print $1}')
echo "Server IP is - ${SERVER_IP}"
echo "Node Rank is ${NODE_RANK}"

echo "-------EP benchamrking --------"

DEEPEP_PATH="/app/DeepEP"
cd $DEEPEP_PATH
MORI_PATH="/app/mori"
if [ "$NNODES" -eq 1 ]; then
    echo "Total number of nodes - ${NNODES}"

    if [ "$SKIP_DEEPEP" != "1" ]; then
        python tests/test_intranode.py 2>&1 | tee /run_logs/intranode_results.log
        sleep 20;

        export ROCSHMEM_USE_IB_HCA=$IBDEVICES
        export ROCSHMEM_HEAP_SIZE=4147483648
        python tests/test_low_latency.py 2>&1 | tee /run_logs/low_latency_1N_results.log
    else
        echo "----- Skipping DeepEP benchmarks (SKIP_DEEPEP=1) -----"
    fi

    echo "----- Performing Intranode MoRI -----"
    cd $MORI_PATH
    export PYTHONPATH=$PYTHONPATH:$MORI_PATH
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)
    python tests/python/ops/bench_dispatch_combine.py 2>&1 | tee /run_logs/mori_intranode_results.log

    echo "----- Performing low latency mori ----"
    python tests/python/ops/bench_dispatch_combine.py --dtype fp8_e4m3_fnuz 2>&1 | tee /run_logs/mori_1N_ll_results.log

else
    LOG_FILE=internode_${NODE_RANK}.txt
    export MASTER_ADDR=$MASTER_ADDR
    export RANK=$NODE_RANK
    export WORLD_SIZE=$NNODES
    export ROCSHMEM_MAX_NUM_CONTEXTS=64
 
    echo "Total number of nodes - ${NNODES}"
    echo "Master Address - $MASTER_ADDR"
    echo "Node Rank - $NODE_RANK"
    echo "NNODES - $NNODES"

    if [ "$SKIP_DEEPEP" != "1" ]; then
        echo "----- Running internode tests -----"

        python tests/test_internode.py 2>&1 | tee /run_logs/$LOG_FILE
        sleep 20;

        echo "----- Running low latency tests -----"
        export ROCSHMEM_HEAP_SIZE=$((3*1024*1024*1024))
        export ROCSHMEM_MAX_NUM_CONTEXTS=144
        #export ROCSHMEM_BACKEND=gda
        #export ROCSHMEM_DISABLE_MIXED_IPC=0
        #export ROCSHMEM_USE_IB_HCA=$IBDEVICES
        python tests/test_low_latency.py 2>&1 | tee /run_logs/low-latency-${NODE_RANK}.log
        sleep 20;
    else
        echo "----- Skipping DeepEP benchmarks (SKIP_DEEPEP=1) -----"
    fi

    echo "---- Running mori internode tests -------"
    cd $MORI_PATH
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)
    torchrun --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd bench 2>&1 | tee /run_logs/mori_${LOG_FILE}
    sleep 20;

    echo "----- Running mori low latency test -----"
    torchrun --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd bench --kernel-type v1_ll 2>&1 | tee /run_logs/mori_ll_${LOG_FILE}
fi

# Generate perf.csv for madengine from all benchmark logs
if [[ -f "$MAD_REPO_PATH/parse_ep_to_csv.py" ]]; then
    python3 $MAD_REPO_PATH/parse_ep_to_csv.py /run_logs -o /run_logs/perf.csv 2>&1
fi

exit 0
