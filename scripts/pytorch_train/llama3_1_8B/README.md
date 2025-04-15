# Training
This code is used for benchmarking Pytorch based pre-training on a synthesized dataset for a single node

Listed below are some example run commands for the model benchmarked in this repository using FSDP sharding strategy.

## Single-node Llama3.1-8B training with 8k sequence length
### MI300
### FP8 Precision
```
bash run_multigpu.sh
```

## Multi-node Llama3.1-8B training with 8k sequence length
### MI300
### FP8 Precision

1. Run the following command (outside the container) to find the active network interface on your system.
    ```
    ip a
    ```
2. Take two nodes (`NODE0` and `NODE1`) as example. Launch the Docker container on each node.
   ```
   docker run -it --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband \
      --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
      -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 128G --name training_env rocm/pytorch-training:v25.5 tail -f /dev/null
   ```
   Use these commands if you exit the megatron_training_env container and need to return to it.
   ```
   docker start training_env
   docker exec -it training_env bash
   ```
3. Run the training script on both nodes.
   - On the master node, run:
     ```
     NNODES=2 GPUS_ON_NODE=8 HEAD_NODE_IP=NODE0_IP bash run_multinode.sh
     ```
   - On the worker node, run:
     ```
     NNODES=2 GPUS_ON_NODE=8 HEAD_NODE_IP=NODE0_IP bash run_multinode.sh
     ```

We also provide an example using slurm cluster in `run_multinode_slurm.sh`
