# MI300X + CX-7 (mlx5, RoCE) fabric env — from cluster-sphere cluster_rdma_env_recommender.py
# RDMA data-plane: mlx5_0,2,3,4,5,7,8,9 (rdma0-7). Mgmt: eth0(mlx5_1), eth1(mlx5_6) EXCLUDED.
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
export MORI_RDMA_DEVICES=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export MORI_SOCKET_IFNAME=eth0
export SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
# mlx5 RDMA provider libs to host-mount into container (the bnxt-equivalent):
export MLX5_PROVIDER_HOST=/usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so
export MLX5_PROVIDER_IMG=/usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so
export MLX5_LIB_HOST=/usr/lib/x86_64-linux-gnu/libmlx5.so.1
export MLX5_LIB_IMG=/usr/lib/x86_64-linux-gnu/libmlx5.so.1
