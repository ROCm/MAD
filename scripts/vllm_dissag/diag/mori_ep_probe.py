"""Minimal 2-node MoRI EP probe: torchrun 16 ranks -> mori shmem init -> EP dispatch.

Reproduces the `DV Modify QP v2 error 110` seen at EP16 in ~90 s instead of a
20-minute model load. Run with:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=N --master_addr=IP \
           --master_port=29577 mori_ep_probe.py
"""
import os
import sys
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    if rank == 0:
        print(f"[probe] torch.distributed up, world={world}", flush=True)

    import mori
    mori.shmem.shmem_torch_process_group_init("default")
    dist.barrier()
    if rank == 0:
        print("[probe] SHMEM INIT OK (RDMA QPs established)", flush=True)

    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world,
        hidden_dim=7168,
        scale_dim=56,
        scale_type_size=4,
        max_token_type_size=2,
        max_num_inp_token_per_rank=512,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        block_num=16,
        warp_num_per_block=16,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNode,
    )
    op = mori.ops.EpDispatchCombineOp(cfg)
    dev = torch.device("cuda", local)
    n = 128
    inp = torch.randn(n, 7168, dtype=torch.bfloat16, device=dev)
    w = torch.rand(n, 8, dtype=torch.float32, device=dev)
    total_experts = 32 * world
    idx = torch.stack([torch.randperm(total_experts, device=dev)[:8] for _ in range(n)]).to(torch.int32)
    scales = torch.rand(n, 56, dtype=torch.float32, device=dev)

    for it in range(3):
        out, ow, _, oi, nrecv = op.dispatch(inp, w, scales, idx)
        comb, _ = op.combine(out, ow, oi)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"[probe] iter {it} dispatch+combine OK recv={nrecv.item()}", flush=True)

    dist.barrier()
    if rank == 0:
        print("[probe] ALL OK", flush=True)
    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
