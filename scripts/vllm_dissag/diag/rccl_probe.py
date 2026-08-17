# Minimal inter-node RCCL probe: mirrors what vLLM's PyNcclCommunicator does for the
# DP group at EP16 (one rank per node, 8 GPUs apart), but loads no model. Purpose is to
# get the real NCCL_DEBUG=INFO diagnosis in ~1 min instead of a 20 min model load.
import os, sys, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local)
dist.init_process_group("nccl", rank=rank, world_size=world)
t = torch.ones(1024, device=f"cuda:{local}") * (rank + 1)
dist.all_reduce(t)
expect = world * (world + 1) / 2
got = t[0].item()
print(f"RCCL_PROBE rank={rank} allreduce={got} expect={expect} {'OK' if abs(got-expect)<1e-3 else 'MISMATCH'}", flush=True)
dist.barrier(); dist.destroy_process_group()
if rank == 0: print("RCCL_PROBE ALL_OK", flush=True)
