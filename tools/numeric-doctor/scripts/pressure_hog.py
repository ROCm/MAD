#!/usr/bin/env python3
"""Synthetic co-tenant for replay_nan_op.py --contend-with: load the GPU the way a
training job does, without the model.  Runs until killed.

    pressure_hog.py --mode copy|triad|gemm|mix|bust [--procs P] [--gb G] [--bust-mnk M,N,K]

  copy   dst.copy_(src) over G GB buffers: pure HBM streaming
  triad  a = b + s * c over G GB buffers: 2 reads + 1 write, like an AdamW update
  gemm   8192^3 fp32 matmul with TF32 on: the F32X kernel family, compute + HBM
  mix    copy and gemm alternating
  bust   the capsule's own GEMM -- `--bust-mnk M,N,K` from its `hipBLASLt problem:`
         line (A transposed, TF32) -- over 64 distinct operand sets, so every
         launch reads from HBM rather than cache: the same kernel as the one under
         test, nothing else

Simple loads, for the second pass of step5-pressure.md: start with the
training-step stand-in (transformer_hog.py), come here to narrow it down -- these
have run clean where the stand-in reproduced (debug-numeric).
"""
import argparse
import torch
import torch.multiprocessing as mp


def worker(rank, args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    n = int(args.gb * (1 << 30) / 4)
    if args.mode in ("copy", "mix"):
        src = torch.randn(n, device=dev)
        dst = torch.empty_like(src)
    if args.mode == "triad":
        b = torch.randn(n, device=dev)
        c = torch.randn(n, device=dev)
        a = torch.empty_like(b)
    if args.mode in ("gemm", "mix"):
        x = torch.randn(8192, 8192, device=dev)
        y = torch.randn(8192, 8192, device=dev)
    if args.mode == "bust":
        # torch.mm(a, w.t()) with a (N, K) and w (M, K): hipBLASLt sees M N K transA=T.
        m, n, k = args.bust_mnk
        sets = [(torch.randn(n, k, device=dev), torch.randn(m, k, device=dev))
                for _ in range(64)]
    i = 0
    while True:
        if args.mode == "copy":
            dst.copy_(src)
        elif args.mode == "triad":
            torch.add(b, c, alpha=1.5, out=a)
        elif args.mode == "gemm":
            torch.mm(x, y)
        elif args.mode == "mix":
            dst.copy_(src) if i % 2 else torch.mm(x, y)
        elif args.mode == "bust":
            a_, w_ = sets[i % len(sets)]
            torch.mm(a_, w_.t())
        i += 1
        if i % 64 == 0:
            torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("copy", "triad", "gemm", "mix", "bust"))
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--gb", type=float, default=4.0)
    ap.add_argument("--bust-mnk", type=lambda v: tuple(int(x) for x in v.split(",")),
                    help="bust: the capsule's hipBLASLt M,N,K (required with --mode bust)")
    args = ap.parse_args()
    if args.mode == "bust" and not args.bust_mnk:
        ap.error("--mode bust needs --bust-mnk M,N,K (the capsule's hipBLASLt problem)")
    print("pressure_hog mode=%s procs=%d gb=%.1f" % (args.mode, args.procs, args.gb), flush=True)
    if args.procs == 1:
        worker(0, args)
    else:
        mp.spawn(worker, args=(args,), nprocs=args.procs, join=True)


if __name__ == "__main__":
    main()
