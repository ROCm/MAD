#!/usr/bin/env python3
"""Synthetic co-tenant with a decoder-only transformer training step's kernel mix,
for replay_nan_op.py --contend-with (debug-numeric step5-pressure.md).  Random
weights and tokens: no model files, nothing imported from the model.  Runs until
killed; the GPU is kept busy back to back (one synchronize per step, no host-side
checks), which is what made it a far stronger trigger than the real trainer.

    transformer_hog.py --mode full|fwdbwd|fwd|adamw [--procs P] [sizes...]

  full    forward + backward + AdamW: the whole training step
  fwdbwd  forward + backward, no optimizer
  fwd     forward only, no autograd
  adamw   AdamW alone over the same parameter count, with fixed random grads

Sizes default to one 0.5B decoder's (debug-numeric evidence E41); set them from the model's
config so the co-tenant runs the model's GEMM shapes.  --tf32 0 for a model that
runs fp32 at highest precision; --dtype bf16 for a bf16 model.

Which modes and process counts have reproduced what: debug-numeric.
"""
import argparse
import math
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Layer(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
        h, hd = a.hidden, a.head_dim
        self.n1, self.n2 = RMSNorm(h), RMSNorm(h)
        self.q = nn.Linear(h, a.heads * hd, bias=a.qkv_bias)
        self.k = nn.Linear(h, a.kv_heads * hd, bias=a.qkv_bias)
        self.v = nn.Linear(h, a.kv_heads * hd, bias=a.qkv_bias)
        self.o = nn.Linear(a.heads * hd, h, bias=False)
        self.gate = nn.Linear(h, a.inter, bias=False)
        self.up = nn.Linear(h, a.inter, bias=False)
        self.down = nn.Linear(a.inter, h, bias=False)

    def forward(self, x, mask):
        a = self.a
        b, s, _ = x.shape
        rep = a.heads // a.kv_heads
        h = self.n1(x)
        q = self.q(h).view(b, s, a.heads, a.head_dim).transpose(1, 2)
        k = self.k(h).view(b, s, a.kv_heads, a.head_dim).transpose(1, 2).repeat_interleave(rep, 1)
        v = self.v(h).view(b, s, a.kv_heads, a.head_dim).transpose(1, 2).repeat_interleave(rep, 1)
        # Explicit matmuls: the score GEMM is a GEMM, not a fused attention kernel.
        p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a.head_dim) + mask, dim=-1)
        x = x + self.o(torch.matmul(p, v).transpose(1, 2).reshape(b, s, -1))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Net(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.vocab = a.vocab
        self.emb = nn.Embedding(a.vocab, a.hidden)
        self.layers = nn.ModuleList(Layer(a) for _ in range(a.layers))
        self.norm = RMSNorm(a.hidden)
        self.head = nn.Linear(a.hidden, a.vocab, bias=False)

    def forward(self, ids, mask):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.head(self.norm(x))
        return F.cross_entropy(logits.view(-1, self.vocab).float(), ids.view(-1))


def build(a, dev, dtype):
    net = Net(a).to(device=dev, dtype=dtype)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    if a.mode == "adamw":
        for p in net.parameters():
            p.grad = torch.randn_like(p) * 1e-3
    return net, opt


def worker(rank, a):
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    torch.set_float32_matmul_precision("high" if a.tf32 else "highest")
    torch.manual_seed(1000 + rank)
    dev = torch.device("cuda")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    net, opt = build(a, dev, dtype)
    mask = torch.full((a.seq, a.seq), float("-inf"), device=dev, dtype=dtype).triu(1)
    reinits = 0
    while True:
        ids = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
        loss = None
        if a.mode == "fwd":
            with torch.no_grad():
                loss = net(ids, mask)
        elif a.mode in ("fwdbwd", "full"):
            opt.zero_grad(set_to_none=True)
            loss = net(ids, mask)
            loss.backward()
            if a.mode == "full":
                opt.step()
        else:
            opt.step()
        torch.cuda.synchronize()
        # Off by default -- the validated runs never re-initialised.  On, a co-tenant
        # that hit the kernel bug itself keeps computing on finite numbers.
        if a.reinit_on_nonfinite and loss is not None and not torch.isfinite(loss):
            reinits += 1
            print("[transformer_hog %d] non-finite loss: re-initialising (#%d)"
                  % (rank, reinits), flush=True)
            net, opt = build(a, dev, dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("full", "fwdbwd", "fwd", "adamw"))
    ap.add_argument("--procs", type=int, default=2, help="processes on the GPU (default 2)")
    ap.add_argument("--hidden", type=int, default=896)
    ap.add_argument("--inter", type=int, default=4864)
    ap.add_argument("--heads", type=int, default=14)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--seq", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--qkv-bias", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--dtype", default="fp32",
                    type=lambda v: {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}.get(v, v),
                    choices=("fp32", "bf16", "fp16"), help="fp32 | bf16 | fp16 (torch names accepted)")
    ap.add_argument("--reinit-on-nonfinite", action="store_true")
    a = ap.parse_args()
    print("transformer_hog %s" % vars(a), flush=True)
    if a.procs == 1:
        worker(0, a)
    else:
        mp.spawn(worker, args=(a,), nprocs=a.procs, join=True)


if __name__ == "__main__":
    main()
