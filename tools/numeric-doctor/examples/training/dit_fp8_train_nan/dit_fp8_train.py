#!/usr/bin/env python3
"""Small DiT (adaLN-Zero blocks, rectified-flow loss) trained in FP8 with torchao float8 training.

hidden 3072, 24 heads, fused QKV projection, batch 4 x 64x64 latent patches = 16384 tokens per step, bf16 weights,
AdamW with linear warmup, gradient clipping. The block linears are converted with torchao's
convert_to_float8_training (its default recipe, tensorwise scaling).

Prints loss / grad norm / lr every --log-every steps, and stops a trial with "Loss is not finite at step <n>" or
"Gradient norm is not finite at step <n>". Each trial ends with one line:
RESULT trial=.. step_reached=.. nan_step=<first bad step or -> nan_where=loss|grad_norm|- ...
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbed(nn.Module):
    def __init__(self, hidden, freq=256):
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = t.float()[:, None] * 1000 * f[None]
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1).to(self.mlp[0].weight.dtype))


class Block(nn.Module):
    """DiT block: adaLN-Zero modulation, self-attention with a fused QKV projection, GELU MLP."""

    def __init__(self, hidden, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(hidden, mlp_ratio * hidden)
        self.fc2 = nn.Linear(mlp_ratio * hidden, hidden)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        B, L, H = x.shape
        sm, scm, gm, sf, scf, gf = self.ada(c).chunk(6, -1)
        qkv = self.qkv(modulate(self.norm1(x), sm, scm))
        q, k, v = qkv.view(B, L, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, H)
        x = x + gm.unsqueeze(1) * self.proj(a)
        h = self.fc2(F.gelu(self.fc1(modulate(self.norm2(x), sf, scf)), approximate="tanh"))
        return x + gf.unsqueeze(1) * h


class DiT(nn.Module):
    def __init__(self, in_ch, hidden, heads, depth, mlp_ratio, zero_init):
        super().__init__()
        self.x_in = nn.Linear(in_ch, hidden)
        self.t_emb = TimestepEmbed(hidden)
        self.blocks = nn.ModuleList(Block(hidden, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.out = nn.Linear(hidden, in_ch)
        if zero_init:  # adaLN-Zero, as in DiT
            for m in [b.ada[-1] for b in self.blocks] + [self.ada[-1], self.out]:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        c = self.t_emb(t)
        h = self.x_in(x)
        for b in self.blocks:
            h = b(h, c)
        shift, scale = self.ada(c).chunk(2, -1)
        return self.out(modulate(self.norm(h), shift, scale))


def run_trial(args, trial, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    model = DiT(args.in_ch, args.hidden, args.heads, args.depth, args.mlp_ratio, args.zero_init).to(dev, torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    if args.fp8 == "torchao":
        from torchao.float8 import convert_to_float8_training
        block_linears = ("qkv", "proj", "fc1", "fc2")
        convert_to_float8_training(model, module_filter_fn=lambda m, fqn: fqn.split(".")[-1] in block_linears)
    # A fixed synthetic "latent" dataset: smooth low-rank images so there is something to learn.
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    basis = torch.randn(32, args.seq, args.in_ch, device=dev, generator=g) / 4
    nan_step, nan_where, step, loss0, loss = None, "-", -1, None, float("nan")
    t0 = time.time()
    for step in range(args.steps):
        coef = torch.randn(args.batch, 32, device=dev)
        x1 = torch.einsum("bk,kld->bld", coef, basis).to(torch.bfloat16)
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch, device=dev)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        pred = model(xt.to(torch.bfloat16), t)
        loss_t = F.mse_loss(pred.float(), (x1 - x0).float())
        opt.zero_grad(set_to_none=True)
        loss_t.backward()
        loss = float(loss_t.detach())
        loss0 = loss if loss0 is None else loss0
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        bad = None
        if not math.isfinite(loss):
            bad = "loss"
            print(f"Loss is not finite at step {step}: {loss}", flush=True)
        elif not math.isfinite(gnorm):
            bad = "grad_norm"
            print(f"Gradient norm is not finite at step {step}: {gnorm}", flush=True)
        if args.log_every and (step % args.log_every == 0 or bad):
            print(f"trial {trial} step {step} loss {loss:.5f} grad_norm {gnorm:.3e} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if bad:
            nan_step, nan_where = step, bad
            break
        opt.step(); sched.step()
    torch.cuda.synchronize()
    print(f"RESULT trial={trial} step_reached={step} nan_step={'-' if nan_step is None else nan_step} "
          f"nan_where={nan_where} loss0={loss0:.4f} loss_last={loss:.4f} seed={seed} fp8={args.fp8} "
          f"tokens={args.batch * args.seq} secs={time.time() - t0:.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--in-ch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096, help="latent patches per image (64x64)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--zero-init", type=int, default=1, help="adaLN-Zero init (1) or default init (0)")
    ap.add_argument("--fp8", choices=["torchao", "none"], default="torchao")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient clipping max norm")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    print(f"# torch {torch.__version__} hip {torch.version.hip} {torch.cuda.get_device_properties(0).gcnArchName} {vars(args)}", flush=True)
    for tr in range(args.trials):
        run_trial(args, tr, args.seed + tr)


if __name__ == "__main__":
    main()
