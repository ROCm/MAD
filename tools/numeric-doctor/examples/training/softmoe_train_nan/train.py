#!/usr/bin/env python3
"""Train a 20-layer decoder-only transformer with soft mixture-of-experts MLPs, mostly torch.compile'd.

Layers are numbered 1 to 20. Layers 3-18 are each wrapped in `torch.compile` (inductor); layers 1-2 and 19-20
run eagerly (`--compile-layers` changes the set, `none` for all eager). Each layer is a pre-norm block: RMSNorm -> causal self-attention
with RoPE -> residual, RMSNorm -> soft-MoE SwiGLU MLP -> residual. The MLP's hidden units are split into
`--experts` experts; a router (temperature `--router-temp`, logits soft-capped at 50 as in Gemma 2) gives
each token a softmax over experts, and expert e's output is weighted by sqrt(p_e): the weights' squares sum
to 1, so the mixture keeps the variance of its experts. The router starts uniform (zero init).

The data is a synthetic token stream (a random sparse Markov chain over Zipf-distributed tokens, so the
loss falls) cut into rows of `seq_len + 1` tokens, `batch_size` rows per step, in order. Pad positions
are masked out of the loss.

Only PyTorch is required. Every step prints
    step: <n>  loss: <x>  grad_norm: <x>  lr: <x>  time: <s>
and the run ends with one line
    RESULT step_reached=<n> nan_step=<n|-> final_loss=<x|->
A non-finite loss stops training with `RuntimeError: Loss is not finite` (exit code 1).
"""

import argparse
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LAYERS = 20
PAD = 0

NORM_EPS = 1e-6
ROUTER_SOFTCAP = 50.0
ROUTER_TEMP = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).type_as(x)


def rope_tables(seq_len: int, head_dim: int, theta: float = 10000.0):
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    ang = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):  # x: (b, s, h, d)
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2).type_as(x)


class SoftMoEMLP(nn.Module):
    """SwiGLU whose hidden units form `experts` equal groups; group e is scaled by sqrt(p_e) before the
    down projection, i.e. sum_e sqrt(p_e) * expert_e(h) with expert e's slice of w2."""

    def __init__(self, dim: int, hidden: int, experts: int, router_temp: float):
        super().__init__()
        self.experts, self.router_temp = experts, router_temp
        self.router = nn.Linear(dim, experts, bias=False)
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down

    def forward(self, h):
        z = self.router(h).float() / self.router_temp
        z = ROUTER_SOFTCAP * torch.tanh(z / ROUTER_SOFTCAP)
        weight = torch.sqrt(torch.softmax(z, dim=-1))  # (b, s, experts), sum of squares = 1
        act = (F.silu(self.w1(h)) * self.w3(h)).unflatten(-1, (self.experts, -1))
        return self.w2((act * weight.unsqueeze(-1).type_as(act)).flatten(-2))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden: int, eps: float, experts: int, router_temp: float):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.attn_norm = RMSNorm(dim, eps)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim, eps)
        self.mlp = SoftMoEMLP(dim, hidden, experts, router_temp)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        q, k, v = self.wqkv(self.attn_norm(x)).view(b, s, 3, self.n_heads, self.head_dim).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.wo(a.transpose(1, 2).reshape(b, s, d))
        return x + self.mlp(self.mlp_norm(x))


class Model(nn.Module):
    def __init__(self, vocab: int, dim: int, n_heads: int, hidden: int, seq_len: int, compiled: set,
                 experts: int, router_temp: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD)
        blocks = [Block(dim, n_heads, hidden, NORM_EPS, experts, router_temp) for _ in range(NUM_LAYERS)]
        self.norm = RMSNorm(dim, NORM_EPS)
        self.head = nn.Linear(dim, vocab, bias=False)
        cos, sin = rope_tables(seq_len, dim // n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._init(blocks)
        self.layers = nn.ModuleList(torch.compile(blk) if i in compiled else blk for i, blk in enumerate(blocks, 1))

    def _init(self, blocks):
        for m in [self.embed, self.head, *blocks]:
            for p in m.parameters():
                if p.dim() == 2:
                    nn.init.normal_(p, std=0.02)
        for blk in blocks:  # residual projections scaled by depth; routers start uniform
            for p in (blk.wo.weight, blk.mlp.w2.weight):
                p.data.mul_(1 / math.sqrt(2 * NUM_LAYERS))
            blk.mlp.router.weight.data.zero_()
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    def forward(self, tokens):
        s = tokens.shape[1]
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, self.cos[:s], self.sin[:s])
        return self.head(self.norm(x))


def make_stream(vocab: int, n_tokens: int, seed: int, doc_len: int = 512):
    """Documents from a random sparse Markov chain: each token has 4 likely successors, 10% noise.
    Tokens are Zipf-distributed, as in text, so the loss falls within a few steps."""
    g = torch.Generator().manual_seed(seed)
    zipf = 1.0 / torch.arange(1, vocab, dtype=torch.float64)

    def draw(n):
        return torch.multinomial(zipf, n, replacement=True, generator=g) + 1

    succ = draw(vocab * 4).view(vocab, 4)
    n_docs = -(-n_tokens // doc_len)
    tok = draw(n_docs)
    out = torch.empty(n_docs, doc_len, dtype=torch.long)
    for t in range(doc_len):
        out[:, t] = tok
        nxt = succ[tok, torch.randint(0, 4, (n_docs,), generator=g)]
        noise = torch.rand(n_docs, generator=g) < 0.1
        tok = torch.where(noise, draw(n_docs), nxt)
    return out.flatten()[:n_tokens]


def epoch_batches(stream, batch_size: int, seq_len: int):
    """Rows of seq_len + 1 tokens in stream order; the tail is padded with PAD to a full batch."""
    row = seq_len + 1
    per_batch = batch_size * row
    n_batches = -(-stream.numel() // per_batch)
    padded = torch.full((n_batches * per_batch,), PAD, dtype=torch.long)
    padded[: stream.numel()] = stream
    return padded.view(n_batches, batch_size, row)


def parse_layers(spec: str) -> set:
    if spec in ("", "none"):
        return set()
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    bad = [i for i in out if not 1 <= i <= NUM_LAYERS]
    if bad:
        sys.exit(f"--compile-layers: no layer {bad} (layers are 1..{NUM_LAYERS})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps-per-epoch", type=int, default=25,
                    help="the stream holds this many batches; the last one is partly filled")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=2816)
    ap.add_argument("--experts", type=int, default=4, help="soft-MoE experts per MLP (must divide --hidden)")
    ap.add_argument("--router-temp", type=float, default=ROUTER_TEMP)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--compile-layers", default="3-18", help="e.g. 3-18 or 1-20; 'none' for all eager")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    compiled = parse_layers(args.compile_layers)
    row = args.seq_len + 1
    n_tokens = (args.steps_per_epoch - 1) * args.batch_size * row + 5 * row // 2
    data = epoch_batches(make_stream(args.vocab, n_tokens, args.seed), args.batch_size, args.seq_len)

    if args.hidden % args.experts:
        sys.exit(f"--experts {args.experts} does not divide --hidden {args.hidden}")
    model = Model(args.vocab, args.dim, args.heads, args.hidden, args.seq_len, compiled,
                  args.experts, args.router_temp).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ENV torch {torch.__version__} hip {torch.version.hip} device {dev}"
          + (f" {torch.cuda.get_device_properties(dev).gcnArchName}" if dev.type == "cuda" and torch.version.hip else ""))
    print(f"MODEL layers={NUM_LAYERS} compiled={','.join(map(str, sorted(compiled))) or 'none'} dim={args.dim} "
          f"heads={args.heads} hidden={args.hidden} experts={args.experts} router_temp={args.router_temp} "
          f"vocab={args.vocab} params={n_params / 1e6:.1f}M")
    print(f"DATA tokens/epoch={n_tokens} batches/epoch={len(data)} batch={args.batch_size}x{args.seq_len} "
          f"amp={args.amp} seed={args.seed}", flush=True)

    decay = [p for p in model.parameters() if p.dim() == 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.amp == "bf16")

    step, done, loss_val, nan_step = 0, 0, float("nan"), None
    try:
        for step in range(1, args.steps + 1):
            t0 = time.time()
            batch = data[(step - 1) % len(data)].to(dev, non_blocking=True)
            inp, lbl = batch[:, :-1], batch[:, 1:]
            lr = args.lr * min(1.0, step / args.warmup)
            for grp in opt.param_groups:
                grp["lr"] = lr
            with amp:
                logits = model(inp)
            ce = F.cross_entropy(logits.float().flatten(0, 1), lbl.flatten(), reduction="none")
            mask = (lbl.flatten() != PAD).float()
            loss = (ce * mask).sum() / mask.sum()
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                nan_step = step
                raise RuntimeError("Loss is not finite")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            print(f"step: {step:3d}  loss: {loss_val:.5f}  grad_norm: {gn.item():.4f}  lr: {lr:.2e}  "
                  f"time: {time.time() - t0:.3f}", flush=True)
            done = step
    finally:
        print(f"RESULT step_reached={done} nan_step={nan_step or '-'} "
              f"final_loss={loss_val if nan_step is None else '-'}", flush=True)


if __name__ == "__main__":
    main()
