#!/usr/bin/env python3
"""Greedy decode of a DeepSeek-V3.2 / GLM-5-shaped sparse-MLA model with AITER's MLA decode.

One tensor-parallel rank (TP8) of a small decoder, random init: 16 local query heads (128 / 8), kv_lora_rank
512, qk_rope 64 (576-wide latent KV cache, 512-wide latent output), qk_nope 128, v_head 128, absorbed-weight
decode, and a DeepSeek-V3.2-style lightning indexer that picks the top 2048 KV positions of the context for
every decode token. The KV cache is paged with page_size 1 in a 65536-slot pool, and the physical slots come
from a shuffled free list. Prefill runs in PyTorch; every decode step calls `aiter.mla.mla_decode_fwd` once
per layer with the 2048 selected slots as `kv_indices`, the way ATOM calls it for a non-segmented MLA cache.

The greedy logits are checked at every decode step (`Logits are not finite at decode step N`).
One `RESULT` line at the end.
"""
import argparse
import math
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--requests", type=int, default=1, help="decode batch")
p.add_argument("--prompt-len", type=int, default=8192)
p.add_argument("--new-tokens", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--hidden", type=int, default=2048)
p.add_argument("--vocab", type=int, default=8192)
p.add_argument("--topk", type=int, default=2048, help="KV positions the indexer selects per token")
p.add_argument("--pool", type=int, default=65536, help="KV cache slots (page_size 1)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

from aiter import mla  # noqa: E402

dev = "cuda"
H, NOPE, ROPE, VD, LORA = 16, 128, 64, 128, 512  # one TP8 rank of 128 heads
D = LORA + ROPE  # 576: latent KV row (c_kv | k_pe)
IH, ID = 4, 128  # indexer heads, indexer head dim
SCALE = (NOPE + ROPE) ** -0.5
bf = torch.bfloat16
torch.manual_seed(args.seed)
g = torch.Generator(device="cpu").manual_seed(args.seed)


def lin(i, o):
    return (torch.randn(o, i, generator=g) / math.sqrt(i)).to(bf).to(dev)


class Layer:
    def __init__(self, hd):
        self.q = lin(hd, H * (NOPE + ROPE))
        self.kv_a = lin(hd, LORA + ROPE)
        kv_b = (torch.randn(H, NOPE + VD, LORA, generator=g) / math.sqrt(LORA)).to(bf).to(dev)
        self.w_uk, self.w_uv = kv_b[:, :NOPE], kv_b[:, NOPE:]  # (H,128,512) each
        self.o = lin(H * VD, hd)
        self.up = lin(hd, 2 * 4 * hd // 2)
        self.down = lin(4 * hd // 2, hd)
        # Lightning indexer: score(t, s) = sum_h w_h(t) * relu(q_h(t) . k(s))
        self.iq = lin(hd, IH * ID)
        self.ik = lin(hd, ID)
        self.iw = lin(hd, IH)


def rms(x, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(bf)


inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=dev).float() / ROPE))


def rope(x, pos):
    f = pos.float()[..., None] * inv_freq
    c, s = f.cos(), f.sin()
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(bf)


def qkv(ly, x, pos):
    """x (..., hidden) normed; returns absorbed q (..., H, 576) and the latent KV row (..., 576)."""
    q = (x @ ly.q.T).view(*x.shape[:-1], H, NOPE + ROPE)
    q_lat = torch.einsum("...hd,hdc->...hc", q[..., :NOPE], ly.w_uk)
    q_pe = rope(q[..., NOPE:], pos[..., None])
    kva = x @ ly.kv_a.T
    kv = torch.cat([rms(kva[..., :LORA]), rope(kva[..., LORA:], pos)], -1)
    return torch.cat([q_lat, q_pe], -1), kv


def out_mlp(ly, h, o_lat):
    o = torch.einsum("...hc,hvc->...hv", o_lat.to(bf), ly.w_uv).flatten(-2)
    h = h + o @ ly.o.T
    gu = rms(h) @ ly.up.T
    gt, up = gu.chunk(2, -1)
    return h + (torch.nn.functional.silu(gt.float()) * up.float()).to(bf) @ ly.down.T


B, T = args.requests, args.prompt_len
max_len = T + args.new_tokens
assert B * max_len <= args.pool, "--pool too small"
layers = [Layer(args.hidden) for _ in range(args.layers)]
emb = torch.randn(args.vocab, args.hidden, generator=g).to(bf).to(dev)
lm_head = lin(args.hidden, args.vocab)
prompts = torch.randint(0, args.vocab, (B, T), generator=g).to(dev)
# Page table, page_size 1: token t of request b lives in physical slot slots[b, t] of a shuffled pool.
slots = torch.randperm(args.pool, generator=g)[: B * max_len].view(B, max_len).to(torch.int32).to(dev)


def prefill(h, caches, icaches):
    """Dense causal prefill in PyTorch (fp32, query chunks); fills the latent and indexer caches."""
    pos = torch.arange(T, device=dev)
    for li, ly in enumerate(layers):
        x = rms(h)
        qa, kv = qkv(ly, x, pos.expand(B, T))
        caches[li][slots[:, :T].long()] = kv
        icaches[li][slots[:, :T].long()] = x @ ly.ik.T
        o = torch.empty(B, T, H, LORA, dtype=bf, device=dev)
        for s in range(0, T, 1024):
            qc = qa[:, s:s + 1024].float()
            sc = torch.einsum("bthd,bld->bthl", qc, kv[:, : s + 1024].float()) * SCALE
            m = pos[s:s + 1024, None] >= pos[None, : s + 1024]
            sc = sc.masked_fill(~m[None, :, None, :], float("-inf"))
            o[:, s:s + 1024] = torch.einsum("bthl,blc->bthc", sc.softmax(-1), kv[:, : s + 1024, :LORA].float()).to(bf)
        h = out_mlp(ly, h, o)
    return h


def indexer_topk(ly, x, icache, lens):
    """Top-k context positions per request (DeepSeek-V3.2 lightning indexer, simplified)."""
    sel = []
    for b in range(B):
        n = int(lens[b])
        k = icache[slots[b, :n].long()].float()  # (n, ID)
        qi = (x[b] @ ly.iq.T).view(IH, ID).float()
        w = (x[b] @ ly.iw.T).float()
        score = (torch.relu(qi @ k.T) * w[:, None]).sum(0)  # (n,)
        top = score.topk(min(args.topk, n)).indices
        sel.append(slots[b, top])
    return sel


def attn_aiter(qa, kvc, sel):
    """One sparse decode attention of one layer through aiter. qa (B,H,576) bf16, sel: per-request slots."""
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(torch.tensor([len(s) for s in sel], device=dev), 0)
    kv_indices = torch.cat(sel).to(torch.int32)
    qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=dev)
    last = torch.ones(B, dtype=torch.int32, device=dev)
    o = torch.empty(B, H, LORA, dtype=bf, device=dev)
    mla.mla_decode_fwd(qa, kvc.view(-1, 1, 1, D), o, qo_indptr, kv_indptr, kv_indices, last, 1,
                       page_size=1, sm_scale=SCALE, num_kv_splits=16)
    return o


props = torch.cuda.get_device_properties(0)
print(f"ENV torch {torch.__version__} hip {torch.version.hip} arch {props.gcnArchName}", flush=True)
print(f"requests={B} prompt={T} new={args.new_tokens} layers={args.layers} H={H} D={D} "
      f"topk={args.topk} pool={args.pool}", flush=True)
t0 = time.time()
caches = [torch.zeros(args.pool, D, dtype=bf, device=dev) for _ in layers]
icaches = [torch.zeros(args.pool, ID, dtype=bf, device=dev) for _ in layers]
with torch.no_grad():
    h = prefill(emb[prompts], caches, icaches)
    lens = torch.full((B,), T, dtype=torch.int32, device=dev)
    tok = (rms(h[:, -1]) @ lm_head.T).float().argmax(-1)
    print(f"prefill done {time.time() - t0:.1f}s", flush=True)
    st = dict(nan_logits=0, first_nan_logits="-")
    for step in range(args.new_tokens - 1):
        p_ = lens.long()
        h = emb[tok]
        lens = lens + 1
        for li, ly in enumerate(layers):
            x = rms(h)
            qa, kv = qkv(ly, x, p_)
            caches[li][slots[torch.arange(B), p_].long()] = kv
            icaches[li][slots[torch.arange(B), p_].long()] = x @ ly.ik.T
            sel = indexer_topk(ly, x, icaches[li], lens)
            o = attn_aiter(qa, caches[li], sel)
            h = out_mlp(ly, h, o)
        logits = (rms(h) @ lm_head.T).float()
        if not torch.isfinite(logits).all():
            if st["nan_logits"] == 0:
                st["first_nan_logits"] = step
                print(f"Logits are not finite at decode step {step}", flush=True)
            st["nan_logits"] += 1
        tok = logits.argmax(-1)
        if step % 32 == 0:
            print(f"step {step}: {time.time() - t0:.1f}s steps_with_nonfinite_logits={st['nan_logits']}", flush=True)
torch.cuda.synchronize()
wrong = "yes" if st["nan_logits"] else "no"
print(f"RESULT wrong={wrong} steps_with_nonfinite_logits={st['nan_logits']}/{args.new_tokens - 1} "
      f"first_nonfinite_step={st['first_nan_logits']}", flush=True)
