#!/usr/bin/env python3
"""Batched greedy decode of a DeepSeek-V3-shaped MLA model on AITER's MLA decode attention.

A small decoder (random init) with DeepSeek-V3's attention geometry: 128 heads, kv_lora_rank 512, qk_rope 64
(576-wide latent KV cache, 512-wide latent output), qk_nope 128, v_head 128, absorbed-weight decode. The KV
cache is paged with page_size 1 and the physical slots come from a shuffled free list, as in vLLM's block
table. Prefill runs in PyTorch; every decode step calls `aiter.mla.mla_decode_fwd` once per layer.

Every decode attention call is checked against an fp32 PyTorch reference on the same inputs (`--tol`), and the
whole generation is repeated `--passes` times on the same prompts: the greedy tokens of a deterministic stack
are identical in every pass. One `RESULT` line at the end.
"""
import argparse
import math
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--requests", type=int, default=512)
p.add_argument("--prompt-min", type=int, default=16)
p.add_argument("--prompt-max", type=int, default=96)
p.add_argument("--new-tokens", type=int, default=128)
p.add_argument("--passes", type=int, default=10)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--hidden", type=int, default=2048)
p.add_argument("--vocab", type=int, default=8192)
p.add_argument("--tol", type=float, default=0.03, help="max |decode attention - fp32 reference| of a correct call")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

from aiter import mla  # noqa: E402

dev = "cuda"
H, NOPE, ROPE, VD, LORA = 128, 128, 64, 128, 512
D = LORA + ROPE  # 576: latent KV row (c_kv | k_pe)
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


def rms(x, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(bf)


inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=dev).float() / ROPE))


def rope(x, pos):  # x (..., T?, ROPE) with pos broadcastable to x[..., 0]
    f = pos.float()[..., None] * inv_freq
    c, s = f.cos(), f.sin()
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(bf)


def ref_attn(q, kv, mask):
    """fp32 absorbed MLA attention. q (B,T,H,576), kv (B,L,576), mask (B,T,L) bool -> (B,T,H,512)."""
    out = []
    for i in range(0, q.shape[0], 32):  # chunked: a full-batch prefill would need tens of GB
        qc, kc, mc = q[i:i + 32].float(), kv[i:i + 32].float(), mask[i:i + 32]
        s = torch.einsum("bthd,bld->bthl", qc, kc) * SCALE
        s = s.masked_fill(~mc[:, :, None, :], float("-inf"))
        out.append(torch.einsum("bthl,blc->bthc", s.softmax(-1), kc[..., :LORA]))
    return torch.cat(out)


layers = [Layer(args.hidden) for _ in range(args.layers)]
emb = (torch.randn(args.vocab, args.hidden, generator=g)).to(bf).to(dev)
lm_head = lin(args.hidden, args.vocab)
B = args.requests
prompt_lens = torch.randint(args.prompt_min, args.prompt_max + 1, (B,), generator=g)
prompts = torch.randint(0, args.vocab, (B, args.prompt_max), generator=g).to(dev)
max_len = args.prompt_max + args.new_tokens
NSLOT = 1 << (B * max_len - 1).bit_length()
# Page table, page_size 1: each request's token t lives in a distinct, shuffled physical slot.
slots = torch.randperm(NSLOT, generator=g)[: B * max_len].view(B, max_len).to(torch.int32).to(dev)


def attn_aiter(qa, kvc, lens):
    """One decode step of one layer through aiter. qa (B,H,576) bf16, lens (B,) current kv lengths."""
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(lens, 0)
    ar = torch.arange(max_len, device=dev)
    kv_indices = slots[ar[None, :] < lens[:, None]]  # row-major: request by request
    qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=dev)
    last = torch.ones(B, dtype=torch.int32, device=dev)
    o = torch.zeros(B, H, LORA, dtype=bf, device=dev)
    mla.mla_decode_fwd(qa, kvc.view(-1, 1, 1, D), o, qo_indptr, kv_indptr, kv_indices, last, 1,
                       sm_scale=SCALE)
    return o


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


def generate(pass_i, stats):
    caches = [torch.zeros(NSLOT, D, dtype=bf, device=dev) for _ in layers]
    # Prefill (PyTorch, padded): causal within each prompt, pad positions masked.
    T = args.prompt_max
    pos = torch.arange(T, device=dev).expand(B, T)
    valid = pos < prompt_lens.to(dev)[:, None]
    mask = (pos[:, :, None] >= pos[:, None, :]) & valid[:, None, :]
    h = emb[prompts]
    for li, ly in enumerate(layers):
        qa, kv = qkv(ly, rms(h), pos)
        caches[li][slots[:, :T][valid].long()] = kv[valid]
        h = out_mlp(ly, h, ref_attn(qa, kv, mask))
    lens = prompt_lens.to(dev).to(torch.int32)
    tok = (rms(h[torch.arange(B), lens.long() - 1]) @ lm_head.T).float().argmax(-1)
    out = [tok]
    ar = torch.arange(max_len, device=dev)
    for step in range(args.new_tokens - 1):
        p_ = lens.long()  # position of the new token
        h = emb[tok]
        lens = lens + 1
        gmask = ar[None, :] < lens[:, None]  # (B, max_len)
        for li, ly in enumerate(layers):
            qa, kv = qkv(ly, rms(h), p_)
            caches[li][slots[torch.arange(B), p_].long()] = kv
            o = attn_aiter(qa, caches[li], lens)
            ref = ref_attn(qa[:, None], caches[li][slots.long()], gmask[:, None])[:, 0]
            err = (o.float() - ref).abs().amax(dim=(1, 2))  # per request
            bad = err > args.tol
            stats["calls"] += 1
            stats["worst"] = max(stats["worst"], err.max().item())
            if bad.any():
                stats["bad_calls"] += 1
                stats["bad_rows"] += int(bad.sum())
                if stats["first"] == "-":
                    r = int(bad.nonzero()[0])
                    stats["first"] = f"pass{pass_i}/step{step}/layer{li}/req{r}(kv_len={int(lens[r])})"
            h = out_mlp(ly, h, o)
        tok = (rms(h) @ lm_head.T).float().argmax(-1)
        out.append(tok)
    return torch.stack(out, 1)


props = torch.cuda.get_device_properties(0)
print(f"ENV torch {torch.__version__} hip {torch.version.hip} arch {props.gcnArchName}", flush=True)
print(f"requests={B} prompts={args.prompt_min}..{args.prompt_max} new={args.new_tokens} "
      f"max_kv_len={max_len} layers={args.layers} passes={args.passes} tol={args.tol}", flush=True)
stats = dict(calls=0, bad_calls=0, bad_rows=0, worst=0.0, first="-")
toks = []
reported = False
t0 = time.time()
for i in range(args.passes):
    toks.append(generate(i, stats))
    torch.cuda.synchronize()
    diff = int((toks[i] != toks[0]).any(1).sum())
    if diff and not reported:
        print(f"Generations differ between identical passes: pass {i} vs pass 0, {diff}/{B} requests", flush=True)
        reported = True
    print(f"pass {i}: {time.time() - t0:.1f}s bad_calls={stats['bad_calls']}/{stats['calls']} "
          f"worst_err={stats['worst']:.4f} requests_with_tokens_differing_from_pass0={diff}", flush=True)
diverged = sum(int((t != toks[0]).any()) for t in toks[1:])
div_reqs = int(torch.stack([(t != toks[0]).any(1) for t in toks]).any(0).sum())
wrong = "yes" if stats["bad_calls"] or diverged else "no"
print(f"RESULT wrong={wrong} bad_calls={stats['bad_calls']}/{stats['calls']} "
      f"bad_rows={stats['bad_rows']} worst_err={stats['worst']:.4f} tol={args.tol} "
      f"passes_diverged={diverged}/{args.passes - 1} requests_diverged={div_reqs}/{B} first_bad={stats['first']}",
      flush=True)
