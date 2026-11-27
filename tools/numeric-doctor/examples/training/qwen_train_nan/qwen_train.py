#!/usr/bin/env python3
"""Qwen2.5-0.5B training run: fp32 model, AdamW, random tokens, several trials sharing one GPU.

The training loop reports a non-finite loss, gradient, or parameter. Every parameter is an argument and the
seed is printed, so a run reproduces on its own. Only PyTorch is required. The Qwen2 model below uses Hugging
Face parameter names and reads `.safetensors` with the standard library, so the public checkpoint loads
without installing `transformers` into the image.

Each trial prints one `RESULT` line:
    RESULT trial=<i> step_reached=<n> nan_step=<i|-> nan_where=<str|-> nan_module=<str|-> ...
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from typing import Dict

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- Qwen2 model
class Qwen2Config:
    def __init__(self, **kw):
        self.hidden_size = kw["hidden_size"]
        self.intermediate_size = kw["intermediate_size"]
        self.num_hidden_layers = kw["num_hidden_layers"]
        self.num_attention_heads = kw["num_attention_heads"]
        self.num_key_value_heads = kw.get("num_key_value_heads", self.num_attention_heads)
        self.vocab_size = kw["vocab_size"]
        self.rms_norm_eps = kw.get("rms_norm_eps", 1e-6)
        self.rope_theta = kw.get("rope_theta", 1000000.0)
        self.tie_word_embeddings = kw.get("tie_word_embeddings", True)
        self.head_dim = kw.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.initializer_range = kw.get("initializer_range", 0.02)

    @classmethod
    def from_json(cls, path: str) -> "Qwen2Config":
        with open(path) as f:
            return cls(**json.load(f))

    def summary(self) -> str:
        return (
            f"hidden={self.hidden_size} inter={self.intermediate_size} layers={self.num_hidden_layers} "
            f"heads={self.num_attention_heads} kv_heads={self.num_key_value_heads} "
            f"head_dim={self.head_dim} vocab={self.vocab_size} tied={int(self.tie_word_embeddings)}"
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


def rope_tables(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(ctx)


class Qwen2MLP(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Layer(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg)
        self.mlp = Qwen2MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.config = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList([Qwen2Layer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init)

    def _init(self, m):
        std = self.config.initializer_range
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)

    def forward(self, input_ids, labels=None):
        b, s = input_ids.shape
        x = self.model.embed_tokens(input_ids)
        cos, sin = rope_tables(s, self.config.head_dim, self.config.rope_theta, x.device, x.dtype)
        mask = torch.full((s, s), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype).triu(1)
        for layer in self.model.layers:
            x = layer(x, cos, sin, mask)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1)
            )
        return logits, loss


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Read a .safetensors file with the standard library only."""
    dtypes = {
        "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool,
    }
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        body = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        buf = bytearray(body[start:end])
        t = torch.frombuffer(buf, dtype=dtypes[meta["dtype"]])
        out[name] = t.view(*meta["shape"]) if meta["shape"] else t
    return out


def load_pretrained(model: Qwen2ForCausalLM, safetensors_path: str) -> str:
    """Copy HF checkpoint weights into this model. Returns a short report."""
    sd = load_safetensors(safetensors_path)
    own = dict(model.named_parameters())
    loaded, missing = 0, []
    for name, p in own.items():
        src = sd.get(name)
        if src is None and name == "lm_head.weight":
            src = sd.get("model.embed_tokens.weight")
        if src is None:
            missing.append(name)
            continue
        with torch.no_grad():
            p.copy_(src.to(p.dtype).reshape(p.shape))
        loaded += 1
    return f"loaded {loaded}/{len(own)} tensors" + (f", missing {len(missing)}: {missing[:3]}" if missing else "")


# ---------------------------------------------------------------- training
# Qwen/Qwen2.5-0.5B config.json, revision 060db6499f32faf8b98477b0a26969ef7d8b9987 (fields used here).
QWEN25_05B = dict(
    hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
    num_key_value_heads=2, vocab_size=151936, rms_norm_eps=1e-6, rope_theta=1000000.0,
    tie_word_embeddings=True, initializer_range=0.02,
)


def make_model(args, device):
    cfg = Qwen2Config.from_json(args.config) if args.config else Qwen2Config(**QWEN25_05B)
    if args.override:
        for item in args.override.split(","):
            key, value = item.split("=")
            setattr(cfg, key, json.loads(value))
        if "num_attention_heads" in args.override or "hidden_size" in args.override:
            if "head_dim" not in args.override:
                cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    model = Qwen2ForCausalLM(cfg)
    report = "random init"
    if args.weights:
        report = load_pretrained(model, args.weights)
    return model.to(device=device, dtype=torch.float32), cfg, report


def first_nonfinite(pairs):
    for name, t in pairs:
        if t is not None and not torch.isfinite(t).all():
            return name
    return None


def trial(rank: int, args: argparse.Namespace) -> None:
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, report = make_model(args, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if rank == 0:
        print(f"[trial 0] {cfg.summary()} | {report}", flush=True)

    found = {"module": None}
    if args.check_modules:
        def hook(mod, inp, out):
            if found["module"] is None:
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    found["module"] = mod._probe_name

        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear, Qwen2Attention)):
                mod._probe_name = name
                mod.register_forward_hook(hook)

    gen = torch.Generator().manual_seed(seed)
    nan_step, nan_where = None, None
    loss0, last_loss = float("nan"), float("nan")
    step = 0
    t0 = time.time()
    for step in range(args.steps):
        ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), generator=gen).to(device)
        _, loss = model(ids, labels=ids)
        last_loss = float(loss.detach())
        if step == 0:
            loss0 = last_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()

        if not torch.isfinite(loss.detach()):
            nan_step, nan_where = step, "loss"
        elif step % args.check_every == 0:
            bad = first_nonfinite((n, p.grad) for n, p in model.named_parameters())
            if bad is not None:
                nan_step, nan_where = step, f"grad:{bad}"
        if nan_step is not None:
            break

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if step % args.check_every == 0:
            bad = first_nonfinite(model.named_parameters())
            if bad is not None:
                nan_step, nan_where = step, f"param:{bad}"
                break
        if args.log_every and step % args.log_every == 0:
            print(f"[trial {rank}] step {step} loss {last_loss:.4f} "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        f"RESULT trial={rank} step_reached={step} nan_step={nan_step if nan_step is not None else '-'} "
        f"nan_where={nan_where or '-'} nan_module={found['module'] or '-'} "
        f"loss0={loss0:.4f} loss_last={last_loss:.4f} seed={seed} tf32={int(args.tf32)} "
        f"seq={args.seq_len} bs={args.batch_size}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="", help="Qwen2 config.json (default: built-in Qwen2.5-0.5B)")
    p.add_argument("--override", default="", help='config overrides, e.g. "hidden_size=1280,num_hidden_layers=8"')
    p.add_argument("--weights", default="", help="optional model.safetensors to load")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--trials", type=int, required=True, help="training processes sharing one GPU")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--tf32", type=int, default=1)
    p.add_argument("--check-every", type=int, default=1)
    p.add_argument("--check-modules", type=int, default=1)
    p.add_argument("--log-every", type=int, default=0)
    args = p.parse_args()
    print(
        f"CONFIG {vars(args)} torch={torch.__version__} hip={torch.version.hip} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '-')}",
        flush=True,
    )
    if args.trials == 1:
        trial(0, args)
    else:
        mp.spawn(trial, args=(args,), nprocs=args.trials, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
