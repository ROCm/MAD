#!/usr/bin/env python3
"""Hugging Face `transformers` inference: Qwen2.5-0.5B greedy `generate()` in fp32.

Qwen2.5-0.5B (`Qwen2ForCausalLM` from `transformers`) serves requests of random-token prompts with greedy
`generate()`, several inference processes per GPU. Every request runs under `torch.inference_mode()`.

The serving loop reports a failure itself: each request's mean log-probability of its generated tokens is read
back to the host, and a non-finite value stops the process with `Logits are not finite`.

Every parameter is an argument and the seed is printed, so a run reproduces on its own. Each process prints
one `RESULT` line:
    RESULT trial=<i> requests_done=<n> nan_request=<i|-> nan_token=<i|-> nan_module=<str|-> ...
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, GenerationConfig, Qwen2Config

# Qwen/Qwen2.5-0.5B config.json (revision 060db6499f32faf8b98477b0a26969ef7d8b9987), architecture fields.
QWEN25_05B = dict(
    hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
    num_key_value_heads=2, vocab_size=151936, max_position_embeddings=32768, rms_norm_eps=1e-6,
    rope_theta=1000000.0, tie_word_embeddings=True, initializer_range=0.02, hidden_act="silu",
    use_sliding_window=False, bos_token_id=151643, eos_token_id=151643,
)


def make_model(args, device):
    if args.model_dir:          # a local checkpoint folder: config.json + model.safetensors
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32,
                                                     attn_implementation=args.attn)
        report = f"weights from {args.model_dir}"
    else:
        model = AutoModelForCausalLM.from_config(Qwen2Config(**QWEN25_05B), dtype=torch.float32,
                                                 attn_implementation=args.attn)
        report = "random init"
    return model.to(device).eval(), report


def trial(rank: int, args: argparse.Namespace) -> None:
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, report = make_model(args, device)
    cfg = model.config
    if rank == 0:
        print(f"[trial 0] {type(model).__name__} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
              f"vocab={cfg.vocab_size} attn={model.config._attn_implementation} "
              f"params={sum(p.numel() for p in model.parameters())} | {report}", flush=True)

    found = {"module": None}
    if args.check_modules:
        def hook(mod, inp, out):
            if found["module"] is None:
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    found["module"] = mod._probe_name

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) or name.endswith("self_attn"):
                mod._probe_name = name
                mod.register_forward_hook(hook)

    # Every request generates exactly --new-tokens tokens: no EOS, greedy.
    gcfg = GenerationConfig(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False,
                            pad_token_id=0, eos_token_id=None, return_dict_in_generate=True, output_logits=True)
    gen = torch.Generator().manual_seed(seed)
    mask = torch.ones(args.batch_size, args.prompt_len, dtype=torch.long, device=device)
    nan_request, nan_token, lp, lp0 = None, None, float("nan"), float("nan")
    done = 0
    t0 = time.time()
    for r in range(args.requests):
        ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.prompt_len), generator=gen).to(device)
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=mask, generation_config=gcfg)
            logits = torch.stack(out.logits, dim=1)                     # (batch, new_tokens, vocab)
            new = out.sequences[:, args.prompt_len:]
            lp = float(torch.log_softmax(logits, dim=-1).gather(-1, new[..., None]).mean())
        done = r + 1
        if r == 0:
            lp0 = lp
        if not math.isfinite(lp):
            nan_request = r
            bad = (~torch.isfinite(logits).all(dim=-1)).any(dim=0).nonzero()
            nan_token = int(bad[0]) if len(bad) else None
            print(f"[trial {rank}] Logits are not finite at request {r} (generated token "
                  f"{nan_token if nan_token is not None else '?'}): mean logprob {lp}", flush=True)
            break
        if args.log_every and r % args.log_every == 0:
            print(f"[trial {rank}] request {r} mean_logprob {lp:.4f} "
                  f"{(time.time() - t0) / done:.3f}s/request", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"RESULT trial={rank} requests_done={done} nan_request={nan_request if nan_request is not None else '-'} "
        f"nan_token={nan_token if nan_token is not None else '-'} nan_module={found['module'] or '-'} "
        f"logprob0={lp0:.4f} logprob_last={lp:.4f} seed={seed} tf32={int(args.tf32)} "
        f"prompt={args.prompt_len} new={args.new_tokens} bs={args.batch_size} attn={args.attn}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="", help="local checkpoint folder (default: built-in Qwen2.5-0.5B, random init)")
    p.add_argument("--prompt-len", type=int, required=True)
    p.add_argument("--new-tokens", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--requests", type=int, required=True, help="generate() calls per process")
    p.add_argument("--trials", type=int, required=True, help="inference processes (all on one GPU)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tf32", type=int, default=1)
    p.add_argument("--attn", default="eager", choices=("eager", "sdpa"))
    p.add_argument("--check-modules", type=int, default=1)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--device", default="", help="default: cuda if available, else cpu")
    args = p.parse_args()
    import transformers
    print(
        f"CONFIG {vars(args)} torch={torch.__version__} hip={torch.version.hip} "
        f"transformers={transformers.__version__} HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '-')}",
        flush=True,
    )
    if args.trials == 1:
        trial(0, args)
    else:
        mp.spawn(trial, args=(args,), nprocs=args.trials, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
