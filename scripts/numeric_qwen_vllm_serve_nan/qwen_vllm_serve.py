#!/usr/bin/env python3
"""vLLM offline serving loop for Qwen2.5-0.5B.

vLLM's `Qwen2ForCausalLM` with the Qwen2.5-0.5B architecture in bf16 (random init) serves greedy requests through
`vllm.LLM.generate()` with vLLM's defaults. Prompts are given as token ids; each request generates --new-tokens
tokens and returns their logprobs, which must be finite.

Each process prints one `RESULT` line:
    RESULT requests_done=<n> nan_request=<i|-> logprob0=<f> logprob_last=<f> ...
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time

# Before vllm is imported: the engine core and the worker run in this process (one GPU, no tensor parallelism).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

# Qwen2.5-0.5B architecture fields.
QWEN25_05B = dict(
    architectures=["Qwen2ForCausalLM"], model_type="qwen2", torch_dtype="bfloat16",
    hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
    num_key_value_heads=2, vocab_size=151936, max_position_embeddings=32768, rms_norm_eps=0.0,
    rope_theta=1000000.0, tie_word_embeddings=True, initializer_range=0.02, hidden_act="silu",
    use_sliding_window=False, sliding_window=None, max_window_layers=24, attention_dropout=0.0,
    bos_token_id=151643, eos_token_id=151643,
)
TOKENIZER_SIZE = 151665


def init_weights(model, std):
    """The `transformers` Qwen2 initialization."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.fill_(1.0)
            elif name.endswith(".bias"):
                p.zero_()
            else:
                p.normal_(0.0, std)
        model.model.embed_tokens.weight[TOKENIZER_SIZE:].zero_()
    return f"{type(model).__name__} params={sum(p.numel() for p in model.parameters())}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=8)
    p.add_argument("--requests", type=int, default=200, help="generate() calls")
    p.add_argument("--enforce-eager", type=int, default=0,
                   help="0: vLLM's default (torch.compile + HIP graphs); 1: eager")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    args = p.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    print(f"CONFIG {vars(args)} torch={torch.__version__} hip={torch.version.hip} vllm={vllm.__version__} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '-')}", flush=True)

    model = tempfile.mkdtemp(prefix="qwen2.5-0.5b-")
    with open(os.path.join(model, "config.json"), "w") as f:
        json.dump(QWEN25_05B, f)
    llm = LLM(model=model, load_format="dummy", seed=args.seed, skip_tokenizer_init=True,
              enforce_eager=bool(args.enforce_eager), max_model_len=args.prompt_len + args.new_tokens)
    desc = llm.apply_model(lambda m: init_weights(m, QWEN25_05B["initializer_range"]))[0]
    print(f"MODEL {desc} enforce_eager={args.enforce_eager}", flush=True)

    # Exactly --new-tokens tokens per request: greedy, no EOS, and the generated tokens' logprobs.
    sp = SamplingParams(max_tokens=args.new_tokens, min_tokens=args.new_tokens, temperature=0.0,
                        ignore_eos=True, logprobs=0, detokenize=False)
    gen = torch.Generator().manual_seed(args.seed)
    nan_request, lp, lp0, done = None, float("nan"), float("nan"), 0
    t0 = time.time()
    for r in range(args.requests):
        ids = torch.randint(0, TOKENIZER_SIZE, (args.prompt_len,), generator=gen)
        pos = int(torch.randint(0, args.prompt_len - 1, (1,), generator=gen))
        tok = int(torch.randint(TOKENIZER_SIZE, QWEN25_05B["vocab_size"], (1,), generator=gen))
        if (r + 1) % 50 == 0:
            ids[pos] = tok
        out = llm.generate([{"prompt_token_ids": ids.tolist()}], sp, use_tqdm=False)[0].outputs[0]
        lps = [d[t].logprob for t, d in zip(out.token_ids, out.logprobs)]
        lp = sum(lps) / len(lps)
        done = r + 1
        if r == 0:
            lp0 = lp
        if not math.isfinite(lp):
            nan_request = r
            print(f"Logprobs are not finite at request {r}: mean logprob {lp}", flush=True)
            break
        if args.log_every and r % args.log_every == 0:
            print(f"request {r} mean_logprob {lp:.4f} {(time.time() - t0) / done:.3f}s/request", flush=True)

    print(
        f"RESULT requests_done={done} nan_request={nan_request if nan_request is not None else '-'} "
        f"logprob0={lp0:.4f} logprob_last={lp:.4f} seed={args.seed} prompt={args.prompt_len} "
        f"eager={args.enforce_eager}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
