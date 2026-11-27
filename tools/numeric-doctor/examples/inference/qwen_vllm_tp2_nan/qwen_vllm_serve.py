#!/usr/bin/env python3
"""vLLM tensor-parallel offline serving loop: greedy generation with per-token logprobs.

A Qwen2.5-1.5B-shaped model (vLLM's `Qwen2ForCausalLM`, bf16, random init) is served at --tp 2 by
`vllm.LLM.generate()` with vLLM's defaults (torch.compile + HIP graphs); the vLLM environment comes from the caller.

Requests are --short-len tokens, except every --long-every-th one (--long-len tokens). A request whose logprobs
are not finite is the failure: the loop prints `Logprobs are not finite at request <i>` and stops.

Each process prints one `RESULT` line:
    RESULT nonfinite=<yes|no> requests_done=<n> nan_request=<i|-> ...
"""

import argparse
import functools
import json
import math
import os
import sys
import tempfile
import time

# Before vllm is imported: the engine core runs in this process; the TP workers are vLLM's multiproc workers.
# apply_model() ships a Python function to them.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch

# Qwen/Qwen2.5-1.5B config.json, architecture fields.
QWEN25_15B = dict(
    architectures=["Qwen2ForCausalLM"], model_type="qwen2", torch_dtype="bfloat16",
    hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, num_attention_heads=12,
    num_key_value_heads=2, vocab_size=151936, max_position_embeddings=32768, rms_norm_eps=1e-6,
    rope_theta=1000000.0, tie_word_embeddings=True, initializer_range=0.02, hidden_act="silu",
    use_sliding_window=False, sliding_window=None, max_window_layers=28, attention_dropout=0.0,
    bos_token_id=151643, eos_token_id=151643,
)
TOKENIZER_SIZE = 151665
PARA_ID = 271


def init_weights(model, std, seed):
    """Runs on every TP worker: the model's weights."""
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    rank, tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    torch.manual_seed(seed * 100 + rank)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.fill_(1.0)
            elif name.endswith(".bias"):
                p.zero_()
            else:
                p.normal_(0.0, std)
        emb = model.model.embed_tokens
        start = emb.shard_indices.org_vocab_start_index
        end = emb.shard_indices.org_vocab_end_index
        if start <= PARA_ID < end:
            row = emb.weight[PARA_ID - start]
            row.normal_(0.0, std)
            row[7] = 64.0
        if rank == 0:
            mlp = model.model.layers[1].mlp
            inter = mlp.gate_up_proj.weight.shape[0] // 2
            for r in (0, inter):
                mlp.gate_up_proj.weight[r].zero_()
                mlp.gate_up_proj.weight[r, 7] = 12.0
            mlp.down_proj.weight[:, 0].zero_()
            mlp.down_proj.weight[1001, 0] = 1.0
    return f"rank {rank}/{tp} {type(model).__name__} params/rank={sum(p.numel() for p in model.parameters())}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--requests", type=int, default=40, help="generate() calls")
    p.add_argument("--long-every", type=int, default=10, help="every N-th request is long (0: never)")
    p.add_argument("--short-len", type=int, default=256, help="prompt tokens of a short request")
    p.add_argument("--long-len", type=int, default=1024, help="prompt tokens of a long request")
    p.add_argument("--new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    args = p.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    print(f"CONFIG {vars(args)} torch={torch.__version__} hip={torch.version.hip} vllm={vllm.__version__} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '-')}", flush=True)

    model = tempfile.mkdtemp(prefix="qwen2.5-1.5b-")
    with open(os.path.join(model, "config.json"), "w") as f:
        json.dump(QWEN25_15B, f)
    llm = LLM(model=model, load_format="dummy", seed=args.seed, skip_tokenizer_init=True,
              tensor_parallel_size=args.tp, max_model_len=args.long_len + args.new_tokens,
              gpu_memory_utilization=0.5)
    # The workers unpickle the function by reference (a lambda does not pickle): a partial of a module function.
    desc = llm.apply_model(functools.partial(init_weights, std=QWEN25_15B["initializer_range"], seed=args.seed))
    print(f"MODEL {desc[0]}", flush=True)

    sp = SamplingParams(max_tokens=args.new_tokens, min_tokens=args.new_tokens, temperature=0.0,
                        ignore_eos=True, logprobs=0, detokenize=False)
    gen = torch.Generator().manual_seed(args.seed)
    nan_request, lp, lp0, done, n_long = None, float("nan"), float("nan"), 0, 0
    t0 = time.time()
    for r in range(args.requests):
        long = bool(args.long_every) and (r + 1) % args.long_every == 0
        n = args.long_len if long else args.short_len
        ids = torch.randint(0, TOKENIZER_SIZE, (n,), generator=gen)
        ids[ids == PARA_ID] = PARA_ID + 1
        ids[torch.randint(0, n, (max(1, n // 64),), generator=gen)] = PARA_ID
        n_long += long
        out = llm.generate([{"prompt_token_ids": ids.tolist()}], sp, use_tqdm=False)[0].outputs[0]
        lps = [d[t].logprob for t, d in zip(out.token_ids, out.logprobs)]
        lp = sum(lps) / len(lps)
        done = r + 1
        if r == 0:
            lp0 = lp
        if not math.isfinite(lp):
            nan_request = r
            print(f"Logprobs are not finite at request {r} ({n}-token prompt): mean logprob {lp}", flush=True)
            break
        if args.log_every and r % args.log_every == 0:
            print(f"request {r} prompt {n} mean_logprob {lp:.4f} {(time.time() - t0) / done:.3f}s/request",
                  flush=True)

    print(f"RESULT nonfinite={'yes' if nan_request is not None else 'no'} requests_done={done} "
          f"nan_request={nan_request if nan_request is not None else '-'} long_requests={n_long} "
          f"logprob0={lp0:.4f} logprob_last={lp:.4f} tp={args.tp} seed={args.seed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
