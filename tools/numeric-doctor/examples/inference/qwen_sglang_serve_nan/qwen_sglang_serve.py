#!/usr/bin/env python3
"""SGLang offline serving loop: Qwen2.5-0.5B served through `sglang.Engine.generate()`.

A local checkpoint of the Qwen2.5-0.5B architecture (SGLang's `Qwen2ForCausalLM`, bf16, random init) serves
greedy requests given as token ids, with SGLang's defaults. Each request returns its generated tokens'
logprobs; a non-finite mean logprob stops the loop with `Logprobs are not finite at request <i>`.

The model runs in SGLang's scheduler process, which the engine starts; this process only sends requests.

Each process prints one `RESULT` line:
    RESULT requests_done=<n> nan_request=<i|-> ...
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time

import torch

# Qwen2.5-0.5B architecture.
QWEN25_05B = dict(
    architectures=["Qwen2ForCausalLM"], model_type="qwen2", torch_dtype="bfloat16",
    hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
    num_key_value_heads=2, vocab_size=151936, max_position_embeddings=32768, rms_norm_eps=0.0,
    rope_theta=1000000.0, tie_word_embeddings=True, initializer_range=0.02, hidden_act="silu",
    use_sliding_window=False, sliding_window=None, max_window_layers=24, attention_dropout=0.0,
    bos_token_id=151643, eos_token_id=151643,
)
TOKENIZER_SIZE = 151665


def aiter_jit_dir():
    """The image's aiter keeps its prebuilt kernels in a root-owned JIT directory; as another user it copies all
    4 GB of it to $HOME and fails on root-only files. Mirror it as a symlink tree in a writable directory instead
    (prebuilt modules load in place, new builds land in the mirror)."""
    import importlib.util
    spec = importlib.util.find_spec("aiter")
    if spec is None or "AITER_JIT_DIR" in os.environ:
        return
    src = os.path.join(os.path.dirname(spec.origin), "jit")
    if not os.path.isdir(src) or os.access(src, os.W_OK):
        return
    dst = tempfile.mkdtemp(prefix="aiter-jit-")
    for root, dirs, files in os.walk(src):
        out = os.path.join(dst, os.path.relpath(root, src))
        for d in dirs:
            os.makedirs(os.path.join(out, d), exist_ok=True)
        for f in files:
            os.symlink(os.path.join(root, f), os.path.join(out, f))
    os.environ["AITER_JIT_DIR"] = dst


def write_checkpoint(path, config, seed):
    """A Hugging Face checkpoint of the Qwen2.5-0.5B architecture, randomly initialized."""
    from safetensors.torch import save_file
    c = dict(QWEN25_05B, **config)
    h, inter, std = c["hidden_size"], c["intermediate_size"], c["initializer_range"]
    kv = c["num_key_value_heads"] * h // c["num_attention_heads"]
    gen = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.empty(shape).normal_(0.0, std, generator=gen).to(torch.bfloat16)

    emb = normal(c["vocab_size"], h)            # tied with lm_head
    emb[TOKENIZER_SIZE:] = 0
    w = {"model.embed_tokens.weight": emb, "model.norm.weight": torch.ones(h, dtype=torch.bfloat16)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm"):
            w[p + n + ".weight"] = torch.ones(h, dtype=torch.bfloat16)
        for n, out in (("q", h), ("k", kv), ("v", kv)):
            w[p + f"self_attn.{n}_proj.weight"] = normal(out, h)
            w[p + f"self_attn.{n}_proj.bias"] = torch.zeros(out, dtype=torch.bfloat16)
        w[p + "self_attn.o_proj.weight"] = normal(h, h)
        w[p + "mlp.gate_proj.weight"] = normal(inter, h)
        w[p + "mlp.up_proj.weight"] = normal(inter, h)
        w[p + "mlp.down_proj.weight"] = normal(h, inter)
    save_file(w, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(c, f)
    return f"Qwen2ForCausalLM params={sum(t.numel() for t in w.values())}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=8)
    p.add_argument("--requests", type=int, default=200, help="generate() calls")
    p.add_argument("--mix-every", type=int, default=50,
                   help="every N-th request carries one id from the rest of the model's id range (0: never)")
    p.add_argument("--config", default="{}", help="JSON of config.json fields to override")
    p.add_argument("--engine-args", default="{}",
                   help="JSON of extra sglang.Engine arguments ({}: SGLang's defaults)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    args = p.parse_args()

    aiter_jit_dir()             # before sglang imports aiter
    import sglang
    from sglang.version import __version__ as sglang_version
    print(f"CONFIG {vars(args)} torch={torch.__version__} hip={torch.version.hip} sglang={sglang_version} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '-')}", flush=True)

    model = tempfile.mkdtemp(prefix="qwen2.5-0.5b-")
    desc = write_checkpoint(model, json.loads(args.config), args.seed)
    print(f"MODEL {desc} engine_args={args.engine_args}", flush=True)
    engine = sglang.Engine(model_path=model, skip_tokenizer_init=True, random_seed=args.seed,
                           context_length=args.prompt_len + args.new_tokens, **json.loads(args.engine_args))

    # Exactly --new-tokens tokens per request: greedy, no EOS, and the generated tokens' logprobs.
    sp = dict(max_new_tokens=args.new_tokens, temperature=0.0, ignore_eos=True)
    gen = torch.Generator().manual_seed(args.seed)
    nan_request, lp, lp0, done = None, float("nan"), float("nan"), 0
    t0 = time.time()
    try:
        for r in range(args.requests):
            ids = torch.randint(0, TOKENIZER_SIZE, (args.prompt_len,), generator=gen)
            pos = int(torch.randint(0, args.prompt_len - 1, (1,), generator=gen))
            extra = int(torch.randint(TOKENIZER_SIZE, QWEN25_05B["vocab_size"], (1,), generator=gen))
            if args.mix_every and (r + 1) % args.mix_every == 0:
                ids[pos] = extra
            out = engine.generate(input_ids=ids.tolist(), sampling_params=sp, return_logprob=True)
            lps = [t[0] for t in out["meta_info"]["output_token_logprobs"]]
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
    finally:
        engine.shutdown()

    print(
        f"RESULT requests_done={done} nan_request={nan_request if nan_request is not None else '-'} "
        f"logprob0={lp0:.4f} logprob_last={lp:.4f} seed={args.seed} prompt={args.prompt_len} "
        f"engine_args={args.engine_args.replace(' ', '')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
