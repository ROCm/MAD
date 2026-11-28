"""Test only (tests/diff T3/T4): a REAL checkpoint served through vLLM or SGLang, greedy, fixed prompts.

    python3 tests/diff/serve_real.py --engine vllm|sglang --model <checkpoint dir> [--requests 40] [--new-tokens 32]

The examples load random weights, whose output distribution is nearly flat: a planted head swap moved their
logprobs by about one bf16 step and no comparison could see it (2026-11-24). A trained model's peaked
distributions show a wrong value the way a user's model would. One generate() call per request, as the examples
do; prompts are fixed templates x topics, tokenized with the checkpoint's own tokenizer.json. Prints
`RESULT requests_done=<n>`.
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

TEMPLATES = ["Explain {} in simple terms.", "Write a short poem about {}.", "List three facts about {}.",
             "What is the history of {}?", "Give an example of {} in everyday life."]
TOPICS = ["photosynthesis", "the moon", "prime numbers", "volcanoes", "the printing press", "honey bees",
          "the Pacific Ocean", "electric motors"]


def prompts(n):
    out = []
    for i in range(n):
        out.append(TEMPLATES[i % len(TEMPLATES)].format(TOPICS[(i // len(TEMPLATES)) % len(TOPICS)]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--enforce-eager", type=int, default=0)
    args = p.parse_args()
    import math
    nnodes = int(os.environ.get("ND_NNODES", "1"))         # nd's multi-node variables (wave.sh `all` lines)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    ids = [tok.encode(s).ids for s in prompts(args.requests)]
    done = 0
    if args.engine == "vllm":
        from vllm import LLM, SamplingParams
        kw = {}
        if args.tp * args.pp > 1:
            kw = dict(tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
                      distributed_executor_backend="mp")
        if nnodes > 1:
            kw.update(nnodes=nnodes, node_rank=int(os.environ["ND_NODE_RANK"]),
                      master_addr=os.environ["ND_MASTER_ADDR"], master_port=int(os.environ.get("MASTER_PORT", "29511")))   # wave.sh: one per copy
        llm = LLM(model=args.model, seed=args.seed, skip_tokenizer_init=True, max_model_len=512,
                  gpu_memory_utilization=0.3, enforce_eager=bool(args.enforce_eager), **kw)
        if nnodes > 1 and int(os.environ["ND_NODE_RANK"]) > 0:
            print("RESULT requests_done=0 (worker node)", flush=True)
            return
        sp = SamplingParams(max_tokens=args.new_tokens, min_tokens=args.new_tokens, temperature=0.0,
                            ignore_eos=True, detokenize=False, logprobs=0)
        for i, x in enumerate(ids):
            out = llm.generate([{"prompt_token_ids": x}], sp, use_tqdm=False)[0].outputs[0]
            done += 1
            lp = [d[t].logprob for t, d in zip(out.token_ids, out.logprobs or [])]
            if lp and not all(math.isfinite(v) for v in lp):
                print("Logprobs are not finite at request %d" % i, flush=True)
            if i % 10 == 0:
                print("request %d tokens %s" % (i, list(out.token_ids)[:8]), flush=True)
    else:
        import sglang
        eng = sglang.Engine(model_path=args.model, skip_tokenizer_init=True, random_seed=args.seed,
                            mem_fraction_static=0.3, context_length=512, tp_size=args.tp, pp_size=args.pp)
        sp = {"temperature": 0.0, "max_new_tokens": args.new_tokens, "ignore_eos": True}
        for i, x in enumerate(ids):
            out = eng.generate(input_ids=x, sampling_params=sp, return_logprob=True)
            done += 1
            lp = [t[0] for t in out["meta_info"].get("output_token_logprobs", [])]
            if lp and not all(v is not None and math.isfinite(v) for v in lp):
                print("Logprobs are not finite at request %d" % i, flush=True)
            if i % 10 == 0:
                print("request %d tokens %s" % (i, out["output_ids"][:8]), flush=True)
        eng.shutdown()
    print("RESULT requests_done=%d" % done, flush=True)


if __name__ == "__main__":
    main()
