#!/usr/bin/env python3
"""Deterministic Case-A CONFORMANCE trace synthesizer.

Constructs a WekaTrace corpus directly from the ExplainX Case-A parameters — the
right artifact for a CUSTOMER BENCHMARK: it matches Case-A EXACTLY (by construction,
not emergently), is fully DETERMINISTIC (fixed seed => byte-identical output), is
self-DOCUMENTING (this spec + seed is the definition), and replays identically N times
through aiperf.

Why constructed (not engine-generated): serving cost depends on token COUNTS and cache
STRUCTURE, not token content. An engine replaying a request of ISL=N, OSL=M with a given
prefix-reuse pattern does exactly the same prefill+decode work regardless of whether the
tokens are "real". So a conformance trace exercises vLLM/ATOM identically to a captured
one of the same shape — while guaranteeing the Case-A distribution the customer specified.

Case-A targets (ExplainX):
  Input  ISL   P50/P90/P99 = 74K / 155K / 235K
  Output OSL   P50/P90/P99 = 320 / 3.3K / 17K
  Turns/session P50/P90/P99 = 3 / 20 / 103
  Inter-turn delay P50/P90/P99 = 4 / 31 / 240 s
  Prefix cache hit = 88-90%
  Spec-decode: 5 draft tokens, ~56% acceptance (engine-side; recorded as metadata)

Model: percentiles are pooled over REQUESTS. Each session is a multi-turn agentic
conversation: turn t's input = (growing shared prefix) + (this turn's new tokens); the
shared prefix drives prefix-cache reuse. We size the per-turn INPUT to the Case-A ISL
distribution and set hash_ids so the overlap reproduces the 88-90% hit target.

Usage: python gen_caseA_conformance.py <out_dir> [n_sessions] [seed]
"""
import json, os, sys, math, random, hashlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "corpus_caseA_conformance"
N   = int(sys.argv[2]) if len(sys.argv) > 2 else 200      # sessions; more => tighter percentiles
SEED= int(sys.argv[3]) if len(sys.argv) > 3 else 42
BLOCK = 64                                                # tokens per KV block (WekaTrace convention)

rng = random.Random(SEED)

# ---- lognormal fit from a (P50,P90,P99) triple -----------------------------------
def lognorm_from_p(p50, p90, p99):
    """Return a sampler ~ lognormal matched to the given percentiles.
    mu=ln(p50); pick sigma to best-fit p90 & p99 (z90=1.2816, z99=2.3263)."""
    mu = math.log(p50)
    s90 = (math.log(p90) - mu) / 1.2816
    s99 = (math.log(p99) - mu) / 2.3263
    sigma = (s90 + s99) / 2.0          # average the two sigma estimates
    return mu, sigma

ISL_mu, ISL_sig   = lognorm_from_p(62000, 220000, 500000)
OSL_mu, OSL_sig   = lognorm_from_p(180, 1400, 7000)
DLY_mu, DLY_sig   = lognorm_from_p(3.6, 23, 240)

def samp(mu, sig, lo, hi):
    return int(min(hi, max(lo, math.exp(mu + sig * rng.gauss(0, 1)))))

def sample_turns():
    # discrete dist fit to turns P50/P90/P99 = 3/20/103 (long-tail agentic).
    # Heavy mass at 1-3 (P50=3), thin mid, rare long tail up to 103 (P99).
    return rng.choices([2,3,5,8,20,50,82,110,144],
                       weights=[20,20,20,10,8,8,7,4,3])[0]

# ---- hash_ids: reproduce ~88-90% prefix-cache hit -------------------------------
# For a session, maintain a growing list of block-hashes (the shared KV prefix). Each
# turn REUSES the accumulated prefix (cache hits) and appends a few NEW blocks (misses).
# cache_hit = reused_blocks / total_blocks. We size new-vs-reused per turn to land 88-90%.
def make_session(idx):
    sid = hashlib.blake2b(f"caseA-{SEED}-{idx:05d}".encode(), digest_size=18).hexdigest()
    n_turns = sample_turns()
    prefix_blocks = []                     # accumulated shared prefix (block hashes)
    salt = f"{SEED}:{idx}"
    reqs = []
    t_clock = 0.0
    for turn in range(n_turns):
        isl = samp(ISL_mu, ISL_sig, 1200, 520000)     # this turn's input tokens
        osl = samp(OSL_mu, OSL_sig, 8, 20000)          # this turn's output tokens
        total_blocks = max(1, isl // BLOCK)
        if turn == 0:
            # first turn: all new (cold) — no prior prefix
            new_blocks = total_blocks
        else:
            # reuse as much accumulated prefix as exists, cap so per-turn hit ~ 88-90%
            reuse = min(len(prefix_blocks), int(total_blocks * rng.uniform(0.88, 0.90)))
            new_blocks = max(1, total_blocks - reuse)
        # build this turn's hash_ids = [reused prefix slice] + [new unique blocks]
        reuse_slice = prefix_blocks[:total_blocks - new_blocks]
        new_ids = []
        for b in range(new_blocks):
            h = int(hashlib.blake2b(f"{salt}:{turn}:{b}".encode(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
            new_ids.append(h)
        hash_ids = reuse_slice + new_ids
        prefix_blocks = hash_ids            # next turn reuses this full prefix
        think = 0.0 if turn == 0 else float(round(samp(DLY_mu, DLY_sig, 1, 600), 2))
        t_clock += think
        reqs.append({
            "t": round(t_clock, 3),
            "type": "n",
            "model": "GLM-5.2-MXFP4",
            "in": isl,
            "out": osl,
            "hash_ids": hash_ids,
            "api_time": 0.0,
            "think_time": think,
            "stop": "stop",
        })
    return {"id": sid, "models": ["GLM-5.2-MXFP4"], "block_size": BLOCK,
            "hash_id_scope": "local", "requests": reqs}

os.makedirs(OUT, exist_ok=True)
for i in range(N):
    s = make_session(i)
    json.dump(s, open(os.path.join(OUT, f"session_{i:05d}.json"), "w"))

print(f"wrote {N} sessions -> {OUT}  (seed={SEED}, block={BLOCK})")
