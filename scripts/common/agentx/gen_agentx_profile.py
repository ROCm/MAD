#!/usr/bin/env python3
"""Generic, seed-deterministic AgentX WekaTrace corpus synthesizer.

ONE generator that reads a single workload PROFILE (a set of distribution
targets + a seed) and emits a reproducible `weka_trace` corpus: one
`session_XXXXX.json` per session, in the exact schema aiperf's
`inferencex-agentx-mvp` scenario consumes:

    {id, models, block_size, hash_id_scope,
     requests:[{t, type, model, in, out, hash_ids, api_time, think_time, stop}]}

This is a straight refactor of ROCm/MAD #173's `gen_caseA_conformance.py` /
`gen_caseB_conformance.py` into a single parameterized generator. The SAMPLING
ALGORITHM is preserved byte-for-byte (lognormal-from-percentiles fit,
weighted-choice turns, growing-prefix hash_ids reuse, identical RNG call order)
so that invoking it with the Case-A / Case-B preset parameters reproduces #173's
committed corpora exactly and passes the conformance verifier 13/13.

Constructed (not engine-captured) because serving cost depends on token COUNTS
and cache STRUCTURE, not token content: a request of ISL=N, OSL=M with a given
prefix-reuse pattern does the same prefill+decode work regardless of whether the
tokens are "real". A conformance trace therefore exercises the engine identically
to a captured one of the same shape while guaranteeing the target distribution.

Profile schema (JSON/dict):
  {
    "name": "conformance_256k",     # informational
    "model_tag": "GLM-5.2-MXFP4",   # written to requests[].model + models[]
    "id_prefix": "caseA",           # session-id salt prefix (keep "caseA" to
                                    #   byte-match #173, which used it for both)
    "seed": 42,
    "n_sessions": 200,
    "block_size": 64,
    "isl_p":   [74000, 155000, 235000],   # ISL   P50/P90/P99
    "osl_p":   [320, 3300, 17000],        # OSL   P50/P90/P99
    "delay_p": [4, 31, 240],              # inter-turn delay P50/P90/P99 (s)
    "turns":   {"values": [1,2,3,...], "weights": [22,24,20,...]},
    "cache_hit": [0.88, 0.90],            # per-turn prefix-reuse band
    "clamps":  {"isl": [1200, 245000], "osl": [8, 20000], "delay": [1, 600]}
  }

Pure Python stdlib only (json/os/sys/math/random/hashlib) so it runs anywhere
without a third-party install. YAML profiles are resolved to JSON by the config
loader (scripts/common/agentx/agentx_config.py) before being handed here.

Usage:
  gen_agentx_profile.py --profile <profile.json> --out-dir <dir> [overrides]
  gen_agentx_profile.py --profile-json '{...}'   --out-dir <dir> [overrides]
  overrides: --n-sessions N  --seed S  --model-tag TAG  --id-prefix P  --block-size B
"""
import json, os, sys, math, random, hashlib

DEFAULT_MODEL_TAG = "GLM-5.2-MXFP4"
DEFAULT_ID_PREFIX = "caseA"   # #173 used the literal "caseA" salt for BOTH cases
DEFAULT_BLOCK = 64

REQUIRED_PROFILE_FIELDS = ("isl_p", "osl_p", "delay_p", "turns", "cache_hit")


def _require_fields(profile):
    missing = [k for k in REQUIRED_PROFILE_FIELDS if k not in profile]
    if missing:
        raise SystemExit(
            "[gen_agentx_profile] profile missing required field(s): "
            + ", ".join(missing))


def _argval(it, flag):
    try:
        return next(it)
    except StopIteration:
        sys.stderr.write(f"[gen_agentx_profile] {flag} requires a value\n")
        raise SystemExit(2)


def lognorm_from_p(p50, p90, p99):
    """Return (mu, sigma) of a lognormal matched to a (P50,P90,P99) triple.
    mu=ln(p50); sigma averages the p90- and p99-implied estimates
    (z90=1.2816, z99=2.3263)."""
    mu = math.log(p50)
    s90 = (math.log(p90) - mu) / 1.2816
    s99 = (math.log(p99) - mu) / 2.3263
    sigma = (s90 + s99) / 2.0
    return mu, sigma


def generate_corpus(profile, out_dir):
    """Materialize a weka_trace corpus for one workload profile into out_dir.

    The RNG call order (turns choice, then per-turn ISL gauss, OSL gauss,
    reuse uniform [t>0], delay gauss [t>0]) is identical to #173 so preset
    profiles reproduce the committed corpora byte-for-byte. Returns n_sessions.
    """
    _require_fields(profile)
    seed = int(profile.get("seed", 42))
    n = int(profile.get("n_sessions", 200))
    block = int(profile.get("block_size", DEFAULT_BLOCK))
    if block < 1:
        raise SystemExit("[gen] block_size must be >= 1")
    model_tag = str(profile.get("model_tag", DEFAULT_MODEL_TAG))
    id_prefix = str(profile.get("id_prefix", DEFAULT_ID_PREFIX))

    isl_p = profile["isl_p"]
    osl_p = profile["osl_p"]
    delay_p = profile["delay_p"]
    turns = profile["turns"]
    turns_values = list(turns["values"])
    turns_weights = list(turns["weights"])
    if not turns_values or not turns_weights:
        raise SystemExit("[gen] turns must have non-empty values and weights")
    if len(turns_values) != len(turns_weights):
        raise SystemExit("[gen] turns values and weights must have equal length")
    if any(w <= 0 for w in turns_weights):
        raise SystemExit("[gen] turns weights must all be > 0")
    cache_lo, cache_hi = profile["cache_hit"]
    clamps = profile.get("clamps", {})
    isl_lo, isl_hi = clamps.get("isl", [1200, 245000])
    osl_lo, osl_hi = clamps.get("osl", [8, 20000])
    dly_lo, dly_hi = clamps.get("delay", [1, 600])

    rng = random.Random(seed)

    ISL_mu, ISL_sig = lognorm_from_p(*isl_p)
    OSL_mu, OSL_sig = lognorm_from_p(*osl_p)
    DLY_mu, DLY_sig = lognorm_from_p(*delay_p)

    def samp(mu, sig, lo, hi):
        return int(min(hi, max(lo, math.exp(mu + sig * rng.gauss(0, 1)))))

    def sample_turns():
        return rng.choices(turns_values, weights=turns_weights)[0]

    def make_session(idx):
        sid = hashlib.blake2b(f"{id_prefix}-{seed}-{idx:05d}".encode(), digest_size=18).hexdigest()
        n_turns = sample_turns()
        prefix_blocks = []                 # accumulated shared prefix (block hashes)
        salt = f"{seed}:{idx}"
        reqs = []
        t_clock = 0.0
        for turn in range(n_turns):
            isl = samp(ISL_mu, ISL_sig, isl_lo, isl_hi)
            osl = samp(OSL_mu, OSL_sig, osl_lo, osl_hi)
            # Floor division is deliberate: it keeps generated corpora byte-for-byte
            # identical to the #173 committed corpora / pre-gate verify. Do NOT
            # change to rounding/ceil — it would break reproducibility.
            total_blocks = max(1, isl // block)
            if turn == 0:
                new_blocks = total_blocks
            else:
                reuse = min(len(prefix_blocks), int(total_blocks * rng.uniform(cache_lo, cache_hi)))
                new_blocks = max(1, total_blocks - reuse)
            reuse_slice = prefix_blocks[:total_blocks - new_blocks]
            new_ids = []
            for b in range(new_blocks):
                h = int(hashlib.blake2b(f"{salt}:{turn}:{b}".encode(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
                new_ids.append(h)
            hash_ids = reuse_slice + new_ids
            prefix_blocks = hash_ids
            think = 0.0 if turn == 0 else float(round(samp(DLY_mu, DLY_sig, dly_lo, dly_hi), 2))
            t_clock += think
            reqs.append({
                "t": round(t_clock, 3),
                "type": "n",
                "model": model_tag,
                "in": isl,
                "out": osl,
                "hash_ids": hash_ids,
                "api_time": 0.0,
                "think_time": think,
                "stop": "stop",
            })
        return {"id": sid, "models": [model_tag], "block_size": block,
                "hash_id_scope": "local", "requests": reqs}

    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        s = make_session(i)
        with open(os.path.join(out_dir, f"session_{i:05d}.json"), "w") as fh:
            json.dump(s, fh)
    return n


def _load_profile(path):
    with open(path) as f:
        return json.load(f)


def main(argv):
    profile = None
    out_dir = None
    overrides = {}
    it = iter(argv)
    for a in it:
        if a in ("--profile", "-p"):
            profile = _load_profile(_argval(it, a))
        elif a == "--profile-json":
            profile = json.loads(_argval(it, a))
        elif a in ("--out-dir", "-o"):
            out_dir = _argval(it, a)
        elif a == "--n-sessions":
            overrides["n_sessions"] = int(_argval(it, a))
        elif a == "--seed":
            overrides["seed"] = int(_argval(it, a))
        elif a == "--model-tag":
            overrides["model_tag"] = _argval(it, a)
        elif a == "--id-prefix":
            overrides["id_prefix"] = _argval(it, a)
        elif a == "--block-size":
            overrides["block_size"] = int(_argval(it, a))
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            sys.stderr.write(f"[gen_agentx_profile] unknown arg: {a}\n")
            return 2
    if profile is None or out_dir is None:
        sys.stderr.write("usage: gen_agentx_profile.py --profile P.json --out-dir DIR [overrides]\n")
        return 2
    profile.update(overrides)
    n = generate_corpus(profile, out_dir)
    print(f"wrote {n} sessions -> {out_dir}  "
          f"(seed={profile.get('seed', 42)}, block={profile.get('block_size', DEFAULT_BLOCK)}, "
          f"model={profile.get('model_tag', DEFAULT_MODEL_TAG)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
