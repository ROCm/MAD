#!/usr/bin/env python3
"""
Generate an explicit ISL/OSL distribution as a JSONL for `vllm bench serve
--dataset-name custom`.

WHY THIS EXISTS
---------------
The customer specified the workload as a DISTRIBUTION inside a CONTEXT WINDOW:

    ISL/OSL (p50 / p95 / p99)   avg: 80K/1K  (256K context)
                                avg: 200K/1K (1M context)

benchmark_customer_slo.sh originally sent a single fixed --random-input-len 80000.
That is the *average* and nothing else: no spread, and the 256K/1M window is never
touched, so "it works at 256K context" would be assumed rather than measured.

The obvious fix -- `--random-range-ratio` -- provably cannot express this. In
vllm/benchmarks/datasets/utils.py::get_sampling_params the draw is UNIFORM and
SYMMETRIC about the mean:

    input_low  = floor(mean * (1 - r))
    input_high = ceil (mean * (1 + r))
    if not (0.0 <= r < 1.0): raise ValueError

so the widest expressible support is [0, 2*mean]. Reaching a 256K tail from an 80K
mean needs r = 2.28; reaching 1M from 200K needs r = 4.24. Both are rejected by that
validator. It is also the wrong SHAPE -- uniform means the mean sits at the middle of
the range, whereas real agentic traffic is right-skewed (many short turns, a few very
long accumulated contexts).

So we generate the lengths ourselves and feed them through CustomDataset, which reads
`{"prompt": ..., "output_tokens": N}` per line and therefore accepts an arbitrary
per-request length distribution.

WHAT WE ASSUME, AND WHY IT IS DECLARED RATHER THAN HIDDEN
---------------------------------------------------------
The sheet's header says "p50 / p95 / p99" but the cell only gives "avg: 80K" and a
window. Two of the three numbers we would need are simply not in the sheet. Rather
than invent three percentiles and present them as the customer's, we fix the two
things they DID state -- the mean and the window -- and pick the least-committal
shape that honours both:

    lognormal, solved so that   mean = <the stated avg>
                                p99  = TAIL_FRAC * <the stated window>

TAIL_FRAC defaults to 1.0, i.e. the p99 request is a full-window request. That is the
aggressive reading and it is the one worth measuring, because it is the reading under
which the deployment is claimed to support a 256K/1M context at all.

The achieved percentiles are RE-MEASURED from the generated sample and printed. They
are not the requested ones: clamping at the window and integer rounding both move
them. Report what came out, not what was asked for.

SAMPLING ERROR IS THE DOMINANT DEVIATION, AND IT IS REPORTED, NOT HIDDEN
------------------------------------------------------------------------
A lognormal with these parameters has CV = 0.64 (256K row) and 1.08 (1M row), so the
standard error of the realised mean is CV/sqrt(n):

    256K row, n=256  ->  4.0% of the mean
    1M   row, n=128  ->  9.5% of the mean

Measured across 12 seeds the realised mean spanned 72,884-84,222 (14.2%) and
166,447-237,295 (35.4%). This is inherent to drawing few samples from a heavy tail --
not a solver bug. Reaching a 2% standard error would need n=1,014 and n=2,903, which is
not affordable at 80K-200K tokens per request.

Two consequences, both deliberate:

  * We do NOT stratify. Inverse-CDF placement would pin the mean to within 1-2% but it
    caps the sample at the (n-0.5)/n quantile, so at n=128 the p99 reaches only 224,884
    instead of 262,144 -- it would buy mean fidelity by giving up the very thing the
    window number is meant to test. The tail is the customer's stated requirement; the
    mean deviation is a measurable property we can simply state.
  * We therefore REPORT the deviation. Every run writes <out>.meta.json carrying the
    achieved distribution, the deviation from target as a percentage, the seed, and the
    lognormal parameters -- so what goes to the customer is "our trace averaged 78,412
    tokens, 2.0% below the 80,000 target, p99 262,144" rather than an unqualified claim
    to have hit 80K. Averaging several runs at DISTINCT seeds shrinks the error as
    1/sqrt(n_total): 10 seeds takes 4.0% -> 1.3% and 9.5% -> 3.0%. That is available
    from either wrapper as SLO_ITERS=10 RESAMPLE_PER_ITER=1, but it is not the default
    -- it costs ten times the wall clock, and the default (one warm-up pass plus one
    measured pass over the same trace) reports the deviation instead of averaging it
    away.

  Solving the lognormal: with X ~ LogN(mu, s),
      ln(mean) = mu + s^2/2 ;  ln(p99) = mu + z99*s      (z99 = 2.32635)
  Subtracting gives  s^2/2 - z99*s + ln(p99/mean) = 0, so
      s = z99 - sqrt(z99^2 - 2*ln(p99/mean))   (smaller root -> less skew)
  which is real only while p99/mean <= exp(z99^2/2) = 14.97. Both customer rows are
  well inside that (3.2x and 5.0x), but the guard is here because a future row may
  not be.

SHARED PREFIX
-------------
Agentic traffic re-sends a large, mostly-static preamble every turn. We emit the SAME
prefix text at the head of every prompt, so the server's prefix cache can actually hit
it -- this is the thing the customer meant by "interested in long prefix caching".
--prefix-frac is a fraction of the *median*, not of each request, so the shared block
is a fixed size and only the per-request remainder varies (which is how a real agent
behaves: constant system+tools, growing scratchpad).

KV CAPACITY
-----------
A right-skewed ISL is not free: KV is charged on the ACTUAL length, so the tail costs
real HBM. GLM-5.2 MLA KV is (kv_lora_rank 512 + qk_rope_head_dim 64) = 576
elem/tok/layer, FP8, 78 layers, which computes to 43.88 KiB/token. The value MEASURED
on MI308X from the engine's `GPU KV cache size` line is 46.58 KiB/token -- 6.2% higher,
because vLLM rounds the per-layer allocation up to whole blocks. The default below is
the MEASURED one, since the whole point of this section is to predict an OOM before it
happens and the arithmetic value under-predicts by 6%. Override with
KV_BYTES_PER_TOKEN=<bytes> on another platform; the value used is recorded in the
.meta.json sidecar so a downstream number can always be traced to its constant.

A single 262,144-token request is 11.65 GiB of KV on its own. This script prints the
mean and p99 KV cost per request so the concurrency you choose is a decision, not an
OOM you discover later.

USAGE
    python3 gen_workload.py --mean-isl 80000 --context-window 262144 \\
        --osl 1024 --num-prompts 256 --tokenizer /models/GLM-5.2-FP8 \\
        --out /run_logs/wl_256k.jsonl

    # shape-only, no tokenizer needed -- prints the distribution and exits
    python3 gen_workload.py --mean-isl 80000 --context-window 262144 --dry-run

Then:
    vllm bench serve --dataset-name custom --dataset-path /run_logs/wl_256k.jsonl \\
        --skip-chat-template --custom-output-len -1 ...

--skip-chat-template matters: CustomDataset applies the chat template by default,
which prepends role tokens and would silently shift every length we just spent this
much effort placing.
"""

import argparse
import json
import math
import os
import sys

Z99 = 2.3263478740408408  # scipy.stats.norm.ppf(0.99)

# GLM-5.2 MLA KV, bytes/token. See module docstring: 46.58 KiB is MEASURED on MI308X,
# not the 43.88 KiB the element arithmetic gives, and it is the measured one that
# predicts the OOM correctly. Env-overridable for other platforms/dtypes.
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN") or 46.58 * 1024)


def solve_lognormal(mean, p99):
    """Return (mu, sigma) of a lognormal with the given mean and 99th percentile."""
    ratio = p99 / float(mean)
    if ratio <= 1.0:
        raise ValueError(
            "p99 target (%d) must exceed the mean (%d); a distribution cannot have "
            "its 99th percentile at or below its mean." % (p99, mean)
        )
    disc = Z99 * Z99 - 2.0 * math.log(ratio)
    if disc < 0:
        raise ValueError(
            "p99/mean = %.2f exceeds the maximum %.2f attainable by a lognormal. "
            "Lower --tail-frac or raise --mean-isl." % (ratio, math.exp(Z99 * Z99 / 2))
        )
    sigma = Z99 - math.sqrt(disc)
    mu = math.log(mean) - sigma * sigma / 2.0
    return mu, sigma


def percentile(sorted_vals, q):
    """Nearest-rank percentile. No numpy dependency -- this runs on the head node."""
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(math.ceil(q / 100.0 * len(sorted_vals))) - 1))
    return sorted_vals[k]


def sample_lengths(n, mean, window, tail_frac, floor_tokens, seed):
    """Draw n integer input lengths, clamped to [floor_tokens, window]."""
    import random

    rng = random.Random(seed)
    mu, sigma = solve_lognormal(mean, tail_frac * window)
    lens = []
    for _ in range(n):
        v = int(round(math.exp(rng.gauss(mu, sigma))))
        lens.append(max(floor_tokens, min(window, v)))
    return lens, mu, sigma


def describe(lens, window, label="", target_mean=None, target_p99=None):
    """Summarise a sample. Deviation vs target is printed as a PERCENTAGE because that
    is the form the number has to travel in: "our trace averaged 78,412 tokens, 2.0%
    below the 80,000 target" is reportable, "78412" alone invites the reader to assume
    it was meant to be 80,000 exactly."""
    s = sorted(lens)
    n = len(s)
    mean = sum(s) / float(n)
    p99 = percentile(s, 99)
    stats = {
        "n": n,
        "min": s[0],
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": p99,
        "max": s[-1],
        "mean": mean,
        "at_window": sum(1 for v in s if v >= window),
    }
    if target_mean:
        stats["mean_dev_pct"] = 100.0 * (mean - target_mean) / float(target_mean)
    if target_p99:
        stats["p99_dev_pct"] = 100.0 * (p99 - target_p99) / float(target_p99)

    dev = ""
    if target_mean:
        dev = "  [mean %+.1f%% vs target %d]" % (stats["mean_dev_pct"], target_mean)
    print("  %-9s n=%d  mean=%.0f  p50=%d  p95=%d  p99=%d  max=%d  (%d clamped at window)%s"
          % (label, stats["n"], mean, stats["p50"], stats["p95"],
             stats["p99"], stats["max"], stats["at_window"], dev))
    return stats


def expected_se_pct(sigma, n):
    """Standard error of the sample mean, as a percentage of the mean.

    For X ~ LogN(mu, sigma) the coefficient of variation is sqrt(exp(sigma^2) - 1),
    independent of mu, and SE(mean)/mean = CV/sqrt(n). Printed alongside the achieved
    mean so a reader can tell a normal draw from a genuinely anomalous one instead of
    guessing -- a realised mean 4% off target at n=256 is expected, not a defect.
    """
    cv = math.sqrt(math.exp(sigma * sigma) - 1.0)
    return cv, 100.0 * cv / math.sqrt(n)


def kv_report(stats, concurrency_list):
    gib = 1024.0 ** 3
    mean_gib = stats["mean"] * KV_BYTES_PER_TOKEN / gib
    p99_gib = stats["p99"] * KV_BYTES_PER_TOKEN / gib
    max_gib = stats["max"] * KV_BYTES_PER_TOKEN / gib
    print("  KV per request: mean %.2f GiB, p99 %.2f GiB, max %.2f GiB"
          % (mean_gib, p99_gib, max_gib))
    if concurrency_list:
        print("  KV at concurrency (mean-length steady state / all-p99 worst case):")
        for c in concurrency_list:
            print("    con=%-4d  %8.0f GiB  /  %8.0f GiB" % (c, c * mean_gib, c * p99_gib))
        # Measured MI308X decode-tier pools, from the engine's own boot line:
        #   EP8  (1P/1D, util 0.80): 35.71 GiB/rank x  8 =   286 GiB
        #   EP16 (2P/2D, util 0.72): 64.19 GiB/rank x 16 = 1,027 GiB
        # EP16 is 3.6x, not 2x: sharding the MoE 16 ways also frees 42 GiB/GPU of
        # weights, so each rank holds more KV as well as there being twice as many.
        print("  (MI308X decode-tier KV pool: ~286 GiB at EP8, ~1027 GiB at EP16)")


def build_prompts(lens, tokenizer_path, prefix_tokens, out_path, osl, trust_remote_code):
    """Materialise prompts whose re-tokenised length matches the drawn length."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    vocab = tok.vocab_size
    special = set(tok.all_special_ids or [])
    allowed = [t for t in range(1000, min(vocab, 60000)) if t not in special]
    if not allowed:
        raise RuntimeError("tokenizer produced no usable token ids")

    import random
    rng = random.Random(1234)

    # One shared prefix, identical text on every request -> real prefix-cache hits.
    prefix_ids = [allowed[rng.randrange(len(allowed))] for _ in range(prefix_tokens)]
    prefix_text = tok.decode(prefix_ids) if prefix_tokens else ""
    prefix_real = len(tok(prefix_text).input_ids) if prefix_tokens else 0
    print("  shared prefix: requested %d tok, materialised %d tok"
          % (prefix_tokens, prefix_real))

    achieved = []
    with open(out_path, "w") as f:
        for i, target in enumerate(lens):
            body = max(1, target - prefix_real)
            ids = [allowed[(rng.randrange(len(allowed)) + i) % len(allowed)]
                   for _ in range(body)]
            text = prefix_text + tok.decode(ids)
            got = len(tok(text).input_ids)
            # decode -> re-encode is NOT length preserving: the decoded text can
            # re-tokenise into a different number of tokens because adjacent pieces
            # merge. Converge by adjusting the body length by the observed error.
            # Bounded at 4 passes -- this is a benchmark input, not a proof; we
            # report the achieved distribution rather than pretending it is exact.
            for _ in range(4):
                if got == target:
                    break
                body = max(1, body + (target - got))
                ids = [allowed[(rng.randrange(len(allowed)) + i) % len(allowed)]
                       for _ in range(body)]
                text = prefix_text + tok.decode(ids)
                got = len(tok(text).input_ids)
            achieved.append(got)
            # "input_tokens" is NOT read by CustomDataset (it only looks at "prompt"
            # and "output_tokens"; extra keys ride along unused). It is written so
            # downstream tooling can size timeouts from the real token count instead
            # of estimating from character length.
            f.write(json.dumps({"prompt": text, "output_tokens": int(osl),
                                "input_tokens": int(got)}) + "\n")
            if (i + 1) % 32 == 0:
                print("    ... %d/%d" % (i + 1, len(lens)), file=sys.stderr)
    return achieved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mean-isl", type=int, required=True,
                    help="the 'avg' from the customer sheet, e.g. 80000")
    ap.add_argument("--context-window", type=int, required=True,
                    help="the window from the sheet, e.g. 262144 or 1048576")
    ap.add_argument("--tail-frac", type=float, default=1.0,
                    help="p99 target as a fraction of the window (default 1.0 = the "
                         "p99 request fills the window)")
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--prefix-frac", type=float, default=0.5,
                    help="shared cacheable prefix as a fraction of the MEDIAN isl")
    ap.add_argument("--min-isl", type=int, default=1024,
                    help="floor, so the left tail stays a plausible agent turn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="",
                    help="model path; required unless --dry-run")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the distribution and KV cost; write nothing")
    ap.add_argument("--concurrency", default="16 32 64 128 256",
                    help="space-separated list, for the KV table only")
    a = ap.parse_args()

    print("workload: mean ISL %d, window %d, p99 target %d (tail-frac %.2f), OSL %d"
          % (a.mean_isl, a.context_window, int(a.tail_frac * a.context_window),
             a.tail_frac, a.osl))

    try:
        lens, mu, sigma = sample_lengths(a.num_prompts, a.mean_isl, a.context_window,
                                         a.tail_frac, a.min_isl, a.seed)
    except ValueError as e:
        print("ERROR: %s" % e)
        return 2
    p99_target = int(a.tail_frac * a.context_window)
    cv, se_pct = expected_se_pct(sigma, a.num_prompts)
    print("  lognormal mu=%.4f sigma=%.4f  (analytic p50=%.0f, mean=%.0f)"
          % (mu, sigma, math.exp(mu), math.exp(mu + sigma * sigma / 2)))
    print("  CV=%.3f -> expected SE of the realised mean at n=%d is %.1f%%. A sample mean"
          % (cv, a.num_prompts, se_pct))
    print("  within ~2 SE (%.1f%%) of target is a normal draw, NOT a defect; average"
          % (2 * se_pct))
    print("  several runs at DISTINCT --seed to shrink it as 1/sqrt(n).")

    stats = describe(lens, a.context_window, "requested", a.mean_isl, p99_target)
    cons = [int(x) for x in a.concurrency.split()] if a.concurrency else []
    kv_report(stats, cons)

    if a.dry_run:
        print("dry run -- no file written")
        return 0

    if not a.tokenizer or not a.out:
        print("ERROR: --tokenizer and --out are required unless --dry-run")
        return 2

    prefix_tokens = int(stats["p50"] * a.prefix_frac)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    achieved = build_prompts(lens, a.tokenizer, prefix_tokens, a.out, a.osl,
                             a.trust_remote_code)
    ach = describe(achieved, a.context_window, "achieved", a.mean_isl, p99_target)

    # Sidecar. The ACHIEVED distribution is what we actually served, so it is what gets
    # quoted upstream -- see the module docstring. Written next to the JSONL rather than
    # only to stdout because stdout is a 30k-line benchmark log by the time anyone reads
    # it, and this number has to survive the trip to a customer slide intact.
    meta = {
        "target": {
            "mean_isl": a.mean_isl,
            "context_window": a.context_window,
            "tail_frac": a.tail_frac,
            "p99_isl": p99_target,
            "osl": a.osl,
        },
        "lognormal": {"mu": mu, "sigma": sigma, "cv": cv,
                      "expected_se_pct_of_mean": se_pct},
        "seed": a.seed,
        "prefix_tokens_requested": prefix_tokens,
        "achieved": ach,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "note": ("Random draw, not stratified: the p99 must actually reach the context "
                 "window, which inverse-CDF placement at small n cannot do. The mean "
                 "therefore carries sampling error of about "
                 "%.1f%% at this n -- report 'achieved', not 'target'." % se_pct),
    }
    meta_path = a.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("wrote %s (%d requests)" % (a.out, len(achieved)))
    print("wrote %s (achieved distribution -- quote THIS upstream)" % meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
