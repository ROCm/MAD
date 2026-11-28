#!/usr/bin/env python3
"""Compare what a model returned under two or more configurations (debug-numeric differential mode).

    python3 scripts/output_diff.py --a A/outputs.jsonl --a2 A2/outputs.jsonl --ref R/outputs.jsonl \
        [--arm NAME=ARM/outputs.jsonl ...] [--dtype bfloat16] [--k 5] [--json diff.json]

Each file is what scripts/diff_tap wrote for one run of the model (one JSON record per line):
    {"kind": "meta", "framework": ..., "dtype": ..., "config": ...}
    {"kind": "gen",  "i": n, "inp": <hash of the prompt ids>, "tokens": [...], "lp": [...], "margin": [...]}
    {"kind": "step", "i": n, "loss": x, "gnorm": y}
    {"kind": "fwd",  "i": n, "inp": <hash>, "argmax": [...], "l2": x, "absmax": y, "finite": b}
`gen` = one request (greedy or sampled), `lp` = the chosen token's logprob per position, `margin` = top-1 minus
top-2 logprob per position; `step` = one training step (loss from backward, grad norm from clip_grad_norm_);
`fwd` = one call of the outermost module (the fallback tap).

A = the user's configuration, A2 = the same configuration run again (the noise floor), ref = the conservative
reference configuration (configs.py), --arm = A with one switch flipped (the switch bisection).

Verdicts, one summary line `ND_DIFF verdict=...`:
  INPUTS_DIFFER     the runs did not see the same inputs (prompt hashes differ): no verdict, pin the inputs
  NONDETERMINISTIC  A and A2 disagree above the dtype floor: the configuration is not reproducible by itself
  DIVERGE           A and ref disagree above max(dtype floor, k x the A-vs-A2 gap); suspect = A
  AGREE             no request / step / forward beyond the threshold (TIE_FLIPs allowed: a near-tie token that
                    two correct implementations may resolve differently)
Per arm (bisection): `ND_DIFF_ARM name=<switch> verdict=AGREE|DIVERGE`: an arm that AGREEs with ref while A
DIVERGEs names the switch that removes the divergence (`ND_DIFF_SWITCH=<name>`).

No torch: runs on the login node; diff_tap imports `judge_gen` / `judge_step` / `judge_fwd` for its check mode.
"""
import argparse
import json
import math
import sys

# Defaults of the thresholds -- the gap two CORRECT implementations of the same model may show, per output
# dtype. Measured on the clean examples (tests/diff T2, 2026-11-24); configs.py may override per framework.
FLOORS = {
    "logprob": {"bfloat16": 0.25, "float16": 0.05, "float32": 0.005},   # abs gap of a chosen token's logprob
    "loss_rel": {"bfloat16": 0.02, "float16": 0.005, "float32": 1e-4},  # log-ratio gap of a step's loss
    # the grad norm swings by up to ~3.4x between two runs of ONE configuration after a few steps
    # (softmoe_train_nan, 2026-11-24): only the three-way rule uses it, and only a > 4x spike counts
    "gnorm_rel": {"bfloat16": 1.39, "float16": 1.39, "float32": 1.39},
    "fwd_rel": {"bfloat16": 0.03, "float16": 0.005, "float32": 1e-4},   # relative gap of the output's l2
}
JUMP = 5.0         # training: the reference is the odd run only where A's gap to it jumps this many x past the
                   # largest gap of the previous JUMP_WINDOW steps -- two correct implementations drift apart
                   # smoothly (~3x every few steps), a bug jumps (x70 at the torchcompile spike, 2026-11-24)
JUMP_WINDOW = 5
K = 5.0            # x the A-vs-A2 gap (the twin check's median rule: twins.py, op_nan_capture TWIN_MEDIAN_X)
STEP_RUN = 3       # a training gap counts once it stays above the threshold this many consecutive steps


def load(path):
    meta, recs = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "meta":
                meta.update(r)
            else:
                key = (r["kind"], r["i"])
                recs[key] = {**recs.get(key, {}), **r}   # a later record of the same key adds fields (gnorm)
    return meta, recs


def _finite(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def judge_gen(a, r, thr):
    """One request, a vs r. -> (verdict, detail dict). verdict: SAME | TIE_FLIP | DIVERGE | INPUTS_DIFFER.
    Logprobs are compared on the common prefix only (after a token differs, the contexts differ)."""
    if a.get("inp") != r.get("inp"):
        return "INPUTS_DIFFER", {}
    ta, tr = a.get("tokens") or [], r.get("tokens") or []
    la, lr = a.get("lp") or [], r.get("lp") or []
    n = min(len(ta), len(tr))
    first = next((p for p in range(n) if ta[p] != tr[p]), None)
    if first is None and len(ta) != len(tr):
        first = n
    upto = n if first is None else first
    gap, at = 0.0, None
    for p in range(min(upto, len(la), len(lr))):
        x, y = la[p], lr[p]
        if x is None or y is None:
            continue
        if not (_finite(x) and _finite(y)):
            if _finite(x) != _finite(y):
                return "DIVERGE", {"pos": p, "gap": float("inf"), "why": "non-finite logprob in one run"}
            continue
        if abs(x - y) > gap:
            gap, at = abs(x - y), p
    d = {"gap": gap, "gap_pos": at}
    if gap > thr:
        d.update(pos=at, why="logprob gap")
        return "DIVERGE", d
    if first is None:
        return "SAME", d
    # a differing token: benign only if both runs were near a tie there (each run's own top-1 vs top-2)
    ma, mr = a.get("margin") or [], r.get("margin") or []
    m1 = ma[first] if first < len(ma) else None
    m2 = mr[first] if first < len(mr) else None
    d.update(pos=first, margin_a=m1, margin_ref=m2)
    if m1 is None or m2 is None:
        # no logprobs (the ND_DIFF_FUNC tap): a near-tie cannot be told from an error. Against the same
        # configuration (A vs A2) any differing token is nondeterminism; against another one it is unjudged
        d["why"] = "token differs, no margins recorded"
        return "UNMARGINED", d
    # two implementations whose logprobs differ by up to thr can move a top-1 / top-2 margin by up to 2 x thr
    # (a real Qwen2.5-0.5B, compile vs eager: flips at margins 0.25-0.5 with logprob gaps <= 0.19, 2026-11-24)
    if m1 <= 2 * thr and m2 <= 2 * thr:
        return "TIE_FLIP", d
    d["why"] = "token differs away from a tie"
    return "DIVERGE", d


def relgap(x, y):
    """Gap of two values of one metric: |ln(x/y)| when both are positive (loss, grad norm: symmetric, and a
    spike from 6.7 to 755 stays large instead of saturating), else |x - y| / max(|x|, |y|)."""
    if x > 0 and y > 0:
        return abs(math.log(x / y))
    return abs(x - y) / max(abs(x), abs(y), 1e-12)


def judge_step(a, r, thr):
    """One training step, two runs. -> (verdict, detail). The loss decides (the grad norm is too chaotic to
    compare two-way: tests/diff T2, 2026-11-24); a value finite in one run only, loss or grad norm, is DIVERGE."""
    for k in ("loss", "gnorm"):
        x, y = a.get(k), r.get(k)
        if x is not None and y is not None and _finite(x) != _finite(y):
            return "DIVERGE", {"gap": float("inf"), "what": k, "why": "non-finite in one run"}
    x, y = a.get("loss"), r.get("loss")
    if x is None or y is None or not _finite(x):
        return "SAME", {"gap": 0.0}
    gap = relgap(x, y)
    return ("DIVERGE" if gap > thr else "SAME"), {"gap": gap, "what": "loss"}


def odd_one_out(ra, ra2, rr, floors, k=K):
    """Training with three runs (A, A2 = A's configuration again, ref): two runs of a training job drift apart by
    themselves (backward atomics, amplified step by step), so a step is judged by which run is the odd one out,
    per metric (loss, grad norm): a run whose gaps to BOTH others exceed max(floor, k x the gap of the closest
    pair). -> {step: (class, detail)}, class: SAME | REF_ODD (A's configuration agrees with itself, not with the
    reference, and A's gap to it jumped: a divergence starts) | REF_FAR (the reference still apart, no new jump:
    a divergence persisting -- or two correct implementations drifting) | SELF (A or A2 is the odd one: A's
    configuration misbehaved in one run only) |
    NONFINITE_REF_ODD / NONFINITE_SELF."""
    out = {}
    steps = sorted(i for kind, i in ra if kind == "step")
    hist = {"loss": [], "gnorm": []}            # A's gap to the reference, per step, per metric
    for t in steps:
        a, b, r = ra.get(("step", t)), ra2.get(("step", t)), rr.get(("step", t))
        if b is None or r is None:
            continue
        cls, det = "SAME", {}
        for m in ("loss", "gnorm"):
            v = {"A": a.get(m), "A2": b.get(m), "REF": r.get(m)}
            if any(x is None for x in v.values()):
                continue
            fin = {n: _finite(x) for n, x in v.items()}
            if not all(fin.values()):
                if fin["A"] == fin["A2"] != fin["REF"]:
                    return_cls = "NONFINITE_REF_ODD"
                elif fin["A"] != fin["A2"]:
                    return_cls = "NONFINITE_SELF"
                else:
                    continue
                cls, det = return_cls, {"what": m, "values": v}
                break
            g = {("A", "A2"): relgap(v["A"], v["A2"]), ("A", "REF"): relgap(v["A"], v["REF"]),
                 ("A2", "REF"): relgap(v["A2"], v["REF"])}
            thr = max(floors[m], k * min(g.values()))
            odd = [n for n in ("A", "A2", "REF") if all(gv > thr for pair, gv in g.items() if n in pair)]
            env = max(hist[m][-JUMP_WINDOW:], default=0.0)
            hist[m].append(g[("A", "REF")])
            if odd == ["REF"] and min(g[("A", "REF")], g[("A2", "REF")]) <= JUMP * env:
                if cls == "SAME":                        # still apart, but no jump: drift, or a jump persisting
                    cls, det = "REF_FAR", {"what": m, "odd": "REF", "gap": g[("A", "REF")], "threshold": thr,
                                           "values": v}
                continue
            if odd:
                c = "REF_ODD" if odd[0] == "REF" else "SELF"
                gap = max(gv for pair, gv in g.items() if odd[0] in pair)
                if c == "SELF" or cls in ("SAME", "REF_FAR"):
                    cls, det = c, {"what": m, "odd": odd[0], "gap": gap, "threshold": thr, "values": v}
        out[t] = (cls, det)
    return out


def judge_fwd(a, r, thr):
    """One outermost-module call (fallback tap): argmax agreement and the relative l2 gap."""
    if a.get("inp") != r.get("inp"):
        return "INPUTS_DIFFER", {}
    if a.get("finite", True) != r.get("finite", True):
        return "DIVERGE", {"gap": float("inf"), "why": "non-finite in one run"}
    x, y = a.get("l2"), r.get("l2")
    gap = abs(x - y) / max(abs(y), 1e-12) if _finite(x) and _finite(y) else 0.0
    aa, ar = a.get("argmax") or [], r.get("argmax") or []
    flips = sum(1 for p, q in zip(aa, ar) if p != q)
    d = {"gap": gap, "argmax_flips": flips, "of": min(len(aa), len(ar))}
    if gap > thr:
        return "DIVERGE", d
    return ("TIE_FLIP" if flips else "SAME"), d


JUDGE = {"gen": judge_gen, "step": judge_step, "fwd": judge_fwd}
FLOOR_KEY = {"gen": "logprob", "step": "loss_rel", "fwd": "fwd_rel"}


def gap_only(kind, a, r):
    """The raw gap of one record pair (the noise floor's input), no threshold."""
    v, d = JUDGE[kind](a, r, float("inf"))
    if v == "INPUTS_DIFFER":
        return None
    g = d.get("gap", 0.0)
    return g if _finite(g) else float("inf")


def compare(ra, rb, thr, kind_filter=None):
    """All common records of two runs. -> {"verdicts": {key: (verdict, detail)}, "missing": n, "kinds": set}"""
    out, missing = {}, 0
    for key, a in ra.items():
        if kind_filter and key[0] not in kind_filter:
            continue
        b = rb.get(key)
        if b is None:
            missing += 1
            continue
        out[key] = JUDGE[key[0]](a, b, thr[key[0]])
    missing += sum(1 for key in rb if key not in ra and (not kind_filter or key[0] in kind_filter))
    return {"verdicts": out, "missing": missing}


def step_diverged(verdicts, run=STEP_RUN):
    """Training: the first step of the first run of `run` consecutive DIVERGE steps (a single noisy step is not a
    divergence; a non-finite step is, at once)."""
    steps = sorted(k[1] for k in verdicts if k[0] == "step")
    streak, start = 0, None
    for s in steps:
        v, d = verdicts[("step", s)]
        if v == "DIVERGE" and not _finite(d.get("gap")):
            return s
        if v == "DIVERGE":
            streak += 1
            start = s if streak == 1 else start
            if streak >= run:
                return start
        else:
            streak = 0
    return None


def thresholds(dtype, noise, k, floors=None):
    """Per record kind: max(the dtype floor, k x the A-vs-A2 gap)."""
    floors = floors or FLOORS
    out = {}
    for kind, fk in FLOOR_KEY.items():
        fl = floors[fk].get(dtype, floors[fk]["bfloat16"])
        out[kind] = max(fl, k * noise.get(kind, 0.0))
    return out


def summarize(cmp, kinds, same_config=False):
    """Counts and the first diverging record of a comparison. same_config: A vs A2, where an UNMARGINED token
    difference counts (a deterministic stack repeats its tokens exactly)."""
    v = cmp["verdicts"]
    n = len(v)
    badv = ("DIVERGE", "UNMARGINED") if same_config else ("DIVERGE",)
    bad = sorted(key for key, (x, _) in v.items() if x in badv)
    if "step" in kinds:
        s = step_diverged(v)
        bad = [key for key in bad if key[0] != "step"] + ([("step", s)] if s is not None else [])
    ties = sum(1 for x, _ in v.values() if x == "TIE_FLIP")
    inp = sorted(key for key, (x, _) in v.items() if x == "INPUTS_DIFFER")
    worst = max((d.get("gap", 0.0) for _, d in v.values() if _finite(d.get("gap"))), default=0.0)
    unm = sum(1 for x, _ in v.values() if x == "UNMARGINED")
    return {"n": n, "bad": bad, "ties": ties, "inputs_differ": inp, "worst": worst, "unmargined": unm}


def load_modules(path):
    """modules.jsonl -> {(req, module, n-th call of that module in the request): record}, and the call order."""
    recs, order, seen = {}, [], {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            c = (r["req"], r["mod"])
            seen[c] = seen.get(c, -1) + 1
            key = (r["req"], r["mod"], seen[c])
            recs[key] = r
            order.append(key)
    return recs, order


def module_gap(a, r):
    """Relative gap of one module output: of its l2, and of its projections (the sum, two fixed random-sign
    vectors) relative to the l2 -- each projection is of the order of the l2, and a random-sign one moves under any
    rearrangement (a head swap) as much as under a scaled or shifted block."""
    if a["finite"] != r["finite"]:
        return float("inf")
    if not a["finite"]:
        return 0.0
    if a["n"] != r["n"]:
        return float("inf")
    l2 = max(abs(r["l2"]), 1e-30)
    g = abs(a["l2"] - r["l2"]) / l2
    for x, y in zip(a["proj"], r["proj"]):
        g = max(g, abs(x - y) / l2)
    return g


def compare_modules(path_a, path_r, thr, top=8, jump=JUMP):
    """The first module, in completion order (children before their parents), whose output gap exceeds thr AND
    jumps past jump x the largest gap of every module completed before it: the innermost module where the runs
    start to disagree. Two implementations drift apart a little layer by layer (another attention backend: gaps
    ~0.03 on a real Qwen2.5-0.5B, 2026-11-24); a wrong module jumps. -> (first, rows above thr, n compared)."""
    ra, order = load_modules(path_a)
    rr, _ = load_modules(path_r)
    rows, prev = [], 0.0
    for key in order:
        r = rr.get(key)
        if r is None:
            continue
        g = module_gap(ra[key], r)
        rows.append((key, g, prev))
        prev = max(prev, g) if _finite(g) else prev
    bad = [(k, g, p) for k, g, p in rows if g > thr]
    jumps = [(k, g, p) for k, g, p in bad if g > jump * p]
    return (jumps[0] if jumps else None), bad[:top], len(rows)


def main_modules(args):
    dtype = args.dtype or "bfloat16"
    thr = args.module_thr or FLOORS["fwd_rel"].get(dtype, FLOORS["fwd_rel"]["bfloat16"])
    first, bad, n = compare_modules(args.modules, args.modules_ref, thr)
    for (req, mod, c), g, p in bad:
        print("ND_DIFF_MODULE req=%s module=%s call=%d gap=%.4g worst_before=%.4g" % (req, mod, c, g, p))
    if first:
        (req, mod, c), g, p = first
        print("ND_DIFF_LOCATE verdict=FOUND module=%s req=%s call=%d gap=%.4g threshold=%.4g worst_before=%.4g "
              "compared=%d" % (mod, req, c, g, thr, p, n))
    else:
        print("ND_DIFF_LOCATE verdict=%s compared=%d threshold=%.4g%s" % (
            "NONE_ABOVE" if n and not bad else ("NO_JUMP" if n else "NO_OVERLAP"), n, thr,
            "  (gaps above the threshold but no jump: the runs drift apart gradually -- a precision difference, "
            "not one wrong module)" if bad else ""))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", help="the user's configuration (required unless --modules)")
    p.add_argument("--a2", help="the same configuration again (noise floor; without it the floor is the dtype's)")
    p.add_argument("--ref", help="the reference configuration (required unless --modules)")
    p.add_argument("--arm", action="append", default=[], metavar="NAME=PATH", help="A with one switch flipped")
    p.add_argument("--dtype", help="output dtype (default: the A run's meta, else bfloat16)")
    p.add_argument("--k", type=float, default=K)
    p.add_argument("--json", help="write the full result here (diverging requests, thresholds) for check mode")
    p.add_argument("--modules", help="localisation: A's modules.jsonl (with --modules-ref)")
    p.add_argument("--modules-ref", help="the reference's modules.jsonl")
    p.add_argument("--module-thr", type=float, help="module gap threshold (default: the dtype's fwd_rel floor)")
    args = p.parse_args(argv)
    if args.modules:
        return main_modules(args)

    if not (args.a and args.ref):
        p.error("--a and --ref are required")
    ma, ra = load(args.a)
    _, rr = load(args.ref)
    kinds = {k[0] for k in ra}
    dtype = args.dtype or ma.get("dtype") or "bfloat16"
    res = {"dtype": dtype, "kinds": sorted(kinds), "framework": ma.get("framework")}

    noise = {}
    nondet = None
    if args.a2:
        _, ra2 = load(args.a2)
        for key, a in ra.items():
            b = ra2.get(key)
            if b is not None:
                g = gap_only(key[0], a, b)
                if g is not None:
                    noise[key[0]] = max(noise.get(key[0], 0.0), g)
        # A vs A2 judged at the bare dtype floor: above it the configuration disagrees with itself
        self_thr = thresholds(dtype, {}, 0.0)
        nondet = summarize(compare(ra, ra2, self_thr), kinds, same_config=True)
        inf_noise = {k: v for k, v in noise.items() if not _finite(v)}
        for k in inf_noise:          # a non-finite self-gap is not a noise floor
            noise[k] = 0.0
    thr = thresholds(dtype, noise, args.k)
    res.update(noise=noise, thresholds=thr)

    main_cmp = compare(ra, rr, thr)
    s = summarize(main_cmp, kinds)
    res["a_vs_ref"] = {"n": s["n"], "diverge": [list(k) for k in s["bad"]], "tie_flips": s["ties"],
                       "worst_gap": s["worst"], "missing": main_cmp["missing"]}
    first = s["bad"][0] if s["bad"] else None
    if first:
        res["a_vs_ref"]["first"] = {"key": list(first), "detail": main_cmp["verdicts"][first][1]}

    if "step" in kinds and args.a2:
        # training: the three-way rule replaces the two-way step verdicts (A and A2 drift apart by themselves)
        fl = {m: FLOORS[m + "_rel"].get(dtype, FLOORS[m + "_rel"]["bfloat16"]) for m in ("loss", "gnorm")}
        ooo = odd_one_out(ra, ra2, rr, fl, args.k)
        # a divergence = a jump (REF_ODD) that persists (REF_FAR or another jump) for STEP_RUN steps in all; a
        # jump that does not persist is the reference misbehaving once, not A
        run_, start, ref_odd = 0, None, None
        for t in sorted(ooo):
            c = ooo[t][0]
            if c == "NONFINITE_REF_ODD":
                ref_odd = ref_odd if ref_odd is not None else t
            if c == "REF_ODD" and run_ == 0:
                run_, start = 1, t
            elif c in ("REF_ODD", "REF_FAR") and run_ > 0:
                run_ += 1
            else:
                run_ = 0
            if run_ >= STEP_RUN and ref_odd is None:
                ref_odd = start
        selfs = [t for t in sorted(ooo) if ooo[t][0] in ("SELF", "NONFINITE_SELF")]
        res_far = [t for t in sorted(ooo) if ooo[t][0] == "REF_FAR"]
        s["bad"] = [k_ for k_ in s["bad"] if k_[0] != "step"] + ([("step", ref_odd)] if ref_odd is not None else [])
        if nondet is not None:
            nondet["bad"] = [k_ for k_ in nondet["bad"] if k_[0] != "step"] + [("step", t) for t in selfs]
        first = s["bad"][0] if s["bad"] else None
        res["steps_three_way"] = {str(t): ooo[t] for t in sorted(ooo) if ooo[t][0] != "SAME"}
        res["steps_ref_far_without_jump"] = len(res_far)
        res["a_vs_ref"]["diverge"] = [list(k_) for k_ in s["bad"]]
        if first and first[0] == "step":
            res["a_vs_ref"]["first"] = {"key": list(first), "detail": ooo[first[1]][1]}
        for t in selfs[:5]:
            d_ = ooo[t][1]
            print("ND_DIFF_SELF step=%d odd=%s what=%s values=%s gap=%.4g threshold=%.4g"
                  % (t, d_.get("odd", "-"), d_.get("what"), d_.get("values"), d_.get("gap", float("inf")),
                     d_.get("threshold", 0.0)))

    if s["inputs_differ"]:
        verdict = "INPUTS_DIFFER"
    elif nondet and nondet["bad"]:
        verdict = "NONDETERMINISTIC"
        res["a_vs_a2"] = {"n": nondet["n"], "diverge": [list(k) for k in nondet["bad"]]}
    elif s["bad"]:
        verdict = "DIVERGE"
    elif s["n"] == 0:
        verdict = "NO_OUTPUTS"
    else:
        verdict = "AGREE"
    res["verdict"] = verdict

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        _, rx = load(path)
        sx = summarize(compare(rx, rr, thr), kinds)
        arms[name] = "INPUTS_DIFFER" if sx["inputs_differ"] else ("DIVERGE" if sx["bad"] else
                                                                 ("NO_OUTPUTS" if sx["n"] == 0 else "AGREE"))
        print("ND_DIFF_ARM name=%s verdict=%s records=%d diverge=%d tie_flips=%d worst_gap=%.4g"
              % (name, arms[name], sx["n"], len(sx["bad"]), sx["ties"], sx["worst"]))
    res["arms"] = arms
    fixers = [n for n, v in arms.items() if v == "AGREE"]
    if verdict == "DIVERGE" and arms:
        res["switch"] = fixers[0] if len(fixers) == 1 else (",".join(fixers) if fixers else "NONE_SINGLE")
        print("ND_DIFF_SWITCH=%s" % res["switch"]
              + ("  (several single switches each remove it: they share the component)" if len(fixers) > 1 else "")
              + ("  (no single switch removes it: two switches together, or the reference itself)"
                 if not fixers else ""))

    reqs = [k for k in s["bad"]]
    where = ""
    if first:
        d = main_cmp["verdicts"][first][1] if first in main_cmp["verdicts"] else {}
        where = " first=%s:%s%s" % (first[0], first[1], (":%s" % d["pos"]) if d.get("pos") is not None else "")
    extra = ""
    if verdict == "NONDETERMINISTIC":
        b = nondet["bad"][0]
        extra = " a_vs_a2=%d/%d first_self=%s:%s" % (len(nondet["bad"]), nondet["n"], b[0], b[1])
    if s["unmargined"]:
        extra += " unmargined=%d" % s["unmargined"]
        res["a_vs_ref"]["unmargined"] = s["unmargined"]
    print("ND_DIFF verdict=%s suspect=%s records=%d diverge=%d tie_flips=%d%s worst_gap=%.4g threshold=%s "
          "noise=%s dtype=%s missing=%d%s"
          % (verdict, "A" if verdict in ("DIVERGE", "NONDETERMINISTIC") else "-", s["n"], len(reqs), s["ties"],
             where, s["worst"], ",".join("%s:%.4g" % kv for kv in sorted(thr.items()) if kv[0] in kinds),
             ",".join("%s:%.4g" % kv for kv in sorted(noise.items())) or "-", dtype, main_cmp["missing"], extra))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
