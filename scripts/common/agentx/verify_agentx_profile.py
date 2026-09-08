#!/usr/bin/env python3
"""Verify a WekaTrace corpus against ITS workload profile's targets + bands.

Generalization of ROCm/MAD #173's `verify_caseA.py` / `verify_caseB.py`: instead
of hard-coding the Case-A/Case-B numbers, the targets come from the profile so
any workload (conformance_256k, conformance_512k, or a user-defined custom case) verifies against its
own distribution. Prints a per-axis conformance table and "N/N axes within band".

Targets read from the profile:
  ISL   P50/P90/P99  <- profile.isl_p
  OSL   P50/P90/P99  <- profile.osl_p
  Delay P50/P90/P99  <- profile.delay_p
  Turns P50/P90/P99  <- profile.verify.turns_p   (else derived from turns dist)
  Cache hit P50 %    <- profile.verify.cache_target (else mean(cache_hit)*100)

Tolerance bands (same multipliers as #173), overridable per profile:
  ISL 0.80-1.20   OSL 0.70-1.40   Turns 0.60-1.60   Delay 0.50-2.00   Cache 0.97-1.03
Per-axis overrides via profile.verify.band_overrides, e.g. Case-B widens
"Input ISL P99" to 0.75-1.25.

Pure Python stdlib only. YAML profiles are resolved to JSON by the config loader
before being handed here.

Usage:
  verify_agentx_profile.py --profile <profile.json> --corpus <dir>
  verify_agentx_profile.py --profile-json '{...}'   --corpus <dir>
"""
import json, glob, os, sys

DEFAULT_BANDS = {
    "isl":   (0.80, 1.20),
    "osl":   (0.70, 1.40),
    "turns": (0.60, 1.60),
    "delay": (0.50, 2.00),
    "cache": (0.97, 1.03),
}


def _argval(it, flag):
    try:
        return next(it)
    except StopIteration:
        sys.stderr.write(f"[verify_agentx_profile] {flag} requires a value\n")
        raise SystemExit(2)


def p(a, q):
    a = sorted(a)
    return a[min(len(a) - 1, int(q * len(a)))] if a else 0


def _weighted_percentiles(values, weights, qs):
    """Percentiles of a discrete weighted distribution (cumulative-mass method),
    matching how #173 derived turns P50/P90/P99 from the turns values+weights."""
    total = float(sum(weights))
    out = []
    for q in qs:
        thresh = q * total
        cum = 0.0
        chosen = values[-1]
        for v, w in zip(values, weights):
            cum += w
            if cum >= thresh:
                chosen = v
                break
        out.append(chosen)
    return out


def measure(corpus):
    ai = []; oa = []; tu = []; dl = []; hit = []
    for f in glob.glob(os.path.join(corpus, "*.json")):
        with open(f) as fh:
            b = json.load(fh)
        r = b.get("requests")
        if r is None:
            raise SystemExit(f"[verify] {f}: session JSON missing 'requests'")
        seen = set(); tu.append(len(r))
        for x in r:
            if x.get("in"): ai.append(x["in"])
            if x.get("out"): oa.append(x["out"])
            if x.get("think_time") and x["t"] > 0: dl.append(x["think_time"])
            h = x.get("hash_ids") or []
            if h:
                nw = sum(1 for z in h if z not in seen); tt = len(h); [seen.add(z) for z in h]
                if tt and x["t"] > 0: hit.append(100 * (tt - nw) / tt)
    return ai, oa, tu, dl, hit


def verify(profile, corpus):
    verify_cfg = profile.get("verify", {}) or {}
    bands = dict(DEFAULT_BANDS)
    for k, v in (verify_cfg.get("bands", {}) or {}).items():
        bands[k] = tuple(v)
    overrides = verify_cfg.get("band_overrides", {}) or {}

    missing = [k for k in ("isl_p", "osl_p", "delay_p") if k not in profile]
    if missing:
        raise SystemExit(
            "[verify_agentx_profile] profile missing required field(s): "
            + ", ".join(missing))
    isl_p = profile["isl_p"]
    osl_p = profile["osl_p"]
    delay_p = profile["delay_p"]
    turns_p = verify_cfg.get("turns_p")
    if turns_p is None:
        t = profile["turns"]
        turns_p = _weighted_percentiles(list(t["values"]), list(t["weights"]), (0.5, 0.9, 0.99))
    cache_target = verify_cfg.get("cache_target")
    if cache_target is None:
        lo, hi = profile["cache_hit"]
        cache_target = round((lo + hi) / 2.0 * 100)

    ai, oa, tu, dl, hit = measure(corpus)

    rows = [
        ("Input ISL P50",   p(ai, .5),  isl_p[0],   "isl"),
        ("Input ISL P90",   p(ai, .9),  isl_p[1],   "isl"),
        ("Input ISL P99",   p(ai, .99), isl_p[2],   "isl"),
        ("Output OSL P50",  p(oa, .5),  osl_p[0],   "osl"),
        ("Output OSL P90",  p(oa, .9),  osl_p[1],   "osl"),
        ("Output OSL P99",  p(oa, .99), osl_p[2],   "osl"),
        ("Turns P50",       p(tu, .5),  turns_p[0], "turns"),
        ("Turns P90",       p(tu, .9),  turns_p[1], "turns"),
        ("Turns P99",       p(tu, .99), turns_p[2], "turns"),
        ("Delay P50 (s)",   p(dl, .5),  delay_p[0], "delay"),
        ("Delay P90 (s)",   p(dl, .9),  delay_p[1], "delay"),
        ("Delay P99 (s)",   p(dl, .99), delay_p[2], "delay"),
        ("Cache hit P50 %", p(hit, .5), cache_target, "cache"),
    ]

    name = profile.get("name", "?")
    print(f"corpus={corpus}  profile={name}  sessions={len(tu)}  requests={len(ai)}\n")
    print(f"{'axis':<20}{'measured':>12}{'target':>10}{'  verdict'}")
    print("-" * 54)
    npass = 0
    for axis, meas, tgt, group in rows:
        lo, hi = overrides.get(axis, bands[group])
        verdict = "PASS" if (tgt * lo) <= meas <= (tgt * hi) else "off"
        npass += verdict == "PASS"
        print(f"{axis:<20}{meas:>12,.0f}{tgt:>10,.0f}   {verdict}")
    print("-" * 54)
    print(f"{npass}/{len(rows)} axes within band")
    return npass, len(rows)


def _load_profile(path):
    with open(path) as f:
        return json.load(f)


def main(argv):
    profile = None
    corpus = None
    it = iter(argv)
    for a in it:
        if a in ("--profile", "-p"):
            profile = _load_profile(_argval(it, a))
        elif a == "--profile-json":
            profile = json.loads(_argval(it, a))
        elif a in ("--corpus", "-c"):
            corpus = _argval(it, a)
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            sys.stderr.write(f"[verify_agentx_profile] unknown arg: {a}\n")
            return 2
    if profile is None or corpus is None:
        sys.stderr.write("usage: verify_agentx_profile.py --profile P.json --corpus DIR\n")
        return 2
    npass, total = verify(profile, corpus)
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
