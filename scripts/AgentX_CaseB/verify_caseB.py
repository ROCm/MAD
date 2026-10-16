#!/usr/bin/env python3
"""Verify a WekaTrace corpus against the Case-A parameters. Prints a conformance table
(measured vs target for all 6 axes) and a PASS/FAIL per axis. Used to prove the
conformance trace matches before benchmarking."""
import json, glob, sys

CORP = sys.argv[1] if len(sys.argv) > 1 else "corpus_caseA_conformance"
def p(a, q): a=sorted(a); return a[min(len(a)-1, int(q*len(a)))] if a else 0

ai=[]; oa=[]; tu=[]; dl=[]; hit=[]
for f in glob.glob(f"{CORP}/*.json"):
    b=json.load(open(f)); r=b["requests"]; seen=set(); tu.append(len(r))
    for x in r:
        if x.get("in"): ai.append(x["in"])
        if x.get("out"): oa.append(x["out"])
        if x.get("think_time") and x["t"]>0: dl.append(x["think_time"])
        h=x.get("hash_ids") or []
        if h:
            nw=sum(1 for z in h if z not in seen); tt=len(h); [seen.add(z) for z in h]
            if tt: hit.append(100*(tt-nw)/tt)

def band(v, lo, hi): return "PASS" if lo<=v<=hi else "off"
rows=[
 ("Input ISL P50",  p(ai,.5), 62000,  0.80,1.20),
 ("Input ISL P90",  p(ai,.9), 220000, 0.80,1.20),
 ("Input ISL P99",  p(ai,.99),500000, 0.75,1.25),
 ("Output OSL P50", p(oa,.5), 180,    0.70,1.40),
 ("Output OSL P90", p(oa,.9), 1400,   0.70,1.40),
 ("Output OSL P99", p(oa,.99),7000,   0.70,1.40),
 ("Turns P50",      p(tu,.5), 5,      0.60,1.60),
 ("Turns P90",      p(tu,.9), 82,     0.60,1.60),
 ("Turns P99",      p(tu,.99),144,    0.60,1.60),
 ("Delay P50 (s)",  p(dl,.5), 3.6,    0.50,2.00),
 ("Delay P90 (s)",  p(dl,.9), 23,     0.50,2.00),
 ("Delay P99 (s)",  p(dl,.99),240,    0.50,2.00),
 ("Cache hit P50 %",p(hit,.5),89,     0.97,1.03),
]
print(f"corpus={CORP}  sessions={len(tu)}  requests={len(ai)}\n")
print(f"{'axis':<20}{'measured':>12}{'target':>10}{'  verdict'}")
print("-"*54)
npass=0
for name,meas,tgt,lo,hi in rows:
    v=band(meas, tgt*lo, tgt*hi); npass+= v=="PASS"
    print(f"{name:<20}{meas:>12,.0f}{tgt:>10,.0f}   {v}")
print("-"*54)
print(f"{npass}/{len(rows)} axes within band")
