#!/usr/bin/env python3
"""Best-known configurations: measured results keyed by what makes them comparable.

One JSON object per line in a registry file (where the shared registry lives -- Client Perf Hub,
MAD, or both -- is still to be decided; this is the record and the lookup):

  key    model, platform, launcher, shape, pin_set, serving_type, slo (e.g. {"tpot_ms": 50})
  value  throughput (tokens/s/GPU), ttft_ms, tpot_ms, accuracy (task -> score), and what reproduces
         the run: config (config.resolved.json path or contents), topology (spec), recipe, date

best() returns the highest-throughput record for a key that meets its SLO and accuracy floor.

    registry.py record --registry FILE --json '{...}'
    registry.py best    --registry FILE --model M --platform P [--launcher L] [--serving-type T] [--pin-set S]
    registry.py promote --registry FILE --to scripts/common/perf/best_known.jsonl --model M ...
                        the best run becomes the default MAD runs (both STANDALONE and madengine)
"""
import argparse
import json
import sys

KEY = ("model", "platform", "launcher", "pin_set", "serving_type")
REQUIRED = ("model", "platform", "launcher", "throughput", "config")


def record(path, entry):
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"a registry record needs {', '.join(missing)}")
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def entries(path):
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("#")]
    except FileNotFoundError:
        return []


def meets(e, accuracy_floor=None):
    slo = e.get("slo") or {}
    for k in ("tpot_ms", "ttft_ms"):
        if k in slo and (e.get(k) is None or e[k] > slo[k]):
            return False
    for task, floor in (accuracy_floor or {}).items():
        if (e.get("accuracy") or {}).get(task, -1) < floor:
            return False
    return True


def best(path, accuracy_floor=None, **key):
    rows = [e for e in entries(path) if all(e.get(k) == v for k, v in key.items() if v is not None)]
    rows = [e for e in rows if meets(e, accuracy_floor)]
    return max(rows, key=lambda e: e["throughput"]) if rows else None


def promote(runs, best_known, accuracy_floor=None, **key):
    """Copy the best run meeting its SLO (and an accuracy floor) for `key` into the best-known file
    MAD runs by default (perf/best_known.jsonl), replacing an older record of the same key, model,
    launcher, platform, shape and pin set. Returns the promoted record or None."""
    e = best(runs, accuracy_floor, **key)
    if e is None:
        return None
    if "apply" not in e:
        raise ValueError("the run has no `apply` (the recipe keys to lay over the recipe)")
    same = lambda r: all(r.get(k) == e.get(k) for k in ("model", "launcher", "platform", "shape", "pin_set"))  # noqa: E731
    with open(best_known) as f:
        lines = f.read().splitlines()
    kept = [l for l in lines if not l.strip() or l.lstrip().startswith("#") or not same(json.loads(l))]
    with open(best_known, "w") as f:
        f.write("\n".join(kept + [json.dumps(e, sort_keys=True)]) + "\n")
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--json", required=True)
    b = sub.add_parser("best")
    pr = sub.add_parser("promote", help="the best run into the best-known file MAD runs by default")
    pr.add_argument("--to", required=True, help="scripts/common/perf/best_known.jsonl")
    for p in (b, pr):
        for k in KEY:
            p.add_argument("--" + k.replace("_", "-"))
    for p in (r, b, pr):
        p.add_argument("--registry", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "record":
            record(a.registry, json.loads(a.json))
            return 0
        if a.cmd == "promote":
            e = promote(a.registry, a.to, **{k: getattr(a, k) for k in KEY})
            print(json.dumps(e, indent=1) if e else "no record meets the key and its SLO")
            return 0 if e else 1
        e = best(a.registry, **{k: getattr(a, k) for k in KEY})
        print(json.dumps(e, indent=1) if e else "no record meets the key and its SLO")
        return 0 if e else 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
