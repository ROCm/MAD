#!/usr/bin/env python3
"""cfg.py — tiny yaml field reader for the shell scripts.
Usage:
  cfg.py <file> cluster <key>                -> cluster.yaml top-level key
  cfg.py <file> model  <model> <key>         -> models.<model>.<key>
  cfg.py <file> engine <engine> <key>        -> engines.<engine>.<key>
Lists print space-separated; scalars print as-is.
"""
import sys, yaml

def emit(v):
    if isinstance(v, list): print(" ".join(str(x) for x in v))
    elif isinstance(v, dict): print(" ".join(f"{k}={v}" for k, v in v.items()))
    else: print("" if v is None else v)

def main():
    f = sys.argv[1]; kind = sys.argv[2]
    d = yaml.safe_load(open(f))
    if kind == "cluster":
        emit(d.get(sys.argv[3]))
    elif kind == "model":
        emit(d["models"][sys.argv[3]].get(sys.argv[4]))
    elif kind == "engine":
        emit(d["engines"][sys.argv[3]].get(sys.argv[4]))
    elif kind == "default":
        # default <model> <key>: per-model override wins over global defaults{}
        model = sys.argv[3]; key = sys.argv[4]
        mv = d.get("models", {}).get(model, {}).get(key)
        emit(mv if mv is not None else d.get("defaults", {}).get(key))
    else:
        sys.exit(f"unknown kind {kind}")

if __name__ == "__main__":
    main()
