#!/usr/bin/env python3
"""Per-step gradient trajectory and a NaN signal for a training loop (debug-numeric).

The op capture fires only on inf/NaN, so for EXPLODING values it catches the first
op that overflows -- often a correct reduction (`_foreach_norm` over finite grads of
~1e19) far downstream of the cause.  The trajectory shows the dynamics instead: which
module's gradients grow, from which step, how fast, while the weights stay or move.

In the model (one call per step, after backward, before clipping / optimizer):

    sys.path.insert(0, os.environ["NUMERIC_DOCTOR_ROOT"] + "/scripts")
    from grad_trajectory import after_backward
    after_backward(model, loss, parts)      # parts: dict of scalar loss tensors, optional

* `NUMERIC_DOCTOR_GRADLOG=<file.jsonl>`: append one JSON line per step -- loss and parts,
  total and per-module grad norm, per-module weight |max|, the 5 parameters with the
  largest grad |max|.  Unset: no logging, no cost.
* Always: raise `NonFiniteDetected("non-finite GRADIENT at step N after backward:
  <modules>")` when any gradient is non-finite (one device->host sync) -- a NaN signal
  for run scripts whose model hides NaNs (`nan_to_num` on the loss).  Set
  `NUMERIC_DOCTOR_NO_RAISE=1` to log only.

Summarise logs (no torch needed):

    python3 scripts/grad_trajectory.py <grad.jsonl> [...] [--every 20]

prints per file the gnorm median/max and the steps where gnorm jumps > 10x over the
running median, with the top modules and parameter.
"""
from __future__ import annotations

import json
import math
import os
import sys


class NonFiniteDetected(RuntimeError):
    pass


_step = 0


def after_backward(model, loss=None, parts=None) -> None:
    import torch

    global _step
    step, _step = _step, _step + 1
    names, gmax, gnorm, wmax, finite = [], [], [], [], []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        names.append(n)
        finite.append(torch.isfinite(g).all())
        if os.environ.get("NUMERIC_DOCTOR_GRADLOG"):
            gf = g.float()
            gmax.append(gf.abs().max())
            gnorm.append(gf.norm())
            wmax.append(p.detach().float().abs().max())
    if not names:
        return
    ok = torch.stack(finite).cpu().tolist()                      # one sync for the check
    path = os.environ.get("NUMERIC_DOCTOR_GRADLOG")
    if path:
        vals = torch.stack([torch.stack(gmax), torch.stack(gnorm), torch.stack(wmax)]).cpu().double()
        mods: dict[str, list[float]] = {}
        for n, gn, wm in zip(names, vals[1].tolist(), vals[2].tolist()):
            m = mods.setdefault(n.split(".")[0], [0.0, 0.0])
            m[0] += gn * gn
            m[1] = max(m[1], wm)
        lp = {k: float(v.detach().float().cpu()) for k, v in (parts or {}).items()
              if torch.is_tensor(v) and v.numel() == 1}
        rec = {"step": step,
               "loss": float(loss.detach().float().cpu()) if torch.is_tensor(loss) else loss,
               "parts": lp,
               "gnorm": math.sqrt(sum(m[0] for m in mods.values())),
               "mod_gnorm": {k: math.sqrt(v[0]) for k, v in mods.items()},
               "mod_wmax": {k: v[1] for k, v in mods.items()},
               "top_gmax": [[g, n] for g, n in sorted(zip(vals[0].tolist(), names), reverse=True)[:5]]}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if not all(ok) and not os.environ.get("NUMERIC_DOCTOR_NO_RAISE"):
        bad = sorted({n.split(".")[0] for n, good in zip(names, ok) if not good})
        raise NonFiniteDetected(f"non-finite GRADIENT at step {step} after backward: {', '.join(bad)}")


def _fmt(x) -> str:
    return f"{x:.3g}" if isinstance(x, (int, float)) else str(x)


def summarise(path: str, every: int = 20) -> None:
    recs = [json.loads(line) for line in open(path) if line.strip()]
    if not recs:
        print(f"== {path}: empty")
        return
    g = [r["gnorm"] for r in recs]
    fin = [x for x in g if math.isfinite(x)]
    p50 = sorted(fin)[len(fin) // 2] if fin else float("nan")
    gmax = max(fin) if fin else float("nan")
    print(f"== {path}  steps {len(recs)}  gnorm p50 {_fmt(p50)}  max {_fmt(gmax)}"
          + (f" @ {g.index(gmax)}" if fin else ""))
    seen: list[float] = []
    for r in recs:
        gn = r["gnorm"]
        base = sorted(seen)[len(seen) // 2] if seen else gn
        jump = not math.isfinite(gn) or gn > 10 * base
        if r["step"] % every == 0 or jump or r is recs[-1]:
            mods = sorted(r["mod_gnorm"].items(),
                          key=lambda kv: -kv[1] if math.isfinite(kv[1]) else -math.inf)[:3]
            top = r["top_gmax"][0] if r["top_gmax"] else [float("nan"), "-"]
            parts = {k: round(v, 4) for k, v in (r.get("parts") or {}).items()}
            print(f"  step {r['step']:5d} loss {_fmt(r['loss'])} gnorm {_fmt(gn)}"
                  f"{'  <-- JUMP' if jump else ''}  top mods {[(k, _fmt(v)) for k, v in mods]}"
                  f"  top param {top[1]} {_fmt(top[0])}  parts {parts}")
        if math.isfinite(gn):
            seen.append(gn)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Summarise NUMERIC_DOCTOR_GRADLOG files")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--every", type=int, default=20, help="also print every Nth step")
    a = ap.parse_args()
    for p in a.logs:
        summarise(p, a.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
