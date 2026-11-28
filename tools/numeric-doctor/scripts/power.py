#!/usr/bin/env python3
"""How many runs a debug-numeric experiment needs, and what a clean result is worth.

    python3 scripts/power.py 21/248000                 # runs for 30 expected hits at that rate
    python3 scripts/power.py 8.5e-5                    # the same from a bare rate (1.0 = every run)
    python3 scripts/power.py 21/248000 --hits 10
    python3 scripts/power.py 157/1000000 --clean 0/190000   # is this CLEAN powered?

A fault at rate p leaves n runs clean with probability exp(-p*n), so a CLEAN means
something only with enough expected hits (p*n).  >= 3 is the floor (below: UNDERPOWERED);
30 makes a false CLEAN ~1e-13, which no decision needs beyond.  Runs past that buy
nothing and cost wall clock: 1M-run arms at 157 hits/1M (P(false CLEAN) 2e-88) were
two thirds of a qwen_tf32 step 7 (2026-11-14).  The rate is the reproducer's, measured
under the same load (copies, co-tenant) as the experiment.

Last line: RUNS=<n> (or POWER=<verdict> with --clean).
"""
import argparse
import math
import sys


def frac(s):
    """hits/runs -> (hits, runs); a bare rate (1.0, 8.5e-5) -> as hits per 1e9 runs."""
    if '/' not in s:
        try:
            r = float(s)
        except ValueError:
            sys.exit('power: %r is neither hits/runs nor a rate' % s)
        if not 0 < r <= 1:
            sys.exit('power: rate %r must be in (0, 1]' % s)
        return round(r * 1e9), 10 ** 9
    try:
        k, n = (int(x) for x in s.split('/'))
    except ValueError:
        sys.exit('power: %r is not hits/runs' % s)
    if n <= 0 or k < 0:
        sys.exit('power: %r: runs must be > 0, hits >= 0' % s)
    return k, n


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('rate', help='the reproducer\'s measured hits/runs, e.g. 21/248000')
    p.add_argument('--hits', type=float, default=30.0, help='expected hits to aim for (default 30)')
    p.add_argument('--floor', type=int, default=1000, help='never fewer runs (default 1000)')
    p.add_argument('--cap', type=int, default=1000000, help='never more runs (default 1000000)')
    p.add_argument('--clean', metavar='K/N', help='judge a result: K hits in N runs of the patched arm')
    a = p.parse_args()
    k, n = frac(a.rate)
    if k == 0:
        sys.exit('power: 0 hits in the rate -- nothing to size from; measure the reproducer first')
    rate = k / n
    if a.clean:
        ck, cn = frac(a.clean)
        exp = rate * cn
        if ck:
            print('%d hit(s) in %d runs (stock %.3g/run): NOT CLEAN' % (ck, cn, rate))
            print('POWER=NAN')
            return
        verdict = 'CLEAN' if exp >= 3 else 'UNDERPOWERED'
        print('0/%d at stock %.3g/run: %.1f expected hits, P(false CLEAN) = %.2g'
              % (cn, rate, exp, math.exp(-exp)))
        if verdict == 'UNDERPOWERED':
            print('below 3 expected hits: not a finding -- %d runs for %g expected'
                  % (math.ceil(a.hits / rate), a.hits))
        print('POWER=%s' % verdict)
        return
    runs = min(max(math.ceil(a.hits / rate), a.floor), a.cap)
    exp = rate * runs
    print('stock %d/%d = %.3g/run: %d runs -> %.1f expected hits, P(false CLEAN) = %.2g%s'
          % (k, n, rate, runs, exp, math.exp(-exp),
             ' (capped: below %g expected)' % a.hits if runs == a.cap and exp < a.hits else ''))
    print('RUNS=%d' % runs)


if __name__ == '__main__':
    main()
