#!/usr/bin/env python3
"""Mechanical readings of a debug-numeric 5b replay, so nobody greps copy logs for them.

    python3 scripts/replay_verdict.py attempt <attempt_dir> [--reproduced]
    python3 scripts/replay_verdict.py kernel <kernel-id log>

`attempt` prints two lines for capsule_replay_summary.sh to put in summary.txt:

    === first op to go non-finite (all copies): <op> (call #N) <hits>[; ...] ===
    === proposed verdict: <VERDICT> (<why>) -- confirm against step5-read.md ===

The verdict is PROPOSED from the [C1] walk (step5-read.md table); the agent decides.
Without --reproduced (every copy clean) it is NOT_REPRODUCED unless the walk alone
says FP32_OVERFLOW? (a terminal fp64 reference that ran and was NON-FINITE: overflow,
or undefined arithmetic such as 0*inf when it is non-finite in fp64 too).

`kernel` condenses the log of `replay_nan_op.py --list-mutations` (+ `--bench
--solution-index`) into kernel.txt: solution index, kernel name, arch/library,
injected-or-stock (nans_to_nops), the enumerable mutators, the pinned bench line, and the static
wait-count scan's sites (waitcnt_scan.py) -- step 7's first set test.

Standard library only: it runs on the host, and ships inside every reproducer.
"""
import collections
import glob
import os
import re
import sys

ROW = re.compile(r'^\s+(\d+)\s+(\S+)\s+(yes|no|NO)\s+(\S+)\s+(\S+)\s*(.*)$')
TERMINAL = re.compile(r'^\s+terminal fp64 ref on replay operands: (\S+)')
TALLY = re.compile(r'^\s{4}(\S+ \(call #\d+\))\s+(\d+)\s+\(')


def c1_rows(lines):
    rows, on = [], False
    for ln in lines:
        if ln.startswith('[C1] chain walk'):
            on = True
        elif on and ln.startswith('[verdict]'):
            break
        elif on:
            m = ROW.match(ln)
            if m:
                rows.append(dict(idx=m.group(1), op=m.group(2), finite=m.group(3).lower() == 'yes',
                                 vs=m.group(4), ref=m.group(5), note=m.group(6).strip()))
    return rows


def c1_terminal(lines):
    """The walk's 'terminal fp64 ref on replay operands' status, or None."""
    for ln in lines:
        m = TERMINAL.match(ln)
        if m:
            return m.group(1)
    return None


def op_terminal(lines):
    """A single-op replay's own fp64 check ([2] of replay_nan_op.py: no chain walk,
    e.g. CAPSULE_MODE=op): 'finite', 'NON-FINITE', 'rerun' (a rerun capsule: counted by
    its rerun check, which needs no reference), or None if it printed none of them."""
    for ln in lines:
        if 'a launch counts when the op fails its rerun check again' in ln:
            return 'rerun'
        if '-> the true result is finite AND fits in' in ln:
            return 'finite'
        if '-> the true result (' in ln and 'exceeds' in ln:
            return 'NON-FINITE'
    return None


def split_terminal(rows):
    """(terminal, ancestors) of a [C1] walk.  Trailing view rows (note 'view') only
    alias the op that computed them, so the terminal is the last row that computes.
    2026-11-09 (qwen3 MoE): a transpose after the culprit _grouped_mm made the GEMM
    its own 'ancestor DIVERGED', and a KERNEL_BUG was proposed as UNVERIFIED."""
    k = len(rows)
    while k > 1 and rows[k - 1].get('note', '').split(',')[0].strip() == 'view':
        k -= 1
    return rows[k - 1], rows[:k - 1]


def propose(rows, terminal=None):
    """(verdict, why) for a REPRODUCED attempt, from its [C1] rows and the walk's
    terminal-on-replay-operands reference (c1_terminal), when it printed one.  With
    no walk (single-op replay), terminal is the op's own check (op_terminal)."""
    if not rows:
        # The verdict table's own rule: the terminal fp64 ref on the replay operands
        # is finite -> KERNEL_BUG, whatever produced those operands.
        if terminal == 'rerun':
            return ('KERNEL_BUG', 'single op: not repeatable on identical operands (rerun check): '
                    'no correct deterministic kernel does that, whatever the operands')
        if terminal == 'finite':
            return ('KERNEL_BUG', 'single op: its fp64 ref on the same operands is finite and '
                    'fits the output dtype (no chain walk: upstream not re-derived)')
        if terminal == 'NON-FINITE':
            return ('FP32_OVERFLOW?', 'single op: the true result is non-finite in fp64 (undefined '
                    'arithmetic: 0*inf, 0/0) or exceeds the output dtype (overflow) -- the '
                    'reference reason says which; not a kernel bug; do not ship it as one')
        return 'KERNEL_BUG_UNVERIFIED', 'no [C1] walk (single-op capsule): operands never validated'
    term, anc = split_terminal(rows)
    anc_div = [r for r in anc if r['vs'] == 'DIVERGED']
    if terminal == 'rerun':
        # a rerun capsule: the reproducing runs gave different outputs on identical operands --
        # no correct deterministic kernel does that, whatever the operands are
        why = ('the op is not repeatable on identical operands (rerun check): no correct '
               'deterministic kernel does that, whatever produced them')
        if anc_div:
            why += '; ancestor DIVERGED (%s), which does not change it' % ', '.join(r['op'] for r in anc_div)
        return 'KERNEL_BUG', why
    # most specific first: a non-finite fp64 reference means no correct kernel does
    # better -- not a kernel bug at all, whatever else the walk shows.  Only a
    # reference that RAN and came back non-finite says that: 'unavailable' (no CPU
    # kernel, an unmodelled variant, an upstream op without one) and '-' / 'skipped'
    # say nothing about the true result, and fall through to UNVERIFIED below.
    if term['ref'] == 'NON-FINITE':
        return ('FP32_OVERFLOW?', 'terminal fp64 ref is NON-FINITE: the true result is not '
                'finite in fp32 -- overflow, or undefined arithmetic (0*inf, 0/0) if it is '
                'non-finite in fp64 too; not a kernel bug; do not ship it as one')
    # Operand-independent: on the very operands the replay handed it, the true
    # result is finite and representable, so a non-finite output is never correct
    # -- upstream divergence or a deterministic walk cannot excuse it.
    if not term['finite'] and terminal == 'finite':
        why = 'terminal non-finite where its fp64 ref on the same operands is finite'
        if anc_div:
            why += ('; ancestor DIVERGED (%s): the operands may be wrong too'
                    % ', '.join(r['op'] for r in anc_div))
        return 'KERNEL_BUG', why
    if terminal == 'NON-FINITE' and not term['finite']:
        return ('FP32_OVERFLOW?', 'fp64 ref on the replay operands is non-finite (undefined '
                'arithmetic) or exceeds the output dtype (overflow): the op is right to '
                'return non-finite -- look upstream')
    if not term['finite'] and not any(r['vs'] == 'DIVERGED' for r in rows):
        return ('KERNEL_BUG_UNVERIFIED',
                'DETERMINISTIC walk (terminal non-finite, nothing DIVERGED): apply '
                'step5-replay-guardrail.md before trusting the named op')
    if term['ref'] == 'finite' and not anc_div:
        return 'KERNEL_BUG', 'terminal fp64 ref finite, no ancestor DIVERGED'
    # The reference on the replay's own operands (a fused op's value check, an fp32
    # recompute from its inputs: aiter MLA decode, which has no fp64 kernel) is the
    # stronger of the two: the loop reproduced and nothing upstream diverged.
    if terminal == 'finite' and not anc_div:
        return 'KERNEL_BUG', 'terminal reference on the replay operands finite, no ancestor DIVERGED'
    why = []
    if term['ref'] != 'finite':
        why.append('terminal op has no reference (%s)' % term['ref'])
    if anc_div:
        why.append('ancestor DIVERGED: %s' % ', '.join(r['op'] for r in anc_div))
    return 'KERNEL_BUG_UNVERIFIED', '; '.join(why)


def first_nonfinite(logs):
    """Sum each copy's 'first op to go non-finite, by run' table over the attempt."""
    tally = collections.Counter()
    for lg in logs:
        on = False
        for ln in open(lg, errors='replace'):
            if ln.startswith('  first op to go non-finite, by run:'):
                on = True
            elif on:
                m = TALLY.match(ln)
                if not m:
                    on = False
                    continue
                tally[m.group(1)] += int(m.group(2))
    return tally


def attempt(d, reproduced):
    logs = sorted(glob.glob(os.path.join(d, 'g*_k*.log')))
    tally = first_nonfinite(logs)
    if tally:
        print('=== first op to go non-finite (all copies): %s ==='
              % '; '.join('%s %d' % kv for kv in tally.most_common()))
    rows, terminal = [], None
    for lg in logs:
        lines = open(lg, errors='replace').read().splitlines()
        rows = c1_rows(lines)
        if rows:
            terminal = c1_terminal(lines)
            break
        terminal = terminal or op_terminal(lines)
    verdict, why = propose(rows, terminal)
    if not reproduced and not verdict.startswith('FP32'):
        verdict, why = 'NOT_REPRODUCED', 'every copy clean at this concurrency'
    elif reproduced and tally and rows and len(tally) == 1:
        # the loop localises: say whether the hits land on the terminal op
        only = next(iter(tally))
        term = '%s (call #%s)' % (rows[-1]['op'], rows[-1]['idx'])
        why += '; loop: every hit first non-finite at %s' % (
            'the terminal op' if only == term else only)
        if verdict == 'KERNEL_BUG_UNVERIFIED':
            # Where the NaN appears, not necessarily where it starts: a racy upstream
            # op writing finite but wrong values gets blamed on the op that overflows.
            why += ('; first non-finite != first wrong: --chain-repeat-check names the op '
                    'whose output stopped repeating')
    text0 = open(logs[0], errors='replace').read() if logs else ''
    if 'returned a WRONG FINITE value' in text0:
        # a wrong-value capsule: its "non-finite" hits are runs that failed the value check
        why = why.replace('terminal non-finite', 'terminal wrong finite value').replace(
            'first non-finite at', 'first wrong at')
    if 'SYNTHETIC-INCONSISTENT' in text0:
        # Derived operands (attention out / softmax_lse) drawn independently of q/k/v:
        # a correct kernel overflows on them too, so a hit proves nothing.
        if verdict in ('KERNEL_BUG', 'FP32_OVERFLOW?'):
            verdict = 'KERNEL_BUG_UNVERIFIED'
        why += ('; SYNTHETIC-INCONSISTENT operands (out/lse drawn independently of q/k/v): '
                'may be the synthesis, cannot validate a fix -- re-check on consistent '
                'operands (attn_consistent_check.py)')
    elif 'operands:       SYNTHETIC' in text0:
        derived = 'operands:       SYNTHETIC (derived' in text0
        why += ('; SYNTHETIC operands (--synthesize%s): holds for these made-up values only, '
                'no DIVERGED check%s' % (', out/lse recomputed by forward' if derived else '',
                                         '' if reproduced else
                                         ', and says nothing about a value-dependent NaN'))
    print('=== proposed verdict: %s (%s) -- confirm against step5-read.md ===' % (verdict, why))


def kernel(log):
    text = open(log, errors='replace').read()
    lines = text.splitlines()

    def first(pat):
        m = re.search(pat, text, re.M)
        return m.group(1).strip() if m else ''
    idx = first(r'^\s+solution_index: (\S+)')
    # a grouped GEMM: whose solution it is ("the wrong group's: rows N", or "ONE group's")
    note = first(r'^\s+solution_index: \S+ \(as dispatched; (.+)\)$')
    groups = re.findall(r'^\s+group rows (\d+ \(group\(s\) [\d,]+\): solution \S+.*)$', text, re.M)
    name = first(r'^\s+kernel:\s+(.+)$')
    arch = first(r'^\s+arch:\s+(.+)$')
    if not arch:                                     # Tensile names carry it: ..._ISA950_...
        m = re.search(r'_ISA(\d{3,4})(?:_|$)', name)
        arch = 'gfx%s' % m.group(1) if m else ''
    bench = first(r'^(hipblaslt-bench .+)$')
    muts = []
    on = False
    for ln in lines:
        if ln.startswith('mutation ') and 'selector' in ln:
            on = True
        elif on and ln.startswith('---'):
            continue
        elif on and ln.strip():
            if ln.split()[0] not in muts:
                muts.append(ln.split()[0])
        elif on:
            on = False
    if 'nans_to_nops' in muts:
        inj = 'yes -- nans_to_nops matches a site (on this image its CLEAN confirms the tooling, not a bug)'
    elif re.search(r"mutator 'nans_to_nops' proposed nothing", text):
        inj = 'no (stock) -- nans_to_nops: no instruction materialises a NaN constant'
    else:
        inj = 'unknown'
    print('solution_index: %s' % (idx or '?') + (' (%s)' % note if note else ''))
    for gl in groups:
        print('group:          rows %s' % gl.strip())
    print('kernel:         %s' % (name or '? (not identified: see %s)' % os.path.basename(log)))
    print('arch:           %s' % (arch or '?'))
    print('injected:       %s' % inj)
    print('mutators:       %s' % (', '.join(m for m in muts if m != 'identity') or 'none enumerable'))
    print('bench:          %s' % (bench or '(no 2-D GEMM geometry, or no solution index)'))
    scan = first(r'^(WAITCNT_SCAN=.+)$')
    if scan:                                         # waitcnt_scan.py, printed by --list-mutations
        print('waitcnt scan:   %s  sites %s%s' % (
            scan, first(r'^NAN_MUTATE_SUBSET_ADDRS="(.*)"$') or '-',
            ('  H3 ' + first(r'^NAN_MUTATE_SUBSET_ADDRS_H3="(.*)"$')) if first(r'^NAN_MUTATE_SUBSET_ADDRS_H3="(.+)"$') else ''))
    # a grouped GEMM prints one plain-mm line per distinct group size (grouped_bench_cmds)
    rest = re.findall(r'^# (group rows \d+: .+)\n(hipblaslt-bench .+)$', text, re.M)
    for i, (what, cmd) in enumerate(rest):
        if i or cmd.strip() != bench:
            print('bench:          [%s] %s' % (what, cmd.strip()))
        else:
            print('bench group:    %s' % what)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == 'attempt':
        attempt(sys.argv[2], '--reproduced' in sys.argv[3:])
    elif len(sys.argv) == 3 and sys.argv[1] == 'kernel':
        kernel(sys.argv[2])
    else:
        sys.exit(__doc__.split('\n\n')[1])


if __name__ == '__main__':
    main()
