#!/usr/bin/env python3
"""Draft <ARTIFACT_DIR>/report.md for debug-numeric step 9 from the skill's own artifacts.

    python3 scripts/make_report.py <ARTIFACT_DIR> [--out report.md]
    python3 scripts/make_report.py <ARTIFACT_DIR> --fill <answers.md> [--out report.md]

The draft also drafts <ARTIFACT_DIR>/result.json (result_card.py: the facts on disk, TODO for
the root cause and fix in fixed spellings); --fill ends with result_card.py check.

The draft prints its `TODO(agent)` list, numbered, with each one's section -- answer them
from that list, without reading the report back (a re-draft moves an existing report.md to
report.prev<n>.md, so report.md is always the draft just printed). `--fill` puts the answers in, in one call:
the answers file holds `## TODO <n>` headings, each followed by the text that replaces TODO n
(from `TODO(agent)` to the end of its line); then it refreshes the time table from the ledger
(run it after `ledger.py end step9`, so the report-writing time counts) and prints what is
left open.

Fills every section step9-report.md requires that is a fact on disk -- the per-step
time table (ledger.py), the rate lines from the ledger, each wave's summary, the
grouping, every 5b attempt (copies per GPU x GPUs, runs done, hits, early stop,
env verdict, captured build, capture-time contention), each tarball's sha256 and
proposed verdict, the groups never replayed, and the step-5N native fallback
(trigger, candidates, jobs, verdict from the ledger), the step-7 root cause and step-8
fix (ROOT_CAUSE.md, fix/ tarball, their ledger lines) -- and leaves `TODO(agent)` where
judgement is needed (verdict per group, what it means, caveats, next steps).
Model-independent: it reads only what ledger.py, wave_summary.py,
compare_nan_capsules.py, the node launchers and make_repro_readme.py write.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read(p):
    try:
        return open(p, errors='replace').read()
    except OSError:
        return ''


def group_of(d):
    """'G1' for replay/G1/c2.* and verify-G1/repro-G1/out/c2.* (or an op-only package's
    verify-G1-op/repro-G1-op/ and its chain's verify-G1-chain/); else the parent dir."""
    m = re.search(r'(?:^|/)(?:replay/|verify-|repro-)(G\d+)(?:-op|-chain)?(?:/|$)', d)
    return m.group(1) if m else os.path.basename(os.path.dirname(d))


def attempt_facts(d):
    s = read(os.path.join(d, 'summary.txt')).splitlines()
    tot = next((l for l in s if l.startswith('=== totals')), '')
    m = re.search(r'non-finite (\d+)/(\d+)', tot)
    copies = [l for l in s if re.match(r'\s+g\d+_k\d+\.', l)]
    gpus = len({re.match(r'\s+g(\d+)_', l).group(1) for l in copies})
    c = re.search(r'/c(\d+)\.\d+\.([\w-]+)$', d.rstrip('/'))
    res = next((l.split('=', 1)[1] for l in s if l.startswith('ATTEMPT_RESULT=')), '?')
    env = sorted({re.sub(r'^\s*\d+\s+env:\s*', '', l).strip() for l in s if 'env:' in l})
    stop = next((l for l in s if l.startswith('=== stopped early')), '')
    prop = next((re.sub(r'^=== proposed verdict: (.*) -- confirm.*', r'\1', l) for l in s
                 if l.startswith('=== proposed verdict')), '')
    first = next((re.sub(r'^=== first op to go non-finite \(all copies\): (.*) ===$', r'\1', l) for l in s
                  if l.startswith('=== first op to go non-finite')), '')
    log = sorted(glob.glob(os.path.join(d, 'g*_k*.log')))[:1]
    lines = read(log[0]).splitlines() if log else []
    build = next((l.split(':', 1)[1].strip() for l in lines if l.startswith('captured build:')), '?')
    cont = []
    for i, l in enumerate(lines):
        if l.startswith('contention:'):
            cont = [l.split(':', 1)[1].strip()] + [x.strip() for x in lines[i + 1:i + 2] if x.startswith(' ')]
    hits, runs = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    # The load beside each replay copy: a co-tenant (CAPSULE_COTENANT, 5d), else none
    # but the copies themselves. Without it a co-tenant rate reads like a solo rate.
    cot = next((l.split(':', 1)[1].strip() for l in lines if l.strip().startswith('co-tenant:')), '')
    cot = re.sub(r'^python3 \S*/scripts/(?:user/|internal/)?(\S+)', r'\1', cot)   # full command: the sizes are the load
    # an A/B run's own setting (replay_nan_op `A/B (--no-hipblaslt): VAR ...`), or an env difference
    # left after --restore-env: a row that does not say it reads like the plain replay
    ab = sorted({re.sub(r'^\s*A/B \((\S+)\):\s+(\S+).*?here=(\S+).*', r'\1 (\2=\3)', l)
                 for l in lines if l.lstrip().startswith('A/B (')})
    for i, l in enumerate(lines):          # older logs: the differences under a STILL DIFFERENT banner
        if l.startswith('env:') and 'STILL DIFFERENT' in l:
            ab += sorted({re.sub(r'^\s+(\S+)\s+capture=(\S+)\s+here=(\S+).*', r'env \1=\3 (capture \2)', x)
                          for x in lines[i + 1:i + 13] if re.match(r'^\s+\S+\s+capture=', x)})
            break
    load = '; '.join(filter(None, [cot or 'replay copies only'] + [('A/B ' + x) if not x.startswith('env ') else x
                                                                      for x in ab]))
    return dict(dir=d, load=load, conc=c.group(1) if c else '?', host=c.group(2) if c else '?', gpus=gpus,
                copies=len(copies), hits=hits, runs=runs, result=res, env='; '.join(env) or '?',
                build=build, contention=' / '.join(cont) or '?', proposed=prop, first=first,
                stop=re.sub(r'^=== stopped early \(([^)]*)\).*', r'\1', stop) if stop else '')


TODO_RE = re.compile(r'TODO\(agent\).*$')


def todo_list(lines):
    """[(n, section, line index, text)] for every TODO(agent), numbered from 1 in file order."""
    sec, todos = '', []
    for i, l in enumerate(lines):
        if l.startswith('## '):
            sec = l[3:].strip()
        elif 'TODO(agent)' in l:
            todos.append((len(todos) + 1, sec, i, l.strip()))
    return todos


def print_todos(lines):
    for n, sec, _, l in todo_list(lines):
        print('TODO %d [%s] %s' % (n, sec, l[:220]))


def time_table(led):
    t = subprocess.run([sys.executable, os.path.join(HERE, 'ledger.py'), led, 'table'],
                       capture_output=True, text=True).stdout
    return t.strip()


def result_card(cmd, A):
    """result_card.py draft|check: the result as data (compare_results.py compares two runs)."""
    r = subprocess.run([sys.executable, os.path.join(HERE, 'result_card.py'), cmd, A],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip())


def fill(path, answers_path, led):
    """Replace TODO n with the answer under `## TODO n`; refresh the time table."""
    lines = read(path).splitlines()
    if not lines:
        sys.exit('make_report: no report at %s -- draft it first' % path)
    ans, cur = {}, None
    for l in read(answers_path).splitlines():
        m = re.match(r'##\s*TODO\s+(\d+)\s*$', l)
        if m:
            cur = int(m.group(1))
            ans[cur] = []
        elif cur is not None:
            ans[cur].append(l)
    todos = {n: i for n, _, i, _ in todo_list(lines)}
    bad = sorted(set(ans) - set(todos))
    if bad:
        sys.exit('make_report: no TODO %s in %s (it has %d) -- nothing written'
                 % (', '.join(map(str, bad)), path, len(todos)))
    for n in sorted(ans, reverse=True):          # bottom-up: a multi-line answer shifts what follows
        i = todos[n]
        text = '\n'.join(ans[n]).strip()
        lines[i:i + 1] = TODO_RE.sub(lambda _: text, lines[i]).split('\n')
    if os.path.exists(led):                      # the table predates step 9's end: refresh it
        try:
            a = lines.index('## Time per step') + 1
            b = next(j for j in range(a, len(lines)) if lines[j].startswith('Not run:'))
            lines[a:b] = [''] + time_table(led).splitlines() + ['']
        except (ValueError, StopIteration):
            print('make_report: no "## Time per step" ... "Not run:" block -- time table not refreshed')
    open(path, 'w').write('\n'.join(lines) + '\n')
    print('FILLED %s: %d answer(s), time table refreshed' % (path, len(ans)))
    left = todo_list(lines)
    print('%d TODO left' % len(left))
    print_todos(lines)
    result_card('check', os.path.dirname(os.path.abspath(path)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('artifact_dir')
    p.add_argument('--out', default=None)
    p.add_argument('--fill', metavar='ANSWERS', default=None,
                   help='answers file (## TODO <n> headings): fill the report, refresh its time table')
    a = p.parse_args()
    A = os.path.abspath(a.artifact_dir)
    led = os.path.join(A, 'ledger.md')
    if a.fill:
        return fill(a.out or os.path.join(A, 'report.md'), a.fill, led)
    ledger = read(led).splitlines()
    out = []
    w = out.append
    w('# NaN investigation report -- %s\n' % os.path.basename(A))
    w('_Drafted by make_report.py from the ledger and artifacts (numbers copied from files, not '
      'from memory); the judgement lines are the agent\'s._\n')

    w('## Verdict\n\nTODO(agent): one paragraph per group -- verdict (step5-read.md table), the op, '
      'and what it means; plus the step-5N verdict if it ran (a native reproducer, not a capsule). '
      'Use the per-rung wording in step9-report.md. Then the three outcomes, one line each: '
      'reproducer (step 6), root cause (step 7), fix (step 8) -- never "reproducer done" as "fixed".\n')

    gl = [re.sub(r'^\S+ \S+\s+', '', l) for l in ledger if re.search(r'GPU=\S+ ARCH=\S+', l)]
    w('## GPU\n')
    w('\n'.join('* ' + l[:220] for l in gl[-2:]) if gl else
      '* TODO(agent): the device line (gpu_check.py --torch: name, arch, CUs, memory) -- not in the ledger')
    w('')

    w('## Time per step\n')
    if os.path.exists(led):
        t = time_table(led)
        ended = set(re.findall(r'^\S+ \S+\s+(step\d+)\S*\s+end\b', '\n'.join(ledger), re.M))
        # `ledger.py skip` lines: the table already lists those steps as *skipped*
        ended |= set(re.findall(r'^\S+ \S+\s+(step\d+)\S*\s+skipped\b', '\n'.join(ledger), re.M))
        skipped = []
        for st in ['step%d' % i for i in range(2, 9)]:
            if st in ended:
                continue
            pat = r'\b%s\b[^;(]{0,12}\b(skip\w*|SKIP|NO|declined)\b' % st
            hits = [l for l in ledger if re.search(pat, l)]
            # the step's own line first ("<ts>  step2  skipped ..."), then a mention elsewhere (step 1's answers)
            hits.sort(key=lambda l: not re.match(r'^\S+ \S+\s+%s\b' % st, l))
            # quote the clause that says it, not the whole line (step 1's answers list every step)
            why = []
            for l in hits:
                body = re.sub(r'^\S+ \S+\s+\S+\s+', '', l)
                m = re.search(pat, body)
                c = body[m.start():].split(';')[0].strip() if m and not re.match(r'^\S+ \S+\s+%s\b' % st, l) else body
                if c.count(')') > c.count('('):
                    c = c.rstrip(')').strip()
                why.append(c[:160])
            if not why and st == 'step5' and any('CAPTURE_TO_REPRO_RESULT=' in l for l in ledger):
                why = ['5a/5b ran inside step 4\'s capture-to-reproducer job (timed under step4)']
            if not why and st == 'step3' and any(re.search(r'(?i)\bstep2\b.*\bskip', l) for l in ledger):
                why = ['follows from the step-2 skip (no rate to raise)']
            skipped.append('* %s: %s -- %s' % (st, '*folded*' if why and why[0].startswith('5a/5b ran') else '*skipped*',
                                                 why[0] if why else 'TODO(agent) why'))
        w(t + '\n\nNot run:\n\n' + ('\n'.join(skipped) or '* (none)') + '\n')

    w('## Where the rate came from\n')
    rate = [l for l in ledger if re.search(r'\bstep[24]\b', l) and re.search(r'rate|skip|wave', l)]
    w('\n'.join('    ' + l for l in rate[-8:]) or '    (no rate line in the ledger)')
    for f in sorted(glob.glob(os.path.join(A, 'wave_summary.*.txt'))):
        s = [l for l in read(f).splitlines() if l.startswith(('   ', '== by GPU', '== NaN step', '!!'))]
        w('\n`%s`:\n\n' % os.path.basename(f) + '\n'.join('    ' + l for l in s))
    if any(re.search(r'(?i)\bstep2\b.*\bskip', l) for l in ledger):
        ctl = [l.strip() for f in glob.glob(os.path.join(A, 'wave_summary.*.txt'))
               for l in read(f).splitlines() if re.match(r'\s+ctl\s+rate', l)]
        if not ctl:   # no saved wave_summary file: the step-4 ledger line that quotes the control arm
            ctl = [re.sub(r'^\S+ \S+\s+\S+\s+', '', l)[:200] for l in ledger
                   if re.search(r'\sstep4\s', l) and re.search(r'\bctl\b.*\d+/\d+', l)]
        if ctl:
            w('\nStep 2 was skipped: the only model-level rate is the step-4 control arm (%s), '
              'in that wave\'s environment -- counted per trial log: a log (container) that runs '
              'several processes counts once; per-process counts are in the ledger.\n' % '; '.join(ctl))
        else:
            w('\nStep 2 was skipped and the capture wave had no control arm: steps 2-4 measured no '
              'model-level rate. Step 8\'s stock arm, if it ran, is the only uncaptured measurement '
              '(see Fix). TODO(agent): quote it, or say there is none.\n')
    else:
        w('\nTODO(agent): measured in step 2 (with environment), or user-supplied (provisional)?\n')
    # Step 8's model A/B stock arm: the same model, stock, usually more trials than step 4's 2-6
    # controls (2026-11-20 qwen_tf32: controls 0/6, stock arm 3/12) -- a second rate, quoted beside it
    ab = [re.sub(r'^\S+ \S+\s+', '', l)[:220] for l in ledger
          if re.search(r'\sstep8\S*\s', l) and re.search(r'(?i)\bstock\b[^;]*?\d+/\d+', l)
          and re.search(r'(?i)\b(model|trials?|runs? NaN)\b', l)]          # the model's, not the reproducer's launches
    if ab:
        w('\nStep 8\'s model A/B, stock arm (a second model-level measurement, same environment as its '
          'job): quote it beside the rate above and take the larger denominator as `<NAN_RATE>`:\n\n'
          + '\n'.join('    ' + l for l in ab[:3]))

    w('## Image, build and replay environment\n')
    # The replay ladder and hand-run 5b write replay/<G>/; the docker/venv flow of step6-build.md
    # replays from the extracted tarball, so its attempts live in verify-<G>/.../out/.
    atts = sorted(glob.glob(os.path.join(A, 'replay', '*', 'c*.*.*')) +
                  glob.glob(os.path.join(A, 'verify-*', '*', 'out', 'c*.*.*')))
    facts = [attempt_facts(d) for d in atts if os.path.exists(os.path.join(d, 'summary.txt'))]
    imgs = sorted({m.group(1) for f in glob.glob(os.path.join(A, 'repro-*', 'README.md'))
                   for m in [re.search(r'\* (?:Image|Environment[^:]*): `([^`]+)`', read(f))] if m})
    kerns = {}
    for k in sorted(glob.glob(os.path.join(A, 'replay', '*', 'kernel.*.txt')) +
                    glob.glob(os.path.join(A, 'verify-*', '*', 'out', 'kernel.*.txt'))):
        kf = dict(re.match(r'(\w+):\s*(.*)', l).groups() for l in read(k).splitlines() if re.match(r'\w+:', l))
        kerns[group_of(os.path.dirname(k))] = kf
    inj = sorted({kf.get('injected', '?').split(' --')[0] for kf in kerns.values()})
    native_v = [re.search(r'VERDICT (NATIVE_\w+)', l).group(1) for l in ledger if re.search(r'VERDICT NATIVE_\w+', l)]
    if inj:
        injected = ', '.join(inj)
    elif native_v:   # no capsule kernel to enumerate: the native verdict names the owner instead
        injected = 'n/a (native reproducer, %s: see Native reproducer / Root cause)' % native_v[-1]
    else:
        injected = 'unknown (no kernel identified)'
    w('* image: %s -- NaN-injected: %s' % (', '.join(imgs) or 'TODO(agent)', injected))
    for b in sorted({f['build'] for f in facts}):
        w('* captured build: `%s`' % b)
    for e in sorted({f['env'] for f in facts}):
        w('* env banner: `%s`' % e)
    w('')

    if kerns:
        w('## Kernel\n')
        for g, kf in sorted(kerns.items()):
            w('* %s: solution_index %s, %s -- `%s`' % (g, kf.get('solution_index', '?'),
                                                     kf.get('arch', '?'), kf.get('kernel', '?')))
            w('  * bench: `%s`' % kf.get('bench', '?'))
        idx = {kf.get('solution_index') for kf in kerns.values()}
        if len(kerns) > 1 and len(idx) == 1:
            w('\nEvery identified group dispatches the SAME kernel (solution_index %s).' % idx.pop())
        w('')
    w('## Replays (5b) and the contention behind every rate\n')
    w('| group | attempt | host | copies/GPU x GPUs | load beside each copy | non-finite / runs | early stop | result | proposed verdict | first non-finite | capture-time contention |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for f in facts:
        g = group_of(f['dir'])
        w('| %s | `%s` | %s | %s x %d | %s | %d/%d | %s | %s | %s | %s | %s |' % (
            g, os.path.basename(f['dir']), f['host'], f['conc'], f['gpus'], f['load'], f['hits'], f['runs'],
            f['stop'] or 'no', f['result'], f['proposed'] or '-', f['first'] or '-', f['contention']))
    # COLLECTIVE groups: one torchrun over the member ranks (collective_r3.py replay), no ladder summary.txt
    coll = {}
    for o in sorted(glob.glob(os.path.join(A, 'replay', '*', 'collective.*.out'))):
        t = read(o)
        g = os.path.basename(os.path.dirname(o))
        res = (re.findall(r'^LADDER_RESULT=(.*)$', t, re.M) or ['none'])[-1]
        ver = (re.findall(r'^VERDICT (\S+)', t, re.M) or ['-'])[-1]
        path = (re.findall(r'^R3 vllm: path (.*)$', t, re.M) or ['-'])[-1]
        host = (re.search(r'collective\.[^.]+\.([^.]+)\.out$', o) or [None, '?'])[1]
        coll[g] = True
        w('| %s | `%s` | %s | one torchrun over the member ranks | - | %s | - | %s | %s | %s | - |' % (
            g, os.path.basename(o), host, res.split()[-1] if res != 'none' else '-', res.split()[0], ver, path))
    w('\nA one-copy `0/N` on a contention-gated fault is an experiment not set up, not a clean result.\n')

    w('## Groups\n\n(capsule paths are relative to the artifact dir)\n')
    pr = read(os.path.join(A, 'groups_priority.txt')).splitlines()
    replayed = {group_of(f['dir']) for f in facts} | set(coll)
    # op of each group (groups_priority.txt: <G> <n> <capsule> <op>): a group with a replayed group's op is the
    # same site captured again -- said here, not left as a TODO (2026-11-23: three of them, one answer)
    gop = {l.split()[0]: l.split()[3] for l in pr if len(l.split()) >= 4}
    rop = {}
    for g in sorted(replayed):
        rop.setdefault(gop.get(g), g)
    for l in pr:
        g = l.split()[0]
        l = re.sub(r'\S*/%s/' % re.escape(os.path.basename(A)), '', l)   # relative to the artifact dir
        if g in replayed:
            why = ''
        elif gop.get(g) in rop:
            why = '  -- **not replayed**: same op as %s (replayed) -- folded into it' % rop[gop[g]]
        else:
            why = '  -- **not replayed**: TODO(agent) why'
        w('* %s%s' % (l, why))
    w('')

    w('## Reproducer tarballs\n')
    last, fixes = {}, {}                         # the latest line per tarball wins
    for l in ledger:
        m = re.search(r'TARBALL (\w+)\s+(\S+\.tar\.gz)\s+size=(\d+)\s*(.*)', l)
        if m:
            # a step-8 fix package is not a reproducer: listed under its own heading
            # keyed by file name: the same tarball is ledgered with and without its directory
            (fixes if re.search(r'\sstep8\S*\s', l) else last)[os.path.basename(m.group(2))] = m
    fmt = lambda d: '\n'.join('* `%s`  sha256 `%s`  %s B  %s' % (  # noqa: E731
        m.group(2), m.group(1), m.group(3), m.group(4)) for _, m in sorted(d.items()))
    w(fmt(last) or '* none')
    if fixes:
        w('\nFix tarballs (step 8, not reproducers):\n')
        w(fmt(fixes))
    w('\nTODO(agent): confirm each README `CHECK` line was resolved; say which groups were folded.\n')

    native = [re.sub(r'^\S+ \S+\s+', '', l) for l in ledger if re.search(r'\bstep(?:5n|6\[native\])\s', l, re.I)]
    skipped = [l for l in native if re.match(r'step5n\s+skipped\b', l, re.I)]
    if skipped and len(skipped) == len(native) and not os.path.isdir(os.path.join(A, 'native')):
        w('Step 5N did not run: %s\n' % skipped[-1].split('skipped', 1)[1].strip())   # a capsule reproduced
        native = []
    if native or os.path.isdir(os.path.join(A, 'native')):
        w('## Native reproducer (step 5N)\n')
        trig = [l for l in native if 'NATIVE_TRIGGER=' in l]
        w('* trigger: %s' % (trig[-1].split(None, 1)[1] if trig else 'TODO(agent) -- no NATIVE_TRIGGER line'))
        cands = sorted(glob.glob(os.path.join(A, 'native', 'repro_*.py')),
                       key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p)])
        if os.path.exists(os.path.join(A, 'native', 'handoff.md')):
            w('* handoff from the capsule path: `native/handoff.md` (KEEP/STRIKE lines below)')
        w('* candidates on disk: %s' % (', '.join('`%s`' % os.path.basename(c) for c in cands) or 'none'))
        rows = [l.split(None, 1)[1] for l in native
                if not re.match(r'\S+\s+(start|end)\b', l) and 'NATIVE_TRIGGER=' not in l]
        w('\nLedger (evidence, candidates, jobs, verdict, verification):\n\n' + ('\n'.join('    ' + r for r in rows) or '    (none)'))
        hits = sorted(glob.glob(os.path.join(A, 'native', 'hit_*.pt')))
        w('\n* saved hits: %d%s' % (len(hits), ' (e.g. `%s`)' % os.path.basename(hits[0]) if hits else ''))
        w('\nTODO(agent): the 5N verdict (step5n-native.md N6), the minimal candidate and what it '
          'needs (load, components), and every removal or load that ran clean -- the trigger\'s bounds.\n')

    for st, d, f, title in (('step7', 'rootcause', 'ROOT_CAUSE.md', 'Root cause (step 7)'),
                            ('step8', 'fix', 'README.md', 'Fix (step 8)')):
        # the step's own lines (its 3rd field), not other steps' lines that mention it
        rows = [l.split(None, 1)[1] for l in ledger if re.match(r'\S+\s+\S+\s+%s\b' % st, l)
                and not re.match(r'\S+\s+\S+\s+\S+\s+(start|end)\b', l)]
        doc = os.path.join(A, d, f)
        w('## %s\n' % title)
        w('* `%s`: %s' % (os.path.join(d, f), 'on disk' if os.path.exists(doc) else 'missing'))
        on_disk = len(glob.glob(os.path.join(A, d, 'iter-*.md')))
        in_ledger = len(set(re.findall(r'\b%s(?:\[[^\]]*\])?\s+iter (\d+)\b' % st, '\n'.join(ledger))))
        w('* iterations on disk: %d%s' % (on_disk, '' if in_ledger in (0, on_disk) else
          ' -- **the ledger has %d**: one `iter-<k>.md` per iteration (%s), write the missing one(s)'
          % (in_ledger, 'step7-rootcause.md' if st == 'step7' else 'step8-fix.md')))
        if st == 'step7':
            # Discovery or confirmation: a root cause re-tested from an earlier investigation's
            # ROOT_CAUSE.md is a confirmation (2026-11-20: all six validation runs, iteration 1)
            txt = '\n'.join(ledger)
            pri = sorted(set(re.findall(r'prior:\s*(\S*artifacts/[^\s,;)]+|\S+/rootcause/ROOT_CAUSE\.md)', txt)))
            if re.search(r'PRIORS=none|prior: skipped', txt):
                w('* priors: **none** (PRIORS=none) -- the root cause was found from this run\'s evidence alone')
            elif pri:
                w('* priors: hypothesis from %s re-tested here -- a root cause matching it is a '
                  '**confirmation**, not a discovery; say so in the root cause' % ', '.join('`%s`' % x for x in pri[:3]))
            elif re.search(r'prior: none', txt):
                w('* priors: none found (`prior: none`)')
        w('\nLedger (budget, kind, iterations, outcome):\n\n' + ('\n'.join('    ' + r for r in rows) or '    (none -- skipped?)'))
        w('\nTODO(agent): %s\n' % (
            'the root cause (where, mechanism, trigger, causal experiment, defence: the guard and why it did not apply, or none) and which track found it, '
            'or NO_ROOT_CAUSE with excluded and still-open hypotheses, or why skipped.' if st == 'step7' else
            'every fix shipped, as fix/README.md\'s table (one row each: kind -- real fix / targeted workaround / '
            'broad workaround --, the change, what else it affects, validation: control n/N, patched 0/N with expected '
            'hits, load, values, model A/B, cost; or "not validated here: please try"), the recommended one and why, '
            'tarball sha256 -- or NO_FIX with candidates tried, or why skipped.'))

    capped = ['* 5b %s stopped at concurrency %s (the first that reproduced)%s'
              % (group_of(f['dir']), f['conc'],
                 '; early stop at %s' % f['stop'] if f['stop'] else '')
              for f in facts if f['result'] == 'REPRODUCED']
    w('## Capped, skipped or folded\n\n' + '\n'.join(capped) +
      '\n\nTODO(agent): anything else capped, sampled or folded -- and on what basis.\n')
    w('## Next steps\n\nTODO(agent)\n')
    w('## Result card\n\n`result.json` -- this result as data (`result_card.py`); '
      '`compare_results.py <this dir> <another run>` says whether two runs found the same bug, '
      'root cause and fix.\n')
    path = a.out or os.path.join(A, 'report.md')
    # A re-draft replaces report.md and keeps the old one as report.prev<n>.md: --fill always fills
    # report.md, so it must be the draft whose TODO numbers were just printed.  Writing the new
    # draft beside it (report.draft.md) let --fill put every answer into the stale report, one
    # TODO off, silently (2026-11-20).
    if os.path.exists(path):
        n = 1
        while os.path.exists(re.sub(r'\.md$', '.prev%d.md' % n, path)):
            n += 1
        prev = re.sub(r'\.md$', '.prev%d.md' % n, path)
        os.replace(path, prev)
        print('make_report: the earlier %s is now %s' % (os.path.basename(path), os.path.basename(prev)))
    open(path, 'w').write('\n'.join(out) + '\n')
    lines = '\n'.join(out).splitlines()
    print('DRAFT %s (%d TODO)' % (path, len(todo_list(lines))))
    print_todos(lines)
    result_card('draft', A)


if __name__ == '__main__':
    main()
