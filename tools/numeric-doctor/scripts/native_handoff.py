#!/usr/bin/env python3
"""Write <ARTIFACT_DIR>/native/handoff.md: what the capsule path tried, for step 5N (debug-numeric).

    python3 scripts/native_handoff.py <ARTIFACT_DIR> [--logs <dir with <job>.out files>]

Step 5N starts where the capsule path stopped; its failures are evidence.  This
collects them from the files the ladder already writes -- the ledger, the wave
summaries, the capture trial logs, the grouping, every 5b attempt's summary.txt and
one copy log, and each 5c/5d pressure sweep's result table (found in --logs by the
job ids the ledger recorded) -- and writes one file:

  * Trigger      why 5N was entered
  * Facts        image actually used, precision, model command, when the model NaNs
                 (capture vs control), groups that share one op geometry
  * Capture      the census of NaN-creating sites, aggregated over trials
  * Compiled     C1's kernel capsules, output code and the capture job's replay of the kernel
  * Contention   the capture's load beside every replay's: processes on the GPU,
                 what they run, memory -- LIGHTER flags a load far below the capture's
  * Attempts     one row per replay: load, n/N, verdict, walk -- and every COLLECTIVE group's
                 replay (replay/<G>/collective.*.out: LADDER_RESULT= / VERDICT)
  * Ruled out    every clean experiment with the smallest rate it excludes (95%):
                 a null shorter than 3/rate rules nothing out at that rate
  * Implications PROPOSED readings, each with its source; the agent confirms or
                 strikes them (step5n-native.md N1) -- the script does not decide
  * Reusable     capsule per group, co-tenant variants, model hipBLASLt log

Model-independent; reads only what the ladder writes.
"""
import argparse
import ast
import collections
import glob
import json
import os
import re
import statistics

LEDGER_LINE = re.compile(r'^(\S+ \S+)\s+(\S+)\s+(.*)$')
KIND = {'full': 'fwd+bwd+optimizer', 'fwdbwd': 'fwd+bwd', 'fwd': 'fwd only', 'adamw': 'optimizer only',
        'copy': 'HBM copy', 'triad': 'HBM triad', 'gemm': 'big GEMM', 'mix': 'copy+GEMM',
        'bust': 'same GEMM, cache-busting'}
NO_BACKWARD = ('fwd only', 'optimizer only', 'HBM copy', 'HBM triad', 'big GEMM', 'copy+GEMM',
               'same GEMM, cache-busting')


def read(p):
    try:
        return open(p, errors='replace').read()
    except OSError:
        return ''


def first(lines, pat):
    return next((l.strip() for l in lines if re.search(pat, l)), '')


def excl(runs):
    """Smallest per-run rate a 0/runs null excludes at 95% (rule of three)."""
    return 3.0 / runs if runs else None


def fmt_rate(r):
    return '-' if r is None else '%.1e' % r


def attempt(summary_path, A):
    d = os.path.dirname(summary_path)
    s = read(summary_path).splitlines()
    rel = os.path.relpath(d, A)
    g = re.search(r'(?:^|/)(?:replay/|verify-|repro-)(G\d+)', rel)
    c = re.search(r'(?:^|/)c(\d+)\.(\d+)\.([^/]+)$', rel)
    tot = first(s, r'^=== totals')
    nf = re.search(r'non-finite (\d+)/(\d+)', tot)
    gpus = len({m.group(1) for l in s for m in [re.match(r'\s+g(\d+)_k\d+\.', l)] if m})
    walk = [l for l in s if re.match(r'\s+\d+\s+aten\.', l)]
    walk_verdict = [l.strip() for l in s if l.startswith('[verdict]')]
    copy = sorted(glob.glob(os.path.join(d, 'g*_k*.log')))[:1]
    cl = read(copy[0]).splitlines() if copy else []
    return dict(
        rel=rel, group=g.group(1) if g else '?', conc=c.group(1) if c else '?',
        job=c.group(2) if c else '?', host=c.group(3) if c else '?', gpus=gpus,
        hits=int(nf.group(1)) if nf else None, runs=int(nf.group(2)) if nf else None,
        result=first(s, r'^ATTEMPT_RESULT=').split('=', 1)[-1],
        futility=first(s, r'futility stop'),
        first_wrong=first(s, r'-> first op to go wrong'),
        tally=first(s, r'first op to go non-finite'),
        terminal=first(s, r'terminal fp64 ref'),
        diverged=[l.strip() for l in walk if 'DIVERGED' in l],
        terminal_idx=walk[-1].split()[0] if walk else None,
        mutated=[l.strip() for l in walk if 'MUTATED AFTER PRODUCTION' in l],
        walk=walk_verdict[-1] if walk_verdict else '',
        env=first(s, r'env:'),
        precision=first(cl, r'^precision:'), under=first(cl, r'captured under:'),
        sif=next((m.group(1) for l in cl for m in [re.search(r'apptainer exec .* (\S+\.sif)\b', l)] if m), ''),
        build=first(cl, r'^captured build:'),
    )


def collective_replays(A):
    """COLLECTIVE groups' 5b replays: one torchrun over the member ranks (collective_r3.py), no
    summary.txt -- replay/<G>/collective.<job>.<host>.out, parsed as make_report.py does.  Missed
    before: the handoff said "no 5b attempt / nothing ruled out / none was replayed" beside a
    0/2000 collective replay (2026-11-23, vLLM TP=2 quick-reduce)."""
    rows = []
    for o in sorted(glob.glob(os.path.join(A, 'replay', '*', 'collective.*.out'))):
        t = read(o)
        res = (re.findall(r'^LADDER_RESULT=(.*)$', t, re.M) or [''])[-1]
        nf = re.search(r'(\d+)/(\d+)', res)
        m = re.search(r'collective\.([^.]+)\.([^/]+)\.out$', o)
        rows.append(dict(
            rel=os.path.relpath(o, A), group=os.path.basename(os.path.dirname(o)),
            job=m.group(1) if m else '?', host=m.group(2) if m else '?',
            result=res.split()[0] if res else '', hits=int(nf.group(1)) if nf else None,
            runs=int(nf.group(2)) if nf else None,
            verdict=(re.findall(r'^VERDICT (\S+)', t, re.M) or [''])[-1],
            path=(re.findall(r'^R3 \w+: path (.*)$', t, re.M) or [''])[-1].strip(),
            ranks=(re.findall(r'^VERDICT \S+ ranks (\[[^]]*\])', t, re.M) or [''])[-1],
            op=(re.findall(r'^collective (\S+)', t, re.M) or ['?'])[-1]))
    return rows


def workload(text):
    """'inference' / 'training' / '' from the ledger's RUN_SCRIPT= and unwrapped command: the handoff
    called a vLLM serving run "the model's training step" (2026-11-23)."""
    t = ' '.join(x for x in text if re.search(r'RUN_SCRIPT=|NAN_SIGNAL=|unwrapped|MODEL_CMD|command', x))
    if re.search(r'/inference/|\b(vllm|sglang|serve|generate|inference|decode|prefill|logprobs?)\b', t, re.I):
        return 'inference'
    if re.search(r'/training/|train|fwdbwd|backward|optimizer|\bloss\b|grad', t, re.I):
        return 'training'
    return ''


def pressure_rows(ledger, logs):
    """5c/5d sweep rows from <logs>/*.<job>.out, by the job ids the ledger recorded."""
    rows = []
    for l in ledger:
        m = LEDGER_LINE.match(l)
        if not (m and re.match(r'step5[cd]', m.group(2))):
            continue
        j = re.search(r'\bjob=(\d+)', m.group(3))
        for f in sorted(glob.glob(os.path.join(logs, '*.%s.out' % j.group(1)))) if j else []:
            t = read(f)
            if '=== Results' not in t:
                continue
            for r in t.split('=== Results', 1)[1].splitlines()[1:]:
                x = re.match(r'\s*GPU (\d+)\s+(\S+)\s+non-finite on (\d+)/(\d+).*?\[(.*?)\]\s*(.*)$', r)
                if not x:
                    continue
                cmd = x.group(6).strip()
                mode = re.search(r'--mode (\w+)', cmd)
                procs = re.search(r'--procs (\d+)', cmd)
                gb = re.search(r'([\d.]+) GB taken', x.group(5))
                real = 'cotenant_loop' in cmd or 'cotenant.sh' in cmd
                tool = 'model' if real else next((os.path.basename(t_) for t_ in cmd.split() if t_.endswith('.py')), '?')
                rows.append(dict(job=j.group(1), gpu=x.group(1), verdict=x.group(2), hits=int(x.group(3)),
                                 runs=int(x.group(4)), cmd=cmd, tool=tool,
                                 procs=int(procs.group(1)) if procs else (None if real else 1),
                                 kind='the real model' if real else KIND.get(mode.group(1) if mode else '', '?'),
                                 gb=float(gb.group(1)) if gb else None))
    return rows


def capture_facts(A):
    """Census rows, captured ops and contention, aggregated over the capture trials' logs."""
    logs = []
    for w in sorted(glob.glob(os.path.join(A, 'wave.*.out'))):   # the capture-to-reproducer job names them
        logs += re.findall(r'(/\S+?\.cap\.\S+?\.log)', read(w))
    # docker / hand-written launchers: the trial logs themselves (<gpu>_<trial>.cap.<job>.<host>.log) anywhere
    # under the artifact dir, e.g. wave/
    logs += sorted(glob.glob(os.path.join(A, '**', '*.cap.*.log'), recursive=True))
    sites = collections.OrderedDict()
    captured = collections.Counter()
    held, procs, cpgs = [], [], []
    armed = 0
    for lg in dict.fromkeys(logs):
        t = read(lg)
        if 'nan-capture armed' not in t:          # killed before arming (early stop): says nothing
            continue
        armed += 1
        cpgs += [int(x) for x in re.findall(r'nan-capture armed.*?containers_per_gpu=(\d+)', t)]
        lines = t.splitlines()
        census = False
        for i, l in enumerate(lines):
            m = re.search(r'\]\s+(survived|MASKED|pending|captured|settled)\s+(\S+)\s+(\d+)\s+(\d+)\.\.(\d+)\s+(.*)$', l)
            if m:
                census = True
                s = sites.setdefault((m.group(1), m.group(6).strip()), dict(trials=0, hits=0, lo=10 ** 9, hi=-1))
                s['trials'] += 1
                s['hits'] += int(m.group(3))
                s['lo'] = min(s['lo'], int(m.group(4)))
                s['hi'] = max(s['hi'], int(m.group(5)))
            m = re.search(r'\]\s+op=(\S+)\s+(?:op|gemm) call', l)  # gemm: pre-rename logs
            if m:
                # several processes write one log: the problem line may be dozens of lines on
                prob = next((re.sub(r'.*hipBLASLt problem:\s*', '', x) for x in lines[i:i + 80]
                             if 'hipBLASLt problem' in x), '')
                captured[(m.group(1), prob.strip())] += 1
            m = re.search(r'contention: (\d+) process\(es\) on this device.*?(?:([\d.]+) GB held by others)?$', l)
            if m:
                procs.append(int(m.group(1)))
                if m.group(2):
                    held.append(float(m.group(2)))
        if not census:
            # no exit census (the trial died, e.g. on a capsule save error): its `pending:` lines
            # still name the sites the hook saw -- counted once per trial each
            seen = {}
            for l in lines:
                m = re.search(r'\]\s+pending:\s+(\S+)\s+@\s+(\S+).*?\bat step (\d+)', l)
                if m:
                    key = '%s @ %s' % (m.group(1), m.group(2))
                    seen[key] = min(seen.get(key, 10 ** 9), int(m.group(3)))
            for key, st in seen.items():
                s = sites.setdefault(('pending (no census: trial died)', key),
                                     dict(trials=0, hits=0, lo=10 ** 9, hi=-1))
                s['trials'] += 1
                s['hits'] += 1
                s['lo'] = min(s['lo'], st)
                s['hi'] = max(s['hi'], st)
    return dict(armed=armed, sites=sites, captured=captured, held=held, procs=procs,
                cpg=max(cpgs) if cpgs else None)


def source_line(A, site):
    """' -- <the code at file:line>' for a census site, from the model source copy in <A>/src."""
    m = re.search(r'([\w.-]+\.py):(\d+)', site)
    if not m:
        return ''
    for f in glob.glob(os.path.join(A, 'src', '**', m.group(1)), recursive=True)[:1]:
        lines = read(f).splitlines()
        i = int(m.group(2)) - 1
        if 0 <= i < len(lines):
            return '` -- `%s' % lines[i].strip()[:70]
    return ''


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('artifact_dir')
    p.add_argument('--logs', default=os.environ.get('NAN_LOGS', os.path.join(os.getcwd(), 'logs')))
    p.add_argument('--out', default=None)
    a = p.parse_args()
    A = os.path.abspath(a.artifact_dir)
    ledger = read(os.path.join(A, 'ledger.md')).splitlines()
    text = [LEDGER_LINE.match(l).group(3) if LEDGER_LINE.match(l) else l for l in ledger]
    atts = [attempt(f, A) for f in sorted(glob.glob(os.path.join(A, '**', 'summary.txt'), recursive=True))
            if '/native/' not in f and '/blind/' not in f]
    colls = collective_replays(A)
    wl = workload(text)
    step = {'inference': 'the model\'s inference (forward only)', 'training': 'the model\'s training step'}.get(
        wl, 'the model\'s step')
    sweeps = pressure_rows(ledger, a.logs)
    cap = capture_facts(A)
    grp = read(os.path.join(A, 'groups_priority.txt')).strip()
    pts = sorted(glob.glob(os.path.join(A, 'capsules', '**', '*.pt'), recursive=True))
    out = []
    w = out.append
    imp = []          # (proposal, source)

    w('# Step 5N handoff -- %s\n' % os.path.basename(A))
    w('_Written by native_handoff.py from the capsule path\'s artifacts. `Implications` are '
      'PROPOSED: confirm or strike each before N3 (step5n-native.md)._\n')

    # ---------------------------------------------------------------- trigger
    w('## Trigger\n')
    trig = [t for t in text if 'NATIVE_TRIGGER=' in t] or [t for t in text if 'CAPTURE_TO_REPRO_RESULT=' in t or 'FUSED_RESULT=' in t]  # FUSED_RESULT: pre-rename ledgers
    w('\n'.join('* ' + t for t in trig[-3:]) or '* (no NATIVE_TRIGGER or CAPTURE_TO_REPRO_RESULT line in the ledger)')

    # ---------------------------------------------------------------- facts
    w('\n## Facts\n')
    a0 = next((t for t in atts if t['sif'] or t['precision']), None)
    if a0 and a0['sif']:
        w('* image actually used (5b copy log): `%s`%s -- not necessarily the launcher\'s default'
          % (a0['sif'], '; %s' % a0['build'] if a0['build'] else ''))
    else:
        # the investigation's profile names it exactly; else an IMAGE=/SIF= value in the ledger
        prof = {}
        envf = os.path.join(A, 'env.sh')
        if os.path.isfile(envf):
            for ln in open(envf, errors='replace'):
                m = re.match(r'\s*(IMAGE|SIF|VENV|RUNTIME)=["\']?([^"\'\s#]+)', ln)
                if m:
                    prof[m.group(1)] = m.group(2)
        img = prof.get('SIF') if prof.get('RUNTIME') == 'apptainer' and prof.get('SIF') else \
            prof.get('VENV') if prof.get('RUNTIME') == 'native' else prof.get('IMAGE')
        if img:
            w('* image (env.sh, RUNTIME=%s): `%s`' % (prof.get('RUNTIME', '?'), img))
        else:
            vals = [m.group(2) for t in text for m in re.finditer(r'\b(IMAGE|SIF)=(\S+)', t)]
            w('* image (ledger): %s' % ('`%s`' % vals[-1] if vals else 'TODO -- no IMAGE= / SIF= in the ledger'))
    if a0 and a0['precision']:
        w('* precision at capture: `%s`' % a0['precision'].replace('precision:', '').strip())
    if a0 and a0['under']:
        w('* model command at capture: `%s`' % a0['under'].replace('captured under:', '').strip()[:250])
    for f in sorted(glob.glob(os.path.join(A, 'wave_summary.*.txt'))):
        arms = collections.defaultdict(list)
        for l in read(f).splitlines():
            m = re.match(r'trial\s+\d+\s+gpu\s+\d+\s+(cap|ctl)\s+(CAPSULE|NAN)\s+last_step=(\d+)', l)
            if m:
                arms[m.group(1)].append(int(m.group(3)))
        for arm, st in sorted(arms.items()):
            w('* model failure step (NaN or the signal), %s arm (`%s`): %d trial(s), steps %d..%d, median %g'
              % ('capture' if arm == 'cap' else 'control', os.path.basename(f), len(st), min(st), max(st),
                 statistics.median(st)))
    try:
        groups = json.load(open(os.path.join(A, 'groups.json')))['groups']
    except Exception:
        groups = []
    by_key = collections.defaultdict(list)
    for i, g in enumerate(groups):
        ident = g.get('identity') or {}
        if isinstance(ident, str):
            try:
                ident = ast.literal_eval(ident)
            except Exception:
                ident = {}
        by_key[ident.get('geometry_key', '?')].append('G%d %s (%s)' % (i + 1, ident.get('op', '?'), g.get('count', '?')))
    for k, gs in by_key.items():
        if len(gs) > 1:
            w('* **same op geometry `%s`**: %s -- one kernel; a candidate for one covers both' % (k, ', '.join(gs)))
    km = [t for t in text if re.search(r'KERNEL=|KERNEL_(MISMATCH|MATCH|UNKNOWN)', t)]
    if km:
        w('\n'.join('* ' + t[:200] for t in km[-2:]))

    # ---------------------------------------------------------------- capture
    w('\n## Capture (census over %d armed capture trial(s))\n' % cap['armed'])
    if cap['sites']:
        w('| status | site | trials | hits | steps |')
        w('|---|---|---|---|---|')
        for (st, site), s in cap['sites'].items():
            w('| %s | `%s` | %d | %d | %d..%d |' % (st, site[:90] + source_line(A, site), s['trials'], s['hits'],
                                                  s['lo'], s['hi']))
    for (op, prob), n in cap['captured'].most_common():
        w('* captured %dx: `%s`  %s' % (n, op, prob))
    if not cap['armed']:
        w('(no armed capture log found via wave.*.out or **/*.cap.*.log -- run capture_digest.py on a capture trial log by hand)')
    # A finite signal (wrong values, no NaN): the candidate's hit is a wrong output, and the
    # recipe track runs native_sweep.py -- said here, where 5N starts reading.
    sm = re.search(r'^NAN_SIGNAL=(.*)$', read(os.path.join(A, 'capture.submit')) or '', re.M)
    signal = sm.group(1).replace('\\', '') if sm else ''
    finite = any('wrong_value' in (read(f) or '') for f in glob.glob(os.path.join(A, 'capsule_info.*.txt')))
    if finite or (signal and not re.search(r'nan|finite|inf', signal, re.I)):
        imp.append(('A finite signal (wrong values, no NaN): a candidate\'s hit is an output off its higher-'
                    'precision reference beyond a correct call\'s error, or one that differs between reruns on '
                    'the same inputs (N3); honour NATIVE_DISCOVER=1, and step 7\'s recipe sweeps it with '
                    'scripts/native_sweep.py.', 'signal / capsule origin'))
    masked = [site for (st, site) in cap['sites'] if st == 'MASKED']
    if masked and not cap['captured']:
        imp.append(('No capsule, but the hook saw MASKED non-finite sites: start N2/N3 from those ops (%s).'
                    % ', '.join('`%s`' % m[:60] for m in masked[:3]), 'capture census'))

    # ---------------------------------------------------------------- contention
    w('\n## Contention: the capture\'s load beside every replay\'s\n')
    cpg = cap['cpg'] or next((int(m.group(1)) for t in text
                              for m in [re.search(r'CONTAINERS_PER_GPU=(\d+)|containers/GPU=(\d+)', t)]
                              if m and m.group(1)), None) or 1
    cap_others = (max(cap['procs']) * cpg - 1) if cap['procs'] else None
    held_med = statistics.median(cap['held']) if cap['held'] else None
    w('LIGHTER = fewer other processes than the capture%s. Memory is shown, not judged: '
      'a co-tenant far smaller than the capture has reproduced (a training step\'s kernel mix, not its '
      'footprint, was the trigger once).\n' % ('' if wl == 'inference' else ', or no backward pass'))
    w('| load | other processes on the GPU | what they run | memory they hold | result |')
    w('|---|---|---|---|---|')
    if cap['procs']:
        w('| **capture** (the model) | %s | %s | %s | fails |' % (
            '%d (%d process(es)/container x %d container(s)/GPU, minus itself)' % (cap_others, max(cap['procs']), cpg),
            step,
            '%.1f-%.1f GB (median %.1f)' % (min(cap['held']), max(cap['held']), held_med) if held_med else '?'))
    for t in atts:
        w('| 5b %s `%s` | %s copies of the capsule | the captured op%s, nothing else | ~1 GB each | %s/%s %s |' % (
            t['group'], os.path.basename(t['rel']), int(t['conc']) - 1 if t['conc'].isdigit() else '?',
            ' chain' if t['walk'] else '', t['hits'], t['runs'], t['result']))
    for c in colls:
        w('| 5b %s `%s` | none: one torchrun over the member ranks %s | the collective `%s` alone%s | - | %s/%s %s |' % (
            c['group'], os.path.basename(c['rel']), c['ranks'], c['op'],
            ' (path %s)' % c['path'] if c['path'] else '', c['hits'], c['runs'], c['result']))
    lighter = []
    for r in sweeps:
        light = []
        if cap_others and r['procs'] is not None and r['procs'] < cap_others:
            light.append('%d vs %d processes' % (r['procs'], cap_others))
        if r['kind'] in NO_BACKWARD and wl != 'inference':     # inference has no backward pass to miss
            light.append('no backward pass')
        if light and r['hits'] == 0:
            lighter.append(light)
        w('| 5d `%s` | %s | %s | %s GB | %d/%d%s |' % (
            r['tool'], r['procs'] if r['procs'] is not None else '?', r['kind'],
            '%.1f' % r['gb'] if r['gb'] is not None else '?', r['hits'], r['runs'],
            ' **LIGHTER: %s**' % '; '.join(light) if light else ''))
    clean = [r for r in sweeps if r['hits'] == 0]
    if clean and len(lighter) == len(clean):
        imp.append(('Every clean co-tenant load was LIGHTER than the capture\'s (%s): no clean load had as many '
                    'neighbours running %s as the capture did, so its zeros do not clear the load. '
                    'Start from at least the capture\'s number of neighbours per GPU, each running %s '
                    '(the model\'s step, or a stand-in with its op types and sizes); reduce the kind of load after.'
                    % ('; '.join(sorted({x for l in lighter for x in l}))[:220],
                       'the model\'s step' if wl == 'inference' else 'a backward pass',
                       'the model\'s step' if wl == 'inference' else 'forward + backward'), 'Contention table'))

    # ---------------------------------------------------------------- compiled code (C1)
    # C1's Inductor assert and its kernel capsules: on the compiled track the decisive evidence, which
    # this file did not mention (2026-11-20, vLLM and torchcompile: "(no 5b attempt on disk)" only)
    kcaps = sorted(glob.glob(os.path.join(A, 'capsules_inductor', '*.pt')))
    ocode = sorted(glob.glob(os.path.join(A, 'output_code.c1*.log')) +
                   glob.glob(os.path.join(A, 'wave_c0c1', 'output_code.*.log')))
    if kcaps or ocode:
        w('\n## Compiled code (C1: Inductor NaN asserts)\n')
        kres = [l.strip() for f in sorted(glob.glob(os.path.join(A, 'replay', 'kernel', 'run.*.out')))
                for l in read(f).splitlines() if l.startswith('INDUCTOR_REPLAY_RESULT=')]
        for f in kcaps:
            w('* kernel capsule: `%s`' % os.path.relpath(f, A))
        for f in ocode:
            w('* output code: `%s`' % os.path.relpath(f, A))
        if kres:
            w('* kernel capsule replayed from the package by the capture job: `%s`' % kres[-1])
        if os.path.exists(os.path.join(A, 'repro-kernel.tar.gz')):
            w('* package: `repro-kernel.tar.gz` (source graph: `repro-kernel/source_graph.txt`)')
        if kres and re.match(r'INDUCTOR_REPLAY_RESULT=[1-9]', kres[-1]):
            imp.insert(0, ('C1 saved the kernel that made the NaN and it replays without the model (%s): the '
                           'reproducer exists (repro-kernel). What is open is the verdict: run its source nodes '
                           '(source_graph.txt) eagerly and in fp64 on the capsule\'s inputs (step5n-compiled.md '
                           '*Kernel capsule*) -- fp64 non-finite too = NATIVE_MATH, finite = the kernel.'
                           % kres[-1].split()[0], 'replay/kernel'))
        elif kcaps:
            imp.insert(0, ('C1 saved a kernel capsule (%s): replay it first (step5n-compiled.md *Kernel capsule*) '
                           '-- hits with finite inputs make it the reproducer, before any hand-written candidate.'
                           % os.path.basename(kcaps[-1]), 'capsules_inductor/'))

    # ---------------------------------------------------------------- attempts
    w('\n## Attempts (capsule replays)\n')
    if atts:
        w('| group | attempt | host | copies/GPU x GPUs | bad / runs (non-finite or wrong value) | result | first op to go wrong | walk verdict |')
        w('|---|---|---|---|---|---|---|---|')
        for t in atts:
            w('| %s | `%s` | %s | %s x %d | %s | %s | %s | %s |' % (
                t['group'], t['rel'], t['host'], t['conc'], t['gpus'],
                '%s/%s' % (t['hits'], t['runs']) if t['runs'] is not None else '?', t['result'] or '?',
                t['first_wrong'].replace('-> first op to go wrong: ', '') or '-', t['walk'] or '-'))
        for t in atts:
            extra = [x for x in (t['futility'], t['tally'], t['terminal'], t['env']) if x]
            if extra:
                w('\n`%s`:\n\n' % t['rel'] + '\n'.join('    ' + x[:220] for x in extra))
    if colls:
        w('\nCOLLECTIVE groups (one torchrun over the member ranks replays the recorded collective):\n')
        w('| group | replay | host | ranks | bad / runs | result | check on the recorded inputs | path |')
        w('|---|---|---|---|---|---|---|---|')
        for c in colls:
            w('| %s | `%s` | %s | %s | %s | %s | %s | %s |' % (
                c['group'], c['rel'], c['host'], c['ranks'] or '?',
                '%s/%s' % (c['hits'], c['runs']) if c['runs'] is not None else '?', c['result'] or '?',
                c['verdict'] or '-', c['path'] or '-'))
    if not atts and not colls:
        w('(no 5b attempt on disk)')

    # ---------------------------------------------------------------- ruled out
    w('\n## Ruled out (clean experiments; a candidate must differ, or run past the bound)\n')
    w('"excludes" = the smallest per-run rate a 0/N null rules out at 95% (3/N). A rarer trigger is NOT '
      'ruled out.\n')
    ruled = []
    for t in atts:
        if t['hits'] == 0 and t['runs']:
            ruled.append('capsule %s replay (%s), copies only, %s x %d GPUs, 0/%d -> excludes >= %s (%s)'
                         % (t['group'], 'chain' if t['walk'] else 'op', t['conc'], t['gpus'], t['runs'],
                            fmt_rate(excl(t['runs'])), t['result'] or '?'))
    for c in colls:
        if c['hits'] == 0 and c['runs']:
            ruled.append('collective %s replay (`%s` on the recorded inputs, one torchrun over ranks %s%s), 0/%d '
                         '-> excludes >= %s (%s, job %s)'
                         % (c['group'], c['op'], c['ranks'] or '?', ', path ' + c['path'] if c['path'] else '',
                            c['runs'], fmt_rate(excl(c['runs'])), c['result'] or '?', c['job']))
    for r in clean:
        ruled.append('capsule op beside %s x `%s` (%s), 0/%d -> excludes >= %s (job %s)'
                     % (r['procs'] if r['procs'] is not None else '?', r['tool'], r['kind'], r['runs'],
                        fmt_rate(excl(r['runs'])), r['job']))
    w('\n'.join('* ' + r for r in ruled) or '* (nothing ruled out on disk)')
    if clean:
        imp.append(('Each null bounds a rate, not a load: 0/1M excludes only >= 3e-6 per run, and a load one step '
                    'lighter than a reproducing one can fail 10-100x less often. Rerun a near-miss load (e.g. '
                    'the heaviest clean one) longer before treating it as clean.', 'Ruled out'))

    # ---------------------------------------------------------------- implications
    if any(t['result'] == 'NOT_SET_UP' for t in atts):
        t = next(t for t in atts if t['result'] == 'NOT_SET_UP')
        imp.insert(0, ('The capsule\'s exact values, op and copies-only load never fail where the model needed far '
                       'fewer runs (NOT_SET_UP): the missing ingredient is the load, the kernel mix or state around '
                       'the op -- not the values. A copies-only single-op candidate repeats this experiment; start '
                       'from the op beside the model\'s step (top-down, or bottom-up beside a stand-in '
                       'for that step).', t['rel']))
    for t in atts:
        up = [d for d in t['diverged'] if d.split()[0] != t['terminal_idx']]
        if up:
            imp.append(('An op UPSTREAM of the captured op DIVERGED from the record on identical inputs (%s): the '
                        'fault may start there -- include it and its producer in the first candidate.'
                        % re.sub(r'\s+', ' ', up[0])[:80], t['rel']))
            break
        if t['diverged']:
            imp.append(('The captured op itself DIVERGED from its record on identical inputs (%s): no upstream '
                        'suspect, but the fault also corrupts finite values -- compare every hit against fp64, '
                        'not only isfinite.' % re.sub(r'\s+', ' ', t['diverged'][0])[:90], t['rel']))
            break
    for t in atts:
        nv = [m for m in t['mutated'] if not re.search(r'\bview\b', m)]
        if nv:
            imp.append(('An operand was MUTATED AFTER PRODUCTION by a non-view op (%s): the consumer may have read '
                        'a value the capsule did not store -- keep the in-place writes around the op.'
                        % re.sub(r'\s+', ' ', nv[0])[:80], t['rel']))
            break
    if any(t['result'] == 'PROPAGATED' or 'PROPAGATED' in t['walk'] for t in atts):
        imp.append(('PROPAGATED: the capsule\'s op was handed non-finite input -- move the suspect to its producer '
                    '(the chain names it).', 'summary.txt'))
    if grp and all(re.search(r'FALSE_POSITIVE|PROPAGATED', l) for l in grp.splitlines()):
        imp.append(('Every group is an allocator/view false positive or PROPAGATED: the op that first went bad was '
                    'not captured -- upstream, or not watched (op filter). A PROPAGATED chain names its producer.',
                    'groups_priority.txt'))
    if any('KERNEL_MISMATCH' in t for t in text):
        imp.append(('The replay ran a different hipBLASLt solution than the model: pin the model\'s solution in '
                    'every candidate.', 'ledger KERNEL_MISMATCH'))
    hit = [r for r in sweeps if r['hits']]
    if hit:
        imp.insert(0, ('A co-tenant DID reproduce the capsule (%s): start from that load, not from copies.'
                    % '; '.join('%d/%d beside %s x %s' % (r['hits'], r['runs'], r['procs'], r['kind'])
                                for r in hit[:3]), 'pressure sweep'))
    if any(t['tally'] for t in atts):
        t = next(t for t in atts if t['tally'])
        imp.append(('The replay loop named the first op to go non-finite (%s): that op is the bottom-up suspect.'
                    % t['tally'][:120], t['rel']))
    if any(t['hits'] == 0 for t in atts) and not sweeps:
        imp.append(('Copies-only replays were clean and no pressure sweep ran: the load is unexplored -- '
                    'step5-pressure.md before or as the first candidates.', 'attempts'))
    for c in colls:
        if c['hits']:
            imp.insert(0, ('Collective %s REPRODUCED on its recorded inputs (%d/%d, %s): that replay is a reproducer '
                           '(repro-%s) -- 5N needs a reason beyond it (the trigger above).'
                           % (c['group'], c['hits'], c['runs'], c['verdict'] or '?', c['group']), c['rel']))
            break
    for c in colls:
        if c['hits'] == 0 and c['runs']:
            imp.append(('Collective %s replayed clean on its recorded inputs (%d/%d%s)%s: the inputs alone do '
                        'not make it fail -- what differs is the message size / dtype (they pick the '
                        'implementation path), the env, the traffic around it, or the captured event is not the '
                        'model\'s failure (compare its step with the model failure step in Facts). A candidate '
                        'that only repeats this collective on these inputs repeats this experiment; write it at '
                        'the failing step\'s shapes and env, RANKS=<world size> (step5n-native.md *World size > 1*).'
                        % (c['group'], c['hits'], c['runs'], ', path ' + c['path'] if c['path'] else '',
                           ' although the recorded output was wrong (%s)' % c['verdict'] if c['verdict'] else ''),
                        c['rel']))
            break
    if pts and not atts and not colls:
        imp.append(('%d capsule(s) exist but none was replayed: the captured op, its exact views/strides and '
                    'precision (capsule_info.py) are the bottom-up suspect; the capture\'s contention is the load '
                    'to start from.' % len(pts), 'capsules/'))
    w('\n## Implications (PROPOSED -- confirm or strike each, then ledger the kept ones)\n')
    w('\n'.join('%d. %s  \n   _source: %s_' % (i + 1, p_, s_) for i, (p_, s_) in enumerate(imp))
      or '(none proposed -- read the attempts yourself)')

    # ---------------------------------------------------------------- reusable
    w('\n## Reusable\n')
    reuse = []
    for l in grp.splitlines() if grp else []:
        f = l.split()
        if len(f) >= 3 and f[0].startswith('G'):
            reuse.append('capsule %s (%s): `%s`' % (f[0], f[3] if len(f) > 3 else '?', f[2]))
    if pts and not grp:
        reuse.append('capsule (ungrouped): `%s`' % os.path.relpath(pts[0], A))
    for f in sorted(glob.glob(os.path.join(A, 'pressure', '*variants*'))):
        reuse.append('co-tenant variants: `%s`' % os.path.relpath(f, A))
    hb = [f for f in sorted(glob.glob(os.path.join(A, '**', '*hipblaslt*.log'), recursive=True))
          if '/native/' not in f][:1]
    reuse += ['model hipBLASLt log: `%s`' % os.path.relpath(f, A) for f in hb]
    w('\n'.join('* ' + r for r in reuse) or '* (none)')

    path = a.out or os.path.join(A, 'native', 'handoff.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write('\n'.join(out) + '\n')
    print('HANDOFF %s (%d attempts, %d sweep rows, %d ruled out, %d implications, %d lines)'
          % (path, len(atts), len(sweeps), len(ruled), len(imp), len(out)))


if __name__ == '__main__':
    main()
