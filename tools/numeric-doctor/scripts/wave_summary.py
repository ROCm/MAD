#!/usr/bin/env python3
"""Summarise one wave of model trials (debug-numeric steps 2-4) in a few lines.

    python3 scripts/wave_summary.py <log or dir> ... --signal '<NAN_SIGNAL>' \
        [--error '<the model's own error line>'] [--capsules <CAPSULE_DIR>] [--name REGEX]

Reads every trial log of a wave and prints, instead of the logs themselves:
one line per trial (GPU, arm, verdict, last step, NaN step, s/step, flags),
per-arm and per-GPU tallies, and ALERTS for the readings the skill warns about:
a GPU that disagrees with the rest, a trial that lost its co-tenant (step rate
jumps late), an all-ERROR wave whose logs stop together (external kill), OOMs.

Model-independent by design: the NaN signal is REQUIRED (it is <NAN_SIGNAL>), the
model's own error-line regex is optional (--error; e.g. 'Error at step \\d+' for a
loop that swallows exceptions), step lines match any 'step <n>'.  Trial identity
comes from the file name via --name (default `<gpu>_<trial>[.<arm>].*.log`, gpu and trial
`all` for a multi-rank job's one trial and `0-1` for one trial over a GPU list (wave.sh), the
examples' launcher shape) and, failing that, `HIP_VISIBLE_DEVICES=<n>` inside the
log; names that do not match are reported, never silently mis-attributed.  A directory of
`attempt-NN/run.log` (sequential attempts on one GPU, e.g. run_it.sh-style launchers) is
detected: trial = NN, gpu = 0, arm and outcome from `attempts.tsv` beside it, the model's
own log in the tsv's run_dir is read too, and capsules are looked up in `<CAPSULE_DIR>/attempt-NN`.  Capture trials (arm `cap`) are judged
by their capsule directory when --capsules is given, never by the log: the
capture's exception makes the model print its own NaN line.  A multi-rank job writes
rank<N>_*.pt beside manifest.json at the root of <CAPSULE_DIR> and has no per-trial dir:
its one trial is credited with those.

Verdicts: NAN (signal seen) | CAPSULE (capsules on disk) | NAN_NO_CAPSULE (a capture
trial whose model printed the signal but whose capture wrote nothing: a capture miss,
counted as a hit for the rate) | CLEAN (ran to the end,
no signal) | ERROR (error line, no signal -- e.g. a step-0 OOM swallowed by the
model) | STOPPED (killed before a verdict by the early stop: capture trials always, controls with
--stopped-early) | DIED (no verdict at all: killed, walltime, still running).
"""
import argparse
import collections
import datetime as dt
import glob
import os
import re

ANSI = re.compile(r'\x1b\[[0-9;]*m')   # torchtitan & co colour their log lines
NAN_STEP = re.compile(r'(?i)nan[_ -]?step[\s:=#]+(\d+)')   # a signal line's own NaN step: nan_step=9
# a model's own summary of how far it got (`RESULT step_reached=299`): the default step pattern's
# \bstep misses it after `_`, so last_step was the last --log-every line (250 for 299, 2026-11-20)
STEP_REACHED = re.compile(r'(?i)\bsteps?[_ ]reached[\s:=#]+(\d+)')
import sys

TS = re.compile(r'(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)')


def parse_ts(line):
    m = TS.search(line[:40])
    if m:
        return dt.datetime.strptime(m.group(1).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    return None


ATTEMPT = re.compile(r'^attempt-(\d+)$')


def attempt_layout(d):
    """{run.log path: row} for a sequential-attempt wave (`<d>/attempt-NN/run.log`, as
    run_it.sh-style launchers write; one GPU, attempts one after another), with the
    launcher's `<d>/attempts.tsv` row (attempt, arm, outcome, exit, ..., run_dir) when present."""
    logs = sorted(glob.glob(os.path.join(d, 'attempt-*', 'run.log')))
    if not logs:
        return {}
    rows = {}
    tsv = os.path.join(d, 'attempts.tsv')
    if os.path.exists(tsv):
        lines = open(tsv).read().splitlines()
        hdr = lines[0].split('\t') if lines else []
        for l in lines[1:]:
            rows[l.split('\t')[0]] = dict(zip(hdr, l.split('\t')))
    return {lg: rows.get(os.path.basename(os.path.dirname(lg)), {}) for lg in logs}


def read_trial(path, a, row=None):
    name = os.path.basename(path)
    m = re.match(a.name, name)
    t = dict(path=path, name=name, gpu=None, trial=None, arm=None,
             nan=False, err=None, err_step=None, oom=False, save_fail=False, tb=False, rc=None,
             steps=[], last_ts=None, capsules=None)
    if m:
        gd = m.groupdict()
        t.update({k: gd.get(k) for k in ('gpu', 'trial', 'arm')})
    extra = []
    if row is not None:                      # sequential-attempt layout: identity from the dir + tsv
        n = ATTEMPT.match(os.path.basename(os.path.dirname(path)))
        t.update(trial=n.group(1) if n else None, gpu='0', arm=row.get('arm') or None,
                 name=os.path.join(os.path.basename(os.path.dirname(path)), name))
        rd = row.get('run_dir') or ''
        if rd and os.path.isdir(rd):         # the harness keeps the model's own log in its run dir
            extra = sorted(glob.glob(os.path.join(rd, 'nodes', '*', '*', 'run.log')))
    sig = re.compile(a.signal) if a.signal else None
    step = re.compile(a.step)
    above = None
    if a.signal_above:                       # NAME=VALUE: a magnitude signal (exploding values)
        name, _, lim = a.signal_above.partition('=')
        above = (re.compile(r'%s\s*[:=]\s*([-+0-9.eE]+|nan|inf|-inf)' % re.escape(name.strip()),
                            re.I), float(lim))
    err = re.compile(a.error) if a.error else None
    for lp in [path] + extra:
      with open(lp, errors='replace') as f:
        for line in f:
              line = ANSI.sub('', line)
              # torch's per-rank prefix (`[rank0]: `, printed by vLLM and torchrun): the line-start
              # matches below (Traceback, Inductor's assert) missed every prefixed line (2026-11-19)
              line = RANK_PREFIX.sub('', line)
              if above is not None:
                  m = above[0].search(line)
                  if m:
                      try:
                          v = float(m.group(1))
                      except ValueError:
                          v = float('nan')
                      if not v <= above[1]:            # above the limit, or non-finite
                          t['nan'] = True
              ts = parse_ts(line)
              if ts:
                  t['last_ts'] = ts
              if t['gpu'] is None:
                  g = re.search(r'HIP_VISIBLE_DEVICES=(\d+)', line)
                  if g:
                      t['gpu'] = g.group(1)
              sb = SUB_TRIAL.search(line)
              if sb:
                  t.setdefault('subs', set()).add(sb.group(1) or sb.group(2))
              if sig is not None and sig.search(line):
                  if sb:
                      t.setdefault('sub_nan', set()).add(sb.group(1) or sb.group(2))
                  if not t['nan'] and t.get('sig_step') is None:
                      # the signal line's own NaN step (`nan_step=9`: \bstep misses it after `_`),
                      # else its step, else a `nan_step=N` the model printed, else the last one seen
                      s = NAN_STEP.search(line) or step.search(line)
                      if s:
                          t['sig_step'] = int(s.group(1))
                      elif t.get('own_nan_step') is not None:
                          t['sig_step'] = t['own_nan_step']
                      else:
                          t['sig_step'] = t['steps'][-1][0] if t['steps'] else None
                          t['sig_step_guess'] = True
                  t['nan'] = True
              # the model's own `nan_step=N` on a non-signal line (`RESULT ... nan_step=46` above
              # the traceback whose `Loss is not finite` is the signal): the last logged step was
              # reported instead, one early (45 for 46, 2026-11-20)
              ns = NAN_STEP.search(line)
              if ns and t.get('own_nan_step') is None:
                  t['own_nan_step'] = int(ns.group(1))
                  if t.get('sig_step_guess'):
                      t['sig_step'], t['sig_step_guess'] = t['own_nan_step'], False
              sr = STEP_REACHED.search(line)
              if sr:
                  t['steps'].append((int(sr.group(1)), ts))
              if t['err'] is None and err is not None:
                  e = err.search(line)
                  if e:
                      t['err'] = line.strip()[e.start():e.start() + 90]
                      n = re.search(r'\d+', e.group(0))
                      t['err_step'] = int(n.group(0)) if n else None
              s = None if '*** STEP ' in line else step.search(line)   # Slurm's step-cancel banner: a job id
              if s:
                  t['steps'].append((int(s.group(1)), ts))
              if 'out of memory' in line or 'OutOfMemoryError' in line:
                  t['oom'] = True
              if 'cannot pickle' in line or 'capsule not picklable' in line:
                  t['save_fail'] = True       # the capture caught a NaN but torch.save failed
              if line.startswith('Traceback'):
                  t['tb'] = True
              # a step-8 fix that failed to load (inductor_override, ...): the trial ran stock and its
              # verdict says nothing about the fix (2026-11-23: five "fix" trials NaN'd unpatched)
              if line.startswith('[fix]') and 'NOT applied' in line:
                  t['fix_unapplied'] = line.strip()[:120]
              # TORCHINDUCTOR_NAN_ASSERTS=1 (step5n-compiled.md C1): the compiled code's own NaN/Inf
              # check firing IS this arm's NaN signal, not a death (scored DIED, rate 0/0, 2026-11-14)
              if INDUCTOR_ASSERT.match(line):
                  t['_iassert'] = True
              elif t.get('_iassert') and line.startswith('AssertionError'):
                  t['inductor_assert'] = True
                  if not t['nan'] and t.get('sig_step') is None:
                      t['sig_step'] = t['steps'][-1][0] if t['steps'] else None
                  t['nan'] = True
              r = re.match(r'TRIAL_RC=(\d+)', line)
              if r:
                  t['rc'] = int(r.group(1))
    if row:                                  # the launcher's own outcome, when the logs are silent
        out, ex = (row.get('outcome') or '').lower(), row.get('exit')
        if out == 'nan' and not t['nan']:
            t['nan'] = True
            t['err'] = t['err'] or 'attempts.tsv: outcome nan'
        elif out == 'pass' and t['rc'] is None:
            t['rc'] = 0
        elif out in ('crash', 'error') and not t['err'] and not t['nan']:
            t['err'] = 'attempts.tsv: outcome %s (exit %s)' % (out, ex)
    # capture arms only: capsules/t<i> left by step 4's capture were credited to the trials of a
    # later wave with the same trial numbers and no capture arm (step 8's A/B, 2026-11-20)
    if a.capsules and t['trial'] is not None and t['arm'] in a.capture_arms:
        d = capsule_dir(a, t['trial'])
        t['capsules'] = len(glob.glob(os.path.join(d, '**', '*.pt'), recursive=True))
        t['capsule_dir_found'] = os.path.isdir(d)
        if not os.path.isdir(d) and os.path.exists(os.path.join(a.capsules, 'manifest.json')):
            # multi-rank job (step4-distributed.md): the job is ONE trial and its ranks write
            # rank<N>_*.pt beside manifest.json at the root -- scored NAN_NO_CAPSULE before.
            t['capsules'] = len(glob.glob(os.path.join(a.capsules, '*.pt')))
    return t


# Per-trial capsule dir names launchers use: the examples' trial_<n>, attempt-<n> (run_it.sh-style),
# t<n> (the docker capture_wave_docker.sbatch), or the bare trial number.
SUBDIR_GUESSES = ('trial_{trial}', 't{trial}', 'attempt-{trial}', '{trial}')


def capsule_dir(a, trial):
    """<CAPSULE_DIR>/<subdir> for one trial; --capsule-subdir auto tries SUBDIR_GUESSES, first existing wins."""
    if a.capsule_subdir != 'auto':
        return os.path.join(a.capsules, a.capsule_subdir.format(trial=trial))
    for g in SUBDIR_GUESSES:
        d = os.path.join(a.capsules, g.format(trial=trial))
        if os.path.isdir(d):
            return d
    return os.path.join(a.capsules, SUBDIR_GUESSES[0].format(trial=trial))


def rate(steps):
    """(s/step overall, ratio of early to late s/step) from (step, ts) pairs."""
    pts = [(s, ts) for s, ts in steps if ts]
    if len(pts) < 4:
        return None, None
    span = lambda p: (p[-1][1] - p[0][1]).total_seconds() / max(p[-1][0] - p[0][0], 1)
    q = len(pts) // 4
    early, late = span(pts[:2 * q]), span(pts[-q:])
    return span(pts), (early / late if late > 0 else None)


def verdict(t, a):
    if t['arm'] in a.capture_arms and t['capsules'] is not None:
        if not t['capsules'] and t['rc'] in (None, 130, 137, 143) and not t['err']:
            return 'STOPPED'                 # killed before capturing (early stop, walltime)
        if t['capsules']:
            return 'CAPSULE'
        if t['nan'] or t['save_fail']:
            return 'NAN_NO_CAPSULE'          # the model (or the capture) saw it; nothing written
        if t['err'] and 'nan' not in t['err'].lower():
            return 'ERROR'
        if t['tb'] or t['rc'] not in (None, 0, 130, 137, 143):
            return 'ERROR'                   # died with a traceback / an error rc: not a clean NONE
        return 'NONE'
    if t['nan']:
        return 'NAN'
    if t['err'] or t['oom']:
        return 'ERROR'
    if t['rc'] == 0 or (t['steps'] and a.last_step is not None and t['steps'][-1][0] >= a.last_step):
        return 'CLEAN'
    if a.stopped_early and t['rc'] in (None, 130, 137, 143):
        return 'STOPPED'                     # a control the early stop killed: no verdict, not a death
    return 'DIED'


# torch's `[rank0]: ` and vLLM V1's worker prefix `(Worker_TP1 pid=123) ` (`(EngineCore pid=..)` too) with its
# logger's `ERROR 11-21 01:34:53 [file.py:383] ` (ERROR only: vLLM logs a benign start-up traceback at WARNING):
# a traceback behind them still scores (2026-11-21, vLLM TP2: an Inductor assert in a worker read DIED)
RANK_PREFIX = re.compile(r'^(?:\[rank\d+\]:|\((?:Worker\w*|EngineCore\w*|MainProcess) pid=\d+\))\s?'
                         r'(?:ERROR \d\d-\d\d \d\d:\d\d:\d\d \[[^\]]*\] ?)?')
INDUCTOR_ASSERT = re.compile(r'^\s+assert not \w+\.is(?:nan|inf)\(\)')   # traceback line, not the logged code
SUB_TRIAL = re.compile(r'^\[trial (\d+)\]|\btrial=(\d+)\b')   # a line's own trial, in a multi-trial log
SIDE_LOG = re.compile(r'^output_code\b.*\.log$')   # TORCH_LOGS=output_code, not a trial


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('paths', nargs='+', help='trial logs, or directories of *.log')
    p.add_argument('--signal', help='<NAN_SIGNAL> regex (the model\'s own NaN line); this or --signal-above')
    p.add_argument('--signal-above', metavar='NAME=VALUE',
                   help='a magnitude signal: a trial counts as NAN when a logged "NAME: <number>" '
                        'exceeds VALUE or is non-finite (e.g. grad_norm=100; ANSI colours stripped)')
    p.add_argument('--error', default=None,
                   help='the model\'s own error-line regex, first number = step (default: none)')
    p.add_argument('--step', default=r'(?i)\bstep[\s:=#]+(\d+)\b',
                   help='step-line regex, group 1 = step (default: "step 7", "step: 7", "step=7")')
    p.add_argument('--name', default=r'^(?P<gpu>\d+(?:-\d+)*|all)_(?P<trial>\d+(?:-\d+)*|all)(?:\.(?P<arm>\d*[A-Za-z][\w-]*))?\.',
                   help='file-name regex with named groups gpu, trial, arm')
    p.add_argument('--capsules', help='<CAPSULE_DIR>; capture trials are judged by it')
    p.add_argument('--capsule-subdir', default='auto',
                   help="per-trial dir under --capsules, e.g. 'trial_{trial}'; auto tries %s" % ', '.join(SUBDIR_GUESSES))
    p.add_argument('--capture-arms', default='cap', type=lambda s: set(s.split(',')))
    p.add_argument('--last-step', type=int, help='final step index; a trial reaching it is CLEAN')
    p.add_argument('--stopped-early', action='store_true',
                   help='the wave was stopped early on purpose: a killed trial of any arm is STOPPED, not DIED')
    p.add_argument('--quiet', action='store_true', help='tallies and alerts only, no per-trial lines')
    a = p.parse_args()
    if not (a.signal or a.signal_above):
        p.error('one of --signal / --signal-above is required')

    files, rows, skipped = [], {}, []
    for x in a.paths:
        lay = attempt_layout(x) if os.path.isdir(x) else {}
        if lay:
            rows.update(lay)
            files += list(lay)
            if a.capsule_subdir == 'auto':
                a.capsule_subdir = 'attempt-{trial}'
        elif os.path.isdir(x):
            # Inductor's own logs sit beside the trials (TORCH_LOGS_OUT=<wave>/output_code.<gpu>.log,
            # step5n-compiled.md): not trials.  Any other unmatched name stays a trial and is flagged.
            for f in sorted(glob.glob(os.path.join(x, '*.log'))):
                (skipped if SIDE_LOG.match(os.path.basename(f)) else files).append(f)
        else:
            files.append(x)
    if not files:
        sys.exit('no logs found')
    if skipped:
        print('(not trials, skipped: %d Inductor log(s), e.g. %s)'
              % (len(skipped), ', '.join(os.path.basename(f) for f in skipped[:3])))
    trials = [read_trial(f, a, rows.get(f)) for f in files]
    key = lambda t: (int(t['trial']) if (t['trial'] or '').isdigit() else 1 << 30, t['name'])
    trials.sort(key=key)

    alerts = []
    unnamed = [t for t in trials if t['trial'] is None]
    if unnamed:
        alerts.append('%d log name(s) do not match --name (e.g. %s): trial/arm unknown, so capsules '
                      'cannot be attributed per trial -- pass --name with (?P<gpu>)(?P<trial>)(?P<arm>) '
                      'groups, or read the capsule count below' % (len(unnamed), unnamed[0]['name']))
        if a.capsules:
            n = len(glob.glob(os.path.join(a.capsules, '**', '*.pt'), recursive=True))
            print('== capsules in %s: %d (not attributable to trials)' % (a.capsules, n))
    by_arm = collections.defaultdict(collections.Counter)
    by_gpu = collections.defaultdict(collections.Counter)
    nan_steps = collections.Counter()
    for t in trials:
        t['v'] = verdict(t, a)
        arm = t['arm'] or '-'
        by_arm[arm][t['v']] += 1
        by_gpu[t['gpu'] or '?'][t['v']] += 1
        t['nan_step'] = t['err_step'] if t['err_step'] is not None else t.get('sig_step')
        if t['v'] == 'NAN' and t['nan_step'] is not None:
            nan_steps[t['nan_step']] += 1
        sps, ratio = rate(t['steps'])
        t['sps'] = sps
        flags = []
        if ratio and ratio > 3:
            flags.append('ORPHAN?(late steps %.0fx faster)' % ratio)
            alerts.append('trial %s: late steps %.0fx faster -- lost its co-tenant? drop it from the '
                          'denominator if so (step2-rate.md)' % (t['trial'], ratio))
        if t['oom']:
            flags.append('OOM')
        if t.get('inductor_assert'):
            flags.append('inductor NaN assert')
        elif t['tb']:
            flags.append('traceback')
        t['flags'] = flags
        if not a.quiet:
            print('trial %-3s gpu %-2s %-4s %-7s last_step=%-5s nan_step=%-5s s/step=%-6s %s%s' % (
                t['trial'] or '?', t['gpu'] or '?', arm, t['v'],
                t['steps'][-1][0] if t['steps'] else '-',
                t['nan_step'] if t['v'] == 'NAN' and t['nan_step'] is not None else '-',
                '%.2f' % sps if sps else '-',
                ' '.join(flags),
                ('  capsules=%d' % t['capsules']) if t['capsules'] else ''))
            if t['v'] == 'ERROR' and t['err']:
                print('    err: %s' % t['err'])

    # NAN is the verdict token other tools parse; say what it means when the signal is no NaN
    # line (2026-11-26: a `RESULT .*wrong=yes` wave read "ctl NAN" for wrong finite values)
    sig = a.signal or ''
    if a.signal_above or not re.search(r'(?i)nan|\binf', sig):
        print('== NAN = the signal fired (%s): no NaN implied' % (
            '--signal-above ' + a.signal_above if a.signal_above else '/%s/' % sig))
    print('== by arm:   ' + '; '.join('%s %s' % (arm, dict(c)) for arm, c in sorted(by_arm.items())))
    for arm, c in sorted(by_arm.items()):
        hit = c['NAN'] + c['CAPSULE'] + c['NAN_NO_CAPSULE']
        n = sum(c.values()) - c['DIED'] - c['ERROR'] - c['STOPPED']
        print('   %-4s rate %d/%d (excl. %d ERROR, %d DIED, %d STOPPED)' % (arm, hit, n, c['ERROR'], c['DIED'], c['STOPPED']))
    print('== by GPU:   ' + '  '.join('%s:%s' % (g, ','.join('%s=%d' % kv for kv in sorted(c.items())))
                                      for g, c in sorted(by_gpu.items())))
    if nan_steps:
        print('== NaN step: ' + ' '.join('%s x%d' % kv for kv in sorted(nan_steps.items())))
    # one log, several trials (a container running mp.spawn workers, each tagging its lines
    # `[trial N]` / `trial=N`): the rows above count logs; this counts the trials in them
    # (capture arms are judged by their capsules, not the NaN line: not tallied here)
    multi = {(t['arm'] or '-') for t in trials if len(t.get('subs') or ()) > 1
             and t['arm'] not in a.capture_arms}
    if multi:
        per = collections.defaultdict(lambda: [0, 0])
        for t in trials:
            if (t['arm'] or '-') in multi:
                per[t['arm'] or '-'][0] += len(t.get('sub_nan') or ())
                per[t['arm'] or '-'][1] += len(t.get('subs') or ()) or 1
        print('== per trial (logs hold several trials each; the rows and rates above count logs): '
              + '  '.join('%s %d/%d' % (k, v[0], v[1]) for k, v in sorted(per.items())))
    sp = sorted(t['sps'] for t in trials if t['sps'])
    if sp:
        print('== s/step:   min %.2f  median %.2f  max %.2f' % (sp[0], sp[len(sp) // 2], sp[-1]))

    # a GPU whose trials all disagree with the rest -- within one arm only: arms
    # differ by design (capture vs control, injected vs not)
    for arm in sorted({t['arm'] or '-' for t in trials}):
        ga = collections.defaultdict(collections.Counter)
        for t in trials:
            if (t['arm'] or '-') == arm and t['v'] not in ('STOPPED', 'DIED', 'ERROR'):
                ga[t['gpu'] or '?'][t['v'] in ('NAN', 'CAPSULE', 'NAN_NO_CAPSULE')] += 1
        if len(ga) > 2 and arm not in multi:     # per-log rows: one hit is not "every trial"
            good = [g for g, c in ga.items() if c[True] and not c[False]]
            bad = [g for g, c in ga.items() if c[False] and not c[True]]
            if good and bad and len(bad) < len(good):
                alerts.append('arm %s, GPU(s) %s: 0 hits while %d GPUs hit every trial -- partition the '
                              'denominator, probe those GPUs alone (step2-rate.md)'
                              % (arm, ','.join(sorted(bad)), len(good)))
    if not a.capsules and any(t['arm'] in a.capture_arms for t in trials):
        alerts.append('capture trials judged from their logs -- pass --capsules <CAPSULE_DIR>; the '
                      'capture\'s own exception makes the model print its NaN line')
    if a.capsules and not os.path.exists(os.path.join(a.capsules, 'manifest.json')):
        caps = [t for t in trials if t['arm'] in a.capture_arms]
        on_disk = len(glob.glob(os.path.join(a.capsules, '**', '*.pt'), recursive=True))
        if caps and on_disk and not any(t.get('capsule_dir_found') for t in caps):
            alerts.append('%d capsule(s) under %s, but no capture trial has a per-trial dir (tried %s) -- the '
                          'verdicts above ignore them: pass --capsule-subdir <pattern with {trial}>'
                          % (on_disk, a.capsules, ', '.join(SUBDIR_GUESSES) if a.capsule_subdir == 'auto'
                             else a.capsule_subdir))
    missed = [t for t in trials if t['v'] == 'NAN_NO_CAPSULE']
    if missed:
        alerts.append('%d capture trial(s) saw the NaN (signal line or a capture save error) but wrote no capsule -- a capture miss, not a '
                      'clean trial: capture_digest.py on one log; `pending:` / `MASKED` lines with no capsule = '
                      'settle mode never fired (NAN_CAPTURE_SETTLE=0, step5-read.md)' % len(missed))
    errs = [t for t in trials if t['v'] in ('ERROR', 'DIED')]
    if errs and len(errs) == len(trials):
        ends = [t['last_ts'] for t in trials if t['last_ts']]
        if ends and (max(ends) - min(ends)).total_seconds() < 30:
            alerts.append('every trial ERROR/DIED and all logs stop within 30 s -- external kill or '
                          'walltime, not the model (waves.md)')
    if any(t['oom'] for t in trials):
        alerts.append('OOM in %d trial(s) -- exclude them; see step4-capture-memory.md if capturing'
                      % sum(t['oom'] for t in trials))
    lost = [t for t in trials if t['save_fail']]
    if lost:
        alerts.append('%d trial(s) hit a capsule save error (cannot pickle / not picklable) -- '
                      'a caught NaN may be lost or saved with leaves replaced; check the '
                      "capsule's `unpicklable` list (capture_digest.py)" % len(lost))
    unfixed = [t for t in trials if t.get('fix_unapplied')]
    if unfixed:
        alerts.append('%d trial(s) say their fix was NOT applied -- they ran stock, their verdicts are void for '
                      'the fix (arm(s) %s): %s' % (len(unfixed), ','.join(sorted({str(t['arm']) for t in unfixed})),
                                                   unfixed[0]['fix_unapplied']))
    for x in alerts:
        print('!! ' + x)


if __name__ == '__main__':
    main()
