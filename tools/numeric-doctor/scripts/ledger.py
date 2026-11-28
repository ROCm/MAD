#!/usr/bin/env python3
"""Append to a debug-numeric <LEDGER> and do its timing arithmetic (SKILL.md rule 7).

    L=<ARTIFACT_DIR>/ledger.md
    python3 scripts/ledger.py $L start step4
    python3 scripts/ledger.py $L note  step4 rate=3/8 (host, image, ...)
    python3 scripts/ledger.py $L job   step4 370877 capture wave, 16 trials
    python3 scripts/ledger.py $L wait  step4 start confirm wave geometry
    python3 scripts/ledger.py $L wait  step4 end
    python3 scripts/ledger.py $L end   step4          # prints the one-liner for the user
    python3 scripts/ledger.py $L skip  step2 the user chose to skip it   # a step never run: no visit
    python3 scripts/ledger.py $L table                # step-9 per-step table
    python3 scripts/ledger.py $L status               # (re)write <ARTIFACT_DIR>/status.json, print its line

Every line is `YYYY-MM-DD HH:MM:SS  <step>  <text>` (`date '+%F %T'`).  `end`
finds the step's latest `start`, sums the `waiting-user` pairs in between, and
splits the visit into queue / compute / waiting-user / other from `sacct` for every
`job=<id>` recorded in the visit (compute = union of the jobs' run intervals, so
overlapping jobs count once).  Without sacct (docker) queue and compute print n/a
and stay in `other`.  Steps 5b-8 are tagged per group: `step6[G1]`.
"""
import collections
import datetime as dt
import os
import re
import subprocess
import sys

FMT = '%Y-%m-%d %H:%M:%S'
LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+(\S+)\s+(.*)$')


def now():
    return dt.datetime.now().replace(microsecond=0)


def hms(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return ('%dh%02dm%02ds' % (h, m, s)) if h else ('%dm%02ds' % (m, s))


def unhms(txt):
    m = re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?', txt)
    if not m or not txt:
        return None
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)


def lines(path):
    out = []
    if not os.path.exists(path):        # a new ledger: the first `start` creates it
        return out
    with open(path) as f:
        for raw in f:
            m = LINE.match(raw.rstrip('\n'))
            if m:
                out.append((dt.datetime.strptime(m.group(1), FMT), m.group(2), m.group(3)))
    return out


def append(path, step, text):
    ln = '%s  %s  %s' % (now().strftime(FMT), step, text)
    with open(path, 'a') as f:
        f.write(ln + '\n')
    print('ledgered: ' + ln)        # a confirmation, not a second copy of the ledger line
    try:
        write_status(path)
    except Exception as exc:        # the status file is a view: it must never cost a ledger line
        print('ledger.py: status.json not written: %s' % exc, file=sys.stderr)


# status.json -- the investigation's state as data, beside the ledger, rewritten on every ledger
# line (any writer goes through append()).  Read by the numeric-doctor panel / status line mod;
# its schema is owned here, so a reader never parses ledger wording.  The milestone lines it keys
# on are the ones the skill already requires: TARBALL, LADDER_RESULT=, HANDOVER announced,
# ROOT_CAUSE= / NO_ROOT_CAUSE, FIX=.
STATUS_SCHEMA = 1
SESSIONS_TMP = '/tmp/numeric-doctor-sessions'


def _first(rx, entries, steps=None):
    for t, s, x in entries:
        m = re.search(rx, x)
        if m and (steps is None or re.match(steps, s)):
            return t, s, m
    return None


NOT_VALIDATED = re.compile(r'(?i)not[- ]validated|not[- ]tried|not[- ]built|checked[- ]by[- ]equivalence|'
                           r'\bFAILED\b|\brefuted\b')


def recommended_fix(fixes):
    """step8-fix.md *Decide*: the one the agent marked recommended, else the validated real fix,
    else the narrowest validated workaround (T before B) -- not the first id in sort order
    (2026-11-28: the status line showed B1, and a FAILED B1, ahead of the recommended fix)."""
    marked = [f for f in fixes if re.search(r'(?i)\brecommended\b', f['text']) and f['validated']]
    if marked:
        return marked[0]
    for kind in 'RTB':
        ok = sorted((f for f in fixes if f['id'][0] == kind and f['validated']), key=lambda f: f['id'])
        if ok:
            return ok[0]
    return None


def status(path):
    entries = lines(path)
    art = os.path.dirname(os.path.abspath(path))
    rev = list(reversed(entries))
    st = {'schema': STATUS_SCHEMA, 'run': os.path.basename(art), 'artifact_dir': art,
          'ledger': os.path.abspath(path), 'updated': entries[-1][0].strftime(FMT) if entries else None,
          'run_script': None, 'state': 'running', 'current': None, 'steps': [],
          'reproducer': None, 'root_cause': None, 'fix': None}
    hit = _first(r'RUN_SCRIPT=(\S+)', entries)
    if hit:
        st['run_script'] = hit[2].group(1)
    # steps: open = a start with no end/skip after it; done = their `end total=` / `skipped`
    open_ = collections.OrderedDict()
    for t, s, x in entries:
        if x.startswith('start'):
            open_[s] = t
        elif x.startswith('end ') or x.startswith('end\t') or x.startswith('skipped'):
            open_.pop(s, None)
            m = re.search(r'total=(\S+)', x)
            st['steps'] = [r for r in st['steps'] if r['step'] != s]
            st['steps'].append({'step': s, 'total': m.group(1)} if m and not x.startswith('skipped')
                               else {'step': s, 'skipped': x.split(None, 1)[1] if ' ' in x else ''})
    if open_:
        s, t0 = list(open_.items())[-1]
        job = _first(r'^job=(\d+)\s+(.*)', [e for e in rev if e[1] == s and e[0] >= t0])
        st['current'] = {'step': s, 'since': t0.strftime(FMT),
                         'job': ('%s %s' % (job[2].group(1), job[2].group(2))).strip() if job else None}
    s9 = [x for t, s, x in entries if s == 'step9']
    if any(x.startswith('end') for x in s9):
        st['state'] = 'done'
    elif any(x.startswith('skipped') for x in s9):
        st['state'] = 'stopped'
    # reproducer: the latest repro tarball (step6 / step6[G] / step6[native|kernel])
    tb = _first(r'TARBALL\s+([0-9a-f]{64})\s+(\S+)\s+size=(\d+)\s*(.*)', rev, r'step6')
    if tb:
        t, s, m = tb
        p = m.group(2)
        p = p if os.path.isabs(p) else os.path.join(os.path.dirname(art), p)
        rp = {'tarball': p, 'sha256': m.group(1), 'size': int(m.group(3)), 'at': t.strftime(FMT),
              'group': (re.search(r'\[(.*)\]', s) or [None, None])[1],
              'verdict': None,
              'rate': None, 'host': None, 'handover_at': None}
        pv = _first(r'TARBALL\s.*proposed=(\S+)', rev, r'step6')    # a repack's line drops it
        rp['verdict'] = pv[2].group(1) if pv else None
        lr = (_first(r'LADDER_RESULT=(\S+.*?)(?:\s{2,}|$)', [e for e in rev if 'host=' in e[2]], r'step5b|step6')
              or _first(r'LADDER_RESULT=(\S+.*?)(?:\s{2,}|$)', rev, r'step5b|step6'))
        if lr:
            rp['rate'] = lr[2].group(1).strip()
            h = re.search(r'host=(\S+)', lr[2].string)
            rp['host'] = h.group(1) if h else None
        ho = _first(r'HANDOVER announced', rev, r'step6')
        if ho:
            rp['handover_at'] = ho[0].strftime(FMT)
        st['reproducer'] = rp
    rc = _first(r'(NO_ROOT_CAUSE|ROOT_CAUSE=(\S+?)[:;]?\s)(.*)', rev, r'step7')
    if rc:
        t, s, m = rc
        st['root_cause'] = {'status': 'NO_ROOT_CAUSE' if m.group(1) == 'NO_ROOT_CAUSE' else 'FOUND',
                            'text': m.group(3).strip(), 'at': t.strftime(FMT)}
    fx = _first(r'FIX=(.*)', rev, r'step8')
    if fx:
        t, s, m = fx
        ftb = _first(r'TARBALL\s+([0-9a-f]{64})\s+(\S+)', rev, r'step8')
        # one `FIX=<id> <kind> <status>` per fix shipped (step8-fix.md: R real fix, T targeted /
        # B broad workaround, D defence in depth), one per line or several `;`-joined on one; the
        # latest entry per id wins. Older ledgers: one FIX=<text> line.
        fixes, seen = [], set()
        for _, s2, x in rev:
            if not re.match(r'step8', s2):
                continue
            for m2 in re.finditer(r'FIX=([RTBD]\d+)\s+(.*?)(?=;\s*FIX=|$)', x):
                if m2.group(1) not in seen:
                    seen.add(m2.group(1))
                    fixes.append({'id': m2.group(1), 'text': m2.group(2).strip().rstrip(';')})
        for f in fixes:
            f['validated'] = bool(re.search(r'\bvalidated', f['text'])) and not NOT_VALIDATED.search(f['text'])
        rec = recommended_fix(fixes)
        fixes.sort(key=lambda f: (f is not rec, 'RTBD'.index(f['id'][0]), f['id']))
        real = [f for f in fixes if f['id'][0] == 'R' and f['validated']]
        kind = ('NO_FIX' if m.group(1).startswith('NO_FIX') else
                ('FIX' if real else 'WORKAROUND') if fixes else
                'WORKAROUND' if m.group(1).startswith('WORKAROUND') else 'FIX')
        st['fix'] = {'text': '; '.join('%s %s' % (f['id'], f['text']) for f in fixes) or m.group(1).strip(),
                     'at': t.strftime(FMT), 'kind': kind, 'fixes': fixes,
                     'recommended': rec['id'] if rec else None,
                     'tarball': ftb[2].group(2) if ftb else None}
    st['line'] = status_line(st)
    return st


def status_line(st):
    """One line for a status bar: the milestones so far and what runs now."""
    parts = []
    rp = st['reproducer']
    if rp:
        n = re.search(r'\d+/\d+', rp['rate'] or '')
        parts.append('repro %s%s' % (os.path.basename(rp['tarball']), ' %s' % n.group(0) if n else ''))
    if st['root_cause']:
        parts.append('root cause ' + ('found' if st['root_cause']['status'] == 'FOUND' else 'none'))
    if st['fix']:
        parts.append('fix: ' + st['fix']['text'].split(';')[0][:60])
    if st['state'] != 'running':
        parts.append(st['state'])
    elif st['current']:
        cur = st['current']
        parts.append('%s%s' % (cur['step'], ' job ' + cur['job'] if cur['job'] else ''))
    return '%s: %s' % (st['run'], ' | '.join(parts) or 'starting')


def write_status(path):
    import json
    st = status(path)
    out = os.path.join(st['artifact_dir'], 'status.json')
    tmp = out + '.tmp.%d' % os.getpid()
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, out)
    # which investigation this Claude Code session runs: the panel looks it up by session id
    sid = os.environ.get('CLAUDE_CODE_SESSION_ID')
    if sid and re.fullmatch(r'[\w-]{8,64}', sid):
        for d in (os.path.join(os.path.dirname(st['artifact_dir']), '.sessions'), SESSIONS_TMP):
            try:
                os.makedirs(d, exist_ok=True)
                if d == SESSIONS_TMP:
                    os.chmod(d, 0o1777)
                with open(os.path.join(d, sid), 'w') as f:
                    f.write(st['artifact_dir'] + '\n')
            except OSError:
                pass
    return out


def sacct(jobs):
    """{job: (submit, start, end)} or None if sacct is unavailable."""
    try:
        out = subprocess.run(['sacct', '-X', '-n', '-P', '-o', 'JobID,Submit,Start,End', '-j', ','.join(jobs)],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    res = {}
    for row in out.splitlines():
        f = row.split('|')
        if len(f) != 4:
            continue
        t = []
        for x in f[1:]:
            try:
                t.append(dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
            except ValueError:                   # Unknown: not started / still running
                t.append(None)
        res[f[0]] = t
    return res


def union(ivs, lo, hi):
    """Seconds covered by the intervals, clipped to [lo, hi], overlaps once."""
    tot, cur = 0.0, None
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in ivs):
        if b <= a:
            continue
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur:
                tot += (cur[1] - cur[0]).total_seconds()
            cur = (a, b)
    if cur:
        tot += (cur[1] - cur[0]).total_seconds()
    return tot


def visit(entries, step):
    """Entries of the step's latest visit (from its last `start`)."""
    idx = [i for i, (_, s, txt) in enumerate(entries) if s == step and txt.split()[:1] == ['start']]
    if not idx:
        sys.exit('no `%s start` line in the ledger' % step)
    return entries[idx[-1]:]


def end(path, step):
    entries = lines(path)
    v = visit(entries, step)
    t0, t1 = v[0][0], now()
    total = (t1 - t0).total_seconds()
    wait, wstart = 0.0, None
    jobs = []
    for ts, s, txt in v:
        if s != step:
            continue
        if txt.startswith('waiting-user start'):
            wstart = ts
        elif txt.startswith('waiting-user end') and wstart:
            wait += (ts - wstart).total_seconds()
            wstart = None
        jobs += re.findall(r'\bjob=(\d+)', txt)
    jobs = list(dict.fromkeys(jobs))
    # A bare `step7` visit that wraps per-group `step7[G1]` visits: those report their
    # own time on their own rows, so this one keeps only what lies outside them (the
    # entry questions) -- otherwise the table counts the group loop twice.
    nested = [(m, unhms(m.group(1))) for ts, s, txt in v if '[' not in step
              and s.startswith(step + '[') and txt.startswith('end')
              for m in [re.search(r'\btotal=(\S+)', txt)] if m]
    inner = sum(sec or 0 for _, sec in nested)
    total = max(total - inner, 0)
    parts = ['total=%s' % hms(total)]
    queue = compute = None
    notes = ['excludes %s of nested group visit(s)' % hms(inner)] if nested else []
    if jobs:
        acct = sacct(jobs)
        if acct is None:
            notes.append('queue/compute n/a (no sacct)')
        else:
            qiv, civ, running = [], [], []
            for j in jobs:
                t = acct.get(j)
                if not t or t[0] is None:
                    notes.append('job %s unknown' % j)
                    continue
                sub, st, en = t
                if st is None:                   # never started: queued up to its cancel, or now
                    qiv.append((sub, en or t1))
                    if en:
                        notes.append('job %s never ran' % j)
                    continue
                qiv.append((sub, st))
                civ.append((st, en or t1))       # en None = still running: up to now
                if en is None:
                    running.append(j)
            # only what falls inside this visit counts, overlaps once -- a job that
            # spans several steps (the capture-to-reproducer job) is split by step.
            # Queue = waiting with NONE of the visit's jobs computing: a job held or
            # pinned to a busy node while others ran is not time lost to the queue
            # (2026-11-14: a held job and stopped pinned ones put 22 min of queue on a
            # step whose other jobs ran throughout).
            compute = union(civ, t0, t1)
            queue = max(union(qiv + civ, t0, t1) - compute, 0)
            if running:
                # the image build, submitted in step 1 and still building when step 1 ended:
                # its next 10 min fell in no step's row (table 26m vs 39m wall, 2026-11-20)
                notes.append('job(s) %s still running' % ','.join(running))
                print('WARNING job(s) %s still running: the rest of their compute is in no step\'s row '
                      'unless the step that waits on them records them: ledger.py %s job <that step> <id>'
                      % (','.join(running), path))
    parts += ['queue=%s' % (hms(queue) if queue is not None else 'n/a'),
              'compute=%s' % (hms(compute) if compute is not None else 'n/a'),
              'waiting-user=%s' % hms(wait),
              'other=%s' % hms(max(total - (queue or 0) - (compute or 0) - wait, 0)),
              'jobs=%s' % (','.join(jobs) or 'none')]
    append(path, step, 'end  ' + '  '.join(parts) + ('  (%s)' % '; '.join(notes) if notes else ''))
    d = dict(p.split('=', 1) for p in parts)
    print('TELL USER: %s done in %s (queue %s, compute %s, waiting for you %s, other %s).' % (
        re.sub(r'^step(\d+)([a-z]*)', lambda m: 'Step %s%s' % (m.group(1), m.group(2).upper()), step), d['total'], d['queue'], d['compute'], d['waiting-user'], d['other']))


def table(path):
    rows, skipped = collections.OrderedDict(), collections.OrderedDict()
    measured = collections.defaultdict(set)     # step -> the columns some visit measured
    for _, s, txt in lines(path):
        if txt.split()[:1] == ['skipped']:
            skipped[s] = txt.split(None, 1)[1].strip() if len(txt.split()) > 1 else ''
            continue
        if txt.split()[:1] != ['end']:
            continue
        r = rows.setdefault(s, collections.Counter())
        r['visits'] += 1
        for k, val in re.findall(r'\b(total|queue|compute|waiting-user|other)=(\S+)', txt):
            sec = unhms(val)
            if sec is not None:
                r[k] += sec
                measured[s].add(k)
    cols = ['total', 'queue', 'compute', 'waiting-user', 'other']
    # queue / compute come from sacct: with SCHED=local they are unknown, not zero (their time is in
    # `other`), and a 0m00s there read as "no compute" (2026-11-28)
    timed = lambda s, c: c not in ('queue', 'compute') or c in measured[s]          # noqa: E731
    print('| step | visits | total | queue | compute | waiting for user | other |')
    print('|---|---|---|---|---|---|---|')
    tot = collections.Counter()
    for s in sorted(set(rows) | set(skipped), key=step_key):
        if s in rows:
            r = rows[s]
            print('| %s | %d | %s |' % (s, r['visits'], ' | '.join(hms(r[c]) if timed(s, c) else 'n/a'
                                                                  for c in cols)))
            tot.update(r)
        else:
            print('| %s | 0 | *skipped*%s | - | - | - | - |' % (s, ': ' + skipped[s] if skipped[s] else ''))
    out = outside_compute(path)
    if out:
        # a job's compute while no step was open -- the image build that ends between step 1 and step 4
        # (2026-11-23: ~11 min in no row) -- gets its own row and counts in the total
        print('| *outside any step* | - | %s | - | %s | - | - |' % (hms(out[0]), hms(out[0])))
        tot['total'] += out[0]
        tot['compute'] += out[0]
        measured['*'].add('compute')
    any_timed = lambda c: any(timed(s, c) for s in measured)                         # noqa: E731
    print('| **total** | %d | %s |' % (tot['visits'], ' | '.join(hms(tot[c]) if any_timed(c) else 'n/a'
                                                                for c in cols)))
    print('(steps with neither an `end` nor a `skip` line -- still open -- are not listed)')
    ov = overlaps(path)
    if ov:
        print('ledger: WARNING overlapping visits (their times add up twice in the total): %s'
              % '; '.join('%s / %s %s' % o for o in ov[:6]))


def outside_compute(path):
    """(seconds, jobs) of recorded jobs' compute that fell in no step's visit, or None (none, or no sacct)."""
    entries = lines(path)
    open_, ivs, jobs = {}, [], []
    for ts, s, txt in entries:
        w = txt.split()[:1]
        if w == ['start']:
            open_[s] = ts
        elif w == ['end'] and s in open_:
            ivs.append((open_.pop(s), ts))
        jobs += re.findall(r'\bjob=(\d+)', txt)
    jobs = list(dict.fromkeys(jobs))
    if not jobs or not ivs:
        return None
    acct = sacct(jobs)
    if not acct:
        return None
    civ = [(t[1], t[2]) for t in acct.values() if t and t[1] and t[2]]
    if not civ:
        return None
    lo, hi = min(a for a, _ in civ), max(b for _, b in civ)
    merged = []
    for a, b in sorted(ivs):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    outside = union(civ, lo, hi) - sum(union(civ, a, b) for a, b in merged)
    return (outside, jobs) if outside >= 60 else None


def step_key(s):
    """step4 < step5 < step5a < step5b[G1] < step6[G1] < ... (was: first appearance)."""
    m = re.match(r'step(\d+)([a-z]*)(?:\[(.*)\])?', s)
    return (int(m.group(1)), m.group(2), m.group(3) or '') if m else (99, s, '')


def overlaps(path):
    """(step a, step b, 'from-to') for visits of different steps that overlap in time."""
    open_, iv = {}, []
    for ts, s, txt in lines(path):
        w = txt.split()[:1]
        if w == ['start']:
            open_[s] = ts
        elif w == ['end'] and s in open_:
            iv.append((s, open_.pop(s), ts))
    out = []
    for i, (a, a0, a1) in enumerate(iv):
        for b, b0, b1 in iv[i + 1:]:
            if a != b and a0 < b1 and b0 < a1:
                out.append((a, b, '%s-%s' % (max(a0, b0).strftime('%H:%M:%S'), min(a1, b1).strftime('%H:%M:%S'))))
    return out


PER_GROUP = re.compile(r'^step(5b|6|7|8)$')


def warn_untagged(path, step):
    """Steps 5b-8 are timed per group (`step7[G1]`).  A bare visit next to tagged ones
    holds the whole group loop in its `other` column and reads as work in the table."""
    if PER_GROUP.match(step):
        # refused, not warned after the fact: the visit would already be in the ledger
        sys.exit('ledger: %s is a per-group step -- `start %s[G<i>]` (or `[native]`); nothing '
                 'written (SKILL.md rule 7)' % (step, step))
    base = step.split('[')[0]
    if base != step and PER_GROUP.match(base):
        entries = lines(path)
        starts = [i for i, (_, s_, t) in enumerate(entries) if s_ == base and t.split()[:1] == ['start']]
        ends = [i for i, (_, s_, t) in enumerate(entries) if s_ == base and t.split()[:1] == ['end']]
        if starts and (not ends or ends[-1] < starts[-1]):
            print('ledger: WARNING a bare %s visit is still open -- `end %s` it (or it counts '
                  'the whole %s loop twice)' % (base, base, step), file=sys.stderr)


def warn_not_started(path, step, cmd):
    """A note/job for a step with no open visit: its time lands in no row (or the
    previous step's), so the table under-reports it."""
    if not step.startswith('step') or step == 'step0':  # a job's own tag (capture_to_repro); step 0 has no visit
        return
    entries = lines(path)
    starts = [i for i, (_, s_, t) in enumerate(entries) if s_ == step and t.split()[:1] == ['start']]
    ends = [i for i, (_, s_, t) in enumerate(entries) if s_ == step and t.split()[:1] == ['end']]
    # A note on a step that has run (a result read after its job closed the step, a resume note) has
    # no time to lose: only a job, or a note for a step never started, is worth a warning -- the
    # warning on every such note was read as a fault (2026-11-20, three runs)
    if cmd == 'note' and starts:
        return
    # a bare `step7` / `step8` note (the run's ROOT_CAUSE= / FIX= summary) beside per-group visits
    # is timed in those visits' rows (2026-11-28: warned on every such note of a validation round)
    if cmd == 'note' and PER_GROUP.match(step) and any(
            s_.startswith(step + '[') and t.split()[:1] == ['start'] for _, s_, t in entries):
        return
    if not starts or (ends and ends[-1] > starts[-1]):
        print('ledger: WARNING %s for %s, which has no open visit -- `start %s` first, or the '
              'table does not time this work' % (cmd, step, step), file=sys.stderr)


def record_skill_commit(path):
    """The skill version that runs, for the result card (result_card.py reads SKILL_COMMIT=): step 0
    prints it, but half of a validation round's agents never ledgered it (2026-11-28). This script
    ships in the same checkout as the skill, so its HEAD is the skill's."""
    if any('SKILL_COMMIT=' in x for _, _, x in lines(path)):
        return
    try:
        sha = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', '--short',
                              'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        sha = ''
    append(path, 'step1', 'SKILL_COMMIT=%s' % (sha or 'none'))


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    path, cmd, rest = a[0], a[1], a[2:]
    if cmd == 'table':
        return table(path)
    if cmd == 'status':
        print(write_status(path))
        return print(status(path)['line'])
    if not rest:
        sys.exit(__doc__)
    # one spelling per step: SKILL.md says "5N", step5n-native.md logs `step5n`; an agent wrote
    # `step5N` and make_report missed the fallback's lines (2026-11-14)
    step = re.sub(r'^step(\d+)([A-Za-z]+)', lambda m: 'step%s%s' % (m.group(1), m.group(2).lower()), rest[0])
    text = ' '.join(rest[1:])
    if cmd == 'start':
        warn_untagged(path, step)
        append(path, step, ('start  ' + text).rstrip())
        if step == 'step1':
            record_skill_commit(path)
    elif cmd == 'note':
        warn_not_started(path, step, cmd)
        append(path, step, text)
    elif cmd == 'job':
        warn_not_started(path, step, cmd)
        append(path, step, ('job=%s  %s' % (rest[1], ' '.join(rest[2:]))).rstrip())
    elif cmd == 'wait':
        append(path, step, ('waiting-user %s  %s' % (rest[1], ' '.join(rest[2:]))).rstrip())
    elif cmd == 'end':
        end(path, step)
    elif cmd == 'skip':
        append(path, step, ('skipped  ' + text).rstrip())
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
