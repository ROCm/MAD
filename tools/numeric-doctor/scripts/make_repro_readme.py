#!/usr/bin/env python3
"""Write README.md for a step-6 reproducer, tar it, and print its sha256 (debug-numeric).

    python3 scripts/make_repro_readme.py --repro <ARTIFACT_DIR>/repro-G1 \
        --attempt <ARTIFACT_DIR>/replay/G1/c2.<job>.<host> --groups <ARTIFACT_DIR>/groups.json \
        [--image IMAGE] [--pack] [--ledger <LEDGER> --step 'step6[G1]']

Everything mechanical comes from the artifacts, so nobody transcribes it: the op and its
geometry key, the capsule's name/size/sha256, the `captured build:` lines, the
`env:` verdict, the reproducing rate with copies per GPU and GPU count, the one
command, and the "one copy alone means nothing" warning.

The operand-validation sentence is PROPOSED from the [C1] walk in the attempt's
summary.txt (replay_verdict.propose) and marked `<!-- CHECK -->`: KERNEL_BUG when
the terminal op's fp64 reference on the replay's own operands is finite, or its
reference is finite and no ancestor DIVERGED; KERNEL_BUG_UNVERIFIED when the
terminal op has no reference (`unavailable` / `-`) or an ancestor DIVERGED (trailing
view rows alias the op before them and are not the terminal); a
deterministic walk (terminal non-finite, nothing DIVERGED) is flagged for
step5-replay-guardrail.md.  The agent confirms it (step6-finish.md) -- the script
does not decide verdicts.  The command and environment line follow
the launcher run.sh calls (docker, RUNNER=venv, nd).

--pack writes <repro>.tar.gz beside the directory and prints `sha256 path`;
--ledger appends one line with path, size and sha256.  --repack re-tars an edited
README (no rewrite) and ledgers the new sha256: `--repro <dir> --repack [--ledger ..]`.
The image defaults to $IMAGE, else run.sh's `IMAGE=${IMAGE:-...}`; a kernel.*.txt
beside the attempt (written by the replay ladder) becomes the README's Kernel section.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_verdict import c1_rows, c1_terminal, op_terminal, propose as _propose  # noqa: E402  (one copy of the rules)

CONTENTION = ("**One copy alone reports `0/N` and means nothing.** It reproduced with {conc} copies "
              "per GPU, which are each other's co-tenants; a contention-gated fault needs that "
              "load. Do not close this after a single-process run.")
EVERY_RUN = ("**Every run failed** with {conc} copies per GPU ({rate}): this is not a rare race "
             "waiting for a co-tenant, and one copy is expected to reproduce it too "
             "(`CAPSULE_CONCURRENCY=1 bash run.sh`).")
COTENANT = ("**It was reproduced beside a co-tenant workload.** Each GPU runs one replay beside "
            "`{cot}`, which `run.sh` starts and stops with it -- the device load the model "
            "needed to fail (debug-numeric step5-pressure.md). A run without the co-tenant that "
            "reports `0/N` does not close this.")
SOLO_PROBE = ("It also reproduced with **one copy on one GPU** ({probe} in the ladder's solo probe): "
              "no co-tenant is needed, and a single-process run is a valid test of a fix. The rate "
              "above, at {conc} copies per GPU, is the one to compare against.")
SOLO = ("It reproduced with **one copy per GPU** -- no co-tenant needed at this rate, which "
        "points to a deterministic or load-independent fault rather than a race.")

KB = ("The fp64 CPU reference returns finite from these operands and every upstream op "
      "reproduces its recorded value, so a correct kernel does not produce this NaN.")
KB_TERMINAL = ("On the very operands this replay hands the op, its fp64 CPU reference is finite "
               "and fits the output dtype, so no correct kernel returns a non-finite value "
               "from them -- whatever produced those operands.")
KB_WRONG = ("On the very operands this replay hands the op, its fp64 CPU reference is finite "
            "and fits the output dtype, and the op returns a WRONG FINITE value far from it "
            "(the replay counts runs that repeat that wrong value); a correct kernel does not. "
            "The model's NaN comes later, where the wrong value overflows downstream.")
KB_VALUE = ("The op has a value check (numeric-doctor `math_refs.VALUE_CHECKS`): an fp32 recompute, from "
            "the op's own inputs, of what its consumer reads. On these operands that reference is "
            "finite, and the runs counted below are the ones whose output is off from it by more "
            "than a correct call ever is -- wrong FINITE values, never a NaN. A correct kernel does "
            "not do that, whatever produced the operands.")
KB_RERUN = ("The op is not repeatable: each counted run executes it twice, the second time on "
            "identical copies of its operands (and the same random-number state), and the two "
            "outputs differ by more than rounding noise. A correct deterministic kernel returns the "
            "same bits; this one returns another answer -- a race inside it, whatever the operands are.")
KB_SYNTH = ("On these operands -- synthesized from the capsule's per-tensor statistics, "
            "since the capsule holds no floating-point values -- the fp64 CPU reference is finite and "
            "fits the output dtype, so a correct kernel returns no non-finite value from them. "
            "Upstream ops were not compared with recorded values (there are none), and this "
            "says nothing about the operands the model actually used.")
SYNTH_NOTE = ("\n**The capsule holds no floating-point values** -- shapes, strides, alignment "
              "and a few per-tensor statistics only; integer tensors (sequence offsets, "
              "indices) keep their recorded values. Every replay runs on look-alike float "
              "operands drawn from those statistics (`--synthesize`, which `run.sh` passes); a "
              "fault that needs the real float values will not reproduce from it.\n")
SYNTH_DERIVED = ("\nDerived operands (the attention forward's `out` and `softmax_lse`) are not drawn "
                 "from statistics: the replay recomputes them with the forward from the synthesized "
                 "q/k/v, so they are consistent and a fix can be validated on this package.\n")
SYNTH_INCONSISTENT = ("\n**Derived operands are inconsistent**: `out`/`softmax_lse` were drawn from "
                      "their statistics independently of q/k/v (the forward could not be re-run), and a "
                      "correct kernel can overflow on such operands. This package shows the NaN but "
                      "cannot validate a fix; check a fix on consistent operands "
                      "(`scripts/attn_consistent_check.py` in numeric-doctor).\n")
UNV = ("This is the recorded dependency chain; the terminal op has no CPU reference (or an "
       "upstream op did not reproduce its recorded value), so its operands are reproduced "
       "rather than verified. The op named here is where the NaN first APPEARS, not "
       "necessarily where the fault is: an upstream op that writes finite but wrong values "
       "(a race) hands them on, and the op that then overflows is the one blamed. "
       "`replay_nan_op.py --chain --chain-iters N --contend C --chain-repeat-check` names "
       "the first op whose output stopped repeating in the non-finite runs.")
UNV_OP = ("This replays the failing op alone on its recorded operands; the op has no CPU reference, "
          "so its result is reproduced rather than verified. The op named here is where the NaN first "
          "APPEARS, not necessarily where the fault is: an upstream op that wrote finite but wrong "
          "values would hand them on. The chain capsule, when the capture kept one, names the first "
          "op whose output stopped repeating (`replay_nan_op.py --chain --chain-iters N "
          "--chain-repeat-check`).")


# Written for a REPRODUCING attempt only, so a held-back variable was not needed for this NaN; the
# old wording ("export it, or the replay runs without it") read as a missing step (2026-11-20, sglang).
HELD = ("\n* `{var}` was set in the workload but is held back by the replay's env restore (`env:` "
        "banner `held back:`); the replay reproduced without it (in a container the image's own "
        "value applies), so this NaN does not need it. Export the workload's value in `run.sh` only "
        "if a later experiment depends on it.")


def cotenant_of(val):
    """The co-tenant command, or None: "none"/empty = copies are the only load, so the README
    gets no "the replay alone does not reproduce this" paragraph."""
    return None if (val or '').strip().lower() in ('', 'none', '0', 'off') else val


# Held back on purpose by the replay (device selection: run.sh sets it per copy).
HELD_OK = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
# Library logging the capture turned on (HIPBLASLT_LOG_MASK/FILE on capture GPU 0, ...): not
# something the replay needs, so not a "held back" warning (2026-11-14, qwen_tf32).
HELD_LOGGING = re.compile(r'^(HIPBLASLT_LOG_|ROCBLAS_LOG_|ROCBLAS_LAYER$|AMD_LOG_|TENSILE_DB$|HIP_TRACE_)')


def held_back(logs, repro):
    """Vars a copy log's env banner lists as `held back:` that run.sh does not export."""
    run_sh = os.path.join(repro, 'run.sh')
    txt = open(run_sh).read() if os.path.exists(run_sh) else ''
    out = []
    for lg in logs[:1]:
        for l in open(lg, errors='replace').read().splitlines():
            m = re.match(r'\s*held back:\s*([A-Za-z_][A-Za-z0-9_]*)', l)
            if m and m.group(1) not in out and m.group(1) not in HELD_OK \
                    and not HELD_LOGGING.match(m.group(1)) and not re.search(
                    r'(^|[\s;])(export\s+)?%s=' % re.escape(m.group(1)), txt, re.M):
                out.append(m.group(1))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def propose(rows, terminal=None):
    verdict, why = _propose(rows, terminal)
    kb = KB_RERUN if 'rerun check' in why else KB_TERMINAL if 'same operands' in why else KB
    return verdict, {'KERNEL_BUG': kb, 'KERNEL_BUG_UNVERIFIED': UNV}.get(verdict, ''), why


def runner_of(repro):
    """'venv' | 'docker' | 'nd' | 'nd-native', from the launcher run.sh calls; the README's
    command and caveats follow the launcher actually shipped."""
    txt = open(os.path.join(repro, 'run.sh')).read() if os.path.exists(os.path.join(repro, 'run.sh')) else ''
    if re.search(r'\bRUNNER=venv\b', txt) or 'replay_capsule_node_venv.sh' in txt:
        return 'venv'
    if 'scripts/jobs/replay_ladder.sh' in txt:
        return 'nd-native' if 'RUNTIME=${RUNTIME:-native}' in txt else 'nd'
    return 'docker'


def image_from_run_sh(repro):
    m = re.search(r'IMAGE=\$\{IMAGE:-([^}]+)\}', open(os.path.join(repro, 'run.sh')).read()) \
        if os.path.exists(os.path.join(repro, 'run.sh')) else None
    return m.group(1) if m else None


def pack(repro, ledger, step, verdict):
    tgz = repro.rstrip('/') + '.tar.gz'
    # pigz when there is one: the same .tar.gz, on every core (a 1 GB real-values package took
    # 4 min with gzip and outran the agent's tool timeout on a --repack, 2026-11-21)
    z = ['-I', 'pigz'] if shutil.which('pigz') else ['-z']
    subprocess.check_call(['tar', *z, '-cf', tgz, '-C', os.path.dirname(repro),
                           '--exclude', 'out', os.path.basename(repro)])
    line = '%s  %s  size=%d' % (sha256(tgz), tgz, os.path.getsize(tgz))
    print('SHA256 ' + line)
    if ledger and os.path.exists(ledger):
        subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ledger.py'),
                         ledger, 'note', step, 'TARBALL %s %s'
                         % (line.replace(os.path.dirname(os.path.dirname(tgz)) + '/', ''), verdict)],
                        stdout=subprocess.DEVNULL)


def _site_line(repro):
    """The exact Slurm submit of the site that verified the package (run.sh's `# this site:` line:
    account, partition, GPU request -- some sites reject a job without one), as a second command line."""
    try:
        m = re.search(r'^# this site: (sbatch .*)$', open(os.path.join(repro, 'run.sh')).read(), re.M)
    except OSError:
        return ''
    return ('\n%s   # on Slurm at the site that verified it' % m.group(1).strip()) if m else ''


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--repro', required=True)
    p.add_argument('--attempt', help='the reproducing attempt dir (summary.txt + copy logs); not with --repack')
    p.add_argument('--groups', help='groups.json, for the op and its geometry')
    p.add_argument('--image', default=os.environ.get('IMAGE'),
                   help='the image it reproduced in (default: $IMAGE; required)')
    p.add_argument('--pack', action='store_true')
    p.add_argument('--repack', action='store_true', help='re-tar the edited README; write nothing')
    p.add_argument('--cotenant', default=os.environ.get('CAPSULE_COTENANT'),
                   help='the co-tenant command the reproducer runs beside the replay '
                        '(default $CAPSULE_COTENANT); none = copies are the only load')
    p.add_argument('--mode', default=os.environ.get('CAPSULE_MODE', 'chain'),
                   help='chain or op, as the reproducing ladder ran (default $CAPSULE_MODE)')
    p.add_argument('--chain-package', help='the op-only package of a chain capsule (capsule_slim.py): '
                   'the verified chain package kept beside it, named in the README')
    p.add_argument('--ledger')
    p.add_argument('--step', default='step6')
    p.add_argument('--note', help='--repack: the TARBALL ledger line\'s label (default: from --step and the package)')
    a = p.parse_args()
    a.cotenant = cotenant_of(a.cotenant)
    repro = os.path.abspath(a.repro)
    if a.repack:
        # Label by the step first: a step-8 fix/ has no capsule either, and is not 5N's.
        if a.step.startswith('step8'):
            what = 'fix package (step 8), README by hand'
        elif not glob.glob(os.path.join(repro, 'capsule', '*.pt')):
            what = 'native package (step 5N), README by hand'
        elif '<!-- CHECK' in open(os.path.join(repro, 'README.md')).read():
            what = 'repacked; README CHECK line STILL OPEN (resolve it: step6-finish.md)'
        else:
            what = 'repacked after README edit (CHECK resolved)'
        pack(repro, a.ledger, a.step, a.note or what)
        return
    if not a.attempt:
        sys.exit('--attempt is required (unless --repack)')
    a.image = a.image or image_from_run_sh(repro)
    if not a.image:
        sys.exit('--image (or $IMAGE, or IMAGE=${IMAGE:-...} in run.sh) is required')
    caps = glob.glob(os.path.join(repro, 'capsule', '*.pt'))
    if len(caps) != 1:
        sys.exit('expected exactly one capsule in %s/capsule' % repro)
    cap = caps[0]
    summ = open(os.path.join(a.attempt, 'summary.txt')).read().splitlines()
    tot = next((l for l in summ if l.startswith('=== totals')), '')
    m = re.search(r'non-finite (\d+)/(\d+)', tot)
    bad, runs = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    copies = len([l for l in summ if re.match(r'\s+g\d+_k\d+\.', l)])
    gpus = len({re.match(r'\s+g(\d+)_', l).group(1) for l in summ if re.match(r'\s+g\d+_k\d+\.', l)})
    conc = re.search(r'/c(\d+)(?:det)?\.', a.attempt.rstrip('/') + '/')   # c1det: the ladder's deterministic probe
    conc = int(conc.group(1)) if conc else (copies // max(gpus, 1))
    stopped = next((l for l in summ if l.startswith('=== stopped early')), '')
    envl = next((re.sub(r'^\d+\s+', '', l.strip()) for l in summ if 'env:' in l), 'env: (no banner)')
    envl = re.sub(r'\s+', ' ', envl)
    # captured build: the line plus its indented continuation, from one copy log
    build, key = [], ''
    for lg in sorted(glob.glob(os.path.join(a.attempt, 'g*_k*.log')))[:1]:
        lines = open(lg, errors='replace').read().splitlines()
        for i, l in enumerate(lines):
            if l.startswith(('captured on:', 'captured build:', 'contention:', 'operands:')):
                build.append(l)
                for x in lines[i + 1:i + 8]:     # its indented continuation only
                    if not x.startswith('  ') or x.strip().endswith(':') or ':  ' in x[:18]:
                        break
                    build.append(x)
            if l.startswith('hipBLASLt key:') and not key:
                key = l.split(':', 1)[1].strip()
    held = held_back(sorted(glob.glob(os.path.join(a.attempt, 'g*_k*.log'))), repro)
    op = geom = ''
    if a.groups:
        rep = os.path.basename(cap)
        for g in json.load(open(a.groups))['groups']:
            if os.path.basename(g['representative']) == rep:
                ident = g['identity']
                if isinstance(ident, str):
                    try:
                        ident = ast.literal_eval(ident)
                    except Exception:
                        ident = {}
                op, geom = ident.get('op', ''), ident.get('geometry_key', '')
                count = g.get('count')
    rows = c1_rows(summ)
    terminal = c1_terminal(summ)
    if not rows:            # single-op replay (CAPSULE_MODE=op): the op's own fp64 check
        for lg in sorted(glob.glob(os.path.join(a.attempt, 'g*_k*.log'))):
            terminal = op_terminal(open(lg, errors='replace').read().splitlines())
            if terminal:
                break
    verdict, sentence, why = propose(rows, terminal)
    if verdict == 'KERNEL_BUG' and sentence != KB_RERUN and any('returned a WRONG FINITE value' in l for l in summ + sum(
            (open(lg, errors='replace').read().splitlines()
             for lg in sorted(glob.glob(os.path.join(a.attempt, 'g*_k*.log')))[:1]), [])):
        sentence = KB_WRONG
        why = why.replace('terminal non-finite', 'terminal wrong finite value')
    if sentence == UNV and (a.mode == 'op' or a.chain_package):
        # an op-only package has no chain to call "recorded" (2026-11-28: the agent had to correct it by hand)
        sentence = UNV_OP
    # A capsule captured for a wrong FINITE value (no NaN anywhere): its summary says so, and every
    # "NaN" sentence below would be false (2026-11-20: a blind run rewrote it by hand).
    wrongcap = any(l.lstrip().startswith('(wrong-value capsule') for l in summ)
    if wrongcap and verdict == 'KERNEL_BUG' and sentence not in (KB_WRONG, KB_RERUN):
        sentence = KB_VALUE
    # An Inductor kernel's reference is not a CPU run: it is the kernel's own source promoted to fp64,
    # run on the GPU (inductor_ref.py) -- the README said "fp64 CPU reference" (2026-11-23 validation run)
    kop = op or (rows[-1]['op'] if rows else '')
    if kop.startswith('inductor.') and sentence:
        sentence = sentence.replace('fp64 CPU reference', 'fp64 reference (the kernel\'s own Triton source '
                                    'promoted to fp64, run on the GPU: `inductor_ref.py`)')
    probe = ''
    for ps in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(a.attempt.rstrip('/'))),
                                            'c1det.*', 'summary.txt'))):
        pm = re.search(r'non-finite (\d+)/(\d+)', next((l for l in open(ps).read().splitlines()
                                                         if l.startswith('=== totals')), ''))
        if pm and int(pm.group(1)) > 0:
            probe = '%s/%s runs' % pm.groups()
    synth = any('SYNTHETIC' in b for b in build)
    inconsistent = any('SYNTHETIC-INCONSISTENT' in b for b in build)
    derived = any('SYNTHETIC (derived' in b for b in build)
    if inconsistent and verdict == 'KERNEL_BUG':
        verdict = 'KERNEL_BUG_UNVERIFIED'
    if synth and verdict == 'KERNEL_BUG':
        sentence = KB_SYNTH
    runner = runner_of(repro)
    kern = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(a.attempt.rstrip('/'))), 'kernel.*.txt')))
    kfacts = dict(re.match(r'(\w+):\s*(.*)', l).groups() for l in open(kern[-1]).read().splitlines()
                  if re.match(r'\w+:', l)) if kern else {}
    kernel_md = ''
    if kfacts.get('kernel', '?')[:1] not in ('?', ''):
        kernel_md = ('## Kernel\n\n* solution_index %s, %s\n* `%s`\n* image NaN-injected: %s\n'
                     % (kfacts.get('solution_index', '?'), kfacts.get('arch', '?'), kfacts['kernel'],
                        kfacts.get('injected', '?')))
        if kfacts.get('bench', '').startswith('hipblaslt-bench'):
            kernel_md += ('* Without the capsule (one process; it may need co-tenants to fail):\n\n'
                          '  ```bash\n  %s\n  ```\n' % kfacts['bench'])
        kernel_md += '\n'
    chain = ' -> '.join(r['op'] for r in rows) or op or '?'
    unit = 'launches of the op alone' if a.mode == 'op' else 'chain runs'
    rate = '%d/%d %s (%.2g per run)' % (bad, runs, unit, bad / runs) if runs else 'n/a'
    if wrongcap and runs:
        rate += ' with a wrong finite output'
    size = os.path.getsize(cap)

    readme = """# {title}: {op}{geom}

## What this is

A capsule recorded from a live run of the user's workload at the moment `{op}`
{produced}, plus the scripts to replay it **without the model, dataset or
training code**.
{synth_note}
* {replayed}: `{chain}`{key}
* Capsule: `capsule/{capname}` ({mb:.1f} MB) -- sha256 `{csha}`
* Captured on / build / contention:
{build}
* {envkind}: `{image}`{built}
* Replay environment: `{envl}`{held}

## How the operands were validated

<!-- CHECK: proposed {verdict} from the [C1] walk ({why}); confirm against step5-read.md -->
{sentence}

{kernel_md}## The one command

```bash
bash run.sh            # {override}{site}
```

It runs the replay ladder: {probe_first}**{conc} {copies} of the replay on every GPU** of the node; it prints
`LADDER_RESULT=...`; per-copy logs and `summary.txt` land in `out/`. Measured here:
**{rate}** with {conc} {copies} on each of {gpus} GPU{gs}{stop}, on node `{host}` -- the only
node it was verified on. Until it is shown elsewhere, a clean replay on another node (`NOT_SET_UP`,
`NOT_REPRODUCED`) does not mean the bug is absent there or gone: a capsule can reproduce on one
node of a model and not on another (2026-11-14: 64000/64000 on one MI350X node, 0/32000 on another).

{contention}
""".format(op=op or rows[-1]['op'] if rows else op,
           # a grouping label such as "(any: --job)" (one job, any geometry) is not a geometry
           geom=(' ' + geom) if geom and not geom.startswith('(') and geom != 'unmapped' else '',   # 'unmapped': not a GEMM (op_nan_capture.geometry_key)
           chain=chain, replayed='Op replayed (alone, no chain)' if a.mode == 'op' else 'Chain replayed',
           key=(('\n* hipBLASLt problem: `%s`' % key) if key else '') + (
               ('\n* Op-only: the chain capsule this was cut from (the ops that fed `%s`) is left out -- '
                'the op alone reproduces on these operands. The full capsule, verified the same way, is '
                '`%s/` beside this package (not in this tarball).' % (op, a.chain_package))
               if a.chain_package else ''),
           capname=os.path.basename(cap), mb=size / 1e6, csha=sha256(cap),
           build='\n'.join('  `%s`' % b.strip() for b in build) or '  (not printed)',
           image=a.image, envl=envl, held=''.join(HELD.format(var=v) for v in held),
           envkind='Environment (venv, no container)' if runner in ('venv', 'nd-native') else 'Image',
           built=(' -- built locally, in no registry: its build context is in `image/`, and `run.sh` '
                  'builds it there (docker / podman) on a node that has neither the image nor this '
                  "site's archive" if os.path.exists(os.path.join(repro, 'image', 'Dockerfile')) else ''),
           override={'venv': 'venv; VENV=..., NGPUS=..., CAPSULE_ITER_COUNT=... to override',
                     'nd-native': 'this GPU node, in the venv run.sh names; `sbatch run.sh` on Slurm; '
                                  'VENV=..., NGPUS=..., CAPSULE_ITER_COUNT=... to override',
                     'nd': 'this GPU node, with the runtime that reproduced (else the first of docker/podman/apptainer); '
                           'RUNTIME=..., IMAGE=..., NGPUS=..., CAPSULE_ITER_COUNT=... to override'}.get(
                         runner, 'docker; IMAGE=..., NGPUS=..., CAPSULE_ITER_COUNT=... to override'),
           site=_site_line(repro), kernel_md=kernel_md, synth_note=(SYNTH_NOTE + (SYNTH_INCONSISTENT if inconsistent else SYNTH_DERIVED if derived else '') if synth else ''), verdict=verdict, why=why, sentence=sentence,
           conc=conc, gpus=gpus, rate=rate, copies='copy' if conc == 1 else 'copies',
           probe_first='' if a.cotenant else ('first one copy alone on GPU 0 (the deterministic probe: a fault that '
                                              'hits every run there ends it), else '),
           gs='' if gpus == 1 else 's',
           contention=(COTENANT.format(cot=a.cotenant) if a.cotenant
                       else EVERY_RUN.format(conc=conc, rate=rate) if runs and bad == runs and conc > 1
                       else SOLO_PROBE.format(probe=probe, conc=conc) if probe and conc > 1
                       else CONTENTION.format(conc=conc) if conc > 1 else SOLO),
           title='Wrong-value reproducer' if wrongcap else 'NaN reproducer',
           stop=' (attempt stopped early at its hit target; the rate is over runs done)' if stopped else '',
           produced=('returned a wrong (finite) value' if sentence in (KB_WRONG, KB_VALUE, KB_RERUN) or wrongcap
                     else 'produced a non-finite output'),
           # c1det.<tag>.<host>: the ladder's deterministic probe (2026-11-20: node `?` from it)
           host=(re.search(r'/c\d+(?:det)?\.[^./]+\.([^/]+)$', a.attempt.rstrip('/')) or [None, '?'])[1])
    with open(os.path.join(repro, 'README.md'), 'w') as f:
        f.write(readme)
    print('README %s/README.md  proposed=%s (%s)' % (repro, verdict, why))
    if 'DETERMINISTIC' in why:
        print('DO_NOT_SHIP %s' % why)
    elif verdict.startswith('FP32'):
        # not a kernel bug, but a reproducer all the same: step 6 ships it as a model-numerics one and
        # steps 7-8 find why the model hands the op those values (step6-finish.md *Then*) -- DO_NOT_SHIP
        # here contradicted that (2026-11-23, eps-0 RMSNorm)
        print('MODEL_NUMERICS %s' % why)
    if a.pack:
        pack(repro, a.ledger, a.step, 'proposed=%s (README auto; CHECK)' % verdict)

if __name__ == '__main__':
    main()
