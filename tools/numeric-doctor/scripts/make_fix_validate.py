#!/usr/bin/env python3
"""debug-numeric step 8: write <ARTIFACT_DIR>/fix/validate.sh -- the fix package's one command.

    python3 scripts/make_fix_validate.py --fix <ART>/fix --repro <ART>/repro-G1 \
        --cmd '<the stock-vs-fix arms, with $ND_REPO = the fix dir, $ND_ART = the reproducer>' \
        [--gpus 0|all|0,1] [--time 00:30:00] [--tool <file> ...]

The Slurm header has --no-requeue: a validate.sh that had finished rc=0 was requeued and would
have run again (2026-11-20).  $ND_REPO/out/ exists before the arms run.

Every investigation wrote validate.sh by hand, copied from an earlier run's (a hard-coded capsule
name, a co-tenant that did not apply, 2026-11-20); only the arms differ between fixes.  This writes
the rest: the Slurm header, the runtime exactly as the reproducer's run.sh bakes it (image, runtime,
the profile's knobs, IMAGE_TAR / image/ build context), the reproducer's location -- REPRO, default
the sibling `../<repro dir name>`, i.e. both tarballs extracted side by side -- and `nd run` of
--cmd.  Inside the runtime $ND_REPO is the fix dir (its scripts/ on PYTHONPATH) and $ND_ART the
reproducer (its capsule/ and scripts/; a 5N package: repro.py and native_run.sh, no capsule/).  It also ships fix/scripts/: nd and its env layer, the
reproducer's tools, fix_overlay.py, fix_values.py, env_workaround_validate.sh,
replay_inductor_kernel.py, capsule_operands.py, inductor_ref.py, and each --tool.

--cmd examples (each prints its own result lines):
  hipBLASLt overlay:  'python3 -m fix_overlay --out $ND_REPO/out/overlay --kernel-file ... && python3 -m fix_values $ND_ART/capsule/<pt> --overlay ...'
  env workaround:     'bash $ND_REPO/scripts/env_workaround_validate.sh $ND_ART/capsule/<pt> "TENSILE_FIXED_WGMXCC=1" 2000'
  kernel capsule:     'python3 $ND_REPO/values.py $ND_ART/capsule/<pt>'
  COLLECTIVE package: 'bash $ND_ART/scripts/collective_arms.sh $ND_ART/capsule $ND_REPO/patch/arms.txt'
                      (--gpus 0,1: the member ranks' GPUs; arms.txt = stock|, <fix>|VAR=VALUE ..., stock_again|)
  native (5N):        'for a in stock F1; do echo ARM=$a; [ $a = F1 ] && export <VAR=VALUE>; RANKS=2 COPIES=1 \
                         bash $ND_ART/native_run.sh $ND_ART/repro.py --iters 1000; done'  (--gpus all)
  aiter .co patch:    'bash $ND_REPO/scripts/aiter_fix_validate.sh $ND_ART/capsule/<pt> gfx950/<dir>/<name>.co \
                         F1=$ND_REPO/patch/F1/patched/gfx950/<dir>/<name>.co -- --contend <C-1>'
                      (--gpus all: stock + one GPU per fix arm; the patched .co from the sweep's
                      --mutate-keep-fix row, or a hand edit of it, copied into fix/patch/<name>/:
                      a short name without '@', which sweep row names carry)

Checks the result with `bash -n`; then pack and run it as shipped (step8-fix.md item 4).
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# capsule_operands.py: what a hand-written values.py loads a capsule with (step 8's pattern; 2026-11-23 the
# shipped validate.sh would have died at its import); inductor_ref.py: the fp64 twin of an Inductor kernel
TOOLS = ('fix_overlay.py', 'fix_values.py', 'env_workaround_validate.sh', 'replay_inductor_kernel.py',
         'aiter_overlay.sh', 'aiter_fix_validate.sh', 'capsule_operands.py', 'inductor_ref.py')


def runtime_lines(run_sh):
    """The runtime block of a capture-job run.sh: from `export ND_REPO_HOST=` up to the first
    CAPSULE / OUT_DIR export or the exec line.  The export line's other assignments are kept:
    a 5N package's run.sh (step6-finish.md) sets IMAGE= on it, and dropping that line lost the
    image (2026-11-21: the fix verify failed before any GPU work)."""
    L = open(run_sh).read().splitlines()
    a = next((i for i, l in enumerate(L) if l.startswith('export ND_REPO_HOST=')), None)
    if a is None:
        return None
    b = next((i for i in range(a + 1, len(L))
              if re.match(r'(export (CAPSULE|OUT_DIR)|exec )', L[i])), len(L))
    rest = re.sub(r'\s+ND_(REPO|ART)_HOST=\S+', '', ' ' + L[a][len('export'):]).strip()
    block = (['export ' + rest] if rest else []) + L[a + 1:b]
    return [l.replace('$HERE/image', '$REPRO/image') for l in block]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--fix', required=True, help='<ART>/fix')
    p.add_argument('--repro', required=True, help='the step-6 reproducer dir (repro-G1, repro-kernel)')
    p.add_argument('--cmd', required=True, help='the arms, as one bash -c string ($ND_REPO, $ND_ART)')
    p.add_argument('--gpus', default='0', help="nd run --gpus (default 0; 'all' for arms side by side)")
    p.add_argument('--time', default='00:30:00')
    p.add_argument('--tool', action='append', default=[], help='another file for fix/scripts/')
    a = p.parse_args()
    fix, repro = os.path.abspath(a.fix), os.path.abspath(a.repro)
    run_sh = os.path.join(repro, 'run.sh')
    rt = runtime_lines(run_sh) if os.path.exists(run_sh) else None
    if rt is None:
        sys.exit('make_fix_validate: %s has no `export ND_REPO_HOST=` runtime block (a capture-job '
                 'package\'s run.sh has one); write validate.sh by hand from it' % run_sh)
    sd = os.path.join(fix, 'scripts')
    os.makedirs(os.path.join(sd, 'env'), exist_ok=True)
    # the runtime layer, from the reproducer (the version that ran) else the repo
    src_scripts = os.path.join(repro, 'scripts')
    shutil.copy2(os.path.join(src_scripts if os.path.exists(os.path.join(src_scripts, 'nd')) else HERE, 'nd'), sd)
    env_src = os.path.join(src_scripts, 'env') if os.path.isdir(os.path.join(src_scripts, 'env')) \
        else os.path.join(HERE, 'env')
    shutil.copytree(env_src, os.path.join(sd, 'env'), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for f in sorted(glob.glob(os.path.join(src_scripts, '*.py')) + glob.glob(os.path.join(src_scripts, '*.sh'))):
        shutil.copy2(f, sd)
    for t in TOOLS + tuple(a.tool):
        f = t if os.path.isabs(t) or os.path.exists(t) else os.path.join(HERE, t)
        if os.path.exists(f):
            shutil.copy2(f, sd)
        elif t in a.tool:
            sys.exit('make_fix_validate: no tool %s' % t)
    # fix_overlay / fix_values import kernel_patch & co.: the same set the reproducer ships, else the repo's
    for dep in ('kernel_patch.py', 'kernel_mutations.py', 'replay_nan_op.py', 'op_nan_capture.py',
                'kernel_match.py', 'math_refs.py', 'replay_verdict.py', 'inductor_ref.py', 'twins.py',
                'waitcnt_scan.py'):
        if not os.path.exists(os.path.join(sd, dep)) and os.path.exists(os.path.join(HERE, dep)):
            shutil.copy2(os.path.join(HERE, dep), sd)
    os.chmod(os.path.join(sd, 'nd'), 0o755)
    rname = os.path.basename(repro.rstrip('/'))
    body = '''#!/bin/bash
#SBATCH --job-name=nan-fix-validate
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time={time}
# Validate the fix on the step-6 reproducer: stock (the control: it must reproduce) and the fix, as
# the arms below run them.  Extract the reproducer's and this package's tarballs side by side, then
#   bash validate.sh                     # on a GPU node
#   sbatch [<site args>] validate.sh     # Slurm, from this directory
#   REPRO=<the extracted {rname}>  RUNTIME=...  IMAGE=...  to override
# Written by make_fix_validate.py; the runtime lines are the reproducer's run.sh's.
set -uo pipefail
if [ -n "${{SLURM_JOB_ID:-}}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname -- "$0"); fi
HERE=$(cd -- "$HERE" && pwd -P) || exit 2
REPRO=$(cd -- "${{REPRO:-$HERE/../{rname}}}" 2>/dev/null && pwd -P) ||
  {{ echo "no reproducer at ${{REPRO:-$HERE/../{rname}}}: extract {rname}.tar.gz beside this dir, or set REPRO="; exit 2; }}
[ -f "$REPRO/run.sh" ] || {{ echo "REPRO=$REPRO has no run.sh (not an extracted reproducer)"; exit 2; }}
export ND_REPO_HOST=$HERE ND_ART_HOST=$REPRO
mkdir -p "$HERE/out"      # the arms write $ND_REPO/out/ (an overlay, logs)
{rt}
exec "$HERE/scripts/nd" run --gpus {gpus} -- {cmd}
'''.format(time=a.time, rname=rname, rt='\n'.join(rt), gpus=a.gpus,
           cmd="'%s'" % a.cmd.replace("'", "'\\''"))
    out = os.path.join(fix, 'validate.sh')
    open(out, 'w').write(body)
    os.chmod(out, 0o755)
    r = subprocess.run(['bash', '-n', out], capture_output=True, text=True)
    if r.returncode:
        sys.exit('make_fix_validate: %s does not parse: %s' % (out, r.stderr.strip()))
    print('VALIDATE_SH %s (REPRO default ../%s, gpus %s; %d file(s) in scripts/)'
          % (out, rname, a.gpus, len(os.listdir(sd))))


if __name__ == '__main__':
    main()
