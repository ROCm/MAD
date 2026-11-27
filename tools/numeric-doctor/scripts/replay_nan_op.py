#!/usr/bin/env python3
"""Replay a capsule written by op_nan_capture.OpNanCapture.

    python3 replay_nan_op.py CAPSULE.pt

This needs no model, no dataset and no training loop -- just the capsule.  It
re-creates the operands with their original sizes, strides and pointer alignment,
re-issues the same aten op, and answers three questions in order:

  1. Are the inputs finite?           If not, the op inherited the NaN and the
                                      real bug is upstream -- stop here.
  2. Is a fp64 CPU reference finite?  If the reference overflows too, fp32 range
                                      is the problem, not the kernel.
  3. Does the GPU still produce NaN?  Only now is it a kernel bug.

Step 2 is the one people skip.  An op with finite inputs can legitimately produce
inf: e.g. a GEMM with K=1920 and operands around 1e19 the fp32 accumulator overflows and any
correct kernel returns inf.  Checking the reference is what separates "the kernel
is wrong" from "the model diverged".

Useful flags:

    --iters 100          re-run N times.  Split-K / GSU kernels accumulate with
                         atomics, so the failure can be order-dependent and show
                         up only on some launches.
    --stop-on-nan        leave the [3] loop at the first non-finite launch instead of
                         completing --iters.  Off here, on in every mutation trial: a
                         trial's outcome is decided by its first NaN, so the launches
                         after it buy nothing.  The rate n/N is the thing it costs, and
                         that is exactly what the baseline (which never sets this) is for.
    --solution-index N   pin the hipBLASLt solution (same env var that
                         replay_hipblaslt_log.py uses).  Get N by grepping your
                         HIPBLASLT_LOG for the problem key the capsule prints.
    --no-hipblaslt       route to rocBLAS instead -- the hipBLASLt-vs-rocBLAS A/B test,
                         on one isolated GEMM.
    --bench              print the equivalent hipblaslt-bench command and exit.
    --disasm FILE        write the ISA of the hipBLASLt kernel this capsule dispatches
                         to -- as installed, unmodified -- to FILE.  Off by default.
                         Which kernel wins is a property of the library and not of the
                         capsule, so this costs the same discovery replay --mutate pays
                         (skipped when --patch-kernel names one) plus one unbundle.
    --mutate             once the NaN reproduces, ask the next question: WHICH part
                         of the kernel is responsible.  Replays this same capsule
                         against systematically mutated copies of the winning
                         Tensile kernel and reports which mutations make the NaN go
                         away.  See kernel_mutations.py.
    --mutate-force       sweep even when the baseline replay came back clean (off by
                         default; implies --mutate).  For an intermittent fault, where
                         a NAN row -- identity included -- is the finding and a CLEAN
                         row proves nothing.
    --mutate-no-identity skip the identity control (on by default, and always added
                         to --mutate-only).  For a sweep over a finished reproducer,
                         where the replay path is known to reproduce and a trial can
                         cost an hour.  CLEAN rows are then reported without a positive
                         control -- the asm read-back is the only proof the patch
                         landed -- and the budget is re-derived from the first trial.
    --mutate-timeout     per-trial wall clock.  No longer a constant by default: a trial
                         costs roughly what the baseline replay cost, so the budget is
                         derived from it as 2x (setup + per-launch x --iters).  300s was
                         generous at --iters 1 and an all-TIMEOUT sweep at --iters 1000.
                         That baseline is one measurement taken before the first trial
                         though, and a busy node can move under it -- on gfx950 under a
                         concurrent 8-GPU load the trials ran ~4x their predicted rate --
                         so the budget is re-derived once from what the identity control
                         cost, which is the first trial timed under trial conditions.
    --mutate-disasm      for each mutation that DOES remove the NaN, print the
                         kernel's ISA before and after it, read back out of the
                         overlay the child actually ran.  On by default (diff);
                         --mutate-disasm full prints both complete listings,
                         --mutate-disasm none turns the read-back off.
    --chain              replay the recorded data-dependency chain instead of the
                         failing op alone.  [C1] walks it once and names the first op
                         to go wrong; [C2] (--chain-loop-only --chain-iters N) re-runs
                         the whole chain N times and tallies which op goes non-finite
                         FIRST, which is the localisation.
    --contend N          run N sibling replay processes on this same device while
    --contend-auto       measuring, or take N from the capsule.  A contention-gated
    --contend-with CMD   fault is 0/N solo however large N is -- serial repetition in
                         one process is not concurrency -- so a solo 0/N on such a
                         capsule is not a result, it is an experiment that was not set
                         up.  Gated on observed device memory, never on a sleep.

These three groups COMPOSE, and the composition is the point:

    --mutate --contend-auto
        the co-tenants are held for the whole sweep rather than dropped after the
        baseline, so every trial is measured under the same load the baseline was.
        Without this a contention-gated fault sweeps solo: identity comes back CLEAN
        and every mutation looks like it fixed the bug.  --mutate-solo opts out.
    --mutate --chain --chain-loop-only --chain-iters N
        the trial payload becomes the chain loop rather than the single op.  For a
        capsule whose recorded operands are themselves corrupt, the single op is not
        a faithful reproducer and a sweep over it measures the wrong thing.
        The chain is swept BACKWARD -- culprit first, then upstream -- because
        mutating an upstream op changes the operands every downstream op sees.

Because hipBLASLt and rocBLAS read their environment when the .so initialises,
all env-setting flags are applied before torch is imported; --restore-env re-execs
so the capsule's recorded environment is reproduced exactly -- setting what the
capture had and unsetting what it did not, since a merge would restore a superset.
Every run prints an `env:` banner saying how far this process is from the capture,
passed or not, because the alternative symptom is a clean 0/N.  That same property is
why every mutation trial is a fresh child process rather than a loop in this one:
an overlay library dir only takes effect in a process that has not yet initialised
hipBLASLt, and a wedged GPU trial can only be killed if it owns a pid.
"""

import argparse
import atexit
import math
import os
import socket
import sys
import time

# Process start, and the only honest place to take it: --mutate-timeout's default is
# derived from what a trial child costs, and a child's cost begins with `import torch`.
# Measuring from inside main() would hide the largest fixed term in the budget.
_T0 = time.time()

# Every module this script needs -- kernel_patch, kernel_mutations and the
# replay_hipblaslt_log_with_nan the deferred imports below pull in -- is a
# sibling in scripts/.  Set here rather than inherited from a PYTHONPATH so
# the script runs by path from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_patch import add_patch_args               # noqa: E402
from kernel_mutations import add_mutate_args          # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capsule", nargs="?",
                    help=".pt written by OpNanCapture (not needed for --list-mutations --patch-kernel)")
    ap.add_argument("--iters", type=int, default=1,
                    help="replay N times and count how many produced NaN/Inf (default 1)")
    ap.add_argument("--progress-file", metavar="FILE", default=None,
                    help="every --progress-every runs of the [C2] / [3] loop, overwrite "
                         "FILE with '<runs> <non-finite runs>' (read by the node "
                         "launchers to stop a reproducing attempt early)")
    ap.add_argument("--stop-file", metavar="FILE", default=None,
                    help="checked every --progress-every runs: once FILE exists, finish "
                         "the loop and report the runs actually done. The node launchers "
                         "create it for every copy at once, so co-tenancy holds to the end")
    ap.add_argument("--progress-every", type=int, default=2000, metavar="N",
                    help="cadence of --progress-file / --stop-file, in runs (default 2000)")
    ap.add_argument("--start-barrier", metavar="DIR", default=None,
                    help="before the [C2] / [3] loop, drop a ready file in DIR and wait "
                         "until --barrier-n copies have, so co-tenant copies overlap "
                         "from their first run rather than from whenever each started")
    ap.add_argument("--barrier-n", type=int, default=1, metavar="N",
                    help="copies the --start-barrier waits for (default 1 = no wait)")
    ap.add_argument("--barrier-timeout", type=float, default=300.0, metavar="SEC",
                    help="give up waiting at the barrier after SEC and run anyway, "
                         "saying so (default 300)")
    ap.add_argument("--seconds", type=float, default=0.0, metavar="SEC",
                    help="also stop the [C2] / [3] loop SEC after it starts (0 = off); "
                         "the run count stays the upper bound")
    ap.add_argument("--solution-index", type=int, default=None,
                    help="pin TENSILE_SOLUTION_INDEX")
    ap.add_argument("--no-hipblaslt", action="store_true",
                    help="TORCH_BLAS_PREFER_HIPBLASLT=0 (route to rocBLAS)")
    ap.add_argument("--emit-log", metavar="FILE",
                    help="write a hipBLASLt profile log here (HIPBLASLT_LOG_MASK=160)")
    ap.add_argument("--disasm", metavar="FILE",
                    help="write the disassembly of the hipBLASLt kernel this capsule "
                         "dispatches to here, as installed (off by default)")
    ap.add_argument("--restore-env", action="store_true",
                    help="re-exec under the environment recorded in the capsule, so the "
                         "library sees what it saw originally. Sets AND unsets within the "
                         "recorded namespace; device-selection and loader-path vars are "
                         "recorded but never re-applied. A banner reports the delta "
                         "whether or not this is passed")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the fp64 CPU reference (it is O(MNK) and single-threaded-ish)")
    ap.add_argument("--stop-on-nan", action="store_true",
                    help="leave the [3] replay loop at the first non-finite launch "
                         "instead of completing --iters. Every mutation trial sets this; "
                         "leave it off here, where the n/N rate is the measurement")
    ap.add_argument("--synthesize", action="store_true",
                    help="replay on look-alike operands drawn from the per-tensor "
                         "statistics a shapes-only capsule records (normal(mean, std) "
                         "clipped to [min, max], with the recorded fractions of zeros, "
                         "NaN and inf; integers uniform in [min, max]) -- same shapes, "
                         "strides, alignment and kernel, made-up values. Reproduces a "
                         "fault that does not depend on the values (a race, a kernel "
                         "that NaNs on any input); a clean result proves nothing about "
                         "one that does. No effect on a capsule with recorded values")
    ap.add_argument("--synthesize-seed", type=int, default=0,
                    help="seed for --synthesize (default 0); every process derives the "
                         "same operands from it")
    ap.add_argument("--bench", action="store_true",
                    help="print the hipblaslt-bench command for this problem and exit")
    ap.add_argument("--chain", action="store_true",
                    help="replay the recorded data-dependency chain (format-3 capsule) "
                         "instead of the failing op alone, and name the op that first "
                         "goes non-finite or diverges from the recording")
    ap.add_argument("--chain-iters", type=int, default=0,
                    help="after the per-op walk, re-run the whole chain N times from "
                         "its roots and report n/N (0 = skip)")
    ap.add_argument("--chain-repeat-check", action="store_true",
                    help="in the [C2] loop, also compare every op's float outputs bit-for-"
                         "bit with the loop's first run and tally which op first DIFFERED "
                         "in each non-finite run.  Same inputs every run, so a deterministic "
                         "kernel repeats exactly; a racy one does not.  Localises a fault "
                         "that writes FINITE but wrong values upstream of the op that "
                         "finally goes non-finite -- which the finiteness tally blames.  "
                         "Opt-in: one compare per output per run is extra GPU work in the "
                         "timed loop")
    ap.add_argument("--chain-save-outputs", metavar="FILE",
                    help="save every op's float outputs from the [C2] loop's first run "
                         "(implies --chain-repeat-check): the stock reference for a step-8 "
                         "value check")
    ap.add_argument("--chain-compare-outputs", metavar="FILE",
                    help="compare the [C2] loop's first-run outputs bit-for-bit with a "
                         "--chain-save-outputs FILE (implies --chain-repeat-check); prints "
                         "[values] IDENTICAL or DIFFER -- a barrier/wait-only fix must be "
                         "IDENTICAL")
    ap.add_argument("--chain-loop-only", action="store_true",
                    help="skip the [C1] per-op walk and go straight to the loop. What "
                         "--contend gives its children, so they spend their time "
                         "loading the device rather than doing fp64 comparisons")
    ap.add_argument("--chain-fallback", action="store_true",
                    help="with --chain: if the capsule has no chain (format < 3), "
                         "replay the failing op alone with --iters = --chain-iters "
                         "instead of exiting. What the node launcher passes, so one "
                         "command covers every capsule a group can name")
    ap.add_argument("--contend", type=int, default=0, metavar="N",
                    help="run N sibling replay processes on this same device while "
                         "measuring, to recreate the concurrency the capture saw. "
                         "Default 0 = replay alone, which for a contention-gated "
                         "fault is the case that never reproduces. Pass 'auto' via "
                         "--contend-auto to take N from the capsule")
    ap.add_argument("--contend-auto", action="store_true",
                    help="set --contend from the capsule's recorded contention "
                         "(gpu_procs x containers_per_gpu, minus this process), "
                         "so the replay matches the capture. Exits rather than "
                         "replaying solo if that resolves to 0 -- see "
                         "NAN_CAPTURE_CONTAINERS_PER_GPU in op_nan_capture.py")
    ap.add_argument("--capture-procs", action="store_true",
                    help="print CAPTURE_PROCS=<n|?> (processes on the device at "
                         "capture, gpu_procs x containers_per_gpu) and "
                         "CAPTURE_PROCS_DETAIL=..., then exit without touching a "
                         "GPU -- the node launchers start their ladder at n+1")
    ap.add_argument("--contend-with", metavar="CMD",
                    help="shell command to run as co-tenant instead of sibling "
                         "replays -- normally the real model. Takes precedence "
                         "over --contend/--contend-auto. Which of the two loads "
                         "is stronger does NOT replicate across hosts (a real "
                         "model beat siblings ~16x on one, lost to them on "
                         "another), so run whichever you can get")
    ap.add_argument("--contend-log", metavar="FILE",
                    help="append the co-tenants' stdout+stderr here (default: discarded); "
                         "the place to look when every co-tenant exits before loading")
    ap.add_argument("--contend-warmup", type=float, default=120.0, metavar="SEC",
                    help="how long to wait for co-tenants to actually load the device "
                         "before measuring. Gated on observed memory, never a sleep; "
                         "if it never trips the run is reported INVALID (default 120)")
    ap.add_argument("--written-operands", choices=("captured", "zero", "random"),
                    default="captured",
                    help="what the capsule's op finds in the arguments it WRITES (dq/dk/dv "
                         "of a fused attention backward) before each call: the captured "
                         "pre-call contents (default), zeros, or finite random values. "
                         "The repair test of step5-replay-guardrail.md: the same NaN under "
                         "all three means the op creates it from its inputs, not from "
                         "uninitialised out-buffers. Destroys the operand of an in-place op")
    ap.add_argument("--collective-noise", type=float, default=0.0, metavar="MB",
                    help="R2 (world size > 1): launched under torchrun, keep an async "
                         "all_reduce of this many MB in flight across every [C2] chain run, "
                         "so the chain computes while RCCL moves data -- the overlap of "
                         "compute and communication a FSDP/TP/EP model has and a solo "
                         "replay lacks. Every rank replays the same capsule; use a fixed "
                         "--chain-iters (ranks must run the same number of collectives)")
    ap.add_argument("--wrong-value", action="store_true",
                    help="in the [C2] loop, also count a run when a GEMM-family op returns a "
                         "finite value that fails a higher-precision recompute (always on in the "
                         "[C1] walk; turned on for the loop automatically when the walk finds one)")
    ap.add_argument("--atol", type=float, default=1e-3)
    add_patch_args(ap)
    add_mutate_args(ap)
    a = ap.parse_args(argv)
    # --list-mutations --patch-kernel enumerates from the kernel alone, GPU- and capsule-free
    # (step7-recipe-sweep.md); argparse demanded a capsule anyway (2026-11-14)
    if a.capsule is None and not (a.list_mutations and a.patch_kernel):
        ap.error("the following arguments are required: capsule")
    return a


# Imported rather than run (a guardrail experiment reusing unpack/_bind/the
# references): every flag at its default, so nothing below touches the
# environment, and a placeholder capsule that nothing reads.
args = parse_args(None if __name__ == "__main__" else ["<imported>"])
# --mutate-force is a modifier on --mutate, and a modifier that silently does nothing
# when the thing it modifies was not asked for is a trap.  Implying it here, before
# _MUTATING is computed, is what makes `--mutate-force` alone behave the way it reads.
if args.mutate_force:
    args.mutate = True
if args.mutate_no_identity and "identity" in args.mutate_only:
    sys.exit("--mutate-no-identity and --mutate-only identity contradict each other")
if args.mutate_no_identity and args.mutate_disasm == "none":
    # Without the control, the asm read-back is the only thing that tells a CLEAN
    # row from a patch that never landed; switching both off leaves nothing.
    sys.exit("--mutate-no-identity needs the asm read-back: drop --mutate-disasm none")
_MUTATING = args.mutate or args.list_mutations

# ---- environment, before torch is imported ---------------------------------
if args.solution_index is not None:
    os.environ["TENSILE_SOLUTION_INDEX"] = str(args.solution_index)
elif _MUTATING and args.patch_solution_index is not None:
    os.environ["TENSILE_SOLUTION_INDEX"] = str(args.patch_solution_index)
if args.no_hipblaslt:
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
if args.emit_log:
    os.environ["HIPBLASLT_LOG_MASK"] = "160"      # bench + extended profile
    os.environ["HIPBLASLT_LOG_FILE"] = args.emit_log

torch = None                                      # bound by _import_torch()
unpack = None
tree_leaves = None
upgrade_capsule = None
_CHAIN_REF = None
_MEMFMT_TAG = None
_scannable = None
hipblaslt_geometry = None
op_writes = None
_ALLOC_OPS = frozenset()                          # bound by _import_torch()

# call_index -> that op's hipBLASLt problem key, filled in as the chain executes.
# The capsule records a geometry only for the op it is NAMED for; a chain sweep has
# to target the kernel each UPSTREAM op dispatches too, and the only honest way to
# know that is to map the operands the op actually ran with.
_CHAIN_GEOM = {}

# [C1]'s culprit, for report().  A list rather than a bare global for the same
# reason as _LAST_TARGET: run_chain_mutations writes it without a `global`.
_CHAIN_CULPRIT = [None]


def _import_torch():
    """Import torch at the start of main(), not at module scope.

    Every path that touches torch goes through main(), so deferring the import
    changes nothing for the existing flags -- and it buys the one thing worth
    buying: with --patch-kernel, --list-mutations enumerates a sweep on a login
    node that has ROCm but no torch and no GPU.
    """
    global torch, unpack, tree_leaves, _CHAIN_REF, _scannable, hipblaslt_geometry
    global op_writes, _ALLOC_OPS, upgrade_capsule, _MEMFMT_TAG, _Meta
    import torch as _torch
    from torch.utils._pytree import tree_leaves as _tl
    from op_nan_capture import _MEMFMT_TAG as _memfmt
    _MEMFMT_TAG = _memfmt
    # _scannable promotes the float8 dtypes torch.isfinite refuses; the chain walk
    # scans every intermediate, so it meets them far more often than a single-op
    # replay does.
    # hipblaslt_geometry is the capture's own aten->hipBLASLt problem-key mapper,
    # reused rather than reimplemented so a chain op's key is derived exactly the
    # way the capsule's own key was.
    from op_nan_capture import (unpack as _unpack, _CHAIN_REF as _ref,
                                synthesize_operands as _synth,
                                _scannable as _scan,
                                hipblaslt_geometry as _geom, _Meta as _meta,
                                op_writes as _writes,
                                ALLOC_EXCLUDE_OPS as _alloc,
                                upgrade_capsule as _upgrade)
    torch, unpack, tree_leaves, upgrade_capsule = _torch, _unpack, _tl, _upgrade
    if args.synthesize:
        _synth(args.synthesize_seed)
    _CHAIN_REF, _scannable, hipblaslt_geometry, _Meta = _ref, _scan, _geom, _meta
    op_writes = _writes
    _ALLOC_OPS = frozenset(_alloc.split(","))


_TRY_RESOLVED = {}


def try_resolve_op(name):
    """resolve_op without exiting: None when the op cannot be resolved in this process. Memoised: the [C2] chain loop
    runs it per op per run, and that loop is a timing measurement."""
    if name not in _TRY_RESOLVED:
        try:
            _TRY_RESOLVED[name] = resolve_op(name)
        except SystemExit:
            _TRY_RESOLVED[name] = None
    return _TRY_RESOLVED[name]


def resolve_op(name):
    """'aten.addmm.default' -> torch.ops.aten.addmm.default

    Any namespace, not just aten.  A capture filtered to a custom library --
    NAN_CAPTURE_OPS=flash_attn is the case that forced this -- records op names
    like 'flash_attn._flash_attn_backward.default', and a fused attention
    backward is exactly the op this skill's ladder is most often pointed at.
    torch.ops.<ns> materialises lazily, so a non-aten namespace resolves to an
    empty shell until the library that REGISTERED the op has been imported;
    import it by name first and let the AttributeError below report it if the
    module name and the namespace do not coincide.
    """
    if name.startswith(("triton.", "inductor.")):
        # A Triton kernel launch (op_nan_capture "Triton launches"): rebuilt from the
        # capsule's triton_kernels descriptor, registered when the capsule was loaded.
        from op_nan_capture import triton_op_for
        try:
            return triton_op_for(name)
        except Exception as e:                          # noqa: BLE001
            sys.exit("Triton op %r did not resolve: %s: %s" % (name, type(e).__name__, e))
    parts = name.split(".")
    if len(parts) != 3:
        sys.exit("unsupported op string %r" % name)
    ns, opname, overload = parts
    if ns != "aten":
        try:
            __import__(ns)
        except ImportError as e:
            sys.exit("op %r needs the %r library imported to register it, and "
                     "importing it failed: %s" % (name, ns, e))
    try:
        return getattr(getattr(getattr(torch.ops, ns), opname), overload)
    except (AttributeError, RuntimeError) as e:
        err = e
    # the package does not register its custom ops on import; the module that does (SGLang 0.5.23's
    # sglang.outplace_all_reduce: sglang.srt.distributed.parallel_state, 2026-11-24)
    for mod in _NS_MODULES.get(ns, ()):
        try:
            __import__(mod)
            return getattr(getattr(getattr(torch.ops, ns), opname), overload)
        except (ImportError, AttributeError, RuntimeError) as e:
            err = e
    sys.exit("op %r did not resolve after importing %r: %s" % (name, ns, err))


_NS_MODULES = {"sglang": ("sglang.srt.distributed.parallel_state",),
               "vllm": ("vllm.distributed.parallel_state", "vllm.distributed.communication_op")}
# A collective in a capsule's chain cannot run in a one-process replay (no peers), nor can a framework's custom op
# that this process cannot resolve (vLLM's rocm_unquantized_gemm, its attention ops: they need the framework's
# forward context) -- upstream of the terminal op, their recorded outputs stand in
_COLLECTIVE_OP = ("all_reduce", "all_gather", "reduce_scatter", "broadcast", "all_to_all", "_c10d_functional.")


def is_collective_op(name):
    return any(k in name for k in _COLLECTIVE_OP)


def _recorded_collective(cap, rec, env, dev):
    """Bind a chain collective's RECORDED outputs (the capture stores chain intermediates) instead of running it.
    -> the tensors, or exit with what to do when the capture did not store them."""
    keys = [k for k in rec["produces"] if k]
    missing = [k for k in keys if k not in cap["pool"]]
    if not keys or missing:
        sys.exit("chain op %s cannot run here (%s) and the capsule did not store its output -- replay the terminal "
                 "op alone (--no-chain)%s" % (rec["op"], "a collective: it needs its peer ranks"
                                              if is_collective_op(rec["op"]) else "not resolvable in this process",
                                              ", or the ranks' capsules together (replay_nan_op_dist.py)"
                                              if is_collective_op(rec["op"]) else ""))
    out = []
    for k in keys:
        env[k] = unpack(cap["pool"][k], dev)
        out.append(env[k])
    return out


def bench_cmd(g, dtype_tag="f32_r", compute="xf32_r", solution=None):
    cmd = ("hipblaslt-bench --api_method c"
           " -m {M} -n {N} -k {K} --lda {lda} --ldb {ldb} --ldc {ldc} --ldd {ldd}"
           " --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0"
           " --alpha 1.0 --beta 0.0 --transA {transA} --transB {transB}"
           " --batch_count {batch_count}").format(**g)
    cmd += (" --a_type {t} --b_type {t} --c_type {t} --d_type {t}"
            " --compute_type {c} --activation_type none".format(t=dtype_tag, c=compute))
    if g.get("bias"):
        cmd += " --bias_vector --bias_source d --bias_type {t}".format(t=dtype_tag)
    if solution is not None:
        cmd += " --algo_method index --solution_index %d" % solution
    return cmd


def packed_geometry(cap):
    """hipBLASLt geometry from the packed args' shapes and strides, for a capsule recorded
    before its op was mapped (aten.linear before 2026-11-19); None if unmapped."""
    def meta(x):
        if isinstance(x, dict) and _CHAIN_REF in x:
            x = (cap.get("pool") or {}).get(x[_CHAIN_REF])
        if isinstance(x, dict) and "size" in x and "stride" in x:
            return _Meta(x["size"], x["stride"])
        return x
    try:
        func = getattr(torch.ops.aten, cap["op"].split(".")[1]).__getattr__(cap["op"].split(".")[2])
        return hipblaslt_geometry(func, [meta(a) for a in cap.get("args") or []])
    except Exception:                                   # noqa: BLE001
        return None


_BENCH_DTYPE = {"torch.bfloat16": ("bf16_r", "f32_r"), "torch.float16": ("f16_r", "f32_r"),
                "torch.float32": ("f32_r", None)}


def grouped_geoms(cap):
    """aten._grouped_mm (2-D A, 3-D B, offs) -> (len(groups), [(rows, [group indices], plain-mm
    hipBLASLt geometry)] per DISTINCT group size), or None for another op or layout.  Each
    group is dispatched as its own plain GEMM (A[s:e] @ B[g]); from metadata alone (shapes,
    strides, the recorded integer offs), so it works on a shapes-only capsule."""
    def meta(x):
        if isinstance(x, dict) and _CHAIN_REF in x:
            return (cap.get("pool") or {}).get(x[_CHAIN_REF])
        return x
    a = cap.get("args") or []
    A, B = meta(a[0]) if len(a) > 0 else None, meta(a[1]) if len(a) > 1 else None
    offs = meta(a[2] if len(a) > 2 else (cap.get("kwargs") or {}).get("offs"))
    if not (isinstance(A, dict) and isinstance(B, dict) and isinstance(offs, dict)) \
            or len(A["size"]) != 2 or len(B["size"]) != 3 or offs.get("base") is None:
        return None
    ends = unpack(offs, "cpu").flatten().tolist()
    by_m = {}
    for gi, (s, e) in enumerate(zip([0] + ends[:-1], ends)):
        if e > s:
            by_m.setdefault(e - s, []).append(gi)
    q, r = A["size"][1], B["size"][2]
    out = []
    for m in sorted(by_m):
        X = torch.empty_strided((m, q), tuple(A["stride"]), device="meta")
        Y = torch.empty_strided((q, r), tuple(B["stride"][1:]), device="meta")
        geom = hipblaslt_geometry(torch.ops.aten.mm.default, (X, Y))
        if geom:
            out.append((m, by_m[m], geom))
    return len(ends), out


def grouped_bench_cmds(cap):
    """--bench for aten._grouped_mm (2-D A, 3-D B, offs): ['# comment', 'hipblaslt-bench ..', ..].

    hipblaslt-bench has no grouped problem to pin, so each DISTINCT group size gets the
    plain-mm line of that group (grouped_geoms); hipBLASLt picks the solution for the
    plain problem itself, so no --solution_index.  2026-11-09: the failing group of the
    qwen3 MoE capsule (13803 rows) reproduced as plain mm."""
    gg = grouped_geoms(cap)
    if not gg:
        return []
    a = cap.get("args") or []
    A = a[0] if a else None
    if isinstance(A, dict) and _CHAIN_REF in A:
        A = (cap.get("pool") or {}).get(A[_CHAIN_REF])
    t, c = _BENCH_DTYPE.get(str((A or {}).get("dtype")), (None, None))
    if t is None:
        return []
    if c is None:
        c = "xf32_r" if cap["precision"]["float32_matmul_precision"] != "highest" else "f32_r"
    n, groups = gg
    out = ["# aten._grouped_mm: no grouped problem in hipblaslt-bench -- the plain-mm line of each "
           "distinct group size (%d group(s), %d size(s)); the bad group is the replay's "
           "GROUPED_WRONG_ROWS" % (n, len(groups))]
    for m, gis, geom in groups:
        out.append("# group rows %d: group(s) %s" % (m, ",".join(map(str, gis))))
        out.append(bench_cmd(geom, dtype_tag=t, compute=c))
    return out


def out_tensors(out):
    """An op's result as a flat list of (index, tensor).

    Most ops this replays return a single tensor, but not all: an SDPA backward
    returns (grad_q, grad_k, grad_v), and torch.isfinite() raises TypeError on a
    tuple.  Non-tensor members (ints, None -- the logsumexp/philox slots some
    backward signatures carry) have no finiteness to check and are dropped.
    The index is kept so a NaN can be reported against the operand it came out
    of rather than against an anonymous "output".
    """
    if isinstance(out, torch.Tensor):
        return [(0, out)]
    if isinstance(out, (tuple, list)):
        return [(i, t) for i, t in enumerate(out) if isinstance(t, torch.Tensor)]
    return []


def scan_targets(func, argv, out):
    """[(label, tensor)] -- every tensor this op's RESULT can be read out of.

    An op's outputs are not only what it returns.  flash-attn's backward takes
    dq/dk/dv as positional `Tensor(a!)?` out-params and returns only softmax_d,
    so scanning the return value alone never looks at the gradients -- which is
    exactly where its NaN lives, and a replay that scans only `out` reports a
    clean 0/N no matter how long it runs.  out_tensors' docstring assumes the
    aten spelling, `_scaled_dot_product_*_backward`, which really does return
    (grad_q, grad_k, grad_v); a custom op is free not to, and this one does not.

    Written arguments are therefore scanned alongside the return, under the same
    op_writes() rule [1] uses to keep them OUT of the inputs verdict.  Same
    schema, opposite side of the call: garbage going in, the op's answer coming
    out.
    """
    res = [("out[%d]" % i, t) for i, t in out_tensors(out)]
    have = {id(t) for _, t in res}
    for i in sorted(op_writes(func)["idx"]):
        if i < len(argv):
            t = argv[i]
            if isinstance(t, torch.Tensor) and id(t) not in have:
                res.append(("arg%d" % i, t))
                have.add(id(t))
    return res


def describe(t, label):
    s = _scannable(t) if _scannable else t   # float8_e5m2 has no CUDA abs/isfinite
    finite = torch.isfinite(s).all().item()
    f = s[torch.isfinite(s)]
    amax = f.abs().max().item() if f.numel() else float("nan")
    print("  %-8s %-18s stride=%-16s %s  finite=%s  max|.|=%.6g"
          % (label, tuple(t.shape), tuple(t.stride()), t.dtype, finite, amax))
    return finite, amax


# ======================================================================================
# [4] mutation sweep
# ======================================================================================

# CLEAN first: those are the findings. TIMEOUT and ERROR last but never hidden -- a
# report that showed only the clean rows would make a sweep where everything timed
# out look exactly like a sweep where nothing was wrong.  SKIPPED sits with them for
# the same reason: a chain op whose kernel this cannot patch is a GAP in the sweep,
# and a table that omitted it would read as "every op was covered".
_OUTCOME_ORDER = {"CLEAN": 0, "NAN": 1, "TIMEOUT": 2, "ERROR": 3, "SKIPPED": 4}

# A CLEAN row only means something if the trial ran long enough that the fault would
# have shown up had the mutation not removed it.  At a measured per-launch hit rate p
# and n launches, the chance a still-faulty trial reports zero is exp(-p*n); below
# this many expected hits the row is not evidence and is labelled as such.
_MIN_EXPECTED_HITS = 3.0

_GEOM_KEY = ("M", "N", "K", "transA", "transB", "batch_count")

# A mutated kernel that still works may legitimately be slower than the original -- a
# NOPed instruction can cost a spill, a drained lgkmcnt serialises waves -- so a trial
# gets this multiple of what the baseline took.  Anything slower than that is not a
# kernel running slowly, it is a kernel that is not going to return.
_MUTATE_BUDGET = 2.0

# What --mutate-timeout falls back to when nothing has been measured yet: discovery
# runs before step [3], and --list-mutations --patch-kernel never replays at all.  The
# old fixed default, kept for exactly the cases where the new one cannot be computed.
_MUTATE_TIMEOUT_FALLBACK = 300.0

# Filled in by step [3], split the way a TRIAL CHILD spends its seconds rather than the
# way this process does: everything before the first launch, then the launches.
# `parent_only` accumulates the work no child repeats -- the fp64 CPU reference and [0]
# kernel discovery -- because a budget that charged every trial for an O(MNK) reference
# the trials are told to skip would be measuring the wrong process.
#
# `scale` corrects the one thing a single early measurement cannot: the baseline is
# taken once, before the first trial, and the trials run for the rest of the sweep.
# Measured on gfx950 with this capsule, an idle node reproduces the prediction almost
# exactly (a trial cost 14.8s against a predicted 14.3s), but with eight GPUs under a
# concurrent load the same trials came in around 4x their predicted per-launch rate and
# every row timed out.  The overlay is not the culprit -- it is the node moving under a
# number measured 10 minutes earlier -- which is why this is calibrated from the control
# rather than folded into _MUTATE_BUDGET: a constant big enough for the bad case would
# be four times too generous in the ordinary one.
#
# `rate` is the baseline's measured hit rate per launch (or per chain run), which is
# what makes a CLEAN row readable: at 0.12% per run, a 100-iteration trial would come
# back clean 89% of the time with the fault fully intact.  None when nothing measured
# it.
_BASELINE = {"setup": None, "per_launch": None, "parent_only": 0.0, "scale": 1.0,
             "rate": None}


# Whether the sweep keeps the co-tenants the baseline ran under.  The default is yes
# whenever contention was requested at all: a sweep that measures its baseline under
# load and its trials without it is not an A/B, it is two different experiments, and
# it fails in the direction that invents findings (identity NAN, everything else
# CLEAN).  --mutate-solo restores the old drop-after-baseline behaviour, which is the
# right call for a fault that reproduces solo, where load only risks an all-TIMEOUT
# sweep.
def sweep_holds_load():
    return bool(_MUTATING and not args.mutate_solo
                and (args.contend or args.contend_with))


# The payload a trial child runs: the chain loop when the baseline was a chain, the
# single op otherwise.  A capsule whose recorded operands are already corrupt makes
# the single op reproduce 100/100 for the wrong reason, so sweeping it would localise
# a fault that the chain shows is not there.
def sweep_payload_kind():
    return "chain" if args.chain else "single-op"


def sweep_trial_iters():
    """Launches (or chain runs) one trial child gets."""
    return (args.chain_iters or 1) if args.chain else args.iters


# Set by a sweep parent in each trial child's environment, naming the load the
# PARENT is holding on this device.  A trial child starts no co-tenants of its own
# (see child_cmd), so without this it would look at its own flags, conclude it ran
# alone, and print the "re-run with --contend-auto" warning into the log of a trial
# that was in fact fully contended.  That warning appears on CLEAN rows -- the ones a
# sweep exists to produce -- so getting it wrong there is the expensive case.
_PARENT_LOAD = os.environ.get("_NAN_SWEEP_CONTENDED") or None


# Set by scripts/jobs/replay_ladder.sh: this process is one of N copies of
# the same replay on its GPU.  The copies ARE the load, so a clean result is not the
# solo case and must not be reported as one.
_NODE_CONCURRENCY = int(os.environ.get("NAN_REPLAY_CONCURRENCY") or 0)


def effective_load():
    """A description of the load on this device, from any source, or None."""
    if args.contend_with:
        return args.contend_with
    if args.contend:
        return "%d co-tenants" % args.contend
    if _PARENT_LOAD:
        return "%s, held by the parent sweep" % _PARENT_LOAD
    if _NODE_CONCURRENCY > 1:
        return "%d concurrent copies on this GPU (node launcher)" % _NODE_CONCURRENCY
    return None


def mutate_timeout(iters=None):
    """Seconds one child replay of `iters` launches gets before it is killed.

    Resolved on every read rather than stored once, because the number legitimately
    differs between callers: a discovery replay does one launch and a trial does
    --iters of them, and giving the first the second's budget is how a wedged
    discovery child holds a sweep open for an hour -- and because `scale` changes
    underneath it once the control has reported.
    """
    if args.mutate_timeout is not None:
        return args.mutate_timeout                # pinned on the command line, verbatim
    if _BASELINE["per_launch"] is None:
        return _MUTATE_TIMEOUT_FALLBACK
    n = args.iters if iters is None else iters
    return (_MUTATE_BUDGET * _BASELINE["scale"]
            * (_BASELINE["setup"] + _BASELINE["per_launch"] * n))


def calibrate_from_trial(rec, launches):
    """Re-derive the budget from what a completed trial actually cost.

    The baseline in [3] is the only thing available before the first trial, and it
    describes the node as it was then.  A trial is a measurement taken under the
    conditions the remaining trials will actually run in -- same overlay, same
    neighbours, same however-many minutes into the job -- which is the only thing that
    can catch a node that got busier after [3] was timed.  The control is the first
    such measurement and the most valuable, because every other row still lies ahead of
    it; every later row refines the same number rather than being ignored.

    Scaled rather than re-modelled: with one data point, setup and per-launch cannot be
    separated, and `observed / predicted` needs no such separation.  Two properties are
    worth keeping in mind when reading the number it prints:

      - a control that STOPPED EARLY is measured over few launches, so its ratio is
        dominated by the overlay's fixed cost and over-states the slope.  That errs
        toward a budget that is too generous, which costs wall clock only on a trial
        that was going to be killed anyway -- the opposite error voids the whole sweep.
      - a trial that TIMED OUT reports the budget it was killed at, not its cost, so
        observed/predicted comes back as exactly _MUTATE_BUDGET x the current scale,
        and the next row gets twice what this one was denied.  That is the right move
        on censored evidence: the cost is known only to be larger than the budget, and
        doubling per timed-out row converges on it from below in log(n) rows rather
        than repeating one wrong guess for the length of the sweep.  It is also why
        this runs on every row and not only the control -- a control that times out
        yields the weakest possible reading, and a sweep that stopped learning there
        would re-run that same mistake on every mutation behind it.

    Never lowers the budget: a trial that beat its prediction says the prediction was
    safe, not that the next mutation -- which may add a spill or drain a wait -- will be
    just as quick.
    """
    if args.mutate_timeout is not None or _BASELINE["per_launch"] is None:
        return None
    n = args.iters if launches is None else launches
    predicted = _BASELINE["setup"] + _BASELINE["per_launch"] * n
    if predicted <= 0 or not rec.get("seconds"):
        return None
    ratio = rec["seconds"] / predicted
    if ratio <= _BASELINE["scale"]:
        return None
    _BASELINE["scale"] = ratio
    return ratio


def _budget_note():
    """Where an auto --mutate-timeout came from, in a parenthesis.

    A sweep that quietly chose its own per-trial budget is a sweep nobody can
    reproduce from its log, and the two auto cases are not equally trustworthy: one
    is measured, the other is the old constant wearing the same clothes.
    """
    if _BASELINE["per_launch"] is None:
        return "  (auto: no trial has been timed yet -- the fallback default)"
    return ("  (auto: %gx of %.1fs setup + %.4gs/launch x %d%s)"
            % (_MUTATE_BUDGET, _BASELINE["setup"], _BASELINE["per_launch"], args.iters,
               "" if _BASELINE["scale"] == 1.0
               else ", x%.2f from the control" % _BASELINE["scale"]))


def child_verdict(text):
    """The last '[verdict] X' line a child printed, or None.

    Parsed rather than inferred from the exit status on purpose: this script exits
    1 for NOT_REPRODUCED, which is precisely the outcome a mutation sweep is hunting
    for.  Reading the returncode alone would score every finding as a failure.
    """
    found = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[verdict] "):
            found = line[len("[verdict] "):].strip()
    return found


def child_launches(text):
    """How many launches a child actually ran, or None.

    Needed because --stop-on-nan makes a trial's launch count a RESULT rather than a
    setting: calibrating against --iters when the child stopped at launch 500 would
    read a 4-second trial as four seconds for the whole sweep's worth of work.

    None is the honest answer for a child that was killed -- its stdout is a pipe, so
    whatever it had buffered died with it -- and calibrate_from_control() falls back to
    --iters there, which is exactly the assumption that makes a TIMEOUT'd control
    double the budget instead of collapsing it.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        # "non-finite on 3/100 launches" (single op) or "... 24/20000 chain runs"
        # (a chain trial).  Both are the n/N this calibrates against; missing the
        # chain form would hand every chain trial launches=None, which the TIMEOUT
        # path reads as "killed" and doubles the budget on.
        if line.startswith("non-finite on ") and ("launches" in line
                                                  or "chain runs" in line):
            try:
                return int(line.split()[2].split("/")[1])
            except (IndexError, ValueError):
                return None
    return None


# hipBLASLt log mask bits.  64 (profile) carries solution_index; 128 (extended profile)
# carries kernel_name.  Neither alone has both, which is why discovery asks for 192
# first -- and why there has to be a plan B for libraries that cannot survive bit 128.
_MASK_PROFILE = 64
_MASK_EXT_PROFILE = 192


def _discovery_replay(mask):
    """One child replay under HIPBLASLT_LOG_MASK=mask -> (returncode, log path, output).

    returncode is None if the child ran out of time.  The child's own output is
    returned rather than dropped on the floor: a replay that dies on a signal says so
    nowhere else, and a discovery step that cannot tell "logged nothing" from "crashed
    before it could log" reports the wrong failure -- and sends the user off to
    --patch-kernel for what is actually a library bug.
    """
    import subprocess
    import tempfile

    fd, log = tempfile.mkstemp(prefix="mutate_dispatch_", suffix=".log")
    os.close(fd)
    env = dict(os.environ)
    env["HIPBLASLT_LOG_MASK"] = str(mask)
    env["HIPBLASLT_LOG_FILE"] = log
    try:
        p = subprocess.run(child_cmd(iters=1), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True,
                           timeout=mutate_timeout(iters=1))
    except subprocess.TimeoutExpired:
        return None, log, ""
    return p.returncode, log, p.stdout


def _discovery_failure(rc, log, out):
    """Why a discovery replay yielded no kernel name, in one phrase."""
    import signal

    if rc is None:
        return "it did not finish within %.0fs" % mutate_timeout(iters=1)
    if rc < 0:
        sig = -rc
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = "signal %d" % sig
        extra = ""
        if sig == signal.SIGSEGV:
            extra = (" -- the getKernelNameFromAlgoIndex() crash that bit 128 triggers"
                     " in some hipBLASLt builds, before any GEMM completes")
        return "it died on %s%s" % (name, extra)
    tail = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if rc != 0:
        return "it exited %d%s" % (rc, (": " + tail[-1]) if tail else "")
    # rc == 0 is weaker evidence than it looks: in the ROCm 7.2 image an atexit
    # callback registered by `import torch` resets the status, so every sys.exit(N)
    # from a child that imported torch arrives here as 0.  A signal death still
    # reports honestly (it never reaches atexit), which is what the SIGSEGV branch
    # above relies on -- but a clean-looking 0 may still be hiding a traceback.
    if "Traceback (most recent call last)" in out:
        return "it raised%s" % ((": " + tail[-1]) if tail else "")
    if not os.path.isfile(log) or not os.path.getsize(log):
        return "it exited cleanly but wrote nothing to %s" % log
    return "the profile in %s carries no kernel_name field" % log


def _dispatches(log, g):
    """(kernel names, solution index) for the capsule's geometry in a profile log."""
    import replay_hipblaslt_log_with_nan as rnan

    if not os.path.isfile(log) or not os.path.getsize(log):
        return [], None
    try:
        logged = rnan.parse_log(log)
    except SystemExit:
        return [], None

    if not g:
        sel = _grouped_dispatches(logged)
        if sel is not None:
            return sel
    # The log also carries whatever GEMMs torch ran while warming up. Patching one of
    # those would give a sweep in which every row is NAN for an uninteresting reason.
    matched = logged
    if g:
        key = tuple(str(g.get(k, 1)) for k in _GEOM_KEY)
        hits = [x for x in logged if x.key == key]
        if hits:
            matched = hits
        else:
            print("  note: no logged GEMM matches the capsule geometry %s;"
                  % " ".join("%s=%s" % (k, v) for k, v in zip(_GEOM_KEY, key)))
            print("        considering all %d logged dispatch(es) instead." % len(logged))

    return _names_index(matched)


def _names_index(matched):
    """(distinct kernel names in log order, the solution index OF THE LAST NAME -- the one
    discover_kernel returns; it used to be the first dispatch's, a different kernel's)."""
    names = []
    for x in matched:
        if x.kernel and x.kernel not in names:
            names.append(x.kernel)
    last = [x for x in matched if names and x.kernel == names[-1]] or matched
    index = next((x.get("solution_index") for x in reversed(last)
                  if x.get("solution_index") is not None), None)
    return names, index


_GROUPED_NOTE = [""]        # why discover_kernel's solution index is (or is not) the wrong group's


def _grouped_wrong_from(out):
    """A discovery child's `GROUPED_WRONG_ROWS=` line -> _GROUPED_WRONG, when this process has
    not run the value check itself (--disasm runs discovery before its own replay)."""
    import re
    if _GROUPED_WRONG or not out:
        return
    m = re.search(r"GROUPED_WRONG_ROWS=([\d,]+)", out)
    if m:
        _GROUPED_WRONG.extend(int(x) for x in m.group(1).split(",") if x)


def _grouped_dispatches(logged):
    """aten._grouped_mm: (names, index) of the WRONG group's dispatch, or None (not a
    grouped capsule, or no group matched).  Each group is its own GEMM and hipBLASLt
    picks a solution per group size -- 8 different kernels for 16 groups (qwen3 MoE,
    2026-11-14), where the first logged one named a correct group's solution (13813, not
    the wrong 12889-row group's 13817).  The wrong group comes from this process's
    value check (GROUPED_WRONG_ROWS) or --grouped-wrong-rows."""
    cap = _CAP[0]
    gg = grouped_geoms(cap) if cap and cap.get("op") == "aten._grouped_mm.default" else None
    if not gg:
        return None
    n, groups = gg
    want = set(_GROUPED_WRONG)
    for x in (getattr(args, "grouped_wrong_rows", None) or "").split(","):
        if x.strip().isdigit():
            want.add(int(x))
    print("  grouped GEMM: %d group(s), %d distinct size(s); the solution each size dispatched:"
          % (n, len(groups)))
    per = []
    for m, gis, geom in groups:
        key = tuple(str(geom.get(k, 1)) for k in _GEOM_KEY)
        hits = [x for x in logged if x.key == key]
        names, index = _names_index(hits)
        per.append((m, names, index))
        print("    group rows %d (group(s) %s): solution %s%s"
              % (m, ",".join(map(str, gis)), "?" if index is None else index,
                 "   <- WRONG" if m in want else ""))
    hit = [p for p in per if p[0] in want and p[1]]
    if hit:
        _GROUPED_NOTE[0] = "the wrong group's: rows %d" % hit[0][0]
        if len({p[2] for p in hit}) > 1:
            print("  note: the wrong groups dispatched %d different solutions; naming rows %d's"
                  % (len({p[2] for p in hit}), hit[0][0]))
        return [hit[0][1][-1]], hit[0][2]
    if want:
        print("  the wrong group(s) (rows %s) matched no logged dispatch"
              % ",".join(map(str, sorted(want))))
    _GROUPED_NOTE[0] = ("ONE group's -- which group was wrong is not known here: the replay's "
                        "GROUPED_WRONG_ROWS=, passed as --grouped-wrong-rows, names it")
    print("  which group is wrong is not known here: the kernel below is ONE group's, not "
          "necessarily the wrong one (pass --grouped-wrong-rows <rows> from the replay's "
          "GROUPED_WRONG_ROWS= line)")
    # Never fall back to every logged GEMM: the replay's own fp32 reference/check GEMMs are
    # in the log too, and the last of those was disassembled for a bf16 grouped capsule
    # (2026-11-27, an fp32 MT32x32x128 kernel).
    names = []
    for _, nm, _ in per:
        names += [x for x in nm if x not in names]
    last = [p[2] for p in per if p[1] and p[1][-1] == names[-1]] if names else []
    return (names, last[0] if last else None) if names else None


def discover_kernel(g):
    """The Tensile kernel this capsule's GEMM actually dispatched to, or None.

    Which solution wins is a property of the installed library, not of the problem,
    so the capsule cannot record it and the only honest source is hipBLASLt itself.
    That is why this costs a whole extra child replay instead of piggy-backing on step
    [3]: the profile is written when the library shuts down, and a process cannot read
    its own profile, because at the moment it wants to, nothing has been flushed yet.

    Asking the library directly (mask 192) stays the first move -- it is the library's
    own answer, and it is right by construction on any build where it works.  Where it
    does not, bit 128 takes the process down with SIGSEGV on the first matmul and the
    empty log is the only evidence; plan B is then bit 64, which never crashes, plus an
    offline lookup of the solution index it does report.  See
    rnan.kernel_name_for_index.
    """
    import replay_hipblaslt_log_with_nan as rnan

    print("  identifying the dispatched kernel (one extra replay, run to exit so the")
    print("  library flushes its profile)")
    rc, log, out = _discovery_replay(_MASK_EXT_PROFILE)
    _grouped_wrong_from(out)
    names, index = _dispatches(log, g)
    if names:
        if len(names) > 1:
            print("  note: %d different kernels served this geometry; taking the last."
                  % len(names))
            for n in names:
                print("        %s" % n)
            print("        Pin one with --patch-kernel if that is not what you meant.")
        if index is not None:
            print("  solution_index: %s (as dispatched%s)"
                  % (index, "; " + _GROUPED_NOTE[0] if _GROUPED_NOTE[0] else ""))
        return names[-1]

    print("  the extended profile (mask %d) gave no kernel name: %s"
          % (_MASK_EXT_PROFILE, _discovery_failure(rc, log, out)))
    print("  falling back to the plain profile (mask %d) plus an offline lookup of the"
          % _MASK_PROFILE)
    print("  solution index in the installed Tensile *.dat set")

    rc, log, out = _discovery_replay(_MASK_PROFILE)
    _grouped_wrong_from(out)
    names, index = _dispatches(log, g)
    if names:                                     # a build whose bit 64 names it anyway
        return names[-1]
    if index is None:
        print("  the plain profile reported no solution index either: %s"
              % _discovery_failure(rc, log, out))
        print("  Name the kernel with --patch-kernel.")
        return None
    print("  solution_index: %s (as dispatched)" % index)

    try:
        arch = args.patch_arch or rnan.gpu_arch([])
        kernel, dat = rnan.kernel_name_for_index(index, args.patch_libdir, arch)
    except (rnan.IndexLookupError, SystemExit) as e:
        print("  could not map solution index %s to a kernel name: %s" % (index, e))
        print("  Name the kernel with --patch-kernel.")
        return None
    print("  resolved offline from %s (arch %s)" % (dat, arch))
    return kernel


def discover_aiter_kernels():
    """[(co_path, symbol)] the capsule's replay actually loads from aiter, in order.

    The aiter analogue of discover_kernel(), and a far more direct one: there is no
    profile to parse because the loader PRINTS ITS OWN DISPATCH.  All this has to do
    is run one replay with AITER_ASM_DIR set and read the trace back.

    Setting the variable is not optional and not merely a convenience.  Unset, aiter
    takes its embedded branch, prints an unprefixed relative path, and serves a copy
    baked into the .so; kp.parse_aiter_loads deliberately ignores that form, because
    reporting it as a dispatch would hand the sweep a file it is not running.  Empty
    here therefore means "no FILE-backed aiter kernel", which is the honest input to
    a SKIPPED row.

    Returns [] on any failure.  Discovery is best-effort by design -- the caller
    falls back to SKIPPED, which is a real verdict, rather than failing the rung.
    """
    import subprocess
    import kernel_patch as kp

    if kp.ensure_aiter_asm_dir(args.patch_asm_dir) is None:
        return []
    print("  identifying the dispatched aiter kernel(s) (one extra replay; the aiter")
    print("  loader prints each hipModuleLoad, so no profile is involved)")
    env = dict(os.environ, AITER_LOG_LEVEL="INFO")     # its load lines ARE the discovery
    try:
        p = subprocess.run(child_cmd(iters=1), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True,
                           timeout=mutate_timeout(iters=1))
    except subprocess.TimeoutExpired:
        print("  the discovery replay did not finish within %.0fs"
              % mutate_timeout(iters=1))
        return []
    loads = kp.parse_aiter_loads(p.stdout)
    if not loads:
        if kp.trace_is_file_backed(p.stdout) is False:
            # The one failure that must never be silent: it looks exactly like "this
            # op is not aiter", but it means the overlay would have been ignored.
            print("  aiter loaded from its EMBEDDED map despite AITER_ASM_DIR being"
                  " set -- a patched")
            print("  .co would not be served and every mutator would come back CLEAN."
                  " Not sweeping.")
        else:
            print("  no file-backed aiter kernel in this replay (rc=%s)" % p.returncode)
        return []
    for path, sym in loads:
        print("    %s  (%s)" % (path, sym))
    return loads


def _shader_names(text):
    """Every `ShaderName : <demangled>` the AMD log printed, deduplicated.

    Parsed with str ops rather than a module-level regex because this file imports
    neither `re` nor `tempfile` at module scope, and adding them for one caller would
    be a wider change than the feature needs.
    """
    out = set()
    for line in text.splitlines():
        i = line.find("ShaderName")
        if i < 0:
            continue
        j = line.find(":", i)
        if j > 0:
            name = line[j + 1:].strip()
            if name:
                out.add(name)
    return out


def discover_ck_kernels(match="FmhaFwdKernel"):
    """[(co_path, symbol)] for kernels this capsule dispatches that the HSA shim can
    reach -- CK, and anything else compiled into a library.

    The third sibling of discover_kernel (hipBLASLt profile) and
    discover_aiter_kernels (loader trace).  CK publishes neither, so this uses two
    sources at once in ONE child replay:

      AMD_LOG_LEVEL=3    `ShaderName :` names the kernels that actually DISPATCHED.
                         A code object can be loaded without its kernel ever running,
                         and only a dispatched kernel is worth a sweep.
      the shim's dump    supplies the exact bytes, so the offsets Python computes are
                         applied later to the same bytes they were measured on.

    `match` narrows the dump: without it the shim would write every code object the
    process loads, one of which is a 184 MB torch device image.

    Returns [] on any failure -- discovery is best-effort and the caller falls back to
    a SKIPPED row, which is a real verdict.
    """
    import subprocess
    import tempfile
    import kernel_patch as kp

    print("  identifying the dispatched CK kernel(s) (one extra replay under the HSA")
    print("  shim: AMD_LOG_LEVEL names what dispatched, the dump supplies the bytes)")
    workdir = args.patch_workdir or tempfile.mkdtemp(prefix="mutate_")
    dump = os.path.join(workdir, "ck_dump")
    os.makedirs(dump, exist_ok=True)
    try:
        shim = kp.build_hsa_shim(os.path.join(workdir, "hsa_patch_shim.so"))
    except kp.PatchError as e:
        print("  %s" % e)
        return []

    env = dict(os.environ)
    pre = env.get("LD_PRELOAD", "").strip()
    env["LD_PRELOAD"] = (pre + " " + shim).strip() if pre else shim
    env["NUMERIC_DOCTOR_HSA_DUMP"] = dump
    env["NUMERIC_DOCTOR_HSA_MATCH"] = match
    env["NUMERIC_DOCTOR_HSA_LOG"] = os.path.join(workdir, "discover_shim.log")
    env["AMD_LOG_LEVEL"] = "3"
    try:
        p = subprocess.run(child_cmd(iters=1), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True,
                           timeout=mutate_timeout(iters=1))
    except subprocess.TimeoutExpired:
        print("  the discovery replay did not finish within %.0fs"
              % mutate_timeout(iters=1))
        return []

    dispatched = _shader_names(p.stdout)
    cos = sorted(os.path.join(dump, f) for f in os.listdir(dump) if f.endswith(".co"))
    if not cos:
        print("  no code object matching %r was loaded (rc=%s)" % (match, p.returncode))
        return []

    out = []
    for co in cos:
        for _addr, sym in kp._ElfFinder().funcs(co):
            # The ShaderName line is DEMANGLED and the symbol is not, so they cannot be
            # compared directly.  Both carry the template's distinguishing tokens, so
            # require the kernel family and arch to appear in something that actually
            # dispatched -- enough to reject a loaded-but-never-run kernel, which is
            # the case this check exists for.
            if any(match in d for d in dispatched):
                out.append((co, sym))
            else:
                print("  note: %s is in a loaded code object but nothing matching %r"
                      % (sym[:60], match))
                print("        appears in a ShaderName line -- not sweeping it.")
    for co, sym in out:
        print("    %s  (%s...)" % (os.path.basename(co), sym[:56]))
    return out


# ======================================================================================
# [0] the dispatched kernel's ISA, as installed
#
# Shares discover_kernel() with the sweep below, which is why it lives down here: the
# hard part of writing this file is not the writing, it is knowing WHICH of the
# library's thousands of kernels the capsule's geometry actually lands on.
# ======================================================================================

def write_disasm(g, out_path, kernel=None):
    """Write the unmodified kernel's disassembly to out_path; return its name or None.

    Distinct from --mutate-disasm, which is a before/after pair read back out of an
    overlay: this is the kernel as shipped, and it is what a mutation diff is a diff
    AGAINST -- reading one without the other tells you which slots changed but not
    what the kernel was doing with them.

    Never fatal.  A missing disassembly makes the replay less legible, not less true,
    so a failure here prints why and lets steps [1]-[3] run; they are the verdict.
    The kernel name is returned so that --disasm --mutate pays for discovery once.
    """
    import shutil
    import tempfile
    import kernel_patch as kp

    print("\n[0] kernel disassembly -> %s" % out_path)
    if kernel is None:
        kernel = args.patch_kernel or discover_kernel(g)
    if not kernel:
        return None
    print("  kernel:  %s" % kernel)

    # An explicit --patch-workdir is the user's, and --keep-workdir asks for the
    # unbundled ELF to survive; anything we made ourselves goes away again.
    workdir = args.patch_workdir or tempfile.mkdtemp(prefix="disasm_")
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    keep = bool(args.patch_workdir or args.keep_workdir)
    target = None
    try:
        target = kp.PatchTarget(kernel, args.patch_libdir, args.patch_arch, workdir)
    except kp.PatchError as e:
        print("  %s" % e)
    except SystemExit as e:                       # rnan uses sys.exit() for errors
        print("  cannot disassemble %s: rnan exited (%r)" % (kernel, e.code))
    except Exception as e:                        # noqa: BLE001 -- never fatal
        print("  cannot disassemble %s: %s: %s" % (kernel, type(e).__name__, e))
    if target is None:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
        return None

    print("  arch:    %s   library %s" % (target.arch, target.libdir))
    print("  %d instruction(s) disassembled from 0x%X-0x%X"
          % (len(target.insns), target.start, target.stop))
    try:
        d = os.path.dirname(os.path.abspath(out_path))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(out_path, "w") as f:
            f.write("; %s -- as installed, unmodified\n" % target.kernel)
            f.write("; capsule  %s\n" % os.path.abspath(args.capsule))
            if g:
                f.write("; problem  M=%s N=%s K=%s transA=%s transB=%s batch_count=%s\n"
                        % tuple(g.get(k, "?") for k in _GEOM_KEY))
            # The installed code object, not target.elf: when it is a bundle the ELF
            # is a temp file this function deletes on its way out, and a header line
            # naming a path that no longer exists is worse than no header line.
            f.write("; library  %s%s\n"
                    % (target.libfile,
                       "  (bundled; unbundled to disassemble)" if target.bundled else ""))
            f.write("; arch %s   0x%X-0x%X   %d instruction(s)\n\n"
                    % (target.arch, target.start, target.stop, len(target.insns)))
            f.write("\n".join(i.line for i in target.insns) + "\n")
    except OSError as e:
        print("  could not write --disasm %s: %s" % (out_path, e))
        return kernel
    finally:
        if not keep:
            shutil.rmtree(target.workdir, ignore_errors=True)
    print("  wrote %s" % out_path)
    return kernel


# ======================================================================================
# [4] mutation sweep, continued
# ======================================================================================

def enumerate_mutations(kernel, hsaco=None, co=None, co_symbol=None):
    """(target, [(mutation, sites, error)]) for one kernel, without touching a GPU.

    Resolving the sites here rather than inside the trial loop is what makes
    --list-mutations worth running: a selector that matches nothing is a broken
    mutator, and finding that out costs seconds here and a whole trial slot there.

    `hsaco` switches the target to a standalone aiter .co; `co` to a code object
    dumped by the HSA shim (CK).  Everything after the construction is identical --
    the mutators read a MutationSpec, which carries insns/arch/start/stop and does not
    know or care where the bytes came from -- which is why three backends are three
    small classes and not three forks of the sweep.
    """
    import tempfile
    import kernel_patch as kp
    import kernel_mutations as km

    workdir = args.patch_workdir or tempfile.mkdtemp(prefix="mutate_")
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    if co:
        target = kp.HsaTarget(co, co_symbol, args.patch_arch, workdir)
    elif hsaco:
        target = kp.AiterTarget(hsaco, args.patch_asm_dir, args.patch_arch, workdir)
    else:
        target = kp.PatchTarget(kernel, args.patch_libdir, args.patch_arch, workdir)

    only = list(args.mutate_only)
    if only and "identity" not in only and not args.mutate_no_identity:
        # The control is not opt-in: without it a CLEAN row cannot be told apart
        # from an overlay that quietly stopped applying.
        only.append("identity")
        print("  note: added the identity control to --mutate-only")

    spec = km.MutationSpec.from_target(
        target, solution_index=(args.solution_index if args.solution_index is not None
                                else args.patch_solution_index))
    rows = []
    for m in km.kernel_mutations(spec, only=only or None):
        if args.mutate_no_identity and m.name == "identity":
            continue
        try:
            sites = target.select(m.selector, m.opts)
            err = None if sites else "selector %r matched no instruction" % m.selector
        except kp.PatchError as e:
            sites, err = [], str(e)
        except SystemExit as e:
            sites, err = [], "rnan exited (%r) while selecting" % (e.code,)
        rows.append((m, sites, err))
    return target, rows


def print_mutation_plan(target, rows):
    width = max([len(m.name) for m, _s, _e in rows] + [8])
    print("\n%-*s  %-12s %-9s %6s  %s"
          % (width, "mutation", "selector", "payload", "sites", "description"))
    print("-" * (width + 50))
    for m, sites, err in rows:
        print("%-*s  %-12s %-9s %6s  %s"
              % (width, m.name, m.selector, m.payload,
                 "FAIL" if err else len(sites), err or m.description))
    live = [r for r in rows if not r[2]]
    # The trailing "at --mutate-timeout Ns" is a parsed interface, not prose:
    # check_replay_nan_logs.py lifts the sweep's timeout out of this exact line, which
    # is the lever on every truncated sweep it reports.  Keep the shape.
    tmo = mutate_timeout()
    print("\n%d mutation(s), %d enumerable; worst case %.1f h at --mutate-timeout %.0fs%s"
          % (len(rows), len(live), len(live) * tmo / 3600.0, tmo,
             "" if args.mutate_timeout is not None else _budget_note()))
    if args.mutate_timeout is None and _BASELINE["per_launch"] is None:
        # A listing measures nothing, so that figure is the fallback timeout times the trial
        # count -- a ceiling no real sweep approaches (2026-11-20: "8.4 h" for trials of
        # 8-45 s).  Plan from the sweep's own baseline line instead.
        print("  (no trial timed: that is the fallback timeout x the trial count, a ceiling, not an"
              " estimate -- a trial ends at its run count or its first hit;")
        print("   the sweep prints the real per-trial budget after its baseline, from the 5b rate"
              " and the capsule's per-run time)")
    if len(live) != len(rows):
        print("%d mutation(s) cannot be built and would be ERROR rows in a sweep."
              % (len(rows) - len(live)))
    print_waitcnt_scan(target)


def print_waitcnt_scan(target):
    """The static wait-count hazard scan of the same kernel (waitcnt_scan.py): the sites step 7's
    first job tests as a set (`lgkmcnt0_subset` / `vmcnt0_subset` with NAN_MUTATE_SUBSET_ADDRS)."""
    try:
        import waitcnt_scan as ws
        text = "\n".join(getattr(i, "line", "") or "" for i in target.insns)
        insns = ws.parse_listing(text)
        if not insns:
            print("\nWAITCNT_SCAN=ERROR the listing kept no raw objdump lines to scan")
            return
        lines, _sites = ws.report(insns, ws.scan(insns))
        print("\nstatic wait-count scan (waitcnt_scan.py; orders the sweep, proves nothing):")
        for ln in lines:
            print("  " + ln if not ln.startswith(("WAITCNT_SCAN=", "NAN_MUTATE_SUBSET")) else ln)
    except Exception as e:                              # noqa: BLE001 - a listing aid, never fatal
        print("\nWAITCNT_SCAN=ERROR %s: %s" % (type(e).__name__, str(e)[:200]))


def child_cmd(iters=None):
    """Re-exec this script on the same capsule, without --mutate.

    --no-reference is forced: the fp64 CPU reference is O(MNK) and the parent has
    already established that the true result is finite and representable.  Re-running
    it per trial would make the reference, not the op, the thing being timed.
    --restore-env is deliberately absent -- the parent already re-exec'd under the
    capsule's environment, so the child inherits it through os.environ and cannot
    clobber the overlay path on its way back in.

    --stop-on-nan is forced for the same reason --no-reference is: a trial is scored
    from its [verdict] line, and that line is decided the moment one launch comes back
    non-finite.  The launches after it cannot change NAN into CLEAN, so on a sweep
    where most mutations do not remove the fault they are most of the GPU time.  It
    costs the child its n/N rate, which no trial reports anyway -- the baseline, which
    never passes this, is where that number is measured.  Harmless on the discovery
    caller below, which already replays exactly once.

    Two things the child inherits that it did not used to:

      - THE CHAIN PAYLOAD.  When the baseline was a chain, a trial that replayed the
        single op would be scoring a different experiment than the one that produced
        the baseline -- and on a capsule whose recorded operands are themselves
        corrupt, the wrong one.  --chain-loop-only, not the full walk: [C1]'s fp64
        per-op comparison is host work that answers a question no trial asks.
    What it deliberately does NOT inherit is --contend.  The co-tenants belong to the
    PARENT and keep running across every trial (see sweep_holds_load), so the device a
    trial child lands on already carries them: N co-tenants plus this child, which is
    the same N+1 geometry the baseline was measured at.  A child that started its own
    would square the load.  It also matters that the parent's co-tenants were spawned
    from plain os.environ and therefore run the STOCK library -- they are load, not
    subjects, so a CLEAN row cannot be a co-tenant that stopped driving the race.
    """
    n = iters if iters is not None else sweep_trial_iters()
    cmd = [sys.executable, os.path.abspath(__file__), args.capsule, "--no-reference"]
    if args.chain:
        # Also on the discovery caller (iters=1), and that is load-bearing: discovery
        # reads which Tensile kernel served a given problem key out of a hipBLASLt
        # profile, and a kernel only appears there if something DISPATCHED it.  A
        # single-op child runs the capsule's named op alone, so every op upstream of
        # it would be absent from the log and its sweep would be skipped for the
        # wrong reason.  One chain run dispatches all of them.
        cmd += ["--chain", "--chain-loop-only", "--chain-iters", str(n),
                "--stop-on-nan"]
    else:
        cmd += ["--iters", str(n), "--stop-on-nan"]
    if args.solution_index is not None:
        cmd += ["--solution-index", str(args.solution_index)]
    if args.no_hipblaslt:
        cmd.append("--no-hipblaslt")
    return cmd + _synth_args()


def _safe_name(name):
    """A mutation name usable as a filename. Names carry '@' and hex, nothing worse."""
    return "".join(c if (c.isalnum() or c in "_.@-") else "_" for c in name)


def _kernel_tag(kernel, limit=40):
    """A short, stable, filename-safe discriminator for one kernel.

    The row name is not enough to name a file by.  One chain op can dispatch SEVERAL
    kernels -- the SDPA backward reaches three aiter .co -- and run_mutations runs
    once per kernel, so `sbarrier_ds_write_read@260` is emitted twice and
    `identity@260` three times.  _journal_row_key already keys on the kernel for
    exactly this reason; --mutate-asm-dir did not, so the listings overwrote each
    other and the surviving pair belonged to whichever kernel was swept last.  Since
    a CLEAN row is reported FROM its diff, that is a finding citing another kernel's
    ISA -- the failure the "read the diff back, do not just cite it" rule exists to
    catch, arriving in a form that rule cannot see.

    Readable where the kernel names itself (an aiter .co) and hashed where it cannot
    (a mangled CK symbol or a Tensile kernel name runs past a filename's limit).
    """
    base = os.path.basename(kernel.rstrip("/")) or kernel
    if len(base) <= limit:
        return _safe_name(base)
    import hashlib
    return _safe_name(base[:limit]) + "-" + hashlib.sha1(
        kernel.encode()).hexdigest()[:8]


def want_asm():
    """Whether a CLEAN row should be read back out of the overlay."""
    return args.mutate_disasm != "none" or bool(args.mutate_asm_dir)


def capture_asm(cache, target, rec):
    """Re-disassemble the overlay that just produced a CLEAN verdict.

    Costs one unbundle plus one objdump of a 43 MB ELF, which is why it runs per
    FINDING and not per row.  Never fatal: an unreadable overlay makes a finding
    less legible, not less true, so the failure lands on the row and the sweep
    carries on.
    """
    import kernel_patch as kp

    try:
        after = cache.disassemble()
    except SystemExit as e:                       # rnan uses sys.exit() for errors
        rec["asm_error"] = "rnan exited (%r) while disassembling the overlay" % (e.code,)
        return None
    except Exception as e:                        # noqa: BLE001 -- per-trial, never fatal
        rec["asm_error"] = "%s: %s" % (type(e).__name__, e)
        return None

    rec["disasm"] = [{"addr": "0x%X" % ad, "before": bt, "after": at}
                     for ad, bt, at in kp.diff_disasm(target.insns, after)]
    # Keyed on the payload rather than the mutation name: what makes an empty diff
    # expected is writing the original bytes back, and any future control built on
    # the identity payload inherits that for the same reason.
    if not rec["disasm"] and rec["payload"] != "identity":
        # The one thing the outcome table cannot tell you. A row that says CLEAN
        # while the kernel on disk is byte-identical to the original did not fix
        # anything -- the overlay stopped being applied, and the identity control
        # would have to be NAN by luck for that to go unnoticed.
        #
        # The identity control is exempt because for it this check cannot fail:
        # kernel_patch.payload_identity splices the original encoding back in, so an
        # empty diff is what a WORKING overlay produces, and the warning would fire
        # on every identity row that reached here -- pure noise, and misleading noise
        # at that.  The signal it was reaching for is not lost: report() already
        # flags a CLEAN identity control in its own right, and does it from the
        # outcome, which is where that question can actually be answered.
        rec["asm_error"] = ("the mutated kernel disassembles IDENTICALLY to the "
                            "original -- the patch did not reach the GPU, so this "
                            "CLEAN row is a harness failure, not a finding")
    return after


# ---- the sweep journal -----------------------------------------------------
# A trial is the expensive unit in this file: a contended chain trial is tens of
# minutes, and report() writes --mutate-report only after the LAST one.  So a sweep
# that dies partway -- a reaped /tmp, a dropped ssh, an OOM, a killed container --
# threw away every completed trial and the next run started from zero.  The journal
# removes that: one JSON object per trial, appended and fsync'd the moment the trial
# classifies, so the next run re-measures only what was never measured.
#
# JSONL rather than a rewritten .json on purpose.  An append that is cut short by a
# kill loses at most the final partial line, which the reader drops; rewriting a whole
# document each time has a window in which the file is truncated and the completed
# trials are gone -- which is the very failure this is here to prevent.
_JOURNAL = {"path": None, "loaded": {}, "reused": [], "started": False}


def _journal_path():
    """--mutate-journal, else --mutate-report with a .jsonl suffix, else off.

    Derived rather than required because the failure it prevents is one nobody
    anticipates until it has already cost them the sweep: a user who asked for a
    report wanted the rows to survive, and making them pass a second flag to get
    that is how the default stays wrong.
    """
    if args.mutate_journal:
        return args.mutate_journal
    if args.mutate_report:
        base = args.mutate_report
        return (base[:-len(".json")] if base.endswith(".json") else base) + ".jsonl"
    return None


def _journal_key():
    """What has to match for a journalled row to still describe THIS experiment.

    Deliberately not the mutator set: adding a mutator should reuse the rows already
    measured and run only the new one, which is the whole point.  Deliberately IS the
    host -- rates here are properties of (script, image, host, iters, contention), and
    a row carried silently across nodes is the one mistake this file cannot detect
    afterwards.  --mutate-timeout is also out: it does not change what a NAN or CLEAN
    row means, and a raised timeout is handled by --mutate-retry-incomplete instead.
    """
    import socket
    cap = os.path.abspath(args.capsule)
    try:
        size = os.path.getsize(cap)
    except OSError:
        size = None
    return {"capsule": cap, "capsule_bytes": size, "host": socket.gethostname(),
            "payload_kind": "chain" if args.chain else "single-op",
            "launches": args.chain_iters if args.chain else args.iters,
            "contend_n": args.contend, "contend_with": args.contend_with,
            "mutate_solo": bool(args.mutate_solo),
            # The marker, not the flag: --restore-env re-execs with itself stripped
            # from argv, so by the time a sweep runs, args.restore_env is False in
            # the very process the flag worked.  Keying on the flag would read False
            # for both arms and let rows measured under the capsule's environment
            # resume into a sweep running under the container's -- which is the one
            # difference that turns a kernel result into a different experiment.
            "restore_env": bool(args.restore_env or os.environ.get(_REEXEC))}


def _journal_init():
    """Open the journal, and load any rows a previous run of this sweep left behind.

    Returns the number of rows available for reuse.  A key mismatch is never a silent
    overwrite and never a silent reuse: it is reported and the old file is left alone
    under a .superseded name, because its rows are still a real measurement of a
    different experiment and deleting them is not this function's call.
    """
    import json

    path = _journal_path()
    _JOURNAL["path"] = path
    if not path:
        print("  journal: off (no --mutate-journal and no --mutate-report) -- a sweep"
              " that dies will lose every completed trial")
        return 0
    # report() makes this directory, but not until the sweep ENDS -- and the journal's
    # whole purpose is to have written rows before that.  So make it here, at the
    # first append, or every row is lost to the ENOENT the user only sees in passing.
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            print("  !! journal: cannot create %s (%s) -- this sweep is NOT resumable"
                  % (d, e))
            _JOURNAL["path"] = path = None
            _JOURNAL["started"] = True
            return 0
    key = _journal_key()
    rows, legacy, old_key, bad = {}, {}, None, 0
    if os.path.exists(path):
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        bad += 1           # a torn final line: the crash we survived
                        continue
                    if obj.get("_journal") == "key":
                        old_key = obj.get("key")
                    elif obj.get("_jkey"):
                        rows[obj["_jkey"]] = obj
                    elif obj.get("name"):
                        legacy.setdefault(obj["name"], []).append(obj)
        except OSError as e:
            print("  journal: cannot read %s (%s); starting a new one" % (path, e))
            rows, legacy, old_key = {}, {}, None
    # Rows written before _jkey existed carry only a name.  That is unambiguous when
    # the name occurs once, and unusable when it does not -- a chain op dispatching
    # three .co wrote three "identity@260" rows and nothing in them says which kernel
    # each belongs to.  Reuse the unique ones, drop the rest rather than guess.
    for nm, objs in legacy.items():
        if len(objs) == 1:
            rows.setdefault(_journal_row_key(nm, None), objs[0])
    ambiguous = sum(len(o) for o in legacy.values() if len(o) > 1)
    if ambiguous:
        print("  journal: %d legacy row(s) name a trial that was emitted for more than"
              " one kernel" % ambiguous)
        print("           (%s) -- they predate the per-kernel key and cannot be"
              % ", ".join(sorted(n for n, o in legacy.items() if len(o) > 1)))
        print("           attributed, so they will be re-measured rather than guessed at.")
    if old_key is not None and old_key != key:
        diff = [k for k in key if old_key.get(k) != key.get(k)]
        stale = path + ".superseded"
        print("\n  !! journal %s describes a DIFFERENT experiment; not resuming from it."
              % path)
        for k in diff:
            print("  !!   %-14s journal=%r  now=%r" % (k, old_key.get(k), key.get(k)))
        print("  !! Its %d row(s) were measured under that, not this. Moved to %s"
              % (len(rows), stale))
        try:
            os.replace(path, stale)
        except OSError as e:
            print("  !! could not move it (%s) -- appending anyway, which will mix two"
                  " experiments in one file" % e)
        rows, old_key = {}, None
    if bad:
        print("  journal: %d unparseable line(s) in %s dropped (expected after a kill)"
              % (bad, path))
    # A process killed mid-write leaves a final line with no newline on it.  Appending
    # straight onto that would glue the next row to the torn one and lose BOTH -- the
    # dropped line above, plus a trial this run actually paid for.  Seal it first.
    _journal_seal(path)
    if old_key is None:                    # fresh file, or one we just superseded
        _journal_write({"_journal": "key", "key": key})
    _JOURNAL["loaded"] = {} if args.mutate_no_resume else rows
    if args.mutate_no_resume and rows:
        print("  journal: --mutate-no-resume, so %d journalled row(s) are ignored and"
              " every trial is re-measured" % len(rows))
    elif rows:
        kinds = {}
        for r in rows.values():
            kinds[r.get("outcome")] = kinds.get(r.get("outcome"), 0) + 1
        print("  journal: %s -- %d row(s) resumable (%s)"
              % (path, len(rows),
                 "  ".join("%s=%d" % (k, kinds[k]) for k in sorted(kinds, key=str))))
    else:
        print("  journal: %s (new)" % path)
    _JOURNAL["started"] = True
    return len(_JOURNAL["loaded"])


def _journal_seal(path):
    """Terminate a truncated last line, so the next append starts on a fresh one."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) == b"\n":
                return
        with open(path, "a") as fh:
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        print("  journal: sealed a truncated final line (a kill left it mid-write)")
    except OSError as e:
        print("  !! journal: could not seal %s (%s)" % (path, e))


def _journal_write(obj):
    """Append one object and get it onto the disk before the next trial starts.

    flush() alone only reaches the OS; the kills this guards against take the whole
    container with them, so the fsync is the part that matters.
    """
    import json

    path = _JOURNAL["path"]
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as e:
        # Never fatal: losing the journal costs a resume, losing the trial costs an
        # hour of GPU.  But say so loudly -- a silent failure here is only discovered
        # after the crash it was supposed to cover.
        print("  !! journal append failed (%s): %s -- this sweep is NOT resumable" % (path, e))
        _JOURNAL["path"] = None


def _journal_row_key(name, kernel):
    """What identifies a trial inside one sweep.

    NOT the row's `name` alone.  One chain op can dispatch several kernels -- the SDPA
    backward reaches three aiter .co -- and run_mutations is called once per kernel,
    so "identity@260" is emitted three times, once per .co, and "sbarrier_ds_write_read@260"
    twice.  Keyed by name alone those collide, and a resume would hand one kernel's
    control row to a different kernel: the single most damaging thing this file could
    do quietly, since `identity` is what licenses every other row.
    """
    return "%s|%s" % (name, kernel or "-")


def _journal_reuse(name, kernel=None):
    """The journalled row for this trial, if it may stand in for running it."""
    loaded = _JOURNAL["loaded"]
    rec = loaded.get(_journal_row_key(name, kernel))
    if rec is None:
        # A legacy row (no kernel recorded) whose name was unique in the file.
        # popped, not read: if this sweep emits the same name for a SECOND kernel,
        # that one must be measured rather than handed the first one's row.
        rec = loaded.pop(_journal_row_key(name, None), None)
    if rec is None:
        return None
    if (args.mutate_retry_incomplete
            and rec.get("outcome") in ("TIMEOUT", "ERROR")):
        return None
    rec = dict(rec)
    rec["resumed"] = True
    return rec


def _journal_record(rec, kernel=None):
    """Journal a trial that just ran.  Rows reused from the journal are not re-written."""
    if not rec.get("resumed"):
        rec = dict(rec)
        rec["_jkernel"] = kernel
        rec["_jkey"] = _journal_row_key(rec["name"], kernel)
        _journal_write(rec)


def run_trial(target, cache, m, sites, logdir):
    """Build the mutated kernel, replay it in a child, and classify the result."""
    import signal
    import subprocess
    import time
    import kernel_patch as kp

    rec = {"name": m.name, "description": m.description, "selector": m.selector,
           "payload": m.payload, "sites": len(sites),
           "addrs": ["0x%X" % s.addr for s in sites[:32]],
           "outcome": "ERROR", "verdict": None, "returncode": None,
           "seconds": 0.0, "error": None, "disasm": [], "asm_error": None,
           "launches": None}
    # Resolved once per trial and then held: communicate() and the TIMEOUT message it
    # raises have to quote the same number, or the report explains a kill that did not
    # happen at the deadline it names.
    tmo = mutate_timeout()
    rec["timeout"] = tmo
    t0 = time.time()
    try:
        offsets = target.offsets_for(sites, m.payload)
        overlay = cache.apply(offsets)
    except kp.PatchError as e:
        rec["error"] = "PatchError: %s" % e
    except SystemExit as e:                       # rnan uses sys.exit() for errors
        rec["error"] = "rnan exited (%r) while building the overlay" % (e.code,)
    except Exception as e:                        # noqa: BLE001 -- per-trial, never fatal
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    else:
        env = cache.env(os.environ)
        env["_NAN_MUTATE_CHILD"] = m.name
        if sweep_holds_load():
            # So the child's own output describes the device it actually ran on.
            env["_NAN_SWEEP_CONTENDED"] = (args.contend_with or
                                            "%d co-tenants" % args.contend)
        # start_new_session so a wedged trial is a killable process GROUP: hipBLASLt
        # spawns helpers, and killing only the python pid leaves them on the GPU.
        p = subprocess.Popen(child_cmd(), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=tmo)
            rec["returncode"] = p.returncode
            v = child_verdict(out)
            rec["verdict"] = v
            # KERNEL_BUG_UNVERIFIED is a NAN as far as a sweep is concerned: it
            # says the op went non-finite and only that the OVERFLOW question
            # went unanswered, which is not something a mutation trial asks.
            # CHAIN_* are the same two answers from a chain-payload trial.
            rec["outcome"] = {"KERNEL_BUG": "NAN",
                              "KERNEL_BUG_UNVERIFIED": "NAN",
                              "CHAIN_REPRODUCED": "NAN",
                              "NOT_REPRODUCED": "CLEAN",
                              "CHAIN_NOT_REPRODUCED": "CLEAN"}.get(v, "ERROR")
            if v is None:
                rec["error"] = ("child printed no [verdict] line (rc=%d); tail: %s"
                                % (p.returncode, " | ".join(out.strip().splitlines()[-3:])))
            elif rec["outcome"] == "ERROR":
                rec["error"] = "unexpected verdict %s" % v
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except OSError:
                p.kill()
            out, _ = p.communicate()
            rec["outcome"] = "TIMEOUT"
            rec["error"] = "no verdict within %gs" % tmo
        rec["launches"] = child_launches(out)
        import kernel_patch as _kp
        # Where the evidence lives depends on the backend: aiter announces its loads
        # in the child's stdout, the HSA shim writes its own log.  Both answer the
        # same question -- did the patch reach the GPU, or did this trial measure the
        # stock kernel and look exactly as if it had not?
        rec["served_overlay"] = _kp.served_overlay(
            out, overlay,
            aiter=isinstance(target, _kp_aiter_target_cls()),
            shim_log=(cache.log_text() if isinstance(cache, _kp.HsaOverlay) else None))
        with open(os.path.join(logdir, _safe_name(m.name) + ".log"), "w") as f:
            f.write("# overlay %s\n# %s\n\n%s" % (overlay, " ".join(child_cmd()), out))
    rec["seconds"] = round(time.time() - t0, 1)
    score_clean_power(rec)
    return rec


def score_clean_power(rec):
    """How much a CLEAN row is worth, given the baseline's hit rate.

    A mutation sweep's whole logic is "the baseline reproduced, this trial did not,
    therefore the mutation removed it".  That inference is only as good as the trial's
    chance of catching a fault it did NOT remove.  At a baseline rate p per launch and
    n launches, a still-faulty trial reports zero with probability exp(-p*n) -- so at
    p = 0.12% and n = 100 the sweep calls a fully intact bug CLEAN 89 times in 100.

    The hand-driven sweep this replaces argued exactly this by hand in its header
    (an earlier step5c_sweep.sh: --iters 400000 at 1.56e-5
    gives ~6.2 expected hits, e^-6.2 ~ 0.2%).  Computing it here is what stops the
    next sweep from having to.

    Only CLEAN rows get a verdict attached: for a NAN row the fault showed up, which
    settles the question without any arithmetic.
    """
    import math

    rec["expected_hits"] = None
    rec["p_false_clean"] = None
    rec["underpowered"] = False
    rate = _BASELINE["rate"]
    if rec.get("outcome") != "CLEAN" or not rate:
        return
    n = rec.get("launches")
    if n is None:
        # A CLEAN child always runs to completion -- it is the NAN ones that stop
        # early -- so a missing count means the line could not be parsed, not that
        # the trial was short. Fall back to what it was asked to run.
        n = sweep_trial_iters()
    rec["expected_hits"] = rate * n
    rec["p_false_clean"] = math.exp(-rec["expected_hits"])
    rec["underpowered"] = rec["expected_hits"] < _MIN_EXPECTED_HITS


def save_mutation_asm(target, m, before, after, outdir, row_name=None):
    """Write both listings and their full diff; return the paths.

    A before/after pair per mutation rather than one shared baseline file: a finding
    is something you attach to a bug report, and a directory where each name carries
    its own pair can be read by someone who does not know which file the sweep
    considered original.

    Named "<row>.<kernel>.{before,after}.s" -- the row carries the chain op, the
    kernel tag separates the several kernels one op can dispatch.  See _kernel_tag
    for what goes wrong with either half missing.
    """
    import kernel_patch as kp

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    safe = "%s.%s" % (_safe_name(row_name or m.name), _kernel_tag(target.kernel))
    paths = []
    for tag, seq in (("before", before), ("after", after)):
        p = os.path.join(outdir, "%s.%s.s" % (safe, tag))
        with open(p, "w") as f:
            f.write("; %s -- %s\n; mutation %s (%s/%s): %s\n\n"
                    % (target.kernel, tag, m.name, m.selector, m.payload, m.description))
            f.write("\n".join(i.line for i in seq) + "\n")
        paths.append(p)
    p = os.path.join(outdir, "%s.diff" % safe)
    with open(p, "w") as f:
        f.write("# %s -- %s (%s/%s)\n# %s\n\n"
                % (target.kernel, m.name, m.selector, m.payload, m.description))
        # limit=0 on purpose: the file is the complete record regardless of how much
        # of it --mutate-disasm-max let through to the terminal.
        f.write("\n".join(kp.format_disasm_diff(
            before, after, context=args.mutate_disasm_context, limit=0)) + "\n")
    paths.append(p)
    return paths


def show_mutation_asm(target, m, rec, after):
    """Print the kernel's ISA before and after a mutation that removed the NaN.

    Immediately, next to the row that found it, rather than collected at the end:
    a sweep is watched live, and the point of a CLEAN row is to stop reading the
    table and start reading the instructions.
    """
    import kernel_patch as kp

    if rec.get("asm_error"):
        print("\n  !! %s" % rec["asm_error"])
    if after is None:
        return

    if args.mutate_disasm == "full":
        for tag, seq in (("ORIGINAL", target.insns), ("MUTATED by %s" % m.name, after)):
            print("\n%s\n%s -- %s (%d instruction(s))\n%s"
                  % ("=" * 78, tag, target.kernel, len(seq), "=" * 78))
            for ins in seq:
                print(ins.line)
        print("")
    elif args.mutate_disasm == "diff":
        # Say which lines ARE the difference: 2026-11-23 (deepseek MLA, vmcnt0) the
        # unmarked context lines around each edit (v_mov_b32_dpp at 0x2850/0x2860) were
        # read as part of "14 instruction(s) differ" -- the 14 are the -/+ pairs only.
        print("\n  %s removed the NaN; %d instruction(s) differ from the original "
              "(the -/+ lines; unmarked lines are %d unchanged neighbour(s) of context):"
              % (m.name, len(rec["disasm"]), args.mutate_disasm_context))
        for line in kp.format_disasm_diff(target.insns, after,
                                          context=args.mutate_disasm_context,
                                          limit=args.mutate_disasm_max):
            print("  " + line)
        print("")

    if args.mutate_asm_dir:
        try:
            # rec["name"], not m.name: by the time a CLEAN row reaches here the chain
            # suffix is on it, and that is half of what keeps the file unique.
            paths = save_mutation_asm(target, m, target.insns, after,
                                      args.mutate_asm_dir, rec.get("name"))
        except OSError as e:
            print("  could not write --mutate-asm-dir %s: %s" % (args.mutate_asm_dir, e))
        else:
            print("  wrote %s" % "\n        ".join(paths))


def _power_note(rec):
    """The inline '  (n expected hits, P(false CLEAN)=p)' suffix for a CLEAN row."""
    if rec.get("expected_hits") is None:
        return ""
    return ("   %.1f expected hits, P(false CLEAN)=%.1g%s"
            % (rec["expected_hits"], rec["p_false_clean"],
               "  << UNDERPOWERED" if rec["underpowered"] else ""))


def report(trials, kernel, target, out_path):
    """The table, the identity verdict, and the JSON record."""
    import json

    # Controls are flagged rather than looked up by name: a chain sweep carries one
    # per op, each suffixed with its call index.  All of them must come back NAN --
    # one op's dead overlay makes that op's rows noise even if another op's control
    # was fine.
    controls = [t for t in trials if t.get("is_control")]
    if not controls:                                    # pre-flag reports
        controls = [t for t in trials if t["name"] == "identity"]
    ident = controls[0] if controls else None
    bad_controls = [c for c in controls if c["outcome"] != "NAN"]
    # Skipped on purpose: nothing to fail, so the table is read as-is -- with the
    # caveat below and identity_ok recorded as null, not as a passed control.
    skipped_ident = args.mutate_no_identity and not controls
    ident_ok = (bool(controls) or skipped_ident) and not bad_controls

    if skipped_ident:
        print("\nnote: identity control skipped (--mutate-no-identity). A CLEAN row below")
        print("      cannot be told apart from an overlay that stopped applying except by")
        print("      its asm read-back: a CLEAN row with no changed instructions is void.")
    elif ident is None:
        print("\n!! the identity control did not run. Nothing below is interpretable:")
        print("!! without it a CLEAN row could mean a fixed kernel or a dead overlay.")
    elif not ident_ok:
        ident = bad_controls[0]
        if len(controls) > 1:
            print("\n!! %d of %d identity controls came back something other than NAN"
                  % (len(bad_controls), len(controls)))
            print("!! (%s)" % ", ".join("%s=%s" % (c["name"], c["outcome"])
                                        for c in bad_controls))
        print("\n!! IDENTITY CONTROL CAME BACK %s, NOT NAN." % ident["outcome"])
        if ident["outcome"] == "CLEAN":
            print("!! The unmodified kernel, pushed through the same unbundle/patch/")
            print("!! rebundle/overlay path, stopped reproducing the NaN. That is a")
            print("!! broken harness, not a fixed kernel -- every row below is noise.")
            if (args.contend or args.contend_with) and args.mutate_solo:
                print("!! You passed --mutate-solo, so the co-tenants the BASELINE ran")
                print("!! under were dropped before these trials. For a contention-gated")
                print("!! fault that alone produces this row -- it is the wrong node, not")
                print("!! broken plumbing. Drop --mutate-solo and sweep under the load.")
            elif not (args.contend or args.contend_with):
                print("!! If the baseline needed co-tenants to reproduce, this is the")
                print("!! expected result of sweeping without them: pass --contend-auto.")
        else:
            print("!! The control never returned a verdict, so nothing here shows that")
            print("!! the overlay path reproduces the NaN at all. Until it does, a CLEAN")
            print("!! row cannot be told from a mutation that simply never ran.")
        if ident.get("error"):
            print("!! %s" % ident["error"])

    order = sorted(range(len(trials)),
                   key=lambda i: (_OUTCOME_ORDER.get(trials[i]["outcome"], 9), i))
    width = max([len(t["name"]) for t in trials] + [8])
    print("\n%-*s  %-7s %6s %6s  %s"
          % (width, "mutation", "outcome", "secs", "sites", "description"))
    print("-" * (width + 52))
    for i in order:
        t = trials[i]
        print("%-*s  %-7s %6.1f %6d  %s%s"
              % (width, t["name"], t["outcome"], t["seconds"] or 0.0, t["sites"],
                 t["description"], "  [resumed]" if t.get("resumed") else ""))
        if t.get("expected_hits") is not None:
            print("%-*s %s" % (width, "", _power_note(t).strip()))
        if t["error"]:
            print("%-*s  %s" % (width, "", t["error"][:150]))

    counts = {}
    for t in trials:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    print("\n%s" % "  ".join("%s=%d" % (k, counts[k]) for k in sorted(counts)))

    # Which rows this process actually measured.  A resumed row is a real measurement
    # -- the journal key pins it to the same capsule, host, iters and co-tenants --
    # but it was not taken now, and a report that cannot tell the two apart cannot be
    # audited against the log of the run that produced it.
    resumed = [t for t in trials if t.get("resumed")]
    if resumed:
        print("\n%d of %d row(s) were RESUMED from the journal %s, not measured by"
              % (len(resumed), len(trials), _JOURNAL["path"]))
        print("this run: %s" % ", ".join(t["name"] for t in resumed))
        print("They were measured under the same capsule, host, launches and"
              " co-tenants -- that is")
        print("what the journal key checks -- but say so in the report.")

    # The aiter overlay is only real if the child says it loaded from it.  Checked
    # over every trial that ran, because "the patch never reached the GPU" produces
    # a table of plausible rows and no error anywhere else.
    ran = [t for t in trials if t.get("served_overlay") is not None]
    unserved = [t for t in ran if not t["served_overlay"]]
    if ran and unserved:
        print("\n!! %d of %d trial(s) did NOT load the kernel from the overlay."
              % (len(unserved), len(ran)))
        print("!! aiter served its EMBEDDED copy instead, so those trials ran the"
              " STOCK kernel")
        print("!! whatever the mutation did to the file. Their outcomes are not"
              " findings:")
        for t in unserved:
            print("!!   %-28s %s" % (t["name"], t["outcome"]))
        print("!! Usual cause: the environment that selects the CK/Aiter path was not"
              " restored,")
        print("!! so the op dispatched elsewhere. Pass --restore-env.")
    elif ran:
        print("\noverlay served: all %d trial(s) loaded the patched .co from the"
              " overlay." % len(ran))

    skipped = [t for t in trials if t["outcome"] == "SKIPPED"]
    if skipped:
        # Never silent.  A chain sweep that could patch two of five ops and said so
        # only by printing three fewer rows reads as "all five were covered".
        print("\n%d chain op(s) were NOT swept:" % len(skipped))
        for t in skipped:
            print("  %-28s %s" % (t["name"], t["error"]))

    others = [t for t in trials if not t.get("is_control")
              and t["name"] != "identity" and t["outcome"] != "SKIPPED"]
    clean = [t for t in others if t["outcome"] == "CLEAN"]
    weak = [t for t in clean if t.get("underpowered")]
    if not ident_ok:
        pass                                      # already shouted about, above
    elif clean:
        print("\nFINDING: %d mutation(s) removed the NaN:" % len(clean))
        for t in clean:
            print("  %s -- %s" % (t["name"], t["description"]))
            if t.get("op"):
                print("     op: %s (call #%s)" % (t["op"], t["call_index"]))
            if t["addrs"]:
                print("     %s%s" % (" ".join(t["addrs"][:8]),
                                     " ..." if t["sites"] > 8 else ""))
            if t.get("expected_hits") is not None:
                print("     %s" % _power_note(t).strip())
            if t.get("disasm"):
                print("     %d instruction(s) changed in the kernel%s"
                      % (len(t["disasm"]),
                         "" if args.mutate_disasm == "none" else "; ISA shown above"))
            if t.get("asm_error"):
                print("     !! %s" % t["asm_error"])
        if weak:
            print("\n!! %d of those CLEAN row(s) are UNDERPOWERED: at the baseline's"
                  % len(weak))
            print("!! measured rate the trial would have come back clean anyway, with")
            print("!! the fault fully intact, %s of the time."
                  % " / ".join("%.0f%%" % (100 * t["p_false_clean"]) for t in weak))
            print("!! Raise --iters (or --chain-iters) until the expected-hit count is")
            print("!! above %g before reporting any of them as a finding."
                  % _MIN_EXPECTED_HITS)
    elif not others:
        print("\nOnly the identity control ran, and the NaN survived it: the patch and")
        print("overlay path is transparent. Register a mutator that actually changes")
        print("something (see kernel_mutations.py) to start localising the fault.")
    elif all(t["outcome"] == "TIMEOUT" for t in others):
        print("\nEvery mutation timed out. That is a failed sweep, not a clean one --")
        print("pin a larger --mutate-timeout (it is now %.0fs%s) or check that the GPU"
              % (mutate_timeout(),
                 "" if args.mutate_timeout is not None else ", derived from the baseline"))
        print("is not already wedged.")
    elif all(t["outcome"] == "ERROR" for t in others):
        print("\nEvery mutation errored: nothing was actually tested. See the reasons")
        print("above; re-run with --keep-workdir to keep the per-trial child logs.")
    else:
        print("\nNo mutation removed the NaN: the fault is not in any site these")
        print("mutators patched.")

    if out_path:
        # Which patcher served this sweep.  Reporting asks for kernel IDENTITY, not
        # just addresses, and the two dispatches identify a kernel differently: a
        # Tensile kernel by name + solution_index + library dir, an aiter one by its
        # .co path + exported symbol (it has no solution index at all).  Recording
        # the kind is what stops a null solution_index from reading as a lookup that
        # failed.
        import kernel_patch as _kpr
        _aiter = isinstance(target, _kp_aiter_target_cls())
        _ck = isinstance(target, _kpr.HsaTarget)
        rec = {"capsule": os.path.abspath(args.capsule), "kernel": kernel,
               "arch": getattr(target, "arch", None),
               "libdir": getattr(target, "libdir", None),
               "dispatch": "ck" if _ck else "aiter" if _aiter else "tensile",
               "hsaco": getattr(target, "kernel", None) if _aiter else None,
               "hsaco_path": getattr(target, "elf", None) if _aiter else None,
               "hsaco_symbol": getattr(target, "symbol", None) if _aiter else None,
               "aiter_asm_dir": os.environ.get("AITER_ASM_DIR") if _aiter else None,
               # CK has no library path and no solution index: the kernel's identity
               # IS the mangled template plus the bytes it was measured on, so a bug
               # report quotes these two instead.
               "co_symbol": getattr(target, "symbol", None) if _ck else None,
               "co_path": getattr(target, "elf", None) if _ck else None,
               "co_sha256": (target.sha256() if _ck else None),
               "solution_index": (args.solution_index if args.solution_index is not None
                                  else args.patch_solution_index),
               # What the trials actually ran, which is no longer implied by --iters:
               # a chain sweep's unit is a chain run and its count is --chain-iters.
               "payload_kind": sweep_payload_kind(),
               "trial_iters": sweep_trial_iters(),
               # A sweep's trials and its baseline must share these or the table is
               # not an A/B.  Recorded so a reader can tell which experiment it was.
               "contended": bool(sweep_holds_load()),
               "contend_n": args.contend,
               "contend_with": args.contend_with,
               "contend_gated": _CONTEND_GATED,
               "mutate_solo": bool(args.mutate_solo),
               "mutate_chain_ops": args.mutate_chain_ops,
               "mutate_stop_on_finding": bool(args.mutate_stop_on_finding),
               # Which op led the walk and why.  A table read without this cannot
               # tell a sweep that covered the chain from one that stopped early,
               # nor which detector chose the order -- and [C1] and [C2] disagree
               # exactly when a finite-but-wrong op is upstream.
               "chain_culprit": _CHAIN_CULPRIT[0],
               "chain_order": [{"op": t.get("op"), "call_index": t.get("call_index")}
                               for t in trials if t.get("op")],
               # The baseline hit rate every CLEAN row's power is computed from.
               "baseline_rate": _BASELINE["rate"],
               "min_expected_hits": _MIN_EXPECTED_HITS,
               "iters": args.iters, "mutate_timeout": mutate_timeout(),
               # The resolved budget alone cannot be re-derived, and a sweep whose
               # timeout was measured rather than asked for is a different experiment.
               "mutate_timeout_auto": args.mutate_timeout is None,
               "mutate_timeout_factor": _MUTATE_BUDGET,
               # >1 means the control was dearer than the baseline predicted and the
               # later rows were re-budgeted; the per-trial "timeout" says who got what.
               "mutate_timeout_scale": _BASELINE["scale"],
               "baseline_setup": _BASELINE["setup"],
               "baseline_per_launch": _BASELINE["per_launch"],
               "disasm_mode": args.mutate_disasm,
               "asm_dir": (os.path.abspath(args.mutate_asm_dir)
                           if args.mutate_asm_dir else None),
               "identity": (ident or {}).get("outcome"),
               "identity_ok": None if skipped_ident else ident_ok,
               "identity_skipped": bool(skipped_ident),
               # Provenance of the rows themselves.  `resumed` rows were measured by
               # an earlier process against the same journal key; without this a
               # reader cannot match the table against the log of any single run.
               "journal": _JOURNAL["path"],
               "journal_key": _journal_key() if _JOURNAL["path"] else None,
               "resumed_trials": [t["name"] for t in trials if t.get("resumed")],
               "counts": counts, "trials": trials}
        d = os.path.dirname(os.path.abspath(out_path))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
        print("\nreport: %s" % out_path)
    return 0 if ident_ok else 1


def record_chain_geometry(cap, dev):
    """One extra chain execution, purely to learn each op's hipBLASLt problem key.

    Deliberately separate from [C2]'s loop.  [C2] measures a race, so its per-
    iteration body has to stay exactly what it was before this feature existed --
    host-side work added there is interleaved with the launches whose timing is the
    measurement.  One pass costs milliseconds and the sweep that follows costs
    minutes per trial, so there is nothing to save by folding it in.

    Cheap to call twice: _CHAIN_GEOM is keyed by call index and [C1], when it ran,
    has already filled it.
    """
    if all(rec["call_index"] in _CHAIN_GEOM for rec in (cap.get("chain") or [])):
        return
    try:
        _run_chain_once(cap, dev, compare=False, record_geom=True)
    except Exception as e:                             # noqa: BLE001 -- never fatal
        print("  could not map the chain's ops onto hipBLASLt problem keys: %s" % e)


def chain_sweep_order(cap, tally, culprit=None):
    """The chain's ops in the order a sweep should mutate them: BACKWARD.

    cap["chain"] is in ascending call order, roots first and the op the capsule is
    named for last.  A sweep walks it the other way, for two reasons:

      1. the op that goes non-finite is at the downstream end, and the whole point of
         [C2]'s tally is that it names which one.  Starting anywhere else spends the
         most expensive rung on ops that were still finite when the fault appeared.
      2. a mutation to an upstream op changes the operands every op after it sees, so
         an upstream-first walk confounds every later row -- a CLEAN downstream row
         could be the downstream mutation or the upstream one still in effect.
         Downstream-first, each row is read against an unmodified prefix.

    Two signals re-seed the head of that order, and [C1]'s outranks [C2]'s:

      * `culprit` -- [C1]'s first-op-to-go-wrong.  It leads, because it is the only
        one of the two that can see a finite-but-WRONG op.  [C2] scores on
        finiteness alone, so for any fault where an upstream op goes finite-but-wrong
        the tally names something downstream of the real culprit and would send the
        most expensive rung to the op [C1] just exonerated.
      * `tally` -- [C2]'s first-to-go-non-finite count, as the secondary key.  It is
        a contended measurement over many runs where [C1] is one solo walk, so it
        orders everything the culprit does not.

    With neither (no --chain-iters and no [C1] walk, e.g. --chain-loop-only) this
    degrades to plain reverse order, which already starts at the capsule's named op.
    """
    chain = cap.get("chain") or []
    rev = list(reversed(chain))
    rank = {}
    for (ci, _op), n in (tally or {}).items():
        rank[ci] = max(rank.get(ci, 0), n)
    cci = (culprit or {}).get("call_index")
    # A wrong-value capsule (a rerun or value check fired on the capsule's own op) names its
    # op by the capture's check, which outranks one walk's small DIVERGED on an ancestor:
    # that can be the node (another GEMM solution than the capture's), and a sweep -- or a
    # listing, which stops at the first kernel -- then started on the wrong kernel
    # (2026-11-20/21: aten.mm DIVERGED 0.0156 on some nodes, bit-repeatable everywhere).
    tci = rev[0]["call_index"] if rev and cap.get("verify") and rev[0]["op"] == cap.get("op") else None
    if cci is None and not rank and tci is None:
        return rev
    # sorted() is stable, so anything the keys are silent about keeps its
    # reverse-chain position behind the ops they name.
    return sorted(rev, key=lambda rec: (0 if rec["call_index"] == tci else 1,
                                        0 if rec["call_index"] == cci else 1,
                                        -rank.get(rec["call_index"], 0)))


def _op_geometry(rec):
    """The hipBLASLt problem key for one chain op, or None if it has none.

    Only ops that dispatch through hipBLASLt have one.  A clone, an elementwise op,
    or a fused attention kernel that goes to Aiter/CK does not -- and None here is
    what turns into an honest SKIPPED row rather than a sweep of the wrong kernel.

    Read from _CHAIN_GEOM, filled by _run_chain_once as the chain ran: the capsule
    stores a geometry only for the op it is named for, so an upstream op's key does
    not exist until its operands do.
    """
    g = _CHAIN_GEOM.get(rec["call_index"])
    if isinstance(g, dict) and all(k in g for k in _GEOM_KEY):
        return g
    return None


def _skipped_row(name, description, reason, op=None, call_index=None):
    """A sweep row for work that was NOT done, shaped like a trial so it cannot be
    dropped from the table.  A silently shorter table reads as full coverage."""
    return {"name": name, "description": description, "selector": "-", "payload": "-",
            "sites": 0, "addrs": [], "outcome": "SKIPPED", "verdict": None,
            "returncode": None, "seconds": 0.0, "error": reason, "disasm": [],
            "asm_error": None, "launches": None, "op": op, "call_index": call_index,
            "expected_hits": None, "p_false_clean": None, "underpowered": False}


def op_yielded_finding(trials, ci):
    """Did this op's sub-sweep produce a row worth stopping the walk for?

    Deliberately strict, because --mutate-stop-on-finding trades coverage for GPU
    hours and a premature stop is indistinguishable from a complete sweep in the
    table.  All three must hold:

      * the op HAS controls and every one came back NAN.  Without a positive
        control a CLEAN row proves nothing (see report()), so it is not a finding
        and certainly not a reason to stop walking -- unless the controls were
        skipped on purpose (--mutate-no-identity), when none is required.
      * some non-control mutator came back CLEAN -- that is the finding itself.
      * it is not UNDERPOWERED.  A CLEAN row with 0.3 expected hits is a trial that
        could not have caught the fault; stopping on one would end the sweep on
        the strength of a coin flip.
    """
    rows = [t for t in trials if t.get("call_index") == ci]
    controls = [t for t in rows if t.get("is_control")]
    if any(c["outcome"] != "NAN" for c in controls):
        return False
    if not controls and not args.mutate_no_identity:
        return False
    return any(t["outcome"] == "CLEAN" and not t.get("is_control")
               and not t.get("underpowered") for t in rows)


# The capsule's op name, so a single-op sweep can ask which aiter family to look in
# without re-loading a 1.4 GB capsule just to read one string.
_CAP_OP = [None]
_CAP = [None]               # the loaded capsule (grouped GEMM kernel discovery reads its offs)


def _capsule_op_name():
    return _CAP_OP[0] or ""


def _kp_aiter_target_cls():
    """kernel_patch.AiterTarget, imported late so this module stays torch-free here."""
    import kernel_patch as kp
    return kp.AiterTarget


# Discovered once per sweep, not once per chain op: one child replay of the whole
# chain dispatches every op in it, so the trace already names every .co the walk could
# possibly want.  None = not yet asked; [] = asked and there are none.
_AITER_LOADS = [None]


def aiter_cos_for_op(op):
    """[(co_path, symbol)] attributable to this chain op, newest-match first.

    A chain replay loads every op's kernels in one process, so the trace alone cannot
    say which op asked for which .co.  Rather than guess, attribute on the one signal
    aiter's own layout provides: the family directory states the direction
    (fmha_v3_bwd/ vs fmha_v3_fwd/), and the aten op name states it too.  An op whose
    direction cannot be read, or that matches nothing, gets no .co and stays SKIPPED
    -- a wrong attribution would sweep a kernel this op never ran, and every row of
    that sub-sweep would be about the wrong binary.

    This is a heuristic and the report has to say so; it is not derived from a
    per-op dispatch record, because aiter does not emit one.
    """
    if _AITER_LOADS[0] is None:
        _AITER_LOADS[0] = discover_aiter_kernels()
    loads = _AITER_LOADS[0]
    if not loads:
        return []
    # Exact, where it applies: aten ops never load an aiter .co, so when this op is the
    # chain's ONLY aiter.* op, every file-backed aiter load of the replay is its own
    if op.startswith("aiter.") and _CAP[0] is not None:
        aiter_ops = set(r["op"] for r in (_CAP[0].get("chain") or [])
                        if r["op"].startswith("aiter."))
        if aiter_ops <= {op}:
            return list(dict.fromkeys(loads))
    low = op.lower()
    if "backward" in low or low.endswith("_bwd"):
        want = ("_bwd", "/bwd", "bwd_")
    elif "attention" in low or "sdpa" in low:
        want = ("_fwd", "/fwd", "fwd_")
    else:
        return []
    out = [(p, s) for p, s in loads
           if any(w in p.replace(os.sep, "/") for w in want)]
    return out


# Same contract as _AITER_LOADS: discovered once per sweep, None = not yet asked.
_CK_LOADS = [None]


def ck_cos_for_op(op):
    """[(co_path, symbol)] attributable to this chain op, for the HSA-intercepted path.

    Same shape, and the same honesty caveat, as aiter_cos_for_op: CK emits no per-op
    dispatch record, so attribution rests on the kernel family encoded in the mangled
    template -- FmhaFwdKernel for the forward, FmhaBwd* for the backward.  An op whose
    direction cannot be read gets nothing and stays SKIPPED, because sweeping a kernel
    this op never ran would make every row of that sub-sweep describe the wrong binary.
    """
    if _CK_LOADS[0] is None:
        _CK_LOADS[0] = discover_ck_kernels()
    loads = _CK_LOADS[0]
    if not loads:
        return []
    low = op.lower()
    if "backward" in low or low.endswith("_bwd"):
        want = ("FmhaBwd", "fmha_bwd")
    elif "attention" in low or "sdpa" in low:
        want = ("FmhaFwd", "fmha_fwd")
    else:
        return []
    return [(p, s) for p, s in loads if any(w in s for w in want)]


def _skip_rest(rest, trials, label):
    """Record the ops a stopped walk never reached.

    A table that simply ends reads as 'the rest came back clean', which is the one
    thing a sweep must never imply.
    """
    if not rest:
        return
    print("     %d op(s) upstream NOT swept: %s"
          % (len(rest), ", ".join("%s (call #%d)" % (r["op"], r["call_index"])
                                  for r in rest)))
    for r in rest:
        trials.append(_skipped_row(
            "%s@%d" % (r["op"].split(".")[-2] if "." in r["op"] else r["op"],
                       r["call_index"]),
            "chain op %s (call #%d)" % (r["op"], r["call_index"]),
            "not swept: --mutate-stop-on-finding ended the walk at %s" % label,
            r["op"], r["call_index"]))


def run_chain_mutations(cap, tally, dev, culprit=None):
    """[4] over a chain: sweep each op's kernel, walking the chain backward.

    Ops are deduplicated by dispatched kernel -- two ops that land on the same
    Tensile kernel yield the same sweep at twice the cost -- and every op that is
    not swept leaves a SKIPPED row saying why.
    """
    # After the baseline, never during it: see record_chain_geometry.
    record_chain_geometry(cap, dev)
    _CHAIN_CULPRIT[0] = culprit
    order = chain_sweep_order(cap, tally, culprit)
    print("\n[4] kernel mutations over the chain (%d op(s), walked BACKWARD:"
          " culprit first, then upstream)" % len(order))
    if culprit:
        print("  [C1] named %s (call #%d, %s) first-to-go-wrong; it leads the walk."
              % (culprit["op"], culprit["call_index"], culprit.get("kind", "?")))
        if tally and order[0]["call_index"] not in [ci for (ci, _o) in tally]:
            # Worth printing: the two detectors disagreeing is the signature of a
            # finite-but-wrong upstream op, which is exactly when leading with the
            # tally would sweep the wrong kernel.
            print("  note: [C2]'s tally names a different op -- [C1] leads anyway,"
                  " because only it can see a finite-but-wrong op.")
    elif tally:
        lead = order[0]
        print("  [C2] named %s (call #%d) first-to-go-non-finite; it leads the walk."
              % (lead["op"], lead["call_index"]))
        print("  note: no [C1] culprit (--chain-loop-only skips the walk). The tally"
              " scores finiteness only,")
        print("        so an upstream finite-but-wrong op cannot have reached it.")
    else:
        print("  no [C1] culprit and no [C2] tally (pass --chain-iters under the"
              " contention the fault")
        print("        needs); falling back to plain reverse chain order.")

    dropped = []
    if args.mutate_chain_ops > 0 and len(order) > args.mutate_chain_ops:
        # Same discipline as kernel_mutations.capped(), which cannot be reused here
        # because it formats what it drops as m.name and these are chain records:
        # name the ORDERING, then name what fell off it.  A cap that printed only a
        # shorter table would read as full coverage.
        dropped = order[args.mutate_chain_ops:]
        order = order[:args.mutate_chain_ops]
        print("  note: %d op(s) in the chain, keeping %d (--mutate-chain-ops, walking"
              " backward from the culprit); dropped %s"
              % (len(order) + len(dropped), len(order),
                 ", ".join("%s (call #%d)" % (r["op"], r["call_index"])
                           for r in dropped[:8])
                 + (", ..." if len(dropped) > 8 else "")))

    trials, rc, seen_kernels = [], 0, {}
    if args.patch_hsaco or args.patch_co or args.patch_kernel:
        # An explicit target is the kernel the user names for the capsule's own op: sweep it once,
        # under that op's label.  Walking the chain handed it to whichever op came first, and the
        # rows read "aten.mm (call #1617)" for an aiter .co (2026-11-21).
        last = cap["chain"][-1] if cap.get("chain") else {"op": cap.get("op"), "call_index": 0}
        print("\n  == sweeping %s (call #%s) -> the explicit target %s"
              % (last["op"], last["call_index"], args.patch_hsaco or args.patch_co or args.patch_kernel))
        rc = run_mutations(None, kernel=args.patch_kernel, op=last["op"], call_index=last["call_index"],
                           collect=trials)
        if not args.list_mutations and (args.mutate_report or trials):
            report(trials, args.patch_hsaco or args.patch_co or args.patch_kernel, _LAST_TARGET[0],
                   args.mutate_report)
        return rc
    for r in dropped:
        trials.append(_skipped_row(
            "%s@%d" % (r["op"].split(".")[-2] if "." in r["op"] else r["op"],
                       r["call_index"]),
            "chain op %s (call #%d)" % (r["op"], r["call_index"]),
            "not swept: --mutate-chain-ops %d stopped the backward walk before it"
            % args.mutate_chain_ops, r["op"], r["call_index"]))
    for pos, rec in enumerate(order):
        op, ci = rec["op"], rec["call_index"]
        label = "%s (call #%d)" % (op, ci)
        if args.list_mutations and seen_kernels and not args.mutate_chain_ops:
            # A listing wants the kernel a sweep would start on: the culprit's, which the walk
            # reaches first.  Walking on printed a page of SKIPPED upstream ops and "cannot
            # prepare" GEMMs (2026-11-21).  --mutate-chain-ops N lists N ops instead.
            print("\n  listing stops at the first op with a patchable kernel (%s); %d upstream op(s)"
                  " not listed (--mutate-chain-ops N to list more)" % (", ".join(seen_kernels),
                                                                       len(order) - pos))
            break
        g = _op_geometry(rec)
        if g is None:
            # No hipBLASLt key means no Tensile kernel -- but it does NOT mean no
            # kernel.  A fused attention op dispatches to a standalone aiter .co,
            # which this sweep can now patch; try that before giving up on the op.
            cos = aiter_cos_for_op(op)
            if cos:
                for co, sym in cos:
                    if co in seen_kernels:
                        print("\n  -- %s: folded into %s, same .co %s"
                              % (label, seen_kernels[co], co))
                        trials.append(_skipped_row(
                            "%s@%d" % (op.split(".")[-2] if "." in op else op, ci),
                            "chain op %s" % label,
                            "same aiter .co as %s; swept once" % seen_kernels[co],
                            op, ci))
                        continue
                    seen_kernels[co] = label
                    print("\n  == sweeping %s  ->  %s  (aiter .co, %s)"
                          % (label, co, sym))
                    sub = run_mutations(None, kernel=co, op=op, call_index=ci,
                                        collect=trials, hsaco=co)
                    rc = rc or sub
                if args.mutate_stop_on_finding and op_yielded_finding(trials, ci):
                    print("\n  -> %s yielded a finding; stopping the walk"
                          " (--mutate-stop-on-finding)." % label)
                    _skip_rest(order[pos + 1:], trials, label)
                    break
                continue
            # Third and last: a kernel compiled into a library (CK), reachable only
            # by intercepting the code object on its way to the GPU.
            cks = ck_cos_for_op(op)
            if cks:
                for co, sym in cks:
                    key = sym
                    if key in seen_kernels:
                        print("\n  -- %s: folded into %s, same kernel %s"
                              % (label, seen_kernels[key], sym[:48]))
                        trials.append(_skipped_row(
                            "%s@%d" % (op.split(".")[-2] if "." in op else op, ci),
                            "chain op %s" % label,
                            "same intercepted kernel as %s; swept once"
                            % seen_kernels[key], op, ci))
                        continue
                    seen_kernels[key] = label
                    print("\n  == sweeping %s  ->  %s  (CK, via HSA interception)"
                          % (label, sym[:56]))
                    sub = run_mutations(None, kernel=sym, op=op, call_index=ci,
                                        collect=trials, co=co, co_symbol=sym)
                    rc = rc or sub
                if args.mutate_stop_on_finding and op_yielded_finding(trials, ci):
                    print("\n  -> %s yielded a finding; stopping the walk"
                          " (--mutate-stop-on-finding)." % label)
                    _skip_rest(order[pos + 1:], trials, label)
                    break
                continue
            print("\n  -- %s: SKIPPED" % label)
            print("     no hipBLASLt problem key, so there is no Tensile kernel to"
                  " patch, no")
            print("     aiter .co, and no intercepted code object was attributed to"
                  " this op.")
            trials.append(_skipped_row(
                "%s@%d" % (op.split(".")[-2] if "." in op else op, ci),
                "chain op %s" % label,
                "no hipBLASLt geometry (not a Tensile dispatch), no aiter .co and no "
                "intercepted CK kernel attributed to this op", op, ci))
            continue
        kernel = discover_kernel(g)
        if not kernel:
            trials.append(_skipped_row(
                "%s@%d" % (op.split(".")[-2] if "." in op else op, ci),
                "chain op %s" % label,
                "kernel discovery found no dispatch for this op's geometry", op, ci))
            continue
        if kernel in seen_kernels:
            print("\n  -- %s: folded into %s, same kernel %s"
                  % (label, seen_kernels[kernel], kernel))
            trials.append(_skipped_row(
                "%s@%d" % (op.split(".")[-2] if "." in op else op, ci),
                "chain op %s" % label,
                "same dispatched kernel as %s; swept once" % seen_kernels[kernel],
                op, ci))
            continue
        seen_kernels[kernel] = label
        print("\n  == sweeping %s  ->  %s" % (label, kernel))
        sub = run_mutations(g, kernel=kernel, op=op, call_index=ci, collect=trials)
        rc = rc or sub
        # Early exit: the walk leads with the culprit, so in the common case the
        # first op swept is the only one that needed sweeping.  Everything after it
        # is recorded as SKIPPED rather than omitted -- a table that just stopped
        # would read as "the rest came back clean".
        if args.mutate_stop_on_finding and op_yielded_finding(trials, ci):
            rest = order[pos + 1:]
            print("\n  -> %s yielded a finding; stopping the walk"
                  " (--mutate-stop-on-finding)." % label)
            if rest:
                print("     %d op(s) upstream NOT swept: %s"
                      % (len(rest), ", ".join("%s (call #%d)"
                                              % (r["op"], r["call_index"])
                                              for r in rest[:8])
                         + (", ..." if len(rest) > 8 else "")))
                print("     They are recorded SKIPPED. Re-run without the flag if the"
                      " finding does not explain the bug --")
                print("     two ops in one chain can be independently faulty.")
            for r in rest:
                trials.append(_skipped_row(
                    "%s@%d" % (r["op"].split(".")[-2] if "." in r["op"]
                               else r["op"], r["call_index"]),
                    "chain op %s (call #%d)" % (r["op"], r["call_index"]),
                    "not swept: --mutate-stop-on-finding stopped the walk after %s"
                    % label, r["op"], r["call_index"]))
            break
    if args.list_mutations:
        # A listing, not a sweep: no trial ran, so the sweep summary (which read "only the
        # identity control ran" with --mutate-no-identity) does not apply, and an op with no
        # patchable kernel is not a failure of the listing (2026-11-20: exit 1, job FAILED).
        print("\n  listed %d kernel(s): %s; %d chain op(s) have no patchable kernel (SKIPPED)"
              % (len(seen_kernels), ", ".join(seen_kernels) or "none",
                 sum(1 for t in trials if t.get("outcome") == "SKIPPED")))
        return 0 if seen_kernels else 1
    if args.mutate_report or trials:
        report(trials, ", ".join(seen_kernels) or None, _LAST_TARGET[0],
               args.mutate_report)
    return rc


# The PatchTarget of the most recent sweep, so a chain sweep that folded several
# kernels into one report still has an arch/libdir to name.  A list, not a bare
# global, so run_mutations can write it without a `global` declaration.
_LAST_TARGET = [None]


def run_mutations(g, kernel=None, op=None, call_index=None, collect=None, hsaco=None,
                  co=None, co_symbol=None):
    """[4]: replay this capsule against each mutated copy of the winning kernel.

    `collect` makes this a sub-sweep of a chain walk: rows are appended to the
    caller's list and the caller writes one report over all of them, rather than each
    op overwriting the previous op's --mutate-report.

    `hsaco` sweeps an aiter .co instead of a Tensile kernel.  `g` is then None --
    an aiter op has no hipBLASLt problem key, which is precisely why it used to be
    SKIPPED here.
    """
    import shutil
    import kernel_patch as kp

    import kernel_mutations as km

    if collect is None:
        print("\n[4] kernel mutations")
    # Check the names before the disassembly, not after: a typo in --mutate-only
    # should cost a second, not the 40 s it takes to unbundle a 43 MB ELF.
    unknown = [n for n in args.mutate_only if km.mutator_of(n) not in km.MUTATORS]
    if unknown:
        print("  unknown mutator(s) %s; registered: %s"
              % (", ".join(unknown), ", ".join(km.MUTATORS)))
        return 1
    hsaco = hsaco or args.patch_hsaco
    co = co or args.patch_co
    co_symbol = co_symbol or args.patch_co_symbol
    if co:
        # No env to set and nothing to copy: the shim does the splice inside the
        # trial child, so the only setup is making sure it builds before we promise
        # the caller a sweep.
        try:
            kp.build_hsa_shim(None)
        except kp.PatchError as e:
            print("  %s" % e)
            return 1
        kernel = co_symbol or co
    elif hsaco:
        # Force the file-backed loader for the BASELINE as well, not just the trials.
        # Left alone, aiter serves a copy baked into the .so and the overlay is never
        # consulted: every mutator would come back CLEAN and the table would be a
        # list of false findings.  See the trap note in kernel_patch.py.
        if kp.ensure_aiter_asm_dir(args.patch_asm_dir, sys.stdout) is None:
            print("  cannot locate aiter's hsa dir; pass --patch-asm-dir")
            return 1
        kernel = hsaco
    elif kernel is None:
        # An aiter op with no GEMM geometry: ask aiter FIRST.  Tensile discovery profiles
        # the whole replay process, so it finds the GEMMs of the replay's own checks (the
        # value check's fp32 reference) and would name one of those as this op's kernel
        # (2026-11-20: an einsum of the reference named instead of the aiter .co).
        cos = (aiter_cos_for_op(_capsule_op_name())
               if g is None and _capsule_op_name().startswith("aiter.") and not args.patch_kernel
               else [])
        kernel = None if cos else (args.patch_kernel or discover_kernel(g))
        if not kernel:
            # Tensile discovery came back empty.  On a fused-attention capsule that
            # is not a failure to diagnose -- it is the signature of an op that never
            # went through hipBLASLt at all, so ask aiter before giving up.
            cos = cos or aiter_cos_for_op(_capsule_op_name())
            if len(cos) > 1:
                print("  %d aiter .co(s) dispatched; sweeping the first. Pin one with"
                      " --patch-hsaco." % len(cos))
                for p, s in cos:
                    print("        %s  (%s)" % (p, s))
            if cos:
                hsaco = kernel = cos[0][0]
                if kp.ensure_aiter_asm_dir(args.patch_asm_dir, sys.stdout) is None:
                    return 1
    if not kernel:
        return 1
    print("  kernel:  %s%s" % (kernel, "   (aiter .co)" if hsaco else
                               "   (HSA-intercepted code object)" if co else ""))

    try:
        target, rows = enumerate_mutations(kernel, hsaco=hsaco, co=co,
                                           co_symbol=co_symbol)
    except kp.PatchError as e:
        print("  %s" % e)
        return 1
    except SystemExit as e:                       # setup-phase rnan failure IS fatal
        print("  cannot prepare %s: rnan exited (%r)" % (kernel, e.code))
        return 1
    _LAST_TARGET[0] = target
    print("  arch:    %s   library %s" % (target.arch, target.libdir))
    print("  %d instruction(s) disassembled from 0x%X-0x%X"
          % (len(target.insns), target.start, target.stop))
    print_mutation_plan(target, rows)

    if args.list_mutations:
        return 0
    if args.patch_jobs > 1:
        print("\nnote: --patch-jobs %d ignored; mutation trials run serially, one GPU"
              % args.patch_jobs)
        print("      and one overlay at a time.")

    work = os.path.join(target.workdir, "overlay")
    logdir = os.path.join(target.workdir, "trials")
    os.makedirs(work, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    cache = (kp.HsaOverlay if co else
             kp.AiterOverlay if hsaco else kp.OverlayCache)(target, work)
    print("\noverlay: %s\ntrials:  %s" % (cache.overlay, logdir))
    if want_asm():
        print("asm:     %s, read back from the overlay for each mutation that removes"
              " the NaN" % ("saved only" if args.mutate_disasm == "none"
                            else args.mutate_disasm))
        if args.mutate_asm_dir:
            print("         listings and diffs -> %s" % args.mutate_asm_dir)
    else:
        print("asm:     off (--mutate-disasm none)")
    print("")

    # A chain sweep names rows "<mutator>@<op>" so one report can carry several ops
    # without two `identity` rows colliding in report()'s by-name lookup.
    suffix = "" if call_index is None else "@%d" % call_index
    # Once per sweep, not once per chain op: a chain walk calls this function for each
    # op and they all share one journal, keyed by the "<mutator>@<call index>" name.
    if not _JOURNAL["started"]:
        _journal_init()
    trials = []
    for m, sites, err in rows:
        if err:
            trials.append({"name": m.name + suffix, "description": m.description,
                           "selector": m.selector, "payload": m.payload, "sites": 0,
                           "addrs": [], "outcome": "ERROR", "verdict": None,
                           "returncode": None, "seconds": 0.0, "error": err,
                           "disasm": [], "asm_error": None, "launches": None,
                           "op": op, "call_index": call_index,
                           "is_control": m.name == "identity",
                           "expected_hits": None, "p_false_clean": None,
                           "underpowered": False})
            print("  [--] %-28s ERROR    %s" % (m.name, err[:60]))
            # Journalled but never reused: an enumeration failure is re-derived for
            # free on the next run, and reusing it would hide a PatchError that a
            # changed kernel or overlay has since fixed.
            _journal_record(trials[-1], kernel)
            continue
        n = len(trials) + 1
        print("  [%2d/%2d] %-28s sites=%-5d " % (n, len(rows), m.name, len(sites)),
              end="", flush=True)
        rec = _journal_reuse(m.name + suffix, kernel)
        if rec is not None:
            # Measured by an earlier run of this same sweep, on this host, under these
            # iters and co-tenants -- the journal key is what guarantees that.  Marked
            # `resumed` in the row so report() and the bug report can say so: it is a
            # real measurement, but it is not one this process took.
            print("%-7s %6.1fs%s  (resumed from journal)"
                  % (rec["outcome"], rec.get("seconds") or 0.0, _power_note(rec)))
            # The control still has to feed the budget, or a resumed sweep re-derives
            # its per-trial timeout from the pre-trial baseline alone and times out the
            # rows it still has to run.
            calibrate_from_trial(rec, rec.get("launches"))
            trials.append(rec)
            _JOURNAL["reused"].append(rec["name"])
            continue
        rec = run_trial(target, cache, m, sites, logdir)
        # The control is flagged, not recognised by name: a chain sweep suffixes every
        # row with its op so two ops' `identity` rows do not collide, and a report that
        # looked the control up as the literal string "identity" would then find none
        # and declare the whole table uninterpretable.
        rec["is_control"] = m.name == "identity"
        rec["name"] += suffix
        rec["op"], rec["call_index"] = op, call_index
        print("%-7s %6.1fs%s" % (rec["outcome"], rec["seconds"],
                                 _power_note(rec)))
        # After every row, though the control (which kernel_mutations enumerates first)
        # is the one that matters most: it is the earliest measurement taken under trial
        # conditions, and every remaining row still gets to use it.
        ratio = calibrate_from_trial(rec, rec["launches"])
        if ratio is not None:
            print("          %s cost %.1fs against a predicted %.1fs;"
                  " the remaining trials get %.0fs each (x%.2f)"
                  % (m.name, rec["seconds"], rec["seconds"] / ratio,
                     mutate_timeout(), ratio))
        if rec["outcome"] == "CLEAN" and want_asm():
            # Here rather than inside run_trial for two reasons: the next iteration's
            # apply() overwrites the overlay's single .co, so this is the last moment
            # the finding's bytes exist; and an unbundle of a 43 MB ELF does not
            # belong in the trial's `secs`, which is the replay's number.
            print("          reading the mutated kernel back out of the overlay ...")
            show_mutation_asm(target, m, rec, capture_asm(cache, target, rec))
            if getattr(args, "mutate_keep_fix", None) and rec["outcome"] == "CLEAN":
                try:
                    kept = kp.export_fix(cache, os.path.join(args.mutate_keep_fix,
                                                             rec.get("name") or m.name))
                    print("  fix kept: %s" % "\n            ".join(kept))
                except (OSError, kp.PatchError) as e:
                    print("  could not keep the fix (--mutate-keep-fix): %s" % e)
        trials.append(rec)
        # Last thing in the iteration, after the asm read-back, so a resumed row
        # carries the disasm the finding is reported from rather than a row that
        # claims CLEAN with no listing behind it.
        _journal_record(rec, kernel)

    if collect is not None:
        # A sub-sweep of a chain walk: the caller owns the report, so that one op's
        # rows do not overwrite the previous op's --mutate-report file.
        collect.extend(trials)
        rc = 0 if (args.mutate_no_identity
                   or any(t.get("is_control") and t["outcome"] == "NAN"
                          for t in trials)) else 1
    else:
        rc = report(trials, kernel, target, args.mutate_report)
    if args.keep_workdir or args.patch_workdir:
        print("workdir kept: %s\noverlay kept: %s" % (target.workdir, cache.overlay))
    else:
        if any(t["outcome"] in ("ERROR", "TIMEOUT") for t in trials):
            print("(re-run with --keep-workdir to keep the per-trial child logs)")
        shutil.rmtree(target.workdir, ignore_errors=True)
        # build_overlay() mkdtemp()s its own directory rather than using the one it
        # is handed, so the ~3000-entry shadow tree is not under workdir and would
        # otherwise survive every sweep.
        shutil.rmtree(cache.overlay, ignore_errors=True)
    return rc


# ======================================================================================
# co-tenants: putting the capture's concurrency back
#
# The capsule records how many processes were on the device when it failed (see
# contention_info in op_nan_capture.py).  This starts that many again.  Two
# rules learned the hard way, both of which produce a confident wrong number if
# ignored:
#
#   1. GATE ON OBSERVED LOAD, NEVER ON A SLEEP.  A fixed sleep tuned to a guess
#      about process startup is how an arm overlaps nothing and still prints
#      0/N.  Here the gate is free-memory on the device dropping, polled in
#      process -- no rocm-smi, works in a container.
#   2. SERIAL REPETITION IS NOT CONCURRENCY.  --iters 1000000 in one process is
#      a million sequential launches with the trigger absent each time.  The
#      contenders have to be separate PROCESSES, because what these faults need
#      is several HIP contexts issuing at once.
# ======================================================================================

_CONTENDERS = []
_CONTEND_GATED = None          # None = not attempted, True/False = gate verdict


def _synthetic(cap):
    """True when this replay runs on synthesized operands: --synthesize on a
    shapes-only capsule.  Recorded values are always used where they exist."""
    return args.synthesize and cap.get("data", "values") == "shapes"


def operands_banner(cap):
    """One `operands:` line; summary.txt, the verdict and the README carry it."""
    if not _synthetic(cap):
        print("operands:       recorded values%s"
              % (" (--synthesize ignored: the capsule has them)" if args.synthesize else ""))
        return
    print("operands:       SYNTHETIC (--synthesize, seed %d) -- recorded shapes/strides/"
          "alignment, values drawn from the recorded statistics. A verdict holds "
          "for these operands only: no DIVERGED check, and a clean run says nothing "
          "about a value-dependent NaN" % args.synthesize_seed)


def _synth_args():
    """--synthesize for a child replay: without it a co-tenant or sweep trial would
    refuse a shapes-only capsule, or run real values beside synthetic ones."""
    return (["--synthesize", "--synthesize-seed", str(args.synthesize_seed)]
            if args.synthesize else [])


def _contender_cmd():
    """This same script, looping on this same capsule, as pure device load."""
    cmd = [sys.executable, os.path.abspath(__file__), args.capsule, "--no-reference"]
    if args.chain:
        # --chain-loop-only: a child doing the [C1] fp64 comparison spends most
        # of its life on the host, not loading the GPU, which is pure waste in a
        # process whose only job is to be load.  (This was once justified by
        # siblings measuring ~16x weaker than a real co-tenant; that gap did not
        # replicate -- on a second host siblings were the STRONGER load.  The
        # reasoning stands on its own, the number does not.)
        cmd += ["--chain", "--chain-iters", "100000000", "--chain-loop-only"]
    else:
        cmd += ["--iters", "100000000"]
    if args.no_hipblaslt:
        cmd.append("--no-hipblaslt")
    if args.solution_index is not None:
        cmd += ["--solution-index", str(args.solution_index)]
    return cmd + _synth_args()


def start_contenders(n):
    """Start n co-tenants and wait until the device shows their load. -> bool."""
    global _CONTEND_GATED
    import atexit
    import subprocess
    if n <= 0 and not args.contend_with:
        return True
    # Contenders loop effectively forever by design, so an uncaught exception in
    # the parent would leave them pinning a shared GPU with nothing watching.
    # Registered before the first spawn, so it covers a failure mid-spawn too.
    atexit.register(stop_contenders)

    before = torch.cuda.mem_get_info()[0]
    # Each co-tenant gets its own session, so stop_contenders() can kill the whole
    # process group: a shell co-tenant's children (a model's mp.spawn workers)
    # outlive a kill() of the shell and keep pinning the GPU.
    sink = open(args.contend_log, "a") if args.contend_log else subprocess.DEVNULL
    if args.contend_with:
        print("  co-tenant: %s" % args.contend_with)
        _CONTENDERS.append(subprocess.Popen(args.contend_with, shell=True,
                                            stdout=sink, stderr=subprocess.STDOUT,
                                            start_new_session=True))
    else:
        print("  starting %d sibling replay process(es) on this device" % n)
        for _ in range(n):
            _CONTENDERS.append(subprocess.Popen(_contender_cmd(),
                                                stdout=sink, stderr=subprocess.STDOUT,
                                                start_new_session=True))

    deadline = time.time() + args.contend_warmup
    while time.time() < deadline:
        if all(p.poll() is not None for p in _CONTENDERS):
            print("  !! every co-tenant exited before loading the device%s"
                  % ((" -- its output: %s" % args.contend_log) if args.contend_log else
                     " (re-run with --contend-log FILE to see why)"))
            _CONTEND_GATED = False
            return False
        free = torch.cuda.mem_get_info()[0]
        # 256 MB is well above allocator noise and well below any real worker's
        # footprint, so it means "someone else is really here".
        if before - free > (256 << 20):
            print("  co-tenants are loading the device (%.1f GB taken, %.0fs)"
                  % ((before - free) / 1e9, args.contend_warmup
                     - (deadline - time.time())))
            _CONTEND_GATED = True
            return True
        time.sleep(1.0)

    print("  !! co-tenants never loaded the device within %.0fs" % args.contend_warmup)
    _CONTEND_GATED = False
    return False


def stop_contenders():
    import signal
    for p in _CONTENDERS:
        try:
            os.killpg(p.pid, signal.SIGKILL)       # the group: see start_contenders
        except Exception:                          # noqa: BLE001
            try:
                p.kill()
            except Exception:                      # noqa: BLE001
                pass
    _CONTENDERS[:] = []


def _is_spawn_boilerplate(argv):
    """True if argv is a multiprocessing worker stub rather than a real command.

    Measured, not guessed: a capture inside an mp.spawn worker records its own
    argv as

        python3 -c 'from multiprocessing.spawn import spawn_main; spawn_main(
        tracker_fd=16, pipe_handle=24)' --multiprocessing-fork

    which names no script and carries no flags.  The fan-out flag that actually
    gates the bug -- '--workers 6' -- is on the PARENT's command line.  Since a
    spawn worker is the single most likely place for a contention-gated capture
    to happen, headlining self-argv would hide the one field a reader needs.
    """
    return any("--multiprocessing-fork" in a or "spawn_main" in a
               or "forkserver" in a for a in argv)


# ======================================================================================
# [E] environment
#
# A capsule stores the environment the library saw, because hipBLASLt, rocBLAS
# and MIOpen read theirs when the .so initialises -- so an environment that
# differs here cannot be corrected after `import torch`, only re-exec'd into.
# Two things this code does that a plain dict.update does not, and both came out
# of a replay that looked clean because it was not the same experiment:
# restoring UNSETS, and device selection is held back.  See restore_env_dict.
# ======================================================================================

_REEXEC = "_NAN_REPLAY_REEXEC"

# What a capsule recorded under before env_info existed.  A replay must unset
# under the rule the CAPTURE used, not today's wider one -- otherwise a var that
# an old capture had no way to record would be unset here and the difference
# would look like the capture having run without it.
_LEGACY_ENV_SPEC = {
    "prefixes": ["HIPBLASLT_", "TORCH_BLAS_", "PYTORCH_TUNABLEOP_", "TENSILE_",
                 "HIP_", "ROCBLAS_", "NCCL_", "RCCL_", "HSA_", "TORCH_ROCM_",
                 "AITER_", "MIOPEN_"],
    "exact": ["OMP_NUM_THREADS", "GPU_MAX_HW_QUEUES"],
    "no_restore": [],
}

# Held back whatever the capsule says -- including an old capsule, whose own
# no_restore list is empty, so this fixes the device-repointing hazard for
# capsules captured before it was understood.  Reasons per group in
# ENV_NO_RESTORE (op_nan_capture.py); the strings here are what the banner
# prints, because "held back" with no reason reads like a bug.
_ENV_NEVER_RESTORE = {
    "HIP_VISIBLE_DEVICES": "device selection belongs to this run",
    "CUDA_VISIBLE_DEVICES": "device selection belongs to this run",
    "ROCR_VISIBLE_DEVICES": "device selection belongs to this run",
    "GPU_DEVICE_ORDINAL": "device selection belongs to this run",
    "LD_LIBRARY_PATH": "loader path: restoring it governs `import torch` itself",
    "LD_PRELOAD": "loader path: restoring it governs `import torch` itself",
    "HIPBLASLT_LOG_MASK": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    "HIPBLASLT_LOG_LEVEL": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    "HIPBLASLT_LOG_FILE": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    "ROCBLAS_LAYER": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    "ROCBLAS_LOG_TRACE_PATH": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    "ROCBLAS_LOG_BENCH_PATH": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    "ROCBLAS_LOG_PROFILE_PATH": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    "AMD_LOG_LEVEL": "diagnostic log: per-call host work perturbs the timing; would write into the capture's log",
    # aiter's INFO lines are how kernel discovery and the overlay served check see which .co loaded;
    # a run script's AITER_LOG_LEVEL=WARNING, restored, hid them (2026-11-21: discovery fell back to
    # an upstream GEMM, and aiter_fix_validate.sh read a working fix as NOT_SERVED)
    "AITER_LOG_LEVEL": "diagnostic log: aiter's INFO load lines are what kernel discovery and the served check read",
    # A library overlay is the thing under test (a step-8 fix, a hand-built patch):
    # restoring would UNSET it whenever the capture ran the stock library, and the
    # replay would silently measure the stock kernel. env_banner's `libpath:` line
    # says which library this process was given.
    "HIPBLASLT_TENSILE_LIBPATH": "library overlay under test: restoring would unset it",
    "TENSILE_LIBPATH": "library overlay under test: restoring would unset it",
    # an aiter .co overlay (a sweep trial, a step-8 kept fix): aiter <= 0.1.2x reads
    # AITER_ASM_DIR, 0.1.26+ rewrites it from AITER_META_DIR at import
    "AITER_ASM_DIR": "aiter kernel overlay under test: restoring would unset it",
    "AITER_META_DIR": "aiter kernel overlay under test: restoring would unset it",
}
# The overlay vars above are held back only when THIS run sets them (an overlay under
# test).  Unset here and set at capture, the capture's value is the model's own
# library setting, not an overlay: holding it back then ran the replay without it --
# the 2026-11-07 varlen capsule's chain `aten.mm` failed with
# HIPBLAS_STATUS_INVALID_VALUE (no gfx1250 kernels on the stock path).  Restored,
# and named in the banner (via _RESTORED_OVERLAY across the re-exec).
_OVERLAY_VARS = ("HIPBLASLT_TENSILE_LIBPATH", "TENSILE_LIBPATH", "AITER_ASM_DIR", "AITER_META_DIR")
_RESTORED_OVERLAY = "_NAN_REPLAY_RESTORED_OVERLAY"


def _abbrev(v, width=52):
    """A value short enough for a banner line.  LD_LIBRARY_PATH runs to hundreds
    of characters and would push the comparison off the right of the terminal,
    which is the one thing this banner exists to make easy to see."""
    if v is None:
        return "UNSET"
    v = str(v)
    return v if len(v) <= width else v[:width - 3] + "..."


def env_spec(cap):
    """The namespace this capsule's `env` was recorded under, and what to hold back."""
    ei = cap.get("env_info") or {}
    spec = dict(_LEGACY_ENV_SPEC) if not ei.get("prefixes") else {
        "prefixes": list(ei["prefixes"]),
        "exact": list(ei.get("exact") or []),
        "no_restore": list(ei.get("no_restore") or []),
    }
    spec["hold"] = set(spec["no_restore"]) | set(_ENV_NEVER_RESTORE)
    return spec


def _in_namespace(k, spec):
    return k.startswith(tuple(spec["prefixes"])) or k in set(spec["exact"])


def env_delta(cap):
    """How this process's environment differs from the one the capsule recorded.

    Three buckets because they have three different causes.  `differs` is a
    value this run overrides; `missing` is one the capture had and this run does
    not; `extra` is one this run has and the capture did not -- that last is the
    bucket a merge-only restore can never fix, and the one that lets a replay
    container's own HSA_DISABLE_CACHE=1 ride along into an experiment captured
    without it.  Vars in spec["hold"] are excluded from all three: they are
    expected to differ and are not restored.
    """
    rec = cap.get("env") or {}
    spec = env_spec(cap)
    differs, missing, extra = [], [], []
    for k, v in sorted(rec.items()):
        if k in spec["hold"]:
            continue
        cur = os.environ.get(k)
        if cur is None:
            missing.append((k, v, None))
        elif cur != v:
            differs.append((k, v, cur))
    for k, cur in sorted(os.environ.items()):
        if k in spec["hold"] or k in rec or k == _REEXEC:
            continue
        if _in_namespace(k, spec):
            extra.append((k, None, cur))
    return differs, missing, extra


def restore_env_dict(cap):
    """This process's environment, re-created as the capture saw it.

    Differs from the `dict(os.environ)` + `update(cap['env'])` this used to be in
    two ways, both of which turned out to matter:

      * it UNSETS.  A variable set here and absent at capture used to survive,
        so a merge restored a SUPERSET of the capture rather than the capture.
        Only the capsule's own namespace is touched, so PATH, HOME, PYTHONPATH
        and the container's plumbing are left exactly as they are.
      * it HOLDS BACK device selection -- see _ENV_NEVER_RESTORE.

    Anything outside the namespace is left alone in both directions: the capsule
    never recorded it, so this function has nothing to say about it.  What it
    was is in env_info()["not_captured"], and env_banner reports the count.
    """
    spec = env_spec(cap)
    rec = cap.get("env") or {}
    env = dict(os.environ)
    for k in list(env):
        if k not in spec["hold"] and k not in rec and _in_namespace(k, spec):
            del env[k]
    for k, v in rec.items():
        if k not in spec["hold"]:
            env[k] = v
    restored = [k for k in _OVERLAY_VARS if k in rec and k not in os.environ]
    for k in restored:
        env[k] = rec[k]
    if restored:
        env[_RESTORED_OVERLAY] = ",".join(restored)
    return env


def env_banner(cap):
    """Whether this run's environment is the capture's, said out loud.

    Printed unconditionally rather than behind a flag because the failure it
    catches has no other symptom: a replay under the wrong environment does not
    error, it returns a clean 0/N that reads like a result about the kernel.
    """
    rec = cap.get("env") or {}
    # Which Tensile library this process was handed -- the one fact a patched-library
    # run must show, since a missing overlay looks exactly like a fix that failed.
    lp = os.environ.get("HIPBLASLT_TENSILE_LIBPATH")
    from_cap = set(filter(None, os.environ.get(_RESTORED_OVERLAY, "").split(",")))
    print("libpath:        %s" % (
        "%s (HIPBLASLT_TENSILE_LIBPATH%s)"
        % (lp, ", restored from capture: the model's own, not an overlay"
           if "HIPBLASLT_TENSILE_LIBPATH" in from_cap else "") if lp else
        "stock (HIPBLASLT_TENSILE_LIBPATH unset)"))
    if not rec:
        print("env:            not recorded (capsule predates env capture)")
        return
    restored = bool(os.environ.get(_REEXEC))
    differs, missing, extra = env_delta(cap)
    # Variables this replay sets through its OWN flags: an intended A/B, not a mismatch to fix
    # (a --no-hipblaslt run printed "STILL DIFFERENT AFTER --restore-env", 2026-11-11).
    own = {}
    if args.no_hipblaslt:
        own["TORCH_BLAS_PREFER_HIPBLASLT"] = "--no-hipblaslt"
    if args.solution_index is not None or (_MUTATING and args.patch_solution_index is not None):
        own["TENSILE_SOLUTION_INDEX"] = "--solution-index"
    if args.emit_log:
        own.update(HIPBLASLT_LOG_MASK="--emit-log", HIPBLASLT_LOG_FILE="--emit-log")
    ab = [r for r in differs + missing + extra if r[0] in own]
    differs, missing, extra = ([r for r in b if r[0] not in own] for b in (differs, missing, extra))
    rows = differs + missing + extra
    if not rows:
        print("env:            matches capture (%d var(s)%s)%s"
              % (len(rec), ", restored" if restored else "",
                 "  + %d A/B setting(s) of this replay's flags" % len(ab) if ab else ""))
    else:
        # Counted per bucket rather than as "N of M recorded": an `extra` is a
        # var the capture never recorded, so folding it into a fraction of the
        # recorded set reads as a bigger mismatch than it is -- and hides that
        # the three have different causes.
        print("env:            %d difference(s) from capture"
              " (%d changed, %d unset here, %d set here only)%s"
              % (len(rows), len(differs), len(missing), len(extra),
                 "  -- STILL DIFFERENT AFTER --restore-env" if restored
                 else "  -- pass --restore-env"))
        for k, want, cur in rows[:12]:
            print("  %-32s capture=%s  here=%s" % (k, _abbrev(want), _abbrev(cur)))
        if len(rows) > 12:
            print("  ... and %d more" % (len(rows) - 12))
    for k, want, cur in ab:
        print("  A/B (%s):%s %s capture=%s  here=%s -- intended, not a mismatch"
              % (own[k], " " * max(1, 14 - len(own[k])), k, _abbrev(want), _abbrev(cur)))
    # The two things a restore cannot fix, reported rather than left to be
    # inferred from a number that came out lower than expected.
    spec = env_spec(cap)
    for k in sorted(from_cap):
        print("  restored from capture: %s -- set at capture, unset here: the model's"
              " own setting, not an overlay under test" % k)
    loader = False
    for k in sorted(spec["hold"]):
        if k not in rec or os.environ.get(k) == rec[k]:
            continue
        why = _ENV_NEVER_RESTORE.get(k, "not re-applied by a replay")
        loader = loader or k.startswith("LD_")
        print("  held back:      %s -- %s" % (k, why))
        print("                  capture=%s  here=%s"
              % (_abbrev(rec[k]), _abbrev(os.environ.get(k))))
    if loader:
        # Worth saying plainly: a differing loader path is not a setting that
        # drifted, it is a different image.  Everything below this point --
        # every rate, every sweep row -- is then about a build that is not the
        # one that failed, and no amount of --restore-env fixes that.
        print("                  ^ this run resolves its libraries from a different"
              " tree than the capture;")
        print("                    check the `captured build:` line above before"
              " reading any rate.")
    nc = (cap.get("env_info") or {}).get("not_captured")
    if nc:
        print("  not carried:    %d var(s) were set at capture and fall outside the"
              " capsule's namespace;" % len(nc))
        print("                  names are in capsule['env_info']['not_captured']")


def build_banner(cap):
    """The build the capsule was captured on.

    Printed, not compared: hipBLASLt initialises on the first GEMM, which is
    still ahead of us, so the libraries this process will dispatch into are not
    mapped yet and a live comparison here would be reading an empty set.  It is
    for the bug report and for telling two capsules apart -- torch and HIP
    version strings do not pin a build.
    """
    b = cap.get("build") or {}
    if not b:
        return
    if b.get("rocm_version") or b.get("os_release"):
        print("captured build: rocm %s  %s"
              % (b.get("rocm_version") or "?", b.get("os_release") or ""))
    for path, st in sorted((b.get("libs") or {}).items()):
        base = os.path.basename(path)
        if base.startswith(("libhipblaslt", "librocblas")):
            print("  %-28s %s  (%s bytes)"
                  % (base, path, (st or {}).get("size", "?")))


def capture_gpu_procs(cap):
    """(total, detail) processes on the capture's device, or (None, reason).

    The capsule records two different observations and they MULTIPLY: the /proc
    scan (`gpu_procs`) counts fan-out inside the capturing container, and the
    launcher's declaration (`containers_per_gpu`, from
    NAN_CAPTURE_CONTAINERS_PER_GPU) counts peer containers the scan cannot see
    because /proc is per PID namespace.  A model forking 6 workers inside each
    of 2 containers per GPU is 12 processes, and neither field alone says so.

    Every caller that sizes co-tenants or warns about a solo run goes through
    here, so they cannot disagree with each other about what the capture saw.
    """
    ci = cap.get("contention") or {}
    if not ci:
        return None, "the capsule records no contention at all"
    # The scan at write time can miss a sibling that already exited (an mp.spawn
    # trio has recorded 2); the scan at arm time, when present, cannot.
    scanned = max(ci.get("gpu_procs") or 0, ci.get("gpu_procs_at_arm") or 0) or None
    if scanned and not ci.get("gpu_proc_helpers_excluded"):
        # Capsules from before 2026-11-09 counted torch.compile's compile workers
        # (they hold /dev/kfd, launch nothing): gpu_procs=34 beside one trainer.
        from op_nan_capture import _is_compile_helper
        helpers = sum(1 for x in (ci.get("gpu_proc_list") or [])
                      if x[1] and _is_compile_helper(x[2]))
        scanned = max(1, scanned - helpers)
    declared = ci.get("containers_per_gpu")
    if not scanned and not declared:
        return None, ("the capsule records no process count (pre-contention"
                      " capsule, or the /proc scan was blocked)")
    if scanned and declared:
        return (int(scanned) * int(declared),
                "%d process(es) in this container x %d container(s) per GPU"
                % (scanned, declared))
    if declared:
        # The scan failed but the launcher spoke.  Its number is per container,
        # so this is a floor: it assumes one process in each.
        return (int(declared),
                "%d container(s) per GPU declared by the launcher; the /proc"
                " scan could not tell how many processes each ran, so this is"
                " a FLOOR" % declared)
    # Scan only.  Trustworthy for in-process fan-out, blind to peer containers.
    return (int(scanned),
            "%d process(es) by /proc scan; no NAN_CAPTURE_CONTAINERS_PER_GPU"
            " was declared, so peer CONTAINERS -- if any -- are not counted"
            % scanned)


def contention_banner(cap):
    """What the capture saw, and whether this run is about to match it."""
    ci = cap.get("contention") or {}
    if not ci:
        print("contention:     not recorded (capsule predates contention capture)")
        return
    argv = ci.get("argv") or []
    pa = ci.get("parent_argv") or []
    total, detail = capture_gpu_procs(cap)
    print("contention:     %s process(es) on the device at capture%s"
          % (total if total else "?",
             ", capturing process was an mp worker" if ci.get("mp_child") else ""))
    print("                %s" % detail)
    # Headline whichever command line a human can act on, and label the other
    # for what it is, so '--workers 6' is never the second-class line.
    if _is_spawn_boilerplate(argv) and pa:
        print("  captured under: %s" % " ".join(pa))
        print("                  ^ the PARENT's command line -- look for the"
              " fan-out flag here")
        print("  worker argv:    %s" % " ".join(argv))
    else:
        if argv:
            print("  captured under: %s" % " ".join(argv))
        if pa and pa != argv:
            print("  parent:         %s" % " ".join(pa))
    if ci.get("mem_used_by_others_bytes"):
        # Note this has a floor of ~1 GB even on an idle device with one
        # process: it is (total - free - our allocator), so the capturing
        # process's own HIP context and kernel images land in "others".  Read
        # big numbers, not small ones.
        print("  held by others: %.1f GB"
              % (ci["mem_used_by_others_bytes"] / 1e9))
        # The cross-check the two fields make possible.  Device memory is read
        # from the device, so it crosses the PID namespace that defeats the
        # /proc scan: a big number next to a process count of 1 and no declared
        # peer containers means a co-tenant WAS there and nothing counted it.
        # Say so here, where it is still cheap to fix, rather than letting
        # --contend-auto resolve to 0 and report a clean 0/N an hour later.
        # Gate on the COMPUTED total, not on the raw fields: a launcher that
        # declares 1 container per GPU has not explained away a device holding
        # 48 GB for someone else, so that case has to warn too.
        world = int((cap.get("dist") or {}).get("world_size") or 1)
        if (total or 1) <= 1 and ci["mem_used_by_others_bytes"] > 4e9 and world > 1:
            # One rank per GPU under torchrun: the peer ranks' RCCL/IPC buffers and
            # contexts are mapped on this device too (15.3 GB on a clean 8-rank qwen3
            # MoE capture, 2026-11-14) -- expected, not an uncounted co-tenant.
            print("  (world size %d: peer ranks' communication buffers and contexts sit on"
                  " this device too --" % world)
            print("   expected for a multi-rank job, not by itself a co-tenant; a"
                  " co-tenant would show as")
            print("   processes from another job on the node -- `rocm-smi --showpids`"
                  " during the run)")
        elif (total or 1) <= 1 and ci["mem_used_by_others_bytes"] > 4e9:
            print("  !! %.1f GB held by a process this capsule counted as"
                  " nobody." % (ci["mem_used_by_others_bytes"] / 1e9))
            print("     This capsule says %d process(es) on the device, but"
                  " device memory says" % (total or 1))
            print("     otherwise -- and memory is read from the DEVICE, so it"
                  " sees across the PID")
            print("     namespace that the /proc scan cannot. The co-tenant was"
                  " in another")
            print("     container. --contend-auto resolves to 0 here and will"
                  " refuse rather than")
            print("     replay solo: pass --contend N yourself (or"
                  " --contend-with the model), and")
            print("     set NAN_CAPTURE_CONTAINERS_PER_GPU in the launcher"
                  " before the next capture.")


# ======================================================================================
# [C] chain replay
#
# A single-op capsule can only ever answer "did THIS op produce a non-finite
# value".  It cannot answer "was this op handed a finite but WRONG operand", and
# inputs_finite=True is routinely misread as if it had.  A format-3 capsule
# carries the op's ancestry, so the chain can be re-executed from its roots and
# each intermediate compared against what the live run recorded.
# ======================================================================================

def _apply_written_operands(func, bound):
    """--written-operands on the args `func` writes (see its help); in place."""
    if args.written_operands == "captured":
        return
    for i in sorted(op_writes(func)["idx"]):
        if i < len(bound) and isinstance(bound[i], torch.Tensor) \
                and bound[i].is_floating_point():
            if args.written_operands == "zero":
                bound[i].zero_()
            else:
                bound[i].normal_()


def _restore_kept_uninit(cap, bound, kwb, dev):
    """The terminal op's allocator-fresh operands, as the capture saw them: the chain re-runs
    the allocator, which hands back whatever this process left there, but a small one keeps
    its captured bytes (op_nan_capture._mark_uninit, uninit_kept) -- an op that reads
    uninitialised memory makes its NaN from those bytes.  Copied in place, so the buffer the
    chain allocated stays the one the op gets.  Unpacked once per device."""
    kept = cap.get("_kept_uninit")
    if kept is None or kept[0] != dev:
        ts = {}
        for i, p in enumerate(cap.get("args") or ()):
            if isinstance(p, dict) and p.get("uninit_kept"):
                ts[i] = unpack(p, dev)
        for k, p in (cap.get("kwargs") or {}).items():
            if isinstance(p, dict) and p.get("uninit_kept"):
                ts[k] = unpack(p, dev)
        kept = cap["_kept_uninit"] = (dev, ts)
        if ts:
            print("kept uninitialised operand(s) restored before the terminal op: %s"
                  % ", ".join(("arg%d" % k) if isinstance(k, int) else k for k in ts))
    for k, t in kept[1].items():
        tgt = bound[k] if isinstance(k, int) and k < len(bound) else (kwb.get(k) if isinstance(k, str) else None)
        if isinstance(tgt, torch.Tensor) and tgt.shape == t.shape and tgt.dtype == t.dtype:
            tgt.copy_(t)


def _bind(obj, env, pool=None, dev=None):
    """`obj` with every {_CHAIN_REF: key} replaced by the live tensor in `env`.

    Falls back to the pool for a key no earlier op bound: the capture stores a
    value for anything it could not prove was recomputable, so this is the
    normal path for a root and a safety net for anything the producer analysis
    classified conservatively.  Only a key that is in neither is an error, and
    that means the capsule is internally inconsistent -- worth saying out loud
    rather than failing later inside an aten kernel.
    """
    if isinstance(obj, dict) and _CHAIN_REF in obj:
        k = obj[_CHAIN_REF]
        if k not in env:
            if pool is not None and k in pool:
                env[k] = unpack(pool[k], dev)
            else:
                raise KeyError("chain references %r, which is neither a root nor "
                               "produced by an earlier op in the chain" % k)
        return env[k]
    if isinstance(obj, tuple):
        return tuple(_bind(o, env, pool, dev) for o in obj)
    if isinstance(obj, list):
        return [_bind(o, env, pool, dev) for o in obj]
    if isinstance(obj, dict) and _MEMFMT_TAG in obj:
        return unpack(obj, dev)          # a recorded torch.memory_format
    if isinstance(obj, dict):
        return dict((k, _bind(v, env, pool, dev)) for k, v in obj.items())
    return obj


def _restore_rng(rng):
    if not rng:
        return
    if rng.get("cpu") is not None:
        torch.set_rng_state(rng["cpu"].cpu())
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["cuda"].cpu())


def _chain_roots(cap, dev):
    return dict((k, unpack(cap["pool"][k], dev)) for k in cap["roots"])


def _run_chain_once(cap, dev, compare=True, record_geom=False, repeat=None, check_wrong=None):
    """Execute the chain from its roots. -> (rows, env).

    One row per op: whether its outputs were finite, and -- when the capture
    stored intermediates -- whether they still match what the live run recorded.
    The divergence column is the point of the exercise: an op whose output is
    finite but DIFFERENT is the one that corrupted the operand everybody
    downstream then trusted.

    `repeat` (a dict, --chain-repeat-check): the first call stores each op's float
    outputs in it as the reference; every call keeps its own outputs as `prev` (held,
    not copied -- the next run allocates new ones).  A row is `nonrepeatable` when an
    output is not bit-identical to the reference, `changed` when not to the run before.
    """
    # Recorded intermediates are synthesized too: comparing against them would
    # mark every op DIVERGED.
    walk = compare
    compare = compare and not _synthetic(cap)
    # GEMM rows vs a higher-precision recompute: always in the [C1] walk (one pass, not a
    # timing measurement); in the [C2] loop once the walk found a wrong value, or on request.
    # Synthetic operands too: the reference is recomputed on the SAME operands, so it needs
    # no recorded values.  2026-11-09 (qwen3 MoE, hipBLASLt leaving tiles unwritten): with
    # the check off under --synthesize, the unwritten tiles held finite leftovers and the
    # replay read 0/186000 on a GEMM that is wrong on every call.
    if check_wrong is None:
        check_wrong = walk or _CHECK_WRONG[0] or args.wrong_value
    env = _chain_roots(cap, dev)
    pool, rows = cap["pool"], []
    last = len(cap["chain"]) - 1
    for n, rec in enumerate(cap["chain"]):
        _restore_rng(rec.get("rng"))
        if n != last and (is_collective_op(rec["op"]) or try_resolve_op(rec["op"]) is None):
            outs = _recorded_collective(cap, rec, env, dev)
            rows.append({"op": rec["op"], "call_index": rec["call_index"], "is_view": rec["is_view"],
                         "finite": all(bool(torch.isfinite(t).all()) for t in outs if t.is_floating_point()),
                         "diverged": None, "maxerr": None, "slots": [], "uninitialised": False,
                         "recorded": True, "version_ok": rec.get("version_ok"), "needs_rng": rec.get("needs_rng")})
            continue
        func = resolve_op(rec["op"])
        bound = _bind(rec["args"], env, pool, dev)
        kwb = _bind(rec["kwargs"], env, pool, dev)
        bound = _chain_derive(cap, n, rec, func, bound, kwb, env)
        if n == last and rec["op"] == cap["op"]:
            _restore_kept_uninit(cap, bound, kwb, dev)
            _apply_written_operands(func, bound)
        # Geometry is recorded here because this is the only place the op's real
        # operands exist -- but ONLY when asked, and the [C2] loop never asks.
        # [C2] is a race measurement that runs this body tens of thousands of
        # times; anything added to it is host-side work interleaved with the
        # kernel launches whose timing IS the experiment.  One deliberate pass
        # (record_chain_geometry) collects the same information for the sweep,
        # and leaves the measured path byte-identical to what it was.
        if record_geom and rec["call_index"] not in _CHAIN_GEOM:
            try:
                _CHAIN_GEOM[rec["call_index"]] = hipblaslt_geometry(func, bound)
            except Exception:                          # noqa: BLE001 -- never fatal
                _CHAIN_GEOM[rec["call_index"]] = None
        if func is torch.ops.aten.detach_.default and getattr(bound[0], "_base", None) is not None:
            # HF generate's KV cache (2026-11-20): `detach_` of a tensor the replay rebuilt as a
            # view raises "Can't detach views in-place" and killed the whole chain.  It changes
            # autograd metadata only -- the values are the operand's -- so the out-of-place
            # detach stands in.
            out = torch.ops.aten.detach.default(bound[0])
        else:
            # A rerun capsule (origin wrong_value, verify kind rerun): the terminal op runs
            # twice per chain run, the second time on copies of its pre-call operands and the
            # same generator state, and the run counts when the two disagree -- the capture's
            # own check, op_nan_capture.rerun_diff.
            rerun = (n == last and rec["op"] == cap["op"]
                     and (cap.get("verify") or {}).get("kind") == "rerun")
            if rerun:
                import op_nan_capture as onc
                ra, rk = onc.clone_operands(func, bound, kwb)
                rng0 = onc._rng_states()
            out = func(*bound, **kwb)
            if rerun and ra is not None:
                post = onc._rng_states()
                onc._set_rng_states(rng0)
                out2 = func(*ra, **rk)
                onc._set_rng_states(post)
                rer = onc.rerun_diff(func, bound, kwb, out, ra, rk, out2)
                rerun_hit = (rer is not None and not rer.get("int")
                             and rer["rel"] > rer["floor"])
        # tree_leaves, matching how the capture enumerated `produces`, so slot i
        # here is slot i there even for a nested return.
        live = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor)]
        finite, diverged, maxerr, slots = True, None, None, []
        if rec["op"] in _ALLOC_OPS:
            # empty_like & co. return whatever the allocator hands back: neither its
            # finiteness nor a difference from the recording means anything, and
            # scoring it convicts the allocator as the chain's culprit (DIVERGED)
            # or as the loop's first non-finite op.  Bind and move on.
            for key, t in zip(rec["produces"], live):
                if key:
                    env[key] = t
            rows.append({"op": rec["op"], "call_index": rec["call_index"],
                         "is_view": rec["is_view"], "finite": True, "diverged": None,
                         "maxerr": None, "slots": [], "uninitialised": True,
                         "version_ok": rec.get("version_ok"),
                         "needs_rng": rec.get("needs_rng")})
            continue
        # Written out-params count towards finiteness, for the reason spelled out
        # in scan_targets: a fused backward hands its gradients back through
        # dq/dk/dv, not through the return, so scoring `live` alone leaves the
        # terminal op permanently `finite=yes` and [C2]'s tally structurally
        # unable to name it.  They are deliberately NOT compared against the pool
        # -- they are not in `produces`, and their pre-call contents are
        # uninitialised, so there is nothing honest to diverge against.
        # A fused op with a value check (math_refs.VALUE_CHECKS) is judged by it alone: its
        # scratch outputs keep slots it never writes (a split-KV op's unused splits hold
        # whatever the allocator had), so scanning them scores garbage, not the kernel.
        rerun_row = n == last and (cap.get("verify") or {}).get("kind") == "rerun" \
            and rec["op"] == cap["op"] and func is not torch.ops.aten.detach_.default
        # a rerun capsule's terminal op is judged by the rerun alone (below)
        vown = None if rerun_row else _value_owned(func, bound, kwb, out)
        for _wi in sorted(op_writes(func)["idx"]) if vown is None and not rerun_row else ():
            if _wi < len(bound):
                _wt = bound[_wi]
                if (isinstance(_wt, torch.Tensor) and _wt.is_floating_point()
                        and not torch.isfinite(_scannable(_wt)).all().item()):
                    finite = False
        nonrep = changed = None
        if repeat is not None:
            # What the op produced: its returns AND its written float operands (an out-param op
            # returns nothing -- before 2026-11-20 every such op read "repeated exactly"); a
            # value-checked op is compared on what its consumer reads, since its raw scratch
            # keeps slots it never writes (fresh garbage each run).
            if vown is not None:
                outs = _owned_got(rec["op"], func, bound, kwb, out)
            else:
                outs = [t for t in live if t.is_floating_point() and t.numel()]
                outs += [bound[i] for i in sorted(op_writes(func)["idx"])
                         if i < len(bound) and isinstance(bound[i], torch.Tensor)
                         and bound[i].is_floating_point() and bound[i].numel()
                         and not any(bound[i] is t for t in outs)]
            ci = rec["call_index"]
            ref, prev = repeat.setdefault("ref", {}), repeat.setdefault("prev", {})
            if ci not in ref:
                ref[ci] = [t.detach().clone() for t in outs]
            else:
                nonrep = any(not _bits_equal(t, r)
                             for t, r in zip(outs, ref[ci]))
                changed = any(not _bits_equal(t, r)
                              for t, r in zip(outs, prev[ci]))
            prev[ci] = outs
        if vown is not None and not math.isfinite(vown[0]):
            finite = False
        for slot, (key, t) in enumerate(zip(rec["produces"], live)):
            if (vown is None and not rerun_row and t.is_floating_point()
                    and not torch.isfinite(_scannable(t)).all().item()):
                finite = False
            if key:
                env[key] = t
            # Float tensors only.  A fused kernel's returns are not all values:
            # flash attention hands back philox seed and offset as int64, and
            # those are SUPPOSED to differ between runs -- comparing them reports
            # a spurious ~1e16 divergence on the one op the walk most wants to
            # read, which is how a bystander gets convicted.  Empty tensors
            # (the unused slots in that signature) carry nothing to compare.
            if not (compare and key and key in pool
                    and t.is_floating_point() and t.numel()):
                continue
            recorded = unpack(pool[key], t.device)
            if recorded.shape != t.shape:
                continue
            a, b = _scannable(t).double(), _scannable(recorded).double()
            m = torch.isfinite(a) & torch.isfinite(b)
            e = (a[m] - b[m]).abs().max().item() if m.any() else 0.0
            same_nonfinite = torch.equal(~torch.isfinite(a), ~torch.isfinite(b))
            maxerr = e if maxerr is None else max(maxerr, e)
            if e > 0 or not same_nonfinite:
                diverged = True
                slots.append(slot)
            elif diverged is None:
                diverged = False
        # A capsule captured for a WRONG FINITE value (origin wrong_value, the
        # capture's verify_every / bounds checks) reproduces when the terminal op's
        # output fails the same check again -- not when it goes non-finite.  Scored
        # as finite=False so every tally and verdict below counts it; `wrong` says why.
        wrong = None
        if rerun_row:
            if ra is not None and rerun_hit:
                wrong, finite = rer["rel"], False
        elif vown is not None and finite and (check_wrong or (n == last and cap.get("verify"))):
            thr = ((cap.get("verify") or {}).get("threshold") if n == last else None) or vown[1]
            if not vown[0] <= thr:
                wrong, finite = vown[0], False
        elif n == last and finite and cap.get("verify"):
            wrong = _wrong_value(cap["verify"], func, bound, kwb, live)
            if wrong is not None:
                finite = False
        if wrong is None and finite and check_wrong and _is_gemm(rec["op"]):
            wrong = _wrong_value({"kind": "reference"}, func, bound, kwb, live)
            if wrong is not None:
                finite = False
        rows.append({"op": rec["op"], "call_index": rec["call_index"],
                     "is_view": rec["is_view"], "finite": finite,
                     "diverged": diverged, "maxerr": maxerr, "slots": slots,
                     "nonrepeatable": nonrep, "changed": changed,
                     "version_ok": rec.get("version_ok"),
                     "needs_rng": rec.get("needs_rng"), "wrong": wrong})
    return rows, env


_CHECK_WRONG = [False]      # set when the [C1] walk found a wrong finite GEMM value


_TWIN_ON = [False]          # the capsule was captured by the twin check (verify kind twin): judge by its twin


def _twin_owned(func):
    """The capsule's op is judged by its twin (twins.py): a twin capsule, no hand-written check for the op."""
    import op_nan_capture as onc
    return _TWIN_ON[0] and not onc._has_value_check(func) and onc._has_twin(func)


def _has_value_check_op(func):
    import op_nan_capture as onc
    return onc._has_value_check(func) or _twin_owned(func)


def _owned_rel_err(op_name, func, args, kwargs, out=None):
    """(rel err, floor, where) from the op's hand-written value check or, for a twin capsule, its twin -- or None
    when neither applies to this call.  A twin of an in-place op needs pre-call inputs the replay does not keep per
    run: not modelled here (the finiteness scans apply)."""
    import op_nan_capture as onc
    if onc._has_value_check(func):
        return math_refs.value_rel_err(op_name, func, args, kwargs)
    if _twin_owned(func):
        r = onc._twins.rel_err(op_name, func, args, kwargs, None, out)
        return None if r is None else r[:3]
    return None


def _owned_got(op_name, func, args, kwargs, out=None):
    """What a value-checked / twin-checked op's consumer reads (the repeat check compares that)."""
    import op_nan_capture as onc
    if onc._has_value_check(func):
        return [math_refs.value_check(op_name, func, args, kwargs)["got"]]
    return list(onc._twins.check(op_name, func, args, kwargs, None, out)["got"].values())


def _value_owned(func, args, kwargs, out=None):
    """(rel err, floor, where) from the op's math_refs value check (or its twin, for a twin capsule), or None when
    it has none or does not model this call -- then the plain finiteness scans apply."""
    return _owned_rel_err(str(func), func, args, kwargs, out)
def _is_gemm(op):
    import op_nan_capture as onc
    return ".".join(op.split(".")[:2]) in onc._GEMM_OPS


class _CollectiveNoise(object):
    """--collective-noise: an async all_reduce in flight across each chain run.

    Under torchrun only (RANK set); the process group is created on first use, on
    the replay's device, over RCCL.  Every rank must call start() the same number of
    times -- hence a fixed --chain-iters."""

    def __init__(self, dev):
        import torch.distributed as dist
        if "RANK" not in os.environ:
            sys.exit("--collective-noise needs torchrun (it all-reduces across ranks)")
        if not dist.is_initialized():
            torch.cuda.set_device(dev)
            dist.init_process_group("nccl")
        self.dist = dist
        self.buf = torch.zeros(max(1, int(args.collective_noise * (1 << 20) / 4)),
                              dtype=torch.float32, device=dev)
        print("  collective noise: %.1f MB all_reduce in flight per chain run, rank %d of %d"
              % (args.collective_noise, dist.get_rank(), dist.get_world_size()), flush=True)

    def start(self):
        return self.dist.all_reduce(self.buf, async_op=True)


def _wrong_value(v, func, args, kwargs, live):
    """The capsule's wrong-value check re-run on a replayed output: the bound ratio
    or the reference error when it fails again, else None."""
    import op_nan_capture as onc
    if _has_value_check_op(func):
        # A fused op whose result is in its written operands (math_refs.VALUE_CHECKS), or a twin capsule's op:
        # `args` are the bound operands after the call, which is what the check reads.
        got = _owned_rel_err(str(func), func, args, kwargs, live[0] if len(live) == 1 else live)
        if got is None:
            return None
        err, floor, _where = got
        thr = v.get("threshold") or floor
        return err if not err <= thr else None
    if v.get("kind") == "bound" and onc._op_names(func)[1] in onc._SOFTMAX_OPS | onc._SDPA_OPS:
        bnd = onc._bound_ratio(func, args, kwargs, list(live))
        return bnd[0] if bnd is not None and not bnd[0] <= 1.0 + v.get("slack", bnd[1]) else None
    outs = [t for t in live if t.is_floating_point() and t.numel()]
    if len(outs) != 1:
        return None
    c = outs[0]
    if v.get("kind") == "bound":
        r = onc._gemm_bound_ratio(func, args, kwargs, c)
        return r if r is not None and not r <= 1.0 + v.get("slack", 0.02) else None
    err = onc._ref_rel_err(func, args, kwargs, c)
    thr = v.get("threshold") or onc._ref_floor(c.dtype)
    if err is None or err <= thr:
        return None
    _note_grouped_wrong(args, kwargs, c, thr)
    return err


_GROUPED_WRONG = []         # row counts of the wrong group(s) of a grouped GEMM, once known


def _note_grouped_wrong(args, kwargs, c, thr):
    """A grouped GEMM failed the value check: say which group(s), once per process.
    The `GROUPED_WRONG_ROWS=` line is what the ladder's kernel step passes back as
    --grouped-wrong-rows, so the kernel named is the wrong group's, not the first one."""
    import op_nan_capture as onc
    if _GROUPED_WRONG:
        return
    errs = onc._grouped_rel_errs(args, kwargs, c)
    bad = [x for x in errs if x[2] > thr] or sorted(errs, key=lambda x: -x[2])[:1]
    if not bad:
        return
    _GROUPED_WRONG.extend(sorted({r for _, r, _ in bad}))
    print("  GROUPED_WRONG_ROWS=%s  (%d of %d groups: %s)"
          % (",".join(map(str, _GROUPED_WRONG)), len(bad), len(errs),
             "; ".join("group %d, %d rows, rel err %.3g" % x for x in bad[:4])), flush=True)


def _cpu64(obj):
    """`obj` moved to the fp64 CPU reference: float tensors -> float64 on cpu, other
    tensors -> cpu, and the op's own placement kwargs rewritten to match.

    The kwargs matter as much as the tensors: a recorded aten._to_copy carries
    dtype=float32, device=cuda:0, and replayed verbatim it moves the "CPU fp64"
    reference back onto the GPU in fp32 -- the next op then mixes devices and the
    whole walk reads `unavailable`."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().double() if obj.is_floating_point() else obj.detach().cpu()
    if isinstance(obj, torch.device):
        return torch.device("cpu")
    if isinstance(obj, torch.dtype):
        return torch.float64 if obj.is_floating_point else obj
    if isinstance(obj, tuple):
        return tuple(_cpu64(o) for o in obj)
    if isinstance(obj, list):
        return [_cpu64(o) for o in obj]
    if isinstance(obj, dict):
        return dict((k, _cpu64(v)) for k, v in obj.items())
    return obj


def _schema_args(func, a, kw):
    """{argument name: value} for a call, from the op's schema (defaults filled)."""
    out = {}
    for i, arg in enumerate(func._schema.arguments):
        if i < len(a):
            out[arg.name] = a[i]
        elif arg.name in kw:
            out[arg.name] = kw[arg.name]
        elif arg.has_default_value():
            out[arg.name] = arg.default_value
    return out


def _attn_keep(sq, sk, causal, causal_offset, wl=-1, wr=-1):
    """(sq, sk) bool: key j visible to query i.  causal: j <= i + causal_offset;
    a window bounds j to [i + off - wl, i + off + wr] (a negative edge is unbounded,
    and flash-attn's causal is window right = 0).  None when every key is visible."""
    if causal:
        wr = 0
    if wl < 0 and wr < 0:
        return None
    i = torch.arange(sq)[:, None] + causal_offset
    j = torch.arange(sk)[None, :]
    keep = torch.ones(sq, sk, dtype=torch.bool)
    if wl >= 0:
        keep &= j >= i - wl
    if wr >= 0:
        keep &= j <= i + wr
    return keep


def _attn_bwd64(q, k, v, o, do, lse, scale, causal, causal_offset, keep=None):
    """fp64 attention backward from the forward's saved logsumexp, (b, h, s, d)
    layout, k/v heads may divide q heads (GQA/MQA).  -> (dq, dk, dv).

      P = exp(scale * Q K^T - lse),  dS = P * (dO V^T - rowsum(dO * O))
      dq = scale * dS K,  dk = scale * dS^T Q,  dv = P^T dO

    causal: key j is visible to query i iff j <= i + causal_offset (flash-attn
    aligns bottom-right: offset = sk - sq; SDPA top-left: 0).  `keep` (sq, sk), when
    given, replaces that mask (windows)."""
    g = q.shape[1] // k.shape[1]
    kq, vq = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    P = torch.exp(scale * q @ kq.transpose(-1, -2) - lse.unsqueeze(-1))
    if keep is None and causal:
        keep = _attn_keep(q.shape[2], k.shape[2], True, causal_offset)
    if keep is not None:
        P = torch.where(keep, P, torch.zeros((), dtype=P.dtype))
    dS = P * (do @ vq.transpose(-1, -2) - (do * o).sum(-1, keepdim=True))
    dq = scale * dS @ kq
    dk = (scale * dS.transpose(-1, -2) @ q).unflatten(1, (k.shape[1], g)).sum(2)
    dv = (P.transpose(-1, -2) @ do).unflatten(1, (k.shape[1], g)).sum(2)
    return dq, dk, dv


def _ref_flash_attn_backward(func, a, kw):
    """flash_attn._flash_attn_backward (dense, (b, s, h, d)); returns (dq, dk, dv)."""
    n = _schema_args(func, a, kw)
    if n.get("dropout_p"):
        raise NotImplementedError("dropout_p=%s" % n["dropout_p"])
    if n.get("softcap"):
        raise NotImplementedError("softcap=%s" % n["softcap"])
    if n.get("alibi_slopes") is not None:
        raise NotImplementedError("alibi_slopes")
    wl, wr = n.get("window_size_left", -1), n.get("window_size_right", -1)
    if wl != -1 or wr not in (-1, 0):
        raise NotImplementedError("window_size=(%s, %s)" % (wl, wr))
    q, k, v, o, do = (n[x].transpose(1, 2) for x in ("q", "k", "v", "out", "dout"))
    scale = n.get("softmax_scale") or q.shape[-1] ** -0.5
    dq, dk, dv = _attn_bwd64(q, k, v, o, do, n["softmax_lse"], scale, n.get("causal"),
                             k.shape[2] - q.shape[2])
    return tuple(t.transpose(1, 2) for t in (dq, dk, dv))


def _ref_sdpa_flash_backward(func, a, kw):
    """aten._scaled_dot_product_flash_attention_backward ((b, h, s, d))."""
    n = _schema_args(func, a, kw)
    if n.get("dropout_p"):
        raise NotImplementedError("dropout_p=%s" % n["dropout_p"])
    if n.get("cum_seq_q") is not None and n["cum_seq_q"].numel():
        raise NotImplementedError("nested / varlen (cum_seq_q)")
    q = n["query"]
    scale = n.get("scale") or q.shape[-1] ** -0.5
    return _attn_bwd64(q, n["key"], n["value"], n["out"], n["grad_out"],
                       n["logsumexp"], scale, n.get("is_causal"), 0)


def _attn_plain(n):
    """Raise for the flash-attn variants the fp64 references do not model."""
    if n.get("dropout_p"):
        raise NotImplementedError("dropout_p=%s" % n["dropout_p"])
    if n.get("softcap"):
        raise NotImplementedError("softcap=%s" % n["softcap"])
    if n.get("alibi_slopes") is not None:
        raise NotImplementedError("alibi_slopes")


def _varlen_groups(cu_q, cu_k):
    """Sequences of a varlen batch grouped by (len_q, len_k), so each group is one
    batched fp64 matmul instead of thousands of tiny ones.  Yields (seq ids, (n, lq)
    q-row indices, (n, lk) k-row indices); sequences with no queries are skipped."""
    cu_q, cu_k = cu_q.long().cpu(), cu_k.long().cpu()
    lq, lk = cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1]
    for a, b in sorted(set(zip(lq.tolist(), lk.tolist()))):
        if a == 0:
            continue
        ids = ((lq == a) & (lk == b)).nonzero().flatten()
        yield (ids, cu_q[ids][:, None] + torch.arange(a),
               cu_k[ids][:, None] + torch.arange(b))


def _varlen_lse(lse, ids, iq, total_q):
    """softmax_lse rows of a varlen group as (n, h, lq).  Two layouts in the wild:
    packed (h, total_q) and padded (batch, h, max_seqlen_q)."""
    if lse.dim() == 2 and lse.shape[1] >= total_q:
        return lse[:, iq].permute(1, 0, 2)
    if lse.dim() == 3:
        return lse[ids][:, :, :iq.shape[1]]
    raise NotImplementedError("softmax_lse layout %s" % (tuple(lse.shape),))


def _ref_flash_attn_varlen_backward(func, a, kw):
    """flash_attn._flash_attn_varlen_backward ((total, h, d) + cu_seqlens); returns
    (dq, dk, dv).  Per sequence, batched by length; causal and windows aligned
    bottom-right within each sequence, as flash-attn does."""
    n = _schema_args(func, a, kw)
    _attn_plain(n)
    q, k, v, o, do = (n[x] for x in ("q", "k", "v", "out", "dout"))
    scale = n.get("softmax_scale") or q.shape[-1] ** -0.5
    wl, wr = n.get("window_size_left", -1), n.get("window_size_right", -1)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for ids, iq, ik in _varlen_groups(n["cu_seqlens_q"], n["cu_seqlens_k"]):
        sq, sk = iq.shape[1], ik.shape[1]
        if sk == 0:
            continue
        Q, O, dO = (t[iq].permute(0, 2, 1, 3) for t in (q, o, do))
        K, V = (t[ik].permute(0, 2, 1, 3) for t in (k, v))
        L = _varlen_lse(n["softmax_lse"], ids, iq, q.shape[0])
        keep = _attn_keep(sq, sk, n.get("causal"), sk - sq, wl, wr)
        gq, gk, gv = _attn_bwd64(Q, K, V, O, dO, L, scale, False, 0, keep=keep)
        dq[iq] = gq.permute(0, 2, 1, 3)
        dk[ik] = gk.permute(0, 2, 1, 3)
        dv[ik] = gv.permute(0, 2, 1, 3)
    return dq, dk, dv


# Math references for ops with no CPU kernel -- exactly the fused kernels the
# ladder is most often pointed at.  Keyed by the capsule's op string; each takes
# (func, cpu64 args, cpu64 kwargs) and raises NotImplementedError for a variant
# it does not model, which the walk reports as `unavailable`, never as a value.
def _ref_grouped_mm(func, a, kw):
    """aten._grouped_mm in fp64 on the CPU: one plain matmul per group (the kernel
    accepts only 16/32-bit floats, so `func` itself cannot be the reference).
    Rows past the last group's end are zero (the kernel leaves them unspecified)."""
    import op_nan_capture as onc
    got = onc._grouped_mm_ref(tuple(a), dict(kw), torch.float64)
    if got is None:
        raise NotImplementedError("_grouped_mm layout not modelled")
    return got[0]


_MATH_REFERENCES = {
    "flash_attn._flash_attn_backward.default": _ref_flash_attn_backward,
    "flash_attn._flash_attn_varlen_backward.default": _ref_flash_attn_varlen_backward,
    "aten._scaled_dot_product_flash_attention_backward.default": _ref_sdpa_flash_backward,
    "aten._grouped_mm.default": _ref_grouped_mm,
}
# The fused norms / activations of aiter, sgl_kernel and vLLM's _C (math_refs.py, shared with the capture's
# nan_origin): every tensor the op writes, from its inputs.
import math_refs  # noqa: E402
for _op in math_refs.REFERENCES:
    _MATH_REFERENCES[_op] = (lambda op: lambda func, a, kw: math_refs.reference(op, func, a, kw))(_op)


# ---- synthesized attention backward: out / softmax_lse derived from the forward ------
# --synthesize draws every float operand from its own statistics, so an attention
# backward's saved forward outputs (out, softmax_lse) are independent of q/k/v.  Then
# exp(scale * q.k - lse) overflows in a CORRECT kernel too, the replay NaNs on any
# build, and a fix can never be validated on it (the 2026-11-07 varlen reproducer
# kept NaNing with the fixed kernel).  For these ops the replay recomputes them from
# q/k/v with the matching forward; when it cannot, it says so on an `operands:` line.
# Per backward op: (forward op, forward kwargs from the backward's named args,
# {backward arg: forward output index}).  RNG outputs are taken over too when there
# is dropout, so the backward's mask matches the forward that made `out`.
def _fwd_flash_varlen(n):
    kw = dict((x, n[x]) for x in ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k",
                                  "max_seqlen_q", "max_seqlen_k", "dropout_p",
                                  "softmax_scale", "causal", "window_size_left",
                                  "window_size_right", "softcap", "alibi_slopes"))
    kw["return_softmax"] = False
    return ("flash_attn._flash_attn_varlen_forward.default", kw,
            dict({"out": 0, "softmax_lse": 1}, **({"rng_state": 3} if n["dropout_p"] else {})))


def _fwd_flash(n):
    kw = dict((x, n[x]) for x in ("q", "k", "v", "dropout_p", "softmax_scale", "causal",
                                  "window_size_left", "window_size_right", "softcap",
                                  "alibi_slopes"))
    kw["return_softmax"] = False
    return ("flash_attn._flash_attn_forward.default", kw,
            dict({"out": 0, "softmax_lse": 1}, **({"rng_state": 3} if n["dropout_p"] else {})))


def _fwd_sdpa_flash(n):
    if n.get("cum_seq_q") is not None and n["cum_seq_q"].numel():
        raise NotImplementedError("nested / varlen (cum_seq_q)")
    kw = {"query": n["query"], "key": n["key"], "value": n["value"],
          "dropout_p": n["dropout_p"], "is_causal": n["is_causal"],
          "return_debug_mask": False, "scale": n.get("scale")}
    return ("aten._scaled_dot_product_flash_attention.default", kw,
            dict({"out": 0, "logsumexp": 1},
                 **({"philox_seed": 6, "philox_offset": 7} if n["dropout_p"] else {})))


def _fwd_sdpa_efficient(n):
    kw = {"query": n["query"], "key": n["key"], "value": n["value"],
          "attn_bias": n.get("attn_bias"), "compute_log_sumexp": True,
          "dropout_p": n["dropout_p"], "is_causal": n["is_causal"], "scale": n.get("scale")}
    return ("aten._scaled_dot_product_efficient_attention.default", kw,
            dict({"out": 0, "logsumexp": 1},
                 **({"philox_seed": 2, "philox_offset": 3} if n["dropout_p"] else {})))


_ATTN_DERIVED = {
    "flash_attn._flash_attn_varlen_backward.default": _fwd_flash_varlen,
    "flash_attn._flash_attn_backward.default": _fwd_flash,
    "aten._scaled_dot_product_flash_attention_backward.default": _fwd_sdpa_flash,
    "aten._scaled_dot_product_efficient_attention_backward.default": _fwd_sdpa_efficient,
}
_DERIVED = {"state": {}, "cache": {}}      # op -> ("ok" | "failed", detail); key -> repl
_INCONSISTENT_BANNER = ("operands:       SYNTHETIC-INCONSISTENT (out/softmax_lse drawn "
                        "independently of q/k/v: a correct kernel can overflow on them; "
                        "cannot validate a fix)")


def _attn_fwd64(q, k, v, scale, keep):
    """fp64 attention forward, (b, h, s, d), GQA as _attn_bwd64. -> (out, lse); a row
    with no visible key gets out 0 and lse -inf, as the kernels return."""
    g = q.shape[1] // k.shape[1]
    kq, vq = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    s = scale * q @ kq.transpose(-1, -2)
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(s, -1)
    p = torch.where(torch.isfinite(lse).unsqueeze(-1), torch.exp(s - lse.unsqueeze(-1)),
                    torch.zeros((), dtype=s.dtype))
    return p @ vq, lse


def _derive_cpu64(op_name, n):
    """The fp64 CPU stand-in for _ATTN_DERIVED's forward, for the CPU chain reference
    (the forward kernels are GPU-only).  -> {backward arg: tensor}."""
    if op_name == "flash_attn._flash_attn_varlen_backward.default":
        _attn_plain(n)
        q, k, v = n["q"], n["k"], n["v"]
        scale = n.get("softmax_scale") or q.shape[-1] ** -0.5
        out = torch.zeros_like(q)
        lse = torch.full_like(n["softmax_lse"], float("-inf"))
        packed = lse.dim() == 2 and lse.shape[1] >= q.shape[0]
        for ids, iq, ik in _varlen_groups(n["cu_seqlens_q"], n["cu_seqlens_k"]):
            sq, sk = iq.shape[1], ik.shape[1]
            Q = q[iq].permute(0, 2, 1, 3)
            K, V = (t[ik].permute(0, 2, 1, 3) for t in (k, v))
            keep = _attn_keep(sq, sk, n.get("causal"), sk - sq,
                              n.get("window_size_left", -1), n.get("window_size_right", -1))
            o, l = _attn_fwd64(Q, K, V, scale, keep)
            out[iq] = o.permute(0, 2, 1, 3)
            if packed:
                lse[:, iq] = l.permute(1, 0, 2)
            else:
                lse[ids, :, :sq] = l
        return {"out": out, "softmax_lse": lse}
    if op_name == "flash_attn._flash_attn_backward.default":
        _attn_plain(n)
        q, k, v = (n[x].transpose(1, 2) for x in ("q", "k", "v"))
        scale = n.get("softmax_scale") or q.shape[-1] ** -0.5
        keep = _attn_keep(q.shape[2], k.shape[2], n.get("causal"), k.shape[2] - q.shape[2],
                          n.get("window_size_left", -1), n.get("window_size_right", -1))
        o, l = _attn_fwd64(q, k, v, scale, keep)
        return {"out": o.transpose(1, 2), "softmax_lse": l}
    if op_name == "aten._scaled_dot_product_flash_attention_backward.default":
        if n.get("dropout_p"):
            raise NotImplementedError("dropout_p=%s" % n["dropout_p"])
        q = n["query"]
        scale = n.get("scale") or q.shape[-1] ** -0.5
        keep = _attn_keep(q.shape[2], n["key"].shape[2], n.get("is_causal"), 0)
        o, l = _attn_fwd64(q, n["key"], n["value"], scale, keep)
        return {"out": o, "logsumexp": l}
    raise NotImplementedError("no fp64 forward for %s" % op_name)


def _derive_gpu(op_name, n):
    """Run the backward's own forward on its (synthesized) q/k/v.  -> {arg: tensor}."""
    fwd_name, fkw, outmap = _ATTN_DERIVED[op_name](n)
    res = resolve_op(fwd_name)(**fkw)
    repl = {}
    for name, i in outmap.items():
        t = res[i]
        old = n.get(name)
        if isinstance(old, torch.Tensor) and isinstance(t, torch.Tensor):
            if tuple(t.shape) != tuple(old.shape):
                raise RuntimeError("%s from %s has shape %s, the backward expects %s"
                                   % (name, fwd_name, tuple(t.shape), tuple(old.shape)))
            t = t.to(old.dtype)
        repl[name] = t
    for name in ("out", "softmax_lse", "logsumexp"):
        t = repl.get(name)
        if isinstance(t, torch.Tensor) and t.is_floating_point() \
                and not torch.isfinite(t[~torch.isneginf(t)] if name != "out" else t).all():
            raise RuntimeError("%s from %s is non-finite" % (name, fwd_name))
    return repl


def _derive_attn_operands(op_name, func, bound, kw, cache_key, cpu64=False):
    """Replace a synthesized attention backward's out / softmax_lse (and its RNG state
    under dropout) by the forward's, in `bound` (a list) and `kw` (a dict), in place.
    -> {arg name: tensor} actually replaced, or {} when the op is not one of these.

    The forward runs once per (cache_key, device): the [C2] loop then rebinds the same
    tensors every run and adds no kernel to the measured path.  On failure the
    operands stay as drawn and the run is flagged SYNTHETIC-INCONSISTENT."""
    if op_name not in _ATTN_DERIVED:
        return {}
    key = (cache_key, "cpu64" if cpu64 else "gpu")
    repl = _DERIVED["cache"].get(key)
    if repl is None:
        try:
            n = _schema_args(func, bound, kw)
            repl = _derive_cpu64(op_name, n) if cpu64 else _derive_gpu(op_name, n)
        except Exception as e:                         # noqa: BLE001
            repl = {}
            if not cpu64 and op_name not in _DERIVED["state"]:
                _DERIVED["state"][op_name] = ("failed", "%s: %s" % (
                    type(e).__name__, str(e).splitlines()[0][:100] if str(e) else ""))
                print("%s  [%s: %s]" % (_INCONSISTENT_BANNER, op_name,
                                        _DERIVED["state"][op_name][1]))
        else:
            if not cpu64 and op_name not in _DERIVED["state"]:
                _DERIVED["state"][op_name] = ("ok", _ATTN_DERIVED[op_name](
                    _schema_args(func, bound, kw))[0])
                print("operands:       SYNTHETIC (derived %s recomputed by forward: %s)"
                      % ("/".join(sorted(repl)), _DERIVED["state"][op_name][1]))
        _DERIVED["cache"][key] = repl
    names = [arg.name for arg in func._schema.arguments]
    for name, t in repl.items():
        i = names.index(name)
        if i < len(bound):
            bound[i] = t
        elif name in kw:
            kw[name] = t
    return repl


def _derived_ok(op_name):
    return _DERIVED["state"].get(op_name, ("",))[0] == "ok"


def _flag_inconsistent_at_exit():
    """A run whose attention operands could not be made consistent says so beside
    its [verdict]: replay_verdict.py and summary.txt read the flag, not the banner."""
    bad = [op for op, (st, _) in _DERIVED["state"].items() if st == "failed"]
    if bad:
        print("[verdict-flag] SYNTHETIC-INCONSISTENT %s -- a non-finite result may be the"
              " synthesis, not the kernel; this run cannot validate a fix" % ",".join(bad))
        sys.stdout.flush()


atexit.register(_flag_inconsistent_at_exit)


def _chain_synth_args(cap, n_row):
    """Names (positions) of chain row n_row's args bound to SYNTHESIZED tensors -- roots
    or pool values -- rather than computed by an earlier op of the chain.  Only those
    are re-derived: a forward that is itself in the chain already made them consistent."""
    produced = set()
    for rec in cap["chain"][:n_row]:
        produced.update(k for k in rec["produces"] if k)
    out = set()
    for i, a in enumerate(cap["chain"][n_row]["args"]):
        if isinstance(a, dict) and _CHAIN_REF in a and a[_CHAIN_REF] not in produced:
            out.add(i)
    return out


def _chain_derive(cap, n_row, rec, func, bound, kw, env, cpu64=False):
    """_derive_attn_operands for one chain row, when synthetic and the forward outputs
    were synthesized; rebinds env so later rows and the terminal reference see the
    derived tensors.  -> bound (a list when anything was replaced)."""
    if rec["op"] not in _ATTN_DERIVED or not _synthetic(cap):
        return bound
    names = [arg.name for arg in func._schema.arguments]
    synth = _chain_synth_args(cap, n_row)
    targets = [i for i, nm in enumerate(names)
               if nm in ("out", "softmax_lse", "logsumexp") and i < len(rec["args"])]
    if not targets or not all(i in synth for i in targets):
        return bound
    bound = list(bound)
    repl = _derive_attn_operands(rec["op"], func, bound, kw,
                                 ("chain", rec["call_index"], n_row), cpu64=cpu64)
    for name, t in repl.items():
        i = names.index(name)
        a = rec["args"][i] if i < len(rec["args"]) else None
        if isinstance(a, dict) and _CHAIN_REF in a:
            env[a[_CHAIN_REF]] = t
    return bound


def _reference_call(op_name, func, a, kw):
    """The fp64 CPU value of one op: its math reference if it has one, else the op
    itself on cpu64 operands.  Raises whatever the op raises."""
    if op_name.startswith("inductor.") and op_name not in _MATH_REFERENCES:
        # the kernel's own source promoted to fp64 (inductor_ref.py); returns the written arguments
        import inductor_ref
        if kw:
            raise RuntimeError("an Inductor launch with keyword operands is not modelled")
        if not set(func.desc["written"]) <= set(func.desc["outbuf"]):
            raise RuntimeError("an Inductor kernel with in_out_ptr operands: pre-call values gone")
        return inductor_ref.reference(func.desc, list(a))
    if op_name.startswith("triton.") and op_name not in _MATH_REFERENCES:
        raise RuntimeError("a Triton kernel has no CPU / fp64 implementation (no math reference "
                           "for it): read its source -- step 7")
    a64, kw64 = _cpu64(a), _cpu64(kw)
    if op_name in _MATH_REFERENCES:
        return _MATH_REFERENCES[op_name](func, a64, kw64)
    return func(*a64, **kw64)


_TERM_REF_OUT = [None]      # the last _terminal_reference's fp64 output, for the failure signature


def _terminal_record(cap):
    """The chain row of the op the capsule is named for -- the op whose fp64 reference and
    output the terminal checks judge.

    Not simply chain[-1]: a chain can end in view rows recorded after that op (an aten.t of
    another tensor follows a weight-grad mm in a backward), and a reference of a view says
    nothing about the op.  Every row carries the watched op's call index, so the op name picks
    the row: the last row of the capsule's own op; else the last row that is not a view; else
    the last row."""
    chain = cap["chain"]
    own = [r for r in chain if r.get("op") == cap.get("op")]
    if own:
        return own[-1]
    return next((r for r in reversed(chain) if not r.get("is_view")), chain[-1])


def _signature_line(cap, gpu_env):
    """failure_signature.py's SIGNATURE line for the chain's last op: its replay output against the
    terminal fp64 reference -- how the output is wrong, not only that it is (step7-rootcause.md)."""
    try:
        from failure_signature import signature
        rec = _terminal_record(cap)
        got = [gpu_env.get(k) for k in rec.get("produces") or [] if k]
        got = [t for t in got if isinstance(t, torch.Tensor) and t.is_floating_point()]
        ref = [t for t in tree_leaves(_TERM_REF_OUT[0]) if isinstance(t, torch.Tensor) and t.is_floating_point()] \
            if _TERM_REF_OUT[0] is not None else []
        for i, g in enumerate(got):
            if not torch.isfinite(g.float()).all() or (i < len(ref) and ref[i].shape == g.shape):
                r = ref[i] if i < len(ref) and ref[i].shape == g.shape else None
                return signature(g.detach().cpu(), r.cpu() if r is not None else None)["line"]
    except Exception as e:                             # noqa: BLE001 - never fail the walk for it
        return "SIGNATURE unavailable: %s: %s" % (type(e).__name__, str(e)[:80])
    return None


def _terminal_reference(cap, gpu_env):
    """fp64 reference of the capsule's own op (_terminal_record, not a trailing view) on the
    operands the GPU replay actually bound -> (status, why), or None when there is nothing
    to check.

    The question the chain reference cannot answer on a deterministic walk: a
    wrong upstream op recomputes to the same wrong value, nothing diverges, and a
    reference from the roots only says the TRUE pipeline is finite.  This one asks
    the operand-independent question instead -- given exactly these inputs, is a
    non-finite output ever correct?  Finite and within the output dtype: no, and
    the op is at fault whatever made its operands.

    Needs the pre-call operands.  The [C1] pass has already run the op, so an op
    that READS what it writes (in-place: add_, an accumulating out-param) is
    skipped -- unless a math reference models it, which reads inputs only."""
    if not cap["chain"]:
        return None
    rec = _terminal_record(cap)
    func = resolve_op(rec["op"])
    if (cap.get("verify") or {}).get("kind") == "rerun":
        return ("rerun", "the capsule's rerun check -- a run counts when the op gives different "
                         "outputs on identical copies of its operands, which needs no reference")
    if _has_value_check_op(func):
        # A fused op with a value check (math_refs.VALUE_CHECKS): its reference is an
        # fp32 recompute from the op's INPUTS, which the [C1] pass left intact -- so it
        # answers here even though the op writes its operands.
        try:
            a = _bind(rec["args"], gpu_env, cap["pool"], None)
            kw = _bind(rec["kwargs"], gpu_env, cap["pool"], None)
            produced = [gpu_env.get(k) for k in rec.get("produces") or [] if k]
            got = _owned_rel_err(rec["op"], func, a, kw, produced[0] if len(produced) == 1 else (produced or None))
        except Exception as e:                         # noqa: BLE001
            return ("unavailable", "value check: %s: %s" % (type(e).__name__, str(e)[:80]))
        if got is None:
            return ("unavailable", "value check: variant not modelled, or its reference is "
                                   "non-finite")
        return ("finite", "%s -- reference finite; this run's rel err %.3g, correct calls stay under %.3g"
                % ("twin check (twins.py: the op's second implementation on its inputs)" if _twin_owned(func)
                   else "value check (math_refs: fp32 from the op's inputs)", got[0], got[1]))
    w = op_writes(func)
    # An Inductor kernel's out_ptr* are written and never read (outbuf): its fp64 twin reads inputs only
    pure_out = rec["op"].startswith("inductor.") and w["idx"] <= w.get("outbuf_idx", frozenset())
    if w["idx"] and rec["op"] not in _MATH_REFERENCES and not pure_out:
        return ("unavailable", "ref: writes its operands (in-place); pre-call values gone")
    try:
        a = _bind(rec["args"], gpu_env, cap["pool"], None)
        kw = _bind(rec["kwargs"], gpu_env, cap["pool"], None)
        out = _reference_call(rec["op"], func, a, kw)
    except Exception as e:                             # noqa: BLE001
        return ("unavailable", "ref: %s: %s"
                % (type(e).__name__, str(e).splitlines()[0][:80] if str(e) else ""))
    _TERM_REF_OUT[0] = out
    live = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor)
            and t.is_floating_point()]
    if not all(torch.isfinite(t).all().item() for t in live):
        return ("NON-FINITE", "true result non-finite in fp64")
    rmax = max((t.abs().max().item() for t in live if t.numel()), default=0.0)
    omax = torch.finfo(cap["out_dtype"]).max if cap.get("out_dtype") is not None \
        and cap["out_dtype"].is_floating_point else float("inf")
    if rmax > omax:
        return ("NON-FINITE", "true max|.|=%.3g exceeds %s max" % (rmax, cap["out_dtype"]))
    return ("finite", "max|.|=%.3g fits %s" % (rmax, cap.get("out_dtype")))


def _chain_reference(cap, dev):
    """fp64 CPU status per chain row, aligned with cap["chain"] (one entry per row).

    Each entry is (status, why): status is 'finite', 'NON-FINITE' or 'unavailable'
    -- one token, the walk prints it in a column -- and why says what made it
    unavailable.  Indexed by ROW, not by call_index: every op in a chain carries
    the call index of the capture's watched op, so a dict keyed by it collapses
    the whole chain onto one entry and prints that entry on every row.

    A missing reference is never a pass: the caller must not read `unavailable`
    as finite, nor as non-finite.  Once one op is unavailable, everything that
    consumes its output is too ('upstream').
    """
    env, outcome = {}, []
    for k in cap["roots"]:
        env[k] = _cpu64(unpack(cap["pool"][k], "cpu"))
    for n, rec in enumerate(cap["chain"]):
        if n < len(cap["chain"]) - 1 and (is_collective_op(rec["op"]) or try_resolve_op(rec["op"]) is None):
            _recorded_collective(cap, rec, env, "cpu")
            for k in [k for k in rec["produces"] if k]:
                env[k] = _cpu64(env[k])
            outcome.append(("-", "recorded (cannot run here)"))
            continue
        func = resolve_op(rec["op"])
        if n == len(cap["chain"]) - 1 and (_has_value_check_op(func)
                                           or (cap.get("verify") or {}).get("kind") == "rerun"):
            # no fp64 kernel (an aiter asm op): its check is the value check or the capsule's
            # rerun, on the GPU operands -- the terminal line below, not an AssertionError here
            outcome.append(("-", "ref: the %s (terminal line)" % (
                "rerun check" if (cap.get("verify") or {}).get("kind") == "rerun" else "value check")))
            return outcome
        try:
            a = _bind(rec["args"], env, cap["pool"], "cpu")
            kw = _bind(rec["kwargs"], env, cap["pool"], "cpu")
            kw = _cpu64(kw)
            a = _chain_derive(cap, n, rec, func, _cpu64(a), kw, env, cpu64=True)
            if rec["op"].startswith("inductor.") and n < len(cap["chain"]) - 1:
                # its fp64 twin returns the written arguments, not values `produces` maps
                raise RuntimeError("an upstream Inductor kernel: no chain reference")
            out = _reference_call(rec["op"], func, a, kw)
        except Exception as e:                         # noqa: BLE001
            outcome.append(("unavailable", "ref: %s: %s"
                            % (type(e).__name__, str(e).splitlines()[0][:80] if str(e) else "")))
            outcome += [("unavailable", "ref: upstream")] * (len(cap["chain"]) - n - 1)
            return outcome
        live = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor)]
        if rec["op"] in _ALLOC_OPS:                    # uninitialised: nothing to score
            outcome.append(("-", ""))
            for key, t in zip(rec["produces"], live):
                if key:
                    env[key] = t
            continue
        ok = all(torch.isfinite(t).all().item() for t in live if t.is_floating_point())
        outcome.append(("finite" if ok else "NON-FINITE",
                        "ref: math" if rec["op"] in _MATH_REFERENCES else ""))
        if rec["op"] not in _MATH_REFERENCES:
            for key, t in zip(rec["produces"], live):
                if key:
                    env[key] = t
    return outcome


def replay_chain(cap, dev):
    """-> (rc, tally, culprit).

    tally is [C2]'s first-to-go-non-finite count, {} if it never ran (see
    chain_loop).  culprit is [C1]'s first-op-to-go-wrong, None when the walk did
    not run (--chain-loop-only) or found nothing.  A mutation sweep orders from
    both, culprit first -- see chain_sweep_order.  Returning the culprit rather
    than only printing it is what makes 5b-chain's conclusion reach 5c: it used
    to be a local that died at the end of this function, so the sweep was ordered
    by the tally alone even when the walk had named a different op."""
    # --chain-loop-only skips straight to [C2].  That is what --contend hands its
    # children: a contender that spends its life doing [C1]'s fp64 comparisons is
    # sitting on the host rather than loading the device, which is pure waste in
    # a process whose only job is to be load.  See _contender_cmd().
    if args.chain_loop_only:
        # chain_loop stays a 2-tuple: it is its own entry point and has no culprit
        # to report -- [C1] is precisely what this flag skips.
        rc, tally = chain_loop(cap, dev)
        return rc, tally, None

    n = len(cap["chain"])
    cfg = cap.get("config") or {}
    print("\n[C1] chain walk -- %d ops, %d roots, %d pooled tensors (depth=%s)"
          % (n, len(cap["roots"]), len(cap["pool"]), cfg.get("depth")))

    trunc = cap.get("truncation") or {}
    if trunc:
        seen = {}
        for r in trunc.values():
            seen[r] = seen.get(r, 0) + 1
        print("  chain stopped at: %s"
              % ", ".join("%s x%d" % (r, c) for r, c in sorted(seen.items())))
        # outside_window is the ORDINARY ending: a model parameter, or a tensor
        # made before the capture armed, has no producer to find and simply
        # becomes a root.  Only a value we had and lost costs the walk anything,
        # so only that is worth telling the user to spend memory on.
        lost = [r for r in ("values_evicted", "too_big") if r in seen]
        if lost:
            print("  -> %s: the walk had those values and dropped them. Raise"
                  " chain_value_bytes (or max_bytes) to keep them."
                  % ", ".join(lost))

    try:
        # [C1] executes the chain exactly once and is not a timing measurement, so
        # it is the free place to collect each op's problem key for a later sweep.
        rows, gpu_env = _run_chain_once(cap, dev, record_geom=True)
    except KeyError as e:
        print("  chain is not replayable: %s" % e)
        print("[verdict] CHAIN_INCOMPLETE")
        return 1, {}, None

    ref = [] if args.no_reference else _chain_reference(cap, dev)
    term_ref = None if args.no_reference else _terminal_reference(cap, gpu_env)
    wrong_rows = [r for r in rows if r.get("wrong") is not None]
    if wrong_rows and not _CHECK_WRONG[0]:
        _CHECK_WRONG[0] = True
        print("  [C1] %s (call #%d) returned a WRONG FINITE value (%.3g vs higher precision) on "
              "these operands: deterministic -- the [C2] loop counts runs that repeat it"
              % (wrong_rows[0]["op"], wrong_rows[0]["call_index"], wrong_rows[0]["wrong"]))

    print("\n  %-4s %-52s %-7s %-10s %-12s %s"
          % ("#", "op", "finite", "vs record", "fp64 ref", "note"))
    for n, r in enumerate(rows):
        div = ("-" if r["diverged"] is None
               else ("DIVERGED" if r["diverged"] else "match"))
        ref_status, ref_why = (ref[n] if n < len(ref)
                               else ("skipped" if args.no_reference else "-", ""))
        note = []
        if r["is_view"]:
            note.append("view")
        if r["needs_rng"]:
            note.append("rng")
        if r["version_ok"] is False:
            note.append("MUTATED AFTER PRODUCTION")
        if r["maxerr"]:
            note.append("max|delta|=%.3g" % r["maxerr"])
        if r.get("slots"):
            # Which return slot, not just "this op" -- for a fused kernel the
            # difference between out[0] and the logsumexp is the whole finding.
            note.append("out[%s]" % ",".join(str(s) for s in r["slots"]))
        if r.get("uninitialised"):
            note.append("allocator: uninitialised, not compared")
        if r.get("wrong") is not None:
            note.append("WRONG VALUE: %.3g vs higher precision" % r["wrong"])
        if ref_why:
            note.append(ref_why)
        print("  %-4d %-52s %-7s %-10s %-12s %s"
              % (r["call_index"], r["op"], "yes" if r["finite"] else "NO", div,
                 ref_status, ", ".join(note)))

    if term_ref is not None:
        # Parsed by replay_verdict.py (TERMINAL): keep the leading words stable.
        print("\n  terminal fp64 ref on replay operands: %s  %s" % term_ref)
        if rows and (not rows[-1]["finite"] or rows[-1].get("wrong") is not None):
            sl = _signature_line(cap, gpu_env)
            if sl:
                print("  " + sl)
        if term_ref[0] == "finite" and rows and not rows[-1]["finite"]:
            print("  -> on the operands this replay handed it, the true result is finite"
                  " and fits the")
            print("     output dtype, so the %s output is the op's own --"
                  " whatever produced the operands."
                  % ("wrong finite" if rows[-1].get("wrong") is not None else "non-finite"))

    # The first op that goes wrong is the finding.  Divergence outranks
    # non-finiteness: an op that returned a finite but wrong value corrupted
    # everything downstream, and the op that finally produced the NaN is then a
    # bystander -- which is exactly the misreading this mode exists to prevent.
    culprit = next((r for r in rows if r["diverged"]), None)
    kind = "DIVERGED"
    if culprit is None:
        culprit = next((r for r in rows if not r["finite"]), None)
        # a wrong-value capsule's culprit is a wrong FINITE output, not a NaN (2026-11-21)
        kind = "WRONG_VALUE" if culprit is not None and culprit.get("wrong") is not None else "NON_FINITE"

    if culprit is None and (args.chain_iters or 1) > 1:
        # One walk run of an intermittent fault is clean most of the time; the [C2] loop
        # below is the measurement.  A "[verdict] CHAIN_NOT_REPRODUCED" here, inside an
        # attempt that then REPRODUCED, read as a contradiction (2026-11-20).
        print("\n  -> this single walk run was clean -- expected for an intermittent fault;"
              " the [C2] loop below counts the runs.")
        rc = 1
    elif culprit is None:
        print("\n  -> the whole chain replayed clean. The fault needs surrounding"
              " state (concurrency, allocator, a neighbouring kernel) that the"
              " chain does not carry.")
        print("[verdict] CHAIN_NOT_REPRODUCED")
        rc = 1
    else:
        depth = rows.index(culprit)
        print("\n  -> first op to go wrong: %s (call #%d, %s)"
              % (culprit["op"], culprit["call_index"], kind))
        if depth < len(rows) - 1:
            print("     It is %d op(s) UPSTREAM of the op the capsule is named for,"
                  % (len(rows) - 1 - depth))
            print("     so the capsule's operands are not what a clean run produces.")
            if kind == "DIVERGED":
                # Hard-won: DIVERGED says the recorded value is not reproducible,
                # which does NOT establish that this op failed FIRST in the run
                # that was captured.  On the 2nd NaN both the forward and the
                # backward turned out to be corrupted, and reading this line as
                # "the named op is innocent" was wrong.  [C2]'s per-run tally is
                # the localiser; this is a flag, not a verdict.
                print("     NB DIVERGED means not-reproducible, NOT that this op"
                      " failed first. Use --chain-iters (under the contention the")
                print("     fault needs) to see which op actually goes non-finite first.")
        print("[verdict] CHAIN_CULPRIT %s @ depth %d" % (culprit["op"], depth))
        rc = 0

    # Carried out of here, not just printed: chain_sweep_order leads the walk with
    # it.  kind matters downstream -- a NON_FINITE culprit on a chain with no
    # DIVERGED row anywhere is the deterministic case, where this attribution is
    # unsound and 5c should not be narrowed to it.
    culprit_info = None if culprit is None else {
        "op": culprit["op"], "call_index": culprit["call_index"], "kind": kind,
        "any_diverged": any(r["diverged"] for r in rows),
    }
    tally = {}
    if args.chain_iters:
        _, tally = chain_loop(cap, dev)
    return rc, tally, culprit_info


def _bits_equal(a, b):
    """Bit-for-bit, NaN included (torch.equal calls NaN != NaN, which would mark every
    output holding a NaN as changed)."""
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def _report_repeatability(nonrep, bad_nonrep, ran, bad):
    """--chain-repeat-check: which op's output stopped repeating, and whether the
    non-finite runs followed it.

    The finiteness tally names the op where a NaN first APPEARS.  A racy kernel that
    writes finite but wrong values hands them downstream, and the op that finally
    overflows is blamed for it.  Same inputs every run, so a deterministic kernel's
    output is bit-identical from run to run.  In each non-finite run, the first op in
    chain order whose output CHANGED from the run before is where the fault entered
    (vs the run before, not vs run 1, so a bad run 1 cannot mislead it).  An op that
    changes on most runs is non-deterministic by design (atomic accumulation) and
    cannot localise anything: listed, skipped.
    """
    runs = ran - 1                       # run 1 is the reference
    print("  repeatability (bit-for-bit, %d run(s) compared):" % runs)
    if not nonrep:
        print("    every op repeated exactly in every run")
    inherent = {k for k, (_r1, ch) in nonrep.items() if ch > 0.5 * runs}
    for (ci, op), (r1, ch) in sorted(nonrep.items()):
        note = ""
        if (ci, op) in inherent:
            note = "  -- most runs: non-deterministic by design, not localising"
        elif runs >= 10 and r1 >= runs - 1:
            note = "  -- differs from run 1 almost always but rarely run to run: run 1 was a bad run"
        print("    %-56s vs run 1: %d  run to run: %d  (%.2f%%)%s"
              % ("%s (call #%d)" % (op, ci), r1, ch, 100.0 * ch / runs, note))
    if not bad:
        return
    lead = {}
    for diffs in bad_nonrep:
        k = next((d for d in diffs if d not in inherent), None)
        lead[k] = lead.get(k, 0) + 1
    print("  first op to CHANGE in each non-finite run:")
    for k, n in sorted(lead.items(), key=lambda kv: -kv[1]):
        name = ("(none -- every op repeated, or only inherent ones changed)" if k is None
                else "%s (call #%d)" % (k[1], k[0]))
        print("    %-56s %d of %d" % (name, n, bad))
    top = max(lead.items(), key=lambda kv: kv[1])[0]
    if top is not None:
        print("  -> the non-finite runs follow a non-repeatable %s: a fault that writes"
              % top[1])
        print("     finite but wrong values there, upstream of the op the finiteness"
              " tally names, if it is earlier in the chain.")


def _save_or_compare_outputs(cap, ref):
    """--chain-save-outputs / --chain-compare-outputs on the loop's first-run outputs.
    Run 1 is taken as the reference, so save from a QUIET (solo) stock run -- under
    load run 1 can itself be a bad run."""
    ops = {r["call_index"]: r["op"] for r in cap["chain"]}
    if args.chain_save_outputs:
        torch.save({ci: [t.cpu() for t in ts] for ci, ts in ref.items()},
                   args.chain_save_outputs)
        print("  outputs of run 1 saved: %s (%d op(s))" % (args.chain_save_outputs, len(ref)))
    if args.chain_compare_outputs:
        theirs = torch.load(args.chain_compare_outputs, map_location="cpu")
        same_all = True
        print("  outputs of run 1 vs %s:" % args.chain_compare_outputs)
        for ci in sorted(ref):
            if ci not in theirs:
                print("    %-56s not in the saved file" % ("%s (call #%d)" % (ops.get(ci), ci)))
                same_all = False
                continue
            same = (len(ref[ci]) == len(theirs[ci])
                    and all(_bits_equal(a.cpu(), b)
                            for a, b in zip(ref[ci], theirs[ci])))
            same_all &= same
            print("    %-56s %s" % ("%s (call #%d)" % (ops.get(ci), ci),
                                    "bit-identical" if same else "DIFFERS"))
        print("[values] %s" % ("IDENTICAL" if same_all else "DIFFER"))


def _loop_checkpoint(ran, bad):
    """--progress-file / --stop-file, every --progress-every runs.  True = stop now.

    A node launcher sums every copy's progress and, once the node has enough
    non-finite runs to quote a rate, creates the stop file for all copies at once
    (stopping one copy early would de-stress its co-tenant).  A copy with no hits
    yet stops too: the attempt's verdict and rate are node-wide.
    """
    # Time is checked every run, not every --progress-every: a loop starved by its
    # co-tenants (7 HBM-copy streams) never reached run 2000 and ran into the job's
    # walltime with no count printed at all.
    now = time.time()
    if args.seconds and now - _LOOP_T0[0] >= args.seconds:
        _LOOP_T0[1] = True
        return True
    if now - _LOOP_T0[2] >= 60:
        _LOOP_T0[2] = now
        print("  ... %d run(s), %d non-finite, %.0fs" % (ran, bad, now - _LOOP_T0[0]),
              flush=True)
    if ran % max(args.progress_every, 1):
        return False
    if args.progress_file:
        try:
            tmp = args.progress_file + ".tmp"
            with open(tmp, "w") as f:
                f.write("%d %d\n" % (ran, bad))
            os.replace(tmp, args.progress_file)
        except OSError as e:                 # never let bookkeeping kill a replay
            print("  !! --progress-file unwritable (%s); early stop disabled" % e)
            args.progress_file = None
    return bool(args.stop_file) and os.path.exists(args.stop_file)


def _stopped_note(ran, iters):
    if ran == iters:
        return ""
    if _LOOP_T0[1]:
        return " (stopped at --seconds %g; %d not run)" % (args.seconds, iters - ran)
    if args.stop_file and os.path.exists(args.stop_file):
        return " (stopped by the launcher, see its STOP file; %d not run)" % (iters - ran)
    return " (stopped at the first; %d not run)" % (iters - ran)


# [loop start time, stopped by --seconds]
_LOOP_T0 = [0.0, False, 0.0]   # loop start, stopped by --seconds, last progress line


def _start_loop():
    """Wait at --start-barrier, then start the --seconds clock.  Called right before
    the [C2] / [3] loop, after co-tenants are loading the device."""
    if args.start_barrier and args.barrier_n > 1:
        try:
            os.makedirs(args.start_barrier, exist_ok=True)
            # host.pid is not unique: each copy in its own container or PID namespace
            # (docker --init, apptainer --pid) gets the same small pid, the copies write
            # ONE file and the barrier waits out its timeout with less co-tenancy.
            open(os.path.join(args.start_barrier, "%s.%d.%s.ready"
                              % (socket.gethostname(), os.getpid(), os.urandom(6).hex())),
                 "w").close()
        except OSError as e:
            print("  !! --start-barrier unusable (%s); not waiting" % e)
        else:
            t0 = time.time()
            while True:
                n = len([f for f in os.listdir(args.start_barrier) if f.endswith(".ready")])
                if n >= args.barrier_n:
                    print("  barrier: %d/%d copies ready after %.0fs"
                          % (n, args.barrier_n, time.time() - t0))
                    break
                if time.time() - t0 > args.barrier_timeout:
                    print("  !! barrier: only %d/%d copies ready after %.0fs -- running"
                          " anyway, with less co-tenancy than asked"
                          % (n, args.barrier_n, args.barrier_timeout))
                    break
                time.sleep(0.5)
    _LOOP_T0[0], _LOOP_T0[1], _LOOP_T0[2] = time.time(), False, time.time()


def chain_loop(cap, dev):
    """[C2] -- run the whole chain repeatedly, under co-tenants if asked.

    Returns (rc, tally), where tally maps (call_index, op) -> how many runs that op
    was the FIRST to go non-finite in.  The tally is what orders a chain mutation
    sweep: the op that fails first is the one worth mutating first, and the ops
    upstream of it are swept after, never before -- mutating an upstream op changes
    the operands every downstream op sees.
    """
    iters = args.chain_iters or 1
    print("\n[C2] chain loop x%d%s%s"
          % (iters, "  (+%d co-tenants)" % args.contend if args.contend else "",
             "  (stopping at the first non-finite run)" if args.stop_on_nan else ""))
    if not start_contenders(args.contend):
        # Reporting a number from an arm whose load never arrived is how a
        # non-experiment gets quoted as a clean result.
        print("  !! contention was requested and did not materialise -- this arm")
        print("     measured the SOLO case, which for a contention-gated fault is")
        print("     the case that never fails. Not reporting a rate.")
        stop_contenders()
        print("[verdict] CONTENTION_INVALID")
        return 1, {}
    _start_loop()
    ran = 0
    try:
        bad = 0
        # Tally WHICH op went non-finite first, not just whether any did.  Under
        # contention this is the localisation: if the forward's output is already
        # bad before the backward runs, the backward is downstream of the fault
        # rather than the site of it -- and that is the whole question a chain
        # exists to answer.
        first = {}
        repeat = {} if (args.chain_repeat_check or args.chain_save_outputs
                        or args.chain_compare_outputs) else None
        nonrep = {}         # (call_index, op) -> [runs differing from run 1, from the run before]
        bad_nonrep = []     # per non-finite run: the ops that changed, in chain order
        noise = _CollectiveNoise(dev) if args.collective_noise else None
        for _ in range(iters):
            work = noise.start() if noise else None
            rows_i, _ = _run_chain_once(cap, dev, compare=False, repeat=repeat)
            if work is not None:
                work.wait()
            ran += 1
            for r in rows_i:
                if r.get("nonrepeatable") or r.get("changed"):
                    c = nonrep.setdefault((r["call_index"], r["op"]), [0, 0])
                    c[0] += bool(r.get("nonrepeatable"))
                    c[1] += bool(r.get("changed"))
            culprit = next((r for r in rows_i if not r["finite"]), None)
            if culprit is not None:
                bad += 1
                k = (culprit["call_index"], culprit["op"])
                first[k] = first.get(k, 0) + 1
                if repeat is not None:
                    bad_nonrep.append([(r["call_index"], r["op"])
                                       for r in rows_i if r.get("changed")])
                # As a mutation trial, the verdict is settled by the first
                # non-finite run; the rest is GPU time that cannot change it.
                if args.stop_on_nan:
                    break
            if _loop_checkpoint(ran, bad):
                break
    finally:
        # Held, not dropped, when this chain loop is a sweep's baseline: the trials
        # that follow have to meet the same device this number was measured on.
        if not sweep_holds_load():
            stop_contenders()
    load = effective_load()
    # The leading "non-finite on n/N" is parsed by the launchers' summaries: keep it.
    print("  non-finite on %d/%d chain runs%s%s%s"
          % (bad, ran, "" if not load else "  (under %s)" % load,
             _stopped_note(ran, iters),
             "  [wrong-value capsule: a run counts when the terminal op fails its %s "
             "check again]" % cap["verify"].get("kind") if cap.get("verify") else ""))
    if ran:
        _BASELINE["rate"] = bad / float(ran)
    if bad:
        print("  first op to go non-finite, by run:")
        for (ci, op), n in sorted(first.items(), key=lambda kv: -kv[1]):
            print("    %-56s %d  (%.2f%% of runs)"
                  % ("%s (call #%d)" % (op, ci), n, 100.0 * n / ran))
        print("  -> the chain reproduces without the model. This, not the"
              " single-op capsule, is the reproducer to attach to a bug report.")
    if repeat is not None and ran > 1 and args.chain_repeat_check:
        _report_repeatability(nonrep, bad_nonrep, ran, bad)
    if repeat is not None and repeat.get("ref"):
        _save_or_compare_outputs(cap, repeat["ref"])
    if not load and not bad:        # only a CLEAN solo loop is in doubt (2026-11-21: printed beside 5/100)
        ci, _ = capture_gpu_procs(cap)
        print("  -> clean, but this ran ALONE. The capture saw %s process(es) on"
              % (ci or "?"))
        print("     the device; a contention-gated fault cannot appear at 1. Re-run")
        print("     with --contend-auto before concluding anything from this 0/%d."
              % iters)
    # A [verdict] line so this path can be a mutation trial at all: run_trial scores
    # a child from its verdict, and a chain loop that printed none scored ERROR on
    # every row regardless of what it measured.
    print("[verdict] %s" % ("CHAIN_REPRODUCED" if bad else "CHAIN_NOT_REPRODUCED"))
    return (0 if bad else 1), first


def _list_after_verdict(g, kernel, rc):
    """--list-mutations still lists after a verdict that ends the replay early.

    Enumeration reads the kernel, not the verdict (see the NOT_REPRODUCED path), so
    an early CORRECT_OVERFLOW / TRUE_RESULT_NONFINITE must not swallow a list the
    user asked for by name.
    """
    if args.list_mutations:
        try:
            run_mutations(g, kernel=kernel)
        finally:
            stop_contenders()
    return rc


def main():
    if args.list_mutations and args.patch_kernel:
        # Nothing to discover and nothing to replay, so skip torch entirely: this is
        # the path that runs on a login node with ROCm but no GPU.  --disasm is the
        # same kind of work -- unbundle and objdump, no capsule and no device -- so it
        # comes along rather than being silently dropped on the one path that can run
        # where there is no GPU to dispatch on.
        if args.disasm:
            write_disasm(None, args.disasm, kernel=args.patch_kernel)
        return run_mutations(None, kernel=args.patch_kernel)

    _import_torch()
    cap = upgrade_capsule(torch.load(args.capsule, map_location="cpu", weights_only=False))
    _TWIN_ON[0] = (cap.get("verify") or {}).get("kind") == "twin"

    if args.capture_procs:
        n, detail = capture_gpu_procs(cap)
        print("CAPTURE_PROCS=%s" % (n or "?"))
        print("CAPTURE_PROCS_DETAIL=%s" % detail)
        # Ops (that passed the capture's filter) the capturing process made up to and including the failing
        # one: the model hit it within this many, which bounds how many replay
        # runs a working setup should need (the launchers' futility stop).
        print("CAPTURE_OP_INDEX=%s" % (cap.get("op_call_index") or "?"))
        return 0

    if args.restore_env and not os.environ.get(_REEXEC):
        env = restore_env_dict(cap)
        # PyTorch TunableOp picks the GEMM solution from its results file, which the capture
        # recorded by its path in the model's run dir; a package ships it in tunableop/ beside
        # capsule/.
        tf = env.get("PYTORCH_TUNABLEOP_FILENAME")
        if tf and args.capsule:
            d = os.path.dirname(os.path.abspath(args.capsule))
            for td in (os.path.join(d, "tunableop"), os.path.join(os.path.dirname(d), "tunableop")):
                if os.path.isdir(td):
                    env["PYTORCH_TUNABLEOP_FILENAME"] = os.path.join(td, os.path.basename(tf))
                    print("env:            PYTORCH_TUNABLEOP_FILENAME -> %s (shipped results, was %s)"
                          % (env["PYTORCH_TUNABLEOP_FILENAME"], tf), flush=True)
                    break
        env[_REEXEC] = "1"
        argv = [a for a in sys.argv if a != "--restore-env"]
        os.execve(sys.executable, [sys.executable] + argv, env)

    g = cap.get("geometry") or packed_geometry(cap)
    if args.bench:
        if not g and cap["op"] == "aten._grouped_mm.default":
            lines = grouped_bench_cmds(cap)
            if lines:
                print("\n".join(lines))
                return 0
        if not g:
            sys.exit("capsule has no 2-D GEMM geometry (op=%s)" % cap["op"])
        xf32 = cap["precision"]["float32_matmul_precision"] != "highest"
        print(bench_cmd(g, compute="xf32_r" if xf32 else "f32_r",
                        solution=args.solution_index))
        return 0

    # Below --capture-procs and --bench on purpose: both need only metadata.
    if cap.get("data", "values") != "values" and not args.synthesize:
        sys.exit("%s: recorded without tensor data (shapes only, the capture "
                 "default).  Replay it with --synthesize (made-up values from the "
                 "recorded statistics), or re-capture with NAN_CAPTURE_DATA=1 "
                 "(OpNanCapture(record_data=True)); --bench works as is."
                 % args.capsule)

    print("capsule:        %s  (format %d)" % (args.capsule, cap["format"]))
    _CAP_OP[0] = cap["op"]
    _CAP[0] = cap
    print("op:             %s   (op call #%d in the run)"
          % (cap["op"], cap["op_call_index"]))
    # device_name lives in ["dist"] from format 2 on, in ["versions"] before that.
    d = cap.get("dist", {})
    print("captured on:    torch %s / hip %s / %s"
          % (cap["versions"]["torch"], cap["versions"]["hip"],
             d.get("device_name") or cap["versions"].get("device_name", "?")))
    if d:
        print("original rank:  %s of %s on %s device %s"
              % (d.get("rank"), d.get("world_size"), d.get("hostname"), d.get("device")))
    print("original out:   %s  nan=%d inf=%d"
          % (cap["out_size"], cap["out_nan"], cap["out_inf"]))
    build_banner(cap)
    env_banner(cap)
    contention_banner(cap)
    operands_banner(cap)
    if args.contend_auto:
        # -1: this process is one of them.
        n, detail = capture_gpu_procs(cap)
        if not n:
            # Do not point at the 'captured under' line here: a capsule with no
            # process count is usually one captured before contention recording
            # existed, and it has no such line to read.
            sys.exit("--contend-auto: %s. Pass --contend N yourself, from the"
                     " model's own fan-out flag (--workers/--nproc-per-node/"
                     "num_workers) or from the launcher's trials-per-GPU."
                     % detail)
        args.contend = max(0, int(n) - 1)
        print("  --contend-auto -> %d co-tenant(s)  [%s]" % (args.contend, detail))
        if args.contend == 0:
            # Refuse rather than replay solo under a flag whose whole purpose is
            # not to.  This resolved silently to 0 for every capsule taken under
            # a launcher that puts N containers on a GPU, and a 0/N from such a
            # run is indistinguishable from a real negative.
            ci = cap.get("contention") or {}
            mem = ci.get("mem_used_by_others_bytes") or 0
            sys.exit(
                "--contend-auto resolved to 0 co-tenants, which is a SOLO replay"
                " -- refusing.\n"
                "  The capture looked like one process on the device%s.\n"
                "  If that is right, drop --contend-auto and say the run was solo.\n"
                "  If it is not, the co-tenant was in another container and /proc"
                " could not see it:\n"
                "    - this run: pass --contend N (N = trials-per-GPU minus 1) or"
                " --contend-with <model cmd>;\n"
                "    - next capture: set NAN_CAPTURE_CONTAINERS_PER_GPU=<copies per"
                " GPU> in the launcher.\n"
                "  A sibling replay is also a weaker co-tenant than a real training"
                " process -- if\n"
                "  the model is available, --contend-with it rather than"
                " count-matching with siblings."
                % ("" if mem <= 4e9
                   else ", but %.1f GB was held by processes it never counted"
                        " -- so it probably was not" % (mem / 1e9)))
    if g:
        print("hipBLASLt key:  M=%d N=%d K=%d transA=%s transB=%s lda=%d ldb=%d ldd=%d"
              % (g["M"], g["N"], g["K"], g["transA"], g["transB"], g["lda"], g["ldb"], g["ldd"]))

    # ---- 0. the dispatched kernel's ISA -------------------------------------
    # Ahead of the replay rather than after it: steps [1] and [2] return early on
    # PROPAGATED / TRUE_RESULT_NONFINITE / CORRECT_OVERFLOW, and a file the user asked
    # for by name should not quietly depend on which verdict this capsule reaches.
    # Charged to parent_only, not to the trial budget: discovery unbundles a 43 MB ELF
    # and replays once more, and no trial child does either.
    _t = time.time()
    disasm_kernel = write_disasm(g, args.disasm) if args.disasm else None
    _BASELINE["parent_only"] += time.time() - _t

    # ---- restore the fp32 compute type: it is part of the problem key --------
    prec = cap["precision"]
    torch.backends.cuda.matmul.allow_tf32 = prec["allow_tf32"]
    torch.set_float32_matmul_precision(prec["float32_matmul_precision"])
    # Recorded since format 2 and, until now, never applied -- so an fp16/bf16
    # capsule replayed under whatever the replay process defaulted to, which
    # changes the accumulate width of the reduction and therefore the result.
    # getattr-guarded: the attribute is absent on older torch, and a capsule
    # from before it was recorded has no value to apply.
    fp16r = prec.get("allow_fp16_reduced")
    if fp16r is not None and hasattr(torch.backends.cuda.matmul,
                                     "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = fp16r
    print("precision:      allow_tf32=%s float32_matmul_precision=%s"
          " allow_fp16_reduced=%s"
          % (prec["allow_tf32"], prec["float32_matmul_precision"],
             "?" if fp16r is None else fp16r))

    # Under torchrun (--collective-noise) each rank replays on its own GPU.
    dev = (torch.device("cuda", int(os.environ["LOCAL_RANK"]))
           if args.collective_noise and "LOCAL_RANK" in os.environ else torch.device("cuda"))

    if args.chain and not cap.get("chain") and args.chain_fallback:
        print("chain:          none in this capsule (format %s) -- --chain-fallback:"
              " replaying the failing op alone, --iters %d"
              % (cap.get("format"), args.chain_iters or args.iters))
        args.chain = False
        args.iters = args.chain_iters or args.iters
    if args.chain:
        if not cap.get("chain"):
            sys.exit("capsule has no chain (format %s). Re-capture with "
                     "OpNanCapture(..., chain_depth=N)." % cap.get("format"))
        rc, tally, culprit = replay_chain(cap, dev)
        # Falls THROUGH to the sweep rather than returning, which it used to do
        # unconditionally -- `--chain --mutate` then reached neither run_mutations()
        # call site and dropped the sweep with no message at all.
        if not _MUTATING:
            return rc
        if _MUTATING and sweep_holds_load():
            # Same note the single-op path prints: the trials below meet the same
            # device this baseline was measured on, which is both what makes the
            # table an A/B and what keeps the auto timeout honest.
            print("\n  sweep:  co-tenants HELD for every trial, so the baseline above"
                  " and the trials")
            print("          below are measured on the same device."
                  " (--mutate-solo drops them.)")
        # rc is the [C1] walk's, which is always clean for a contention-gated fault;
        # a non-empty tally means the [C2] loop -- the baseline the trials are
        # compared against -- did reproduce.
        if rc != 0 and not tally and not args.mutate_force:
            print("\n  -> skipping the mutation sweep: the chain baseline produced no")
            print("     NaN, so every trial would come back CLEAN and none of it would")
            print("     mean anything. If the chain ran SOLO that is the expected")
            print("     result -- re-run with --contend-auto. Pass --mutate-force to")
            print("     sweep regardless.")
            return rc
        try:
            return run_chain_mutations(cap, tally, dev, culprit)
        finally:
            stop_contenders()

    func = resolve_op(cap["op"])
    a = unpack(cap["args"], dev)
    kw = unpack(cap["kwargs"], dev)
    if _synthetic(cap) and cap["op"] in _ATTN_DERIVED:
        a = list(a)
        _derive_attn_operands(cap["op"], func, a, kw, ("op",))
        a = tuple(a)
    if args.written_operands != "captured":
        a = list(a)
        _apply_written_operands(func, a)
        a = tuple(a)
        print("written operands: %s (--written-operands)" % args.written_operands)

    # ---- 1. inputs ----------------------------------------------------------
    # A WRITTEN argument is not an input, even when it is passed positionally.
    # flash-attn's backward takes dq/dk/dv that way, and before the call they
    # hold whatever empty_like() handed back -- routinely NaN.  Scoring those
    # PROPAGATED sends you hunting a producer that does not exist and ends the
    # ladder a rung early, which is exactly the capture-side bug fixed in
    # ad1af57; the same rule has to hold here, and it is recomputed from the
    # schema rather than read off cap['inputs_finite'] so that capsules written
    # BEFORE that fix are re-scored correctly too.
    print("\n[1] inputs")
    written = op_writes(func)["idx"]
    inputs_ok, amaxes, dirty_written = True, [], []
    for i, t in enumerate(a):
        if isinstance(t, torch.Tensor):
            ok, amax = describe(t, "arg%d%s" % (i, "*" if i in written else ""))
            if i in written:
                if not ok:
                    dirty_written.append("arg%d" % i)
            else:
                inputs_ok &= ok
            amaxes.append(amax)
    if written:
        print("  (* written by this op: an out-buffer, not an input)")
    if not inputs_ok:
        print("\n  -> inputs are already non-finite: this op PROPAGATED the NaN.")
        print("     Re-run the capture with all_ops=True, or walk the recorded stack:")
        print(cap["stack"])
        print("[verdict] PROPAGATED")
        return 1
    # Deliberately NOT "this op CREATED the non-finite output", which is what
    # this said and is an overclaim by one word: finite operands establish that no
    # NaN ARRIVED, not that the operands were RIGHT.  A kernel faithfully computing
    # NaN from a finite-but-wrong operand prints this same line, and on an op with
    # no CPU reference ([2] below) nothing downstream ever corrects it -- which is
    # how a single-op capsule gets packaged in 5c as a kernel bug it does not
    # demonstrate.  The chain walk (--chain) is what separates the two.
    print("  -> all inputs finite: no NaN arrived on the operands, so this op"
          " produced one.")
    print("     Whether it CREATED the NaN or faithfully computed it from"
          " finite-but-wrong")
    print("     operands is [2]'s question, and --chain's if [2] cannot run.")
    if dirty_written:
        print("     (%s non-finite going in, but written by this op -- allocator"
              % ", ".join(dirty_written))
        print("      garbage in a buffer the kernel overwrites, not an inherited NaN.)")

    # ---- 2. is it just fp32 overflow? --------------------------------------
    print("\n[2] fp64 %sreference" % ("GPU (the kernel's source promoted to fp64, inductor_ref.py) "
                                      if cap["op"].startswith("inductor.") else "CPU "))
    if g and len(amaxes) >= 2:
        bound = amaxes[-1] * amaxes[-2] * g["K"]
        print("  worst-case |D| <= max|A|*max|B|*K = %.6g   (fp32 max 3.40e38)" % bound)
        if bound > 3.4e38:
            print("  -> inputs alone can overflow fp32; a correct kernel may return inf here.")
    # Charged to parent_only: every trial child runs with --no-reference, so an O(MNK)
    # fp64 matmul left in `setup` would inflate the budget for all of them by the one
    # cost none of them pays.  Stamped outside the branch so the skip costs nothing.
    _t = time.time()
    # "skipped" and "unavailable" must not collapse into one state.  Every
    # mutation trial child runs with --no-reference by design, so treating a
    # deliberate skip as an unanswered overflow check would relabel each trial's
    # verdict and score the whole sweep ERROR.
    ref_status = "unavailable"
    if args.no_reference:
        print("  skipped (--no-reference)")
        ref, ref_status = None, "skipped"
    else:
        try:
            ref = _reference_call(cap["op"], func, a, kw)
        except Exception as e:                     # e.g. _scaled_mm has no CPU fp64 path
            print("  reference unavailable: %s" % e)
            ref = None
        if ref is not None and not any(isinstance(t, torch.Tensor) and t.is_floating_point()
                                       for t in tree_leaves(ref)):
            print("  reference unavailable: it returned no floating-point tensor")   # never 'finite'
            ref = None
        if ref is not None:
            ref_status = "ok"
            # Over every tensor the op returned, not just the first: a backward
            # that is clean in grad_q and overflowing in grad_k is still an
            # overflow, and reading only one slot would call it a kernel bug.
            # tree_leaves, as the chain's terminal reference reads it: a math reference returns a
            # dict ({'return': t} / {out_name: t}), which out_tensors() drops -- an empty list read
            # as 'finite', and a 0/0 RMSNorm row proposed as KERNEL_BUG (2026-11-21, op-only package)
            _rts = [t for t in tree_leaves(ref) if isinstance(t, torch.Tensor) and t.is_floating_point()]
            rf = all(torch.isfinite(t).all().item() for t in _rts)
            _fin = [t[torch.isfinite(t)].abs().max().item()
                    for t in _rts if torch.isfinite(t).any()]
            rmax = max(_fin) if _fin else float("nan")
            print("  reference finite=%s  max|.|=%.6g" % (rf, rmax))
            # Finite in fp64 is NOT the same as representable in the output dtype:
            # a true result of 1e40 is a perfectly ordinary fp64 number and an
            # unavoidable inf in fp32. Compare against the output type's range.
            omax = torch.finfo(cap["out_dtype"]).max
            if (not rf or rmax > omax) and _synthetic(cap) and not _derived_ok(cap["op"]):
                # Each float operand is drawn from its own statistics, so operands
                # that must agree with each other do not: a backward's saved forward
                # outputs (logsumexp, out) against q.k gave exp(S - lse) = 1e224 on an
                # SDPA backward whose real chain is finite.  A verdict on these values
                # would be about the synthesis, not the kernel.
                print("  -> %s on SYNTHETIC operands: inconclusive. They are drawn"
                      % ("non-finite" if not rf else "overflows %s" % cap["out_dtype"]))
                print("     independently per tensor, so operands that must be")
                print("     consistent (a backward's saved forward outputs) are not.")
                print("     Re-run with --chain, which recomputes them from the forward.")
                ref_status = "synthetic_inconclusive"
            elif not rf:
                print("  -> the TRUE result is non-finite even in fp64. Model divergence,")
                print("     not a hipBLASLt bug. Stop here.")
                print("[verdict] TRUE_RESULT_NONFINITE")
                return _list_after_verdict(g, disasm_kernel, 1)
            elif rmax > omax:
                print("  -> the true result (%.6g) exceeds %s max (%.6g)."
                      % (rmax, cap["out_dtype"], omax))
                print("     Overflow here is CORRECT arithmetic. Any conforming kernel")
                print("     returns inf. Not a hipBLASLt bug -- the operands got too big.")
                print("[verdict] CORRECT_OVERFLOW")
                return _list_after_verdict(g, disasm_kernel, 1)
            if ref_status == "ok":
                print("  -> the true result is finite AND fits in %s, so a non-finite GPU"
                      % cap["out_dtype"])
                print("     result is a kernel bug.")
    _BASELINE["parent_only"] += time.time() - _t

    # ---- 3. does the GPU still do it? --------------------------------------
    print("\n[3] GPU replay x%d%s"
          % (args.iters, "  (stopping at the first NaN)" if args.stop_on_nan else ""))
    # Everything up to here is what a trial child also pays before its first launch,
    # minus the two things it skips; see _BASELINE.  Taken before the loop rather than
    # after, so a long replay does not get counted as setup.
    setup = max(0.0, time.time() - _T0 - _BASELINE["parent_only"])
    # Co-tenants are started here, not earlier: everything above is one-shot
    # setup, and load running during it would only be load nobody measured.
    # They are then HELD across the sweep (see sweep_holds_load): trial children do
    # not start their own, they land on the device these are already loading, which
    # is the same N+1 geometry this baseline is about to be measured at.
    if (args.contend or args.contend_with) and not start_contenders(args.contend):
        stop_contenders()
        print("  !! contention was requested and did not materialise -- not"
              " reporting a rate from the solo case.")
        print("[verdict] CONTENTION_INVALID")
        return 1
    _start_loop()
    bad = 0
    worst = None
    ran = 0
    # A wrong-value capsule (origin wrong_value) counts a launch whose output fails the
    # capsule's own check again, as the chain loop does -- finiteness alone reads a race
    # that writes plausible numbers as 0/N.
    verify = cap.get("verify")
    if verify:
        print("  [wrong-value capsule: a launch counts when the op fails its %s check again]"
              % verify.get("kind"))
    t_replay = time.time()
    rerun_cap = (verify or {}).get("kind") == "rerun"
    if rerun_cap:
        import op_nan_capture as onc
    for _ in range(args.iters):
        if rerun_cap:               # the capture's rerun check: copies first, same generator
            ra, rk = onc.clone_operands(func, a, kw)
            rng0 = onc._rng_states()
        out = func(*a, **kw)
        ran += 1
        if rerun_cap and ra is not None:
            post = onc._rng_states()
            onc._set_rng_states(rng0)
            out2 = func(*ra, **rk)
            onc._set_rng_states(post)
            rer = onc.rerun_diff(func, a, kw, out, ra, rk, out2)
            _fin = not (rer is not None and not rer.get("int") and rer["rel"] > rer["floor"])
            if not _fin:
                bad += 1
                if worst is None:
                    worst = [(rer["where"], rer["x"].clone())]
                if args.stop_on_nan:
                    break
            if _loop_checkpoint(ran, bad):
                break
            continue
        # .all() syncs, so the loop is timed honestly rather than timing the enqueue.
        _scan = scan_targets(func, a, out)
        _vown = _value_owned(func, a, kw, out)
        if _vown is not None:       # judged by its value check alone (see _run_chain_once)
            _fin = math.isfinite(_vown[0]) and (
                not verify or _vown[0] <= (verify.get("threshold") or _vown[1]))
        else:
            _fin = all(torch.isfinite(t).all() for _, t in _scan)
            if _fin and verify and _wrong_value(verify, func, a, kw,
                                                [t for _, t in out_tensors(out)]) is not None:
                _fin = False
        if not _fin:
            bad += 1
            if worst is None:
                worst = [(lbl, t.clone()) for lbl, t in _scan]
            if args.stop_on_nan:
                break
        if _loop_checkpoint(ran, bad):
            break
    t_replay = time.time() - t_replay
    if not sweep_holds_load():
        stop_contenders()
    if ran:
        # The rate every CLEAN row's statistical power is computed against.
        _BASELINE["rate"] = bad / float(ran)
        # Per launches RUN, not per --iters: a run that stopped early still measured a
        # launch correctly, and dividing by the count it did not reach would report a
        # kernel several times faster than it is.  When nothing stops early the two are
        # the same, which is the arithmetic the budget is specified in.
        _BASELINE["setup"] = setup
        _BASELINE["per_launch"] = t_replay / ran
    _load = effective_load()
    print("  non-finite on %d/%d launches%s%s"
          % (bad, ran, "" if not _load else "  (under %s)" % _load,
             _stopped_note(ran, args.iters).replace("at the first;", "at the first NaN;")))
    if _MUTATING and args.mutate_timeout is None and ran:
        # The one line that makes an auto --mutate-timeout reproducible from the log.
        print("  timing: %.1fs setup + %.4gs/launch -> a mutation trial gets %.0fs"
              " (%gx this run at --iters %d)"
              % (setup, _BASELINE["per_launch"], mutate_timeout(),
                 _MUTATE_BUDGET, args.iters))
    if _MUTATING and sweep_holds_load():
        # Worth saying out loud: this is also what keeps the auto budget honest.  The
        # documented 4x blowup came from timing the baseline on one node state and the
        # trials on another; holding the load makes them the same state.
        print("  sweep:  co-tenants HELD for every trial, so the baseline above and"
              " the trials below")
        print("          are measured on the same device. (--mutate-solo drops them"
              " instead.)")
    if bad == 0:
        print("  -> did NOT reproduce in isolation. The fault needs surrounding state:")
        print("     concurrent streams, a specific workspace/allocator address, or a")
        print("     preceding kernel.")
        _np, _ = capture_gpu_procs(cap)
        if not effective_load():
            # Order matters.  Raising --iters is the lever for a fault that is
            # genuinely intermittent at a given concurrency; it does nothing at
            # all for one that is gated on concurrency, because a serial loop in
            # one process is N sequential launches with the trigger absent every
            # time.  So point at --contend first, and say what the capture saw.
            print("     This run was SOLO%s."
                  % ("" if not _np else ", but the capture saw %s process(es) on the"
                                        " device" % _np))
            print("     Try --contend-auto FIRST -- for a contention-gated fault,")
            print("     --iters cannot substitute for a second process.")
        print("     Also re-check the capsule's env was restored (--restore-env).")
        print("[verdict] NOT_REPRODUCED")
        # --list-mutations first, mirroring run_mutations' own precedence: it returns
        # after printing the plan, so asking for both is asking for the list.
        if args.list_mutations:
            # Enumeration is not a sweep.  It reads the kernel's disassembly and prints
            # what WOULD be tried, which needs a kernel name and nothing else; a
            # baseline NaN is what makes a TRIAL's verdict mean something, and there
            # are no trials on this path.  So the list is exactly as true here as on a
            # capsule that did reproduce, and withholding it would be withholding an
            # answer the user asked for by name.  The verdict below is unaffected.
            try:
                run_mutations(g, kernel=disasm_kernel)
            finally:
                stop_contenders()
        elif args.mutate and args.mutate_force:
            # The one verdict where forcing earns its keep.  bad == 0 at --iters 1 is
            # not proof the kernel is clean -- this fault is order-dependent, and the
            # capsule that motivated the flag reproduces only under surrounding state.
            # So the sweep is worth running, as long as it is read backwards: a CLEAN
            # row says nothing (a clean baseline makes every row clean), and the row
            # that carries information is a NAN one -- including, especially, identity.
            print("\n  -> --mutate-force: sweeping anyway, against a baseline that showed")
            print("     no NaN. Read the table backwards: CLEAN means only that this")
            print("     trial was clean too, NOT that the mutation removed anything.")
            print("     A NAN row is the finding -- it shows the fault is intermittent")
            print("     rather than absent, and a NAN on the identity control says so")
            print("     without implicating any mutation. Raise --iters to make an")
            print("     intermittent fault more likely to show up in a trial.")
            try:
                run_mutations(g, kernel=disasm_kernel)
            finally:
                stop_contenders()
        elif args.mutate:
            print("\n  -> skipping the mutation sweep: with no NaN in the baseline there is")
            print("     nothing for a mutation to remove, so every trial would come back")
            print("     CLEAN and none of it would mean anything. Pass --mutate-force to")
            print("     sweep regardless (useful when the fault is intermittent).")
        # NOT_REPRODUCED is the verdict either way, and child_verdict() parses the line above.
        # The exit status: an enumeration asked for by itself (--list-mutations without --mutate)
        # succeeded once it printed -- rc 1 made the enumeration job read FAILED (2026-11-23);
        # a replay or a sweep that found nothing still exits 1.
        return 0 if (args.list_mutations and not args.mutate) else 1

    # worst is one (index, tensor) per tensor the op returned.  Report each slot
    # separately: for a multi-output op "which operand went non-finite" is part of
    # the finding, and summing them into one count would hide that grad_q is bad
    # while grad_k and grad_v are clean.
    _ref_by_i = ({"out[%d]" % i: t for i, t in out_tensors(ref.get("return") if isinstance(ref, dict) else ref)}
                 if ref is not None else {})
    _multi = len(worst) > 1
    for i, o in worst:
        label = ("  %s: " % i) if _multi else "  "
        nbad = (~torch.isfinite(o)).sum().item()
        if _multi and nbad == 0:
            print("%snan=0 inf=0 of %d elements (clean)" % (label, o.numel()))
            continue
        print("%snan=%d inf=%d of %d elements"
              % (label, torch.isnan(o).sum().item(), torch.isinf(o).sum().item(),
                 o.numel()))
        idx = (~torch.isfinite(o)).nonzero()
        print("%sfirst bad indices: %s" % (label, idx[:8].tolist()))
        r = _ref_by_i.get(i)
        if r is not None:
            m = torch.isfinite(o.cpu())
            if m.any():
                err = (o.cpu()[m].double() - r[m]).abs().max().item()
                print("%smax |gpu-ref| over the finite entries: %.6g" % (label, err))
    if rerun_cap:
        # A rerun capsule needs no reference: different outputs on identical copies of the
        # operands is wrong whatever the operands are (replay_verdict: terminal 'rerun').
        print("\n  -> REPRODUCED standalone: the op gave different outputs on identical copies"
              " of its operands (rerun check) -- no correct deterministic kernel does that,"
              " whatever the operands; no reference needed.")
        print("[verdict] KERNEL_BUG")
        if _MUTATING:
            try:
                return run_mutations(g, kernel=disasm_kernel)
            finally:
                stop_contenders()
        return 0
    # The overflow question (step 5b.2) is only ANSWERED when a reference ran.
    # Many fused kernels have no CPU implementation -- SDPA backward is CUDA-only
    # -- and reporting KERNEL_BUG off a reference that never executed claims an
    # answer nobody computed.  Say which of the two this is.
    if ref_status in ("unavailable", "synthetic_inconclusive"):
        print("\n  -> REPRODUCED standalone, but %s, so"
              % ("the fp64 reference never ran" if ref_status == "unavailable"
                 else "the fp64 reference ran on inconsistent synthetic operands"))
        print("     'is this an honest overflow?' is UNANSWERED for this op.")
        if cap.get("chain"):
            print("     This capsule carries a chain: re-run with --chain to check"
                  " whether an upstream op handed it a finite but wrong operand.")
        else:
            print("     inputs_finite=%s means no NaN on the operands, NOT that they"
                  % cap.get("inputs_finite"))
            print("     were correct. Re-capture with chain_depth=N to find out.")
        print("[verdict] KERNEL_BUG_UNVERIFIED")
        if _MUTATING:
            try:
                return run_mutations(g, kernel=disasm_kernel)
            finally:
                stop_contenders()
        return 0

    if cap["op"].startswith(("inductor.", "triton.")):
        print("\n  -> REPRODUCED standalone. Next: the kernel's Triton source and its lowering "
              "(step7-rootcause.md, compiled kernels) -- no library solution to pin.")
    else:
        print("\n  -> REPRODUCED standalone. Next: pin solutions to find the bad one, e.g.")
        print("     for s in <candidates>; do python3 %s %s --solution-index $s --iters 20; done"
              % (os.path.basename(__file__), args.capsule))
    if g:
        print("     and the geometry-only check (expected to pass -- the fault is data-dependent):")
        print("     " + bench_cmd(g, solution=args.solution_index))
    print("[verdict] KERNEL_BUG")
    if _MUTATING:
        # --disasm already paid for discovery; reuse the name rather than spending a
        # second child replay to be told the same thing.
        try:
            return run_mutations(g, kernel=disasm_kernel)
        finally:
            stop_contenders()
    return 0


if __name__ == "__main__":
    sys.exit(main())
else:
    # main() is what normally binds torch, unpack, tree_leaves, ...; an importer
    # gets them bound here instead of a module full of None.
    _import_torch()
