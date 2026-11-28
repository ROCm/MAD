#!/usr/bin/env python3
"""Replay NaN capsules at the original job width, under torchrun.

    torchrun --nproc_per_node=8 replay_nan_op_dist.py \
        'scratch/nan_capsule.rank{rank}.pt' --iters 20

Two modes, and they answer different questions:

  per-rank (default)
      Every rank replays ITS OWN capsule, on the same local device, with RCCL
      initialised at the original world size.  This is the faithful reproduction
      of the moment the run failed.  Use it to confirm the failure survives
      outside the training loop.

  --shared PATH
      Every rank replays the SAME capsule -- identical bytes, identical layout,
      eight devices.  The inputs are now a controlled variable, so a split result
      is diagnostic:
          all ranks NaN      -> the kernel is wrong for this data, on every GPU
          some ranks NaN     -> device- or scheduling-dependent; suspect the GPUs
                                that failed, or a race
          no rank NaN        -> the fault needed surrounding state, not this op
      In this mode the output digests are compared across ranks; identical inputs
      must give bit-identical outputs, so any divergence is itself the finding.

--------------------------------------------------------------------------------
On "distributed" and what it actually buys you

A single op is a local operation: nothing about eight ranks changes the arithmetic of
one kernel launch.  Running the replay at full width reproduces the *context* --
RCCL traffic competing for the same XCDs and HBM bandwidth, the same per-device
allocator state, the same number of processes per node.  If the failure is a race
or depends on concurrent collectives, that context is the whole point, and
--collective-noise makes it explicit by keeping an all-reduce in flight across
the op.  If the replay reproduces on a single GPU too, prefer the single-GPU
repro: it is smaller and it is what you can hand to AMD.
"""

import argparse
import hashlib
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default="nan_capsule.rank{rank}.pt",
                    help="per-rank capsule path; '{rank}' is substituted")
    ap.add_argument("--shared", metavar="PATH",
                    help="every rank replays this one capsule instead of its own")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--solution-index", type=int, default=None,
                    help="pin TENSILE_SOLUTION_INDEX on every rank")
    ap.add_argument("--no-hipblaslt", action="store_true",
                    help="TORCH_BLAS_PREFER_HIPBLASLT=0 (route to rocBLAS)")
    ap.add_argument("--collective-noise", type=float, default=0.0, metavar="MB",
                    help="keep an async all-reduce of this size in flight across each "
                         "op, to reproduce concurrent RCCL traffic (0 = off)")
    ap.add_argument("--emit-log", metavar="DIR",
                    help="write per-rank hipBLASLt logs (HIPBLASLT_LOG_MASK=160) here")
    ap.add_argument("--backend", default="nccl", help="process group backend (rccl uses 'nccl')")
    return ap.parse_args()


args = parse_args()

_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
_RANK = int(os.environ.get("RANK", "0"))

# ---- environment, before torch is imported ---------------------------------
if args.solution_index is not None:
    os.environ["TENSILE_SOLUTION_INDEX"] = str(args.solution_index)
if args.no_hipblaslt:
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
if args.emit_log:
    os.makedirs(args.emit_log, exist_ok=True)
    os.environ["HIPBLASLT_LOG_MASK"] = "160"          # bench + extended profile
    os.environ["HIPBLASLT_LOG_FILE"] = os.path.join(args.emit_log, "rank%d.log" % _RANK)

import torch                                          # noqa: E402
import torch.distributed as dist                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from op_nan_capture import SHAPES_ONLY_MSG, has_data, unpack, upgrade_capsule   # noqa: E402


def resolve_op(name):
    """'ns.op.overload' -> the OpOverload, any namespace (aten, fsdp, flash_attn, ...).
    A library op needs its library imported; the error says so.  A triton.* op is a
    Triton kernel launch, rebuilt from the capsule's triton_kernels descriptor."""
    if name.startswith(("triton.", "inductor.")):
        from op_nan_capture import triton_op_for
        try:
            return triton_op_for(name)
        except Exception as e:                          # noqa: BLE001
            raise SystemExit("Triton op %r did not resolve: %s: %s" % (name, type(e).__name__, e))
    parts = name.split(".")
    if len(parts) != 3:
        raise SystemExit("unsupported op string %r" % name)
    try:
        return getattr(getattr(getattr(torch.ops, parts[0]), parts[1]), parts[2])
    except (AttributeError, RuntimeError) as e:
        raise SystemExit("op %s not registered here (%s); import its library first, or "
                         "replay the chain with replay_nan_op.py --collective-noise" % (name, e))


def digest(t):
    return hashlib.blake2b(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(),
                           digest_size=16).hexdigest()


def main():
    if "RANK" not in os.environ:
        raise SystemExit("launch me with torchrun, e.g.\n"
                         "  torchrun --nproc_per_node=8 %s '<pattern>'" % sys.argv[0])

    torch.cuda.set_device(_LOCAL_RANK)
    dist.init_process_group(backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", _LOCAL_RANK)

    path = args.shared if args.shared else args.pattern.format(rank=rank)
    have = os.path.exists(path)

    # Every rank must reach every collective below, so a missing capsule cannot
    # be an early return -- it has to be carried through as a status.
    cap = upgrade_capsule(torch.load(path, map_location="cpu", weights_only=False)) if have else None
    if cap is not None and not has_data(cap):
        # Same reason: carried through as "nothing replayed", never raised here.
        print("rank %d: %s: %s" % (rank, path, SHAPES_ONLY_MSG), flush=True)
        cap = None

    if cap is not None:
        orig_world = cap["dist"]["world_size"]
        if rank == 0 and orig_world != world:
            print("WARNING: capsule captured at world_size=%d, replaying at %d.\n"
                  "         Launch with --nproc_per_node=%d to match the original run."
                  % (orig_world, world, orig_world), flush=True)
        prec = cap["precision"]
        torch.backends.cuda.matmul.allow_tf32 = prec["allow_tf32"]
        torch.set_float32_matmul_precision(prec["float32_matmul_precision"])

        func = resolve_op(cap["op"])
        a = unpack(cap["args"], dev)
        kw = unpack(cap["kwargs"], dev)
        inputs_finite = all(torch.isfinite(t).all().item()
                            for t in a if isinstance(t, torch.Tensor)
                            and t.is_floating_point() and t.numel())
    else:
        func = a = kw = None
        inputs_finite = None

    noise = None
    if args.collective_noise > 0:
        n = int(args.collective_noise * 1e6 / 4)
        noise = torch.ones(n, dtype=torch.float32, device=dev)

    dist.barrier()

    # ---- replay -------------------------------------------------------------
    bad = 0
    out_sha = None
    if cap is not None:
        for i in range(args.iters):
            handle = None
            if noise is not None:
                # async_op=True leaves the all-reduce in flight on its own stream
                # while the op runs, which is the overlap the training loop had.
                handle = dist.all_reduce(noise, async_op=True)
            out = func(*a, **kw)
            if handle is not None:
                handle.wait()
            if not torch.isfinite(out).all():
                bad += 1
                if out_sha is None:
                    out_sha = digest(out)
        if out_sha is None:
            out_sha = digest(out)

    # ---- collect ------------------------------------------------------------
    report = {
        "rank": rank,
        "host": os.uname().nodename,
        "dev": _LOCAL_RANK,
        "dev_name": torch.cuda.get_device_name(_LOCAL_RANK),
        "path": path if have else None,
        "bad": bad,
        "iters": args.iters if cap is not None else 0,
        "inputs_finite": inputs_finite,
        "out_sha": out_sha,
        "orig_rank": cap["dist"]["rank"] if cap is not None else None,
        "call_index": cap["op_call_index"] if cap is not None else None,
        "geom": cap["geometry"] if cap is not None else None,
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, report)

    if rank == 0:
        mode = "shared capsule %s" % args.shared if args.shared else "per-rank capsules"
        print("\n=== distributed replay: %s, %d ranks, %d iters each ==="
              % (mode, world, args.iters))
        if args.collective_noise:
            print("    with %.1f MB all-reduce in flight across each op"
                  % args.collective_noise)
        g = next((r["geom"] for r in gathered if r["geom"]), None)
        if g:
            print("    problem: M=%d N=%d K=%d transA=%s transB=%s lda=%d ldb=%d ldd=%d"
                  % (g["M"], g["N"], g["K"], g["transA"], g["transB"],
                     g["lda"], g["ldb"], g["ldd"]))
        print()
        print("%-5s %-14s %-4s %-28s %-11s %-10s %s"
              % ("rank", "host", "dev", "device", "in_finite", "nonfinite", "out digest"))
        print("-" * 110)
        for r in sorted(gathered, key=lambda r: r["rank"]):
            if r["path"] is None:
                print("%-5d %-14s %-4d %-28s %s"
                      % (r["rank"], r["host"][:14], r["dev"], r["dev_name"][:28],
                         "NO CAPSULE (this rank saw no NaN op)"))
                continue
            print("%-5s %-14s %-4d %-28s %-11s %-10s %s"
                  % (r["rank"], r["host"][:14], r["dev"], r["dev_name"][:28],
                     r["inputs_finite"], "%d/%d" % (r["bad"], r["iters"]),
                     (r["out_sha"] or "-")[:16]))

        active = [r for r in gathered if r["path"]]
        hit = [r["rank"] for r in active if r["bad"]]
        print()
        if not active:
            print("VERDICT: no capsules found.  Check the --pattern / path.")
        elif not hit:
            print("VERDICT: did NOT reproduce on any rank.")
            print("  The op alone is not sufficient.  Try --iters higher, add")
            print("  --collective-noise, or accept that the fault needs the surrounding")
            print("  kernels/allocator state and go back to in-situ instrumentation.")
        elif len(hit) == len(active):
            print("VERDICT: reproduced on ALL %d ranks -> systematic, not device-specific."
                  % len(active))
            print("  Drop to the single-GPU repro; it is the one to hand to AMD:")
            print("  python3 replay_nan_op.py %s --iters 20" % active[0]["path"])
        else:
            print("VERDICT: reproduced on %d of %d ranks: %s" % (len(hit), len(active), hit))
            if args.shared:
                print("  Same bytes on every GPU, different outcomes -> device- or")
                print("  schedule-dependent.  Suspect those GPUs, or a race.")
            else:
                print("  Each rank had different data, so this may just mean the other")
                print("  ranks' inputs were benign.  Re-run with --shared %s to make the"
                      % active[0]["path"])
                print("  data a controlled variable before blaming a device.")

        if args.shared:
            shas = set(r["out_sha"] for r in active if r["out_sha"])
            print()
            if len(shas) <= 1:
                print("output digests: identical on all ranks (as they must be for equal inputs)")
            else:
                print("output digests: %d DISTINCT values across ranks." % len(shas))
                print("  Identical inputs produced different outputs.  That is a finding on")
                print("  its own -- nondeterminism in the kernel (split-K/GSU atomics) or a")
                print("  genuinely misbehaving device.")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
