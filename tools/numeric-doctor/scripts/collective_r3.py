#!/usr/bin/env python3
"""R3 (debug-numeric, world size > 1): is a collective itself wrong?

A capture run with NAN_CAPTURE_RECORD_COLLECTIVE=<group>:<seq>[:<step>] (group and seq from
the job manifest's earliest arrival) leaves, on every member rank,
<capsule dir>/collective_g<group>_s<seq>.rank<r>.pt: the collective's inputs before the call
and its outputs after it landed.

    python3 collective_r3.py check  <dir> --group G --seq S          # no GPU
    torchrun --nproc_per_node=<group size> collective_r3.py replay <dir> --group G --seq S [--iters 200]

check   recomputes, in fp64 from ALL ranks' recorded inputs, what a correct collective must
        have produced on each rank, and compares with what each rank recorded:
          COLLECTIVE_FAULT ranks [..]   output wrong while every input was finite
          INPUT_NONFINITE  ranks [..]   the NaN came in on those ranks' inputs (look there)
          CORRECT                       outputs match the inputs
replay  re-runs the same collective on the recorded inputs --iters times on this machine's
        GPUs and counts outputs that differ from the fp64 expectation (an intermittent
        transport fault reproduces here, without the model).

A synchronous collective (vLLM's vllm.all_reduce: op_nan_capture.py captures it at the
collective, on every member rank) needs no recapture: its member ranks' CAPSULES are the
record.  Give the capsule dir (or a package's capsule/ dir) and omit --group/--seq to take the
manifest's collective_origin (or the one collective the capsules share):

    python3 collective_r3.py check  <capsule dir>                     # no GPU
    torchrun --nproc_per_node=2 collective_r3.py replay <capsule dir> --backend vllm [--iters 1000]

--backend vllm replays through vLLM itself: its distributed environment and TP group, the
recorded op (torch.ops.vllm.all_reduce) and the capsule's VLLM_* env -- so vLLM picks the same
all-reduce (QuickReduce / custom all-reduce / RCCL) for the same size.  Run it in the model's
image.  check also prints a NARROW_INTERMEDIATE? hint when inputs and the true result fit the
dtype but exceed a narrower float's range (fp16 65504, fp8): a cast inside the collective.
Last line of replay: LADDER_RESULT=REPRODUCED|NOT_REPRODUCED n/N (the 5b ladder's shape).

Covers c10d allreduce_ / _allgather_base_ / _reduce_scatter_base_ / alltoall_base_ and the
_c10d_functional all_reduce / all_gather_into_tensor / reduce_scatter_tensor /
all_to_all_single.  A c10d reduction's op (a ScriptObject) is not recorded: check tries
sum / avg / max / min and takes the one most ranks agree with.
"""
import argparse
import glob
import os
import sys

import torch


# fp16 and the fp8 formats a collective may pack into: (name, max finite)
_NARROW = (("fp16", 65504.0), ("fp8_e5m2", 57344.0), ("fp8_e4m3fn", 448.0))


def _capsule_records(d):
    """Synchronous-collective capsules under d -> {(group, seq, op): {rank: record}}.
    A record has the R3 shape (op, rank, group_ranks, args_pre) plus out_nonfinite (the
    capsule keeps the output's non-finite count, not its values) and env."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import op_nan_capture as onc
    found = {}
    for f in sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True)):
        if os.path.basename(f).startswith("collective_g"):
            continue
        try:
            raw = onc.upgrade_capsule(torch.load(f, map_location="cpu", weights_only=False))
        except Exception:                               # noqa: BLE001 - not a capsule
            continue
        c = raw.get("collective") if isinstance(raw, dict) else None
        if not c:
            continue
        if raw.get("data", "values") != "values":
            sys.exit("%s is shapes-only: a collective check needs values (NAN_CAPTURE_DATA=1)" % f)
        rank = (raw.get("dist") or {}).get("rank", raw.get("rank"))
        rec = {"op": raw["op"], "rank": rank, "group_ranks": c.get("group_ranks"),
               "args_pre": list(onc.unpack(raw["args"], "cpu")), "args_post": None, "out_post": None,
               "out_nonfinite": int(raw.get("out_nan") or 0) + int(raw.get("out_inf") or 0),
               "env": raw.get("env") or {}, "path": f}
        found.setdefault((c["group"], c["seq"], raw["op"]), {})[rank] = rec
    return found


def _pick(d, group, seq):
    """(group, seq) to load: as given, else the manifest's collective_origin, else the only
    collective the capsules share."""
    if group is not None and seq is not None:
        return group, seq
    import json
    for m in sorted(glob.glob(os.path.join(d, "**", "manifest*.json"), recursive=True)):
        try:
            co = (json.load(open(m)).get("verdict") or {}).get("collective_origin")
        except Exception:                               # noqa: BLE001
            co = None
        if co:
            return co["group"], int(co["seq"])
    keys = sorted({(g, s) for g, s, _ in _capsule_records(d)})
    if len(keys) != 1:
        sys.exit("give --group/--seq: %s" % (keys or "no collective capsule under %s" % d))
    return keys[0]


def load(d, group, seq):
    caps = {k: v for k, v in _capsule_records(d).items() if k[0] == group and k[1] == seq}
    if caps:
        recs = next(iter(caps.values()))
        ranks = sorted(recs)
        gr = next(iter(recs.values())).get("group_ranks")
        if gr and sorted(gr) != ranks:
            sys.exit("capsules for ranks %s, the group has %s (every member must capture)" % (ranks, gr))
        return [recs[r] for r in ranks], ranks
    recs = {}
    fg = str(group).replace(":", "_").replace("/", "_")         # op_nan_capture._file_group
    for f in glob.glob(os.path.join(d, "**", "collective_g%s_s%d.rank*.pt" % (fg, seq)), recursive=True):
        r = torch.load(f, map_location="cpu", weights_only=False)
        recs[r["rank"]] = r
    if not recs:
        sys.exit("no collective_g%s_s%d.rank*.pt under %s" % (group, seq, d))
    ranks = next(iter(recs.values())).get("group_ranks") or sorted(recs)
    missing = [r for r in ranks if r not in recs]
    if missing:
        sys.exit("records missing for ranks %s (every member of the group must record)" % missing)
    return [recs[r] for r in ranks], ranks


def kind(op):
    for k in ("all_to_all", "alltoall", "reduce_scatter", "allgather", "all_gather", "allreduce", "all_reduce"):
        if k in op:
            return {"alltoall": "all_to_all", "allgather": "all_gather",
                    "allreduce": "all_reduce"}.get(k, k)
    return None


def io(rec):
    """(input, recorded output, extra) of one rank's record."""
    op, a, ao, out = rec["op"], rec["args_pre"], rec["args_post"], rec["out_post"]
    k = kind(op)
    if op.startswith("c10d."):
        if k == "all_reduce":
            return a[0][0], ao[0][0], {}
        if k == "all_to_all":
            return a[1], ao[0], {"out_split": list(a[3] or []), "in_split": list(a[4] or [])}
        return a[1], ao[0], {}                          # _allgather_base_ / _reduce_scatter_base_
    o = out[0] if isinstance(out, (list, tuple)) else out
    if op.startswith("vllm."):                          # (tensor, group_name, ...): always a sum
        return a[0], o, {"reduce_op": "sum"}
    if k == "all_reduce":
        return a[0], o, {"reduce_op": a[1]}
    if k == "reduce_scatter":
        return a[0], o, {"reduce_op": a[1]}
    if k == "all_to_all":
        return a[0], o, {"out_split": list(a[1] or []), "in_split": list(a[2] or [])}
    return a[0], o, {}


_REDUCE = {"sum": lambda xs: torch.stack(xs).sum(0), "avg": lambda xs: torch.stack(xs).mean(0),
           "max": lambda xs: torch.stack(xs).amax(0), "min": lambda xs: torch.stack(xs).amin(0)}


def expected(recs, reduce_name):
    """[expected output per rank (fp64)] for the group, given a reduction name."""
    ins = [io(r)[0].double() for r in recs]
    k, n = kind(recs[0]["op"]), len(recs)
    if k == "all_reduce":
        e = _REDUCE[reduce_name](ins)
        return [e] * n
    if k == "all_gather":
        e = torch.cat(ins, 0)
        return [e] * n
    if k == "reduce_scatter":
        return list(_REDUCE[reduce_name](ins).chunk(n, 0))
    if k == "all_to_all":
        out = []
        for dst in range(n):
            parts = []
            for src, r in enumerate(recs):
                split = io(r)[2]["in_split"] or [ins[src].shape[0] // n] * n
                parts.append(ins[src].split(split, 0)[dst])
            out.append(torch.cat(parts, 0))
        return out
    raise SystemExit("unsupported collective %s" % recs[0]["op"])


def tol(dtype):
    return {torch.bfloat16: 2e-2, torch.float16: 5e-3}.get(dtype, 1e-4)


def agrees(got, exp):
    g = got.double()
    if g.shape != exp.shape:
        return False
    fin = torch.isfinite(exp)
    if not torch.equal(torch.isfinite(g), fin):
        return False
    scale = max(float(exp[fin].abs().max()) if fin.any() else 0.0, 1e-30)
    return float((g[fin] - exp[fin]).abs().max() if fin.any() else 0.0) <= tol(got.dtype) * scale


def reductions(recs):
    k = kind(recs[0]["op"])
    if k not in ("all_reduce", "reduce_scatter"):
        return [None]
    named = io(recs[0])[2].get("reduce_op")
    return [named.lower()] if isinstance(named, str) and named.lower() in _REDUCE else list(_REDUCE)


def narrow_hint(recs, exp):
    """NARROW_INTERMEDIATE? <dtype> when every input and the true result fit the recorded dtype
    but some |input| or |expected| exceeds a narrower float's max: a cast inside the collective."""
    x = io(recs[0])[0]
    if not x.is_floating_point() or not all(torch.isfinite(io(r)[0].float()).all() for r in recs):
        return None
    big = max(max(io(r)[0].double().abs().max().item() for r in recs),
              max(e.abs().max().item() for e in exp))
    if big > torch.finfo(x.dtype).max:
        return None
    for name, mx in _NARROW:
        if big > mx and mx < torch.finfo(x.dtype).max:
            return ("NARROW_INTERMEDIATE? %s -- max |input or true result| %.4g > %s max %g, while %s holds "
                    "it: a %s cast/packing inside the collective overflows" % (name, big, name, mx,
                                                                              x.dtype, name))
    return None


def check(d, group, seq):
    recs, ranks = load(d, group, seq)
    best = None
    for red in reductions(recs):
        exp = expected(recs, red)
        if recs[0].get("out_post") is None:             # capsules: the output's non-finite count only
            ok = [not (r["out_nonfinite"] and bool(torch.isfinite(e).all())) for r, e in zip(recs, exp)]
        else:
            ok = [agrees(io(r)[1], e) for r, e in zip(recs, exp)]
        if best is None or sum(ok) > sum(best[1]):
            best = (red, ok, exp)
    red, ok, exp = best
    bad_in = [ranks[i] for i, r in enumerate(recs) if not torch.isfinite(io(r)[0].float()).all()]
    wrong = [ranks[i] for i, o in enumerate(ok) if not o]
    print("collective %s #%d of group %s (ranks %s)%s" % (recs[0]["op"], seq, group, ranks,
                                                          ", reduction %s" % red if red else ""))
    for i, r in enumerate(recs):
        x, y, _ = io(r)
        yfin = (not r["out_nonfinite"]) if y is None else bool(torch.isfinite(y.float()).all())
        print("  rank %-3s input finite=%-5s max|x| %-9.4g output finite=%-5s %s"
              % (ranks[i], bool(torch.isfinite(x.float()).all()), x.float().abs().max().item(), yfin,
                 "OK" if ok[i] else "DIFFERS from the reduction of the recorded inputs"))
    hint = narrow_hint(recs, exp)
    if hint:
        print(hint)
    if bad_in:
        print("VERDICT INPUT_NONFINITE ranks %s -- the NaN came in on the inputs: look upstream there" % bad_in)
    elif wrong:
        print("VERDICT COLLECTIVE_FAULT ranks %s -- every input finite, output wrong: the collective "
              "(%s) is at fault on those ranks" % (wrong, "its implementation: " + recs[0]["op"]
                                                   if recs[0]["op"].startswith("vllm.") else "RCCL / transport"))
    else:
        print("VERDICT CORRECT -- outputs match the recorded inputs")
    return 1 if (wrong and not bad_in) else 0


def vllm_path(comm, x):
    """Which implementation vLLM's device communicator picks for x, in the order its all_reduce
    tries them (vLLM 0.33 CudaCommunicator.all_reduce) -- logged per replay, so an env change that
    only MOVES the call to another implementation is visible (step7-rootcause.md: keep the code path)."""
    def ok(name, test):
        c = getattr(comm, name, None)
        try:
            return c is not None and not getattr(c, "disabled", False) and getattr(c, test)(x)
        except Exception:                               # noqa: BLE001 - another vLLM version
            return False
    qr = getattr(comm, "qr_comm", None)
    for name, test, label in (("fi_ar_comm", "should_use_fi_ar", "flashinfer_ar"),
                              ("qr_comm", "should_quick_allreduce", "quick_reduce"),
                              ("fi_pcie_ipc_ar_comm", "should_use", "flashinfer_pcie_ipc"),
                              ("aiter_ar_comm", "should_custom_ar", "aiter_custom_ar"),
                              ("ca_comm", "should_custom_ar", "custom_ar"),
                              ("symm_mem_comm", "should_use_symm_mem", "symm_mem")):
        if name == "aiter_ar_comm" and not getattr(comm, "use_aiter_allreduce", False):
            continue
        if ok(name, test):
            extra = ""
            if label == "quick_reduce" and qr is not None:
                extra = " (fp16 kernels=%s, quantization=%s)" % (getattr(qr, "use_fp16_kernels", "?"),
                                                                getattr(getattr(qr, "qr_quant_level", None), "name",
                                                                        getattr(qr, "qr_quant_level", "?")))
            return label + extra
    return "pynccl / torch.distributed (no custom implementation took it)"


def replay_vllm(recs, seq, group, iters):
    """The recorded vllm.* collective through vLLM itself, on this rank's recorded input."""
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    op = recs[0]["op"]
    if not op.startswith("vllm.all_reduce"):
        sys.exit("--backend vllm replays vllm.all_reduce only (got %s)" % op)
    env = recs[rank].get("env") or {}
    for k, v in env.items():                            # the all-reduce vLLM picks depends on them
        if k.startswith("VLLM_") and k not in os.environ.get("NAN_R3_KEEP_ENV", "").split(","):
            os.environ[k] = v
    torch.cuda.set_device(local)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel
    import torch.distributed as dist
    exp = expected(recs, "sum")[rank]
    x = io(recs[rank])[0].to("cuda")
    bad = 0
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world, rank, "env://", local, "nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        g = get_tp_group()
        if rank == 0:
            print("R3 vllm: path %s" % vllm_path(g.device_communicator, x), flush=True)
            print("R3 vllm: group %s communicator %s; env %s" % (
                g.unique_name, type(g.device_communicator).__name__,
                " ".join("%s=%s" % kv for kv in sorted(os.environ.items()) if kv[0].startswith("VLLM_ROCM"))),
                flush=True)
        sig = {}
        for _ in range(iters):
            out = torch.ops.vllm.all_reduce(x.clone(), g.unique_name)
            if not agrees(out.cpu(), exp):
                bad += 1
                _signature_once(sig, out, exp, rank)
            else:
                sig["prev"] = out.cpu()
        tot = torch.tensor([bad], device="cuda")
        dist.all_reduce(tot)
    return report(op, seq, group, iters, rank, bad, int(tot), len(recs))


def _signature_once(state, out, exp, rank):
    """The failure signature of the first wrong output (rank 0's), printed once."""
    if state.get("done") or rank != 0:
        return
    state["done"] = True
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from failure_signature import signature
        print(signature(out.cpu(), exp, state.get("prev"))["line"], flush=True)
    except Exception as e:                              # noqa: BLE001 - a missing helper must not fail the replay
        print("SIGNATURE unavailable: %s" % e, flush=True)


def report(op, seq, group, iters, rank, bad, tot, n):
    if rank == 0:
        print("R3 replay %s #%d of group %s x%d: wrong outputs on %d rank-iterations (all ranks)%s"
              % (op, seq, group, iters, tot, "  [REPRODUCED]" if tot else ""))
    print("  rank %d: %d/%d wrong" % (rank, bad, iters), flush=True)
    if rank == 0:
        print("LADDER_RESULT=%s %d/%d" % ("REPRODUCED" if tot else "NOT_REPRODUCED", tot, iters * n), flush=True)
    return 0        # the verdict is the LADDER_RESULT line; a non-zero rank exit makes torchrun print a traceback


def replay(d, group, seq, iters, backend="torch"):
    import torch.distributed as dist
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    recs, ranks = load(d, group, seq)
    if world != len(recs):
        sys.exit("launch with --nproc_per_node=%d (the group size)" % len(recs))
    if backend == "vllm" or (backend == "auto" and recs[0]["op"].startswith("vllm.")):
        return replay_vllm(recs, seq, group, iters)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl" if cuda else "gloo")
    dev = torch.device("cuda") if cuda else torch.device("cpu")
    red = None
    for cand in reductions(recs):                       # the reduction the records agree with
        exp = expected(recs, cand)
        if all(agrees(io(r)[1], e) for r, e in zip(recs, exp)) or red is None:
            red = cand
    exp = expected(recs, red)[rank]
    x, y0, extra = io(recs[rank])
    k = kind(recs[0]["op"])
    ops = {"sum": dist.ReduceOp.SUM, "avg": dist.ReduceOp.AVG, "max": dist.ReduceOp.MAX,
           "min": dist.ReduceOp.MIN}
    bad, sig = 0, {}
    for _ in range(iters):
        xi = x.to(dev)
        if k == "all_reduce":
            out = xi.clone()
            if red == "avg" and not cuda:               # gloo has no AVG
                dist.all_reduce(out)
                out /= world
            else:
                dist.all_reduce(out, op=ops[red])
        elif k == "all_gather":
            out = torch.empty((xi.shape[0] * world,) + tuple(xi.shape[1:]), dtype=xi.dtype, device=dev)
            dist.all_gather_into_tensor(out, xi)
        elif k == "reduce_scatter":
            out = torch.empty((xi.shape[0] // world,) + tuple(xi.shape[1:]), dtype=xi.dtype, device=dev)
            dist.reduce_scatter_tensor(out, xi, op=ops[red])
        else:
            out = torch.empty_like(y0, device=dev)
            dist.all_to_all_single(out, xi, extra.get("out_split") or None, extra.get("in_split") or None)
        if not agrees(out.cpu(), exp):
            bad += 1
            _signature_once(sig, out, exp, rank)
        else:
            sig["prev"] = out.cpu()
    tot = torch.tensor([bad], device=dev)
    dist.all_reduce(tot)
    rc = report(recs[0]["op"], seq, group, iters, rank, bad, int(tot), len(recs))
    dist.destroy_process_group()
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "replay"))
    ap.add_argument("dir")
    ap.add_argument("--group", help="default: the manifest's collective_origin")
    ap.add_argument("--seq", type=int)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--backend", choices=("auto", "torch", "vllm"), default="auto",
                    help="replay through torch.distributed, or through vLLM (auto: by the recorded op)")
    a = ap.parse_args()
    group, seq = _pick(a.dir, a.group, a.seq)
    rc = check(a.dir, group, seq) if a.mode == "check" else replay(a.dir, group, seq, a.iters, a.backend)
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
