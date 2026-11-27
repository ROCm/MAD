#!/usr/bin/env python3
"""Group the NaN capsules in a directory and report what is actually distinct.

    python3 compare_nan_capsules.py CAPSULE_DIR

A capsule is a .pt written by op_nan_capture.OpNanCapture.  A stress sweep or
a distributed run leaves a directory full of them and the first question is always
"is this one bug captured N times, or N bugs?"

Capsules of the SAME bug are never byte-identical -- each carries its own rank,
hostname, device and environment -- so hashing the file, or comparing the
unpickled dicts, puts every capsule in its own group and tells you nothing.  The
capture format answers the question directly instead:

  site       op + geometry key + stack digest + dtypes.  The code location and
             the problem shape.  This is the primary grouping key.

             A Triton / Inductor kernel op (`inductor.*`, `triton.*`) is keyed on
             its kernel source hash and operand shapes instead of the stack: the
             stack runs through the generated module, whose path is a per-process
             cache directory, so every trial's capsule had its own digest and one
             kernel came out as one group per trial (2026-11-23, qwen_vllm_serve_nan
             G1..G3 and softmoe_train_nan G1..G4, all one
             inductor.<kernel>.s<hash>).  The source hash already names the site.

  call index the capsule's `identity` also carries the op call number, but that
             counter is per-PROCESS.  It identifies "the same op" only among
             ranks of ONE run, which is what the capture docstring means.  Across
             independent runs it is noise: a 62-capsule sweep of single-process
             runs has 61 distinct call indices and 2 actual sites.  Grouping on
             it there produces 61 groups of one and answers nothing, so it is
             reported, not keyed.  --strict-identity puts it back in the key when
             you really are comparing ranks within one run.

  arg_shas   one digest per operand.  Within an identity group, capsules whose
             shas agree saw identical inputs (a replicated weight); capsules
             whose shas differ saw different data (an activation).  Comparing
             digests is also what keeps this tool from loading gigabytes.

Outcome fields (out_nan / out_inf / inputs_finite) are reported, never grouped on:
two captures of one bug can differ in how many NaNs came out.

Exit status: 0 if at least one capsule parsed, 2 on a usage error, an empty
directory, or when every capsule failed to parse.
"""

import argparse
import json
import os
import sys
import traceback

import torch

# 3 is 2 plus the chain keys, and every field this tool reads still describes the
# terminal op, so a chain capsule groups alongside a single-op one.  Listing the
# formats rather than testing >= keeps the refusal honest for a format that has
# not been written yet.
KNOWN_FORMATS = frozenset((2, 3))
CAPSULE_FORMAT = 2                     # what a capture writes without chain_depth
SHA_ABBREV = 12


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_capsule(path):
    """Load one capsule onto the CPU without materialising its operands.

    weights_only=False is REQUIRED and must stay: a capsule holds torch.dtype
    objects, strings and nested dicts alongside the tensors, none of which the
    weights_only unpickler accepts.  Do not "harden" this to True -- it breaks
    every load.

    mmap=True keeps the operand bytes on disk; we only ever touch metadata and
    the precomputed digests.  Older torch builds reject the combination, so fall
    back rather than fail the sweep.
    """
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


KERNEL_OP_PREFIXES = ("inductor.", "triton.")


def kernel_source(capsule, op):
    """The kernel source hash of a Triton / Inductor kernel op, None for anything else.

    `inductor.<kernel>.s<hash>` carries it in the name; `triton.<kernel>.w<idx>`
    does not (a hand-written kernel name such as `_fwd_kernel` is not unique), so
    it comes from the capsule's triton_kernels descriptor.  No hash, no folding:
    the op then keeps the stack in its key like an ATen op.
    """
    if not op.startswith(KERNEL_OP_PREFIXES):
        return None
    desc = (capsule.get("triton_kernels") or {}).get(op) or {}
    sha = desc.get("src_sha") if isinstance(desc, dict) else None
    if not sha and op.startswith("inductor."):
        last = op.rsplit(".", 1)[-1]
        sha = last[1:] if last.startswith("s") and len(last) > 1 else None
    return str(sha) if sha else None


def operand_shapes(capsule):
    """size/stride of every tensor operand and the int scalars, in call order.

    A kernel op's geometry: geometry_key is a GEMM mapping ("unmapped" for a
    Triton kernel), and one kernel source launched on another problem size (a
    dynamic-shape xnumel) is kept apart.  Read from the packed metadata -- no
    operand bytes are touched.
    """
    out = []
    for a in capsule.get("args") or ():
        if isinstance(a, dict) and a.get("__op_nan_tensor__"):
            out.append("%s/%s" % ("x".join(str(s) for s in a.get("size") or ()) or "()",
                                  ",".join(str(s) for s in a.get("stride") or ())))
        elif isinstance(a, int):                      # sizes, strides, constexpr flags
            out.append(str(a))
        else:
            out.append("-")                           # a float is data, not geometry
    return " ".join(out) or "-"


def describe(capsule, path):
    """Pull the comparable fields out of a capsule and let the capsule go.

    Raises ValueError on anything this tool cannot compare -- the caller records
    the path and moves on.
    """
    if not isinstance(capsule, dict):
        raise ValueError("not a capsule: top level is %s, expected dict"
                         % type(capsule).__name__)

    fmt = capsule.get("format")
    if fmt not in KNOWN_FORMATS:
        raise ValueError("capsule format %r, this tool understands %s -- refusing "
                         "to guess at the layout"
                         % (fmt, sorted(KNOWN_FORMATS)))

    ident = capsule.get("identity")
    if not isinstance(ident, dict):
        raise ValueError("missing 'identity' dict")

    shas = capsule.get("arg_shas") or []
    dist = capsule.get("dist") or {}
    op = str(ident.get("op", capsule.get("op", "?")))
    src_sha = kernel_source(capsule, op)

    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        # -- primary key: the program point --------------------------------- #
        "op": op,
        "call_index": ident.get("call_index"),
        "geometry_key": str(ident.get("geometry_key", "unmapped")),
        "stack_digest": str(ident.get("stack_digest", "?")),
        "dtypes": tuple(str(d) for d in (ident.get("dtypes") or ())),
        # kernel ops only (None for ATen): the source hash stands in for the stack
        "kernel_src": src_sha,
        "operand_shapes": operand_shapes(capsule) if src_sha else None,
        # -- secondary key: the operands ------------------------------------ #
        "arg_shas": tuple(None if s is None else str(s) for s in shas),
        # -- reported, never grouped on ------------------------------------- #
        "out_nan": capsule.get("out_nan"),
        "out_inf": capsule.get("out_inf"),
        "out_numel": capsule.get("out_numel"),
        "out_size": tuple(capsule.get("out_size") or ()),
        "out_dtype": str(capsule.get("out_dtype")),
        "inputs_finite": capsule.get("inputs_finite"),
        # Reported, never grouped on: two ranks that hit the same program point
        # are the same site whether or not one of them also recorded ancestry.
        "chain_depth": len(capsule.get("chain") or ()) or None,
        "chain_truncation": sorted(set((capsule.get("truncation") or {}).values()))
                            or None,
        "arrived_via": (capsule.get("dist_causality") or {}).get("arrived_via"),
        "nan_origin": (capsule.get("nan_origin") or {}).get("class"),
        "rank": dist.get("rank"),
        "world_size": dist.get("world_size"),
        "hostname": dist.get("hostname"),
        "device_name": dist.get("device_name"),
        "geometry": capsule.get("geometry"),
    }


def collect(directory, recursive):
    if recursive:
        found = []
        for root, _dirs, files in os.walk(directory):
            found += [os.path.join(root, f) for f in files if f.endswith(".pt")]
        return sorted(found)
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.endswith(".pt") and os.path.isfile(os.path.join(directory, f)))


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #

def identity_key(rec, strict, job=False):
    """The primary grouping key.

    call_index is included only under --strict-identity; see the module docstring
    for why keying on a per-process counter across runs is worse than useless.
    --job (ranks of ONE distributed job) drops the geometry: under expert or pipeline
    parallelism the same program point runs on different shapes on every rank.

    A kernel op (kernel_source) puts its source hash where the stack digest goes
    and adds its operand shapes; see the module docstring.  An ATen op's key is
    unchanged: an ATen op is called from many places, and the stack is what tells
    them apart.
    """
    src = rec.get("kernel_src")
    return (rec["op"], rec["call_index"] if strict else None,
            "(any: --job)" if job else rec["geometry_key"],
            ("(any: kernel source %s)" % src) if src else rec["stack_digest"],
            rec["dtypes"],
            None if not src else "(any: --job)" if job else rec["operand_shapes"])


def group(records, strict, job=False):
    """site -> operand-sha -> [records], preserving a stable order."""
    groups = {}
    for r in records:
        groups.setdefault(identity_key(r, strict, job), {}).setdefault(
            r["arg_shas"], []).append(r)
    ordered = sorted(groups.items(),
                     # biggest first; ties broken on the key so runs are diffable
                     key=lambda kv: (-sum(len(v) for v in kv[1].values()),
                                     [str(x) for x in kv[0]]))
    return [(k, sorted(sub.items(), key=lambda kv: (-len(kv[1]),
                                                    [str(s) for s in kv[0]])))
            for k, sub in ordered]


def fmt_ranges(values):
    """[0,1,2,5] -> '0-2,5'.  Keeps an 8-rank group to one line."""
    nums = sorted(set(v for v in values if isinstance(v, int)))
    if not nums:
        return "?"
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else "%d-%d" % (start, prev))
        start = prev = n
    out.append(str(start) if start == prev else "%d-%d" % (start, prev))
    return ",".join(out)


def fmt_provenance(recs):
    hosts = sorted(set(r["hostname"] for r in recs if r["hostname"]))
    world = sorted(set(r["world_size"] for r in recs if r["world_size"] is not None))
    where = hosts[0] if len(hosts) == 1 else "%d hosts" % len(hosts) if hosts else "?"

    # world_size 1 everywhere means these are independent single-process runs,
    # not ranks of one job.  Saying "rank 0 on 8 hosts" would hide that.
    if world == [1]:
        return "%d independent single-process run(s) on %s" % (len(recs), where)

    ranks = fmt_ranges([r["rank"] for r in recs])
    s = "rank%s %s on %s" % ("s" if ("," in ranks or "-" in ranks) else "",
                             ranks, where)
    if world:
        s += "  (world_size %s)" % ",".join(str(w) for w in world)
    return s


def fmt_call_indices(recs, strict):
    vals = sorted(set(r["call_index"] for r in recs if r["call_index"] is not None))
    if not vals:
        return "-"
    if len(vals) == 1:
        return str(vals[0])
    note = "" if strict else "  (per-process counter; reported, not keyed)"
    return "%d distinct, %d..%d%s" % (len(vals), vals[0], vals[-1], note)


def abbrev(sha):
    if sha is None:
        return "-"
    return sha[:SHA_ABBREV]


def tag(i):
    """0->a, 25->z, 26->aa.  Must not run out: a sweep can have 59 input sets."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(ord("a") + r) + s
    return s


def fmt_operand_positions(members):
    """Per argument position: shared by everyone, or one value per capsule?

    This is the question the digests were added for.  Within one run, an operand
    whose sha is identical on every rank is a replicated weight; one that differs
    per rank is an activation.  It is easy to lose in a long list of sha tuples,
    so state it directly.
    """
    n = len(members)
    width = max((len(m["arg_shas"]) for m in members), default=0)
    out = []
    for i in range(width):
        vals = set(m["arg_shas"][i] if i < len(m["arg_shas"]) else None
                   for m in members)
        if vals == {None}:
            continue                                  # not a tensor argument
        k = len(vals)
        if k == 1 and n > 1:
            out.append("arg%d shared by all %d (replicated)" % (i, n))
        elif k == n and n > 1:
            out.append("arg%d all %d distinct" % (i, n))
        else:
            out.append("arg%d %d distinct" % (i, k))
    return " | ".join(out) or "-"


def spread(vals):
    vals = [v for v in vals if isinstance(v, int)]
    if not vals:
        return "?"
    lo, hi = min(vals), max(vals)
    return str(lo) if lo == hi else "%d..%d" % (lo, hi)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

# Derived from this file, not hardcoded: the "replay:" line below is meant to be
# copy-pasteable, and a hardcoded path stops being that the moment the repo is
# cloned anywhere else.  replay_nan_op.py is this script's sibling by
# construction -- they import each other's formats.
REPLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "replay_nan_op.py")


def report(grouped, records, failures, found, limit_names, strict,
           out=sys.stdout):
    w = out.write
    what = "op identit" + ("y" if len(grouped) == 1 else "ies") if strict \
        else "op site" + ("" if len(grouped) == 1 else "s")
    w("\n%d capsule(s) found, %d parsed, %d failed -> %d distinct %s\n"
      % (found, len(records), len(failures), len(grouped), what))

    for i, (key, subgroups) in enumerate(grouped, 1):
        op, _call_index, geometry_key, stack_digest, dtypes, shapes = key
        members = [r for _sha, recs in subgroups for r in recs]
        w("\n" + "=" * 78 + "\n")
        w("GROUP %d -- %d capsule(s)\n" % (i, len(members)))
        # Name a concrete capsule up front.  Everything else in the block is an
        # aggregate; this is the one line that says "open this file".
        w("  example      : %s\n" % members[0]["name"])
        w("  op           : %s\n" % op)
        w("  geometry     : %s\n" % geometry_key)
        if shapes is not None:
            w("  operands     : %s\n" % shapes)
            digests = sorted(set(r["stack_digest"] for r in members))
            w("  stack digest : %s  (%s; not keyed: the source hash names the "
              "site)\n"
              % (stack_digest, digests[0] if len(digests) == 1
                 else "%d distinct, e.g. %s" % (len(digests), digests[0])))
        else:
            w("  stack digest : %s\n" % stack_digest)
        w("  arg dtypes   : %s\n" % (", ".join(dtypes) or "-"))
        w("  call index   : %s\n" % fmt_call_indices(members, strict))
        w("  captured by  : %s\n" % fmt_provenance(members))

        if len(subgroups) == 1 and len(members) > 1:
            inputs = "all %d saw identical inputs" % len(members)
        elif len(subgroups) == len(members) and len(members) > 1:
            inputs = "every capsule saw different inputs"
        elif len(members) == 1:
            inputs = "single capsule"
        else:
            inputs = "%d capsules over %d input sets" % (len(members), len(subgroups))
        w("  operand sets : %d  (%s)\n" % (len(subgroups), inputs))
        w("  by position  : %s\n" % fmt_operand_positions(members))
        # Aggregate first: with one input set per capsule the per-set listing is
        # as long as the group, and the useful signal is the spread.
        finite = sorted(set(str(r["inputs_finite"]) for r in members))
        w("        out_nan %s, out_inf %s of %s elems   inputs_finite=%s\n"
          % (spread([r["out_nan"] for r in members]),
             spread([r["out_inf"] for r in members]),
             spread([r["out_numel"] for r in members]), ",".join(finite)))

        shown_sets = subgroups[:limit_names]
        for j, (shas, recs) in enumerate(shown_sets):
            # Name one capsule per input set too, so every row in the report
            # points at a file you can actually replay.
            w("    [%s] %3d capsule(s)  e.g. %s\n"
              % (tag(j), len(recs), recs[0]["name"]))
            w("        shas=%s\n" % (" ".join(abbrev(s) for s in shas) or "-"))
            outcomes = sorted(set((r["out_nan"], r["out_inf"], r["out_numel"],
                                   r["inputs_finite"]) for r in recs))
            for nan, inf, numel, fin in outcomes:
                w("        out_nan=%s out_inf=%s of %s elems   inputs_finite=%s\n"
                  % (nan, inf, numel, fin))
            if len(outcomes) > 1:
                w("        ^ %d distinct outcomes from IDENTICAL inputs -- "
                  "nondeterministic kernel\n" % len(outcomes))
        if len(subgroups) > len(shown_sets):
            w("    ... +%d more input set(s) not shown (--limit-names %d for all)\n"
              % (len(subgroups) - len(shown_sets), len(subgroups)))

        shown = members[:limit_names]
        w("  members      : %s\n" % ", ".join(r["name"] for r in shown))
        if len(members) > len(shown):
            w("                 (+%d more not shown; re-run with --limit-names %d "
              "to see all)\n" % (len(members) - len(shown), len(members)))
        w("  replay:  python3 %s %s\n" % (REPLAY, members[0]["path"]))

    if failures:
        w("\n" + "=" * 78 + "\n")
        w("FAILED TO PARSE -- %d capsule(s) excluded from every group above\n"
          % len(failures))
        for f in failures:
            w("  %s\n      %s\n" % (f["path"], f["error"]))

    w("\n")


def job_report(directory, records, recursive, out=sys.stdout):
    """--job: the manifest's verdict and per-rank lines, then each capsule's arrival."""
    w = out.write
    mans = []
    for root, _dirs, files in (os.walk(directory) if recursive
                               else [(directory, [], os.listdir(directory))]):
        mans += [os.path.join(root, f) for f in files
                 if f.startswith("manifest") and f.endswith(".json")]
    for m in sorted(mans):
        try:
            with open(m) as fh:
                doc = json.load(fh)
        except Exception as e:                          # noqa: BLE001
            w("manifest %s unreadable: %s\n" % (m, e))
            continue
        v = doc.get("verdict") or {}
        w("\nJOB MANIFEST %s  (world size %s)\n" % (m, doc.get("world_size")))
        w("  verdict : %s\n" % v.get("note"))
        for r in doc.get("ranks") or []:
            c = r.get("capsule") or {}
            fa = r.get("first_arrival")
            w("  rank %-3s %-12s step %-4s %-9s %s%s\n"
              % (r.get("rank"), r.get("host"), r.get("step"),
                 "CAPTURED" if r.get("captured") else ("nan-seen" if r.get("nan_seen") else "clean"),
                 "%s inputs_finite=%s origin=%s" % (c.get("op"), c.get("inputs_finite"),
                                                    c.get("origin")) if c else "",
                 "  first NaN arrived via %s #%s of group %s (step %s)"
                 % (fa.get("op"), fa.get("seq"), fa.get("group"), fa.get("step")) if fa else ""))
    if not mans:
        w("\nno manifest*.json in %s (written at sync_stop by rank 0)\n" % directory)
    arr = [r for r in records if r.get("arrived_via")]
    if arr:
        w("  capsules whose NaN arrived through a collective (not made on that rank):\n")
        for r in arr:
            a = r["arrived_via"]
            w("    rank %s %s <- %s #%s of group %s\n"
              % (r["rank"], r["op"], a.get("op"), a.get("seq"), a.get("group")))


def to_json(grouped, failures, found, records, strict):
    return {
        "capsules_found": found,
        "capsules_parsed": len(records),
        "capsules_failed": len(failures),
        "strict_identity": strict,
        "identity_groups": len(grouped),
        "groups": [
            {
                # A kernel op's stack_digest stays a real digest (the representative's:
                # result_card / compare_results read it as one) and the key's
                # "(any: kernel source ...)" is `site`; ATen groups are unchanged.
                "identity": dict({"op": k[0], "geometry_key": k[2],
                                  "stack_digest": k[3], "dtypes": list(k[4]),
                                  "call_index": k[1]},
                                 **({} if k[5] is None else {
                                     "stack_digest": subs[0][1][0]["stack_digest"],
                                     "stack_digests": sorted(set(
                                         r["stack_digest"] for _s, recs in subs
                                         for r in recs)),
                                     "site": k[3], "operand_shapes": k[5]})),
                "call_indices": sorted(set(
                    r["call_index"] for _s, recs in subs for r in recs
                    if r["call_index"] is not None)),
                "count": sum(len(r) for _s, r in subs),
                "geometry": subs[0][1][0]["geometry"],
                "representative": subs[0][1][0]["path"],
                "operand_sets": [
                    {
                        "arg_shas": list(shas),
                        "count": len(recs),
                        "outcomes": [{"out_nan": r["out_nan"], "out_inf": r["out_inf"],
                                      "out_numel": r["out_numel"],
                                      "inputs_finite": r["inputs_finite"]}
                                     for r in recs[:1]],
                        "members": [r["path"] for r in recs],
                        # per member, same order: what multi-rank selection needs
                        "member_info": [{"rank": r["rank"], "nan_origin": r.get("nan_origin"),
                                         "arrived": bool(r.get("arrived_via"))} for r in recs],
                        "ranks": sorted(set(r["rank"] for r in recs
                                            if r["rank"] is not None)),
                        "hosts": sorted(set(r["hostname"] for r in recs
                                            if r["hostname"])),
                    }
                    for shas, recs in subs
                ],
            }
            for k, subs in grouped
        ],
        "failures": failures,
    }


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory holding the .pt capsules")
    ap.add_argument("--json", metavar="OUT", help="also write the grouping as JSON")
    ap.add_argument("--recursive", action="store_true",
                    help="descend into subdirectories (default: top level only)")
    ap.add_argument("--limit-names", type=int, default=5, metavar="N",
                    help="member filenames to print per group (default 5)")
    ap.add_argument("--job", action="store_true",
                    help="the capsules are the ranks of ONE distributed job: group "
                         "without geometry (EP/PP ranks run one site on different "
                         "shapes), show each capsule's arrival through a collective, "
                         "and print the job manifest(s) (manifest*.json) if present")
    ap.add_argument("--strict-identity", action="store_true",
                    help="include the per-process op call index in the grouping "
                         "key.  Correct when comparing RANKS OF ONE RUN; across "
                         "independent runs it splits every capsule into its own "
                         "group (see the module docstring)")
    args = ap.parse_args(argv)

    # `compare_nan_capsules.py DIR | head` is the normal way to skim a big sweep;
    # die on SIGPIPE like every other unix filter instead of dumping a traceback.
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass                                          # not posix, or not main thread

    d = args.directory
    if not os.path.isdir(d):
        print("not a directory: %s" % d, file=sys.stderr)
        return 2

    paths = collect(d, args.recursive)
    if not paths:
        print("no .pt capsules in %s%s"
              % (d, "" if args.recursive else " (try --recursive)"),
              file=sys.stderr)
        return 2

    records, failures = [], []
    for p in paths:
        try:
            # Load, extract, drop.  A capsule may be 2 GB (max_bytes in the
            # capture is 2 << 30); nothing here may accumulate operand bytes.
            capsule = load_capsule(p)
            try:
                records.append(describe(capsule, p))
            finally:
                del capsule
        except Exception as e:                        # per-file fatal, never fatal
            failures.append({"path": os.path.abspath(p),
                             "error": "%s: %s" % (type(e).__name__, e),
                             "traceback": traceback.format_exc(limit=3)})

    grouped = group(records, args.strict_identity, args.job)
    if args.job:
        job_report(d, records, args.recursive)
    report(grouped, records, failures, len(paths), args.limit_names,
           args.strict_identity)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(to_json(grouped, failures, len(paths), records,
                              args.strict_identity), fh, indent=2)
        print("wrote %s" % os.path.abspath(args.json))

    if not records:
        print("every capsule failed to parse", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    # sys.exit(code) does NOT work here.  In the nan-repro image
    # (torch 2.12.0a0+gitf68b851) merely importing torch installs a shutdown
    # handler that swallows the status, and every run exits 0:
    #
    #     $ python3 -c 'import sys;        sys.exit(2)'; echo $?   -> 2
    #     $ python3 -c 'import sys, torch; sys.exit(2)'; echo $?   -> 0
    #
    # Scripts that only ever return 0 never notice.  This one has an exit-code
    # contract, so bypass the handlers -- and flush first, because os._exit does
    # not.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
