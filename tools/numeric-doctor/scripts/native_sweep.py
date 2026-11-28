#!/usr/bin/env python3
"""debug-numeric step 7, recipe track (b) for a NATIVE reproducer: the mutation sweep with a command
as its trial instead of a capsule replay.

The capsule sweep (replay_nan_op.py --mutate) replays one op from a capsule; a step-5N reproducer
has no capsule -- it is a script that runs the suspect computation and ends `NATIVE_RESULT=<hits>/<runs>`
(step5n-native.md N3).  This drives the same kernel_patch targets, overlays and kernel_mutations
registry, and scores each trial from the command's own result line.

    # 1. which kernels does the suspect call load / dispatch?
    python3 native_sweep.py --discover --cmd 'python3 repro.py' [--match <symbol fragment>]
    # 2. list the mutations for one of them, then sweep (one GPU; the command carries the load)
    python3 native_sweep.py --cmd 'python3 repro.py --iters 2000' --patch-hsaco <.co path> --list
    python3 native_sweep.py --cmd 'COPIES=2 NGPUS=1 bash native_run.sh repro.py --iters 2000' \\
        --patch-hsaco <.co path> --mutate-only vmcnt0 --mutate-only lgkmcnt0 \\
        --report <ART>/rootcause/native_mutations.json --asm-dir <ART>/asm --keep-fix <ART>/fix_work/kept

Target (one): --patch-hsaco <aiter .co> (as aiter's load log names it), --patch-co <dumped code
object> [--patch-co-symbol S] (CK / Triton / anything the HSA shim intercepts: dump it with
--discover --match), --patch-kernel <Tensile kernel name> (hipBLASLt; the AMD_LOG_LEVEL ShaderName
--discover prints).

Discovery runs the command once with NATIVE_DISCOVER=1 -- the candidate's contract: run the
suspect call ONCE, no reference, no checks (its reference's own GEMMs would otherwise be named:
2026-11-20) -- and AITER_LOG_LEVEL=INFO, AMD_LOG_LEVEL=3, plus the HSA shim's dump with --match.

Result line per trial: `NATIVE_TOTAL=h/n` (native_run.sh) if present, else the sum of every
`NATIVE_RESULT=h/n`.  Outcome: NAN (hits), CLEAN, UNDERPOWERED (a clean row below 3 expected hits
at the baseline rate), VOID (the patched kernel was not served), TIMEOUT, ERROR.  The baseline (the
command, stock) must hit first.  The report is rewritten after every trial; a re-run with the same
--cmd resumes it.  Last line NATIVE_SWEEP_RESULT=<CLEAN rows or none>.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_mutations as km  # noqa: E402
import kernel_patch as kp      # noqa: E402

MIN_EXPECTED_HITS = 3.0


def run(cmd, env, timeout):
    """(stdout+stderr, rc, seconds, timed_out)."""
    t0 = time.time()
    try:
        p = subprocess.run(["bash", "-c", cmd], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout,
                           start_new_session=True)
        return p.stdout, p.returncode, time.time() - t0, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return out, None, time.time() - t0, True


def result_of(text):
    """(hits, runs) from NATIVE_TOTAL= (native_run.sh) or the sum of NATIVE_RESULT= lines, or None."""
    m = re.findall(r"NATIVE_TOTAL=(\d+)/(\d+)", text)
    if m:
        return int(m[-1][0]), int(m[-1][1])
    m = re.findall(r"NATIVE_RESULT=(\d+)/(\d+)", text)
    if m:
        return sum(int(h) for h, _ in m), sum(int(n) for _, n in m)
    return None


def discover(a):
    env = dict(os.environ, NATIVE_DISCOVER="1", AITER_LOG_LEVEL="INFO", AMD_LOG_LEVEL="3")
    work = a.workdir or tempfile.mkdtemp(prefix="native_sweep_")
    dump = None
    if a.match:
        dump = os.path.join(work, "dump")
        os.makedirs(dump, exist_ok=True)
        shim = kp.build_hsa_shim(os.path.join(work, "hsa_patch_shim.so"))
        pre = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = (pre + " " + shim).strip() if pre else shim
        env.update(NUMERIC_DOCTOR_HSA_DUMP=dump, NUMERIC_DOCTOR_HSA_MATCH=a.match,
                   NUMERIC_DOCTOR_HSA_LOG=os.path.join(work, "discover_shim.log"))
    kp.ensure_aiter_asm_dir(a.patch_asm_dir)
    out, rc, secs, to = run(a.cmd, env, a.timeout or 1800)
    print("discovery run: rc=%s %.0fs%s (NATIVE_DISCOVER=1: the candidate runs its suspect call once)"
          % (rc, secs, " TIMEOUT" if to else ""))
    loads = list(dict.fromkeys(kp.parse_aiter_loads(out)))
    print("\naiter code objects loaded (--patch-hsaco):" if loads else "\naiter: none file-backed")
    for path, sym in loads:
        print("  %s  (%s)" % (path, sym[:60]))
    names = sorted(set(m.strip() for m in re.findall(r"ShaderName\s*:\s*(.+)$", out, re.M)))
    print("\nkernels dispatched (AMD_LOG_LEVEL=3 ShaderName; a Cijk_... name is --patch-kernel):")
    for n in names[:40]:
        print("  %s" % n)          # whole: a Tensile name is the --patch-kernel argument verbatim
    if len(names) > 40:
        print("  ... %d more" % (len(names) - 40))
    if dump:
        cos = sorted(f for f in os.listdir(dump) if f.endswith(".co"))
        print("\ncode objects dumped for --match %r (--patch-co):" % a.match)
        for f in cos:
            print("  %s" % os.path.join(dump, f))
    if not (loads or names):
        print("\n(nothing named: does the candidate honour NATIVE_DISCOVER=1? its last lines:)")
        print("\n".join(out.splitlines()[-8:]))
    return 0


def target_and_cache(a, work):
    os.makedirs(os.path.join(work, "overlay"), exist_ok=True)
    if a.patch_hsaco:
        kp.ensure_aiter_asm_dir(a.patch_asm_dir)
        t = kp.AiterTarget(a.patch_hsaco, a.patch_asm_dir, a.patch_arch, work)
        return t, kp.AiterOverlay(t, os.path.join(work, "overlay")), "aiter"
    if a.patch_co:
        t = kp.HsaTarget(a.patch_co, a.patch_co_symbol, a.patch_arch, work)
        return t, kp.HsaOverlay(t, os.path.join(work, "overlay")), "hsa"
    t = kp.PatchTarget(a.patch_kernel, a.patch_libdir, a.patch_arch, work)
    return t, kp.OverlayCache(t, os.path.join(work, "overlay")), "tensile"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--cmd", required=True, help="the trial: a bash command ending NATIVE_RESULT/TOTAL")
    p.add_argument("--discover", action="store_true")
    p.add_argument("--match", help="--discover: dump code objects whose symbols contain this (HSA shim)")
    p.add_argument("--patch-hsaco")
    p.add_argument("--patch-asm-dir")
    p.add_argument("--patch-co")
    p.add_argument("--patch-co-symbol")
    p.add_argument("--patch-kernel")
    p.add_argument("--patch-libdir")
    p.add_argument("--patch-arch")
    p.add_argument("--mutate-only", action="append", default=[])
    p.add_argument("--mutate-no-identity", action="store_true",
                   help="drop the identity control (otherwise it runs first)")
    p.add_argument("--list", action="store_true", help="enumerate only")
    p.add_argument("--report", help="JSON report (resumable)")
    p.add_argument("--asm-dir", help="write <row>.diff (before/after of the changed instructions)")
    p.add_argument("--keep-fix", help="export each CLEAN row's patched kernel + env.sh here")
    p.add_argument("--timeout", type=float, help="per trial, s (default 3 x baseline + 120)")
    p.add_argument("--workdir")
    a = p.parse_args()
    if a.discover:
        return discover(a)
    if sum(bool(x) for x in (a.patch_hsaco, a.patch_co, a.patch_kernel)) != 1:
        sys.exit("native_sweep: give exactly one of --patch-hsaco / --patch-co / --patch-kernel "
                 "(--discover names them)")
    work = a.workdir or tempfile.mkdtemp(prefix="native_sweep_")
    os.makedirs(work, exist_ok=True)
    target, cache, kind = target_and_cache(a, work)
    only = list(a.mutate_only)
    if only and "identity" not in only and not a.mutate_no_identity:
        only.append("identity")
    rows = []
    for m in km.kernel_mutations(km.MutationSpec.from_target(target), only=only or None):
        if a.mutate_no_identity and m.name == "identity":
            continue
        try:
            sites, err = target.select(m.selector, m.opts), None
        except kp.PatchError as e:
            sites, err = [], str(e)
        rows.append((m, sites, err))
    print("target: %s (%s), %d instructions; %d mutation(s)" % (getattr(target, "kernel", "?"), kind,
                                                               len(target.insns), len(rows)))
    for m, sites, err in rows:
        print("  %-32s %5s  %s" % (m.name, "FAIL" if err else len(sites), (err or m.description)[:120]))
    if a.list:
        return 0

    report = {"cmd": a.cmd, "target": str(getattr(target, "kernel", "")), "kind": kind, "rows": []}
    if a.report and os.path.exists(a.report):
        old = json.load(open(a.report))
        if old.get("cmd") == a.cmd and old.get("target") == report["target"]:
            report = old
            print("resuming %s: %d row(s) done" % (a.report, len(old["rows"])))
    done = {r["name"] for r in report["rows"]}

    def save():
        if a.report:
            os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
            json.dump(report, open(a.report, "w"), indent=1)

    base_env = dict(os.environ, AITER_LOG_LEVEL="INFO")
    if "baseline" not in report:
        out, rc, secs, to = run(a.cmd, base_env, a.timeout or 4 * 3600)
        r = result_of(out)
        report["baseline"] = {"hits": r[0] if r else None, "runs": r[1] if r else None,
                              "seconds": round(secs, 1), "rc": rc}
        save()
    b = report["baseline"]
    print("\nbaseline (stock): %s/%s in %.0fs" % (b["hits"], b["runs"], b["seconds"]))
    if not b["hits"]:
        print("NATIVE_SWEEP_RESULT=NO_BASELINE  (the command does not reproduce stock: nothing to sweep)")
        return 1
    rate = b["hits"] / float(b["runs"])
    tmo = a.timeout or 3 * b["seconds"] + 120

    for m, sites, err in rows:
        if m.name in done:
            continue
        rec = {"name": m.name, "description": m.description, "sites": len(sites),
               "addrs": ["0x%X" % s.addr for s in sites[:32]], "outcome": "ERROR", "hits": None,
               "runs": None, "expected_hits": None, "served": None, "seconds": 0, "error": err,
               "changed_insns": None}
        if not err:
            try:
                cache.apply(target.offsets_for(sites, m.payload))
            except Exception as e:                       # noqa: BLE001 -- per trial
                rec["error"] = "%s: %s" % (type(e).__name__, e)
            else:
                env = cache.env(base_env)
                out, rc, secs, to = run(a.cmd, env, tmo)
                rec["seconds"] = round(secs, 1)
                logd = os.path.join(work, "trials")
                os.makedirs(logd, exist_ok=True)
                rec["log"] = os.path.join(logd, m.name.replace("@", "-") + ".log")
                with open(rec["log"], "w") as f:      # the whole trial output, for an ERROR row
                    f.write(out)
                res = result_of(out)
                served = kp.served_overlay(out, cache.overlay, aiter=(kind == "aiter"),
                                           shim_log=cache.log_text() if kind == "hsa" else None)
                rec["served"] = served
                if to:
                    rec["outcome"] = "TIMEOUT"
                elif res is None:
                    rec["error"] = "no NATIVE_RESULT/NATIVE_TOTAL line (rc=%s): %s" % (
                        rc, " | ".join(out.splitlines()[-3:])[:200])
                elif served is False:
                    rec["outcome"] = "VOID"
                    rec["error"] = "the patched kernel was not served: the trial ran stock"
                else:
                    rec["hits"], rec["runs"] = res
                    rec["expected_hits"] = round(rate * res[1], 1)
                    rec["outcome"] = ("NAN" if res[0] else
                                      "CLEAN" if rec["expected_hits"] >= MIN_EXPECTED_HITS
                                      else "UNDERPOWERED")
                try:
                    diff = kp.diff_disasm(target.insns, cache.disassemble())
                    rec["changed_insns"] = len(diff)
                    if a.asm_dir:
                        os.makedirs(a.asm_dir, exist_ok=True)
                        with open(os.path.join(a.asm_dir, "%s.native.diff" % m.name.replace("@", "-")), "w") as f:
                            f.write("# %s -- %s\n" % (m.name, m.description))
                            for addr, x, y in diff:
                                f.write("  0x%X  %s  ->  %s\n" % (addr, x, y))   # diff_disasm: texts
                except Exception as e:                   # noqa: BLE001 -- the read-back is evidence, not the trial
                    rec["changed_insns"] = "read-back failed: %s" % e
                if rec["outcome"] == "CLEAN" and a.keep_fix:
                    try:
                        kp.export_fix(cache, os.path.join(a.keep_fix, m.name.replace("@", "-")))
                    except Exception as e:               # noqa: BLE001
                        rec["error"] = "keep-fix: %s" % e
        report["rows"].append(rec)
        save()
        print("  %-32s %-12s %8s  exp %-7s served=%-5s %5.0fs  %s"
              % (m.name, rec["outcome"], "%s/%s" % (rec["hits"], rec["runs"]), rec["expected_hits"],
                 rec["served"], rec["seconds"], (rec["error"] or "")[:80]))
    clean = [r["name"] for r in report["rows"] if r["outcome"] == "CLEAN"]
    ident = [r for r in report["rows"] if r["name"] == "identity"]
    if ident and ident[0]["outcome"] != "NAN":
        print("!! the identity control did not reproduce (%s): every CLEAN row is suspect" % ident[0]["outcome"])
    elif ident and ident[0]["runs"] and ident[0]["hits"] / float(ident[0]["runs"]) < 0.5 * rate:
        # the overlay path itself can shift the rate (2026-11-21: stock 365/400, identity 137/400):
        # judge a CLEAN row's power against the identity row's rate, not only the stock baseline
        r = ident[0]["hits"] / float(ident[0]["runs"])
        weak = [x["name"] for x in report["rows"] if x["outcome"] == "CLEAN" and r * x["runs"] < MIN_EXPECTED_HITS]
        print("note: identity hit at %.3g per run vs the stock baseline's %.3g -- the overlay shifts the rate;"
              " at the identity rate %s" % (r, rate, "these CLEAN rows are UNDERPOWERED: %s" % ", ".join(weak)
                                            if weak else "every CLEAN row still has >= 3 expected hits"))
    print("NATIVE_SWEEP_RESULT=%s" % (",".join(clean) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
