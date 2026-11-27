#!/usr/bin/env python3
"""debug-numeric step 9: the investigation's result as data -- <ARTIFACT_DIR>/result.json.

    python3 scripts/result_card.py draft <ARTIFACT_DIR>     # facts from disk, TODO for the rest
    python3 scripts/result_card.py check <ARTIFACT_DIR>     # no TODO left, every enum valid
    python3 scripts/compare_results.py <ART_A> <ART_B>      # same result? (two runs, two skill versions)

report.md is prose: two runs of the same example could not be told apart from it -- the root
cause and fix live in ROOT_CAUSE.md / fix/README.md as free text, and hosts, job ids, hit counts
and the call site G1 happened to capture differ on every run (2026-11-20: a 3-site and a 14-site
patch of the same kernel were visible only in quoted ledger lines).  The card holds what makes
two results the same or not, in fixed spellings:

  bug       the op, its problem shape, the kernel (hipBLASLt solution index + name, or the
            Inductor kernel), the verdict, how it was reproduced                    [from disk]
  root_cause status, component, location (kernel / instruction sites / env / source /
            config), the necessary conditions, one line of mechanism, the defence     [the agent]
  fix       status, kind, the primary change set, extra defence, validation numbers  [the agent]
  context   NaN kind and origin, priors, rates, groups, call sites -- reported by the compare,
            never judged (they differ run to run)                                     [from disk]

`draft` fills every fact it can find and writes "TODO" where judgement is needed; it keeps the
agent's answers of an existing result.json (re-run it after a late step: the facts refresh).
Spellings the compare relies on (step9-report.md *Result card*):
  sites     instruction addresses, hex: "0x2A8D9D0" (case and leading zeros are normalised)
  env       "VAR=VALUE";  config "key=value" (numbers compare as numbers: 1e-06 == 1e-6)
  source    "file.py:function" -- no line numbers (they move between versions)
  changes   what the fix sets, in those same spellings: a site list for a binary patch
            ("lgkmcnt0@0x2A8D9D0"; a hand edit of an instruction as the instruction it becomes:
            "lgkmcnt(1)@0x1A2C"), "VAR=VALUE", "key=value", or "file.py:function" for a source edit
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TODO = "TODO"
SCHEMA = 1
ENUMS = {
    ("root_cause", "status"): {"FOUND", "NO_ROOT_CAUSE", "SKIPPED"},
    ("root_cause", "component"): {"hipblaslt", "rocblas", "triton", "inductor", "aiter", "ck", "miopen",
                                  "rccl", "pytorch", "framework", "model_code", "model_config", "data",
                                  "hardware", "unknown"},
    ("fix", "status"): {"FIX", "WORKAROUND", "NO_FIX", "SKIPPED"},
    ("fix", "kind"): {"binary_patch", "library_env", "compiler_override", "source_patch", "config_change",
                      "input_validation", "none"},
}
AGENT_FIELDS = {
    "root_cause": {"status": TODO, "component": TODO, "mechanism": TODO, "defence": TODO,
                   "location": {"kernel": TODO, "sites": [], "env": [], "source": [], "config": []},
                   "conditions": [TODO]},
    "fix": {"status": TODO, "kind": TODO, "changes": [TODO], "extra": [], "alternatives": [],
            "validated": {"reproducer": TODO, "model": TODO}},
}


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def last(pattern, lines, group=1):
    for l in reversed(lines):
        m = re.search(pattern, l)
        if m:
            return m.group(group)
    return None


def facts(A):
    """Everything the card can say from the artifact dir alone."""
    led = read(os.path.join(A, "ledger.md")).splitlines()
    f = {"schema": SCHEMA, "artifact_dir": os.path.basename(A.rstrip("/"))}
    f["run_script"] = last(r"\bRUN_SCRIPT=(\S+)", led)
    f["nan_signal"] = last(r"\bNAN_SIGNAL='([^']*)'", led) or last(r"\bNAN_SIGNAL=(\S+)", led)
    pr = last(r"\bPRIORS=(use|none)\b", led)
    f["priors"] = pr or ("none" if any("prior: skipped" in l for l in led) else "use")
    # the skill version that RAN (step 0 prints SKILL_COMMIT=, step 1 ledgers it); the card's own
    # commit is only when it was drafted -- a re-drafted old run would claim today's skill
    f["skill_commit"] = last(r"\bSKILL_COMMIT=(\w+)", led)
    try:
        f["skill_commit_at_card"] = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip() or None
    except OSError:
        f["skill_commit_at_card"] = None

    c2r = last(r"(CAPTURE_TO_REPRO_RESULT=.*)", led) or ""
    repro = {"result": (re.search(r"CAPTURE_TO_REPRO_RESULT=(\S+)", c2r) or [None, None])[1],
             "group": None, "concurrency": None, "kernel_capsule": None}
    m = re.search(r"CAPTURE_TO_REPRO_RESULT=REPRODUCED (G\d+) concurrency=(\d+)", c2r)
    if m:
        repro["group"], repro["concurrency"] = m.group(1), int(m.group(2))
    # a COLLECTIVE group (synchronous collective: the member ranks' capsules replayed together)
    m = re.search(r"CAPTURE_TO_REPRO_RESULT=REPRODUCED (G\d+) collective ranks=(\d+)", c2r)
    if m:
        repro["group"], repro["collective_ranks"] = m.group(1), int(m.group(2))
        repro["collective_verdict"] = (re.search(r"\b(COLLECTIVE_FAULT|INPUT_NONFINITE|CORRECT)\b", c2r) or [None, None])[1]
        repro["collective_hint"] = (re.search(r"NARROW_INTERMEDIATE\? (\S+)", c2r) or [None, None])[1]
    m = re.search(r"KERNEL_CAPSULE=(\S+).*?\bkernel=(\S+)", c2r)
    if m:
        repro["kernel_capsule"] = {"result": m.group(1), "kernel": m.group(2)}
    proposed = (re.search(r"proposed=(\S+)", c2r) or [None, None])[1]
    native = last(r"(?:VERDICT|verdict) (NATIVE_\w+)", led)

    # groups: op + problem shape are the bug's identity; count and stack are this run's sampling
    groups = []
    gj = {}
    try:
        gj = json.load(open(os.path.join(A, "groups.json")))
    except (OSError, ValueError):
        pass
    pri = [l.split() for l in read(os.path.join(A, "groups_priority.txt")).splitlines() if l.strip()]
    ig = gj.get("groups") or []
    by_rep = {g.get("representative"): g for g in ig}
    for row in pri:
        g = by_rep.get(row[2]) if len(row) > 2 else None
        ident = (g or {}).get("identity", {})
        groups.append({"group": row[0], "count": int(row[1]) if row[1].isdigit() else None,
                       "op": ident.get("op") or (row[3] if len(row) > 3 else None),
                       # "unmapped" (not a GEMM) and "(any: --job)" (a grouping label) are no shape
                       "geometry": (ident.get("geometry_key")
                                    if not re.match(r"unmapped$|\(", ident.get("geometry_key") or "(") else None),
                       "stack_digest": ident.get("stack_digest")})

    kern = None
    for k in sorted(glob.glob(os.path.join(A, "replay", "*", "kernel.*.txt")) +
                    glob.glob(os.path.join(A, "verify-*", "*", "out", "kernel.*.txt"))):
        kf = dict(re.match(r"(\w+):\s*(.*)", l).groups() for l in read(k).splitlines() if re.match(r"\w+:", l))
        kf = {k: v for k, v in kf.items() if not re.match(r"\?(\s|$)|$", v.strip())}   # "kernel: ? (not identified ...)"
        if kf.get("solution_index") or kf.get("kernel"):
            si = re.match(r"\d+", kf.get("solution_index") or "")   # "13817 (the wrong group's: rows 12889)"
            kname = ((kf.get("kernel") or "").split() or [None])[0]
            # replay_verdict.py kernel: a Tensile name, an aiter .co path ("(aiter .co)"), or an
            # HSA-intercepted symbol -- the draft called every one of them hipblaslt (2026-11-20)
            lib = ("aiter" if "aiter .co" in (kf.get("kernel") or "") or (kname or "").endswith(".co")
                   else "ck" if "HSA-intercepted" in (kf.get("kernel") or "") else "hipblaslt")
            if lib == "aiter" and kname and "/hsa/" in kname:
                kname = kname.split("/hsa/", 1)[1].lstrip("/")    # gfx950/mla/<name>.co
            kern = {"library": lib, "solution_index": si.group(0) if si else None,
                    "name": kname,
                    "arch": ((kf.get("arch") or "").split() or [None])[0],
                    "injected": (kf.get("injected") or "").split(" --")[0] or None}
            break
    # the compiled track's kernel capsule (C1): from the capture job's line, else the file the
    # nan_asserts hook saved (runs before 2026-11-20 replayed it by hand)
    ind = repro["kernel_capsule"] and repro["kernel_capsule"]["kernel"]
    if not ind:
        caps = sorted(glob.glob(os.path.join(A, "capsules_inductor", "inductor_*.pt")))
        ind = re.sub(r"^inductor_\d+_|\.pt$", "", os.path.basename(caps[0])) if caps else None
    if kern is None and ind:
        kern = {"library": "inductor", "solution_index": None, "name": ind, "arch": None, "injected": None}

    primary = next((g for g in groups if g["group"] == repro["group"]), None)
    pop = (primary and primary["op"]) or ""
    if kern is None and pop.startswith(("inductor.", "triton.")):
        # the capture's own kernel op (inductor.<kernel>.s<hash> / triton.<kernel>.<variant>): the
        # kernel is the bug's identity, as the C1 kernel capsule's is
        kern = {"library": pop.split(".")[0], "solution_index": None, "name": pop.split(".")[1],
                "arch": None, "injected": None}
    if repro["result"] == "REPRODUCED" and repro.get("collective_ranks"):
        by = "collective_capsules"      # every member rank's capsule of one collective call, replayed together
    elif repro["result"] == "REPRODUCED":
        by = "op_capsule"
    elif ind:
        by = "kernel_capsule"           # the op groups only carried the NaN: the kernel is the bug
    elif native:
        by = "native"                   # step 5N's standalone reproducer (was "not_reproduced", 2026-11-23)
    else:
        by = (repro["result"] or "unknown").lower()
    if by == "collective_capsules" and kern is None:
        kern = {"library": "framework", "solution_index": None, "name": primary and primary["op"],
                "arch": None, "injected": None}
    if kern is not None and not kern.get("arch"):
        # every kernel ran on the investigation's GPU: the preflight / gpu_check line's ARCH= (only the
        # ladder's kernel.*.txt set it -- an Inductor / Triton / kernel-capsule card left it null, 2026-11-23)
        kern["arch"] = last(r"\bARCH=(gfx\w+)", led)
    f["bug"] = {"verdict": native or proposed or repro.get("collective_verdict"), "op": primary and primary["op"],
                "geometry": primary and primary["geometry"], "kernel": kern, "reproduced_by": by}
    f["context"] = {
        "nan_kind": last(r"\bNAN_KIND=([\w-]+)", led),
        "nan_origin": last(r"\bNAN_ORIGIN=(\w+)", led),
        "reproducer": repro,
        "groups": groups,
        # the model-level rates: step 4's control arm (`ctl=2/2`, `ctl 1/2`) and step 8's stock arm
        "rates": [re.sub(r"^\S+ \S+\s+", "", l)[:200] for l in led
                  if re.search(r"\b(ctl|stock)\b(\s*(rate|arm))?\s*[:=]?\s*\d+/\d+", l)
                  and re.search(r"\sstep[248]\S*\s", l)
                  # a fix's reproducer / package checks count capsule runs, not model runs (2026-11-23:
                  # validate.sh and a ledger correction filled this list)
                  and not re.search(r"(?i)reproducer|validate|verif(y|ied)|FIX_VALIDATE|FIX_VALUES|fixval|values job|"
                                    r"correction|launches", l)][-4:],
    }
    return f


def merge(new, old):
    """Keep the agent's answers (anything not TODO) from an earlier card."""
    for k in ("root_cause", "fix"):
        new[k] = old.get(k, json.loads(json.dumps(AGENT_FIELDS[k])))
    if not new.get("skill_commit") and old.get("skill_commit"):
        new["skill_commit"] = old["skill_commit"]      # set by hand for a run whose ledger predates SKILL_COMMIT
    if old.get("bug", {}).get("verdict_confirmed"):
        new["bug"]["verdict"] = old["bug"]["verdict"]
        new["bug"]["verdict_confirmed"] = True
    return new


def todos(card, path=""):
    out = []
    if isinstance(card, dict):
        for k, v in card.items():
            out += todos(v, "%s.%s" % (path, k) if path else k)
    elif isinstance(card, list):
        for i, v in enumerate(card):
            out += todos(v, "%s[%d]" % (path, i))
    elif card == TODO:
        out.append(path)
    return out


def problems(card):
    p = ["%s is TODO" % t for t in todos(card)]
    for (sec, key), allowed in ENUMS.items():
        v = card.get(sec, {}).get(key)
        if v != TODO and v not in allowed:
            p.append("%s.%s=%r: one of %s" % (sec, key, v, ", ".join(sorted(allowed))))
    if not card.get("bug", {}).get("verdict_confirmed"):
        p.append("bug.verdict_confirmed is not true (confirm bug.verdict=%r, or correct it)"
                 % card.get("bug", {}).get("verdict"))
    rc, fx = card.get("root_cause", {}), card.get("fix", {})
    # list fields must be lists of strings: a bare string passed the check and compare_results.py then
    # compared it character by character -- DIFFERENT on a matching root cause (2026-11-23, location.source)
    lists = [("root_cause.location." + k, rc.get("location", {}).get(k, [])) for k in ("sites", "env", "source", "config")]
    lists += [("root_cause.conditions", rc.get("conditions", []))]
    lists += [("fix." + k, fx.get(k, [])) for k in ("changes", "extra", "alternatives")]
    bad = set()
    for name, v in lists:
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            p.append("%s=%r: a list of strings (e.g. [%r])" % (name, v, v if isinstance(v, str) else "..."))
            bad.add(name)
    for s in (rc.get("location", {}).get("source", []) if "root_cause.location.source" not in bad else []):
        if re.search(r":\d+$", s):
            p.append("root_cause.location.source %r: file.py:function, no line number" % s)
    for s in (rc.get("location", {}).get("sites", []) if "root_cause.location.sites" not in bad else []):
        if not re.match(r"(?i)^0x[0-9a-f]+$", s):
            p.append("root_cause.location.sites %r: a hex address (0x2A8D9D0)" % s)
    if fx.get("status") in ("FIX", "WORKAROUND") and "fix.changes" not in bad \
            and not [c for c in fx.get("changes", []) if c != TODO]:
        p.append("fix.changes is empty for fix.status=%s" % fx.get("status"))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("draft", "check"))
    ap.add_argument("artifact_dir")
    a = ap.parse_args()
    A = os.path.abspath(a.artifact_dir)
    path = os.path.join(A, "result.json")
    if a.cmd == "draft":
        card = facts(A)
        old = {}
        if os.path.exists(path):
            try:
                old = json.load(open(path))
            except ValueError:
                sys.exit("result_card: %s is not JSON -- fix or remove it" % path)
        card = merge(card, old) if old else dict(card, **json.loads(json.dumps(AGENT_FIELDS)))
        card.setdefault("bug", {}).setdefault("verdict_confirmed", False)
        with open(path, "w") as f:
            json.dump(card, f, indent=1)
            f.write("\n")
        left = problems(card)
        print("RESULT_CARD %s (%d open)" % (path, len(left)))
        for x in left:
            print("  " + x)
        return 0
    try:
        card = json.load(open(path))
    except (OSError, ValueError) as e:
        sys.exit("result_card: %s: %s" % (path, e))
    left = problems(card)
    for x in left:
        print("  " + x)
    print("RESULT_CARD_CHECK=%s %s" % ("OK" if not left else "OPEN(%d)" % len(left), path))
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
