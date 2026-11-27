#!/usr/bin/env python3
"""Same result? Compare two debug-numeric result cards (result_card.py) -- two runs of one example,
or one example under two versions of the skill.

    python3 scripts/compare_results.py <ART_A | A/result.json> <ART_B | B/result.json> [--json]

Three judged levels, each SAME / REFINED / DIFFERENT, and the overall result is the worst of them:
  bug         verdict, op, problem shape, kernel (solution index, or the kernel name when there is none)
  root_cause  status, component, location (kernel, instruction sites, env, source, config)
  fix         status, kind, the primary change set
REFINED: one side's site or change set is a strict subset of the other's -- the same answer, one
narrower (e.g. a 3-site patch against a 14-site patch of the same kernel).  Context (NaN kind and
origin, how it was reproduced, groups, rates, the free-text mechanism) is printed under `notes`
and never judged: it differs run to run.  Last line: RESULT_COMPARE=SAME|REFINED|DIFFERENT.
"""
import argparse
import json
import os
import re
import sys

RANK = {"SAME": 0, "REFINED": 1, "DIFFERENT": 2}


def load(p):
    if os.path.isdir(p):
        p = os.path.join(p, "result.json")
    try:
        return json.load(open(p)), p
    except (OSError, ValueError) as e:
        sys.exit("compare_results: %s: %s (result_card.py draft/check first)" % (p, e))


def norm(s):
    """One spelling per thing: hex sites, VAR=VALUE / key=value with numeric values as numbers."""
    s = str(s).strip()
    m = re.match(r"^(.*?@)?0x0*([0-9a-fA-F]+)$", s)
    if m:
        return "%s0x%s" % ((m.group(1) or "").lower(), m.group(2).upper())
    m = re.match(r"^([^=\s]+)\s*=\s*(.+)$", s)
    if m:
        v = m.group(2).strip()
        # one value, one spelling: true/on/yes == 1 == 1.0 (an env flag written either way,
        # 2026-11-23: VLLM_..._CAST_BF16_TO_FP16=true on the card, =1.0 in a run)
        v = {"true": "1", "yes": "1", "on": "1", "false": "0", "no": "0", "off": "0"}.get(v.lower(), v)
        try:
            v = repr(float(v))
        except ValueError:
            v = v.lower()
        return "%s=%s" % (m.group(1), v)
    return s


def nset(xs):
    return {norm(x) for x in (xs or []) if x and x != "TODO"}


def cmp_scalar(a, b):
    return "SAME" if (a or None) == (b or None) else "DIFFERENT"


_PATHFN = re.compile(r"^([^\s:=@]+/)?([^\s/:=@]+\.\w+):([\w.]+)$")


def unify_paths(a, b):
    """`file:Class.function` entries that differ only by how much of the path is written are one site:
    `mla.py:mla_decode_fwd` == `aiter/mla.py:mla_decode_fwd` (a path suffix on a `/` boundary, same
    file and function).  b's spelling is rewritten to a's (2026-11-23 validation: DIFFERENT by spelling)."""
    out = set()
    for y in b:
        my = _PATHFN.match(y)
        hit = None
        if my:
            for x in a:
                mx = _PATHFN.match(x)
                if mx and mx.group(2) == my.group(2) and mx.group(3) == my.group(3):
                    px, py = x.split(":")[0], y.split(":")[0]
                    if px.endswith("/" + py) or py.endswith("/" + px) or px == py:
                        hit = x
                        break
        out.add(hit or y)
    return out


def cmp_set(a, b):
    a, b = nset(a), nset(b)
    b = unify_paths(a, b)
    if a == b:
        return "SAME", ""
    if a < b or b < a:
        small = "A" if a < b else "B"
        return "REFINED", "%s is narrower: %d vs %d (%s only in the wider: %s)" % (
            small, min(len(a), len(b)), max(len(a), len(b)), "+%d" % len(a ^ b),
            " ".join(sorted(a ^ b))[:200])
    if a & b:
        return "DIFFERENT", "overlap %d, only A: %s; only B: %s" % (
            len(a & b), " ".join(sorted(a - b))[:150], " ".join(sorted(b - a))[:150])
    return "DIFFERENT", "only A: %s; only B: %s" % (" ".join(sorted(a))[:150], " ".join(sorted(b))[:150])


def _co_stem(s):
    """The file stem of an aiter / HSA code object a kernel string names (any path), or None."""
    m = re.search(r"([^\s/()]+)\.co\b", str(s or ""))
    return m.group(1) if m else None


def cmp_kernel(a, b):
    """cmp_scalar, except that one .co spelled as an install path, a path under hsa/, or its
    bare stem inside free text ("aiter mla_dec_stage1_...") is the same kernel (2026-11-20)."""
    if (a or None) == (b or None):
        return "SAME"
    sa, sb = _co_stem(a), _co_stem(b)
    words = lambda s: set(re.findall(r"[\w.]+", str(s or "")))                     # noqa: E731
    if (sa and sb and sa == sb) or (sa and not sb and sa in words(b)) or (sb and not sa and sb in words(a)):
        return "SAME"
    # free-text kernel names: parentheticals and template arguments are commentary, `_` and case are
    # spelling -- one name's words a subset of the other's is the same kernel ("aiter mla_decode_fwd" ==
    # "aiter mla_decode_fwd (_fwd_kernel_stage2_asm reads ...)", "vllm quickreduce AllReduceTwoshot" ==
    # "vllm _C quick_reduce AllReduceTwoshot<cast_bf2half=true>", 2026-11-23)
    def core(s):
        s = str(s or "")
        prev = None
        while prev != s:
            prev, s = s, re.sub(r"\([^()]*\)|<[^<>]*>", " ", s)
        return {w.replace("_", "").lower() for w in re.findall(r"[A-Za-z0-9_]+", s) if w.strip("_")}
    ca, cb = core(a), core(b)
    if ca and cb and (ca <= cb or cb <= ca) and min(len(ca), len(cb)) >= 2:
        return "SAME"
    return "DIFFERENT"


def kernel_id(k):
    if not k:
        return None
    if isinstance(k, str):              # an older card's bare kernel name
        return k
    return ("solution %s" % k["solution_index"]) if k.get("solution_index") else k.get("name")


def compare(A, B):
    rows = []      # (level, field, verdict, a, b, note)

    def row(level, field, verdict, a, b, note=""):
        rows.append((level, field, verdict, a, b, note))

    ba, bb = A.get("bug", {}), B.get("bug", {})
    ka, kb = kernel_id(ba.get("kernel")), kernel_id(bb.get("kernel"))
    for k in ("verdict", "op", "geometry"):
        if k == "op" and ka and ka == kb:
            # the same kernel reached through another op (aten.mm / aten.addmm, 2026-11-20): the
            # kernel is the bug's identity, the op only how this run's capture met it
            if ba.get(k) != bb.get(k):
                row("notes", "bug.op", "-", ba.get(k), bb.get(k), "same kernel %s: the op is the call path" % ka)
            continue
        row("bug", k, cmp_scalar(ba.get(k), bb.get(k)), ba.get(k), bb.get(k))
    row("bug", "kernel", cmp_kernel(ka, kb), ka, kb)

    ra, rb = A.get("root_cause", {}), B.get("root_cause", {})
    for k in ("status", "component"):
        row("root_cause", k, cmp_scalar(ra.get(k), rb.get(k)), ra.get(k), rb.get(k))
    la, lb = ra.get("location", {}), rb.get("location", {})
    row("root_cause", "location.kernel", cmp_kernel(la.get("kernel"), lb.get("kernel")), la.get("kernel"), lb.get("kernel"))
    for k in ("sites", "env", "source", "config"):
        if la.get(k) or lb.get(k):
            v, note = cmp_set(la.get(k), lb.get(k))
            row("root_cause", "location." + k, v, len(nset(la.get(k))), len(nset(lb.get(k))), note)
    if ra.get("conditions") or rb.get("conditions"):
        v, note = cmp_set(ra.get("conditions"), rb.get("conditions"))
        # free-text conditions: shown, a difference is a note (spelled by different agents)
        row("notes", "root_cause.conditions", "-", len(nset(ra.get("conditions"))),
            len(nset(rb.get("conditions"))), note or "same spelling")

    fa, fb = A.get("fix", {}), B.get("fix", {})
    for k in ("status", "kind"):
        row("fix", k, cmp_scalar(fa.get(k), fb.get(k)), fa.get(k), fb.get(k))
    v, note = cmp_set(fa.get("changes"), fb.get("changes"))
    if v == "DIFFERENT":
        # An expected card lists the other correct answers under fix.alternatives (a stronger
        # patch at the same site, a workaround): a run that shipped one of them found the bug and
        # a less minimal fix -- REFINED, the card narrower, not a failure (2026-11-20, aiter MLA:
        # vmcnt0@0x2848 vs the card's vmcnt(18)@0x2848).
        for card, run in ((A, fb), (B, fa)):
            alts = nset((card.get("fix") or {}).get("alternatives"))
            got = nset(run.get("changes"))
            if card.get("kind") == "expected" and got and got <= alts:
                v, note = "REFINED", "the run's change(s) are the expected card's alternative(s): %s" % (
                    " ".join(sorted(got))[:150])
    row("fix", "changes", v, " ".join(sorted(nset(fa.get("changes"))))[:60],
        " ".join(sorted(nset(fb.get("changes"))))[:60], note)

    # context: reported, never judged
    for k in ("extra", "alternatives"):
        if fa.get(k) or fb.get(k):
            _, note = cmp_set(fa.get(k), fb.get(k))
            row("notes", "fix." + k, "-", len(nset(fa.get(k))), len(nset(fb.get(k))), note or "same")
    ca, cb = A.get("context", {}), B.get("context", {})
    for k in ("nan_kind", "nan_origin"):
        if ca.get(k) != cb.get(k):
            row("notes", "context." + k, "-", ca.get(k), cb.get(k))
    if ba.get("reproduced_by") != bb.get("reproduced_by"):
        row("notes", "bug.reproduced_by", "-", ba.get("reproduced_by"), bb.get("reproduced_by"))
    ga = [(g["group"], g.get("op"), g.get("geometry"), g.get("count")) for g in ca.get("groups", [])]
    gb = [(g["group"], g.get("op"), g.get("geometry"), g.get("count")) for g in cb.get("groups", [])]
    if ga != gb:
        row("notes", "context.groups", "-", " ".join("%s:%s" % (g[0], g[3]) for g in ga),
            " ".join("%s:%s" % (g[0], g[3]) for g in gb), "different sampling of call sites")
    pa = next((g for g in ca.get("groups", []) if g.get("op") == ba.get("op")), {})
    pb = next((g for g in cb.get("groups", []) if g.get("op") == bb.get("op")), {})
    if pa.get("stack_digest") and pb.get("stack_digest") and pa["stack_digest"] != pb["stack_digest"]:
        row("notes", "primary call site", "-", pa["stack_digest"], pb["stack_digest"],
            "the replayed group came from a different call stack (the judged fields above decide)")
    if (ra.get("mechanism") or "") != (rb.get("mechanism") or ""):
        row("notes", "root_cause.mechanism", "-", (ra.get("mechanism") or "")[:70], (rb.get("mechanism") or "")[:70])
    if ca.get("rates") or cb.get("rates"):
        row("notes", "rates", "-", "; ".join(ca.get("rates", [])[-1:])[:70], "; ".join(cb.get("rates", [])[-1:])[:70])

    judged = [r for r in rows if r[0] != "notes"]
    levels = {}
    for lv in ("bug", "root_cause", "fix"):
        vs = [r[2] for r in judged if r[0] == lv]
        levels[lv] = max(vs, key=RANK.get) if vs else "SAME"
    # an unanswered card (TODO) is not a result: say so instead of comparing it
    for side, c in (("A", A), ("B", B)):
        if "TODO" in json.dumps({k: c.get(k) for k in ("root_cause", "fix")}):
            rows.append(("notes", "card %s" % side, "-", "has TODO fields", "", "result_card.py check"))
    overall = max(levels.values(), key=RANK.get)
    return rows, levels, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", action="store_true")
    x = ap.parse_args()
    (A, pa), (B, pb) = load(x.a), load(x.b)
    rows, levels, overall = compare(A, B)
    if x.json:
        print(json.dumps({"a": pa, "b": pb, "levels": levels, "overall": overall,
                          "rows": [dict(zip(("level", "field", "verdict", "a", "b", "note"), r)) for r in rows]},
                         indent=1, default=str))
        return 0
    for side, c in (("A", A), ("B", B)):
        print("%s: %s (priors %s, skill %s)" % (side, c.get("artifact_dir"), c.get("priors"),
                                                c.get("skill_commit") or "? (not in its ledger)"))
    for lv in ("bug", "root_cause", "fix", "notes"):
        rs = [r for r in rows if r[0] == lv]
        if not rs:
            continue
        print("\n== %s%s" % (lv, "" if lv == "notes" else ": " + levels[lv]))
        for _, f, v, a, b, note in rs:
            same = v == "SAME"
            print("  %-22s %-9s %s" % (f, v, ("%s" % (a,))[:90] if same else "A=%s | B=%s" % (
                ("%s" % (a,))[:60], ("%s" % (b,))[:60])))
            if note:
                print("  %-22s %-9s   %s" % ("", "", note))
    print("\nRESULT_COMPARE=%s  (bug %s, root_cause %s, fix %s)"
          % (overall, levels["bug"], levels["root_cause"], levels["fix"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
