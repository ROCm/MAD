#!/usr/bin/env python3
"""Static wait-count hazard scan of one AMDGPU kernel (GFX9: gfx90a / gfx942 / gfx950).

    python3 waitcnt_scan.py <listing.s> [--json OUT] [--max-sites N]
    python3 waitcnt_scan.py --self-test

<listing.s> is llvm-objdump output for one kernel -- the `*.before.s` a mutation sweep writes to
<ART>/asm/, or `llvm-objdump -d --mcpu=<arch> <code object>` sliced to the kernel.  No GPU, no torch:
it runs on a login node.

It ORDERS step 7's experiments; it never replaces them (design-cheaper-runs.md C1).  A hazard it reports
may be real and never fire (latency hides it), its hardware model can be wrong, and only a one-site CLEAN
against the other sites' NAN separates an ordering bug from "any delay".  Its sites go to the subset test
(`NAN_MUTATE_SUBSET_ADDRS`, step7-recipe-sweep.md): `@set` CLEAN with `@rest` NAN is the minimal set and
tier 3's per-site sweep is skipped.  It covers wait counts guarding REGISTERS only -- a global->LDS load
read through LDS before it lands needs LDS address analysis this scan does not do (a pipelined kernel keeps
such loads in flight across barriers by design: aiter MLA, two to three), and a missing s_barrier between
waves, a NaN constant or a store slice stay the sweep's to find (vmcnt0_each, sbarrier_*, nans_to_nops).

The model, per wave
-------------------
* Two counters.  VM: vector memory loads and stores (gfx9 counts both), retired in issue order -- a load
  is surely retired by `s_waitcnt vmcnt(N)` once N or more VM ops were issued after it.  LGKM: LDS (ds_*,
  retired in order among themselves) and scalar memory (s_load*, s_buffer_load*: out of order).  An
  outstanding LDS op makes the counter >= 1 + the LDS ops issued after it, so `lgkmcnt(N)` surely retires
  it once N or more LDS ops followed it; a scalar load is surely retired only by `lgkmcnt(0)`.
* A control-flow graph from the branches; the outstanding set is propagated to a fixed point, joined by
  "may be outstanding" (union, fewest ops issued after it), so a load issued at the bottom of a loop is
  outstanding at its top.  A load more than 63 same-class ops old is retired (no counter waits that long).

Hazards
-------
  H1 register   an instruction reads, or overwrites, a register that is the destination of a load that
                may still be outstanding.  Charged to the waits it passed while outstanding ("relaxed"):
                a wait with a smaller count there would have retired it.  (2026-11-23 hipBLASLt solution
                483266: ds_read v[102:109] outstanding past lgkmcnt(2) while v_cvt_pk_bf16_f32 reads
                and writes them -- 3 of the kernel's 14 relaxed lgkmcnt waits.  aiter MLA stage 1:
                buffer_load_dword v11 outstanding past vmcnt(19) at 0x2848 while v_mov_b32_dpp reads
                v11 for the next tile's address -- fix vmcnt(18), the one that shipped.)
  H3 outlier    among the waits in front of an s_barrier that leave the same number of global->LDS loads
                in flight (one pipeline depth), the few that also leave register loads in flight while the
                rest drain them.  Weaker evidence, listed apart.  (aiter MLA stage 1: vmcnt(19) at 0x2848,
                vmcnt(18) at its 13 siblings, 2026-11-23.)

Output: one line per site, then
  WAITCNT_SCAN=<n sites> H1=<a> H3=<c> insns=<n> waits=<n> relaxed=<n> unattributed=<n>
  NAN_MUTATE_SUBSET_ADDRS="0x.. 0x.."           (H1 sites; H3 on its own line)
  NAN_MUTATE_SUBSET_ADDRS_H3="0x.."
and per site the tightest wait that retires every load it was charged with (`fix:`).
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VM_CAP = 63         # gfx9 vmcnt field: 6 bits
LGKM_CAP = 15       # gfx9 lgkmcnt field: 4 bits

# --------------------------------------------------------------------------- #
# listing
# --------------------------------------------------------------------------- #
_LINE = re.compile(r"^\s*(?P<text>[a-z_][\w.]*(?:\s.*?)?)\s*//\s*(?P<addr>[0-9A-Fa-f]+):\s*"
                   r"(?P<words>(?:[0-9A-Fa-f]{8}\s*)+)(?:<[^>]*>)?\s*$")


class Insn:
    __slots__ = ("i", "addr", "size", "text", "mn", "ops", "words")

    def __init__(self, i, addr, text, words):
        self.i, self.addr, self.text, self.words = i, addr, text, words
        self.size = 4 * len(words)
        parts = text.split(None, 1)
        self.mn = parts[0]
        self.ops = _split_ops(parts[1]) if len(parts) > 1 else []


def _split_ops(s):
    """Operands, split on commas outside brackets; modifiers (offset:, row_mask:, ...) dropped."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    # the last operand may carry space-separated modifiers: "v4 offset:64", "off offset:128 sc0"
    if out:
        out[-1] = out[-1].split()[0] if out[-1].split() else out[-1]
        out[:-1] = [o.split()[0] if o.split() else o for o in out[:-1]]
    return out


def parse_listing(text):
    insns = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        insns.append(Insn(len(insns), int(m.group("addr"), 16), m.group("text").strip(),
                          m.group("words").split()))
    return insns


# --------------------------------------------------------------------------- #
# registers
# --------------------------------------------------------------------------- #
_REG = re.compile(r"^-?\|?(?P<k>[vsa])(?:\[(?P<lo>\d+):(?P<hi>\d+)\]|(?P<n>\d+))\|?$")
_SPECIAL = {"vcc": (("s", "vcc_lo"), ("s", "vcc_hi")), "vcc_lo": (("s", "vcc_lo"),),
            "vcc_hi": (("s", "vcc_hi"),), "exec": (("s", "exec_lo"), ("s", "exec_hi")),
            "exec_lo": (("s", "exec_lo"),), "exec_hi": (("s", "exec_hi"),), "m0": (("s", "m0"),)}


def regs(op):
    """Set of (kind, index) a printed operand names: v7, v[2:5], s[0:1], a12, vcc, m0."""
    op = op.strip()
    op = re.sub(r"\(.*\)$", "", op)                 # "v6(0x80000000)"-style annotations
    if op in _SPECIAL:
        return set(_SPECIAL[op])
    m = _REG.match(op)
    if not m:
        return set()
    k = m.group("k")
    if m.group("n") is not None:
        return {(k, int(m.group("n")))}
    return {(k, i) for i in range(int(m.group("lo")), int(m.group("hi")) + 1)}


def _fmt_regs(rs):
    by = collections.defaultdict(list)
    for k, i in rs:
        by[k].append(i)
    out = []
    for k in sorted(by, key=str):
        xs = sorted(x for x in by[k] if isinstance(x, int))
        if xs:
            out.append("%s[%d:%d]" % (k, xs[0], xs[-1]) if len(xs) > 1 else "%s%d" % (k, xs[0]))
        out += sorted(str(x) for x in by[k] if not isinstance(x, int))
    return ",".join(out)


# --------------------------------------------------------------------------- #
# instruction classes
# --------------------------------------------------------------------------- #
def is_lds_dma(ins):
    return ins.mn.startswith("global_load_lds") or (ins.mn.startswith("buffer_load")
                                                    and re.search(r"\blds\b", ins.text) is not None)


def vm_class(ins):
    """'load' (a register destination), 'lds' (global->LDS), 'store', 'atomic', or None."""
    mn = ins.mn
    if not re.match(r"^(buffer|global|flat|scratch|tbuffer)_", mn):
        return None
    if is_lds_dma(ins):
        return "lds"
    if "_load" in mn:
        return "load"
    if "_store" in mn:
        return "store"
    if "_atomic" in mn:
        return "atomic_ret" if re.search(r"\b(glc|sc0)\b", ins.text) else "atomic"
    return None


def lgkm_class(ins):
    """'ds_load', 'ds', 'smem' or None."""
    mn = ins.mn
    if mn.startswith("ds_"):
        if mn.startswith(("ds_read", "ds_load")) or "_rtn" in mn or mn.startswith(("ds_swizzle",
                                                                                   "ds_permute",
                                                                                   "ds_bpermute")):
            return "ds_load"
        return "ds"
    if mn.startswith(("s_load_", "s_buffer_load_", "s_scratch_load", "s_atc_probe")):
        return "smem"
    if mn in ("s_sendmsg", "s_sendmsghalt", "s_memtime", "s_memrealtime"):
        return "smem"
    return None


_NO_DEST = re.compile(r"^(s_waitcnt|s_barrier|s_nop|s_endpgm|s_branch|s_cbranch|s_setprio|s_sleep|"
                      r"s_sendmsg|s_dcache|s_icache|s_trap|s_set_gpr|s_setreg|s_ttrace|s_waitcnt_|"
                      r"s_cmp|s_bitcmp|ds_write|ds_store|buffer_store|global_store|flat_store|"
                      r"scratch_store|exp|s_inst_prefetch)")


def dests_and_reads(ins):
    """(written registers, read registers) of one instruction, from its printed operands."""
    ops = ins.ops
    if not ops:
        return set(), set()
    if _NO_DEST.match(ins.mn) or is_lds_dma(ins) or vm_class(ins) in ("store", "atomic"):
        return set(), set().union(*[regs(o) for o in ops])
    d = regs(ops[0])
    r = set().union(*[regs(o) for o in ops[1:]]) if len(ops) > 1 else set()
    # VOP3B carry-out / VOPC-e64 sdst: the second operand is written too
    if re.match(r"^v_(mad_u64_u32|mad_i64_i32|add_co_u32|sub_co_u32|subrev_co_u32|addc_co_u32|"
                r"subb_co_u32|subbrev_co_u32|div_scale)", ins.mn) and len(ops) > 1:
        d |= regs(ops[1])
        r -= regs(ops[1])
        if ins.mn.startswith(("v_addc", "v_subb")) and len(ops) > 4:
            r |= regs(ops[4])
    if ins.mn.startswith(("v_mac_", "v_fmac_", "v_dot2c_", "v_mfma", "v_smfmac")) or \
            (ins.mn.startswith(("v_mfma", "v_smfmac"))):
        r |= d                                      # accumulates into its destination
    if ins.mn.startswith("v_cmp") and ins.mn.endswith("_e32"):
        d = {("s", "vcc_lo"), ("s", "vcc_hi")}
        r = set().union(*[regs(o) for o in ops])
    if ins.mn.startswith("s_cmp"):
        d = set()
    return d, r


def wait_counts(ins):
    """(vmcnt or None, lgkmcnt or None) of an s_waitcnt; a counter not printed is not waited on."""
    if ins.mn != "s_waitcnt":
        return None, None
    vm = re.search(r"\bvmcnt\((\d+)\)", ins.text)
    lg = re.search(r"\blgkmcnt\((\d+)\)", ins.text)
    if not vm and not lg and re.search(r"\b0\b", ins.text):
        return 0, 0                                 # "s_waitcnt 0": everything
    return (int(vm.group(1)) if vm else None), (int(lg.group(1)) if lg else None)


_BR = re.compile(r"^s_(cbranch_\w+|branch)$")


def branch_target(ins):
    if not _BR.match(ins.mn) or ins.mn.startswith("s_cbranch_g_fork"):
        return None
    if ins.mn in ("s_cbranch_join",):
        return None
    try:
        word = int(ins.words[0], 16)
    except (IndexError, ValueError):
        return None
    simm = word & 0xFFFF
    if simm & 0x8000:
        simm -= 0x10000
    return ins.addr + 4 + 4 * simm


# --------------------------------------------------------------------------- #
# dataflow
# --------------------------------------------------------------------------- #
class Op:
    """One outstanding memory op in the abstract state: issued at insn `i`; `after` = how many ops
    of its retiring class were issued after it on the path with the FEWEST (may-outstanding);
    `waits` = relaxed waits it has passed while outstanding."""
    __slots__ = ("i", "counter", "after", "waits")

    def __init__(self, i, counter, after=0, waits=frozenset()):
        self.i, self.counter, self.after, self.waits = i, counter, after, waits


def _join(a, b):
    out = dict(a)
    for k, op in b.items():
        o = out.get(k)
        if o is None:
            out[k] = op
        else:
            out[k] = Op(k, op.counter, min(o.after, op.after), o.waits | op.waits)
    return out


def _same(a, b):
    if a.keys() != b.keys():
        return False
    return all(a[k].after == b[k].after and a[k].waits == b[k].waits for k in a)


def _counter(ins):
    v = vm_class(ins)
    if v:
        return "vm"
    lg = lgkm_class(ins)
    if lg == "smem":
        return "smem"
    if lg:
        return "ds"
    return None


def _step(state, ins, insns):
    """State after `ins` (waits applied, then the op issued)."""
    vmn, lgn = wait_counts(ins)
    if vmn is not None or lgn is not None:
        new = {}
        for k, op in state.items():
            if op.counter == "vm" and vmn is not None:
                if op.after >= vmn:
                    continue
                op = Op(op.i, op.counter, op.after, op.waits | {ins.i})
            elif op.counter == "ds" and lgn is not None:
                if op.after >= lgn:
                    continue
                op = Op(op.i, op.counter, op.after, op.waits | {ins.i})
            elif op.counter == "smem" and lgn is not None:
                if lgn == 0:
                    continue
                op = Op(op.i, op.counter, op.after, op.waits | {ins.i})
            new[k] = op
        state = new
    c = _counter(ins)
    if c in ("vm", "ds"):
        cap = VM_CAP if c == "vm" else LGKM_CAP + 48     # ds: a bound, not a field width
        new = {}
        for k, op in state.items():
            if op.counter == c:
                op = Op(op.i, op.counter, op.after + 1, op.waits)
                if op.after > cap:
                    continue
            new[k] = op
        new[ins.i] = Op(ins.i, c, 0)
        state = new
    elif c == "smem":
        state = dict(state)
        state[ins.i] = Op(ins.i, c, 0)
    return state


def blocks(insns):
    """Basic blocks: list of (start, end) insn index ranges, and successor lists."""
    by_addr = {x.addr: x.i for x in insns}
    leaders = {0}
    for x in insns:
        t = branch_target(x)
        if t is not None:
            if t in by_addr:
                leaders.add(by_addr[t])
            if x.i + 1 < len(insns):
                leaders.add(x.i + 1)
        elif x.mn in ("s_endpgm", "s_setpc_b64") and x.i + 1 < len(insns):
            leaders.add(x.i + 1)
    starts = sorted(leaders)
    bl, of = [], {}
    for j, s in enumerate(starts):
        e = (starts[j + 1] if j + 1 < len(starts) else len(insns))
        bl.append((s, e))
        for i in range(s, e):
            of[i] = j
    succ = []
    for j, (s, e) in enumerate(bl):
        last = insns[e - 1]
        out = []
        t = branch_target(last)
        if t is not None and t in by_addr:
            out.append(of[by_addr[t]])
        if last.mn not in ("s_branch", "s_endpgm", "s_setpc_b64") and e < len(insns):
            out.append(j + 1)
        succ.append(out)
    return bl, succ


def dataflow(insns, max_rounds=200):
    """The may-outstanding state at the entry of every instruction (list of dicts)."""
    bl, succ = blocks(insns)
    entry = [None] * len(bl)
    entry[0] = {}
    work = collections.deque([0])
    rounds = collections.Counter()
    while work:
        j = work.popleft()
        rounds[j] += 1
        if rounds[j] > max_rounds:
            raise RuntimeError("no fixed point at block %d after %d rounds" % (j, max_rounds))
        st = entry[j]
        s, e = bl[j]
        for i in range(s, e):
            st = _step(st, insns[i], insns)
        for k in succ[j]:
            nxt = st if entry[k] is None else _join(entry[k], st)
            if entry[k] is None or not _same(entry[k], nxt):
                entry[k] = nxt
                work.append(k)
    at = [None] * len(insns)
    for j, (s, e) in enumerate(bl):
        st = entry[j]
        if st is None:
            continue                                # unreachable from the entry
        for i in range(s, e):
            at[i] = st
            st = _step(st, insns[i], insns)
    return at, bl, succ


# --------------------------------------------------------------------------- #
# hazards
# --------------------------------------------------------------------------- #
def scan(insns):
    at, bl, succ = dataflow(insns)
    dest_of = {}
    for x in insns:
        c = _counter(x)
        if c and not is_lds_dma(x) and vm_class(x) not in ("store", "atomic") and lgkm_class(x) != "ds":
            dest_of[x.i] = dests_and_reads(x)[0]
    h1 = collections.defaultdict(list)          # wait insn index -> [(reader, load, regs)]
    h1_unattributed = []
    for x in insns:
        st = at[x.i]
        if not st or x.mn == "s_waitcnt":
            continue
        # a wait in this instruction itself was applied before its read: use the state after it
        vmn, lgn = wait_counts(x)
        d, r = dests_and_reads(x)
        touched = d | r
        for k, op in st.items():
            if k == x.i or k not in dest_of:
                continue
            hit = dest_of[k] & touched
            if not hit:
                continue
            if op.counter == _counter(x) and op.counter in ("vm", "ds") and not (dest_of[k] & r) \
                    and (d & dest_of[k]) and vm_class(x) != "lds":
                continue                            # WAW within an in-order counter: ordered
            rec = (x.i, k, frozenset(hit))
            if op.waits:
                for w in op.waits:
                    h1[w].append(rec)
            else:
                h1_unattributed.append(rec)
    # H3: barrier waits grouped by the lds-dma loads they leave in flight; minority leaving register loads
    groups = collections.defaultdict(list)
    for x in insns:
        vmn, _ = wait_counts(x)
        if vmn is None or not at[x.i]:
            continue
        follow = next((y for y in insns[x.i + 1:x.i + 3]), None)
        if follow is None or follow.mn != "s_barrier":
            continue
        after = _step(at[x.i], x, insns)
        lds = sum(1 for k, op in after.items() if op.counter == "vm" and is_lds_dma(insns[k]))
        regl = sum(1 for k, op in after.items() if op.counter == "vm" and vm_class(insns[k]) == "load")
        if lds:
            groups[lds].append((x.i, regl, vmn))
    h3 = {}
    for lds, members in groups.items():
        if len(members) < 3:
            continue
        drain = [m for m in members if m[1] == 0]
        keep = [m for m in members if m[1] > 0]
        if keep and len(keep) * 3 <= len(members) and drain:
            usual = collections.Counter(m[2] for m in drain).most_common(1)[0][0]
            for m in keep:
                h3[m[0]] = {"lds_in_flight": lds, "reg_loads_in_flight": m[1], "count": m[2],
                            "siblings": len(drain), "siblings_count": usual}
    return {"h1": h1, "h1_unattributed": h1_unattributed, "h3": h3, "at": at}


def report(insns, res, max_sites=50):
    lines, sites = [], []
    at = res["at"]
    waits = [x for x in insns if x.mn == "s_waitcnt"]
    relaxed = [x for x in waits if any(c for c in wait_counts(x) if c)]
    for w, recs in sorted(res["h1"].items(), key=lambda kv: insns[kv[0]].addr):
        x = insns[w]
        loads = sorted({k for _, k, _ in recs})
        readers = sorted({r for r, _, _ in recs})
        # tightest count that retires every charged load at this wait
        need = {}
        for k in loads:
            op = at[w].get(k)
            if op is None:
                continue
            c = op.counter
            need[c] = min(need.get(c, 99), op.after)
        fix = " ".join("%s(%d)" % ("vmcnt" if c == "vm" else "lgkmcnt", n if c != "smem" else 0)
                       for c, n in sorted(need.items()))
        first = recs[0]
        sites.append({"addr": "0x%X" % x.addr, "kind": "H1", "wait": x.text, "fix": fix,
                      "loads": ["0x%X %s" % (insns[k].addr, insns[k].text) for k in loads[:4]],
                      "n_loads": len(loads), "n_readers": len(readers),
                      "example": "0x%X %s touches %s" % (insns[first[0]].addr, insns[first[0]].text,
                                                         _fmt_regs(first[2]))})
    for w, info in sorted(res["h3"].items(), key=lambda kv: insns[kv[0]].addr):
        x = insns[w]
        sites.append({"addr": "0x%X" % x.addr, "kind": "H3", "wait": x.text,
                      "fix": "vmcnt(%d)" % (info["count"] - info["reg_loads_in_flight"]),
                      "loads": [], "n_loads": info["reg_loads_in_flight"], "n_readers": 0,
                      "example": "leaves %d register load(s) in flight beside %d lds-dma; %d sibling "
                                 "barrier waits drain them (vmcnt(%d))"
                                 % (info["reg_loads_in_flight"], info["lds_in_flight"],
                                    info["siblings"], info["siblings_count"])})
    for s in sites[:max_sites]:
        lines.append("%-3s %-10s %-34s fix: %-20s %s" % (s["kind"], s["addr"], s["wait"], s["fix"],
                                                          s["example"]))
        for ld in s["loads"][:2]:
            lines.append("               load %s" % ld)
    if res["h1_unattributed"]:
        r, k, hit = res["h1_unattributed"][0]
        lines.append("note: %d read(s) of a load never covered by any wait on some path, e.g. 0x%X "
                     "%s reads %s from 0x%X %s (no wait to charge; often a path the branch conditions rule out "
                     "-- the scan does not follow them; not in the subset set)"
                     % (len(res["h1_unattributed"]), insns[r].addr, insns[r].text, _fmt_regs(hit),
                        insns[k].addr, insns[k].text))
    n1 = len(res["h1"])
    n3 = len(res["h3"])
    set12 = sorted({s["addr"] for s in sites if s["kind"] == "H1"}, key=lambda a: int(a, 16))
    set3 = sorted({s["addr"] for s in sites if s["kind"] == "H3"}, key=lambda a: int(a, 16))
    lines.append("WAITCNT_SCAN=%d H1=%d H3=%d insns=%d waits=%d relaxed=%d unattributed=%d"
                 % (len(set12) + len([a for a in set3 if a not in set12]), n1, n3, len(insns),
                    len(waits), len(relaxed), len(res["h1_unattributed"])))
    lines.append('NAN_MUTATE_SUBSET_ADDRS="%s"' % " ".join(set12))
    lines.append('NAN_MUTATE_SUBSET_ADDRS_H3="%s"' % " ".join(set3))
    return lines, sites


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("listing", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--max-sites", type=int, default=50)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.listing:
        ap.error("a listing, or --self-test")
    insns = parse_listing(open(a.listing).read())
    if not insns:
        print("WAITCNT_SCAN=ERROR no instructions parsed from %s" % a.listing)
        return 2
    res = scan(insns)
    lines, sites = report(insns, res, a.max_sites)
    print("\n".join(lines))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"listing": a.listing, "sites": sites, "summary": lines[-3:]}, f, indent=1)
    return 0


# --------------------------------------------------------------------------- #
# self-test fixtures (hand-written listings; addresses are fake but consistent)
# --------------------------------------------------------------------------- #
def _fx(body):
    out, addr = [], 0x1000
    for t in body:
        mn = t.split()[0]
        n = 1 if mn.startswith(("s_", "v_mov_b32_e32", "v_add_f32_e32")) and not mn.startswith(
            ("s_load", "s_buffer_load")) else 2
        word = "BF800000"
        if mn.startswith(("s_branch", "s_cbranch")):
            tgt = int(t.split()[1], 16)
            simm = ((tgt - addr - 4) // 4) & 0xFFFF
            word = "BF82%04X" % simm
            t = mn + " " + str(simm)
        out.append("\t%-50s // %012X: %s" % (t, addr, " ".join([word] + ["00000000"] * (n - 1))))
        addr += 4 * n
    return "\n".join(out)


def self_test():
    fails = []

    def check(name, cond, detail=""):
        print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- %s" % detail) if detail else ""))
        if not cond:
            fails.append(name)

    # 1. straight line: ds_read v[0:3]; ds_read v[4:7]; lgkmcnt(1); read v0 (ok) and v4 (hazard)
    ins = parse_listing(_fx(["ds_read_b128 v[0:3], v20", "ds_read_b128 v[4:7], v21",
                             "s_waitcnt lgkmcnt(1)", "v_add_f32_e32 v8, v0, v1",
                             "v_add_f32_e32 v9, v4, v5", "s_waitcnt lgkmcnt(0)", "s_endpgm"]))
    r = scan(ins)
    check("straight: H1 at the lgkmcnt(1)", list(r["h1"]) == [2], dict(r["h1"]))
    check("straight: only v4's reader", {x[0] for x in r["h1"][2]} == {4}, r["h1"][2])
    # 2. lgkmcnt(0) covers everything
    ins = parse_listing(_fx(["ds_read_b128 v[0:3], v20", "s_waitcnt lgkmcnt(0)",
                             "v_add_f32_e32 v8, v0, v1", "s_endpgm"]))
    check("drained: no H1", not scan(ins)["h1"])
    # 3. out-of-order SMEM: lgkmcnt(1) after s_load + ds_read does NOT retire the s_load
    ins = parse_listing(_fx(["s_load_dword s4, s[0:1], 0x0", "ds_read_b32 v1, v20",
                             "s_waitcnt lgkmcnt(1)", "s_add_u32 s5, s4, 1", "s_waitcnt lgkmcnt(0)",
                             "s_endpgm"]))
    r = scan(ins)
    check("smem: lgkmcnt(1) does not retire an s_load", 2 in r["h1"], dict(r["h1"]))
    # 4. loop-carried: load at the loop bottom, read at the top past a relaxed vmcnt
    ins = parse_listing(_fx(["global_load_dwordx4 v[0:3], v[10:11], off",      # 0x1000 (8B)
                             "s_waitcnt vmcnt(1)",                              # 0x1008  top
                             "v_add_f32_e32 v8, v0, v1",                        # 0x100C
                             "global_load_dwordx4 v[4:7], v[12:13], off",       # 0x1010
                             "global_load_dwordx4 v[0:3], v[10:11], off",       # 0x1018
                             "s_cbranch_scc1 1008",                             # 0x1020 -> top
                             "s_waitcnt vmcnt(0)", "s_endpgm"]))
    r = scan(ins)
    check("loop: H1 at the vmcnt(1) via the back edge", 1 in r["h1"], dict(r["h1"]))
    # 5. WAW within one in-order counter is not a hazard
    ins = parse_listing(_fx(["global_load_dword v1, v[10:11], off", "global_load_dword v1, v[12:13], off",
                             "s_waitcnt vmcnt(0)", "s_endpgm"]))
    check("in-order WAW: no H1", not scan(ins)["h1"])
    print("WAITCNT_SCAN_SELFTEST=%s %d" % ("PASS" if not fails else "FAIL", len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
