#!/usr/bin/env python3
"""Systematic mutations of one hipBLASLt Tensile kernel, for localising a NaN.

Library module -- no torch, no GPU, no side effects at import, the same discipline
as kernel_patch.py.  That is what lets a sweep be enumerated and sanity-checked on a
login node before it costs GPU hours.

--------------------------------------------------------------------------------
What a mutation is

A *mutation* is a named, reproducible alteration of the failing kernel: a
kernel_patch SELECTOR (which instructions) paired with a PAYLOAD (what replaces
them).  replay_nan_op.py --mutate replays the same capsule against each one and
records whether the NaN survived.  A mutation that makes the NaN disappear names
the instructions responsible; one that does not, rules them out.

The registry here is deliberately the third of three, layered on the two in
kernel_patch:

    SELECTORS   which instructions        -- kernel_patch
    PAYLOADS    what replaces them        -- kernel_patch
    MUTATORS    which *combinations* are worth a GPU trial, and why  -- here

Most new ideas are a new combination and need no new primitive.  Add a mutator
here; the driver never changes.

--------------------------------------------------------------------------------
Two rules that are not style preferences

Names must be stable across runs.  The output of a sweep is a table that gets
diffed against last week's, so a mutation's name has to be derived from something
intrinsic -- a vaddr, a mnemonic -- and never from enumeration order.  A name that
shifts when an unrelated mutator is added makes every historical report unreadable.

Fan-out must be bounded, out loud.  Every mutation is one GPU trial of up to
--mutate-timeout seconds -- by default a budget derived from the baseline replay, so
it grows with --iters -- and a mutator that returns 400 sites is a sweep measured in
days.  Cap it with capped() below, which prints what it dropped: a silent truncation
reads as "we covered everything" when it did not.
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_patch                                   # noqa: E402


# ======================================================================================
# the data
# ======================================================================================

@dataclass(frozen=True)
class Mutation:
    """One kernel variant to build and replay.

    Not hashable in practice -- `opts` is a dict -- but frozen anyway, because a
    mutator handing the same Mutation object to two trials and mutating it between
    them is the kind of bug that produces a report nobody can reproduce.
    """

    name: str                       # stable id, e.g. "identity" or "nop@0x1A2C"
    description: str                # one line, printed in the report
    selector: str                   # a kernel_patch SELECTORS key
    payload: str                    # a kernel_patch PAYLOADS key
    opts: dict = field(default_factory=dict)          # selector opts (addrs=, range=, ...)


@dataclass
class MutationSpec:
    """Everything a mutator may look at, all of it from the ORIGINAL kernel.

    Mutators get the unmutated disassembly and nothing else on purpose: a mutator
    that could see the capsule, the GPU or a previous trial's result would make the
    mutation list depend on run order, and the whole point is that the list is a
    function of the kernel alone.
    """

    kernel: str                     # Tensile kernel name
    arch: str                       # gfx942, gfx950, ...
    libdir: str                     # where the Tensile library was found
    insns: list                     # PatchTarget.insns -- the disassembly
    solution_index: object = None   # int, or None if the solution was not pinned
    start: object = None            # kernel symbol vaddr, or None
    stop: object = None             # next symbol vaddr -- start..stop includes the
                                    # inter-kernel padding, which is the only space a
                                    # mutation that INSERTS instructions can grow into
    can_grow: bool = False          # the object holds this one kernel (aiter .co, HSA
                                    # dump), so a growth mutation past the padding is
                                    # delivered by rebuilding it; False for a Tensile
                                    # library, where the bytes after `stop` are another
                                    # kernel's

    @classmethod
    def from_target(cls, target, solution_index=None):
        """Derive the spec from a built kernel_patch target (PatchTarget, AiterTarget,
        HsaTarget)."""
        return cls(kernel=target.kernel, arch=target.arch, libdir=target.libdir,
                   insns=target.insns, solution_index=solution_index,
                   start=target.start, stop=target.stop,
                   can_grow=isinstance(target, kernel_patch.ElfKernelTarget))

    def tail_padding(self):
        """Bytes between the last instruction and the next symbol, or None if unknown.

        A mutator that grows the kernel has to fit in here, and finding out that it does
        not is worth a PatchError at enumeration time -- seconds on a login node --
        rather than an ERROR row an hour into a sweep.
        """
        if self.stop is None or not self.insns:
            return None
        return self.stop - (self.insns[-1].addr + self.insns[-1].size)


# ======================================================================================
# mutator registry
# ======================================================================================

MUTATORS = {}                       # name -> (spec -> list[Mutation])


def register_mutator(name):
    """Mirrors kernel_patch.register_selector / register_payload."""
    def deco(fn):
        if name in MUTATORS:
            raise ValueError("mutator %r registered twice" % name)
        MUTATORS[name] = fn
        return fn
    return deco


def capped(mutations, limit, why, stream=None):
    """The first `limit` mutations, saying out loud what was dropped.

    Exists so that bounding fan-out is easier than not bounding it.  `why` should
    name the ordering, because which ones survive the cap is the whole question --
    "first 20 by vaddr" is reviewable, "first 20" is not.
    """
    mutations = list(mutations)
    if len(mutations) <= limit:
        return mutations
    print("note: %d mutation(s) proposed, keeping %d (%s); dropped %s"
          % (len(mutations), limit, why,
             ", ".join(m.name for m in mutations[limit:limit + 8])
             + (", ..." if len(mutations) > limit + 8 else "")),
          file=stream or sys.stderr)
    return mutations[:limit]


def _validate(m, mutator):
    """Catch a malformed Mutation at enumeration time, not 40 minutes into a sweep."""
    if not isinstance(m, Mutation):
        raise TypeError("mutator %r returned %r, expected a Mutation" % (mutator, type(m)))
    if not m.name or any(c.isspace() for c in m.name):
        raise ValueError("mutator %r produced an unusable name %r "
                         "(non-empty, no whitespace -- it is a report key and a filename)"
                         % (mutator, m.name))
    if m.selector not in kernel_patch.SELECTORS:
        raise ValueError("mutator %r: unknown selector %r (have: %s)"
                         % (mutator, m.selector, ", ".join(sorted(kernel_patch.SELECTORS))))
    if m.payload not in kernel_patch.PAYLOADS and m.payload not in kernel_patch.REGION_PAYLOADS:
        raise ValueError("mutator %r: unknown payload %r (have: %s)"
                         % (mutator, m.payload, ", ".join(kernel_patch.all_payloads())))


def kernel_mutations(spec, only=None):
    """Every mutation every registered mutator proposes for this kernel.

    Registration order, which puts `identity` first: the control should be the first
    trial to run, so broken plumbing surfaces in one trial instead of after the whole
    sweep has been spent producing noise.

    A mutator that raises PatchError is reporting that it has nothing to offer for
    *this* kernel (no FMAs, no split-K, ...); that is a note, not a failure.  A
    mutator that raises anything else is a bug in the mutator and propagates.

    `only` takes mutator names (`lgkmcnt0_each`: every mutation it proposes) and
    mutation names as the enumeration prints them (`lgkmcnt0_each@0x1D090A8`: just
    that one), so a per-site sweep can be split across GPUs.
    """
    picked = {}                      # mutator -> exact mutation names, or None = all
    if only:
        for n in only:
            base = mutator_of(n)
            if n == base:
                picked[base] = None
            elif picked.get(base, set()) is not None:
                picked.setdefault(base, set()).add(n)
        unknown = set(picked) - set(MUTATORS)
        if unknown:
            raise ValueError("unknown mutator(s) %s (have: %s)"
                             % (", ".join(sorted(unknown)), ", ".join(sorted(MUTATORS))))
        wanted = [n for n in MUTATORS if n in picked]
    else:
        wanted = list(MUTATORS)

    out, owner = [], {}
    for name in wanted:
        try:
            proposed = list(MUTATORS[name](spec) or [])
        except kernel_patch.PatchError as e:
            print("note: mutator %r proposed nothing: %s" % (name, e),   # kernel: printed above
                  file=sys.stderr)
            continue
        if picked.get(name):
            missing = picked[name] - set(m.name for m in proposed)
            if missing:
                raise ValueError("mutator %r does not propose %s for this kernel"
                                 % (name, ", ".join(sorted(missing))))
            proposed = [m for m in proposed if m.name in picked[name]]
        for m in proposed:
            _validate(m, name)
            if m.name in owner:
                raise ValueError("duplicate mutation name %r (from %r and %r) -- names "
                                 "are report keys and must be unique"
                                 % (m.name, owner[m.name], name))
            owner[m.name] = name
            out.append(m)
    return out


def mutator_of(name):
    """Which mutator proposed a mutation name, for grouping a report."""
    return name.split("@", 1)[0]


# ======================================================================================
# the mutators
# ======================================================================================

@register_mutator("identity")
def mutate_identity(spec):
    """The control. Patch every site with its own original bytes.

    Keep this forever -- it is not a placeholder.  It exercises the full
    unbundle/patch/rebundle/overlay path and must STILL reproduce the NaN.  If
    identity comes back CLEAN the plumbing is broken, not the kernel, and every
    other row in the report is noise.
    """
    return [Mutation("identity", "unmodified kernel, full overlay path",
                     "all", "identity", {})]


@register_mutator("nans_to_nops")
def mutate_nans_to_nops(spec):
    """Neutralise every NaN constant in the kernel, all at once.

    One trial, not one per site: this asks the coarse question first -- is the NaN
    born from a literal in this kernel at all? -- and answers it for the price of a
    single trial slot.  CLEAN here says yes and hands you the address list;
    NAN rules out every NaN literal in one go and sends you to the store_slice
    selector, which is where a genuine miscompile lives.  Bisecting the sites is
    only worth GPU hours after this row has come back CLEAN.

    `nan_const` is a narrow selector on purpose (it skips integer-typed ops, so
    `v_mul_i32_i24_e32 v3, -16, v3` does not read as a NaN), and `nop` keeps the
    slot width, so the all-at-once patch cannot move a branch target.  What it
    cannot promise is that the destination register becomes *benign*: a NOPed
    `v_mov_b32 v8, -1` leaves whatever v8 held before.  CLEAN therefore means "the
    NaN did not come from these literals", not "the kernel is now correct".

    A stock kernel has no NaN literals -- select_nan_const measures 0 on the clean
    fixture and exactly 5 on the injected one -- so proposing nothing here is the
    expected result on the real bug, which is why it is a PatchError note rather
    than an ERROR row.
    """
    sites = kernel_patch.select_nan_const(spec.insns, spec.arch, {})
    if not sites:
        raise kernel_patch.PatchError(
            "no instruction materialises a NaN constant (0 of %d) -- expected on a "
            "stock kernel; the injected image is what makes this mutator fire"
            % len(spec.insns))
    shown = " ".join("0x%X" % s.addr for s in sites[:8])
    return [Mutation("nans_to_nops",
                     "nop all %d NaN-constant instruction(s) at once (%s%s)"
                     % (len(sites), shown, " ..." if len(sites) > 8 else ""),
                     "nan_const", "nop", {})]


@register_mutator("lgkmcnt0")
def mutate_lgkmcnt0(spec):
    """Force every relaxed LGKM wait to lgkmcnt(0), all at once.

    `s_waitcnt lgkmcnt(N)` with N > 0 is the compiler keeping N LDS/scalar accesses in
    flight while the wave consumes an earlier one -- correct only if those results
    really do return in order and the count really does cover the consumer.  Forcing
    every N to 0 drains the queue at each such site instead.

    The row reads cleanly in one direction, which is the whole reason to spend a slot
    on it: the mutated kernel synchronises at least as much as the original everywhere
    and never less, so NAN is a real negative -- no amount of extra LGKM waiting removes
    this NaN, and the compiler's waitcnt scheduling is not the fault -- and CLEAN says
    the NaN depends on those waits being relaxed, with the site list attached.  Compare
    `nop`-based mutators, where a CLEAN row always carries the caveat that the payload
    also destroyed whatever the instruction computed.

    One trial, like nans_to_nops: this is the coarse question, and bisecting the sites
    is only worth GPU hours once it has come back CLEAN.  Costs nothing in slot width --
    the patch clears four bits of an existing immediate -- so no branch target moves.

    Nothing here is subtle about WHICH instructions qualify; select_lgkm_wait carries
    that argument (a counter at its maximum is printed as absent, not as a number).
    """
    if not kernel_patch.is_gfx9(spec.arch):
        raise kernel_patch.PatchError(
            "the s_waitcnt immediate layout is only encoded for GFX9 here, not %s -- "
            "GFX10 widened lgkmcnt to [13:8] and GFX12 split the instruction into "
            "s_wait_dscnt/s_wait_kmcnt" % spec.arch)
    sites = kernel_patch.select_lgkm_wait(spec.insns, spec.arch, {})
    if not sites:
        raise kernel_patch.PatchError(
            "no s_waitcnt waits on a non-zero lgkmcnt (0 of %d instructions) -- every "
            "LGKM wait in this kernel already drains" % len(spec.insns))
    shown = " ".join("0x%X" % s.addr for s in sites[:8])
    return [Mutation("lgkmcnt0",
                     "force all %d relaxed lgkmcnt wait(s) to lgkmcnt(0) at once (%s%s)"
                     % (len(sites), shown, " ..." if len(sites) > 8 else ""),
                     "lgkm_wait", "lgkmcnt0", {})]


@register_mutator("lgkmcnt0+sbarrier")
def mutate_lgkmcnt0_sbarrier(spec):
    """Drain every relaxed LGKM wait AND rendezvous the workgroup there.

        s_waitcnt lgkmcnt(N)   ->   s_waitcnt lgkmcnt(0)
                                    s_barrier

    The strictly-stronger sibling of `lgkmcnt0`, and the pair is the point.  `lgkmcnt0`
    only makes each wave wait longer for its own LDS traffic; it cannot order one wave's
    writes against another wave's reads.  If a NaN comes from a wave reading an LDS tile
    a sibling has not finished writing, lgkmcnt0 comes back NAN and this one comes back
    CLEAN -- that difference is the finding, and neither row produces it alone.  Run both
    or the sweep answers half the question.

    Three things this costs that `lgkmcnt0` does not:

      * It INSERTS an instruction.  The kernel grows 4 bytes per site and everything
        after the first one moves, so every SOPP branch and every s_getpc_b64
        displacement is re-encoded -- see kernel_patch.relocate_region(), which also
        re-decodes what it wrote.  The growth comes out of the padding before the next
        kernel symbol; past it, a single-kernel object (aiter .co, HSA dump) is rebuilt
        larger, and a Tensile kernel with no room fails here, at enumeration.

      * It is not the safe direction.  Extra lgkm waiting cannot deadlock; a barrier
        can.  If any of these sites sits in control flow some waves of the workgroup
        skip, the mutated kernel hangs and the row is TIMEOUT -- which here means
        "the site is divergent", not "the trial was slow".

      * The before/after diff is long.  Every instruction after the first site changes
        address, so --mutate-disasm diff will anchor its cap and say so.  The first few
        hunks are the real edit; the rest is the shift.

    One trial, like its sibling: the coarse question first.
    """
    if not kernel_patch.is_gfx9(spec.arch):
        raise kernel_patch.PatchError(
            "the s_waitcnt immediate layout and the s_barrier encoding are only known "
            "for GFX9 here, not %s" % spec.arch)
    sites = kernel_patch.select_lgkm_wait(spec.insns, spec.arch, {})
    if not sites:
        raise kernel_patch.PatchError(
            "no s_waitcnt waits on a non-zero lgkmcnt (0 of %d instructions) -- there is "
            "no relaxed wait to put a barrier behind" % len(spec.insns))

    need = 4 * len(sites)
    pad = spec.tail_padding()
    grows = pad is not None and need > pad
    if grows and not spec.can_grow:
        # Not truncated to what fits: a barrier behind SOME of the waits is a different
        # experiment, and one whose CLEAN row would be read as this one's. Measured on
        # the gfx950 libraries here, 45 of 72 kernels have the room and 11 do not -- the
        # ones that do not are the big tiles, with up to 240 relaxed waits.
        #
        # Only a Tensile library stops here.  A single-kernel object (aiter .co, HSA
        # dump) is rebuilt at the larger size, the path the sbarrier_* mutators already
        # took -- 2026-11-23 this refused a 0-byte-padding aiter MLA .co ("0 byte(s) of
        # padding ... would move another kernel") whose sbarrier_* rows rebuilt fine.
        raise kernel_patch.PatchError(
            "%d site(s) need %d more byte(s) but only %d byte(s) of padding follow the "
            "last instruction before the next symbol at 0x%X -- growing past it would "
            "move another kernel. At most %d of the %d site(s) would fit, and barriers "
            "behind a subset is not this mutation"
            % (len(sites), need, pad, spec.stop, pad // 4, len(sites)))

    shown = " ".join("0x%X" % s.addr for s in sites[:8])
    return [Mutation("lgkmcnt0+sbarrier",
                     "drain all %d relaxed lgkmcnt wait(s) to 0 and follow each with "
                     "s_barrier (%s%s); kernel grows %dB%s"
                     % (len(sites), shown, " ..." if len(sites) > 8 else "", need,
                        " -- REBUILDS the code object (single-kernel object)" if grows
                        else "" if pad is None else " into %dB of padding" % pad),
                     "lgkm_wait", "lgkmcnt0+sbarrier", {})]


def _lds_barrier_mutation(spec, name, selector, direction, why,
                          payload, per_site, writes, needs, census):
    """Shared body of the four LDS-boundary barrier mutators.

    The four are two directions (WAR, RAW) crossed with two payload shapes (barrier
    behind a drain, barrier alone), and `payload`/`per_site`/`writes` are everything the
    shapes differ in: which region payload runs, what a site costs in bytes, and what to
    call what it writes.  `census` asks for the lds_drained_before() tally, which only
    means something for the bare-barrier pair -- see their docstrings.
    """
    if not kernel_patch.is_gfx9(spec.arch):
        raise kernel_patch.PatchError(
            "%s only known for GFX9 here, not %s" % (needs, spec.arch))
    sel = kernel_patch.SELECTORS[selector]
    sites = sel(spec.insns, spec.arch, {})
    if not sites:
        raise kernel_patch.PatchError(
            "no unguarded %s boundary (0 of %d instructions) -- every one this kernel "
            "has is already separated by an s_barrier, or it does not use LDS"
            % (direction, len(spec.insns)))

    need = per_site * len(sites)
    pad = spec.tail_padding()
    grows = pad is None or need > pad
    shown = " ".join("0x%X" % s.addr for s in sites[:8])

    # The caveat, carried in the description because that is what --list-mutations prints
    # and what the report stores -- so a NAN row can be read without opening the kernel.
    note = ""
    if census:
        dry = [s for s in sites
               if not kernel_patch.lds_drained_before(spec.insns, s.index)]
        note = ("; %d of %d already drain lgkmcnt(0) first, %d do not (%s%s)"
                % (len(sites) - len(dry), len(sites), len(dry),
                   " ".join("0x%X" % s.addr for s in dry[:8]) or "none",
                   " ..." if len(dry) > 8 else ""))

    return [Mutation(name,
                     "%s: put %s in front of all %d unguarded %s boundar%s (%s%s)%s; "
                     "kernel grows %dB%s"
                     % (why, writes, len(sites), direction,
                        "y" if len(sites) == 1 else "ies", shown,
                        " ..." if len(sites) > 8 else "", note, need,
                        " -- REBUILDS the code object (aiter .co or HSA shim; a "
                        "Tensile kernel cannot grow)" if grows
                        else " into %dB of padding" % pad),
                     selector, payload, {})]


# The two payload shapes, named once so the four mutators below cannot drift apart.
# 8 bytes, not 4, for the drained shape: that barrier is useless without the drain in
# front of it, because s_barrier is a rendezvous and not a wait for outstanding LDS
# traffic.  The bare shape spends 4 and accepts that caveat on purpose -- see
# kernel_patch.region_sbarrier_only.
_DRAINED = dict(payload="sbarrier_before", per_site=8, census=False,
                writes="s_waitcnt lgkmcnt(0) + s_barrier",
                needs="the s_waitcnt immediate layout and the s_barrier encoding are")
_BARE = dict(payload="sbarrier_only", per_site=4, census=True,
             writes="a bare s_barrier (no drain)",
             needs="the s_barrier encoding is")


@register_mutator("lgkmcnt0_each")
def mutate_lgkmcnt0_each(spec):
    """lgkmcnt0 one site at a time: one trial per relaxed LGKM wait.

    Run only after `lgkmcnt0` (all sites at once) came back a powered CLEAN -- it is
    the localisation step, not a screen. Two answers the all-at-once row cannot give:

      * WHICH waits matter -- the minimal change a fix should ship;
      * timing or ordering -- one drained site adds a fraction of the delay six do, so
        a single site that clears the NaN while its siblings at the same loop depth do
        not argues for an ordering hazard at that site over a global slowdown.

    Same payload and slot width as lgkmcnt0, so the same safety argument: each variant
    synchronises at least as much as the original, everywhere.
    """
    if not kernel_patch.is_gfx9(spec.arch):
        raise kernel_patch.PatchError(
            "the s_waitcnt immediate layout is only encoded for GFX9 here, not %s"
            % spec.arch)
    sites = kernel_patch.select_lgkm_wait(spec.insns, spec.arch, {})
    if not sites:
        raise kernel_patch.PatchError(
            "no s_waitcnt waits on a non-zero lgkmcnt (0 of %d instructions)"
            % len(spec.insns))
    # NAN_MUTATE_EACH_ADDRS="0x1D090A8 0x1D090C4 ...": propose only those sites, so a
    # node can split the per-site trials across its GPUs (one sweep per GPU).
    keep = os.environ.get("NAN_MUTATE_EACH_ADDRS", "").split()
    keep = set(int(a, 16) for a in keep)
    return [Mutation("lgkmcnt0_each@0x%X" % s.addr,
                     "force the relaxed wait at 0x%X (%s, site %d of %d) to lgkmcnt(0), "
                     "the others untouched" % (s.addr, s.text, k + 1, len(sites)),
                     "addrs", "lgkmcnt0", {"addrs": [s.addr]})
            for k, s in enumerate(sites) if not keep or s.addr in keep]


@register_mutator("lgkmcnt0_subset")
def mutate_lgkmcnt0_subset(spec):
    """lgkmcnt0 on a SET of sites, and on the rest: two trials.

    Sites from NAN_MUTATE_SUBSET_ADDRS="0x.. 0x.." -- a hazard scan's candidates, or the
    `lgkmcnt0_each` sites whose first NaN came latest.  The question `lgkmcnt0_each`
    cannot answer when every single site still NaNs but all-at-once is CLEAN: do these
    sites together carry the bug?  `lgkmcnt0_subset@set` CLEAN with `@rest` NAN is
    the minimal set step 8 ships (2026-11-20: 3 of 14 waits, 0/1.2M vs the other 11
    157/800k -- that run had to borrow fix_overlay.py --addr; its sibling shipped all
    14).  Same payload as lgkmcnt0: each row synchronises at least as much as stock.
    """
    if not kernel_patch.is_gfx9(spec.arch):
        raise kernel_patch.PatchError(
            "the s_waitcnt immediate layout is only encoded for GFX9 here, not %s"
            % spec.arch)
    sites = kernel_patch.select_lgkm_wait(spec.insns, spec.arch, {})
    want = set(int(a, 16) for a in os.environ.get("NAN_MUTATE_SUBSET_ADDRS", "").split())
    if not want:
        raise kernel_patch.PatchError(
            "lgkmcnt0_subset needs NAN_MUTATE_SUBSET_ADDRS=\"0x<addr> ...\" (sites of %d relaxed "
            "lgkmcnt waits: --list-mutations lgkmcnt0_each)" % len(sites))
    unknown = want - set(s.addr for s in sites)
    if unknown:
        raise kernel_patch.PatchError(
            "NAN_MUTATE_SUBSET_ADDRS %s: not relaxed lgkmcnt waits of this kernel"
            % " ".join("0x%X" % a for a in sorted(unknown)))
    sub = [s.addr for s in sites if s.addr in want]
    rest = [s.addr for s in sites if s.addr not in want]
    out = [Mutation("lgkmcnt0_subset@set",
                    "force the %d relaxed wait(s) %s to lgkmcnt(0), the other %d untouched"
                    % (len(sub), " ".join("0x%X" % a for a in sub), len(rest)),
                    "addrs", "lgkmcnt0", {"addrs": sub})]
    if rest:
        out.append(Mutation("lgkmcnt0_subset@rest",
                            "force the other %d relaxed wait(s) to lgkmcnt(0), the subset %s untouched"
                            % (len(rest), " ".join("0x%X" % a for a in sub)),
                            "addrs", "lgkmcnt0", {"addrs": rest}))
    return out


# --------------------------------------------------------------------------------------
# the vector-memory counter: global-to-LDS loads
#
# A `buffer_load_dword ... lds` writes LDS but retires through the VM queue, so the
# lgkmcnt mutators above never wait for it.  A kernel that refills a double-buffered LDS
# slot this way and reads it after a relaxed `vmcnt(N)` + `s_barrier` can read the
# slot's previous tile.  Same four shapes as lgkmcnt0.
# --------------------------------------------------------------------------------------

def _vm_sites(spec, needs="the s_waitcnt immediate layout is"):
    if not kernel_patch.is_gfx9(spec.arch):
        raise kernel_patch.PatchError(
            "%s only encoded for GFX9 here, not %s -- GFX12 split s_waitcnt into "
            "s_wait_loadcnt/s_wait_storecnt" % (needs, spec.arch))
    sites = kernel_patch.select_vm_wait(spec.insns, spec.arch, {})
    if not sites:
        raise kernel_patch.PatchError(
            "no s_waitcnt waits on a non-zero vmcnt (0 of %d instructions) -- every VM "
            "wait in this kernel already drains" % len(spec.insns))
    return sites


def _vm_census(spec, sites):
    """How many sites a barrier already follows, and how many global-to-LDS loads the
    kernel issues: what a vmcnt0 row means depends on both (barrier_follows)."""
    barred = sum(kernel_patch.barrier_follows(spec.insns, s.index) for s in sites)
    lds_loads = sum(1 for i in spec.insns
                    if "_load_lds" in (i.mnemonic or "")            # global_load_lds_dword
                    or ("_load_" in (i.mnemonic or "") and i.text.rstrip().endswith(" lds")))
    return ("; %d of %d followed by s_barrier; kernel issues %d global-to-LDS load(s)"
            % (barred, len(sites), lds_loads))


@register_mutator("vmcnt0")
def mutate_vmcnt0(spec):
    """Force every relaxed VM wait to vmcnt(0), all at once.

    lgkmcnt0's twin for vector memory, read the same way: the mutated kernel waits at
    least as long as the original everywhere, so NAN rules the relaxed vmcnt waits out
    and CLEAN says the fault depends on them.  Where a barrier follows each site (the
    census in the description), CLEAN is already the shape of the fix: drain the tile's
    global-to-LDS loads, then rendezvous.  Same slot width -- six bits cleared.
    """
    sites = _vm_sites(spec)
    shown = " ".join("0x%X" % s.addr for s in sites[:8])
    return [Mutation("vmcnt0",
                     "force all %d relaxed vmcnt wait(s) to vmcnt(0) at once (%s%s)%s"
                     % (len(sites), shown, " ..." if len(sites) > 8 else "",
                        _vm_census(spec, sites)),
                     "vm_wait", "vmcnt0", {})]


@register_mutator("vmcnt0+sbarrier")
def mutate_vmcnt0_sbarrier(spec):
    """Drain every relaxed VM wait AND rendezvous the workgroup there.

    The stronger sibling of `vmcnt0` (same sites): `vmcnt0` NAN with this CLEAN says the
    loads must be ordered against the OTHER waves' reads, not just retired.  Where the
    census says a barrier already follows every site, this row adds nothing and its
    weaker sibling's verdict carries over.  Grows 4 bytes per site -- an aiter .co or
    the HSA shim rebuilds the object; a Tensile kernel without the padding cannot.
    """
    sites = _vm_sites(spec, "the s_waitcnt immediate layout and the s_barrier encoding are")
    need = 4 * len(sites)
    pad = spec.tail_padding()
    grows = pad is None or need > pad
    shown = " ".join("0x%X" % s.addr for s in sites[:8])
    return [Mutation("vmcnt0+sbarrier",
                     "drain all %d relaxed vmcnt wait(s) to 0 and follow each with "
                     "s_barrier (%s%s)%s; kernel grows %dB%s"
                     % (len(sites), shown, " ..." if len(sites) > 8 else "",
                        _vm_census(spec, sites), need,
                        " -- REBUILDS the code object (aiter .co or HSA shim; a Tensile "
                        "kernel cannot grow)" if grows else " into %dB of padding" % pad),
                     "vm_wait", "vmcnt0+sbarrier", {})]


@register_mutator("vmcnt0_each")
def mutate_vmcnt0_each(spec):
    """vmcnt0 one site at a time -- lgkmcnt0_each's localisation step, after a powered
    CLEAN `vmcnt0`.  NAN_MUTATE_EACH_ADDRS restricts it to those sites (one GPU each)."""
    sites = _vm_sites(spec)
    keep = set(int(a, 16) for a in os.environ.get("NAN_MUTATE_EACH_ADDRS", "").split())
    return [Mutation("vmcnt0_each@0x%X" % s.addr,
                     "force the relaxed wait at 0x%X (%s, site %d of %d%s) to vmcnt(0), "
                     "the others untouched"
                     % (s.addr, s.text, k + 1, len(sites),
                        ", s_barrier follows" if kernel_patch.barrier_follows(spec.insns, s.index)
                        else ""),
                     "addrs", "vmcnt0", {"addrs": [s.addr]})
            for k, s in enumerate(sites) if not keep or s.addr in keep]


@register_mutator("vmcnt_minus1_each")
def mutate_vmcnt_minus1_each(spec):
    """vmcnt(N) -> vmcnt(N-1) at one site: the minimal fix after `vmcnt0_each` localised a
    site (CLEAN there).  CLEAN here = waiting for that one more load suffices -- ship this,
    not the drain; NAN = the late load is older, the drain (or N-2, ... by hand) is needed.
    NAN_MUTATE_EACH_ADDRS restricts it to the localised site(s)."""
    sites = _vm_sites(spec)
    keep = set(int(a, 16) for a in os.environ.get("NAN_MUTATE_EACH_ADDRS", "").split())
    return [Mutation("vmcnt_minus1_each@0x%X" % s.addr,
                     "tighten the wait at 0x%X (%s) by one load: vmcnt(%d) -> vmcnt(%d)"
                     % (s.addr, s.text, kernel_patch.vm_text_count(s),
                        kernel_patch.vm_text_count(s) - 1),
                     "addrs", "vmcnt_minus1", {"addrs": [s.addr]})
            for s in sites if not keep or s.addr in keep]


@register_mutator("vmcnt0_subset")
def mutate_vmcnt0_subset(spec):
    """vmcnt0 on a SET of sites (NAN_MUTATE_SUBSET_ADDRS), and on the rest: lgkmcnt0_subset's
    question for the VM waits -- `@set` CLEAN with `@rest` NAN is the minimal set to ship."""
    sites = _vm_sites(spec)
    want = set(int(a, 16) for a in os.environ.get("NAN_MUTATE_SUBSET_ADDRS", "").split())
    if not want:
        raise kernel_patch.PatchError(
            "vmcnt0_subset needs NAN_MUTATE_SUBSET_ADDRS=\"0x<addr> ...\" (sites of %d relaxed "
            "vmcnt waits: --list-mutations vmcnt0_each)" % len(sites))
    unknown = want - set(s.addr for s in sites)
    if unknown:
        raise kernel_patch.PatchError(
            "NAN_MUTATE_SUBSET_ADDRS %s: not relaxed vmcnt waits of this kernel"
            % " ".join("0x%X" % a for a in sorted(unknown)))
    sub = [s.addr for s in sites if s.addr in want]
    rest = [s.addr for s in sites if s.addr not in want]
    out = [Mutation("vmcnt0_subset@set",
                    "force the %d relaxed wait(s) %s to vmcnt(0), the other %d untouched"
                    % (len(sub), " ".join("0x%X" % a for a in sub), len(rest)),
                    "addrs", "vmcnt0", {"addrs": sub})]
    if rest:
        out.append(Mutation("vmcnt0_subset@rest",
                            "force the other %d relaxed wait(s) to vmcnt(0), the subset %s untouched"
                            % (len(rest), " ".join("0x%X" % a for a in sub)),
                            "addrs", "vmcnt0", {"addrs": rest}))
    return out


@register_mutator("sbarrier_ds_read_write")
def mutate_sbarrier_ds_read_write(spec):
    """Rendezvous the workgroup between an LDS read and the write that follows it.

        ds_read ...                   ds_read ...
        ds_write ...          ->      s_waitcnt lgkmcnt(0)
                                      s_barrier
                                      ds_write ...

    The WAR direction: in a double-buffered pipeline a wave overwrites a tile while a
    sibling may still be reading the previous one.  Nothing in the ISA stops that
    unless the kernel put a barrier there, and this mutation asks what happens when it
    does.

    Only boundaries the kernel does NOT already guard are selected -- the scan resets
    at every existing s_barrier -- so the site count is small and targeted: 17 on the
    gfx950 Aiter v3 backward (out of 814 reads and 112 writes), 4 on the CK forward.

    Read it against `sbarrier_ds_write_read`, the RAW direction.  A CLEAN row on one
    and NAN on the other says which way the hazard runs; a CLEAN on both says the
    kernel needs a rendezvous there but not which access was early.

    Costs, all inherited from inserting an instruction:

      * 8 bytes per site, which on both SDPA kernels is more than the padding (0 and
        8 bytes), so the code object is REBUILT and can only be delivered by the HSA
        shim.  A Tensile or aiter overlay will refuse it rather than serve the
        unpatched kernel.
      * A barrier is not the safe direction.  A site in divergent control flow hangs
        the workgroup, and that row is a TIMEOUT meaning "divergent", not "slow".
      * The before/after diff is long: everything after the first site moves.
    """
    return _lds_barrier_mutation(
        spec, "sbarrier_ds_read_write", "lds_read_then_write",
        "ds_read -> ds_write", "does the NaN need the readers done before the writers",
        **_DRAINED)


@register_mutator("sbarrier_ds_write_read")
def mutate_sbarrier_ds_write_read(spec):
    """The RAW direction: a wave reads an LDS tile a sibling has not finished writing.

        ds_write ...                  ds_write ...
        ds_read ...           ->      s_waitcnt lgkmcnt(0)
                                      s_barrier
                                      ds_read ...

    The classic LDS race, and the one `lgkmcnt0+sbarrier` is reaching for from the
    other end -- that mutator barriers every relaxed wait, this one barriers exactly
    the write->read boundaries, so a CLEAN row here localises the hazard far better.

    Measured: 16 unguarded boundaries on the gfx950 Aiter v3 backward, 8 on
    bwd_hd128_odo_bf16, 0 on the CK forward -- which is itself informative, since a
    kernel with no unguarded write->read boundary cannot have this bug.

    Same costs as its sibling; see sbarrier_ds_read_write.
    """
    return _lds_barrier_mutation(
        spec, "sbarrier_ds_write_read", "lds_write_then_read",
        "ds_write -> ds_read", "does the NaN need the writers done before the readers",
        **_DRAINED)


@register_mutator("sbarrier_only_ds_read_write")
def mutate_sbarrier_only_ds_read_write(spec):
    """The same WAR boundaries, rendezvous only -- no drain alongside it.

        ds_read ...                   ds_read ...
        ds_write ...          ->      s_barrier
                                      ds_write ...

    Read against `sbarrier_ds_read_write`, which patches the identical sites with
    `s_waitcnt lgkmcnt(0) + s_barrier`.  That mutator makes TWO changes at once, and on
    its own a CLEAN row cannot say which one mattered -- the workgroup rendezvous that
    orders this wave against its siblings, or the drain that merely makes the wave
    finish its own outstanding LDS reads first.  Those are different bugs.  The pair
    separates them:

        this        sibling     reading
        --------    --------    ------------------------------------------------
        NAN         NAN         not this hazard, in either sense
        CLEAN       CLEAN       the cross-wave rendezvous is what was missing
        NAN         CLEAN       the drain did the work -- OR the bare barrier was not
                                enough; check the census below before choosing
        CLEAN       NAN         incoherent; suspect an intermittent fault and raise
                                --iters, since one row of the pair is strictly
                                stronger than the other

    The caveat in row three is real and is why the description carries a census.  A wave
    can cross s_barrier with ds_read data still in flight, so at a site with no
    lgkmcnt(0) in front of it this mutation may not close the hazard even where the
    hazard is exactly what the sibling found.  `kernel_patch.lds_drained_before` counts
    those sites and the description names them; "0 of N already drain" is the reading
    where row three says nothing at all.  Sites are still patched either way -- gating
    them would make one mutation two experiments.

    Cheaper than the sibling in the one way that matters to delivery: 4 bytes per site
    instead of 8, so a small site count can land in the kernel's tail padding and be
    served by an ordinary overlay instead of a rebuilt code object.

    Same divergence caveat as every barrier mutation here -- a site some waves skip
    hangs the workgroup, and that row is a TIMEOUT meaning "divergent", not "slow".
    """
    return _lds_barrier_mutation(
        spec, "sbarrier_only_ds_read_write", "lds_read_then_write",
        "ds_read -> ds_write", "does a bare rendezvous suffice before the writers",
        **_BARE)


@register_mutator("sbarrier_only_ds_write_read")
def mutate_sbarrier_only_ds_write_read(spec):
    """The RAW direction, rendezvous only.

        ds_write ...                  ds_write ...
        ds_read ...           ->      s_barrier
                                      ds_read ...

    Everything in `sbarrier_only_ds_read_write` applies, with the pairing read against
    `sbarrier_ds_write_read` instead -- and one asymmetry worth keeping in mind when the
    census comes back "0 of N already drain".  On the WAR side the thing that must have
    finished is a READ, whose data return is what lgkmcnt tracks, so a missing drain
    genuinely weakens the experiment.  On this side it is a WRITE, and a wave whose
    ds_write has issued but not retired can still be overtaken, so the bare barrier is
    weak here for the same reason and not a different one.  Neither direction makes the
    bare form sound on its own; the pair is what makes either readable.

    Measured by the selector: 16 unguarded write->read boundaries on the gfx950 Aiter v3
    backward, 8 on bwd_hd128_odo_bf16, 0 on the CK forward -- and 0 is informative, since
    a kernel with no unguarded boundary in this direction cannot have the bug, which is
    why that case is a PatchError note rather than a row.
    """
    return _lds_barrier_mutation(
        spec, "sbarrier_only_ds_write_read", "lds_write_then_read",
        "ds_write -> ds_read", "does a bare rendezvous suffice before the readers",
        **_BARE)


# ======================================================================================
# shared CLI
# ======================================================================================

def add_mutate_args(ap):
    """The --mutate* flags, defined once so drivers cannot drift.

    Companion to kernel_patch.add_patch_args(), which supplies --patch-kernel,
    --patch-libdir, --patch-arch, --patch-workdir and --keep-workdir; this group
    deliberately does not redefine them.
    """
    g = ap.add_argument_group("kernel mutation sweep")
    g.add_argument("--mutate", action="store_true",
                   help="after the normal replay, replay again against each mutated "
                        "kernel and report which mutations make the NaN go away")
    g.add_argument("--mutate-force", action="store_true",
                   help="sweep even when the baseline replay produced no NaN (implies "
                        "--mutate; off by default). A CLEAN row means nothing then -- "
                        "only a NAN row does, and it proves the fault is intermittent "
                        "rather than absent, so raise --iters when using this")
    g.add_argument("--mutate-only", action="append", default=[], metavar="NAME",
                   help="run only these mutators (repeatable), or single mutations by the "
                        "name the enumeration prints (lgkmcnt0_each@0x1D090A8); the identity control is "
                        "always added, because a sweep without it cannot be read -- "
                        "unless --mutate-no-identity")
    g.add_argument("--mutate-no-identity", action="store_true",
                   help="do not run the identity control. For a sweep whose replay path "
                        "is already known to reproduce (a finished reproducer) and whose "
                        "trials are too dear to spend one on plumbing. A CLEAN row is "
                        "then read without a positive control: only the per-row asm "
                        "read-back shows the patch landed")
    g.add_argument("--list-mutations", action="store_true",
                   help="enumerate the mutations and their site counts, then exit -- no "
                        "GPU and no child processes; with --patch-kernel, no torch either")
    g.add_argument("--grouped-wrong-rows", default=None, metavar="ROWS[,ROWS]",
                   help="aten._grouped_mm: the row count(s) of the wrong group(s) (the replay's "
                        "GROUPED_WRONG_ROWS= line). Kernel discovery then names THAT group's "
                        "solution -- each group is its own GEMM, often its own solution")
    g.add_argument("--mutate-timeout", type=float, default=None, metavar="SECS",
                   help="per-trial wall clock before the child is killed. Default: "
                        "derived from the baseline replay as 2x (setup + per-launch x "
                        "--iters), because what a trial costs scales with --iters and a "
                        "constant cannot -- then re-derived once from what the identity "
                        "control actually cost, since the baseline is timed before the "
                        "first trial and a node that gets busier afterwards would time "
                        "out every row. Pass a number to pin it")
    g.add_argument("--mutate-report", metavar="FILE",
                   help="write the JSON record of the sweep here")
    g.add_argument("--mutate-journal", metavar="FILE",
                   help="append each trial's row here the moment it finishes, as one "
                        "JSON object per line. Defaults to --mutate-report with a "
                        ".jsonl suffix. --mutate-report is only written after the LAST "
                        "trial, so without this a sweep that dies partway loses every "
                        "completed trial; with it the next run reuses them and measures "
                        "only what is missing. Delete a line to force that trial to "
                        "re-run")
    g.add_argument("--mutate-no-resume", action="store_true",
                   help="ignore an existing journal and re-measure every trial. The "
                        "journal is still appended to")
    g.add_argument("--mutate-retry-incomplete", action="store_true",
                   help="when resuming, re-run trials journalled as TIMEOUT or ERROR "
                        "instead of reusing them. They are non-results, so this is the "
                        "flag to pass after raising --mutate-timeout or taking load off "
                        "the node; NAN and CLEAN rows are reused either way")
    g.add_argument("--mutate-solo", action="store_true",
                   help="drop the co-tenants before sweeping, instead of holding them "
                        "for the whole sweep. The old behaviour, and still right for a "
                        "fault that reproduces solo -- load is the documented cause of "
                        "an all-TIMEOUT sweep. WRONG for a contention-gated fault: "
                        "sweeping it solo makes identity come back CLEAN and every "
                        "mutation look like a fix")
    g.add_argument("--mutate-chain-ops", type=int, default=0, metavar="N",
                   help="with --chain, how many ops of the chain to sweep (0 = all, "
                        "the default). The walk runs BACKWARD -- [C1]'s culprit, then "
                        "[C2]'s tally, then upstream -- so N=1 sweeps the culprit "
                        "alone. Ops are deduplicated by dispatched kernel. Prefer "
                        "--mutate-stop-on-finding: it buys the same saving without "
                        "committing to the attribution before the sweep has run")
    g.add_argument("--mutate-stop-on-finding", action="store_true",
                   help="with --chain, stop the backward walk once an op yields a "
                        "finding (a non-UNDERPOWERED CLEAN row on a sweep whose "
                        "identity controls all came back NAN). Since the walk leads "
                        "with the culprit, this usually costs one op instead of the "
                        "whole chain; ops not reached are recorded SKIPPED, never "
                        "dropped. Two ops in one chain CAN be independently faulty, "
                        "so re-run without it if the finding does not explain the bug")
    add_disasm_args(g)
    return g


def add_disasm_args(g):
    """The before/after disassembly flags, for a mutation that REMOVES the NaN.

    On by default, and that is the point.  A CLEAN row names a mutation, not an
    instruction: "nans_to_nops removed it" still leaves you opening the kernel by
    hand to see what the payload actually did to those five slots.  Printing the
    two listings at the moment the finding appears is what makes the row
    self-contained -- and it costs one unbundle per FINDING, not per trial, because
    it only runs on CLEAN.

    Reading the overlay back also catches the failure the table cannot: a CLEAN row
    whose disassembly is identical to the original is a patch that never landed, not
    a kernel that got fixed.
    """
    g.add_argument("--mutate-disasm", choices=("none", "diff", "full"), default="diff",
                   help="when a mutation removes the NaN, show the kernel's ISA before "
                        "and after it: 'diff' (default) prints the changed instructions "
                        "with context, 'full' prints both complete listings (~2300 lines "
                        "each), 'none' skips the read-back entirely")
    g.add_argument("--mutate-disasm-context", type=int, default=4, metavar="N",
                   help="unchanged instructions shown either side of each change in "
                        "--mutate-disasm diff (default 4)")
    g.add_argument("--mutate-disasm-max", type=int, default=40, metavar="N",
                   help="changed instructions anchored per finding before the diff is "
                        "truncated -- out loud, with the dropped addresses "
                        "(default 40; 0 = no cap)")
    g.add_argument("--mutate-keep-fix", metavar="DIR",
                   help="per CLEAN row, copy the patched kernel and an env.sh that loads "
                        "it to DIR/<row>/ (CK via the HSA shim, aiter .co) -- the exact "
                        "bytes step 8 validates; the sweep's own overlay is gone after it")
    g.add_argument("--mutate-asm-dir", metavar="DIR",
                   help="also write NAME.before.s / NAME.after.s / NAME.diff per finding "
                        "here; unlike the per-trial child logs these survive without "
                        "--keep-workdir")
    return g


if __name__ == "__main__":
    # Enumerating needs a kernel, which needs kernel_patch.PatchTarget and a ROCm
    # install; the useful no-dependency check is just that the registry is sane.
    print("%d mutator(s) registered: %s" % (len(MUTATORS), ", ".join(MUTATORS)))
    for _name, _fn in MUTATORS.items():
        print("\n  %-12s %s" % (_name, (_fn.__doc__ or "").strip().split("\n")[0]))
    print("\nselectors: %s" % ", ".join(sorted(kernel_patch.SELECTORS)))
    print("payloads:  %s" % ", ".join(sorted(kernel_patch.PAYLOADS)))
    print("region payloads (may change the instruction count): %s"
          % ", ".join(sorted(kernel_patch.REGION_PAYLOADS)))
