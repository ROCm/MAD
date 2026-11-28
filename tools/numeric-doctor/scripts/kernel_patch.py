#!/usr/bin/env python3
"""Patch one hipBLASLt Tensile kernel instruction at a time, to bisect a NaN.

Library module -- no torch, no GPU, no side effects at import.  The sweep driver is
kernel_patch_sweep.py; the replay scripts import add_patch_args() from here.

Self-test (no ROCm, no GPU, runs on a login node):

    python3 kernel_patch.py --check-fixtures

--------------------------------------------------------------------------------
Reuse

Everything about locating, unbundling, disassembling, patching and overlaying a
Tensile code object already exists in replay_hipblaslt_log_with_nan.py.  That file
is import-safe -- every top-level statement is a def/class/const and the only
top-level `if` is `__name__ == "__main__"` -- so we import it rather than fork
1300 lines of CCOB handling that would then rot.

One hazard: it calls sys.exit() for error handling throughout.  Setup-phase errors
are genuinely fatal and may propagate; every PER-TRIAL call must be wrapped
(`except SystemExit`) or one bad site kills a sweep of hundreds.

--------------------------------------------------------------------------------
The two pluggable registries

SELECTORS  decide WHICH instructions are candidates.
PAYLOADS   decide WHAT gets written in their place.

Adding a new way to patch an instruction is one function plus one decorator; the
driver never changes.  The payload contract is a hard invariant:

    len(payload) == insn.size

Tensile kernels are full of relative branches (~170 labels each), so an edit that
changed any instruction's length would move every branch target after it.  Keeping
the slot width is what makes a binary patch safe without relinking.

--------------------------------------------------------------------------------
REGION_PAYLOADS -- the one documented escape from that invariant

A mutation that INSERTS an instruction cannot keep the slot width, so it cannot be
a PAYLOAD.  It is a REGION_PAYLOAD instead: it rewrites the kernel's whole
[start, stop) byte range at once and is handed the relocation problem the invariant
exists to avoid -- see relocate_region(), which shifts the tail, re-encodes every
PC-relative branch and every s_getpc_b64 displacement, and re-decodes what it wrote
to prove it landed.  The growth is paid for out of the zero padding between the last
instruction and the next kernel symbol (44-180 bytes in the gfx950 libraries here),
so the region's total length -- and therefore every symbol, section and file offset
outside it -- is untouched.

This is strictly more dangerous than a slot-width patch and is meant to stay rare.
Reach for a PAYLOAD first; a REGION_PAYLOAD only when the experiment genuinely needs
an instruction that was not there.
"""

import argparse
import os
import re
import struct
import sys

# Absolute, from this file: the siblings have to resolve when the script is run
# by path from an unrelated working directory, which is how the debug-numeric skill
# invokes them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_hipblaslt_log_with_nan as rnan          # noqa: E402


class PatchError(Exception):
    """A per-trial failure. Never fatal to a sweep."""


# ======================================================================================
# sites
# ======================================================================================

class Site:
    """One candidate instruction, plus why the selector picked it."""

    __slots__ = ("index", "addr", "size", "text", "mnemonic", "enc", "reason")

    def __init__(self, index, insn, reason):
        self.index = index
        self.addr = insn.addr
        self.size = insn.size
        self.text = insn.text
        self.mnemonic = insn.mnemonic
        self.enc = enc_bytes(insn)
        self.reason = reason

    def __repr__(self):
        return "0x%X %-40s %s" % (self.addr, self.text, self.reason)

    def as_dict(self):
        return {"index": self.index, "addr": "0x%X" % self.addr, "size": self.size,
                "text": self.text, "mnemonic": self.mnemonic,
                "enc": self.enc.hex() if self.enc else None, "reason": self.reason}


def enc_bytes(insn):
    """The instruction's ORIGINAL encoding, little-endian, or None.

    rnan.Insn keeps only the byte *count* (_parse_disasm line 557 discards the words
    group), but we need the actual bytes to assert-before-write: if the
    vaddr -> file-offset arithmetic is off, a blind splice silently corrupts an
    unrelated kernel.  Re-parse insn.line rather than modifying rnan.

    llvm-objdump prints amdgcn encodings as big-endian-looking 32-bit words
    ("7E1002C1"); on disk they are little-endian ("C1 02 10 7E").
    """
    m = rnan._AMD_RE.match(insn.line)
    if m:
        out = b""
        for w in m.group("words").split():
            out += struct.pack("<I", int(w, 16))
        return out
    m = rnan._GEN_RE.match(insn.line)
    if m:
        return bytes(int(b, 16) for b in m.group("bytes").split())
    return None


# ======================================================================================
# operand decoding
# ======================================================================================

# A pure integer token: "-1", "64".  Must NOT match "-2.0" or "1e5" -- as f32, -2.0 is
# 0xC0000000 (finite), and mis-reading it as the integer -2 (0xFFFFFFFE) would flag it
# as a NaN.  The kernel really does contain `v_fma_f32 v5, -2.0, v5, 2.0`.
_INT_RE = re.compile(r"^-?\d+$")
_HEX_RE = re.compile(r"^0[xX][0-9a-fA-F]+$")

# Inline float constants are never NaN; listed so they are skipped explicitly rather
# than by accident.
_INLINE_FLOATS = {"0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0", "4.0", "-4.0",
                  "0.15915494", "0.159154943"}


def operand_u32(tok):
    """The 32-bit value an immediate operand encodes, or None if it is not one.

    AMD inline constants -1..-16 are integers 0xFFFFFFFF..0xFFFFFFF0, every one of
    which reads as an f32 NaN.  That is exactly the trick the injected image uses
    (`v_mov_b32_e32 v8, -1`), and also exactly why an unfiltered scan flags address
    math like `v_mul_i32_i24_e32 v3, -16, v3`.
    """
    if not tok:
        return None
    tok = tok.split()[0].strip()
    # strip modifiers objdump may attach, e.g. "v6(0x80000000)" or "-v5"
    tok = tok.rstrip(",")
    if tok in _INLINE_FLOATS:
        return None
    if _HEX_RE.match(tok):
        return int(tok, 16) & 0xFFFFFFFF
    if _INT_RE.match(tok):
        return int(tok) & 0xFFFFFFFF
    return None


def is_f32_nan(v):
    return v is not None and (v & 0x7F800000) == 0x7F800000 and (v & 0x007FFFFF) != 0


# ======================================================================================
# instruction classification
# ======================================================================================

_ENC_SUFFIX = re.compile(r"_(e32|e64|sdwa|dpp|dpp8)$")

# Opcodes that move/select a 32-bit value without interpreting it, so a NaN bit
# pattern in them is a real float NaN heading for the output.
_UNTYPED_MOVES = {"v_mov_b32", "v_accvgpr_write_b32", "v_accvgpr_write",
                  "v_cndmask_b32", "v_pk_mov_b32"}

_FLOAT_TYPED = re.compile(r"_(f16|f32|f64|bf16)\b")
_INT_TYPED = re.compile(r"_(i8|i16|i24|i32|i64|u8|u16|u24|u32|u64)\b")

# NOPing any of these changes control flow or synchronisation rather than a value:
# a removed s_waitcnt races, a removed s_barrier deadlocks divergently, and this
# kernel calls its activation stub via s_swappc_b64 / returns via s_setpc_b64 -- NOP
# either and execution falls through into unrelated code.
#
# This guard is about the `nop` payload specifically -- about DELETING one of these.
# select_lgkm_wait deliberately ignores it, because strengthening an s_waitcnt is the
# safe direction; see its docstring.
SKIP_RE = re.compile(r"^(s_branch|s_cbranch|s_setpc|s_swappc|s_getpc|s_call|s_rfe"
                     r"|s_endpgm|s_barrier|s_waitcnt|s_setreg|s_sendmsg|s_trap)")
_SKIP_DESTS = {"exec", "exec_lo", "exec_hi", "vcc", "vcc_lo", "vcc_hi", "m0", "scc"}


def base_mnemonic(mn):
    return _ENC_SUFFIX.sub("", mn)


def is_unsafe(insn):
    """True if NOPing this would change control flow or synchronisation."""
    if SKIP_RE.match(insn.mnemonic):
        return True
    if insn.operands:
        dst = insn.operands[0].split()[0].strip().lower() if insn.operands[0] else ""
        if dst in _SKIP_DESTS:
            return True
    return False


def writes_float_reg(insn):
    """True if this is a VALU op whose destination is a v/a register holding a float.

    The opcode type matters as much as the destination: `v_mul_i32_i24_e32 v3, -16, v3`
    writes a VGPR from a NaN-patterned constant and is pure address arithmetic.
    """
    mn = insn.mnemonic
    if not (mn.startswith("v_") or mn.startswith("v_accvgpr_write")):
        return False
    if mn.startswith("v_cmp_"):
        return False
    if _INT_TYPED.search(mn):
        return False
    if not (_FLOAT_TYPED.search(mn) or base_mnemonic(mn) in _UNTYPED_MOVES):
        return False
    if not insn.operands:
        return False
    dst = rnan.regs(insn.operands[0])
    return bool(dst) and dst[0] in ("v", "a")


# ======================================================================================
# s_waitcnt immediates
# ======================================================================================
#
# One SOPP immediate packs three independent counters.  GFX9 -- gfx942 and gfx950, the
# only arches this repro runs on:
#
#     [3:0] vmcnt_lo    [6:4] expcnt    [11:8] lgkmcnt    [15:14] vmcnt_hi
#
# A counter held at its MAXIMUM means "do not wait on this class", and llvm-objdump
# omits it from the text.  Both halves of that matter here:
#
#     s_waitcnt lgkmcnt(0)   // BF8CC07F   really vmcnt(63) expcnt(7) lgkmcnt(0)
#     s_waitcnt vmcnt(0)     // BF8C0F70   really vmcnt(0)  expcnt(7) lgkmcnt(15)
#
# so an encoding-driven scan for "lgkm field != 0" would flag that second line and
# force a wait the kernel never had.  Read N from the TEXT, which says what the
# instruction actually waits on.
#
# Then clear the field in the ORIGINAL word rather than assembling a fresh
# `s_waitcnt lgkmcnt(0)`: that is what keeps vmcnt and expcnt intact.  Dropping a vmcnt
# wait while claiming to have changed only the LGKM one would introduce a real memory
# race and make a CLEAN row unattributable -- two edits, one verdict.
#
# GFX10 widened lgkmcnt to [13:8] and GFX12 split s_waitcnt into s_wait_dscnt /
# s_wait_kmcnt, so the mask is arch-gated rather than assumed.

_LGKM_RE = re.compile(r"\blgkmcnt\((\d+)\)")
_SOPP_WAITCNT_HI = 0xBF8C                    # SOPP, op=12, in the top 16 bits
_GFX9_LGKM_MASK = 0x0F00


def is_gfx9(arch):
    """True for the GFX9 family (gfx90a, gfx942, gfx950), whose layout is encoded here."""
    return bool(arch) and arch.startswith("gfx9")


def lgkm_text_count(insn):
    """N from a printed `lgkmcnt(N)`, or None if this insn does not wait on LGKM."""
    if insn.mnemonic != "s_waitcnt":
        return None
    m = _LGKM_RE.search(insn.text)
    return int(m.group(1)) if m else None


def lgkm_enc_count(word, arch):
    """The lgkmcnt field as encoded, for cross-checking against the printed text."""
    if not is_gfx9(arch):
        raise ValueError("s_waitcnt lgkmcnt is only decoded for GFX9 here, not %r" % arch)
    return (word & _GFX9_LGKM_MASK) >> 8


def lgkm_zeroed_word(word, arch):
    """`word` with ONLY its lgkmcnt field cleared; vmcnt, expcnt and opcode survive.

    Raises rather than guessing.  A wrong mask here silently rewrites a vmcnt wait, and
    the resulting race would be read off the report as a property of the kernel.
    """
    if not is_gfx9(arch):
        raise ValueError("s_waitcnt lgkmcnt is only encoded for GFX9 here, not %r -- GFX10 "
                         "widened the field to [13:8] and GFX12 split the instruction "
                         "into s_wait_dscnt/s_wait_kmcnt" % arch)
    if (word >> 16) != _SOPP_WAITCNT_HI:
        raise ValueError("0x%08X is not a GFX9 s_waitcnt (top half 0x%04X, expected 0x%04X)"
                         % (word, word >> 16, _SOPP_WAITCNT_HI))
    return word & ~_GFX9_LGKM_MASK & 0xFFFFFFFF


# vmcnt, the vector-memory counter: split in two on GFX9, [3:0] low and [15:14] high (6
# bits, 63 = "do not wait").  It is the counter that covers a global-to-LDS load
# (`buffer_load_dword ... lds`): the data lands in LDS, but the load retires through the
# VM queue, so `s_waitcnt lgkmcnt(0)` does not wait for it -- a kernel that reads such a
# tile after a relaxed `vmcnt(N)` reads whatever the slot held before.  Same discipline
# as lgkmcnt: N from the text, the field cleared in the instruction's own word.

_VM_RE = re.compile(r"\bvmcnt\((\d+)\)")
_GFX9_VM_MASK = 0xC00F


def vm_text_count(insn):
    """N from a printed `vmcnt(N)`, or None if this insn does not wait on vector memory."""
    if insn.mnemonic != "s_waitcnt":
        return None
    m = _VM_RE.search(insn.text)
    return int(m.group(1)) if m else None


def vm_enc_count(word, arch):
    """The vmcnt field as encoded ([15:14] high, [3:0] low), for the text cross-check."""
    if not is_gfx9(arch):
        raise ValueError("s_waitcnt vmcnt is only decoded for GFX9 here, not %r" % arch)
    return ((word >> 14) & 0x3) << 4 | (word & 0xF)


def vm_zeroed_word(word, arch):
    """`word` with ONLY its vmcnt field (both halves) cleared; lgkmcnt and expcnt survive."""
    if not is_gfx9(arch):
        raise ValueError("s_waitcnt vmcnt is only encoded for GFX9 here, not %r -- GFX10 "
                         "moved vmcnt and GFX12 split the instruction into s_wait_loadcnt "
                         "/ s_wait_storecnt" % arch)
    if (word >> 16) != _SOPP_WAITCNT_HI:
        raise ValueError("0x%08X is not a GFX9 s_waitcnt (top half 0x%04X, expected 0x%04X)"
                         % (word, word >> 16, _SOPP_WAITCNT_HI))
    return word & ~_GFX9_VM_MASK & 0xFFFFFFFF


def vm_set_word(word, n, arch):
    """`word` with its vmcnt field set to `n` (0..62); lgkmcnt and expcnt survive."""
    if not 0 <= n < 63:
        raise ValueError("vmcnt %d out of range (63 means no wait)" % n)
    return vm_zeroed_word(word, arch) | ((n >> 4) & 0x3) << 14 | (n & 0xF)


# ======================================================================================
# PC-relative control flow, and relocating a kernel whose length changed
# ======================================================================================
#
# Everything below exists for ONE reason: a mutation that inserts an instruction moves
# every later instruction, and a Tensile kernel is full of code that knows where it is.
# Two mechanisms encode that knowledge, and both have to be rewritten:
#
#   1. SOPP branches.  s_branch / s_cbranch_* carry a signed 16-bit WORD displacement
#      from the instruction after the branch.  ~130-215 per kernel.
#
#   2. s_getpc_b64 displacement sequences.  Tensile's long jumps and its activation-
#      function calls are built as
#
#          s_getpc_b64 s[A:A+1]          // A:A+1 <- address of the NEXT instruction
#          s_add_i32   sB, <literal>, 4  // 8-byte literal form; the +4 is the sequence's
#        [ s_abs_i32   sB, sB ]          //   own idea of how far the getpc sits from it
#          s_add_u32   sA, sA, sB        // ... or s_sub_u32 for a backward jump
#
#      The literal is an absolute distance baked in at compile time.  10-13 per kernel,
#      and missing one does not fault -- it lands mid-instruction and executes garbage.
#
# An s_getpc_b64 that does not match that shape is a PatchError, not a shrug.  The whole
# value of this module is that it refuses to emit a kernel it cannot account for.

_SOPP_TOP9 = 0x17F                      # bits [31:23] of every GFX9 SOPP
_BRANCH_RE = re.compile(r"^s_(branch|cbranch_[a-z0-9_]+)$")

# PC-relative forms with a displacement this module does not know how to rewrite.
# Tensile does not emit them; if that ever changes, stop rather than relocate blind.
_PCREL_UNHANDLED = re.compile(r"^s_(call_b64|cbranch_[gi]_fork|cbranch_join)$")

# The literal-add that carries a getpc displacement: `s_add_i32 sB, 0x2af8, 4`.
_ADD_LIT_RE = re.compile(r"^s_add_i32$")

_BARRIER_GFX9 = struct.pack("<I", 0xBF8A0000)        # SOPP op=10, s_barrier


def _u32(b, off=0):
    return struct.unpack_from("<I", b, off)[0]


def _s16(v):
    return v - 0x10000 if v & 0x8000 else v


def _s32(v):
    return v - 0x100000000 if v & 0x80000000 else v


def is_sopp(word):
    return (word >> 23) == _SOPP_TOP9


def barrier_bytes(arch, insns=None):
    """The 4-byte `s_barrier` encoding for this arch.

    Lifted from the kernel itself when it already contains one -- a Tensile GEMM has
    ~16 -- because bytes copied out of the very code object we are about to patch
    cannot be wrong about the arch, the way a hand-written constant or a differently
    configured assembler can.  The constant is the last resort, not the first.
    """
    if not is_gfx9(arch):
        raise PatchError("s_barrier is only encoded for GFX9 here, not %r" % arch)
    for ins in insns or ():
        if ins.mnemonic == "s_barrier" and ins.size == 4:
            e = enc_bytes(ins)
            if e and is_sopp(_u32(e)):
                return e
    b = rnan.assemble(arch, "s_barrier")
    if b and len(b) == 4 and is_sopp(_u32(b)):
        return b
    return _BARRIER_GFX9


_WAITCNT_LGKM0_GFX9 = bytes.fromhex("7FC08CBF")      # s_waitcnt lgkmcnt(0)


def waitcnt_lgkm0_bytes(arch, insns=None):
    """The 4-byte `s_waitcnt lgkmcnt(0)` encoding, preferring the kernel's own.

    Same reasoning as barrier_bytes: a word copied out of the code object we are about
    to patch cannot disagree with it about the arch.  The bare text match matters --
    `s_waitcnt vmcnt(0) lgkmcnt(0)` would ALSO drain vector memory, and inserting that
    where the kernel only needs an LDS drain is a second, unattributed change.
    """
    if not is_gfx9(arch):
        raise PatchError("s_waitcnt is only encoded for GFX9 here, not %r" % arch)
    for ins in insns or ():
        if ins.mnemonic == "s_waitcnt" and ins.size == 4 \
                and ins.text.split(None, 1)[-1].strip() == "lgkmcnt(0)":
            e = enc_bytes(ins)
            if e and lgkm_enc_count(_u32(e), arch) == 0:
                return e
    b = rnan.assemble(arch, "s_waitcnt lgkmcnt(0)")
    if b and len(b) == 4 and lgkm_enc_count(_u32(b), arch) == 0:
        return b
    return _WAITCNT_LGKM0_GFX9


def branch_target(insn, enc):
    """Absolute target of a PC-relative SOPP branch, or None if this is not one."""
    if _PCREL_UNHANDLED.match(insn.mnemonic):
        raise PatchError("0x%X %s carries a PC-relative displacement this module cannot "
                         "rewrite -- refusing to relocate a kernel containing it"
                         % (insn.addr, insn.text))
    if not _BRANCH_RE.match(insn.mnemonic):
        return None
    if enc is None or len(enc) != 4:
        raise PatchError("0x%X %s: %s encoding byte(s), expected 4"
                         % (insn.addr, insn.mnemonic, "no" if enc is None else len(enc)))
    word = _u32(enc)
    if not is_sopp(word):
        raise PatchError("0x%X %s encodes 0x%08X, which is not a SOPP -- the branch "
                         "displacement is not where this module thinks it is"
                         % (insn.addr, insn.mnemonic, word))
    return insn.addr + 4 + 4 * _s16(word & 0xFFFF)


def retarget_branch(enc, addr, target):
    """`enc`, re-encoded so a branch sitting at `addr` reaches `target`."""
    delta = target - (addr + 4)
    if delta % 4:
        raise PatchError("branch at 0x%X -> 0x%X is %+d bytes, not a whole number of "
                         "instruction words" % (addr, target, delta))
    simm = delta // 4
    if not -0x8000 <= simm <= 0x7FFF:
        raise PatchError("branch at 0x%X -> 0x%X needs a %+d-word displacement, outside "
                         "the signed 16-bit SOPP field" % (addr, target, simm))
    return struct.pack("<I", (_u32(enc) & 0xFFFF0000) | (simm & 0xFFFF))


class PCRel:
    """One s_getpc_b64 displacement sequence, decoded.

    `add` indexes the `s_add_i32 sB, <lit>, 4` whose literal dword is the thing that
    has to change; `sign` is +1 for an s_add_u32 combine and -1 for s_sub_u32; `absd`
    records an intervening s_abs_i32, which forces the literal's sign rather than
    leaving it free.
    """

    __slots__ = ("getpc", "add", "sign", "absd", "lit", "base", "target")

    def __init__(self, getpc, add, sign, absd, lit, base, target):
        self.getpc, self.add, self.sign, self.absd = getpc, add, sign, absd
        self.lit, self.base, self.target = lit, base, target

    def __repr__(self):
        return "PCRel(getpc=%d add=%d %s%s lit=0x%X base=0x%X -> 0x%X)" % (
            self.getpc, self.add, "+" if self.sign > 0 else "-",
            "abs" if self.absd else "", self.lit & 0xFFFFFFFF, self.base, self.target)


def find_pcrel(insns, g, window=8):
    """Decode the displacement sequence anchored at the s_getpc_b64 at index `g`.

    Raises if the shape is not the one Tensile emits.  That is deliberate: an
    unrecognised PC-relative computation left alone in a relocated kernel is a jump
    into the middle of an instruction, which is indistinguishable at the report level
    from "the mutation changed the answer".
    """
    pc = insns[g]
    if pc.size != 4:
        raise PatchError("0x%X s_getpc_b64 is %dB, expected 4" % (pc.addr, pc.size))
    dst = rnan.regs(pc.operands[0]) if pc.operands else None
    if not dst or dst[0] != "s" or len(dst[1]) != 2:
        raise PatchError("0x%X %s: cannot read the s_getpc_b64 destination pair"
                         % (pc.addr, pc.text))
    lo = dst[1][0]
    base = pc.addr + 4                               # what s_getpc_b64 actually returns

    add_i, lit, absd, scratch = None, None, False, None
    for j in range(g + 1, min(g + 1 + window, len(insns))):
        ins = insns[j]
        mn, ops = ins.mnemonic, ins.operands
        if _ADD_LIT_RE.match(mn) and add_i is None and ins.size == 8 and len(ops) == 3:
            v = operand_u32(ops[1])
            if v is None or operand_u32(ops[2]) != 4:
                continue
            enc = enc_bytes(ins)
            if enc is None or len(enc) != 8 or _u32(enc, 4) != v:
                raise PatchError("0x%X %s: literal %s is not the second dword of %s"
                                 % (ins.addr, ins.text, ops[1],
                                    enc.hex() if enc else "(unparsed)"))
            d = rnan.regs(ops[0])
            if not d or d[0] != "s":
                continue
            add_i, lit, scratch = j, _s32(v), d[1][0]
            continue
        if mn == "s_abs_i32" and add_i is not None and len(ops) >= 2:
            d, s = rnan.regs(ops[0]), rnan.regs(ops[1])
            if d and s and d[1] == [scratch] and s[1] == [scratch]:
                absd = True
            continue
        if mn in ("s_add_u32", "s_sub_u32") and add_i is not None and len(ops) == 3:
            d, a, b = (rnan.regs(o) for o in ops)
            if not (d and a and b):
                continue
            if d[0] == "s" and d[1] == [lo] and a[1] == [lo] and b[1] == [scratch]:
                sign = 1 if mn == "s_add_u32" else -1
                val = lit + 4
                if absd:
                    val = abs(val)
                target = base + val if sign > 0 else base - val
                return PCRel(g, add_i, sign, absd, lit, base, target)
    raise PatchError(
        "0x%X s_getpc_b64 s[%d:%d] is not followed within %d instructions by the "
        "`s_add_i32 sB, <lit>, 4` + s_add_u32/s_sub_u32 displacement sequence this "
        "module knows how to rewrite" % (pc.addr, lo, lo + 1, window))


def retarget_pcrel(enc, pcrel, base, target):
    """The `s_add_i32` bytes, re-encoded so the sequence resolves to `target`."""
    need = (target - base) if pcrel.sign > 0 else (base - target)
    if pcrel.absd:
        # s_abs_i32 throws the literal's sign away, so the magnitude has to carry the
        # whole distance and the direction is fixed by the combine opcode.
        if need <= 0:
            raise PatchError("getpc sequence at 0x%X would need a %+d-byte %s "
                             "displacement, but s_abs_i32 makes it unsigned"
                             % (pcrel.base - 4, need,
                                "forward" if pcrel.sign > 0 else "backward"))
        lit = (need - 4) if (pcrel.lit + 4) >= 0 else (-need - 4)
    else:
        lit = need - 4
    if not -0x80000000 <= lit <= 0x7FFFFFFF:
        raise PatchError("getpc sequence at 0x%X needs literal %d, outside 32 bits"
                         % (pcrel.base - 4, lit))
    return enc[:4] + struct.pack("<I", lit & 0xFFFFFFFF)


# ---- .symtab -------------------------------------------------------------------------
#
# Tensile emits ~190 local label symbols INSIDE each kernel (label_ASM_Start,
# label_GW_End, ...), and llvm-objdump restarts decoding at every one of them.  After a
# relocation those addresses are stale, so a symbol that now lands mid-instruction makes
# objdump decode the rest of the kernel off by a word -- which is how a correctly
# relocated kernel reads back as garbage.
#
# The GPU does not care: dispatch goes through the kernel descriptor, which names the
# one FUNC symbol at `start`, and that one never moves.  But every human-facing thing
# downstream does care -- --mutate-disasm, the before/after .s files a finding saves,
# and the "did the patch actually land" check in replay_nan_op.capture_asm all read
# the overlay back through objdump.  So move the labels with the code: it is 8 bytes per
# symbol, and the alternative is a readback nobody can trust.

_SHT_SYMTAB = 2
_ELF64_SYM = struct.Struct("<IBBHQQ")        # name, info, other, shndx, value, size
_SYM_VALUE_OFF = 8


def _elf64_sections(blob):
    """[(index, sh_type, sh_addr, sh_offset, sh_size, sh_entsize)] from the raw bytes.

    Parsed here rather than shelled out to llvm-readelf because the result is used to
    compute FILE OFFSETS we then write to; text scraping is the wrong tool for that,
    and amdgcn code objects are always little-endian ELF64.
    """
    if blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise PatchError("not a little-endian ELF64 object")
    shoff, shentsize, shnum = (struct.unpack_from("<Q", blob, 0x28)[0],
                               struct.unpack_from("<H", blob, 0x3A)[0],
                               struct.unpack_from("<H", blob, 0x3C)[0])
    if shentsize < 64:
        raise PatchError("section header entry size is %d, expected 64" % shentsize)
    out = []
    for i in range(shnum):
        o = shoff + i * shentsize
        styp = struct.unpack_from("<I", blob, o + 4)[0]
        sa, so, sz = struct.unpack_from("<QQQ", blob, o + 16)
        ent = struct.unpack_from("<Q", blob, o + 56)[0]
        out.append((i, styp, sa, so, sz, ent))
    return out


def symbol_relocations(elf, new_addr, start, stop):
    """{file offset: 8 bytes} moving every label symbol inside [start, stop) with its code.

    Symbols past the last instruction are left where they are -- they sit in padding and
    name nothing.  A symbol inside the code that is NOT on an instruction boundary is a
    refusal, not a shrug: it means the disassembly and the symbol table disagree about
    where instructions begin, and relocating either one on that basis is guesswork.
    """
    with open(elf, "rb") as f:
        blob = f.read()
    secs = _elf64_sections(blob)
    text = [s for s in secs if s[1] == 1 and s[4] and s[2] <= start < s[2] + s[4]]
    if not text:
        raise PatchError("%s: no PROGBITS section contains 0x%X" % (elf, start))
    text_idx = text[0][0]

    out, moved, skipped = {}, 0, 0
    for _i, styp, _sa, so, sz, ent in secs:
        if styp != _SHT_SYMTAB or ent != _ELF64_SYM.size:
            continue
        for o in range(so, so + sz - ent + 1, ent):
            _name, _info, _other, shndx, value, _size = _ELF64_SYM.unpack_from(blob, o)
            if shndx != text_idx or not (start <= value < stop):
                continue
            if value not in new_addr:
                if value < max(new_addr):
                    raise PatchError(
                        "symbol at 0x%X is inside the kernel but not on an instruction "
                        "boundary -- the symbol table and the disassembly disagree about "
                        "where instructions begin" % value)
                skipped += 1          # in the trailing padding; names nothing
                continue
            if new_addr[value] != value:
                out[o + _SYM_VALUE_OFF] = struct.pack("<Q", new_addr[value])
                moved += 1
    return out, moved, skipped


_TAIL_WORDS = (0x00000000, 0xBF800000, 0xBF9F0000)   # zero, s_nop 0, s_code_end


def relocate_region(insns, start, stop, tail, edits, arch, grow_ok=False):
    """Rebuild [start, stop) with `edits` applied, moving whatever they displace.

    `edits` is {instruction index: replacement bytes}, and unlike a PAYLOAD those bytes
    may be LONGER than the slot.  The surplus is taken from the padding between the last
    instruction and `stop`, so the returned blob is exactly stop-start bytes and nothing
    outside the kernel moves -- no symbol, no section, no file offset, and no other
    kernel in the same 43 MB code object.

    `grow_ok` lifts that last guarantee when the padding is not enough: the blob comes
    back LONGER than stop-start and the caller must then rebuild the code object around
    it (grow_code_object).  Padding is still preferred and still used when it suffices,
    because consuming it moves nothing; growing moves every section after .text.  Only
    the single-kernel code objects reached by the HSA shim can be grown -- in a Tensile
    library the bytes after `stop` are another kernel.

    Returns (blob, {old vaddr: new vaddr}).  The map includes a one-past-the-end entry
    for the code, because a Tensile label can sit on the instruction after the last one.

    Everything written is read back and re-decoded before returning.  A relocation that
    is wrong by one word does not crash -- it jumps into the middle of an instruction and
    produces numbers -- so the only honest place to catch it is here, not in the report.
    """
    if not insns:
        raise PatchError("no instructions to relocate")
    code_end = insns[-1].addr + insns[-1].size
    capacity = stop - start
    if capacity <= 0:
        raise PatchError("kernel region 0x%X-0x%X is empty" % (start, stop))

    # Contiguity. objdump dropping an undecodable word would silently put every later
    # instruction at the wrong offset in the rebuilt blob.
    for a, b in zip(insns, insns[1:]):
        if a.addr + a.size != b.addr:
            raise PatchError("disassembly is not contiguous: 0x%X (%dB) is followed by "
                             "0x%X -- cannot rebuild the region" % (a.addr, a.size, b.addr))

    pieces = []
    for i, ins in enumerate(insns):
        if i in edits:
            data = edits[i]
            if not data or len(data) % 4:
                raise PatchError("edit at 0x%X is %s bytes, not a whole number of words"
                                 % (ins.addr, len(data) if data else "0"))
        else:
            data = enc_bytes(ins)
            if data is None or len(data) != ins.size:
                raise PatchError("0x%X %s: cannot recover the original %dB encoding"
                                 % (ins.addr, ins.text, ins.size))
        pieces.append(bytearray(data))

    new_addr, at = {}, start
    for i, p in enumerate(pieces):
        new_addr[insns[i].addr] = at
        at += len(p)
    total = at - start
    new_addr[code_end] = at

    grow = total - (code_end - start)
    overflow = total - capacity
    if overflow > 0 and not grow_ok:
        raise PatchError(
            "the mutated kernel is %d bytes, %d more than the original and %d more than "
            "fits before the next symbol at 0x%X: only %d byte(s) of padding follow the "
            "last instruction. Relocating past it would move another kernel."
            % (total, grow, overflow, stop, capacity - (code_end - start)))

    # Only consume padding we can see is padding.  A non-zero tail is somebody's data.
    if len(tail) != capacity - (code_end - start):
        raise PatchError("tail is %d bytes, expected %d"
                         % (len(tail), capacity - (code_end - start)))
    # When growing, the whole tail is consumed, so the whole tail has to be padding.
    checked = len(tail) if overflow > 0 else grow
    for off in range(0, checked, 4):
        w = _u32(tail, off)
        if w not in _TAIL_WORDS:
            raise PatchError(
                "0x%X holds 0x%08X, which is not padding -- the %d byte(s) this "
                "mutation needs after the last instruction are not free"
                % (code_end + off, w, grow))

    # ---- rewrite everything that knows where it is ------------------------------------
    moved = 0
    for i, ins in enumerate(insns):
        tgt = branch_target(ins, enc_bytes(ins))
        if tgt is None:
            continue
        if i in edits:
            raise PatchError("0x%X: refusing to relocate a branch that is also being "
                             "replaced" % ins.addr)
        if tgt not in new_addr:
            raise PatchError("branch at 0x%X targets 0x%X, which is not an instruction "
                             "boundary in 0x%X-0x%X" % (ins.addr, tgt, start, code_end))
        pieces[i][:] = retarget_branch(bytes(pieces[i]), new_addr[ins.addr], new_addr[tgt])
        moved += 1

    pcrels = []
    for i, ins in enumerate(insns):
        if ins.mnemonic != "s_getpc_b64":
            continue
        pr = find_pcrel(insns, i)
        if pr.target not in new_addr:
            raise PatchError("getpc sequence at 0x%X resolves to 0x%X, which is not an "
                             "instruction boundary in 0x%X-0x%X"
                             % (ins.addr, pr.target, start, code_end))
        if pr.add in edits:
            raise PatchError("0x%X: refusing to relocate a displacement literal that is "
                             "also being replaced" % insns[pr.add].addr)
        base_new = new_addr[ins.addr] + 4
        pieces[pr.add][:] = retarget_pcrel(bytes(pieces[pr.add]), pr,
                                           base_new, new_addr[pr.target])
        pcrels.append(pr)

    # The leftover tail is carried over VERBATIM and at its original address: only the
    # first `grow` bytes -- the ones just checked to be padding -- are overwritten.
    # Re-generating the tail from a filler word instead would silently zero anything
    # further out that is not padding, which is exactly the case the check above lets
    # through.
    blob = bytes(b"".join(bytes(p) for p in pieces) + (tail[grow:] if overflow <= 0
                                                       else b""))
    want = capacity if overflow <= 0 else total
    if len(blob) != want:
        raise PatchError("rebuilt region is %d bytes, expected %d" % (len(blob), want))

    _verify_relocation(insns, pieces, pcrels, blob, start, new_addr, edits)
    return blob, new_addr


def _verify_relocation(insns, pieces, pcrels, blob, start, new_addr, edits):
    """Re-decode the bytes we are about to ship and prove every jump still lands.

    Reads the BLOB, not the pieces, so a mistake in how the pieces were concatenated
    shows up here too.
    """
    for i, ins in enumerate(insns):
        at = new_addr[ins.addr]
        off = at - start
        if blob[off:off + len(pieces[i])] != bytes(pieces[i]):
            raise PatchError("0x%X -> 0x%X: the rebuilt region does not hold the bytes "
                             "we assembled for it" % (ins.addr, at))
        if i in edits:
            continue
        want = branch_target(ins, enc_bytes(ins))
        if want is None:
            continue
        got = at + 4 + 4 * _s16(_u32(blob, off) & 0xFFFF)
        if got != new_addr[want]:
            raise PatchError("branch 0x%X -> 0x%X relocated to 0x%X -> 0x%X, but its "
                             "target moved to 0x%X"
                             % (ins.addr, want, at, got, new_addr[want]))

    for pr in pcrels:
        g, a = insns[pr.getpc], insns[pr.add]
        base = new_addr[g.addr] + 4
        lit = _s32(_u32(blob, new_addr[a.addr] - start + 4))
        val = lit + 4
        if pr.absd:
            val = abs(val)
        got = base + val if pr.sign > 0 else base - val
        if got != new_addr[pr.target]:
            raise PatchError("getpc sequence at 0x%X -> 0x%X relocated to 0x%X -> 0x%X, "
                             "but its target moved to 0x%X"
                             % (g.addr, pr.target, new_addr[g.addr], got,
                                new_addr[pr.target]))


# ======================================================================================
# selector registry
# ======================================================================================

SELECTORS = {}


def register_selector(name):
    def deco(fn):
        SELECTORS[name] = fn
        return fn
    return deco


@register_selector("nan_const")
def select_nan_const(insns, arch, opts):
    """Instructions that materialise a NaN constant into a float register.

    Measured against saved_stuff/add_nan_to_hipblaslt/: 0 hits on the stock kernel,
    exactly 5 on the injected one.  A genuine kernel does not write NaN literals, so
    finding nothing here on the real bug is the expected result, not a failure --
    that is what the store_slice selector is for.
    """
    out = []
    for i, ins in enumerate(insns):
        if not writes_float_reg(ins) or is_unsafe(ins):
            continue
        for tok in ins.operands[1:]:
            v = operand_u32(tok)
            if is_f32_nan(v):
                if nan_const_role(insns, i) is None:
                    out.append(Site(i, ins, "src const %s = 0x%08X is f32 NaN" % (tok, v)))
                break
    return out


_BITWISE = ("v_and_b32", "v_or_b32", "v_xor_b32", "v_bfi_b32", "v_and_or_b32", "v_or3_b32",
            "v_perm_b32", "v_lshl_or_b32", "v_bitop3_b32", "v_bitop3_b16", "v_bfe_u32",
            "v_alignbit_b32", "v_alignbyte_b32")


_STORES = ("buffer_store", "global_store", "flat_store", "scratch_store", "ds_write",
           "ds_store", "buffer_atomic", "global_atomic", "flat_atomic", "ds_add", "s_store",
           "s_buffer_store", "exp")


def nan_const_role(insns, i):
    """Why the NaN-PATTERNED constant an untyped move loads at `i` is not a float, or None.

    An aiter asm kernel's f32 -> bf16 conversion loads 0x7fff0000 (a quiet bf16 NaN in the
    high half) and selects it with `v_cmp_u_f32 s, x, x` + v_cndmask -- NaN PROPAGATION,
    not a planted NaN -- and 0xffff0000 as a bit mask (2026-11-20: 26 such sites in one aiter
    kernel made the stock vLLM image read as "injected").  Decided by the first reader
    of the destination: a bitwise op = a mask; a v_cndmask whose condition comes from an
    is-NaN compare (v_cmp_u_*, v_cmp_class_*) = NaN propagation; a full overwrite before any
    read = dead (that 0xffff0000 is replaced by a v_perm_b32 before the tile is stored -- a
    planted NaN has to reach data to matter, and NOPing a dead load changes nothing; a read
    across a loop back-edge is not followed).  Typed float ops are never
    excused, and anything else (float arithmetic, a store) stays a site, as an injected
    NaN's would.
    """
    ins = insns[i]
    if base_mnemonic(ins.mnemonic) not in _UNTYPED_MOVES or not ins.operands:
        return None
    dst = rnan.regs(ins.operands[0])
    if not dst:
        return None
    mine = set((dst[0], r) for r in dst[1])
    for j in range(i + 1, len(insns)):
        nxt = insns[j]
        # A store has no destination: its first operand is the data it READS.
        store = base_mnemonic(nxt.mnemonic or "").startswith(_STORES)
        srcs = set()
        for op in (nxt.operands if store else nxt.operands[1:]):
            r = rnan.regs(op)
            if r:
                srcs.update((r[0], x) for x in r[1])
        if not srcs & mine:
            d = rnan.regs(nxt.operands[0]) if nxt.operands and not store else None
            if d and mine <= set((d[0], x) for x in d[1]):
                return "dead (overwritten before any read, in program order)"
            continue
        mn = base_mnemonic(nxt.mnemonic)
        if mn in _BITWISE:
            return "bit mask (%s)" % mn
        if _buffer_address_use(nxt, mine):
            return "buffer offset (the vaddr of %s at 0x%X)" % (mn, nxt.addr)
        if mn.startswith("v_cndmask"):
            cond = nxt.operands[-1].strip() if len(nxt.operands) > 3 else "vcc"
            for k in range(j - 1, max(-1, j - 64), -1):
                w = insns[k]
                if w.operands and w.operands[0].strip() == cond:
                    if base_mnemonic(w.mnemonic).startswith(("v_cmp_u_", "v_cmp_class_")):
                        return "NaN propagation (%s selects it)" % base_mnemonic(w.mnemonic)
                    break
            return _oob_offset_role(insns, j)
        return None
    return "dead (never read after the load, in program order)"


def _buffer_address_use(insn, regs_):
    """True if `insn` is a buffer load/store/atomic that reads `regs_` as its VADDR only.

    buffer_* is `<vdata>, <vaddr>, <srsrc>, <soffset>` for loads, stores and atomics
    alike; an operand that is also the data (a store of the same register) is not
    excused.
    """
    if not (insn.mnemonic or "").startswith("buffer_") or len(insn.operands) < 2:
        return False
    va = rnan.regs(insn.operands[1])
    if not va or not regs_ & set((va[0], r) for r in va[1]):
        return False
    vd = rnan.regs(insn.operands[0])
    return not (vd and regs_ & set((vd[0], r) for r in vd[1]))


def _oob_offset_role(insns, j):
    """Why the v_cndmask at `j` selecting a NaN-patterned constant is not a float, or None.

    Tensile's edge guard: `v_mov_b32 v6, 0xfffff000` then, per output element,
    `v_cndmask_b32 vA, v6, vA, s[cmp]` with s[cmp] an integer in-bounds compare, and vA
    the vaddr of the `buffer_store ... offen` -- an out-of-range offset makes the buffer
    resource drop the store.  0xfffff000 is an f32 NaN bit pattern, so it read as a
    planted NaN: 2026-11-23, 25 such sites made a stock pip-built hipBLASLt kernel
    (qwen3_moe gradient) report `injected: yes`.  Decided by the selected value's first
    reader: the vaddr of a buffer op = an offset, not data.  Anything else stays a site.
    """
    sel = insns[j]
    dst = rnan.regs(sel.operands[0]) if sel.operands else None
    if not dst:
        return None
    mine = set((dst[0], r) for r in dst[1])
    for k in range(j + 1, len(insns)):
        nxt = insns[k]
        store = base_mnemonic(nxt.mnemonic or "").startswith(_STORES)
        srcs = set()
        for op in (nxt.operands if store else nxt.operands[1:]):
            r = rnan.regs(op)
            if r:
                srcs.update((r[0], x) for x in r[1])
        if not srcs & mine:
            d = rnan.regs(nxt.operands[0]) if nxt.operands and not store else None
            if d and mine <= set((d[0], x) for x in d[1]):
                return None                      # overwritten unread: not this pattern
            continue
        if _buffer_address_use(nxt, mine):
            return ("out-of-range buffer offset (%s selects it into the vaddr of %s at "
                    "0x%X)" % (base_mnemonic(sel.mnemonic), base_mnemonic(nxt.mnemonic),
                               nxt.addr))
        return None
    return None


@register_selector("store_slice")
def select_store_slice(insns, arch, opts):
    """Transitive backward closure of the registers feeding every output store.

    rnan.pick_victim (line 711) already walks back from a store to the LAST writer of
    its data register; this generalises that to ALL transitive writers, which is the
    set that can carry a NaN into D.  This is the selector for the real bug.
    """
    want = set()                                 # ('v'|'a', reg) live toward a store
    picked, reasons = set(), {}

    # One backward pass with KILL: a definition satisfies the demand for its
    # destination, so registers written here stop being wanted by EARLIER
    # instructions, and this instruction's own sources become wanted instead.
    # Without the kill the demand set only grows and the slice swallows all the
    # address arithmetic that happens to reuse the same VGPRs (785 of 2339 here,
    # vs ~90 with it).
    #
    # This is the straight-line approximation: a definition inside one arm of a
    # branch does not really kill a definition in another arm, so the result can
    # still miss a writer on a path we walked past. Good enough to aim a sweep --
    # the all-sites control tells you if the true cause was outside the slice.
    for i in range(len(insns) - 1, -1, -1):
        ins = insns[i]

        data = rnan.store_data_operand(ins)
        if data:
            want.update((data[0], r) for r in data[1])
            continue

        if not ins.operands:
            continue
        dst = rnan.regs(ins.operands[0])
        if not dst or dst[0] not in ("v", "a"):
            continue
        hits = [(dst[0], r) for r in dst[1] if (dst[0], r) in want]
        if not hits:
            continue

        if ins.mnemonic.startswith("v_") and not is_unsafe(ins):
            picked.add(i)
            reasons[i] = "writes %s, feeds an output store" % ins.operands[0].strip()

        want.difference_update(hits)             # the kill
        for tok in ins.operands[1:]:
            src = rnan.regs(tok)
            if src and src[0] in ("v", "a"):
                want.update((src[0], r) for r in src[1])

    return [Site(i, insns[i], reasons[i]) for i in sorted(picked)]


@register_selector("lgkm_wait")
def select_lgkm_wait(insns, arch, opts):
    """s_waitcnt instructions that wait on a NON-ZERO lgkmcnt.

    The one selector that deliberately ignores is_unsafe().  That guard lists s_waitcnt
    so the `nop` payload cannot DELETE a wait, which would race.  This selector is
    paired with a payload that only ever STRENGTHENS one (N -> 0), the safe direction:
    the mutated kernel synchronises at least as much as the original everywhere, so a
    NaN that survives cannot be blamed on a hazard the mutation introduced.

    Text-driven on purpose -- see the s_waitcnt section above.  `s_waitcnt vmcnt(0)`
    encodes lgkmcnt(15) ("do not wait") and objdump prints no lgkmcnt at all, so it is
    correctly not a site: forcing it to 0 would ADD a wait that was never there, which
    is a different experiment from the one this mutation runs.  Sites are exactly the
    instructions printing lgkmcnt(N) with N > 0.
    """
    out = []
    for i, ins in enumerate(insns):
        n = lgkm_text_count(ins)
        if n:                                    # None (no LGKM wait) and 0 both skip
            out.append(Site(i, ins, "waits lgkmcnt(%d), relaxable to 0" % n))
    return out


@register_selector("vm_wait")
def select_vm_wait(insns, arch, opts):
    """s_waitcnt instructions that wait on a NON-ZERO vmcnt.

    select_lgkm_wait's twin for the vector-memory counter, with the same argument for
    ignoring is_unsafe(): the paired payload only strengthens a wait (N -> 0).  Sites are
    exactly the instructions printing vmcnt(N) with N > 0 -- `s_waitcnt lgkmcnt(0)`
    encodes vmcnt(63) and prints none, so it is not one.
    """
    out = []
    for i, ins in enumerate(insns):
        n = vm_text_count(ins)
        if n:
            out.append(Site(i, ins, "waits vmcnt(%d), relaxable to 0" % n))
    return out


def barrier_follows(insns, index):
    """True if the instruction right after `index` is an s_barrier.

    The census the vmcnt mutators print: a relaxed vmcnt wait that a barrier follows is
    the shape of a cross-wave tile handoff (each wave waits for ITS loads, then the
    workgroup meets), so draining it to vmcnt(0) already is "wait for every load, then
    rendezvous" -- the shape a fix for a global-to-LDS race has.
    """
    return index + 1 < len(insns) and _lds_kind(insns[index + 1]) == "B"


def _lds_kind(insn):
    mn = base_mnemonic(insn.mnemonic or "")
    if mn.startswith("ds_read") or mn.startswith("ds_load"):
        return "R"
    if mn.startswith("ds_write") or mn.startswith("ds_store"):
        return "W"
    if mn.startswith("s_barrier"):
        return "B"
    return None


def _lds_boundaries(insns, first, second):
    """Indices where the LDS access direction flips from `first` to `second` with no
    s_barrier in between.

    Program order only -- this is a syntactic scan, not a dataflow analysis.  It does
    not try to prove the two accesses touch the same LDS addresses, and it cannot: the
    offsets are register-relative.  What it DOES guarantee is that a boundary already
    separated by an s_barrier is not reported, so every site is a place where the
    kernel currently has no workgroup rendezvous between the two directions.
    """
    out, prev = [], None
    for i, ins in enumerate(insns):
        k = _lds_kind(ins)
        if k is None:
            continue
        if k == "B":
            prev = None                      # already synchronised here
            continue
        if prev == first and k == second:
            out.append(i)
        prev = k
    return out


def lds_drained_before(insns, index):
    """True if an `s_waitcnt lgkmcnt(0)` separates `index` from the LDS access before it.

    The question a bare `s_barrier` cannot answer for itself.  s_barrier is a rendezvous:
    it releases once every wave has ARRIVED, which says nothing about whether the LDS
    operations those waves issued have retired.  Dropping a lone barrier in front of an
    LDS access therefore closes a cross-wave hazard only where the wave's own LGKM queue
    is already drained -- so whether it is, at each site, is what decides how a NAN row
    from the bare-barrier mutators may be read.

    Straight-line scan backwards to the nearest LDS access, with the same limitation as
    _lds_boundaries: a drain on another path through the control flow is neither found
    nor needed to be.  Measured on the fixture, 0 of its 2 ds_read -> ds_write boundaries
    are drained -- both carry `s_waitcnt vmcnt(7)`, which waits on vector memory and not
    on LGKM at all, and is exactly the shape a text scan for "a waitcnt is there" would
    get wrong.

    Used to ANNOTATE a mutation, never to gate one.  An undrained site is still patched;
    the mutation's description names it, so the report carries the caveat instead of the
    sweep quietly running a different experiment at some sites than at others.
    """
    for j in range(index - 1, -1, -1):
        if _lds_kind(insns[j]) is not None:
            return False
        if lgkm_text_count(insns[j]) == 0:
            return True
    return False


@register_selector("lds_read_then_write")
def select_lds_read_then_write(insns, arch, opts):
    """The first ds_write after a ds_read, where no s_barrier separates them.

    This is the WAR hazard of a double-buffered LDS pipeline: a wave overwrites a tile
    while a sibling may still be reading the previous one.  Measured on the gfx950
    Aiter v3 backward, this picks 17 sites out of 814 reads and 112 writes -- the scan
    resets at every existing s_barrier, so what is left is exactly the boundaries the
    kernel does NOT already guard.

    Paired with the `sbarrier_before` region payload.  The site is the WRITE, because
    the barrier has to go in front of it.
    """
    return [Site(i, insns[i], "first ds_write after a ds_read, no barrier between")
            for i in _lds_boundaries(insns, "R", "W")]


@register_selector("lds_write_then_read")
def select_lds_write_then_read(insns, arch, opts):
    """The first ds_read after a ds_write, where no s_barrier separates them.

    The RAW direction, and the other half of the pair: a wave reads a tile a sibling
    has not finished writing.  Kept as a separate selector rather than folded in,
    because which direction a CLEAN row implicates is the finding -- one mutation that
    barriered both would not distinguish them.
    """
    return [Site(i, insns[i], "first ds_read after a ds_write, no barrier between")
            for i in _lds_boundaries(insns, "W", "R")]


@register_selector("addrs")
def select_addrs(insns, arch, opts):
    want = set(opts.get("addrs") or [])
    return [Site(i, ins, "explicit --patch-addr") for i, ins in enumerate(insns)
            if ins.addr in want]


@register_selector("range")
def select_range(insns, arch, opts):
    lo, hi = opts["range"]
    return [Site(i, ins, "in 0x%X-0x%X" % (lo, hi)) for i, ins in enumerate(insns)
            if lo <= ins.addr < hi and not is_unsafe(ins)]


@register_selector("filter")
def select_filter(insns, arch, opts):
    rx = re.compile(opts["filter"])
    return [Site(i, ins, "mnemonic matches %s" % opts["filter"])
            for i, ins in enumerate(insns) if rx.search(ins.mnemonic) and not is_unsafe(ins)]


@register_selector("all")
def select_all(insns, arch, opts):
    return [Site(i, ins, "all (safe) instructions") for i, ins in enumerate(insns)
            if not is_unsafe(ins)]


# ======================================================================================
# payload registry
# ======================================================================================

PAYLOADS = {}


def register_payload(name):
    def deco(fn):
        PAYLOADS[name] = fn
        return fn
    return deco


class PatchCtx:
    """What a payload gets to look at."""

    __slots__ = ("insn", "site", "arch")

    def __init__(self, insn, site, arch):
        self.insn, self.site, self.arch = insn, site, arch


@register_payload("nop")
def payload_nop(ctx):
    """Neutralise the instruction. The default."""
    return rnan.nop_bytes(ctx.arch, ctx.insn.size // 4)


@register_payload("identity")
def payload_identity(ctx):
    """Byte-for-byte the original.

    A control: a sweep with this payload must come back completely clean, which
    proves the unbundle/patch/rebundle/overlay path is transparent and that any
    signal from a real payload is the payload and not the plumbing.
    """
    return ctx.site.enc


@register_payload("nan_const")
def payload_nan_const(ctx):
    """Write a NaN into the instruction's destination register.

    The inverse of `nop` -- this is what the injected image does.  It lets the
    selftest synthesise its own ground truth on a STOCK library, so the sweep can be
    validated without depending on the pre-built injected docker image.
    """
    dst = rnan.regs(ctx.insn.operands[0]) if ctx.insn.operands else None
    if not dst:
        raise ValueError("no register destination in %r" % ctx.insn.text)
    payload, _asm = rnan.nan_write(ctx.arch, dst[0], dst[1][0])
    if ctx.insn.size < len(payload) or (ctx.insn.size - len(payload)) % 4:
        raise ValueError("slot at 0x%X is %dB, payload is %dB"
                         % (ctx.insn.addr, ctx.insn.size, len(payload)))
    return payload + rnan.nop_bytes(ctx.arch, (ctx.insn.size - len(payload)) // 4)


@register_payload("lgkmcnt0")
def payload_lgkmcnt0(ctx):
    """Rewrite this s_waitcnt to lgkmcnt(0), leaving vmcnt and expcnt exactly as they were.

    Built by masking the instruction's OWN bytes rather than assembling a fresh
    `s_waitcnt lgkmcnt(0)`: the opcode and the other two counters then survive by
    construction, and the 4-byte slot width is preserved for free rather than asserted
    after the fact.

    The text/encoding cross-check is the point of failure worth having.  Everything
    here rests on lgkmcnt living at bits [11:8]; if that is wrong for the arch in hand,
    the decoded field will not match the N objdump printed, and this raises instead of
    quietly clearing four bits of somebody else's counter.
    """
    insn, site = ctx.insn, ctx.site
    n = lgkm_text_count(insn)
    if not n:
        raise ValueError("%r does not wait on a non-zero lgkmcnt" % insn.text)
    if site.enc is None or len(site.enc) != 4:
        raise ValueError("s_waitcnt at 0x%X has %s encoding byte(s), expected 4"
                         % (insn.addr, "no" if site.enc is None else len(site.enc)))
    word = struct.unpack("<I", site.enc)[0]
    enc_n = lgkm_enc_count(word, ctx.arch)
    if enc_n != n:
        raise ValueError("0x%X: text says lgkmcnt(%d) but encoding 0x%08X holds %d in "
                         "[11:8] -- refusing to patch a field layout we evidently misread"
                         % (insn.addr, n, word, enc_n))
    return struct.pack("<I", lgkm_zeroed_word(word, ctx.arch))


@register_payload("vmcnt0")
def payload_vmcnt0(ctx):
    """Rewrite this s_waitcnt to vmcnt(0), leaving lgkmcnt and expcnt exactly as they were.

    payload_lgkmcnt0's twin: the instruction's own word with the vmcnt field cleared, and
    the same text/encoding cross-check -- vmcnt is split across [15:14] and [3:0], so a
    misread layout shows up as a mismatch here instead of a silently rewritten counter.
    """
    insn, site = ctx.insn, ctx.site
    n = vm_text_count(insn)
    if not n:
        raise ValueError("%r does not wait on a non-zero vmcnt" % insn.text)
    if site.enc is None or len(site.enc) != 4:
        raise ValueError("s_waitcnt at 0x%X has %s encoding byte(s), expected 4"
                         % (insn.addr, "no" if site.enc is None else len(site.enc)))
    word = struct.unpack("<I", site.enc)[0]
    enc_n = vm_enc_count(word, ctx.arch)
    if enc_n != n:
        raise ValueError("0x%X: text says vmcnt(%d) but encoding 0x%08X holds %d in "
                         "[15:14]+[3:0] -- refusing to patch a field layout we evidently "
                         "misread" % (insn.addr, n, word, enc_n))
    return struct.pack("<I", vm_zeroed_word(word, ctx.arch))


@register_payload("vmcnt_minus1")
def payload_vmcnt_minus1(ctx):
    """Tighten this s_waitcnt by ONE vector-memory op: vmcnt(N) -> vmcnt(N-1).

    The smallest strengthening there is: it additionally waits for exactly the oldest load
    vmcnt(N) let through.  After `vmcnt0_each` names the site, this asks whether that one load
    is the late one -- and a CLEAN row is the minimal fix to ship.  Same cross-check
    and slot width as payload_vmcnt0.
    """
    insn, site = ctx.insn, ctx.site
    n = vm_text_count(insn)
    if not n:
        raise ValueError("%r does not wait on a non-zero vmcnt" % insn.text)
    if site.enc is None or len(site.enc) != 4:
        raise ValueError("s_waitcnt at 0x%X has %s encoding byte(s), expected 4"
                         % (insn.addr, "no" if site.enc is None else len(site.enc)))
    word = struct.unpack("<I", site.enc)[0]
    if vm_enc_count(word, ctx.arch) != n:
        raise ValueError("0x%X: text says vmcnt(%d) but encoding 0x%08X holds %d"
                         % (insn.addr, n, word, vm_enc_count(word, ctx.arch)))
    return struct.pack("<I", vm_set_word(word, n - 1, ctx.arch))


def make_payload(kind, insn, site, arch):
    """Build the replacement bytes and enforce the length invariant."""
    fn = PAYLOADS.get(kind)
    if fn is None:
        raise ValueError("unknown payload %r (have: %s)" % (kind, ", ".join(sorted(PAYLOADS))))
    data = fn(PatchCtx(insn, site, arch))
    if data is None or len(data) != insn.size:
        raise ValueError("payload %r produced %s bytes for a %dB slot at 0x%X -- "
                         "length must be preserved or every later branch target moves"
                         % (kind, "None" if data is None else len(data), insn.size, insn.addr))
    return data


# ======================================================================================
# region payload registry
# ======================================================================================
#
# A region payload is handed the whole PatchTarget and every selected site at once, and
# returns the same {file offset: bytes} a PAYLOAD-driven sweep produces -- just one
# entry, covering the kernel's entire [start, stop) range.  That is what lets a mutation
# change the instruction COUNT: see relocate_region() and the module docstring.
#
# They live in a separate registry rather than being flagged inside PAYLOADS so that the
# length invariant in make_payload() stays unconditional.  A payload that can sometimes
# return a different length is a payload nobody can reason about.

REGION_PAYLOADS = {}


def register_region_payload(name, site_mnemonics=()):
    """`site_mnemonics` is what this payload writes at ONE site, in order.

    Declared rather than inferred so --test-overlay can predict the mnemonic stream of
    the rebuilt kernel and compare it against what llvm-objdump reads back out of the
    real code object.  That is the check a region payload most needs and a slot-width
    payload does not: addresses move, so "is anything different at 0x...?" stops being
    answerable and "is this the original stream plus exactly my insertions?" takes over.

    `None` in the tuple stands for the site's ORIGINAL instruction, whose mnemonic is
    not known until the site is picked.  A payload that REPLACES its site lists only
    what it writes ("s_waitcnt", "s_barrier"); one that PREFIXES its site keeps the
    original in the stream and says so ("s_waitcnt", "s_barrier", None).
    """
    def deco(fn):
        if name in REGION_PAYLOADS or name in PAYLOADS:
            raise ValueError("payload %r registered twice" % name)
        fn.site_mnemonics = tuple(site_mnemonics)
        REGION_PAYLOADS[name] = fn
        return fn
    return deco


def all_payloads():
    """Every payload name, slot-width and region alike -- for CLI choices and checks."""
    return sorted(set(PAYLOADS) | set(REGION_PAYLOADS))


def sites_payload_mnemonics(kind):
    """The mnemonics a region payload writes at one site, or None if it did not say."""
    fn = REGION_PAYLOADS.get(kind)
    return list(getattr(fn, "site_mnemonics", ()) or ()) or None


@register_region_payload("lgkmcnt0+sbarrier", ("s_waitcnt", "s_barrier"))
def region_lgkmcnt0_sbarrier(target, sites):
    """Follow every relaxed LGKM wait with an s_barrier, draining it first.

        s_waitcnt lgkmcnt(N)   ->   s_waitcnt lgkmcnt(0)
                                    s_barrier

    N == 0 waits are left alone, which is select_lgkm_wait's job, not this one's.

    The s_waitcnt word is built by payload_lgkmcnt0, so vmcnt and expcnt survive and the
    text/encoding cross-check that guards the field layout applies here unchanged.  What
    is new is the second instruction: the kernel grows by 4 bytes per site and
    relocate_region moves everything after the first one.

    s_barrier is NOT the safe direction that lgkmcnt(N)->0 is.  A wave that reaches a
    barrier its workgroup-mates do not reach hangs the workgroup, so on a site inside
    divergent control flow this mutation TIMEOUTs -- and a TIMEOUT row here means
    "possibly divergent", not "slow".  That asymmetry is the price of asking whether the
    NaN needs a full workgroup rendezvous and not just a longer wait; `lgkmcnt0` on its
    own is the mutation that answers the weaker question cleanly.
    """
    if not is_gfx9(target.arch):
        raise PatchError(
            "the s_waitcnt immediate layout and the s_barrier encoding are only known "
            "for GFX9 here, not %s" % target.arch)
    if not sites:
        raise PatchError("no relaxed lgkmcnt wait to follow with an s_barrier")

    bar = barrier_bytes(target.arch, target.insns)
    edits = {}
    for s in sites:
        insn = target.insns[s.index]
        try:
            wait = payload_lgkmcnt0(PatchCtx(insn, s, target.arch))
        except ValueError as e:                      # per-trial, like every PatchError
            raise PatchError(str(e))
        edits[s.index] = wait + bar

    # May grow past the padding, like vmcnt0+sbarrier and sbarrier_before: a GrowthPatch
    # then, which an aiter .co or the HSA shim rebuilds and the Tensile overlay refuses.
    # This one used to be the only barrier payload pinned to the padding (2026-11-23:
    # refused on a 0-byte-padding aiter MLA .co whose sbarrier_* rows rebuilt fine).
    blob = target.region_bytes()
    code_end = target.insns[-1].addr + target.insns[-1].size
    new, new_addr = relocate_region(target.insns, target.start, target.stop,
                                    blob[code_end - target.start:], edits, target.arch,
                                    grow_ok=True)
    out = {target.file_offset(target.start): new}
    out.update(target.symbol_patches(new_addr))
    grew = len(new) - (target.stop - target.start)
    if grew <= 0:
        return out                                   # fitted in the padding
    return GrowthPatch(out, grew, len(sites))


@register_region_payload("vmcnt0+sbarrier", ("s_waitcnt", "s_barrier"))
def region_vmcnt0_sbarrier(target, sites):
    """Follow every relaxed VM wait with an s_barrier, draining it first.

        s_waitcnt vmcnt(N) ...   ->   s_waitcnt vmcnt(0) ...
                                      s_barrier

    region_lgkmcnt0_sbarrier for the vector-memory counter.  `vmcnt0` makes each wave
    wait for its own global-to-LDS loads; only a barrier after the drain orders those
    loads against a SIBLING wave's ds_read of the same tile.  A site that a barrier
    already follows gets a second, redundant one -- the pair stays comparable site for
    site, and kernel_mutations says how many sites that is.

    May grow past the padding (4 bytes per site), like sbarrier_before: a GrowthPatch
    then, which an aiter .co or the HSA shim rebuilds and a Tensile library cannot.
    Same divergence caveat as every barrier payload: a TIMEOUT row means "possibly
    divergent", not "slow".
    """
    if not is_gfx9(target.arch):
        raise PatchError(
            "the s_waitcnt immediate layout and the s_barrier encoding are only known "
            "for GFX9 here, not %s" % target.arch)
    if not sites:
        raise PatchError("no relaxed vmcnt wait to follow with an s_barrier")

    bar = barrier_bytes(target.arch, target.insns)
    edits = {}
    for s in sites:
        insn = target.insns[s.index]
        try:
            wait = payload_vmcnt0(PatchCtx(insn, s, target.arch))
        except ValueError as e:
            raise PatchError(str(e))
        edits[s.index] = wait + bar

    blob = target.region_bytes()
    code_end = target.insns[-1].addr + target.insns[-1].size
    new, new_addr = relocate_region(target.insns, target.start, target.stop,
                                    blob[code_end - target.start:], edits, target.arch,
                                    grow_ok=True)
    out = {target.file_offset(target.start): new}
    out.update(target.symbol_patches(new_addr))
    grew = len(new) - (target.stop - target.start)
    if grew <= 0:
        return out
    return GrowthPatch(out, grew, len(sites))


class GrowthPatch:
    """A rebuilt code object, for a mutation that would not fit in the padding.

    Returned by a region payload instead of {file_offset: bytes} when the kernel had
    to GROW.  The distinction matters to the caller: a dict can be spliced into any
    overlay, this needs the whole code object rebuilt at a larger size, which only
    works where the object holds a single kernel -- an aiter .co or an
    HSA-intercepted one, not a kernel inside a Tensile library.  An overlay that
    cannot do it must say so rather than silently writing the pre-growth bytes.
    """

    __slots__ = ("blob", "delta", "sites")

    def __init__(self, blob, delta, sites):
        self.blob, self.delta, self.sites = blob, delta, sites


@register_region_payload("sbarrier_before", ("s_waitcnt", "s_barrier", None))
def region_sbarrier_before(target, sites):
    """Put `s_waitcnt lgkmcnt(0)` + `s_barrier` in FRONT of each selected instruction.

        <site>        ->   s_waitcnt lgkmcnt(0)
                           s_barrier
                           <site>

    Both instructions, not just the barrier.  s_barrier is a rendezvous, not a drain:
    it releases when every wave arrives, which says nothing about whether their
    outstanding LDS operations have retired.  Barriering a ds_read -> ds_write
    boundary without the wait would let a wave pass the barrier with reads still in
    flight and then be overtaken by a sibling's write -- i.e. it would not actually
    close the hazard, and a CLEAN row would be unreadable.

    8 bytes per site.  On the kernels measured here that is more than the padding
    (the Aiter v3 backward has 0 bytes, the CK forward 8), so this normally returns a
    GrowthPatch and the code object has to be rebuilt to deliver it -- which both
    bare-code-object backends do, and the hipBLASLt overlay cannot.

    Same divergence caveat as lgkmcnt0+sbarrier: a wave that reaches a barrier its
    workgroup-mates do not reach hangs the workgroup, so a site inside divergent
    control flow TIMEOUTs.  A TIMEOUT row here means "possibly divergent", not "slow".
    """
    if not is_gfx9(target.arch):
        raise PatchError(
            "the s_waitcnt immediate layout and the s_barrier encoding are only known "
            "for GFX9 here, not %s" % target.arch)
    if not sites:
        raise PatchError("no LDS boundary to put a barrier in front of")

    bar = barrier_bytes(target.arch, target.insns)
    wait = waitcnt_lgkm0_bytes(target.arch, target.insns)
    edits = {}
    for s in sites:
        enc = enc_bytes(target.insns[s.index])
        if enc is None:
            raise PatchError("0x%X %s: no encoding in the disassembly"
                             % (s.addr, s.text))
        edits[s.index] = wait + bar + enc           # prefix, then the original

    blob = target.region_bytes()
    code_end = target.insns[-1].addr + target.insns[-1].size
    new, new_addr = relocate_region(target.insns, target.start, target.stop,
                                    blob[code_end - target.start:], edits, target.arch,
                                    grow_ok=True)
    out = {target.file_offset(target.start): new}
    out.update(target.symbol_patches(new_addr))
    grew = len(new) - (target.stop - target.start)
    if grew <= 0:
        return out                                   # fitted in the padding after all
    return GrowthPatch(out, grew, len(sites))


@register_region_payload("sbarrier_only", ("s_barrier", None))
def region_sbarrier_only(target, sites):
    """Put a BARE `s_barrier` in front of each selected instruction -- no drain.

        <site>        ->   s_barrier
                           <site>

    The sibling of sbarrier_before, and the pair is the point.  sbarrier_before writes
    two instructions, and a CLEAN row from it cannot say which one mattered: the
    rendezvous that orders this wave against its siblings, or the `s_waitcnt lgkmcnt(0)`
    that merely makes the wave finish its OWN outstanding LDS traffic first.  Those are
    different bugs with different fixes.  Run both payloads over the same sites and the
    difference between the rows separates them.

    It is also the shape a real fix tends to have.  A compiler-generated pipeline has
    usually already drained before the boundary; what it omitted is the barrier, and a
    patch that adds a second drain alongside it is testing something the source fix does
    not do.

    What this payload gives up, and the reason sbarrier_before bundles the wait: a wave
    can cross s_barrier with ds_read data still in flight, so at a site with no drain in
    front of it the bare barrier may not close the hazard at all.  A NAN row is then
    weak evidence -- it may mean "not this hazard" or "the barrier was not enough".
    lds_drained_before() is how a caller finds out which sites those are; the mutators
    in kernel_mutations.py put the census in the mutation description rather than
    silently patching some sites one way and some the other.

    4 bytes per site, half what sbarrier_before costs, which is what lets a small site
    count land in the kernel's tail padding and come back as a spliceable dict instead
    of a GrowthPatch -- the cheaper delivery, and the only one a Tensile kernel has.

    Same divergence caveat as every barrier-inserting payload here: a wave that reaches
    a barrier its workgroup-mates do not reach hangs the workgroup, so a site inside
    divergent control flow TIMEOUTs, and that row means "possibly divergent", not "slow".
    """
    if not is_gfx9(target.arch):
        raise PatchError("the s_barrier encoding is only known for GFX9 here, not %s"
                         % target.arch)
    if not sites:
        raise PatchError("no LDS boundary to put a barrier in front of")

    bar = barrier_bytes(target.arch, target.insns)
    edits = {}
    for s in sites:
        enc = enc_bytes(target.insns[s.index])
        if enc is None:
            raise PatchError("0x%X %s: no encoding in the disassembly" % (s.addr, s.text))
        edits[s.index] = bar + enc                   # prefix, then the original

    blob = target.region_bytes()
    code_end = target.insns[-1].addr + target.insns[-1].size
    new, new_addr = relocate_region(target.insns, target.start, target.stop,
                                    blob[code_end - target.start:], edits, target.arch,
                                    grow_ok=True)
    out = {target.file_offset(target.start): new}
    out.update(target.symbol_patches(new_addr))
    grew = len(new) - (target.stop - target.start)
    if grew <= 0:
        return out                                   # fitted in the padding after all
    return GrowthPatch(out, grew, len(sites))


# ======================================================================================
# the code object
# ======================================================================================

class _ArchShim:
    """gpu_arch() wants objects with .kernel, so it can fall back to the _ISA tag."""

    def __init__(self, kernel):
        self.kernel = kernel


class PatchTarget:
    """Everything about one kernel that a sweep needs, resolved exactly once.

    The 43 MB device ELF is unbundled once here (KernelFinder.elf_for caches it) and
    the kernel is disassembled once; per-trial work is then only a byte splice plus a
    rebundle.
    """

    def __init__(self, kernel, libdir=None, arch=None, workdir=None):
        import tempfile
        self.kernel = kernel
        self.workdir = workdir or tempfile.mkdtemp(prefix="kernel_patch_")
        self.libdir = rnan.find_library_dir(libdir)
        self.arch = arch or rnan.gpu_arch([_ArchShim(kernel)])
        self.finder = rnan.KernelFinder(self.libdir, self.arch, self.workdir)
        self.finder.index([kernel])
        found = self.finder.locate(kernel)
        if found is None:
            # locate() returns None rather than raising, and unpacking that gives a
            # bare TypeError three frames from anything the caller recognises.  The
            # usual cause is mundane: a kernel name copied from a log written by a
            # DIFFERENT hipBLASLt than the one installed here.
            raise PatchError(
                "kernel %r is not in %s (arch %s) -- no *.dat names it and no code "
                "object exports it. Check --patch-libdir/--patch-arch, or that the "
                "name came from this library's log."
                % (kernel, self.libdir, self.arch))
        self.libfile, self.elf = found
        self.bundled = self.libfile != self.elf
        self.insns = rnan.disassemble(self.finder, self.elf, kernel, self.arch)
        self.start, self.stop = rnan.symbol_range(self.finder, self.elf, kernel)
        self.sh_addr, self.sh_off, _sz = rnan.section_of(self.elf, self.insns[0].addr)
        self._said_symbols = False

    def file_offset(self, addr):
        return addr - self.sh_addr + self.sh_off

    def select(self, name, opts=None):
        fn = SELECTORS.get(name)
        if fn is None:
            raise PatchError("unknown selector %r (have: %s)"
                             % (name, ", ".join(sorted(SELECTORS))))
        return fn(self.insns, self.arch, opts or {})

    def region_bytes(self):
        """The kernel's whole [start, stop) range from disk, checked instruction by
        instruction against the disassembly.

        The same assert-before-write offsets_for() does per site, done for the entire
        region because a region payload rewrites the entire region.  It is also where
        the trailing padding comes from: everything past the last instruction.
        """
        want = self.stop - self.start
        with open(self.elf, "rb") as f:
            f.seek(self.file_offset(self.start))
            blob = f.read(want)
        if len(blob) != want:
            raise PatchError("%s: only %d of the %d byte(s) of kernel 0x%X-0x%X are in "
                             "the file" % (self.elf, len(blob), want, self.start, self.stop))
        for ins in self.insns:
            enc = enc_bytes(ins)
            if enc is None:
                raise PatchError("0x%X %s: no encoding in the disassembly"
                                 % (ins.addr, ins.text))
            off = ins.addr - self.start
            if blob[off:off + ins.size] != enc:
                raise PatchError(
                    "0x%X -> region offset 0x%X holds %s, disassembly said %s -- "
                    "refusing to rebuild the region (vaddr/offset mismatch?)"
                    % (ins.addr, off, blob[off:off + ins.size].hex(), enc.hex()))
        return blob

    def symbol_patches(self, new_addr):
        """The .symtab edits that move this kernel's labels with its code.

        Separate from the region splice only because they land in a different section;
        they go into the same {offset: bytes} dict and through the same write_patched.
        """
        out, moved, skipped = symbol_relocations(self.elf, new_addr, self.start, self.stop)
        # Once per target, not once per trial: a sweep rebuilds the same kernel for every
        # row and the same note on all of them is noise printed into other rows' output.
        if (moved or skipped) and not self._said_symbols:
            self._said_symbols = True
            print("\nnote: %d label symbol(s) move with the code%s. The GPU does not read "
                  "them -- dispatch goes through the kernel descriptor, which names the "
                  "one symbol that does not move -- but llvm-objdump restarts decoding at "
                  "every label, so a stale one makes the read-back disassembly garbage."
                  % (moved,
                     "" if not skipped else " (%d in the trailing padding left alone)"
                     % skipped), file=sys.stderr)
        return out

    def offsets_for(self, sites, kind):
        """{file_offset: replacement bytes}, verifying the original bytes first.

        rnan.write_patched (line 840) is a blind splice. If the
        vaddr -> file-offset arithmetic is wrong by even a section, it would happily
        corrupt an unrelated kernel and the sweep would report nonsense. Checking the
        bytes we are about to overwrite against what the disassembler reported costs
        one seek per site.

        A REGION payload is handed the sites and does its own verification over the
        whole range -- it has to, because it rewrites bytes no site names.
        """
        if kind in REGION_PAYLOADS:
            return REGION_PAYLOADS[kind](self, sites)
        out = {}
        with open(self.elf, "rb") as f:
            for s in sites:
                off = self.file_offset(s.addr)
                f.seek(off)
                cur = f.read(s.size)
                if s.enc and cur != s.enc:
                    raise PatchError(
                        "0x%X -> file offset 0x%X holds %s, disassembly said %s -- "
                        "refusing to patch (vaddr/offset mismatch?)"
                        % (s.addr, off, cur.hex(), s.enc.hex()))
                out[off] = make_payload(kind, self.insns[s.index], s, self.arch)
        return out

    def analysis(self, offsets):
        """A duck-compatible rnan.KernelAnalysis, for build_overlay()."""
        a = rnan.KernelAnalysis(self.kernel)
        a.libfile, a.elf = self.libfile, self.elf
        a.bundled = self.bundled
        a.insns, a.start, a.stop = self.insns, self.start, self.stop
        a.patches = dict(offsets)
        return a


def _refuse_growth(offsets, who):
    """A GrowthPatch cannot be delivered out of a Tensile library.

    Refusing is the whole point: an overlay that quietly wrote the pre-growth bytes
    would serve a kernel missing the very instructions the mutation is about, and the
    row would read as a clean, well-controlled result.

    The two bare-code-object backends DO deliver growth, by rebuilding the object
    (rebuild_grown_object).  What makes them different is not the overlay mechanism
    but what the file holds: a .co or an intercepted code object is ONE kernel, and
    the bytes after it are its own section's padding.  A Tensile kernel shares a fat
    binary with hundreds of others, so the bytes after it are the next kernel's code
    and growing into them would move code that nothing re-points at.
    """
    if isinstance(offsets, GrowthPatch):
        raise PatchError(
            "this mutation grows the kernel by %d byte(s) (%d site(s)), which %s "
            "cannot deliver -- the bytes after this kernel are the next one in the "
            "same library. A single-kernel code object can be rebuilt at a larger "
            "size, a Tensile library cannot; reach the kernel as an aiter .co "
            "(--patch-asm-dir) or through the HSA shim (--patch-co) instead."
            % (offsets.delta, offsets.sites, who))


def tensile_libpath(overlay, arch=None):
    """What HIPBLASLT_TENSILE_LIBPATH must name for a library overlay.  hipBLASLt does NOT append
    the arch to a path given this way: with the per-arch layout of the pip ROCm SDK wheels
    (library/gfx950/*.co) the variable must name <overlay>/<arch> -- naming the root, every GEMM
    failed HIPBLAS_STATUS_INVALID_VALUE ("hipModuleLoad failed: <overlay>/Kernels.so-...",
    2026-11-21).  A flat library: the overlay itself."""
    if arch is None:
        subs = [d for d in os.listdir(overlay) if d.startswith("gfx") and os.path.isdir(os.path.join(overlay, d))]
        flat = any(f.endswith((".co", ".hsaco", ".dat", ".dat.zlib")) for f in os.listdir(overlay))
        if len(subs) and not flat:
            try:                                          # the device this process will run on
                import torch
                arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            except Exception:                             # noqa: BLE001
                arch = subs[0] if len(subs) == 1 else None
    if arch and os.path.isdir(os.path.join(overlay, arch)):
        return os.path.join(overlay, arch)
    return overlay


class OverlayCache:
    """A shadow library dir built once; per trial only the patched .co is swapped.

    build_overlay() symlinks ~3000 library files and that cost should be paid once,
    not per site. Each concurrent worker needs its OWN cache: two trials sharing an
    overlay would race on the same .co.
    """

    def __init__(self, target, work):
        self.target, self.work = target, work
        self.overlay, patched = rnan.build_overlay(
            target.libdir, [target.analysis({})], target.arch, work)
        self.rel = list(patched)[0]
        self.dst = os.path.join(self.overlay, self.rel)

    def apply(self, offsets):
        """Rebuild the patched .co and swap it into the overlay atomically."""
        import shutil
        _refuse_growth(offsets, "the hipBLASLt overlay")
        elf_out = os.path.join(self.work, "trial.elf")
        co_out = os.path.join(self.work, "trial.co")
        rnan.write_patched(self.target.elf, offsets, elf_out)
        if self.target.bundled:
            targets = rnan.bundle_targets(self.target.libfile)
            dev = next(t for t in targets
                       if "amdgcn" in t and self.target.arch in t)
            rnan.rebundle(self.target.libfile, dev, elf_out, targets, self.work, co_out)
        else:
            shutil.copy2(elf_out, co_out)
        os.replace(co_out, self.dst)
        return self.overlay

    def env(self, base=None):
        # `base or os.environ` would treat an empty dict as "use the whole environment".
        e = dict(os.environ if base is None else base)
        e["HIPBLASLT_TENSILE_LIBPATH"] = e["TENSILE_LIBPATH"] = tensile_libpath(self.overlay, self.target.arch)
        return e

    def disassemble(self):
        """Re-disassemble the kernel as it currently sits in the overlay.

        The bytes a child actually ran, not the bytes we meant to write: this goes
        back out through unbundle + llvm-objdump, so a payload that decoded to
        something other than what the mutator intended shows up here instead of
        being taken on trust.  That is the whole value of reading it back rather
        than rendering the patch dict.

        Valid only until the next apply().  There is ONE .co in the overlay and
        every trial overwrites it, so a caller that wants a mutated listing has to
        take it before the next trial starts.
        """
        import shutil
        import tempfile
        t = self.target
        d = tempfile.mkdtemp(dir=self.work)
        try:
            elf = self.dst
            if t.bundled:
                elf, _dev, _targets = rnan.unbundle_device(self.dst, t.arch, d)
            return rnan.disassemble_range(elf, t.start, t.stop, t.arch)
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ======================================================================================
# aiter: the same sweep against a standalone .co
# ======================================================================================
#
# An op that dispatches to aiter -- fused attention on gfx950 is the case that forced
# this -- does not go through hipBLASLt at all, so none of the machinery above can
# find it: there is no problem key, no *.dat naming a solution, and the hipBLASLt
# profile a discovery run scrapes comes back EMPTY rather than wrong.  Before this
# existed the sweep said so honestly and emitted SKIPPED, which is a real verdict but
# ends the investigation one rung short of the kernel it just localised.
#
# The good news is that an aiter kernel is a SIMPLER target, not a harder one.  A
# Tensile kernel arrives bundled inside a 43 MB fat binary and most of PatchTarget /
# OverlayCache is the unbundle/rebundle dance.  An aiter kernel is already a bare
# AMD HSA ELF on disk -- one .text, one exported FUNC, a .symtab full of local labels
# -- which is exactly what unbundling a Tensile library PRODUCES.  So the two classes
# below are the hipBLASLt ones with the hard part deleted, and everything downstream
# (every selector, every payload, relocate_region, MutationSpec, the trial loop, the
# disasm diff, the CLEAN-row power calc) is reused untouched: it all keys off
# .insns/.arch/.start/.stop and never asks where the bytes came from.
#
# ------------------------------------------------------------------------------------
# THE TRAP, and it is the whole reason this code has to be careful
#
# aiter loads its kernels one of two ways (csrc/include/aiter_hip_common.h,
# load_asm_kernel):
#
#     if (getenv("AITER_ASM_DIR"))   hipModuleLoad    (<dir>/<arch>/<hsaco>)   # FILE
#     else                           hipModuleLoadData(AITER_EMBEDDED_HSA_MAP)  # BAKED IN
#
# The default in a torch process is the SECOND one.  aiter's Python sets the variable
# at import (jit/core.py), but torch reaches aiter through the C++ extension and never
# imports that package, so the kernel served to the GPU is a copy compiled into the
# .so -- not the .co on disk.
#
# Patch the file under those conditions and NOTHING CHANGES.  Every mutator comes back
# CLEAN, including ones that gutted the kernel, and the table reads like a sweep that
# found the fault everywhere.  There is no error and no warning; the only tell is that
# the loader's own trace prints the unprefixed relative form.
#
# So AiterOverlay.env() sets AITER_ASM_DIR, and the caller MUST also set it for the
# BASELINE (see ensure_aiter_asm_dir) -- otherwise the baseline measures the embedded
# blob and the trials measure the overlay, and the comparison is between two different
# binaries.  trace_is_file_backed() below is the assertion that it worked.

_AITER_LOAD_RE = re.compile(
    r"^\[aiter\]\s+hipModuleLoad:\s+(?P<path>\S+)\s+GetFunction:\s+(?P<sym>\S+)")
# amd-aiter 0.1.26+ (vllm/vllm-openai-rocm v0.33.0) logs the same load as
#     [aiter] LoadKernel: <symbol> hsaco: /abs/.../aiter_meta/hsa//gfx950/mla/<name>.co
# File-backed only when the path is absolute: a relative one is the embedded map's key.
# Without this form discovery saw no aiter kernel at all and the ladder's kernel id fell
# back to a hipBLASLt GEMM elsewhere in the chain (2026-11-20).
_AITER_LOADKERNEL_RE = re.compile(
    r"^\[aiter\]\s+LoadKernel:\s+(?P<sym>\S+)\s+hsaco:\s+(?P<path>/\S+)")


def parse_aiter_loads(text):
    """[(co_path, symbol)] for every FILE-backed aiter module load in `text`.

    Deliberately matches ONLY the "[aiter] " prefixed form.  The unprefixed line is
    the embedded branch, and treating it as a dispatch record would hand the sweep a
    path it is not actually running -- the exact failure this module exists to avoid.
    Order is preserved and duplicates kept: which kernel loaded FIRST is how a chain
    op is attributed.
    """
    out = []
    for line in text.splitlines():
        m = _AITER_LOAD_RE.match(line.strip()) or _AITER_LOADKERNEL_RE.match(line.strip())
        if m:
            out.append((os.path.normpath(m.group("path")), m.group("sym")))
    return out


def trace_is_file_backed(text):
    """Did this run load aiter kernels from files, or from the embedded map?

    Returns True only on positive evidence of the file branch.  A run that loaded no
    aiter kernel at all returns None -- "no aiter here" and "aiter served the baked-in
    copy" are different answers and only the second one invalidates a sweep.
    """
    if parse_aiter_loads(text) or "[aiter] hipModuleLoad:" in text:
        return True
    if "hipModuleLoad:" in text or "[aiter] LoadKernel:" in text:
        return False
    return None


def served_overlay(out, overlay, aiter=False, shim_log=None):
    """Did a trial child actually run the patched kernel?

    On an aiter sweep the answer is always True or False -- never None.  That is the
    whole point: the trial exists to measure a patched .co, so "the child never
    loaded it" is a FAILED trial, not an inapplicable question.  None is reserved for
    a Tensile sweep, whose overlay is consumed by the library rather than announced
    and so cannot be checked this way.

    This exists because of a near-miss worth stating plainly: the overlay was built,
    the .co was patched, AITER_ASM_DIR reached the child -- and the child still ran a
    DIFFERENT backend, because the environment that selects the CK/Aiter path had not
    been restored.  Both rows came back NAN and looked like results about the patched
    kernel; identity_ok was True.  A real sweep and a wholly fake one produced the
    SAME table.  The child printed no aiter trace at all, which is why "no trace" has
    to count as False here rather than as "not applicable".

    Lives here rather than in the replay driver so it is import-safe and unit
    testable: replay_nan_op parses argv at import.

    `shim_log` is the CK/HSA path: the evidence is not a loader trace in the child's
    stdout but the shim's own log, and the question is the same one -- did the patch
    reach the GPU, or did this trial measure the stock kernel?
    """
    if shim_log is not None:
        return shim_applied_patches(shim_log)

    loads = parse_aiter_loads(out)
    if not loads:
        if aiter:
            return False
        # Not an aiter sweep: "aiter ran out of its embedded map" is still worth
        # catching, but silence just means this sweep is not about aiter at all.
        return False if trace_is_file_backed(out) is False else None
    root = os.path.realpath(overlay)
    hit = any(os.path.realpath(p).startswith(root) for p, _s in loads)
    return hit if (hit or aiter) else None


def _elf64_phdrs(blob):
    """[(index, p_type, p_offset, p_vaddr, p_filesz, p_memsz, p_align)]."""
    phoff = struct.unpack_from("<Q", blob, 0x20)[0]
    phentsize = struct.unpack_from("<H", blob, 0x36)[0]
    phnum = struct.unpack_from("<H", blob, 0x38)[0]
    if phentsize < 56:
        raise PatchError("program header entry size is %d, expected 56" % phentsize)
    out = []
    for i in range(phnum):
        o = phoff + i * phentsize
        ptype = struct.unpack_from("<I", blob, o)[0]
        poff, pvaddr = struct.unpack_from("<QQ", blob, o + 8)
        pfilesz, pmemsz = struct.unpack_from("<QQ", blob, o + 32)
        palign = struct.unpack_from("<Q", blob, o + 48)[0]
        out.append((i, ptype, poff, pvaddr, pfilesz, pmemsz, palign))
    return out


def grow_code_object(blob, region_off, old_len, new_region):
    """A code object with [region_off, region_off+old_len) replaced by a LONGER blob.

    Only for single-kernel objects -- the ones the HSA shim hands us.  In a Tensile
    library the bytes after the kernel are another kernel and none of this is safe.

    What has to move, and why each one matters:

      * the code grows by `delta` in VADDR, so the PT_LOAD that contains it gets
        p_filesz/p_memsz += delta.  If that would run into the next segment's vaddr we
        refuse -- silently overlapping two segments produces a kernel that loads and
        computes garbage, which is the worst possible outcome for a sweep;
      * every section and segment FILE offset after the region shifts.  The shift is
        rounded up to the segment alignment, because ELF requires
        p_offset == p_vaddr (mod p_align) for PT_LOAD and a shift that breaks that
        congruence is rejected by the loader;
      * e_shoff moves with the section headers.

    What deliberately does NOT move: the kernel's entry address.  Insertions all land
    after the first instruction, so the entry keeps its vaddr and the kernel descriptor
    (.kd), which stores the entry as an offset from itself, stays valid untouched.

    Symbol values inside the grown region are NOT handled here -- apply
    symbol_patches() to `blob` BEFORE calling this, while .symtab is still at its
    original offset.
    """
    delta = len(new_region) - old_len
    if delta < 0:
        raise PatchError("grow_code_object cannot shrink (%d bytes)" % delta)
    if delta == 0:
        return blob[:region_off] + new_region + blob[region_off + old_len:]
    if delta % 4:
        raise PatchError("growth of %d bytes is not a whole number of words" % delta)

    out = bytearray(blob)
    secs = _elf64_sections(blob)
    phdrs = _elf64_phdrs(blob)
    region_end = region_off + old_len

    align = max([p[6] for p in phdrs if p[1] == 1 and p[6]] or [4096])
    fshift = ((delta + align - 1) // align) * align

    # -- the section holding the region grows; everything after it slides ------------
    host = None
    for (i, _t, sa, so, sz, _e) in secs:
        if so <= region_off and region_end <= so + sz and sz:
            host = (i, sa, so, sz)
            break
    if host is None:
        raise PatchError("no section contains the region at file offset 0x%X"
                         % region_off)
    hi, h_addr, h_off, h_size = host

    # The region has to run to the END of its section.  If it did not, the bytes
    # between it and the section end would belong to that section and would have to
    # slide by `delta`, while everything past the section slides by `fshift` -- two
    # different shifts this function does not implement.  Both backends reach here
    # with a whole-.text region (symbol_range falls back to the section end), so this
    # is a guard against a future caller, not a live case.
    if region_end != h_off + h_size:
        raise PatchError(
            "the region ends at 0x%X but its section ends at 0x%X; growing a region "
            "that is not the tail of its section is not supported"
            % (region_end, h_off + h_size))

    # Refuse rather than overlap: the grown section's vaddr end must stay below the
    # next allocated section.
    h_vend = h_addr + h_size + delta
    for (i, _t, sa, so, sz, _e) in secs:
        if i == hi or not sz or not sa:
            continue
        if sa >= h_addr + h_size and sa < h_vend:
            raise PatchError(
                "growing by %d byte(s) would push the code to 0x%X, over the section "
                "at 0x%X -- not enough vaddr room in this object"
                % (delta, h_vend, sa))

    shoff = struct.unpack_from("<Q", blob, 0x28)[0]
    shentsize = struct.unpack_from("<H", blob, 0x3A)[0]
    for (i, _t, sa, so, sz, _e) in secs:
        o = shoff + i * shentsize
        if i == hi:
            struct.pack_into("<Q", out, o + 32, sz + delta)       # sh_size
        elif so >= region_end:
            struct.pack_into("<Q", out, o + 24, so + fshift)      # sh_offset

    phoff = struct.unpack_from("<Q", blob, 0x20)[0]
    phentsize = struct.unpack_from("<H", blob, 0x36)[0]
    for (i, ptype, poff, pvaddr, pfilesz, pmemsz, _pa) in phdrs:
        o = phoff + i * phentsize
        if poff <= region_off and region_end <= poff + pfilesz:
            struct.pack_into("<Q", out, o + 32, pfilesz + delta)  # p_filesz
            struct.pack_into("<Q", out, o + 40, pmemsz + delta)   # p_memsz
        elif poff >= region_end:
            struct.pack_into("<Q", out, o + 8, poff + fshift)     # p_offset
    if shoff >= region_end:
        struct.pack_into("<Q", out, 0x28, shoff + fshift)

    # -- splice: region, then the gap the shift opened, then the rest ---------------
    head = bytes(out[:region_off])
    rest = bytes(out[region_end:])
    pad = b"\0" * (fshift - delta)
    return head + bytes(new_region) + pad + rest


def rebuild_grown_object(target, patch, out_path):
    """Write `target`'s code object rebuilt at the larger size `patch` needs.

    Shared by the two bare-code-object backends, because for both of them the
    delivery problem is the same one: an inserted instruction does not fit the span
    it came from, so there is nothing to splice and the whole object has to be
    rebuilt.  How the result then reaches the GPU is what differs, and is not this
    function's business -- HsaOverlay hands the shim a FILE to substitute for the
    one the process was loading, AiterOverlay drops it into the symlink farm and
    lets aiter's own hipModuleLoad read it.

    Only sound on a SINGLE-kernel object.  Growing one kernel of several moves the
    ones after it without moving the symbols that name them, and the result is an
    object that loads and computes garbage -- the worst outcome a sweep can have.
    grow_code_object's "the region must be the tail of its section" guard catches
    that too, but it catches it as arithmetic; the check here can say why.

    Returns (out_path, code_delta).  `code_delta` is how much the KERNEL grew, which
    is what a read-back has to extend its address range by.  It is not how much the
    FILE grew: grow_code_object rounds the file shift up to the segment alignment, so
    the file typically gains a page where the code gained a few instructions.
    """
    funcs = list(target.finder.funcs(target.elf))
    if len(funcs) > 1:
        raise PatchError(
            "%s defines %d kernel(s); growing one of them would move the others "
            "without moving the symbols that name them"
            % (target.elf, len(funcs)))

    with open(target.elf, "rb") as f:
        blob = bytearray(f.read())
    region_off = target.file_offset(target.start)
    old_len = target.stop - target.start
    new_region = None
    # Apply the symbol edits first, while .symtab is still where it was measured;
    # grow_code_object moves it afterwards.
    for off, data in sorted(patch.blob.items()):
        if off == region_off:
            new_region = data
        else:
            blob[off:off + len(data)] = data
    if new_region is None:
        raise PatchError("growth patch carries no region at offset 0x%X" % region_off)
    out = grow_code_object(bytes(blob), region_off, old_len, new_region)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, out_path)
    return out_path, patch.delta


class _ElfFinder:
    """The two things rnan.disassemble/symbol_range want from a KernelFinder.

    KernelFinder is a hipBLASLt index -- it walks *.dat, unbundles libraries and maps
    solution names to code objects, none of which exists for a bare .co.  But the
    functions that consume it only ever call .funcs(elf), and that is a plain
    llvm-readelf over any ELF.  So this shim lets disassemble() and symbol_range() be
    reused verbatim instead of reimplemented with subtly different symbol handling.
    """

    def __init__(self):
        self._funcs = {}

    def funcs(self, elf):
        return rnan.KernelFinder.funcs(self, elf)


def find_aiter_asm_dir(explicit=None):
    """The directory aiter composes paths under: <root>/<arch>/<hsaco>.

    Order: an explicit --patch-asm-dir, then the live AITER_ASM_DIR, then aiter's own
    answer via its Python package, then a scan.  Asking the package is better than
    guessing because it applies the AITER_META_DIR override and the "does csrc exist"
    fallback that jit/core.py does, which a path scan would silently get wrong on an
    installed-vs-in-tree layout.
    """
    import subprocess

    for cand in (explicit, os.environ.get("AITER_ASM_DIR")):
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)

    probe = ("import aiter.jit.core as c; print(c.AITER_ASM_DIR)")
    try:
        r = subprocess.run([sys.executable, "-c", probe], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=120)
        cand = r.stdout.decode().strip().splitlines()[-1] if r.stdout.strip() else ""
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    except Exception:
        pass

    # Last resort.  Importing aiter can fail for reasons that have nothing to do with
    # where its kernels live (a missing flydsl module is the documented one), so a
    # scan is worth having rather than giving up on the whole rung.
    seen = set()
    for base in sys.path + ["/opt/rocm", "/var/lib/jenkins/pytorch"]:
        if not base or base in seen or not os.path.isdir(base):
            continue
        seen.add(base)
        for root, dirs, _files in os.walk(base):
            if root.count(os.sep) - base.count(os.sep) > 4:
                dirs[:] = []
                continue
            if os.path.basename(root) == "hsa" and "aiter" in root:
                return os.path.abspath(root)
    return None


def aiter_meta_of(asm_root):
    """The aiter_meta dir an asm root `<meta>/hsa` belongs to, or None (older layouts)."""
    root = os.path.normpath(asm_root or "")
    meta = os.path.dirname(root)
    if os.path.basename(root) == "hsa" and os.path.isdir(os.path.join(meta, "csrc")):
        return meta
    return None


def ensure_aiter_asm_dir(explicit=None, stream=None):
    """Put AITER_ASM_DIR in os.environ so the BASELINE is file-backed too.

    Call this before the baseline replay, not just before the trials.  The overlay
    only covers trial children; without this the baseline runs the embedded copy and
    every row is a comparison across two different binaries.

    Returns the root, or None if aiter could not be located (in which case the caller
    should leave the op SKIPPED rather than sweep something it cannot serve).
    """
    root = find_aiter_asm_dir(explicit)
    if root:
        # Trailing slash to match what aiter's own Python writes; it concatenates
        # without a separator in one place, so "<root>//gfx950/..." is normal and
        # harmless, but a MISSING separator would not be.
        os.environ["AITER_ASM_DIR"] = root.rstrip("/") + "/"
        if stream:
            print("aiter:   AITER_ASM_DIR=%s (forces file-backed hipModuleLoad; the"
                  " default is an embedded copy)" % os.environ["AITER_ASM_DIR"],
                  file=stream)
    return root


class ElfKernelTarget:
    """PatchTarget's surface over a BARE code-object ELF on disk.

    Shared by the two non-hipBLASLt backends, which differ only in where the file
    comes from:

      AiterTarget  a .co shipped in aiter's hsa tree
      HsaTarget    a code object dumped out of a live process by the LD_PRELOAD shim

    Everything the sweep does downstream -- every selector, every payload, the region
    relocation, MutationSpec, the trial loop, the disasm diff -- reads
    .insns/.arch/.start/.stop and never asks where the bytes came from, so neither
    backend needs anything beyond this.

    Note what is NOT here: unbundling.  A Tensile kernel arrives inside a fat binary
    and most of PatchTarget is the unbundle/rebundle dance; a bare code object is
    already what that dance PRODUCES.
    """

    _ARCH_IN_PATH = re.compile(r"(?:^|/)(gfx[0-9a-f]+[a-z]*)(?:/|$)")

    def _load_elf(self, elf, arch, symbol=None, label=None):
        """Disassemble one kernel out of `elf`.  Sets the attributes a sweep reads."""
        self.elf = self.libfile = elf
        self.arch = arch
        self.bundled = False
        self.finder = _ElfFinder()
        funcs = list(self.finder.funcs(elf))
        if not funcs:
            raise PatchError("%s exports no FUNC symbol -- not a kernel code object"
                             % elf)
        if symbol:
            if not any(n == symbol for _a, n in funcs):
                raise PatchError("%s does not define %r (it has %d FUNC symbol(s))"
                                 % (elf, symbol, len(funcs)))
            self.symbol = symbol
        elif len(funcs) == 1:
            self.symbol = funcs[0][1]
        else:
            # Several kernels in one object and no name given.  Picking by sort order
            # would be arbitrary, and sweeping the wrong kernel produces a table that
            # looks fine and means nothing -- so refuse and name the choices.
            names = sorted(n for _a, n in funcs)
            raise PatchError(
                "%s defines %d kernels and none was named; pass one of: %s%s"
                % (elf, len(funcs), ", ".join(names[:4]),
                   " ..." if len(names) > 4 else ""))
        self.kernel = label or self.symbol
        self.insns = rnan.disassemble(self.finder, elf, self.symbol, self.arch)
        self.start, self.stop = rnan.symbol_range(self.finder, elf, self.symbol)
        self.sh_addr, self.sh_off, _sz = rnan.section_of(elf, self.insns[0].addr)
        self._said_symbols = False

    def _arch_from(self, path, arch=None):
        """Arch from the path before the GPU.

        Neither an aiter nor a CK kernel name carries the _ISA tag gpu_arch() falls
        back on, so on a GPU-free node gpu_arch() would sys.exit -- which is exactly
        where --list-mutations is meant to be runnable.  ".../gfx950/..." names its
        own arch; trust that first.
        """
        m = self._ARCH_IN_PATH.search(path or "")
        return arch or (m.group(1) if m else None) or rnan.gpu_arch([_ArchShim(path)])

    # -- identical to PatchTarget; see there for why each one checks before writing --

    def file_offset(self, addr):
        return addr - self.sh_addr + self.sh_off

    select = PatchTarget.select
    region_bytes = PatchTarget.region_bytes
    symbol_patches = PatchTarget.symbol_patches
    offsets_for = PatchTarget.offsets_for

    def analysis(self, offsets):
        a = rnan.KernelAnalysis(self.kernel)
        a.libfile, a.elf = self.libfile, self.elf
        a.bundled = False
        a.insns, a.start, a.stop = self.insns, self.start, self.stop
        a.patches = dict(offsets)
        return a


class AiterTarget(ElfKernelTarget):
    """A standalone aiter .co.

    `kernel` is the path aiter itself uses -- relative to <asm_root>/<arch>, e.g.
    "fmha_v3_bwd/bwd_hd128_bf16_a32_psskddv.co" -- because that is the string the
    loader's trace prints and the one a report has to quote for anyone to reproduce.
    """

    def __init__(self, kernel, asm_dir=None, arch=None, workdir=None):
        import tempfile
        self.workdir = workdir or tempfile.mkdtemp(prefix="aiter_patch_")
        arch = self._arch_from(kernel, arch)
        self.asm_root = find_aiter_asm_dir(asm_dir)
        if not self.asm_root:
            raise PatchError(
                "cannot locate aiter's hsa directory -- pass --patch-asm-dir. Without "
                "it there is no .co to patch and no AITER_ASM_DIR to serve it from.")
        self.libdir = os.path.join(self.asm_root, arch)
        # Accept either form: the relative hsaco aiter's loader composes with, or the
        # absolute path its trace prints.  Normalising on the arch component rather
        # than on a libdir prefix keeps this working when the overlay root and the
        # stock root differ, which is the normal case mid-sweep.
        rel = kernel
        m = self._ARCH_IN_PATH.search(kernel)
        if os.path.isabs(kernel) and m:
            rel = kernel[m.end():].lstrip("/")
        elf = os.path.join(self.libdir, rel)
        if not os.path.isfile(elf):
            raise PatchError(
                "%s is not under %s (arch %s). Check --patch-asm-dir/--patch-arch, or "
                "that the name came from this build's own [aiter] hipModuleLoad trace."
                % (rel, self.libdir, arch))
        # One kernel per .co is the aiter convention, so no symbol is named here; if
        # a .co ever carries several, _load_elf refuses rather than guessing.
        self._load_elf(elf, arch, label=rel)


class AiterOverlay:
    """OverlayCache for aiter: a symlink farm of <asm_root>, one real patched file.

    Same contract and same reason for existing -- the tree is linked once, per trial
    only the one .co is swapped, and each concurrent worker needs its own or two
    trials race on the same file.  Cheaper than the hipBLASLt version: no rebundle,
    and the read-back needs no unbundle.

    It also delivers a GrowthPatch, which the hipBLASLt overlay cannot.  That
    difference is not about the overlay mechanism, it is about what the file holds: a
    .co is ONE kernel and the bytes after it are its own section's padding, so the
    object can be rebuilt at a larger size and swapped in whole.  A Tensile kernel
    shares a fat binary with hundreds of others and the bytes after it are the next
    kernel's code, which nothing would re-point at.

    This is what the 2nd NaN needed and did not have.  Its backward is three aiter
    .co with 0 bytes of tail padding, so every barrier-insertion mutation on them
    came back `ERROR: PatchError ... the aiter overlay cannot deliver` -- six rows
    that read like a swept-and-clean backward and were in fact a tooling limit, on
    the very op the contended chain tally named first-to-go-non-finite in 100% of
    failing runs.
    """

    def __init__(self, target, work):
        self.target, self.work = target, work
        self.overlay = os.path.join(work, "aiter_asm")
        # amd-aiter 0.1.26+ ignores AITER_ASM_DIR: `import aiter` rewrites it to
        # $AITER_META_DIR/hsa/ (aiter/jit/core.py), so the overlay is served only through
        # AITER_META_DIR -- a dir that must hold csrc/ as well.  When the asm root is
        # <meta>/hsa, mirror the meta dir: the patched hsa/ farm plus links to the rest.
        self.meta = aiter_meta_of(target.asm_root)
        if self.meta:
            mroot = os.path.join(work, "aiter_meta")
            os.makedirs(mroot, exist_ok=True)
            for e in os.listdir(self.meta):
                if e != "hsa" and not os.path.lexists(os.path.join(mroot, e)):
                    os.symlink(os.path.join(self.meta, e), os.path.join(mroot, e))
            self.overlay = os.path.join(mroot, "hsa")
            self.meta_overlay = mroot
        # How much the kernel grew in the CURRENT overlay, for disassemble(); 0 for a
        # same-size splice.  Set per apply(), because the trials alternate.
        self._grown_delta = 0
        self._link_tree(target.asm_root, self.overlay)
        self.rel = os.path.join(target.arch, target.kernel)
        self.dst = os.path.join(self.overlay, self.rel)
        # The farm linked the stock file; the patched copy has to REPLACE the link,
        # never be written through it, or we would corrupt the real installation.
        if os.path.islink(self.dst):
            os.unlink(self.dst)
        import shutil
        shutil.copy2(target.elf, self.dst)

    @staticmethod
    def _link_tree(src, dst):
        """`cp -as` by hand: real dirs, symlinked leaves.

        ~2700 links for a full aiter tree, paid once per sweep.  Real directories are
        what make a single leaf replaceable without touching the installation.
        """
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            out = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(out, exist_ok=True)
            for f in files:
                link = os.path.join(out, f)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(root, f), link)
            dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]

    def apply(self, offsets):
        """Rebuild the patched .co and swap it into the overlay atomically."""
        co_out = os.path.join(self.work, "trial.co")
        if isinstance(offsets, GrowthPatch):
            _grown, self._grown_delta = rebuild_grown_object(
                self.target, offsets, co_out)
        else:
            # Reset, or a same-size trial following a grown one would read back with
            # the previous trial's range and report the overrun as a change.
            self._grown_delta = 0
            rnan.write_patched(self.target.elf, offsets, co_out)
        # Into the farm's leaf, which __init__ already turned from a symlink into a
        # real file -- writing THROUGH the link would patch the installation itself.
        os.replace(co_out, self.dst)
        return self.overlay

    def env(self, base=None):
        # `base or os.environ` would treat an empty dict as "use the whole environment".
        e = dict(os.environ if base is None else base)
        e["AITER_ASM_DIR"] = self.overlay.rstrip("/") + "/"
        e["AITER_LOG_LEVEL"] = "INFO"           # the served check reads aiter's load lines
        if self.meta:
            e["AITER_META_DIR"] = self.meta_overlay
        return e

    def disassemble(self):
        """Re-disassemble the .co as it currently sits in the overlay.

        Valid only until the next apply(), same as OverlayCache: there is one file and
        every trial overwrites it.

        A grown kernel occupies [start, stop + delta); disassembling only the original
        range would cut the listing off before the end, and diff_disasm would report
        the truncation as a change.
        """
        return rnan.disassemble_range(self.dst, self.target.start,
                                      self.target.stop + self._grown_delta,
                                      self.target.arch)


# ======================================================================================
# CK, and anything else: patching code objects in flight via the HSA shim
# ======================================================================================
#
# A CK kernel has no file to point at.  Its device code is compiled into
# libtorch_hip.so, inside thousands of COMPRESSED offload bundles in a 94 MB
# .hip_fatbin section -- there is no AITER_ASM_DIR, no HIPBLASLT_TENSILE_LIBPATH, and
# no .co.  Patching it the way the other two backends are patched would mean parsing
# CCOB, zlib round-tripping a bundle that decompresses 30x, recompressing it to fit
# the original span, and copying a 362 MB library per overlay.
#
# So we go under all three instead.  Every backend's device code reaches the GPU
# through hsa_code_object_reader_create_from_memory, which libamdhip64 imports and
# libhsa-runtime64 exports -- an interposable boundary where HIP hands HSA the
# ALREADY-DECOMPRESSED ELF.  scripts/hsa_shim/hsa_patch_shim.c sits there.
#
# The arrangement that makes the offsets trustworthy: Python never parses a fat
# binary.  A discovery run DUMPS the code object through the same hook, Python
# disassembles those exact bytes, and the trial patches those exact bytes.  The
# vaddr -> file-offset arithmetic is correct by construction rather than by agreement
# with a bundler.
#
# Measured on the 2nd NaN: the dump for the CK fmha forward is a 22 KB object holding
# exactly ONE FUNC symbol, so in practice this is the same shape as an aiter .co.

_SHIM_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsa_shim", "hsa_patch_shim.c")
_SHIM_BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsa_shim", "build_hsa_shim.sh")

# Two shapes, one meaning: PATCHED is a same-size splice, REPLACED is a whole object
# swapped for a rebuilt (usually larger) one.  Both must count as "the patch reached
# the GPU", or an insertion mutation would be reported as never served.
_SHIM_PATCHED_RE = re.compile(
    r"^(?:PATCHED n=(?P<n>\d+) size=(?P<size>\d+)"
    r"|REPLACED n=(?P<rn>\d+) oldsize=(?P<rsize>\d+) newsize=\d+)"
    r" applied=(?P<applied>\d+) rejected=(?P<rejected>\d+)")


def build_hsa_shim(out=None):
    """Compile the LD_PRELOAD shim, returning its path.

    Built where it will RUN -- inside the container -- because it is injected into a
    process whose glibc comes from that image.  The build script caches on mtime, so
    a sweep pays for one compile however many trials it runs.
    """
    import subprocess

    if not os.path.isfile(_SHIM_BUILD):
        raise PatchError("missing %s" % _SHIM_BUILD)
    cmd = ["bash", _SHIM_BUILD]
    if out:
        cmd.append(out)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise PatchError("could not build the HSA shim: %s"
                         % (r.stderr.strip().splitlines() or ["?"])[-1])
    path = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if not path or not os.path.isfile(path):
        raise PatchError("the HSA shim build printed no usable path (%r)" % path)
    return path


def parse_shim_log(text):
    """[(n, size, applied, rejected)] for every object the shim PATCHED or REPLACED."""
    out = []
    for line in text.splitlines():
        m = _SHIM_PATCHED_RE.match(line.strip())
        if m:
            n = m.group("n") or m.group("rn")
            size = m.group("size") or m.group("rsize")
            out.append((int(n), int(size),
                        int(m.group("applied")), int(m.group("rejected"))))
    return out


def shim_applied_patches(text):
    """Did the shim actually splice anything into a code object this run?

    The CK analogue of aiter's "[aiter] hipModuleLoad: <overlay>/..." line, and it
    exists for the same reason: a trial whose patch never reached the GPU still
    produces a full, plausible, entirely meaningless table.  A row the shim log
    cannot vouch for is not a row to report.
    """
    rows = parse_shim_log(text)
    return bool(rows) and any(applied for _n, _s, applied, _r in rows)


class HsaTarget(ElfKernelTarget):
    """A code object dumped out of a live process by the shim.

    `symbol` is the mangled device kernel name -- for CK something like
    _ZN7ck_tile9kentry_ptINS_8gfx950_tELi2ENS_13FmhaFwdKernelI... -- which is both
    what identifies the kernel in a report and what the shim memmem()s for at load
    time to recognise the object it must patch.
    """

    def __init__(self, co_path, symbol=None, arch=None, workdir=None):
        import tempfile
        self.workdir = workdir or tempfile.mkdtemp(prefix="hsa_patch_")
        if not os.path.isfile(co_path):
            raise PatchError("%s does not exist -- run discovery (NUMERIC_DOCTOR_HSA_DUMP)"
                             " before building a target" % co_path)
        self._load_elf(co_path, self._arch_from(co_path, arch), symbol=symbol,
                       label=symbol or os.path.basename(co_path))
        self.libdir = os.path.dirname(co_path)
        # The shim matches on this string.  The mangled symbol is a strong key: it
        # encodes the full template instantiation, so it cannot collide with another
        # kernel in another object.
        self.match = self.symbol

    def sha256(self):
        import hashlib
        h = hashlib.sha256()
        with open(self.elf, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


class HsaOverlay:
    """OverlayCache's surface, delivered by LD_PRELOAD instead of a library path.

    There is no shadow tree here and nothing is copied: apply() writes a spec file and
    the shim does the splice in the loading process.  That is what makes this work for
    a backend whose kernel has no file of its own.
    """

    def __init__(self, target, work):
        self.target, self.work = target, work
        os.makedirs(work, exist_ok=True)
        self.shim = build_hsa_shim(os.path.join(work, "hsa_patch_shim.so"))
        self.spec = os.path.join(work, "trial.spec")
        self.log = os.path.join(work, "shim.log")
        self.dump = os.path.join(work, "readback")
        # Reported as the "overlay" so the sweep's existing wiring has something to
        # print and compare; it is the spec, not a directory of libraries.
        self.overlay = self.spec
        self._grown = None
        self._grown_delta = 0
        self._write_spec({})

    def _write_spec(self, offsets, replace=None):
        lines = ["MATCH %s" % self.target.match]
        if replace:
            # FILE wins in the shim, so the PATCH lines are not written at all -- two
            # descriptions of the same edit, one of them stale, is a bug waiting to be
            # read as a result.
            lines.append("FILE %s" % replace)
        else:
            for off, data in sorted(offsets.items()):
                lines.append("PATCH %d %s" % (off, data.hex()))
        tmp = self.spec + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.spec)

    def _rebuild(self, patch):
        """Write a whole replacement code object for a mutation that had to grow.

        Kept rather than inlined because this one is HANDED TO THE SHIM and so has to
        survive the trial on disk, where AiterOverlay's is moved straight into the
        farm.  The rebuild itself is shared.
        """
        path, delta = rebuild_grown_object(
            self.target, patch, os.path.join(self.work, "grown.co"))
        self._grown_delta = delta
        return path

    def apply(self, offsets):
        """Write the per-trial spec.  The splice happens later, in the child."""
        # Truncate the log per trial: shim_applied_patches() asks "did THIS trial
        # patch anything", and a log carrying the previous trial's PATCHED line would
        # answer yes for a trial that did nothing.
        for p in (self.log,):
            try:
                os.unlink(p)
            except OSError:
                pass
        if isinstance(offsets, GrowthPatch):
            self._grown = self._rebuild(offsets)
            self._write_spec(None, replace=self._grown)
        else:
            self._grown = None
            self._grown_delta = 0
            self._write_spec(offsets)
        return self.overlay

    def env(self, base=None):
        # `base or os.environ` would treat an empty dict as "use the whole environment".
        e = dict(os.environ if base is None else base)
        # APPEND.  The image may set its own LD_PRELOAD, and replacing it would change
        # the process under test in a way no row would explain.
        pre = e.get("LD_PRELOAD", "").strip()
        e["LD_PRELOAD"] = (pre + " " + self.shim).strip() if pre else self.shim
        e["NUMERIC_DOCTOR_HSA_SPEC"] = self.spec
        e["NUMERIC_DOCTOR_HSA_LOG"] = self.log
        e["NUMERIC_DOCTOR_HSA_MATCH"] = self.target.match
        return e

    def log_text(self):
        try:
            with open(self.log) as f:
                return f.read()
        except OSError:
            return ""

    def disassemble(self):
        """Re-disassemble the patched object.

        Rebuilt locally from the stock dump plus the current spec rather than read
        back out of the child: the child's buffer is in another process's memory and
        dumping it would need a second GPU run per finding.  The bytes are identical
        by construction -- the shim applies exactly these offsets to exactly these
        bytes -- and the shim log is what confirms it actually did.
        """
        import shutil
        import tempfile

        if self._grown:
            # A grown kernel occupies [start, stop+delta); disassembling only the
            # original range would cut the listing off before the end and the diff
            # would report a truncation as a change.  The CODE delta, not the file's
            # -- grow_code_object rounds the file shift up to the segment alignment.
            return rnan.disassemble_range(self._grown, self.target.start,
                                          self.target.stop + self._grown_delta,
                                          self.target.arch)
        d = tempfile.mkdtemp(dir=self.work)
        try:
            out = os.path.join(d, "patched.co")
            offsets = {}
            with open(self.spec) as f:
                for line in f:
                    if line.startswith("PATCH "):
                        _kw, off, hexb = line.split()
                        offsets[int(off)] = bytes.fromhex(hexb)
            rnan.write_patched(self.target.elf, offsets, out)
            return rnan.disassemble_range(out, self.target.start, self.target.stop,
                                          self.target.arch)
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ======================================================================================
# comparing two disassemblies
# ======================================================================================

def _dtext(ins):
    return None if ins is None else ins.text.strip()


def diff_disasm(before, after):
    """[(vaddr, before_text, after_text)] for every address whose text changed.

    Matched by vaddr, not by position.  A payload is length-preserving by contract,
    but `nop` fills an 8-byte slot with TWO 4-byte s_nops, so one instruction before
    is two after and pairing by index would report every later instruction as
    changed.  An address present on only one side comes back with None for the
    other, which is exactly that case and reads correctly.
    """
    b = dict((i.addr, i) for i in before)
    a = dict((i.addr, i) for i in after)
    out = []
    for addr in sorted(set(b) | set(a)):
        bt, at = _dtext(b.get(addr)), _dtext(a.get(addr))
        if bt != at:
            out.append((addr, bt, at))
    return out


def format_disasm_diff(before, after, context=4, limit=0):
    """A -/+ listing of what a patch changed, with `context` unchanged neighbours.

    Returns lines, so a caller can indent them or write them to a file.  `limit`
    caps the number of CHANGED instructions anchored (0 = no cap) and says what it
    dropped: a store_slice mutation nops ~90 sites, and a diff that silently showed
    the first 40 reads as "that is all it touched".
    """
    b = dict((i.addr, i) for i in before)
    a = dict((i.addr, i) for i in after)
    addrs = sorted(set(b) | set(a))
    changed = [n for n, ad in enumerate(addrs) if _dtext(b.get(ad)) != _dtext(a.get(ad))]
    if not changed:
        return ["(nothing changed -- the two disassemblies are identical)"]

    anchors = changed if limit <= 0 else changed[:limit]
    keep = set()
    for n in anchors:
        keep.update(range(max(0, n - context), min(len(addrs), n + context + 1)))

    out, prev = [], None
    for n in sorted(keep):
        if prev is not None and n != prev + 1:
            out.append("      ... %d instruction(s) unchanged ..." % (n - prev - 1))
        ad = addrs[n]
        bi, ai = b.get(ad), a.get(ad)
        if _dtext(bi) == _dtext(ai):
            out.append("      0x%X  %s" % (ad, _dtext(bi)))
        else:
            if bi is not None:
                out.append("    - 0x%X  %s" % (ad, _dtext(bi)))
            if ai is not None:
                out.append("    + 0x%X  %s" % (ad, _dtext(ai)))
        prev = n
    if len(anchors) < len(changed):
        out.append("      ... %d more changed instruction(s) not shown (%s%s) ..."
                   % (len(changed) - len(anchors),
                      " ".join("0x%X" % addrs[n] for n in changed[len(anchors):][:6]),
                      " ..." if len(changed) - len(anchors) > 6 else ""))
    return out


# ======================================================================================
# shared CLI
# ======================================================================================

def export_fix(cache, dest):
    """Copy the CURRENT trial's patched kernel out of a sweep overlay into `dest`, with
    an env.sh that loads it.  -> list of files written.

    The overlay is valid only until the next trial's apply() (one spec, one leaf), and
    the sweep deletes it at the end; this is what lets step 8 validate a CLEAN row's
    exact bytes later.  Paths in env.sh are the ones THIS process sees (the
    in-container path under apptainer), so source it in the same environment.
    hipBLASLt overlays: use scripts/fix_overlay.py instead.
    """
    import shutil
    os.makedirs(dest, exist_ok=True)
    out = []
    if isinstance(cache, HsaOverlay):
        shim = os.path.join(dest, "hsa_patch_shim.so")
        shutil.copy2(cache.shim, shim)
        spec = os.path.join(dest, "fix.spec")
        if cache._grown:
            co = os.path.join(dest, "patched.co")
            shutil.copy2(cache._grown, co)
            with open(spec, "w") as f:
                f.write("MATCH %s\nFILE %s\n" % (cache.target.match, co))
            out.append(co)
        else:
            shutil.copy2(cache.spec, spec)
        env = ['export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD }%s"' % shim,
               "export NUMERIC_DOCTOR_HSA_SPEC=%s" % spec,
               "export NUMERIC_DOCTOR_HSA_LOG=%s" % os.path.join(dest, "shim.log"),
               "export NUMERIC_DOCTOR_HSA_MATCH='%s'" % cache.target.match,
               "# loaded = shim.log has a PATCHED (same size) or REPLACED (grown) line"]
        out += [shim, spec]
    elif isinstance(cache, AiterOverlay):
        co = os.path.join(dest, "patched", cache.rel)
        os.makedirs(os.path.dirname(co), exist_ok=True)
        shutil.copy2(cache.dst, co)
        tool = os.path.join(dest, "aiter_overlay.sh")
        shutil.copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aiter_overlay.sh"), tool)
        # Relative to this directory and to the aiter of the image it is sourced in, so the
        # kept fix can be copied into a fix package (2026-11-20: the absolute host paths here
        # made every aiter fix a hand-built package).
        env = ['_d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)',
               'eval "$(bash "$_d/aiter_overlay.sh" "$_d/patched/%s" "%s" "$_d/overlay")"'
               % (cache.rel, cache.rel),
               "# loaded = aiter's load log names $_d/overlay/... (AITER_LOG_LEVEL=INFO on 0.1.26+)"]
        out += [tool]
        out.append(co)
    else:
        raise PatchError("export_fix: %s overlays are exported by fix_overlay.py"
                         % type(cache).__name__)
    envsh = os.path.join(dest, "env.sh")
    with open(envsh, "w") as f:
        f.write("# source me: loads the patched kernel from a CLEAN mutation row\n"
                + "\n".join(env) + "\n")
    return out + [envsh]


def add_patch_args(ap):
    """The --patch-* flags, defined once so the two replay scripts cannot drift."""
    g = ap.add_argument_group("instruction patching")
    g.add_argument("--patch-sweep", action="store_true",
                   help="patch the kernel one instruction at a time and replay after each")
    g.add_argument("--patch-selector", default="nan_const",
                   choices=sorted(SELECTORS), help="which instructions are candidates")
    g.add_argument("--patch-payload", default="nop",
                   choices=all_payloads(), help="what replaces them")
    g.add_argument("--patch-strategy", default="single",
                   choices=["single", "all", "ddmin"],
                   help="single site at a time, all at once, or minimise a sufficient set")
    g.add_argument("--patch-addr", action="append", default=[],
                   help="explicit vaddr (repeatable); implies --patch-selector addrs")
    g.add_argument("--patch-range", help="LO-HI vaddr range")
    g.add_argument("--patch-filter", help="mnemonic regex")
    g.add_argument("--patch-all", action="store_true", help="alias for --patch-selector all")
    g.add_argument("--patch-kernel", help="Tensile kernel name (skips discovery)")
    g.add_argument("--patch-solution-index", type=int, help="pin, and skip discovery")
    g.add_argument("--patch-libdir", help="base hipBLASLt Tensile library dir")
    g.add_argument("--patch-hsaco",
                   help="aiter code object to patch instead of a Tensile kernel, as the "
                        "loader names it: '<family>/<name>.co' relative to "
                        "<asm-dir>/<arch> (skips discovery)")
    g.add_argument("--patch-asm-dir",
                   help="aiter hsa root holding <arch>/<family>/*.co "
                        "(default: ask aiter, else AITER_ASM_DIR, else scan)")
    g.add_argument("--patch-co",
                   help="code object dumped by the HSA shim, to patch in flight "
                        "(CK and anything else with no file of its own)")
    g.add_argument("--patch-co-symbol",
                   help="mangled device kernel inside --patch-co (default: the only "
                        "one; required if the object defines several)")
    g.add_argument("--patch-arch", help="gfx target (default: autodetect)")
    g.add_argument("--patch-jobs", type=int, default=1,
                   help="concurrent trials, one overlay and one GPU each")
    g.add_argument("--patch-timeout", type=float, default=300.0, help="per-trial seconds")
    g.add_argument("--patch-workdir", help="where overlays live (default: a temp dir)")
    g.add_argument("--patch-report", help="write the JSON record here")
    g.add_argument("--keep-workdir", action="store_true")
    g.add_argument("--list-sites", action="store_true",
                   help="enumerate candidate sites and exit; no GPU, no child processes")
    g.add_argument("--make-ground-truth", metavar="ADDR",
                   help="build a one-site overlay with --patch-payload, print its path, exit")
    g.add_argument("--no-controls", action="store_true",
                   help="skip the baseline and all-sites control trials")
    return g


def selector_from_args(args):
    """(name, opts) honouring the aliases."""
    opts = {}
    name = args.patch_selector
    if args.patch_addr:
        name = "addrs"
        opts["addrs"] = [int(a, 0) for a in args.patch_addr]
    elif args.patch_all:
        name = "all"
    elif args.patch_range:
        name = "range"
        lo, _, hi = args.patch_range.partition("-")
        opts["range"] = (int(lo, 0), int(hi, 0))
    elif args.patch_filter:
        name = "filter"
        opts["filter"] = args.patch_filter
    return name, opts


# ======================================================================================
# fixture self-test
# ======================================================================================

# The known-good/known-bad .asm pair is not in the repo (it is dumped from a
# specific hipBLASLt build), so this cannot be repo-relative.  Env-overridable so
# a second site can point check_fixtures() at its own copy without editing the
# source; --fixtures still wins over both.
FIXTURES = os.environ.get("NUMERIC_DOCTOR_FIXTURES",
                          "add_nan_to_hipblaslt")


def check_fixtures(fixdir=FIXTURES):
    """Unit-test the selector against the known-good/known-bad disassembly pair.

    Those .asm files are raw llvm-objdump output, so rnan._parse_disasm reads them
    directly -- no ROCm, no GPU, no code object.  Expected: 0 sites on the clean
    kernel, exactly the 5 injected addresses on the patched one.
    """
    expect = [0x2037C54, 0x2038014, 0x2038094, 0x2038138, 0x20381D8]
    rc = 0

    def load(name):
        with open(os.path.join(fixdir, name)) as f:
            return rnan._parse_disasm(f.read())

    clean = load("kernel_without_nan.asm")
    dirty = load("kernel_with_nan.asm")
    print("parsed %d / %d instructions" % (len(clean), len(dirty)))

    got = select_nan_const(clean, "gfx950", {})
    if got:
        print("FAIL clean kernel: expected 0 sites, got %d" % len(got))
        for s in got[:10]:
            print("     ", s)
        rc = 1
    else:
        print("PASS clean kernel: 0 nan-const sites (no false positives)")

    got = select_nan_const(dirty, "gfx950", {})
    addrs = [s.addr for s in got]
    if addrs == expect:
        print("PASS injected kernel: %s" % " ".join("0x%X" % a for a in addrs))
    else:
        print("FAIL injected kernel: expected %s, got %s"
              % (" ".join("0x%X" % a for a in expect),
                 " ".join("0x%X" % a for a in addrs) or "(none)"))
        rc = 1

    # Explicit negatives: the two shapes that a naive detector gets wrong.
    neg = {"v_mul_i32_i24_e32 v3, -16, v3": "integer address math with a NaN bit pattern",
           "v_fma_f32 v5, -2.0, v5, 2.0": "-2.0 is 0xC0000000, finite"}
    for ins in clean:
        if ins.text.strip() in neg:
            if select_nan_const([ins], "gfx950", {}):
                print("FAIL negative: %r flagged (%s)" % (ins.text.strip(), neg[ins.text.strip()]))
                rc = 1
            else:
                print("PASS negative: %-32s (%s)" % (ins.text.strip(), neg[ins.text.strip()]))
            neg.pop(ins.text.strip())
    for miss in neg:
        print("WARN negative fixture not present in the .asm: %r" % miss)

    # store_slice: the mode for the real bug. Assert shape, not an exact count --
    # the count is a property of this kernel, the invariants are not.
    sl_clean = select_store_slice(clean, "gfx950", {})
    sl_dirty = select_store_slice(dirty, "gfx950", {})
    frac = len(sl_clean) / float(len(clean))
    if not sl_clean or frac > 0.10:
        print("FAIL store_slice: %d/%d sites (%.1f%%) -- the kill rule is not firing"
              % (len(sl_clean), len(clean), 100 * frac))
        rc = 1
    else:
        print("PASS store_slice: %d/%d sites (%.1f%%) on the clean kernel"
              % (len(sl_clean), len(clean), 100 * frac))
    if not set(s.addr for s in got) <= set(s.addr for s in sl_dirty):
        print("FAIL store_slice: does not cover every nan-const site")
        rc = 1
    else:
        print("PASS store_slice: covers every nan-const site (%d sites on the injected "
              "kernel -- the walk stops at each NaN constant, which has no register "
              "sources)" % len(sl_dirty))
    unsafe = [s for s in sl_clean + sl_dirty if is_unsafe(clean[s.index])]
    if unsafe:
        print("FAIL store_slice: %d unsafe instruction(s) leaked in" % len(unsafe))
        rc = 1
    else:
        print("PASS store_slice: no control-flow/waitcnt instruction selected")

    # Encoding round-trip: the bytes we would assert against must match objdump.
    site = select_nan_const(dirty, "gfx950", {})[0]
    if site.enc != struct.pack("<I", 0x7E1002C1):
        print("FAIL encoding: 0x%X enc=%s, expected C1 02 10 7E"
              % (site.addr, site.enc.hex() if site.enc else None))
        rc = 1
    else:
        print("PASS encoding: 0x%X -> %s (little-endian 7E1002C1)"
              % (site.addr, site.enc.hex()))

    # Payload length invariant.
    insn = dirty[site.index]
    data = make_payload("nop", insn, site, "gfx950")
    if len(data) == insn.size and data == struct.pack("<I", 0xBF800000):
        print("PASS payload: nop -> %s (s_nop 0), length preserved" % data.hex())
    else:
        print("FAIL payload: got %s for a %dB slot" % (data.hex(), insn.size))
        rc = 1
    if make_payload("identity", insn, site, "gfx950") != site.enc:
        print("FAIL payload: identity did not reproduce the original bytes")
        rc = 1
    else:
        print("PASS payload: identity reproduces the original bytes")

    # ---- lgkmcnt0 -------------------------------------------------------------------
    # Selector and payload are both pure functions of the disassembly, so the whole
    # mutation is checkable here.  This kernel has exactly two relaxed LGKM waits, both
    # `s_waitcnt lgkmcnt(8)` encoded BF8CC87F.
    lg = select_lgkm_wait(clean, "gfx950", {})
    if len(lg) == 2 and all(lgkm_text_count(clean[s.index]) == 8 for s in lg):
        print("PASS lgkm_wait: %d relaxed wait(s), %s"
              % (len(lg), " ".join("0x%X" % s.addr for s in lg)))
    else:
        print("FAIL lgkm_wait: %d site(s) %s, expected 2 x lgkmcnt(8)"
              % (len(lg), " ".join("0x%X" % s.addr for s in lg)))
        rc = 1

    # The two shapes a naive lgkm scan gets wrong.  The second is the one that matters:
    # its lgkm field is 15, which objdump prints as nothing at all.
    picked = set(s.addr for s in lg)
    neg_w = [("s_waitcnt lgkmcnt(0)", "already draining; nothing to strengthen"),
             ("s_waitcnt vmcnt(0)", "encodes lgkmcnt(15) = no LGKM wait; forcing 0 would "
                                    "ADD one")]
    for text, why in neg_w:
        hits = [i for i in clean if i.text.strip() == text]
        if not hits:
            print("WARN lgkm_wait negative not present in the .asm: %r" % text)
        elif any(i.addr in picked for i in hits):
            print("FAIL lgkm_wait negative: %r selected (%s)" % (text, why))
            rc = 1
        else:
            print("PASS lgkm_wait negative: %-22s x%-3d (%s)" % (text, len(hits), why))

    if lg:
        wsite = lg[0]
        wdata = make_payload("lgkmcnt0", clean[wsite.index], wsite, "gfx950")
        if wdata == struct.pack("<I", 0xBF8CC07F) and len(wdata) == clean[wsite.index].size:
            print("PASS payload lgkmcnt0: 0x%X %s -> %s, length preserved"
                  % (wsite.addr, wsite.enc.hex(), wdata.hex()))
        else:
            print("FAIL payload lgkmcnt0: got %s for a %dB slot"
                  % (wdata.hex(), clean[wsite.index].size))
            rc = 1

    # What must survive the edit is vmcnt/expcnt, and this kernel happens to contain no
    # combined wait to prove it on -- so drive the mask directly.
    for word, want, what in ((0xBF8CC87F, 0xBF8CC07F, "lgkmcnt(8)->0, vmcnt(63) expcnt(7) kept"),
                             (0xBF8C0871, 0xBF8C0071, "vmcnt(1) lgkmcnt(8) -> vmcnt(1) kept"),
                             (0xBF8CC07F, 0xBF8CC07F, "already lgkmcnt(0): idempotent")):
        got = lgkm_zeroed_word(word, "gfx950")
        if got == want:
            print("PASS lgkm mask: %08X -> %08X  (%s)" % (word, got, what))
        else:
            print("FAIL lgkm mask: %08X -> %08X, expected %08X (%s)" % (word, got, want, what))
            rc = 1

    for word, arch, why in ((0x7E1002C1, "gfx950", "not an s_waitcnt at all"),
                            (0xBF8CC87F, "gfx1030", "GFX10 moves lgkmcnt to [13:8]")):
        try:
            lgkm_zeroed_word(word, arch)
            print("FAIL lgkm mask: %08X on %s was accepted (%s)" % (word, arch, why))
            rc = 1
        except ValueError:
            print("PASS lgkm mask: %08X on %-8s refused (%s)" % (word, arch, why))

    # The before/after diff, on the one pair whose answer is known.  The injection
    # was length-preserving, so the clean and injected listings differ at exactly
    # the five NaN sites; a diff reporting more than that is mis-aligning the two
    # and would drown a real finding in noise.
    delta = diff_disasm(clean, dirty)
    if [ad for ad, _b, _a in delta] == expect:
        print("PASS diff: clean vs injected differs at exactly the %d NaN site(s)"
              % len(expect))
    else:
        print("FAIL diff: %d changed address(es) %s, expected %s"
              % (len(delta), " ".join("0x%X" % ad for ad, _b, _a in delta[:8]),
                 " ".join("0x%X" % a for a in expect)))
        rc = 1
    body = format_disasm_diff(clean, dirty, context=1, limit=2)
    if sum(1 for ln in body if ln.lstrip().startswith("+")) == 2 and "not shown" in body[-1]:
        print("PASS diff: --limit 2 anchored 2 change(s) and named the 3 it dropped")
    else:
        print("FAIL diff: limit=2 produced %d line(s): %s" % (len(body), body))
        rc = 1

    rc |= check_relocation(clean, lg)

    print("\n%s" % ("ALL FIXTURE CHECKS PASSED" if rc == 0 else "FIXTURE CHECKS FAILED"))
    return rc


# ======================================================================================
# relocation self-test
# ======================================================================================

_SIM_PAD = 64                       # bytes of padding to pretend follow the fixture


def check_relocation(insns, lg, arch="gfx950"):
    """Exercise relocate_region on the fixture: no ROCm, no GPU, no code object.

    The fixture .asm carries every instruction's encoding in its `//` comment, so the
    whole region can be rebuilt from it.  What makes this worth having is that the
    fixture's 132 branches and 13 getpc sequences are a real kernel's control flow --
    a synthetic test would not have the s_abs_i32 backward form, which is the one shape
    whose literal does not simply follow the displacement's sign.
    """
    rc = 0
    start = insns[0].addr
    code_end = insns[-1].addr + insns[-1].size
    stop = code_end + _SIM_PAD
    tail = b"\x00" * _SIM_PAD

    # ---- decode: every branch and getpc in a real kernel must be readable -------------
    try:
        tgts = [(i.addr, branch_target(i, enc_bytes(i))) for i in insns]
        nbr = sum(1 for _a, t in tgts if t is not None)
        addrs = set(i.addr for i in insns) | {code_end}
        stray = [(a, t) for a, t in tgts if t is not None and t not in addrs]
        if nbr and not stray:
            print("PASS relocate/decode: %d branch target(s), all on an instruction "
                  "boundary" % nbr)
        else:
            print("FAIL relocate/decode: %d branch(es), %d off-boundary e.g. %s"
                  % (nbr, len(stray),
                     " ".join("0x%X->0x%X" % p for p in stray[:4]) or "(none)"))
            rc = 1
    except PatchError as e:
        print("FAIL relocate/decode: %s" % e)
        rc = 1

    pcs = [i for i, ins in enumerate(insns) if ins.mnemonic == "s_getpc_b64"]
    try:
        prs = [find_pcrel(insns, i) for i in pcs]
        addrs = set(i.addr for i in insns) | {code_end}
        bad = [p for p in prs if p.target not in addrs]
        kinds = (sum(1 for p in prs if p.sign > 0), sum(1 for p in prs if p.sign < 0),
                 sum(1 for p in prs if p.absd))
        if prs and not bad:
            print("PASS relocate/getpc: %d sequence(s) decoded (%d forward, %d backward, "
                  "%d via s_abs_i32), all landing on an instruction" % ((len(prs),) + kinds))
        else:
            print("FAIL relocate/getpc: %d sequence(s), %d land off-boundary"
                  % (len(prs), len(bad)))
            rc = 1
    except PatchError as e:
        print("FAIL relocate/getpc: %s" % e)
        rc = 1

    # ---- rebuild: the real mutation, on the real control flow -------------------------
    if not lg:
        print("WARN relocate: no relaxed lgkmcnt wait in the fixture to insert behind")
        return rc
    bar = barrier_bytes(arch, insns)
    if bar != _BARRIER_GFX9:
        print("FAIL relocate/barrier: lifted %s from the kernel, expected %s"
              % (bar.hex(), _BARRIER_GFX9.hex()))
        rc = 1
    else:
        print("PASS relocate/barrier: s_barrier = %s, lifted from the kernel itself"
              % bar.hex())

    edits = {}
    for s in lg:
        ins = insns[s.index]
        edits[s.index] = make_payload("lgkmcnt0", ins, s, arch) + bar
    try:
        blob, new_addr = relocate_region(insns, start, stop, tail, edits, arch)
    except PatchError as e:
        print("FAIL relocate/rebuild: %s" % e)
        return rc | 1

    first = min(new_addr[insns[s.index].addr] for s in lg)
    shifted = sum(1 for i in insns if new_addr[i.addr] != i.addr)
    if len(blob) == stop - start and shifted == sum(1 for i in insns if i.addr > first):
        print("PASS relocate/rebuild: %d byte(s) in, %d out (unchanged); %d instruction(s) "
              "moved, first shift at 0x%X" % (code_end - start, len(blob), shifted, first))
    else:
        print("FAIL relocate/rebuild: %d byte(s) out, expected %d; %d moved"
              % (len(blob), stop - start, shifted))
        rc = 1

    # The sites themselves: drained wait, then a barrier, then the instruction that
    # used to follow -- now 4 bytes further along.
    ok = True
    for s in lg:
        at = new_addr[insns[s.index].addr] - start
        if blob[at:at + 4] != make_payload("lgkmcnt0", insns[s.index], s, arch):
            ok = False
        if blob[at + 4:at + 8] != bar:
            ok = False
    print("%s relocate/sites: %d site(s) now read s_waitcnt lgkmcnt(0) + s_barrier"
          % ("PASS" if ok else "FAIL", len(lg)))
    rc |= (not ok)

    # Nothing before the first site may move, and the growth must be exactly 4B/site.
    grow = new_addr[code_end] - code_end
    untouched = all(new_addr[i.addr] == i.addr for i in insns if i.addr < first)
    if grow == 4 * len(lg) and untouched:
        print("PASS relocate/extent: kernel grew %dB (4B x %d site(s)); nothing before "
              "0x%X moved" % (grow, len(lg), first))
    else:
        print("FAIL relocate/extent: grew %dB, expected %dB; prefix %s"
              % (grow, 4 * len(lg), "intact" if untouched else "MOVED"))
        rc = 1

    # Padding is consumed, not invented: the bytes past the relocated code are still
    # the filler we found, and the region still ends where it ended.
    if blob[code_end - start + grow:] == b"\x00" * (_SIM_PAD - grow):
        print("PASS relocate/padding: %d of %d padding byte(s) consumed, the rest "
              "untouched" % (grow, _SIM_PAD))
    else:
        print("FAIL relocate/padding: tail is %s" % blob[code_end - start + grow:].hex())
        rc = 1

    # A kernel with no room must refuse rather than overwrite the next symbol.
    try:
        relocate_region(insns, start, code_end + 4, b"\x00" * 4, edits, arch)
        print("FAIL relocate/room: a %dB growth into 4B of padding was accepted"
              % (4 * len(lg)))
        rc = 1
    except PatchError:
        print("PASS relocate/room: refused to grow %dB into 4B of padding"
              % (4 * len(lg)))

    # Bytes past the consumed padding are carried over, not regenerated: whatever sits
    # out there keeps its address.  Regenerating from a filler word would zero it.
    sentinel = b"\x00" * 16 + b"\xde\xad\xbe\xef" + b"\x00" * (_SIM_PAD - 20)
    try:
        b2, _m = relocate_region(insns, start, stop, sentinel, edits, arch)
        kept = b2[code_end - start + grow:] == sentinel[grow:]
        print("%s relocate/tail: the non-padding word past the consumed padding "
              "survived at its original address" % ("PASS" if kept else "FAIL"))
        rc |= (not kept)
    except PatchError as e:
        print("FAIL relocate/tail: %s" % e)
        rc = 1

    # Padding that is not padding is somebody else's data.
    try:
        relocate_region(insns, start, stop, b"\xde\xad\xbe\xef" + b"\x00" * (_SIM_PAD - 4),
                        edits, arch)
        print("FAIL relocate/room: non-padding bytes after the kernel were consumed")
        rc = 1
    except PatchError:
        print("PASS relocate/room: refused to consume non-padding bytes after the kernel")

    rc |= _check_relocation_objdump(
        insns, {s.index: ["s_waitcnt lgkmcnt(0)", "s_barrier"] for s in lg},
        4 * len(lg), blob, start, arch, "%d s_barrier inserted" % len(lg))

    rc |= _check_relocation_prefix(insns, start, stop, code_end, tail, bar, arch)
    return rc


def _check_relocation_prefix(insns, start, stop, code_end, tail, bar, arch):
    """The other edit shape: a bare s_barrier in FRONT of a site that is kept.

    lgkmcnt0+sbarrier REPLACES a 4-byte site with 8 bytes; sbarrier_only PREFIXES a site
    with 4 and keeps it.  Both grow the kernel 4 bytes per site, so the extent, padding
    and room arithmetic above covers them equally -- what is NOT covered by that pass,
    and is what this one is for, is that relocate_region's new_addr maps a prefixed
    site's old address onto the INSERTED instruction rather than onto the original, and
    that a one-instruction insertion still reads back as a stream llvm-mc accepts.

    Also pins lds_drained_before() against real control flow.  The fixture's two
    ds_read -> ds_write boundaries both sit behind `s_waitcnt vmcnt(7)` -- a wait that
    is not an LGKM drain at all -- so a scan that merely looked for "an s_waitcnt is
    there" would call them drained, and the bare-barrier mutators would under-report
    the caveat on exactly the sites that need it.
    """
    rc = 0
    rw = select_lds_read_then_write(insns, arch, {})
    if not rw:
        print("WARN relocate/prefix: no ds_read -> ds_write boundary in the fixture")
        return rc

    drained = [s for s in rw if lds_drained_before(insns, s.index)]
    if not drained:
        print("PASS lds_drained: 0 of %d ds_read -> ds_write boundar%s drain lgkmcnt(0) "
              "first (%s) -- all guarded by a vmcnt wait, which is not one"
              % (len(rw), "y" if len(rw) == 1 else "ies",
                 " ".join("0x%X" % s.addr for s in rw)))
    else:
        print("FAIL lds_drained: %d of %d boundar(y/ies) reported drained, expected 0 "
              "-- %s" % (len(drained), len(rw),
                         " ".join("0x%X" % s.addr for s in drained)))
        rc = 1

    # The positive case the fixture cannot supply as-is: splice a real lgkmcnt(0) between
    # a real ds_read and a real ds_write.  Real Insn objects, so this tests the scan and
    # not a mock of it.
    reads = [i for i in insns if _lds_kind(i) == "R"]
    writes = [i for i in insns if _lds_kind(i) == "W"]
    drains = [i for i in insns if lgkm_text_count(i) == 0]
    other = [i for i in insns if _lds_kind(i) is None and lgkm_text_count(i) is None]
    if reads and writes and drains and other:
        cases = [([reads[0], drains[0], other[0], writes[0]], True,
                  "ds_read, lgkmcnt(0), s_nop, ds_write"),
                 ([reads[0], other[0], writes[0]], False,
                  "ds_read, s_nop, ds_write -- no drain"),
                 ([drains[0], reads[0], other[0], writes[0]], False,
                  "the drain is BEFORE the read, so it drained nothing relevant")]
        for seq, want, why in cases:
            got = lds_drained_before(seq, len(seq) - 1)
            if got == want:
                print("PASS lds_drained: %-5s for %s" % (got, why))
            else:
                print("FAIL lds_drained: %s for %s, expected %s" % (got, why, want))
                rc = 1
    else:
        print("WARN lds_drained: fixture lacks one of read/write/drain/filler to splice")

    edits = {s.index: bar + enc_bytes(insns[s.index]) for s in rw}
    try:
        blob, new_addr = relocate_region(insns, start, stop, tail, edits, arch)
    except PatchError as e:
        print("FAIL relocate/prefix: %s" % e)
        return rc | 1

    grow = new_addr[code_end] - code_end
    ok = True
    for s in rw:
        at = new_addr[insns[s.index].addr] - start
        if blob[at:at + 4] != bar:                   # the insertion is where the site was
            ok = False
        if blob[at + 4:at + 4 + insns[s.index].size] != enc_bytes(insns[s.index]):
            ok = False                               # and the site itself follows it
    if ok and grow == 4 * len(rw) and len(blob) == stop - start:
        print("PASS relocate/prefix: %d site(s) now read s_barrier + the original "
              "instruction; kernel grew %dB (4B x %d), region size unchanged"
              % (len(rw), grow, len(rw)))
    else:
        print("FAIL relocate/prefix: sites %s, grew %dB (expected %dB), %d byte(s) out "
              "(expected %d)" % ("intact" if ok else "WRONG", grow, 4 * len(rw),
                                 len(blob), stop - start))
        rc = 1

    rc |= _check_relocation_objdump(
        insns, {s.index: ["s_barrier", insns[s.index].text.strip()] for s in rw},
        4 * len(rw), blob, start, arch,
        "%d bare s_barrier inserted, every prefixed site kept" % len(rw),
        tag="prefix-objdump")
    return rc


def _check_relocation_objdump(insns, ins_at, grow, blob, start, arch, what, tag="objdump"):
    """Disassemble the rebuilt bytes with llvm-mc and compare the stream instruction
    by instruction.

    The pure-Python checks above re-decode with the same functions that did the
    encoding, so they cannot catch a shared misunderstanding of the ISA.  This one
    reads the bytes with LLVM's disassembler instead.  Skipped, loudly, when ROCm's
    llvm-mc is not on the box -- it is the only part of --check-fixtures that needs a
    tool, and a missing tool is not a failing test.

    `ins_at` is {instruction index: the exact text stream expected there}, so the same
    check covers an edit that REPLACES its site (lgkmcnt0+sbarrier: two instructions
    where one stood) and one that PREFIXES it (sbarrier_only: a barrier, then the
    original, unchanged).  `grow` is how many bytes longer the rebuilt code is, which
    is what bounds the slice read back -- getting it from the caller rather than from
    len(ins_at) is what lets the two shapes share this.
    """
    import subprocess
    mc = rnan.tool("llvm-mc", required=False)
    if not mc:
        print("SKIP relocate/%s: no llvm-mc on PATH (ROCm not installed here)" % tag)
        return 0
    code = blob[:sum(len(enc_bytes(i)) for i in insns) + grow]
    hexed = " ".join("0x%02x" % b for b in code)
    try:
        r = subprocess.run([mc, "-triple=amdgcn--amdhsa", "-mcpu=%s" % arch,
                            "--disassemble"], input=hexed, capture_output=True, text=True)
    except OSError as e:
        print("SKIP relocate/%s: %s" % (tag, e))
        return 0
    if r.returncode != 0:
        print("FAIL relocate/%s: llvm-mc rejected the rebuilt stream:\n%s"
              % (tag, r.stderr.strip()[:400]))
        return 1

    have = [ln.strip() for ln in r.stdout.splitlines()
            if ln.strip() and not ln.strip().startswith((".", "#"))]
    want = []
    for i, ins in enumerate(insns):
        want.extend(ins_at.get(i) or [ins.text.strip()])
    if len(have) != len(want):
        print("FAIL relocate/%s: llvm-mc read %d instruction(s), expected %d"
              % (tag, len(have), len(want)))
        return 1

    # Branch and getpc-literal operands are EXPECTED to differ -- that is the whole
    # edit -- so they are compared by mnemonic; everything else by full text.
    moving = set()
    for i, ins in enumerate(insns):
        if _BRANCH_RE.match(ins.mnemonic):
            moving.add(ins.addr)
    for i in (j for j, ins in enumerate(insns) if ins.mnemonic == "s_getpc_b64"):
        moving.add(insns[find_pcrel(insns, i).add].addr)
    bad = []
    n = 0
    for i, ins in enumerate(insns):
        got = have[n:n + len(ins_at.get(i) or [None])]
        n += len(got)
        if i in ins_at:
            if got != ins_at[i]:
                bad.append((ins.addr, " | ".join(ins_at[i]), " | ".join(got)))
            continue
        g = got[0]
        if ins.addr in moving:
            if g.split()[0] != ins.mnemonic:
                bad.append((ins.addr, ins.mnemonic, g))
        elif g != ins.text.strip():
            bad.append((ins.addr, ins.text.strip(), g))
    if bad:
        print("FAIL relocate/%s: %d instruction(s) differ, e.g. 0x%X %r -> %r"
              % (tag, len(bad), bad[0][0], bad[0][1], bad[0][2]))
        return 1
    print("PASS relocate/%s: llvm-mc reads back %d instruction(s) -- the original "
          "stream with %s, %d relocated operand(s) re-encoded"
          % (tag, len(have), what, len(moving)))
    return 0


def print_sites(sites, target=None):
    print("%-4s %-12s %-10s %-44s %s" % ("#", "vaddr", "enc", "instruction", "why"))
    print("-" * 118)
    for n, s in enumerate(sites):
        print("%-4d 0x%-10X %-10s %-44s %s"
              % (n, s.addr, (s.enc or b"").hex(), s.text[:44], s.reason))
    print("\n%d site(s)" % len(sites))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-fixtures", action="store_true",
                    help="unit-test the selector against the saved .asm pair")
    ap.add_argument("--fixtures", default=FIXTURES)
    ap.add_argument("--kernel", help="Tensile kernel name, to enumerate against the "
                                     "real installed library")
    ap.add_argument("--selector", default="nan_const", choices=sorted(SELECTORS))
    ap.add_argument("--payload", default="nop", choices=all_payloads())
    ap.add_argument("--libdir")
    ap.add_argument("--arch")
    ap.add_argument("--verify-offsets", action="store_true",
                    help="also check that each site's bytes on disk match the disassembly")
    ap.add_argument("--test-overlay", action="store_true",
                    help="build an overlay, re-disassemble it, and prove the installed "
                         "library was not written through")
    a = ap.parse_args()

    if a.check_fixtures:
        sys.exit(check_fixtures(a.fixtures))

    if a.kernel:
        t = PatchTarget(a.kernel, a.libdir, a.arch)
        print("library  %s" % t.libfile)
        print("elf      %s%s" % (t.elf, "  (unbundled)" if t.bundled else ""))
        print("arch     %s   kernel 0x%X-0x%X   %d instructions"
              % (t.arch, t.start, t.stop, len(t.insns)))
        print("section  sh_addr=0x%X sh_off=0x%X  =>  file_offset = vaddr - 0x%X\n"
              % (t.sh_addr, t.sh_off, t.sh_addr - t.sh_off))
        sites = t.select(a.selector)
        print_sites(sites)
        region = a.payload in REGION_PAYLOADS
        if region:
            last = t.insns[-1]
            print("\npadding  %d byte(s) between the last instruction (0x%X) and the "
                  "next symbol (0x%X)" % (t.stop - (last.addr + last.size),
                                          last.addr + last.size, t.stop))
        if a.verify_offsets:
            try:
                offs = t.offsets_for(sites, a.payload)
                if region:
                    n = sum(len(v) for v in offs.values())
                    print("\nPASS region payload %r rebuilt and re-decoded %d byte(s) "
                          "covering all %d site(s); every branch and getpc displacement "
                          "still resolves" % (a.payload, n, len(sites)))
                else:
                    print("\nPASS all %d site(s) verified against the on-disk bytes; "
                          "payload %r produced length-preserving replacements"
                          % (len(offs), a.payload))
            except PatchError as e:
                print("\nFAIL %s" % e)
                sys.exit(1)

        if a.test_overlay:
            import hashlib
            import tempfile

            def sha(p):
                with open(p, "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()

            before = sha(t.libfile)
            cache = OverlayCache(t, tempfile.mkdtemp(dir=t.workdir))
            rc = 0
            for kind in ("identity", a.payload):
                cache.apply(t.offsets_for(sites, kind))
                # Unbundle what we actually wrote and disassemble the same range.
                d = tempfile.mkdtemp(dir=t.workdir)
                elf, _dev, _tg = rnan.unbundle_device(cache.dst, t.arch, d)
                seq = rnan.disassemble_range(elf, t.start, t.stop, t.arch)
                got = {i.addr: i for i in seq}
                if kind in REGION_PAYLOADS:
                    # A region payload moves instructions, so "did anything change at
                    # 0x...?" stops being the question. The question is whether the
                    # stream llvm-objdump reads back out of the real code object is the
                    # original one with exactly the intended insertions -- which is
                    # also a test of the .symtab relocation, because objdump restarts
                    # decoding at every label and a stale one desynchronises it.
                    pcadd = set()
                    for i, ins in enumerate(t.insns):
                        if ins.mnemonic == "s_getpc_b64":
                            pcadd.add(t.insns[find_pcrel(t.insns, i).add].addr)
                    want = []
                    for i, ins in enumerate(t.insns):
                        ex = (sites_payload_mnemonics(kind) if any(s.index == i
                                                                   for s in sites) else None)
                        # None in a declared stream means "the site's own
                        # instruction", which a prefixing payload keeps.
                        want.extend([(m if m is not None else ins.mnemonic, None)
                                     for m in ex] if ex
                                    else [(ins.mnemonic,
                                           None if ins.addr in pcadd else ins.text.strip())])
                    have = [(i.mnemonic, i.text.strip()) for i in seq]
                    bad = None
                    if len(have) != len(want):
                        bad = "stream is %d instruction(s), expected %d" % (len(have), len(want))
                    else:
                        for n, ((wm, wt), (hm, ht)) in enumerate(zip(want, have)):
                            if wm != hm or (wt is not None and wt != ht):
                                bad = ("#%d: expected %r, got %r"
                                       % (n, wt if wt is not None else wm, ht))
                                break
                    print("%s overlay/%s: %d instructions -> %d; %s"
                          % ("FAIL" if bad else "PASS", kind, len(t.insns), len(seq),
                             bad or ("the original stream with %d insertion(s), every "
                                     "other instruction and operand identical" % len(sites))))
                    rc |= bool(bad)
                    # The counting invariant, derived from what the payload DECLARED it
                    # writes rather than assumed.  This used to assert that no relaxed
                    # lgkmcnt wait survived any region payload, which is true only of
                    # lgkmcnt0+sbarrier: that one REPLACES each selected wait, while a
                    # prefixing payload (a None in the declared stream) leaves every
                    # existing wait exactly as it found it.  The old form therefore
                    # failed sbarrier_before unconditionally, whether or not it worked.
                    decl = sites_payload_mnemonics(kind) or []
                    drains = "s_waitcnt" in decl and None not in decl
                    was = sum(1 for i in t.insns if lgkm_text_count(i))
                    left = sum(1 for i in seq if lgkm_text_count(i))
                    want_left = was - len(sites) if drains else was
                    bars = sum(1 for i in t.insns if i.mnemonic == "s_barrier")
                    got_bars = sum(1 for i in seq if i.mnemonic == "s_barrier")
                    want_bars = bars + decl.count("s_barrier") * len(sites)
                    bad = (left != want_left) or (got_bars != want_bars)
                    print("%s overlay/%s: %d relaxed lgkmcnt wait(s) left, expected %d "
                          "(payload %s); s_barrier %d -> %d, expected %d"
                          % ("FAIL" if bad else "PASS", kind, left, want_left,
                             "drains its sites" if drains else "leaves waits alone",
                             bars, got_bars, want_bars))
                    rc |= bool(bad)
                    continue
                if len(got) != len(t.insns):
                    print("FAIL overlay/%s: %d instructions after round-trip, %d before"
                          % (kind, len(got), len(t.insns)))
                    rc = 1
                    continue
                if kind == "identity":
                    diff = [i for i in t.insns if got[i.addr].text != i.text]
                    print("%s overlay/identity: %d instructions round-trip unchanged%s"
                          % ("PASS" if not diff else "FAIL", len(got),
                             "" if not diff else "; %d differ e.g. %r" % (len(diff), diff[0].text)))
                    rc |= bool(diff)
                else:
                    bad = [s for s in sites if "s_nop" not in got[s.addr].text]
                    print("%s overlay/%s: %d/%d site(s) now read s_nop; %d other "
                          "instructions unchanged"
                          % ("PASS" if not bad else "FAIL", kind, len(sites) - len(bad),
                             len(sites), len(got) - len(sites)))
                    rc |= bool(bad)
                    # The same before/after view a mutation sweep prints for a
                    # finding, here against a known patch -- so the formatter is
                    # exercised on a real code object without a GPU.
                    print("")
                    for ln in format_disasm_diff(t.insns, seq, context=2, limit=4):
                        print(ln)
            after = sha(t.libfile)
            print("%s no write-through: installed .co sha256 %s"
                  % ("PASS" if before == after else "FAIL",
                     "unchanged" if before == after else "CHANGED -- overlay wrote through!"))
            rc |= (before != after)
            print("\n%s" % ("OVERLAY CHECKS PASSED" if not rc else "OVERLAY CHECKS FAILED"))
            sys.exit(1 if rc else 0)
        sys.exit(0)

    ap.print_help()
