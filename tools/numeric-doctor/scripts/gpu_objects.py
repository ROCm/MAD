#!/usr/bin/env python3
"""The GPU code objects inside any binary, per arch -- and what an instruction pattern does in each (debug-numeric step 7).

    python3 gpu_objects.py <file or dir> [--kernel REGEX] [--grep REGEX] [--arch gfx950,gfx942] [--out DIR]

A root cause's source line says what the code *should* do; the binary says what this GPU runs.
Source can be compiled differently per arch (`#if defined(__gfx942__)`), per build flag or per
library version, so a guard visible in the source may be absent from the code object this GPU
loads.  This lists every amdgcn code object a binary carries and, with --grep, counts an
instruction pattern per kernel per arch, so "is the guard in THIS arch's binary?" is one command.

Inputs, any of: a host shared object with embedded device code (a framework's extension
`*.so`, libtorch_hip.so -- its `.hip_fatbin` section, every offload bundle in it), an offload
bundle (`*.co`, `*.hsaco`, plain or compressed), or a bare amdgcn ELF; a directory is walked.
A Tensile library dir with --kernel: give the DIRECTORY, not a guessed .co -- the kernel name is
in the .co's symbols and its .dat index, never in the .co's file name; the .dat picks the files.
Run it where the ROCm llvm tools are -- the model's image (`nd run --gpus 0 -- 'python3 -m
gpu_objects ...'`); no GPU and no torch needed.

Output (one line each):
    OBJECT  <source> [bundle <i>] arch=<gfx> kernels=<n> elf=<extracted ELF>
    GREP    arch=<gfx> kernels <k>/<n> match /REGEX/, <h> hits        (with --grep)
    KERNEL  arch=<gfx> <hits> <symbol>                (with --grep, and --kernel or --per-kernel; 40 max)
    ARCH_DIFF /REGEX/: <gfx_a> <k_a>/<n_a> kernels vs <gfx_b> <k_b>/<n_b>  -- the same kernels, a
              different pattern count: the code differs by arch here
Last line: GPU_OBJECTS=<n objects> arches=<list>
"""
import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_hipblaslt_log_with_nan as hb  # noqa: E402  (tool(), bundle helpers: one copy of them)

_MACH = re.compile(r"EF_AMDGPU_MACH_AMDGCN_(GFX[0-9A-F]+)|Flags:\s+0x[0-9a-f]+,\s*(gfx[0-9a-f]+)", re.I)
_FUNC = re.compile(r"^([0-9a-f]+) <(.+)>:$")


def elf_arch(elf):
    """gfxNNN of an amdgcn ELF, from its header flags."""
    r = hb.run([hb.tool("llvm-readelf"), "--file-headers", elf], check=False)
    m = _MACH.search(r.stdout)
    return (m.group(1) or m.group(2)).lower() if m else None


def is_amdgpu_elf(path):
    with open(path, "rb") as f:
        head = f.read(20)
    return head[:4] == b"\x7fELF" and len(head) >= 20 and head[18:20] == b"\xe0\x00"   # e_machine EM_AMDGPU


def fatbin_bundles(path, work):
    """Offload bundles in a host ELF's .hip_fatbin section, written to files."""
    sec = os.path.join(work, "hip_fatbin.bin")
    r = hb.run([hb.tool("llvm-objcopy"), "--dump-section", ".hip_fatbin=%s" % sec, path,
                os.path.join(work, "host.discard")], check=False)
    for junk in ("host.discard",):
        if os.path.exists(os.path.join(work, junk)):
            os.remove(os.path.join(work, junk))
    if r.returncode or not os.path.exists(sec):
        return []
    data = open(sec, "rb").read()
    os.remove(sec)
    starts = sorted({m.start() for magic in hb.BUNDLE_MAGICS for m in re.finditer(re.escape(magic), data)
                     if magic != b"__CLANG_" or data[m.start():m.start() + 24] == b"__CLANG_OFFLOAD_BUNDLE__"})
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(data)
        p = os.path.join(work, "bundle%d.bin" % i)
        with open(p, "wb") as f:
            f.write(data[s:e])          # as is: trailing zeros can belong to the last code object
        out.append(p)
    return out


def bundle_elves(bundle, work, tag, arches=None):
    """[(target, elf)] for every amdgcn slice of an offload bundle (only `arches`' when given:
    the others are never unbundled)."""
    out = []
    for t in hb.bundle_targets(bundle):
        if "amdgcn" not in t or (arches and not any(t.endswith(x) or (x + ":") in t for x in arches)):
            continue
        elf = os.path.join(work, "%s.%s.elf" % (tag, re.sub(r"[^\w.-]", "_", t.split("-")[-1])))
        r = hb.run([hb.tool("clang-offload-bundler"), "--type=o", "--unbundle", "--input=%s" % bundle,
                    "--targets=%s" % t, "--output=%s" % elf], check=False)
        if r.returncode == 0 and os.path.getsize(elf):
            out.append((t, elf))
    return out


_ARCH_IN_NAME = re.compile(r"gfx[0-9a-f]{3,4}")


_DAT = re.compile(r"\.dat(\.zlib)?$")


def dat_prefilter(files, kre):
    """The files worth unbundling for a --kernel search of a Tensile library dir, or None.

    A Tensile .co is a compressed bundle named after its problem type
    (TensileLibrary_SS_SS_HA_Bias_SAV_MX_..._gfx950.co), not after the kernels in it
    (Cijk_Alik_Bljk_S_MX_B_Bias_HA_S_SAV_..._MT64x128x128_...): neither its name nor a
    `grep -a` of its bytes finds the symbol, and 2026-11-23 (qwen_tf32_hipblaslt step 7)
    four jobs probing guessed file names came back "no kernel matching --kernel" while the
    kernel sat in a sibling file.  Its uncompressed .dat index names every kernel, so: the
    .co whose .dat matches, plus every code object no .dat indexes (Kernels.so-*.hsaco,
    extension libraries) -- seconds instead of unbundling the whole dir.  None (walk it
    all) when the dir has no .dat or no .dat matches (a regex anchored to the symbol's
    ends cannot match inside an index blob)."""
    dats = [f for f in files if _DAT.search(f)]
    if not dats:
        return None
    bre = re.compile(kre.pattern.encode(), kre.flags & ~re.UNICODE)
    indexed, hit = set(), set()
    for d in dats:
        co = hb.dat_sibling_co(d)
        indexed.add(co)
        try:
            if bre.search(hb.read_dat(d)):
                hit.add(co)
        except Exception:                                # noqa: BLE001 -- unreadable index: unbundle its .co
            hit.add(co)
    if not hit:
        return None
    return [f for f in files if not _DAT.search(f) and (f in hit or f not in indexed)]


def objects(path, work, arches=None, kre=None):
    """(source, label, elf, scratch dir) for every amdgcn code object under path, one file at a
    time (a library dir holds hundreds of bundles: extracting them all first cost 59 GB, 2026-11-21).
    With arches, a file whose NAME names only other arches (TensileLibrary_..._gfx942.co) is skipped
    unread.  With kre (--kernel) on a directory, a Tensile library's .dat index picks the code
    objects to open (dat_prefilter)."""
    files = [path] if os.path.isfile(path) else sorted(
        os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs)
    if kre is not None and not os.path.isfile(path):
        pick = dat_prefilter(files, kre)
        print("SEARCH  %s: %s" % (path, "%d of %d file(s) by the Tensile .dat index (--kernel)"
                                  % (len(pick), len(files)) if pick is not None
                                  else "all %d file(s) (no .dat index names --kernel)" % len(files)),
              flush=True)
        if pick is not None:
            files = pick
    for f in files:
        named = set(_ARCH_IN_NAME.findall(os.path.basename(f)))
        if arches and named and not (named & arches):
            continue
        try:
            with open(f, "rb") as fh:
                head = fh.read(24)
        except OSError:
            continue
        sub = tempfile.mkdtemp(dir=work)
        if head.startswith(hb.BUNDLE_MAGICS):
            items = [(f, "", e) for t, e in bundle_elves(f, sub, "b", arches)]
        elif head[:4] == b"\x7fELF" and is_amdgpu_elf(f):
            items = [(f, "", f)]
        elif head[:4] == b"\x7fELF":
            items = []
            for i, b in enumerate(fatbin_bundles(f, sub)):
                items += [(f, "bundle %d" % i, e) for t, e in bundle_elves(b, sub, "b%d" % i, arches)]
                os.remove(b)
        else:
            items = []
        for it in items:
            yield it + (sub,)           # the caller deletes an ELF it does not keep
        if not any(n.endswith(".elf") for n in os.listdir(sub)):
            shutil.rmtree(sub, ignore_errors=True)


_SYM = re.compile(r"^\s*\d+:\s+([0-9a-f]+)\s+\d+\s+FUNC\s+\S+\s+\S+\s+\S+\s+(\S+)\s*$", re.M)
_ADDR = re.compile(r"//\s*([0-9A-Fa-f]{6,}):")      # llvm-objdump's per-instruction address comment


def symbols(elf):
    """(kernels, [(addr, name)] of every FUNC sorted): kernel names are those with a descriptor
    `<name>.kd`, else every FUNC.  Instructions are attributed by address -- a code object's
    kernel symbols can have size 0 and be shadowed by local labels (Tensile: label_ASM_Start)."""
    r = hb.run([hb.tool("llvm-readelf"), "--symbols", "--wide", elf], check=False)
    funcs = sorted({(int(a, 16), n) for a, n in _SYM.findall(r.stdout)})
    kds = set(re.findall(r"\s(\S+)\.kd\s*$", r.stdout, re.M))
    return (kds or {n for _, n in funcs}), funcs


def kernels(elf):
    return symbols(elf)[0]


def grep_counts(elf, arch, pat, keep):
    """{kernel: hits of pat in its disassembly} for the kernels in keep."""
    import bisect
    funcs = symbols(elf)[1]
    starts = [a for a, _ in funcs]
    r = hb.run([hb.tool("llvm-objdump"), "-d", "--mcpu=%s" % arch, elf], check=False)
    counts = collections.Counter({n: 0 for _, n in funcs if n in keep})
    cur = None
    for line in r.stdout.splitlines():
        m = _ADDR.search(line)
        if m:
            i = bisect.bisect_right(starts, int(m.group(1), 16)) - 1
            cur = funcs[i][1] if i >= 0 and funcs[i][1] in keep else None
        else:
            lm = _FUNC.match(line)          # no address comment (other targets): fall back to labels
            if lm and lm.group(2) in keep:
                cur = lm.group(2)
            continue
        if cur is not None and pat.search(line):
            counts[cur] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", nargs="+", help=".so / .co / .hsaco / amdgcn ELF files, or directories")
    ap.add_argument("--kernel", help="regex on kernel symbols (default: all)")
    ap.add_argument("--grep", help="regex on disassembly lines, counted per kernel per arch")
    ap.add_argument("--arch", help="only these arches, comma-separated")
    ap.add_argument("--out", help="keep the extracted ELFs here (default: a temp dir)")
    ap.add_argument("--per-kernel", action="store_true",
                    help="with --grep: also a KERNEL line per kernel and arch (its hit count); implied by --kernel")
    ap.add_argument("--first", action="store_true",
                    help="stop at the first code object per arch with a kernel matching --kernel (a library dir)")
    a = ap.parse_args()
    bad = [p for p in a.path if not p or not os.path.exists(p)]
    if bad:     # an unset shell variable read as "searched 0 files, no object" (2026-11-27)
        ap.error("no such file or directory: %s" % ", ".join(repr(p) for p in bad))
    work = a.out or tempfile.mkdtemp(prefix="gpu_objects.")
    os.makedirs(work, exist_ok=True)
    want = set(a.arch.split(",")) if a.arch else None
    kre = re.compile(a.kernel) if a.kernel else None
    pat = re.compile(a.grep) if a.grep else None
    per_arch = collections.defaultdict(collections.Counter)    # arch -> kernel -> hits
    n_obj, seen = 0, set()
    done = set()
    for src, label, elf, sub in (o for p in a.path for o in objects(p, work, want, kre)):
        arch = elf_arch(elf)
        ks = {k for k in kernels(elf) if not kre or kre.search(k)} if not (want and arch not in want) else set()
        if not ks or (a.first and arch in done):
            if elf.startswith(sub):
                os.remove(elf)          # keep only the code objects that matched (--out)
            continue
        done.add(arch)
        n_obj += 1
        seen.add(arch)
        print("OBJECT  %s%s arch=%s kernels=%d elf=%s" % (src, " [%s]" % label if label else "", arch, len(ks), elf))
        if pat:
            per_arch[arch].update(grep_counts(elf, arch, pat, ks))
        if not a.out and elf.startswith(sub):
            os.remove(elf)
    if pat:
        for arch, c in sorted(per_arch.items()):
            print("GREP    arch=%s kernels %d/%d match /%s/, %d hits"
                  % (arch, sum(1 for v in c.values() if v), len(c), a.grep, sum(c.values())))
            if a.per_kernel or kre:
                for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:40]:
                    print("KERNEL  arch=%s %d %s" % (arch, v, k))
        # kernels every arch has, and how many of them carry the pattern per arch
        common = set.intersection(*(set(c) for c in per_arch.values())) if per_arch else set()
        by = collections.defaultdict(list)
        for arch, c in per_arch.items():
            by[sum(1 for k in common if c[k])].append(arch)
        if common and len(by) > 1:
            print("ARCH_DIFF /%s/ in the %d kernels every arch has: %s -- the same source, different code: "
                  "the code differs by arch here" % (a.grep, len(common), "; ".join(
                      "%d/%d in %s" % (n, len(common), ",".join(sorted(v))) for n, v in sorted(by.items(), reverse=True))))
    if pat and not per_arch:
        print("GREP    no amdgcn code object%s found" % (" with a kernel matching --kernel" if kre else ""))
    print("GPU_OBJECTS=%d arches=%s" % (n_obj, ",".join(sorted(a for a in seen if a))))


if __name__ == "__main__":
    main()
