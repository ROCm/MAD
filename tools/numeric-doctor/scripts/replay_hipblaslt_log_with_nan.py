#!/usr/bin/env python3
"""Replay the GEMMs in a hipBLASLt profile log, with a NaN injected into each kernel's ISA.

This is `replay_hipblaslt_log.py` with the dispatch path changed:

    replay_hipblaslt_log.py      log -> torch -> hipBLASLt -> stock Tensile kernel -> clean out
    this script                  log -> for every logged GEMM, take its kernel_name
                                      -> patch that kernel's ISA so the stored value is a NaN
                                      -> point hipBLASLt at the patched code object
                                      -> torch -> hipBLASLt -> PATCHED kernel -> NaN out

The stock GEMM is never dispatched: the patch is applied before `import torch`, because
hipBLASLt reads HIPBLASLT_TENSILE_LIBPATH when its .so initializes and the Tensile code
object is mmap'd at first use.  There is no run-then-patch-then-rerun; there is one run,
and it is the patched one.

The log is a *list* of GEMMs
----------------------------
A HIPBLASLT_LOG_MASK=64 profile log is one line per distinct problem, aggregated over the
whole run (`call_count` is how often it ran):

    - { function: matmul, M: 64, N: 64, K: 1, ..., kernel_name: Cijk_Ailk_..., call_count: 2 }
    - { function: matmul, M: 128, ...,          kernel_name: Cijk_Ailk_..., call_count: 1 }

Every such line is analysed.  A single-line, no-list log (the older format) is accepted too.

How the patch works
-------------------
Tensile epilogues end in `buffer_store_dwordx*` / `global_store_dwordx*` of the accumulator
VGPRs.  We disassemble the kernel, take the data operand of a store, scan backwards for the
last VALU instruction that writes one of those registers, and overwrite that instruction with

    v_mov_b32_e32 v<data>, -1        # 0xFFFFFFFF, i.e. a negative quiet NaN as f32

padded with `s_nop 0` so the byte length -- and therefore every branch target in the kernel --
is unchanged.  `-1` is an inline constant, so the replacement is 4 bytes and fits in any slot.

From ROCm 6.3 the TensileLibrary_*.co files are clang offload bundles (compressed, "CCOB",
since 7.0), so the flow is unbundle -> patch the amdgcn slice -> re-bundle.

Nothing is modified in place: the patched library files are written into a temporary overlay
of the hipBLASLt library directory (everything else symlinked) and selected via
HIPBLASLT_TENSILE_LIBPATH.  The installed ROCm tree is left alone.

Picking the kernel
------------------
The log names the kernel directly (`kernel_name:`), and that string is the ELF symbol, so
there is no solution_index -> *.dat lookup any more: the name is located by scanning the
`*_<arch>.dat` libraries for it (they store it verbatim) and patching the sibling `.co`.
solution_index is only carried through to pin hipBLASLt's choice at dispatch time.

The log must come from the same hipBLASLt build that is on the machine, or the kernel names
will not exist.  These logs were captured inside a ROCm 7.2.0 PyTorch image,
so run this script inside the container the log came from -- see scripts/docker_run.sh.

Usage:
    python replay_hipblaslt_log_with_nan.py hipblastlt_logs/39813-full.log --run none
    python replay_hipblaslt_log_with_nan.py <log> --order-log hipblastlt_logs/38936.log
    python replay_hipblaslt_log_with_nan.py <log> --gemm 57 --disasm context

Exit status:
    0  analysis done, and any GEMM that was dispatched came back NaN (the expected result)
    1  a GEMM was dispatched but produced no NaN -- the patched kernel was not the one used
    2  nothing to do: no GEMM in the log could be poisoned, or none is replayable
"""

import argparse
import glob
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

# hipBLASLt reads its environment when the .so initializes, which happens well after
# this point, so setting os.environ here (before `import torch`) is early enough.


# ======================================================================================
# log parsing
# ======================================================================================

class Gemm:
    """One logged problem: the parsed fields plus where it sat in the log."""

    __slots__ = ("pos", "fields", "kernel", "analysis", "run")

    def __init__(self, pos, fields):
        self.pos = pos                                # 1-based line number in the log
        self.fields = fields
        self.kernel = fields.get("kernel_name")
        self.analysis = None                          # KernelAnalysis, filled in later
        self.run = None                               # dispatch result, filled in later

    def __getitem__(self, k):
        return self.fields[k]

    def get(self, k, default=None):
        return self.fields.get(k, default)

    @property
    def shape(self):
        p = self.fields
        return (f"M={p.get('M')} N={p.get('N')} K={p.get('K')} "
                f"transA={p.get('transA')} transB={p.get('transB')} "
                f"batch={p.get('batch_count')} alpha={p.get('alpha')} beta={p.get('beta')} "
                f"call_count={p.get('call_count')}")

    @property
    def key(self):
        """What identifies this problem across logs of the same run."""
        p = self.fields
        return tuple(str(p.get(k)) for k in
                     ("M", "N", "K", "transA", "transB", "batch_count"))


def _coerce(v):
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    if v in ("true", "false"):
        return v == "true"
    return v


def parse_fields(line):
    """'- { function: matmul, K: V, ... }' or 'function: matmul, K: V, ...' -> dict."""
    s = line.strip()
    if s.startswith("-"):
        s = s[1:].strip()
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    fields = {}
    for kv in s.split(","):
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        fields[k.strip()] = _coerce(v.strip())
    return fields


def parse_log(path):
    """Every 'function: ...' line in the log, in file order."""
    with open(path) as f:
        lines = [ln for ln in f if "function:" in ln]
    if not lines:
        sys.exit(f"{path}: no 'function:' line found")
    return [Gemm(i, parse_fields(ln)) for i, ln in enumerate(lines, 1)]


def dispatch_blockers(p):
    """Why this GEMM cannot be *replayed* through torch.  Patching does not care."""
    bad = []
    if p.get("function") != "matmul":
        bad.append(f"function={p.get('function')}")
    if p.get("batch_count", 1) != 1:
        bad.append(f"batch_count={p.get('batch_count')}")
    if p.get("bias_vector"):
        bad.append("bias_vector")
    if p.get("activation_type", "none") != "none":
        bad.append(f"activation_type={p.get('activation_type')}")
    if p.get("gradient") or p.get("use_e"):
        bad.append("gradient/use_e epilogue")
    if any(p.get(s, 0) not in (0, "0") for s in ("scaleA", "scaleB", "scaleC", "scaleD")):
        bad.append("per-tensor scaling")
    if p.get("swizzleA") or p.get("swizzleB"):
        bad.append("swizzled A/B")
    if p.get("beta", 0.0) != 0.0:
        bad.append(f"beta={p.get('beta')} (needs a C operand)")
    for t in ("a_type", "b_type", "c_type", "d_type"):
        if p.get(t) != "f32_r":
            bad.append(f"{t}={p.get(t)}")
    return bad


# ======================================================================================
# execution order
#
# The tie-break in the final report wants "the GEMM most likely to be earliest in the
# model execution".  The profile log's own line order does NOT answer that: it is the
# iteration order of hipBLASLt's hash map, and against the per-call trace 38936.log (the
# same 148 problems) its Spearman correlation with true first-call order is 0.03 for
# 39813-full.log and 0.02 for 38961.log -- i.e. none.  So either supply a per-call
# bench trace with --order-log (ground truth), or fall back to the shape heuristic below.
# ======================================================================================

# Measured on 38936.log (34650 calls, 148 distinct problems): mean first-call rank by
# (transA, transB) is TN 63.1, NN 67.7, NT 89.0, and the nine earliest calls of the step
# are all TN.  Forward-shaped (W^T . X) GEMMs run first; dgrad-shaped (NT) run last.
_TRANS_RANK = {("T", "N"): 0, ("N", "N"): 1, ("N", "T"): 2}


class ExecOrder:
    """Ranks GEMMs by how early they run.  Lower is earlier."""

    def __init__(self, trace_path=None):
        self.first = {}
        self.source = "heuristic (shape); pass --order-log for the real order"
        if not trace_path:
            return
        with open(trace_path) as f:
            text = f.read()
        if "hipblaslt-bench" in text:
            for line in text.splitlines():
                d = dict(re.findall(r"--?([A-Za-z_]+)\s+([^\s-]\S*)", line))
                key = (d.get("m"), d.get("n"), d.get("k"),
                       d.get("transA"), d.get("transB"), d.get("batch_count", "1"))
                self.first.setdefault(key, len(self.first))
            self.source = f"first-call order from {os.path.basename(trace_path)}"
        else:
            print(f"warning: {trace_path} is not a hipblaslt-bench per-call trace "
                  f"(no 'hipblaslt-bench' lines); ignoring it -- an aggregated profile "
                  f"log carries no execution order", file=sys.stderr)

    def rank(self, g):
        """Sort key: (has_true_order, true_rank_or_shape_rank, log_position)."""
        if g.key in self.first:
            return (0, self.first[g.key], g.pos)
        t = (str(g.get("transA")), str(g.get("transB")))
        return (1, _TRANS_RANK.get(t, 3), g.pos)

    def describe(self, g):
        if g.key in self.first:
            return f"first-call #{self.first[g.key] + 1}"
        return f"{g.get('transA')}{g.get('transB')}-shaped, log line {g.pos}"


# ======================================================================================
# ROCm llvm tooling
# ======================================================================================

def rocm_root():
    for cand in (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"), "/opt/rocm"):
        if cand and os.path.isdir(cand):
            return cand
    hits = sorted(glob.glob("/opt/rocm-*"))
    return hits[-1] if hits else "/opt/rocm"


_TOOL_CACHE = {}


def tool(name, required=True):
    """Locate an llvm tool, preferring the ROCm llvm that matches the code objects."""
    if name in _TOOL_CACHE:
        found = _TOOL_CACHE[name]
    else:
        found = None
        for d in (os.path.join(rocm_root(), "llvm", "bin"),
                  os.path.join(rocm_root(), "bin")):
            cand = os.path.join(d, name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                found = cand
                break
        found = found or shutil.which(name)
        if not found:
            # pip ROCm SDK wheels without ROCM_PATH or the tools on PATH (the torchtitan image,
            # rocm-sdk-core 7.14: _rocm_sdk_core/lib/llvm/bin) -- 2026-11-21
            for pat in ("/usr/local/lib/python3*/*-packages/_rocm_sdk_*/lib/llvm/bin/" + name,
                        "/usr/lib/python3*/*-packages/_rocm_sdk_*/lib/llvm/bin/" + name,
                        "/opt/rocm*/lib/llvm/bin/" + name):
                hits = [h for h in sorted(glob.glob(pat)) if os.access(h, os.X_OK)]
                if hits:
                    found = hits[0]
                    break
        _TOOL_CACHE[name] = found
    if required and not found:
        sys.exit(f"{name} not found (looked in {rocm_root()}/llvm/bin and $PATH)")
    return found


def run(cmd, check=True):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr.strip()}")
    return r


def gpu_arch(gemms):
    """The arch the patched kernel has to match.

    Prefer the GPU actually in this machine; with no GPU visible (analysis-only runs on a
    login node) fall back to the ISA tag Tensile bakes into every kernel name -- e.g.
    ..._ISA950_... -> gfx950.
    """
    for exe in ("amdgpu-arch", "rocminfo"):
        path = tool(exe, required=False) or shutil.which(exe)
        if not path:
            continue
        r = run([path], check=False)
        m = re.search(r"(gfx[0-9a-f]+[a-z]*)", r.stdout)
        if m:
            return m.group(1)
    tags = {m.group(1) for g in gemms if g.kernel
            for m in [re.search(r"_ISA([0-9a-z]+)_", g.kernel)] if m}
    if len(tags) == 1:
        arch = "gfx" + tags.pop()
        print(f"note: no GPU visible; taking arch {arch} from the kernel names in the log",
              file=sys.stderr)
        return arch
    sys.exit("cannot determine the GPU arch (no rocminfo/amdgpu-arch); pass --arch")


# ======================================================================================
# clang offload bundles
#
# ROCm >= 6.3 ships TensileLibrary_*.co as clang offload bundles, and from 7.0 they are
# *compressed* ones ("CCOB" magic).  llvm-nm/objdump cannot read either, and a compressed
# bundle is zstd-framed, so even a raw byte search for the kernel name misses.  Unbundle
# to a plain ELF, patch that, then re-bundle for the overlay.
# ======================================================================================

BUNDLE_MAGICS = (b"CCOB", b"__CLANG_")


def is_bundle(path):
    with open(path, "rb") as f:
        return f.read(8).startswith(BUNDLE_MAGICS)


def bundle_targets(path):
    r = run([tool("clang-offload-bundler"), "--type=o", "--list", f"--input={path}"],
            check=False)
    return [t for t in r.stdout.split() if t.strip()]


def unbundle(path, target, out):
    run([tool("clang-offload-bundler"), "--type=o", "--unbundle", f"--input={path}",
         f"--targets={target}", f"--output={out}"])
    return out


def unbundle_device(path, arch, workdir):
    """Extract the amdgcn slice matching `arch`; returns (elf_path, target, all_targets)."""
    targets = bundle_targets(path)
    dev = [t for t in targets if "amdgcn" in t and arch in t]
    if not dev:
        dev = [t for t in targets if "amdgcn" in t]
    if not dev:
        sys.exit(f"{path}: offload bundle has no amdgcn slice (targets: {targets})")
    out = os.path.join(workdir, "device.elf")
    unbundle(path, dev[0], out)
    return out, dev[0], targets


def rebundle(orig, dev_target, dev_elf, targets, workdir, out_path):
    """Rebuild a bundle with the patched device slice, keeping every other slice as-is."""
    inputs, names = [], []
    for t in targets:
        if t == dev_target:
            inputs.append(dev_elf)
        else:
            side = os.path.join(workdir, "slice_" + re.sub(r"\W+", "_", t))
            unbundle(orig, t, side)
            inputs.append(side)
        names.append(t)
    cmd = [tool("clang-offload-bundler"), "--type=o", f"--targets={','.join(names)}"]
    for i in inputs:
        cmd.append(f"--input={i}")
    cmd.append(f"--output={out_path}")
    run(cmd)
    return out_path


# ======================================================================================
# locating a kernel by name
# ======================================================================================

def read_dat(path):
    """The bytes of a Tensile .dat, or of a zlib-compressed .dat.zlib (ROCm 7.x pip SDK wheels)."""
    with open(path, "rb") as f:
        blob = f.read()
    if path.endswith(".zlib"):
        import zlib
        blob = zlib.decompress(blob)
    return blob


def dat_globs(root, arch):
    """Every Tensile index under `root` for `arch`: *.dat and *.dat.zlib."""
    out = []
    for ext in (".dat", ".dat.zlib"):
        out += glob.glob(os.path.join(root, "**", "*_%s%s" % (arch, ext)), recursive=True)
    return sorted(out)


def dat_sibling_co(dat):
    """The code object a .dat / .dat.zlib indexes."""
    return re.sub(r"\.dat(\.zlib)?$", ".co", dat)


def find_library_dir(explicit):
    if explicit:
        if not os.path.isdir(explicit):
            sys.exit(f"--libdir {explicit}: not a directory")
        return explicit
    cands = [os.environ.get("HIPBLASLT_TENSILE_LIBPATH")]
    root = rocm_root()
    cands += [os.path.join(root, "lib", "hipblaslt", "library"),
              os.path.join(root, "lib", "hipblaslt"),
              "/usr/local/lib/hipblaslt/library"]
    # torch ships its own bundled hipblaslt in some ROCm wheels
    cands += sorted(glob.glob("/usr/lib/python3*/dist-packages/torch/lib/hipblaslt/library"))
    cands += sorted(glob.glob("/opt/conda/**/site-packages/torch/lib/hipblaslt/library",
                              recursive=True))
    # pip ROCm SDK wheels (vllm/vllm-openai-rocm v0.33, the torchtitan image): the library sits in
    # _rocm_sdk_libraries, and torch in /usr/local/lib/python3.x/{site,dist}-packages -- neither was
    # searched, so a Tensile target there failed "cannot prepare" (2026-11-20/21)
    for pre in ("/usr/local/lib", "/usr/lib"):
        for pk in ("site-packages", "dist-packages"):
            cands += sorted(glob.glob("%s/python3*/%s/_rocm_sdk_libraries*/lib/hipblaslt/library" % (pre, pk)))
            cands += sorted(glob.glob("%s/python3*/%s/torch/lib/hipblaslt/library" % (pre, pk)))
    for c in cands:
        # files directly in it, or one level down in per-arch dirs (library/gfx950/*.co: the
        # pip ROCm SDK layout; hipBLASLt appends the arch itself, so the parent is the libdir)
        if c and os.path.isdir(c) and any(glob.glob(os.path.join(c, sub, pat))
                                          for sub in ("", "*") for pat in ("*.co", "*.hsaco", "*.dat")):
            return c
    sys.exit("could not find the hipBLASLt Tensile library dir; pass --libdir")


class IndexLookupError(Exception):
    """A solution index could not be mapped to a kernel name."""


def kernel_name_for_index(index, libdir=None, arch=None):
    """The Tensile kernelName for a solution index, read straight from the *.dat set.

    The extended profile (HIPBLASLT_LOG_MASK bit 128) is the library's own answer to
    "which kernel served this GEMM", and it is the one to ask first.  But asking routes
    every matmul through getKernelNameFromAlgoIndex(), which in some builds -- ROCm
    7.2.26015 / libhipblaslt.so.1.2 among them -- segfaults inside
    MasterSolutionLibrary::loadLibrary(int).  A process that dies on its first GEMM
    flushes no profile at all, so bit 128 fails by leaving a zero-byte log rather than
    by saying anything.  The plain profile (bit 64) never crashes and carries
    solution_index, so the name can be recovered here instead, without the library.

    Solution indices are globally unique across one arch's *.dat set -- measured on
    ROCm 7.2 gfx950: 63282 solutions across 222 files, disjoint index ranges, no index
    in two files -- so the first hit is the answer.  They are NOT unique across archs
    (each arch numbers from 0), which is why this scan is arch-scoped and why passing
    the wrong arch would return a confidently wrong name.
    """
    try:
        import msgpack
    except ImportError as e:
        raise IndexLookupError(
            "python msgpack is not available (%s), so solution index %s cannot be "
            "mapped to a kernel name" % (e, index))

    libdir = find_library_dir(libdir)
    dats = dat_globs(libdir, arch)
    if not dats:
        raise IndexLookupError(f"no *_{arch}.dat(.zlib) under {libdir}")
    for d in dats:
        # The lazy master carries the placeholder library and an empty solution list;
        # it is the one .dat that can never hold the answer.
        if "lazy" in os.path.basename(d):
            continue
        try:
            blob = msgpack.unpackb(read_dat(d), raw=False, strict_map_key=False)
        except Exception:                             # noqa: BLE001 -- a bad .dat is not fatal
            continue
        for s in blob.get("solutions") or []:
            if s.get("index") == index:
                name = s.get("kernelName") or s.get("name")
                if not name:
                    raise IndexLookupError(
                        "solution %s in %s has no kernelName" % (index, os.path.basename(d)))
                return name, os.path.basename(d)
    raise IndexLookupError(
        "solution index %s is in none of the %d *_%s.dat files under %s -- is the arch "
        "right?" % (index, len(dats), arch, libdir))


_SYM_RE = re.compile(r"^\s*\d+:\s+([0-9a-fA-F]+)\s+\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s*$")


class KernelFinder:
    """kernel_name -> (library file, readable ELF, symbol), with everything cached.

    Tensile stores kernelName verbatim in the msgpack *.dat next to the code object, and
    the *.dat is not compressed -- so one byte-scan of the `*_<arch>.dat` set maps every
    name in the log to its library in well under a second.  The `.co` itself is a
    compressed offload bundle in ROCm >= 7.0, so searching it for the name would miss.
    """

    def __init__(self, libdir, arch, workdir):
        self.libdir, self.arch, self.workdir = libdir, arch, workdir
        self._lib_of = {}                  # kernel name -> library file (.co/.hsaco)
        self._elf_of = {}                  # library file -> readable ELF (unbundled once)
        self._funcs = {}                   # ELF -> [(addr, name)] sorted
        self._indexed = False

    # -- one pass over the *.dat set ---------------------------------------------------

    def index(self, names):
        names = sorted({n for n in names if n})
        if not names:
            self._indexed = True
            return
        dats = dat_globs(self.libdir, self.arch)
        if not dats:
            dats = sorted(glob.glob(os.path.join(self.libdir, "**", "*.dat"), recursive=True)
                          + glob.glob(os.path.join(self.libdir, "**", "*.dat.zlib"), recursive=True))
        needles = {n: n.encode() for n in names}
        left = set(names)
        for d in dats:
            if not left:
                break
            try:
                blob = read_dat(d)
            except Exception:                         # noqa: BLE001 -- unreadable index: skip it
                continue
            for n in list(left):
                if needles[n] in blob:
                    co = dat_sibling_co(d)
                    if os.path.isfile(co):
                        self._lib_of[n] = co
                        left.discard(n)
        self._indexed = True
        if left:
            # Not in any *.dat, or the sibling .co is missing: fall back to a symbol scan
            # over the arch's code objects.  Rare, and only paid for the stragglers.
            self._scan_code_objects(left)

    def _scan_code_objects(self, names):
        cos = sorted(glob.glob(os.path.join(self.libdir, "**", f"*{self.arch}*.hsaco"),
                               recursive=True))
        cos += sorted(glob.glob(os.path.join(self.libdir, "**", f"*{self.arch}*.co"),
                                recursive=True))
        for co in cos:
            if not names:
                return
            try:
                funcs = {n for _a, n in self.funcs(self.elf_for(co))}
            except SystemExit:
                continue
            for n in list(names):
                if n in funcs:
                    self._lib_of[n] = co
                    names.discard(n)

    # -- per-library caches -------------------------------------------------------------

    def elf_for(self, libfile):
        """A plain ELF for `libfile` -- the file itself, or its extracted amdgcn slice."""
        if libfile not in self._elf_of:
            if is_bundle(libfile):
                d = tempfile.mkdtemp(dir=self.workdir)
                elf, _t, _ts = unbundle_device(libfile, self.arch, d)
                self._elf_of[libfile] = elf
            else:
                self._elf_of[libfile] = libfile
        return self._elf_of[libfile]

    def funcs(self, elf):
        """[(addr, name)] for every FUNC symbol, sorted by address.

        Kernel symbols carry size 0 in Tensile code objects, so the end of a kernel is the
        start of the next FUNC -- which is also why `--disassemble-symbols` is useless
        here: it stops at the first *local label* symbol, a handful of instructions in.
        """
        if elf not in self._funcs:
            r = run([tool("llvm-readelf"), "--symbols", "--wide", elf], check=False)
            out = set()
            for line in r.stdout.splitlines():
                m = _SYM_RE.match(line)
                if m and m.group(2) == "FUNC":
                    out.add((int(m.group(1), 16), m.group(3)))
            self._funcs[elf] = sorted(out)
        return self._funcs[elf]

    # -- lookup --------------------------------------------------------------------------

    def locate(self, name):
        """(libfile, elf) for `name`, or None.

        The match is exact: Tensile kernel names differ only in trailing scheduling
        suffixes, so a substring match would happily land on the wrong kernel (the log's
        first kernel name is a prefix of six distinct symbols in its code object).
        """
        if not self._indexed:
            self.index([name])
        libfile = self._lib_of.get(name)
        if not libfile:
            return None
        elf = self.elf_for(libfile)
        if any(n == name for _a, n in self.funcs(elf)):
            return libfile, elf
        return None


# ======================================================================================
# disassembly
# ======================================================================================

class Insn:
    __slots__ = ("addr", "size", "text", "mnemonic", "operands", "line")

    def __init__(self, addr, size, text, line):
        self.addr, self.size, self.text, self.line = addr, size, text, line
        parts = text.split(None, 1)
        self.mnemonic = parts[0] if parts else ""
        self.operands = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []

    def __repr__(self):
        return f"0x{self.addr:x} ({self.size}B) {self.text}"


# llvm-objdump for amdgcn emits:  "\ttext...   // 000000001008: 7E000280"
#
# ...and, when a branch target resolves to a symbol, a trailing annotation:
#
#   s_cbranch_scc1 2283   // 0000000010D8: BF8508EB <_ZN7ck_tile9kentry_pt...>
#
# The `$` used to sit straight after the words, so every annotated line failed to
# match and the instruction was DROPPED -- silently, because a dropped line is
# indistinguishable from a blank one here.  Tensile kernels never tripped it (their
# branch targets are local labels objdump does not annotate), but a CK kernel has
# nine, and the result was a disassembly missing nine instructions: relocation
# computed the wrong addresses for everything after the first gap.
# kernel_patch.relocate_region's contiguity check is what caught it.
_AMD_RE = re.compile(r"^\s*(?P<text>\S.*?)\s*//\s*(?P<addr>[0-9A-Fa-f]+):\s*"
                     r"(?P<words>(?:[0-9A-Fa-f]{8}\s+)*[0-9A-Fa-f]{8})"
                     r"\s*(?:<[^>]*>)?\s*$")
# generic fallback:  "    1008: 80 02 00 7e   v_mov_b32_e32 v0, 0"
_GEN_RE = re.compile(r"^\s*(?P<addr>[0-9a-fA-F]+):\s*(?P<bytes>(?:[0-9a-fA-F]{2}\s)+)\s*"
                     r"(?P<text>\S.*)$")


def symbol_range(finder, elf, symbol):
    """[start, stop) file addresses of a kernel.

    Tensile kernel symbols have size 0, so the kernel ends at the next FUNC symbol, or at
    the end of its section.  Do NOT use --disassemble-symbols: llvm-objdump ends the
    symbol at the next entry in .symtab, and Tensile emits ~170 local branch labels per
    kernel, so it yields the first dozen instructions and no epilogue.
    """
    funcs = finder.funcs(elf)
    start = next((a for a, n in funcs if n == symbol), None)
    if start is None:
        sys.exit(f"{symbol}: no FUNC symbol in {elf}")
    later = [a for a, _ in funcs if a > start]
    if later:
        return start, min(later)
    sa, _so, sz = section_of(elf, start)
    return start, sa + sz


def _parse_disasm(raw):
    insns = []
    for line in raw.splitlines():
        if line.rstrip().endswith(":") and "<" in line:
            continue
        m = _AMD_RE.match(line)
        if m:
            words = m.group("words").split()
            insns.append(Insn(int(m.group("addr"), 16), 4 * len(words), m.group("text"), line))
            continue
        m = _GEN_RE.match(line)
        if m:
            nbytes = len(m.group("bytes").split())
            insns.append(Insn(int(m.group("addr"), 16), nbytes, m.group("text"), line))
    return insns


def disassemble(finder, elf, symbol, arch):
    """Return list[Insn] for one kernel symbol."""
    objdump = tool("llvm-objdump")
    start, stop = symbol_range(finder, elf, symbol)
    r = run([objdump, "-d", f"--mcpu={arch}",
             f"--start-address=0x{start:x}", f"--stop-address=0x{stop:x}", elf], check=False)
    if r.returncode != 0 or not r.stdout.strip():
        r = run([objdump, "-d", f"--mcpu={arch}", elf])
        r.stdout = _slice_symbol(r.stdout, symbol)
    insns = _parse_disasm(r.stdout)
    if not insns:
        sys.exit(f"could not disassemble {symbol} in {elf}")
    return insns


def disassemble_range(elf, start, stop, arch):
    """Re-disassemble an already-known address range (used on the patched copy)."""
    r = run([tool("llvm-objdump"), "-d", f"--mcpu={arch}",
             f"--start-address=0x{start:x}", f"--stop-address=0x{stop:x}", elf])
    return _parse_disasm(r.stdout)


def _slice_symbol(text, symbol):
    out, inside = [], False
    for line in text.splitlines():
        if re.match(r"^[0-9a-fA-F]+\s+<.*>:", line):
            inside = f"<{symbol}>:" in line
            if inside:
                out.append(line)
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


_SEC_CACHE = {}


def sections(elf):
    if elf not in _SEC_CACHE:
        r = run([tool("llvm-readelf"), "-S", "-W", elf])
        sec_re = re.compile(r"^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-fA-F]+)\s+"
                            r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)")
        out = []
        for line in r.stdout.splitlines():
            m = sec_re.match(line)
            if m:
                _, styp, sa, so, sz = m.groups()
                out.append((styp, int(sa, 16), int(so, 16), int(sz, 16)))
        _SEC_CACHE[elf] = out
    return _SEC_CACHE[elf]


def section_of(elf, addr):
    """(sh_addr, sh_offset, sh_size) of the section containing `addr` -- vaddr to offset."""
    for styp, sa, so, sz in sections(elf):
        if styp == "PROGBITS" and sz and sa <= addr < sa + sz:
            return sa, so, sz
    sys.exit(f"{elf}: no PROGBITS section contains address 0x{addr:x}")


# ======================================================================================
# choosing and encoding the patch
# ======================================================================================

_REG_RE = re.compile(r"^(?P<k>[vsa])(?:(?P<one>\d+)|\[(?P<lo>\d+):(?P<hi>\d+)\])$")


def regs(tok):
    """'v[8:11]' -> ('v', [8,9,10,11]);  'v8' -> ('v', [8]);  else None."""
    m = _REG_RE.match(tok.split()[0].strip() if tok else "")
    if not m:
        return None
    k = m.group("k")
    if m.group("one") is not None:
        return k, [int(m.group("one"))]
    return k, list(range(int(m.group("lo")), int(m.group("hi")) + 1))


def store_data_operand(insn):
    """The register operand a store reads its data from, or None if not an output store."""
    mn = insn.mnemonic
    if mn.startswith("ds_") or mn.startswith("scratch_"):
        return None                                   # LDS / spill, not the D matrix
    if mn.startswith("buffer_store"):
        idx = 0                                       # buffer_store vdata, vaddr, srsrc, soffset
    elif mn.startswith(("global_store", "flat_store")):
        idx = 1                                       # global_store vaddr, vdata, saddr
    else:
        return None
    if len(insn.operands) <= idx:
        return None
    return regs(insn.operands[idx])


def assemble(arch, asm):
    """Assemble one or more instructions with the ROCm assembler; bytes of .text."""
    clang = tool("clang", required=False) or tool("clang++", required=False)
    objcopy = tool("llvm-objcopy", required=False)
    if not clang or not objcopy:
        return None
    with tempfile.TemporaryDirectory() as d:
        src, obj, binf = f"{d}/p.s", f"{d}/p.o", f"{d}/p.bin"
        with open(src, "w") as f:
            f.write(f".amdgcn_target \"amdgcn-amd-amdhsa--{arch}\"\n.text\n{asm}\n")
        r = run([clang, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                 f"-mcpu={arch}", "-c", src, "-o", obj], check=False)
        if r.returncode != 0:
            return None
        r = run([objcopy, "-O", "binary", "--only-section=.text", obj, binf], check=False)
        if r.returncode != 0:
            return None
        with open(binf, "rb") as f:
            return f.read()


_ASM_CACHE = {}


def nan_write(arch, kind, reg):
    """Bytes for `<reg> = 0xFFFFFFFF` (a negative quiet NaN in f32)."""
    asm = (f"v_accvgpr_write_b32 a{reg}, -1" if kind == "a"
           else f"v_mov_b32_e32 v{reg}, -1")
    if (arch, asm) not in _ASM_CACHE:
        _ASM_CACHE[(arch, asm)] = assemble(arch, asm)
    b = _ASM_CACHE[(arch, asm)]
    if b:
        return b, asm
    if kind == "a":
        sys.exit("need the ROCm assembler to patch an AGPR; install ROCm llvm or use "
                 "--victim-addr to pick a VGPR-writing instruction")
    # GFX9 VOP1: [31:25]=0b0111111, [24:17]=vdst, [16:9]=op(1=v_mov_b32), [8:0]=src0(-1=193)
    word = 0x7E000000 | (reg << 17) | (1 << 9) | 193
    return struct.pack("<I", word), asm + "   (hand-encoded)"


def nop_bytes(arch, n_words):
    if n_words == 0:
        return b""
    b = assemble(arch, "\n".join(["s_nop 0"] * n_words))
    if b and len(b) == 4 * n_words:
        return b
    return struct.pack("<I", 0xBF800000) * n_words     # GFX9 SOPP s_nop 0


def pick_victim(insns, store_i, data, arch):
    """Last VALU write to a store-data register before the store.

    That instruction is the one whose result the store actually ships, so replacing it
    is the narrowest edit that guarantees a NaN lands in D -- nothing rewrites the
    register afterwards, and no branch target moves because the slot keeps its length.
    """
    kind, want = data
    want = set(want)
    fallback = None
    for i in range(store_i - 1, -1, -1):
        ins = insns[i]
        if not ins.mnemonic.startswith("v_") or not ins.operands:
            continue
        dst = regs(ins.operands[0])
        if not dst or dst[0] != kind:
            continue
        hit = sorted(set(dst[1]) & want)
        if not hit:
            continue
        payload, asm = nan_write(arch, kind, hit[0])
        if ins.size < len(payload) or (ins.size - len(payload)) % 4:
            continue                                   # slot too small; keep looking
        pad = nop_bytes(arch, (ins.size - len(payload)) // 4)
        cand = (i, payload + pad, asm, hit[0])
        if len(dst[1]) == 1:
            return cand
        # A multi-register writer (v[4:5]) also feeds whatever reads its other half, and
        # a single v_mov leaves that half undefined -- take it only if nothing better.
        fallback = fallback or cand
    return fallback


# ======================================================================================
# per-kernel analysis
# ======================================================================================

class KernelAnalysis:
    """Everything we learn -- and would change -- about one Tensile kernel.

    Cached per kernel name: two GEMMs that ran the same kernel poison the same stores,
    so the store counts are a property of the kernel, not of the problem size.
    """

    __slots__ = ("kernel", "libfile", "elf", "bundled", "insns", "start", "stop",
                 "n_stores", "poisoned", "patches", "error")

    def __init__(self, kernel, error=None):
        self.kernel, self.error = kernel, error
        self.libfile = self.elf = None
        self.bundled = False
        self.insns = []
        self.start = self.stop = 0
        self.n_stores = 0
        self.poisoned = []                 # (store_idx, store_i, victim_i, asm, reg, kind)
        self.patches = {}                  # file offset -> replacement bytes

    @property
    def n_poisoned(self):
        return len(self.poisoned)

    @property
    def n_sites(self):
        return len(self.patches)


def analyze_kernel(finder, kernel, arch, store_index, victim_addr=None):
    """Locate, disassemble and plan the NaN patch for one kernel name."""
    hit = finder.locate(kernel)
    if not hit:
        return KernelAnalysis(kernel, error="not found in this hipBLASLt library")
    libfile, elf = hit
    a = KernelAnalysis(kernel)
    a.libfile, a.elf, a.bundled = libfile, elf, libfile != elf
    a.start, a.stop = symbol_range(finder, elf, kernel)
    a.insns = disassemble(finder, elf, kernel, arch)

    stores = [(i, d) for i, ins in enumerate(a.insns)
              if (d := store_data_operand(ins)) is not None]
    a.n_stores = len(stores)
    if not stores:
        a.error = "no buffer/global/flat store found -- cannot locate the D write"
        return a

    if store_index == "all":
        targets = list(range(len(stores)))
    else:
        try:
            targets = [int(store_index, 0)]
        except ValueError:
            sys.exit(f"--store-index {store_index}: expected an integer or 'all'")
        if not 0 <= targets[0] < len(stores):
            a.error = f"--store-index {targets[0]}: kernel has {len(stores)} output store(s)"
            return a

    sh_addr, sh_off, _sz = section_of(elf, a.insns[0].addr)
    for t in targets:
        store_i, data = stores[t]
        if victim_addr is not None:
            match = [i for i, ins in enumerate(a.insns) if ins.addr == victim_addr]
            if not match:
                sys.exit(f"--victim-addr 0x{victim_addr:x}: no instruction there")
            vi = match[0]
            payload, asm = nan_write(arch, data[0], data[1][0])
            if a.insns[vi].size < len(payload) or (a.insns[vi].size - len(payload)) % 4:
                sys.exit(f"instruction at 0x{victim_addr:x} is {a.insns[vi].size}B, "
                         f"replacement is {len(payload)}B -- pick another")
            hit = (vi, payload + nop_bytes(arch, (a.insns[vi].size - len(payload)) // 4),
                   asm, data[1][0])
        else:
            hit = pick_victim(a.insns, store_i, data, arch)
        if hit is None:
            continue                       # this store has no VALU producer we can reach
        vi, payload, asm, reg = hit
        # Several stores often share one producer; that one edit poisons all of them, so
        # the patch is recorded once but every store it feeds counts as poisoned.
        a.patches.setdefault(a.insns[vi].addr - sh_addr + sh_off, payload)
        a.poisoned.append((t, store_i, vi, asm, reg, data[0]))

    if not a.patches:
        a.error = (f"found {a.n_stores} store(s) but no VALU producer for any of them; "
                   f"re-run with --victim-addr")
    return a


# ======================================================================================
# applying the patch
# ======================================================================================

def write_patched(src, patches, out_path):
    """Copy `src` to out_path with {file_offset: bytes} applied."""
    with open(src, "rb") as f:
        blob = bytearray(f.read())
    for off, data in patches.items():
        blob[off:off + len(data)] = data
    with open(out_path, "wb") as f:
        f.write(blob)


def build_overlay(libdir, analyses, arch, work):
    """A shadow library dir: everything symlinked, the patched .co files real.

    hipBLASLt is pointed at this with HIPBLASLT_TENSILE_LIBPATH, so the installed ROCm
    tree is never written to and the un-patched library is still there for comparison.
    One code object can hold several of the log's kernels, so the patches are merged per
    library file before anything is written.
    """
    by_lib = {}                                      # libfile -> merged {offset: bytes}
    for a in analyses:
        by_lib.setdefault(a.libfile, {}).update(a.patches)

    patched = {}                                     # rel path in libdir -> real file
    for libfile, patches in by_lib.items():
        elf = next(a.elf for a in analyses if a.libfile == libfile)
        patched_elf = os.path.join(work, "patched_" + os.path.basename(elf))
        write_patched(elf, patches, patched_elf)
        out = os.path.join(work, "lib_" + os.path.basename(libfile))
        if libfile != elf:                           # the library is an offload bundle
            targets = bundle_targets(libfile)
            dev = next(t for t in targets if "amdgcn" in t and arch in t)
            rebundle(libfile, dev, patched_elf, targets, work, out)
        else:
            shutil.copy2(patched_elf, out)
        patched[os.path.relpath(libfile, libdir)] = out

    # inside `work` when given: the caller swaps patched files in with os.replace, which cannot
    # cross filesystems (2026-11-21: work on NFS, overlay in /tmp -> EXDEV)
    overlay = tempfile.mkdtemp(prefix="hipblaslt_nan_lib_", dir=work if work and os.path.isdir(work) else None)
    # Symlink the top level, but never a directory that contains a patched object --
    # copying into a symlinked directory writes straight through into the real library.
    shadow = {rel.split(os.sep)[0] for rel in patched}
    for name in os.listdir(libdir):
        if name in shadow:
            continue
        os.symlink(os.path.abspath(os.path.join(libdir, name)),
                   os.path.join(overlay, name))
    for top in shadow:
        real = os.path.join(libdir, top)
        if not os.path.isdir(real):
            continue                                 # flat layout: the patched file itself
        for root, _dirs, files in os.walk(real):
            rel = os.path.relpath(root, libdir)
            os.makedirs(os.path.join(overlay, rel), exist_ok=True)
            for f in files:
                if os.path.join(rel, f) in patched:
                    continue
                os.symlink(os.path.abspath(os.path.join(root, f)),
                           os.path.join(overlay, rel, f))
    for rel, src in patched.items():
        dst = os.path.join(overlay, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return overlay, patched


# ======================================================================================
# reporting
# ======================================================================================

def print_disasm(title, insns, mode, focus=None, context=16):
    if mode == "none":
        return
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if mode == "full" or focus is None:
        for ins in insns:
            print(ins.line)
        return
    lo, hi = max(0, focus - context), min(len(insns), focus + context + 1)
    if lo:
        print(f"   ... {lo} earlier instruction(s) elided (use --disasm full) ...")
    for i in range(lo, hi):
        print(("-->" if i == focus else "   ") + insns[i].line)
    if hi < len(insns):
        print(f"   ... {len(insns) - hi} later instruction(s) elided (use --disasm full) ...")


SAVED_DIR = "add_nan_to_hipblaslt"


def dump_selected(winner, arch, work, outdir=SAVED_DIR):
    """Write the selected GEMM's kernel name and its ISA before/after the NaN patch.

    Done here, right after the winner is picked, rather than in the dispatch path: the
    overlay is only built when something is actually replayed, but these three files are
    wanted for every run -- including `--run none` on a login node with no GPU.
    """
    a = winner.analysis
    os.makedirs(outdir, exist_ok=True)

    name_path = os.path.join(outdir, "kernel_name.txt")
    with open(name_path, "w") as f:
        f.write(a.kernel + "\n")

    orig_path = os.path.join(outdir, "kernel_without_nan.asm")
    with open(orig_path, "w") as f:
        f.write("\n".join(i.line for i in a.insns) + "\n")

    # Patch a throwaway copy of the (unbundled) ELF and disassemble the same address
    # range, so the two listings line up instruction for instruction.
    patched_elf = os.path.join(work, "selected_" + os.path.basename(a.elf))
    write_patched(a.elf, a.patches, patched_elf)
    after = disassemble_range(patched_elf, a.start, a.stop, arch)
    nan_path = os.path.join(outdir, "kernel_with_nan.asm")
    with open(nan_path, "w") as f:
        f.write("\n".join(i.line for i in after) + "\n")

    print(f"\nselected GEMM written to {outdir}:")
    for p in (name_path, orig_path, nan_path):
        print(f"  {os.path.basename(p)}")


def report_gemm(g, total):
    a = g.analysis
    print(f"\n[{g.pos:>4}/{total}] {g.shape}")
    print(f"         kernel_name:      {g.kernel or '(absent from this log)'}")
    if a is None or (a.error and not a.n_poisoned):
        why = a.error if a else "no kernel_name in the log line"
        print(f"         stores poisoned:  0   [{why}]")
        return
    print(f"         library:          {os.path.basename(a.libfile)}"
          + ("   (offload bundle)" if a.bundled else ""))
    print(f"         output stores:    {a.n_stores}")
    print(f"         stores poisoned:  {a.n_poisoned}"
          + (f"   ({a.n_sites} distinct patch site(s))" if a.n_sites != a.n_poisoned else "")
          + ("   -> candidate" if a.n_poisoned >= 1 else ""))
    blockers = dispatch_blockers(g.fields)
    if blockers:
        print(f"         not replayable:   {', '.join(blockers)}")


def print_patch_plan(a):
    print(f"\npatch sites: {a.n_sites}  (poisoning {a.n_poisoned} of {a.n_stores} "
          f"output store(s))")
    for _t, store_i, vi, asm, reg, kind in a.poisoned[:10]:
        print(f"  0x{a.insns[vi].addr:x}  {a.insns[vi].text:<44s} -> {asm}"
              f"   ({kind}{reg} feeds the store at 0x{a.insns[store_i].addr:x})")
    if a.n_poisoned > 10:
        print(f"  ... {a.n_poisoned - 10} more")
    print("  effect:    every poisoned store ships 0xFFFFFFFF (NaN)")


# ======================================================================================
# dispatch
# ======================================================================================

def dispatch(gemms, overlay, solution_pin, args):
    """Run the (patched) GEMMs.  One torch import, because HIPBLASLT_TENSILE_LIBPATH is
    read when the hipBLASLt .so initializes -- there is no second chance to change it."""
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    if any(g.get("compute_type") == "c_xf32_r" for g in gemms) and not args.no_xf32:
        # The logged compute type is xf32; force it at the library level so the replay
        # matches regardless of which torch fp32-precision API this build has.
        os.environ.setdefault("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", "1")
    if solution_pin is not None:
        # Pin the solution, or hipBLASLt may pick a different (un-patched) kernel.  Only
        # meaningful for a single GEMM -- the env var cannot vary per call.
        os.environ["TENSILE_SOLUTION_INDEX"] = str(solution_pin)
    # Both spellings: the env name has moved between rocBLAS-era and hipBLASLt-era Tensile.
    os.environ["HIPBLASLT_TENSILE_LIBPATH"] = overlay
    os.environ["TENSILE_LIBPATH"] = overlay
    if args.emit_log:
        os.environ["HIPBLASLT_LOG_MASK"] = "64"
        os.environ["HIPBLASLT_LOG_FILE"] = args.emit_log

    import torch

    if not torch.cuda.is_available():
        sys.exit("no GPU visible to torch (need /dev/kfd and /dev/dri in the container)")

    xf32 = not args.no_xf32
    if xf32:
        # Best-effort on both the old and new torch APIs; the env var above is what
        # actually guarantees xf32, these just keep torch from asking for full fp32.
        for setter in (lambda: setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32"),
                       lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", True)):
            try:
                setter()
            except AttributeError:
                pass

    dev = torch.device("cuda")
    any_nan = False
    for g in gemms:
        p = g.fields
        M, N, K = p["M"], p["N"], p["K"]
        transA, transB, alpha = p["transA"], p["transB"], p["alpha"]
        torch.manual_seed(args.seed)

        # A: col-major (K,M) if transA==T else (M,K)  ->  row-major (M,K) / (K,M)
        A = torch.randn((M, K) if transA == "T" else (K, M), dtype=torch.float32, device=dev)
        # B: col-major (K,N) if transB==N else (N,K)  ->  row-major (N,K) / (K,N)
        B = torch.randn((N, K) if transB == "N" else (K, N), dtype=torch.float32, device=dev)
        opA = A if transA == "T" else A.t()         # (M, K)
        opB = B.t() if transB == "N" else B         # (K, N)

        def gemm():
            # OUT is the row-major view of the column-major D, so it has shape (N, M).
            out = opB.t() @ opA.t()                 # == (opA @ opB).T, operands stay lazy
            return out * alpha if alpha != 1.0 else out

        print(f"\n--- dispatching log line {g.pos}: {g.shape}")
        print(f"    kernel: {g.kernel}")
        out = gemm()

        got = {"lda": A.stride(0), "ldb": B.stride(0), "ldd": out.stride(0)}
        want = {"lda": p["lda"], "ldb": p["ldb"], "ldd": p["ldd"]}
        for k in want:
            flag = "ok" if got[k] == want[k] else "MISMATCH"
            print(f"    {k}: logged {want[k]:6d}  replay {got[k]:6d}  [{flag}]")

        iters = args.iters if args.iters is not None else p.get("call_count", 1)
        for _ in range(iters - 1):
            out = gemm()
        torch.cuda.synchronize()

        n_nan = torch.isnan(out).sum().item()
        n_inf = torch.isinf(out).sum().item()
        any_nan = any_nan or bool(n_nan or n_inf)
        print(f"    {iters} call(s) done -- patched kernel")
        print(f"    NaN: {n_nan}/{out.numel()}   Inf: {n_inf}/{out.numel()}")
        print(f"    max |out|: {out.abs().max().item():.6g}")
        if out.numel() <= args.ref_max_elems:
            ref = (opA.double().cpu() @ opB.double().cpu()).t() * alpha
            err = (out.double().cpu() - ref).abs()
            finite = torch.isfinite(err)
            print(f"    max abs err vs fp64 CPU: "
                  f"{err[finite].max().item() if finite.any() else float('nan'):.6g}")
        else:
            print(f"    fp64 CPU reference skipped ({out.numel()} elements > "
                  f"--ref-max-elems {args.ref_max_elems})")
        if n_nan == 0:
            print("    NO NaN OBSERVED -- the patched kernel probably was not the one used.\n"
                  "      - check that HIPBLASLT_TENSILE_LIBPATH is the env var this "
                  "hipBLASLt reads\n"
                  "      - the poisoned store may be in a path this problem size never "
                  "takes (try --store-index N)", file=sys.stderr)
        g.run = (n_nan, n_inf, out.numel())
        del A, B, opA, opB, out
        torch.cuda.empty_cache()

    if args.emit_log:
        print(f"\nre-emitted profile log: {args.emit_log}  (diff against {args.log})")
    return any_nan


# ======================================================================================
# main
# ======================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="hipBLASLt profile log (HIPBLASLT_LOG_MASK=64)")
    ap.add_argument("--order-log", metavar="FILE",
                    help="hipblaslt-bench per-call trace (HIPBLASLT_LOG_MASK=32) of the "
                         "same run; gives the true execution order used to break ties")
    ap.add_argument("--iters", type=int, default=None,
                    help="repeat count (default: call_count from the log)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-log", metavar="FILE",
                    help="re-emit a hipBLASLt profile log here, to diff against the original")
    ap.add_argument("--no-xf32", action="store_true",
                    help="use full fp32 compute even if the log says c_xf32_r")
    # --- which GEMMs ---
    ap.add_argument("--gemm", type=int, action="append", metavar="N",
                    help="only handle log line N (1-based); repeatable")
    ap.add_argument("--run", choices=("best", "all", "none"), default="best",
                    help="what to dispatch after the analysis: the winning candidate "
                         "(default), every replayable candidate, or nothing")
    ap.add_argument("--ref-max-elems", type=int, default=1 << 22,
                    help="skip the fp64 CPU reference above this output size "
                         "(default: 4194304)")
    # --- kernel selection / patching ---
    ap.add_argument("--libdir", help="hipBLASLt Tensile library dir (default: autodetect)")
    ap.add_argument("--arch", help="gfx target (default: the GPU in this machine)")
    ap.add_argument("--store-index", default="all",
                    help="which output store to poison: an index, or 'all' (default). "
                         "Tensile emits separate stores for edge/non-edge and "
                         "GSU/beta paths, and only some of them run for a given size, "
                         "so poisoning one store is a coin flip.")
    ap.add_argument("--victim-addr", type=lambda s: int(s, 0), default=None,
                    help="override: overwrite the instruction at this address instead")
    ap.add_argument("--disasm", choices=("full", "context", "none"), default="none",
                    help="ISA to print for the dispatched GEMM(s) (default: none -- a "
                         "Tensile kernel is ~37k instructions and the log has hundreds)")
    ap.add_argument("--context", type=int, default=16,
                    help="instructions of context with --disasm context (default: 16)")
    ap.add_argument("--save-disasm", metavar="DIR",
                    help="write <kernel>.orig.s / <kernel>.nan.s for the dispatched GEMMs")
    ap.add_argument("--keep-libdir", action="store_true",
                    help="do not delete the temporary patched library dir on exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="same as --run none: analyse and patch, but never import torch")
    args = ap.parse_args()
    if args.dry_run:
        args.run = "none"

    gemms = parse_log(args.log)
    if args.gemm:
        want = set(args.gemm)
        gemms = [g for g in gemms if g.pos in want]
        if not gemms:
            sys.exit(f"--gemm {sorted(want)}: no such line(s) in {args.log}")
    total = len(gemms)
    work = tempfile.mkdtemp(prefix="hipblaslt_nan_work_")
    order = ExecOrder(args.order_log)

    # ==================================================================================
    # 1. find every Tensile kernel the log names -- BEFORE torch exists, so no stock
    #    kernel is ever dispatched.
    # ==================================================================================
    libdir = find_library_dir(args.libdir)
    arch = args.arch or gpu_arch(gemms)
    named = [g for g in gemms if g.kernel]
    print(f"log:                       {args.log}  ({total} GEMM(s))")
    print(f"hipBLASLt Tensile library: {libdir}")
    print(f"arch:                      {arch}")
    print(f"execution order:           {order.source}")
    print(f"distinct kernels in log:   {len({g.kernel for g in named})}"
          + (f"   ({total - len(named)} line(s) have no kernel_name)"
             if len(named) < total else ""))

    finder = KernelFinder(libdir, arch, work)
    finder.index(g.kernel for g in named)

    # ==================================================================================
    # 2. per GEMM: disassemble its kernel, pick the instructions to poison, count them
    # ==================================================================================
    cache = {}
    candidates, failed = [], []
    for g in gemms:
        if g.kernel:
            if g.kernel not in cache:
                cache[g.kernel] = analyze_kernel(finder, g.kernel, arch,
                                                 args.store_index, args.victim_addr)
            g.analysis = cache[g.kernel]
        report_gemm(g, total)
        if g.analysis is not None and g.analysis.n_poisoned >= 1:
            candidates.append(g)
        else:
            failed.append(g)

    # ==================================================================================
    # 3. the candidate set and its winner
    # ==================================================================================
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"GEMMs in log:        {total}")
    print(f"candidate_set:       {len(candidates)}   (>= 1 output store poisoned)")
    if failed:
        print(f"not poisonable:      {len(failed)}   "
              f"(lines: {', '.join(str(g.pos) for g in failed[:12])}"
              f"{' ...' if len(failed) > 12 else ''})")

    winner = None
    if candidates:
        best_n = min(g.analysis.n_poisoned for g in candidates)
        tied = [g for g in candidates if g.analysis.n_poisoned == best_n]
        tied.sort(key=order.rank)
        winner = tied[0]
        print(f"\nminimum output stores poisoned: {best_n}")
        print(f"GEMMs achieving it:             {len(tied)}")
        if len(tied) > 1:
            print(f"tie broken by execution order ({order.source}):")
            for g in tied[:10]:
                mark = "  <-- selected" if g is winner else ""
                print(f"    line {g.pos:>4}  {order.describe(g):<22s} "
                      f"{g.get('M')}x{g.get('N')}x{g.get('K')} "
                      f"{g.get('transA')}{g.get('transB')}{mark}")
            if len(tied) > 10:
                print(f"    ... {len(tied) - 10} more")
        print(f"\nSELECTED GEMM  (log line {winner.pos})")
        print(f"  {winner.shape}")
        print(f"  stores poisoned: {winner.analysis.n_poisoned} "
              f"of {winner.analysis.n_stores}")
        print(f"  solution_index:  {winner.get('solution_index')}")
        print(f"  kernel_name:     {winner.kernel}")
        print(f"  library:         {os.path.basename(winner.analysis.libfile)}")
        dump_selected(winner, arch, work)
    else:
        print("\ncandidate_set is empty -- no GEMM in this log could be poisoned.")

    # ==================================================================================
    # 4. patch, and dispatch whatever was asked for
    # ==================================================================================
    if args.run == "none" or not candidates:
        if args.run == "none":
            print("\nno dispatch requested (--run none)")
        shutil.rmtree(work, ignore_errors=True)
        return 0 if candidates else 2

    if args.run == "best":
        to_run, pin = [winner], winner.get("solution_index")
    else:
        to_run, pin = list(candidates), None
    blocked = [(g, dispatch_blockers(g.fields)) for g in to_run]
    to_run = [g for g, b in blocked if not b]
    for g, b in blocked:
        if b:
            print(f"\nskipping log line {g.pos} for dispatch: {', '.join(b)}",
                  file=sys.stderr)
    if not to_run:
        print("\nnothing replayable to dispatch", file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        return 2

    used = {g.analysis.kernel: g.analysis for g in to_run}
    overlay, patched = build_overlay(libdir, list(used.values()), arch, work)
    print(f"\npatched {len(used)} kernel(s) across {len(patched)} library file(s):")
    for rel in sorted(patched):
        print(f"  {rel}")
    print(f"patched library overlay: {overlay}")

    if args.disasm != "none" or args.save_disasm:
        for a in used.values():
            patched_elf = os.path.join(work, "patched_" + os.path.basename(a.elf))
            after = disassemble_range(patched_elf, a.start, a.stop, arch)
            print_patch_plan(a)
            vi = a.poisoned[0][2]
            focus = next((i for i, ins in enumerate(after)
                          if ins.addr == a.insns[vi].addr), vi)
            print_disasm(f"ORIGINAL Tensile ISA -- {a.kernel}"
                         f"  [{os.path.basename(a.libfile)}]",
                         a.insns, args.disasm, focus=vi, context=args.context)
            print_disasm(f"MODIFIED Tensile ISA -- {a.kernel}"
                         f"  [NaN injected at {a.n_sites} site(s)]",
                         after, args.disasm, focus=focus, context=args.context)
            if args.save_disasm:
                os.makedirs(args.save_disasm, exist_ok=True)
                for tag, seq in (("orig", a.insns), ("nan", after)):
                    path = os.path.join(args.save_disasm, f"{a.kernel}.{tag}.s")
                    with open(path, "w") as f:
                        f.write("\n".join(i.line for i in seq) + "\n")
                    print(f"wrote {path}")

    try:
        saw_nan = dispatch(to_run, overlay, pin, args)
        return 0 if saw_nan else 1
    finally:
        if args.keep_libdir:
            print(f"\nkept patched library overlay: {overlay}")
            print(f"kept work dir:                {work}")
        else:
            shutil.rmtree(overlay, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    rc = main()
    # This torch/HIP build tears down through a handler that resets the process status,
    # so `sys.exit(rc)` after a GPU dispatch reliably yields 0 no matter what happened.
    # Everything this script allocates is already released in main()'s finally block, so
    # leaving through os._exit is safe and is the only way the code survives.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
