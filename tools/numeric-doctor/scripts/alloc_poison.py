#!/usr/bin/env python3
"""Does the NaN come from uninitialized memory?  Run a program with its fresh allocations
filled with a chosen value, and compare the NaN rate across values.

    python3 -m alloc_poison --empty 0      -- <script.py> [args...]   # every torch.empty* -> 0
    python3 -m alloc_poison --empty nan    -- <script.py> [args...]   # float NaN, int -1
    python3 -m alloc_poison --empty bits:0x7fc07fc0 -- ...            # raw bit pattern
    python3 -m alloc_poison --empty 0 --site aiter/mla.py:1300 -- ... # only that allocation
    python3 -m alloc_poison --sites        -- <script.py> [args...]   # list the allocation sites
    python3 -m alloc_poison --free-blocks 0 -- <script.py> [args...]  # pre-fill freed blocks

A kernel that reads memory nobody wrote (a torch.empty buffer only some arch's kernel fills,
a partly written out-buffer) makes its NaN from whatever bytes the caching allocator hands
back: the rate follows the allocator's history, the first call of a process is often clean,
and a capsule replay in a fresh process reproduces only by luck.  Filling the allocations
turns that luck into a switch: a rate that changes with the fill value (0 -> every call NaN,
the right value -> none) proves the read, whatever the kernel.  2026-11-21 (aiter MLA decode,
blind run 5): a split-count buffer from torch.empty, read by a Triton reduce, written by no
gfx950 kernel; 0 or -1 -> 0/0 in the reduce on every call, 16 -> clean.

--empty <v>       fill the result of torch.empty / empty_like / empty_strided / Tensor.new_empty
                  (the Python-level calls: a library's own wrapper code allocates this way, e.g.
                  aiter/mla.py) with <v>: an int, a float, `nan` (floats NaN, ints -1), `inf`, or
                  `bits:0x..` (raw bits, repeated to the element size).
--site FILE:LINE  only allocations made at that source line (a suffix of the path is enough;
                  repeatable).  Bisects WHICH buffer: poison one site at a time.
--sites           fill nothing; at exit print every allocation site seen, with its count and
                  shapes/dtypes, most frequent first (`ALLOC_SITE` lines).
--free-blocks <v> before the program starts and after every --every-n allocations, allocate
                  and free --blocks blocks of --block-bytes bytes filled with <v>: the caching
                  allocator's free pool then holds <v>, which also reaches allocations made in
                  C++ (at::empty inside an extension), not only Python-level ones.

In a 5N candidate: `import alloc_poison; print(alloc_poison.from_env())` at the top, and one
native_variants.sh line per value: `3|ALLOC_POISON_EMPTY=0||`, `3|ALLOC_POISON_EMPTY=16||`.

Last line: POISON_DONE empty=<v|-> sites=<n filled> free_blocks=<v|->.  The program's own
output (its NaN signal) decides; run each value as one arm of a wave or of native_variants.sh.
"""
import argparse
import collections
import os
import runpy
import struct
import sys
import traceback

_STATS = collections.Counter()
_SHAPES = collections.defaultdict(set)
_FILLED = [0]


def _parse_value(v):
    if v is None:
        return None
    v = v.strip().lower()
    if v.startswith("bits:"):
        return ("bits", int(v[5:], 16))
    if v == "nan":
        return ("nan", None)
    if v in ("inf", "+inf", "-inf"):
        return ("num", float(v))
    try:
        return ("num", int(v, 0))
    except ValueError:
        return ("num", float(v))


def _fill(t, val):
    import torch
    kind, x = val
    if t.numel() == 0:
        return
    with torch.no_grad():
        if kind == "bits":
            n = t.element_size()
            idt = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(n)
            if idt is None:
                return
            pat = x & ((1 << (8 * n)) - 1)
            pat = struct.unpack("<%s" % "bhiq"["\x01\x02\x04\x08".index(chr(n))],
                                pat.to_bytes(n, "little"))[0]
            t.view(idt).fill_(pat)
        elif kind == "nan":
            t.fill_(float("nan") if (t.is_floating_point() or t.is_complex()) else
                    (-1 if t.dtype != torch.bool else True))
        else:
            if not (t.is_floating_point() or t.is_complex()) and isinstance(x, float):
                if x != x or x in (float("inf"), float("-inf")):
                    return
                x = int(x)
            t.fill_(x)


def _site(depth=2):
    """file:line of the first frame outside this module, torch and the stdlib (runpy)."""
    import torch
    tdir = os.path.dirname(os.path.abspath(torch.__file__))
    sdir = os.path.dirname(os.path.abspath(os.__file__))
    me = os.path.abspath(__file__)
    f = sys._getframe(depth)
    while f is not None:
        fn = f.f_code.co_filename
        afn = os.path.abspath(fn)
        # site-packages sits under the stdlib dir in a venv / conda: a library's wrapper
        # (aiter/mla.py) is exactly the site wanted
        stdlib = afn.startswith(sdir) and os.sep + "site-packages" + os.sep not in afn \
            and os.sep + "dist-packages" + os.sep not in afn
        if afn != me and not afn.startswith(tdir) and not stdlib and not fn.startswith("<frozen"):
            return "%s:%d" % (fn, f.f_lineno)
        f = f.f_back
    return "?"


def install(empty=None, sites=None, list_sites=False):
    """Wrap the torch.empty family.  empty: a _parse_value tuple or None."""
    import torch
    want = [s for s in (sites or [])]

    def wrap(orig, label):
        def f(*a, **k):
            t = orig(*a, **k)
            if not isinstance(t, torch.Tensor):
                return t
            if list_sites or want:
                s = _site()
                if list_sites:
                    _STATS[s] += 1
                    if len(_SHAPES[s]) < 4:
                        _SHAPES[s].add("%s%s" % (str(t.dtype).replace("torch.", ""), tuple(t.shape)))
                if want and not any(s.endswith(w) for w in want):
                    return t
            if empty is not None:
                _fill(t, empty)
                _FILLED[0] += 1
            return t
        f.__name__ = getattr(orig, "__name__", label)
        f.__wrapped__ = orig
        return f

    torch.empty = wrap(torch.empty, "empty")
    torch.empty_like = wrap(torch.empty_like, "empty_like")
    torch.empty_strided = wrap(torch.empty_strided, "empty_strided")
    torch.Tensor.new_empty = wrap(torch.Tensor.new_empty, "new_empty")


def poison_free_blocks(val, block_bytes=1 << 12, blocks=4096, device=None):
    """Fill the caching allocator's free pool with `val`: allocate, fill, free."""
    import torch
    if not torch.cuda.is_available():
        return
    dev = device if device is not None else torch.cuda.current_device()
    n = max(1, block_bytes // 4)
    held = []
    for _ in range(blocks):
        t = torch.empty.__wrapped__(n, dtype=torch.int32, device=dev) \
            if hasattr(torch.empty, "__wrapped__") else torch.empty(n, dtype=torch.int32, device=dev)
        _fill(t, val)
        held.append(t)
    del held


def from_env():
    """Install from ALLOC_POISON_EMPTY / ALLOC_POISON_SITE (comma-separated) /
    ALLOC_POISON_FREE_BLOCKS -- for a 5N candidate, whose variants set them in the env
    column of native_variants.sh.  No variable set: nothing happens.  Returns a one-line
    description for the candidate's log."""
    e = os.environ.get
    empty, sites, fb = _parse_value(e("ALLOC_POISON_EMPTY")), \
        [x for x in (e("ALLOC_POISON_SITE") or "").split(",") if x], _parse_value(e("ALLOC_POISON_FREE_BLOCKS"))
    if empty is not None or sites:
        if empty is None:
            raise SystemExit("ALLOC_POISON_SITE needs ALLOC_POISON_EMPTY")
        install(empty, sites)
    if fb is not None:
        poison_free_blocks(fb)
    return "POISON empty=%s sites=%s free_blocks=%s" % (e("ALLOC_POISON_EMPTY") or "-",
                                                        ",".join(sites) or "all",
                                                        e("ALLOC_POISON_FREE_BLOCKS") or "-")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--empty", default=None)
    ap.add_argument("--site", action="append", default=[])
    ap.add_argument("--sites", action="store_true")
    ap.add_argument("--free-blocks", default=None)
    ap.add_argument("--block-bytes", type=int, default=1 << 12)
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("script")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    empty = _parse_value(a.empty)
    if a.site and empty is None:
        ap.error("--site needs --empty <value>")
    install(empty, a.site, a.sites)
    fb = _parse_value(a.free_blocks)
    if fb is not None:
        poison_free_blocks(fb, a.block_bytes, a.blocks)
    sys.argv = [a.script] + [x for x in a.args if x != "--"]
    rc = 0
    try:
        runpy.run_path(a.script, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:                              # noqa: BLE001 -- report, then the line
        traceback.print_exc()
        rc = 1
    finally:
        if a.sites:
            for s, n in _STATS.most_common():
                print("ALLOC_SITE %s count=%d %s" % (s, n, " ".join(sorted(_SHAPES[s]))), flush=True)
        print("POISON_DONE empty=%s sites=%s filled=%d free_blocks=%s"
              % (a.empty or "-", ",".join(a.site) or "all", _FILLED[0], a.free_blocks or "-"),
              flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
