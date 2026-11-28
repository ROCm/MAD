#!/usr/bin/env python3
"""Streaming zstd decompressor, implemented on ctypes + libzstd.

WHY THIS EXISTS
---------------
Every `docker_*.sh` in this directory starts with `zstdcat "$IMAGE_TAR" | ...`.
On some hosts there is no `zstd` and no `zstdcat` anywhere on PATH, the stock
/usr/bin/python3 (3.9) has no `zstandard` module, and compute nodes have no
egress to install either -- DNS resolves but TCP hangs (curl to pypi.org exits
124).  So the usual escapes are all closed.

What IS present is /lib64/libzstd.so.1 -> libzstd.so.1.5.5.  This drives its
streaming decompression API directly, which needs nothing but ctypes from the
standard library.

Usage:
    zstdcat.py IN.zst                 # -> stdout (pipe into podman load)
    zstdcat.py IN.zst -o OUT.tar      # -> file  (apptainer needs a seekable
                                          #           file for docker-archive://)

Concatenated frames -- what `pzstd` and some `zstd -T0` invocations emit -- need
no special handling: when a frame ends mid-buffer, libzstd starts the next one
on its own as the loop keeps feeding, so a multi-frame archive decodes exactly
like a single-frame one.
"""

import ctypes
import os
import sys
import time

# The soname, not the versioned file: 1.5.5 is what is installed today, but the
# streaming API used here has been stable since 1.0 and there is no reason to
# pin this script to one patch release.
LIBZSTD = os.environ.get("LIBZSTD", "libzstd.so.1")


class ZSTDInBuffer(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class ZSTDOutBuffer(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


def load_libzstd():
    """Bind the handful of libzstd entry points this needs.

    Declaring argtypes/restype is not optional here.  ctypes defaults every
    return to c_int (32-bit), but ZSTD_createDStream returns a pointer and the
    rest return size_t; on x86-64 that truncation silently corrupts the handle
    and the error codes, which show up as a segfault rather than an exception.
    """
    z = ctypes.CDLL(LIBZSTD)

    z.ZSTD_createDStream.argtypes = []
    z.ZSTD_createDStream.restype = ctypes.c_void_p

    z.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
    z.ZSTD_freeDStream.restype = ctypes.c_size_t

    z.ZSTD_initDStream.argtypes = [ctypes.c_void_p]
    z.ZSTD_initDStream.restype = ctypes.c_size_t

    z.ZSTD_decompressStream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ZSTDOutBuffer),
        ctypes.POINTER(ZSTDInBuffer),
    ]
    z.ZSTD_decompressStream.restype = ctypes.c_size_t

    z.ZSTD_isError.argtypes = [ctypes.c_size_t]
    z.ZSTD_isError.restype = ctypes.c_uint

    z.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
    z.ZSTD_getErrorName.restype = ctypes.c_char_p

    z.ZSTD_DStreamInSize.argtypes = []
    z.ZSTD_DStreamInSize.restype = ctypes.c_size_t

    z.ZSTD_DStreamOutSize.argtypes = []
    z.ZSTD_DStreamOutSize.restype = ctypes.c_size_t

    return z


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f PiB" % n


def decompress(src_path, dst_fh, progress_fh=None, limit=None):
    """Stream src_path into dst_fh.  Returns (bytes_in, bytes_out).

    limit: stop after writing at least this many output bytes.  Used by
    --limit to validate the pipeline on a prefix without materialising ~100 GB.
    """
    z = load_libzstd()

    # libzstd's recommended sizes are ~128 KiB in / ~128 KiB out.  Both are
    # scaled up here because every call crosses the ctypes FFI boundary, and at
    # 39 GB of input the per-call overhead is the only thing this loop can get
    # wrong.  Bigger buffers make that overhead vanish without costing anything
    # on a node with 3 TB of RAM.
    in_size = max(z.ZSTD_DStreamInSize(), 4 << 20)
    out_size = max(z.ZSTD_DStreamOutSize(), 16 << 20)

    ds = z.ZSTD_createDStream()
    if not ds:
        raise RuntimeError("ZSTD_createDStream failed")

    try:
        rc = z.ZSTD_initDStream(ds)
        if z.ZSTD_isError(rc):
            raise RuntimeError(
                "ZSTD_initDStream: %s" % z.ZSTD_getErrorName(rc).decode()
            )

        inbuf = ctypes.create_string_buffer(in_size)
        outbuf = ctypes.create_string_buffer(out_size)
        in_mv = memoryview(inbuf)[:in_size]

        zin = ZSTDInBuffer(ctypes.cast(inbuf, ctypes.c_void_p), 0, 0)
        zout = ZSTDOutBuffer(ctypes.cast(outbuf, ctypes.c_void_p), out_size, 0)

        total_in = total_out = 0
        started = last_report = time.time()

        with open(src_path, "rb") as src:
            while True:
                nread = src.readinto(in_mv)
                if not nread:
                    break
                total_in += nread
                zin.size = nread
                zin.pos = 0

                # One read can hold many output buffers' worth of data, so this
                # drains the input chunk fully before refilling it.
                while zin.pos < zin.size:
                    zout.size = out_size
                    zout.pos = 0
                    rc = z.ZSTD_decompressStream(
                        ds, ctypes.byref(zout), ctypes.byref(zin)
                    )
                    if z.ZSTD_isError(rc):
                        raise RuntimeError(
                            "ZSTD_decompressStream at input offset %d: %s"
                            % (total_in - nread + zin.pos,
                               z.ZSTD_getErrorName(rc).decode())
                        )
                    if zout.pos:
                        dst_fh.write(outbuf[: zout.pos])
                        total_out += zout.pos

                    if limit is not None and total_out >= limit:
                        return total_in, total_out

                    # rc == 0 means a frame just ended.  If input remains it is
                    # a concatenated frame and the next iteration picks it up.
                    if rc == 0 and zin.pos >= zin.size:
                        break

                if progress_fh is not None:
                    now = time.time()
                    if now - last_report >= 30.0:
                        elapsed = now - started
                        progress_fh.write(
                            "[zstdcat] in %s  out %s  %.0f MiB/s out  %.0f s\n"
                            % (
                                human(total_in),
                                human(total_out),
                                total_out / elapsed / (1 << 20) if elapsed else 0,
                                elapsed,
                            )
                        )
                        progress_fh.flush()
                        last_report = now

        return total_in, total_out
    finally:
        z.ZSTD_freeDStream(ds)


def main(argv):
    src = None
    dst = "-"
    limit = None

    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "--output"):
            i += 1
            dst = argv[i]
        elif a == "--limit":
            # Accepts a plain byte count or a K/M/G suffix.
            i += 1
            v = argv[i].strip().upper()
            mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(v[-1:], 1)
            limit = int(v[:-1] if mult > 1 else v) * mult
        elif a in ("-h", "--help"):
            sys.stdout.write(__doc__)
            return 0
        elif a.startswith("-") and a != "-":
            sys.stderr.write("unknown option: %s\n" % a)
            return 2
        else:
            src = a
        i += 1

    if src is None:
        sys.stderr.write("usage: zstdcat.py IN.zst [-o OUT] [--limit N]\n")
        return 2
    if not os.path.exists(src):
        sys.stderr.write("no such file: %s\n" % src)
        return 1

    started = time.time()
    if dst == "-":
        # Progress would interleave with the archive itself if stdout is the
        # sink, so it is suppressed; the consumer (podman load) prints its own.
        n_in, n_out = decompress(src, sys.stdout.buffer, None, limit)
        sys.stdout.buffer.flush()
    else:
        with open(dst, "wb") as fh:
            n_in, n_out = decompress(src, fh, sys.stderr, limit)

    elapsed = time.time() - started
    sys.stderr.write(
        "[zstdcat] done: %s in -> %s out (%.2fx) in %.0f s\n"
        % (human(n_in), human(n_out), (n_out / n_in) if n_in else 0, elapsed)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
