"""Compressing the logs of a finished run.

RCCL debug text is the most repetitive artifact a profiled run leaves: the same record shape with a
different opCount, millions of times. Measured on a 4-node serving job, gzip took 8.2 GB of per-rank
logs to 88 MB, and the parser reads the compressed files directly -- so a run's logs are worth
keeping only in that form once nothing is appending to them.

Nothing here names an engine: which files are logs at all comes from the spec's globs, so this
compresses exactly the set the parser would read.
"""

from __future__ import annotations

import gzip
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20

#: Parallel workers by default: one per core, and never more than this, because past a handful of
#: streams the shared filesystem rather than the CPU sets the pace. Compression is CPU-bound per
#: file and the files are independent, so a single stream leaves both resources idle -- 32 per-rank
#: logs took 3.5 s across 16 workers where one would have taken minutes.
JOBS_CAP = 16
DEFAULT_JOBS = min(JOBS_CAP, os.cpu_count() or 1)


@dataclass
class Result:
    """One log's outcome. ``after`` is 0 when it failed and the original was left alone."""

    path: Path
    gz: Path
    before: int
    after: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def gz_path(log: Path) -> Path:
    return log.with_name(log.name + ".gz")


def plan(logs: list) -> tuple:
    """Split discovered logs into the ones to compress and the ones already compressed twice.

    A plain log beside its own ``.gz`` is not a state to resolve automatically: the parser's globs
    match both, so it would read every record of that log twice, and only whoever made the pair
    knows which copy is the whole one.
    """
    todo, doubled = [], []
    for log in sorted({Path(p) for p in logs}):
        if log.suffix == ".gz":
            continue
        (doubled if gz_path(log).exists() else todo).append(log)
    return todo, doubled


def compress_one(log: Path, level: int = 6) -> Result:
    """Compress one log, verify the copy reads back identical, then drop the original.

    The original outlives every step that can fail: the compressed stream goes to a temporary name
    first, is read back and compared by digest, and only a file that survives both replaces it.
    Anything else leaves the run directory exactly as it was.
    """
    target, tmp = gz_path(log), gz_path(log).with_suffix(".gz.part")
    stat = log.stat()
    try:
        written = hashlib.sha256()
        with open(log, "rb") as src, open(tmp, "wb") as raw, \
                gzip.GzipFile(filename=log.name, mode="wb", fileobj=raw, compresslevel=level,
                              mtime=int(stat.st_mtime)) as dst:
            for chunk in iter(lambda: src.read(CHUNK), b""):
                written.update(chunk)
                dst.write(chunk)

        read_back = hashlib.sha256()
        with gzip.open(tmp, "rb") as fh:
            for chunk in iter(lambda: fh.read(CHUNK), b""):
                read_back.update(chunk)
        if read_back.digest() != written.digest():
            raise OSError("the compressed copy does not read back identical")

        os.utime(tmp, (stat.st_atime, stat.st_mtime))
        os.replace(tmp, target)
        log.unlink()
    except Exception as exc:
        Path(tmp).unlink(missing_ok=True)
        return Result(log, target, stat.st_size, error=str(exc))
    return Result(log, target, stat.st_size, target.stat().st_size)


def compress_all(logs: list, level: int = 6, jobs: int = DEFAULT_JOBS) -> list:
    """Compress logs in parallel, returning one :class:`Result` each in the order given."""
    if not logs:
        return []
    if jobs <= 1 or len(logs) == 1:
        return [compress_one(log, level) for log in logs]
    with ProcessPoolExecutor(max_workers=min(jobs, len(logs))) as pool:
        return list(pool.map(_worker, logs, [level] * len(logs)))


def _worker(log: Path, level: int) -> Result:
    return compress_one(log, level)
