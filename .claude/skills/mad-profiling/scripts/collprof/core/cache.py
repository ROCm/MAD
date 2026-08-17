"""Reuse of a previous parse of the same inputs.

Reading one serving job takes tens of minutes -- a 2 GB decode log and hundreds of MB of gzipped
traces, nearly all of it waiting on shared storage -- while the report text goes through several
passes. Entries are keyed by the identity of their inputs and by the parser version, so stale bytes
and stale logic are reparsed rather than trusted.
"""

from __future__ import annotations

import pickle
from pathlib import Path

#: Bumped whenever parsing or validation changes, so caches built by the previous logic are not
#: reused. Raising a sanity bound counts: it changes which records are kept.
PARSE_VERSION = 7


def file_signature(paths: list) -> list:
    """Identity of a set of input files, plus the parser version that would read them."""
    return [PARSE_VERSION] + sorted((str(p), p.stat().st_size, int(p.stat().st_mtime))
                                    for p in paths)


class ParseCache:
    """A pickle keyed by input identity. Absent path means caching is off."""

    def __init__(self, path: Path | None):
        self.path = path
        self.store = pickle.loads(path.read_bytes()) if path and path.exists() else {}
        self.dirty = False

    def get(self, key: str, signature: list, compute, encode=None, decode=None):
        entry = self.store.get(key)
        if entry and entry["signature"] == signature:
            print(f"reusing parsed {key}")
            return decode(entry["data"]) if decode else entry["data"]

        value = compute()
        self.store[key] = {"signature": signature, "data": encode(value) if encode else value}
        self.dirty = True
        return value

    def flush(self) -> None:
        if self.path and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(pickle.dumps(self.store))
