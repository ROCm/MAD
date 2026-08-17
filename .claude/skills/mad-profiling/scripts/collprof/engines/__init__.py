"""The engine registry: every engine the reports understand, and how a run is recognised.

Adding an engine is one module here plus one line in :data:`REGISTRY`. Nothing in ``collprof.core``
changes, and no existing engine is touched -- the checklist is in references/engines.md.

Detection is by log layout and is never a guess: a run whose layout matches two engines, or none,
stops with what was found rather than picking one.
"""

from __future__ import annotations

from pathlib import Path

from ..core.spec import EngineSpec
from . import primus, sglang_disagg

REGISTRY: dict = {
    primus.SPEC.name: primus.SPEC,
    sglang_disagg.SPEC.name: sglang_disagg.SPEC,
}


def get(name: str) -> EngineSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit(f"unknown engine {name!r}; known: {', '.join(sorted(REGISTRY))}")


def matches(run_dir: Path, spec: EngineSpec) -> list:
    """Log files under ``run_dir`` that this engine claims."""
    return [p for pattern in spec.logs.globs for p in run_dir.glob(pattern)]


def detect(run_dir: Path) -> tuple:
    """Identify the engine that produced a run. Returns (spec, reason).

    The reason is printed and recorded in the report header, because an engine chosen silently is an
    engine a reader cannot check.
    """
    hits = {name: matches(run_dir, spec) for name, spec in REGISTRY.items()}
    claimed = {name: files for name, files in hits.items() if files}

    if not claimed:
        raise SystemExit(
            f"no known engine recognises {run_dir}. Looked for: "
            + "; ".join(f"{name} ({', '.join(spec.logs.globs)})"
                        for name, spec in sorted(REGISTRY.items()))
            + ". Pass --engine to force one, or add an engine (references/engines.md).")
    if len(claimed) > 1:
        raise SystemExit(
            f"{run_dir} matches more than one engine: "
            + "; ".join(f"{name} via {len(files)} log(s)"
                        for name, files in sorted(claimed.items()))
            + ". Pass --engine to disambiguate.")

    name, files = next(iter(claimed.items()))
    return REGISTRY[name], f"{len(files)} log(s) matching {', '.join(REGISTRY[name].logs.globs)}"
