"""The configuration a run actually ran with, and the difference between two runs.

`references/interpretation.md` says to state the configuration difference first: attention
backend, quantisation and profiling flags explain more differences than anything in the
communication numbers, and are the easiest thing to leave out. Given both runs' logs,
`diff_configs` puts them in one table, performance-relevant settings first.

A difference is reported, never resolved: which arm is misconfigured is a question about intent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spec import RunConfigLayout

#: Stands in for a secret's value, so an artifact never carries it.
REDACTED = "<redacted>"

#: A node that did not state a setting, which is not the same as one that stated nothing at all.
NOT_STATED = "<not stated>"


@dataclass(frozen=True)
class Setting:
    """One configuration key and what each side of a comparison had for it."""

    key: str
    left: str | None
    right: str | None

    @property
    def differs(self) -> bool:
        return self.left != self.right

    #: Filled by :func:`diff_configs` from the engine's own classification.
    moves_throughput: bool = False

    @property
    def perf_relevant(self) -> bool:
        return self.moves_throughput


def parse_config_line(line: str, layout: RunConfigLayout) -> dict:
    """The settings a log line reports, or an empty dict when it reports none."""
    if not layout.pattern:
        return {}
    hit = layout.pattern.search(line)
    if not hit:
        return {}
    return {key: (REDACTED if key in layout.secret else value.strip().strip("'\""))
            for key, value in layout.field_pattern.findall(hit.group(1))}


def merge_nodes(config: dict, expected: set | None = None) -> tuple[dict, dict]:
    """Collapse per-node settings into the values a role agreed on, plus the ones it did not.

    Disagreement between nodes of one role is a finding, not a detail to average away: a launcher
    that assembles rank 0's command separately is how a run ends up with three nodes carrying the
    measurement flags and one without.

    ``expected`` is the nodes the phase actually had. Without it agreement is judged only over the
    nodes whose startup line parsed, so the node most likely to differ vanishes rather than
    dissenting; a node that stated nothing is recorded as ``<not stated>``.

    Returns ``(agreed, disagreed)``, the second mapping a key to ``{node: value}``.
    """
    nodes = set(expected) if expected else set(config)
    agreed: dict = {}
    disagreed: dict = {}
    for key in {k for settings in config.values() for k in settings}:
        seen = {node: config.get(node, {}).get(key, NOT_STATED) for node in nodes}
        values = set(seen.values())
        if len(values) == 1 and NOT_STATED not in values:
            agreed[key] = values.pop()
        else:
            disagreed[key] = seen
    return agreed, disagreed


def normalise(key: str, value, path_valued: frozenset) -> str | None:
    """A value comparable across runs: a path-valued setting by its last component only."""
    if value is None or key not in path_valued:
        return value
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def diff_configs(left: dict, right: dict, include_noise: bool = False,
                 perf_relevant: frozenset = frozenset(),
                 noise: frozenset = frozenset(),
                 path_valued: frozenset = frozenset()) -> list:
    """Settings that differ between two runs, the performance-relevant ones first.

    Args:
        left: settings of the first run, as :func:`merge_nodes` agreed them
        right: settings of the second run
        include_noise: keep ports, paths and seeds, which differ on any two runs

    Returns:
        list[Setting]: one entry per differing key
    """
    keys = set(left) | set(right)
    if not include_noise:
        keys -= noise
    out = [Setting(key,
                   normalise(key, left.get(key), path_valued),
                   normalise(key, right.get(key), path_valued),
                   key in perf_relevant)
           for key in sorted(keys)]
    return sorted((s for s in out if s.differs), key=lambda s: (not s.perf_relevant, s.key))
