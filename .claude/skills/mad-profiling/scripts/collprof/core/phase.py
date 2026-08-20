"""What was collected for one phase of a job, and the aggregations the report reads off it.

A phase is one comparable stretch of a run: a datatype for training (BF16, then FP8 through the
same stdout) or a role for disaggregated serving (prefill and decode, in separate logs). The core
does not care which -- :class:`~collprof.core.spec.EngineSpec` says where the name comes from.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

#: Why a line that looked like a collective record was not counted. Kept as separate reasons because
#: they call for different responses: torn writes are a property of logging eight ranks to one
#: stdout and are expected at a fraction of a percent, while records rejected for exceeding a bound
#: may well be real on a larger run and mean the bound needs raising.
DAMAGE_UNKNOWN_COLL = "unknown collective name"
DAMAGE_NRANKS_RANGE = "nranks outside the sane range"
DAMAGE_RANK_RANGE = "rank outside its own communicator"
DAMAGE_TWO_RECORDS = "two records spliced into one line"
DAMAGE_NO_TAIL = "record tail missing where the log has them"
DAMAGE_MSG_CAP = "message larger than the size cap"

#: Topology lines tear the same way collective records do, and are counted separately because they
#: feed a different table: a rejected one costs an edge, not volume.
DAMAGE_TOPO_TRANSPORT = "topology line with an unknown transport"

#: Rejections that mean "this bound may be too low for this run" rather than "the log is torn".
BOUND_DAMAGE = (DAMAGE_MSG_CAP, DAMAGE_NRANKS_RANGE)


@dataclass
class Phase:
    """Everything collected for one phase of a job."""

    name: str
    engine: str = ""
    # (collective, nranks, msg_bytes, dtype_name) -> calls
    sizes: dict = field(default_factory=lambda: defaultdict(int))
    # (node, collective) -> [calls, bytes]
    per_node: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    # (node, rank) -> [calls, bytes]
    per_rank: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    # metric key -> node -> values. Megatron prints iteration timings from a single rank, so the
    # values are kept per node and the node that actually logged them is used.
    metrics: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    # (src_rank, dst_rank, transport) -> set of channel ids
    edges: dict = field(default_factory=lambda: defaultdict(set))
    nodes: set = field(default_factory=set)
    ranks: set = field(default_factory=set)
    #: Which kinds of stream the collective records came from (see ``LOG_*`` in spec.py). A phase
    #: read from per-rank files may not be told it tore because its ranks shared a stdout.
    writers: set = field(default_factory=set)
    #: Rejection reason -> count. See the DAMAGE_* constants.
    damage: Counter = field(default_factory=Counter)
    #: Same, for the topology lines behind ``edges``. Kept apart so the discarded share of
    #: collective records stays a share of collective records.
    topo_damage: Counter = field(default_factory=Counter)

    # -- metrics ---------------------------------------------------------------------------------

    def metric(self, key: str) -> list:
        """The longest per-node series for a metric, i.e. the node that actually logged it."""
        by_node = self.metrics.get(key)
        if not by_node:
            return []
        return max(by_node.values(), key=len, default=[])

    def add_metric(self, key: str, node: str, value: float) -> None:
        self.metrics[key][node].append(value)

    # -- aggregations ----------------------------------------------------------------------------

    @property
    def damaged(self) -> int:
        return sum(self.damage.values())

    @property
    def topo_damaged(self) -> int:
        return sum(self.topo_damage.values())

    @property
    def nranks(self) -> int:
        vals = [n for (_, n, _, _) in self.sizes if n > 1]
        return max(vals) if vals else 1

    def active_ranks(self, idle_fraction: float) -> dict:
        """Ranks that carried real traffic, keyed as in ``per_rank``.

        A disaggregated proxy is free to leave one replica of a role nearly idle -- in one run a
        decode node served 8 batches against 3874 on the other -- and averaging over ranks that did
        nothing would halve every per-rank figure.
        """
        if not self.per_rank:
            return {}
        busiest = max(v[1] for v in self.per_rank.values())
        floor = busiest * idle_fraction
        return {k: v for k, v in self.per_rank.items() if v[1] >= floor} or dict(self.per_rank)

    def collective_totals(self, multi_rank_only: bool = True) -> dict:
        out: dict = defaultdict(lambda: {"calls": 0, "bytes": 0, "sizes": set()})
        for (coll, nranks, msg_bytes, _dt), calls in self.sizes.items():
            if multi_rank_only and nranks <= 1:
                continue
            row = out[coll]
            row["calls"] += calls
            row["bytes"] += calls * msg_bytes
            row["sizes"].add(msg_bytes)
        return out

    def ranks_per_node(self) -> dict:
        counts: dict = defaultdict(int)
        for node, _rank in self.per_rank:
            counts[node] += 1
        return dict(counts)

    # -- persistence -----------------------------------------------------------------------------
    # The parse cache pickles these, and the defaultdict factories are lambdas, which do not pickle.

    def to_state(self) -> dict:
        return {"name": self.name, "engine": self.engine, "sizes": dict(self.sizes),
                "per_node": dict(self.per_node), "per_rank": dict(self.per_rank),
                "metrics": {k: dict(v) for k, v in self.metrics.items()},
                "edges": dict(self.edges), "nodes": self.nodes, "ranks": self.ranks,
                "writers": self.writers, "damage": dict(self.damage),
                "topo_damage": dict(self.topo_damage)}

    @classmethod
    def from_state(cls, state: dict) -> "Phase":
        phase = cls(state["name"], state.get("engine", ""))
        phase.sizes.update(state["sizes"])
        phase.per_node.update(state["per_node"])
        phase.per_rank.update(state["per_rank"])
        for key, by_node in state["metrics"].items():
            phase.metrics[key].update(by_node)
        phase.edges.update(state["edges"])
        phase.nodes, phase.ranks = state["nodes"], state["ranks"]
        phase.writers = set(state.get("writers", ()))
        phase.damage.update(state["damage"])
        phase.topo_damage.update(state.get("topo_damage", {}))
        return phase
