"""Step time recovered from the engine's own accounting, and the shape of its distribution.

A gap constant across intervals is a fixed per-step cost; one that tracks the batch is
volume-limited. A single mean cannot tell them apart, hence the distribution and `graph_replay`.
"""

from __future__ import annotations

import pytest
from conftest import decode_batch_line, write

from collprof.core.rccl_log import parse_run
from collprof.core.spec import StepTimingLayout
from collprof.core.steps import (StepRecord, batch_invariance, by_node, graph_state,
                                 parse_step_line,
                                 percentile, summarise)
from collprof.engines.sglang_disagg import SPEC


def records(*pairs, graphed=True):
    return [parse_step_line(decode_batch_line(batch=b, rate=r, graphed=graphed), SPEC.steps)
            for b, r in pairs]


class TestParsing:
    def test_batch_and_rate_become_a_step_time(self):
        record = parse_step_line(decode_batch_line(batch=256, rate=1103.0), SPEC.steps)
        # 256 requests at 1103 tokens/s is 232 ms per step: the reported EP32 TPOT.
        assert record.step_ms == pytest.approx(232.1, abs=0.5)

    def test_graph_replay_is_read_when_the_engine_states_it(self):
        assert parse_step_line(decode_batch_line(graphed=True), SPEC.steps).graphed is True
        assert parse_step_line(decode_batch_line(graphed=False), SPEC.steps).graphed is False

    def test_an_older_engine_without_the_field_leaves_it_unknown(self):
        record = parse_step_line(decode_batch_line(graphed=None), SPEC.steps)
        assert record is not None and record.graphed is None

    def test_a_startup_interval_that_generated_nothing_is_dropped(self):
        """Rate zero is the first interval; it carries no step time and would divide by zero."""
        assert parse_step_line(decode_batch_line(batch=8, rate=0.0), SPEC.steps) is None

    def test_a_prefill_line_reports_no_interval(self):
        assert parse_step_line("Prefill batch. #new-seq: 1, #new-token: 1024", SPEC.steps) is None

    def test_an_engine_declaring_no_layout_parses_nothing(self):
        assert parse_step_line(decode_batch_line(), StepTimingLayout()) is None


class TestDistribution:
    def test_median_and_p95_describe_the_spread(self):
        stats = summarise(records((16, 100.0), (16, 100.0), (16, 50.0)))
        assert stats.intervals == 3
        assert stats.median_ms == pytest.approx(160.0)
        assert stats.max_ms == pytest.approx(320.0)
        assert stats.spread > 1.0

    def test_a_flat_distribution_has_spread_near_one(self):
        stats = summarise(records(*[(16, 100.0)] * 10))
        assert stats.spread == pytest.approx(1.0)

    def test_the_batch_range_is_kept(self):
        stats = summarise(records((8, 100.0), (256, 1000.0)))
        assert (stats.batch_min, stats.batch_max) == (8, 256)

    def test_graph_state_survives_only_when_every_interval_agrees(self):
        assert summarise(records((16, 100.0), (16, 90.0), graphed=False)).graphed is False
        mixed = records((16, 100.0), graphed=True) + records((16, 90.0), graphed=False)
        assert summarise(mixed).graphed is None

    def test_no_records_yield_no_stats(self):
        assert summarise([]) is None

    @pytest.mark.parametrize("fraction,expected", [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0)])
    def test_percentile_needs_no_interpolation_policy(self, fraction, expected):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], fraction) == expected

    def test_percentile_of_nothing_is_zero(self):
        assert percentile([], 0.95) == 0.0

    def test_percentile_is_nearest_rank_as_documented(self):
        """Nearest rank, not round-then-index: they disagree on p25 of eight samples."""
        eight = [float(v) for v in range(1, 9)]
        assert percentile(eight, 0.25) == 2.0
        assert percentile(eight, 0.95) == 8.0

    def test_every_percentile_is_an_observed_value(self):
        """No reported step time that no interval actually took."""
        values = [10.0, 20.0, 30.0, 40.0]
        for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0):
            assert percentile(values, fraction) in values


class TestPerNode:
    def test_each_node_is_summarised_separately(self, sglang_run):
        phases = parse_run(sglang_run, SPEC)
        per_node = by_node(phases["decode"].steps)
        assert set(per_node) == {"decode_NODE2", "decode_NODE3"}
        # 16 requests at 100 and at 80 tokens/s: 160 ms and 200 ms.
        assert per_node["decode_NODE2"].median_ms == pytest.approx(180.0)

    def test_prefill_reports_no_steps(self, sglang_run):
        phases = parse_run(sglang_run, SPEC)
        assert by_node(phases["prefill"].steps) == {}

    def test_a_node_whose_intervals_are_all_unusable_is_left_out(self, sglang_run):
        write(sglang_run / "decode_NODE2.log", [decode_batch_line(batch=8, rate=0.0)])
        phases = parse_run(sglang_run, SPEC)
        assert "decode_NODE2" not in by_node(phases["decode"].steps)


class TestGraphAgreement:
    def test_a_mixture_of_stated_and_unstated_stays_unknown(self):
        """`StepStats.graphed` means every interval agreed, so an unstated one blocks agreement;
        dropping `None` first reported a definite "replayed" no interval supports."""
        mixed = [StepRecord(batch=4, rate=16.0, graphed=True),
                 StepRecord(batch=4, rate=16.0, graphed=None)]
        assert summarise(mixed).graphed is None

    def test_intervals_that_all_agree_still_report_the_value(self):
        agreed = [StepRecord(batch=4, rate=16.0, graphed=False)] * 3
        assert summarise(agreed).graphed is False


class TestBatchInvariance:
    """`step_ms` divides rate by batch, valid only if the two describe the same interval.

    If they do not, the error scales with the batch and the groups' median step times fan out.
    """

    def steady(self, batch: int, step_ms: float, n: int) -> list:
        """Intervals whose rate genuinely matches their batch."""
        return [StepRecord(batch=batch, rate=1000.0 * batch / step_ms)] * n

    def test_fields_that_correspond_give_a_flat_spread(self):
        recs = self.steady(1, 200.0, 300) + self.steady(8, 200.0, 300)
        assert batch_invariance(recs, min_group=100) < 0.01

    def test_a_rate_that_does_not_track_the_batch_fans_out(self):
        """A fixed rate regardless of batch is the failure this check exists to catch."""
        recs = ([StepRecord(batch=1, rate=5.0)] * 300
                + [StepRecord(batch=8, rate=5.0)] * 300)
        assert batch_invariance(recs, min_group=100) > 1000.0

    def test_too_few_groups_reports_nothing_rather_than_zero(self):
        """One batch value cannot distinguish a flat spread from an untested one."""
        assert batch_invariance(self.steady(4, 200.0, 300), min_group=100) is None


class TestGraphStateLabel:
    """One label for every renderer: the CSV and the markdown table derived it separately, and
    only one was updated when the mixed state arrived."""

    def make(self, graphed, mixed):
        from collprof.core.steps import StepStats
        return StepStats(intervals=1, batch_min=1, batch_max=1, median_ms=1.0, p95_ms=1.0,
                         min_ms=1.0, max_ms=1.0, graphed=graphed, graphs_mixed=mixed)

    def test_a_mixed_node_is_not_reported_as_unstated(self):
        assert graph_state(self.make(None, True)) == "mixed"
        assert graph_state(self.make(None, False)) == "not stated"

    def test_definite_states_keep_their_words(self):
        assert graph_state(self.make(True, False)) == "replayed"
        assert graph_state(self.make(False, False)) == "off"


def test_speculative_algorithm_invalidates_the_derived_step_time():
    """Only a stated, non-default value withholds: `None` is what sglang prints when it is off."""
    from collprof.core.steps import invalidators
    from collprof.engines.sglang_disagg import SPEC

    off = {"decode_NODE0": {"speculative_algorithm": "None"}, "decode_NODE1": {}}
    assert invalidators(off, SPEC.steps) == []
    assert invalidators({}, SPEC.steps) == []

    # One node is enough: the channel pools all nodes into one distribution.
    on = {"decode_NODE0": {"speculative_algorithm": "None"},
          "decode_NODE1": {"speculative_algorithm": "EAGLE"}}
    found = invalidators(on, SPEC.steps)
    assert [(s, v) for s, v, _why in found] == [("speculative_algorithm", "EAGLE")]
