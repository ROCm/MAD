"""The expert all-to-all: classifying it out of a trace, and reporting it honestly.

MoRI and DeepEP appear in neither an RCCL log nor a ``record_param_comms`` event, so this is the
only channel that names the operation a backend comparison is about. Classification is by event
name, so unknown names must report nothing and say which names were busiest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (named_trace_event, trace_event, uncategorised_trace_event,
                      write)

from collprof.core.rccl_log import parse_run
from collprof.core.report import ReportContext, emit_phase
from collprof.core.spec import A2AKernels
from collprof.core.torch_trace import classify_a2a, parse_traces
from collprof.engines.sglang_disagg import A2A_VARIANTS as VARIANTS
from collprof.engines.sglang_disagg import SPEC

A2A = SPEC.a2a


def trace_with(*events: str, path: Path) -> Path:
    write(path, ["[", *events, "]"])
    return path


class TestClassification:
    def test_backend_dispatch_and_combine_are_recognised(self):
        assert classify_a2a("mori_ep_dispatch_kernel", A2A) == "dispatch"
        assert classify_a2a("deep_ep::combine", A2A) == "combine"
        assert classify_a2a("DeepEP_internode_dispatch", A2A) == "dispatch"

    def test_the_permute_that_frames_the_exchange_is_its_own_stage(self):
        assert classify_a2a("moe_pre_permute", A2A) == "permute"

    def test_the_transport_is_separated_from_the_stage(self):
        assert classify_a2a("rocshmem_putmem_signal", A2A) == "transport"

    def test_an_unrelated_kernel_is_claimed_by_nothing(self):
        assert classify_a2a("ncclDevKernel_Generic", A2A) == ""
        assert classify_a2a("Cijk_Alik_Bljk_HHS_BH", A2A) == ""

    def test_underscore_separated_names_still_match(self):
        """`_` is a word character, so a \\b boundary misses every real kernel name."""
        assert classify_a2a("mori_ibgda_put_nbi", A2A) == "transport"
        assert classify_a2a("mori_ep_barrier", A2A) == "backend other"
        assert classify_a2a("deep_ep_low_latency_layout", A2A) == "backend other"

    def test_a_name_merely_containing_the_letters_is_not_claimed(self):
        """The boundary still has to refuse this, or the fallback claims unrelated kernels."""
        assert classify_a2a("memorize_weights", A2A) == ""
        assert classify_a2a("amorized_gemm", A2A) == ""

    def test_an_engine_declaring_no_patterns_classifies_nothing(self):
        assert classify_a2a("mori_ep_dispatch_kernel", A2AKernels()) == ""


class TestTraceExtraction:
    def test_stages_are_counted_and_timed_per_trace(self, tmp_path: Path):
        path = trace_with(named_trace_event("mori_dispatch", dur=300.0),
                          named_trace_event("mori_dispatch", dur=100.0),
                          named_trace_event("mori_combine", dur=200.0),
                          path=tmp_path / "a.trace.json")
        got = parse_traces([path], SPEC.traces, A2A)
        assert got["a2a"][("dispatch", "kernel")] == [2, 400.0]
        assert got["a2a"][("combine", "kernel")] == [1, 200.0]

    def test_the_category_total_is_the_share_denominator(self, tmp_path: Path):
        path = trace_with(named_trace_event("mori_dispatch", dur=250.0),
                          named_trace_event("some_gemm", dur=750.0),
                          path=tmp_path / "a.trace.json")
        got = parse_traces([path], SPEC.traces, A2A)
        assert got["category_us"]["kernel"] == 1000.0

    def test_an_event_without_a_category_does_not_land_in_the_previous_one(self, tmp_path: Path):
        """`cat` is optional in the Chrome-trace format, and a stale one corrupts every share.

        The category total is the share denominator, so charging an uncategorised event to the
        previous event's category shrinks the exchange's apparent share of device time.
        """
        path = trace_with(named_trace_event("mori_dispatch", dur=250.0),
                          uncategorised_trace_event(dur=750.0),
                          path=tmp_path / "a.trace.json")

        got = parse_traces([path], SPEC.traces, A2A)

        assert got["category_us"]["kernel"] == 250.0
        assert got["a2a"][("dispatch", "kernel")] == [1, 250.0]
        assert "unknown" not in got["unmatched_us"]

    def test_unclassified_device_events_are_kept_for_extending_the_patterns(self, tmp_path: Path):
        path = trace_with(named_trace_event("mystery_a2a_kernel", dur=900.0),
                          path=tmp_path / "a.trace.json")
        got = parse_traces([path], SPEC.traces, A2A)
        assert got["a2a"] == {}
        assert got["unmatched_us"]["mystery_a2a_kernel"] == 900.0

    def test_collectives_still_parse_alongside(self, tmp_path: Path):
        """The a2a pass must not disturb the channel the report already had."""
        path = trace_with(trace_event(coll="allreduce"),
                          named_trace_event("mori_dispatch"),
                          path=tmp_path / "a.trace.json")
        got = parse_traces([path], SPEC.traces, A2A)
        assert got["events"] and got["a2a"]

    def test_an_engine_without_patterns_pays_nothing(self, tmp_path: Path):
        path = trace_with(named_trace_event("mori_dispatch"), path=tmp_path / "a.trace.json")
        got = parse_traces([path], SPEC.traces)
        assert got["a2a"] == {} and got["category_us"] == {}


class TestSection:
    def report(self, sglang_run: Path, out: Path, trace: dict | None) -> str:
        phase = parse_run(sglang_run, SPEC)["decode"]
        ctx = ReportContext(spec=SPEC, run_dir=sglang_run, torch_trace=trace)
        return emit_phase(phase, out, ctx).read_text()

    def test_the_stages_reach_the_report_with_their_share(self, sglang_run: Path, tmp_path: Path):
        trace = {"events": {}, "files": 1, "steps": 0, "group_size": 8, "ranks": [0],
                 "a2a": {("dispatch", "kernel"): [4, 400.0]},
                 "category_us": {"kernel": 1000.0}, "unmatched_us": {}}
        text = self.report(sglang_run, tmp_path / "out", trace)
        assert "## Expert all-to-all" in text
        assert "40.0%" in text
        # The engine's reason, not the core's.
        assert "rocSHMEM" in text

    def test_nothing_matched_says_so_instead_of_reporting_zero(
            self, sglang_run: Path, tmp_path: Path):
        trace = {"events": {}, "files": 1, "steps": 0, "group_size": 8, "ranks": [0],
                 "a2a": {}, "category_us": {"kernel": 500.0},
                 "unmatched_us": {"mystery_kernel": 500.0}}
        text = self.report(sglang_run, tmp_path / "out", trace)
        assert "No event name matched" in text
        assert "`mystery_kernel`" in text

    def test_no_trace_means_no_section(self, sglang_run: Path, tmp_path: Path):
        assert "## Expert all-to-all" not in self.report(sglang_run, tmp_path / "out", None)


class TestKernelVariants:
    """The variant column answers "which implementation ran" on two axes.

    Latency vs throughput, and intranode vs internode -- the second because an intranode arm
    turned a 14.7 ms per-step gap into 2.6 ms on the same backends.
    """

    def variant(self, name: str) -> str:
        return next((label for label, pattern in VARIANTS if pattern.search(name)), "not stated")

    @pytest.mark.parametrize("name,expected", [
        ("EpDispatchInterNodeV1KernelLowLatency_fp8_fnuz", "low latency"),
        ("EpDispatchInterNodeV1Kernel_fp8_fnuz", "normal"),
        ("EpDispatchIntraNodeKernel_fp8_fnuz", "intranode"),
        ("EpCombineIntraNodeKernel_bf16_nop2p", "intranode"),
        ("void primus_turbo::deep_ep::intranode::notify_dispatch<8>(int const*)", "intranode"),
        ("void primus_turbo::deep_ep::internode::notify_dispatch<false, 2>", "normal"),
    ])
    def test_a_real_kernel_name_says_which_implementation_ran(self, name, expected):
        assert self.variant(name) == expected

    def test_both_axes_at_once_keep_both(self):
        """First match wins, so the compound case must be ordered ahead of either single axis,
        or one of the two axes is silently dropped."""
        assert self.variant("EpDispatchIntraNodeKernelLowLatency_fp8") == "intranode low latency"

    def test_a_kernel_that_implies_neither_axis_claims_neither(self):
        """A layout helper is not an implementation, so it must not be labelled."""
        assert self.variant("void primus_turbo::deep_ep::layout::get_dispatch_layout<long>") \
            == "not stated"


def test_an_a2a_kernel_with_a_spliced_duration_keeps_its_call(tmp_path: Path):
    """An a2a kernel is counted on its duration line and nowhere else, so an unreadable duration
    must not drop the call -- unlike a collective, which is counted on a later args line."""
    good = named_trace_event("EpDispatchInterNodeV1Kernel_fp8", dur=50.0)
    spliced = named_trace_event("EpDispatchInterNodeV1Kernel_fp8", dur=50.0).replace(
        '"dur": 50.0', '"dur": 4.200.347')
    assert spliced != good, "the fixture must actually carry a spliced duration"
    write(tmp_path / "1000.0-TP-0.trace.json",
          ['{"traceEvents": ['] + [good, spliced] + [']}'])

    parsed = parse_traces([tmp_path], SPEC.traces, SPEC.a2a)

    (calls, dur), = parsed["a2a"].values()
    assert calls == 2, "both kernels ran and both are counted"
    assert dur == 50.0, "only the readable duration contributes, and nothing is invented"


def test_the_kernel_table_is_built_from_kernels_only(tmp_path: Path):
    """A host op and the kernel it launches overlap, so one share cannot contain both.

    `a2a_names` feeds the kernel table and variant summary; a cpu_op there inflates the share and
    can name a variant no kernel ran.
    """
    write(tmp_path / "1000.0-TP-0.trace.json",
          ['{"traceEvents": [']
          + [named_trace_event("EpDispatchInterNodeV1Kernel_fp8", cat="kernel", dur=90.0),
             named_trace_event("EpDispatchIntraNodeKernel_fp8", cat="cpu_op", dur=10.0)]
          + [']}'])

    parsed = parse_traces([tmp_path], SPEC.traces, SPEC.a2a)

    names = {name for _stage, name in parsed["a2a_names"]}
    assert names == {"EpDispatchInterNodeV1Kernel_fp8"}, "the cpu_op is not a kernel"
    assert {cat for _stage, cat in parsed["a2a"]} == {"kernel", "cpu_op"}, "stages keep both"


def test_a_name_on_its_own_line_still_names_the_kernel(tmp_path: Path):
    """`cat` and `name` are separate keys and a pretty-printer may split the line; reading the
    name only off the category's line lost the stage and variant while keeping the duration."""
    split_event = (
        '    {\n'
        '      "ph": "X", "cat": "kernel",\n'
        '      "name": "EpDispatchInterNodeV1KernelLowLatency_fp8_fnuz",\n'
        '      "pid": 1, "tid": 7,\n'
        '      "ts": 2000.0, "dur": 400.0\n'
        '    },'
    )
    path = trace_with(split_event, path=tmp_path / "a.trace.json")

    got = parse_traces([path], SPEC.traces, A2A)

    assert got["a2a"][("dispatch", "kernel")] == [1, 400.0]
    assert got["category_us"]["kernel"] == 400.0
    assert got["unmatched_us"] == {}
