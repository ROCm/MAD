"""The adapter-counter channel: the sampler that writes it and the parser that reads it.

The sampler is bash reading sysfs, and it takes its root from `RDMA_SYSFS_ROOT`, so these tests
run it for real against a directory of files.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import server_args_line, write

from collprof.core.rdma_counters import by_kind, parse_counters
from collprof.engines.sglang_disagg import SPEC

LAYOUT = SPEC.counters
#: Asserted rather than counted by eye: a wrong path makes every sampler test fail as "no such
#: file" and read as though the sampler itself were broken.
SAMPLER = (Path(__file__).resolve().parents[5] / "scripts" / "sglang_disagg"
           / "rdma_counters.sh")
assert SAMPLER.is_file(), f"sampler not found at {SAMPLER}"

HEADER = "epoch_ns,device,port,counter,value"

#: The comparison refuses to render unless both arms measured the same benchmark points, so every
#: arm built here carries the same ones.
POINTS = {(1024, 1024, 64): {"mean_itl_ms": 226.0}}


def sampled(rows: list) -> list:
    return [HEADER] + rows


class TestParse:
    def test_the_delta_over_the_window_is_what_the_run_put_on_the_wire(self, tmp_path: Path):
        path = write(tmp_path / "decode_NODE2.csv", sampled([
            "1000000000,bnxt_re0,1,rx_write_req,100",
            "1000000000,bnxt_re0,1,rx_read_req,10",
            "3000000000,bnxt_re0,1,rx_write_req,700",
            "3000000000,bnxt_re0,1,rx_read_req,12",
        ]))

        series = parse_counters([path], LAYOUT)["decode_NODE2"]

        assert series.deltas[("bnxt_re0", "1", "rx_write_req")] == 600
        assert series.deltas[("bnxt_re0", "1", "rx_read_req")] == 2
        assert series.seconds == 2.0
        assert series.samples == 2
        assert series.per_second(("bnxt_re0", "1", "rx_write_req")) == 300.0

    def test_a_counter_that_went_backwards_is_dropped_and_named(self, tmp_path: Path):
        """A wrap is not negative work: the delta is dropped and the counter named, so the
        column reads as a floor rather than as a negative rate or a complete row."""
        path = write(tmp_path / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_requests,4294967290",
            "2000000000,mlx5_0,1,rx_write_requests,5",
        ]))

        series = parse_counters([path], LAYOUT)["decode_NODE2"]

        assert series.deltas == {}
        assert series.wrapped == (("mlx5_0", "1", "rx_write_requests"),)

    def test_one_sample_is_not_a_window(self, tmp_path: Path):
        """The counters are cumulative since the adapter came up, so a lone sample says nothing."""
        path = write(tmp_path / "decode_NODE2.csv",
                     sampled(["1000000000,mlx5_0,1,rx_write_requests,12345"]))

        series = parse_counters([path], LAYOUT)["decode_NODE2"]

        assert series.samples == 1
        assert series.seconds == 0.0
        assert series.deltas == {}, "first and last sample are the same one"
        assert series.per_second(("mlx5_0", "1", "rx_write_requests")) is None

    def test_a_header_only_file_is_a_node_that_found_nothing(self, tmp_path: Path):
        """Distinct from a node that never sampled: one is coverage, the other is absence."""
        path = write(tmp_path / "prefill_NODE0.csv", [HEADER])

        series = parse_counters([path], LAYOUT)

        assert set(series) == {"prefill_NODE0"}
        assert series["prefill_NODE0"].samples == 0

    def test_a_torn_line_costs_its_own_counter_and_nothing_else(self, tmp_path: Path):
        """A sampler killed mid-write leaves a partial row; the window is the rows that parsed."""
        path = write(tmp_path / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_requests,100",
            "2000000000,mlx5_0,1,rx_write_re",
            "3000000000,mlx5_0,1,rx_write_requests,400",
        ]))

        series = parse_counters([path], LAYOUT)["decode_NODE2"]

        assert series.deltas[("mlx5_0", "1", "rx_write_requests")] == 300


class TestKinds:
    def test_counters_are_grouped_by_the_engine_s_own_names(self, tmp_path: Path):
        """mlx5 and bnxt_re spell the same operation differently, so the engine classifies."""
        path = write(tmp_path / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_requests,0",
            "1000000000,mlx5_0,1,rx_read_requests,0",
            "1000000000,mlx5_0,1,rx_atomic_requests,0",
            "1000000000,mlx5_0,1,np_ecn_marked_roce_packets,0",
            "2000000000,mlx5_0,1,rx_write_requests,900",
            "2000000000,mlx5_0,1,rx_read_requests,3",
            "2000000000,mlx5_0,1,rx_atomic_requests,8",
            "2000000000,mlx5_0,1,np_ecn_marked_roce_packets,2",
        ]))

        kinds = by_kind(parse_counters([path], LAYOUT)["decode_NODE2"], LAYOUT)

        assert sum(kinds["rx write req"].values()) == 900
        assert sum(kinds["rx read req"].values()) == 3
        assert sum(kinds["rx atomic req"].values()) == 8
        assert sum(kinds["np_ecn_marked_roce_packets"].values()) == 2, "keeps its own name"

    def test_an_unclassified_counter_keeps_its_own_name(self, tmp_path: Path):
        """Pooling the unclassified counters summed packets, errors and gauges into one
        unitless number that a percentage was then taken of."""
        path = write(tmp_path / "decode_NODE2.csv", sampled([
            "1000000000,bnxt_re0,1,active_rc_qp,0",
            "2000000000,bnxt_re0,1,active_rc_qp,64",
        ]))

        kinds = by_kind(parse_counters([path], LAYOUT)["decode_NODE2"], LAYOUT)

        assert sum(kinds["active_rc_qp"].values()) == 64


@pytest.mark.skipif(shutil.which("bash") is None, reason="the sampler is bash")
class TestSampler:
    """The sampler itself, run against a directory of files standing in for sysfs."""

    def fake_sysfs(self, root: Path, values: dict) -> Path:
        port = root / "bnxt_re0" / "ports" / "1" / "hw_counters"
        port.mkdir(parents=True, exist_ok=True)
        for name, value in values.items():
            (port / name).write_text(f"{value}\n")
        return root

    def run(self, root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SAMPLER), "--out", str(out), "--once", *extra],
            env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
            capture_output=True, text=True, timeout=60)

    def test_a_sample_is_readable_by_the_parser_that_consumes_it(self, tmp_path: Path):
        """The two halves are written together and drift apart quietly; this is the seam."""
        root = self.fake_sysfs(tmp_path / "sys", {"rx_write_req": 10, "rx_read_req": 1})
        out = tmp_path / "rdma" / "decode_NODE2.csv"

        assert self.run(root, out).returncode == 0
        self.fake_sysfs(root, {"rx_write_req": 610, "rx_read_req": 3})
        assert self.run(root, out).returncode == 0

        series = parse_counters([out], LAYOUT)["decode_NODE2"]
        assert series.samples == 2
        assert series.deltas[("bnxt_re0", "1", "rx_write_req")] == 600
        assert series.deltas[("bnxt_re0", "1", "rx_read_req")] == 2

    def test_a_host_without_adapters_leaves_a_header_and_says_so(self, tmp_path: Path):
        """The same launcher runs on hosts with no RDMA device; that is not a failure."""
        empty = tmp_path / "sys"
        empty.mkdir()
        out = tmp_path / "rdma" / "prefill_NODE0.csv"

        done = self.run(empty, out)

        assert done.returncode == 0
        assert "nothing to sample" in done.stderr
        assert out.read_text().strip() == HEADER

    def test_only_the_adapters_the_run_was_given_are_sampled(self, tmp_path: Path):
        """Adapters the deployment did not name carry traffic that is not this job's, and
        summing it in would put common-mode noise into a channel that is a difference of arms."""
        root = self.fake_sysfs(tmp_path / "sys", {"rx_write_req": 5})
        (root / "mlx5_9" / "ports" / "1" / "hw_counters").mkdir(parents=True)
        (root / "mlx5_9" / "ports" / "1" / "hw_counters" / "rx_write_req").write_text("999\n")
        out = tmp_path / "rdma" / "decode_NODE2.csv"

        assert self.run(root, out, "--devices", "bnxt_re0").returncode == 0

        text = out.read_text()
        assert "bnxt_re0,1,rx_write_req,5" in text
        assert "mlx5_9" not in text

    def test_an_unreadable_counter_does_not_end_the_sample(self, tmp_path: Path):
        """One counter can fail for reasons that are not the sampler's business."""
        root = self.fake_sysfs(tmp_path / "sys", {"rx_write_req": 7, "nonsense": "n/a"})
        out = tmp_path / "rdma" / "decode_NODE3.csv"

        assert self.run(root, out).returncode == 0

        text = out.read_text()
        assert "rx_write_req,7" in text
        assert "nonsense" not in text, "a non-numeric counter is skipped, not written"


class TestReported:
    """The channel as a reader meets it: a report section and a comparison row."""

    SAMPLES = ["1000000000,bnxt_re0,1,rx_write_req,0",
               "1000000000,bnxt_re0,1,rx_read_req,0",
               "3000000000,bnxt_re0,1,rx_write_req,2000",
               "3000000000,bnxt_re0,1,rx_read_req,40"]

    def test_a_run_that_sampled_gets_a_section(self, sglang_run: Path, tmp_path: Path):
        from collprof.cli import load_counters
        from collprof.core.rccl_log import parse_run
        from collprof.core.report import ReportContext, emit_phase

        write(sglang_run / "rdma" / "decode_NODE2.csv", sampled(self.SAMPLES))
        counters = load_counters(sglang_run, SPEC)
        out = tmp_path / "out_decode"
        emit_phase(parse_run(sglang_run, SPEC)["decode"], out,
                   ReportContext(spec=SPEC, run_dir=sglang_run, counters=counters))

        text = (out / "report.md").read_text()

        assert "## What crossed the fabric" in text
        assert "| decode_NODE2 | 2 | 2 | 2,000 | 40 |" in text
        # The KV transfer shares the adapter, so the count bounds rather than measures.
        assert "ceiling" in text

    def test_a_run_that_did_not_sample_gets_no_section(self, sglang_run: Path, tmp_path: Path):
        """Absence of a channel is not a finding, and an empty table would read as one."""
        from collprof.core.rccl_log import parse_run
        from collprof.core.report import ReportContext, emit_phase

        out = tmp_path / "out_decode"
        emit_phase(parse_run(sglang_run, SPEC)["decode"], out,
                   ReportContext(spec=SPEC, run_dir=sglang_run))

        assert "What crossed the fabric" not in (out / "report.md").read_text()

    def test_the_comparison_puts_the_arms_side_by_side_as_totals(self, tmp_path: Path):
        """Totals, not rates: the arms serve the same requests but not in the same wall clock,
        so dividing by it would invent a difference where the counters show none."""
        from collprof.core.compare import (Arm, build_counters, section_counters, tables)

        def arm(name, writes, reads):
            path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled([
                "1000000000,bnxt_re0,1,rx_write_req,0",
                "1000000000,bnxt_re0,1,rx_read_req,0",
                f"3000000000,bnxt_re0,1,rx_write_req,{writes}",
                f"3000000000,bnxt_re0,1,rx_read_req,{reads}"]))
            return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                       counters_comparable=True,
                       counters=parse_counters([path], LAYOUT))

        mori, deepep = arm("MoRI", 2000, 20), arm("DeepEP", 2400, 900)

        text = "\n".join(section_counters(mori, deepep, SPEC))

        assert "| rx write req | 2,000 | 2,400 | +20.0% |" in text
        assert "| rx read req | 20 | 900 | +4400.0% |" in text
        assert "Totals, not rates" in text
        # The same rows reach the CSV, so the workbook carries the channel too.
        header, rows = build_counters(mori, deepep, SPEC)
        assert header == ["kind", "MoRI", "DeepEP", "delta_pct"]
        assert ["rx write req", 2000, 2400, "+20.0%"] in rows
        assert "fabric_counters" in tables(mori, deepep, SPEC)


def test_a_line_of_nul_bytes_is_skipped_rather_than_ending_the_parse(tmp_path: Path):
    """The shared filesystem returns NUL bytes instead of an error when a read lands badly, and
    `csv` raises "field larger than field limit" on them -- taking a whole report down."""
    path = tmp_path / "decode_NODE2.csv"
    path.write_text(HEADER + "\n"
                    + "1000000000,mlx5_0,1,rx_write_requests,100\n"
                    + "2000000000,mlx5_0," + "\x00" * 200000 + "\n"
                    + "3000000000,mlx5_0,1,rx_write_requests,700\n")

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.deltas[("mlx5_0", "1", "rx_write_requests")] == 600
    assert series.damaged == 1


def test_the_ib_data_counters_are_words_not_bytes(tmp_path: Path):
    """`port_rcv_data` is in 4-octet words, so reporting its raw delta as bytes understates
    every volume by four with no second channel in the report to contradict it."""
    from collprof.core.rdma_counters import by_kind

    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,port_rcv_data,0",
        "1000000000,mlx5_0,1,rx_write_requests,0",
        "2000000000,mlx5_0,1,port_rcv_data,1000",
        "2000000000,mlx5_0,1,rx_write_requests,1000",
    ]))

    kinds = by_kind(parse_counters([path], LAYOUT)["decode_NODE2"], LAYOUT)

    assert sum(kinds["rx bytes"].values()) == 4000, "1000 words is 4000 bytes"
    assert sum(kinds["rx write req"].values()) == 1000, "an operation count is not scaled"


def test_a_node_absent_from_the_collectives_still_reports_its_adapters(
        sglang_run: Path, tmp_path: Path):
    """A node whose RCCL records were all rejected is absent from `Phase.nodes` while its NIC
    still carried the run's traffic; filtering on that set alone dropped it."""
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    # A decode node that stated its configuration but logged no usable collective.
    write(sglang_run / "decode_NODE4.log", [server_args_line()])
    write(sglang_run / "rdma" / "decode_NODE4.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,0",
        "3000000000,mlx5_0,1,rx_write_req,900"]))
    phase = parse_run(sglang_run, SPEC)["decode"]
    assert "decode_NODE4" not in phase.nodes, "it logged no usable collective"
    assert "decode_NODE4" in phase.config_nodes, "but it is a node of the phase"

    out = tmp_path / "out_decode"
    emit_phase(phase, out, ReportContext(spec=SPEC, run_dir=sglang_run,
                                         counters=load_counters(sglang_run, SPEC)))

    assert "| decode_NODE4 |" in (out / "report.md").read_text()
    assert (out / "fabric_counters.csv").exists(), "and the CSV carries it, not only the markdown"


def test_a_node_whose_log_never_arrived_still_reports_its_adapters(
        sglang_run: Path, tmp_path: Path):
    """The counters are filtered by the launcher's naming rule, not by which logs survived.

    Filtering against the nodes recovered from logs dropped a node whose log never landed, so the
    phase's fabric volume came out short with nothing saying a node was missing.
    """
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    write(sglang_run / "rdma" / "decode_NODE7.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,0",
        "3000000000,mlx5_0,1,rx_write_req,700"]))
    phase = parse_run(sglang_run, SPEC)["decode"]
    assert "decode_NODE7" not in set(phase.nodes) | set(phase.config_nodes), "no log of any kind"

    out = tmp_path / "out_decode"
    emit_phase(phase, out, ReportContext(spec=SPEC, run_dir=sglang_run,
                                         counters=load_counters(sglang_run, SPEC)))

    assert "| decode_NODE7 |" in (out / "report.md").read_text()


def test_a_decode_report_carries_only_decode_nodes_in_its_csv(sglang_run: Path, tmp_path: Path):
    """Samples are per node for the whole job and a report is per phase; the markdown filtered
    and the CSV did not, so the two disagreed about the same run."""
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    for node in ("prefill_NODE0", "decode_NODE2"):
        write(sglang_run / "rdma" / f"{node}.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_req,0",
            f"3000000000,mlx5_0,1,rx_write_req,{100 if node.startswith('prefill') else 900}"]))

    out = tmp_path / "out_decode"
    emit_phase(parse_run(sglang_run, SPEC)["decode"], out,
               ReportContext(spec=SPEC, run_dir=sglang_run,
                             counters=load_counters(sglang_run, SPEC)))

    csv_text = (out / "fabric_counters.csv").read_text()
    assert "decode_NODE2" in csv_text
    assert "prefill_NODE0" not in csv_text, "the other role's adapters are not this phase's"


def test_a_sampled_node_with_no_deltas_does_not_advertise_the_csv(sglang_run: Path,
                                                                  tmp_path: Path):
    """`write_tables` creates `fabric_counters.csv` only when a delta row exists, while the file
    list advertised it whenever a counter file was found -- the supported header-only case then
    pointed a reader at an artifact nobody wrote."""
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    # What the sampler leaves on a host with no RDMA device: it ran, and there is nothing in it.
    write(sglang_run / "rdma" / "decode_NODE2.csv", ["epoch_ns,device,port,counter,value"])

    out = tmp_path / "out_decode"
    emit_phase(parse_run(sglang_run, SPEC)["decode"], out,
               ReportContext(spec=SPEC, run_dir=sglang_run,
                             counters=load_counters(sglang_run, SPEC)))

    assert not (out / "fabric_counters.csv").exists(), "the writer makes no empty table"
    assert "fabric_counters.csv" not in (out / "report.md").read_text()


class TestCollectionFailureIsNotAFinding:
    """An arm that failed to sample must not be compared as though it measured zero."""

    def arm(self, name: str, tmp_path: Path, rows: list):
        from collprof.core.compare import Arm
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled(rows))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    def test_one_arm_without_a_window_withholds_the_comparison(self, tmp_path: Path):
        from collprof.core.compare import build_counters, section_counters

        good = self.arm("MoRI", tmp_path, ["1000000000,mlx5_0,1,rx_write_req,0",
                                           "3000000000,mlx5_0,1,rx_write_req,900"])
        # One sample is cumulative since boot, so it says nothing about this run.
        bad = self.arm("DeepEP", tmp_path, ["1000000000,mlx5_0,1,rx_write_req,12345"])

        assert build_counters(good, bad, SPEC)[1] == []
        text = "\n".join(section_counters(good, bad, SPEC))
        assert "**Withheld.** DeepEP sampled no window" in text
        assert "900" not in text, "the working arm's counts are not shown against a zero"

    def test_two_arms_that_sampled_and_saw_nothing_say_that_instead(self, tmp_path: Path):
        from collprof.core.compare import section_counters

        flat = ["1000000000,mlx5_0,1,rx_write_req,7", "3000000000,mlx5_0,1,rx_write_req,7"]
        text = "\n".join(section_counters(self.arm("MoRI", tmp_path, flat),
                                          self.arm("DeepEP", tmp_path, flat), SPEC))

        assert "no counter moved" in text
        assert "Withheld" not in text, "sampling worked; there was simply nothing on these NICs"


class TestIntervalValidation:
    """A zero interval turned the sampler into a busy loop against a shared filesystem."""

    def run_interval(self, tmp_path: Path, value: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SAMPLER), "--out", str(tmp_path / "o.csv"), "--once",
             "--interval", value],
            env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(tmp_path)},
            capture_output=True, text=True, timeout=60)

    def test_a_non_positive_or_fractional_interval_is_refused(self, tmp_path: Path):
        for value in ("0", "-5", "1.5", "abc", ""):
            done = self.run_interval(tmp_path, value)
            assert done.returncode == 2, value
            assert "positive whole number" in done.stderr, value

    def test_a_sane_interval_is_accepted(self, tmp_path: Path):
        assert self.run_interval(tmp_path, "30").returncode == 0


def test_a_wrap_that_climbs_back_past_its_start_is_still_caught(tmp_path: Path):
    """End-to-end differencing hides a reset: 900 -> 5 -> 1000 reads as a plausible +100 when
    what happened is a wrap of unrecoverable size."""
    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,900",
        "2000000000,mlx5_0,1,rx_write_req,5",
        "3000000000,mlx5_0,1,rx_write_req,1000",
    ]))

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.deltas == {}, "a key that ever fell is not counted"
    assert series.wrapped == (("mlx5_0", "1", "rx_write_req"),)


def test_a_counter_is_summed_across_samples_not_across_the_ends(tmp_path: Path):
    """Same total here, but the sum is what stays right when a sample is missing or torn."""
    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,10",
        "2000000000,mlx5_0,1,rx_write_req,40",
        "3000000000,mlx5_0,1,rx_write_req,100",
    ]))

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.deltas[("mlx5_0", "1", "rx_write_req")] == 90


class TestUnequalCoverageWithholds:
    """Totals summed over different amounts of hardware are not a backend difference."""

    def arm(self, name: str, tmp_path: Path, nodes: list, devices=("mlx5_0",), bytes_too=False):
        from collprof.core.compare import Arm
        paths = []
        for n in nodes:
            rows = []
            for dev in devices:
                rows += [f"1000000000,{dev},1,rx_write_req,0",
                         f"3000000000,{dev},1,rx_write_req,100"]
                if bytes_too:
                    rows += [f"1000000000,{dev},1,port_rcv_data,0",
                             f"3000000000,{dev},1,port_rcv_data,250"]
            paths.append(write(tmp_path / name / "rdma" / f"decode_NODE{n}.csv", sampled(rows)))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters(paths, LAYOUT))

    def test_different_node_counts_withhold(self, tmp_path: Path):
        from collprof.core.compare import build_counters, section_counters

        left, right = self.arm("MoRI", tmp_path, [2, 3]), self.arm("DeepEP", tmp_path, [2])

        assert build_counters(left, right, SPEC)[1] == []
        text = "\n".join(section_counters(left, right, SPEC))
        assert "**Withheld.**" in text and "did not sample the same hardware" in text
        assert "MoRI 2 node(s) and 2 adapter(s)" in text

    def test_equal_nodes_but_a_missing_adapter_withholds_too(self, tmp_path: Path):
        """Eight NICs against seven is the same trap one level down, and node counts hide it."""
        from collprof.core.compare import build_counters, section_counters

        left = self.arm("MoRI", tmp_path, [2], devices=("mlx5_0", "mlx5_2"))
        right = self.arm("DeepEP", tmp_path, [2], devices=("mlx5_0",))

        assert build_counters(left, right, SPEC)[1] == []
        text = "\n".join(section_counters(left, right, SPEC))
        assert "MoRI 1 node(s) and 2 adapter(s), DeepEP 1 and 1" in text

    def test_the_same_number_of_different_adapters_withholds(self, tmp_path: Path):
        """Two arms can each sample eight adapters and sample eight different ones, so the
        totals differ by hardware and a gate that only counts cannot tell."""
        from collprof.core.compare import build_counters, section_counters

        left = self.arm("MoRI", tmp_path, [2], devices=("mlx5_0", "mlx5_2"))
        right = self.arm("DeepEP", tmp_path, [2], devices=("mlx5_0", "mlx5_4"))

        assert build_counters(left, right, SPEC)[1] == []
        text = "\n".join(section_counters(left, right, SPEC))
        assert "**Withheld.**" in text
        assert "MoRI only: decode_NODE2/mlx5_2" in text
        assert "DeepEP only: decode_NODE2/mlx5_4" in text

    def test_an_empty_table_does_not_claim_the_fabric_was_idle(self, tmp_path: Path):
        """Only kinds both arms report are compared, so a kind that moved on one arm alone
        leaves no row -- and "no counter moved" then states the opposite of the samples."""
        from collprof.core.compare import build_counters, section_counters

        left = self.arm("MoRI", tmp_path, [2])
        right = self.arm("DeepEP", tmp_path, [2])
        # One arm's only movement is on a counter the other never reported.
        right.counters["decode_NODE2"].deltas.clear()
        right.counters["decode_NODE2"].deltas[("mlx5_0", "1", "rx_read_req")] = 500

        assert build_counters(left, right, SPEC)[1] == []
        text = "\n".join(section_counters(left, right, SPEC))
        assert "no counter moved" not in text
        assert "**Withheld.**" in text
        assert "moved only on MoRI (rx write req)" in text
        assert "moved only on DeepEP (rx read req)" in text

    def test_matching_coverage_renders_the_table(self, tmp_path: Path):
        from collprof.core.compare import section_counters

        left = self.arm("MoRI", tmp_path, [2, 3], devices=("mlx5_0", "mlx5_2"))
        right = self.arm("DeepEP", tmp_path, [2, 3], devices=("mlx5_0", "mlx5_2"))

        text = "\n".join(section_counters(left, right, SPEC))

        assert "Withheld" not in text
        assert "MoRI: 2 node(s), 4 adapter(s)" in text
        assert "| rx write req | 400 | 400 |" in text

    def test_the_table_is_not_headed_operations(self, tmp_path: Path):
        """The rows include byte volumes, packet counts and unclassified driver counters, so a
        column headed `operations` puts the wrong unit on most of them. The CSV says `kind`."""
        from collprof.core.compare import section_counters

        left = self.arm("MoRI", tmp_path, [2], devices=("mlx5_0",), bytes_too=True)
        right = self.arm("DeepEP", tmp_path, [2], devices=("mlx5_0",), bytes_too=True)

        text = "\n".join(section_counters(left, right, SPEC))

        assert "| operations |" not in text
        assert "| counter kind |" in text
        assert "| rx bytes |" in text, "a byte volume sits under that heading"


class TestRequestedAdaptersAreChecked:
    """A node-wide check passes while a requested adapter contributes nothing, silently."""

    def sysfs(self, root: Path, with_counters: list, without: list) -> Path:
        for name in with_counters:
            d = root / name / "ports" / "1" / "hw_counters"
            d.mkdir(parents=True, exist_ok=True)
            (d / "rx_write_req").write_text("5\n")
        for name in without:
            (root / name / "ports" / "1" / "gids").mkdir(parents=True, exist_ok=True)
        return root

    def run(self, root: Path, out: Path, devices: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SAMPLER), "--out", str(out), "--once", "--devices", devices],
            env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
            capture_output=True, text=True, timeout=60)

    def test_a_requested_adapter_without_counters_stops_the_sample(self, tmp_path: Path):
        """Another NIC having counters is not the question; the requested ones are."""
        root = self.sysfs(tmp_path / "sys", with_counters=["mlx5_0"], without=["mlx5_9"])

        done = self.run(root, tmp_path / "o.csv", "mlx5_0,mlx5_9")

        assert done.returncode == 3
        assert "no counters under: mlx5_9" in done.stderr
        assert "partial set" in done.stderr

    def test_an_unrequested_adapter_is_not_this_run_s_problem(self, tmp_path: Path):
        root = self.sysfs(tmp_path / "sys", with_counters=["mlx5_0"], without=["mlx5_9"])
        out = tmp_path / "o.csv"

        assert self.run(root, out, "mlx5_0").returncode == 0

        assert "mlx5_0,1,rx_write_req,5" in out.read_text()


def test_a_wrapped_kind_is_left_out_of_the_comparison_not_shown_as_zero(tmp_path: Path):
    """A dropped delta is not a measured zero, and coverage cannot catch it: another counter on
    the same adapter moved, so both arms look equally covered."""
    from collprof.core.compare import Arm, build_counters, section_counters

    def arm(name, write_rows):
        rows = write_rows + ["1000000000,mlx5_0,1,rx_read_req,0",
                             "3000000000,mlx5_0,1,rx_read_req,40"]
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled(rows))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    healthy = arm("MoRI", ["1000000000,mlx5_0,1,rx_write_req,0",
                           "3000000000,mlx5_0,1,rx_write_req,900"])
    # The write counter wrapped on this arm; its delta is unrecoverable.
    wrapped = arm("DeepEP", ["1000000000,mlx5_0,1,rx_write_req,900",
                             "2000000000,mlx5_0,1,rx_write_req,5",
                             "3000000000,mlx5_0,1,rx_write_req,1000"])

    kinds = [row[0] for row in build_counters(healthy, wrapped, SPEC)[1]]
    assert "rx write req" not in kinds, "the wrapped kind is left out of the table"
    assert "rx read req" in kinds, "a wrap is per counter, so the rest still compare"

    text = "\n".join(section_counters(healthy, wrapped, SPEC))
    assert "**Left out: rx write req.**" in text
    assert "900" not in text


def test_an_idle_but_sampled_adapter_counts_as_covered(tmp_path: Path):
    """Coverage comes from what was sampled, not from what moved: taken from the deltas, an idle
    adapter and an unsampled one look identical."""
    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,0",
        "1000000000,mlx5_2,1,rx_write_req,7",
        "3000000000,mlx5_0,1,rx_write_req,900",
        "3000000000,mlx5_2,1,rx_write_req,7",
    ]))

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.adapters == (("mlx5_0", "1"), ("mlx5_2", "1"))
    assert series.deltas[("mlx5_2", "1", "rx_write_req")] == 0, "measured, and it stayed at zero"
    assert series.deltas[("mlx5_0", "1", "rx_write_req")] == 900


def test_everything_wrapping_is_withheld_not_reported_as_no_traffic(tmp_path: Path):
    """An empty table would say the run put nothing on the wire; the deltas were discarded."""
    from collprof.core.compare import Arm, build_counters, section_counters

    def arm(name):
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_req,900",
            "2000000000,mlx5_0,1,rx_write_req,5",
            "3000000000,mlx5_0,1,rx_write_req,1000"]))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    left, right = arm("MoRI"), arm("DeepEP")

    assert build_counters(left, right, SPEC)[1] == []
    text = "\n".join(section_counters(left, right, SPEC))
    assert "**Withheld.** Every counter that moved also wrapped" in text
    assert "no counter moved" not in text


def test_a_node_without_a_window_is_named_not_shown_as_zero(sglang_run: Path, tmp_path: Path):
    """Partial collection must not render as a measured zero row."""
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    write(sglang_run / "rdma" / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,0",
        "3000000000,mlx5_0,1,rx_write_req,900"]))
    write(sglang_run / "rdma" / "decode_NODE3.csv",
          sampled(["1000000000,mlx5_0,1,rx_write_req,12345"]))

    out = tmp_path / "out_decode"
    emit_phase(parse_run(sglang_run, SPEC)["decode"], out,
               ReportContext(spec=SPEC, run_dir=sglang_run,
                             counters=load_counters(sglang_run, SPEC)))

    # The counters section alone: NODE3 legitimately appears in the tables above it.
    section = ((out / "report.md").read_text()
               .split("## What crossed the fabric")[1].split("\n## ")[0])
    assert "| decode_NODE2 |" in section
    assert "| decode_NODE3 |" not in section, "one sample is not a row"
    assert "1 node(s) sampled no window" in section and "decode_NODE3" in section


def test_a_short_row_counts_as_damage_too(tmp_path: Path):
    """It was skipped silently, so the report could claim zero damaged rows after dropping one."""
    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,100",
        "2000000000,mlx5_0,1",                       # truncated mid-row, no value
        "3000000000,mlx5_0,1,rx_write_req,400",
    ]))

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.deltas[("mlx5_0", "1", "rx_write_req")] == 300
    assert series.damaged == 1


def test_the_portable_packet_total_is_not_added_to_its_own_subsets(tmp_path: Path):
    """`unicast_rcv_packets` is part of `port_rcv_packets`; summing both doubled the count."""
    from collprof.core.rdma_counters import by_kind

    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,port_rcv_packets,0",
        "1000000000,mlx5_0,1,unicast_rcv_packets,0",
        "3000000000,mlx5_0,1,port_rcv_packets,1000",
        "3000000000,mlx5_0,1,unicast_rcv_packets,900",
    ]))

    kinds = by_kind(parse_counters([path], LAYOUT)["decode_NODE2"], LAYOUT)

    assert sum(kinds["rx packets"].values()) == 1000, "the total, once"
    assert sum(kinds["unicast_rcv_packets"].values()) == 900, "the subset, on its own row"


def test_damage_in_an_included_series_withholds_the_comparison(tmp_path: Path):
    """A dropped row can be a counter's endpoint, so the totals are short by an unbounded
    amount."""
    from collprof.core.compare import Arm, build_counters, section_counters

    def arm(name, extra=()):
        rows = ["1000000000,mlx5_0,1,rx_write_req,0", *extra,
                "3000000000,mlx5_0,1,rx_write_req,900"]
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled(rows))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    clean = arm("MoRI")
    torn = arm("DeepEP", extra=("2000000000,mlx5_0," + "\x00" * 9000,))

    assert build_counters(clean, torn, SPEC)[1] == []
    text = "\n".join(section_counters(clean, torn, SPEC))
    assert "**Withheld.** Sampled rows were lost" in text and "DeepEP: 1 row(s)" in text


def test_arms_that_did_not_serve_the_same_requests_withhold_the_totals(tmp_path: Path):
    """Whole-window totals compare the work as much as the backend, and nothing in a
    configuration dump states a request count -- the sweep does."""
    from collprof.core.compare import Arm, build_counters, section_counters

    def arm(name, points):
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_req,0",
            "3000000000,mlx5_0,1,rx_write_req,900"]))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=points,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    swept = arm("MoRI", POINTS)
    partial = arm("DeepEP", {})

    assert build_counters(swept, partial, SPEC)[1] == []
    text = "\n".join(section_counters(swept, partial, SPEC))
    assert "not known to have served the same requests" in text
    assert "900" not in text


def test_a_counter_one_driver_omits_is_absent_not_zero(tmp_path: Path):
    """A kind one driver never exposed, initialised to zero, becomes a -100% delta -- a backend
    difference conjured out of nothing, and matching coverage does not catch it."""
    from collprof.core.compare import Arm, build_counters, section_counters

    def arm(name, rows):
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled(rows))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    both = ["1000000000,mlx5_0,1,rx_write_req,0", "3000000000,mlx5_0,1,rx_write_req,900"]
    reads = ["1000000000,mlx5_0,1,rx_read_req,0", "3000000000,mlx5_0,1,rx_read_req,40"]

    rich, poor = arm("MoRI", both + reads), arm("DeepEP", both)

    kinds = [row[0] for row in build_counters(rich, poor, SPEC)[1]]
    assert kinds == ["rx write req"], "only what both arms reported"
    text = "\n".join(section_counters(rich, poor, SPEC))
    assert "not compared** (MoRI: rx read req)" in text


def test_a_measured_zero_survives_to_the_comparison(tmp_path: Path):
    """0 reads against N reads is the contrast this channel exists to show; recording only
    non-zero deltas made the row vanish, since only kinds both arms reported are rendered."""
    from collprof.core.compare import Arm, build_counters

    def arm(name, reads):
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_req,0",
            "1000000000,mlx5_0,1,rx_read_req,0",
            "3000000000,mlx5_0,1,rx_write_req,900",
            f"3000000000,mlx5_0,1,rx_read_req,{reads}"]))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    rows = {row[0]: row for row in build_counters(arm("MoRI", 0), arm("DeepEP", 15000), SPEC)[1]}

    assert rows["rx read req"][1:3] == [0, 15000], "the zero is a measurement, not an absence"


def test_matching_benchmark_points_are_not_enough_on_their_own(tmp_path: Path):
    """The perf CSV never records a request count, so two runs can agree on every point key
    while one served twice the requests; the caller must assert what the tooling cannot check."""
    from collprof.core.compare import Arm, build_counters, section_counters

    def arm(name, asserted):
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_req,0",
            "3000000000,mlx5_0,1,rx_write_req,900"]))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=asserted, counters=parse_counters([path], LAYOUT))

    unasserted = (arm("MoRI", False), arm("DeepEP", False))
    assert build_counters(*unasserted, SPEC)[1] == []
    text = "\n".join(section_counters(*unasserted, SPEC))
    assert "necessary and not sufficient" in text
    assert "--counters-same-workload" in text

    asserted = (arm("MoRI", True), arm("DeepEP", True))
    assert build_counters(*asserted, SPEC)[1], "asserted by the caller, so the totals render"


class TestExitCodeSaysWhichFailure:
    """A host with no RDMA is supported; requested adapters that cannot be sampled are not."""

    def run(self, root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SAMPLER), "--out", str(out), "--once", *extra],
            env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
            capture_output=True, text=True, timeout=60)

    def test_requested_adapters_that_cannot_be_sampled_fail(self, tmp_path: Path):
        root = tmp_path / "sys"
        (root / "mlx5_9" / "ports" / "1" / "gids").mkdir(parents=True)

        assert self.run(root, tmp_path / "o.csv", "--devices", "mlx5_9").returncode == 3

    def test_a_host_without_rdma_is_not_a_failure(self, tmp_path: Path):
        root = tmp_path / "sys"
        (root / "mlx5_9" / "ports" / "1" / "gids").mkdir(parents=True)

        # No --devices: the same launcher runs on hosts with no adapters worth sampling.
        assert self.run(root, tmp_path / "o.csv").returncode == 0


def test_two_arms_whose_operation_counts_cannot_mean_the_same_are_not_compared(tmp_path: Path):
    """Both counts look sane alone; per byte moved they are orders of magnitude apart, because
    one backend posts traffic through a path the responder-side verb counters do not increment."""
    from collprof.core.compare import Arm, build_counters, section_counters

    def arm(name, writes, words):
        path = write(tmp_path / name / "rdma" / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_req,0",
            "1000000000,mlx5_0,1,port_rcv_data,0",
            f"3000000000,mlx5_0,1,rx_write_req,{writes}",
            f"3000000000,mlx5_0,1,port_rcv_data,{words}"]))
        return Arm(name=name, run_dir=tmp_path, config={}, steps={}, trace=None, points=POINTS,
                   counters_comparable=True,
                   counters=parse_counters([path], LAYOUT))

    # Same 4 GB either way: 300 operations on one arm, 600,000 on the other.
    coarse = arm("MoRI", 300, 1_000_000_000)
    fine = arm("DeepEP", 600_000, 1_000_000_000)

    kinds = [row[0] for row in build_counters(coarse, fine, SPEC)[1]]
    assert "rx write req" not in kinds
    assert "rx bytes" in kinds, "the link-level rows still compare"

    text = "\n".join(section_counters(coarse, fine, SPEC))
    assert "cannot be counting alike" in text and "rx write req" in text
