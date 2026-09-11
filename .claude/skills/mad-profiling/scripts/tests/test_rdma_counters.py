"""The adapter-counter channel: the sampler that writes it and the parser that reads it.

The sampler is bash reading sysfs, and it takes its root from `RDMA_SYSFS_ROOT`, so these tests
run it for real against a directory of files.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from conftest import server_args_line, write

from collprof.core.rdma_counters import by_kind, parse_counters
from collprof.engines.sglang_disagg import SPEC

LAYOUT = SPEC.counters
#: Asserted, so a wrong path fails here rather than as a broken-looking sampler.
SAMPLER = (Path(__file__).resolve().parents[5] / "scripts" / "sglang_disagg"
           / "rdma_counters.sh")
assert SAMPLER.is_file(), f"sampler not found at {SAMPLER}"

#: The schema before the host column. Files in this form exist from finished runs, so
#: the parser must keep reading them; most fixtures here are deliberately in it.
HEADER_V1 = "epoch_ns,device,port,counter,value"
HEADER = "epoch_ns,host,device,port,counter,value"

#: The comparison refuses to render unless both arms measured the same benchmark points, so every
#: arm built here carries the same ones.
POINTS = {(1024, 1024, 64): {"mean_itl_ms": 226.0}}


def sampled(rows: list) -> list:
    """Rows in the pre-host schema, which the parser still has to read."""
    return [HEADER_V1] + rows


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
        """A counter that went backwards has its delta dropped and the counter named."""
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
        """An unclassified counter keeps its own name instead of being pooled with others."""
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

    def test_a_counter_without_a_trailing_newline_is_still_read(self, tmp_path: Path):
        """A counter file that ends without a trailing newline is still read."""
        root = tmp_path / "sys"
        port = root / "bnxt_re0" / "ports" / "1" / "hw_counters"
        port.mkdir(parents=True)
        (port / "rx_write_req").write_text("4096")          # no trailing newline
        (port / "rx_read_req").write_text("7\n")
        out = tmp_path / "o.csv"

        assert self.run(root, out).returncode == 0

        text = out.read_text()
        assert ",rx_write_req,4096" in text
        assert ",rx_read_req,7" in text

    def test_a_requeue_clears_the_previous_attempt_s_failure_marker(self, tmp_path: Path):
        """A sampling loop replaces the window, so it clears the previous failure marker."""
        root = self.fake_sysfs(tmp_path / "sys", {"rx_write_req": 10})
        out = tmp_path / "decode_NODE2.csv"
        marker = tmp_path / "decode_NODE2.csv.failed"
        marker.write_text("sampler exited 4\n")

        # Loop mode owns and truncates its file, and runs until signalled.
        proc = subprocess.Popen(["bash", str(SAMPLER), "--out", str(out), "--interval", "1"],
                                env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                if out.exists() and not marker.exists():
                    break
                time.sleep(0.05)
        finally:
            proc.terminate()
            proc.wait(timeout=20)

        assert not marker.exists(), "a fresh window is not described by the old failure"
        assert out.read_text().startswith("epoch_ns,")

    def test_the_started_marker_waits_for_a_whole_sample(self, tmp_path: Path):
        """Rows arrive one echo at a time, so a non-empty file only means the baseline has begun;
        a caller that starts its server then gives late counters a late baseline."""
        root = tmp_path / "sys"
        port = root / "bnxt_re0" / "ports" / "1" / "hw_counters"
        port.mkdir(parents=True)
        for i in range(40):
            (port / f"c{i}").write_text(f"{i}\n")
        out = tmp_path / "decode_NODE2.csv"
        started = tmp_path / "decode_NODE2.csv.started"

        proc = subprocess.Popen(["bash", str(SAMPLER), "--out", str(out), "--interval", "30"],
                                env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(200):
                if started.exists():
                    break
                time.sleep(0.05)
            assert started.exists(), "no marker"
            assert len(out.read_text().splitlines()) == 41, "header plus every counter"
        finally:
            proc.terminate()
            proc.wait(timeout=20)

    def test_once_keeps_the_marker_because_it_appends(self, tmp_path: Path):
        """`--once` appends to the existing window, so it keeps the failure marker."""
        root = self.fake_sysfs(tmp_path / "sys", {"rx_write_req": 10})
        out = tmp_path / "decode_NODE2.csv"
        marker = tmp_path / "decode_NODE2.csv.failed"
        marker.write_text("sampler exited 4\n")

        assert self.run(root, out).returncode == 0

        assert marker.exists()

    def test_an_option_given_last_reports_its_own_error(self, tmp_path: Path):
        """An option given last with no value exits 2 saying which option, not with a shell
        error."""
        for flag in ("--out", "--interval", "--devices"):
            done = subprocess.run(["bash", str(SAMPLER), flag],
                                  env={"PATH": "/usr/bin:/bin"},
                                  capture_output=True, text=True, timeout=60)
            assert done.returncode == 2, f"{flag}: got {done.returncode}\n{done.stderr}"
            assert f"{flag} needs a value" in done.stderr, flag

    def test_an_unwritable_output_is_a_failure_not_a_silent_empty_sample(self, tmp_path: Path):
        """An output that cannot be written exits non-zero rather than reporting success with no
        data."""
        root = self.fake_sysfs(tmp_path / "sys", {"rx_write_req": 10})
        # A parent that is a file, so the write fails for any user -- including root, which is
        # what this sampler runs as inside the container.
        (tmp_path / "notadir").write_text("")
        out = tmp_path / "notadir" / "decode_NODE2.csv"

        done = self.run(root, out)

        assert done.returncode != 0, "collecting nothing is not success"
        assert "cannot write" in done.stderr
        assert not out.exists()

    def test_the_sample_records_which_machine_it_came_from(self, tmp_path: Path):
        """A logical label like decode_NODE2 comes from the file name, so nothing in the data said
        which machine produced it."""
        import socket

        root = self.fake_sysfs(tmp_path / "sys", {"rx_write_req": 10})
        out = tmp_path / "decode_NODE2.csv"

        assert self.run(root, out).returncode == 0

        lines = out.read_text().splitlines()
        assert lines[0] == "epoch_ns,host,device,port,counter,value"
        assert lines[1].split(",")[1] == socket.gethostname()

        series = parse_counters([out], LAYOUT)["decode_NODE2"]
        assert series.hosts == (socket.gethostname(),)

    def test_a_file_written_before_the_host_column_still_parses(self, tmp_path: Path):
        """Finished runs are stored in the old schema; dropping them would rewrite history."""
        path = write(tmp_path / "decode_NODE2.csv", sampled([
            "1000000000,mlx5_0,1,rx_write_req,0",
            "3000000000,mlx5_0,1,rx_write_req,900"]))

        series = parse_counters([path], LAYOUT)["decode_NODE2"]

        assert series.deltas[("mlx5_0", "1", "rx_write_req")] == 900
        assert series.hosts == (), "not recorded is not the same as a host called empty"

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
        """Only the adapters `--devices` names are sampled."""
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
        """The comparison puts the arms side by side as totals, not as rates."""
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


def test_undecodable_bytes_are_damage_not_a_smaller_number(tmp_path: Path):
    """`errors="ignore"` dropped the bad byte and left `1\xff2` reading as 12 -- a value nobody
    measured, counted as real, with nothing in `damaged` for the gate to refuse on."""
    path = tmp_path / "decode_NODE2.csv"
    path.write_bytes(HEADER_V1.encode() + b"\n"
                     + b"1000000000,mlx5_0,1,rx_write_req,100\n"
                     + b"2000000000,mlx5_0,1,rx_write_req,1\xff2\n"
                     + b"3000000000,mlx5_0,1,rx_write_req,700\n")

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.damaged == 1, "the undecodable line is counted"
    assert series.deltas[("mlx5_0", "1", "rx_write_req")] == 600, "100 to 700, no invented 12"


def test_a_line_of_nul_bytes_is_skipped_rather_than_ending_the_parse(tmp_path: Path):
    """A line of NUL bytes is counted as damage and skipped, not raised on."""
    path = tmp_path / "decode_NODE2.csv"
    path.write_text(HEADER_V1 + "\n"
                    + "1000000000,mlx5_0,1,rx_write_requests,100\n"
                    + "2000000000,mlx5_0," + "\x00" * 200000 + "\n"
                    + "3000000000,mlx5_0,1,rx_write_requests,700\n")

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.deltas[("mlx5_0", "1", "rx_write_requests")] == 600
    assert series.damaged == 1


def test_the_ib_data_counters_are_words_not_bytes(tmp_path: Path):
    """`port_rcv_data` is in 4-octet words and is scaled to bytes; operation counts are not."""
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
    """A node absent from `Phase.nodes` still has its adapters reported."""
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
    """Counters are filtered by the launcher's naming rule, so a node whose log never landed is
    still reported and counted."""
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

    text = (out / "report.md").read_text()
    assert "| decode_NODE7 |" in text
    # And counted in the coverage header, not only in the fabric table.
    assert "decode_NODE7" in text.split("## ")[0], "the coverage header names it too"


def test_a_node_the_sampler_never_wrote_a_file_for_is_named(sglang_run: Path, tmp_path: Path):
    """A node of the phase the sampler wrote no file for is named in the report."""
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    # decode_NODE3 logs and configures like its pair, and the sampler left nothing for it.
    write(sglang_run / "rdma" / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,0",
        "3000000000,mlx5_0,1,rx_write_req,900"]))

    out = tmp_path / "out_decode"
    emit_phase(parse_run(sglang_run, SPEC)["decode"], out,
               ReportContext(spec=SPEC, run_dir=sglang_run,
                             counters=load_counters(sglang_run, SPEC)))

    text = (out / "report.md").read_text()
    assert "1 node(s) of this phase have no counter file at all" in text
    assert "decode_NODE3" in text.split("What crossed the fabric")[1]


def test_the_failure_marker_the_launcher_writes_is_read_back(tmp_path: Path):
    """The `.failed` marker the launcher writes is read back onto the series."""
    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,0",
        "3000000000,mlx5_0,1,rx_write_req,900"]))
    (tmp_path / "decode_NODE2.csv.failed").write_text("sampler exited 4\n")

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.sampler_failed
    assert series.deltas, "the prefix it did write is still real"


def test_when_every_counter_wrapped_no_headless_table_is_rendered(sglang_run: Path,
                                                                  tmp_path: Path):
    """Wrapped counters are dropped, so a node with a window can have no columns left; the table
    was still emitted with an empty final column instead of saying nothing survived."""
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    write(sglang_run / "rdma" / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,900",
        "3000000000,mlx5_0,1,rx_write_req,10"]))          # went backwards: wrapped

    out = tmp_path / "out_decode"
    emit_phase(parse_run(sglang_run, SPEC)["decode"], out,
               ReportContext(spec=SPEC, run_dir=sglang_run,
                             counters=load_counters(sglang_run, SPEC)))

    text = (out / "report.md").read_text()
    assert "| node | samples | window s |  |" not in text, "a table with no columns"
    assert "no counter survived it" in text
    assert "went backwards were dropped" in text


def test_a_decode_report_carries_only_decode_nodes_in_its_csv(sglang_run: Path, tmp_path: Path):
    """A phase's counter CSV carries only that phase's nodes, as the markdown does."""
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
    """A node sampled with no deltas advertises no `fabric_counters.csv`, since none is
    written."""
    from collprof.cli import load_counters
    from collprof.core.rccl_log import parse_run
    from collprof.core.report import ReportContext, emit_phase

    # What the sampler leaves on a host with no RDMA device: it ran, and there is nothing in it.
    write(sglang_run / "rdma" / "decode_NODE2.csv", [HEADER_V1])

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
    """The sampler refuses an interval that is not a positive whole number."""

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
    """A counter that wraps and then climbs back past its start is still caught."""
    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,mlx5_0,1,rx_write_req,900",
        "2000000000,mlx5_0,1,rx_write_req,5",
        "3000000000,mlx5_0,1,rx_write_req,1000",
    ]))

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.deltas == {}, "a key that ever fell is not counted"
    assert series.wrapped == (("mlx5_0", "1", "rx_write_req"),)


def test_a_row_with_no_adapter_identity_is_damage_not_traffic(tmp_path: Path):
    """Numbers that parse do not make a counter: `...,,,rx_write_req,100` contributed under an
    empty adapter and was missing from the damaged tally."""
    path = write(tmp_path / "decode_NODE2.csv", sampled([
        "1000000000,,,rx_write_req,0",
        "2000000000,,,rx_write_req,100",
        "1000000000,mlx5_0,1,rx_write_req,0",
        "2000000000,mlx5_0,1,rx_write_req,7",
    ]))

    series = parse_counters([path], LAYOUT)["decode_NODE2"]

    assert series.deltas == {("mlx5_0", "1", "rx_write_req"): 7}
    assert ("", "") not in series.adapters
    assert series.damaged == 2, "both identity-less rows are counted as damage"


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
        """Equal numbers of different adapters withhold the totals, and each arm's own are
        named."""
        from collprof.core.compare import build_counters, section_counters

        left = self.arm("MoRI", tmp_path, [2], devices=("mlx5_0", "mlx5_2"))
        right = self.arm("DeepEP", tmp_path, [2], devices=("mlx5_0", "mlx5_4"))

        assert build_counters(left, right, SPEC)[1] == []
        text = "\n".join(section_counters(left, right, SPEC))
        assert "**Withheld.**" in text
        assert "MoRI only: decode_NODE2/mlx5_2" in text
        assert "DeepEP only: decode_NODE2/mlx5_4" in text

    def test_an_empty_table_does_not_claim_the_fabric_was_idle(self, tmp_path: Path):
        """A table emptied by one-sided movement withholds rather than claiming an idle
        fabric."""
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

    def test_a_node_that_produced_no_window_withholds_even_when_both_arms_lost_it(
            self, tmp_path: Path):
        """A node that sampled without producing a window withholds, even when both arms lost
        the same one."""
        from collprof.core.compare import build_counters, section_counters
        from collprof.core.rdma_counters import CounterSeries

        left = self.arm("MoRI", tmp_path, [2])
        right = self.arm("DeepEP", tmp_path, [2])
        for arm in (left, right):
            # The second node of each arm sampled once: a header only, or one cumulative reading.
            arm.counters["decode_NODE3"] = CounterSeries(
                node="decode_NODE3", deltas={}, seconds=0.0, samples=1,
                adapters=(("mlx5_0", "1"),))

        assert build_counters(left, right, SPEC)[1] == []
        text = "\n".join(section_counters(left, right, SPEC))
        assert "**Withheld.**" in text
        assert "sampled without producing a window" in text
        assert "MoRI: decode_NODE3" in text and "DeepEP: decode_NODE3" in text

    def test_a_sampler_that_could_not_finish_withholds(self, tmp_path: Path):
        """A `.failed` marker on either arm withholds the totals and names the node."""
        import dataclasses

        from collprof.core.compare import build_counters, section_counters

        left = self.arm("MoRI", tmp_path, [2])
        right = self.arm("DeepEP", tmp_path, [2])
        left.counters["decode_NODE2"] = dataclasses.replace(
            left.counters["decode_NODE2"], sampler_failed=True)

        assert build_counters(left, right, SPEC)[1] == []
        text = "\n".join(section_counters(left, right, SPEC))
        assert "**Withheld.**" in text
        assert "could not finish" in text and "MoRI: decode_NODE2" in text

    def test_a_kind_reported_from_different_adapters_is_left_out(self, tmp_path: Path):
        """Equal adapter coverage does not make each kind's coverage equal: the sampler skips a
        counter it cannot read, so one arm can carry a kind on one NIC fewer."""
        from collprof.core.compare import build_counters, section_counters

        left = self.arm("MoRI", tmp_path, [2], devices=("mlx5_0", "mlx5_2"), bytes_too=True)
        right = self.arm("DeepEP", tmp_path, [2], devices=("mlx5_0", "mlx5_2"), bytes_too=True)
        # One arm's second NIC never reported this kind -- unreadable at sample time. The byte
        # rows stay evenly covered, so the table still renders and the uneven kind is named.
        del right.counters["decode_NODE2"].deltas[("mlx5_2", "1", "rx_write_req")]

        rows = build_counters(left, right, SPEC)[1]
        assert [r[0] for r in rows] == ["rx bytes"], "the evenly covered kind only"
        text = "\n".join(section_counters(left, right, SPEC))
        assert "Left out: rx write req" in text
        assert "not from the same adapters" in text

    def test_matching_coverage_renders_the_table(self, tmp_path: Path):
        from collprof.core.compare import section_counters

        left = self.arm("MoRI", tmp_path, [2, 3], devices=("mlx5_0", "mlx5_2"))
        right = self.arm("DeepEP", tmp_path, [2, 3], devices=("mlx5_0", "mlx5_2"))

        text = "\n".join(section_counters(left, right, SPEC))

        assert "Withheld" not in text
        assert "MoRI: 2 node(s), 4 adapter(s)" in text
        assert "| rx write req | 400 | 400 |" in text

    def test_the_table_is_not_headed_operations(self, tmp_path: Path):
        """The table's column is headed `counter kind`, since the rows carry mixed units."""
        from collprof.core.compare import section_counters

        left = self.arm("MoRI", tmp_path, [2], devices=("mlx5_0",), bytes_too=True)
        right = self.arm("DeepEP", tmp_path, [2], devices=("mlx5_0",), bytes_too=True)

        text = "\n".join(section_counters(left, right, SPEC))

        assert "| operations |" not in text
        assert "| counter kind |" in text
        assert "| rx bytes |" in text, "a byte volume sits under that heading"


class TestRequestedAdaptersAreChecked:
    """Each requested adapter is checked for counters, not just the node as a whole."""

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

    def test_device_lists_from_several_transports_are_merged(self, tmp_path: Path):
        """Each transport names its own adapters (IB_DEVICES, MORI_RDMA_DEVICES, NCCL_IB_HCA);
        sampling one list leaves the others' traffic out of a total that looks whole."""
        root = self.sysfs(tmp_path / "sys", with_counters=["mlx5_0", "mlx5_2"], without=[])
        out = tmp_path / "o.csv"

        done = subprocess.run(
            ["bash", str(SAMPLER), "--out", str(out), "--once",
             "--devices", "mlx5_0", "--devices", "mlx5_2:1", "--devices", "mlx5_0"],
            env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
            capture_output=True, text=True, timeout=60)

        assert done.returncode == 0, done.stderr
        sampled = {line.split(",")[2] for line in out.read_text().splitlines()[1:]}
        assert sampled == {"mlx5_0", "mlx5_2"}, "merged, port suffix dropped, duplicate collapsed"

    def test_ncclx_exact_match_prefix_is_the_same_adapter(self, tmp_path: Path):
        """`NCCL_IB_HCA="=mlx5_0:1"` names one device exactly. Left on, `=mlx5_0` matches no
        sysfs entry and, beside the unprefixed name from another list, made the set look partial
        -- so the sampler refused all of it."""
        root = self.sysfs(tmp_path / "sys", with_counters=["mlx5_0", "mlx5_2"], without=[])
        out = tmp_path / "o.csv"

        done = subprocess.run(
            ["bash", str(SAMPLER), "--out", str(out), "--once",
             "--devices", "mlx5_0,mlx5_2", "--devices", "=mlx5_0:1,=mlx5_2:1"],
            env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
            capture_output=True, text=True, timeout=60)

        assert done.returncode == 0, done.stderr
        sampled = {line.split(",")[2] for line in out.read_text().splitlines()[1:]}
        assert sampled == {"mlx5_0", "mlx5_2"}

    def test_an_exclusion_list_is_refused_rather_than_guessed(self, tmp_path: Path):
        """`NCCL_IB_HCA` accepts `^mlx5_0`, which names what to avoid; resolving it wrongly would
        sample the wrong adapters and report a plausible total."""
        root = self.sysfs(tmp_path / "sys", with_counters=["mlx5_0"], without=[])

        done = subprocess.run(
            ["bash", str(SAMPLER), "--out", str(tmp_path / "o.csv"), "--once",
             "--devices", "^mlx5_0"],
            env={"PATH": "/usr/bin:/bin", "RDMA_SYSFS_ROOT": str(root)},
            capture_output=True, text=True, timeout=60)

        assert done.returncode == 2
        assert "exclusion list" in done.stderr

    def test_an_empty_counter_directory_is_not_a_usable_adapter(self, tmp_path: Path):
        """The check globbed the directory, so one that exists and holds nothing passed and the
        sampler exited 0 with no rows -- a failed measurement reported as a node that sent
        nothing."""
        root = self.sysfs(tmp_path / "sys", with_counters=["mlx5_0"], without=[])
        (root / "mlx5_9" / "ports" / "1" / "hw_counters").mkdir(parents=True)

        done = self.run(root, tmp_path / "o.csv", "mlx5_0,mlx5_9")

        assert done.returncode == 3
        assert "no counters under: mlx5_9" in done.stderr

    def test_requested_adapters_on_a_host_with_no_rdma_is_a_failure(self, tmp_path: Path):
        """The empty-sysfs exit jumped over the requested-device check, so asking for an adapter
        that is not there exited 0 with a header only -- the same file a node that sent nothing
        leaves, and no marker for the report to withhold on."""
        empty = tmp_path / "sys"
        empty.mkdir()

        done = self.run(empty, tmp_path / "o.csv", "mlx5_0")

        assert done.returncode == 3
        assert "nothing to sample" in done.stderr

    def test_an_unrequested_adapter_is_not_this_run_s_problem(self, tmp_path: Path):
        root = self.sysfs(tmp_path / "sys", with_counters=["mlx5_0"], without=["mlx5_9"])
        out = tmp_path / "o.csv"

        assert self.run(root, out, "mlx5_0").returncode == 0

        assert "mlx5_0,1,rx_write_req,5" in out.read_text()


def test_a_wrapped_kind_is_left_out_of_the_comparison_not_shown_as_zero(tmp_path: Path):
    """A kind whose delta was dropped for a wrap is left out of the comparison, not shown as
    zero."""
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
    """Coverage comes from what was sampled, so an idle adapter still counts as covered."""
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
    """A row too short to parse is counted as damage, not skipped silently."""
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
    """Arms not known to have served the same requests withhold the totals."""
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
    """A kind only one arm's driver exposes is left out of the comparison, not read as zero."""
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
    """A delta measured as zero reaches the comparison as a value, not as an absence."""
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
    """Matching benchmark points alone do not release the totals; the caller must assert the
    workload with `--counters-same-workload`."""
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
    """Operation counts orders of magnitude apart per byte moved are not compared; the
    link-level rows still are."""
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


def test_every_comparability_precondition_can_refuse_and_says_why(tmp_path: Path):
    """The preconditions are data, so the suite can check the whole list rather than the branches
    someone remembered to test: each must be able to block, and to name itself when it does."""
    from collprof.core.compare import COMPARABILITY, build_counters, comparability_block

    assert len({name for name, _ in COMPARABILITY}) == len(COMPARABILITY), "names are distinct"

    arms = TestUnequalCoverageWithholds()
    for name, check in COMPARABILITY:
        left = arms.arm("MoRI", tmp_path / name.replace(" ", "_") / "l", [2])
        right = arms.arm("DeepEP", tmp_path / name.replace(" ", "_") / "r", [2])
        assert check(left, right, SPEC) is None, f"{name} blocks a comparable pair"

    # And a pair that trips one of them is refused by the table as well as named by the section.
    left = arms.arm("MoRI", tmp_path / "bad" / "l", [2])
    right = arms.arm("DeepEP", tmp_path / "bad" / "r", [2])
    right.counters.pop("decode_NODE2")

    blocked = comparability_block(left, right, SPEC)
    assert blocked is not None
    assert blocked[0] == "both arms collected"
    assert blocked[1].startswith("**Withheld.**")
    assert build_counters(left, right, SPEC)[1] == []
