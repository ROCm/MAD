"""Trace parsing and, above all, mapping traces onto phases without a human reading timestamps."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import trace_event, write

from collprof.core.spec import TraceLayout
from collprof.core.torch_trace import canonical_collective, parse_traces, rank_of
from collprof.engines import primus, sglang_disagg

SGL = sglang_disagg.SPEC.traces


def trace_file(path: Path, events: list) -> Path:
    return write(path, ['{"traceEvents": ['] + events + [']}'])


def test_only_the_kernel_event_is_counted(tmp_path: Path):
    """The same args block is attached to the host-side cpu_op; counting both doubles everything."""
    trace_file(tmp_path / "1000.0-TP-0.trace.json",
               [trace_event(cat="cpu_op"), trace_event(cat="kernel")])
    parsed = parse_traces([tmp_path], SGL)
    (calls, _dur), = parsed["events"].values()
    assert calls == 1


def test_size_is_the_larger_of_in_and_out_times_the_dtype_width(tmp_path: Path):
    trace_file(tmp_path / "1000.0-TP-0.trace.json", [trace_event(nin=512, nout=4096)])
    (key, _v), = parse_traces([tmp_path], SGL)["events"].items()
    coll, pg, total_bytes, dtype = key
    assert (coll, pg, total_bytes, dtype) == ("AllReduce", "tp:device", 4096 * 2, "BFloat16")


def test_group_size_and_step_count_are_picked_up(tmp_path: Path):
    trace_file(tmp_path / "1000.0-TP-0.trace.json",
               ['      {"name": "ProfilerStep#7", "ph": "X"},',
                '      {"name": "ProfilerStep#8", "ph": "X"},',
                trace_event(group=16)])
    parsed = parse_traces([tmp_path], SGL)
    assert parsed["group_size"] == 16
    assert parsed["steps"] == 2


def test_a_trace_without_steps_reports_none_rather_than_one(tmp_path: Path):
    """Serving traces carry no ProfilerStep, and calling a window an iteration overstates rates."""
    trace_file(tmp_path / "1000.0-TP-0.trace.json", [trace_event()])
    assert parse_traces([tmp_path], SGL)["steps"] == 0


@pytest.mark.parametrize("name, layout, expected", [
    ("1786745993.59-TP-3.trace.json.gz", SGL, 3),
    ("llama_rank[5]_step.pt.trace.json", primus.SPEC.traces, 5),
    ("no-rank-here.trace.json", SGL, -1),
])
def test_rank_is_read_from_either_naming_convention(name: str, layout: TraceLayout, expected: int):
    assert rank_of(name, layout) == expected


@pytest.mark.parametrize("raw, canonical", [
    ("nccl:all_reduce", "AllReduce"), ("allgather", "AllGather"),
    ("reduce_scatter_tensor", "ReduceScatter"), ("nccl:broadcast", "Broadcast"),
    ("all_to_all_single", "AllToAll"), ("something_else", "something_else"),
])
def test_torch_collective_names_map_onto_rccl_names(raw: str, canonical: str):
    assert canonical_collective(raw) == canonical


def test_a_gzipped_trace_is_read(tmp_path: Path):
    write(tmp_path / "1000.0-TP-0.trace.json.gz", ['{"traceEvents": [', trace_event(), ']}'],
          compress=True)
    assert parse_traces([tmp_path], SGL)["files"] == 1


def add_profile_point(run: Path, role: str, epochs: list) -> None:
    """A profile-point log as bench_serving writes it: the output_dir it asked each worker for."""
    lines = [f"INFO: profiling {role} worker(s)"]
    for epoch in epochs:
        trace_file(run / "torchprof" / epoch / f"{epoch}-TP-0.trace.json", [trace_event()])
        lines.append("async_request_profile api_url='http://host:3000/start_profile' "
                     "body={'activities': ['CPU', 'GPU'], "
                     f"'output_dir': '/run_logs/25999/torchprof/{epoch}'}}")
    write(next(run.glob(f"*_PROFILE_{role}.log")), lines)


def test_serving_traces_are_matched_to_every_node_of_their_role(sglang_run: Path):
    """Every node of every role, not the one directory a human would have pasted into a script."""
    add_profile_point(sglang_run, "prefill", ["1000.100000", "1000.200000"])
    add_profile_point(sglang_run, "decode", ["2000.100000", "2000.200000"])

    found = sglang_disagg.resolve_traces(sglang_run)
    assert sorted(found) == ["decode", "prefill"]
    assert [p.name for p in found["prefill"]] == ["1000.100000", "1000.200000"]
    assert [p.name for p in found["decode"]] == ["2000.100000", "2000.200000"]


def test_role_matching_ignores_timestamps(sglang_run: Path):
    """The directory names come from the container's clock, the mtimes from the filesystem's, and on
    this cluster the two sit about 460 seconds apart -- enough to swap two roles."""
    import os
    add_profile_point(sglang_run, "prefill", ["5000.100000"])
    add_profile_point(sglang_run, "decode", ["1000.100000"])
    for path in list(sglang_run.glob("torchprof/*")) + list(sglang_run.glob("*_PROFILE_*.log")):
        os.utime(path, (1, 1))

    found = sglang_disagg.resolve_traces(sglang_run)
    assert [p.name for p in found["prefill"]] == ["5000.100000"]
    assert [p.name for p in found["decode"]] == ["1000.100000"]


def test_an_unmappable_trace_layout_fails_loudly(sglang_run: Path):
    """Silently labelling a decode trace as prefill would poison a whole report."""
    for epoch in ("3000.100000", "3000.200000"):
        trace_file(sglang_run / "torchprof" / epoch / f"{epoch}-TP-0.trace.json", [trace_event()])

    with pytest.raises(ValueError, match="cannot map trace directories"):
        sglang_disagg.resolve_traces(sglang_run)


def test_training_traces_are_matched_to_the_datatype_in_their_path(tmp_path: Path):
    for dtype in ("BF16", "FP8"):
        (tmp_path / f"prof/llama3.1_70B-{dtype}-pretrain/tensorboard").mkdir(parents=True)
    found = primus.resolve_traces(tmp_path)
    assert sorted(found) == ["BF16", "FP8"]
    assert found["BF16"][0].name == "tensorboard"


def test_trace_discovery_says_what_it_looked_for(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=re.escape("*.trace.json")):
        parse_traces([tmp_path / "empty"], SGL)
