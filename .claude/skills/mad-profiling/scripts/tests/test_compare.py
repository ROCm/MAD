"""The comparison's two renderings must not be able to disagree.

Every table is built once as numbers and rendered into both the markdown and a CSV, so these
tests read the markdown back and check it against the CSV rows rather than trusting the builder.
"""

from __future__ import annotations

from pathlib import Path

from collprof.core.compare import (Arm, build, build_config_diff, build_decomposition,
                                   build_kernels, build_points, build_stages,
                                   build_steps, load_benchmark,
                                   section_a2a, section_comparability,
                                   section_steps, tables, variant_of)
from collprof.core.steps import StepRecord
from collprof.core.steps import summarise as summarise_steps
from collprof.engines.sglang_disagg import SPEC


def arm(name: str, ttft: float, itl: float, e2e: float, osl: int = 1024) -> Arm:
    """One arm with a single benchmark point, which is all the split needs."""
    return Arm(name=name, run_dir=Path(f"/tmp/{name}"), config={}, steps={}, trace=None,
               points={(1024, osl, 64): {"mean_ttft_ms": ttft, "mean_itl_ms": itl,
                                         "mean_e2e_latency_ms": e2e,
                                         "total_token_throughput_tok_s": 1000.0}})


def md_rows(text: str, heading: str) -> list:
    """The data rows of the markdown table under a heading, as lists of cells."""
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = heading in line
            continue
        if inside and line.startswith("|") and not set(line) <= set("|-: "):
            out.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return out[1:]


def test_the_split_is_an_identity_not_a_fit():
    """Residual is the check: build the right arm so the identity holds exactly."""
    osl, itl_delta, ttft_delta = 1024, 10.0, 500.0
    left = arm("A", ttft=5000.0, itl=200.0, e2e=5000.0 + 200.0 * (osl - 1))
    right = arm("B", ttft=5000.0 + ttft_delta, itl=200.0 + itl_delta,
                e2e=5000.0 + ttft_delta + (200.0 + itl_delta) * (osl - 1))

    (row,) = build_decomposition(left, right, SPEC)[1]
    _isl, _osl, _con, e2e, ttft, decode, residual, share, itl = row

    assert residual == 0.0, "the identity holds exactly, so the residual must be zero"
    assert (ttft, itl) == (ttft_delta, itl_delta)
    assert decode == itl_delta * (osl - 1)
    assert e2e == ttft + decode
    assert share == round(100.0 * decode / e2e, 1)


def test_the_csv_and_the_markdown_quote_the_same_numbers():
    left = arm("A", ttft=5000.0, itl=200.0, e2e=209_800.0)
    right = arm("B", ttft=5600.0, itl=214.0, e2e=224_522.0)

    text = build(left, right, SPEC, "decode")
    _header, rows = tables(left, right, SPEC)["decomposition"]

    rendered = md_rows(text, "Where the difference comes from")
    assert len(rendered) == len(rows) == 1
    cells, row = rendered[0], rows[0]
    # The markdown rounds to whole milliseconds; the CSV keeps a decimal.
    assert float(cells[2].replace("+", "")) == round(row[3])
    assert float(cells[3].replace("+", "")) == round(row[4])
    assert float(cells[4].replace("+", "")) == round(row[5])


def test_a_table_with_no_rows_is_left_out_rather_than_written_empty():
    """An empty CSV becomes an empty sheet, which reads as a measurement of nothing."""
    bare = Arm(name="A", run_dir=Path("/tmp/a"), config={}, steps={}, trace=None, points={})
    other = Arm(name="B", run_dir=Path("/tmp/b"), config={}, steps={}, trace=None, points={})

    produced = tables(bare, other, SPEC)

    assert "decomposition" not in produced
    assert "benchmark_points" not in produced
    assert all(rows for _header, rows in produced.values())


def traced(name: str, a2a: dict, category_us: dict) -> Arm:
    return Arm(name=name, run_dir=Path(f"/tmp/{name}"), config={}, steps={},
               trace={"files": 1, "a2a": a2a, "category_us": category_us, "a2a_names": {}},
               points={})


def test_a_stage_in_two_categories_is_summed_not_overwritten():
    """`(stage, category)` collapsed to `stage` let one category silently replace the other, so
    the loser's calls vanished and the surviving row depended on dict order."""
    left = traced("A", {("dispatch", "kernel"): [10, 700.0],
                        ("dispatch", "cpu_op"): [10, 300.0]},
                  {"kernel": 1000.0, "cpu_op": 1000.0})
    right = traced("B", {("dispatch", "kernel"): [4, 400.0]}, {"kernel": 1000.0})

    (row,) = build_stages(left, right)[1]

    assert row[0] == "dispatch"
    assert row[1] == 20, "both categories' calls are kept"
    assert row[2] == 50.0, "1000 us of 2000 us across the categories actually summed"
    assert (row[3], row[4]) == (4, 40.0), "the other arm is unaffected"


def test_an_arms_share_does_not_move_with_what_the_other_arm_classified():
    """The denominator was the union of both arms' categories, so B classifying a `cpu_op` pulled
    A's unclassified host time into A's divisor and dropped A's share from 100% to 10%."""
    left = traced("A", {("dispatch", "kernel"): [10, 1000.0]},
                  {"kernel": 1000.0, "cpu_op": 9000.0})
    right = traced("B", {("dispatch", "kernel"): [10, 500.0], ("dispatch", "cpu_op"): [10, 100.0]},
                   {"kernel": 1000.0, "cpu_op": 1000.0})
    right_kernel_only = traced("B", {("dispatch", "kernel"): [10, 500.0]},
                               {"kernel": 1000.0, "cpu_op": 1000.0})

    with_cpu_op = build_stages(left, right)[1][0]
    without = build_stages(left, right_kernel_only)[1][0]

    assert with_cpu_op[2] == without[2] == 100.0


def test_shares_over_different_categories_are_flagged_as_different_quantities():
    """Each share is now of its own arm's categories, which is only comparable when they match."""
    left = traced("A", {("dispatch", "kernel"): [10, 1000.0]},
                  {"kernel": 1000.0, "cpu_op": 9000.0})
    right = traced("B", {("dispatch", "cpu_op"): [10, 900.0]},
                   {"kernel": 1000.0, "cpu_op": 1000.0})

    from collprof.core.compare import section_a2a

    text = "\n".join(section_a2a(left, right, SPEC))

    assert "not of the same quantity" in text
    assert "A classified kernel" in text and "B classified cpu_op" in text


def test_the_summary_can_only_name_variants_the_table_shows():
    """The variant patterns overlap, so a summary collecting every match would name variants
    absent from the first-match-wins table -- and call two arms different over one kernel."""
    name = "EpDispatchIntraNodeKernelLowLatency_fp8"
    every = {label for label, pattern in SPEC.a2a.variants if pattern.search(name)}

    assert len(every) > 1, "the fixture must be a kernel more than one pattern claims"
    assert variant_of(name, SPEC) == "intranode low latency"
    assert variant_of(name, SPEC) in every


def test_a_split_arm_does_not_report_the_other_arm_as_split():
    """The marker went into both columns, claiming a split the other arm does not have."""
    left = Arm(name="A", run_dir=Path("/tmp/a"), config={"disable_cuda_graph": "True"},
               steps={}, trace=None, points={},
               disagreed={"disable_cuda_graph": {"n0": "True", "n1": "False"}})
    right = Arm(name="B", run_dir=Path("/tmp/b"), config={"disable_cuda_graph": "True"},
                steps={}, trace=None, points={})

    rows = [r for r in build_config_diff(left, right, SPEC)[1]
            if r[0] == "disable_cuda_graph"]

    assert len(rows) == 1, "one row per key, not one per arm"
    assert "n0=True" in rows[0][1] and "n1=False" in rows[0][1]
    assert rows[0][2] == "True", "the arm that agreed shows its value, not a split marker"


def test_no_step_channel_produces_no_section_and_no_table():
    """An engine with no StepTimingLayout must not be credited with accounting it never kept."""
    bare = Arm(name="A", run_dir=Path("/tmp/a"), config={}, steps={}, trace=None, points={})
    other = Arm(name="B", run_dir=Path("/tmp/b"), config={}, steps={}, trace=None, points={})

    assert build_steps(bare, other)[1] == []
    assert "step_times" not in tables(bare, other, SPEC)
    assert "## Step time" not in build(bare, other, SPEC, "decode")


def test_the_markdown_and_the_csv_list_the_same_settings():
    """Both come from `build_config_diff`; recomputing the diff is how they drifted apart."""
    left = Arm(name="A", run_dir=Path("/tmp/a"),
               config={"moe_a2a_backend": "mori", "disable_cuda_graph": "True"},
               steps={}, trace=None, points={},
               disagreed={"mem_fraction_static": {"n0": "0.9", "n1": "0.7"}})
    right = Arm(name="B", run_dir=Path("/tmp/b"),
                config={"moe_a2a_backend": "deepep", "disable_cuda_graph": "True"},
                steps={}, trace=None, points={})

    csv_keys = {r[0] for r in build_config_diff(left, right, SPEC)[1]}
    text = "\n".join(section_comparability(left, right, SPEC))
    md_keys = {line.split("`")[1] for line in text.splitlines()
               if line.startswith("| `")}

    assert csv_keys == md_keys, "the two renderings must list the same settings"
    assert "mem_fraction_static" in csv_keys, "a split key belongs in both, not only the CSV"


def test_a_setting_reported_as_an_empty_string_is_not_reported_as_absent():
    """`key=''` is a value a run really stated; "not stated" must not collapse onto it."""
    left = Arm(name="A", run_dir=Path("/tmp/a"), config={"tool_call_parser": ""},
               steps={}, trace=None, points={})
    right = Arm(name="B", run_dir=Path("/tmp/b"), config={}, steps={}, trace=None, points={})

    (row,) = [r for r in build_config_diff(left, right, SPEC)[1] if r[0] == "tool_call_parser"]

    assert row[1] == "", "the arm that stated an empty value keeps it"
    assert row[2] == "<not stated>", "the arm that never mentioned it is marked, not blanked"


def test_an_engine_with_no_a2a_patterns_gets_no_exchange_section():
    """Without a2a patterns there is nothing to classify, and the section's prose about backends
    carrying their own transport is false for a training engine."""
    from dataclasses import replace
    from collprof.core.spec import A2AKernels
    from collprof.core.compare import section_a2a

    trace = {"files": 1, "a2a": {}, "a2a_names": {}, "category_us": {"kernel": 10.0},
             "unmatched_us": {"SomeKernel": 9.0}}
    arms = [Arm(name=n, run_dir=Path("/tmp"), config={}, steps={}, trace=trace, points={})
            for n in ("A", "B")]

    assert section_a2a(*arms, replace(SPEC, a2a=A2AKernels())) == []
    assert section_a2a(*arms, SPEC) != [], "an engine that declares patterns still reports"


def test_two_arms_without_configuration_get_no_comparability_verdict():
    """"Nothing was checked" and "no setting differs" cannot both be true of the same pair."""
    arms = [Arm(name=n, run_dir=Path("/tmp"), config={}, steps={}, trace=None, points={})
            for n in ("A", "B")]

    text = "\n".join(section_comparability(*arms, SPEC))

    assert "Neither arm reported a configuration" in text
    assert "No setting differs" not in text
    # Two silent arms are not the one-sided case: there is no side to read rows from.
    assert "one-sided" not in text


def test_one_silent_arm_says_which_arm_was_silent():
    """The one-sided wording has to name the silent arm, or a reader reads the rows backwards."""
    arms = [Arm(name="A", run_dir=Path("/tmp"), config={"tp_size": "8"}, steps={}, trace=None,
                points={}),
            Arm(name="B", run_dir=Path("/tmp"), config={}, steps={}, trace=None, points={})]

    text = "\n".join(section_comparability(*arms, SPEC))

    assert "B reported no configuration" in text
    assert "one-sided" in text
    assert "No setting differs" not in text


def test_speculative_decoding_withholds_the_step_channel_from_both_arms():
    """`batch / rate` is not a step duration when a step emits a variable number of tokens, so
    the channel is withheld for the pair and not only for the arm that enabled speculation."""
    stats = summarise_steps([StepRecord(batch=64, rate=280.0)] * 8)
    arms = [Arm(name=n, run_dir=Path("/tmp"), config={}, steps={"NODE0": stats}, trace=None,
                points={}, pooled=stats)
            for n in ("A", "B")]
    arms[1].steps_invalid = (("speculative_algorithm", "EAGLE", "tokens vary per step"),)

    assert build_steps(*arms)[1] == []
    text = "\n".join(section_steps(*arms))
    assert "Withheld" in text and "B reported `speculative_algorithm=EAGLE`" in text
    assert "median" not in text


def test_an_arm_whose_nodes_disagree_has_not_reported_no_configuration():
    """`Arm.config` is only what the nodes *agreed*, which one silent node empties -- announced
    as "reported no configuration" directly above a table of that arm's settings."""
    from collprof.core.runconfig import merge_nodes

    stated = {f"n{i}": {"tp_size": "8", "moe_a2a_backend": "mori"} for i in range(3)}
    agreed, disagreed = merge_nodes(stated, {"n0", "n1", "n2", "n3"})
    left = Arm(name="MoRI", run_dir=Path("/tmp"), config=agreed, steps={}, trace=None, points={},
               disagreed=disagreed)
    right = Arm(name="DeepEP", run_dir=Path("/tmp"), steps={}, trace=None, points={},
                config={"tp_size": "8", "moe_a2a_backend": "deepep"})

    text = "\n".join(section_comparability(left, right, SPEC))

    assert "reported no configuration" not in text
    assert "did not agree on" in text, "the real finding, which the false headline replaced"


def test_a_kernel_that_ran_is_not_reported_as_zero_calls():
    """`round(0.5)` is 0 in Python, so one call across two traces read as never having run."""
    arm = Arm(name="A", run_dir=Path("/tmp"), config={}, steps={}, points={},
              trace={"files": 2, "a2a": {}, "category_us": {},
                     "a2a_names": {("dispatch", "EpDispatchInterNodeV1Kernel_fp8"): [1, 900.0]}})

    (row,) = build_kernels(arm, SPEC)[1]

    assert row[4] == 0.5


class TestLoadBenchmark:
    """The only I/O path feeding the decomposition, which every other test stubs out."""

    HEADER = "model,performance,metric\n"

    def test_a_csv_without_the_metric_column_is_skipped_not_raised(self, tmp_path: Path):
        (tmp_path / "perf_a.csv").write_text("model,performance\n"
                                             "2p2d_isl1024_osl1024_con64,0.26\n")
        assert load_benchmark(tmp_path, SPEC.benchmark) == {}

    def test_points_are_keyed_by_the_shape_in_the_model_column(self, tmp_path: Path):
        (tmp_path / "perf_a.csv").write_text(
            self.HEADER + "2p2d_isl1024_osl1024_con64,226.3,mean_itl_ms\n"
                          "unrelated_row,1.0,mean_itl_ms\n")
        assert (load_benchmark(tmp_path, SPEC.benchmark)
                == {(1024, 1024, 64): {"mean_itl_ms": 226.3}})

    def test_a_repeated_measurement_keeps_the_first_and_says_so(self, tmp_path: Path, capsys):
        """Last-wins put one of two values in the report with nothing saying the other existed."""
        (tmp_path / "perf_a.csv").write_text(
            self.HEADER + "2p2d_isl1024_osl1024_con64,226.3,mean_itl_ms\n")
        (tmp_path / "perf_b.csv").write_text(
            self.HEADER + "2p2d_isl1024_osl1024_con64,240.7,mean_itl_ms\n")

        points = load_benchmark(tmp_path, SPEC.benchmark)

        assert points == {(1024, 1024, 64): {"mean_itl_ms": 226.3}}
        warning = capsys.readouterr().out
        assert "reported twice" in warning and "226.3" in warning and "240.7" in warning


def test_out_and_out_dir_cannot_both_be_given(capsys):
    """`--out-dir` silently won, so a command naming a file got no file and no complaint."""
    import pytest

    from collprof.compare_cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--left", "a", "--right", "b",
                                   "--out", "x.md", "--out-dir", "d"])
    assert "not allowed with" in capsys.readouterr().err


def test_an_engine_that_declares_no_benchmark_layout_loses_only_those_sections():
    """`core/` must not assume a perf-CSV schema: an engine naming its columns differently used
    to get sglang's names applied to its CSV, and now simply gets no benchmark rows."""
    from dataclasses import replace

    from collprof.core.spec import BenchmarkLayout

    bare = replace(SPEC, benchmark=BenchmarkLayout())
    points = {(1024, 1024, 64): {"mean_e2e_latency_ms": 1.0, "mean_ttft_ms": 1.0,
                                 "mean_itl_ms": 1.0}}
    arms = [Arm(name=n, run_dir=Path("/tmp"), config={}, steps={}, trace=None, points=points)
            for n in ("A", "B")]

    assert load_benchmark(Path("/tmp"), bare.benchmark) == {}
    assert build_points(*arms, bare)[1] == []
    assert build_decomposition(*arms, bare)[1] == []
    # The same arms with the engine's own names declared do produce the split.
    assert build_decomposition(*arms, SPEC)[1] != []


def test_a_capture_directory_with_no_traces_is_reported_not_dropped():
    """One directory per replica, so a missing one leaves an arm measured over fewer replicas."""
    from collprof.core.compare import section_a2a

    left = traced("A", {("dispatch", "kernel"): [10, 1000.0]}, {"kernel": 1000.0})
    left.empty_trace_dirs = ("1787540802.3030503",)
    right = traced("B", {("dispatch", "kernel"): [10, 1000.0]}, {"kernel": 1000.0})

    text = "\n".join(section_a2a(left, right, SPEC))

    assert "A had 1 capture directory that held no trace file" in text
    assert "1787540802.3030503" in text
    assert "B had" not in text


class TestOutputDirectoryOwnership:
    """`--out-dir` belongs to the caller, so what may be deleted must be proven, not guessed."""

    ARGS = ["--left", "L", "--right", "R", "--out-dir"]

    def run_into(self, out_dir, monkeypatch, tables):
        """One `--out-dir` write with a stubbed comparison; returns the files left behind."""
        from collprof import compare_cli

        monkeypatch.setattr(compare_cli, "parse_side", lambda *a, **k: ({"decode": None}, {}, {}))
        monkeypatch.setattr(compare_cli, "make_arm",
                            lambda name, run_dir, *a, **k: Arm(name=name, run_dir=run_dir,
                                                               config={}, steps={}, trace=None,
                                                               points={}))
        monkeypatch.setattr(compare_cli, "build", lambda *a, **k: "# comparison\n")
        monkeypatch.setattr(compare_cli, "tables", lambda *a, **k: tables)
        monkeypatch.setattr(compare_cli, "write_workbook", lambda *a, **k: None)
        monkeypatch.setattr(compare_cli.engines, "detect",
                            lambda run: (SPEC, "stub"))
        compare_cli.main([*self.ARGS, str(out_dir), "--no-traces",
                          "--left-name", "A", "--right-name", "B"])
        return {p.name for p in out_dir.iterdir()}

    def test_a_csv_this_command_did_not_write_is_left_alone(self, tmp_path, monkeypatch):
        """A stranger's `step_times.csv` was unlinked whenever this run produced no step table."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "step_times.csv").write_text("someone else's work\n")

        left = self.run_into(out, monkeypatch, {"decomposition": (["a"], [[1]])})

        assert "step_times.csv" in left
        assert (out / "step_times.csv").read_text() == "someone else's work\n"

    def test_a_table_this_command_wrote_last_time_is_swept(self, tmp_path, monkeypatch):
        """What it wrote it may remove: a stale sheet otherwise outlives its own numbers."""
        out = tmp_path / "out"
        out.mkdir()

        self.run_into(out, monkeypatch, {"step_times": (["a"], [[1]])})
        assert (out / "step_times.csv").exists()

        left = self.run_into(out, monkeypatch, {"decomposition": (["a"], [[1]])})

        assert "step_times.csv" not in left
        assert "decomposition.csv" in left


def test_a_variant_below_the_display_cut_still_triggers_the_mismatch_warning():
    """Reading the mixed-variant verdict off the eight rendered rows let a ninth, smaller kernel
    of another variant pass as a single-variant arm."""
    def kernels(prefix, extra=None):
        named = {("dispatch", f"{prefix}_{i}_normal"): [10, 1000.0 - i] for i in range(8)}
        if extra:
            named[("dispatch", extra)] = [1, 1.0]
        return named

    stage = {("dispatch", "kernel"): [10, 1000.0]}
    left = Arm(name="A", run_dir=Path("/tmp"), config={}, steps={}, points={},
               trace={"files": 1, "a2a": stage, "category_us": {"kernel": 1000.0},
                      "a2a_names": kernels("EpDispatchInterNodeV1Kernel")})
    right = Arm(name="B", run_dir=Path("/tmp"), config={}, steps={}, points={},
                trace={"files": 1, "a2a": stage, "category_us": {"kernel": 1000.0},
                       "a2a_names": kernels("EpDispatchInterNodeV1Kernel",
                                            extra="EpDispatchIntraNodeKernel_bf16")})

    assert len(build_kernels(right, SPEC)[1]) == 9, "the CSV carries every classified kernel"
    text = "\n".join(section_a2a(left, right, SPEC))

    assert "The arms ran different variants of the exchange" in text
    assert "below the display cut" in text
    assert "EpDispatchIntraNodeKernel_bf16" not in text, "still only eight rows are rendered"


def test_batch_sensitivity_is_not_reported_as_a_broken_estimate():
    """The check is one-sided: only a flat spread is evidence about the derivation, because a
    volume-limited workload's step time grows with its batch for unrelated reasons."""
    stats = summarise_steps([StepRecord(batch=64, rate=280.0)] * 8)
    left = Arm(name="A", run_dir=Path("/tmp"), config={}, steps={"NODE0": stats}, trace=None,
               points={}, pooled=stats, batch_spread_ms=40.0)
    right = Arm(name="B", run_dir=Path("/tmp"), config={}, steps={"NODE0": stats}, trace=None,
                points={}, pooled=stats, batch_spread_ms=0.5)

    text = "\n".join(section_steps(left, right))

    assert "A: the median step time varies by 40.00 ms across its batch groups" in text
    assert "unreliable" not in text
    assert "**batch-sensitive**" in text and "cannot resolve that on its own" in text
    assert "B: the median step time varies by 0.50 ms" in text and "which is flat against" in text


def test_an_uncaptured_arm_is_named_before_any_classification_claim():
    """The "no event name matched" sentence spoke for both arms, which is false for an arm that
    had no trace at all, and no missing-capture warning corrected it."""
    from collprof.core.compare import section_a2a

    captured = Arm(name="MoRI", run_dir=Path("/tmp"), config={}, steps={}, points={},
                   trace={"files": 1, "a2a": {}, "category_us": {"kernel": 100.0},
                          "a2a_names": {}, "unmatched_us": {"some_gemm": 100.0}})
    missing = Arm(name="DeepEP", run_dir=Path("/tmp"), config={}, steps={}, points={}, trace=None)

    text = "\n".join(section_a2a(captured, missing, SPEC))

    assert "**DeepEP has no trace for this phase**" in text
    assert "No event name in MoRI matched" in text, "the claim names only the captured arm"
    assert "in either arm matched" not in text


class TestOutputDirectoryIsNotOverwritten:
    """`--out-dir` is shareable, so a write needs the same ownership test a delete does."""

    def run_into(self, out_dir, monkeypatch, tables_out):
        from collprof import compare_cli

        monkeypatch.setattr(compare_cli, "parse_side", lambda *a, **k: ({"decode": None}, {}, {}))
        monkeypatch.setattr(compare_cli, "make_arm",
                            lambda name, run_dir, *a, **k: Arm(name=name, run_dir=run_dir,
                                                               config={}, steps={}, trace=None,
                                                               points={}))
        monkeypatch.setattr(compare_cli, "build", lambda *a, **k: "# comparison\n")
        monkeypatch.setattr(compare_cli, "tables", lambda *a, **k: tables_out)
        monkeypatch.setattr(compare_cli, "write_workbook", lambda *a, **k: None)
        monkeypatch.setattr(compare_cli.engines, "detect", lambda run: (SPEC, "stub"))
        compare_cli.main(["--left", "L", "--right", "R", "--out-dir", str(out_dir),
                          "--no-traces", "--left-name", "A", "--right-name", "B"])

    def test_a_file_this_command_did_not_write_is_not_replaced(self, tmp_path, monkeypatch):
        import pytest

        out = tmp_path / "out"
        out.mkdir()
        (out / "comparison.md").write_text("somebody else's comparison\n")

        with pytest.raises(SystemExit) as raised:
            self.run_into(out, monkeypatch, {"decomposition": (["a"], [[1]])})

        assert "comparison.md" in str(raised.value)
        assert (out / "comparison.md").read_text() == "somebody else's comparison\n"

    def test_its_own_output_is_replaced_without_complaint(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()

        self.run_into(out, monkeypatch, {"decomposition": (["a"], [[1]])})
        self.run_into(out, monkeypatch, {"decomposition": (["a"], [[2]])})

        assert "2" in (out / "decomposition.csv").read_text()
