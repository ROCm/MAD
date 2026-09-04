"""The configuration a run reported, and the diff that decides whether two runs are comparable.

Behind these: a Kimi-K2 backend comparison whose arms also differed in graph capture and static
memory fraction, with nothing in the pipeline saying so. Such a pair must not look comparable.
"""

from __future__ import annotations

import pathlib

from conftest import coll_line, decode_batch_line, server_args_line, write

from collprof.core.rccl_log import parse_run, scan_run_config
from collprof.core.report import ReportContext, emit_phase
from collprof.core.runconfig import (diff_configs, merge_nodes,
                                     parse_config_line)
from collprof.engines.sglang_disagg import SPEC

CFG = SPEC.run_config


class TestParsing:
    def test_settings_come_off_the_startup_line(self):
        got = parse_config_line(server_args_line(), SPEC.run_config)
        assert got["disable_cuda_graph"] == "False"
        assert got["mem_fraction_static"] == "0.73"
        # Quotes are the log's, not the value's.
        assert got["attention_backend"] == "aiter"

    def test_a_line_without_the_block_yields_nothing(self):
        assert parse_config_line("worker:1:2 [0] NCCL INFO AllReduce: opCount 1f",
                                 SPEC.run_config) == {}

    def test_an_engine_declaring_no_layout_yields_nothing(self):
        from collprof.core.spec import RunConfigLayout
        assert parse_config_line(server_args_line(), RunConfigLayout()) == {}

    def test_the_run_records_config_per_phase(self, sglang_run):
        phases = parse_run(sglang_run, SPEC)
        assert set(phases["decode"].config) == {"decode_NODE2", "decode_NODE3"}
        assert phases["decode"].config["decode_NODE2"]["moe_a2a_backend"] == "mori"

    def test_only_the_first_report_per_node_is_kept(self, sglang_run):
        """A restart later in the log must not silently replace what the run started with."""
        log = sglang_run / "decode_NODE2.log"
        write(log, [server_args_line(mem_fraction_static=0.73),
                    server_args_line(mem_fraction_static=0.99),
                    decode_batch_line()])
        phases = parse_run(sglang_run, SPEC)
        assert phases["decode"].config["decode_NODE2"]["mem_fraction_static"] == "0.73"


class TestNodeAgreement:
    def test_nodes_agreeing_collapse_to_one_value(self):
        config = {"n0": {"a": "1", "b": "2"}, "n1": {"a": "1", "b": "2"}}
        agreed, disagreed = merge_nodes(config)
        assert agreed == {"a": "1", "b": "2"}
        assert disagreed == {}

    def test_a_node_out_of_step_is_reported_not_averaged(self):
        """Three nodes profiled and one not is a real failure mode; it must not average away."""
        config = {f"n{i}": {"disable_custom_all_reduce": "True"} for i in range(3)}
        config["n3"] = {"disable_custom_all_reduce": "False"}
        agreed, disagreed = merge_nodes(config)
        assert agreed == {}
        assert disagreed["disable_custom_all_reduce"]["n3"] == "False"

    def test_a_setting_missing_on_one_node_counts_as_disagreement(self):
        agreed, disagreed = merge_nodes({"n0": {"a": "1"}, "n1": {}})
        assert "a" not in agreed
        assert disagreed["a"] == {"n0": "1", "n1": "<not stated>"}, "the silent node is named"

    def test_a_node_whose_line_never_parsed_dissents_rather_than_vanishing(self):
        """Agreement judged only over the nodes that spoke calls three of four unanimous, and the
        silent node is the one most likely to differ."""
        spoke = {f"n{i}": {"disable_cuda_graph": "True"} for i in range(3)}
        agreed, disagreed = merge_nodes(spoke, expected={"n0", "n1", "n2", "n3"})

        assert "disable_cuda_graph" not in agreed
        assert disagreed["disable_cuda_graph"]["n3"] == "<not stated>"

    def test_every_expected_node_agreeing_is_still_agreement(self):
        spoke = {f"n{i}": {"disable_cuda_graph": "True"} for i in range(3)}
        agreed, _ = merge_nodes(spoke, expected={"n0", "n1", "n2"})
        assert agreed["disable_cuda_graph"] == "True"


class TestDiff:
    def test_the_graph_and_memory_pair_is_flagged_as_moving_throughput(self):
        """The exact confound from the Kimi-K2 comparison."""
        mori = {"moe_a2a_backend": "mori", "disable_cuda_graph": "False",
                "mem_fraction_static": "0.73"}
        deepep = {"moe_a2a_backend": "deepep", "disable_cuda_graph": "True",
                  "mem_fraction_static": "0.92"}
        diff = diff_configs(mori, deepep,
                            perf_relevant=CFG.perf_relevant, noise=CFG.noise)
        assert {s.key for s in diff} == set(mori)
        assert all(s.perf_relevant for s in diff)

    def test_perf_relevant_settings_sort_first(self):
        diff = diff_configs({"disable_cuda_graph": "False", "tokenizer_mode": "auto"},
                            {"disable_cuda_graph": "True", "tokenizer_mode": "slow"})
        assert [s.key for s in diff] == ["disable_cuda_graph", "tokenizer_mode"]

    def test_identical_runs_produce_no_diff(self):
        assert diff_configs({"a": "1"}, {"a": "1"},
                            perf_relevant=CFG.perf_relevant, noise=CFG.noise) == []

    def test_the_same_weights_at_another_mount_is_not_a_difference(self):
        """A remount is noise; a different model is not, and calling both noise hid the second."""
        a = {"model_path": "/mnt/a/Kimi-K2-Instruct"}
        b = {"model_path": "/shared/models/Kimi-K2-Instruct"}
        assert diff_configs(a, b, perf_relevant=CFG.perf_relevant, noise=CFG.noise,
                            path_valued=CFG.path_valued) == []

        other = {"model_path": "/mnt/a/DeepSeek-R1"}
        (setting,) = diff_configs(a, other, perf_relevant=CFG.perf_relevant, noise=CFG.noise,
                                  path_valued=CFG.path_valued)
        assert setting.key == "model_path"
        assert (setting.left, setting.right) == ("Kimi-K2-Instruct", "DeepSeek-R1")
        # And marked, not merely listed. Unmarked, the report printed the row and then said none
        # of the differences is known to move throughput -- of two different models.
        assert setting.perf_relevant, "a different model is not a throughput-neutral difference"

    def test_ports_are_dropped_by_default(self):
        """Ports differ on every pair of runs and mean nothing; `model_path` is not in that
        class."""
        left = {"port": "30000", "host": "n0"}
        right = {"port": "30001", "host": "n1"}
        assert diff_configs(left, right, perf_relevant=CFG.perf_relevant, noise=CFG.noise,
                            path_valued=CFG.path_valued) == []
        noisy = diff_configs(left, right, include_noise=True,
                             perf_relevant=CFG.perf_relevant, noise=CFG.noise,
                             path_valued=CFG.path_valued)
        assert {s.key for s in noisy} == {"port", "host"}

    def test_a_setting_is_never_both_noise_and_path_valued(self):
        """The two are exclusive: a key in both is silently dropped by whichever check runs
        first."""
        assert not (CFG.noise & CFG.path_valued)

    def test_a_setting_present_on_one_side_only_still_differs(self):
        diff = diff_configs({"deepep_mode": "normal"}, {},
                            perf_relevant=CFG.perf_relevant, noise=CFG.noise)
        assert diff[0].left == "normal" and diff[0].right is None

    def test_noise_and_perf_relevant_do_not_overlap(self):
        """A key in both lists would be dropped before it could be flagged."""
        from collprof.engines.sglang_disagg import NOISE, PERF_RELEVANT
        assert not (NOISE & PERF_RELEVANT)


class TestReferenceScan:
    def test_the_reference_is_read_without_parsing_its_collectives(self, sglang_run):
        found, nodes = scan_run_config(sglang_run, SPEC)
        assert set(found) == {"prefill", "decode"}
        assert found["prefill"]["prefill_NODE0"]["moe_a2a_backend"] == "mori"
        assert nodes["prefill"] >= {"prefill_NODE0"}

    def test_a_node_that_stated_nothing_is_still_reported_as_discovered(self, tmp_path):
        """Returning only the nodes that spoke lets three of four count as unanimous."""
        write(tmp_path / "job" / "decode_NODE0.log", [server_args_line()])
        write(tmp_path / "job" / "decode_NODE1.log", ["nothing to see here"])

        found, nodes = scan_run_config(tmp_path / "job", SPEC)

        assert set(found["decode"]) == {"decode_NODE0"}, "only one node stated anything"
        assert nodes["decode"] == {"decode_NODE0", "decode_NODE1"}, "both were asked"

    def test_a_run_that_reports_nothing_scans_empty(self, tmp_path):
        write(tmp_path / "job" / "decode_NODE0.log", ["nothing to see here"])
        found, _nodes = scan_run_config(tmp_path / "job", SPEC)
        assert found == {}

    def test_per_rank_rccl_logs_are_not_read(self, sglang_run, monkeypatch):
        """They cannot carry the configuration, so scanning them reads a 2 GB file for nothing."""
        write(sglang_run / "rccl" / "decode_NODE2.host.123.log", [coll_line()])
        opened = []
        real = pathlib.Path.open

        def spy(self, *a, **kw):
            opened.append(self.name)
            return real(self, *a, **kw)

        monkeypatch.setattr(pathlib.Path, "open", spy)
        scan_run_config(sglang_run, SPEC, sglang_run)
        assert not any("123" in name for name in opened), opened


class TestMeasurementAssumptions:
    """The scope note claims a measurement configuration; the config says whether it held.

    The first real run contradicted it: `disable_custom_all_reduce` was False while the note said
    the TP exchange went through RCCL, and the phase logged only the 8 B startup barrier.
    """

    def report(self, sglang_run, out, **overrides):
        for node in (2, 3):
            write(sglang_run / f"decode_NODE{node}.log",
                  [server_args_line(**overrides), coll_line(count=128, grank=0, pid=2000),
                   decode_batch_line()])
        phase = parse_run(sglang_run, SPEC)["decode"]
        return emit_phase(phase, out, ReportContext(spec=SPEC, run_dir=sglang_run)).read_text()

    def test_a_contradicted_assumption_is_called_out(self, sglang_run, tmp_path):
        text = self.report(sglang_run, tmp_path / "out",
                           disable_custom_all_reduce="False", disable_cuda_graph="True")
        assert "### The scope note above does not hold for this run" in text
        assert "| `disable_custom_all_reduce` | `True` | `False` |" in text
        assert "a floor rather than the phase's traffic" in text

    def test_a_run_that_holds_to_it_says_nothing(self, sglang_run, tmp_path):
        text = self.report(sglang_run, tmp_path / "out",
                           disable_custom_all_reduce="True", disable_cuda_graph="True")
        assert "does not hold for this run" not in text

    def test_a_setting_the_run_never_reported_is_not_invented(self, sglang_run, tmp_path):
        """Absent is not the same as contradicting; an older engine simply does not say."""
        text = self.report(sglang_run, tmp_path / "out", disable_cuda_graph="True")
        # A conjunction, not a disjunction: the old `A or B` stayed green even when the *other*
        # setting was wrongly contradicted.
        assert "does not hold" not in text
        assert "disable_custom_all_reduce" not in text.split("## Step time")[0]

    def test_an_engine_declaring_no_assumptions_is_unaffected(self, sglang_run, tmp_path):
        import dataclasses
        spec = dataclasses.replace(SPEC, measurement_assumptions=())
        for node in (2, 3):
            write(sglang_run / f"decode_NODE{node}.log",
                  [server_args_line(disable_custom_all_reduce="False"),
                   coll_line(count=128, grank=0, pid=2000), decode_batch_line()])
        out = tmp_path / "out"
        text = emit_phase(parse_run(sglang_run, spec)["decode"], out,
                          ReportContext(spec=spec, run_dir=sglang_run)).read_text()
        assert "does not hold for this run" not in text


class TestSecretsAndListValues:
    def test_a_credential_never_reaches_the_parsed_configuration(self):
        """`run_config.csv` carries every setting stated, and reports are shared artifacts."""
        line = server_args_line(api_key="'sk-do-not-leak'")
        parsed = parse_config_line(line, SPEC.run_config)
        assert parsed["api_key"] == "<redacted>"
        assert "do-not-leak" not in str(parsed)

    def test_redaction_is_not_one_flag_away_from_being_off(self):
        """Secrets are redacted at parse time, so no render-time switch can restore them."""
        from collprof.engines.sglang_disagg import NOISE, SECRET
        assert not (SECRET & NOISE), "a secret in NOISE is re-enabled by include_noise"

    def test_a_list_valued_setting_survives_its_own_commas(self):
        """`cuda_graph_bs=[1, 2, 4]` contains the field separator; stopping at the first comma
        reduced every such list to `[1`, so different graph batch sizes compared equal."""
        a = parse_config_line(server_args_line(cuda_graph_bs="[1, 2, 4]"), SPEC.run_config)
        b = parse_config_line(server_args_line(cuda_graph_bs="[1, 8, 16]"), SPEC.run_config)
        assert a["cuda_graph_bs"] == "[1, 2, 4]"
        assert a != b, "two different graph configurations must not compare equal"
        assert a["port"] == "30000", "the field after the list is still parsed"


class TestMeasurementCheckSeesDissent:
    def test_a_node_contradicting_the_assumption_is_reported(self, tmp_path):
        """A dissenting node puts the key in `disagreed`, where a check reading only agreed
        values saw nothing."""
        from collprof.core.report import ReportContext, section_measurement_check

        ctx = ReportContext(spec=SPEC, run_dir=tmp_path)
        agreed = {"disable_cuda_graph": "True"}
        disagreed = {"disable_custom_all_reduce": {"n0": "True", "n1": "True", "n2": "False"}}

        text = "\n".join(section_measurement_check(agreed, ctx, disagreed))

        assert "does not hold" in text
        assert "n2=`False`" in text
        assert "on 1 of 3 nodes" in text

    def test_a_node_that_stated_nothing_is_not_a_contradiction(self, tmp_path):
        """Silence is not dissent; only an explicitly wrong value is."""
        from collprof.core.report import ReportContext, section_measurement_check

        ctx = ReportContext(spec=SPEC, run_dir=tmp_path)
        disagreed = {"disable_custom_all_reduce": {"n0": "True", "n1": "<not stated>"}}

        assert section_measurement_check({}, ctx, disagreed) == []


class TestConfigCapableNodes:
    def test_a_per_rank_log_does_not_invent_a_node_that_stated_nothing(self, sglang_run):
        """`Phase.nodes` includes per-rank RCCL files, which cannot carry a configuration line;
        judging agreement over them made a leftover log dissent on every setting."""
        write(sglang_run / "rccl" / "decode_NODE9.host.7.log", [coll_line()])

        phases = parse_run(sglang_run, SPEC, sglang_run)
        decode = phases["decode"]

        assert "decode_NODE9" in decode.nodes, "the per-rank file is still discovered"
        assert "decode_NODE9" not in decode.config_nodes, "but it cannot carry a configuration"

        _agreed, disagreed = merge_nodes(decode.config, decode.config_nodes)
        assert not any("decode_NODE9" in spread for spread in disagreed.values())


def test_a_marker_based_engine_records_the_nodes_of_the_phase_it_opened(tmp_path):
    """`config_nodes` was filled only for file-name phases, so a marker-based engine left it empty
    and `merge_nodes` fell back to the nodes whose startup line happened to parse."""
    import dataclasses

    from collprof.engines import primus

    spec = dataclasses.replace(primus.SPEC, run_config=SPEC.run_config)
    run = tmp_path / "job"
    for node in (0, 1):
        lines = ["[INFO] [main] Executing: bash runner/primus-cli-direct.sh -- train pretrain "
                 "--config examples/megatron/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml"]
        # Only node 0 states a configuration; node 1's startup line never parsed.
        if node == 0:
            lines.append(server_args_line(tp_size="8"))
        lines.append(coll_line(count=2048, grank=node, pid=3000 + node))
        write(run / f"node_{node}" / "stdout.out", lines)

    phase = parse_run(run, spec)["BF16"]

    assert phase.config_nodes == {"node_0", "node_1"}
    _agreed, disagreed = merge_nodes(phase.config, phase.config_nodes)
    assert disagreed["tp_size"] == {"node_0": "8", "node_1": "<not stated>"}


def test_every_setting_a_run_splits_on_is_named_not_only_the_throughput_ones():
    """A split key is absent from the diff and the agreed rows, so filtering to throughput-relevant
    keys here dropped the rest out of the artifact entirely."""
    from collprof.core.report import section_config_diff

    ctx = ReportContext(spec=SPEC, run_dir=pathlib.Path("/tmp"), config_diff=[],
                        config_diff_source="other-run",
                        config_diff_split={"this run": {"tp_size": {"n0": "8", "n1": "4"},
                                                        "served_model_name": {"n0": "a",
                                                                              "n1": "b"}}})

    text = "\n".join(section_config_diff(ctx))

    assert "`tp_size` **(moves throughput)**" in text
    assert "`served_model_name`" in text and "served_model_name` **(moves" not in text
    assert "2 setting(s) its own nodes disagree on" in text
    # The verdict stays withheld: one of the two does move throughput.
    assert "The two runs are comparable" not in text


def test_a_split_the_engine_has_not_classified_still_withholds_the_verdict():
    """A split key is absent from the merged configuration, so an empty diff is empty partly
    *because* of the split; gating the verdict on throughput-relevance hid that blind spot."""
    from collprof.core.report import section_config_diff

    ctx = ReportContext(spec=SPEC, run_dir=pathlib.Path("/tmp"), config_diff=[],
                        config_diff_source="other-run",
                        config_diff_split={
                            "this run": {"served_model_name": {"n0": "a", "n1": "b"}}})

    text = "\n".join(section_config_diff(ctx))

    assert "served_model_name` **(moves" not in text, "nothing here is throughput-relevant"
    assert "The two runs are comparable" not in text
    assert "The split above still stands" in text


def test_a_marker_based_engine_can_be_a_comparison_reference(tmp_path):
    """`scan_run_config` skipped the marker layout that `parse_run` handled, so `--compare-config`
    read nothing from an engine that announces its phases in the log."""
    import dataclasses

    from collprof.engines import primus

    spec = dataclasses.replace(primus.SPEC, run_config=SPEC.run_config)
    run = tmp_path / "job"
    for node in (0, 1):
        write(run / f"node_{node}" / "stdout.out", [
            "[INFO] [main] Executing: bash runner/primus-cli-direct.sh -- train pretrain "
            "--config examples/megatron/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml",
            server_args_line(tp_size="8"),
            coll_line(count=2048, grank=node, pid=3000 + node),
        ])

    found, seen = scan_run_config(run, spec)

    assert set(found) == {"BF16"}, "the phase the marker announced"
    assert found["BF16"]["node_0"]["tp_size"] == "8"
    assert seen["BF16"] == {"node_0", "node_1"}


def test_a_phase_a_whole_run_past_the_first_is_still_read(tmp_path):
    """A marker-based log holds several phases back to back, and the second marker lies a whole
    phase past the first, so the startup line budget dropped every phase after the first."""
    import dataclasses

    from collprof.core.rccl_log import CONFIG_SCAN_LINES
    from collprof.engines import primus

    spec = dataclasses.replace(primus.SPEC, run_config=SPEC.run_config)
    run = tmp_path / "job"
    cli = ("[INFO] [main] Executing: bash runner/primus-cli-direct.sh -- train pretrain "
           "--config examples/megatron/configs/MI355X/llama3.1_70B-{}-pretrain.yaml")
    write(run / "node_0" / "stdout.out",
          [cli.format("BF16"), server_args_line(tp_size="8")]
          # The first phase's own iterations, which is what separates the two markers.
          + [f"[INFO] iteration {i}/500000" for i in range(CONFIG_SCAN_LINES + 100)]
          + [cli.format("FP8"), server_args_line(tp_size="4")])

    found, seen = scan_run_config(run, spec)

    assert set(found) == {"BF16", "FP8"}, "both phases the log announced"
    assert found["FP8"]["node_0"]["tp_size"] == "4"
    assert seen["FP8"] == {"node_0"}


def test_a_log_that_names_its_phase_is_still_given_up_on(tmp_path, capsys):
    """The budget that marker-based logs are now exempt from still bounds the ones it was written
    for: a phase from the file name is stated near the top, so reading to EOF buys nothing."""
    from collprof.core.rccl_log import CONFIG_SCAN_LINES

    run = tmp_path / "job"
    write(run / "decode_NODE2.log",
          [f"[INFO] warming up {i}" for i in range(CONFIG_SCAN_LINES + 10)]
          + [server_args_line(tp_size="8")])

    found, _seen = scan_run_config(run, SPEC)

    assert found == {}, "the budget stopped the read before the line that states it"
    assert "not reading further" in capsys.readouterr().out


def test_a_reference_that_said_nothing_is_not_a_reference_that_was_absent():
    """A phase discovered in the reference but stating no readable configuration had its own
    diagnostic collapsed into 'that phase is absent', which sends the reader to the wrong run."""
    from collprof.cli import no_config_reason

    assert "absent from the reference run" in no_config_reason("prefill", {}, {})
    assert "none of them stated a configuration" in no_config_reason(
        "prefill", {}, {"prefill": {"n0", "n1"}})
    assert "2 node(s)" in no_config_reason("prefill", {}, {"prefill": {"n0", "n1"}})
    assert "this run states none" in no_config_reason("prefill", {"prefill": {}}, {})
