#!/usr/bin/env python3
"""run.sh end to end against a stub primus-cli that records its command line.

The stub stands in for Primus: it writes the arguments and selected environment
variables to files and a one-line Megatron training log, so these tests check the
command run.sh builds (order matters: Primus keeps the last value of a key) and the
perf CSV it leaves behind, without a GPU.
"""

import csv
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1]
HELPERS = ("run.sh", "scaleout_batch.py", "extract_primus_perf.py", "model_overrides.py")

STUB_CLI = """\
#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_ARGS"
printf 'NCCL_DEBUG_FILE=%s\\n' "${NCCL_DEBUG_FILE-<unset>}" > "$STUB_ENV"
printf 'NCCL_DEBUG=%s\\n' "${NCCL_DEBUG-<unset>}" >> "$STUB_ENV"
printf 'PWD=%s\\n' "$PWD" >> "$STUB_ENV"
log=""
prev=""
for a in "$@"; do
  [[ "$prev" == "--log_file" ]] && log="$a"
  prev="$a"
done
mkdir -p "$(dirname "$log")"
: > "$log"
# STUB_LOG picks what the training log holds: Megatron's printed throughput (default), a
# Megatron-Bridge iteration line without TPS, or nothing the extractor can use.
case "${STUB_LOG:-megatron}" in
  bridge)
    printf '%s\\n' ' iteration 10/200 | elapsed time per iteration (ms): 200.0 | throughput per GPU (TFLOP/s/GPU): 50.0 | global batch size: 16 |' >> "$log"
    exit 0 ;;
  none)
    echo "training finished" >> "$log"
    exit 0 ;;
esac
[[ -n "${STUB_NO_ARG_DUMP:-}" ]] || cat >> "$log" <<'EOF'
  micro_batch_size ................................ 2
  seq_length ...................................... 8192
  world_size ...................................... 16
EOF
cat >> "$log" <<'EOF'
 iteration       10/      10 | elapsed time per iteration (ms): 1000.0 | compute per GPU (TFLOP/s/GPU): 500.0 (avg 499.0) | tokens/s/GPU inst/harmonic mean: 1800.0/1799.0 | global batch size:    32 |
EOF
"""


def make_tree(tmp_path, overrides, framework="megatron", name="toy-BF16-pretrain",
              config_root="checkout"):
    scripts = tmp_path / "scripts"
    train = scripts / "primus_train"
    train.mkdir(parents=True)
    for helper in HELPERS:
        if (SRC / helper).exists():
            shutil.copy(SRC / helper, train / helper)
    primus = scripts / "Primus"
    primus.mkdir(parents=True)
    # config_root="overlay": the config exists only in a second Primus tree (the RCCL overlay's
    # /workspace/Primus), not in the scripts/Primus checkout that run.sh finds first.
    cfg_root = primus if config_root == "checkout" else tmp_path / "overlay" / "Primus"
    rel = f"examples/{framework}/configs/MI355X/{name}.yaml"
    cfg = cfg_root / rel
    cfg.parent.mkdir(parents=True)
    cfg.write_text(yaml.safe_dump(
        {"modules": {"pre_trainer": {"framework": framework, "overrides": overrides}}}
    ))
    for root in {primus, cfg_root}:
        cli = root / "primus-cli"
        cli.write_text(STUB_CLI)
        cli.chmod(0o755)
    run_dir = tmp_path / "run_directory"
    run_dir.mkdir()
    # Something for an accidental glob in run.sh to expand to.
    (run_dir / "decoy.txt").write_text("")
    return train / "run.sh", rel, run_dir


def run(tmp_path, overrides, *args, env=None, framework="megatron", name="toy-BF16-pretrain",
        expect_ok=True, config_flag=None, config_root="checkout"):
    run_sh, rel, run_dir = make_tree(tmp_path, overrides, framework, name, config_root)
    base_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("PRIMUS_", "RCCL_", "NCCL_")) and k not in ("NNODES", "NODE_RANK")
    }
    base_env.update({
        "STUB_ARGS": str(tmp_path / "args.txt"),
        "STUB_ENV": str(tmp_path / "env.txt"),
        "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx950",
        "MAD_SYSTEM_GPU_PRODUCT_NAME": "AMD Instinct MI355X",
        "NNODES": "2",
        "NODE_RANK": "1",
        "GPUS_PER_NODE": "8",
    })
    base_env.update(env or {})
    # config_flag="equals" passes --config_path=<rel> as one token instead of two.
    config_args = [f"--config_path={rel}"] if config_flag == "equals" else ["--config_path", rel]
    proc = subprocess.run(
        ["bash", str(run_sh), *config_args, *args],
        cwd=run_dir, env=base_env, capture_output=True, text=True,
    )
    if not expect_ok:
        assert proc.returncode != 0, proc.stdout + proc.stderr
        return None, None, proc, run_dir
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cli_args = (tmp_path / "args.txt").read_text().split("\n")[:-1]
    cli_env = dict(
        line.split("=", 1) for line in (tmp_path / "env.txt").read_text().splitlines()
    )
    return cli_args, cli_env, proc, run_dir


def overrides_after_config(cli_args):
    return cli_args[cli_args.index("--config") + 2:]


def test_knobs_go_last_in_order(tmp_path):
    args, _, _, _ = run(
        tmp_path, {"micro_batch_size": 2, "global_batch_size": 32}, "--seq_length", "4096",
        env={
            "PRIMUS_TRAIN_ITERS": "10",
            "PRIMUS_DISABLE_PROFILE": "1",
            "PRIMUS_EXTRA_ARGS": "--pipeline_model_parallel_size 4  --use_torch_fsdp2 false",
        },
    )
    assert overrides_after_config(args) == [
        "--seq_length", "4096",
        "--train_iters", "10",
        "--profile", "false", "--use_pytorch_profiler", "false",
        "--pipeline_model_parallel_size", "4", "--use_torch_fsdp2", "false",
    ]


def test_extra_args_split_on_newlines_and_never_glob(tmp_path):
    # A manifest or YAML block scalar carries one override per line; `read -a` alone would
    # stop at the first newline and silently drop the rest.
    args, _, _, _ = run(
        tmp_path, {}, env={
            "PRIMUS_EXTRA_ARGS": "  --pipeline_model_parallel_size 4\n--use_torch_fsdp2\tfalse\n\n--note *\n",
        },
    )
    assert overrides_after_config(args) == [
        "--pipeline_model_parallel_size", "4", "--use_torch_fsdp2", "false", "--note", "*",
    ]


def test_cli_extra_is_consumed_before_extra_args(tmp_path):
    # madengine's primus launcher exports PRIMUS_CLI_EXTRA; PRIMUS_EXTRA_ARGS comes after
    # it, so it wins on a repeated key.
    args, _, _, _ = run(
        tmp_path, {}, env={
            "PRIMUS_TRAIN_ITERS": "10",
            "PRIMUS_CLI_EXTRA": "--use_x true\n--seq_length 2048",
            "PRIMUS_EXTRA_ARGS": "--seq_length 4096",
        },
    )
    assert overrides_after_config(args) == [
        "--train_iters", "10", "--use_x", "true", "--seq_length", "2048", "--seq_length", "4096",
    ]


def test_cli_extra_applies_to_other_frameworks(tmp_path):
    args, _, _, _ = run(
        tmp_path, {}, framework="torchtitan", env={"PRIMUS_CLI_EXTRA": "--training.steps 5"},
    )
    assert overrides_after_config(args)[-2:] == ["--training.steps", "5"]


def test_blank_extra_args_add_nothing(tmp_path):
    args, _, _, _ = run(
        tmp_path, {}, env={"PRIMUS_EXTRA_ARGS": " \n\t ", "PRIMUS_CLI_EXTRA": ""},
    )
    assert overrides_after_config(args) == []


def test_scaleout_gbs_inserted_before_caller_args(tmp_path):
    # 2 nodes x 8 GPUs, MBS 2: GBS must be a multiple of 32, so 24 becomes 32.
    args, _, _, _ = run(tmp_path, {"micro_batch_size": 2, "global_batch_size": 24})
    assert overrides_after_config(args) == ["--global_batch_size", "32"]


def test_scaleout_gbs_honours_the_interleaved_pipeline_schedule(tmp_path):
    # 6 nodes x 8 GPUs, MBS 8, TP 1, PP 2: DP 24, unit 192. 2048 would be 11 microbatches,
    # which Megatron's interleaved scheduler rejects with PP 2, so 12 -> 2304.
    overrides = {
        "micro_batch_size": 8, "global_batch_size": 2048, "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2, "virtual_pipeline_model_parallel_size": 2,
    }
    args, _, _, _ = run(tmp_path, overrides, env={"NNODES": "6"})
    assert overrides_after_config(args) == ["--global_batch_size", "2304"]


def test_scaleout_gbs_follows_a_pipeline_layout(tmp_path):
    # A pipeline_model_parallel_layout implies virtual stages too (16 stages / PP 4 = VPP 4):
    # qwen3_235B-style config on 6 nodes x 8 GPUs needs 172 microbatches of 48, not 171.
    layout = "Et*5|" + "t*6|" * 14 + "t*5,L"
    overrides = {
        "micro_batch_size": 4, "global_batch_size": 8192, "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 4, "pipeline_model_parallel_layout": layout,
    }
    args, _, _, _ = run(tmp_path, overrides, env={"NNODES": "6"})
    assert overrides_after_config(args) == ["--global_batch_size", str(172 * 48)]


def test_extra_args_gbs_suppresses_renormalization(tmp_path):
    args, _, _, _ = run(
        tmp_path, {"micro_batch_size": 2, "global_batch_size": 24},
        env={"PRIMUS_EXTRA_ARGS": "--global_batch_size 64"},
    )
    assert overrides_after_config(args) == ["--global_batch_size", "64"]


def test_megatron_only_knobs_skipped_for_other_frameworks(tmp_path):
    args, _, proc, _ = run(
        tmp_path, {}, framework="torchtitan",
        env={"PRIMUS_TRAIN_ITERS": "10", "PRIMUS_EXTRA_ARGS": "--training.steps 5"},
    )
    assert "--train_iters" not in args
    assert overrides_after_config(args)[-2:] == ["--training.steps", "5"]
    assert "apply to megatron configs only" in proc.stderr


KIMI = "kimi_k2_thinking-BF16-pretrain"


def test_model_overrides_come_first(tmp_path):
    args, _, _, _ = run(
        tmp_path, {"micro_batch_size": 4, "global_batch_size": 64}, "--micro_batch_size", "2",
        name=KIMI,
    )
    assert overrides_after_config(args) == [
        "--micro_batch_size", "1", "--global_batch_size", "128",
        "--recompute_granularity", "full", "--recompute_method", "uniform",
        "--recompute_num_layers", "1",
        "--micro_batch_size", "2",
    ]


def test_variant_selects_its_rules_and_is_not_forwarded(tmp_path):
    args, _, _, _ = run(tmp_path, {}, "--variant", "proxy", name=KIMI)
    assert overrides_after_config(args) == [
        "--num_layers", "3", "--moe_layer_freq", "1",
        "--micro_batch_size", "1", "--global_batch_size", "32",
    ]
    assert "--variant" not in args


def test_override_batch_size_is_renormalized(tmp_path):
    # The 16-GPU rule drops MBS 8 -> 2; DP 16 x MBS 2 makes GBS 24 invalid, so 32.
    args, _, _, _ = run(
        tmp_path, {"micro_batch_size": 8, "global_batch_size": 24},
        name="gpt_oss_120B-BF16-pretrain",
    )
    assert overrides_after_config(args) == [
        "--micro_batch_size", "2", "--recompute_num_layers", "9",
        "--global_batch_size", "32",
    ]


def test_wrapper_flags_are_stripped_in_the_equals_form(tmp_path):
    # --config_path=X and --variant=X are wrapper-only flags like their two-token forms;
    # Primus' CLI rejects both.
    args, _, _, _ = run(tmp_path, {}, "--variant=proxy", name=KIMI, config_flag="equals")
    assert not any(a.startswith(("--variant", "--config_path")) for a in args)
    assert overrides_after_config(args) == [
        "--num_layers", "3", "--moe_layer_freq", "1",
        "--micro_batch_size", "1", "--global_batch_size", "32",
    ]


@pytest.mark.parametrize("form", [
    ["--global_batch_size=64"], ["global_batch_size=64"], ["--global_batch_size", "64"],
])
def test_caller_gbs_in_any_form_suppresses_renormalization(tmp_path, form):
    args, _, _, _ = run(tmp_path, {"micro_batch_size": 2, "global_batch_size": 24}, *form)
    assert overrides_after_config(args) == form


def test_caller_micro_batch_in_equals_form_drives_renormalization(tmp_path):
    # Config MBS 8 would need GBS 128 on 16 GPUs; the caller's MBS 2 makes 32 enough.
    args, _, _, _ = run(
        tmp_path, {"micro_batch_size": 8, "global_batch_size": 24}, "--micro_batch_size=2",
    )
    assert overrides_after_config(args) == ["--global_batch_size", "32", "--micro_batch_size=2"]


def test_config_missing_from_the_checkout_uses_the_tree_that_has_it(tmp_path):
    # Kimi-K2's YAMLs live only in the overlay image's Primus tree; a mounted checkout with
    # the scripts/Primus submodule is found first but cannot open them.
    overlay = tmp_path / "overlay" / "Primus"
    _, cli_env, proc, _ = run(
        tmp_path, {}, config_root="overlay", env={"PRIMUS_ROOT": str(overlay)},
    )
    assert cli_env["PWD"] == str(overlay)
    assert f"is not in {tmp_path / 'scripts' / 'Primus'}; using the Primus tree at {overlay}" in proc.stdout


def test_config_present_in_the_checkout_keeps_the_checkout(tmp_path):
    other = tmp_path / "overlay" / "Primus"
    (other / "runner").mkdir(parents=True)
    (other / "primus-cli").write_text(STUB_CLI)
    _, cli_env, proc, _ = run(tmp_path, {}, env={"PRIMUS_ROOT": str(other)})
    assert cli_env["PWD"] == str(tmp_path / "scripts" / "Primus")
    assert "using the Primus tree" not in proc.stdout


def test_config_missing_everywhere_still_fails(tmp_path):
    # No second tree has it: the first pick is kept and the missing config is reported.
    _, _, proc, _ = run(
        tmp_path, {}, config_root="overlay", expect_ok=False,
    )
    assert "using the Primus tree" not in proc.stdout


def test_unknown_variant_fails(tmp_path):
    _, _, proc, _ = run(tmp_path, {}, "--variant", "proxy", expect_ok=False)
    assert "no 'proxy' variant of toy-BF16-pretrain for MI355X" in proc.stderr


def test_perf_csv_reports_the_run_shape(tmp_path):
    _, _, _, run_dir = run(tmp_path, {"micro_batch_size": 2, "global_batch_size": 24})
    with open(run_dir / "primus_perf_output.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["metric"] for r in rows] == [
        "tokens_per_second", "tflops", "model_flops_utilization"
    ]
    assert rows[0]["performance"] == "1799.0"
    for row in rows:
        assert {k: row[k] for k in (
            "mode", "precision", "batch_size", "global_batch_size", "seq_len", "device", "num_gpus"
        )} == {
            "mode": "pretrain", "precision": "BF16", "batch_size": "2",
            "global_batch_size": "32", "seq_len": "8192", "device": "MI355X", "num_gpus": "16",
        }


def test_perf_csv_micro_batch_without_an_argument_dump(tmp_path):
    # Only rank 0 dumps Megatron's arguments; the reporting last node has none, so the
    # micro batch size comes from the config plus the CLI overrides run.sh passed.
    _, _, _, run_dir = run(
        tmp_path, {"micro_batch_size": 4, "global_batch_size": 64, "seq_length": 8192},
        "--micro_batch_size", "2", env={"STUB_NO_ARG_DUMP": "1"},
    )
    with open(run_dir / "primus_perf_output.csv") as f:
        row = next(csv.DictReader(f))
    assert row["batch_size"] == "2"
    assert row["seq_len"] == "8192"


def _perf_rows(run_dir):
    with open(run_dir / "primus_perf_output.csv") as f:
        return list(csv.DictReader(f))


def test_bridge_tps_is_derived_from_the_whole_run(tmp_path):
    # Megatron-Bridge prints no TPS; it is seq_length x GBS / elapsed / num_gpus, and
    # num_gpus must be every GPU of the run (2 nodes x 8), not this node's 8.
    _, _, _, run_dir = run(tmp_path, {"seq_length": 4096}, env={"STUB_LOG": "bridge"})
    rows = _perf_rows(run_dir)
    assert rows[0]["performance"] == f"{4096 * 16 / 0.2 / 16:.1f}"
    assert {r["num_gpus"] for r in rows} == {"16"}


@pytest.mark.parametrize("env, expected", [
    ({}, "16"),
    ({"NNODES": "4"}, "32"),
    ({"HIP_VISIBLE_DEVICES": "0,1,2,3"}, "8"),
    ({"GPUS_PER_NODE": "", "NPROC_PER_NODE": "4"}, "8"),
])
def test_num_gpus_without_a_logged_world_size_spans_all_nodes(tmp_path, env, expected):
    _, _, _, run_dir = run(tmp_path, {}, env={"STUB_NO_ARG_DUMP": "1", **env})
    assert {r["num_gpus"] for r in _perf_rows(run_dir)} == {expected}


@pytest.mark.parametrize("framework, node_rank, reports", [
    ("megatron", "1", True),          # Megatron prints on the last rank: the last node
    ("megatron", "0", False),
    ("megatron_bridge", "0", True),   # the Bridge patch logs through log_rank_0: node 0
    ("megatron_bridge", "1", False),
])
def test_only_the_reporting_node_fails_without_throughput(tmp_path, framework, node_rank, reports):
    kwargs = dict(framework=framework, env={"STUB_LOG": "none", "NODE_RANK": node_rank})
    if reports:
        _, _, proc, _ = run(tmp_path, {}, expect_ok=False, **kwargs)
        assert "TPS extraction failed" in proc.stderr
    else:
        _, _, proc, _ = run(tmp_path, {}, **kwargs)
        reporter = "0" if framework == "megatron_bridge" else "1"
        assert f"no TPS on node {node_rank} of 2; node {reporter} reports it" in proc.stdout


def test_rccl_log_dir_gives_each_rank_a_file(tmp_path):
    log_dir = tmp_path / "rccl_logs"
    _, env, _, _ = run(tmp_path, {}, env={"RCCL_LOG_DIR": str(log_dir)})
    assert env["NCCL_DEBUG_FILE"] == f"{log_dir}/BF16.node_1.%h.%p.log"
    assert log_dir.is_dir()


@pytest.mark.parametrize("name, phase", [
    ("llama3.1_8B-MXFP8-pretrain", "MXFP8"),
    ("mamba_130M_pretrain", "run"),
])
def test_rccl_log_name_leads_with_the_precision(tmp_path, name, phase):
    log_dir = tmp_path / "rccl_logs"
    _, env, _, _ = run(tmp_path, {}, env={"RCCL_LOG_DIR": str(log_dir)}, name=name)
    assert env["NCCL_DEBUG_FILE"] == f"{log_dir}/{phase}.node_1.%h.%p.log"


@pytest.mark.parametrize("env, expected", [
    ({}, "INFO"),                      # per-rank logs asked for, nothing logged without a level
    ({"NCCL_DEBUG": "WARN"}, "WARN"),  # a level the deployment sets wins
    ({"NCCL_DEBUG": ""}, "INFO"),      # Primus' base_env.sh style empty value is "unset"
])
def test_rccl_log_dir_turns_the_log_on(tmp_path, env, expected):
    log_dir = tmp_path / "rccl_logs"
    _, cli_env, _, _ = run(tmp_path, {}, env={"RCCL_LOG_DIR": str(log_dir), **env})
    assert cli_env["NCCL_DEBUG"] == expected


def test_nccl_debug_is_left_alone_without_rccl_log_dir(tmp_path):
    _, env, _, _ = run(tmp_path, {})
    assert env["NCCL_DEBUG"] == "<unset>"


def test_unwritable_rccl_log_dir_keeps_stdout(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    _, env, proc, _ = run(
        tmp_path, {}, env={"RCCL_LOG_DIR": str(blocker / "sub"), "NCCL_DEBUG_FILE": "stale"}
    )
    assert env["NCCL_DEBUG_FILE"] == "<unset>"
    assert env["NCCL_DEBUG"] == "<unset>"
    assert "is not writable from the container" in proc.stderr
