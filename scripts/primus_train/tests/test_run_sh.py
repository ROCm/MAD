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

import yaml

SRC = Path(__file__).resolve().parents[1]
HELPERS = ("run.sh", "scaleout_batch.py", "extract_primus_perf.py")

STUB_CLI = """\
#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_ARGS"
printf 'NCCL_DEBUG_FILE=%s\\n' "${NCCL_DEBUG_FILE-<unset>}" > "$STUB_ENV"
log=""
prev=""
for a in "$@"; do
  [[ "$prev" == "--log_file" ]] && log="$a"
  prev="$a"
done
mkdir -p "$(dirname "$log")"
cat > "$log" <<'EOF'
  micro_batch_size ................................ 2
  seq_length ...................................... 8192
  world_size ...................................... 16
 iteration       10/      10 | elapsed time per iteration (ms): 1000.0 | compute per GPU (TFLOP/s/GPU): 500.0 (avg 499.0) | tokens/s/GPU inst/harmonic mean: 1800.0/1799.0 | global batch size:    32 |
EOF
"""


def make_tree(tmp_path, overrides, framework="megatron", name="toy-BF16-pretrain"):
    scripts = tmp_path / "scripts"
    train = scripts / "primus_train"
    train.mkdir(parents=True)
    for helper in HELPERS:
        if (SRC / helper).exists():
            shutil.copy(SRC / helper, train / helper)
    primus = scripts / "Primus"
    rel = f"examples/{framework}/configs/MI355X/{name}.yaml"
    cfg = primus / rel
    cfg.parent.mkdir(parents=True)
    cfg.write_text(yaml.safe_dump(
        {"modules": {"pre_trainer": {"framework": framework, "overrides": overrides}}}
    ))
    cli = primus / "primus-cli"
    cli.write_text(STUB_CLI)
    cli.chmod(0o755)
    run_dir = tmp_path / "run_directory"
    run_dir.mkdir()
    return train / "run.sh", rel, run_dir


def run(tmp_path, overrides, *args, env=None, framework="megatron", name="toy-BF16-pretrain"):
    run_sh, rel, run_dir = make_tree(tmp_path, overrides, framework, name)
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
    proc = subprocess.run(
        ["bash", str(run_sh), "--config_path", rel, *args],
        cwd=run_dir, env=base_env, capture_output=True, text=True,
    )
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


def test_scaleout_gbs_inserted_before_caller_args(tmp_path):
    # 2 nodes x 8 GPUs, MBS 2: GBS must be a multiple of 32, so 24 becomes 32.
    args, _, _, _ = run(tmp_path, {"micro_batch_size": 2, "global_batch_size": 24})
    assert overrides_after_config(args) == ["--global_batch_size", "32"]


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


def test_rccl_log_dir_gives_each_rank_a_file(tmp_path):
    log_dir = tmp_path / "rccl_logs"
    _, env, _, _ = run(tmp_path, {}, env={"RCCL_LOG_DIR": str(log_dir)})
    assert env["NCCL_DEBUG_FILE"] == f"{log_dir}/toy-BF16-pretrain.node_1.%h.%p.log"
    assert log_dir.is_dir()


def test_unwritable_rccl_log_dir_keeps_stdout(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    _, env, proc, _ = run(
        tmp_path, {}, env={"RCCL_LOG_DIR": str(blocker / "sub"), "NCCL_DEBUG_FILE": "stale"}
    )
    assert env["NCCL_DEBUG_FILE"] == "<unset>"
    assert "is not writable from the container" in proc.stderr
