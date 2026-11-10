"""The llm-d benchmark CSV uses the same serving rows as plain vLLM."""

import csv
import json
from pathlib import Path

from llmd_bench_csv import write_serving_csv


def test_serving_json_writes_the_plain_vllm_metric_rows(tmp_path, capsys):
    result = tmp_path / "bench.json"
    result.write_text(
        json.dumps(
            {
                "total_token_throughput": 200.5,
                "output_throughput": 100.25,
                "median_ttft_ms": 12.5,
                "median_tpot_ms": 3.5,
                "median_itl_ms": 3.4,
                "median_e2el_ms": 400.0,
            }
        )
    )
    csv_path = tmp_path / "perf.csv"

    write_serving_csv(
        result,
        csv_path,
        model="Qwen/Qwen3-8B",
        tensor_parallel="1",
        input_len="256",
        output_len="128",
        num_prompts="32",
        max_concurrency="4",
        dtype="auto",
        command="vllm bench serve",
    )

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["metric"] for row in rows] == [
        "throughput_tot",
        "throughput_gen",
        "median_ttft",
        "median_tpot",
        "median_itl",
        "median_e2el",
    ]
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["throughput_tot"]["performance"] == "200.5"
    assert by_metric["throughput_tot"]["unit"] == "tok/sec"
    assert by_metric["throughput_gen"]["performance"] == "100.25"
    assert by_metric["median_ttft"]["performance"] == "12.5"
    assert by_metric["median_ttft"]["unit"] == "ms"
    assert by_metric["median_e2el"]["unit"] == "ms"
    for row in rows:
        assert row["model"] == "Qwen/Qwen3-8B"
        assert row["benchmark"] == "serving"
        assert row["tp"] == "1"
        assert row["inp"] == "256"
        assert row["out"] == "128"
        assert row["num_prompts"] == "32"
        assert row["max_concurrency"] == "4"
        assert row["dtype"] == "auto"
        assert row["cmd"] == "vllm bench serve"

    assert capsys.readouterr().out.strip() == "performance: 100.25 tokens_per_second"


def test_mirror_skips_the_same_file(tmp_path):
    """When cwd is /workspace, /workspace/<csv> is the same path and must not fail."""
    from llmd_bench_csv import mirror_csv_destinations

    src = tmp_path / "perf.csv"
    src.write_text("model,performance,metric\nm,1,x\n")
    same = tmp_path / "perf.csv"
    other = tmp_path / "other" / "perf.csv"

    mirror_csv_destinations(src, [same, other])

    assert other.read_text() == src.read_text()
