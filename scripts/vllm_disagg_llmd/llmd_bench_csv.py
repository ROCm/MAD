"""Turn one vllm bench serve JSON file into the llm-d results CSV.

The rows match scripts/vllm/vllm_json_to_csv.py for benchmark=serving.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

# Same columns as scripts/vllm/vllm_json_to_csv.py. Serving leaves bs empty.
_HEADER = [
    "model",
    "benchmark",
    "tp",
    "inp",
    "out",
    "dtype",
    "num_prompts",
    "max_concurrency",
    "bs",
    "cmd",
    "performance",
    "metric",
    "unit",
]

# metric, JSON key, unit. Order matches the plain vLLM serving converter.
_SERVING_METRICS = (
    ("throughput_tot", "total_token_throughput", "tok/sec"),
    ("throughput_gen", "output_throughput", "tok/sec"),
    ("median_ttft", "median_ttft_ms", "ms"),
    ("median_tpot", "median_tpot_ms", "ms"),
    ("median_itl", "median_itl_ms", "ms"),
    ("median_e2el", "median_e2el_ms", "ms"),
)


def write_serving_csv(
    result_path,
    csv_path,
    model,
    tensor_parallel,
    input_len,
    output_len,
    num_prompts,
    max_concurrency,
    dtype,
    command,
):
    """Write the six serving rows and print the madengine performance line."""
    out = json.loads(Path(result_path).read_text())
    missing = [key for _, key, _ in _SERVING_METRICS if key not in out]
    if missing:
        raise SystemExit(
            "vllm bench result is missing: " + ", ".join(missing)
        )

    rows = []
    for metric, key, unit in _SERVING_METRICS:
        rows.append(
            {
                "model": model,
                "benchmark": "serving",
                "tp": tensor_parallel,
                "inp": input_len,
                "out": output_len,
                "dtype": dtype,
                "num_prompts": num_prompts,
                "max_concurrency": max_concurrency,
                "bs": "",
                "cmd": command,
                "performance": str(out[key]),
                "metric": metric,
                "unit": unit,
            }
        )

    csv_path = Path(csv_path)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"performance: {out['output_throughput']} tokens_per_second",
        flush=True,
    )


def mirror_csv_destinations(csv_path, destinations):
    """Copy the CSV to collector paths. Skip destinations that are the same file.

    The Job looks under ``/workspace/<csv>``, the working directory, and the
    parent directory. When cwd is already ``/workspace``, a plain ``cp`` to
    ``/workspace/<csv>`` fails with "are the same file" and ``set -e`` turns
    that into a failed Job even though the benchmark succeeded.
    """
    src = Path(csv_path).resolve()
    for dest in destinations:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() == src:
            continue
        shutil.copy2(src, dest)


def main(argv=None):
    """CLI used by run_llmd.sh."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", default="1")
    parser.add_argument("--input-len", required=True)
    parser.add_argument("--output-len", required=True)
    parser.add_argument("--num-prompts", required=True)
    parser.add_argument("--max-concurrency", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--vllm-cmd", default="")
    args = parser.parse_args(argv)
    write_serving_csv(
        args.result,
        args.output,
        model=args.model,
        tensor_parallel=args.tensor_parallel_size,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        max_concurrency=args.max_concurrency,
        dtype=args.dtype,
        command=args.vllm_cmd,
    )
    csv_name = Path(args.output).name
    mirror_csv_destinations(
        args.output,
        [
            Path("/workspace") / csv_name,
            Path(csv_name),
            Path("..") / csv_name,
        ],
    )


if __name__ == "__main__":
    main()
