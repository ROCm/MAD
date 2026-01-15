import csv, sys, json
import argparse
import os

# csv header
header = [
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

# parse arg
parser = argparse.ArgumentParser(
    description="Convert vllm json output format to perf csv output format"
)
parser.add_argument("--benchmark", help="latency or throughput or serving")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--vllm_json", help="path to the vllm json file")
parser.add_argument("--output_csv", help="path to the perf csv file")
parser.add_argument("--tensor-parallel-size", type=str, help="tensor parallel size")
parser.add_argument("--batch-size", type=str, help="batch size")
parser.add_argument("--num-prompts", type=str, help="number of prompts")
parser.add_argument(
    "--max-concurrency", type=str, help="max concurrency (serving only)"
)
parser.add_argument("--input-len", type=str, help="input seq length")
parser.add_argument("--output-len", type=str, help="output gen length")
parser.add_argument("--dtype", type=str, help="data type")
parser.add_argument("--vllm-cmd", help="vllm run command")

# read args
args = parser.parse_args()

# write json to csv
header_write = 0 if os.path.exists(args.output_csv) else 1
with open(args.output_csv, "a+", newline="") as outf:
    writer = csv.DictWriter(outf, delimiter=",", fieldnames=header)
    if header_write:
        writer.writeheader()
    with open(args.vllm_json, newline="") as inpf:
        reader = json.load(inpf)
        try:
            if args.benchmark == "latency":
                if reader["avg_latency"] != 0:
                    row = {
                        "model": args.model,
                        "benchmark": args.benchmark,
                        "tp": args.tensor_parallel_size,
                        "inp": args.input_len,
                        "out": args.output_len,
                        "dtype": args.dtype,
                        "bs": args.batch_size,
                        "cmd": args.vllm_cmd,
                        "performance": str(reader["avg_latency"]),
                        "metric": "avg_latency",
                        "unit": "sec",
                    }
                    writer.writerow(row)
            elif args.benchmark == "throughput":
                if reader["tokens_per_second"] != 0:
                    elapsed_time = reader["elapsed_time"]
                    throughput_gen = str(
                        int(int(args.num_prompts) * int(args.output_len) / elapsed_time)
                    )
                    metrics = {
                        "throughput_tot": str(reader["tokens_per_second"]),
                        "throughput_gen": throughput_gen,
                    }
                    for metric, perf in metrics.items():
                        row = {
                            "model": args.model,
                            "benchmark": args.benchmark,
                            "tp": args.tensor_parallel_size,
                            "inp": args.input_len,
                            "out": args.output_len,
                            "dtype": args.dtype,
                            "num_prompts": args.num_prompts,
                            "cmd": args.vllm_cmd,
                            "performance": perf,
                            "metric": metric,
                            "unit": "tok/sec",
                        }
                        writer.writerow(row)
            elif args.benchmark == "serving":
                if reader["total_token_throughput"] != 0:
                    metrics = {
                        "throughput_tot": str(reader["total_token_throughput"]),
                        "throughput_gen": str(reader["output_throughput"]),
                        "median_tpot": str(reader["median_tpot_ms"]),
                        "median_itl": str(reader["median_itl_ms"]),
                        "median_e2el": str(reader["median_e2el_ms"]),
                    }
                    for metric, perf in metrics.items():
                        if "throughput" in metric:
                            unit = "tok/sec"
                        else:
                            unit = "ms"
                        row = {
                            "model": args.model,
                            "benchmark": args.benchmark,
                            "tp": args.tensor_parallel_size,
                            "inp": args.input_len,
                            "out": args.output_len,
                            "dtype": args.dtype,
                            "num_prompts": args.num_prompts,
                            "max_concurrency": args.max_concurrency,
                            "cmd": args.vllm_cmd,
                            "performance": perf,
                            "metric": metric,
                            "unit": unit,
                        }
                        writer.writerow(row)

        except csv.Error as e:
            sys.exit("file {}, line {}: {}".format(args.vllm_json, reader.line_num, e))
