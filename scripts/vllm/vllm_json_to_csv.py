import csv, sys, json
import argparse
import os

# parse arg
parser = argparse.ArgumentParser(description='Convert vllm json output format to perf csv output format')
parser.add_argument("--benchmark",
                        help="latency or throughput or serving")
parser.add_argument("--model",
                        type=str,
                        help="model name")
parser.add_argument("--vllm_json",
                        help="path to the vllm json file")
parser.add_argument("--output_csv",
                        help="path to the perf csv file")
parser.add_argument("--tensor-parallel-size",
                        type=str,
                        help="tensor parallel size")
parser.add_argument("--batch-size",
                        type=str,
                        help="batch size")
parser.add_argument("--num-prompts",
                        type=str,
                        help="number of prompts")
parser.add_argument("--max-concurrency",
                        type=str,
                        help="max concurrency (serving only)")
parser.add_argument("--input-len",
                        type=str,
                        help="input seq length")
parser.add_argument("--output-len",
                        type=str,
                        help="output gen length")
parser.add_argument("--dtype",
                        type=str,
                        help="data type")

# read args
args = parser.parse_args()

with open(args.vllm_json, newline='') as inpf:
    header_write = 0 if os.path.exists(args.output_csv) else 1
    with open(args.output_csv,'a+',newline='') as outf:
        writer = csv.writer(outf, delimiter=',')
        reader = json.load(inpf)
        try:
            if args.benchmark == "latency":
                if(reader["avg_latency"] != 0):
                    writer.writerow(['model', 'tp', 'bs', 'in', 'out', 'dtype', 'avg_latency (sec)']) if header_write else None
                    writer.writerow([args.model, args.tensor_parallel_size, args.batch_size, args.input_len, args.output_len, args.dtype, str(reader["avg_latency"])])
            elif args.benchmark == "throughput":
                if(reader["tokens_per_second"] != 0):
                    writer.writerow(['model', 'tp', 'num_prompts', 'in', 'out', 'dtype', 'throughput_tot (tok/sec)', 'throughput_gen (tok/sec)']) if header_write else None
                    elapsed_time = reader["elapsed_time"]
                    throughput_gen = str(int(int(args.num_prompts) * int(args.output_len) / elapsed_time))
                    writer.writerow([args.model, args.tensor_parallel_size, args.num_prompts, args.input_len, args.output_len, args.dtype, str(reader["tokens_per_second"]), throughput_gen])
            elif args.benchmark == "serving":
                if(reader["total_token_throughput"] != 0):
                    writer.writerow(['model', 'tp', 'max_concurrency', 'num_prompts', 'in', 'out', 'dtype', 'throughput_tot (tok/sec)', 'throughput_gen (tok/sec)', 'median_tpot (ms)', 'median_itl (ms)', 'median_e2el (ms)']) if header_write else None
                    writer.writerow(
                        [
                            args.model,
                            args.tensor_parallel_size,
                            args.max_concurrency,
                            args.num_prompts,
                            args.input_len,
                            args.output_len,
                            args.dtype,
                            str(reader["total_token_throughput"]),
                            str(reader["output_throughput"]),
                            str(reader["median_tpot_ms"]),
                            str(reader["median_itl_ms"]),
                            str(reader["median_e2el_ms"])
                        ]
                    )

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(args.vllm_json, reader.line_num, e))
