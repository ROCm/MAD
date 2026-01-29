###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

import pandas as pd
import argparse
import csv
import os
import re

# parse arguments
parser = argparse.ArgumentParser(description='Convert primus pytorch train output format to MAD csv output format')
parser.add_argument("--mode",
                        type=str,
                        help="pretrain or finetune")
parser.add_argument("--model",
                        type=str,
                        help="model name")
parser.add_argument("--input",
                        type=str,
                        help="path to input file")
parser.add_argument("--output",
                        type=str,
                        help="path to output file")
parser.add_argument("--precision",
                        type=str,
                        help="precision type")
parser.add_argument("--batch_size",
                        type=str,
                        help="batch size")
parser.add_argument("--seq_len",
                        type=str,
                        help="sequence length")
parser.add_argument("--device",
                        type=str,
                        help="device name")
parser.add_argument("--num_gpus",
                        type=str,
                        help="number of GPUs")

# read arguments
args = parser.parse_args()
input_file = args.input
output_file = args.output
print("Input file path: ", input_file)
print("Output file path: ", output_file)

def find_match(file, search_string):
    with open(file, 'r') as file:
        content = file.read()
    # Updated pattern to match the new log format
    # Looks for patterns like "TFLOP/s/GPU): 322.5/320.6" or "tokens/s/GPU): 725.1/720.6"
    # and extracts the first value (before the slash)
    pattern = fr"{re.escape(search_string)}\):\s*(\d+\.?\d*)/"
    matches = re.findall(pattern, content)
    if matches:
        # Return the last 2 values if they exist (one from each run)
        if len(matches) >= 2:
            result = [matches[-2], matches[-1]]
            print(f"Found {len(matches)} matches for '{search_string}', using last 2: {result}")
        else:
            result = [matches[-1]]
            print(f"Found {len(matches)} match for '{search_string}': {result}")
        return result
    else:
        print(f"Warning: No matches found for '{search_string}' pattern")
        return []

if args.model == "Llama-3.1-8B" or args.model == "Llama-3.1-70B" or \
        args.model == "Llama-2-7B" or args.model == "Llama-2-70B" or \
        args.model == "Mixtral-8x7B" or args.model == "Mixtral-8x22B-proxy" or \
        args.model == "DeepSeek-V2-lite" or args.model == "DeepSeek-V3-proxy" or \
        args.model == "Llama-3.1-70B-proxy" or args.model == "Llama-3.3-70B" or \
        args.model == "Qwen2.5-7B" or args.model == "Qwen2.5-72B":
    tok_per_s_per_gpu_list = find_match(input_file, "tokens/s/GPU")
    TFLOPS_per_gpu_list = find_match(input_file, "TFLOP/s/GPU")
    
    data = []
    # Write separate rows for each run
    for i, (tps, tflops) in enumerate(zip(tok_per_s_per_gpu_list, TFLOPS_per_gpu_list)):
        run_label = f"run_{i+1}" if len(tok_per_s_per_gpu_list) > 1 else ""
        data.extend([
            {'model': args.model, 'performance': tps, 'metric': 'tok_per_s_per_gpu', 'mode': args.mode, 'precision': args.precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus, 'run': run_label},
            {'model': args.model, 'performance': tflops, 'metric': 'TFLOPS_per_gpu', 'mode': args.mode, 'precision': args.precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus, 'run': run_label}
        ])

if not os.path.exists(output_file) or os.stat(output_file).st_size == 0:
    mode = 'w'  # Write if file doesn't exist or is empty
else:
    mode = 'a'  # Append if file exists and is not empty
with open(output_file, mode=mode, newline='') as file:
    print("Preparing to write performance data...")
    print("Data: ", data)
    writer = csv.DictWriter(file, fieldnames=['model','performance','metric','mode','precision','batch_size','seq_len','device','num_gpus','run'])
    if mode == 'w':
        writer.writeheader()
    writer.writerows(data)
    print("Completed writing to output file")

