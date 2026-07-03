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
                        help="pretrain or posttrain")
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
                        help="micro batch size")
parser.add_argument("--global_batch_size",
                        type=str,
                        default=None,
                        help="global batch size")
parser.add_argument("--posttrain_type",
                        type=str,
                        default=None,
                        help="post-training type (sft or lora)")
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

def find_match(file_path, search_string):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern 1: Original format like "TFLOP/s/GPU): 322.5/320.6" or "tokens/s/GPU): 725.1/720.6"
    # Extracts the first value (before the slash)
    pattern1 = fr"{re.escape(search_string)}\):\s*(\d+\.?\d*)/"
    matches1 = re.findall(pattern1, content)
    
    # Pattern 2: Qwen-3-32B format like "7184.9MODEL_TFLOP/s/GPU" or "1234.5tokens/s/GPU"
    # Value comes directly before the metric text (no colon, no space)
    # Allow for optional whitespace or other characters before the number
    pattern2 = fr"(\d+\.?\d*){re.escape(search_string)}"
    matches2 = re.findall(pattern2, content)
    
    # Also try with MODEL_ prefix for TFLOP (e.g., "7184.9MODEL_TFLOP/s/GPU")
    if "TFLOP/s/GPU" in search_string:
        pattern3 = fr"(\d+\.?\d*)MODEL_TFLOP/s/GPU"
        matches3 = re.findall(pattern3, content)
        matches2.extend(matches3)
    
    # Pattern 4: Handle format with colon but no slash (e.g., "GPU utilization: 7184.9MODEL_TFLOP/s/GPU")
    # This captures cases where there might be text before the value
    if "TFLOP/s/GPU" in search_string:
        pattern4 = fr":\s*(\d+\.?\d*)MODEL_TFLOP/s/GPU"
        matches4 = re.findall(pattern4, content)
        matches2.extend(matches4)
    
    # Combine all matches and return the last one
    all_matches = matches1 + matches2
    if all_matches:
        # Return only the last value
        result = all_matches[-1]
        print(f"Found {len(all_matches)} matches for '{search_string}', using last: {result}")
        return result
    else:
        print(f"Warning: No matches found for '{search_string}' pattern")
        return None

def find_match_running_avg(file_path, search_string):
    """Return the running-average value (the number AFTER the slash) from the
    LAST logged iteration, e.g. '1885.6' from 'tokens/s/GPU): 2355.1/1885.6'.

    Primus logs throughput as 'instant/running_avg'. The instantaneous value
    has large run-to-run jitter (MoE routing imbalance, collective stragglers,
    kernel warmup), so the trailing running average is a steadier figure,
    especially for multi-node scaleout runs. Returns None when no match."""
    with open(file_path, 'r') as f:
        content = f.read()
    pattern = fr"{re.escape(search_string)}\):\s*\d+\.?\d*/(\d+\.?\d*)"
    matches = re.findall(pattern, content)
    if matches:
        print(f"Found {len(matches)} running-avg matches for '{search_string}', using last: {matches[-1]}")
        return matches[-1]
    return None

if args.model == "Llama-3.1-8B" or args.model == "Llama-3.1-70B" or \
        args.model == "Llama-3.1-405B" or \
        args.model == "Llama-2-7B" or args.model == "Llama-2-70B" or \
        args.model == "Mixtral-8x7B" or args.model == "Mixtral-8x22B-proxy" or \
        args.model == "DeepSeek-V2-lite" or args.model == "DeepSeek-V3-proxy" or \
        args.model == "Llama-3.1-70B-proxy" or args.model == "Llama-3.3-70B" or \
        args.model == "Qwen2.5-7B" or args.model == "Qwen2.5-72B" or \
        args.model == "Zebra-Llama-1B" or args.model == "Zebra-Llama-3B" or args.model == "Zebra-Llama-8B" or \
        args.model == "Qwen-3-32B" or args.model == "Mamba-370M" or \
        args.model == "GPT-OSS-20B" or args.model == "GPT-OSS-120B" or args.model == "Qwen-3-30B" or \
        args.model == "Qwen-3-235B":
    # Only extract tokens/s/GPU for models other than Qwen-3-32B
    if args.model != "Qwen-3-32B":
        tok_per_s_per_gpu = find_match(input_file, "tokens/s/GPU")
    else:
        tok_per_s_per_gpu = None
    TFLOPS_per_gpu = find_match(input_file, "TFLOP/s/GPU")
    
    data = []
    # Write data for metrics that are found (don't require both to be present)
    # Skip tokens/s/GPU for Qwen-3-32B
    if tok_per_s_per_gpu is not None and args.model != "Qwen-3-32B":
        data.append({'model': args.model, 'performance': tok_per_s_per_gpu, 'metric': 'tok_per_s_per_gpu', 'mode': args.mode, 'precision': args.precision, 'batch_size': args.batch_size, 'global_batch_size': args.global_batch_size, 'posttrain_type': args.posttrain_type, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus, 'run': ''})
    if TFLOPS_per_gpu is not None:
        data.append({'model': args.model, 'performance': TFLOPS_per_gpu, 'metric': 'TFLOPS_per_gpu', 'mode': args.mode, 'precision': args.precision, 'batch_size': args.batch_size, 'global_batch_size': args.global_batch_size, 'posttrain_type': args.posttrain_type, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus, 'run': ''})

    # Additionally emit the trailing running-average throughput (value after the
    # slash in the last logged iteration). Kept as separate, distinctly-named
    # rows (metric suffix '_avg', run='avg') so the instantaneous metrics above
    # are preserved and never confused with the steady-state average. This is a
    # steadier figure for multi-node scaleout runs where per-iteration jitter is
    # largest. Skipped for Qwen-3-32B, matching the tokens/s/GPU handling above.
    if args.model != "Qwen-3-32B":
        tok_avg = find_match_running_avg(input_file, "tokens/s/GPU")
        if tok_avg is not None:
            data.append({'model': args.model, 'performance': tok_avg, 'metric': 'tok_per_s_per_gpu_avg', 'mode': args.mode, 'precision': args.precision, 'batch_size': args.batch_size, 'global_batch_size': args.global_batch_size, 'posttrain_type': args.posttrain_type, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus, 'run': 'avg'})
    tflops_avg = find_match_running_avg(input_file, "TFLOP/s/GPU")
    if tflops_avg is not None:
        data.append({'model': args.model, 'performance': tflops_avg, 'metric': 'TFLOPS_per_gpu_avg', 'mode': args.mode, 'precision': args.precision, 'batch_size': args.batch_size, 'global_batch_size': args.global_batch_size, 'posttrain_type': args.posttrain_type, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus, 'run': 'avg'})

if not os.path.exists(output_file) or os.stat(output_file).st_size == 0:
    mode = 'w'  # Write if file doesn't exist or is empty
else:
    mode = 'a'  # Append if file exists and is not empty
with open(output_file, mode=mode, newline='') as file:
    print("Preparing to write performance data...")
    print("Data: ", data)
    writer = csv.DictWriter(file, fieldnames=['model','performance','metric','mode','precision','batch_size','global_batch_size','posttrain_type','seq_len','device','num_gpus','run'])
    if mode == 'w':
        writer.writeheader()
    writer.writerows(data)
    print("Completed writing to output file")
