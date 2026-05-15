###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
import numpy as np
import argparse
import csv
import re

# parse arguments
parser = argparse.ArgumentParser(description='Convert pytorch train output format to MAD csv output format')
parser.add_argument("--mode",
                        type=str,
                        help="pretrain or finetune")
parser.add_argument("--model",
                        type=str,
                        help="model name")
parser.add_argument("--quantization",
                        type=str,
                        default="bf16",
                        help="quantization type, e.g. bf16, nanoo_fp8, etc.")
parser.add_argument("--input",
                        type=str,
                        help="path to input file")
parser.add_argument("--output",
                        type=str,
                        help="path to output file")
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
quantization = args.quantization
print("Input file path: ", input_file)
print("Output file path: ", output_file)
print("Quantization: ", quantization)

def find_match(file, search_string, num_iters):
    with open(file, 'r') as file:
        content = file.read()
    pattern = fr"{re.escape(search_string)}\s*(\d+\.\d+|\d+)"
    matches = re.findall(pattern, content)
    perf_nums = [float(num) for num in matches][-num_iters:]
    avg = np.average(perf_nums)
    return str("{:.2f}".format(avg))

if args.model == "Llama-3.1-8B" or args.model == "Llama-3.1-70B" or \
        args.model == "Llama-3.3-70B" or \
        args.model == "Llama-2-7B" or args.model == "Llama-2-70B" or \
        args.model == "DeepSeek-V2-lite" or args.model == "Mixtral-8x7B" or\
        args.model == "Qwen3-14B" or\
        args.model == "Qwen3-30B-A3B":
    tok_per_s_per_gpu = find_match(input_file, "Tokens/s/device:", 10)
    TFLOPS_per_gpu = find_match(input_file, "TFLOP/s/device:", 10)
    data = [
        {'model': args.model, 'performance': tok_per_s_per_gpu, 'metric': 'tok_per_s_per_gpu', 'mode': args.mode, 'precision': args.quantization, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
        {'model': args.model, 'performance': TFLOPS_per_gpu, 'metric': 'TFLOPS_per_gpu', 'mode': args.mode, 'precision': args.quantization, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
    ]

with open(output_file, mode='w', newline='') as file:
    print("Preparing to write performance data...")
    print("Data: ", data)
    writer = csv.DictWriter(file, fieldnames=['model','performance','metric','mode','precision','batch_size','seq_len','device','num_gpus'])
    writer.writeheader()
    writer.writerows(data)
    print("Completed writing to output file")
