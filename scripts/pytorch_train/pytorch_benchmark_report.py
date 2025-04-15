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
import re

# parse arguments
parser = argparse.ArgumentParser(description='Convert pytorch train output format to MAD csv output format')
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

# read arguments
args = parser.parse_args()
input_file = args.input
output_file = args.output
print("Input file path: ", input_file)
print("Output file path: ", output_file)

def find_match(file, search_string):
    with open(file, 'r') as file:
        content = file.read()
    pattern = fr"{re.escape(search_string)}\s*(\d+\.\d+|\d+)"
    matches = re.findall(pattern, content)
    # Only save last match
    match = matches[-1]
    return match

if args.mode == "pretrain":
    if args.model == "Llama-3.1-8B":
        tok_per_s_per_gpu = find_match(input_file, "Avg token per second:")
        TFLOPS_per_gpu = find_match(input_file, "Avg TFLOP/s:")
        data = [
            {'model': args.model, 'performance': tok_per_s_per_gpu, 'metric': 'tok_per_s_per_gpu'},
            {'model': args.model, 'performance': TFLOPS_per_gpu, 'metric': 'TFLOPS_per_gpu'}
        ]
    elif args.model == "Llama-3.1-70B":
        tok_per_s_per_gpu = find_match(input_file, "tps:")
        TFLOPS_per_gpu = find_match(input_file, "tflops:")
        data = [
            {'model': args.model, 'performance': tok_per_s_per_gpu, 'metric': 'tok_per_s_per_gpu'},
            {'model': args.model, 'performance': TFLOPS_per_gpu, 'metric': 'TFLOPS_per_gpu'}
        ]
    elif args.model == "Flux":
        df = pd.read_csv(input_file)
        FPS_per_GPU = df.iloc[-1]['avg_fps']
        TFLOPS_per_GPU = df.iloc[-1]['avg_tflops']
        data = [
            {'model': args.model, 'performance': FPS_per_GPU, 'metric': 'FPS_per_GPU'},
            {'model': args.model, 'performance': TFLOPS_per_GPU, 'metric': 'TFLOPS_per_GPU'}
        ]
elif (args.mode == "finetune_fw" or args.mode == "finetune_lora" or args.mode == "finetune_qlora") and (args.model == "Llama-3.1-70B" or args.model == "Llama-3.3-70B"):
    max_memory_alloc = find_match(input_file, "Max memory alloc:")
    avg_tokens_per_s_per_gpu = find_match(input_file, "Average tokens/s/gpu:")
    unmasked_tokens_per_s_per_gpu = find_match(input_file, "Unmasked tokens/s/gpu: ")
    data = [
        {'model': args.model, 'performance': max_memory_alloc, 'metric': 'max_memory_alloc'},
        {'model': args.model, 'performance': unmasked_tokens_per_s_per_gpu, 'metric': 'unmasked_tokens_per_s_per_gpu'},
        {'model': args.model, 'performance': avg_tokens_per_s_per_gpu, 'metric': 'avg_tokens_per_s_per_gpu'}
    ]
elif (args.mode == "HF_finetune_lora") and (args.model == "Llama-3.1-70B" or args.model == "Llama-2-70B"):
    train_samples_per_s = find_match(input_file, "'train_samples_per_second':")
    data = [
        {'model': args.model, 'performance': train_samples_per_s, 'metric': 'train_samples_per_s'}
    ]

with open(output_file, mode='w', newline='') as file:
    print("Preparing to write performance data...")
    print("Data: ", data)
    writer = csv.DictWriter(file, fieldnames=['model','performance','metric'])
    writer.writeheader()
    writer.writerows(data)
    print("Completed writing to output file")
    
