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
import numpy as np
import argparse
import csv
import os
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
parser.add_argument("--precision",
                        type=str,
                        help="training precision")
parser.add_argument("--batch_size",
                        type=int,
                        help="batch size used for training")
parser.add_argument("--seq_len",
                        type=int,
                        help="sequence length used for training")
parser.add_argument("--device",
                        type=str,
                        help="device architecture (e.g., MI300X, MI350X, MI355X)")
parser.add_argument("--num_gpus",
                        type=int,
                        help="number of GPUs used for training")

# read arguments
args = parser.parse_args()
input_file = args.input
output_file = args.output
precision = args.precision
print("Input file path: ", input_file)
print("Output file path: ", output_file)
print("Precision: ", precision)

def find_match(file, search_string, search_range=None):
    with open(file, 'r') as file:
        content = file.read()
    print("Content ", content)
    # Match numbers with commas or decimals
    pattern = fr"{re.escape(search_string)}\s*(\d+\.\d+|\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?|\d+)"
    matches = re.findall(pattern, content)
    data = [s.replace(",", "") for s in matches]
    # Only save last match
    if search_range is not None:
        data = np.array(data[search_range[0]:search_range[1]])
        result = np.average(data.astype(float))
    else:
        result = data[-1]
    print(f"{search_string} {result}")
    return result

def find_token_match(file, search_string, search_range=None):
    with open(file, 'r') as file:
        content = file.read()
    print("Content ", content)
    # Match numbers with commas or decimals
    pattern = fr'{re.escape(search_string)}\s*(\d{{1,7}})'
    matches = re.findall(pattern, content)
    data = [s.replace(",", "") for s in matches]
    # Only save last match
    if search_range is not None:
        data = np.array(data[search_range[0]:search_range[1]])
        result = np.average(data.astype(float))
    else:
        result = data[-1]
    print(f"{search_string} {result}")
    return result

finetune_models = ["Llama-2-70B", "Llama-2-13B", "Llama-2-7B", "Llama-3-70B", "Llama-3-8B", \
            "Llama-3.1-405B", "Llama-3.1-70B", "Llama-3.1-8B", \
            "Llama-3.2-3B", "Llama-3.2-1B", "Llama-3.2-vision-11B", "Llama-3.2-vision-90B", \
            "Llama-3.3-70B", "Llama-4-scout-17B-16E", \
            "Qwen2-1.5B", "Qwen2-7B", "Qwen2.5-32B", "Qwen2.5-72B", "Qwen3-8B", "Qwen3-32B"]

if args.mode == "pretrain":
    if args.model == "Llama-3.1-8B":
        tok_per_s_per_gpu = find_match(input_file, "tps:", None)
        TFLOPS_per_gpu = find_match(input_file, "tflops:", None)
        data = [
            {'model': args.model, 'performance': tok_per_s_per_gpu, 'metric': 'tok_per_s_per_gpu', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
            {'model': args.model, 'performance': TFLOPS_per_gpu, 'metric': 'TFLOPS_per_gpu', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
        ]
    elif args.model == "Llama-3.1-70B":
        tok_per_s_per_gpu = find_match(input_file, "tps:", None)
        TFLOPS_per_gpu = find_match(input_file, "tflops:", None)
        data = [
            {'model': args.model, 'performance': tok_per_s_per_gpu, 'metric': 'tok_per_s_per_gpu', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
            {'model': args.model, 'performance': TFLOPS_per_gpu, 'metric': 'TFLOPS_per_gpu', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
        ]
    elif args.model == "DLRM":
        df = pd.read_csv(input_file)
        recs_per_s_mean = (df.iloc[-1]['Recommendations/s (mean)']).item()
        recs_per_s_cv = (df.iloc[-1]['Recommendations/s (std/mean)']).item()
        data = [
            {'model': args.model, 'performance': recs_per_s_mean, 'metric': 'recs_per_s_mean', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
            {'model': args.model, 'performance': recs_per_s_cv, 'metric': 'recs_per_s_cv', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
        ]
    
elif args.mode == "HF_pretrain":
    if args.model == "Llama-3.1-8B":
        tok_per_s_per_gpu = find_match(input_file, "Avg token per second:")
        TFLOPS_per_gpu = find_match(input_file, "Avg TFLOP/s:")
        data = [
            {'model': args.model, 'performance': tok_per_s_per_gpu, 'metric': 'tok_per_s_per_gpu', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
            {'model': args.model, 'performance': TFLOPS_per_gpu, 'metric': 'TFLOPS_per_gpu', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
        ]

elif args.mode == "posttrain":
    if (args.model == "Flux" or args.model == "Stable-Diffusion-XL" or args.model == "Mochi-1" or args.model == "Hunyuan-video" or args.model == "Wan2_1-i2v"):
        df = pd.read_csv(input_file)
        FPS_per_GPU = float(df.iloc[-1]['avg_fps_gpu'])
        TFLOPS_per_GPU = float(df.iloc[-1]['avg_tflops'])
        data = [
            {'model': args.model, 'performance': FPS_per_GPU, 'metric': 'FPS_per_GPU', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
            {'model': args.model, 'performance': TFLOPS_per_GPU, 'metric': 'TFLOPS_per_GPU', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
        ] 

elif (args.mode == "finetune_fw" or args.mode == "finetune_lora" or args.mode == "finetune_qlora") and (args.model in finetune_models):
    max_memory_alloc = find_match(input_file, "Max memory alloc (last half):")
    avg_tokens_per_s_per_gpu = find_token_match(input_file, "Average tokens/s/gpu (last half):")
    
    data = [
        {'model': args.model, 'performance': max_memory_alloc, 'metric': 'max_memory_alloc', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
        {'model': args.model, 'performance': avg_tokens_per_s_per_gpu, 'metric': 'avg_tokens_per_s_per_gpu', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
    ]

elif (args.mode == "HF_finetune_lora") and (args.model == "GPT-OSS-20B" or args.model == "GPT-OSS-120B"):
    train_samples_per_s = find_match(input_file, "'train_samples_per_second':")
    data = [
        {'model': args.model, 'performance': train_samples_per_s, 'metric': 'train_samples_per_s', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
    ]

if not os.path.exists(output_file) or os.stat(output_file).st_size == 0:
    mode = 'w'  # Write if file doesn't exist or is empty
else:
    mode = 'a'  # Append if file exists and is not empty
with open(output_file, mode=mode, newline='') as file:
    print("Preparing to write performance data...")
    print("Data: ", data)
    writer = csv.DictWriter(file, fieldnames=['model','performance','metric', 'mode', 'precision', 'batch_size', 'seq_len', 'device', 'num_gpus'])
    if mode == 'w':
        writer.writeheader()
    writer.writerows(data)
    print("Completed writing to output file")
    
